"""
Tests for HealthMonitor component (NodeClient decomposition).

Tests cover:
- Health observation tracking
- Health gossip protocol
- Misbehavior detection
- Eclipse attack anomaly detection
- Keepalive tracking

Note: This tests the HealthMonitor from health_monitor.py, not the one
from seed.py (which is tested in test_health_monitor.py).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.network.health_monitor import (
    HealthMonitor,
    HealthMonitorConfig,
)
from valence.network.messages import (
    HealthGossip,
    RouterHealthObservation,
    MisbehaviorType,
    RouterBehaviorMetrics,
    MisbehaviorEvidence,
    MisbehaviorReport,
    NetworkBaseline,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_discovery():
    """Create a mock DiscoveryClient."""
    discovery = MagicMock()
    discovery.report_misbehavior = AsyncMock()
    return discovery


@pytest.fixture
def config():
    """Create a test configuration."""
    return HealthMonitorConfig(
        gossip_interval=5.0,
        gossip_ttl=2,
        observation_max_age=60.0,
        keepalive_interval=1.0,
        ping_timeout=1.0,
        missed_pings_threshold=2,
        misbehavior_detection_enabled=True,
        min_messages_for_detection=5,  # Low for testing
        delivery_rate_threshold=0.75,
        ack_failure_threshold=0.30,
        mild_severity_threshold=0.3,
        severe_severity_threshold=0.7,
        anomaly_detection_enabled=True,
        anomaly_window=10.0,
        anomaly_threshold=2,  # Low for testing
    )


@pytest.fixture
def health_monitor(mock_discovery, config):
    """Create a HealthMonitor for testing."""
    return HealthMonitor(
        node_id="a" * 64,
        discovery=mock_discovery,
        config=config,
    )


@pytest.fixture
def mock_router_connection():
    """Create a mock RouterConnection."""
    from valence.network.discovery import RouterInfo
    
    router = RouterInfo(
        router_id="b" * 64,
        endpoints=["192.168.1.1:8471"],
        capacity={"current_load_pct": 25},
        health={},
        regions=[],
        features=[],
    )
    
    conn = MagicMock()
    conn.router = router
    conn.ping_latency_ms = 50.0
    conn.ack_success_rate = 0.9
    conn.ack_success = 9
    conn.ack_failure = 1
    return conn


# =============================================================================
# Unit Tests - HealthMonitorConfig
# =============================================================================


class TestHealthMonitorConfig:
    """Tests for HealthMonitorConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = HealthMonitorConfig()
        assert config.gossip_interval == 30.0
        assert config.gossip_ttl == 2
        assert config.keepalive_interval == 2.0
        assert config.misbehavior_detection_enabled is True
        assert config.anomaly_detection_enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = HealthMonitorConfig(
            gossip_interval=10.0,
            misbehavior_detection_enabled=False,
        )
        assert config.gossip_interval == 10.0
        assert config.misbehavior_detection_enabled is False


# =============================================================================
# Unit Tests - HealthMonitor Properties
# =============================================================================


class TestHealthMonitorProperties:
    """Tests for HealthMonitor property methods."""

    def test_get_stats_initial(self, health_monitor):
        """Test initial statistics."""
        stats = health_monitor.get_stats()
        assert stats["gossip_sent"] == 0
        assert stats["gossip_received"] == 0
        assert stats["anomalies_detected"] == 0
        assert stats["routers_flagged"] == 0
        assert stats["own_observations"] == 0
        assert stats["peer_observation_sources"] == 0


# =============================================================================
# Unit Tests - Health Observations
# =============================================================================


class TestHealthObservations:
    """Tests for health observation tracking."""

    def test_update_observation(self, health_monitor, mock_router_connection):
        """Test updating router health observation."""
        health_monitor.update_observation(
            router_id=mock_router_connection.router.router_id,
            conn=mock_router_connection,
        )
        
        # Observation should be stored
        assert mock_router_connection.router.router_id in health_monitor._own_observations
        obs = health_monitor._own_observations[mock_router_connection.router.router_id]
        assert obs.latency_ms == 50.0
        assert obs.success_rate == 0.9

    def test_get_aggregated_health_own_only(self, health_monitor, mock_router_connection):
        """Test aggregated health with only own observations."""
        router_id = mock_router_connection.router.router_id
        
        # Add own observation
        health_monitor.update_observation(router_id, mock_router_connection)
        
        score = health_monitor.get_aggregated_health(router_id)
        
        # Should return a score between 0 and 1
        assert 0.0 <= score <= 1.0

    def test_get_aggregated_health_no_data(self, health_monitor):
        """Test aggregated health with no data."""
        score = health_monitor.get_aggregated_health("unknown-router")
        
        # Should return default score
        assert score == 0.5

    def test_get_aggregated_health_with_peer_observations(self, health_monitor, mock_router_connection):
        """Test aggregated health combining own and peer observations."""
        router_id = mock_router_connection.router.router_id
        
        # Add own observation
        health_monitor.update_observation(router_id, mock_router_connection)
        
        # Add peer observation
        peer_obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=60.0,
            success_rate=0.85,
            last_seen=time.time(),
        )
        health_monitor._peer_observations["peer1"] = {router_id: peer_obs}
        
        score = health_monitor.get_aggregated_health(router_id)
        
        # Should blend both observations
        assert 0.0 <= score <= 1.0

    def test_get_health_observations(self, health_monitor, mock_router_connection):
        """Test getting health observations summary."""
        # Add observation
        health_monitor.update_observation(
            mock_router_connection.router.router_id,
            mock_router_connection,
        )
        
        result = health_monitor.get_health_observations()
        
        assert "own_observations" in result
        assert "peer_observation_count" in result
        assert "aggregated_health_scores" in result


# =============================================================================
# Unit Tests - Health Gossip
# =============================================================================


class TestHealthGossip:
    """Tests for health gossip protocol."""

    def test_create_gossip_message(self, health_monitor, mock_router_connection):
        """Test creating gossip message."""
        # Add observation first
        health_monitor.update_observation(
            mock_router_connection.router.router_id,
            mock_router_connection,
        )
        
        gossip = health_monitor.create_gossip_message()
        
        assert gossip.source_node_id == health_monitor.node_id
        assert gossip.ttl == health_monitor.config.gossip_ttl
        assert len(gossip.observations) > 0

    def test_create_gossip_message_empty(self, health_monitor):
        """Test creating gossip message with no observations."""
        gossip = health_monitor.create_gossip_message()
        
        assert gossip.source_node_id == health_monitor.node_id
        assert len(gossip.observations) == 0

    def test_handle_gossip_from_peer(self, health_monitor):
        """Test handling incoming gossip from peer."""
        gossip = HealthGossip(
            source_node_id="peer123",
            timestamp=time.time(),
            observations=[
                RouterHealthObservation(
                    router_id="router1",
                    latency_ms=45.0,
                    success_rate=0.95,
                    last_seen=time.time(),
                ),
            ],
            ttl=2,
        )
        
        health_monitor.handle_gossip(gossip)
        
        # Peer observations should be stored
        assert "peer123" in health_monitor._peer_observations
        assert health_monitor._stats["gossip_received"] == 1

    def test_handle_gossip_from_self(self, health_monitor):
        """Test that gossip from self is ignored."""
        gossip = HealthGossip(
            source_node_id=health_monitor.node_id,
            timestamp=time.time(),
            observations=[],
            ttl=2,
        )
        
        health_monitor.handle_gossip(gossip)
        
        # Should not count as received
        assert health_monitor._stats["gossip_received"] == 0

    def test_handle_gossip_expired_ttl(self, health_monitor):
        """Test that gossip with expired TTL is ignored."""
        gossip = HealthGossip(
            source_node_id="peer123",
            timestamp=time.time(),
            observations=[],
            ttl=0,
        )
        
        health_monitor.handle_gossip(gossip)
        
        # Should not process
        assert "peer123" not in health_monitor._peer_observations


# =============================================================================
# Unit Tests - Misbehavior Detection
# =============================================================================


class TestMisbehaviorDetection:
    """Tests for misbehavior detection."""

    def test_record_delivery_outcome_success(self, health_monitor):
        """Test recording successful delivery."""
        health_monitor.record_delivery_outcome(
            router_id="router1",
            message_id="msg1",
            delivered=True,
            latency_ms=50.0,
        )
        
        metrics = health_monitor._router_behavior_metrics["router1"]
        assert metrics.messages_sent == 1
        assert metrics.messages_delivered == 1
        assert metrics.avg_latency_ms == 50.0

    def test_record_delivery_outcome_failure(self, health_monitor):
        """Test recording failed delivery."""
        health_monitor.record_delivery_outcome(
            router_id="router1",
            message_id="msg1",
            delivered=False,
        )
        
        metrics = health_monitor._router_behavior_metrics["router1"]
        assert metrics.messages_sent == 1
        assert metrics.messages_dropped == 1

    def test_record_ack_outcome(self, health_monitor):
        """Test recording ACK outcomes."""
        health_monitor.record_ack_outcome("router1", success=True)
        health_monitor.record_ack_outcome("router1", success=False)
        
        metrics = health_monitor._router_behavior_metrics["router1"]
        assert metrics.ack_success_count == 1
        assert metrics.ack_failure_count == 1
        assert health_monitor._stats["misbehavior_ack_success"] == 1
        assert health_monitor._stats["misbehavior_ack_failure"] == 1

    def test_misbehavior_detection_disabled(self, health_monitor):
        """Test that misbehavior detection can be disabled."""
        health_monitor.config.misbehavior_detection_enabled = False
        
        health_monitor.record_delivery_outcome("router1", "msg1", False)
        
        # Should not track metrics
        assert "router1" not in health_monitor._router_behavior_metrics

    def test_is_router_flagged_not_flagged(self, health_monitor):
        """Test checking if router is flagged (not flagged)."""
        assert health_monitor.is_router_flagged("router1") is False

    def test_is_router_flagged_flagged(self, health_monitor):
        """Test checking if router is flagged (flagged)."""
        # Add flagged router
        report = MagicMock()
        health_monitor._flagged_routers["router1"] = report
        
        assert health_monitor.is_router_flagged("router1") is True

    def test_get_flagged_routers_empty(self, health_monitor):
        """Test getting flagged routers when none exist."""
        result = health_monitor.get_flagged_routers()
        assert result == {}

    def test_clear_router_flag(self, health_monitor):
        """Test clearing router flag."""
        # Add flagged router
        report = MagicMock()
        health_monitor._flagged_routers["router1"] = report
        
        result = health_monitor.clear_router_flag("router1")
        
        assert result is True
        assert "router1" not in health_monitor._flagged_routers

    def test_clear_router_flag_not_flagged(self, health_monitor):
        """Test clearing flag for non-flagged router."""
        result = health_monitor.clear_router_flag("router1")
        assert result is False

    def test_get_misbehavior_detection_stats(self, health_monitor):
        """Test getting misbehavior detection statistics."""
        stats = health_monitor.get_misbehavior_detection_stats()
        
        assert stats["enabled"] is True
        assert stats["routers_tracked"] == 0
        assert stats["routers_flagged"] == 0
        assert "thresholds" in stats

    def test_check_router_behavior_insufficient_data(self, health_monitor):
        """Test behavior check with insufficient data."""
        # Record just one message (below min_messages_for_detection)
        health_monitor.record_delivery_outcome("router1", "msg1", False)
        
        # Should not trigger detection yet
        assert health_monitor.is_router_flagged("router1") is False

    def test_check_router_behavior_flags_misbehaving_router(self, health_monitor):
        """Test that misbehaving router gets flagged."""
        router_id = "router1"
        
        # Record many failures to exceed threshold
        for i in range(10):
            health_monitor.record_delivery_outcome(router_id, f"msg{i}", delivered=False)
            health_monitor.record_ack_outcome(router_id, success=False)
        
        # Should be flagged
        assert health_monitor.is_router_flagged(router_id) is True
        assert health_monitor._stats["routers_flagged"] == 1


# =============================================================================
# Unit Tests - Eclipse Anomaly Detection
# =============================================================================


class TestEclipseAnomalyDetection:
    """Tests for eclipse attack anomaly detection."""

    def test_record_failure_event(self, health_monitor):
        """Test recording failure event."""
        health_monitor.record_failure_event(
            router_id="router1",
            failure_type="connection",
            error_code="TIMEOUT",
        )
        
        assert len(health_monitor._failure_events) == 1
        assert health_monitor._failure_events[0]["router_id"] == "router1"

    def test_record_failure_event_disabled(self, health_monitor):
        """Test that failure recording is disabled when anomaly detection off."""
        health_monitor.config.anomaly_detection_enabled = False
        
        health_monitor.record_failure_event("router1", "connection")
        
        assert len(health_monitor._failure_events) == 0

    def test_record_failure_event_prunes_old(self, health_monitor):
        """Test that old failure events are pruned."""
        # Add old event
        health_monitor._failure_events.append({
            "router_id": "old-router",
            "failure_type": "connection",
            "timestamp": time.time() - 100,  # Old event
        })
        
        # Add new event
        health_monitor.record_failure_event("router1", "connection")
        
        # Old event should be pruned
        assert len(health_monitor._failure_events) == 1
        assert health_monitor._failure_events[0]["router_id"] == "router1"

    def test_detect_anomaly_correlated_failures(self, health_monitor):
        """Test detecting correlated failures (eclipse anomaly)."""
        # Record multiple failures across different routers
        # with same failure type
        for i in range(3):
            health_monitor.record_failure_event(
                router_id=f"router{i}",
                failure_type="connection",
            )
        
        # Should detect anomalies (triggers at threshold=2 and again at 3)
        # Each time we exceed threshold with new unique routers, a new alert is generated
        assert health_monitor._stats["anomalies_detected"] >= 1
        assert len(health_monitor._anomaly_alerts) >= 1

    def test_get_anomaly_alerts_empty(self, health_monitor):
        """Test getting anomaly alerts when none exist."""
        alerts = health_monitor.get_anomaly_alerts()
        assert alerts == []

    def test_clear_anomaly_alerts(self, health_monitor):
        """Test clearing anomaly alerts."""
        # Add alert
        health_monitor._anomaly_alerts.append({"type": "test"})
        
        count = health_monitor.clear_anomaly_alerts()
        
        assert count == 1
        assert len(health_monitor._anomaly_alerts) == 0


# =============================================================================
# Unit Tests - Keepalive Tracking
# =============================================================================


class TestKeepaliveTracking:
    """Tests for keepalive ping tracking."""

    def test_track_ping_response_success(self, health_monitor):
        """Test tracking successful ping."""
        # First record some misses
        health_monitor._missed_pings["router1"] = 1
        
        should_fail = health_monitor.track_ping_response("router1", success=True)
        
        assert should_fail is False
        assert health_monitor._missed_pings["router1"] == 0

    def test_track_ping_response_failure(self, health_monitor):
        """Test tracking failed ping."""
        should_fail = health_monitor.track_ping_response("router1", success=False)
        
        assert should_fail is False
        assert health_monitor._missed_pings["router1"] == 1

    def test_track_ping_response_threshold_exceeded(self, health_monitor):
        """Test that exceeding missed pings threshold triggers failure."""
        # Miss pings up to threshold
        health_monitor.track_ping_response("router1", success=False)
        should_fail = health_monitor.track_ping_response("router1", success=False)
        
        assert should_fail is True
        assert health_monitor._missed_pings["router1"] == 2

    def test_clear_ping_state(self, health_monitor):
        """Test clearing ping state."""
        health_monitor._missed_pings["router1"] = 3
        
        health_monitor.clear_ping_state("router1")
        
        assert "router1" not in health_monitor._missed_pings


# =============================================================================
# Integration Tests
# =============================================================================


class TestHealthMonitorIntegration:
    """Integration tests for HealthMonitor."""

    def test_full_misbehavior_detection_flow(self, health_monitor):
        """Test full misbehavior detection flow."""
        router_id = "misbehaving-router"
        
        # Simulate router exhibiting misbehavior
        for i in range(10):
            health_monitor.record_delivery_outcome(
                router_id=router_id,
                message_id=f"msg-{i}",
                delivered=False,  # All messages dropped
            )
            health_monitor.record_ack_outcome(router_id, success=False)
        
        # Check detection
        assert health_monitor.is_router_flagged(router_id) is True
        
        # Get stats
        stats = health_monitor.get_misbehavior_detection_stats()
        assert stats["routers_flagged"] == 1
        assert router_id in stats["flagged_routers"]
        
        # Clear flag
        health_monitor.clear_router_flag(router_id)
        assert health_monitor.is_router_flagged(router_id) is False

    def test_full_gossip_flow(self, health_monitor, mock_router_connection):
        """Test full gossip flow."""
        router_id = mock_router_connection.router.router_id
        
        # Update observation
        health_monitor.update_observation(router_id, mock_router_connection)
        
        # Create gossip
        gossip = health_monitor.create_gossip_message()
        assert len(gossip.observations) > 0
        
        # Simulate receiving gossip from peer
        peer_gossip = HealthGossip(
            source_node_id="peer-node",
            timestamp=time.time(),
            observations=[
                RouterHealthObservation(
                    router_id=router_id,
                    latency_ms=60.0,
                    success_rate=0.88,
                    last_seen=time.time(),
                ),
            ],
            ttl=2,
        )
        health_monitor.handle_gossip(peer_gossip)
        
        # Aggregated health should incorporate both
        score = health_monitor.get_aggregated_health(router_id)
        assert 0.0 <= score <= 1.0

    def test_on_router_flagged_callback(self, health_monitor):
        """Test that on_router_flagged callback is called."""
        callback_args = []
        
        def on_flagged(router_id, report):
            callback_args.append((router_id, report))
        
        health_monitor.on_router_flagged = on_flagged
        
        # Trigger flagging
        router_id = "bad-router"
        for i in range(10):
            health_monitor.record_delivery_outcome(router_id, f"msg-{i}", False)
            health_monitor.record_ack_outcome(router_id, success=False)
        
        # Callback should have been called
        assert len(callback_args) == 1
        assert callback_args[0][0] == router_id
