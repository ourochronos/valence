"""
Tests for Malicious Router Detection (Issue #119).

Tests cover:
1. RouterBehaviorMetrics tracking
2. MisbehaviorReport message types
3. Network baseline calculation
4. Anomaly detection and flagging
5. Automatic avoidance of flagged routers
6. Misbehavior report submission to seeds
7. Seed handling of misbehavior reports
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.network.messages import (
    MisbehaviorType,
    RouterBehaviorMetrics,
    MisbehaviorEvidence,
    MisbehaviorReport,
    NetworkBaseline,
)
from valence.network.node import NodeClient, RouterConnection
from valence.network.discovery import RouterInfo

pytestmark = pytest.mark.skip(
    reason="Needs update for NodeClient decomposition - see #167"
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_private_key():
    """Create a mock Ed25519 private key."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    return Ed25519PrivateKey.generate()


@pytest.fixture
def mock_encryption_key():
    """Create a mock X25519 private key."""
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    return X25519PrivateKey.generate()


@pytest.fixture
def node_client(mock_private_key, mock_encryption_key):
    """Create a NodeClient for testing."""
    node_id = mock_private_key.public_key().public_bytes_raw().hex()
    return NodeClient(
        node_id=node_id,
        private_key=mock_private_key,
        encryption_private_key=mock_encryption_key,
        misbehavior_detection_enabled=True,
        min_messages_for_detection=5,  # Low for testing
    )


@pytest.fixture
def mock_router_info():
    """Create mock router info."""
    return RouterInfo(
        router_id="a" * 64,
        endpoints=["192.168.1.1:8471"],
        capacity={"max_connections": 100, "current_load_pct": 30},
        health={"uptime_pct": 99, "avg_latency_ms": 50},
        regions=["us-west"],
        features=["relay"],
    )


# =============================================================================
# ROUTER BEHAVIOR METRICS TESTS
# =============================================================================


class TestRouterBehaviorMetrics:
    """Tests for RouterBehaviorMetrics class."""
    
    def test_metrics_creation(self):
        """Test creating behavior metrics."""
        metrics = RouterBehaviorMetrics(router_id="test_router_123")
        
        assert metrics.router_id == "test_router_123"
        assert metrics.messages_sent == 0
        assert metrics.messages_delivered == 0
        assert metrics.delivery_rate == 1.0  # Default when no messages
        assert metrics.ack_success_rate == 1.0  # Default when no ACKs
        assert not metrics.flagged
    
    def test_delivery_rate_calculation(self):
        """Test delivery rate calculation."""
        metrics = RouterBehaviorMetrics(router_id="test")
        
        # No messages yet
        assert metrics.delivery_rate == 1.0
        
        # Record deliveries
        metrics.record_delivery(True)  # Delivered
        metrics.record_delivery(True)  # Delivered
        metrics.record_delivery(False)  # Dropped
        
        assert metrics.messages_sent == 3
        assert metrics.messages_delivered == 2
        assert metrics.messages_dropped == 1
        assert abs(metrics.delivery_rate - 0.667) < 0.01
    
    def test_ack_success_rate(self):
        """Test ACK success rate calculation."""
        metrics = RouterBehaviorMetrics(router_id="test")
        
        # No ACKs yet
        assert metrics.ack_success_rate == 1.0
        
        # Record ACKs
        metrics.record_ack(True)
        metrics.record_ack(True)
        metrics.record_ack(True)
        metrics.record_ack(False)
        
        assert metrics.ack_success_count == 3
        assert metrics.ack_failure_count == 1
        assert metrics.ack_success_rate == 0.75
    
    def test_latency_tracking(self):
        """Test latency recording and averaging."""
        metrics = RouterBehaviorMetrics(router_id="test")
        
        # No samples yet
        assert metrics.avg_latency_ms == 0.0
        
        # Record latencies
        metrics.record_latency(100.0)
        assert metrics.avg_latency_ms == 100.0
        
        metrics.record_latency(200.0)
        assert metrics.avg_latency_ms == 150.0
        
        metrics.record_latency(300.0)
        assert abs(metrics.avg_latency_ms - 200.0) < 0.01
    
    def test_serialization(self):
        """Test metrics serialization and deserialization."""
        metrics = RouterBehaviorMetrics(
            router_id="test_router",
            messages_sent=100,
            messages_delivered=95,
            messages_dropped=5,
            ack_success_count=90,
            ack_failure_count=10,
        )
        metrics.record_latency(150.0)
        metrics.flagged = True
        metrics.flag_reason = "message_drop"
        
        data = metrics.to_dict()
        
        assert data["router_id"] == "test_router"
        assert data["messages_sent"] == 100
        assert data["messages_delivered"] == 95
        assert data["delivery_rate"] == 0.95
        assert data["flagged"] is True
        
        # Deserialize
        metrics2 = RouterBehaviorMetrics.from_dict(data)
        assert metrics2.router_id == metrics.router_id
        assert metrics2.messages_sent == metrics.messages_sent
        assert metrics2.flagged == metrics.flagged


# =============================================================================
# MISBEHAVIOR EVIDENCE TESTS
# =============================================================================


class TestMisbehaviorEvidence:
    """Tests for MisbehaviorEvidence class."""
    
    def test_evidence_creation(self):
        """Test creating misbehavior evidence."""
        evidence = MisbehaviorEvidence(
            misbehavior_type=MisbehaviorType.MESSAGE_DROP,
            delivery_rate_baseline=0.95,
            delivery_rate_observed=0.60,
            description="Delivery rate 60% below threshold 75%"
        )
        
        assert evidence.misbehavior_type == "message_drop"
        assert evidence.delivery_rate_baseline == 0.95
        assert evidence.delivery_rate_observed == 0.60
    
    def test_evidence_serialization(self):
        """Test evidence serialization."""
        evidence = MisbehaviorEvidence(
            misbehavior_type=MisbehaviorType.MESSAGE_DELAY,
            expected_latency_ms=100.0,
            actual_latency_ms=500.0,
        )
        
        data = evidence.to_dict()
        
        assert data["misbehavior_type"] == "message_delay"
        assert data["expected_latency_ms"] == 100.0
        
        evidence2 = MisbehaviorEvidence.from_dict(data)
        assert evidence2.misbehavior_type == evidence.misbehavior_type
        assert evidence2.actual_latency_ms == evidence.actual_latency_ms


# =============================================================================
# MISBEHAVIOR REPORT TESTS
# =============================================================================


class TestMisbehaviorReport:
    """Tests for MisbehaviorReport class."""
    
    def test_report_creation(self):
        """Test creating a misbehavior report."""
        metrics = RouterBehaviorMetrics(router_id="bad_router")
        evidence = MisbehaviorEvidence(
            misbehavior_type=MisbehaviorType.ACK_FAILURE,
            description="ACK failure rate 40% exceeds threshold 30%"
        )
        
        report = MisbehaviorReport(
            reporter_id="node_123",
            router_id="bad_router",
            misbehavior_type=MisbehaviorType.ACK_FAILURE,
            evidence=[evidence],
            metrics=metrics,
            severity=0.5,
        )
        
        assert report.type == "misbehavior_report"
        assert report.reporter_id == "node_123"
        assert report.router_id == "bad_router"
        assert report.severity == 0.5
        assert len(report.evidence) == 1
    
    def test_report_serialization(self):
        """Test report serialization and deserialization."""
        evidence = MisbehaviorEvidence(
            misbehavior_type=MisbehaviorType.MESSAGE_DROP,
        )
        metrics = RouterBehaviorMetrics(router_id="bad_router")
        
        report = MisbehaviorReport(
            reporter_id="node_abc",
            router_id="bad_router",
            misbehavior_type=MisbehaviorType.MESSAGE_DROP,
            evidence=[evidence],
            metrics=metrics,
            severity=0.7,
        )
        
        data = report.to_dict()
        
        assert data["type"] == "misbehavior_report"
        assert data["reporter_id"] == "node_abc"
        assert data["severity"] == 0.7
        
        # Deserialize
        report2 = MisbehaviorReport.from_dict(data)
        assert report2.reporter_id == report.reporter_id
        assert report2.router_id == report.router_id
        assert report2.severity == report.severity
        assert len(report2.evidence) == 1


# =============================================================================
# NETWORK BASELINE TESTS
# =============================================================================


class TestNetworkBaseline:
    """Tests for NetworkBaseline class."""
    
    def test_baseline_defaults(self):
        """Test baseline default values."""
        baseline = NetworkBaseline()
        
        assert baseline.avg_delivery_rate == 0.95
        assert baseline.avg_latency_ms == 100.0
        assert baseline.avg_ack_success_rate == 0.95
    
    def test_delivery_rate_anomaly_detection(self):
        """Test detecting anomalous delivery rates."""
        baseline = NetworkBaseline(
            avg_delivery_rate=0.95,
            delivery_rate_stddev=0.05,
        )
        
        # Normal rate (within 2 stddev)
        assert not baseline.is_delivery_rate_anomalous(0.90)
        assert not baseline.is_delivery_rate_anomalous(0.88)
        
        # Anomalous rate (below 2 stddev)
        assert baseline.is_delivery_rate_anomalous(0.80)
        assert baseline.is_delivery_rate_anomalous(0.50)
    
    def test_latency_anomaly_detection(self):
        """Test detecting anomalous latencies."""
        baseline = NetworkBaseline(
            avg_latency_ms=100.0,
            latency_stddev_ms=50.0,
        )
        
        # Normal latency (within 2 stddev)
        assert not baseline.is_latency_anomalous(150.0)
        assert not baseline.is_latency_anomalous(180.0)
        
        # Anomalous latency (above 2 stddev)
        assert baseline.is_latency_anomalous(250.0)
        assert baseline.is_latency_anomalous(500.0)


# =============================================================================
# NODE BEHAVIOR TRACKING TESTS
# =============================================================================


class TestNodeBehaviorTracking:
    """Tests for node-level behavior tracking."""
    
    def test_record_message_sent(self, node_client):
        """Test recording message sent through router."""
        router_id = "test_router_" + "a" * 52
        
        node_client._record_message_sent(router_id, "msg_001")
        
        assert router_id in node_client._router_behavior_metrics
        metrics = node_client._router_behavior_metrics[router_id]
        assert metrics.messages_sent == 1
    
    def test_record_delivery_outcome(self, node_client):
        """Test recording delivery outcomes."""
        router_id = "test_router_" + "b" * 52
        
        # Record successful delivery
        node_client._record_delivery_outcome(router_id, "msg_001", True, 100.0)
        
        metrics = node_client._router_behavior_metrics[router_id]
        assert metrics.messages_delivered == 1
        assert metrics.avg_latency_ms == 100.0
        
        # Record failed delivery
        node_client._record_delivery_outcome(router_id, "msg_002", False)
        
        assert metrics.messages_dropped == 1
    
    def test_record_ack_outcome(self, node_client):
        """Test recording ACK outcomes."""
        router_id = "test_router_" + "c" * 52
        
        node_client._record_ack_outcome(router_id, True)
        node_client._record_ack_outcome(router_id, True)
        node_client._record_ack_outcome(router_id, False)
        
        metrics = node_client._router_behavior_metrics[router_id]
        assert metrics.ack_success_count == 2
        assert metrics.ack_failure_count == 1
    
    def test_update_network_baseline(self, node_client):
        """Test network baseline calculation."""
        # Add metrics for multiple routers
        for i in range(5):
            router_id = f"router_{i}" + "x" * 55
            metrics = node_client._get_router_metrics(router_id)
            
            # Simulate varying performance
            for _ in range(20):
                metrics.record_delivery(True)
                metrics.record_ack(True)
            
            # Add some failures
            for _ in range(i):
                metrics.record_delivery(False)
                metrics.record_ack(False)
            
            metrics.record_latency(50.0 + i * 20)
        
        # Update baseline
        node_client._update_network_baseline()
        
        baseline = node_client._network_baseline
        assert baseline is not None
        assert baseline.sample_count >= 3
        assert baseline.avg_delivery_rate > 0.8
    
    def test_detection_requires_minimum_samples(self, node_client):
        """Test that detection only triggers after minimum samples."""
        router_id = "test_router_" + "d" * 52
        
        # Record failures but below minimum threshold
        for i in range(3):
            node_client._record_delivery_outcome(router_id, f"msg_{i}", False)
        
        # Should not be flagged (not enough samples)
        metrics = node_client._router_behavior_metrics[router_id]
        assert not metrics.flagged
        assert not node_client.is_router_flagged(router_id)


# =============================================================================
# ANOMALY DETECTION TESTS
# =============================================================================


class TestAnomalyDetection:
    """Tests for anomaly detection and router flagging."""
    
    def test_flag_router_for_low_delivery_rate(self, node_client):
        """Test flagging router with low delivery rate."""
        router_id = "bad_router_" + "e" * 52
        
        # Set up baseline with good routers
        for i in range(3):
            good_router = f"good_router_{i}" + "g" * 49
            metrics = node_client._get_router_metrics(good_router)
            for _ in range(20):
                metrics.record_delivery(True)
            metrics.record_latency(100.0)
        
        node_client._update_network_baseline()
        
        # Simulate bad router with many dropped messages
        for i in range(10):
            delivered = i < 3  # Only 3 out of 10 delivered (30%)
            node_client._record_delivery_outcome(router_id, f"msg_{i}", delivered)
        
        # Check if flagged
        assert node_client.is_router_flagged(router_id)
        
        flagged = node_client.get_flagged_routers()
        assert router_id in flagged
    
    def test_flag_router_for_high_ack_failure(self, node_client):
        """Test flagging router with high ACK failure rate."""
        router_id = "unreliable_" + "f" * 54
        
        # Set threshold lower for testing
        node_client.ack_failure_threshold = 0.20
        
        # Record ACK outcomes
        for i in range(20):
            success = i < 10  # 50% failure rate
            node_client._record_ack_outcome(router_id, success)
        
        # Also need some delivery data to trigger check
        metrics = node_client._get_router_metrics(router_id)
        for _ in range(10):
            metrics.record_delivery(True)
        
        # Force check
        node_client._check_router_behavior(router_id)
        
        assert node_client.is_router_flagged(router_id)
    
    def test_clear_router_flag(self, node_client):
        """Test clearing a router's misbehavior flag."""
        router_id = "test_router_" + "h" * 52
        
        # Flag the router manually
        metrics = node_client._get_router_metrics(router_id)
        metrics.flagged = True
        metrics.flag_reason = "test"
        node_client._flagged_routers[router_id] = MisbehaviorReport(
            reporter_id=node_client.node_id,
            router_id=router_id,
            misbehavior_type="test",
            severity=0.5,
        )
        
        assert node_client.is_router_flagged(router_id)
        
        # Clear the flag
        result = node_client.clear_router_flag(router_id)
        
        assert result is True
        assert not node_client.is_router_flagged(router_id)
        
        # Clear non-existent flag
        result = node_client.clear_router_flag("nonexistent" * 6)
        assert result is False


# =============================================================================
# ROUTER SELECTION WITH FLAGGING TESTS
# =============================================================================


class TestRouterSelectionWithFlagging:
    """Tests for router selection penalizing flagged routers."""
    
    def test_flagged_router_penalty(self, node_client):
        """Test that flagged routers are penalized in selection."""
        # Create two mock connections
        good_router_id = "good_router_" + "a" * 52
        bad_router_id = "bad_router_" + "b" * 53
        
        good_router = RouterInfo(
            router_id=good_router_id,
            endpoints=["192.168.1.1:8471"],
            capacity={"max_connections": 100, "current_load_pct": 30},
            health={},
            regions=[],
            features=[],
        )
        
        bad_router = RouterInfo(
            router_id=bad_router_id,
            endpoints=["192.168.2.1:8471"],
            capacity={"max_connections": 100, "current_load_pct": 30},
            health={},
            regions=[],
            features=[],
        )
        
        # Create mock connections
        good_conn = MagicMock()
        good_conn.router = good_router
        good_conn.websocket = MagicMock()
        good_conn.websocket.closed = False
        good_conn.ping_latency_ms = 50
        good_conn.ack_success_rate = 0.95
        good_conn.health_score = 0.9
        good_conn.is_under_back_pressure = False
        
        bad_conn = MagicMock()
        bad_conn.router = bad_router
        bad_conn.websocket = MagicMock()
        bad_conn.websocket.closed = False
        bad_conn.ping_latency_ms = 50
        bad_conn.ack_success_rate = 0.95
        bad_conn.health_score = 0.9
        bad_conn.is_under_back_pressure = False
        
        node_client.connections = {
            good_router_id: good_conn,
            bad_router_id: bad_conn,
        }
        
        # Flag the bad router
        node_client._flagged_routers[bad_router_id] = MisbehaviorReport(
            reporter_id=node_client.node_id,
            router_id=bad_router_id,
            misbehavior_type="test",
            severity=0.7,
        )
        
        # Select routers many times
        selections = {"good": 0, "bad": 0}
        for _ in range(100):
            selected = node_client._select_router()
            if selected.router_id == good_router_id:
                selections["good"] += 1
            else:
                selections["bad"] += 1
        
        # Good router should be selected much more often
        assert selections["good"] > selections["bad"] * 5


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestMisbehaviorStatistics:
    """Tests for misbehavior detection statistics."""
    
    def test_get_misbehavior_detection_stats(self, node_client):
        """Test getting detection statistics."""
        # Add some router metrics
        router_id = "test_router_" + "z" * 52
        node_client._get_router_metrics(router_id)
        
        stats = node_client.get_misbehavior_detection_stats()
        
        assert stats["enabled"] is True
        assert stats["routers_tracked"] >= 1
        assert "thresholds" in stats
        assert "delivery_rate" in stats["thresholds"]
    
    def test_get_router_behavior_metrics(self, node_client):
        """Test getting metrics for a specific router."""
        router_id = "test_router_" + "y" * 52
        
        # Record some activity
        node_client._record_delivery_outcome(router_id, "msg_001", True, 150.0)
        
        metrics = node_client.get_router_behavior_metrics(router_id)
        
        assert metrics is not None
        assert metrics["router_id"] == router_id
        assert metrics["avg_latency_ms"] == 150.0
        
        # Non-existent router
        assert node_client.get_router_behavior_metrics("nonexistent") is None
    
    def test_get_all_router_metrics(self, node_client):
        """Test getting metrics for all routers."""
        # Add metrics for multiple routers
        for i in range(3):
            router_id = f"router_{i}" + "w" * 55
            node_client._record_ack_outcome(router_id, True)
        
        all_metrics = node_client.get_all_router_metrics()
        
        assert len(all_metrics) == 3


# =============================================================================
# SEED MISBEHAVIOR HANDLING TESTS
# =============================================================================


class TestSeedMisbehaviorHandling:
    """Tests for seed node misbehavior report handling."""
    
    @pytest.fixture
    def seed_node(self):
        """Create a seed node for testing."""
        from valence.network.seed import SeedNode, SeedConfig
        
        config = SeedConfig(
            misbehavior_reports_enabled=True,
            misbehavior_min_reports_to_flag=2,  # Low for testing
            misbehavior_verify_reporter_signature=False,  # Skip for testing
        )
        return SeedNode(config=config)
    
    @pytest.mark.asyncio
    async def test_handle_misbehavior_report(self, seed_node):
        """Test handling a misbehavior report."""
        from aiohttp.test_utils import make_mocked_request
        
        # Mock request with report data
        report_data = {
            "report_id": "report_001",
            "reporter_id": "node_" + "a" * 60,
            "router_id": "bad_router_" + "b" * 52,
            "misbehavior_type": "message_drop",
            "severity": 0.7,
            "timestamp": time.time(),
        }
        
        request = MagicMock()
        request.json = AsyncMock(return_value=report_data)
        
        response = await seed_node.handle_misbehavior_report(request)
        
        # Check response
        assert response.status == 200
        response_data = response.body.decode() if hasattr(response, 'body') else "{}"
        
        # Check report was stored
        router_id = report_data["router_id"]
        assert router_id in seed_node._misbehavior_reports
    
    @pytest.mark.asyncio
    async def test_flag_router_after_multiple_reports(self, seed_node):
        """Test flagging router after receiving reports from multiple nodes."""
        router_id = "bad_router_" + "c" * 52
        
        # Submit reports from different reporters
        for i in range(3):
            report_data = {
                "report_id": f"report_{i}",
                "reporter_id": f"node_{i}" + "d" * 58,
                "router_id": router_id,
                "misbehavior_type": "message_drop",
                "severity": 0.6,
                "timestamp": time.time(),
            }
            
            request = MagicMock()
            request.json = AsyncMock(return_value=report_data)
            
            await seed_node.handle_misbehavior_report(request)
        
        # Router should be flagged after min_reports_to_flag reports
        assert seed_node.is_router_misbehavior_flagged(router_id)
    
    def test_get_misbehavior_stats(self, seed_node):
        """Test getting misbehavior statistics from seed."""
        # Flag a router manually
        router_id = "flagged_router_" + "e" * 49
        seed_node._misbehavior_flagged_routers[router_id] = time.time()
        
        stats = seed_node.get_misbehavior_stats()
        
        assert stats["enabled"] is True
        assert stats["flagged_routers"] == 1
        assert router_id in stats["flagged_router_ids"]
    
    def test_clear_router_misbehavior_flag(self, seed_node):
        """Test clearing misbehavior flag on seed."""
        router_id = "flagged_router_" + "f" * 49
        seed_node._misbehavior_flagged_routers[router_id] = time.time()
        
        assert seed_node.is_router_misbehavior_flagged(router_id)
        
        result = seed_node.clear_router_misbehavior_flag(router_id)
        
        assert result is True
        assert not seed_node.is_router_misbehavior_flagged(router_id)


# =============================================================================
# MISBEHAVIOR REPORT SUBMISSION TESTS
# =============================================================================


class TestMisbehaviorReportSubmission:
    """Tests for submitting reports to seeds."""
    
    @pytest.mark.asyncio
    async def test_report_misbehavior_to_seeds(self, node_client):
        """Test reporting misbehavior to seed nodes."""
        # Create a report
        report = MisbehaviorReport(
            reporter_id=node_client.node_id,
            router_id="bad_router_" + "g" * 52,
            misbehavior_type=MisbehaviorType.MESSAGE_DROP,
            severity=0.7,
        )
        
        # Mock the discovery client
        node_client.discovery.report_misbehavior = AsyncMock(return_value=True)
        
        # Submit the report
        result = await node_client._report_misbehavior_to_seeds(report)
        
        assert result is True
        node_client.discovery.report_misbehavior.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_report_cooldown(self, node_client):
        """Test that report cooldown prevents rapid re-reports."""
        router_id = "bad_router_" + "h" * 52
        
        # Create reports
        report1 = MisbehaviorReport(
            reporter_id=node_client.node_id,
            router_id=router_id,
            misbehavior_type=MisbehaviorType.ACK_FAILURE,
            severity=0.5,
        )
        
        report2 = MisbehaviorReport(
            reporter_id=node_client.node_id,
            router_id=router_id,
            misbehavior_type=MisbehaviorType.ACK_FAILURE,
            severity=0.6,
        )
        
        # Mock discovery
        node_client.discovery.report_misbehavior = AsyncMock(return_value=True)
        
        # First report should succeed
        result1 = await node_client._report_misbehavior_to_seeds(report1)
        assert result1 is True
        
        # Second report should be blocked by cooldown
        result2 = await node_client._report_misbehavior_to_seeds(report2)
        assert result2 is False
        
        # Verify only one call was made
        assert node_client.discovery.report_misbehavior.call_count == 1
