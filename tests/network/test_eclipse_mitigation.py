"""
Tests for Eclipse Attack Mitigations (Issue #118).

These tests verify:
- Router diversity enforcement (different IPs, subnets, ASNs)
- Periodic router rotation to prevent long-term eclipse
- Anomaly detection for coordinated router failures
- Out-of-band seed verification
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from valence.network.node import (
    NodeClient,
    RouterConnection,
    create_node_client,
)
from valence.network.discovery import RouterInfo, DiscoveryClient

# Note: Some tests need update to use component methods
# Methods moved: _check_ip_diversity -> connection_manager.check_ip_diversity, etc.


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ed25519_keypair():
    """Generate an Ed25519 keypair for testing."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def x25519_keypair():
    """Generate an X25519 keypair for testing."""
    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.closed = False
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive = AsyncMock()
    return ws


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def node_client(ed25519_keypair, x25519_keypair):
    """Create a NodeClient with eclipse mitigation settings for testing."""
    private_key, public_key = ed25519_keypair
    enc_private, _ = x25519_keypair
    
    client = NodeClient(
        node_id=public_key.public_bytes_raw().hex(),
        private_key=private_key,
        encryption_private_key=enc_private,
        min_connections=2,
        target_connections=3,
        max_connections=5,
    )
    
    # Configure eclipse mitigation settings
    client.min_diverse_subnets = 2
    client.min_diverse_asns = 2
    client.asn_diversity_enabled = True
    client.rotation_enabled = True
    client.rotation_interval = 3600.0
    client.rotation_max_age = 7200.0
    client.anomaly_detection_enabled = True
    client.anomaly_window = 60.0
    client.anomaly_threshold = 3
    
    return client


def create_mock_router(
    router_id: str,
    ip: str,
    asn: str = None,
    port: int = 8471,
) -> RouterInfo:
    """Create a mock RouterInfo with specified network properties."""
    capacity = {"max_connections": 100, "current_load_pct": 25}
    if asn:
        capacity["asn"] = asn
    
    return RouterInfo(
        router_id=router_id,
        endpoints=[f"{ip}:{port}"],
        capacity=capacity,
        health={"uptime_pct": 99.9, "avg_latency_ms": 50},
        regions=["us-west"],
        features=["relay-v1"],
        asn=asn,
    )


# =============================================================================
# Tests - IP/Subnet Diversity
# =============================================================================


class TestSubnetDiversity:
    """Tests for IP subnet diversity enforcement."""

    def test_check_ip_diversity_different_subnets(self, node_client):
        """Test that routers from different /16 subnets pass diversity check."""
        # Connect to first router
        router1 = create_mock_router("a" * 64, "192.168.1.1")
        node_client._connected_subnets.add("192.168.0.0/16")
        
        # Router from different /16 should pass
        router2 = create_mock_router("b" * 64, "10.0.1.1")
        assert node_client._check_ip_diversity(router2) is True

    def test_check_ip_diversity_same_subnet_rejected(self, node_client):
        """Test that routers from the same /16 subnet are rejected."""
        # Connect to first router
        node_client._connected_subnets.add("192.168.0.0/16")
        
        # Router from same /16 should fail
        router2 = create_mock_router("b" * 64, "192.168.50.100")
        assert node_client._check_ip_diversity(router2) is False
        assert node_client._stats["diversity_rejections"] == 1

    def test_check_ip_diversity_hostname_allowed(self, node_client):
        """Test that hostnames bypass IP diversity check."""
        node_client._connected_subnets.add("192.168.0.0/16")
        
        # Router with hostname should pass (can't determine subnet)
        router = create_mock_router("b" * 64, "router.example.com")
        router.endpoints = ["router.example.com:8471"]
        assert node_client._check_ip_diversity(router) is True

    def test_check_diversity_requirements_met(self, node_client):
        """Test diversity requirements check when met."""
        node_client._connected_subnets.add("192.168.0.0/16")
        node_client._connected_subnets.add("10.0.0.0/16")
        node_client._connected_asns.add("AS12345")
        node_client._connected_asns.add("AS67890")
        
        # Add mock connections
        router1 = create_mock_router("a" * 64, "192.168.1.1", "AS12345")
        router2 = create_mock_router("b" * 64, "10.0.1.1", "AS67890")
        
        ws1, ws2 = AsyncMock(), AsyncMock()
        ws1.closed = ws2.closed = False
        session = AsyncMock()
        
        node_client.connections = {
            router1.router_id: RouterConnection(
                router=router1, websocket=ws1, session=session,
                connected_at=time.time(), last_seen=time.time(),
            ),
            router2.router_id: RouterConnection(
                router=router2, websocket=ws2, session=session,
                connected_at=time.time(), last_seen=time.time(),
            ),
        }
        
        assert node_client._check_diversity_requirements() is True

    def test_check_diversity_requirements_not_met(self, node_client):
        """Test diversity requirements check when not met."""
        node_client._connected_subnets.add("192.168.0.0/16")  # Only one subnet
        
        # Add mock connections (need 2 connections but same subnet to fail diversity)
        router1 = create_mock_router("a" * 64, "192.168.1.1")
        router2 = create_mock_router("b" * 64, "192.168.2.2")
        ws1, ws2 = AsyncMock(), AsyncMock()
        ws1.closed = ws2.closed = False
        session = AsyncMock()
        
        node_client.connections = {
            router1.router_id: RouterConnection(
                router=router1, websocket=ws1, session=session,
                connected_at=time.time(), last_seen=time.time(),
            ),
            router2.router_id: RouterConnection(
                router=router2, websocket=ws2, session=session,
                connected_at=time.time(), last_seen=time.time(),
            ),
        }
        
        # We have 2 connections but only 1 subnet (min_diverse_subnets=2)
        # So len(subnets)=1 < min(min_diverse_subnets=2, len(connections)=2) = 2
        assert node_client._check_diversity_requirements() is False


# =============================================================================
# Tests - ASN Diversity
# =============================================================================


class TestASNDiversity:
    """Tests for ASN (Autonomous System Number) diversity enforcement."""

    def test_check_asn_diversity_different_asns(self, node_client):
        """Test that routers from different ASNs pass diversity check."""
        node_client._connected_asns.add("AS12345")
        
        router = create_mock_router("b" * 64, "10.0.1.1", "AS67890")
        assert node_client._check_asn_diversity(router) is True

    def test_check_asn_diversity_same_asn_with_enough_diversity(self, node_client):
        """Test that same ASN is rejected when we have enough diversity."""
        node_client._connected_asns.add("AS12345")
        node_client._connected_asns.add("AS67890")
        node_client.min_diverse_asns = 2
        
        # We already have 2 ASNs (meeting requirement), so same ASN should be rejected
        router = create_mock_router("c" * 64, "172.16.1.1", "AS12345")
        assert node_client._check_asn_diversity(router) is False

    def test_check_asn_diversity_same_asn_with_low_diversity(self, node_client):
        """Test that same ASN is allowed when we need more diversity."""
        node_client._connected_asns.add("AS12345")
        node_client.min_diverse_asns = 3
        
        # We only have 1 ASN, need 3, so allow same ASN for now
        router = create_mock_router("c" * 64, "172.16.1.1", "AS12345")
        assert node_client._check_asn_diversity(router) is True

    def test_check_asn_diversity_no_asn_info(self, node_client):
        """Test that routers without ASN info are allowed."""
        node_client._connected_asns.add("AS12345")
        
        # Router without ASN should pass (can't determine)
        router = create_mock_router("b" * 64, "10.0.1.1", asn=None)
        assert node_client._check_asn_diversity(router) is True

    def test_check_asn_diversity_disabled(self, node_client):
        """Test that ASN diversity check can be disabled."""
        node_client.asn_diversity_enabled = False
        node_client._connected_asns.add("AS12345")
        node_client._connected_asns.add("AS67890")
        
        # Same ASN should pass when diversity check is disabled
        router = create_mock_router("c" * 64, "172.16.1.1", "AS12345")
        assert node_client._check_asn_diversity(router) is True


# =============================================================================
# Tests - Anomaly Detection
# =============================================================================


class TestAnomalyDetection:
    """Tests for eclipse attack anomaly detection."""

    def test_record_failure_event(self, node_client):
        """Test recording failure events."""
        node_client._record_failure_event("router1", "connection", "ETIMEDOUT")
        
        assert len(node_client._failure_events) == 1
        assert node_client._failure_events[0]["router_id"] == "router1"
        assert node_client._failure_events[0]["failure_type"] == "connection"
        assert node_client._failure_events[0]["error_code"] == "ETIMEDOUT"

    def test_detect_correlated_failures(self, node_client):
        """Test detection of correlated failures (eclipse indicator)."""
        # Simulate multiple routers failing with same error
        for i in range(3):
            node_client._record_failure_event(
                f"router{i}",
                "connection",
                "ETIMEDOUT"
            )
        
        assert len(node_client._anomaly_alerts) == 1
        assert node_client._anomaly_alerts[0]["type"] == "correlated_failures"
        assert node_client._anomaly_alerts[0]["count"] >= 3
        assert node_client._stats["anomalies_detected"] == 1

    def test_detect_same_error_code(self, node_client):
        """Test detection of same error code across routers."""
        # Simulate multiple routers with same error code but different failure types
        # to ensure the error_code detection triggers instead of failure_type
        for i in range(3):
            node_client._record_failure_event(
                f"router{i}",
                f"error_type_{i}",  # Different failure types
                "MALICIOUS_RESPONSE"  # Same error code
            )
        
        # Check that an anomaly was detected
        assert len(node_client._anomaly_alerts) == 1
        # The anomaly could be "correlated_failures" or "same_error"
        anomaly = node_client._anomaly_alerts[0]
        assert anomaly["count"] >= 3
        # When failure types differ but error code is same, "same_error" is detected
        assert anomaly["type"] == "same_error"
        assert anomaly["error_code"] == "MALICIOUS_RESPONSE"

    def test_no_anomaly_below_threshold(self, node_client):
        """Test that no anomaly is detected below threshold."""
        node_client.anomaly_threshold = 5
        
        # Only 3 failures - below threshold
        for i in range(3):
            node_client._record_failure_event(f"router{i}", "connection", None)
        
        assert len(node_client._anomaly_alerts) == 0

    def test_failure_events_pruned_by_window(self, node_client):
        """Test that old failure events are pruned."""
        node_client.anomaly_window = 60.0
        
        # Record event with old timestamp
        node_client._failure_events.append({
            "router_id": "old_router",
            "failure_type": "connection",
            "error_code": None,
            "timestamp": time.time() - 120,  # 2 minutes ago
        })
        
        # Record new event (triggers pruning)
        node_client._record_failure_event("new_router", "connection", None)
        
        # Old event should be pruned
        assert len(node_client._failure_events) == 1
        assert node_client._failure_events[0]["router_id"] == "new_router"

    def test_anomaly_detection_disabled(self, node_client):
        """Test that anomaly detection can be disabled."""
        node_client.anomaly_detection_enabled = False
        
        # Record failures
        for i in range(5):
            node_client._record_failure_event(f"router{i}", "connection", "ERROR")
        
        # No events should be recorded
        assert len(node_client._failure_events) == 0
        assert len(node_client._anomaly_alerts) == 0

    def test_clear_anomaly_alerts(self, node_client):
        """Test clearing anomaly alerts."""
        # Create some alerts
        for i in range(3):
            node_client._record_failure_event(f"router{i}", "connection", None)
        
        assert len(node_client._anomaly_alerts) > 0
        
        count = node_client.clear_anomaly_alerts()
        assert count > 0
        assert len(node_client._anomaly_alerts) == 0


# =============================================================================
# Tests - Router Rotation
# =============================================================================


class TestRouterRotation:
    """Tests for periodic router rotation."""

    def test_connection_timestamp_tracking(self, node_client, mock_websocket, mock_session):
        """Test that connection timestamps are tracked."""
        router = create_mock_router("a" * 64, "192.168.1.1")
        
        now = time.time()
        conn = RouterConnection(
            router=router,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=now,
            last_seen=now,
        )
        
        node_client.connections[router.router_id] = conn
        node_client._connection_timestamps[router.router_id] = now
        
        assert router.router_id in node_client._connection_timestamps
        assert node_client._connection_timestamps[router.router_id] == now

    @pytest.mark.asyncio
    async def test_rotate_router_success(self, node_client, mock_websocket, mock_session):
        """Test successful router rotation."""
        old_router = create_mock_router("a" * 64, "192.168.1.1", "AS12345")
        new_router = create_mock_router("b" * 64, "10.0.1.1", "AS67890")
        
        conn = RouterConnection(
            router=old_router,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time() - 7200,  # 2 hours ago
            last_seen=time.time(),
        )
        
        node_client.connections[old_router.router_id] = conn
        node_client._connection_timestamps[old_router.router_id] = time.time() - 7200
        node_client._add_subnet(old_router)
        
        # Mock discovery and connection
        node_client.discovery = AsyncMock()
        node_client.discovery.discover_routers = AsyncMock(return_value=[new_router])
        
        # Mock _connect_to_router
        async def mock_connect(router):
            node_client.connections[router.router_id] = RouterConnection(
                router=router,
                websocket=AsyncMock(),
                session=AsyncMock(),
                connected_at=time.time(),
                last_seen=time.time(),
            )
        
        node_client._connect_to_router = mock_connect
        
        # Perform rotation
        result = await node_client._rotate_router(old_router.router_id, "test")
        
        assert result is True
        assert node_client._stats["routers_rotated"] == 1
        assert old_router.router_id not in node_client.connections

    def test_get_eclipse_mitigation_status(self, node_client, mock_websocket, mock_session):
        """Test eclipse mitigation status reporting."""
        router = create_mock_router("a" * 64, "192.168.1.1", "AS12345")
        
        conn = RouterConnection(
            router=router,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time() - 3600,
            last_seen=time.time(),
        )
        
        node_client.connections[router.router_id] = conn
        node_client._connection_timestamps[router.router_id] = time.time() - 3600
        node_client._connected_subnets.add("192.168.0.0/16")
        node_client._connected_asns.add("AS12345")
        
        status = node_client.get_eclipse_mitigation_status()
        
        assert "diversity" in status
        assert "rotation" in status
        assert "anomaly_detection" in status
        assert "oob_verification" in status
        assert status["diversity"]["subnets_connected"] == 1
        assert status["diversity"]["asns_connected"] == 1
        assert status["rotation"]["enabled"] is True


# =============================================================================
# Tests - Discovery Client Diversity Selection
# =============================================================================


class TestDiscoveryDiversitySelection:
    """Tests for DiscoveryClient diverse router selection."""

    def test_select_diverse_routers_by_subnet(self):
        """Test selecting routers from different subnets."""
        client = DiscoveryClient()
        
        routers = [
            create_mock_router("a" * 64, "192.168.1.1"),
            create_mock_router("b" * 64, "192.168.2.1"),  # Same /16
            create_mock_router("c" * 64, "10.0.1.1"),  # Different /16
            create_mock_router("d" * 64, "172.16.1.1"),  # Different /16
        ]
        
        selected = client.select_diverse_routers(routers, count=3)
        
        # Should select routers from different /16 subnets
        assert len(selected) == 3
        
        # Get unique subnets
        subnets = set()
        for r in selected:
            subnet = client._get_router_subnet(r)
            if subnet:
                subnets.add(subnet)
        
        assert len(subnets) >= 2  # At least 2 different subnets

    def test_select_diverse_routers_by_asn(self):
        """Test selecting routers from different ASNs."""
        client = DiscoveryClient()
        
        routers = [
            create_mock_router("a" * 64, "192.168.1.1", "AS12345"),
            create_mock_router("b" * 64, "10.0.1.1", "AS12345"),  # Same ASN
            create_mock_router("c" * 64, "172.16.1.1", "AS67890"),  # Different ASN
            create_mock_router("d" * 64, "203.0.113.1", "AS11111"),  # Different ASN
        ]
        
        selected = client.select_diverse_routers(routers, count=3)
        
        assert len(selected) == 3
        
        # Get unique ASNs
        asns = set()
        for r in selected:
            if r.asn:
                asns.add(r.asn)
        
        assert len(asns) >= 2  # At least 2 different ASNs

    def test_select_diverse_routers_with_exclusions(self):
        """Test selecting routers while excluding certain subnets/ASNs."""
        client = DiscoveryClient()
        
        routers = [
            create_mock_router("a" * 64, "192.168.1.1", "AS12345"),
            create_mock_router("b" * 64, "10.0.1.1", "AS67890"),
            create_mock_router("c" * 64, "172.16.1.1", "AS11111"),
        ]
        
        # Exclude the first router's subnet and ASN
        selected = client.select_diverse_routers(
            routers,
            count=2,
            exclude_subnets={"192.168.0.0/16"},
            exclude_asns={"AS12345"},
        )
        
        assert len(selected) == 2
        
        # First router should not be selected
        selected_ids = {r.router_id for r in selected}
        assert routers[0].router_id not in selected_ids

    def test_select_diverse_routers_insufficient_candidates(self):
        """Test behavior when not enough diverse candidates exist."""
        client = DiscoveryClient()
        
        # All routers from same subnet and ASN
        routers = [
            create_mock_router("a" * 64, "192.168.1.1", "AS12345"),
            create_mock_router("b" * 64, "192.168.1.2", "AS12345"),
        ]
        
        # Request 3 but only 2 available (both from same network)
        selected = client.select_diverse_routers(routers, count=3)
        
        # Should return all available even if not diverse
        assert len(selected) == 2


# =============================================================================
# Tests - OOB Verification (Mock)
# =============================================================================


class TestOOBVerification:
    """Tests for out-of-band verification."""

    @pytest.mark.asyncio
    async def test_oob_verification_success(self, node_client, mock_websocket, mock_session):
        """Test successful OOB verification."""
        node_client.oob_verification_enabled = True
        node_client.oob_verification_url = "https://verify.example.com/check"
        
        router = create_mock_router("a" * 64, "192.168.1.1")
        conn = RouterConnection(
            router=router,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[router.router_id] = conn
        
        # Mock the HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "verified": [router.router_id],
            "unknown": [],
            "suspicious": [],
        })
        
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_cm.__aexit__ = AsyncMock()
        
        mock_session_cm = AsyncMock()
        mock_session_cm.post = MagicMock(return_value=mock_cm)
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session_cm)
        mock_session_cm.__aexit__ = AsyncMock()
        
        with patch("aiohttp.ClientSession", return_value=mock_session_cm):
            result = await node_client._perform_oob_verification()
        
        assert result is True
        assert node_client._stats["oob_verifications"] == 1
        assert router.router_id in node_client._oob_verified_routers

    def test_oob_verification_disabled_by_default(self, node_client):
        """Test that OOB verification is disabled by default."""
        assert node_client.oob_verification_enabled is False
        assert node_client.oob_verification_url is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestEclipseMitigationIntegration:
    """Integration tests for eclipse mitigation features."""

    def test_get_stats_includes_eclipse_metrics(self, node_client):
        """Test that get_stats() includes eclipse mitigation metrics."""
        stats = node_client.get_stats()
        
        assert "connected_asns" in stats
        assert "diversity_met" in stats
        assert "recent_anomalies" in stats
        assert "oob_verified_routers" in stats
        assert "routers_rotated" in stats
        assert "diversity_rejections" in stats
        assert "anomalies_detected" in stats

    def test_subnet_and_asn_tracking(self, node_client):
        """Test that subnet and ASN are tracked together."""
        router = create_mock_router("a" * 64, "192.168.1.1", "AS12345")
        
        node_client._add_subnet(router)
        
        assert "192.168.0.0/16" in node_client._connected_subnets
        assert "AS12345" in node_client._connected_asns
        
        node_client._remove_subnet(router)
        
        assert "192.168.0.0/16" not in node_client._connected_subnets
        assert "AS12345" not in node_client._connected_asns
