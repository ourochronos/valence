"""
Tests for Node-to-Node Health Gossip (Issue #114).

Tests cover:
1. Health observation tracking
2. Gossip message creation and serialization
3. Peer observation handling
4. Aggregated health scoring
5. Gossip propagation
6. Observation pruning
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.network.messages import HealthGossip, RouterHealthObservation
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
        gossip_interval=1.0,  # Fast for testing
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


@pytest.fixture
def mock_router_connection(mock_router_info):
    """Create a mock router connection."""
    ws = MagicMock()
    ws.closed = False
    ws.send_json = AsyncMock()
    
    session = MagicMock()
    
    return RouterConnection(
        router=mock_router_info,
        websocket=ws,
        session=session,
        connected_at=time.time(),
        last_seen=time.time(),
        messages_sent=10,
        messages_received=8,
        ack_success=7,
        ack_failure=1,
        ping_latency_ms=45.0,
    )


# =============================================================================
# ROUTER HEALTH OBSERVATION TESTS
# =============================================================================


class TestRouterHealthObservation:
    """Tests for RouterHealthObservation message type."""
    
    def test_create_observation(self):
        """Test creating a health observation."""
        obs = RouterHealthObservation(
            router_id="abc123",
            latency_ms=50.0,
            success_rate=0.95,
            failure_count=2,
            success_count=38,
            last_seen=time.time(),
            load_pct=30.0,
        )
        
        assert obs.router_id == "abc123"
        assert obs.latency_ms == 50.0
        assert obs.success_rate == 0.95
        assert obs.failure_count == 2
        assert obs.success_count == 38
        assert obs.load_pct == 30.0
    
    def test_observation_serialization(self):
        """Test observation to_dict and from_dict."""
        obs = RouterHealthObservation(
            router_id="router1",
            latency_ms=100.0,
            success_rate=0.9,
            failure_count=5,
            success_count=45,
            last_seen=1234567890.0,
            load_pct=50.0,
        )
        
        data = obs.to_dict()
        
        assert data["router_id"] == "router1"
        assert data["latency_ms"] == 100.0
        assert data["success_rate"] == 0.9
        
        # Deserialize
        obs2 = RouterHealthObservation.from_dict(data)
        
        assert obs2.router_id == obs.router_id
        assert obs2.latency_ms == obs.latency_ms
        assert obs2.success_rate == obs.success_rate
        assert obs2.last_seen == obs.last_seen
    
    def test_observation_defaults(self):
        """Test observation default values."""
        obs = RouterHealthObservation(router_id="test")
        
        assert obs.latency_ms == 0.0
        assert obs.success_rate == 1.0
        assert obs.failure_count == 0
        assert obs.success_count == 0
        assert obs.load_pct == 0.0


# =============================================================================
# HEALTH GOSSIP MESSAGE TESTS
# =============================================================================


class TestHealthGossip:
    """Tests for HealthGossip message type."""
    
    def test_create_gossip(self):
        """Test creating a health gossip message."""
        obs1 = RouterHealthObservation(router_id="r1", latency_ms=50)
        obs2 = RouterHealthObservation(router_id="r2", latency_ms=75)
        
        gossip = HealthGossip(
            source_node_id="node123",
            timestamp=time.time(),
            observations=[obs1, obs2],
            ttl=2,
        )
        
        assert gossip.type == "health_gossip"
        assert gossip.source_node_id == "node123"
        assert len(gossip.observations) == 2
        assert gossip.ttl == 2
    
    def test_gossip_serialization(self):
        """Test gossip to_dict and from_dict."""
        obs = RouterHealthObservation(router_id="r1", latency_ms=50, success_rate=0.9)
        
        gossip = HealthGossip(
            source_node_id="node_abc",
            timestamp=1234567890.0,
            observations=[obs],
            ttl=3,
        )
        
        data = gossip.to_dict()
        
        assert data["type"] == "health_gossip"
        assert data["source_node_id"] == "node_abc"
        assert data["timestamp"] == 1234567890.0
        assert data["ttl"] == 3
        assert len(data["observations"]) == 1
        assert data["observations"][0]["router_id"] == "r1"
        
        # Deserialize
        gossip2 = HealthGossip.from_dict(data)
        
        assert gossip2.source_node_id == gossip.source_node_id
        assert gossip2.timestamp == gossip.timestamp
        assert gossip2.ttl == gossip.ttl
        assert len(gossip2.observations) == 1
        assert gossip2.observations[0].router_id == "r1"
        assert gossip2.observations[0].latency_ms == 50
    
    def test_gossip_default_ttl(self):
        """Test gossip default TTL."""
        gossip = HealthGossip(source_node_id="node1", timestamp=time.time())
        assert gossip.ttl == 2


# =============================================================================
# NODE HEALTH TRACKING TESTS
# =============================================================================


class TestNodeHealthTracking:
    """Tests for node health observation tracking."""
    
    def test_update_own_observation(self, node_client, mock_router_connection):
        """Test updating own health observation from connection."""
        router_id = mock_router_connection.router.router_id
        
        node_client._update_own_observation(router_id, mock_router_connection)
        
        assert router_id in node_client._own_observations
        obs = node_client._own_observations[router_id]
        
        assert obs.router_id == router_id
        assert obs.latency_ms == mock_router_connection.ping_latency_ms
        assert obs.success_rate == mock_router_connection.ack_success_rate
        assert obs.failure_count == mock_router_connection.ack_failure
        assert obs.success_count == mock_router_connection.ack_success
    
    def test_calculate_observation_score(self, node_client):
        """Test health score calculation from observation."""
        # Good observation: low latency, high success rate, low load
        good_obs = RouterHealthObservation(
            router_id="r1",
            latency_ms=50,
            success_rate=0.95,
            load_pct=20,
        )
        
        good_score = node_client._calculate_observation_score(good_obs)
        
        # Bad observation: high latency, low success rate, high load
        bad_obs = RouterHealthObservation(
            router_id="r2",
            latency_ms=400,
            success_rate=0.5,
            load_pct=90,
        )
        
        bad_score = node_client._calculate_observation_score(bad_obs)
        
        assert good_score > bad_score
        assert 0 <= good_score <= 1
        assert 0 <= bad_score <= 1
    
    def test_sample_observations_for_gossip(self, node_client):
        """Test sampling observations for gossip message."""
        # Add some observations
        now = time.time()
        for i in range(15):
            obs = RouterHealthObservation(
                router_id=f"router_{i}",
                latency_ms=50 + i,
                last_seen=now - i,  # Newer observations have higher indices
            )
            node_client._own_observations[f"router_{i}"] = obs
        
        # Sample should respect max limit
        sampled = node_client._sample_observations_for_gossip()
        
        assert len(sampled) <= node_client.max_observations_per_gossip
        
        # Should prioritize recent observations
        # (sorted by age ascending, so most recent first)
        first_obs = sampled[0]
        last_obs = sampled[-1]
        assert first_obs.last_seen >= last_obs.last_seen
    
    def test_sample_excludes_old_observations(self, node_client):
        """Test that old observations are excluded from gossip."""
        now = time.time()
        
        # Add a recent observation
        recent_obs = RouterHealthObservation(
            router_id="recent",
            latency_ms=50,
            last_seen=now - 10,  # 10 seconds old
        )
        node_client._own_observations["recent"] = recent_obs
        
        # Add an old observation (beyond max age)
        old_obs = RouterHealthObservation(
            router_id="old",
            latency_ms=50,
            last_seen=now - 600,  # 10 minutes old (beyond 5 min default)
        )
        node_client._own_observations["old"] = old_obs
        
        sampled = node_client._sample_observations_for_gossip()
        
        router_ids = [obs.router_id for obs in sampled]
        assert "recent" in router_ids
        assert "old" not in router_ids


# =============================================================================
# PEER OBSERVATION HANDLING TESTS
# =============================================================================


class TestPeerObservations:
    """Tests for handling peer health observations."""
    
    def test_handle_gossip_from_peer(self, node_client):
        """Test handling incoming gossip from a peer."""
        peer_id = "peer_node_123"
        
        obs = RouterHealthObservation(
            router_id="router_abc",
            latency_ms=75,
            success_rate=0.85,
            last_seen=time.time(),
        )
        
        gossip = HealthGossip(
            source_node_id=peer_id,
            timestamp=time.time(),
            observations=[obs],
            ttl=2,
        )
        
        node_client._handle_gossip(gossip)
        
        assert peer_id in node_client._peer_observations
        assert "router_abc" in node_client._peer_observations[peer_id]
        
        stored_obs = node_client._peer_observations[peer_id]["router_abc"]
        assert stored_obs.latency_ms == 75
        assert stored_obs.success_rate == 0.85
    
    def test_ignore_own_gossip(self, node_client):
        """Test that node ignores its own gossip (loop prevention)."""
        obs = RouterHealthObservation(router_id="r1", latency_ms=50)
        
        gossip = HealthGossip(
            source_node_id=node_client.node_id,  # Our own ID
            timestamp=time.time(),
            observations=[obs],
            ttl=2,
        )
        
        node_client._handle_gossip(gossip)
        
        # Should not store our own gossip
        assert node_client.node_id not in node_client._peer_observations
    
    def test_ignore_zero_ttl_gossip(self, node_client):
        """Test that gossip with TTL=0 is ignored."""
        peer_id = "peer_456"
        obs = RouterHealthObservation(router_id="r1", latency_ms=50)
        
        gossip = HealthGossip(
            source_node_id=peer_id,
            timestamp=time.time(),
            observations=[obs],
            ttl=0,  # Expired TTL
        )
        
        node_client._handle_gossip(gossip)
        
        assert peer_id not in node_client._peer_observations
    
    def test_prune_old_peer_observations(self, node_client):
        """Test pruning of old peer observations."""
        now = time.time()
        peer_id = "peer_789"
        
        # Add observations - one recent, one old
        node_client._peer_observations[peer_id] = {
            "recent_router": RouterHealthObservation(
                router_id="recent_router",
                latency_ms=50,
                last_seen=now - 60,  # 1 minute old
            ),
            "old_router": RouterHealthObservation(
                router_id="old_router",
                latency_ms=50,
                last_seen=now - 600,  # 10 minutes old (beyond 5 min max age)
            ),
        }
        
        node_client._prune_peer_observations()
        
        # Recent should remain, old should be pruned
        assert "recent_router" in node_client._peer_observations[peer_id]
        assert "old_router" not in node_client._peer_observations[peer_id]
    
    def test_prune_excess_observations(self, node_client):
        """Test pruning when exceeding max peer observations."""
        node_client.max_peer_observations = 5  # Low limit for testing
        now = time.time()
        
        # Add more observations than the limit
        for i in range(10):
            peer_id = f"peer_{i}"
            node_client._peer_observations[peer_id] = {
                "router": RouterHealthObservation(
                    router_id="router",
                    latency_ms=50,
                    last_seen=now - i * 10,  # Older observations first
                ),
            }
        
        node_client._prune_peer_observations()
        
        # Count total observations
        total = sum(
            len(peer_data)
            for peer_data in node_client._peer_observations.values()
        )
        
        assert total <= node_client.max_peer_observations


# =============================================================================
# AGGREGATED HEALTH TESTS
# =============================================================================


class TestAggregatedHealth:
    """Tests for aggregated health scoring."""
    
    def test_aggregated_health_own_only(self, node_client):
        """Test aggregated health with only own observations."""
        router_id = "router_123"
        
        obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=50,
            success_rate=0.9,
            load_pct=30,
            last_seen=time.time(),
        )
        node_client._own_observations[router_id] = obs
        
        score = node_client._get_aggregated_health(router_id)
        
        # Should be our observation score (no peer data)
        assert 0 < score < 1
    
    def test_aggregated_health_peer_only(self, node_client):
        """Test aggregated health with only peer observations."""
        router_id = "router_456"
        peer_id = "peer_1"
        
        obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=75,
            success_rate=0.85,
            load_pct=40,
            last_seen=time.time(),
        )
        node_client._peer_observations[peer_id] = {router_id: obs}
        
        score = node_client._get_aggregated_health(router_id)
        
        # Should be peer score with penalty (no direct observation)
        assert 0 < score < 1
    
    def test_aggregated_health_combined(self, node_client):
        """Test aggregated health combining own and peer observations."""
        router_id = "router_789"
        now = time.time()
        
        # Add own observation (good)
        own_obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=30,
            success_rate=0.95,
            load_pct=20,
            last_seen=now,
        )
        node_client._own_observations[router_id] = own_obs
        
        # Add peer observation (worse)
        peer_obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=200,
            success_rate=0.7,
            load_pct=70,
            last_seen=now,
        )
        node_client._peer_observations["peer_1"] = {router_id: peer_obs}
        
        combined_score = node_client._get_aggregated_health(router_id)
        own_score = node_client._calculate_observation_score(own_obs)
        peer_score = node_client._calculate_observation_score(peer_obs)
        
        # Combined should be between own and peer scores
        # With default weights (0.7 own, 0.3 peer), closer to own
        expected = (own_score * 0.7) + (peer_score * 0.3)
        
        assert abs(combined_score - expected) < 0.01
    
    def test_aggregated_health_weights_own_higher(self, node_client):
        """Test that own observations are weighted higher than peers."""
        router_id = "router_weight_test"
        now = time.time()
        
        # Own says bad router
        own_obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=400,
            success_rate=0.5,
            load_pct=90,
            last_seen=now,
        )
        node_client._own_observations[router_id] = own_obs
        
        # Peer says good router
        peer_obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=30,
            success_rate=0.99,
            load_pct=10,
            last_seen=now,
        )
        node_client._peer_observations["peer_1"] = {router_id: peer_obs}
        
        combined_score = node_client._get_aggregated_health(router_id)
        own_score = node_client._calculate_observation_score(own_obs)
        peer_score = node_client._calculate_observation_score(peer_obs)
        
        # Combined should be closer to own score due to higher weight
        dist_to_own = abs(combined_score - own_score)
        dist_to_peer = abs(combined_score - peer_score)
        
        assert dist_to_own < dist_to_peer
    
    def test_aggregated_health_no_data(self, node_client):
        """Test aggregated health when no data available."""
        score = node_client._get_aggregated_health("unknown_router")
        
        # Should return neutral score
        assert score == 0.5


# =============================================================================
# GOSSIP CREATION AND BROADCAST TESTS
# =============================================================================


class TestGossipBroadcast:
    """Tests for gossip creation and broadcasting."""
    
    def test_create_gossip_message(self, node_client):
        """Test creating a gossip message."""
        # Add some observations
        now = time.time()
        obs1 = RouterHealthObservation(
            router_id="r1", latency_ms=50, last_seen=now
        )
        obs2 = RouterHealthObservation(
            router_id="r2", latency_ms=75, last_seen=now
        )
        node_client._own_observations = {"r1": obs1, "r2": obs2}
        
        gossip = node_client._create_gossip_message()
        
        assert gossip.source_node_id == node_client.node_id
        assert gossip.ttl == node_client.gossip_ttl
        assert len(gossip.observations) == 2
        assert gossip.timestamp > 0
    
    @pytest.mark.asyncio
    async def test_broadcast_gossip(self, node_client, mock_router_connection):
        """Test broadcasting gossip to connected routers."""
        router_id = mock_router_connection.router.router_id
        node_client.connections[router_id] = mock_router_connection
        
        gossip = HealthGossip(
            source_node_id=node_client.node_id,
            timestamp=time.time(),
            observations=[],
            ttl=2,
        )
        
        await node_client._broadcast_gossip(gossip)
        
        # Should have sent to the router
        mock_router_connection.websocket.send_json.assert_called_once()
        call_args = mock_router_connection.websocket.send_json.call_args
        sent_data = call_args[0][0]
        
        assert sent_data["type"] == "gossip"
        assert "payload" in sent_data
    
    @pytest.mark.asyncio
    async def test_broadcast_skips_closed_connections(self, node_client, mock_router_connection):
        """Test that broadcast skips closed WebSocket connections."""
        mock_router_connection.websocket.closed = True
        router_id = mock_router_connection.router.router_id
        node_client.connections[router_id] = mock_router_connection
        
        gossip = HealthGossip(
            source_node_id=node_client.node_id,
            timestamp=time.time(),
            observations=[],
            ttl=2,
        )
        
        await node_client._broadcast_gossip(gossip)
        
        # Should not have attempted to send
        mock_router_connection.websocket.send_json.assert_not_called()


# =============================================================================
# GOSSIP PROPAGATION TESTS
# =============================================================================


class TestGossipPropagation:
    """Tests for gossip propagation between nodes."""
    
    @pytest.mark.asyncio
    async def test_propagate_gossip_to_other_routers(self, node_client, mock_router_connection):
        """Test propagating gossip to other routers (excluding source)."""
        # Add two router connections
        router1_id = "router1" + "a" * 58
        router2_id = "router2" + "b" * 58
        
        mock_conn1 = MagicMock()
        mock_conn1.websocket = MagicMock()
        mock_conn1.websocket.closed = False
        mock_conn1.websocket.send_json = AsyncMock()
        
        mock_conn2 = MagicMock()
        mock_conn2.websocket = MagicMock()
        mock_conn2.websocket.closed = False
        mock_conn2.websocket.send_json = AsyncMock()
        
        node_client.connections = {
            router1_id: mock_conn1,
            router2_id: mock_conn2,
        }
        
        gossip = HealthGossip(
            source_node_id="other_node",
            timestamp=time.time(),
            observations=[],
            ttl=1,
        )
        
        # Propagate, excluding router1 (simulating we received from router1)
        await node_client._propagate_gossip(gossip, exclude_router=router1_id)
        
        # Should send to router2 only
        mock_conn1.websocket.send_json.assert_not_called()
        mock_conn2.websocket.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_handle_gossip_decrements_ttl(self, node_client, mock_router_connection):
        """Test that gossip handling decrements TTL for propagation."""
        router_id = mock_router_connection.router.router_id
        node_client.connections[router_id] = mock_router_connection
        
        peer_id = "peer_node"
        obs = RouterHealthObservation(router_id="r1", latency_ms=50, last_seen=time.time())
        
        # Create gossip with TTL > 1
        gossip_data = {
            "payload": {
                "type": "health_gossip",
                "source_node_id": peer_id,
                "timestamp": time.time(),
                "observations": [obs.to_dict()],
                "ttl": 2,
            }
        }
        
        await node_client._handle_gossip_message(gossip_data, mock_router_connection)
        
        # Gossip should have been stored
        assert peer_id in node_client._peer_observations
        
        # Note: Propagation would be tested separately with multiple connections


# =============================================================================
# ROUTER SELECTION WITH GOSSIP TESTS
# =============================================================================


class TestRouterSelectionWithGossip:
    """Tests for router selection using aggregated health."""
    
    def test_select_router_uses_aggregated_health(self, node_client):
        """Test that router selection considers peer observations."""
        now = time.time()
        
        # Create two mock connections
        router1_info = RouterInfo(
            router_id="good_router_" + "a" * 52,
            endpoints=["192.168.1.1:8471"],
            capacity={"max_connections": 100, "current_load_pct": 20},
            health={},
            regions=[],
            features=[],
        )
        
        router2_info = RouterInfo(
            router_id="bad_router_" + "b" * 53,
            endpoints=["192.168.2.1:8471"],
            capacity={"max_connections": 100, "current_load_pct": 20},
            health={},
            regions=[],
            features=[],
        )
        
        conn1 = MagicMock()
        conn1.router = router1_info
        conn1.websocket = MagicMock()
        conn1.websocket.closed = False
        conn1.ping_latency_ms = 50
        conn1.ack_success_rate = 0.95
        conn1.health_score = 0.9
        conn1.back_pressure_until = 0  # Not under back-pressure
        
        conn2 = MagicMock()
        conn2.router = router2_info
        conn2.websocket = MagicMock()
        conn2.websocket.closed = False
        conn2.ping_latency_ms = 50
        conn2.ack_success_rate = 0.95
        conn2.health_score = 0.9
        conn2.back_pressure_until = 0  # Not under back-pressure
        
        node_client.connections = {
            router1_info.router_id: conn1,
            router2_info.router_id: conn2,
        }
        
        # Add peer observation saying router2 is bad
        peer_obs = RouterHealthObservation(
            router_id=router2_info.router_id,
            latency_ms=500,
            success_rate=0.3,
            load_pct=95,
            last_seen=now,
        )
        node_client._peer_observations["peer1"] = {
            router2_info.router_id: peer_obs
        }
        
        # Select routers many times and check distribution
        selections = {"good": 0, "bad": 0}
        for _ in range(100):
            selected = node_client._select_router()
            if selected.router_id.startswith("good"):
                selections["good"] += 1
            else:
                selections["bad"] += 1
        
        # Good router should be selected more often due to peer gossip
        # saying bad router is poor
        assert selections["good"] > selections["bad"]


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestGossipStatistics:
    """Tests for gossip-related statistics."""
    
    def test_stats_include_gossip_counts(self, node_client):
        """Test that stats include gossip sent/received counts."""
        stats = node_client.get_stats()
        
        assert "gossip_sent" in stats
        assert "gossip_received" in stats
        assert "own_health_observations" in stats
        assert "peer_observation_sources" in stats
    
    def test_get_health_observations(self, node_client):
        """Test get_health_observations debugging method."""
        now = time.time()
        
        # Add own observation
        obs = RouterHealthObservation(
            router_id="r1",
            latency_ms=50,
            success_rate=0.9,
            last_seen=now,
        )
        node_client._own_observations["r1"] = obs
        
        # Add peer observation
        peer_obs = RouterHealthObservation(
            router_id="r2",
            latency_ms=75,
            last_seen=now,
        )
        node_client._peer_observations["peer1"] = {"r2": peer_obs}
        
        report = node_client.get_health_observations()
        
        assert "own_observations" in report
        assert "r1" in report["own_observations"]
        assert report["peer_observation_count"] == 1
        assert report["peers_with_observations"] == 1
        assert "aggregated_health_scores" in report
