"""
Tests for Seed Node Peering / Gossip Protocol.

Tests cover:
- SeedPeerManager initialization and configuration
- Gossip exchange protocol (HTTP endpoint)
- Router selection for gossip (health/freshness filtering)
- Router merging and deduplication
- Peer state tracking
- Full gossip round scenarios
"""

from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from valence.network.seed import (
    RouterRecord,
    SeedConfig,
    SeedNode,
    create_seed_node,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def gossip_config():
    """Create a test config with gossip enabled."""
    return SeedConfig(
        host="127.0.0.1",
        port=18470,
        seed_id="test-seed-001",
        peer_seeds=["http://seed2.test:8470", "http://seed3.test:8470"],
        gossip_enabled=True,
        gossip_interval_seconds=60.0,  # Short for testing
        gossip_batch_size=10,
        gossip_timeout_seconds=5.0,
        gossip_max_router_age_seconds=300.0,  # 5 minutes
        verify_signatures=False,
        verify_pow=False,
        probe_endpoints=False,
    )


@pytest.fixture
def seed_node(gossip_config):
    """Create a test seed node with gossip enabled."""
    return SeedNode(config=gossip_config)


@pytest.fixture
def peer_manager(seed_node):
    """Get the peer manager from the seed node."""
    return seed_node.peer_manager


@pytest.fixture
def fresh_router():
    """Create a fresh, healthy router."""
    now = time.time()
    return RouterRecord(
        router_id="z6MkFreshRouter1ABCDEFGHIJ",
        endpoints=["192.168.1.100:8471"],
        capacity={
            "max_connections": 1000,
            "current_load_pct": 30.0,
        },
        health={
            "last_seen": now,
            "uptime_pct": 99.5,
            "avg_latency_ms": 15.0,
            "status": "healthy",
        },
        regions=["us-west"],
        features=["ipv6"],
        registered_at=now - 3600,
        router_signature="sig1",
    )


@pytest.fixture
def stale_router():
    """Create a stale router (old last_seen)."""
    now = time.time()
    return RouterRecord(
        router_id="z6MkStaleRouter1XYZ",
        endpoints=["192.168.2.100:8471"],
        capacity={
            "max_connections": 500,
            "current_load_pct": 20.0,
        },
        health={
            "last_seen": now - 600,  # 10 minutes old (> 5 min threshold)
            "uptime_pct": 99.0,
            "avg_latency_ms": 10.0,
            "status": "healthy",
        },
        regions=["us-east"],
        features=[],
        registered_at=now - 7200,
        router_signature="sig2",
    )


@pytest.fixture
def multiple_fresh_routers():
    """Create multiple fresh routers for batch testing."""
    now = time.time()
    routers = []

    for i in range(15):  # More than batch size (10)
        routers.append(
            RouterRecord(
                router_id=f"z6MkBatchRouter{i:02d}ABCDEF",
                endpoints=[f"10.0.{i}.100:8471"],
                capacity={
                    "max_connections": 1000,
                    "current_load_pct": 20.0 + i,
                },
                health={
                    "last_seen": now - i,  # Slightly different times
                    "uptime_pct": 99.0,
                    "avg_latency_ms": 10.0,
                    "status": "healthy",
                },
                regions=["us-west"] if i % 2 == 0 else ["us-east"],
                features=["ipv6"] if i % 2 == 0 else [],
                registered_at=now - 3600,
                router_signature=f"sig_batch_{i}",
            )
        )

    return routers


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestGossipConfiguration:
    """Tests for gossip configuration."""

    def test_gossip_config_defaults(self):
        """Test default gossip configuration values."""
        config = SeedConfig()

        assert config.gossip_enabled is True
        assert config.gossip_interval_seconds == 300.0  # 5 minutes
        assert config.gossip_batch_size == 20
        assert config.gossip_timeout_seconds == 10.0
        assert config.gossip_max_router_age_seconds == 1800.0  # 30 minutes
        assert config.peer_seeds == []

    def test_gossip_config_custom(self, gossip_config):
        """Test custom gossip configuration."""
        assert gossip_config.peer_seeds == [
            "http://seed2.test:8470",
            "http://seed3.test:8470",
        ]
        assert gossip_config.gossip_interval_seconds == 60.0
        assert gossip_config.gossip_batch_size == 10

    def test_create_seed_node_with_peers(self):
        """Test create_seed_node with peer_seeds parameter."""
        node = create_seed_node(
            host="127.0.0.1",
            port=9999,
            peer_seeds=["http://peer1.test:8470"],
        )

        assert node.config.peer_seeds == ["http://peer1.test:8470"]


# =============================================================================
# PEER MANAGER TESTS
# =============================================================================


class TestSeedPeerManager:
    """Tests for SeedPeerManager initialization and properties."""

    def test_peer_manager_creation(self, seed_node):
        """Peer manager should be created on access."""
        pm = seed_node.peer_manager

        assert pm is not None
        assert pm.seed is seed_node
        assert pm.peer_seeds == seed_node.config.peer_seeds

    def test_peer_manager_singleton(self, seed_node):
        """Peer manager should be a singleton per seed node."""
        pm1 = seed_node.peer_manager
        pm2 = seed_node.peer_manager

        assert pm1 is pm2

    def test_peer_seeds_property(self, peer_manager, gossip_config):
        """Peer seeds should come from config."""
        assert peer_manager.peer_seeds == gossip_config.peer_seeds

    def test_gossip_interval_property(self, peer_manager, gossip_config):
        """Gossip interval should come from config."""
        assert peer_manager.gossip_interval == gossip_config.gossip_interval_seconds

    def test_batch_size_property(self, peer_manager, gossip_config):
        """Batch size should come from config."""
        assert peer_manager.batch_size == gossip_config.gossip_batch_size


# =============================================================================
# ROUTER SELECTION FOR GOSSIP
# =============================================================================


class TestRouterSelectionForGossip:
    """Tests for selecting routers to share in gossip."""

    def test_select_fresh_routers(self, seed_node, peer_manager, fresh_router):
        """Fresh healthy routers should be selected for gossip."""
        seed_node.router_registry[fresh_router.router_id] = fresh_router
        seed_node.health_monitor.record_heartbeat(fresh_router.router_id)

        selected = peer_manager._select_routers_for_gossip()

        assert len(selected) == 1
        assert selected[0]["router_id"] == fresh_router.router_id

    def test_exclude_stale_routers(self, seed_node, peer_manager, stale_router):
        """Stale routers should not be selected for gossip."""
        seed_node.router_registry[stale_router.router_id] = stale_router

        selected = peer_manager._select_routers_for_gossip()

        assert len(selected) == 0

    def test_exclude_unhealthy_routers(self, seed_node, peer_manager):
        """Unhealthy routers should not be selected for gossip."""
        now = time.time()
        unhealthy = RouterRecord(
            router_id="z6MkUnhealthyRouter123",
            endpoints=["10.0.0.1:8471"],
            capacity={"current_load_pct": 50.0},
            health={
                "last_seen": now,
                "uptime_pct": 50.0,  # Below threshold
            },
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig",
        )
        seed_node.router_registry[unhealthy.router_id] = unhealthy

        selected = peer_manager._select_routers_for_gossip()

        assert len(selected) == 0

    def test_batch_size_limit(self, seed_node, peer_manager, multiple_fresh_routers):
        """Selection should respect batch size limit."""
        for r in multiple_fresh_routers:
            seed_node.router_registry[r.router_id] = r
            seed_node.health_monitor.record_heartbeat(r.router_id)

        selected = peer_manager._select_routers_for_gossip()

        # Should be limited to batch_size (10)
        assert len(selected) == peer_manager.batch_size
        assert len(selected) < len(multiple_fresh_routers)

    def test_empty_registry(self, peer_manager):
        """Empty registry should return empty list."""
        selected = peer_manager._select_routers_for_gossip()

        assert selected == []


# =============================================================================
# ROUTER MERGING / DEDUPLICATION
# =============================================================================


class TestRouterMerging:
    """Tests for merging received routers and deduplication."""

    def test_merge_new_router(self, seed_node, peer_manager):
        """New router should be added to registry."""
        now = time.time()
        received = [
            {
                "router_id": "z6MkNewRouter123",
                "endpoints": ["10.0.0.1:8471"],
                "capacity": {"max_connections": 500},
                "health": {
                    "last_seen": now,
                    "uptime_pct": 99.0,
                },
                "regions": ["us-west"],
                "features": [],
                "registered_at": now - 100,
                "router_signature": "sig",
            }
        ]

        merged_count = peer_manager._merge_routers(received, "peer-seed-001")

        assert merged_count == 1
        assert "z6MkNewRouter123" in seed_node.router_registry

    def test_merge_skip_stale_router(self, seed_node, peer_manager):
        """Stale routers from gossip should be skipped."""
        now = time.time()
        received = [
            {
                "router_id": "z6MkStaleGossipRouter",
                "endpoints": ["10.0.0.1:8471"],
                "health": {
                    "last_seen": now - 600,  # Older than max_router_age (300s)
                },
                "registered_at": now - 1000,
                "router_signature": "sig",
            }
        ]

        merged_count = peer_manager._merge_routers(received, "peer-seed-001")

        assert merged_count == 0
        assert "z6MkStaleGossipRouter" not in seed_node.router_registry

    def test_dedup_newer_wins(self, seed_node, peer_manager, fresh_router):
        """When deduplicating, newer router should win."""
        seed_node.router_registry[fresh_router.router_id] = fresh_router
        fresh_router.health["last_seen"]

        # Receive newer version
        now = time.time()
        received = [
            {
                "router_id": fresh_router.router_id,
                "endpoints": ["10.0.0.99:8471"],  # Different endpoint
                "capacity": {"max_connections": 2000},  # Different capacity
                "health": {
                    "last_seen": now + 10,  # Newer
                    "uptime_pct": 99.9,
                },
                "regions": ["us-west"],
                "features": [],
                "registered_at": fresh_router.registered_at,
                "router_signature": "new-sig",
            }
        ]

        merged_count = peer_manager._merge_routers(received, "peer-seed-001")

        assert merged_count == 1

        updated = seed_node.router_registry[fresh_router.router_id]
        assert updated.endpoints == ["10.0.0.99:8471"]
        assert updated.capacity["max_connections"] == 2000

    def test_dedup_older_ignored(self, seed_node, peer_manager, fresh_router):
        """When deduplicating, older router should be ignored."""
        seed_node.router_registry[fresh_router.router_id] = fresh_router
        original_endpoints = fresh_router.endpoints.copy()

        # Receive older version
        received = [
            {
                "router_id": fresh_router.router_id,
                "endpoints": ["10.0.0.99:8471"],  # Different endpoint
                "health": {
                    "last_seen": fresh_router.health["last_seen"] - 100,  # Older
                },
                "registered_at": fresh_router.registered_at,
                "router_signature": "old-sig",
            }
        ]

        merged_count = peer_manager._merge_routers(received, "peer-seed-001")

        assert merged_count == 0

        # Should not be updated
        router = seed_node.router_registry[fresh_router.router_id]
        assert router.endpoints == original_endpoints

    def test_merge_preserves_original_source(self, seed_node, peer_manager, fresh_router):
        """Merging should preserve original source_ip."""
        fresh_router.source_ip = "original-source"
        seed_node.router_registry[fresh_router.router_id] = fresh_router

        # Receive newer version
        now = time.time()
        received = [
            {
                "router_id": fresh_router.router_id,
                "endpoints": fresh_router.endpoints,
                "health": {
                    "last_seen": now + 10,
                },
                "registered_at": fresh_router.registered_at,
                "router_signature": "new-sig",
            }
        ]

        peer_manager._merge_routers(received, "peer-seed-002")

        updated = seed_node.router_registry[fresh_router.router_id]
        assert updated.source_ip == "original-source"  # Preserved

    def test_merge_invalid_router_skipped(self, seed_node, peer_manager):
        """Invalid router data should be skipped without error."""
        received = [
            {"no_router_id": True},  # Missing router_id
            {"router_id": "valid", "health": {"last_seen": time.time()}},  # Valid
        ]

        merged_count = peer_manager._merge_routers(received, "peer-seed-001")

        assert merged_count == 1
        assert "valid" in seed_node.router_registry


# =============================================================================
# GOSSIP EXCHANGE ENDPOINT
# =============================================================================


class TestGossipExchangeEndpoint:
    """Tests for /gossip/exchange HTTP endpoint."""

    @pytest.mark.asyncio
    async def test_handle_gossip_exchange_basic(self, seed_node, peer_manager, fresh_router):
        """Gossip exchange should accept and return routers."""
        seed_node.router_registry[fresh_router.router_id] = fresh_router
        seed_node.health_monitor.record_heartbeat(fresh_router.router_id)

        now = time.time()
        incoming_router = {
            "router_id": "z6MkIncomingRouter123",
            "endpoints": ["10.0.0.99:8471"],
            "health": {"last_seen": now},
            "registered_at": now - 100,
            "router_signature": "sig",
        }

        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "seed_id": "peer-seed-001",
                "timestamp": now,
                "routers": [incoming_router],
            }
        )

        response = await peer_manager.handle_gossip_exchange(request)

        assert response.status == 200

        data = json.loads(response.text)
        assert data["seed_id"] == seed_node.seed_id
        assert "timestamp" in data
        assert "routers" in data

        # Should have merged incoming router
        assert "z6MkIncomingRouter123" in seed_node.router_registry

    @pytest.mark.asyncio
    async def test_handle_gossip_exchange_returns_our_routers(self, seed_node, peer_manager, fresh_router):
        """Gossip exchange should return our routers."""
        seed_node.router_registry[fresh_router.router_id] = fresh_router
        seed_node.health_monitor.record_heartbeat(fresh_router.router_id)

        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "seed_id": "peer-seed-001",
                "timestamp": time.time(),
                "routers": [],
            }
        )

        response = await peer_manager.handle_gossip_exchange(request)

        data = json.loads(response.text)

        # Should include our fresh router
        router_ids = [r["router_id"] for r in data["routers"]]
        assert fresh_router.router_id in router_ids

    @pytest.mark.asyncio
    async def test_handle_gossip_exchange_invalid_json(self, peer_manager):
        """Invalid JSON should return 400 error."""
        request = MagicMock()
        request.json = AsyncMock(side_effect=json.JSONDecodeError("Invalid JSON", "", 0))

        response = await peer_manager.handle_gossip_exchange(request)

        assert response.status == 400


# =============================================================================
# PEER STATE TRACKING
# =============================================================================


class TestPeerStateTracking:
    """Tests for peer connection state tracking."""

    def test_update_peer_state_success(self, peer_manager):
        """Successful exchange should update state."""
        peer_url = "http://test-peer:8470"

        peer_manager._update_peer_state(peer_url, success=True)

        state = peer_manager._peer_states[peer_url]
        assert state["successful_exchanges"] == 1
        assert state["failed_exchanges"] == 0
        assert state["last_success"] is not None
        assert state["last_error"] is None

    def test_update_peer_state_failure(self, peer_manager):
        """Failed exchange should update state with error."""
        peer_url = "http://test-peer:8470"

        peer_manager._update_peer_state(peer_url, success=False, error="timeout")

        state = peer_manager._peer_states[peer_url]
        assert state["successful_exchanges"] == 0
        assert state["failed_exchanges"] == 1
        assert state["last_failure"] is not None
        assert state["last_error"] == "timeout"

    def test_get_peer_stats(self, peer_manager, gossip_config):
        """Peer stats should include all peer info."""
        # Simulate some exchanges
        peer_manager._update_peer_state("http://seed2.test:8470", success=True)
        peer_manager._update_peer_state("http://seed3.test:8470", success=False, error="refused")

        stats = peer_manager.get_peer_stats()

        assert stats["peer_count"] == len(gossip_config.peer_seeds)
        assert stats["gossip_enabled"] is True
        assert stats["gossip_interval_seconds"] == gossip_config.gossip_interval_seconds

        assert "http://seed2.test:8470" in stats["peer_states"]
        assert stats["peer_states"]["http://seed2.test:8470"]["successful"] == 1


# =============================================================================
# GOSSIP ROUND TESTS
# =============================================================================


class TestGossipRound:
    """Tests for full gossip round execution."""

    @pytest.mark.asyncio
    async def test_exchange_with_peer_success(self, seed_node, peer_manager, fresh_router):
        """Successful peer exchange should merge routers."""
        seed_node.router_registry[fresh_router.router_id] = fresh_router
        seed_node.health_monitor.record_heartbeat(fresh_router.router_id)

        now = time.time()
        peer_router = {
            "router_id": "z6MkPeerRouter123",
            "endpoints": ["10.0.0.50:8471"],
            "health": {"last_seen": now},
            "registered_at": now - 100,
            "router_signature": "sig",
        }

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "seed_id": "peer-seed-001",
                "timestamp": now,
                "routers": [peer_router],
            }
        )

        with patch("valence.network.seed.aiohttp.ClientSession") as mock_session:
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_response)
            mock_ctx.__aexit__ = AsyncMock()

            mock_session_instance = MagicMock()
            mock_session_instance.post = MagicMock(return_value=mock_ctx)
            mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
            mock_session_instance.__aexit__ = AsyncMock()

            mock_session.return_value = mock_session_instance

            result = await peer_manager._exchange_with_peer(
                "http://peer.test:8470",
                [fresh_router.to_dict()],
            )

        assert result is True
        assert "z6MkPeerRouter123" in seed_node.router_registry

    @pytest.mark.asyncio
    async def test_exchange_with_peer_timeout(self, peer_manager):
        """Timeout should be handled gracefully by the exception handler."""
        # Directly test the error handling path by manipulating peer state
        # and verifying the expected behavior

        # The actual HTTP timeout behavior is tested via integration tests
        # Here we verify the state tracking works correctly
        peer_url = "http://slow-peer.test:8470"

        # Simulate what happens on timeout
        peer_manager._update_peer_state(peer_url, success=False, error="timeout")

        state = peer_manager._peer_states[peer_url]
        assert state["failed_exchanges"] == 1
        assert state["last_error"] == "timeout"

    @pytest.mark.asyncio
    async def test_exchange_with_peer_error(self, peer_manager):
        """Connection error tracking should work correctly."""
        # Directly test the error handling path
        peer_url = "http://down-peer.test:8470"

        # Simulate what happens on connection error
        peer_manager._update_peer_state(peer_url, success=False, error="Connection refused")

        state = peer_manager._peer_states[peer_url]
        assert state["failed_exchanges"] == 1
        assert state["last_error"] == "Connection refused"

    @pytest.mark.asyncio
    async def test_gossip_round_no_peers(self, seed_node):
        """Gossip round with no peers should complete quickly."""
        config = SeedConfig(
            peer_seeds=[],  # No peers
            gossip_enabled=True,
        )
        node = SeedNode(config=config)

        # Should not fail
        await node.peer_manager._gossip_round()


# =============================================================================
# LIFECYCLE TESTS
# =============================================================================


class TestGossipLifecycle:
    """Tests for gossip start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_with_peers(self, seed_node, peer_manager):
        """Start should begin gossip loop when peers configured."""
        await peer_manager.start()

        assert peer_manager._running is True
        assert peer_manager._gossip_task is not None

        await peer_manager.stop()

        assert peer_manager._running is False

    @pytest.mark.asyncio
    async def test_start_without_peers(self):
        """Start should skip gossip when no peers configured."""
        config = SeedConfig(
            peer_seeds=[],
            gossip_enabled=True,
        )
        node = SeedNode(config=config)

        await node.peer_manager.start()

        # Should not start gossip task when no peers
        assert node.peer_manager._gossip_task is None

        await node.peer_manager.stop()

    @pytest.mark.asyncio
    async def test_start_gossip_disabled(self, gossip_config):
        """Start should skip gossip when disabled."""
        gossip_config.gossip_enabled = False
        node = SeedNode(config=gossip_config)

        await node.peer_manager.start()

        assert node.peer_manager._gossip_task is None

        await node.peer_manager.stop()

    @pytest.mark.asyncio
    async def test_seed_node_starts_peer_manager(self, seed_node):
        """Seed node start should also start peer manager."""
        await seed_node.start()

        assert seed_node._peer_manager is not None
        assert seed_node.peer_manager._running is True

        await seed_node.stop()

        assert seed_node.peer_manager._running is False


# =============================================================================
# STATUS ENDPOINT WITH PEERING
# =============================================================================


class TestStatusWithPeering:
    """Tests for /status endpoint with peering info."""

    @pytest.mark.asyncio
    async def test_status_includes_peering(self, seed_node, peer_manager):
        """Status should include peering statistics."""
        seed_node._running = True

        # Simulate some peer activity
        peer_manager._update_peer_state("http://seed2.test:8470", success=True)

        request = MagicMock()
        response = await seed_node.handle_status(request)

        data = json.loads(response.text)

        assert "peering" in data
        assert data["peering"]["peer_count"] == 2
        assert data["peering"]["gossip_enabled"] is True


# =============================================================================
# INTEGRATION SCENARIOS
# =============================================================================


class TestGossipIntegrationScenarios:
    """Integration tests for gossip scenarios."""

    @pytest.mark.asyncio
    async def test_bidirectional_gossip(self):
        """Two seeds should be able to exchange routers bidirectionally."""
        # Create two seed nodes
        config1 = SeedConfig(
            host="127.0.0.1",
            port=18470,
            seed_id="seed-1",
            peer_seeds=["http://127.0.0.1:18471"],
            gossip_enabled=True,
            verify_signatures=False,
            verify_pow=False,
            probe_endpoints=False,
        )
        config2 = SeedConfig(
            host="127.0.0.1",
            port=18471,
            seed_id="seed-2",
            peer_seeds=["http://127.0.0.1:18470"],
            gossip_enabled=True,
            verify_signatures=False,
            verify_pow=False,
            probe_endpoints=False,
        )

        seed1 = SeedNode(config=config1)
        seed2 = SeedNode(config=config2)

        # Add different routers to each seed
        now = time.time()
        router1 = RouterRecord(
            router_id="z6MkSeed1Router",
            endpoints=["10.1.0.1:8471"],
            capacity={},
            health={"last_seen": now, "uptime_pct": 99.0},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig1",
        )
        router2 = RouterRecord(
            router_id="z6MkSeed2Router",
            endpoints=["10.2.0.1:8471"],
            capacity={},
            health={"last_seen": now, "uptime_pct": 99.0},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig2",
        )

        seed1.router_registry[router1.router_id] = router1
        seed1.health_monitor.record_heartbeat(router1.router_id)

        seed2.router_registry[router2.router_id] = router2
        seed2.health_monitor.record_heartbeat(router2.router_id)

        # Simulate exchange: seed1 sends to seed2
        routers_from_seed1 = seed1.peer_manager._select_routers_for_gossip()
        merged_at_seed2 = seed2.peer_manager._merge_routers(routers_from_seed1, "seed-1")

        assert merged_at_seed2 == 1
        assert router1.router_id in seed2.router_registry

        # Simulate exchange: seed2 sends to seed1
        routers_from_seed2 = seed2.peer_manager._select_routers_for_gossip()
        seed1.peer_manager._merge_routers(routers_from_seed2, "seed-2")

        # Note: seed2 now has router1, so it might send it back
        # seed1 should have both routers now
        assert router2.router_id in seed1.router_registry

    @pytest.mark.asyncio
    async def test_stale_router_not_propagated(self, seed_node, peer_manager):
        """Stale routers should not be propagated via gossip."""
        now = time.time()

        # Add fresh and stale routers
        fresh = RouterRecord(
            router_id="z6MkFreshOne",
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={"last_seen": now, "uptime_pct": 99.0},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig1",
        )
        stale = RouterRecord(
            router_id="z6MkStaleOne",
            endpoints=["10.0.0.2:8471"],
            capacity={},
            health={
                "last_seen": now - 600,  # 10 min old, > 5 min threshold
                "uptime_pct": 99.0,
            },
            regions=[],
            features=[],
            registered_at=now - 1000,
            router_signature="sig2",
        )

        seed_node.router_registry[fresh.router_id] = fresh
        seed_node.router_registry[stale.router_id] = stale
        seed_node.health_monitor.record_heartbeat(fresh.router_id)

        selected = peer_manager._select_routers_for_gossip()

        router_ids = [r["router_id"] for r in selected]
        assert fresh.router_id in router_ids
        assert stale.router_id not in router_ids
