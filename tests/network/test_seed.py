"""
Tests for Seed Node implementation.

Tests cover:
- RouterRecord data model
- Router selection algorithm
- Health filtering
- IP diversity enforcement
- Discovery protocol (HTTP endpoints)
- Registration and heartbeat handling
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.network.seed import (
    SeedNode,
    SeedConfig,
    RouterRecord,
    create_seed_node,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def seed_config():
    """Create a test seed config."""
    return SeedConfig(
        host="127.0.0.1",
        port=18470,  # Use non-standard port for tests
        seed_id="test-seed-001",
        known_seeds=["https://seed2.test.local"],
        min_uptime_pct=90.0,
        max_stale_seconds=600.0,
    )


@pytest.fixture
def seed_node(seed_config):
    """Create a test seed node."""
    return SeedNode(config=seed_config)


@pytest.fixture
def healthy_router():
    """Create a healthy router record."""
    now = time.time()
    return RouterRecord(
        router_id="z6MkRouter1TestABCDEFGHIJKLMNOP",
        endpoints=["192.168.1.100:8471"],
        capacity={
            "max_connections": 1000,
            "current_load_pct": 30.0,
            "bandwidth_mbps": 100,
        },
        health={
            "last_seen": now,
            "uptime_pct": 99.5,
            "avg_latency_ms": 15.0,
        },
        regions=["us-west", "us-central"],
        features=["ipv6", "quic"],
        registered_at=now - 3600,
        router_signature="sig1",
    )


@pytest.fixture
def unhealthy_router():
    """Create an unhealthy router (low uptime)."""
    now = time.time()
    return RouterRecord(
        router_id="z6MkRouter2UnhealthyXYZ",
        endpoints=["192.168.2.100:8471"],
        capacity={
            "max_connections": 500,
            "current_load_pct": 80.0,
            "bandwidth_mbps": 50,
        },
        health={
            "last_seen": now,
            "uptime_pct": 50.0,  # Below threshold
            "avg_latency_ms": 100.0,
        },
        regions=["us-east"],
        features=[],
        registered_at=now - 7200,
        router_signature="sig2",
    )


@pytest.fixture
def stale_router():
    """Create a stale router (old heartbeat)."""
    now = time.time()
    return RouterRecord(
        router_id="z6MkRouter3StaleABC",
        endpoints=["192.168.3.100:8471"],
        capacity={
            "max_connections": 1000,
            "current_load_pct": 10.0,
            "bandwidth_mbps": 100,
        },
        health={
            "last_seen": now - 1000,  # Stale (> 600s)
            "uptime_pct": 99.0,
            "avg_latency_ms": 10.0,
        },
        regions=["eu-west"],
        features=["ipv6"],
        registered_at=now - 86400,
        router_signature="sig3",
    )


@pytest.fixture
def multiple_routers():
    """Create multiple routers with varying IPs for diversity testing."""
    now = time.time()
    routers = []
    
    # Different /16 subnets
    subnets = ["10.0", "10.1", "192.168", "172.16", "172.17"]
    
    for i, subnet in enumerate(subnets):
        routers.append(RouterRecord(
            router_id=f"z6MkRouterMulti{i}ABCDEFG",
            endpoints=[f"{subnet}.{i}.100:8471"],
            capacity={
                "max_connections": 1000,
                "current_load_pct": 20.0 + i * 10,
                "bandwidth_mbps": 100,
            },
            health={
                "last_seen": now,
                "uptime_pct": 95.0 + i * 0.5,
                "avg_latency_ms": 10.0 + i,
            },
            regions=["us-west"] if i % 2 == 0 else ["us-east"],
            features=["ipv6"] if i % 2 == 0 else [],
            registered_at=now - 3600,
            router_signature=f"sig{i}",
        ))
    
    return routers


@pytest.fixture
def same_subnet_routers():
    """Create routers in the same /16 subnet."""
    now = time.time()
    routers = []
    
    for i in range(5):
        routers.append(RouterRecord(
            router_id=f"z6MkRouterSameSubnet{i}XYZ",
            endpoints=[f"192.168.{i}.100:8471"],  # All in 192.168.x.x
            capacity={
                "max_connections": 1000,
                "current_load_pct": 20.0 + i * 5,
                "bandwidth_mbps": 100,
            },
            health={
                "last_seen": now,
                "uptime_pct": 99.0 - i * 0.1,
                "avg_latency_ms": 10.0,
            },
            regions=["us-west"],
            features=["ipv6"],
            registered_at=now - 3600,
            router_signature=f"sig_same_{i}",
        ))
    
    return routers


# =============================================================================
# ROUTER RECORD TESTS
# =============================================================================


class TestRouterRecord:
    """Tests for RouterRecord data model."""
    
    def test_to_dict(self, healthy_router):
        """Test RouterRecord serialization."""
        d = healthy_router.to_dict()
        
        assert d["router_id"] == healthy_router.router_id
        assert d["endpoints"] == healthy_router.endpoints
        assert d["capacity"] == healthy_router.capacity
        assert d["health"] == healthy_router.health
        assert d["regions"] == healthy_router.regions
        assert d["features"] == healthy_router.features
        assert d["registered_at"] == healthy_router.registered_at
        assert d["router_signature"] == healthy_router.router_signature
    
    def test_from_dict(self, healthy_router):
        """Test RouterRecord deserialization."""
        d = healthy_router.to_dict()
        restored = RouterRecord.from_dict(d)
        
        assert restored.router_id == healthy_router.router_id
        assert restored.endpoints == healthy_router.endpoints
        assert restored.regions == healthy_router.regions
    
    def test_from_dict_minimal(self):
        """Test RouterRecord creation with minimal data."""
        d = {"router_id": "test-id"}
        router = RouterRecord.from_dict(d)
        
        assert router.router_id == "test-id"
        assert router.endpoints == []
        assert router.capacity == {}
        assert router.regions == []


# =============================================================================
# HEALTH FILTERING TESTS
# =============================================================================


class TestHealthFiltering:
    """Tests for router health filtering."""
    
    def test_healthy_router_passes(self, seed_node, healthy_router):
        """Healthy router should pass health check."""
        now = time.time()
        assert seed_node._is_healthy(healthy_router, now) is True
    
    def test_low_uptime_fails(self, seed_node, unhealthy_router):
        """Router with low uptime should fail health check."""
        now = time.time()
        assert seed_node._is_healthy(unhealthy_router, now) is False
    
    def test_stale_heartbeat_fails(self, seed_node, stale_router):
        """Router with stale heartbeat should fail health check."""
        now = time.time()
        assert seed_node._is_healthy(stale_router, now) is False
    
    def test_custom_thresholds(self, healthy_router):
        """Test health filtering with custom thresholds."""
        config = SeedConfig(
            min_uptime_pct=99.9,  # Very strict
            max_stale_seconds=60.0,  # Very short
        )
        node = SeedNode(config=config)
        
        now = time.time()
        # Healthy router has 99.5% uptime, should fail strict 99.9% threshold
        assert node._is_healthy(healthy_router, now) is False


# =============================================================================
# ROUTER SELECTION TESTS
# =============================================================================


class TestRouterSelection:
    """Tests for router selection algorithm."""
    
    def test_select_returns_healthy_only(
        self,
        seed_node,
        healthy_router,
        unhealthy_router,
        stale_router,
    ):
        """Selection should only return healthy routers."""
        seed_node.router_registry = {
            healthy_router.router_id: healthy_router,
            unhealthy_router.router_id: unhealthy_router,
            stale_router.router_id: stale_router,
        }
        
        selected = seed_node.select_routers(count=5)
        
        assert len(selected) == 1
        assert selected[0].router_id == healthy_router.router_id
    
    def test_select_respects_count(self, seed_node, multiple_routers):
        """Selection should respect requested count."""
        for r in multiple_routers:
            seed_node.router_registry[r.router_id] = r
        
        selected = seed_node.select_routers(count=3)
        assert len(selected) == 3
        
        selected = seed_node.select_routers(count=1)
        assert len(selected) == 1
    
    def test_select_empty_registry(self, seed_node):
        """Selection from empty registry returns empty list."""
        selected = seed_node.select_routers(count=5)
        assert selected == []
    
    def test_select_region_preference(self, seed_node, multiple_routers):
        """Selection should prefer routers matching region."""
        for r in multiple_routers:
            seed_node.router_registry[r.router_id] = r
        
        # Request us-west routers
        selected = seed_node.select_routers(
            count=5,
            preferences={"region": "us-west"},
        )
        
        # Should get routers (even indices have us-west)
        assert len(selected) > 0
        # The scoring should favor us-west routers
        us_west_count = sum(1 for r in selected if "us-west" in r.regions)
        assert us_west_count >= 1
    
    def test_select_with_features(self, seed_node, multiple_routers):
        """Selection should handle feature preferences."""
        for r in multiple_routers:
            seed_node.router_registry[r.router_id] = r
        
        selected = seed_node.select_routers(
            count=5,
            preferences={"features": ["ipv6"]},
        )
        
        assert len(selected) > 0


# =============================================================================
# IP DIVERSITY TESTS
# =============================================================================


class TestIPDiversity:
    """Tests for IP diversity enforcement in selection."""
    
    def test_get_subnet_ipv4(self, seed_node):
        """Test /16 subnet extraction."""
        assert seed_node._get_subnet("192.168.1.100:8471") == "192.168"
        assert seed_node._get_subnet("10.0.5.200:8471") == "10.0"
        assert seed_node._get_subnet("172.16.0.1:8471") == "172.16"
    
    def test_get_subnet_no_port(self, seed_node):
        """Test subnet extraction without port."""
        assert seed_node._get_subnet("192.168.1.100") == "192.168"
    
    def test_get_subnet_invalid(self, seed_node):
        """Test subnet extraction with invalid input."""
        assert seed_node._get_subnet("invalid") is None
        assert seed_node._get_subnet("hostname.example.com:8471") is None
    
    def test_diversity_different_subnets(self, seed_node, multiple_routers):
        """Selection from diverse subnets should include all."""
        for r in multiple_routers:
            seed_node.router_registry[r.router_id] = r
        
        selected = seed_node.select_routers(count=5)
        
        # Should get 5 routers from 5 different /16 subnets
        assert len(selected) == 5
        
        subnets = set()
        for r in selected:
            subnet = seed_node._get_subnet(r.endpoints[0])
            subnets.add(subnet)
        
        # All should be unique
        assert len(subnets) == 5
    
    def test_diversity_same_subnet(self, seed_node, same_subnet_routers):
        """Selection from same subnet should limit to one."""
        for r in same_subnet_routers:
            seed_node.router_registry[r.router_id] = r
        
        selected = seed_node.select_routers(count=5)
        
        # Should only get 1 router (all in same /16)
        assert len(selected) == 1


# =============================================================================
# SCORING TESTS
# =============================================================================


class TestRouterScoring:
    """Tests for router scoring algorithm."""
    
    def test_score_healthy_high(self, seed_node, healthy_router):
        """Healthy router should have high score."""
        score = seed_node._score_router(healthy_router, {})
        assert score > 0.5
    
    def test_score_region_boost(self, seed_node, healthy_router):
        """Region match should boost score."""
        score_no_match = seed_node._score_router(healthy_router, {"region": "eu-north"})
        score_match = seed_node._score_router(healthy_router, {"region": "us-west"})
        
        # us-west is in healthy_router.regions
        assert score_match > score_no_match
        assert score_match - score_no_match >= seed_node.config.weight_region - 0.01
    
    def test_score_load_impact(self, seed_node):
        """Higher load should result in lower score."""
        now = time.time()
        
        low_load = RouterRecord(
            router_id="low-load",
            endpoints=["10.0.1.1:8471"],
            capacity={"current_load_pct": 10.0},
            health={"last_seen": now, "uptime_pct": 95.0},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig",
        )
        
        high_load = RouterRecord(
            router_id="high-load",
            endpoints=["10.0.2.1:8471"],
            capacity={"current_load_pct": 90.0},
            health={"last_seen": now, "uptime_pct": 95.0},
            regions=[],
            features=[],
            registered_at=now,
            router_signature="sig",
        )
        
        score_low = seed_node._score_router(low_load, {})
        score_high = seed_node._score_router(high_load, {})
        
        assert score_low > score_high


# =============================================================================
# HTTP ENDPOINT TESTS
# =============================================================================


class TestDiscoveryEndpoint:
    """Tests for /discover endpoint."""
    
    @pytest.mark.asyncio
    async def test_discover_returns_routers(self, seed_node, healthy_router):
        """Discovery should return registered routers."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        
        # Create mock request
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "requested_count": 5,
            "preferences": {},
        })
        
        response = await seed_node.handle_discover(request)
        
        assert response.status == 200
        import json
        data = json.loads(response.text)
        
        assert data["seed_id"] == seed_node.seed_id
        assert "timestamp" in data
        assert len(data["routers"]) == 1
        assert data["routers"][0]["router_id"] == healthy_router.router_id
    
    @pytest.mark.asyncio
    async def test_discover_empty_body(self, seed_node, healthy_router):
        """Discovery should handle empty request body."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        
        request = MagicMock()
        request.json = AsyncMock(side_effect=Exception("No JSON"))
        
        response = await seed_node.handle_discover(request)
        
        assert response.status == 200
        import json
        data = json.loads(response.text)
        assert "routers" in data
    
    @pytest.mark.asyncio
    async def test_discover_includes_other_seeds(self, seed_node):
        """Discovery response should include other seeds."""
        request = MagicMock()
        request.json = AsyncMock(return_value={})
        
        response = await seed_node.handle_discover(request)
        
        import json
        data = json.loads(response.text)
        assert "other_seeds" in data
        assert data["other_seeds"] == seed_node.known_seeds


class TestRegistrationEndpoint:
    """Tests for /register endpoint."""
    
    @pytest.mark.asyncio
    async def test_register_new_router(self, seed_node):
        """Registration should add router to registry."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "router_id": "z6MkNewRouter123",
            "endpoints": ["10.0.0.1:8471"],
            "capacity": {"max_connections": 500},
            "regions": ["us-west"],
            "features": ["ipv6"],
            "router_signature": "test-sig",
        })
        
        response = await seed_node.handle_register(request)
        
        assert response.status == 200
        import json
        data = json.loads(response.text)
        
        assert data["status"] == "ok"
        assert data["action"] == "registered"
        assert "z6MkNewRouter123" in seed_node.router_registry
    
    @pytest.mark.asyncio
    async def test_register_update_existing(self, seed_node, healthy_router):
        """Registration should update existing router."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        original_registered_at = healthy_router.registered_at
        
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "router_id": healthy_router.router_id,
            "endpoints": ["10.0.0.2:8471"],  # New endpoint
            "router_signature": "new-sig",
        })
        
        response = await seed_node.handle_register(request)
        
        import json
        data = json.loads(response.text)
        
        assert data["action"] == "updated"
        
        # Should update endpoint but preserve registered_at
        updated = seed_node.router_registry[healthy_router.router_id]
        assert updated.endpoints == ["10.0.0.2:8471"]
        assert updated.registered_at == original_registered_at
    
    @pytest.mark.asyncio
    async def test_register_missing_router_id(self, seed_node):
        """Registration without router_id should fail."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "endpoints": ["10.0.0.1:8471"],
        })
        
        response = await seed_node.handle_register(request)
        
        assert response.status == 400
    
    @pytest.mark.asyncio
    async def test_register_missing_endpoints(self, seed_node):
        """Registration without endpoints should fail."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "router_id": "test-router",
        })
        
        response = await seed_node.handle_register(request)
        
        assert response.status == 400


class TestHeartbeatEndpoint:
    """Tests for /heartbeat endpoint."""
    
    @pytest.mark.asyncio
    async def test_heartbeat_updates_health(self, seed_node, healthy_router):
        """Heartbeat should update router health."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        old_last_seen = healthy_router.health["last_seen"]
        
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "router_id": healthy_router.router_id,
            "current_load_pct": 45.0,
            "uptime_pct": 99.8,
            "avg_latency_ms": 12.0,
        })
        
        await asyncio.sleep(0.01)  # Small delay to ensure time changes
        response = await seed_node.handle_heartbeat(request)
        
        assert response.status == 200
        
        updated = seed_node.router_registry[healthy_router.router_id]
        assert updated.health["last_seen"] >= old_last_seen
        assert updated.health["uptime_pct"] == 99.8
        assert updated.capacity["current_load_pct"] == 45.0
    
    @pytest.mark.asyncio
    async def test_heartbeat_unregistered_router(self, seed_node):
        """Heartbeat from unregistered router should fail."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "router_id": "unknown-router",
            "current_load_pct": 50.0,
        })
        
        response = await seed_node.handle_heartbeat(request)
        
        assert response.status == 404
    
    @pytest.mark.asyncio
    async def test_heartbeat_missing_router_id(self, seed_node):
        """Heartbeat without router_id should fail."""
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "current_load_pct": 50.0,
        })
        
        response = await seed_node.handle_heartbeat(request)
        
        assert response.status == 400


class TestStatusEndpoint:
    """Tests for /status endpoint."""
    
    @pytest.mark.asyncio
    async def test_status_returns_info(self, seed_node, healthy_router, unhealthy_router):
        """Status should return seed node info."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        seed_node.router_registry[unhealthy_router.router_id] = unhealthy_router
        seed_node._running = True
        
        request = MagicMock()
        response = await seed_node.handle_status(request)
        
        assert response.status == 200
        import json
        data = json.loads(response.text)
        
        assert data["seed_id"] == seed_node.seed_id
        assert data["status"] == "running"
        assert data["routers"]["total"] == 2
        assert data["routers"]["healthy"] == 1  # Only healthy_router passes


# =============================================================================
# LIFECYCLE TESTS
# =============================================================================


class TestSeedNodeLifecycle:
    """Tests for seed node start/stop lifecycle."""
    
    @pytest.mark.asyncio
    async def test_start_stop(self, seed_node):
        """Seed node should start and stop cleanly."""
        await seed_node.start()
        
        assert seed_node._running is True
        assert seed_node._app is not None
        assert seed_node._runner is not None
        
        await seed_node.stop()
        
        assert seed_node._running is False
        assert seed_node._app is None
    
    @pytest.mark.asyncio
    async def test_double_start(self, seed_node):
        """Double start should not fail."""
        await seed_node.start()
        await seed_node.start()  # Should log warning but not fail
        
        assert seed_node._running is True
        
        await seed_node.stop()
    
    @pytest.mark.asyncio
    async def test_double_stop(self, seed_node):
        """Double stop should not fail."""
        await seed_node.start()
        await seed_node.stop()
        await seed_node.stop()  # Should be safe
        
        assert seed_node._running is False


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_seed_node(self):
        """create_seed_node should create with custom config."""
        node = create_seed_node(
            host="127.0.0.1",
            port=9999,
            known_seeds=["https://seed.example.com"],
        )
        
        assert node.config.host == "127.0.0.1"
        assert node.config.port == 9999
        assert "https://seed.example.com" in node.config.known_seeds
