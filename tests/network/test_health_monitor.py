"""
Tests for Health Monitoring Protocol.

Tests cover:
- HealthStatus enum and HealthState dataclass
- HealthMonitor heartbeat tracking
- Status state machine transitions
- Active probing
- Integration with discovery
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from valence.network.seed import (
    HealthState,
    HealthStatus,
    RouterRecord,
    SeedConfig,
    SeedNode,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def seed_config():
    """Create a test seed config."""
    return SeedConfig(
        host="127.0.0.1",
        port=18470,
        seed_id="test-seed-001",
        min_uptime_pct=90.0,
        max_stale_seconds=600.0,
        verify_signatures=False,
        verify_pow=False,
        probe_endpoints=False,
    )


@pytest.fixture
def seed_node(seed_config):
    """Create a test seed node."""
    return SeedNode(config=seed_config)


@pytest.fixture
def health_monitor(seed_node):
    """Create a health monitor for testing."""
    return seed_node.health_monitor


@pytest.fixture
def healthy_router():
    """Create a healthy router record."""
    now = time.time()
    return RouterRecord(
        router_id="router-healthy-001",
        endpoints=["192.168.1.100:8471"],
        capacity={"max_connections": 1000, "current_load_pct": 30.0},
        health={"last_seen": now, "uptime_pct": 99.5},
        regions=["us-west"],
        features=["ipv6"],
        registered_at=now - 3600,
        router_signature="sig1",
    )


@pytest.fixture
def multiple_routers():
    """Create multiple router records for testing."""
    now = time.time()
    routers = []
    for i in range(5):
        routers.append(
            RouterRecord(
                router_id=f"router-multi-{i:03d}",
                endpoints=[f"10.{i}.0.100:8471"],
                capacity={"max_connections": 1000, "current_load_pct": 20.0 + i * 10},
                health={"last_seen": now, "uptime_pct": 95.0 + i * 0.5},
                regions=["us-west"],
                features=["ipv6"],
                registered_at=now - 3600,
                router_signature=f"sig{i}",
            )
        )
    return routers


# =============================================================================
# HEALTH STATUS TESTS
# =============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """HealthStatus should have all expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.REMOVED.value == "removed"

    def test_status_comparison(self):
        """HealthStatus values should be comparable."""
        assert HealthStatus.HEALTHY != HealthStatus.WARNING
        assert HealthStatus.HEALTHY == HealthStatus.HEALTHY


class TestHealthState:
    """Tests for HealthState dataclass."""

    def test_health_state_creation(self):
        """HealthState should be creatable with all fields."""
        now = time.time()
        state = HealthState(
            status=HealthStatus.HEALTHY,
            missed_heartbeats=0,
            last_heartbeat=now,
            last_probe=now - 100,
            probe_latency_ms=15.5,
            warnings=["test_warning"],
        )

        assert state.status == HealthStatus.HEALTHY
        assert state.missed_heartbeats == 0
        assert state.last_heartbeat == now
        assert state.probe_latency_ms == 15.5
        assert "test_warning" in state.warnings

    def test_health_state_to_dict(self):
        """HealthState should serialize to dict."""
        now = time.time()
        state = HealthState(
            status=HealthStatus.WARNING,
            missed_heartbeats=1,
            last_heartbeat=now,
            last_probe=now - 50,
            probe_latency_ms=25.0,
            warnings=["high_latency"],
        )

        d = state.to_dict()

        assert d["status"] == "warning"
        assert d["missed_heartbeats"] == 1
        assert d["last_heartbeat"] == now
        assert d["probe_latency_ms"] == 25.0
        assert "high_latency" in d["warnings"]

    def test_health_state_default_warnings(self):
        """HealthState should default to empty warnings list."""
        state = HealthState(
            status=HealthStatus.HEALTHY,
            missed_heartbeats=0,
            last_heartbeat=time.time(),
            last_probe=0,
            probe_latency_ms=0,
        )

        assert state.warnings == []


# =============================================================================
# HEARTBEAT TRACKING TESTS
# =============================================================================


class TestHeartbeatTracking:
    """Tests for heartbeat tracking in HealthMonitor."""

    def test_record_heartbeat_new_router(self, health_monitor):
        """Recording heartbeat for new router creates healthy state."""
        router_id = "new-router-001"

        state = health_monitor.record_heartbeat(router_id)

        assert state.status == HealthStatus.HEALTHY
        assert state.missed_heartbeats == 0
        assert state.last_heartbeat > 0
        assert router_id in health_monitor.health_states

    def test_record_heartbeat_existing_router(self, health_monitor):
        """Recording heartbeat for existing router updates state."""
        router_id = "existing-router-001"

        # First heartbeat
        health_monitor.record_heartbeat(router_id)
        first_heartbeat = health_monitor.health_states[router_id].last_heartbeat

        # Simulate time passing
        time.sleep(0.01)

        # Second heartbeat
        state = health_monitor.record_heartbeat(router_id)

        assert state.last_heartbeat > first_heartbeat
        assert state.status == HealthStatus.HEALTHY
        assert state.missed_heartbeats == 0

    def test_record_heartbeat_resets_missed_count(self, health_monitor):
        """Recording heartbeat resets missed heartbeat count."""
        router_id = "missed-router-001"

        # Create state with missed heartbeats
        health_monitor.health_states[router_id] = HealthState(
            status=HealthStatus.DEGRADED,
            missed_heartbeats=2,
            last_heartbeat=time.time() - 1000,
            last_probe=0,
            probe_latency_ms=0,
        )

        # Record heartbeat
        state = health_monitor.record_heartbeat(router_id)

        assert state.missed_heartbeats == 0
        assert state.status == HealthStatus.HEALTHY

    def test_record_heartbeat_clears_transient_warnings(self, health_monitor):
        """Recording heartbeat clears transient probe warnings."""
        router_id = "warned-router-001"

        # Create state with warnings
        health_monitor.health_states[router_id] = HealthState(
            status=HealthStatus.WARNING,
            missed_heartbeats=1,
            last_heartbeat=time.time() - 400,
            last_probe=0,
            probe_latency_ms=0,
            warnings=["probe_failed", "probe_timeout", "high_latency"],
        )

        # Record heartbeat
        state = health_monitor.record_heartbeat(router_id)

        # Transient warnings should be cleared
        assert "probe_failed" not in state.warnings
        assert "probe_timeout" not in state.warnings
        # Persistent warning should remain
        assert "high_latency" in state.warnings


# =============================================================================
# STATUS TRANSITION TESTS
# =============================================================================


class TestStatusTransitions:
    """Tests for health status state machine transitions."""

    def test_compute_status_healthy(self, health_monitor):
        """0 missed heartbeats = HEALTHY."""
        assert health_monitor._compute_status(0) == HealthStatus.HEALTHY

    def test_compute_status_warning(self, health_monitor):
        """1 missed heartbeat = WARNING."""
        assert health_monitor._compute_status(1) == HealthStatus.WARNING

    def test_compute_status_degraded(self, health_monitor):
        """2 missed heartbeats = DEGRADED."""
        assert health_monitor._compute_status(2) == HealthStatus.DEGRADED

    def test_compute_status_unhealthy(self, health_monitor):
        """3-5 missed heartbeats = UNHEALTHY."""
        assert health_monitor._compute_status(3) == HealthStatus.UNHEALTHY
        assert health_monitor._compute_status(4) == HealthStatus.UNHEALTHY
        assert health_monitor._compute_status(5) == HealthStatus.UNHEALTHY

    def test_compute_status_removed(self, health_monitor):
        """6+ missed heartbeats = REMOVED."""
        assert health_monitor._compute_status(6) == HealthStatus.REMOVED
        assert health_monitor._compute_status(10) == HealthStatus.REMOVED
        assert health_monitor._compute_status(100) == HealthStatus.REMOVED

    @pytest.mark.asyncio
    async def test_heartbeat_checker_updates_status(self, seed_node, healthy_router):
        """Heartbeat checker should update status based on missed heartbeats."""
        # Register router and record initial heartbeat
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        health_monitor = seed_node.health_monitor

        # Set up old heartbeat (simulating 1 missed)
        health_monitor.health_states[healthy_router.router_id] = HealthState(
            status=HealthStatus.HEALTHY,
            missed_heartbeats=0,
            last_heartbeat=time.time() - 350,  # Just over 1 heartbeat interval
            last_probe=0,
            probe_latency_ms=0,
        )

        # Reduce check interval for test
        health_monitor.check_interval = 0.01

        # Start and let checker run once
        await health_monitor.start()
        await asyncio.sleep(0.05)
        await health_monitor.stop()

        state = health_monitor.health_states[healthy_router.router_id]
        assert state.missed_heartbeats >= 1
        assert state.status in (HealthStatus.WARNING, HealthStatus.DEGRADED)

    @pytest.mark.asyncio
    async def test_heartbeat_checker_removes_expired_routers(self, seed_node, healthy_router):
        """Heartbeat checker should remove routers after 6+ missed heartbeats."""
        # Register router
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        health_monitor = seed_node.health_monitor

        # Set up very old heartbeat (simulating 6+ missed)
        health_monitor.health_states[healthy_router.router_id] = HealthState(
            status=HealthStatus.UNHEALTHY,
            missed_heartbeats=5,
            last_heartbeat=time.time() - 2000,  # Way over 6 intervals
            last_probe=0,
            probe_latency_ms=0,
        )

        # Reduce check interval for test
        health_monitor.check_interval = 0.01

        # Start and let checker run
        await health_monitor.start()
        await asyncio.sleep(0.05)
        await health_monitor.stop()

        # Router should be removed from both registries
        assert healthy_router.router_id not in seed_node.router_registry
        assert healthy_router.router_id not in health_monitor.health_states


# =============================================================================
# ACTIVE PROBING TESTS
# =============================================================================


class TestActiveProbing:
    """Tests for active router probing."""

    @pytest.mark.asyncio
    async def test_probe_router_success(self, seed_node, healthy_router):
        """Successful probe should update latency."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        health_monitor = seed_node.health_monitor
        health_monitor.record_heartbeat(healthy_router.router_id)

        # Mock successful HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            await health_monitor._probe_router(healthy_router)

        state = health_monitor.health_states[healthy_router.router_id]
        assert state.last_probe > 0
        assert state.probe_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_probe_router_timeout(self, seed_node, healthy_router):
        """Probe timeout should add warning."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        health_monitor = seed_node.health_monitor
        health_monitor.record_heartbeat(healthy_router.router_id)

        # Mock timeout
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(side_effect=TimeoutError())
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            await health_monitor._probe_router(healthy_router)

        state = health_monitor.health_states[healthy_router.router_id]
        assert "probe_timeout" in state.warnings

    @pytest.mark.asyncio
    async def test_probe_router_failure(self, seed_node, healthy_router):
        """Probe failure should add warning."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        health_monitor = seed_node.health_monitor
        health_monitor.record_heartbeat(healthy_router.router_id)

        # Mock connection error
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(side_effect=aiohttp.ClientError("Connection refused"))
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_session

            await health_monitor._probe_router(healthy_router)

        state = health_monitor.health_states[healthy_router.router_id]
        assert "probe_failed" in state.warnings

    @pytest.mark.asyncio
    async def test_probe_router_high_latency(self, seed_node, healthy_router):
        """High latency probe should add warning."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router
        health_monitor = seed_node.health_monitor
        health_monitor.record_heartbeat(healthy_router.router_id)
        health_monitor.high_latency_threshold_ms = 10  # Low threshold for testing

        # Mock slow response using context manager pattern
        mock_response = MagicMock()
        mock_response.status = 200

        class SlowContextManager:
            async def __aenter__(self):
                await asyncio.sleep(0.05)  # 50ms delay
                return mock_response

            async def __aexit__(self, *args):
                pass

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=SlowContextManager())

        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session_cls.return_value = mock_ctx

            await health_monitor._probe_router(healthy_router)

        state = health_monitor.health_states[healthy_router.router_id]
        assert "high_latency" in state.warnings
        assert state.probe_latency_ms >= 50

    @pytest.mark.asyncio
    async def test_probe_router_no_endpoints(self, health_monitor):
        """Probe should handle router with no endpoints gracefully."""
        router = RouterRecord(
            router_id="no-endpoints",
            endpoints=[],
            capacity={},
            health={},
            regions=[],
            features=[],
            registered_at=time.time(),
            router_signature="sig",
        )

        # Should not raise
        await health_monitor._probe_router(router)


# =============================================================================
# DISCOVERY INTEGRATION TESTS
# =============================================================================


class TestDiscoveryIntegration:
    """Tests for health monitor integration with discovery."""

    def test_is_healthy_for_discovery_no_state(self, health_monitor):
        """Router with no health state should be considered healthy."""
        assert health_monitor.is_healthy_for_discovery("unknown-router") is True

    def test_is_healthy_for_discovery_healthy(self, health_monitor):
        """Router with HEALTHY status should pass discovery filter."""
        router_id = "healthy-router"
        health_monitor.health_states[router_id] = HealthState(
            status=HealthStatus.HEALTHY,
            missed_heartbeats=0,
            last_heartbeat=time.time(),
            last_probe=0,
            probe_latency_ms=0,
        )

        assert health_monitor.is_healthy_for_discovery(router_id) is True

    def test_is_healthy_for_discovery_warning(self, health_monitor):
        """Router with WARNING status should pass discovery filter."""
        router_id = "warning-router"
        health_monitor.health_states[router_id] = HealthState(
            status=HealthStatus.WARNING,
            missed_heartbeats=1,
            last_heartbeat=time.time() - 350,
            last_probe=0,
            probe_latency_ms=0,
        )

        assert health_monitor.is_healthy_for_discovery(router_id) is True

    def test_is_healthy_for_discovery_degraded(self, health_monitor):
        """Router with DEGRADED status should fail discovery filter."""
        router_id = "degraded-router"
        health_monitor.health_states[router_id] = HealthState(
            status=HealthStatus.DEGRADED,
            missed_heartbeats=2,
            last_heartbeat=time.time() - 700,
            last_probe=0,
            probe_latency_ms=0,
        )

        assert health_monitor.is_healthy_for_discovery(router_id) is False

    def test_is_healthy_for_discovery_unhealthy(self, health_monitor):
        """Router with UNHEALTHY status should fail discovery filter."""
        router_id = "unhealthy-router"
        health_monitor.health_states[router_id] = HealthState(
            status=HealthStatus.UNHEALTHY,
            missed_heartbeats=4,
            last_heartbeat=time.time() - 1500,
            last_probe=0,
            probe_latency_ms=0,
        )

        assert health_monitor.is_healthy_for_discovery(router_id) is False

    def test_select_routers_filters_unhealthy(self, seed_node, multiple_routers):
        """select_routers should filter out unhealthy routers."""
        health_monitor = seed_node.health_monitor

        # Register all routers
        for router in multiple_routers:
            seed_node.router_registry[router.router_id] = router
            health_monitor.record_heartbeat(router.router_id)

        # Make some routers unhealthy
        health_monitor.health_states[multiple_routers[1].router_id].status = HealthStatus.DEGRADED
        health_monitor.health_states[multiple_routers[3].router_id].status = HealthStatus.UNHEALTHY

        # Select routers
        selected = seed_node.select_routers(count=10)

        # Should only get the 3 healthy/warning routers
        assert len(selected) == 3
        selected_ids = {r.router_id for r in selected}
        assert multiple_routers[0].router_id in selected_ids
        assert multiple_routers[2].router_id in selected_ids
        assert multiple_routers[4].router_id in selected_ids

    def test_select_routers_include_unhealthy_flag(self, seed_node, multiple_routers):
        """select_routers with include_unhealthy=True should skip filter."""
        health_monitor = seed_node.health_monitor

        # Register all routers
        for router in multiple_routers:
            seed_node.router_registry[router.router_id] = router
            health_monitor.record_heartbeat(router.router_id)

        # Make some routers unhealthy
        health_monitor.health_states[multiple_routers[1].router_id].status = HealthStatus.DEGRADED
        health_monitor.health_states[multiple_routers[3].router_id].status = HealthStatus.UNHEALTHY

        # Select routers with include_unhealthy=True
        selected = seed_node.select_routers(count=10, include_unhealthy=True)

        # Should get all 5 routers
        assert len(selected) == 5


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestHealthMonitorStats:
    """Tests for health monitor statistics."""

    def test_get_stats_empty(self, health_monitor):
        """Stats should work with no routers."""
        stats = health_monitor.get_stats()

        assert stats["total"] == 0
        assert stats["healthy"] == 0
        assert stats["warning"] == 0
        assert stats["degraded"] == 0
        assert stats["unhealthy"] == 0
        assert stats["removed"] == 0

    def test_get_stats_with_routers(self, health_monitor):
        """Stats should count routers by status."""
        now = time.time()

        # Add routers with various statuses
        health_monitor.health_states["r1"] = HealthState(
            status=HealthStatus.HEALTHY,
            missed_heartbeats=0,
            last_heartbeat=now,
            last_probe=0,
            probe_latency_ms=0,
        )
        health_monitor.health_states["r2"] = HealthState(
            status=HealthStatus.HEALTHY,
            missed_heartbeats=0,
            last_heartbeat=now,
            last_probe=0,
            probe_latency_ms=0,
        )
        health_monitor.health_states["r3"] = HealthState(
            status=HealthStatus.WARNING,
            missed_heartbeats=1,
            last_heartbeat=now - 350,
            last_probe=0,
            probe_latency_ms=0,
        )
        health_monitor.health_states["r4"] = HealthState(
            status=HealthStatus.DEGRADED,
            missed_heartbeats=2,
            last_heartbeat=now - 700,
            last_probe=0,
            probe_latency_ms=0,
        )
        health_monitor.health_states["r5"] = HealthState(
            status=HealthStatus.UNHEALTHY,
            missed_heartbeats=4,
            last_heartbeat=now - 1500,
            last_probe=0,
            probe_latency_ms=0,
        )

        stats = health_monitor.get_stats()

        assert stats["total"] == 5
        assert stats["healthy"] == 2
        assert stats["warning"] == 1
        assert stats["degraded"] == 1
        assert stats["unhealthy"] == 1
        assert stats["removed"] == 0


# =============================================================================
# LIFECYCLE TESTS
# =============================================================================


class TestHealthMonitorLifecycle:
    """Tests for health monitor start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_stop(self, health_monitor):
        """Health monitor should start and stop cleanly."""
        await health_monitor.start()

        assert health_monitor._running is True
        assert len(health_monitor._tasks) == 2

        await health_monitor.stop()

        assert health_monitor._running is False
        assert len(health_monitor._tasks) == 0

    @pytest.mark.asyncio
    async def test_double_start(self, health_monitor):
        """Double start should not create duplicate tasks."""
        await health_monitor.start()
        await health_monitor.start()  # Should be safe

        assert health_monitor._running is True
        assert len(health_monitor._tasks) == 2

        await health_monitor.stop()

    @pytest.mark.asyncio
    async def test_seed_node_lifecycle_includes_health_monitor(self, seed_node):
        """SeedNode start/stop should manage health monitor."""
        await seed_node.start()

        assert seed_node._running is True
        assert seed_node.health_monitor._running is True

        await seed_node.stop()

        assert seed_node._running is False
        assert seed_node.health_monitor._running is False


# =============================================================================
# ENDPOINT INTEGRATION TESTS
# =============================================================================


class TestHeartbeatEndpointIntegration:
    """Tests for heartbeat endpoint with health monitor."""

    @pytest.mark.asyncio
    async def test_heartbeat_records_with_health_monitor(self, seed_node, healthy_router):
        """Heartbeat endpoint should record with health monitor."""
        seed_node.router_registry[healthy_router.router_id] = healthy_router

        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "router_id": healthy_router.router_id,
                "current_load_pct": 45.0,
                "uptime_pct": 99.8,
            }
        )

        response = await seed_node.handle_heartbeat(request)

        assert response.status == 200

        # Health monitor should have recorded heartbeat
        state = seed_node.health_monitor.get_health_state(healthy_router.router_id)
        assert state is not None
        assert state.status == HealthStatus.HEALTHY


class TestStatusEndpointIntegration:
    """Tests for status endpoint with health monitor."""

    @pytest.mark.asyncio
    async def test_status_includes_health_stats(self, seed_node, multiple_routers):
        """Status endpoint should include health monitor stats."""
        # Register routers with various health states
        health_monitor = seed_node.health_monitor

        for i, router in enumerate(multiple_routers):
            seed_node.router_registry[router.router_id] = router
            health_monitor.record_heartbeat(router.router_id)

        # Make one router degraded
        health_monitor.health_states[multiple_routers[0].router_id].status = HealthStatus.DEGRADED

        seed_node._running = True

        request = MagicMock()
        response = await seed_node.handle_status(request)

        assert response.status == 200

        import json

        data = json.loads(response.text)

        assert "health_monitor" in data
        assert data["health_monitor"]["total"] == 5
        assert data["health_monitor"]["healthy"] == 4
        assert data["health_monitor"]["degraded"] == 1
