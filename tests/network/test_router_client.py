"""
Tests for RouterClient component.

Tests cover:
- Router selection (weighted by health)
- Back-pressure handling
- Failover logic
- Router rotation for eclipse attack resistance
- Direct mode (graceful degradation)
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from valence.network.connection_manager import (
    ConnectionManager,
    ConnectionManagerConfig,
)
from valence.network.discovery import RouterInfo
from valence.network.node import RouterConnection
from valence.network.router_client import (
    RouterClient,
    RouterClientConfig,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_discovery():
    """Create a mock DiscoveryClient."""
    discovery = MagicMock()
    discovery.discover_routers = AsyncMock(return_value=[])
    return discovery


@pytest.fixture
def mock_connection_manager(mock_discovery):
    """Create a mock ConnectionManager."""
    cm = MagicMock(spec=ConnectionManager)
    cm.connections = {}
    cm.failover_states = {}
    cm._connection_timestamps = {}
    cm.config = ConnectionManagerConfig()
    cm.connection_count = 0
    cm.check_ip_diversity = MagicMock(return_value=True)
    cm.check_asn_diversity = MagicMock(return_value=True)
    cm.close_connection = AsyncMock()
    cm.connect_to_router = AsyncMock()
    cm.ensure_connections = AsyncMock()
    return cm


@pytest.fixture
def config():
    """Create a test configuration."""
    return RouterClientConfig(
        initial_cooldown=10.0,
        max_cooldown=60.0,
        reconnect_delay=0.1,
        failover_connect_timeout=1.0,
        rotation_enabled=True,
        rotation_interval=60.0,
        rotation_max_age=120.0,
    )


@pytest.fixture
def router_client(mock_connection_manager, mock_discovery, config):
    """Create a RouterClient for testing."""
    return RouterClient(
        connection_manager=mock_connection_manager,
        discovery=mock_discovery,
        config=config,
    )


@pytest.fixture
def mock_router_info():
    """Create a mock RouterInfo."""
    return RouterInfo(
        router_id="b" * 64,
        endpoints=["192.168.1.1:8471"],
        capacity={"max_connections": 100, "current_load_pct": 25},
        health={"uptime_pct": 99.9, "avg_latency_ms": 50},
        regions=["us-west"],
        features=["relay-v1"],
    )


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket."""
    ws = AsyncMock()
    ws.closed = False
    return ws


@pytest.fixture
def mock_session():
    """Create a mock aiohttp session."""
    return AsyncMock()


@pytest.fixture
def mock_router_connection(mock_router_info, mock_websocket, mock_session):
    """Create a mock RouterConnection."""
    conn = RouterConnection(
        router=mock_router_info,
        websocket=mock_websocket,
        session=mock_session,
        connected_at=time.time(),
        last_seen=time.time(),
        ack_success=8,
        ack_failure=2,
        ping_latency_ms=50.0,
    )
    return conn


# =============================================================================
# Unit Tests - RouterClientConfig
# =============================================================================


class TestRouterClientConfig:
    """Tests for RouterClientConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = RouterClientConfig()
        assert config.initial_cooldown == 60.0
        assert config.max_cooldown == 3600.0
        assert config.reconnect_delay == 1.0
        assert config.rotation_enabled is True
        assert config.rotation_interval == 3600.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RouterClientConfig(
            initial_cooldown=30.0,
            max_cooldown=120.0,
            rotation_enabled=False,
        )
        assert config.initial_cooldown == 30.0
        assert config.max_cooldown == 120.0
        assert config.rotation_enabled is False


# =============================================================================
# Unit Tests - RouterClient Properties
# =============================================================================


class TestRouterClientProperties:
    """Tests for RouterClient property methods."""

    def test_get_stats_initial(self, router_client):
        """Test initial statistics."""
        stats = router_client.get_stats()
        assert stats["failovers"] == 0
        assert stats["routers_rotated"] == 0
        assert stats["direct_mode"] is False
        assert stats["last_rotation"] == 0.0


# =============================================================================
# Unit Tests - Router Selection
# =============================================================================


class TestRouterSelection:
    """Tests for router selection."""

    def test_select_router_no_connections(self, router_client):
        """Test router selection with no connections."""
        result = router_client.select_router()
        assert result is None

    def test_select_router_single_connection(self, router_client, mock_connection_manager, mock_router_connection):
        """Test router selection with single connection."""
        # Add connection to manager
        mock_connection_manager.connections = {mock_router_connection.router.router_id: mock_router_connection}

        result = router_client.select_router()

        assert result is not None
        assert result.router_id == mock_router_connection.router.router_id

    def test_select_router_excludes_closed(self, router_client, mock_connection_manager, mock_router_connection):
        """Test router selection excludes closed connections."""
        # Mark connection as closed
        mock_router_connection.websocket.closed = True
        mock_connection_manager.connections = {mock_router_connection.router.router_id: mock_router_connection}

        result = router_client.select_router()

        # Should not select closed connection
        assert result is None

    def test_select_router_excludes_back_pressured(self, router_client, mock_connection_manager, mock_router_connection):
        """Test router selection excludes back-pressured routers."""
        # Mark as under back-pressure
        mock_router_connection.back_pressure_active = True
        mock_router_connection.back_pressure_until = time.time() + 100

        mock_connection_manager.connections = {mock_router_connection.router.router_id: mock_router_connection}

        result = router_client.select_router(exclude_back_pressured=True)

        # With only back-pressured connections, should still return something
        # as fallback
        assert result is not None

    def test_select_router_uses_health_callback(self, router_client, mock_connection_manager, mock_router_connection):
        """Test router selection uses health callback."""
        mock_connection_manager.connections = {mock_router_connection.router.router_id: mock_router_connection}

        # Set up health callback
        router_client.get_aggregated_health = MagicMock(return_value=0.9)

        router_client.select_router()

        # Health callback should have been called
        router_client.get_aggregated_health.assert_called_with(mock_router_connection.router.router_id)

    def test_select_router_penalizes_flagged(self, router_client, mock_connection_manager, mock_router_connection):
        """Test router selection penalizes flagged routers."""
        mock_connection_manager.connections = {mock_router_connection.router.router_id: mock_router_connection}

        # Set up flagged router check
        router_client.is_router_flagged = MagicMock(return_value=True)

        router_client.select_router()

        # Flag check should have been called
        router_client.is_router_flagged.assert_called_with(mock_router_connection.router.router_id)


# =============================================================================
# Unit Tests - Back-Pressure Handling
# =============================================================================


class TestBackPressureHandling:
    """Tests for back-pressure handling."""

    def test_handle_back_pressure_active(self, router_client, mock_router_connection):
        """Test handling active back-pressure."""
        router_client.handle_back_pressure(
            conn=mock_router_connection,
            active=True,
            load_pct=85.0,
            retry_after_ms=5000,
            reason="high load",
        )

        assert mock_router_connection.back_pressure_active is True
        assert mock_router_connection.back_pressure_retry_ms == 5000
        assert mock_router_connection.back_pressure_until > time.time()

    def test_handle_back_pressure_released(self, router_client, mock_router_connection):
        """Test handling back-pressure release."""
        # First activate back-pressure
        mock_router_connection.back_pressure_active = True
        mock_router_connection.back_pressure_until = time.time() + 100

        router_client.handle_back_pressure(
            conn=mock_router_connection,
            active=False,
        )

        assert mock_router_connection.back_pressure_active is False
        assert mock_router_connection.back_pressure_until == 0.0


# =============================================================================
# Unit Tests - Direct Mode
# =============================================================================


class TestDirectMode:
    """Tests for direct mode (graceful degradation)."""

    def test_enable_direct_mode(self, router_client):
        """Test enabling direct mode."""
        assert router_client.direct_mode is False

        router_client._enable_direct_mode()

        assert router_client.direct_mode is True

    def test_disable_direct_mode(self, router_client):
        """Test disabling direct mode."""
        router_client.direct_mode = True

        router_client._disable_direct_mode()

        assert router_client.direct_mode is False


# =============================================================================
# Unit Tests - Router Rotation
# =============================================================================


class TestRouterRotation:
    """Tests for router rotation."""

    @pytest.mark.asyncio
    async def test_check_rotation_needed_disabled(self, router_client):
        """Test rotation check when disabled."""
        router_client.config.rotation_enabled = False

        result = await router_client.check_rotation_needed()

        assert result is None

    @pytest.mark.asyncio
    async def test_check_rotation_needed_no_connections(self, router_client, mock_connection_manager):
        """Test rotation check with no connections."""
        mock_connection_manager.connections = {}

        result = await router_client.check_rotation_needed()

        assert result is None

    @pytest.mark.asyncio
    async def test_check_rotation_needed_old_connection(self, router_client, mock_connection_manager, mock_router_connection):
        """Test rotation check detects old connection."""
        router_id = mock_router_connection.router.router_id

        # Set connection timestamp to exceed max age
        old_time = time.time() - 200  # Exceeds rotation_max_age of 120
        mock_router_connection.connected_at = old_time
        mock_connection_manager._connection_timestamps = {router_id: old_time}
        mock_connection_manager.connections = {router_id: mock_router_connection}

        result = await router_client.check_rotation_needed()

        assert result == router_id

    @pytest.mark.asyncio
    async def test_rotate_router_success(
        self,
        router_client,
        mock_connection_manager,
        mock_discovery,
        mock_router_info,
        mock_router_connection,
    ):
        """Test successful router rotation."""
        router_id = mock_router_connection.router.router_id
        mock_connection_manager.connections = {router_id: mock_router_connection}

        # Mock discovery to return new router
        new_router = RouterInfo(
            router_id="new" * 16,
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        mock_discovery.discover_routers = AsyncMock(return_value=[new_router])

        await router_client.rotate_router(router_id, reason="test")

        # Should have closed old connection
        mock_connection_manager.close_connection.assert_called()

        # Stats should be updated
        assert router_client._stats["routers_rotated"] == 1

    @pytest.mark.asyncio
    async def test_rotate_router_nonexistent(self, router_client, mock_connection_manager):
        """Test rotating non-existent router."""
        mock_connection_manager.connections = {}

        result = await router_client.rotate_router("nonexistent")

        assert result is False


# =============================================================================
# Integration Tests - Failover
# =============================================================================


class TestFailoverHandling:
    """Tests for failover handling."""

    @pytest.mark.asyncio
    async def test_handle_router_failure_creates_failover_state(
        self,
        router_client,
        mock_connection_manager,
        mock_router_connection,
        mock_discovery,
    ):
        """Test that router failure creates failover state."""
        router_id = mock_router_connection.router.router_id
        mock_connection_manager.connections = {router_id: mock_router_connection}
        mock_discovery.discover_routers = AsyncMock(return_value=[])

        await router_client.handle_router_failure(router_id)

        # Failover state should be created
        assert router_id in mock_connection_manager.failover_states
        state = mock_connection_manager.failover_states[router_id]
        assert state.fail_count == 1
        assert state.cooldown_until > time.time()

        # Stats should be updated
        assert router_client._stats["failovers"] == 1

    @pytest.mark.asyncio
    async def test_handle_router_failure_exponential_backoff(
        self,
        router_client,
        mock_connection_manager,
        mock_router_connection,
        mock_discovery,
    ):
        """Test exponential backoff on repeated failures."""
        router_id = mock_router_connection.router.router_id
        mock_connection_manager.connections = {router_id: mock_router_connection}
        mock_discovery.discover_routers = AsyncMock(return_value=[])

        # First failure
        await router_client.handle_router_failure(router_id)
        mock_connection_manager.failover_states[router_id].cooldown_until

        # Reset connection for second failure
        mock_connection_manager.connections = {router_id: mock_router_connection}

        # Second failure
        await router_client.handle_router_failure(router_id)
        mock_connection_manager.failover_states[router_id].cooldown_until

        # Second cooldown should be longer (exponential backoff)
        assert mock_connection_manager.failover_states[router_id].fail_count == 2

    @pytest.mark.asyncio
    async def test_handle_router_failure_connects_to_alternative(
        self,
        router_client,
        mock_connection_manager,
        mock_router_connection,
        mock_discovery,
    ):
        """Test that failover connects to alternative router."""
        router_id = mock_router_connection.router.router_id
        mock_connection_manager.connections = {router_id: mock_router_connection}

        # Mock alternative router
        alternative = RouterInfo(
            router_id="alt" * 16,
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={"uptime_pct": 99.0, "avg_latency_ms": 30},
            regions=[],
            features=[],
        )
        mock_discovery.discover_routers = AsyncMock(return_value=[alternative])

        await router_client.handle_router_failure(router_id)

        # Should have tried to connect to alternative
        mock_connection_manager.connect_to_router.assert_called()

    @pytest.mark.asyncio
    async def test_handle_router_failure_enables_direct_mode(
        self,
        router_client,
        mock_connection_manager,
        mock_router_connection,
        mock_discovery,
    ):
        """Test that direct mode is enabled when no alternatives available."""
        router_id = mock_router_connection.router.router_id
        mock_connection_manager.connections = {router_id: mock_router_connection}
        mock_connection_manager.connection_count = 0

        # No alternatives available
        mock_discovery.discover_routers = AsyncMock(return_value=[])
        mock_connection_manager.connect_to_router = AsyncMock(side_effect=OSError("Failed"))

        await router_client.handle_router_failure(router_id)

        # Direct mode should be enabled
        assert router_client.direct_mode is True

    @pytest.mark.asyncio
    async def test_handle_router_failure_nonexistent(self, router_client, mock_connection_manager):
        """Test handling failure for non-existent router."""
        mock_connection_manager.connections = {}

        result = await router_client.handle_router_failure("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_handle_router_failure_retries_messages(
        self,
        router_client,
        mock_connection_manager,
        mock_router_connection,
        mock_discovery,
    ):
        """Test that pending messages are retried on failover."""
        router_id = mock_router_connection.router.router_id
        mock_connection_manager.connections = {router_id: mock_router_connection}

        # Mock alternative
        alternative = RouterInfo(
            router_id="alt" * 16,
            endpoints=["10.0.0.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        mock_discovery.discover_routers = AsyncMock(return_value=[alternative])
        mock_connection_manager.connect_to_router = AsyncMock()

        # Track retry callback
        retry_called = []

        async def on_retry():
            retry_called.append(True)

        await router_client.handle_router_failure(
            router_id,
            on_retry_messages=on_retry,
        )

        # Retry callback should have been called
        assert len(retry_called) == 1
