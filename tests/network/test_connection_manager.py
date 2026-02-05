"""
Tests for ConnectionManager component.

Tests cover:
- Connection lifecycle (establishment, closure)
- IP diversity enforcement
- ASN diversity enforcement  
- Failover state management
- Connection statistics
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.network.connection_manager import (
    ConnectionManager,
    ConnectionManagerConfig,
)
from valence.network.discovery import RouterInfo


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
def config():
    """Create a test configuration."""
    return ConnectionManagerConfig(
        min_connections=2,
        target_connections=3,
        max_connections=5,
        enforce_ip_diversity=True,
        ip_diversity_prefix=16,
        min_diverse_subnets=2,
        min_diverse_asns=2,
        asn_diversity_enabled=True,
    )


@pytest.fixture
def connection_manager(mock_discovery, config):
    """Create a ConnectionManager for testing."""
    return ConnectionManager(
        node_id="a" * 64,
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
def mock_router_info_ipv6():
    """Create a mock RouterInfo with IPv6 endpoint."""
    return RouterInfo(
        router_id="c" * 64,
        endpoints=["[2001:db8::1]:8471"],
        capacity={"max_connections": 100, "current_load_pct": 25},
        health={"uptime_pct": 99.9, "avg_latency_ms": 50},
        regions=["us-west"],
        features=["relay-v1"],
    )


# =============================================================================
# Unit Tests - ConnectionManagerConfig
# =============================================================================


class TestConnectionManagerConfig:
    """Tests for ConnectionManagerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ConnectionManagerConfig()
        assert config.min_connections == 3
        assert config.target_connections == 5
        assert config.max_connections == 8
        assert config.enforce_ip_diversity is True
        assert config.ip_diversity_prefix == 16
        assert config.asn_diversity_enabled is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ConnectionManagerConfig(
            min_connections=1,
            target_connections=2,
            max_connections=3,
            enforce_ip_diversity=False,
        )
        assert config.min_connections == 1
        assert config.target_connections == 2
        assert config.max_connections == 3
        assert config.enforce_ip_diversity is False


# =============================================================================
# Unit Tests - ConnectionManager Properties
# =============================================================================


class TestConnectionManagerProperties:
    """Tests for ConnectionManager property methods."""

    def test_connection_count_empty(self, connection_manager):
        """Test connection count with no connections."""
        assert connection_manager.connection_count == 0

    def test_is_healthy_below_minimum(self, connection_manager):
        """Test health check with insufficient connections."""
        assert connection_manager.is_healthy is False

    def test_get_stats_initial(self, connection_manager):
        """Test initial statistics."""
        stats = connection_manager.get_stats()
        assert stats["connections_established"] == 0
        assert stats["connections_failed"] == 0
        assert stats["diversity_rejections"] == 0
        assert stats["active_connections"] == 0
        assert stats["connected_subnets"] == 0
        assert stats["connected_asns"] == 0


# =============================================================================
# Unit Tests - IP Diversity
# =============================================================================


class TestIPDiversity:
    """Tests for IP diversity enforcement."""

    def test_check_ip_diversity_no_endpoints(self, connection_manager):
        """Test IP diversity check with router without endpoints."""
        router = RouterInfo(
            router_id="x" * 64,
            endpoints=[],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        # No endpoints means we can't check IP, so allow it
        assert connection_manager.check_ip_diversity(router) is False

    def test_check_ip_diversity_first_connection(self, connection_manager, mock_router_info):
        """Test IP diversity check for first connection (should pass)."""
        assert connection_manager.check_ip_diversity(mock_router_info) is True

    def test_check_ip_diversity_same_subnet(self, connection_manager, mock_router_info):
        """Test IP diversity check for router in same /16 subnet."""
        # Simulate that we already have a connection to this subnet
        connection_manager._connected_subnets.add("192.168.0.0/16")
        
        # Same subnet should be rejected
        assert connection_manager.check_ip_diversity(mock_router_info) is False
        assert connection_manager._stats["diversity_rejections"] == 1

    def test_check_ip_diversity_different_subnet(self, connection_manager, mock_router_info):
        """Test IP diversity check for router in different subnet."""
        # Simulate connection to a different subnet
        connection_manager._connected_subnets.add("10.0.0.0/16")
        
        # Different subnet should pass
        assert connection_manager.check_ip_diversity(mock_router_info) is True

    def test_check_ip_diversity_hostname(self, connection_manager):
        """Test IP diversity check with hostname endpoint."""
        router = RouterInfo(
            router_id="x" * 64,
            endpoints=["router.example.com:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        # Hostnames are allowed (we can't check diversity)
        assert connection_manager.check_ip_diversity(router) is True

    def test_check_ip_diversity_ipv6(self, connection_manager, mock_router_info_ipv6):
        """Test IP diversity check for IPv6 address."""
        # First IPv6 should pass
        assert connection_manager.check_ip_diversity(mock_router_info_ipv6) is True

    def test_check_ip_diversity_ipv6_same_block(self, connection_manager, mock_router_info_ipv6):
        """Test IP diversity check for IPv6 in same /48."""
        # Simulate that we already have a connection to this /48
        connection_manager._connected_subnets.add("2001:db8::/48")
        
        # Same /48 should be rejected
        assert connection_manager.check_ip_diversity(mock_router_info_ipv6) is False


# =============================================================================
# Unit Tests - ASN Diversity
# =============================================================================


class TestASNDiversity:
    """Tests for ASN diversity enforcement."""

    def test_check_asn_diversity_no_asn(self, connection_manager, mock_router_info):
        """Test ASN diversity check when router has no ASN."""
        # No ASN info means we can't check diversity, so allow it
        assert connection_manager.check_asn_diversity(mock_router_info) is True

    def test_check_asn_diversity_first_asn(self, connection_manager):
        """Test ASN diversity check for first ASN."""
        router = RouterInfo(
            router_id="x" * 64,
            endpoints=["1.2.3.4:8471"],
            capacity={"asn": "AS12345"},
            health={},
            regions=[],
            features=[],
        )
        assert connection_manager.check_asn_diversity(router) is True

    def test_check_asn_diversity_same_asn(self, connection_manager):
        """Test ASN diversity check for same ASN when we have min_diverse_asns already."""
        router = RouterInfo(
            router_id="x" * 64,
            endpoints=["1.2.3.4:8471"],
            capacity={"asn": "AS12345"},
            health={},
            regions=[],
            features=[],
        )
        
        # Simulate we already have this ASN and hit the min diverse
        connection_manager._connected_asns.add("AS12345")
        connection_manager._connected_asns.add("AS67890")
        
        # Same ASN when we have min_diverse_asns should be rejected
        assert connection_manager.check_asn_diversity(router) is False
        assert connection_manager._stats["diversity_rejections"] == 1

    def test_check_asn_diversity_disabled(self, connection_manager):
        """Test ASN diversity check when disabled."""
        connection_manager.config.asn_diversity_enabled = False
        router = RouterInfo(
            router_id="x" * 64,
            endpoints=["1.2.3.4:8471"],
            capacity={"asn": "AS12345"},
            health={},
            regions=[],
            features=[],
        )
        
        # Should always pass when disabled
        assert connection_manager.check_asn_diversity(router) is True


# =============================================================================
# Unit Tests - Subnet Tracking
# =============================================================================


class TestSubnetTracking:
    """Tests for subnet and ASN tracking."""

    def test_add_subnet_ipv4(self, connection_manager, mock_router_info):
        """Test adding IPv4 subnet tracking."""
        connection_manager._add_subnet(mock_router_info)
        
        assert "192.168.0.0/16" in connection_manager._connected_subnets

    def test_add_subnet_with_asn(self, connection_manager):
        """Test adding subnet with ASN tracking."""
        router = RouterInfo(
            router_id="x" * 64,
            endpoints=["1.2.3.4:8471"],
            capacity={"asn": "AS12345"},
            health={},
            regions=[],
            features=[],
        )
        connection_manager._add_subnet(router)
        
        assert "AS12345" in connection_manager._connected_asns

    def test_remove_subnet(self, connection_manager, mock_router_info):
        """Test removing subnet tracking."""
        # Add then remove
        connection_manager._add_subnet(mock_router_info)
        assert len(connection_manager._connected_subnets) == 1
        
        connection_manager._remove_subnet(mock_router_info)
        assert len(connection_manager._connected_subnets) == 0


# =============================================================================
# Unit Tests - Diversity Requirements
# =============================================================================


class TestDiversityRequirements:
    """Tests for diversity requirement checking."""

    def test_check_diversity_requirements_no_connections(self, connection_manager):
        """Test diversity requirements with no connections."""
        # With no connections, requirements should pass (vacuously true)
        assert connection_manager.check_diversity_requirements() is True

    def test_check_diversity_requirements_insufficient_subnets(self, connection_manager):
        """Test diversity requirements with insufficient subnet diversity."""
        # Simulate 3 connections but only 1 subnet
        connection_manager.connections = {"a": MagicMock(), "b": MagicMock(), "c": MagicMock()}
        connection_manager._connected_subnets = {"192.168.0.0/16"}
        
        assert connection_manager.check_diversity_requirements() is False


# =============================================================================
# Unit Tests - Failover State
# =============================================================================


class TestFailoverState:
    """Tests for failover state management."""

    def test_clear_router_cooldown_nonexistent(self, connection_manager):
        """Test clearing cooldown for non-existent router."""
        result = connection_manager.clear_router_cooldown("nonexistent")
        assert result is False

    def test_get_failover_states_empty(self, connection_manager):
        """Test getting failover states when empty."""
        states = connection_manager.get_failover_states()
        assert states == {}


# =============================================================================
# Unit Tests - Connection Management
# =============================================================================


class TestConnectionManagement:
    """Tests for connection management operations."""

    def test_get_connection_nonexistent(self, connection_manager):
        """Test getting a non-existent connection."""
        conn = connection_manager.get_connection("nonexistent")
        assert conn is None

    def test_get_healthy_connections_empty(self, connection_manager):
        """Test getting healthy connections when none exist."""
        conns = connection_manager.get_healthy_connections()
        assert conns == []

    @pytest.mark.asyncio
    async def test_close_all_empty(self, connection_manager):
        """Test closing all connections when none exist."""
        await connection_manager.close_all()
        
        assert connection_manager.connection_count == 0
        assert len(connection_manager._connected_subnets) == 0
        assert len(connection_manager._connected_asns) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestConnectionManagerIntegration:
    """Integration tests for ConnectionManager."""

    @pytest.mark.asyncio
    async def test_ensure_connections_no_routers(self, connection_manager, mock_discovery):
        """Test ensuring connections when no routers are discovered."""
        mock_discovery.discover_routers = AsyncMock(return_value=[])
        
        await connection_manager.ensure_connections()
        
        assert connection_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_ensure_connections_with_routers(self, connection_manager, mock_discovery, mock_router_info):
        """Test ensuring connections with available routers."""
        mock_discovery.discover_routers = AsyncMock(return_value=[mock_router_info])
        
        # Mock the connect_to_router method to avoid actual network calls
        with patch.object(connection_manager, 'connect_to_router', new_callable=AsyncMock) as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            await connection_manager.ensure_connections()
            
            # Should have tried to connect
            mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_close_connection_with_callback(self, connection_manager, mock_router_info):
        """Test closing connection triggers callback."""
        callback_called = []
        
        def on_lost(router_id):
            callback_called.append(router_id)
        
        connection_manager.on_connection_lost = on_lost
        
        # Create mock connection
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_session = AsyncMock()
        
        from valence.network.node import RouterConnection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        
        connection_manager.connections[mock_router_info.router_id] = conn
        connection_manager._add_subnet(mock_router_info)
        
        await connection_manager.close_connection(mock_router_info.router_id, conn)
        
        assert mock_router_info.router_id in callback_called
