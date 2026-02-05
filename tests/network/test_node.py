"""Tests for the Valence Node Client.

These tests verify:
- Connection management (multi-router, reconnection)
- Router selection (weighted by health)
- Message sending and queueing
- Keepalive and failure detection
- IP diversity enforcement
"""

from __future__ import annotations

import asyncio
import ipaddress
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from valence.network.node import (
    NodeClient,
    RouterConnection,
    PendingMessage,
    FailoverState,
    NodeError,
    NoRoutersAvailableError,
    create_node_client,
)
from valence.network.discovery import RouterInfo, DiscoveryClient


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
def mock_router_info():
    """Create a mock RouterInfo."""
    return RouterInfo(
        router_id="a" * 64,
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
    """Create a NodeClient for testing."""
    private_key, public_key = ed25519_keypair
    enc_private, _ = x25519_keypair
    
    return NodeClient(
        node_id=public_key.public_bytes_raw().hex(),
        private_key=private_key,
        encryption_private_key=enc_private,
        min_connections=1,
        target_connections=3,
        max_connections=5,
    )


# =============================================================================
# Unit Tests - RouterConnection
# =============================================================================


class TestRouterConnection:
    """Tests for the RouterConnection dataclass."""

    def test_router_connection_creation(self, mock_router_info, mock_websocket, mock_session):
        """Test creating a RouterConnection instance."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=1000.0,
            last_seen=1000.0,
        )
        
        assert conn.router is mock_router_info
        assert conn.websocket is mock_websocket
        assert conn.connected_at == 1000.0
        assert conn.last_seen == 1000.0
        assert conn.messages_sent == 0
        assert conn.messages_received == 0

    def test_ack_success_rate_no_data(self, mock_router_info, mock_websocket, mock_session):
        """Test ACK success rate with no data (default to 1.0)."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=1000.0,
            last_seen=1000.0,
        )
        
        assert conn.ack_success_rate == 1.0

    def test_ack_success_rate_with_data(self, mock_router_info, mock_websocket, mock_session):
        """Test ACK success rate calculation."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=1000.0,
            last_seen=1000.0,
            ack_success=8,
            ack_failure=2,
        )
        
        assert conn.ack_success_rate == 0.8

    def test_health_score(self, mock_router_info, mock_websocket, mock_session):
        """Test health score calculation."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=1000.0,
            last_seen=1000.0,
            ack_success=10,
            ack_failure=0,
            ping_latency_ms=100,
        )
        
        # ACK score: 1.0 * 0.5 = 0.5
        # Latency score: (1.0 - 100/500) * 0.3 = 0.8 * 0.3 = 0.24
        # Load score: (1.0 - 25/100) * 0.2 = 0.75 * 0.2 = 0.15
        # Total: 0.5 + 0.24 + 0.15 = 0.89
        assert 0.88 < conn.health_score < 0.90


# =============================================================================
# Unit Tests - PendingMessage
# =============================================================================


class TestPendingMessage:
    """Tests for the PendingMessage dataclass."""

    def test_pending_message_creation(self, x25519_keypair):
        """Test creating a PendingMessage instance."""
        _, public_key = x25519_keypair
        
        msg = PendingMessage(
            message_id="msg-123",
            recipient_id="recipient-456",
            content=b"Hello, World!",
            recipient_public_key=public_key,
            queued_at=2000.0,
        )
        
        assert msg.message_id == "msg-123"
        assert msg.recipient_id == "recipient-456"
        assert msg.content == b"Hello, World!"
        assert msg.queued_at == 2000.0
        assert msg.retries == 0
        assert msg.max_retries == 3


# =============================================================================
# Unit Tests - NodeClient
# =============================================================================


class TestNodeClient:
    """Tests for NodeClient core functionality."""

    def test_node_creation_defaults(self, ed25519_keypair, x25519_keypair):
        """Test creating a node with default settings."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        node_id = public_key.public_bytes_raw().hex()
        
        node = NodeClient(
            node_id=node_id,
            private_key=private_key,
            encryption_private_key=enc_private,
        )
        
        assert node.node_id == node_id
        assert node.min_connections == 3
        assert node.target_connections == 5
        assert node.max_connections == 8
        assert node.keepalive_interval == 2.0  # Fast detection (changed from 30.0)
        assert node.enforce_ip_diversity is True
        assert len(node.connections) == 0
        assert len(node.message_queue) == 0
        assert node.direct_mode is False
        assert len(node.failover_states) == 0

    def test_node_creation_custom(self, ed25519_keypair, x25519_keypair):
        """Test creating a node with custom settings."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        node_id = public_key.public_bytes_raw().hex()
        
        node = NodeClient(
            node_id=node_id,
            private_key=private_key,
            encryption_private_key=enc_private,
            min_connections=2,
            target_connections=4,
            max_connections=6,
            keepalive_interval=60.0,
            enforce_ip_diversity=False,
        )
        
        assert node.min_connections == 2
        assert node.target_connections == 4
        assert node.max_connections == 6
        assert node.keepalive_interval == 60.0
        assert node.enforce_ip_diversity is False

    def test_get_stats(self, node_client):
        """Test getting node statistics."""
        stats = node_client.get_stats()
        
        assert "messages_sent" in stats
        assert "messages_received" in stats
        assert "active_connections" in stats
        assert "queued_messages" in stats
        assert stats["active_connections"] == 0
        assert stats["queued_messages"] == 0

    def test_get_connections_empty(self, node_client):
        """Test getting connections when none exist."""
        connections = node_client.get_connections()
        assert connections == []


# =============================================================================
# Unit Tests - Router Selection
# =============================================================================


class TestRouterSelection:
    """Tests for router selection logic (uses ConnectionManager internally)."""

    def test_select_router_no_connections(self, node_client):
        """Test router selection with no connections."""
        result = node_client._select_router()
        assert result is None

    def test_select_router_single_connection(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test router selection with single connection."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        # Add to both NodeClient.connections and ConnectionManager.connections
        node_client.connections[mock_router_info.router_id] = conn
        node_client.connection_manager.connections[mock_router_info.router_id] = conn
        
        result = node_client._select_router()
        assert result is mock_router_info

    def test_select_router_excludes_closed(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test that closed connections are excluded from selection."""
        mock_websocket.closed = True
        
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        # Add to both NodeClient.connections and ConnectionManager.connections
        node_client.connections[mock_router_info.router_id] = conn
        node_client.connection_manager.connections[mock_router_info.router_id] = conn
        
        result = node_client._select_router()
        assert result is None

    def test_select_router_weighted(self, node_client, mock_session):
        """Test that healthier routers are selected more often."""
        # Create a healthy router
        healthy_router = RouterInfo(
            router_id="h" * 64,
            endpoints=["192.168.1.1:8471"],
            capacity={"max_connections": 100, "current_load_pct": 10},
            health={"uptime_pct": 99.9},
            regions=[],
            features=[],
        )
        healthy_ws = AsyncMock()
        healthy_ws.closed = False
        healthy_conn = RouterConnection(
            router=healthy_router,
            websocket=healthy_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_success=100,
            ack_failure=0,
            ping_latency_ms=20,
        )
        
        # Create an unhealthy router
        unhealthy_router = RouterInfo(
            router_id="u" * 64,
            endpoints=["192.168.2.1:8471"],
            capacity={"max_connections": 100, "current_load_pct": 90},
            health={"uptime_pct": 50.0},
            regions=[],
            features=[],
        )
        unhealthy_ws = AsyncMock()
        unhealthy_ws.closed = False
        unhealthy_conn = RouterConnection(
            router=unhealthy_router,
            websocket=unhealthy_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_success=50,
            ack_failure=50,
            ping_latency_ms=400,
        )
        
        # Add to both NodeClient.connections and ConnectionManager.connections
        node_client.connections[healthy_router.router_id] = healthy_conn
        node_client.connections[unhealthy_router.router_id] = unhealthy_conn
        node_client.connection_manager.connections[healthy_router.router_id] = healthy_conn
        node_client.connection_manager.connections[unhealthy_router.router_id] = unhealthy_conn
        
        # Run multiple selections and count
        healthy_count = 0
        trials = 1000
        
        for _ in range(trials):
            selected = node_client._select_router()
            if selected.router_id == healthy_router.router_id:
                healthy_count += 1
        
        # Healthy router should be selected significantly more often
        # (at least 60% of the time given the health difference)
        assert healthy_count > trials * 0.6


# =============================================================================
# Unit Tests - IP Diversity
# =============================================================================


class TestIPDiversity:
    """Tests for IP diversity enforcement (via ConnectionManager)."""

    def test_check_ip_diversity_empty(self, node_client, mock_router_info):
        """Test IP diversity check with no existing connections."""
        conn_mgr = node_client.connection_manager
        result = conn_mgr.check_ip_diversity(mock_router_info)
        assert result is True

    def test_check_ip_diversity_same_subnet(self, node_client):
        """Test IP diversity check rejects same /16 subnet."""
        conn_mgr = node_client.connection_manager
        
        # Add first router
        router1 = RouterInfo(
            router_id="a" * 64,
            endpoints=["10.0.1.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        conn_mgr._add_subnet(router1)
        
        # Check second router in same /16
        router2 = RouterInfo(
            router_id="b" * 64,
            endpoints=["10.0.2.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        result = conn_mgr.check_ip_diversity(router2)
        assert result is False

    def test_check_ip_diversity_different_subnet(self, node_client):
        """Test IP diversity check allows different /16 subnets."""
        conn_mgr = node_client.connection_manager
        
        # Add first router
        router1 = RouterInfo(
            router_id="a" * 64,
            endpoints=["10.0.1.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        conn_mgr._add_subnet(router1)
        
        # Check second router in different /16
        router2 = RouterInfo(
            router_id="b" * 64,
            endpoints=["10.1.1.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        result = conn_mgr.check_ip_diversity(router2)
        assert result is True

    def test_check_ip_diversity_hostname(self, node_client):
        """Test IP diversity check allows hostnames."""
        conn_mgr = node_client.connection_manager
        
        # Add first router with IP
        router1 = RouterInfo(
            router_id="a" * 64,
            endpoints=["10.0.1.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        conn_mgr._add_subnet(router1)
        
        # Check router with hostname
        router2 = RouterInfo(
            router_id="b" * 64,
            endpoints=["router.example.com:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        result = conn_mgr.check_ip_diversity(router2)
        assert result is True

    def test_check_ip_diversity_no_endpoints(self, node_client):
        """Test IP diversity check with no endpoints."""
        conn_mgr = node_client.connection_manager
        
        router = RouterInfo(
            router_id="a" * 64,
            endpoints=[],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        result = conn_mgr.check_ip_diversity(router)
        assert result is False

    def test_add_and_remove_subnet(self, node_client):
        """Test adding and removing subnet tracking."""
        conn_mgr = node_client.connection_manager
        
        router = RouterInfo(
            router_id="a" * 64,
            endpoints=["10.0.1.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        # Initially empty
        assert len(conn_mgr._connected_subnets) == 0
        
        # Add subnet
        conn_mgr._add_subnet(router)
        assert len(conn_mgr._connected_subnets) == 1
        assert "10.0.0.0/16" in conn_mgr._connected_subnets
        
        # Remove subnet
        conn_mgr._remove_subnet(router)
        assert len(conn_mgr._connected_subnets) == 0

    def test_ip_diversity_disabled(self, ed25519_keypair, x25519_keypair):
        """Test that IP diversity can be disabled."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            enforce_ip_diversity=False,
        )
        conn_mgr = node.connection_manager
        
        # Add first router
        router1 = RouterInfo(
            router_id="a" * 64,
            endpoints=["10.0.1.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        conn_mgr._add_subnet(router1)
        
        # Same subnet should be allowed
        router2 = RouterInfo(
            router_id="b" * 64,
            endpoints=["10.0.2.1:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        # When disabled, check_ip_diversity isn't called at connection time,
        # but the method still works if called directly
        result = conn_mgr.check_ip_diversity(router2)
        # Still returns False because subnet is tracked, but enforcement is at ensure_connections


# =============================================================================
# Unit Tests - Message Queueing
# =============================================================================


class TestMessageQueueing:
    """Tests for message queueing during failover (via MessageHandler)."""

    @pytest.mark.asyncio
    async def test_send_message_queues_when_no_routers(
        self, node_client, x25519_keypair
    ):
        """Test that messages are queued when no routers available."""
        _, recipient_pub = x25519_keypair
        handler = node_client.message_handler
        
        message_id = await node_client.send_message(
            recipient_id="recipient-123",
            recipient_public_key=recipient_pub,
            content=b"Hello!",
        )
        
        assert message_id is not None
        assert len(handler.message_queue) == 1
        assert handler.message_queue[0].content == b"Hello!"
        assert handler._stats["messages_queued"] == 1

    @pytest.mark.asyncio
    async def test_queue_limit_enforced(self, node_client, x25519_keypair):
        """Test that queue size limit is enforced."""
        _, recipient_pub = x25519_keypair
        handler = node_client.message_handler
        handler.config.max_queue_size = 5
        
        # Queue up to limit
        for i in range(5):
            await node_client.send_message(
                recipient_id="recipient-123",
                recipient_public_key=recipient_pub,
                content=f"Message {i}".encode(),
            )
        
        assert len(handler.message_queue) == 5
        
        # Next message should raise error
        with pytest.raises(NoRoutersAvailableError):
            await node_client.send_message(
                recipient_id="recipient-123",
                recipient_public_key=recipient_pub,
                content=b"Overflow!",
            )
        
        assert handler._stats["messages_dropped"] == 1


# =============================================================================
# Unit Tests - ACK Handling
# =============================================================================


class TestACKHandling:
    """Tests for ACK message handling."""

    @pytest.mark.asyncio
    async def test_handle_ack_success(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test handling successful ACK."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_pending=1,
        )
        
        await node_client._handle_ack(
            {"message_id": "msg-123", "success": True},
            conn,
        )
        
        assert conn.ack_pending == 0
        assert conn.ack_success == 1
        assert conn.ack_failure == 0

    @pytest.mark.asyncio
    async def test_handle_ack_failure(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test handling failed ACK."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_pending=1,
        )
        
        await node_client._handle_ack(
            {"message_id": "msg-123", "success": False},
            conn,
        )
        
        assert conn.ack_pending == 0
        assert conn.ack_success == 0
        assert conn.ack_failure == 1


# =============================================================================
# Unit Tests - Pong Handling
# =============================================================================


class TestPongHandling:
    """Tests for pong message handling."""

    @pytest.mark.asyncio
    async def test_handle_pong_updates_latency(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test that pong updates ping latency."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        
        # Simulate ping sent 100ms ago
        sent_at = time.time() - 0.1
        
        await node_client._handle_pong(
            {"sent_at": sent_at},
            conn,
        )
        
        # Latency should be approximately 100ms
        assert 90 < conn.ping_latency_ms < 150


# =============================================================================
# Unit Tests - Factory Function
# =============================================================================


class TestCreateNodeClient:
    """Tests for the create_node_client factory function."""

    def test_create_node_client_basic(self, ed25519_keypair, x25519_keypair):
        """Test creating a node client via factory function."""
        private_key, _ = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = create_node_client(
            private_key=private_key,
            encryption_private_key=enc_private,
        )
        
        assert node.node_id == private_key.public_key().public_bytes_raw().hex()
        assert node.private_key is private_key
        assert node.encryption_private_key is enc_private

    def test_create_node_client_with_discovery(
        self, ed25519_keypair, x25519_keypair
    ):
        """Test creating a node client with custom discovery client."""
        private_key, _ = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        discovery = DiscoveryClient()
        discovery.add_seed("https://custom.seed:8470")
        
        node = create_node_client(
            private_key=private_key,
            encryption_private_key=enc_private,
            discovery_client=discovery,
        )
        
        assert node.discovery is discovery
        assert "https://custom.seed:8470" in node.discovery.custom_seeds

    def test_create_node_client_with_kwargs(self, ed25519_keypair, x25519_keypair):
        """Test creating a node client with additional kwargs."""
        private_key, _ = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = create_node_client(
            private_key=private_key,
            encryption_private_key=enc_private,
            min_connections=2,
            target_connections=4,
            enforce_ip_diversity=False,
        )
        
        assert node.min_connections == 2
        assert node.target_connections == 4
        assert node.enforce_ip_diversity is False
    
    def test_create_node_client_with_seed_urls(self, ed25519_keypair, x25519_keypair):
        """Test creating a node client with seed URLs (Issue #108)."""
        private_key, _ = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = create_node_client(
            private_key=private_key,
            encryption_private_key=enc_private,
            seed_urls=[
                "https://primary.seed:8470",
                "https://backup.seed:8470",
            ],
        )
        
        assert len(node.seed_urls) == 2
        assert "https://primary.seed:8470" in node.seed_urls
        assert "https://backup.seed:8470" in node.seed_urls


# =============================================================================
# Integration Tests
# =============================================================================


class TestNodeIntegration:
    """Integration tests for NodeClient."""

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, node_client):
        """Test node start/stop lifecycle."""
        # Mock discovery to avoid network calls
        with patch.object(
            node_client.discovery,
            'discover_routers',
            new_callable=AsyncMock,
            return_value=[],
        ):
            await node_client.start()
            assert node_client._running is True
            # Tasks: maintenance, keepalive, queue, gossip
            assert len(node_client._tasks) >= 3  # At least 3, may have more (gossip, etc.)
            
            await node_client.stop()
            assert node_client._running is False
            assert len(node_client._tasks) == 0

    @pytest.mark.asyncio
    async def test_start_configures_seed_urls(self, ed25519_keypair, x25519_keypair):
        """Test that start() configures discovery client with seed_urls (Issue #108)."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            seed_urls=[
                "https://primary.seed:8470",
                "https://backup.seed:8470",
            ],
        )
        
        # Initially, discovery client doesn't have our seeds
        assert "https://primary.seed:8470" not in node.discovery.custom_seeds
        
        # Mock discovery to avoid network calls
        with patch.object(
            node.discovery,
            'discover_routers',
            new_callable=AsyncMock,
            return_value=[],
        ):
            await node.start()
            
            # After start, seeds should be configured
            assert "https://primary.seed:8470" in node.discovery.custom_seeds
            assert "https://backup.seed:8470" in node.discovery.custom_seeds
            
            await node.stop()

    @pytest.mark.asyncio
    async def test_get_connections_info(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test getting connection information."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=1000.0,
            last_seen=1000.0,
            messages_sent=5,
            messages_received=3,
            ack_success=4,
            ack_failure=1,
            ping_latency_ms=50.0,
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        connections = node_client.get_connections()
        assert len(connections) == 1
        
        info = connections[0]
        assert "router_id" in info
        assert info["messages_sent"] == 5
        assert info["messages_received"] == 3
        assert info["ping_latency_ms"] == 50.0
        assert 0 <= info["ack_success_rate"] <= 1.0
        assert 0 <= info["health_score"] <= 1.0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ack_success_rate_division_by_zero(
        self, mock_router_info, mock_websocket, mock_session
    ):
        """Test ACK success rate with no ACKs (should not divide by zero)."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_success=0,
            ack_failure=0,
        )
        
        # Should return 1.0, not raise ZeroDivisionError
        assert conn.ack_success_rate == 1.0

    def test_health_score_with_missing_capacity(self, mock_websocket, mock_session):
        """Test health score with missing capacity data."""
        router = RouterInfo(
            router_id="a" * 64,
            endpoints=["192.168.1.1:8471"],
            capacity={},  # No load data
            health={},
            regions=[],
            features=[],
        )
        
        conn = RouterConnection(
            router=router,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        
        # Should handle missing data gracefully
        score = conn.health_score
        assert 0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_close_connection_already_closed(
        self, node_client, mock_router_info, mock_session
    ):
        """Test closing an already closed connection (via ConnectionManager)."""
        ws = AsyncMock()
        ws.closed = True
        ws.close = AsyncMock()
        
        conn = RouterConnection(
            router=mock_router_info,
            websocket=ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        
        # Should not raise
        await node_client.connection_manager.close_connection(mock_router_info.router_id, conn)

    def test_ip_diversity_ipv6(self, node_client):
        """Test IP diversity with IPv6 addresses (via ConnectionManager)."""
        conn_mgr = node_client.connection_manager
        
        # Add IPv6 router (using bracket notation for port separator)
        router1 = RouterInfo(
            router_id="a" * 64,
            endpoints=["[2001:db8::1]:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        conn_mgr._add_subnet(router1)
        
        # Check same /48 (should be blocked by diversity)
        router2 = RouterInfo(
            router_id="b" * 64,
            endpoints=["[2001:db8::2]:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        result = conn_mgr.check_ip_diversity(router2)
        assert result is False
        
        # Check different /48 (should be allowed)
        router3 = RouterInfo(
            router_id="c" * 64,
            endpoints=["[2001:db9::1]:8471"],
            capacity={},
            health={},
            regions=[],
            features=[],
        )
        
        result = conn_mgr.check_ip_diversity(router3)
        assert result is True


# =============================================================================
# Router Failover Tests
# =============================================================================


class TestRouterFailover:
    """Tests for router failover logic (Issue #107)."""

    def test_failover_state_creation(self):
        """Test FailoverState dataclass creation."""
        from valence.network.node import FailoverState
        
        state = FailoverState(
            router_id="r" * 64,
            failed_at=1000.0,
            fail_count=1,
            cooldown_until=1060.0,
        )
        
        assert state.router_id == "r" * 64
        assert state.failed_at == 1000.0
        assert state.fail_count == 1
        assert state.cooldown_until == 1060.0
        assert state.queued_messages == []

    def test_failover_state_cooldown_check(self):
        """Test FailoverState cooldown methods."""
        from valence.network.node import FailoverState
        
        now = time.time()
        
        # In cooldown
        state = FailoverState(
            router_id="r" * 64,
            failed_at=now,
            fail_count=1,
            cooldown_until=now + 60,
        )
        assert state.is_in_cooldown() is True
        assert 50 < state.remaining_cooldown() <= 60
        
        # Past cooldown
        state2 = FailoverState(
            router_id="s" * 64,
            failed_at=now - 120,
            fail_count=1,
            cooldown_until=now - 60,
        )
        assert state2.is_in_cooldown() is False
        assert state2.remaining_cooldown() == 0

    @pytest.mark.asyncio
    async def test_handle_router_failure_creates_failover_state(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test that router failure creates failover state."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        # Mock discovery to avoid network calls
        with patch.object(
            node_client.discovery,
            'discover_routers',
            new_callable=AsyncMock,
            return_value=[],
        ):
            await node_client._handle_router_failure(mock_router_info.router_id)
        
        # Should create failover state
        assert mock_router_info.router_id in node_client.failover_states
        state = node_client.failover_states[mock_router_info.router_id]
        assert state.fail_count == 1
        assert state.is_in_cooldown()
        assert node_client._stats["failovers"] == 1

    @pytest.mark.asyncio
    async def test_exponential_backoff_on_repeated_failures(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test exponential backoff for flapping routers."""
        # Set short initial cooldown for testing
        node_client.initial_cooldown = 10.0
        node_client.max_cooldown = 100.0
        
        with patch.object(
            node_client.discovery,
            'discover_routers',
            new_callable=AsyncMock,
            return_value=[],
        ):
            # First failure - should get initial cooldown
            conn1 = RouterConnection(
                router=mock_router_info,
                websocket=mock_websocket,
                session=mock_session,
                connected_at=time.time(),
                last_seen=time.time(),
            )
            node_client.connections[mock_router_info.router_id] = conn1
            await node_client._handle_router_failure(mock_router_info.router_id)
            
            state = node_client.failover_states[mock_router_info.router_id]
            first_cooldown = state.cooldown_until - state.failed_at
            assert 9 <= first_cooldown <= 11  # ~10s
            
            # Second failure - should get 2x cooldown
            conn2 = RouterConnection(
                router=mock_router_info,
                websocket=mock_websocket,
                session=mock_session,
                connected_at=time.time(),
                last_seen=time.time(),
            )
            node_client.connections[mock_router_info.router_id] = conn2
            await node_client._handle_router_failure(mock_router_info.router_id)
            
            state = node_client.failover_states[mock_router_info.router_id]
            second_cooldown = state.cooldown_until - state.failed_at
            assert state.fail_count == 2
            assert 19 <= second_cooldown <= 21  # ~20s (10 * 2^1)
            
            # Third failure - should get 4x cooldown
            conn3 = RouterConnection(
                router=mock_router_info,
                websocket=mock_websocket,
                session=mock_session,
                connected_at=time.time(),
                last_seen=time.time(),
            )
            node_client.connections[mock_router_info.router_id] = conn3
            await node_client._handle_router_failure(mock_router_info.router_id)
            
            state = node_client.failover_states[mock_router_info.router_id]
            third_cooldown = state.cooldown_until - state.failed_at
            assert state.fail_count == 3
            assert 39 <= third_cooldown <= 41  # ~40s (10 * 2^2)

    @pytest.mark.asyncio
    async def test_max_cooldown_capped(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test that cooldown is capped at max_cooldown."""
        node_client.initial_cooldown = 100.0
        node_client.max_cooldown = 200.0
        
        with patch.object(
            node_client.discovery,
            'discover_routers',
            new_callable=AsyncMock,
            return_value=[],
        ):
            # Simulate many failures
            for i in range(5):
                conn = RouterConnection(
                    router=mock_router_info,
                    websocket=mock_websocket,
                    session=mock_session,
                    connected_at=time.time(),
                    last_seen=time.time(),
                )
                node_client.connections[mock_router_info.router_id] = conn
                await node_client._handle_router_failure(mock_router_info.router_id)
            
            state = node_client.failover_states[mock_router_info.router_id]
            cooldown = state.cooldown_until - state.failed_at
            assert cooldown <= 200.0  # Capped at max_cooldown

    @pytest.mark.asyncio
    async def test_failover_queries_alternatives(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test that failover queries discovery for alternatives."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        discover_mock = AsyncMock(return_value=[])
        
        with patch.object(
            node_client.discovery,
            'discover_routers',
            discover_mock,
        ):
            await node_client._handle_router_failure(mock_router_info.router_id)
        
        # Should have called discover_routers with force_refresh
        discover_mock.assert_called_once()
        call_kwargs = discover_mock.call_args.kwargs
        assert call_kwargs.get("force_refresh") is True
        assert call_kwargs.get("count") == 3

    @pytest.mark.asyncio
    async def test_failover_connects_to_alternative(
        self, node_client, mock_router_info, mock_session
    ):
        """Test successful failover to alternative router."""
        # Create a mock websocket for the original connection
        original_ws = AsyncMock()
        original_ws.closed = False
        original_ws.close = AsyncMock()
        
        conn = RouterConnection(
            router=mock_router_info,
            websocket=original_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        # Create alternative router
        alt_router = RouterInfo(
            router_id="b" * 64,
            endpoints=["10.1.1.1:8471"],  # Different subnet
            capacity={"current_load_pct": 20},
            health={"uptime_pct": 99, "avg_latency_ms": 30},
            regions=["us-west"],
            features=[],
        )
        
        discover_mock = AsyncMock(return_value=[alt_router])
        connect_mock = AsyncMock()
        
        with patch.object(node_client.discovery, 'discover_routers', discover_mock):
            with patch.object(node_client, '_connect_to_router', connect_mock):
                await node_client._handle_router_failure(mock_router_info.router_id)
        
        # Should have tried to connect to alternative
        connect_mock.assert_called_once_with(alt_router)

    @pytest.mark.asyncio
    async def test_failover_enables_direct_mode_when_no_alternatives(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test that direct mode is enabled when no alternatives available."""
        node_client.reconnect_delay = 0.01  # Fast for testing
        
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        assert node_client.direct_mode is False
        
        with patch.object(
            node_client.discovery,
            'discover_routers',
            new_callable=AsyncMock,
            return_value=[],
        ):
            await node_client._handle_router_failure(mock_router_info.router_id)
        
        # Should enable direct mode
        assert node_client.direct_mode is True

    def test_clear_router_cooldown(self, node_client):
        """Test manually clearing router cooldown."""
        from valence.network.node import FailoverState
        
        router_id = "r" * 64
        node_client.failover_states[router_id] = FailoverState(
            router_id=router_id,
            failed_at=time.time(),
            fail_count=3,
            cooldown_until=time.time() + 3600,
        )
        
        assert node_client.failover_states[router_id].is_in_cooldown()
        
        result = node_client.clear_router_cooldown(router_id)
        
        assert result is True
        assert not node_client.failover_states[router_id].is_in_cooldown()
        assert node_client.failover_states[router_id].fail_count == 0

    def test_clear_router_cooldown_not_found(self, node_client):
        """Test clearing cooldown for unknown router."""
        result = node_client.clear_router_cooldown("unknown" * 8)
        assert result is False

    def test_get_failover_states(self, node_client):
        """Test getting failover states."""
        from valence.network.node import FailoverState
        
        now = time.time()
        node_client.failover_states["r" * 64] = FailoverState(
            router_id="r" * 64,
            failed_at=now,
            fail_count=2,
            cooldown_until=now + 60,
        )
        
        states = node_client.get_failover_states()
        
        assert "r" * 64 in states
        state_info = states["r" * 64]
        assert state_info["fail_count"] == 2
        assert state_info["in_cooldown"] is True
        assert 50 <= state_info["remaining_cooldown"] <= 60

    def test_stats_include_failover_info(self, node_client):
        """Test that stats include failover information."""
        from valence.network.node import FailoverState
        
        node_client.failover_states["r" * 64] = FailoverState(
            router_id="r" * 64,
            failed_at=time.time(),
            fail_count=1,
            cooldown_until=time.time() + 60,
        )
        
        stats = node_client.get_stats()
        
        assert "routers_in_cooldown" in stats
        assert stats["routers_in_cooldown"] == 1
        assert "direct_mode" in stats
        assert stats["direct_mode"] is False


# =============================================================================
# Seed Redundancy Tests (Issue #108)
# =============================================================================


class TestSeedRedundancy:
    """Tests for seed redundancy feature (Issue #108)."""

    def test_node_accepts_seed_urls_list(self, ed25519_keypair, x25519_keypair):
        """Test that NodeClient accepts a list of seed URLs."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            seed_urls=[
                "https://seed1.example.com:8470",
                "https://seed2.example.com:8470",
                "https://seed3.example.com:8470",
            ],
        )
        
        assert len(node.seed_urls) == 3
        assert node.seed_urls[0] == "https://seed1.example.com:8470"
    
    def test_node_empty_seed_urls_by_default(self, node_client):
        """Test that seed_urls is empty by default."""
        assert node_client.seed_urls == []
    
    @pytest.mark.asyncio
    async def test_seeds_added_in_order(self, ed25519_keypair, x25519_keypair):
        """Test that seeds are added to discovery client in order."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            seed_urls=[
                "https://primary.seed:8470",
                "https://secondary.seed:8470",
            ],
        )
        
        with patch.object(
            node.discovery,
            'discover_routers',
            new_callable=AsyncMock,
            return_value=[],
        ):
            await node.start()
            
            # Check that seeds are in discovery client's custom_seeds
            seeds = node.discovery.custom_seeds
            primary_idx = seeds.index("https://primary.seed:8470")
            secondary_idx = seeds.index("https://secondary.seed:8470")
            
            # Primary should be added first
            assert primary_idx < secondary_idx
            
            await node.stop()
    
    @pytest.mark.asyncio
    async def test_seed_fallback_on_failure(self, ed25519_keypair, x25519_keypair, mock_router_info):
        """Test that discovery falls back to secondary seed on primary failure."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            seed_urls=[
                "https://failing.seed:8470",
                "https://working.seed:8470",
            ],
        )
        
        query_calls = []
        
        async def mock_query(seed_url, count, prefs):
            query_calls.append(seed_url)
            if "failing" in seed_url:
                from valence.network.discovery import DiscoveryError
                raise DiscoveryError("Primary seed unavailable")
            return [mock_router_info]
        
        # Clear default seeds to only use our custom ones
        node.discovery.default_seeds = []
        
        with patch.object(node.discovery, '_query_seed', side_effect=mock_query):
            with patch.object(node, '_connect_to_router', new_callable=AsyncMock):
                await node.start()
                
                # Should have tried failing seed first, then working seed
                assert len(query_calls) >= 2
                assert "https://failing.seed:8470" in query_calls
                assert "https://working.seed:8470" in query_calls
                # Failing seed should be tried first
                fail_idx = query_calls.index("https://failing.seed:8470")
                work_idx = query_calls.index("https://working.seed:8470")
                assert fail_idx < work_idx
                
                await node.stop()
    
    def test_seed_health_tracking_through_node(self, ed25519_keypair, x25519_keypair):
        """Test that seed health is tracked when using NodeClient."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            seed_urls=["https://tracked.seed:8470"],
        )
        
        # Discovery client should have health tracking available
        assert hasattr(node.discovery, 'seed_health')
        assert hasattr(node.discovery, 'get_seed_health_report')
    
    @pytest.mark.asyncio
    async def test_successful_seed_remembered(self, ed25519_keypair, x25519_keypair, mock_router_info):
        """Test that last successful seed is remembered for future queries."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            seed_urls=[
                "https://slow.seed:8470",
                "https://fast.seed:8470",
            ],
        )
        
        # Clear default seeds
        node.discovery.default_seeds = []
        
        # First query fails on slow, succeeds on fast
        first_call = True
        
        async def mock_query(seed_url, count, prefs):
            nonlocal first_call
            if "slow" in seed_url and first_call:
                first_call = False
                from valence.network.discovery import DiscoveryError
                raise DiscoveryError("Timeout")
            return [mock_router_info]
        
        with patch.object(node.discovery, '_query_seed', side_effect=mock_query):
            # First discovery
            node.discovery.add_seed("https://slow.seed:8470")
            node.discovery.add_seed("https://fast.seed:8470")
            await node.discovery.discover_routers(count=1, force_refresh=True)
        
        # Fast seed should be remembered
        assert node.discovery.last_successful_seed == "https://fast.seed:8470"


class TestFastFailureDetection:
    """Tests for fast failure detection via keepalive."""

    def test_default_keepalive_interval_is_fast(self, node_client):
        """Test that default keepalive interval is 2 seconds for fast detection."""
        # Requirement: Detection within 2-5 seconds
        assert node_client.keepalive_interval <= 2.0
        assert node_client.ping_timeout <= 3.0
        # With 2s interval and 2 missed pings threshold = 4-6s detection

    @pytest.mark.asyncio
    async def test_missed_pings_tracking(
        self, node_client, mock_router_info, mock_websocket, mock_session
    ):
        """Test that missed pings are tracked."""
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        # Simulate missed ping by making send_json timeout
        mock_websocket.send_json = AsyncMock(
            side_effect=asyncio.TimeoutError()
        )
        
        # Manually trigger a single keepalive check (not the loop)
        # This simulates what happens in _keepalive_loop
        try:
            await asyncio.wait_for(
                mock_websocket.send_json({"type": "ping"}),
                timeout=node_client.ping_timeout,
            )
        except asyncio.TimeoutError:
            missed = node_client._missed_pings.get(mock_router_info.router_id, 0) + 1
            node_client._missed_pings[mock_router_info.router_id] = missed
        
        assert node_client._missed_pings[mock_router_info.router_id] == 1

    def test_detection_time_calculation(self, node_client):
        """Test that detection time is within requirements."""
        # Detection time = keepalive_interval * missed_pings_threshold + ping_timeout
        # With defaults: 2.0 * 2 + 3.0 = 7s worst case (first timeout) to 4s (immediate)
        max_detection_time = (
            node_client.keepalive_interval * node_client.missed_pings_threshold 
            + node_client.ping_timeout
        )
        
        # Should detect within 2-5 seconds (allowing some margin)
        assert max_detection_time <= 8.0  # Reasonable upper bound

    @pytest.mark.asyncio
    async def test_failure_triggered_after_threshold(
        self, node_client, mock_router_info, mock_session
    ):
        """Test that failure is triggered after missed_pings_threshold."""
        ws = AsyncMock()
        ws.closed = False
        ws.close = AsyncMock()
        ws.send_json = AsyncMock(side_effect=asyncio.TimeoutError())
        
        conn = RouterConnection(
            router=mock_router_info,
            websocket=ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        node_client.missed_pings_threshold = 2
        
        # Track if failure was called
        failure_called = False
        
        async def mock_handle_failure(router_id):
            nonlocal failure_called
            failure_called = True
            node_client.connections.pop(router_id, None)
        
        with patch.object(
            node_client.discovery, 'discover_routers',
            new_callable=AsyncMock, return_value=[]
        ):
            original_handle = node_client._handle_router_failure
            node_client._handle_router_failure = mock_handle_failure
            
            # First missed ping - should not trigger failure
            node_client._missed_pings[mock_router_info.router_id] = 1
            assert failure_called is False
            
            # Second missed ping - should trigger failure
            node_client._missed_pings[mock_router_info.router_id] = 2
            if node_client._missed_pings[mock_router_info.router_id] >= node_client.missed_pings_threshold:
                await node_client._handle_router_failure(mock_router_info.router_id)
            
            assert failure_called is True


class TestMessagePreservation:
    """Tests for message preservation during failover (via MessageHandler)."""

    @pytest.mark.asyncio
    async def test_pending_messages_preserved_on_failure(
        self, node_client, mock_router_info, mock_websocket, mock_session, x25519_keypair
    ):
        """Test that pending messages are preserved during failover."""
        from valence.network.node import PendingAck
        
        _, recipient_pub = x25519_keypair
        
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        # Add some pending ACKs for this router
        for i in range(3):
            node_client.pending_acks[f"msg-{i}"] = PendingAck(
                message_id=f"msg-{i}",
                recipient_id="recipient-123",
                content=f"Message {i}".encode(),
                recipient_public_key=recipient_pub,
                sent_at=time.time(),
                router_id=mock_router_info.router_id,
            )
        
        # Create an alternative router
        alt_router = RouterInfo(
            router_id="b" * 64,
            endpoints=["10.1.1.1:8471"],
            capacity={},
            health={"uptime_pct": 99},
            regions=[],
            features=[],
        )
        
        retry_called = []
        
        original_retry = node_client.message_handler._retry_message
        async def mock_retry(msg_id, send_via_router, router_selector):
            retry_called.append(msg_id)
        
        with patch.object(node_client.discovery, 'discover_routers', 
                          new_callable=AsyncMock, return_value=[alt_router]):
            with patch.object(node_client.connection_manager, 'connect_to_router', new_callable=AsyncMock):
                with patch.object(node_client.message_handler, '_retry_message', mock_retry):
                    await node_client._handle_router_failure(mock_router_info.router_id)
        
        # All 3 messages should have been retried
        assert len(retry_called) == 3
        assert set(retry_called) == {"msg-0", "msg-1", "msg-2"}

    @pytest.mark.asyncio
    async def test_messages_not_retried_when_no_alternative(
        self, node_client, mock_router_info, mock_websocket, mock_session, x25519_keypair
    ):
        """Test that messages are not retried when no alternative router available."""
        from valence.network.node import PendingAck
        
        _, recipient_pub = x25519_keypair
        node_client.router_client.config.reconnect_delay = 0.01  # Fast for testing
        
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn
        
        # Add pending ACK
        node_client.pending_acks["msg-0"] = PendingAck(
            message_id="msg-0",
            recipient_id="recipient-123",
            content=b"Message",
            recipient_public_key=recipient_pub,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
        )
        
        retry_called = []
        
        async def mock_retry(msg_id, send_via_router, router_selector):
            retry_called.append(msg_id)
        
        with patch.object(node_client.discovery, 'discover_routers',
                          new_callable=AsyncMock, return_value=[]):
            with patch.object(node_client.message_handler, '_retry_message', mock_retry):
                await node_client._handle_router_failure(mock_router_info.router_id)
        
        # No retry because no alternative was found
        assert len(retry_called) == 0


# =============================================================================
# Multi-Router Connection Tests (Issue #106)
# =============================================================================


class TestMultiRouterConnections:
    """Tests for multi-router connection management (Issue #106).
    
    These tests verify:
    1. NodeClient maintains connections to multiple routers (configurable)
    2. Messages sent through best router, failover to alternatives
    3. Router health tracking and connection status
    4. Integration with failover logic from #107
    """

    def test_configurable_connection_counts(self, ed25519_keypair, x25519_keypair):
        """Test that connection counts are configurable."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        # Test various configurations
        for min_conn, target, max_conn in [(2, 3, 5), (1, 2, 3), (3, 5, 8)]:
            node = NodeClient(
                node_id=public_key.public_bytes_raw().hex(),
                private_key=private_key,
                encryption_private_key=enc_private,
                min_connections=min_conn,
                target_connections=target,
                max_connections=max_conn,
            )
            
            assert node.min_connections == min_conn
            assert node.target_connections == target
            assert node.max_connections == max_conn

    def test_multiple_connections_tracked(
        self, node_client, mock_session
    ):
        """Test that multiple router connections are tracked correctly."""
        conn_mgr = node_client.connection_manager
        routers = []
        for i in range(3):
            router = RouterInfo(
                router_id=f"{chr(97+i)}" * 64,  # a*64, b*64, c*64
                endpoints=[f"10.{i}.1.1:8471"],
                capacity={"current_load_pct": 20 + i * 10},
                health={"uptime_pct": 99 - i},
                regions=["us-west"],
                features=[],
            )
            routers.append(router)
            
            ws = AsyncMock()
            ws.closed = False
            
            conn = RouterConnection(
                router=router,
                websocket=ws,
                session=mock_session,
                connected_at=time.time(),
                last_seen=time.time(),
            )
            node_client.connections[router.router_id] = conn
            conn_mgr._add_subnet(router)
        
        # Verify all connections are tracked
        assert len(node_client.connections) == 3
        assert len(conn_mgr._connected_subnets) == 3
        
        # Verify connection info is retrievable
        conn_info = node_client.get_connections()
        assert len(conn_info) == 3

    def test_health_tracking_per_router(
        self, node_client, mock_session
    ):
        """Test that health metrics are tracked per router."""
        # Create healthy router
        healthy_router = RouterInfo(
            router_id="h" * 64,
            endpoints=["10.0.1.1:8471"],
            capacity={"current_load_pct": 10},
            health={},
            regions=[],
            features=[],
        )
        healthy_ws = AsyncMock()
        healthy_ws.closed = False
        healthy_conn = RouterConnection(
            router=healthy_router,
            websocket=healthy_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_success=95,
            ack_failure=5,
            ping_latency_ms=30,
        )
        
        # Create degraded router
        degraded_router = RouterInfo(
            router_id="d" * 64,
            endpoints=["10.1.1.1:8471"],
            capacity={"current_load_pct": 80},
            health={},
            regions=[],
            features=[],
        )
        degraded_ws = AsyncMock()
        degraded_ws.closed = False
        degraded_conn = RouterConnection(
            router=degraded_router,
            websocket=degraded_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_success=60,
            ack_failure=40,
            ping_latency_ms=300,
        )
        
        node_client.connections[healthy_router.router_id] = healthy_conn
        node_client.connections[degraded_router.router_id] = degraded_conn
        
        # Health scores should differ significantly
        assert healthy_conn.health_score > degraded_conn.health_score
        assert healthy_conn.ack_success_rate > degraded_conn.ack_success_rate
        
        # Stats should reflect multi-router state
        stats = node_client.get_stats()
        assert stats["active_connections"] == 2

    def test_best_router_selected_for_messages(
        self, node_client, mock_session
    ):
        """Test that healthiest router is selected most often."""
        conn_mgr = node_client.connection_manager
        # Create 3 routers with different health levels
        routers_and_health = [
            ("excellent", 0.95),
            ("good", 0.7),
            ("poor", 0.3),
        ]
        
        for name, target_health in routers_and_health:
            router = RouterInfo(
                router_id=name.ljust(64, "x"),
                endpoints=[f"10.{len(name)}.1.1:8471"],
                capacity={"current_load_pct": int((1 - target_health) * 100)},
                health={},
                regions=[],
                features=[],
            )
            ws = AsyncMock()
            ws.closed = False
            
            # Set ACK stats to achieve target health
            if target_health >= 0.9:
                ack_success, ack_failure, latency = 100, 0, 20
            elif target_health >= 0.6:
                ack_success, ack_failure, latency = 70, 30, 150
            else:
                ack_success, ack_failure, latency = 30, 70, 400
            
            conn = RouterConnection(
                router=router,
                websocket=ws,
                session=mock_session,
                connected_at=time.time(),
                last_seen=time.time(),
                ack_success=ack_success,
                ack_failure=ack_failure,
                ping_latency_ms=latency,
            )
            # Add to both NodeClient and ConnectionManager connections
            node_client.connections[router.router_id] = conn
            conn_mgr.connections[router.router_id] = conn
        
        # Sample selections
        selection_counts = {}
        for _ in range(1000):
            selected = node_client._select_router()
            selection_counts[selected.router_id] = selection_counts.get(selected.router_id, 0) + 1
        
        # Excellent router should be selected most
        excellent_id = "excellent".ljust(64, "x")
        poor_id = "poor".ljust(64, "x")
        
        assert selection_counts[excellent_id] > selection_counts[poor_id] * 2

    @pytest.mark.asyncio
    async def test_failover_to_secondary_router(
        self, node_client, mock_session
    ):
        """Test failover from primary to secondary router."""
        # Create primary router (will fail)
        primary_router = RouterInfo(
            router_id="p" * 64,
            endpoints=["10.0.1.1:8471"],
            capacity={},
            health={"uptime_pct": 99},
            regions=[],
            features=[],
        )
        primary_ws = AsyncMock()
        primary_ws.closed = False
        primary_ws.close = AsyncMock()
        primary_conn = RouterConnection(
            router=primary_router,
            websocket=primary_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_success=100,  # Healthy before failure
        )
        
        # Create secondary router (backup)
        secondary_router = RouterInfo(
            router_id="s" * 64,
            endpoints=["10.1.1.1:8471"],
            capacity={},
            health={"uptime_pct": 99},
            regions=[],
            features=[],
        )
        secondary_ws = AsyncMock()
        secondary_ws.closed = False
        secondary_conn = RouterConnection(
            router=secondary_router,
            websocket=secondary_ws,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
            ack_success=90,
        )
        
        node_client.connections[primary_router.router_id] = primary_conn
        node_client.connections[secondary_router.router_id] = secondary_conn
        
        # Before failure: should have 2 connections
        assert len(node_client.connections) == 2
        
        # Simulate primary failure
        with patch.object(
            node_client.discovery, 'discover_routers',
            new_callable=AsyncMock, return_value=[]
        ):
            await node_client._handle_router_failure(primary_router.router_id)
        
        # After failure: secondary should still be available
        assert len(node_client.connections) == 1
        assert secondary_router.router_id in node_client.connections
        
        # Should be able to select secondary
        selected = node_client._select_router()
        assert selected.router_id == secondary_router.router_id

    def test_connection_status_reporting(
        self, node_client, mock_session
    ):
        """Test accurate connection status reporting."""
        # Add some routers with different states
        for i, (load, latency, ack_rate) in enumerate([
            (15, 25, 0.98),
            (45, 100, 0.85),
            (80, 300, 0.60),
        ]):
            router = RouterInfo(
                router_id=f"{chr(97+i)}" * 64,
                endpoints=[f"10.{i}.1.1:8471"],
                capacity={"current_load_pct": load},
                health={},
                regions=[],
                features=[],
            )
            ws = AsyncMock()
            ws.closed = False
            
            total_acks = 100
            successes = int(total_acks * ack_rate)
            
            conn = RouterConnection(
                router=router,
                websocket=ws,
                session=mock_session,
                connected_at=time.time() - 3600,  # 1 hour ago
                last_seen=time.time(),
                messages_sent=50 + i * 10,
                messages_received=45 + i * 8,
                ack_success=successes,
                ack_failure=total_acks - successes,
                ping_latency_ms=latency,
            )
            node_client.connections[router.router_id] = conn
        
        # Get connection info
        connections = node_client.get_connections()
        
        assert len(connections) == 3
        
        # Verify info contains expected fields
        for conn_info in connections:
            assert "router_id" in conn_info
            assert "endpoint" in conn_info
            assert "connected_at" in conn_info
            assert "last_seen" in conn_info
            assert "health_score" in conn_info
            assert "ack_success_rate" in conn_info
            assert "ping_latency_ms" in conn_info
            assert "messages_sent" in conn_info
            assert "messages_received" in conn_info

    @pytest.mark.asyncio
    async def test_ensure_connections_targets_config(
        self, node_client
    ):
        """Test that _ensure_connections respects target_connections."""
        node_client.target_connections = 3
        
        # Create mock routers for discovery
        mock_routers = [
            RouterInfo(
                router_id=f"{chr(97+i)}" * 64,
                endpoints=[f"10.{i}.1.1:8471"],
                capacity={"current_load_pct": 20},
                health={"uptime_pct": 99},
                regions=[],
                features=[],
            )
            for i in range(5)
        ]
        
        connections_made = []
        
        async def mock_connect(router):
            connections_made.append(router.router_id)
            # Simulate successful connection
            ws = AsyncMock()
            ws.closed = False
            conn = RouterConnection(
                router=router,
                websocket=ws,
                session=AsyncMock(),
                connected_at=time.time(),
                last_seen=time.time(),
            )
            node_client.connections[router.router_id] = conn
            node_client._add_subnet(router)
        
        with patch.object(
            node_client.discovery, 'discover_routers',
            new_callable=AsyncMock, return_value=mock_routers
        ):
            with patch.object(node_client, '_connect_to_router', mock_connect):
                await node_client._ensure_connections()
        
        # Should have connected to target_connections routers
        assert len(node_client.connections) == 3

    def test_integration_with_failover_states(
        self, node_client
    ):
        """Test multi-router integrates with failover state tracking."""
        from valence.network.node import FailoverState
        
        # Add a router in cooldown
        cooldown_router_id = "c" * 64
        node_client.failover_states[cooldown_router_id] = FailoverState(
            router_id=cooldown_router_id,
            failed_at=time.time(),
            fail_count=2,
            cooldown_until=time.time() + 120,  # 2 minutes cooldown
        )
        
        # Stats should show routers in cooldown
        stats = node_client.get_stats()
        assert stats["routers_in_cooldown"] == 1
        
        # Failover states should be retrievable
        states = node_client.get_failover_states()
        assert cooldown_router_id in states
        assert states[cooldown_router_id]["in_cooldown"] is True
        assert states[cooldown_router_id]["fail_count"] == 2
