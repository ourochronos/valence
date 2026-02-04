"""Tests for the Valence Router Node.

These tests verify:
- Connection handling (identify, disconnect)
- Message relay (direct delivery, offline queueing)
- TTL handling (decrement, expiry)
- Queue management (size limits, age limits)
- Health/status endpoints
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.network.router import Connection, QueuedMessage, RouterNode


# =============================================================================
# Unit Tests - Data Classes
# =============================================================================


class TestConnection:
    """Tests for the Connection dataclass."""

    def test_connection_creation(self):
        """Test creating a Connection instance."""
        ws = MagicMock()
        conn = Connection(
            node_id="node-123",
            websocket=ws,
            connected_at=1000.0,
            last_seen=1000.0,
        )
        assert conn.node_id == "node-123"
        assert conn.websocket is ws
        assert conn.connected_at == 1000.0
        assert conn.last_seen == 1000.0


class TestQueuedMessage:
    """Tests for the QueuedMessage dataclass."""

    def test_queued_message_creation(self):
        """Test creating a QueuedMessage instance."""
        msg = QueuedMessage(
            message_id="msg-456",
            payload="encrypted_payload_data",
            queued_at=2000.0,
            ttl=5,
        )
        assert msg.message_id == "msg-456"
        assert msg.payload == "encrypted_payload_data"
        assert msg.queued_at == 2000.0
        assert msg.ttl == 5


# =============================================================================
# Unit Tests - RouterNode
# =============================================================================


class TestRouterNode:
    """Tests for RouterNode core functionality."""

    def test_router_creation_defaults(self):
        """Test creating a router with default settings."""
        router = RouterNode()
        assert router.host == "0.0.0.0"
        assert router.port == 8471
        assert router.max_connections == 100
        assert router.seed_url is None
        assert router.heartbeat_interval == 300
        assert len(router.connections) == 0
        assert len(router.offline_queues) == 0

    def test_router_creation_custom(self):
        """Test creating a router with custom settings."""
        router = RouterNode(
            host="127.0.0.1",
            port=9000,
            max_connections=50,
            seed_url="https://seed.example.com:8470",
            heartbeat_interval=120,
        )
        assert router.host == "127.0.0.1"
        assert router.port == 9000
        assert router.max_connections == 50
        assert router.seed_url == "https://seed.example.com:8470"
        assert router.heartbeat_interval == 120

    def test_queue_message(self):
        """Test queueing a message for an offline node."""
        router = RouterNode()
        msg = QueuedMessage(
            message_id="msg-1",
            payload="payload",
            queued_at=time.time(),
            ttl=5,
        )

        result = router._queue_message("node-offline", msg)

        assert result is True
        assert "node-offline" in router.offline_queues
        assert len(router.offline_queues["node-offline"]) == 1
        assert router.offline_queues["node-offline"][0].message_id == "msg-1"
        assert router.messages_queued == 1

    def test_queue_message_size_limit(self):
        """Test that queue enforces size limits."""
        router = RouterNode()
        router.MAX_QUEUE_SIZE = 3  # Small limit for testing

        # Queue up to the limit
        for i in range(5):
            msg = QueuedMessage(
                message_id=f"msg-{i}",
                payload=f"payload-{i}",
                queued_at=time.time(),
                ttl=5,
            )
            router._queue_message("node-offline", msg)

        # Should only have MAX_QUEUE_SIZE messages (oldest dropped)
        assert len(router.offline_queues["node-offline"]) == 3
        # The oldest messages (0, 1) should have been dropped
        ids = [m.message_id for m in router.offline_queues["node-offline"]]
        assert ids == ["msg-2", "msg-3", "msg-4"]


class TestRouterNodeHandlers:
    """Tests for RouterNode message handlers."""

    @pytest.fixture
    def router(self):
        """Create a router for testing."""
        return RouterNode()

    @pytest.mark.asyncio
    async def test_handle_identify_success(self, router):
        """Test successful node identification."""
        ws = AsyncMock()
        ws.closed = False

        result = await router._handle_identify({"node_id": "test-node"}, ws)

        assert result == "test-node"
        assert "test-node" in router.connections
        assert router.connections["test-node"].node_id == "test-node"

        # Should have sent identified response
        ws.send_json.assert_called()
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "identified"
        assert call_args["node_id"] == "test-node"

    @pytest.mark.asyncio
    async def test_handle_identify_missing_node_id(self, router):
        """Test identification with missing node_id."""
        ws = AsyncMock()

        result = await router._handle_identify({}, ws)

        assert result is None
        assert len(router.connections) == 0

        # Should have sent error response
        ws.send_json.assert_called_with(
            {"type": "error", "message": "Missing node_id"}
        )

    @pytest.mark.asyncio
    async def test_handle_identify_already_connected(self, router):
        """Test identification when node is already connected."""
        existing_ws = AsyncMock()
        existing_ws.closed = False

        router.connections["test-node"] = Connection(
            node_id="test-node",
            websocket=existing_ws,
            connected_at=time.time(),
            last_seen=time.time(),
        )

        new_ws = AsyncMock()
        result = await router._handle_identify({"node_id": "test-node"}, new_ws)

        assert result is None
        # Should have sent error
        new_ws.send_json.assert_called_with(
            {"type": "error", "message": "Node already connected"}
        )

    @pytest.mark.asyncio
    async def test_handle_relay_direct_delivery(self, router):
        """Test relaying a message to a connected node."""
        target_ws = AsyncMock()
        router.connections["target-node"] = Connection(
            node_id="target-node",
            websocket=target_ws,
            connected_at=time.time(),
            last_seen=time.time(),
        )

        await router._handle_relay(
            {
                "message_id": "msg-123",
                "next_hop": "target-node",
                "payload": "encrypted_data",
                "ttl": 10,
            }
        )

        assert router.messages_relayed == 1
        assert router.messages_delivered == 1

        # Should have delivered to target
        target_ws.send_json.assert_called_with(
            {
                "type": "deliver",
                "message_id": "msg-123",
                "payload": "encrypted_data",
                "ttl": 9,  # Decremented
            }
        )

    @pytest.mark.asyncio
    async def test_handle_relay_offline_queue(self, router):
        """Test relaying a message to an offline node (queued)."""
        await router._handle_relay(
            {
                "message_id": "msg-456",
                "next_hop": "offline-node",
                "payload": "encrypted_data",
                "ttl": 10,
            }
        )

        assert router.messages_relayed == 1
        assert router.messages_queued == 1
        assert "offline-node" in router.offline_queues
        assert len(router.offline_queues["offline-node"]) == 1

        queued = router.offline_queues["offline-node"][0]
        assert queued.message_id == "msg-456"
        assert queued.ttl == 9  # Decremented

    @pytest.mark.asyncio
    async def test_handle_relay_ttl_expired(self, router):
        """Test that messages with TTL <= 0 are dropped."""
        await router._handle_relay(
            {
                "message_id": "msg-expired",
                "next_hop": "some-node",
                "payload": "encrypted_data",
                "ttl": 0,
            }
        )

        # Should not be relayed or queued
        assert router.messages_relayed == 0
        assert router.messages_queued == 0
        assert len(router.offline_queues) == 0

    @pytest.mark.asyncio
    async def test_handle_relay_missing_fields(self, router):
        """Test that relay rejects messages with missing fields."""
        await router._handle_relay(
            {
                "message_id": "msg-incomplete",
                # Missing next_hop and payload
            }
        )

        assert router.messages_relayed == 0

    @pytest.mark.asyncio
    async def test_deliver_queued_messages(self, router):
        """Test delivering queued messages when node connects."""
        # Queue some messages
        now = time.time()
        router.offline_queues["reconnecting-node"] = [
            QueuedMessage("msg-1", "payload-1", now, 5),
            QueuedMessage("msg-2", "payload-2", now, 3),
        ]

        ws = AsyncMock()
        delivered = await router._deliver_queued("reconnecting-node", ws)

        assert delivered == 2
        assert router.messages_delivered == 2
        assert "reconnecting-node" not in router.offline_queues  # Queue cleared

        # Check both messages were sent
        calls = ws.send_json.call_args_list
        assert len(calls) == 2
        assert calls[0][0][0]["message_id"] == "msg-1"
        assert calls[1][0][0]["message_id"] == "msg-2"

    @pytest.mark.asyncio
    async def test_deliver_queued_expired_messages(self, router):
        """Test that expired queued messages are not delivered."""
        # Queue messages - one fresh, one expired
        now = time.time()
        router.offline_queues["reconnecting-node"] = [
            QueuedMessage("msg-fresh", "payload", now, 5),
            QueuedMessage("msg-expired", "payload", now - 7200, 5),  # 2 hours old
        ]

        ws = AsyncMock()
        delivered = await router._deliver_queued("reconnecting-node", ws)

        # Only fresh message should be delivered
        assert delivered == 1


class TestRouterNodeEndpoints:
    """Tests for RouterNode HTTP endpoints."""

    @pytest.fixture
    def router(self):
        """Create a router for testing."""
        return RouterNode()

    @pytest.mark.asyncio
    async def test_handle_health(self, router):
        """Test health endpoint response."""
        request = MagicMock()
        response = await router.handle_health(request)

        # Parse response
        data = json.loads(response.body)

        assert data["status"] == "healthy"
        assert data["connections"] == 0
        assert "metrics" in data
        assert data["metrics"]["messages_relayed"] == 0

    @pytest.mark.asyncio
    async def test_handle_status(self, router):
        """Test status endpoint response."""
        router.messages_relayed = 100
        router.connections_total = 50

        request = MagicMock()
        response = await router.handle_status(request)

        data = json.loads(response.body)

        assert data["host"] == "0.0.0.0"
        assert data["port"] == 8471
        assert data["connections"]["current"] == 0
        assert data["metrics"]["messages_relayed"] == 100


class TestRouterNodeLifecycle:
    """Tests for RouterNode start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self):
        """Test starting and stopping the router."""
        router = RouterNode(port=0)  # Port 0 = auto-assign

        await router.start()

        assert router._running is True
        assert router._app is not None
        assert router._runner is not None

        await router.stop()

        assert router._running is False
        assert router._app is None

    @pytest.mark.asyncio
    async def test_stop_closes_connections(self):
        """Test that stop() closes all WebSocket connections."""
        router = RouterNode()
        router._running = True

        # Add some mock connections
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        router.connections = {
            "node-1": Connection("node-1", ws1, time.time(), time.time()),
            "node-2": Connection("node-2", ws2, time.time(), time.time()),
        }

        await router.stop()

        ws1.close.assert_called_once()
        ws2.close.assert_called_once()
        assert len(router.connections) == 0


class TestRouterNodeSeedIntegration:
    """Tests for RouterNode seed node integration."""

    @pytest.mark.asyncio
    async def test_register_with_seed_no_url(self):
        """Test registration when no seed URL configured."""
        router = RouterNode(seed_url=None)

        result = await router._register_with_seed()

        assert result is False

    @pytest.mark.asyncio
    async def test_send_heartbeat_no_url(self):
        """Test heartbeat when no seed URL configured."""
        router = RouterNode(seed_url=None)

        result = await router._send_heartbeat()

        assert result is False

    @pytest.mark.asyncio
    async def test_register_with_seed_connection_error(self):
        """Test registration when seed is unreachable."""
        router = RouterNode(seed_url="http://localhost:99999")  # Invalid port

        # This should fail gracefully without throwing
        result = await router._register_with_seed()

        assert result is False

    @pytest.mark.asyncio
    async def test_send_heartbeat_connection_error(self):
        """Test heartbeat when seed is unreachable."""
        router = RouterNode(seed_url="http://localhost:99999")  # Invalid port

        # This should fail gracefully without throwing
        result = await router._send_heartbeat()

        assert result is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestRouterIntegration:
    """Integration tests that test multiple components together."""

    @pytest.mark.asyncio
    async def test_full_message_flow(self):
        """Test complete message flow: connect -> relay -> deliver."""
        router = RouterNode()

        # Simulate two nodes connecting
        ws_sender = AsyncMock()
        ws_sender.closed = False
        ws_receiver = AsyncMock()
        ws_receiver.closed = False

        # Connect sender
        await router._handle_identify({"node_id": "sender"}, ws_sender)
        assert "sender" in router.connections

        # Connect receiver
        await router._handle_identify({"node_id": "receiver"}, ws_receiver)
        assert "receiver" in router.connections

        # Send message from sender to receiver
        await router._handle_relay(
            {
                "message_id": "test-msg",
                "next_hop": "receiver",
                "payload": "secret_encrypted_data",
                "ttl": 5,
            }
        )

        # Verify delivery
        assert router.messages_relayed == 1
        assert router.messages_delivered == 1

        ws_receiver.send_json.assert_any_call(
            {
                "type": "deliver",
                "message_id": "test-msg",
                "payload": "secret_encrypted_data",
                "ttl": 4,
            }
        )

    @pytest.mark.asyncio
    async def test_offline_reconnect_flow(self):
        """Test message queuing and delivery on reconnect."""
        router = RouterNode()

        # Send messages to offline node
        for i in range(3):
            await router._handle_relay(
                {
                    "message_id": f"msg-{i}",
                    "next_hop": "offline-node",
                    "payload": f"data-{i}",
                    "ttl": 10,
                }
            )

        # Verify messages are queued
        assert router.messages_queued == 3
        assert len(router.offline_queues["offline-node"]) == 3

        # Node comes online
        ws = AsyncMock()
        ws.closed = False
        await router._handle_identify({"node_id": "offline-node"}, ws)

        # Verify all messages were delivered
        assert router.messages_delivered == 3
        assert "offline-node" not in router.offline_queues

        # Check all 3 messages + 1 identified response = 4 calls
        assert ws.send_json.call_count == 4
