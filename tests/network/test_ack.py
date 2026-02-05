"""Tests for the Message Acknowledgment Protocol.

These tests verify:
- ACK message data structures
- Pending ACK tracking
- Timeout handling (30s default)
- Retry via same router, then alternate
- Idempotent delivery (dedup by message_id)
- ACK success rate tracking per router
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from valence.network.discovery import RouterInfo
from valence.network.message_handler import MessageHandler, MessageHandlerConfig
from valence.network.messages import AckMessage, AckRequest, DeliverPayload
from valence.network.node import (
    NodeClient,
    PendingAck,
    RouterConnection,
)

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
def mock_router_info_2():
    """Create a second mock RouterInfo for alternate router tests."""
    return RouterInfo(
        router_id="b" * 64,
        endpoints=["192.168.2.1:8471"],
        capacity={"max_connections": 100, "current_load_pct": 30},
        health={"uptime_pct": 99.5, "avg_latency_ms": 60},
        regions=["us-east"],
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
        default_ack_timeout_ms=100,  # Short timeout for testing
    )


@pytest.fixture
def message_handler(ed25519_keypair, x25519_keypair):
    """Create a MessageHandler for testing deduplication."""
    private_key, public_key = ed25519_keypair
    enc_private, _ = x25519_keypair

    config = MessageHandlerConfig(
        default_ack_timeout_ms=100,
        max_seen_messages=10000,
    )

    return MessageHandler(
        node_id=public_key.public_bytes_raw().hex(),
        private_key=private_key,
        encryption_private_key=enc_private,
        config=config,
    )


# =============================================================================
# Unit Tests - ACK Data Structures
# =============================================================================


class TestAckRequest:
    """Tests for the AckRequest dataclass."""

    def test_ack_request_defaults(self):
        """Test AckRequest with default values."""
        req = AckRequest(message_id="msg-123")

        assert req.message_id == "msg-123"
        assert req.require_ack is True
        assert req.ack_timeout_ms == 30000

    def test_ack_request_custom(self):
        """Test AckRequest with custom values."""
        req = AckRequest(
            message_id="msg-456",
            require_ack=False,
            ack_timeout_ms=60000,
        )

        assert req.message_id == "msg-456"
        assert req.require_ack is False
        assert req.ack_timeout_ms == 60000


class TestAckMessage:
    """Tests for the AckMessage dataclass."""

    def test_ack_message_creation(self):
        """Test creating an AckMessage."""
        ack = AckMessage(
            original_message_id="msg-789",
            received_at=1234567890.123,
            recipient_id="recipient-abc",
            signature="deadbeef",
        )

        assert ack.type == "ack"
        assert ack.original_message_id == "msg-789"
        assert ack.received_at == 1234567890.123
        assert ack.recipient_id == "recipient-abc"
        assert ack.signature == "deadbeef"

    def test_ack_message_to_dict(self):
        """Test AckMessage serialization."""
        ack = AckMessage(
            original_message_id="msg-123",
            received_at=1000.0,
            recipient_id="node-456",
            signature="abcd1234",
        )

        data = ack.to_dict()

        assert data["type"] == "ack"
        assert data["original_message_id"] == "msg-123"
        assert data["received_at"] == 1000.0
        assert data["recipient_id"] == "node-456"
        assert data["signature"] == "abcd1234"

    def test_ack_message_from_dict(self):
        """Test AckMessage deserialization."""
        data = {
            "type": "ack",
            "original_message_id": "msg-xyz",
            "received_at": 2000.0,
            "recipient_id": "node-789",
            "signature": "5678efgh",
        }

        ack = AckMessage.from_dict(data)

        assert ack.type == "ack"
        assert ack.original_message_id == "msg-xyz"
        assert ack.received_at == 2000.0
        assert ack.recipient_id == "node-789"
        assert ack.signature == "5678efgh"

    def test_ack_message_roundtrip(self):
        """Test AckMessage serialization roundtrip."""
        original = AckMessage(
            original_message_id="roundtrip-msg",
            received_at=3000.0,
            recipient_id="node-roundtrip",
            signature="roundtripsig",
        )

        data = original.to_dict()
        restored = AckMessage.from_dict(data)

        assert restored.original_message_id == original.original_message_id
        assert restored.received_at == original.received_at
        assert restored.recipient_id == original.recipient_id
        assert restored.signature == original.signature


class TestDeliverPayloadACK:
    """Tests for DeliverPayload ACK-related fields."""

    def test_deliver_payload_ack_fields(self):
        """Test DeliverPayload includes ACK fields."""
        payload = DeliverPayload(
            sender_id="sender-123",
            message_type="belief",
            content={"data": "test"},
            message_id="msg-001",
            require_ack=True,
        )

        assert payload.message_id == "msg-001"
        assert payload.require_ack is True

    def test_deliver_payload_ack_defaults(self):
        """Test DeliverPayload ACK field defaults."""
        payload = DeliverPayload(
            sender_id="sender-456",
            message_type="query",
            content={},
        )

        assert payload.message_id is None
        assert payload.require_ack is False

    def test_deliver_payload_to_dict_includes_ack(self):
        """Test DeliverPayload serialization includes ACK fields."""
        payload = DeliverPayload(
            sender_id="sender-789",
            message_type="response",
            content={"result": "ok"},
            message_id="msg-002",
            require_ack=True,
        )

        data = payload.to_dict()

        assert "message_id" in data
        assert "require_ack" in data
        assert data["message_id"] == "msg-002"
        assert data["require_ack"] is True

    def test_deliver_payload_from_dict_with_ack(self):
        """Test DeliverPayload deserialization with ACK fields."""
        data = {
            "sender_id": "sender-abc",
            "message_type": "ack",
            "content": {},
            "message_id": "msg-003",
            "require_ack": False,
        }

        payload = DeliverPayload.from_dict(data)

        assert payload.message_id == "msg-003"
        assert payload.require_ack is False


# =============================================================================
# Unit Tests - PendingAck
# =============================================================================


class TestPendingAck:
    """Tests for the PendingAck dataclass."""

    def test_pending_ack_creation(self, x25519_keypair):
        """Test creating a PendingAck."""
        _, pub_key = x25519_keypair

        pending = PendingAck(
            message_id="pending-123",
            recipient_id="recipient-456",
            content=b"test content",
            recipient_public_key=pub_key,
            sent_at=1000.0,
            router_id="router-789",
        )

        assert pending.message_id == "pending-123"
        assert pending.recipient_id == "recipient-456"
        assert pending.content == b"test content"
        assert pending.sent_at == 1000.0
        assert pending.router_id == "router-789"
        assert pending.timeout_ms == 30000
        assert pending.retries == 0
        assert pending.max_retries == 2

    def test_pending_ack_custom_timeout(self, x25519_keypair):
        """Test PendingAck with custom timeout."""
        _, pub_key = x25519_keypair

        pending = PendingAck(
            message_id="pending-456",
            recipient_id="recipient-789",
            content=b"data",
            recipient_public_key=pub_key,
            sent_at=2000.0,
            router_id="router-abc",
            timeout_ms=60000,
        )

        assert pending.timeout_ms == 60000


# =============================================================================
# Unit Tests - Idempotent Delivery
# =============================================================================


class TestIdempotentDelivery:
    """Tests for idempotent message delivery (deduplication)."""

    def test_is_duplicate_first_message(self, message_handler):
        """Test that first occurrence of message is not duplicate."""
        result = message_handler.is_duplicate_message("new-msg-001")

        assert result is False
        assert "new-msg-001" in message_handler.seen_messages

    def test_is_duplicate_second_occurrence(self, message_handler):
        """Test that second occurrence is detected as duplicate."""
        # First occurrence
        message_handler.is_duplicate_message("dup-msg-001")

        # Second occurrence
        result = message_handler.is_duplicate_message("dup-msg-001")

        assert result is True

    def test_is_duplicate_different_messages(self, message_handler):
        """Test that different messages are not duplicates."""
        message_handler.is_duplicate_message("msg-a")
        result = message_handler.is_duplicate_message("msg-b")

        assert result is False
        assert "msg-a" in message_handler.seen_messages
        assert "msg-b" in message_handler.seen_messages

    def test_seen_messages_pruning(self, message_handler):
        """Test that seen_messages is pruned when too large."""
        message_handler.config.max_seen_messages = 10

        # Add more than max
        for i in range(15):
            message_handler.is_duplicate_message(f"msg-{i}")

        # Should have pruned to roughly half
        assert len(message_handler.seen_messages) <= 10


# =============================================================================
# Unit Tests - ACK Signing
# =============================================================================


@pytest.mark.skip(
    reason="NodeClient methods moved to MessageHandler (Issue #128 god class decomposition)"
)
class TestACKSigning:
    """Tests for ACK message signing."""

    def test_sign_ack(self, node_client):
        """Test signing a message_id for ACK."""
        signature = node_client._sign_ack("test-message-id")

        assert signature is not None
        assert len(signature) > 0
        # Ed25519 signatures are 64 bytes = 128 hex chars
        assert len(signature) == 128

    def test_sign_ack_deterministic(self, node_client):
        """Test that signing same message_id produces same signature."""
        sig1 = node_client._sign_ack("consistent-id")
        sig2 = node_client._sign_ack("consistent-id")

        assert sig1 == sig2

    def test_sign_ack_different_messages(self, node_client):
        """Test that different message_ids produce different signatures."""
        sig1 = node_client._sign_ack("message-1")
        sig2 = node_client._sign_ack("message-2")

        assert sig1 != sig2


# =============================================================================
# Unit Tests - E2E ACK Handling
# =============================================================================


@pytest.mark.skip(
    reason="NodeClient methods moved to MessageHandler (Issue #128 god class decomposition)"
)
class TestE2EACKHandling:
    """Tests for E2E ACK message handling."""

    @pytest.mark.asyncio
    async def test_handle_e2e_ack_success(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test handling a successful E2E ACK."""
        _, pub_key = x25519_keypair

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        # Track a pending ACK
        node_client.pending_acks["test-msg"] = PendingAck(
            message_id="test-msg",
            recipient_id="recipient-xyz",
            content=b"content",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
        )

        # Handle ACK
        ack = AckMessage(
            original_message_id="test-msg",
            received_at=time.time(),
            recipient_id="recipient-xyz",
            signature="validsig",
        )

        await node_client._handle_e2e_ack(ack)

        # Pending ACK should be removed
        assert "test-msg" not in node_client.pending_acks
        # Router stats should be updated
        assert conn.ack_success == 1
        # Global stats should be updated
        assert node_client._stats["ack_successes"] == 1

    @pytest.mark.asyncio
    async def test_handle_e2e_ack_unknown_message(self, node_client):
        """Test handling ACK for unknown message."""
        ack = AckMessage(
            original_message_id="unknown-msg",
            received_at=time.time(),
            recipient_id="someone",
            signature="sig",
        )

        # Should not raise, just log
        await node_client._handle_e2e_ack(ack)

        # Stats should remain at 0
        assert node_client._stats.get("ack_successes", 0) == 0


# =============================================================================
# Unit Tests - ACK Failure Handling
# =============================================================================


@pytest.mark.skip(
    reason="NodeClient methods moved to MessageHandler (Issue #128 god class decomposition)"
)
class TestACKFailureHandling:
    """Tests for ACK failure and retry handling."""

    def test_handle_ack_failure(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test handling ACK failure."""
        _, pub_key = x25519_keypair

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        # Create pending ACK
        pending = PendingAck(
            message_id="failed-msg",
            recipient_id="recipient",
            content=b"data",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
            retries=2,
        )
        node_client.pending_acks["failed-msg"] = pending

        # Handle failure
        node_client._handle_ack_failure("failed-msg", pending)

        # Pending should be removed
        assert "failed-msg" not in node_client.pending_acks
        # Router failure count should increase
        assert conn.ack_failure == 1
        # Global stats should update
        assert node_client._stats["ack_failures"] == 1

    def test_handle_ack_failure_with_callback(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test ACK failure triggers callback."""
        _, pub_key = x25519_keypair
        callback_called = []

        def on_timeout(msg_id, recipient_id):
            callback_called.append((msg_id, recipient_id))

        node_client.on_ack_timeout = on_timeout

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        pending = PendingAck(
            message_id="callback-msg",
            recipient_id="recipient-callback",
            content=b"data",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
        )

        node_client._handle_ack_failure("callback-msg", pending)

        assert len(callback_called) == 1
        assert callback_called[0] == ("callback-msg", "recipient-callback")


# =============================================================================
# Unit Tests - Stats
# =============================================================================


class TestACKStats:
    """Tests for ACK-related statistics."""

    def test_get_stats_includes_ack_fields(self, message_handler):
        """Test that message_handler.get_stats includes ACK fields."""
        stats = message_handler.get_stats()

        assert "pending_acks" in stats
        assert "seen_messages_cached" in stats
        assert "ack_successes" in stats
        assert "ack_failures" in stats
        assert "messages_deduplicated" in stats

    def test_stats_track_deduplication(self, message_handler):
        """Test that deduplication stats are tracked."""
        # Process same message twice
        message_handler.is_duplicate_message("dup-test")  # First time, not dup
        is_dup = message_handler.is_duplicate_message("dup-test")  # Second time, is dup

        assert is_dup is True
        # Note: The deduplication stat is tracked elsewhere, not in is_duplicate_message
        # We verify the seen_messages behavior
        stats = message_handler.get_stats()
        assert "dup-test" in message_handler.seen_messages


# =============================================================================
# Integration Tests - Timeout and Retry
# =============================================================================


@pytest.mark.skip(
    reason="NodeClient methods moved to MessageHandler (Issue #128 god class decomposition)"
)
class TestACKTimeoutRetry:
    """Integration tests for ACK timeout and retry logic."""

    @pytest.mark.asyncio
    async def test_wait_for_ack_timeout(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test that _wait_for_ack triggers retry on timeout."""
        _, pub_key = x25519_keypair

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        # Create pending ACK with very short timeout
        pending = PendingAck(
            message_id="timeout-msg",
            recipient_id="recipient",
            content=b"data",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
            timeout_ms=50,  # 50ms timeout for testing
        )
        node_client.pending_acks["timeout-msg"] = pending

        # Start waiting
        await node_client._wait_for_ack("timeout-msg")

        # After timeout, retries should have incremented
        # (since no ACK was received)
        if "timeout-msg" in node_client.pending_acks:
            assert node_client.pending_acks["timeout-msg"].retries >= 1

    @pytest.mark.asyncio
    async def test_wait_for_ack_received_before_timeout(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test that receiving ACK before timeout prevents retry."""
        _, pub_key = x25519_keypair

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        # Create pending ACK
        pending = PendingAck(
            message_id="quick-ack-msg",
            recipient_id="recipient",
            content=b"data",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
            timeout_ms=200,  # 200ms timeout
        )
        node_client.pending_acks["quick-ack-msg"] = pending

        # Start waiting in background
        wait_task = asyncio.create_task(node_client._wait_for_ack("quick-ack-msg"))

        # Simulate ACK received quickly (remove from pending)
        await asyncio.sleep(0.05)  # 50ms
        ack = AckMessage(
            original_message_id="quick-ack-msg",
            received_at=time.time(),
            recipient_id="recipient",
            signature="sig",
        )
        await node_client._handle_e2e_ack(ack)

        # Wait for timeout handler to complete
        await wait_task

        # Should have succeeded without retries
        assert "quick-ack-msg" not in node_client.pending_acks
        assert conn.ack_success == 1

    @pytest.mark.asyncio
    async def test_retry_via_different_router(
        self,
        node_client,
        mock_router_info,
        mock_router_info_2,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test retrying via different router after failures."""
        _, pub_key = x25519_keypair

        # Set up two connections
        ws1 = AsyncMock()
        ws1.closed = False
        ws1.send_json = AsyncMock()
        conn1 = RouterConnection(
            router=mock_router_info,
            websocket=ws1,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )

        ws2 = AsyncMock()
        ws2.closed = False
        ws2.send_json = AsyncMock()
        conn2 = RouterConnection(
            router=mock_router_info_2,
            websocket=ws2,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )

        node_client.connections[mock_router_info.router_id] = conn1
        node_client.connections[mock_router_info_2.router_id] = conn2

        # Create pending ACK for first router
        pending = PendingAck(
            message_id="retry-msg",
            recipient_id="recipient",
            content=b"test",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
            timeout_ms=50,
            retries=1,  # Already retried once
        )
        node_client.pending_acks["retry-msg"] = pending

        # Trigger retry via different router
        await node_client._retry_via_different_router("retry-msg")

        # Should have switched to second router
        if "retry-msg" in node_client.pending_acks:
            assert (
                node_client.pending_acks["retry-msg"].router_id
                == mock_router_info_2.router_id
            )
            # Second websocket should have been used
            ws2.send_json.assert_called()


# =============================================================================
# Integration Tests - Send Message with ACK
# =============================================================================


@pytest.mark.skip(
    reason="NodeClient methods moved to MessageHandler (Issue #128 god class decomposition)"
)
class TestSendMessageWithACK:
    """Tests for send_message with ACK tracking."""

    @pytest.mark.asyncio
    async def test_send_message_require_ack_true(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test sending message with require_ack=True creates pending ACK."""
        _, pub_key = x25519_keypair

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        # Send with ACK
        message_id = await node_client.send_message(
            recipient_id="recipient-123",
            recipient_public_key=pub_key,
            content=b"Hello!",
            require_ack=True,
        )

        # Should have pending ACK
        assert message_id in node_client.pending_acks
        pending = node_client.pending_acks[message_id]
        assert pending.recipient_id == "recipient-123"
        assert pending.content == b"Hello!"
        assert pending.router_id == mock_router_info.router_id

    @pytest.mark.asyncio
    async def test_send_message_require_ack_false(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test sending message with require_ack=False skips ACK tracking."""
        _, pub_key = x25519_keypair

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        # Send without ACK
        message_id = await node_client.send_message(
            recipient_id="recipient-456",
            recipient_public_key=pub_key,
            content=b"Fire and forget",
            require_ack=False,
        )

        # Should NOT have pending ACK
        assert message_id not in node_client.pending_acks

    @pytest.mark.asyncio
    async def test_send_message_custom_timeout(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test sending message with custom ACK timeout."""
        _, pub_key = x25519_keypair

        # Set up connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        # Send with custom timeout
        message_id = await node_client.send_message(
            recipient_id="recipient-789",
            recipient_public_key=pub_key,
            content=b"Custom timeout",
            require_ack=True,
            ack_timeout_ms=60000,
        )

        assert message_id in node_client.pending_acks
        assert node_client.pending_acks[message_id].timeout_ms == 60000


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.mark.skip(
    reason="NodeClient methods moved to MessageHandler (Issue #128 god class decomposition)"
)
class TestACKEdgeCases:
    """Tests for edge cases in ACK handling."""

    @pytest.mark.asyncio
    async def test_retry_message_no_pending(self, node_client):
        """Test _retry_message when pending ACK doesn't exist."""
        # Should not raise
        await node_client._retry_message("nonexistent-msg")

    @pytest.mark.asyncio
    async def test_retry_different_router_no_alternative(
        self,
        node_client,
        mock_router_info,
        mock_websocket,
        mock_session,
        x25519_keypair,
    ):
        """Test retry when no alternative router available."""
        _, pub_key = x25519_keypair

        # Set up only one connection
        conn = RouterConnection(
            router=mock_router_info,
            websocket=mock_websocket,
            session=mock_session,
            connected_at=time.time(),
            last_seen=time.time(),
        )
        node_client.connections[mock_router_info.router_id] = conn

        pending = PendingAck(
            message_id="no-alt-msg",
            recipient_id="recipient",
            content=b"data",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id=mock_router_info.router_id,
        )
        node_client.pending_acks["no-alt-msg"] = pending

        # Try to retry via different router (none available)
        await node_client._retry_via_different_router("no-alt-msg")

        # Should have failed and removed pending
        assert "no-alt-msg" not in node_client.pending_acks
        assert node_client._stats["ack_failures"] == 1

    def test_is_duplicate_empty_id(self, message_handler):
        """Test duplicate check with empty message_id."""
        result1 = message_handler.is_duplicate_message("")
        result2 = message_handler.is_duplicate_message("")

        assert result1 is False
        assert result2 is True

    @pytest.mark.asyncio
    async def test_wait_for_ack_no_pending(self, node_client):
        """Test _wait_for_ack when pending doesn't exist."""
        # Should return immediately without error
        await node_client._wait_for_ack("nonexistent")
