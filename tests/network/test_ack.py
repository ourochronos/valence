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
