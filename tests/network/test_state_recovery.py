"""Tests for Connection State Recovery (Issue #111).

These tests verify:
- State persistence to disk
- State recovery on restart
- Disconnect/reconnect scenarios
- Stale state handling
- State conflict detection
- Router reconnection tracking
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from valence.network.node import (
    NodeClient,
    PendingAck,
    PendingMessage,
    FailoverState,
    ConnectionState,
    StateConflictError,
    StaleStateError,
    STATE_VERSION,
    create_node_client,
)
from valence.network.router import (
    RouterNode,
    Connection,
    NodeConnectionHistory,
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
def temp_state_file():
    """Create a temporary state file path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def node_client_with_state(ed25519_keypair, x25519_keypair, temp_state_file):
    """Create a NodeClient with state persistence enabled."""
    private_key, public_key = ed25519_keypair
    enc_private, _ = x25519_keypair
    
    return NodeClient(
        node_id=public_key.public_bytes_raw().hex(),
        private_key=private_key,
        encryption_private_key=enc_private,
        min_connections=1,
        target_connections=3,
        max_connections=5,
        state_file=temp_state_file,
        state_save_interval=1.0,  # Fast for testing
        max_state_age=3600.0,
    )


# =============================================================================
# ConnectionState Tests
# =============================================================================


class TestConnectionState:
    """Tests for ConnectionState serialization."""

    def test_connection_state_creation(self):
        """Test creating a ConnectionState instance."""
        state = ConnectionState(
            version=STATE_VERSION,
            node_id="a" * 64,
            saved_at=1000.0,
            sequence_number=42,
        )
        
        assert state.version == STATE_VERSION
        assert state.node_id == "a" * 64
        assert state.saved_at == 1000.0
        assert state.sequence_number == 42
        assert state.pending_acks == []
        assert state.message_queue == []

    def test_connection_state_to_dict(self):
        """Test ConnectionState serialization to dict."""
        state = ConnectionState(
            version=STATE_VERSION,
            node_id="a" * 64,
            saved_at=1000.0,
            sequence_number=42,
            pending_acks=[{"message_id": "msg-1"}],
            seen_messages=["seen-1", "seen-2"],
        )
        
        data = state.to_dict()
        
        assert data["version"] == STATE_VERSION
        assert data["node_id"] == "a" * 64
        assert data["sequence_number"] == 42
        assert len(data["pending_acks"]) == 1
        assert len(data["seen_messages"]) == 2

    def test_connection_state_from_dict(self):
        """Test ConnectionState deserialization from dict."""
        data = {
            "version": STATE_VERSION,
            "node_id": "b" * 64,
            "saved_at": 2000.0,
            "sequence_number": 99,
            "pending_acks": [{"message_id": "msg-2"}],
            "message_queue": [],
            "seen_messages": ["s1"],
            "failover_states": {},
            "stats": {"messages_sent": 10},
        }
        
        state = ConnectionState.from_dict(data)
        
        assert state.node_id == "b" * 64
        assert state.sequence_number == 99
        assert len(state.pending_acks) == 1
        assert state.stats["messages_sent"] == 10

    def test_connection_state_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = ConnectionState(
            version=STATE_VERSION,
            node_id="c" * 64,
            saved_at=3000.0,
            sequence_number=123,
            pending_acks=[{"message_id": "msg-3"}],
            seen_messages=["s1", "s2", "s3"],
        )
        
        json_str = original.to_json()
        restored = ConnectionState.from_json(json_str)
        
        assert restored.version == original.version
        assert restored.node_id == original.node_id
        assert restored.sequence_number == original.sequence_number
        assert restored.seen_messages == original.seen_messages


# =============================================================================
# PendingAck Serialization Tests
# =============================================================================


class TestPendingAckSerialization:
    """Tests for PendingAck serialization."""

    def test_pending_ack_to_dict(self, x25519_keypair):
        """Test PendingAck serialization."""
        _, pub_key = x25519_keypair
        
        ack = PendingAck(
            message_id="msg-123",
            recipient_id="recipient-456",
            content=b"Hello, World!",
            recipient_public_key=pub_key,
            sent_at=1000.0,
            router_id="router-789",
            timeout_ms=5000,
            retries=1,
        )
        
        data = ack.to_dict()
        
        assert data["message_id"] == "msg-123"
        assert data["recipient_id"] == "recipient-456"
        assert data["content"] == b"Hello, World!".hex()
        assert "recipient_public_key_hex" in data
        assert data["sent_at"] == 1000.0
        assert data["router_id"] == "router-789"
        assert data["timeout_ms"] == 5000
        assert data["retries"] == 1

    def test_pending_ack_from_dict(self, x25519_keypair):
        """Test PendingAck deserialization."""
        _, pub_key = x25519_keypair
        pub_key_hex = pub_key.public_bytes_raw().hex()
        
        data = {
            "message_id": "msg-abc",
            "recipient_id": "recipient-def",
            "content": b"Test content".hex(),
            "recipient_public_key_hex": pub_key_hex,
            "sent_at": 2000.0,
            "router_id": "router-xyz",
            "timeout_ms": 10000,
            "retries": 2,
            "max_retries": 3,
        }
        
        ack = PendingAck.from_dict(data)
        
        assert ack.message_id == "msg-abc"
        assert ack.content == b"Test content"
        assert ack.timeout_ms == 10000
        assert ack.retries == 2

    def test_pending_ack_roundtrip(self, x25519_keypair):
        """Test PendingAck serialization roundtrip."""
        _, pub_key = x25519_keypair
        
        original = PendingAck(
            message_id="msg-roundtrip",
            recipient_id="recipient-rt",
            content=b"Roundtrip test",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id="router-rt",
        )
        
        data = original.to_dict()
        restored = PendingAck.from_dict(data)
        
        assert restored.message_id == original.message_id
        assert restored.content == original.content
        assert restored.recipient_id == original.recipient_id


# =============================================================================
# PendingMessage Serialization Tests
# =============================================================================


class TestPendingMessageSerialization:
    """Tests for PendingMessage serialization."""

    def test_pending_message_to_dict(self, x25519_keypair):
        """Test PendingMessage serialization."""
        _, pub_key = x25519_keypair
        
        msg = PendingMessage(
            message_id="msg-pm",
            recipient_id="recipient-pm",
            content=b"Pending content",
            recipient_public_key=pub_key,
            queued_at=1500.0,
            retries=1,
        )
        
        data = msg.to_dict()
        
        assert data["message_id"] == "msg-pm"
        assert data["content"] == b"Pending content".hex()
        assert data["queued_at"] == 1500.0

    def test_pending_message_roundtrip(self, x25519_keypair):
        """Test PendingMessage serialization roundtrip."""
        _, pub_key = x25519_keypair
        
        original = PendingMessage(
            message_id="msg-pm-rt",
            recipient_id="recipient-pm-rt",
            content=b"Roundtrip message",
            recipient_public_key=pub_key,
            queued_at=time.time(),
        )
        
        data = original.to_dict()
        restored = PendingMessage.from_dict(data)
        
        assert restored.message_id == original.message_id
        assert restored.content == original.content


# =============================================================================
# FailoverState Serialization Tests
# =============================================================================


class TestFailoverStateSerialization:
    """Tests for FailoverState serialization."""

    def test_failover_state_to_dict(self):
        """Test FailoverState serialization."""
        state = FailoverState(
            router_id="router-fs",
            failed_at=1000.0,
            fail_count=3,
            cooldown_until=1060.0,
        )
        
        data = state.to_dict()
        
        assert data["router_id"] == "router-fs"
        assert data["fail_count"] == 3
        assert data["cooldown_until"] == 1060.0

    def test_failover_state_roundtrip(self):
        """Test FailoverState serialization roundtrip."""
        original = FailoverState(
            router_id="router-fs-rt",
            failed_at=time.time(),
            fail_count=2,
            cooldown_until=time.time() + 60,
        )
        
        data = original.to_dict()
        restored = FailoverState.from_dict(data)
        
        assert restored.router_id == original.router_id
        assert restored.fail_count == original.fail_count


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for state persistence to/from disk."""

    @pytest.mark.asyncio
    async def test_save_state(self, node_client_with_state, x25519_keypair):
        """Test saving state to disk."""
        node = node_client_with_state
        _, pub_key = x25519_keypair
        
        # Add some pending state
        node.pending_acks["msg-1"] = PendingAck(
            message_id="msg-1",
            recipient_id="r1",
            content=b"test",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id="router-1",
        )
        
        node.message_queue.append(PendingMessage(
            message_id="msg-2",
            recipient_id="r2",
            content=b"queued",
            recipient_public_key=pub_key,
            queued_at=time.time(),
        ))
        
        # Save state
        await node._save_state()
        
        # Verify file exists and is valid JSON
        assert Path(node.state_file).exists()
        
        with open(node.state_file) as f:
            data = json.load(f)
        
        assert data["version"] == STATE_VERSION
        assert data["node_id"] == node.node_id
        assert len(data["pending_acks"]) == 1
        assert len(data["message_queue"]) == 1

    @pytest.mark.asyncio
    async def test_load_state(self, node_client_with_state, x25519_keypair):
        """Test loading state from disk."""
        node = node_client_with_state
        _, pub_key = x25519_keypair
        
        # Create a state file
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 10,  # 10 seconds ago
            sequence_number=5,
            pending_acks=[],
            message_queue=[],
            seen_messages=["s1", "s2"],
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        # Load state
        loaded = await node._load_state()
        
        assert loaded is not None
        assert loaded.node_id == node.node_id
        assert loaded.sequence_number == 5
        assert loaded.seen_messages == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_load_state_version_mismatch(self, node_client_with_state):
        """Test that version mismatch returns None."""
        node = node_client_with_state
        
        # Create state with wrong version
        state_data = {
            "version": STATE_VERSION + 1,  # Wrong version
            "node_id": node.node_id,
            "saved_at": time.time(),
            "sequence_number": 1,
            "pending_acks": [],
            "message_queue": [],
            "seen_messages": [],
            "failover_states": {},
            "stats": {},
        }
        
        Path(node.state_file).write_text(json.dumps(state_data))
        
        # Should return None for version mismatch
        loaded = await node._load_state()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_state_wrong_node_id(self, node_client_with_state):
        """Test that wrong node_id raises StateConflictError."""
        node = node_client_with_state
        
        # Create state with different node_id
        state = ConnectionState(
            version=STATE_VERSION,
            node_id="wrong" * 16,  # Different node
            saved_at=time.time(),
            sequence_number=1,
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        # Should raise StateConflictError
        with pytest.raises(StateConflictError):
            await node._load_state()

    @pytest.mark.asyncio
    async def test_load_state_too_old(self, node_client_with_state):
        """Test that stale state raises StaleStateError."""
        node = node_client_with_state
        node.max_state_age = 60.0  # 1 minute max age
        
        # Create state that's too old
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 120,  # 2 minutes ago
            sequence_number=1,
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        # Should raise StaleStateError
        with pytest.raises(StaleStateError):
            await node._load_state()


# =============================================================================
# State Recovery Tests
# =============================================================================


class TestStateRecovery:
    """Tests for state recovery on startup."""

    @pytest.mark.asyncio
    async def test_recover_pending_acks(self, node_client_with_state, x25519_keypair):
        """Test recovering pending ACKs from saved state."""
        node = node_client_with_state
        _, pub_key = x25519_keypair
        
        # Create state with pending ACKs
        ack_data = {
            "message_id": "recover-ack-1",
            "recipient_id": "r1",
            "content": b"test".hex(),
            "recipient_public_key_hex": pub_key.public_bytes_raw().hex(),
            "sent_at": time.time() - 5,  # 5 seconds ago (not expired)
            "router_id": "router-1",
            "timeout_ms": 30000,
            "retries": 0,
            "max_retries": 2,
        }
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=10,
            pending_acks=[ack_data],
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        # Mock _wait_for_ack to prevent actual waiting
        with patch.object(node, '_wait_for_ack', new_callable=AsyncMock):
            recovered = await node._recover_state()
        
        assert recovered is True
        assert "recover-ack-1" in node.pending_acks
        assert node._state_sequence == 10

    @pytest.mark.asyncio
    async def test_recover_message_queue(self, node_client_with_state, x25519_keypair):
        """Test recovering message queue from saved state."""
        node = node_client_with_state
        _, pub_key = x25519_keypair
        
        # Create state with queued messages
        msg_data = {
            "message_id": "recover-msg-1",
            "recipient_id": "r1",
            "content": b"queued".hex(),
            "recipient_public_key_hex": pub_key.public_bytes_raw().hex(),
            "queued_at": time.time() - 60,  # 1 minute ago (not expired)
            "retries": 0,
            "max_retries": 3,
        }
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=20,
            message_queue=[msg_data],
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        recovered = await node._recover_state()
        
        assert recovered is True
        assert len(node.message_queue) == 1
        assert node.message_queue[0].message_id == "recover-msg-1"

    @pytest.mark.asyncio
    async def test_recover_seen_messages(self, node_client_with_state):
        """Test recovering seen messages for deduplication."""
        node = node_client_with_state
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=30,
            seen_messages=["seen-1", "seen-2", "seen-3"],
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        recovered = await node._recover_state()
        
        assert recovered is True
        assert "seen-1" in node.seen_messages
        assert "seen-2" in node.seen_messages
        assert "seen-3" in node.seen_messages

    @pytest.mark.asyncio
    async def test_recover_failover_states(self, node_client_with_state):
        """Test recovering failover states."""
        node = node_client_with_state
        
        # Create state with active cooldown
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=40,
            failover_states={
                "router-1": {
                    "router_id": "router-1",
                    "failed_at": time.time() - 30,
                    "fail_count": 2,
                    "cooldown_until": time.time() + 30,  # Still in cooldown
                    "queued_messages": [],
                }
            },
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        recovered = await node._recover_state()
        
        assert recovered is True
        assert "router-1" in node.failover_states
        assert node.failover_states["router-1"].is_in_cooldown()

    @pytest.mark.asyncio
    async def test_recovery_deletes_state_file(self, node_client_with_state):
        """Test that state file is deleted after successful recovery."""
        node = node_client_with_state
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=50,
        )
        
        Path(node.state_file).write_text(state.to_json())
        assert Path(node.state_file).exists()
        
        await node._recover_state()
        
        # State file should be deleted
        assert not Path(node.state_file).exists()

    @pytest.mark.asyncio
    async def test_no_recovery_without_state_file(self, node_client_with_state):
        """Test that recovery returns False when no state file exists."""
        node = node_client_with_state
        
        # Ensure no state file
        if Path(node.state_file).exists():
            Path(node.state_file).unlink()
        
        recovered = await node._recover_state()
        
        assert recovered is False

    @pytest.mark.asyncio
    async def test_recovery_callback(self, node_client_with_state):
        """Test that on_state_recovered callback is called."""
        node = node_client_with_state
        
        callback_called = []
        
        def on_recovered(state):
            callback_called.append(state)
        
        node.on_state_recovered = on_recovered
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=60,
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        await node._recover_state()
        
        assert len(callback_called) == 1
        assert callback_called[0].sequence_number == 60


# =============================================================================
# Node Lifecycle with State Tests
# =============================================================================


class TestNodeLifecycleWithState:
    """Tests for node start/stop with state persistence."""

    @pytest.mark.asyncio
    async def test_start_recovers_state(self, node_client_with_state):
        """Test that start() attempts state recovery."""
        node = node_client_with_state
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=70,
            seen_messages=["startup-seen"],
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        with patch.object(node.discovery, 'discover_routers', new_callable=AsyncMock, return_value=[]):
            await node.start()
        
        try:
            # Should have recovered seen messages
            assert "startup-seen" in node.seen_messages
        finally:
            await node.stop()

    @pytest.mark.asyncio
    async def test_stop_saves_state(self, node_client_with_state, x25519_keypair):
        """Test that stop() saves state."""
        node = node_client_with_state
        _, pub_key = x25519_keypair
        
        with patch.object(node.discovery, 'discover_routers', new_callable=AsyncMock, return_value=[]):
            await node.start()
        
        # Add some pending state
        node.pending_acks["stop-test"] = PendingAck(
            message_id="stop-test",
            recipient_id="r1",
            content=b"test",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id="router-1",
        )
        
        await node.stop()
        
        # State file should exist with our pending ACK
        assert Path(node.state_file).exists()
        
        with open(node.state_file) as f:
            data = json.load(f)
        
        assert len(data["pending_acks"]) == 1
        assert data["pending_acks"][0]["message_id"] == "stop-test"

    @pytest.mark.asyncio
    async def test_state_persistence_loop(self, node_client_with_state, x25519_keypair):
        """Test background state persistence task."""
        node = node_client_with_state
        node.state_save_interval = 0.1  # Fast for testing
        _, pub_key = x25519_keypair
        
        with patch.object(node.discovery, 'discover_routers', new_callable=AsyncMock, return_value=[]):
            await node.start()
        
        # Add pending state
        node.pending_acks["periodic-test"] = PendingAck(
            message_id="periodic-test",
            recipient_id="r1",
            content=b"test",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id="router-1",
        )
        
        # Wait for periodic save
        await asyncio.sleep(0.3)
        
        await node.stop()
        
        # Should have been saved by the loop
        assert Path(node.state_file).exists()


# =============================================================================
# Router Reconnection Tracking Tests
# =============================================================================


class TestRouterReconnectionTracking:
    """Tests for router reconnection tracking (Issue #111)."""

    def test_node_connection_history_creation(self):
        """Test NodeConnectionHistory creation."""
        history = NodeConnectionHistory(
            node_id="node-1",
            first_seen=1000.0,
            last_connected=1000.0,
            last_disconnected=0.0,
            connection_count=1,
        )
        
        assert history.node_id == "node-1"
        assert history.connection_count == 1
        assert history.time_since_disconnect() > 0  # last_disconnected=0

    @pytest.mark.asyncio
    async def test_router_tracks_new_connection(self):
        """Test that router tracks new node connections."""
        router = RouterNode()
        ws = AsyncMock()
        ws.closed = False
        
        node_id = await router._handle_identify({"node_id": "new-node"}, ws)
        
        assert node_id == "new-node"
        assert "new-node" in router.connection_history
        assert router.connection_history["new-node"].connection_count == 1

    @pytest.mark.asyncio
    async def test_router_tracks_reconnection(self):
        """Test that router recognizes reconnecting nodes."""
        router = RouterNode()
        ws1 = AsyncMock()
        ws1.closed = False
        ws2 = AsyncMock()
        ws2.closed = False
        
        # First connection
        await router._handle_identify({"node_id": "reconnect-node"}, ws1)
        assert router.connection_history["reconnect-node"].connection_count == 1
        
        # Disconnect
        del router.connections["reconnect-node"]
        router.connection_history["reconnect-node"].last_disconnected = time.time()
        
        # Reconnect
        await router._handle_identify({"node_id": "reconnect-node"}, ws2)
        
        assert router.connection_history["reconnect-node"].connection_count == 2
        assert router.reconnections_total == 1

    @pytest.mark.asyncio
    async def test_router_sends_reconnection_info(self):
        """Test that router sends reconnection info in identify response."""
        router = RouterNode()
        
        # First connection
        ws1 = AsyncMock()
        ws1.closed = False
        await router._handle_identify({"node_id": "info-node"}, ws1)
        
        # Check first response
        call_args = ws1.send_json.call_args[0][0]
        assert call_args["is_reconnection"] is False
        
        # Disconnect and reconnect
        del router.connections["info-node"]
        router.connection_history["info-node"].last_disconnected = time.time() - 10
        
        ws2 = AsyncMock()
        ws2.closed = False
        await router._handle_identify({"node_id": "info-node"}, ws2)
        
        # Check reconnection response
        call_args = ws2.send_json.call_args[0][0]
        assert call_args["is_reconnection"] is True
        assert "time_since_disconnect" in call_args
        assert call_args["queued_messages"] == 0

    @pytest.mark.asyncio
    async def test_router_includes_queued_count_in_response(self):
        """Test that router includes queued message count in response."""
        router = RouterNode()
        
        # First connection
        ws1 = AsyncMock()
        ws1.closed = False
        await router._handle_identify({"node_id": "queue-node"}, ws1)
        
        # Disconnect
        del router.connections["queue-node"]
        router.connection_history["queue-node"].last_disconnected = time.time()
        
        # Queue some messages
        await router._handle_relay({
            "message_id": "q1",
            "next_hop": "queue-node",
            "payload": "data1",
            "ttl": 5,
        })
        await router._handle_relay({
            "message_id": "q2",
            "next_hop": "queue-node",
            "payload": "data2",
            "ttl": 5,
        })
        
        # Reconnect
        ws2 = AsyncMock()
        ws2.closed = False
        await router._handle_identify({"node_id": "queue-node"}, ws2)
        
        # Should show 2 queued messages in response
        call_args = ws2.send_json.call_args_list[0][0][0]  # First call is identify response
        assert call_args["queued_messages"] == 2

    @pytest.mark.asyncio
    async def test_router_delivers_queued_on_reconnect(self):
        """Test that router delivers queued messages on reconnection."""
        router = RouterNode()
        
        # Queue messages for offline node
        await router._handle_relay({
            "message_id": "deliver-1",
            "next_hop": "offline-node",
            "payload": "payload1",
            "ttl": 5,
        })
        await router._handle_relay({
            "message_id": "deliver-2",
            "next_hop": "offline-node",
            "payload": "payload2",
            "ttl": 5,
        })
        
        assert len(router.offline_queues.get("offline-node", [])) == 2
        
        # Node connects
        ws = AsyncMock()
        ws.closed = False
        await router._handle_identify({"node_id": "offline-node"}, ws)
        
        # Queued messages should have been delivered
        assert "offline-node" not in router.offline_queues
        
        # Check that messages were sent (identified + 2 delivers)
        assert ws.send_json.call_count == 3

    def test_router_prunes_old_history(self):
        """Test that router prunes old connection history."""
        router = RouterNode()
        router.max_history_entries = 5
        router.history_max_age = 60.0
        
        # Add old history entries
        now = time.time()
        for i in range(10):
            router.connection_history[f"node-{i}"] = NodeConnectionHistory(
                node_id=f"node-{i}",
                first_seen=now - 120,  # 2 minutes ago
                last_connected=now - 120,
                last_disconnected=now - 100,
                connection_count=1,
            )
        
        assert len(router.connection_history) == 10
        
        # Prune
        router._prune_connection_history()
        
        # Should have removed old entries
        assert len(router.connection_history) < 10


# =============================================================================
# Delete State File Tests
# =============================================================================


class TestDeleteStateFile:
    """Tests for delete_state_file utility."""

    def test_delete_existing_state_file(self, node_client_with_state):
        """Test deleting an existing state file."""
        node = node_client_with_state
        
        # Create file
        Path(node.state_file).write_text("{}")
        assert Path(node.state_file).exists()
        
        result = node.delete_state_file()
        
        assert result is True
        assert not Path(node.state_file).exists()

    def test_delete_nonexistent_state_file(self, node_client_with_state):
        """Test deleting a nonexistent state file."""
        node = node_client_with_state
        
        # Ensure file doesn't exist
        if Path(node.state_file).exists():
            Path(node.state_file).unlink()
        
        result = node.delete_state_file()
        
        assert result is False

    def test_delete_state_file_disabled(self, ed25519_keypair, x25519_keypair):
        """Test delete_state_file when state persistence is disabled."""
        private_key, public_key = ed25519_keypair
        enc_private, _ = x25519_keypair
        
        node = NodeClient(
            node_id=public_key.public_bytes_raw().hex(),
            private_key=private_key,
            encryption_private_key=enc_private,
            state_file=None,  # Disabled
        )
        
        result = node.delete_state_file()
        
        assert result is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in state recovery."""

    @pytest.mark.asyncio
    async def test_expired_pending_acks_not_recovered(self, node_client_with_state, x25519_keypair):
        """Test that expired pending ACKs are not recovered."""
        node = node_client_with_state
        _, pub_key = x25519_keypair
        
        # Create state with expired ACK
        ack_data = {
            "message_id": "expired-ack",
            "recipient_id": "r1",
            "content": b"test".hex(),
            "recipient_public_key_hex": pub_key.public_bytes_raw().hex(),
            "sent_at": time.time() - 120,  # 2 minutes ago (way past 30s timeout)
            "router_id": "router-1",
            "timeout_ms": 30000,
            "retries": 0,
            "max_retries": 2,
        }
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=80,
            pending_acks=[ack_data],
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        with patch.object(node, '_wait_for_ack', new_callable=AsyncMock):
            await node._recover_state()
        
        # Expired ACK should not be recovered
        assert "expired-ack" not in node.pending_acks

    @pytest.mark.asyncio
    async def test_expired_queue_messages_not_recovered(self, node_client_with_state, x25519_keypair):
        """Test that expired queued messages are not recovered."""
        node = node_client_with_state
        node.MAX_QUEUE_AGE = 60.0  # 1 minute max age
        _, pub_key = x25519_keypair
        
        # Create state with expired message
        msg_data = {
            "message_id": "expired-msg",
            "recipient_id": "r1",
            "content": b"test".hex(),
            "recipient_public_key_hex": pub_key.public_bytes_raw().hex(),
            "queued_at": time.time() - 120,  # 2 minutes ago
            "retries": 0,
            "max_retries": 3,
        }
        
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=90,
            message_queue=[msg_data],
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        await node._recover_state()
        
        # Expired message should not be recovered
        assert len(node.message_queue) == 0

    @pytest.mark.asyncio
    async def test_expired_failover_states_not_recovered(self, node_client_with_state):
        """Test that expired failover states are not recovered."""
        node = node_client_with_state
        
        # Create state with expired cooldown
        state = ConnectionState(
            version=STATE_VERSION,
            node_id=node.node_id,
            saved_at=time.time() - 2,
            sequence_number=100,
            failover_states={
                "router-expired": {
                    "router_id": "router-expired",
                    "failed_at": time.time() - 120,
                    "fail_count": 1,
                    "cooldown_until": time.time() - 60,  # Already past cooldown
                    "queued_messages": [],
                }
            },
        )
        
        Path(node.state_file).write_text(state.to_json())
        
        await node._recover_state()
        
        # Expired failover state should not be recovered
        assert "router-expired" not in node.failover_states

    @pytest.mark.asyncio
    async def test_corrupted_state_file_handled(self, node_client_with_state):
        """Test that corrupted state file is handled gracefully."""
        node = node_client_with_state
        
        # Write invalid JSON
        Path(node.state_file).write_text("not valid json {{{")
        
        # Should return None, not raise
        loaded = await node._load_state()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_atomic_save_on_crash(self, node_client_with_state, x25519_keypair):
        """Test that save is atomic (temp file pattern)."""
        node = node_client_with_state
        _, pub_key = x25519_keypair
        
        # Add state
        node.pending_acks["atomic-test"] = PendingAck(
            message_id="atomic-test",
            recipient_id="r1",
            content=b"test",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id="router-1",
        )
        
        # Save state
        await node._save_state()
        
        # No temp file should remain
        temp_path = Path(node.state_file).with_suffix('.tmp')
        assert not temp_path.exists()
        
        # Main file should exist
        assert Path(node.state_file).exists()
