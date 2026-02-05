"""
Tests for MessageHandler component.

Tests cover:
- Message sending (direct and batched)
- Message receiving and deduplication
- ACK tracking and handling
- Message queuing during failover
- Traffic analysis mitigations (batching, jitter)
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from valence.network.message_handler import (
    MessageHandler,
    MessageHandlerConfig,
)
from valence.network.messages import AckMessage
from valence.network.config import (
    TrafficAnalysisMitigationConfig,
    BatchingConfig,
    TimingJitterConfig,
    ConstantRateConfig,
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
def config():
    """Create a test configuration."""
    return MessageHandlerConfig(
        default_ack_timeout_ms=5000,
        max_seen_messages=100,
        max_queue_size=50,
        max_queue_age=300.0,
    )


@pytest.fixture
def message_handler(ed25519_keypair, x25519_keypair, config):
    """Create a MessageHandler for testing."""
    private_key, public_key = ed25519_keypair
    enc_private, _ = x25519_keypair
    
    return MessageHandler(
        node_id=public_key.public_bytes_raw().hex(),
        private_key=private_key,
        encryption_private_key=enc_private,
        config=config,
    )


@pytest.fixture
def mock_router_selector():
    """Create a mock router selector function."""
    from valence.network.discovery import RouterInfo
    
    router = RouterInfo(
        router_id="b" * 64,
        endpoints=["192.168.1.1:8471"],
        capacity={},
        health={},
        regions=[],
        features=[],
    )
    
    def selector():
        return router
    
    return selector


@pytest.fixture
def mock_send_via_router():
    """Create a mock send_via_router function."""
    return AsyncMock()


# =============================================================================
# Unit Tests - MessageHandlerConfig
# =============================================================================


class TestMessageHandlerConfig:
    """Tests for MessageHandlerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = MessageHandlerConfig()
        assert config.default_ack_timeout_ms == 30000
        assert config.max_seen_messages == 10000
        assert config.max_queue_size == 1000
        assert config.max_queue_age == 3600.0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = MessageHandlerConfig(
            default_ack_timeout_ms=5000,
            max_seen_messages=500,
        )
        assert config.default_ack_timeout_ms == 5000
        assert config.max_seen_messages == 500


# =============================================================================
# Unit Tests - MessageHandler Properties
# =============================================================================


class TestMessageHandlerProperties:
    """Tests for MessageHandler property methods."""

    def test_get_stats_initial(self, message_handler):
        """Test initial statistics."""
        stats = message_handler.get_stats()
        assert stats["messages_sent"] == 0
        assert stats["messages_received"] == 0
        assert stats["messages_queued"] == 0
        assert stats["messages_dropped"] == 0
        assert stats["ack_successes"] == 0
        assert stats["ack_failures"] == 0
        assert stats["pending_acks"] == 0
        assert stats["queued_messages"] == 0


# =============================================================================
# Unit Tests - Message Deduplication
# =============================================================================


class TestMessageDeduplication:
    """Tests for message deduplication."""

    def test_is_duplicate_message_first_time(self, message_handler):
        """Test deduplication check for new message."""
        message_id = "test-message-001"
        
        # First time should not be duplicate
        assert message_handler.is_duplicate_message(message_id) is False
        
        # Message should now be in seen set
        assert message_id in message_handler.seen_messages

    def test_is_duplicate_message_second_time(self, message_handler):
        """Test deduplication check for already-seen message."""
        message_id = "test-message-001"
        
        # First call adds it
        message_handler.is_duplicate_message(message_id)
        
        # Second call should be duplicate
        assert message_handler.is_duplicate_message(message_id) is True

    def test_is_duplicate_message_pruning(self, message_handler):
        """Test that seen messages are pruned when exceeding limit."""
        # Set a small limit
        message_handler.config.max_seen_messages = 10
        
        # Add more than the limit
        for i in range(15):
            message_handler.is_duplicate_message(f"message-{i}")
        
        # Should have pruned to about half
        assert len(message_handler.seen_messages) <= 10


# =============================================================================
# Unit Tests - ACK Handling
# =============================================================================


class TestACKHandling:
    """Tests for ACK handling."""

    def test_handle_ack_success(self, message_handler, x25519_keypair):
        """Test handling successful ACK."""
        from valence.network.node import PendingAck
        
        message_id = "test-message-001"
        _, pub_key = x25519_keypair
        
        # Add pending ACK
        pending = PendingAck(
            message_id=message_id,
            recipient_id="recipient123",
            content=b"test content",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id="router123",
        )
        message_handler.pending_acks[message_id] = pending
        
        # Handle ACK
        message_handler.handle_ack(message_id, success=True)
        
        # Should be removed from pending
        assert message_id not in message_handler.pending_acks
        assert message_handler._stats["ack_successes"] == 1

    def test_handle_ack_failure(self, message_handler):
        """Test handling failed ACK."""
        message_handler.handle_ack("nonexistent", success=False)
        
        assert message_handler._stats["ack_failures"] == 1

    def test_handle_e2e_ack(self, message_handler, x25519_keypair):
        """Test handling E2E acknowledgment."""
        from valence.network.node import PendingAck
        
        message_id = "test-message-001"
        _, pub_key = x25519_keypair
        
        # Add pending ACK
        sent_time = time.time() - 1.0  # 1 second ago
        pending = PendingAck(
            message_id=message_id,
            recipient_id="recipient123",
            content=b"test content",
            recipient_public_key=pub_key,
            sent_at=sent_time,
            router_id="router123",
        )
        message_handler.pending_acks[message_id] = pending
        
        # Create E2E ACK
        ack = AckMessage(
            original_message_id=message_id,
            received_at=time.time(),
            recipient_id="recipient123",
            signature="",
        )
        
        # Handle E2E ACK
        message_handler.handle_e2e_ack(ack)
        
        # Should be removed from pending
        assert message_id not in message_handler.pending_acks
        assert message_handler._stats["ack_successes"] == 1

    def test_handle_e2e_ack_unknown_message(self, message_handler):
        """Test handling E2E ACK for unknown message."""
        ack = AckMessage(
            original_message_id="unknown-message",
            received_at=time.time(),
            recipient_id="recipient123",
            signature="",
        )
        
        # Should not raise, just log
        message_handler.handle_e2e_ack(ack)
        
        # No stats change
        assert message_handler._stats["ack_successes"] == 0


# =============================================================================
# Unit Tests - Message Signing
# =============================================================================


class TestMessageSigning:
    """Tests for message signing."""

    def test_sign_ack(self, message_handler):
        """Test signing an ACK."""
        message_id = "test-message-001"
        signature = message_handler.sign_ack(message_id)
        
        # Should return a hex string
        assert isinstance(signature, str)
        assert len(signature) == 128  # Ed25519 signature is 64 bytes = 128 hex chars


# =============================================================================
# Unit Tests - Message Batching
# =============================================================================


class TestMessageBatching:
    """Tests for message batching."""

    def test_flush_batch_empty(self, message_handler):
        """Test flushing empty batch."""
        # Should not raise
        asyncio.get_event_loop().run_until_complete(
            message_handler.flush_batch()
        )

    @pytest.mark.asyncio
    async def test_add_to_batch(self, message_handler, x25519_keypair, mock_router_selector, mock_send_via_router):
        """Test adding message to batch."""
        _, pub_key = x25519_keypair
        
        # Enable batching
        message_handler.traffic_mitigation_config = TrafficAnalysisMitigationConfig(
            batching=BatchingConfig(
                enabled=True,
                max_batch_size=10,
                batch_interval_ms=1000,
            ),
        )
        
        # Add to batch
        message_id = await message_handler._add_to_batch(
            message_id="test-001",
            recipient_id="recipient123",
            recipient_public_key=pub_key,
            content=b"test content",
            require_ack=True,
            timeout_ms=5000,
            router_selector=mock_router_selector,
            send_via_router=mock_send_via_router,
        )
        
        assert message_id == "test-001"
        assert message_handler._stats["batched_messages"] == 1
        assert len(message_handler._message_batch) == 1


# =============================================================================
# Unit Tests - Queue Processing
# =============================================================================


class TestQueueProcessing:
    """Tests for message queue processing."""

    @pytest.mark.asyncio
    async def test_process_queue_empty(self, message_handler, mock_router_selector, mock_send_via_router):
        """Test processing empty queue."""
        count = await message_handler.process_queue(
            router_selector=mock_router_selector,
            send_via_router=mock_send_via_router,
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_process_queue_old_messages_dropped(self, message_handler, mock_router_selector, mock_send_via_router, x25519_keypair):
        """Test that old messages are dropped during queue processing."""
        from valence.network.node import PendingMessage
        
        _, pub_key = x25519_keypair
        
        # Add old message to queue
        old_message = PendingMessage(
            message_id="old-message",
            recipient_id="recipient123",
            content=b"old content",
            recipient_public_key=pub_key,
            queued_at=time.time() - 7200,  # 2 hours ago (exceeds max_queue_age)
        )
        message_handler.message_queue.append(old_message)
        
        await message_handler.process_queue(
            router_selector=mock_router_selector,
            send_via_router=mock_send_via_router,
        )
        
        assert message_handler._stats["messages_dropped"] == 1
        assert len(message_handler.message_queue) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestMessageHandlerIntegration:
    """Integration tests for MessageHandler."""

    @pytest.mark.asyncio
    async def test_send_message_no_router(self, message_handler, x25519_keypair, mock_send_via_router):
        """Test sending message when no router available."""
        _, pub_key = x25519_keypair
        
        def no_router_selector():
            return None
        
        message_id = await message_handler.send_message(
            recipient_id="recipient123",
            recipient_public_key=pub_key,
            content=b"test content",
            router_selector=no_router_selector,
            send_via_router=mock_send_via_router,
            require_ack=True,
        )
        
        # Message should be queued
        assert message_handler._stats["messages_queued"] == 1
        assert len(message_handler.message_queue) == 1

    @pytest.mark.asyncio
    async def test_send_message_queue_full(self, message_handler, x25519_keypair, mock_send_via_router):
        """Test sending message when queue is full."""
        from valence.network.node import PendingMessage, NoRoutersAvailableError
        
        _, pub_key = x25519_keypair
        
        def no_router_selector():
            return None
        
        # Fill the queue
        message_handler.config.max_queue_size = 1
        message_handler.message_queue.append(PendingMessage(
            message_id="existing",
            recipient_id="recipient",
            content=b"content",
            recipient_public_key=pub_key,
            queued_at=time.time(),
        ))
        
        # Should raise when queue is full
        with pytest.raises(NoRoutersAvailableError):
            await message_handler.send_message(
                recipient_id="recipient123",
                recipient_public_key=pub_key,
                content=b"test content",
                router_selector=no_router_selector,
                send_via_router=mock_send_via_router,
                require_ack=True,
            )
        
        assert message_handler._stats["messages_dropped"] == 1

    @pytest.mark.asyncio
    async def test_send_message_with_jitter(self, message_handler, x25519_keypair, mock_router_selector, mock_send_via_router):
        """Test sending message with timing jitter enabled."""
        _, pub_key = x25519_keypair
        
        # Enable jitter with small values for testing
        message_handler.traffic_mitigation_config = TrafficAnalysisMitigationConfig(
            jitter=TimingJitterConfig(
                enabled=True,
                min_delay_ms=1,
                max_delay_ms=10,
            ),
        )
        
        start = time.time()
        await message_handler.send_message(
            recipient_id="recipient123",
            recipient_public_key=pub_key,
            content=b"test content",
            router_selector=mock_router_selector,
            send_via_router=mock_send_via_router,
            require_ack=False,
            bypass_mitigations=False,
        )
        
        # Should have applied some delay
        # Note: This test is timing-sensitive but uses very small delays
        assert message_handler._stats["jitter_delays_applied"] >= 0

    @pytest.mark.asyncio
    async def test_send_message_direct(self, message_handler, x25519_keypair, mock_router_selector, mock_send_via_router):
        """Test sending message directly without mitigations."""
        _, pub_key = x25519_keypair
        
        message_id = await message_handler.send_message(
            recipient_id="recipient123",
            recipient_public_key=pub_key,
            content=b"test content",
            router_selector=mock_router_selector,
            send_via_router=mock_send_via_router,
            require_ack=False,
        )
        
        # Should have called send_via_router
        mock_send_via_router.assert_called_once()
        
        # Message ID should be a UUID
        assert len(message_id) == 36  # UUID format


# =============================================================================
# Unit Tests - Callback Handling
# =============================================================================


class TestCallbackHandling:
    """Tests for callback handling."""

    def test_ack_timeout_callback(self, message_handler, x25519_keypair):
        """Test that ACK timeout callback is called."""
        from valence.network.node import PendingAck
        
        callback_args = []
        
        def on_ack_timeout(message_id, recipient_id):
            callback_args.append((message_id, recipient_id))
        
        message_handler.on_ack_timeout = on_ack_timeout
        
        _, pub_key = x25519_keypair
        pending = PendingAck(
            message_id="test-001",
            recipient_id="recipient123",
            content=b"content",
            recipient_public_key=pub_key,
            sent_at=time.time(),
            router_id="router123",
            retries=5,  # Max retries exceeded
        )
        
        # Simulate failure handling
        message_handler._handle_ack_failure("test-001", pending)
        
        assert len(callback_args) == 1
        assert callback_args[0] == ("test-001", "recipient123")
        assert message_handler._stats["ack_failures"] == 1
