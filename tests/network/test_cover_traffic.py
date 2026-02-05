"""Tests for Cover Traffic functionality (Issue #116).

Tests verify:
- Message padding to fixed bucket sizes
- Unpadding of padded messages
- Cover traffic configuration
- Cover message generation and handling
- Idle detection
- Runtime enable/disable of cover traffic
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from valence.network.messages import (
    MESSAGE_SIZE_BUCKETS,
    CoverMessage,
    calculate_padding_overhead,
    generate_cover_content,
    get_padded_size,
    pad_message,
    unpad_message,
)
from valence.network.node import (
    CoverTrafficConfig,
    NodeClient,
)

# Note: Some tests need update to use component methods


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


@pytest.fixture
def node_client_with_cover_traffic(ed25519_keypair, x25519_keypair):
    """Create a NodeClient with cover traffic enabled."""
    private_key, public_key = ed25519_keypair
    enc_private, _ = x25519_keypair
    
    cover_config = CoverTrafficConfig(
        enabled=True,
        rate_per_minute=6.0,  # 1 msg per 10 seconds for testing
        idle_threshold_seconds=5.0,  # Short threshold for tests
        pad_messages=True,
    )
    
    return NodeClient(
        node_id=public_key.public_bytes_raw().hex(),
        private_key=private_key,
        encryption_private_key=enc_private,
        min_connections=1,
        target_connections=3,
        max_connections=5,
        cover_traffic=cover_config,
    )


# =============================================================================
# Message Padding Tests
# =============================================================================


class TestMessagePadding:
    """Tests for message padding utilities."""
    
    def test_get_padded_size_small_message(self):
        """Small messages should pad to 1KB bucket."""
        assert get_padded_size(10) == 1024
        assert get_padded_size(100) == 1024
        assert get_padded_size(500) == 1024
        assert get_padded_size(1000) == 1024
    
    def test_get_padded_size_medium_message(self):
        """Medium messages should pad to 4KB bucket."""
        assert get_padded_size(1024) == 4096
        assert get_padded_size(2000) == 4096
        assert get_padded_size(4000) == 4096
    
    def test_get_padded_size_large_message(self):
        """Large messages should pad to 16KB or 64KB bucket."""
        assert get_padded_size(5000) == 16384
        assert get_padded_size(10000) == 16384
        assert get_padded_size(20000) == 65536
        assert get_padded_size(50000) == 65536
    
    def test_get_padded_size_very_large_message(self):
        """Very large messages exceed all buckets - return as-is."""
        size = 100000
        assert get_padded_size(size) == size
    
    def test_pad_message_basic(self):
        """Padding should produce exact bucket size."""
        content = b"Hello, World!"
        padded = pad_message(content)
        
        assert len(padded) == 1024  # Smallest bucket
        assert padded.startswith(content)
    
    def test_pad_message_exact_bucket(self):
        """Padding respects bucket boundaries."""
        # Just under 1KB threshold
        content = b"x" * 1000
        padded = pad_message(content)
        assert len(padded) == 1024
        
        # Over 1KB, should go to 4KB
        content = b"x" * 1024
        padded = pad_message(content)
        assert len(padded) == 4096
    
    def test_pad_message_custom_target(self):
        """Padding to custom target size."""
        content = b"Hello"
        padded = pad_message(content, target_size=4096)
        assert len(padded) == 4096
    
    def test_unpad_message_basic(self):
        """Unpadding should recover original content."""
        original = b"Hello, World!"
        padded = pad_message(original)
        unpadded = unpad_message(padded)
        
        assert unpadded == original
    
    def test_unpad_message_roundtrip_all_buckets(self):
        """Roundtrip padding/unpadding for all bucket sizes."""
        for bucket in MESSAGE_SIZE_BUCKETS:
            # Create content that fits in this bucket
            content_size = bucket - 100
            original = b"x" * content_size
            padded = pad_message(original)
            unpadded = unpad_message(padded)
            
            assert unpadded == original, f"Roundtrip failed for bucket {bucket}"
    
    def test_unpad_message_not_padded(self):
        """Unpadding non-padded message returns original."""
        original = b"Not padded at all"
        assert unpad_message(original) == original
    
    def test_pad_message_empty(self):
        """Empty message can be padded and unpadded."""
        original = b""
        padded = pad_message(original)
        assert len(padded) == 1024
        unpadded = unpad_message(padded)
        assert unpadded == original
    
    def test_padding_deterministic_size(self):
        """Same-size inputs produce same-size outputs."""
        content1 = b"Hello World!!"
        content2 = b"Goodbye now!!"
        assert len(content1) == len(content2)
        
        padded1 = pad_message(content1)
        padded2 = pad_message(content2)
        assert len(padded1) == len(padded2)
    
    def test_calculate_padding_overhead(self):
        """Padding overhead calculation is correct."""
        padded_size, overhead_pct = calculate_padding_overhead(100)
        assert padded_size == 1024
        assert overhead_pct > 0
        
        # Very large messages have no overhead
        padded_size, overhead_pct = calculate_padding_overhead(100000)
        assert padded_size == 100000
        assert overhead_pct == 0.0
    
    def test_bucket_sizes_ascending(self):
        """Bucket sizes should be in ascending order."""
        for i in range(len(MESSAGE_SIZE_BUCKETS) - 1):
            assert MESSAGE_SIZE_BUCKETS[i] < MESSAGE_SIZE_BUCKETS[i + 1]


# =============================================================================
# Cover Message Tests
# =============================================================================


class TestCoverMessage:
    """Tests for CoverMessage dataclass."""
    
    def test_cover_message_creation(self):
        """CoverMessage should have expected fields."""
        msg = CoverMessage()
        
        assert msg.type == "cover"
        assert msg.message_id is not None
        assert msg.timestamp > 0
        assert msg.nonce is not None
        assert len(msg.nonce) == 64  # 32 bytes hex
    
    def test_cover_message_to_dict(self):
        """CoverMessage serializes to dict correctly."""
        msg = CoverMessage()
        data = msg.to_dict()
        
        assert data["type"] == "cover"
        assert "message_id" in data
        assert "timestamp" in data
        assert "nonce" in data
    
    def test_cover_message_from_dict(self):
        """CoverMessage deserializes from dict correctly."""
        original = CoverMessage()
        data = original.to_dict()
        restored = CoverMessage.from_dict(data)
        
        assert restored.type == original.type
        assert restored.message_id == original.message_id
        assert restored.timestamp == original.timestamp
    
    def test_cover_message_bytes_roundtrip(self):
        """CoverMessage bytes serialization roundtrip."""
        original = CoverMessage()
        data = original.to_bytes()
        restored = CoverMessage.from_bytes(data)
        
        assert restored.type == original.type
        assert restored.message_id == original.message_id
    
    def test_is_cover_message_detection(self):
        """is_cover_message correctly identifies cover traffic."""
        cover_data = {"type": "cover", "message_id": "123"}
        assert CoverMessage.is_cover_message(cover_data) is True
        
        real_data = {"type": "relay", "message_id": "123"}
        assert CoverMessage.is_cover_message(real_data) is False
        
        empty_data = {}
        assert CoverMessage.is_cover_message(empty_data) is False
    
    def test_generate_cover_content(self):
        """generate_cover_content produces random bytes of correct size."""
        content = generate_cover_content(target_bucket=1024)
        assert len(content) > 0
        assert len(content) < 1024  # Should leave room for JSON overhead
        
        content = generate_cover_content(target_bucket=4096)
        assert len(content) > 0
        assert len(content) < 4096
    
    def test_generate_cover_content_random_bucket(self):
        """generate_cover_content with no target picks random bucket."""
        contents = [generate_cover_content() for _ in range(10)]
        
        # Should produce content (not crash)
        for content in contents:
            assert len(content) > 0


# =============================================================================
# Cover Traffic Configuration Tests
# =============================================================================


class TestCoverTrafficConfig:
    """Tests for CoverTrafficConfig dataclass."""
    
    def test_default_config_disabled(self):
        """Default config has cover traffic disabled."""
        config = CoverTrafficConfig()
        
        assert config.enabled is False
        assert config.rate_per_minute == 2.0
        assert config.idle_threshold_seconds == 30.0
        assert config.pad_messages is True
    
    def test_enabled_config(self):
        """Enabled config with custom settings."""
        config = CoverTrafficConfig(
            enabled=True,
            rate_per_minute=5.0,
            idle_threshold_seconds=60.0,
            pad_messages=False,
        )
        
        assert config.enabled is True
        assert config.rate_per_minute == 5.0
        assert config.idle_threshold_seconds == 60.0
        assert config.pad_messages is False
    
    def test_get_next_interval_disabled(self):
        """Disabled config returns infinite interval."""
        config = CoverTrafficConfig(enabled=False)
        interval = config.get_next_interval()
        assert interval == float('inf')
    
    def test_get_next_interval_enabled(self):
        """Enabled config returns reasonable interval."""
        config = CoverTrafficConfig(
            enabled=True,
            rate_per_minute=6.0,  # 10 seconds base
            randomize_timing=False,
        )
        interval = config.get_next_interval()
        assert 5.0 <= interval <= 60.0  # Within bounds
    
    def test_get_next_interval_randomized(self):
        """Randomized timing produces varied intervals."""
        config = CoverTrafficConfig(
            enabled=True,
            rate_per_minute=6.0,
            randomize_timing=True,
            min_interval_seconds=5.0,
            max_interval_seconds=30.0,
        )
        
        intervals = [config.get_next_interval() for _ in range(20)]
        
        # All should be within bounds
        for interval in intervals:
            assert 5.0 <= interval <= 30.0
        
        # Should have some variation (not all same)
        assert len(set(intervals)) > 1
    
    def test_target_peers_configuration(self):
        """Target peers can be configured."""
        peers = ["peer1", "peer2", "peer3"]
        config = CoverTrafficConfig(
            enabled=True,
            target_peers=peers,
        )
        
        assert config.target_peers == peers


# =============================================================================
# NodeClient Cover Traffic Integration Tests
# =============================================================================


class TestNodeClientCoverTraffic:
    """Tests for cover traffic integration in NodeClient."""
    
    def test_node_has_cover_traffic_config(self, node_client):
        """Node client should have cover traffic config."""
        assert hasattr(node_client, 'cover_traffic')
        assert isinstance(node_client.cover_traffic, CoverTrafficConfig)
    
    def test_node_cover_traffic_disabled_by_default(self, node_client):
        """Cover traffic is disabled by default."""
        assert node_client.cover_traffic.enabled is False
    
    def test_node_with_cover_traffic_enabled(self, node_client_with_cover_traffic):
        """Node can be created with cover traffic enabled."""
        assert node_client_with_cover_traffic.cover_traffic.enabled is True
        assert node_client_with_cover_traffic.cover_traffic.rate_per_minute == 6.0
    
    def test_is_idle_no_messages(self, node_client_with_cover_traffic):
        """Node with no messages sent is considered idle."""
        assert node_client_with_cover_traffic._is_idle() is True
    
    def test_is_idle_after_message(self, node_client_with_cover_traffic):
        """Node is not idle immediately after sending a message."""
        node_client_with_cover_traffic._last_real_message_time = time.time()
        assert node_client_with_cover_traffic._is_idle() is False
    
    def test_is_idle_after_threshold(self, node_client_with_cover_traffic):
        """Node becomes idle after threshold."""
        # Set last message time to past the threshold
        node_client_with_cover_traffic._last_real_message_time = (
            time.time() - node_client_with_cover_traffic.cover_traffic.idle_threshold_seconds - 1
        )
        assert node_client_with_cover_traffic._is_idle() is True
    
    def test_is_idle_disabled_returns_false(self, node_client):
        """is_idle returns False when cover traffic disabled."""
        assert node_client._is_idle() is False
    
    def test_get_cover_target_with_peers(self, node_client_with_cover_traffic):
        """get_cover_target selects from configured peers."""
        peers = ["peer1", "peer2", "peer3"]
        node_client_with_cover_traffic.cover_traffic.target_peers = peers
        
        targets = [node_client_with_cover_traffic._get_cover_target() for _ in range(10)]
        for target in targets:
            assert target in peers
    
    def test_get_cover_target_no_peers(self, node_client_with_cover_traffic):
        """get_cover_target generates random ID without peers."""
        node_client_with_cover_traffic.cover_traffic.target_peers = []
        
        target = node_client_with_cover_traffic._get_cover_target()
        assert target is not None
        assert len(target) == 64  # 32 bytes hex
    
    def test_cover_traffic_stats(self, node_client_with_cover_traffic):
        """get_cover_traffic_stats returns expected data."""
        stats = node_client_with_cover_traffic.get_cover_traffic_stats()
        
        assert stats["enabled"] is True
        assert stats["rate_per_minute"] == 6.0
        assert stats["pad_messages"] is True
        assert "cover_messages_sent" in stats
        assert "cover_messages_received" in stats
        assert "bytes_padded" in stats
    
    def test_enable_cover_traffic_runtime(self, node_client):
        """Cover traffic can be enabled at runtime."""
        assert node_client.cover_traffic.enabled is False
        
        node_client.enable_cover_traffic(
            rate_per_minute=3.0,
            pad_messages=True,
            target_peers=["peer1", "peer2"],
        )
        
        assert node_client.cover_traffic.enabled is True
        assert node_client.cover_traffic.rate_per_minute == 3.0
        assert node_client.cover_traffic.target_peers == ["peer1", "peer2"]
    
    def test_disable_cover_traffic_runtime(self, node_client_with_cover_traffic):
        """Cover traffic can be disabled at runtime."""
        assert node_client_with_cover_traffic.cover_traffic.enabled is True
        
        node_client_with_cover_traffic.disable_cover_traffic()
        
        assert node_client_with_cover_traffic.cover_traffic.enabled is False


# =============================================================================
# Cover Traffic Message Flow Tests
# =============================================================================


class TestCoverTrafficFlow:
    """Tests for cover traffic message flow."""
    
    @pytest.mark.asyncio
    async def test_generate_cover_message_no_connections(
        self, node_client_with_cover_traffic
    ):
        """Cover message generation handles no connections gracefully."""
        # No connections - should not crash
        assert len(node_client_with_cover_traffic.connections) == 0
        await node_client_with_cover_traffic._generate_cover_message()
        
        # Should not have sent anything
        assert node_client_with_cover_traffic._stats["cover_messages_sent"] == 0
    
    @pytest.mark.asyncio
    async def test_cover_traffic_loop_cancellation(
        self, node_client_with_cover_traffic
    ):
        """Cover traffic loop can be cancelled cleanly."""
        node_client_with_cover_traffic._running = True
        
        # Start the loop
        task = asyncio.create_task(
            node_client_with_cover_traffic._cover_traffic_loop()
        )
        
        # Give it a moment to start
        await asyncio.sleep(0.1)
        
        # Cancel it
        task.cancel()
        
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected
    
    @pytest.mark.asyncio  
    async def test_cover_traffic_stats_tracking(
        self, node_client_with_cover_traffic
    ):
        """Cover traffic stats are tracked correctly."""
        initial_sent = node_client_with_cover_traffic._stats["cover_messages_sent"]
        initial_padded = node_client_with_cover_traffic._stats["bytes_padded"]
        
        # These stats start at 0
        assert initial_sent == 0
        assert initial_padded == 0
    
    def test_stats_include_cover_traffic(self, node_client_with_cover_traffic):
        """Node stats include cover traffic counters."""
        stats = node_client_with_cover_traffic.get_stats()
        
        assert "cover_messages_sent" in node_client_with_cover_traffic._stats
        assert "cover_messages_received" in node_client_with_cover_traffic._stats
        assert "bytes_padded" in node_client_with_cover_traffic._stats


# =============================================================================
# Padding Integration Tests
# =============================================================================


class TestPaddingIntegration:
    """Tests for padding integration with message sending."""
    
    def test_padding_enabled_in_config(self, node_client_with_cover_traffic):
        """Padding should be enabled when cover traffic is configured."""
        assert node_client_with_cover_traffic.cover_traffic.pad_messages is True
    
    def test_padding_disabled_by_default(self, node_client):
        """Padding is enabled by default but cover traffic is off."""
        # Default config has padding on but cover traffic off
        assert node_client.cover_traffic.pad_messages is True
        assert node_client.cover_traffic.enabled is False
    
    def test_message_indistinguishable_from_cover(self):
        """Real and cover messages should have same size after padding."""
        real_content = b"This is a real message with actual content"
        cover_msg = CoverMessage()
        cover_content = cover_msg.to_bytes()
        
        # Both should pad to same bucket
        real_padded = pad_message(real_content)
        cover_padded = pad_message(cover_content)
        
        # Both small messages should be 1KB
        assert len(real_padded) == 1024
        assert len(cover_padded) == 1024
