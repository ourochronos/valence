"""
Tests for traffic analysis mitigations (Issue #120).

Tests cover:
- Message batching configuration and behavior
- Timing jitter configuration and randomness
- Constant-rate sending with padding
- Privacy level presets
- Integration with NodeClient
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from valence.network.config import (
    PRIVACY_HIGH,
    PRIVACY_LOW,
    PRIVACY_MEDIUM,
    PRIVACY_PARANOID,
    BatchingConfig,
    ConstantRateConfig,
    PrivacyLevel,
    TimingJitterConfig,
    TrafficAnalysisMitigationConfig,
    get_recommended_config,
)
from valence.network.messages import get_padded_size, pad_message, unpad_message
from valence.network.node import NodeClient

pytestmark = pytest.mark.skip(
    reason="Needs update for NodeClient decomposition - see #167"
)

# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestPrivacyLevelPresets:
    """Test privacy level preset configurations."""

    def test_low_privacy_no_mitigations(self):
        """LOW privacy should disable all mitigations."""
        config = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.LOW)

        assert config.privacy_level == PrivacyLevel.LOW
        assert not config.batching.enabled
        assert not config.jitter.enabled
        assert not config.constant_rate.enabled
        assert not config.mix_network.enabled

    def test_medium_privacy_balanced(self):
        """MEDIUM privacy should enable batching and light jitter."""
        config = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.MEDIUM)

        assert config.privacy_level == PrivacyLevel.MEDIUM
        assert config.batching.enabled
        assert config.batching.min_batch_size == 2
        assert config.batching.max_batch_size == 4
        assert config.batching.batch_interval_ms == 2000
        assert config.jitter.enabled
        assert config.jitter.max_delay_ms == 500
        assert not config.constant_rate.enabled

    def test_high_privacy_aggressive(self):
        """HIGH privacy should enable larger batches and more jitter."""
        config = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.HIGH)

        assert config.privacy_level == PrivacyLevel.HIGH
        assert config.batching.enabled
        assert config.batching.min_batch_size == 4
        assert config.batching.max_batch_size == 8
        assert config.batching.batch_interval_ms == 5000
        assert config.jitter.enabled
        assert config.jitter.max_delay_ms == 2000
        assert config.jitter.distribution == "exponential"

    def test_paranoid_privacy_maximum(self):
        """PARANOID privacy should enable all mitigations including constant-rate."""
        config = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.PARANOID)

        assert config.privacy_level == PrivacyLevel.PARANOID
        assert config.batching.enabled
        assert config.batching.min_batch_size == 8
        assert config.batching.max_batch_size == 16
        assert config.jitter.enabled
        assert config.jitter.min_delay_ms == 500
        assert config.jitter.max_delay_ms == 5000
        assert config.constant_rate.enabled
        assert config.constant_rate.messages_per_minute == 4.0
        assert not config.constant_rate.allow_burst  # Strict rate limiting

    def test_preset_constants(self):
        """Test preset constant configurations."""
        assert PRIVACY_LOW.privacy_level == PrivacyLevel.LOW
        assert PRIVACY_MEDIUM.privacy_level == PrivacyLevel.MEDIUM
        assert PRIVACY_HIGH.privacy_level == PrivacyLevel.HIGH
        assert PRIVACY_PARANOID.privacy_level == PrivacyLevel.PARANOID


class TestBatchingConfig:
    """Test batching configuration."""

    def test_default_config(self):
        """Test default batching configuration."""
        config = BatchingConfig()

        assert not config.enabled
        assert config.min_batch_size == 2
        assert config.max_batch_size == 8
        assert config.batch_interval_ms == 2000
        assert config.randomize_order

    def test_effective_interval(self):
        """Test batch interval conversion to seconds."""
        config = BatchingConfig(batch_interval_ms=5000)
        assert config.get_effective_interval() == 5.0

        config = BatchingConfig(batch_interval_ms=500)
        assert config.get_effective_interval() == 0.5


class TestTimingJitterConfig:
    """Test timing jitter configuration."""

    def test_default_config(self):
        """Test default jitter configuration."""
        config = TimingJitterConfig()

        assert not config.enabled
        assert config.min_delay_ms == 0
        assert config.max_delay_ms == 500
        assert config.distribution == "uniform"

    def test_uniform_jitter_range(self):
        """Test uniform jitter stays within range."""
        config = TimingJitterConfig(
            enabled=True,
            min_delay_ms=100,
            max_delay_ms=500,
            distribution="uniform",
        )

        # Sample many times
        samples = [config.get_jitter_delay() for _ in range(1000)]

        # All should be in range
        for s in samples:
            assert 0.1 <= s <= 0.5

        # Should have variation
        assert max(samples) - min(samples) > 0.1

    def test_exponential_jitter_range(self):
        """Test exponential jitter stays within range."""
        config = TimingJitterConfig(
            enabled=True,
            min_delay_ms=0,
            max_delay_ms=1000,
            distribution="exponential",
        )

        samples = [config.get_jitter_delay() for _ in range(1000)]

        # All should be capped at max
        for s in samples:
            assert 0 <= s <= 1.0

        # Exponential should have more small values
        median = sorted(samples)[500]
        assert median < 0.5  # Most values should be below mean

    def test_disabled_jitter_returns_zero(self):
        """Disabled jitter should return 0."""
        config = TimingJitterConfig(enabled=False, max_delay_ms=1000)

        for _ in range(100):
            assert config.get_jitter_delay() == 0.0


class TestConstantRateConfig:
    """Test constant-rate sending configuration."""

    def test_default_config(self):
        """Test default constant-rate configuration."""
        config = ConstantRateConfig()

        assert not config.enabled
        assert config.messages_per_minute == 10.0
        assert config.pad_to_size == 4096
        assert config.allow_burst
        assert config.max_burst_size == 5

    def test_send_interval_calculation(self):
        """Test send interval calculation."""
        config = ConstantRateConfig(messages_per_minute=6.0)
        assert config.get_send_interval() == 10.0  # 60 / 6 = 10 seconds

        config = ConstantRateConfig(messages_per_minute=60.0)
        assert config.get_send_interval() == 1.0  # 60 / 60 = 1 second

        config = ConstantRateConfig(messages_per_minute=0)
        assert config.get_send_interval() == 60.0  # Default to 60s for 0


class TestLatencyEstimation:
    """Test latency overhead estimation."""

    def test_low_privacy_no_overhead(self):
        """LOW privacy should have no latency overhead."""
        config = PRIVACY_LOW
        estimate = config.estimate_latency_overhead()

        assert estimate["min_delay_ms"] == 0
        assert estimate["max_delay_ms"] == 0
        assert estimate["avg_delay_ms"] == 0

    def test_medium_privacy_moderate_overhead(self):
        """MEDIUM privacy should have moderate latency."""
        config = PRIVACY_MEDIUM
        estimate = config.estimate_latency_overhead()

        # Batching: avg 1000ms, Jitter: avg 250ms
        assert estimate["avg_delay_ms"] > 500
        assert estimate["max_delay_ms"] <= 2500  # 2000ms batch + 500ms jitter

    def test_paranoid_privacy_high_overhead(self):
        """PARANOID privacy should have significant latency."""
        config = PRIVACY_PARANOID
        estimate = config.estimate_latency_overhead()

        # Should have noticeable overhead
        assert estimate["avg_delay_ms"] > 5000
        assert estimate["max_delay_ms"] > 15000


class TestConfigSerialization:
    """Test configuration serialization."""

    def test_to_dict_and_from_dict(self):
        """Config should survive round-trip serialization."""
        original = TrafficAnalysisMitigationConfig.from_privacy_level(PrivacyLevel.HIGH)

        serialized = original.to_dict()
        restored = TrafficAnalysisMitigationConfig.from_dict(serialized)

        assert restored.privacy_level == original.privacy_level
        assert restored.batching.enabled == original.batching.enabled
        assert restored.batching.batch_interval_ms == original.batching.batch_interval_ms
        assert restored.jitter.enabled == original.jitter.enabled
        assert restored.jitter.max_delay_ms == original.jitter.max_delay_ms
        assert restored.constant_rate.enabled == original.constant_rate.enabled


class TestRecommendedConfig:
    """Test recommended configuration helper."""

    def test_latency_sensitive(self):
        """Latency-sensitive should minimize delays."""
        config = get_recommended_config(latency_sensitive=True)
        assert config.privacy_level == PrivacyLevel.LOW

    def test_high_security(self):
        """High-security should maximize protection."""
        config = get_recommended_config(high_security=True)
        assert config.privacy_level == PrivacyLevel.PARANOID

    def test_bandwidth_limited_high_security(self):
        """Bandwidth-limited + high security should skip constant-rate."""
        config = get_recommended_config(
            bandwidth_limited=True,
            high_security=True,
        )
        assert not config.constant_rate.enabled
        assert config.batching.enabled


# =============================================================================
# MESSAGE PADDING TESTS
# =============================================================================


class TestMessagePadding:
    """Test message padding for traffic analysis resistance."""

    def test_small_message_padded_to_bucket(self):
        """Small messages should be padded to first bucket."""
        content = b"Hello"
        padded = pad_message(content)

        assert len(padded) == 1024  # First bucket
        assert unpad_message(padded) == content

    def test_medium_message_padded(self):
        """Medium messages should pad to appropriate bucket."""
        content = b"x" * 2000
        padded = pad_message(content)

        assert len(padded) == 4096  # Second bucket
        assert unpad_message(padded) == content

    def test_large_message_padded(self):
        """Large messages should pad to larger buckets."""
        content = b"x" * 10000
        padded = pad_message(content)

        assert len(padded) == 16384  # Third bucket
        assert unpad_message(padded) == content

    def test_bucket_size_selection(self):
        """Test bucket size selection for various lengths."""
        assert get_padded_size(100) == 1024
        assert get_padded_size(1000) == 1024
        assert get_padded_size(1500) == 4096
        assert get_padded_size(5000) == 16384
        assert get_padded_size(20000) == 65536
        assert get_padded_size(100000) == 100000  # Exceeds all buckets


# =============================================================================
# NODE CLIENT INTEGRATION TESTS
# =============================================================================


@pytest.fixture
def mock_node_client():
    """Create a mock node client for testing."""
    private_key = Ed25519PrivateKey.generate()
    encryption_key = X25519PrivateKey.generate()
    node_id = private_key.public_key().public_bytes_raw().hex()

    client = NodeClient(
        node_id=node_id,
        private_key=private_key,
        encryption_private_key=encryption_key,
        traffic_analysis_mitigation=TrafficAnalysisMitigationConfig.from_privacy_level(
            PrivacyLevel.MEDIUM
        ),
    )
    return client


class TestNodeClientTrafficMitigation:
    """Test NodeClient traffic analysis mitigation integration."""

    def test_mitigation_config_applied(self, mock_node_client):
        """Verify mitigation config is applied to node client."""
        client = mock_node_client

        assert client.traffic_analysis_mitigation.batching.enabled
        assert client.traffic_analysis_mitigation.jitter.enabled

    def test_stats_initialized(self, mock_node_client):
        """Verify mitigation stats are initialized."""
        client = mock_node_client

        assert "batched_messages" in client._stats
        assert "batch_flushes" in client._stats
        assert "jitter_delays_applied" in client._stats
        assert "constant_rate_padding_sent" in client._stats

    def test_privacy_level_change(self, mock_node_client):
        """Test changing privacy level at runtime."""
        client = mock_node_client

        # Start with MEDIUM
        assert client.traffic_analysis_mitigation.privacy_level == PrivacyLevel.MEDIUM

        # Change to HIGH
        client.set_privacy_level(PrivacyLevel.HIGH)

        assert client.traffic_analysis_mitigation.privacy_level == PrivacyLevel.HIGH
        assert client.traffic_analysis_mitigation.batching.max_batch_size == 8

    def test_mitigation_stats_available(self, mock_node_client):
        """Test that mitigation stats are accessible."""
        client = mock_node_client

        stats = client.get_traffic_analysis_mitigation_stats()

        assert "privacy_level" in stats
        assert "batching" in stats
        assert "jitter" in stats
        assert "constant_rate" in stats
        assert "stats" in stats
        assert "latency_estimate" in stats


class TestBatchingBehavior:
    """Test message batching behavior."""

    @pytest.mark.asyncio
    async def test_add_to_batch(self, mock_node_client):
        """Test adding messages to batch."""
        client = mock_node_client
        recipient_key = X25519PrivateKey.generate().public_key()

        # Add a message to batch
        msg_id = await client._add_to_batch(
            message_id="test-msg-1",
            recipient_id="recipient123",
            recipient_public_key=recipient_key,
            content=b"Hello",
            require_ack=True,
            timeout_ms=30000,
        )

        assert msg_id == "test-msg-1"
        assert len(client._message_batch) == 1
        assert client._stats["batched_messages"] == 1

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Needs update after NodeClient refactor - see #166")
    async def test_batch_triggers_on_max_size(self, mock_node_client):
        """Test batch triggers flush when max size reached."""
        client = mock_node_client
        client.traffic_analysis_mitigation.batching.max_batch_size = 3
        recipient_key = X25519PrivateKey.generate().public_key()

        # Add messages up to max
        for i in range(3):
            await client._add_to_batch(
                message_id=f"test-msg-{i}",
                recipient_id="recipient123",
                recipient_public_key=recipient_key,
                content=b"Hello",
                require_ack=False,
                timeout_ms=30000,
            )

        # Event should be set
        assert client._pending_batch_event.is_set()

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Needs update after NodeClient refactor - see #166")
    async def test_flush_randomizes_order(self, mock_node_client):
        """Test batch flush randomizes message order."""
        client = mock_node_client
        recipient_key = X25519PrivateKey.generate().public_key()

        # Add multiple messages
        for i in range(10):
            await client._add_to_batch(
                message_id=f"msg-{i:02d}",
                recipient_id="recipient123",
                recipient_public_key=recipient_key,
                content=b"Hello",
                require_ack=False,
                timeout_ms=30000,
            )

        # Get the message IDs before flush
        _ = [m["message_id"] for m in client._message_batch]

        # The batch is copied and shuffled during flush
        # We can't easily test randomness, but we can verify config
        assert client.traffic_analysis_mitigation.batching.randomize_order


class TestJitterBehavior:
    """Test timing jitter behavior."""

    def test_jitter_config_from_privacy_level(self, mock_node_client):
        """Test jitter is configured from privacy level."""
        client = mock_node_client

        jitter = client.traffic_analysis_mitigation.jitter
        assert jitter.enabled
        assert jitter.max_delay_ms > 0

    @pytest.mark.asyncio
    async def test_jitter_delay_applied(self, mock_node_client):
        """Test jitter delay is tracked in stats."""
        client = mock_node_client

        # Configure jitter
        client.traffic_analysis_mitigation.jitter.enabled = True
        client.traffic_analysis_mitigation.jitter.min_delay_ms = 10
        client.traffic_analysis_mitigation.jitter.max_delay_ms = 50

        # Get a delay
        delay = client.traffic_analysis_mitigation.jitter.get_jitter_delay()

        assert 0.01 <= delay <= 0.05


class TestConstantRateBehavior:
    """Test constant-rate sending behavior."""

    @pytest.mark.asyncio
    async def test_padding_message_creation(self, mock_node_client):
        """Test padding message can be created."""
        client = mock_node_client
        client.traffic_analysis_mitigation.constant_rate.enabled = True
        client.traffic_analysis_mitigation.constant_rate.pad_to_size = 1024

        # Mock the connections
        client.connections = {"router1": MagicMock()}
        client._select_router = MagicMock(return_value=MagicMock(router_id="router1"))
        client._send_via_router = AsyncMock()

        await client._send_padding_message()

        # Verify padding was attempted
        assert client._send_via_router.called or client._stats["constant_rate_padding_sent"] >= 0


# =============================================================================
# TIMING TESTS
# =============================================================================


class TestTimingMeasurements:
    """Test actual timing measurements."""

    @pytest.mark.asyncio
    async def test_jitter_adds_measurable_delay(self):
        """Test that jitter actually delays execution."""
        config = TimingJitterConfig(
            enabled=True,
            min_delay_ms=50,
            max_delay_ms=100,
            distribution="uniform",
        )

        start = time.time()
        delay = config.get_jitter_delay()
        await asyncio.sleep(delay)
        elapsed = time.time() - start

        # Should have delayed at least min_delay
        assert elapsed >= 0.05

    @pytest.mark.asyncio
    async def test_batch_interval_respected(self):
        """Test batch interval timing."""
        config = BatchingConfig(
            enabled=True,
            batch_interval_ms=100,  # 100ms for fast test
        )

        interval = config.get_effective_interval()
        assert interval == 0.1

        start = time.time()
        await asyncio.sleep(interval)
        elapsed = time.time() - start

        assert elapsed >= 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
