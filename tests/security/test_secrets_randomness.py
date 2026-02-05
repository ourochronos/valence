"""
Tests for cryptographically secure randomness (Issue #170).

Verifies that security-sensitive operations use the secrets module
instead of the standard random module, which is predictable if seeded.

Security-sensitive operations that require secrets:
- Timing jitter (traffic analysis resistance)
- Message bucket selection (cover traffic)
- Router selection (prevents predictable routing)
- Router sampling (prevents predictable patterns)
- Privacy weight decisions

Non-security operations that can use random/np.random:
- Differential privacy noise (needs specific distributions)
- Test data generation
- UX shuffling
"""

import secrets
from unittest.mock import patch

import pytest


class TestSecretsModuleUsage:
    """Verify security-sensitive code uses secrets module."""

    def test_config_uses_secure_random(self):
        """TimingJitterConfig should use secrets.SystemRandom."""
        from valence.network.config import _secure_random
        
        assert isinstance(_secure_random, secrets.SystemRandom), (
            "config._secure_random must be secrets.SystemRandom"
        )

    def test_messages_uses_secure_random(self):
        """Messages module should use secrets.SystemRandom."""
        from valence.network.messages import _secure_random
        
        assert isinstance(_secure_random, secrets.SystemRandom), (
            "messages._secure_random must be secrets.SystemRandom"
        )

    def test_router_client_uses_secure_random(self):
        """RouterClient should use secrets.SystemRandom."""
        from valence.network.router_client import _secure_random
        
        assert isinstance(_secure_random, secrets.SystemRandom), (
            "router_client._secure_random must be secrets.SystemRandom"
        )

    def test_seed_uses_secure_random(self):
        """Seed module should use secrets.SystemRandom."""
        from valence.network.seed import _secure_random
        
        assert isinstance(_secure_random, secrets.SystemRandom), (
            "seed._secure_random must be secrets.SystemRandom"
        )

    def test_privacy_uses_secure_random(self):
        """Privacy module should use secrets.SystemRandom."""
        from valence.federation.privacy import _secure_random
        
        assert isinstance(_secure_random, secrets.SystemRandom), (
            "privacy._secure_random must be secrets.SystemRandom"
        )


class TestJitterRandomness:
    """Test timing jitter uses secure randomness."""

    def test_jitter_uniform_distribution(self):
        """Uniform jitter should produce values in expected range."""
        from valence.network.config import TimingJitterConfig
        
        config = TimingJitterConfig(
            enabled=True,
            min_delay_ms=100,
            max_delay_ms=500,
            distribution="uniform"
        )
        
        # Generate multiple samples
        delays = [config.get_jitter_delay() for _ in range(100)]
        
        # All should be in range (converted to seconds)
        for delay in delays:
            assert 0.1 <= delay <= 0.5, f"Delay {delay} out of range"
        
        # Should have some variance (not constant)
        assert len(set(delays)) > 1, "Jitter should produce varying delays"

    def test_jitter_exponential_distribution(self):
        """Exponential jitter should produce values in expected range."""
        from valence.network.config import TimingJitterConfig
        
        config = TimingJitterConfig(
            enabled=True,
            min_delay_ms=100,
            max_delay_ms=500,
            distribution="exponential"
        )
        
        # Generate multiple samples
        delays = [config.get_jitter_delay() for _ in range(100)]
        
        # All should be in range (min to max, converted to seconds)
        for delay in delays:
            assert 0.1 <= delay <= 0.5, f"Delay {delay} out of range"

    def test_jitter_disabled_returns_zero(self):
        """Disabled jitter should return 0."""
        from valence.network.config import TimingJitterConfig
        
        config = TimingJitterConfig(enabled=False)
        assert config.get_jitter_delay() == 0.0


class TestCoverTrafficRandomness:
    """Test cover traffic uses secure randomness."""

    def test_cover_content_bucket_selection(self):
        """Cover content should select from valid buckets."""
        from valence.network.messages import generate_cover_content, MESSAGE_SIZE_BUCKETS
        
        # Generate multiple cover messages without specifying bucket
        # This exercises the weighted random selection
        sizes = set()
        for _ in range(50):
            content = generate_cover_content()
            # Content size + overhead should be <= bucket size
            assert len(content) > 0
            sizes.add(len(content))
        
        # Should have some variety in sizes selected
        assert len(sizes) > 1, "Cover content should vary in size"

    def test_cover_content_specific_bucket(self):
        """Cover content with specific bucket should match."""
        from valence.network.messages import generate_cover_content, MESSAGE_SIZE_BUCKETS
        
        for bucket in MESSAGE_SIZE_BUCKETS:
            content = generate_cover_content(target_bucket=bucket)
            # Content should fit in bucket (with overhead)
            assert len(content) <= bucket


class TestRouterSelectionRandomness:
    """Test router selection uses secure randomness."""

    def test_router_selection_weighted(self):
        """Router selection should respect weights but be unpredictable."""
        from valence.network.router_client import _secure_random
        
        # Create mock candidates with varying weights
        candidates = ['router_a', 'router_b', 'router_c']
        weights = [0.1, 0.1, 0.8]  # router_c heavily weighted
        
        # Select many times
        selections = [
            _secure_random.choices(candidates, weights=weights, k=1)[0]
            for _ in range(100)
        ]
        
        # Should have variety (not deterministic)
        assert len(set(selections)) > 1, "Selection should vary"
        
        # router_c should be selected most often (but not exclusively due to randomness)
        c_count = selections.count('router_c')
        assert c_count > 30, f"Heavy-weighted router should appear often, got {c_count}"


class TestSeedSamplingRandomness:
    """Test seed router sampling uses secure randomness."""

    def test_sample_unpredictable(self):
        """Router sampling should be unpredictable."""
        from valence.network.seed import _secure_random
        
        population = list(range(100))
        
        # Sample multiple times
        samples = [tuple(_secure_random.sample(population, 10)) for _ in range(20)]
        
        # Should have variety
        unique_samples = set(samples)
        assert len(unique_samples) > 1, "Samples should vary"


class TestPrivacyDecisionRandomness:
    """Test privacy decisions use secure randomness."""

    def test_random_produces_valid_range(self):
        """Privacy random() should produce values in [0, 1)."""
        from valence.federation.privacy import _secure_random
        
        values = [_secure_random.random() for _ in range(100)]
        
        for v in values:
            assert 0.0 <= v < 1.0, f"Value {v} out of range"
        
        # Should have variance
        assert len(set(values)) > 1, "Should produce varying values"


class TestNoStandardRandomInSecurityCode:
    """Verify standard random module is not used for security operations."""

    def test_config_no_standard_random_import(self):
        """config.py should not use standard random for security operations."""
        import inspect
        from valence.network import config
        
        source = inspect.getsource(config)
        
        # Should import secrets, not just random
        assert "import secrets" in source, "Should import secrets module"
        
        # _secure_random should be used
        assert "_secure_random" in source, "Should use _secure_random"

    def test_messages_no_inline_random_import(self):
        """messages.py should not have inline random import for security ops."""
        import inspect
        from valence.network import messages
        
        source = inspect.getsource(messages)
        
        # Should NOT have inline "import random" in the cover traffic function
        # Check the generate_cover_content function doesn't do "import random"
        assert "import secrets" in source, "Should import secrets module"
        
        # The old inline import should be removed
        assert source.count("import random") == 0, (
            "Should not have 'import random' in security-sensitive code"
        )

    def test_router_client_no_standard_random(self):
        """router_client.py should use secrets, not standard random."""
        import inspect
        from valence.network import router_client
        
        source = inspect.getsource(router_client)
        
        assert "import secrets" in source
        assert "_secure_random" in source
        # Standard random should not be imported
        assert "import random" not in source.replace("_secure_random", ""), (
            "Should not import standard random module"
        )
