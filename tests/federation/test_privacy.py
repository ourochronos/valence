"""Tests for differential privacy implementation.

Tests cover:
- Privacy configuration validation
- Privacy budget tracking and enforcement
- Noise mechanisms (Laplace, Gaussian)
- Temporal smoothing for membership changes
- Histogram suppression
- Sensitive domain detection
- Rate limiting
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import numpy as np
import pytest
from valence.federation.privacy import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    DEFAULT_MIN_CONTRIBUTORS,
    MAX_EPSILON,
    MAX_QUERIES_PER_FEDERATION_PER_DAY,
    MAX_QUERIES_PER_TOPIC_PER_DAY,
    # Constants
    MIN_EPSILON,
    SENSITIVE_MIN_CONTRIBUTORS,
    BudgetCheckResult,
    NoiseMechanism,
    PrivacyBudget,
    # Classes
    PrivacyConfig,
    # Enums
    PrivacyLevel,
    PrivateAggregateResult,
    RequesterBudget,
    TemporalSmoother,
    add_gaussian_noise,
    # Functions
    add_laplace_noise,
    add_noise,
    build_noisy_histogram,
    compute_private_aggregate,
    compute_topic_hash,
    is_sensitive_domain,
    should_include_histogram,
)

# =============================================================================
# PRIVACY CONFIG TESTS
# =============================================================================


class TestPrivacyConfig:
    """Tests for PrivacyConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PrivacyConfig()

        assert config.epsilon == DEFAULT_EPSILON
        assert config.delta == DEFAULT_DELTA
        assert config.min_contributors == DEFAULT_MIN_CONTRIBUTORS
        assert not config.sensitive_domain

    def test_epsilon_bounds_validation(self) -> None:
        """Test that epsilon must be within bounds."""
        # Below minimum
        with pytest.raises(ValueError, match="epsilon must be in"):
            PrivacyConfig(epsilon=0.001)

        # Above maximum
        with pytest.raises(ValueError, match="epsilon must be in"):
            PrivacyConfig(epsilon=5.0)

        # At bounds - should work
        PrivacyConfig(epsilon=MIN_EPSILON)
        PrivacyConfig(epsilon=MAX_EPSILON)

    def test_delta_validation(self) -> None:
        """Test that delta must be small."""
        with pytest.raises(ValueError, match="delta must be < 10"):
            PrivacyConfig(delta=0.001)  # 10⁻³ is too large

        # Valid delta
        PrivacyConfig(delta=1e-5)

    def test_min_contributors_validation(self) -> None:
        """Test that min_contributors >= 5."""
        with pytest.raises(ValueError, match="min_contributors must be >= 5"):
            PrivacyConfig(min_contributors=3)

        # Valid
        PrivacyConfig(min_contributors=5)
        PrivacyConfig(min_contributors=10)

    def test_from_level_maximum(self) -> None:
        """Test MAXIMUM privacy level."""
        config = PrivacyConfig.from_level(PrivacyLevel.MAXIMUM)

        assert config.epsilon == 0.1
        assert config.delta == 1e-8
        assert config.min_contributors == 10

    def test_from_level_standard(self) -> None:
        """Test STANDARD privacy level."""
        config = PrivacyConfig.from_level(PrivacyLevel.STANDARD)

        assert config.epsilon == 1.0
        assert config.delta == 1e-6
        assert config.min_contributors == 5

    def test_effective_min_contributors_normal(self) -> None:
        """Test effective min contributors for normal federation."""
        config = PrivacyConfig(min_contributors=5, sensitive_domain=False)
        assert config.effective_min_contributors == 5

    def test_effective_min_contributors_sensitive(self) -> None:
        """Test effective min contributors for sensitive federation."""
        config = PrivacyConfig(min_contributors=5, sensitive_domain=True)
        assert config.effective_min_contributors == SENSITIVE_MIN_CONTRIBUTORS

        # Higher configured value preserved
        config = PrivacyConfig(min_contributors=15, sensitive_domain=True)
        assert config.effective_min_contributors == 15

    def test_to_dict(self) -> None:
        """Test config serialization."""
        config = PrivacyConfig(epsilon=0.5, sensitive_domain=True)
        d = config.to_dict()

        assert d["epsilon"] == 0.5
        assert d["sensitive_domain"] is True
        assert "effective_min_contributors" in d


# =============================================================================
# PRIVACY BUDGET TESTS
# =============================================================================


class TestPrivacyBudget:
    """Tests for PrivacyBudget class."""

    @pytest.fixture
    def budget(self) -> PrivacyBudget:
        """Create a fresh budget for testing."""
        return PrivacyBudget(federation_id=uuid4())

    def test_initial_budget(self, budget: PrivacyBudget) -> None:
        """Test initial budget state."""
        assert budget.spent_epsilon == 0.0
        assert budget.spent_delta == 0.0
        assert budget.queries_today == 0

    def test_check_budget_ok(self, budget: PrivacyBudget) -> None:
        """Test budget check when sufficient budget."""
        can_query, result = budget.check_budget(
            epsilon=1.0,
            delta=1e-6,
            topic_hash="test_topic",
        )

        assert can_query is True
        assert result == BudgetCheckResult.OK

    def test_consume_budget(self, budget: PrivacyBudget) -> None:
        """Test budget consumption."""
        budget.consume(
            epsilon=1.0,
            delta=1e-6,
            topic_hash="test_topic",
        )

        assert budget.spent_epsilon == 1.0
        assert budget.spent_delta == 1e-6
        assert budget.queries_today == 1

    def test_epsilon_exhausted(self, budget: PrivacyBudget) -> None:
        """Test behavior when epsilon budget exhausted."""
        # Exhaust the budget
        budget.spent_epsilon = budget.daily_epsilon_budget

        can_query, result = budget.check_budget(
            epsilon=1.0,
            delta=1e-6,
            topic_hash="test",
        )

        assert can_query is False
        assert result == BudgetCheckResult.DAILY_EPSILON_EXHAUSTED

    def test_topic_rate_limit(self, budget: PrivacyBudget) -> None:
        """Test per-topic rate limiting."""
        topic = "repeated_topic"

        # Exhaust topic queries
        for _ in range(MAX_QUERIES_PER_TOPIC_PER_DAY):
            can_query, result = budget.check_budget(1.0, 1e-6, topic)
            assert can_query is True
            budget.consume(1.0, 1e-6, topic)

        # Next query should be rate limited
        can_query, result = budget.check_budget(1.0, 1e-6, topic)
        assert can_query is False
        assert result == BudgetCheckResult.TOPIC_RATE_LIMITED

    def test_federation_rate_limit(self, budget: PrivacyBudget) -> None:
        """Test federation-wide rate limiting."""
        # Set queries near limit
        budget.queries_today = MAX_QUERIES_PER_FEDERATION_PER_DAY

        can_query, result = budget.check_budget(0.01, 1e-7, "topic")
        assert can_query is False
        assert result == BudgetCheckResult.FEDERATION_RATE_LIMITED

    def test_requester_rate_limit(self, budget: PrivacyBudget) -> None:
        """Test per-requester rate limiting."""
        requester = "attacker_123"

        # Create requester budget at limit
        budget.requester_budgets[requester] = RequesterBudget(
            requester_id=requester,
            queries_this_hour=20,
        )

        can_query, result = budget.check_budget(1.0, 1e-6, "topic", requester_id=requester)
        assert can_query is False
        assert result == BudgetCheckResult.REQUESTER_RATE_LIMITED

    def test_budget_reset(self, budget: PrivacyBudget) -> None:
        """Test budget resets after period."""
        # Spend some budget
        budget.spent_epsilon = 5.0
        budget.queries_today = 50

        # Set period start to yesterday
        budget.period_start = datetime.now(UTC) - timedelta(hours=25)

        # Check should trigger reset
        remaining = budget.remaining_epsilon()

        assert remaining == budget.daily_epsilon_budget
        assert budget.spent_epsilon == 0.0
        assert budget.queries_today == 0

    def test_topic_queries_remaining(self, budget: PrivacyBudget) -> None:
        """Test remaining topic queries calculation."""
        topic = "test_topic"

        # New topic should have full allowance
        assert budget.topic_queries_remaining(topic) == MAX_QUERIES_PER_TOPIC_PER_DAY

        # After some queries
        budget.consume(1.0, 1e-6, topic)
        budget.consume(1.0, 1e-6, topic)

        assert budget.topic_queries_remaining(topic) == MAX_QUERIES_PER_TOPIC_PER_DAY - 2


# =============================================================================
# NOISE MECHANISM TESTS
# =============================================================================


class TestNoiseMechanisms:
    """Tests for noise mechanisms."""

    def test_laplace_noise_distribution(self) -> None:
        """Test Laplace noise has correct distribution."""
        np.random.seed(42)

        true_value = 100.0
        sensitivity = 1.0
        epsilon = 1.0

        # Generate many samples
        samples = [add_laplace_noise(true_value, sensitivity, epsilon) for _ in range(10000)]

        # Mean should be close to true value
        assert abs(np.mean(samples) - true_value) < 0.5

        # Variance should be 2(sensitivity/epsilon)²
        expected_variance = 2 * (sensitivity / epsilon) ** 2
        assert abs(np.var(samples) - expected_variance) < 0.5

    def test_gaussian_noise_distribution(self) -> None:
        """Test Gaussian noise has correct distribution."""
        np.random.seed(42)

        true_value = 100.0
        sensitivity = 1.0
        epsilon = 1.0
        delta = 1e-5

        # Generate samples
        samples = [add_gaussian_noise(true_value, sensitivity, epsilon, delta) for _ in range(10000)]

        # Mean should be close to true value
        assert abs(np.mean(samples) - true_value) < 0.5

    def test_noise_invalid_epsilon(self) -> None:
        """Test noise functions reject invalid epsilon."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            add_laplace_noise(1.0, 1.0, 0.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            add_laplace_noise(1.0, 1.0, -1.0)

    def test_gaussian_invalid_delta(self) -> None:
        """Test Gaussian noise rejects invalid delta."""
        with pytest.raises(ValueError, match="delta must be positive"):
            add_gaussian_noise(1.0, 1.0, 1.0, 0.0)

    def test_add_noise_uses_config(self) -> None:
        """Test add_noise respects config mechanism."""
        np.random.seed(42)

        config_laplace = PrivacyConfig(noise_mechanism=NoiseMechanism.LAPLACE)
        config_gaussian = PrivacyConfig(noise_mechanism=NoiseMechanism.GAUSSIAN)

        # Both should produce noisy output
        noisy_laplace = add_noise(100.0, 1.0, config_laplace)
        noisy_gaussian = add_noise(100.0, 1.0, config_gaussian)

        assert noisy_laplace != 100.0
        assert noisy_gaussian != 100.0

    def test_lower_epsilon_more_noise(self) -> None:
        """Test that lower epsilon produces more noise (better privacy)."""
        np.random.seed(42)

        true_value = 100.0
        sensitivity = 1.0

        # High privacy (low epsilon)
        high_privacy_samples = [add_laplace_noise(true_value, sensitivity, 0.1) for _ in range(1000)]

        # Low privacy (high epsilon)
        low_privacy_samples = [add_laplace_noise(true_value, sensitivity, 2.0) for _ in range(1000)]

        # High privacy should have higher variance
        high_var = np.var(high_privacy_samples)
        low_var = np.var(low_privacy_samples)

        assert high_var > low_var


# =============================================================================
# TEMPORAL SMOOTHING TESTS
# =============================================================================


class TestTemporalSmoother:
    """Tests for temporal smoothing."""

    @pytest.fixture
    def smoother(self) -> TemporalSmoother:
        """Create a smoother with 24h window."""
        return TemporalSmoother(smoothing_hours=24)

    def test_founding_member_full_weight(self, smoother: TemporalSmoother) -> None:
        """Test founding members have full weight."""
        weight = smoother.get_contribution_weight(
            member_id="founder",
            joined_at=None,  # Founding member
            departed_at=None,
        )
        assert weight == 1.0

    def test_active_member_full_weight(self, smoother: TemporalSmoother) -> None:
        """Test long-standing active members have full weight."""
        old_join = datetime.now(UTC) - timedelta(days=30)

        weight = smoother.get_contribution_weight(
            member_id="veteran",
            joined_at=old_join,
            departed_at=None,
        )
        assert weight == 1.0

    def test_new_member_ramping_weight(self, smoother: TemporalSmoother) -> None:
        """Test new members have ramping weight."""
        # Joined 12 hours ago (half of 24h window)
        recent_join = datetime.now(UTC) - timedelta(hours=12)

        weight = smoother.get_contribution_weight(
            member_id="newbie",
            joined_at=recent_join,
            departed_at=None,
        )

        # Should be approximately 0.5
        assert 0.4 <= weight <= 0.6

    def test_very_new_member_low_weight(self, smoother: TemporalSmoother) -> None:
        """Test very new members have low weight."""
        just_joined = datetime.now(UTC) - timedelta(hours=1)

        weight = smoother.get_contribution_weight(
            member_id="newest",
            joined_at=just_joined,
            departed_at=None,
        )

        # Should be approximately 1/24
        assert weight < 0.1

    def test_recently_departed_included(self, smoother: TemporalSmoother) -> None:
        """Test recently departed members may be included."""
        # Set seed for deterministic test
        np.random.seed(42)

        departed_6h_ago = datetime.now(UTC) - timedelta(hours=6)

        # Check multiple times - should sometimes include
        included_count = 0
        for _ in range(100):
            if smoother.should_include_member(
                member_id="departed",
                joined_at=datetime.now(UTC) - timedelta(days=30),
                departed_at=departed_6h_ago,
            ):
                included_count += 1

        # Should be included roughly 75% of the time ((24-6)/24)
        assert 50 < included_count < 95

    def test_long_departed_excluded(self, smoother: TemporalSmoother) -> None:
        """Test members departed beyond window are excluded."""
        departed_long_ago = datetime.now(UTC) - timedelta(hours=25)

        included = smoother.should_include_member(
            member_id="long_gone",
            joined_at=datetime.now(UTC) - timedelta(days=30),
            departed_at=departed_long_ago,
        )

        assert included is False


# =============================================================================
# HISTOGRAM TESTS
# =============================================================================


class TestHistogramHandling:
    """Tests for histogram suppression and noise."""

    def test_histogram_suppressed_below_threshold(self) -> None:
        """Test histogram suppressed when contributors below threshold."""
        assert should_include_histogram(5) is False
        assert should_include_histogram(19) is False

    def test_histogram_included_above_threshold(self) -> None:
        """Test histogram included when contributors above threshold."""
        assert should_include_histogram(20) is True
        assert should_include_histogram(100) is True

    def test_custom_threshold(self) -> None:
        """Test custom histogram threshold."""
        assert should_include_histogram(15, threshold=10) is True
        assert should_include_histogram(8, threshold=10) is False

    def test_noisy_histogram_structure(self) -> None:
        """Test noisy histogram has correct structure."""
        values = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8]

        histogram = build_noisy_histogram(values, epsilon=1.0, bins=5)

        # Should have 5 bins
        assert len(histogram) == 5

        # Bins should cover [0, 1]
        assert "0.0-0.2" in histogram
        assert "0.8-1.0" in histogram

    def test_noisy_histogram_counts_noisy(self) -> None:
        """Test histogram counts have noise."""
        np.random.seed(42)

        # Create uniform distribution
        values = [i / 10 for i in range(100)]

        histogram1 = build_noisy_histogram(values, epsilon=1.0, bins=5)
        histogram2 = build_noisy_histogram(values, epsilon=1.0, bins=5)

        # Different noise should give different results
        assert histogram1 != histogram2

    def test_histogram_values_clamped(self) -> None:
        """Test out-of-range values are clamped."""
        values = [-0.5, 0.5, 1.5]  # Includes out-of-range

        # Should not raise
        histogram = build_noisy_histogram(values, epsilon=1.0, bins=5)

        # All values should be in some bin
        assert sum(histogram.values()) > 0


# =============================================================================
# SENSITIVE DOMAIN TESTS
# =============================================================================


class TestSensitiveDomains:
    """Tests for sensitive domain detection."""

    def test_health_domain_sensitive(self) -> None:
        """Test health-related domains are sensitive."""
        assert is_sensitive_domain(["health"]) is True
        assert is_sensitive_domain(["medical", "diagnosis"]) is True
        assert is_sensitive_domain(["mental_health", "therapy"]) is True

    def test_finance_domain_sensitive(self) -> None:
        """Test finance-related domains are sensitive."""
        assert is_sensitive_domain(["finance"]) is True
        assert is_sensitive_domain(["banking", "accounts"]) is True
        assert is_sensitive_domain(["salary", "compensation"]) is True

    def test_legal_domain_sensitive(self) -> None:
        """Test legal-related domains are sensitive."""
        assert is_sensitive_domain(["legal", "contracts"]) is True
        assert is_sensitive_domain(["criminal", "defense"]) is True

    def test_general_domain_not_sensitive(self) -> None:
        """Test general domains are not sensitive."""
        assert is_sensitive_domain(["technology"]) is False
        assert is_sensitive_domain(["cooking", "recipes"]) is False
        assert is_sensitive_domain(["sports", "football"]) is False

    def test_case_insensitive(self) -> None:
        """Test detection is case insensitive."""
        assert is_sensitive_domain(["HEALTH"]) is True
        assert is_sensitive_domain(["Medical"]) is True
        assert is_sensitive_domain(["FINANCE"]) is True

    def test_substring_matching(self) -> None:
        """Test domains containing sensitive terms are detected."""
        assert is_sensitive_domain(["healthcare"]) is True
        assert is_sensitive_domain(["finance_planning"]) is True  # Contains "finance"
        assert is_sensitive_domain(["legal_documents"]) is True


# =============================================================================
# TOPIC HASH TESTS
# =============================================================================


class TestTopicHash:
    """Tests for topic hashing."""

    def test_deterministic_hash(self) -> None:
        """Test same input produces same hash."""
        domains = ["tech", "programming"]
        query = "python best practices"

        hash1 = compute_topic_hash(domains, query)
        hash2 = compute_topic_hash(domains, query)

        assert hash1 == hash2

    def test_different_domains_different_hash(self) -> None:
        """Test different domains produce different hashes."""
        hash1 = compute_topic_hash(["tech"])
        hash2 = compute_topic_hash(["health"])

        assert hash1 != hash2

    def test_domain_order_normalized(self) -> None:
        """Test domain order doesn't affect hash."""
        hash1 = compute_topic_hash(["a", "b", "c"])
        hash2 = compute_topic_hash(["c", "a", "b"])

        assert hash1 == hash2

    def test_semantic_query_affects_hash(self) -> None:
        """Test semantic query changes hash."""
        domains = ["tech"]

        hash1 = compute_topic_hash(domains, "python")
        hash2 = compute_topic_hash(domains, "javascript")
        hash3 = compute_topic_hash(domains, None)

        assert hash1 != hash2
        assert hash1 != hash3


# =============================================================================
# PRIVATE AGGREGATE TESTS
# =============================================================================


class TestPrivateAggregate:
    """Tests for compute_private_aggregate."""

    @pytest.fixture
    def config(self) -> PrivacyConfig:
        """Create a standard config."""
        return PrivacyConfig()

    def test_insufficient_contributors_returns_none(self, config: PrivacyConfig) -> None:
        """Test aggregate returns None when k-anonymity not satisfied."""
        # Only 3 contributors, need 5
        confidences = [0.5, 0.6, 0.7]

        result = compute_private_aggregate(confidences, config)

        assert result is None

    def test_sufficient_contributors_returns_result(self, config: PrivacyConfig) -> None:
        """Test aggregate returns result when k-anonymity satisfied."""
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]  # 5 contributors

        result = compute_private_aggregate(confidences, config)

        assert result is not None
        assert isinstance(result, PrivateAggregateResult)

    def test_noisy_mean_close_to_true(self, config: PrivacyConfig) -> None:
        """Test noisy mean is reasonably close to true mean."""
        np.random.seed(42)

        # Many contributors for more stable test
        confidences = [0.5] * 50

        result = compute_private_aggregate(confidences, config)

        # Should be within reasonable range of 0.5
        assert 0.3 <= result.collective_confidence <= 0.7

    def test_noisy_count_close_to_true(self, config: PrivacyConfig) -> None:
        """Test noisy count is reasonably close to true count."""
        np.random.seed(42)

        confidences = [0.5] * 100

        result = compute_private_aggregate(confidences, config)

        # Should be within reasonable range of 100
        assert 80 <= result.contributor_count <= 120

    def test_histogram_suppressed_for_small_groups(self, config: PrivacyConfig) -> None:
        """Test histogram suppressed when contributors below threshold."""
        confidences = [0.5] * 10  # Below HISTOGRAM_SUPPRESSION_THRESHOLD

        result = compute_private_aggregate(confidences, config)

        assert result.confidence_distribution is None
        assert result.histogram_suppressed is True

    def test_histogram_included_for_large_groups(self, config: PrivacyConfig) -> None:
        """Test histogram included when contributors above threshold."""
        confidences = [0.5] * 25  # Above threshold

        result = compute_private_aggregate(confidences, config)

        assert result.confidence_distribution is not None
        assert result.histogram_suppressed is False

    def test_privacy_guarantees_in_result(self, config: PrivacyConfig) -> None:
        """Test privacy guarantees are included in result."""
        confidences = [0.5] * 10

        result = compute_private_aggregate(confidences, config)

        assert result.epsilon_used == config.epsilon
        assert result.delta == config.delta
        assert result.noise_mechanism == config.noise_mechanism.value
        assert result.k_anonymity_satisfied is True

    def test_sensitive_config_higher_k(self) -> None:
        """Test sensitive federation requires higher k."""
        sensitive_config = PrivacyConfig(sensitive_domain=True)

        # 7 contributors - not enough for sensitive (needs 10)
        confidences = [0.5] * 7

        result = compute_private_aggregate(confidences, sensitive_config)

        assert result is None

        # 10 contributors - enough
        confidences = [0.5] * 10
        result = compute_private_aggregate(confidences, sensitive_config)

        assert result is not None

    def test_agreement_score_computed(self, config: PrivacyConfig) -> None:
        """Test agreement score is computed."""
        # High agreement (all same)
        confidences = [0.8] * 20
        result = compute_private_aggregate(confidences, config)

        # Agreement should be high (close to 1)
        # Note: noise can affect this
        assert result.agreement_score is not None

    def test_result_to_dict(self, config: PrivacyConfig) -> None:
        """Test result serialization."""
        confidences = [0.5] * 25

        result = compute_private_aggregate(confidences, config)
        d = result.to_dict()

        assert "collective_confidence" in d
        assert "contributor_count" in d
        assert "privacy_guarantees" in d
        assert d["privacy_guarantees"]["epsilon"] == config.epsilon


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPrivacyIntegration:
    """Integration tests combining multiple privacy features."""

    def test_full_query_flow(self) -> None:
        """Test complete query flow with budget tracking."""
        federation_id = uuid4()
        budget = PrivacyBudget(federation_id=federation_id)
        config = PrivacyConfig()

        topic_hash = compute_topic_hash(["tech", "programming"])

        # Check budget
        can_query, result = budget.check_budget(config.epsilon, config.delta, topic_hash)
        assert can_query is True

        # Simulate aggregation
        confidences = [0.7, 0.8, 0.6, 0.9, 0.75]
        aggregate = compute_private_aggregate(confidences, config)
        assert aggregate is not None

        # Consume budget
        budget.consume(config.epsilon, config.delta, topic_hash)

        # Verify budget consumed
        assert budget.spent_epsilon == config.epsilon
        assert budget.queries_today == 1

    def test_sensitive_domain_elevated_privacy(self) -> None:
        """Test sensitive domains get elevated privacy."""
        domains = ["medical", "diagnosis"]

        assert is_sensitive_domain(domains) is True

        # Create config with sensitive flag
        config = PrivacyConfig(sensitive_domain=True)

        # Effective k should be elevated
        assert config.effective_min_contributors == SENSITIVE_MIN_CONTRIBUTORS

    def test_membership_changes_smoothed(self) -> None:
        """Test membership changes don't immediately affect aggregates."""
        smoother = TemporalSmoother(smoothing_hours=24)

        # Use fixed reference time for deterministic test
        now = datetime.now(UTC)

        # Member just departed
        just_departed = now - timedelta(minutes=5)

        weight = smoother.get_contribution_weight(
            member_id="recent_departure",
            joined_at=now - timedelta(days=30),
            departed_at=just_departed,
            query_time=now,  # Use same reference time
        )

        # Should still have significant weight
        assert weight > 0.9


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_confidences(self) -> None:
        """Test handling of empty confidence list."""
        config = PrivacyConfig()
        result = compute_private_aggregate([], config)

        assert result is None  # Can't satisfy k-anonymity

    def test_single_confidence(self) -> None:
        """Test handling of single confidence."""
        config = PrivacyConfig()
        result = compute_private_aggregate([0.5], config)

        assert result is None

    def test_exact_k_threshold(self) -> None:
        """Test behavior at exact k threshold."""
        config = PrivacyConfig(min_contributors=5)

        # Exactly k contributors
        confidences = [0.5, 0.6, 0.7, 0.8, 0.9]
        result = compute_private_aggregate(confidences, config)

        assert result is not None

    def test_confidence_bounds(self) -> None:
        """Test confidence values at bounds."""
        config = PrivacyConfig()

        # All at 0
        confidences = [0.0] * 10
        result = compute_private_aggregate(confidences, config)
        assert result is not None

        # All at 1
        confidences = [1.0] * 10
        result = compute_private_aggregate(confidences, config)
        assert result is not None

    def test_high_variance_confidences(self) -> None:
        """Test handling of high variance in confidences."""
        np.random.seed(42)
        config = PrivacyConfig()

        # Maximum variance
        confidences = [0.0, 1.0] * 10
        result = compute_private_aggregate(confidences, config)

        assert result is not None
        # Agreement should be low
        # (exact value depends on noise)
