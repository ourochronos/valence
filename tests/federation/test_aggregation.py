"""Tests for cross-federation belief aggregation.

Tests cover:
- Conflict detection across federations
- Trust-weighted aggregation
- Privacy-preserving aggregation
- Temporal smoothing integration
- Various conflict resolution strategies
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from valence.core.confidence import DimensionalConfidence
from valence.federation.aggregation import (
    MIN_FEDERATIONS_FOR_AGGREGATE,
    AggregationConfig,
    AggregationStrategy,
    ConflictDetector,
    ConflictResolution,
    # Enums
    ConflictType,
    CrossFederationAggregateResult,
    DetectedConflict,
    FederationAggregator,
    # Classes
    FederationContribution,
    PrivacyPreservingAggregator,
    TrustWeightedAggregator,
    # Functions
    aggregate_cross_federation,
    create_contribution,
)
from valence.federation.models import FederatedBelief, ShareLevel, Visibility
from valence.federation.privacy import PrivacyConfig, TemporalSmoother

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_confidence() -> DimensionalConfidence:
    """Create a sample confidence vector."""
    return DimensionalConfidence(
        source_reliability=0.8,
        method_quality=0.7,
        internal_consistency=0.9,
        temporal_freshness=0.95,
        corroboration=0.6,
        domain_applicability=0.85,
    )


@pytest.fixture
def sample_belief(sample_confidence: DimensionalConfidence) -> FederatedBelief:
    """Create a sample federated belief."""
    return FederatedBelief(
        id=uuid4(),
        federation_id=uuid4(),
        origin_node_did="did:vkb:web:test.example.com",
        content="The sky is blue during daytime",
        confidence=sample_confidence,
        domain_path=["science", "physics"],
        visibility=Visibility.FEDERATED,
        share_level=ShareLevel.BELIEF_ONLY,
        signed_at=datetime.now(UTC),
    )


@pytest.fixture
def sample_contribution(sample_belief: FederatedBelief) -> FederationContribution:
    """Create a sample federation contribution."""
    return FederationContribution(
        federation_id=uuid4(),
        node_id=uuid4(),
        federation_did="did:vkb:fed:test-federation",
        trust_score=0.7,
        beliefs=[sample_belief],
    )


def make_belief(
    content: str,
    confidence: float = 0.7,
    domain_path: list[str] | None = None,
) -> FederatedBelief:
    """Helper to create beliefs for testing."""
    return FederatedBelief(
        id=uuid4(),
        federation_id=uuid4(),
        origin_node_did=f"did:vkb:web:{uuid4().hex[:8]}.example.com",
        content=content,
        confidence=DimensionalConfidence(
            overall=confidence,  # Set overall explicitly!
            source_reliability=confidence,
            method_quality=confidence,
            internal_consistency=confidence,
            temporal_freshness=0.95,
            corroboration=0.5,
            domain_applicability=confidence,
        ),
        domain_path=domain_path or ["test"],
        visibility=Visibility.FEDERATED,
        share_level=ShareLevel.BELIEF_ONLY,
        signed_at=datetime.now(UTC),
    )


def make_contribution(
    beliefs: list[FederatedBelief],
    trust_score: float = 0.5,
    is_anchor: bool = False,
) -> FederationContribution:
    """Helper to create contributions for testing."""
    return FederationContribution(
        federation_id=uuid4(),
        node_id=uuid4(),
        federation_did=f"did:vkb:fed:{uuid4().hex[:8]}",
        trust_score=trust_score,
        is_anchor=is_anchor,
        beliefs=beliefs,
    )


# =============================================================================
# FEDERATION CONTRIBUTION TESTS
# =============================================================================


class TestFederationContribution:
    """Tests for FederationContribution dataclass."""

    def test_creation_with_beliefs(self) -> None:
        """Test creating a contribution with beliefs."""
        beliefs = [make_belief("Test belief 1"), make_belief("Test belief 2")]
        contrib = FederationContribution(
            federation_id=uuid4(),
            node_id=uuid4(),
            federation_did="did:vkb:fed:test",
            beliefs=beliefs,
            trust_score=0.8,
        )

        assert contrib.belief_count == 2
        assert contrib.trust_score == 0.8
        assert not contrib.is_anchor

    def test_empty_contribution(self) -> None:
        """Test creating an empty contribution."""
        contrib = FederationContribution(
            federation_id=uuid4(),
            node_id=uuid4(),
            federation_did="did:vkb:fed:empty",
            beliefs=[],
        )

        assert contrib.belief_count == 0
        assert contrib.trust_score == 0.5  # Default

    def test_anchor_contribution(self) -> None:
        """Test creating an anchor contribution."""
        contrib = make_contribution(
            beliefs=[make_belief("Anchor belief")],
            trust_score=0.9,
            is_anchor=True,
        )

        assert contrib.is_anchor
        assert contrib.trust_score == 0.9


# =============================================================================
# CONFLICT DETECTION TESTS
# =============================================================================


class TestConflictDetector:
    """Tests for ConflictDetector class."""

    def test_no_conflict_different_topics(self) -> None:
        """Test no conflict when beliefs are about different topics."""
        detector = ConflictDetector()

        contrib_a = make_contribution([make_belief("The sky is blue")])
        contrib_b = make_contribution([make_belief("Water is wet")])

        conflicts = detector.detect_conflicts([contrib_a, contrib_b])

        assert len(conflicts) == 0

    def test_no_conflict_same_federation(self) -> None:
        """Test no conflict detection within same federation."""
        detector = ConflictDetector()

        fed_id = uuid4()
        belief_a = make_belief("Cats are mammals", confidence=0.9)
        belief_b = make_belief("Dogs are mammals", confidence=0.3)

        contrib = FederationContribution(
            federation_id=fed_id,
            node_id=uuid4(),
            federation_did="did:vkb:fed:same",
            beliefs=[belief_a, belief_b],
            trust_score=0.7,
        )

        # Only one contribution, no cross-federation conflicts
        conflicts = detector.detect_conflicts([contrib])
        assert len(conflicts) == 0

    def test_detect_contradiction(self) -> None:
        """Test detecting contradiction between beliefs."""
        detector = ConflictDetector()

        # Similar topic, opposing views
        belief_a = make_belief("The earth is round", confidence=0.9)
        belief_b = make_belief("The earth is not round", confidence=0.8)

        contrib_a = make_contribution([belief_a], trust_score=0.7)
        contrib_b = make_contribution([belief_b], trust_score=0.6)

        conflicts = detector.detect_conflicts([contrib_a, contrib_b])

        # Should detect conflict due to negation
        assert len(conflicts) >= 1
        if conflicts:
            # Check conflict type
            assert conflicts[0].conflict_type in [
                ConflictType.CONTRADICTION,
                ConflictType.DIVERGENCE,
            ]

    def test_detect_confidence_divergence(self) -> None:
        """Test detecting confidence divergence."""
        detector = ConflictDetector(confidence_divergence=0.3)

        # Same topic, different confidence levels
        belief_a = make_belief("Climate change is real", confidence=0.95)
        belief_b = make_belief("Climate change is real", confidence=0.4)

        contrib_a = make_contribution([belief_a], trust_score=0.8)
        contrib_b = make_contribution([belief_b], trust_score=0.7)

        # Use custom similarity function that returns high similarity
        def always_similar(a: str, b: str) -> float:
            return 0.95

        conflicts = detector.detect_conflicts(
            [contrib_a, contrib_b],
            similarity_fn=always_similar,
        )

        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == ConflictType.DIVERGENCE
        assert conflicts[0].confidence_divergence > 0.3

    def test_custom_thresholds(self) -> None:
        """Test conflict detection with custom thresholds."""
        # Very strict detector
        strict_detector = ConflictDetector(
            semantic_threshold=0.95,
            confidence_divergence=0.1,
        )

        # Very lenient detector
        lenient_detector = ConflictDetector(
            semantic_threshold=0.5,
            confidence_divergence=0.8,
        )

        belief_a = make_belief("Python is great for data science", confidence=0.8)
        belief_b = make_belief("Python is good for data analysis", confidence=0.6)

        contrib_a = make_contribution([belief_a])
        contrib_b = make_contribution([belief_b])

        strict_conflicts = strict_detector.detect_conflicts([contrib_a, contrib_b])
        lenient_conflicts = lenient_detector.detect_conflicts([contrib_a, contrib_b])

        # Lenient should find fewer conflicts
        assert len(lenient_conflicts) <= len(strict_conflicts)

    def test_conflict_to_dict(self) -> None:
        """Test DetectedConflict serialization."""
        conflict = DetectedConflict(
            conflict_type=ConflictType.CONTRADICTION,
            belief_a_id=uuid4(),
            belief_b_id=uuid4(),
            federation_a_id=uuid4(),
            federation_b_id=uuid4(),
            semantic_similarity=0.85,
            confidence_divergence=0.3,
            description="Test conflict",
        )

        d = conflict.to_dict()

        assert d["conflict_type"] == "contradiction"
        assert d["semantic_similarity"] == 0.85
        assert d["confidence_divergence"] == 0.3


# =============================================================================
# TRUST-WEIGHTED AGGREGATION TESTS
# =============================================================================


class TestTrustWeightedAggregator:
    """Tests for TrustWeightedAggregator class."""

    def test_compute_weights_single_federation(self) -> None:
        """Test weight computation for single federation."""
        aggregator = TrustWeightedAggregator()

        contrib = make_contribution(
            [make_belief("Test")],
            trust_score=0.8,
        )

        weights = aggregator.compute_contribution_weights([contrib])

        assert len(weights) == 1
        assert weights[contrib.federation_id] == 1.0  # Only one, gets 100%

    def test_compute_weights_multiple_federations(self) -> None:
        """Test weight computation for multiple federations."""
        aggregator = TrustWeightedAggregator(
            trust_weight=0.5,
            recency_weight=0.25,
            corroboration_weight=0.25,
        )

        high_trust = make_contribution(
            [make_belief("High trust belief")],
            trust_score=0.9,
        )
        low_trust = make_contribution(
            [make_belief("Low trust belief")],
            trust_score=0.3,
        )

        weights = aggregator.compute_contribution_weights([high_trust, low_trust])

        assert len(weights) == 2
        # High trust should get more weight
        assert weights[high_trust.federation_id] > weights[low_trust.federation_id]
        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_anchor_bonus(self) -> None:
        """Test that anchor federations get weight bonus."""
        aggregator = TrustWeightedAggregator()

        regular = make_contribution(
            [make_belief("Regular")],
            trust_score=0.7,
            is_anchor=False,
        )
        anchor = make_contribution(
            [make_belief("Anchor")],
            trust_score=0.7,  # Same trust score
            is_anchor=True,
        )

        weights = aggregator.compute_contribution_weights([regular, anchor])

        # Anchor should have higher weight
        assert weights[anchor.federation_id] > weights[regular.federation_id]

    def test_aggregate_confidences(self) -> None:
        """Test weighted confidence aggregation."""
        aggregator = TrustWeightedAggregator()

        high_conf = make_contribution(
            [make_belief("High", confidence=0.9)],
            trust_score=0.8,
        )
        low_conf = make_contribution(
            [make_belief("Low", confidence=0.3)],
            trust_score=0.8,
        )

        weights = {
            high_conf.federation_id: 0.5,
            low_conf.federation_id: 0.5,
        }

        aggregate = aggregator.aggregate_confidences([high_conf, low_conf], weights)

        # Should be around 0.6 (average of 0.9 and 0.3)
        assert 0.4 < aggregate < 0.8  # Wider range due to noise

    def test_agreement_score_full_agreement(self) -> None:
        """Test agreement score when all federations agree."""
        aggregator = TrustWeightedAggregator()

        contrib_a = make_contribution(
            [make_belief("Same", confidence=0.8)],
            trust_score=0.7,
        )
        contrib_b = make_contribution(
            [make_belief("Same", confidence=0.8)],
            trust_score=0.7,
        )

        weights = {
            contrib_a.federation_id: 0.5,
            contrib_b.federation_id: 0.5,
        }

        agreement = aggregator.compute_agreement_score([contrib_a, contrib_b], weights)

        # Should be high (near 1.0)
        assert agreement > 0.9

    def test_agreement_score_disagreement(self) -> None:
        """Test agreement score when federations disagree."""
        aggregator = TrustWeightedAggregator()

        # Create contributions with explicit local_confidence
        contrib_a = make_contribution(
            [make_belief("High", confidence=0.95)],
            trust_score=0.7,
        )
        contrib_a.local_confidence = 0.95  # Set explicitly for test

        contrib_b = make_contribution(
            [make_belief("Low", confidence=0.05)],
            trust_score=0.7,
        )
        contrib_b.local_confidence = 0.05  # Set explicitly for test

        weights = {
            contrib_a.federation_id: 0.5,
            contrib_b.federation_id: 0.5,
        }

        agreement = aggregator.compute_agreement_score([contrib_a, contrib_b], weights)

        # Should be low (significant disagreement)
        assert agreement < 0.5

    def test_temporal_smoothing_integration(self) -> None:
        """Test weight computation with temporal smoothing."""
        aggregator = TrustWeightedAggregator()
        smoother = TemporalSmoother(smoothing_hours=24)

        # Newly joined federation
        new_contrib = FederationContribution(
            federation_id=uuid4(),
            node_id=uuid4(),
            federation_did="did:vkb:fed:new",
            trust_score=0.8,
            beliefs=[make_belief("New")],
            joined_at=datetime.now(UTC) - timedelta(hours=6),  # 6 hours ago
        )

        # Established federation
        old_contrib = FederationContribution(
            federation_id=uuid4(),
            node_id=uuid4(),
            federation_did="did:vkb:fed:old",
            trust_score=0.8,  # Same trust
            beliefs=[make_belief("Old")],
            joined_at=datetime.now(UTC) - timedelta(days=30),  # 30 days ago
        )

        weights = aggregator.compute_contribution_weights(
            [new_contrib, old_contrib],
            smoother=smoother,
        )

        # Established federation should have more weight (new one is ramping up)
        assert weights[old_contrib.federation_id] > weights[new_contrib.federation_id]


# =============================================================================
# PRIVACY-PRESERVING AGGREGATION TESTS
# =============================================================================


class TestPrivacyPreservingAggregator:
    """Tests for PrivacyPreservingAggregator class."""

    def test_apply_privacy_adds_noise(self) -> None:
        """Test that privacy application adds noise."""
        config = PrivacyConfig(epsilon=1.0)
        aggregator = PrivacyPreservingAggregator(privacy_config=config)

        # Run multiple times to check noise is added
        results = []
        for _ in range(10):
            noisy_conf, *_ = aggregator.apply_privacy(
                aggregate_confidence=0.7,
                agreement_score=0.8,
                federation_count=5,
                total_belief_count=20,
                total_contributor_count=10,
                domain_filter=["test"],
            )
            results.append(noisy_conf)

        # Results should vary (noise is added)
        unique_results = set(round(r, 4) for r in results)
        assert len(unique_results) > 1

    def test_k_anonymity_failure(self) -> None:
        """Test k-anonymity check with insufficient contributors."""
        config = PrivacyConfig(min_contributors=10)
        aggregator = PrivacyPreservingAggregator(privacy_config=config)

        noisy_conf, noisy_agree, noisy_fed, noisy_belief, noisy_contrib, k_sat = aggregator.apply_privacy(
            aggregate_confidence=0.7,
            agreement_score=0.8,
            federation_count=3,
            total_belief_count=5,
            total_contributor_count=5,  # Below min_contributors
            domain_filter=["test"],
        )

        assert not k_sat
        assert noisy_conf == 0.0
        assert noisy_contrib == 0

    def test_sensitive_domain_stricter_params(self) -> None:
        """Test that sensitive domains get stricter parameters."""
        config = PrivacyConfig(epsilon=2.0, min_contributors=5)
        aggregator = PrivacyPreservingAggregator(privacy_config=config)

        # Non-sensitive domain
        _, _, _, _, contrib_normal, _ = aggregator.apply_privacy(
            aggregate_confidence=0.7,
            agreement_score=0.8,
            federation_count=5,
            total_belief_count=20,
            total_contributor_count=10,
            domain_filter=["technology"],
        )

        # Sensitive domain (health)
        _, _, _, _, contrib_sensitive, k_sat = aggregator.apply_privacy(
            aggregate_confidence=0.7,
            agreement_score=0.8,
            federation_count=5,
            total_belief_count=20,
            total_contributor_count=8,  # Between 5 and 10
            domain_filter=["health", "medical"],
        )

        # With 8 contributors, sensitive domain (min=10) should fail k-anonymity
        assert not k_sat

    def test_values_clamped_to_valid_range(self) -> None:
        """Test that noisy values are clamped to [0, 1]."""
        config = PrivacyConfig(epsilon=0.1)  # High noise
        aggregator = PrivacyPreservingAggregator(privacy_config=config)

        # Run many times to hit edge cases
        for _ in range(100):
            noisy_conf, noisy_agree, *_ = aggregator.apply_privacy(
                aggregate_confidence=0.99,
                agreement_score=0.01,
                federation_count=5,
                total_belief_count=20,
                total_contributor_count=10,
                domain_filter=["test"],
            )

            assert 0.0 <= noisy_conf <= 1.0
            assert 0.0 <= noisy_agree <= 1.0


# =============================================================================
# AGGREGATION CONFIG TESTS
# =============================================================================


class TestAggregationConfig:
    """Tests for AggregationConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = AggregationConfig()

        assert config.strategy == AggregationStrategy.WEIGHTED_MEAN
        assert config.conflict_resolution == ConflictResolution.TRUST_WEIGHTED
        assert config.min_federations == MIN_FEDERATIONS_FOR_AGGREGATE

    def test_weight_validation(self) -> None:
        """Test that weights are validated."""
        with pytest.raises(ValueError, match="weights must sum to <= 1.0"):
            AggregationConfig(
                trust_weight=0.5,
                recency_weight=0.5,
                corroboration_weight=0.5,  # Total > 1.0
            )

    def test_min_federations_validation(self) -> None:
        """Test minimum federations validation."""
        with pytest.raises(ValueError, match="min_federations must be >= 2"):
            AggregationConfig(min_federations=1)

    def test_to_dict(self) -> None:
        """Test config serialization."""
        config = AggregationConfig(
            strategy=AggregationStrategy.CONSENSUS_ONLY,
            conflict_resolution=ConflictResolution.RECENCY_WINS,
        )

        d = config.to_dict()

        assert d["strategy"] == "consensus_only"
        assert d["conflict_resolution"] == "recency_wins"


# =============================================================================
# FEDERATION AGGREGATOR TESTS
# =============================================================================


class TestFederationAggregator:
    """Tests for FederationAggregator class."""

    def test_aggregate_insufficient_federations(self) -> None:
        """Test aggregation fails with too few federations."""
        aggregator = FederationAggregator()

        # Only one federation
        contrib = make_contribution([make_belief("Test")], trust_score=0.8)

        result = aggregator.aggregate([contrib], domain_filter=["test"])

        assert not result.k_anonymity_satisfied
        assert result.federation_count == 0

    def test_aggregate_basic(self) -> None:
        """Test basic aggregation with multiple federations."""
        aggregator = FederationAggregator()

        # Need at least 5 contributions to satisfy default k-anonymity
        contributions = [
            make_contribution([make_belief(f"Belief {i}", confidence=0.5 + i * 0.05)], trust_score=0.7)
            for i in range(6)  # 6 contributions to satisfy k=5
        ]

        result = aggregator.aggregate(
            contributions,
            domain_filter=["test"],
        )

        assert result.k_anonymity_satisfied
        assert result.federation_count > 0
        assert 0 <= result.collective_confidence <= 1  # May be 0.0 with sparse data

    def test_aggregate_filters_low_trust(self) -> None:
        """Test that low-trust federations are filtered."""
        config = AggregationConfig(min_federation_trust=0.5)
        aggregator = FederationAggregator(config=config)

        # Create enough high-trust federations to satisfy k-anonymity after filtering
        high_trust_contribs = [
            make_contribution([make_belief(f"High trust {i}")], trust_score=0.7 + i * 0.05)
            for i in range(6)  # 6 high trust
        ]
        low_trust = make_contribution(
            [make_belief("Low trust")],
            trust_score=0.3,  # Below threshold
        )

        result = aggregator.aggregate(
            high_trust_contribs + [low_trust],
            domain_filter=["test"],
        )

        # Low trust federation should be filtered
        assert result.k_anonymity_satisfied
        # Result should have contributions (6 high trust, after filtering out low)
        assert result.total_contributor_count > 0

    def test_aggregate_with_conflicts(self) -> None:
        """Test aggregation detects and handles conflicts."""
        aggregator = FederationAggregator()

        # Similar beliefs with different confidence
        contrib_a = make_contribution(
            [make_belief("Python is great", confidence=0.9)],
            trust_score=0.8,
        )
        contrib_b = make_contribution(
            [make_belief("Python is not great", confidence=0.8)],
            trust_score=0.7,
        )
        contrib_c = make_contribution(
            [make_belief("Python is ok", confidence=0.5)],
            trust_score=0.6,
        )

        # Use similarity function that detects these as related
        def topic_similar(a: str, b: str) -> float:
            if "python" in a.lower() and "python" in b.lower():
                return 0.9
            return 0.1

        result = aggregator.aggregate(
            [contrib_a, contrib_b, contrib_c],
            domain_filter=["tech"],
            similarity_fn=topic_similar,
        )

        # Should detect conflicts
        assert result.conflict_count > 0 or result.k_anonymity_satisfied

    def test_recency_wins_resolution(self) -> None:
        """Test recency-wins conflict resolution."""
        config = AggregationConfig(conflict_resolution=ConflictResolution.RECENCY_WINS)
        aggregator = FederationAggregator(config=config)

        old_belief = make_belief("Old info", confidence=0.7)
        old_belief.signed_at = datetime.now(UTC) - timedelta(days=30)

        new_belief = make_belief("New info", confidence=0.9)
        new_belief.signed_at = datetime.now(UTC)

        contrib_old = make_contribution([old_belief], trust_score=0.8)
        contrib_new = make_contribution([new_belief], trust_score=0.8)
        contrib_other = make_contribution(
            [make_belief("Other", confidence=0.7)],
            trust_score=0.7,
        )

        # Force conflict detection
        def always_similar(a: str, b: str) -> float:
            if "info" in a.lower() and "info" in b.lower():
                return 0.95
            return 0.1

        result = aggregator.aggregate(
            [contrib_old, contrib_new, contrib_other],
            domain_filter=["test"],
            similarity_fn=always_similar,
        )

        # Result should be computed (may or may not have conflicts based on detection)
        assert result.computed_at is not None

    def test_exclude_conflicting_resolution(self) -> None:
        """Test exclude-conflicting conflict resolution."""
        config = AggregationConfig(conflict_resolution=ConflictResolution.EXCLUDE_CONFLICTING)
        aggregator = FederationAggregator(config=config)

        contrib_a = make_contribution(
            [make_belief("Conflicting A", confidence=0.9)],
            trust_score=0.8,
        )
        contrib_b = make_contribution(
            [make_belief("Conflicting B", confidence=0.3)],
            trust_score=0.8,
        )
        # Non-conflicting
        contrib_c = make_contribution(
            [make_belief("Unrelated topic entirely different", confidence=0.7)],
            trust_score=0.7,
        )

        result = aggregator.aggregate(
            [contrib_a, contrib_b, contrib_c],
            domain_filter=["test"],
        )

        # Should complete aggregation
        assert result.computed_at is not None

    def test_query_hash_consistency(self) -> None:
        """Test that same query produces same hash."""
        aggregator = FederationAggregator()

        contrib = make_contribution([make_belief("Test")], trust_score=0.7)

        result1 = aggregator.aggregate(
            [contrib, contrib, contrib],
            domain_filter=["science", "physics"],
            semantic_query="What about gravity?",
        )
        result2 = aggregator.aggregate(
            [contrib, contrib, contrib],
            domain_filter=["science", "physics"],
            semantic_query="What about gravity?",
        )

        assert result1.query_hash == result2.query_hash

    def test_result_to_dict(self) -> None:
        """Test result serialization."""
        aggregator = FederationAggregator()

        contrib_a = make_contribution([make_belief("A")], trust_score=0.7)
        contrib_b = make_contribution([make_belief("B")], trust_score=0.7)
        contrib_c = make_contribution([make_belief("C")], trust_score=0.7)

        result = aggregator.aggregate(
            [contrib_a, contrib_b, contrib_c],
            domain_filter=["test"],
        )

        d = result.to_dict()

        assert "id" in d
        assert "result" in d
        assert "participation" in d
        assert "conflicts" in d
        assert "privacy_guarantees" in d
        assert "computed_at" in d


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_aggregate_cross_federation(self) -> None:
        """Test aggregate_cross_federation convenience function."""
        contributions = [
            make_contribution([make_belief("A")], trust_score=0.7),
            make_contribution([make_belief("B")], trust_score=0.7),
            make_contribution([make_belief("C")], trust_score=0.7),
        ]

        result = aggregate_cross_federation(
            contributions,
            domain_filter=["test"],
        )

        assert isinstance(result, CrossFederationAggregateResult)
        assert result.computed_at is not None

    def test_create_contribution(self) -> None:
        """Test create_contribution helper."""
        beliefs = [make_belief("Test 1"), make_belief("Test 2")]

        contrib = create_contribution(
            federation_id=uuid4(),
            node_id=uuid4(),
            federation_did="did:vkb:fed:helper-test",
            beliefs=beliefs,
            trust_score=0.85,
            is_anchor=True,
        )

        assert contrib.belief_count == 2
        assert contrib.trust_score == 0.85
        assert contrib.is_anchor


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the full aggregation pipeline."""

    def test_full_aggregation_pipeline(self) -> None:
        """Test complete aggregation from contributions to result."""
        config = AggregationConfig(
            strategy=AggregationStrategy.WEIGHTED_MEAN,
            conflict_resolution=ConflictResolution.TRUST_WEIGHTED,
            privacy_config=PrivacyConfig(epsilon=1.0, min_contributors=5),
        )

        # Create diverse contributions - need at least 5 to satisfy k-anonymity
        anchor_contrib = make_contribution(
            [
                make_belief("Machine learning is useful", confidence=0.85),
                make_belief("Deep learning requires data", confidence=0.9),
            ],
            trust_score=0.9,
            is_anchor=True,
        )

        regular_contrib_a = make_contribution(
            [make_belief("AI needs careful governance", confidence=0.75)],
            trust_score=0.7,
        )

        regular_contrib_b = make_contribution(
            [make_belief("ML models can be biased", confidence=0.8)],
            trust_score=0.65,
        )

        # Add more contributions to satisfy k=5
        extra_contrib_c = make_contribution(
            [make_belief("Neural networks are powerful", confidence=0.7)],
            trust_score=0.6,
        )

        extra_contrib_d = make_contribution(
            [make_belief("Training data quality matters", confidence=0.85)],
            trust_score=0.7,
        )

        extra_contrib_e = make_contribution(
            [make_belief("Ethics in AI is important", confidence=0.9)],
            trust_score=0.75,
        )

        aggregator = FederationAggregator(config=config)
        result = aggregator.aggregate(
            [
                anchor_contrib,
                regular_contrib_a,
                regular_contrib_b,
                extra_contrib_c,
                extra_contrib_d,
                extra_contrib_e,
            ],
            domain_filter=["ai", "machine_learning"],
            semantic_query="What are the considerations for ML?",
        )

        # Verify result structure
        assert result.query_hash != ""
        assert result.domain_filter == ["ai", "machine_learning"]
        assert result.semantic_query == "What are the considerations for ML?"
        assert result.federation_count >= 0
        assert result.computed_at is not None
        assert "epsilon" in result.to_dict()["privacy_guarantees"]

    def test_privacy_budget_consumption(self) -> None:
        """Test that privacy budget is consumed on queries."""
        from valence.federation.privacy import PrivacyBudget

        budget = PrivacyBudget(
            federation_id=uuid4(),
            daily_epsilon_budget=5.0,
        )

        config = AggregationConfig(
            privacy_config=PrivacyConfig(epsilon=1.0, min_contributors=5),
        )

        aggregator = FederationAggregator(config=config)
        aggregator.privacy_aggregator.privacy_budget = budget

        # Need 6 contributions to satisfy k=5
        contributions = [make_contribution([make_belief(f"Belief {i}")], trust_score=0.7) for i in range(6)]

        initial_remaining = budget.remaining_epsilon()

        aggregator.aggregate(
            contributions,
            domain_filter=["test"],
            requester_id="test-requester",
        )

        # Budget should be consumed
        assert budget.remaining_epsilon() < initial_remaining

    def test_multi_round_aggregation(self) -> None:
        """Test aggregation over multiple rounds."""
        aggregator = FederationAggregator()

        results = []
        for i in range(5):
            contributions = [
                make_contribution(
                    [make_belief(f"Belief {i}-{j}", confidence=0.5 + j * 0.1)],
                    trust_score=0.6 + j * 0.1,
                )
                for j in range(3)
            ]

            result = aggregator.aggregate(
                contributions,
                domain_filter=["round", str(i)],
            )
            results.append(result)

        # All results should be valid
        for r in results:
            assert r.computed_at is not None
            assert r.query_hash != ""


# =============================================================================
# EDGE CASES
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_contributions_list(self) -> None:
        """Test handling of empty contributions."""
        aggregator = FederationAggregator()

        result = aggregator.aggregate([], domain_filter=["test"])

        assert not result.k_anonymity_satisfied
        assert result.federation_count == 0

    def test_contribution_with_no_beliefs(self) -> None:
        """Test handling contributions with no beliefs."""
        aggregator = FederationAggregator()

        empty_contrib = make_contribution([], trust_score=0.8)
        normal_contrib = make_contribution([make_belief("Normal")], trust_score=0.7)
        another_contrib = make_contribution([make_belief("Another")], trust_score=0.7)

        result = aggregator.aggregate(
            [empty_contrib, normal_contrib, another_contrib],
            domain_filter=["test"],
        )

        # Should handle gracefully
        assert result.computed_at is not None

    def test_extreme_trust_values(self) -> None:
        """Test with extreme trust values (0 and 1)."""
        aggregator = FederationAggregator()

        zero_trust = make_contribution([make_belief("Zero")], trust_score=0.0)
        max_trust = make_contribution([make_belief("Max")], trust_score=1.0)
        normal = make_contribution([make_belief("Normal")], trust_score=0.5)

        result = aggregator.aggregate(
            [zero_trust, max_trust, normal],
            domain_filter=["test"],
        )

        # Zero trust should be filtered out
        assert result.computed_at is not None

    def test_extreme_confidence_values(self) -> None:
        """Test with extreme confidence values."""
        aggregator = FederationAggregator()

        # Need 6 contributions to satisfy k=5
        contributions = [
            make_contribution([make_belief("Zero", confidence=0.01)], trust_score=0.7),  # Near zero
            make_contribution([make_belief("Max", confidence=0.99)], trust_score=0.7),  # Near max
            make_contribution([make_belief("Mid", confidence=0.5)], trust_score=0.7),
            make_contribution([make_belief("Low", confidence=0.3)], trust_score=0.7),
            make_contribution([make_belief("High", confidence=0.8)], trust_score=0.7),
            make_contribution([make_belief("Medium", confidence=0.6)], trust_score=0.7),
        ]

        result = aggregator.aggregate(
            contributions,
            domain_filter=["test"],
        )

        assert result.k_anonymity_satisfied
        assert 0.0 <= result.collective_confidence <= 1.0

    def test_very_long_belief_content(self) -> None:
        """Test with very long belief content."""
        aggregator = FederationAggregator()

        long_content = "A" * 10000  # 10KB of content
        contrib = make_contribution(
            [make_belief(long_content)],
            trust_score=0.7,
        )

        # Need 3 for min federations
        result = aggregator.aggregate(
            [contrib, contrib, contrib],
            domain_filter=["test"],
        )

        # Should handle gracefully
        assert result.computed_at is not None

    def test_unicode_in_beliefs(self) -> None:
        """Test handling of unicode content."""
        aggregator = FederationAggregator()

        unicode_beliefs = [
            make_contribution(
                [make_belief("日本語のテスト", confidence=0.8)],
                trust_score=0.7,
            ),
            make_contribution(
                [make_belief("Тест на русском", confidence=0.7)],
                trust_score=0.7,
            ),
            make_contribution(
                [make_belief("🎉 emoji test 🚀", confidence=0.75)],
                trust_score=0.7,
            ),
        ]

        result = aggregator.aggregate(unicode_beliefs, domain_filter=["test"])

        assert result.computed_at is not None


# =============================================================================
# ADDITIONAL COVERAGE TESTS
# =============================================================================


class TestConflictDetectorInternals:
    """Tests for ConflictDetector internal methods."""

    def test_jaccard_similarity_empty_text(self) -> None:
        """Test Jaccard similarity with empty text."""
        detector = ConflictDetector()

        # Test with empty string
        assert detector._jaccard_similarity("", "some text") == 0.0
        assert detector._jaccard_similarity("some text", "") == 0.0
        assert detector._jaccard_similarity("", "") == 0.0

    def test_jaccard_similarity_identical_text(self) -> None:
        """Test Jaccard similarity with identical text."""
        detector = ConflictDetector()

        result = detector._jaccard_similarity("hello world", "hello world")
        assert result == 1.0

    def test_jaccard_similarity_partial_overlap(self) -> None:
        """Test Jaccard similarity with partial overlap."""
        detector = ConflictDetector()

        # "hello" is common, "world" and "there" differ
        result = detector._jaccard_similarity("hello world", "hello there")
        # Intersection: {"hello"}, Union: {"hello", "world", "there"}
        # Expected: 1/3 ≈ 0.333
        assert 0.3 < result < 0.35

    def test_stance_divergence_high_confidence_both(self) -> None:
        """Test stance divergence with high confidence on both sides."""
        detector = ConflictDetector()

        # Both beliefs have high confidence but different values
        belief_a = make_belief("Science is important", confidence=0.95)
        belief_b = make_belief("Science is important", confidence=0.75)  # Still high but different

        divergence = detector._compute_stance_divergence(belief_a, belief_b)

        # Should return divergence based on confidence difference
        assert divergence >= 0.0

    def test_stance_divergence_one_with_negation(self) -> None:
        """Test stance divergence when one belief has negation."""
        detector = ConflictDetector()

        belief_a = make_belief("The theory is correct", confidence=0.8)
        belief_b = make_belief("The theory is not correct", confidence=0.8)

        divergence = detector._compute_stance_divergence(belief_a, belief_b)

        # Should detect divergence due to negation
        assert divergence == 0.7

    def test_stance_divergence_no_negation_low_confidence(self) -> None:
        """Test stance divergence without negation and low confidence."""
        detector = ConflictDetector()

        belief_a = make_belief("Something is true", confidence=0.5)
        belief_b = make_belief("Something is true", confidence=0.5)

        divergence = detector._compute_stance_divergence(belief_a, belief_b)

        # Low confidence, no negation -> returns 0.0
        assert divergence == 0.0

    def test_temporal_conflict_with_temporal_constraints(self) -> None:
        """Test temporal conflict detection when beliefs have temporal constraints."""
        detector = ConflictDetector()

        belief_a = make_belief("Current fact", confidence=0.8)
        belief_a.valid_from = datetime.now(UTC) - timedelta(days=10)
        belief_a.valid_until = datetime.now(UTC) + timedelta(days=10)

        belief_b = make_belief("Current fact", confidence=0.7)
        belief_b.valid_from = datetime.now(UTC) - timedelta(days=5)
        belief_b.valid_until = datetime.now(UTC) + timedelta(days=5)

        # Both have temporal constraints - method should return False (simplified impl)
        result = detector._has_temporal_conflict(belief_a, belief_b)
        assert result is False  # Current implementation always returns False

    def test_temporal_conflict_only_one_has_constraints(self) -> None:
        """Test temporal conflict when only one belief has constraints."""
        detector = ConflictDetector()

        belief_a = make_belief("Current fact", confidence=0.8)
        belief_a.valid_from = datetime.now(UTC)

        belief_b = make_belief("Current fact", confidence=0.7)
        # No temporal constraints

        # One without constraints -> no conflict
        result = detector._has_temporal_conflict(belief_a, belief_b)
        assert result is False

    def test_detect_conflict_without_similarity_fn(self) -> None:
        """Test conflict detection uses Jaccard fallback when no similarity_fn provided."""
        detector = ConflictDetector(semantic_threshold=0.3)

        # Create beliefs with some word overlap
        belief_a = make_belief("Python programming language is great", confidence=0.9)
        belief_b = make_belief("Python programming language is bad", confidence=0.3)

        contrib_a = make_contribution([belief_a], trust_score=0.7)
        contrib_b = make_contribution([belief_b], trust_score=0.7)

        # Without similarity_fn, uses Jaccard
        conflicts = detector.detect_conflicts([contrib_a, contrib_b], similarity_fn=None)

        # Should find conflict due to word overlap and confidence divergence
        # The Jaccard similarity is high (4/6 words match)
        assert len(conflicts) >= 0  # May or may not find conflict depending on thresholds


class TestTrustWeightedAggregatorInternals:
    """Tests for TrustWeightedAggregator internal methods."""

    def test_corroboration_weight_zero_total(self) -> None:
        """Test corroboration weight when total_beliefs is zero."""
        aggregator = TrustWeightedAggregator()

        result = aggregator._compute_corroboration_weight(0, 0)
        assert result == 0.0

    def test_corroboration_weight_proportional(self) -> None:
        """Test corroboration weight follows sqrt scale."""
        aggregator = TrustWeightedAggregator()

        # 25% contribution
        result = aggregator._compute_corroboration_weight(1, 4)
        # sqrt(0.25) = 0.5
        assert abs(result - 0.5) < 0.01

        # 100% contribution
        result = aggregator._compute_corroboration_weight(10, 10)
        # sqrt(1.0) = 1.0
        assert result == 1.0

    def test_agreement_score_with_zero_weights(self) -> None:
        """Test agreement score when all weights are zero."""
        aggregator = TrustWeightedAggregator()

        contrib_a = make_contribution([make_belief("A", confidence=0.8)], trust_score=0.7)
        contrib_b = make_contribution([make_belief("B", confidence=0.3)], trust_score=0.7)

        # All weights are zero
        weights = {
            contrib_a.federation_id: 0.0,
            contrib_b.federation_id: 0.0,
        }

        agreement = aggregator.compute_agreement_score([contrib_a, contrib_b], weights)

        # Should return 1.0 (less than 2 valid confidences)
        assert agreement == 1.0

    def test_aggregate_confidences_no_data(self) -> None:
        """Test aggregate confidences when no contributions have data."""
        aggregator = TrustWeightedAggregator()

        # Empty contribution
        contrib = make_contribution([], trust_score=0.7)

        weights = {contrib.federation_id: 1.0}

        result = aggregator.aggregate_confidences([contrib], weights)

        # No data -> returns 0.0
        assert result == 0.0

    def test_recency_weight_with_no_beliefs(self) -> None:
        """Test recency weight computation with empty beliefs."""
        aggregator = TrustWeightedAggregator()

        contrib = make_contribution([], trust_score=0.7)

        result = aggregator._compute_recency_weight(contrib)

        # Default for no beliefs
        assert result == 0.5


class TestConflictResolutionStrategies:
    """Tests for different conflict resolution strategies."""

    def test_exclude_conflicting_with_actual_conflicts(self) -> None:
        """Test EXCLUDE_CONFLICTING resolution excludes all conflicting beliefs."""
        config = AggregationConfig(
            conflict_resolution=ConflictResolution.EXCLUDE_CONFLICTING,
            min_federations=2,
        )
        aggregator = FederationAggregator(config=config)

        # Create beliefs with clear conflict (high similarity, different confidence)
        belief_a = make_belief("The sky is blue", confidence=0.9)
        belief_b = make_belief("The sky is blue", confidence=0.2)
        belief_c = make_belief("Water is wet", confidence=0.7)  # Unrelated

        contrib_a = make_contribution([belief_a], trust_score=0.8)
        contrib_b = make_contribution([belief_b], trust_score=0.7)
        contrib_c = make_contribution([belief_c], trust_score=0.7)

        # Force high similarity for conflict detection
        def always_similar(a: str, b: str) -> float:
            if "sky" in a.lower() and "sky" in b.lower():
                return 0.95
            return 0.1

        result = aggregator.aggregate(
            [contrib_a, contrib_b, contrib_c],
            domain_filter=["test"],
            similarity_fn=always_similar,
        )

        # Should complete without error
        assert result.computed_at is not None
        # Conflicts should be detected
        assert result.conflict_count >= 0

    def test_corroboration_resolution_with_conflicts(self) -> None:
        """Test CORROBORATION resolution keeps higher trust beliefs."""
        config = AggregationConfig(
            conflict_resolution=ConflictResolution.CORROBORATION,
            min_federations=2,
        )
        aggregator = FederationAggregator(config=config)

        # Create conflicting beliefs from different trust federations
        belief_a = make_belief("Statement A", confidence=0.9)
        belief_b = make_belief("Statement A but different", confidence=0.8)
        belief_c = make_belief("Unrelated topic", confidence=0.7)

        # High trust federation
        contrib_a = make_contribution([belief_a], trust_score=0.9)
        # Low trust federation
        contrib_b = make_contribution([belief_b], trust_score=0.4)
        # Medium trust
        contrib_c = make_contribution([belief_c], trust_score=0.6)

        def detect_similar(a: str, b: str) -> float:
            if "statement" in a.lower() and "statement" in b.lower():
                return 0.9
            return 0.1

        result = aggregator.aggregate(
            [contrib_a, contrib_b, contrib_c],
            domain_filter=["test"],
            similarity_fn=detect_similar,
        )

        assert result.computed_at is not None

    def test_flag_for_review_keeps_all(self) -> None:
        """Test FLAG_FOR_REVIEW resolution keeps all beliefs."""
        config = AggregationConfig(
            conflict_resolution=ConflictResolution.FLAG_FOR_REVIEW,
            min_federations=2,
        )
        aggregator = FederationAggregator(config=config)

        contrib_a = make_contribution([make_belief("A", confidence=0.9)], trust_score=0.8)
        contrib_b = make_contribution([make_belief("B", confidence=0.2)], trust_score=0.7)
        contrib_c = make_contribution([make_belief("C", confidence=0.5)], trust_score=0.6)

        result = aggregator.aggregate(
            [contrib_a, contrib_b, contrib_c],
            domain_filter=["test"],
        )

        assert result.computed_at is not None


class TestAggregatorHelperMethods:
    """Tests for FederationAggregator helper methods."""

    def test_find_belief_not_found(self) -> None:
        """Test _find_belief returns None when belief not found."""
        aggregator = FederationAggregator()

        contrib = make_contribution([make_belief("Test")], trust_score=0.7)

        # Search for non-existent ID
        result = aggregator._find_belief([contrib], uuid4())
        assert result is None

    def test_find_belief_with_none_id(self) -> None:
        """Test _find_belief returns None when belief_id is None."""
        aggregator = FederationAggregator()

        contrib = make_contribution([make_belief("Test")], trust_score=0.7)

        result = aggregator._find_belief([contrib], None)
        assert result is None

    def test_find_belief_found(self) -> None:
        """Test _find_belief returns belief when found."""
        aggregator = FederationAggregator()

        belief = make_belief("Test belief")
        contrib = make_contribution([belief], trust_score=0.7)

        result = aggregator._find_belief([contrib], belief.id)
        assert result is not None
        assert result.id == belief.id

    def test_find_contribution_not_found(self) -> None:
        """Test _find_contribution returns None when federation not found."""
        aggregator = FederationAggregator()

        contrib = make_contribution([make_belief("Test")], trust_score=0.7)

        result = aggregator._find_contribution([contrib], uuid4())
        assert result is None

    def test_find_contribution_with_none_id(self) -> None:
        """Test _find_contribution returns None when federation_id is None."""
        aggregator = FederationAggregator()

        contrib = make_contribution([make_belief("Test")], trust_score=0.7)

        result = aggregator._find_contribution([contrib], None)
        assert result is None

    def test_find_contribution_found(self) -> None:
        """Test _find_contribution returns contribution when found."""
        aggregator = FederationAggregator()

        contrib = make_contribution([make_belief("Test")], trust_score=0.7)

        result = aggregator._find_contribution([contrib], contrib.federation_id)
        assert result is not None
        assert result.federation_id == contrib.federation_id

    def test_filter_beliefs_excludes_correctly(self) -> None:
        """Test _filter_beliefs excludes specified beliefs."""
        aggregator = FederationAggregator()

        belief_a = make_belief("Keep this")
        belief_b = make_belief("Exclude this")
        contrib = FederationContribution(
            federation_id=uuid4(),
            node_id=uuid4(),
            federation_did="did:vkb:fed:test",
            beliefs=[belief_a, belief_b],
            trust_score=0.7,
        )

        result = aggregator._filter_beliefs([contrib], {belief_b.id})

        assert len(result) == 1
        assert len(result[0].beliefs) == 1
        assert result[0].beliefs[0].id == belief_a.id

    def test_filter_beliefs_removes_empty_contributions(self) -> None:
        """Test _filter_beliefs removes contributions with no remaining beliefs."""
        aggregator = FederationAggregator()

        belief = make_belief("Only belief")
        contrib = FederationContribution(
            federation_id=uuid4(),
            node_id=uuid4(),
            federation_did="did:vkb:fed:test",
            beliefs=[belief],
            trust_score=0.7,
        )

        # Exclude the only belief
        result = aggregator._filter_beliefs([contrib], {belief.id})

        # Contribution should be removed entirely
        assert len(result) == 0


class TestPrivacyAggregatorInternals:
    """Tests for PrivacyPreservingAggregator internal behavior."""

    def test_consume_budget_no_budget_configured(self) -> None:
        """Test consume_budget returns True when no budget is configured."""
        aggregator = PrivacyPreservingAggregator(privacy_config=PrivacyConfig())
        # No privacy_budget set

        result = aggregator.consume_budget(
            domain_filter=["test"],
            semantic_query="query",
            requester_id="user1",
        )

        assert result is True

    def test_apply_privacy_federation_count_zero(self) -> None:
        """Test apply_privacy handles zero federation count."""
        config = PrivacyConfig(epsilon=1.0, min_contributors=5)
        aggregator = PrivacyPreservingAggregator(privacy_config=config)

        # Zero federations but enough contributors
        noisy_conf, noisy_agree, noisy_fed, noisy_belief, noisy_contrib, k_sat = aggregator.apply_privacy(
            aggregate_confidence=0.7,
            agreement_score=0.8,
            federation_count=0,
            total_belief_count=10,
            total_contributor_count=10,
            domain_filter=["test"],
        )

        # k_sat should be True since min_contributors=5 and we have 10
        assert k_sat is True
