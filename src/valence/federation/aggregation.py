"""Cross-federation belief aggregation.

Implements federated aggregation of beliefs across multiple federation nodes
with conflict detection, trust-weighted aggregation, and privacy preservation.

Reference: spec/components/federation-layer/SPEC.md
Related: privacy.py for differential privacy primitives

Issue #15: Federation Layer with Aggregation
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable
from uuid import UUID, uuid4

from ..core.confidence import DimensionalConfidence
from .models import (
    FederationNode,
    FederatedBelief,
    NodeTrust,
    AggregatedBelief,
    AggregationSource,
    AggregationQuery,
    AggregationResult,
    LocalSummary,
    PrivacyParameters,
)
from .privacy import (
    PrivacyConfig,
    PrivacyBudget,
    TemporalSmoother,
    compute_private_aggregate,
    compute_topic_hash,
    is_sensitive_domain,
    add_noise,
    PrivateAggregateResult,
)
from .trust import get_effective_trust


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Conflict detection thresholds
CONFLICT_SEMANTIC_THRESHOLD = 0.75  # Embedding similarity for same topic
CONFLICT_CONFIDENCE_DIVERGENCE = 0.4  # Confidence diff to flag as conflict
CONFLICT_STANCE_THRESHOLD = 0.6  # Stance divergence threshold

# Aggregation weights
DEFAULT_TRUST_WEIGHT = 0.5  # How much trust affects aggregation
DEFAULT_RECENCY_WEIGHT = 0.2  # How much recency affects aggregation
DEFAULT_CORROBORATION_WEIGHT = 0.3  # How much corroboration affects

# Trust thresholds
MIN_FEDERATION_TRUST = 0.1  # Minimum trust to include federation
ANCHOR_TRUST_BOOST = 0.2  # Bonus for anchor federations

# Privacy defaults
MIN_FEDERATIONS_FOR_AGGREGATE = 2  # Need beliefs from at least 2 federations
DEFAULT_PRIVACY_EPSILON = 1.0


# =============================================================================
# ENUMS
# =============================================================================


class ConflictType(str, Enum):
    """Types of belief conflicts across federations."""
    
    CONTRADICTION = "contradiction"  # Beliefs directly contradict
    DIVERGENCE = "divergence"  # Significant confidence gap
    TEMPORAL = "temporal"  # Same topic, different temporal validity
    SCOPE = "scope"  # Different domain applicability
    NONE = "none"  # No conflict detected


class ConflictResolution(str, Enum):
    """How to resolve detected conflicts."""
    
    TRUST_WEIGHTED = "trust_weighted"  # Weight by federation trust
    RECENCY_WINS = "recency_wins"  # Most recent belief wins
    CORROBORATION = "corroboration"  # Most corroborated wins
    FLAG_FOR_REVIEW = "flag_for_review"  # Don't resolve, flag for human review
    EXCLUDE_CONFLICTING = "exclude_conflicting"  # Exclude conflicting beliefs


class AggregationStrategy(str, Enum):
    """Strategy for aggregating beliefs."""
    
    WEIGHTED_MEAN = "weighted_mean"  # Trust-weighted mean confidence
    CONFIDENCE_FUSION = "confidence_fusion"  # Bayesian confidence fusion
    CONSENSUS_ONLY = "consensus_only"  # Only include high-agreement beliefs
    MAJORITY_STANCE = "majority_stance"  # Take majority stance


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class FederationContribution:
    """A federation's contribution to an aggregation query."""
    
    federation_id: UUID
    node_id: UUID
    federation_did: str
    
    # Trust metrics
    trust_score: float = 0.5
    is_anchor: bool = False
    
    # Beliefs contributed
    beliefs: list[FederatedBelief] = field(default_factory=list)
    belief_count: int = 0
    
    # Local aggregate (before cross-federation aggregation)
    local_confidence: float | None = None
    local_agreement: float | None = None
    
    # Membership info for temporal smoothing
    joined_at: datetime | None = None
    departed_at: datetime | None = None
    
    # Privacy
    contribution_weight: float = 1.0
    
    def __post_init__(self) -> None:
        self.belief_count = len(self.beliefs)


@dataclass
class DetectedConflict:
    """A detected conflict between federation beliefs."""
    
    id: UUID = field(default_factory=uuid4)
    conflict_type: ConflictType = ConflictType.NONE
    
    # Conflicting beliefs
    belief_a_id: UUID | None = None
    belief_b_id: UUID | None = None
    
    # Source federations
    federation_a_id: UUID | None = None
    federation_b_id: UUID | None = None
    
    # Conflict metrics
    semantic_similarity: float = 0.0  # How similar the topics are
    confidence_divergence: float = 0.0  # How much confidences differ
    stance_divergence: float = 0.0  # How much stances differ
    
    # Details
    description: str = ""
    belief_a_content: str = ""
    belief_b_content: str = ""
    
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "conflict_type": self.conflict_type.value,
            "belief_a_id": str(self.belief_a_id) if self.belief_a_id else None,
            "belief_b_id": str(self.belief_b_id) if self.belief_b_id else None,
            "federation_a_id": str(self.federation_a_id) if self.federation_a_id else None,
            "federation_b_id": str(self.federation_b_id) if self.federation_b_id else None,
            "semantic_similarity": self.semantic_similarity,
            "confidence_divergence": self.confidence_divergence,
            "stance_divergence": self.stance_divergence,
            "description": self.description,
            "detected_at": self.detected_at.isoformat(),
        }


@dataclass
class AggregationConfig:
    """Configuration for cross-federation aggregation."""
    
    # Strategy
    strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_MEAN
    conflict_resolution: ConflictResolution = ConflictResolution.TRUST_WEIGHTED
    
    # Weights
    trust_weight: float = DEFAULT_TRUST_WEIGHT
    recency_weight: float = DEFAULT_RECENCY_WEIGHT
    corroboration_weight: float = DEFAULT_CORROBORATION_WEIGHT
    
    # Thresholds
    min_federation_trust: float = MIN_FEDERATION_TRUST
    min_federations: int = MIN_FEDERATIONS_FOR_AGGREGATE
    min_total_beliefs: int = 3  # Minimum total beliefs across federations
    
    # Conflict detection
    conflict_semantic_threshold: float = CONFLICT_SEMANTIC_THRESHOLD
    conflict_confidence_divergence: float = CONFLICT_CONFIDENCE_DIVERGENCE
    
    # Privacy
    privacy_config: PrivacyConfig = field(default_factory=PrivacyConfig)
    
    # Temporal smoothing
    apply_temporal_smoothing: bool = True
    smoothing_hours: int = 24
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.trust_weight + self.recency_weight + self.corroboration_weight > 1.0:
            raise ValueError("Aggregation weights must sum to <= 1.0")
        if self.min_federations < 2:
            raise ValueError("min_federations must be >= 2")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "strategy": self.strategy.value,
            "conflict_resolution": self.conflict_resolution.value,
            "trust_weight": self.trust_weight,
            "recency_weight": self.recency_weight,
            "corroboration_weight": self.corroboration_weight,
            "min_federation_trust": self.min_federation_trust,
            "min_federations": self.min_federations,
            "min_total_beliefs": self.min_total_beliefs,
            "privacy_epsilon": self.privacy_config.epsilon,
        }


@dataclass
class CrossFederationAggregateResult:
    """Result of cross-federation belief aggregation."""
    
    id: UUID = field(default_factory=uuid4)
    query_hash: str = ""
    
    # Query context
    domain_filter: list[str] = field(default_factory=list)
    semantic_query: str | None = None
    
    # Aggregated results
    collective_confidence: float = 0.0
    agreement_score: float | None = None
    stance_summary: str | None = None
    key_factors: list[str] = field(default_factory=list)
    
    # Federation participation
    federation_count: int = 0
    total_belief_count: int = 0
    total_contributor_count: int = 0
    
    # Conflicts
    conflicts_detected: list[DetectedConflict] = field(default_factory=list)
    conflict_count: int = 0
    conflicts_resolved: int = 0
    
    # Privacy guarantees
    privacy_epsilon: float = 0.0
    privacy_delta: float = 0.0
    k_anonymity_satisfied: bool = True
    
    # Metadata
    computed_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    config_used: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "query_hash": self.query_hash,
            "domain_filter": self.domain_filter,
            "semantic_query": self.semantic_query,
            "result": {
                "collective_confidence": self.collective_confidence,
                "agreement_score": self.agreement_score,
                "stance_summary": self.stance_summary,
                "key_factors": self.key_factors,
            },
            "participation": {
                "federation_count": self.federation_count,
                "total_belief_count": self.total_belief_count,
                "total_contributor_count": self.total_contributor_count,
            },
            "conflicts": {
                "detected": self.conflict_count,
                "resolved": self.conflicts_resolved,
                "details": [c.to_dict() for c in self.conflicts_detected],
            },
            "privacy_guarantees": {
                "epsilon": self.privacy_epsilon,
                "delta": self.privacy_delta,
                "k_anonymity_satisfied": self.k_anonymity_satisfied,
            },
            "computed_at": self.computed_at.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "config_used": self.config_used,
        }


# =============================================================================
# CONFLICT DETECTION
# =============================================================================


class ConflictDetector:
    """Detects conflicts between beliefs from different federations."""
    
    def __init__(
        self,
        semantic_threshold: float = CONFLICT_SEMANTIC_THRESHOLD,
        confidence_divergence: float = CONFLICT_CONFIDENCE_DIVERGENCE,
        stance_threshold: float = CONFLICT_STANCE_THRESHOLD,
    ):
        self.semantic_threshold = semantic_threshold
        self.confidence_divergence = confidence_divergence
        self.stance_threshold = stance_threshold
    
    def detect_conflicts(
        self,
        contributions: list[FederationContribution],
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> list[DetectedConflict]:
        """Detect conflicts across federation contributions.
        
        Args:
            contributions: List of federation contributions
            similarity_fn: Optional function to compute semantic similarity
            
        Returns:
            List of detected conflicts
        """
        conflicts: list[DetectedConflict] = []
        
        # Collect all beliefs with their federation context
        all_beliefs: list[tuple[FederatedBelief, FederationContribution]] = []
        for contrib in contributions:
            for belief in contrib.beliefs:
                all_beliefs.append((belief, contrib))
        
        # Compare pairs of beliefs from different federations
        for i, (belief_a, contrib_a) in enumerate(all_beliefs):
            for belief_b, contrib_b in all_beliefs[i + 1:]:
                # Skip if from same federation
                if contrib_a.federation_id == contrib_b.federation_id:
                    continue
                
                # Check for conflict
                conflict = self._detect_belief_conflict(
                    belief_a, belief_b,
                    contrib_a, contrib_b,
                    similarity_fn,
                )
                if conflict.conflict_type != ConflictType.NONE:
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_belief_conflict(
        self,
        belief_a: FederatedBelief,
        belief_b: FederatedBelief,
        contrib_a: FederationContribution,
        contrib_b: FederationContribution,
        similarity_fn: Callable[[str, str], float] | None = None,
    ) -> DetectedConflict:
        """Detect conflict between two specific beliefs."""
        # Compute semantic similarity
        if similarity_fn:
            similarity = similarity_fn(belief_a.content, belief_b.content)
        else:
            # Simple Jaccard similarity fallback
            similarity = self._jaccard_similarity(belief_a.content, belief_b.content)
        
        # Not similar enough to be about same topic
        if similarity < self.semantic_threshold:
            return DetectedConflict(conflict_type=ConflictType.NONE)
        
        # Compute confidence divergence
        conf_a = belief_a.confidence.overall if belief_a.confidence else 0.5
        conf_b = belief_b.confidence.overall if belief_b.confidence else 0.5
        confidence_div = abs(conf_a - conf_b)
        
        # Detect stance divergence (simplified: using confidence direction)
        # In a full implementation, this would use stance analysis
        stance_div = self._compute_stance_divergence(belief_a, belief_b)
        
        # Determine conflict type
        conflict_type = ConflictType.NONE
        description = ""
        
        if stance_div > self.stance_threshold:
            conflict_type = ConflictType.CONTRADICTION
            description = "Beliefs express contradictory stances on the same topic"
        elif confidence_div > self.confidence_divergence:
            conflict_type = ConflictType.DIVERGENCE
            description = f"Significant confidence divergence ({confidence_div:.2f})"
        elif self._has_temporal_conflict(belief_a, belief_b):
            conflict_type = ConflictType.TEMPORAL
            description = "Temporal validity periods conflict"
        
        return DetectedConflict(
            conflict_type=conflict_type,
            belief_a_id=belief_a.id,
            belief_b_id=belief_b.id,
            federation_a_id=contrib_a.federation_id,
            federation_b_id=contrib_b.federation_id,
            semantic_similarity=similarity,
            confidence_divergence=confidence_div,
            stance_divergence=stance_div,
            description=description,
            belief_a_content=belief_a.content[:200],
            belief_b_content=belief_b.content[:200],
        )
    
    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """Compute Jaccard similarity between two texts."""
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_stance_divergence(
        self,
        belief_a: FederatedBelief,
        belief_b: FederatedBelief,
    ) -> float:
        """Compute stance divergence between beliefs.
        
        Uses simple heuristics. In production, would use NLP stance detection.
        """
        # Check for negation keywords
        negation_words = {"not", "no", "never", "false", "incorrect", "wrong", "deny"}
        
        words_a = set(belief_a.content.lower().split())
        words_b = set(belief_b.content.lower().split())
        
        neg_a = len(words_a & negation_words)
        neg_b = len(words_b & negation_words)
        
        # If one has negation and other doesn't, likely divergent
        if (neg_a > 0) != (neg_b > 0):
            return 0.7
        
        # Otherwise use confidence divergence as proxy
        conf_a = belief_a.confidence.overall if belief_a.confidence else 0.5
        conf_b = belief_b.confidence.overall if belief_b.confidence else 0.5
        
        # High confidence on both sides with different values suggests stance divergence
        if conf_a > 0.7 and conf_b > 0.7:
            return abs(conf_a - conf_b) * 0.5
        
        return 0.0
    
    def _has_temporal_conflict(
        self,
        belief_a: FederatedBelief,
        belief_b: FederatedBelief,
    ) -> bool:
        """Check if beliefs have conflicting temporal validity."""
        # If neither has temporal constraints, no conflict
        if not belief_a.valid_from and not belief_a.valid_until:
            return False
        if not belief_b.valid_from and not belief_b.valid_until:
            return False
        
        # Check for overlap with conflicting confidence
        # (simplified - in production would check actual overlap)
        return False


# =============================================================================
# WEIGHTED AGGREGATION
# =============================================================================


class TrustWeightedAggregator:
    """Aggregates beliefs using trust-weighted computation."""
    
    def __init__(
        self,
        trust_weight: float = DEFAULT_TRUST_WEIGHT,
        recency_weight: float = DEFAULT_RECENCY_WEIGHT,
        corroboration_weight: float = DEFAULT_CORROBORATION_WEIGHT,
    ):
        self.trust_weight = trust_weight
        self.recency_weight = recency_weight
        self.corroboration_weight = corroboration_weight
    
    def compute_contribution_weights(
        self,
        contributions: list[FederationContribution],
        smoother: TemporalSmoother | None = None,
    ) -> dict[UUID, float]:
        """Compute weight for each federation's contribution.
        
        Args:
            contributions: List of federation contributions
            smoother: Optional temporal smoother for membership changes
            
        Returns:
            Dictionary mapping federation_id to weight
        """
        weights: dict[UUID, float] = {}
        total_weight = 0.0
        
        for contrib in contributions:
            # Base weight from trust
            trust_component = contrib.trust_score * self.trust_weight
            
            # Anchor bonus
            if contrib.is_anchor:
                trust_component += ANCHOR_TRUST_BOOST * self.trust_weight
            
            # Recency component (based on most recent belief)
            recency_component = self._compute_recency_weight(contrib)
            
            # Corroboration component (based on belief count)
            corroboration_component = self._compute_corroboration_weight(
                contrib.belief_count,
                sum(c.belief_count for c in contributions),
            )
            
            # Temporal smoothing
            temporal_weight = 1.0
            if smoother and (contrib.joined_at or contrib.departed_at):
                temporal_weight = smoother.get_contribution_weight(
                    member_id=str(contrib.federation_id),
                    joined_at=contrib.joined_at,
                    departed_at=contrib.departed_at,
                )
            
            # Combine components
            weight = (
                trust_component +
                recency_component * self.recency_weight +
                corroboration_component * self.corroboration_weight
            ) * temporal_weight
            
            weights[contrib.federation_id] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            for fed_id in weights:
                weights[fed_id] /= total_weight
        
        return weights
    
    def aggregate_confidences(
        self,
        contributions: list[FederationContribution],
        weights: dict[UUID, float],
    ) -> float:
        """Compute weighted aggregate confidence.
        
        Args:
            contributions: Federation contributions
            weights: Weight per federation
            
        Returns:
            Weighted aggregate confidence
        """
        weighted_sum = 0.0
        total_weight = 0.0
        
        for contrib in contributions:
            weight = weights.get(contrib.federation_id, 0.0)
            
            if contrib.local_confidence is not None:
                weighted_sum += contrib.local_confidence * weight
                total_weight += weight
            elif contrib.beliefs:
                # Compute local confidence from beliefs
                local_conf = sum(
                    b.confidence.overall if b.confidence else 0.5
                    for b in contrib.beliefs
                ) / len(contrib.beliefs)
                weighted_sum += local_conf * weight
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def compute_agreement_score(
        self,
        contributions: list[FederationContribution],
        weights: dict[UUID, float],
    ) -> float:
        """Compute agreement score across federations.
        
        Lower variance = higher agreement.
        
        Returns:
            Agreement score in [0, 1]
        """
        if len(contributions) < 2:
            return 1.0  # Perfect agreement with self
        
        # Collect weighted confidences
        confidences: list[tuple[float, float]] = []  # (confidence, weight)
        
        for contrib in contributions:
            weight = weights.get(contrib.federation_id, 0.0)
            if weight <= 0:
                continue
            
            if contrib.local_confidence is not None:
                confidences.append((contrib.local_confidence, weight))
            elif contrib.beliefs:
                local_conf = sum(
                    b.confidence.overall if b.confidence else 0.5
                    for b in contrib.beliefs
                ) / len(contrib.beliefs)
                confidences.append((local_conf, weight))
        
        if len(confidences) < 2:
            return 1.0
        
        # Compute weighted mean
        total_weight = sum(w for _, w in confidences)
        if total_weight <= 0:
            return 0.5
        
        weighted_mean = sum(c * w for c, w in confidences) / total_weight
        
        # Compute weighted variance
        weighted_variance = sum(
            w * (c - weighted_mean) ** 2
            for c, w in confidences
        ) / total_weight
        
        # Convert to agreement score: 1 - normalized stddev
        # Max possible stddev is 0.5 (for values in [0,1])
        normalized_stddev = math.sqrt(weighted_variance) / 0.5
        agreement = 1.0 - min(1.0, normalized_stddev)
        
        return agreement
    
    def _compute_recency_weight(self, contrib: FederationContribution) -> float:
        """Compute recency weight based on belief timestamps."""
        if not contrib.beliefs:
            return 0.5
        
        # Get most recent belief
        now = datetime.now(UTC)
        max_age_days = 365
        
        most_recent = max(
            (b.signed_at for b in contrib.beliefs),
            default=now - timedelta(days=max_age_days),
        )
        
        age_days = (now - most_recent).days
        
        # Exponential decay with half-life of 30 days
        half_life = 30
        return math.exp(-math.log(2) * age_days / half_life)
    
    def _compute_corroboration_weight(
        self,
        belief_count: int,
        total_beliefs: int,
    ) -> float:
        """Compute corroboration weight based on belief contribution."""
        if total_beliefs <= 0:
            return 0.0
        
        # Diminishing returns: sqrt scale
        proportion = belief_count / total_beliefs
        return math.sqrt(proportion)


# =============================================================================
# PRIVACY-PRESERVING AGGREGATION
# =============================================================================


class PrivacyPreservingAggregator:
    """Wraps aggregation with differential privacy guarantees."""
    
    def __init__(
        self,
        privacy_config: PrivacyConfig | None = None,
        privacy_budget: PrivacyBudget | None = None,
    ):
        self.privacy_config = privacy_config or PrivacyConfig()
        self.privacy_budget = privacy_budget
    
    def apply_privacy(
        self,
        aggregate_confidence: float,
        agreement_score: float,
        federation_count: int,
        total_belief_count: int,
        total_contributor_count: int,
        domain_filter: list[str],
    ) -> tuple[float, float, int, int, int, bool]:
        """Apply differential privacy to aggregate statistics.
        
        Args:
            aggregate_confidence: True aggregate confidence
            agreement_score: True agreement score
            federation_count: True federation count
            total_belief_count: True belief count
            total_contributor_count: True contributor count
            domain_filter: Domains being queried
            
        Returns:
            Tuple of (noisy_confidence, noisy_agreement, noisy_fed_count,
                     noisy_belief_count, noisy_contributor_count, k_satisfied)
        """
        # Check if domain is sensitive
        config = self.privacy_config
        if is_sensitive_domain(domain_filter):
            # Use stricter parameters
            config = PrivacyConfig(
                epsilon=min(config.epsilon, 0.5),
                delta=min(config.delta, 1e-7),
                min_contributors=max(config.min_contributors, 10),
                sensitive_domain=True,
            )
        
        # Check k-anonymity
        k_satisfied = total_contributor_count >= config.effective_min_contributors
        
        if not k_satisfied:
            # Return suppressed values
            return 0.0, 0.0, 0, 0, 0, False
        
        # Add noise to statistics
        noisy_confidence = add_noise(
            aggregate_confidence,
            sensitivity=1.0 / total_contributor_count,
            config=config,
        )
        noisy_confidence = max(0.0, min(1.0, noisy_confidence))
        
        noisy_agreement = add_noise(
            agreement_score,
            sensitivity=1.0 / federation_count if federation_count > 0 else 1.0,
            config=config,
        )
        noisy_agreement = max(0.0, min(1.0, noisy_agreement))
        
        # Noise counts with sensitivity 1
        noisy_fed_count = max(0, round(add_noise(
            float(federation_count), 1.0, config
        )))
        noisy_belief_count = max(0, round(add_noise(
            float(total_belief_count), 1.0, config
        )))
        noisy_contributor_count = max(0, round(add_noise(
            float(total_contributor_count), 1.0, config
        )))
        
        return (
            noisy_confidence,
            noisy_agreement,
            noisy_fed_count,
            noisy_belief_count,
            noisy_contributor_count,
            k_satisfied,
        )
    
    def consume_budget(
        self,
        domain_filter: list[str],
        semantic_query: str | None = None,
        requester_id: str | None = None,
    ) -> bool:
        """Consume privacy budget for a query.
        
        Args:
            domain_filter: Domains being queried
            semantic_query: Optional semantic query
            requester_id: ID of requester
            
        Returns:
            True if budget allows query, False if exhausted
        """
        if not self.privacy_budget:
            return True
        
        topic_hash = compute_topic_hash(domain_filter, semantic_query)
        
        can_query, reason = self.privacy_budget.check_budget(
            self.privacy_config.epsilon,
            self.privacy_config.delta,
            topic_hash,
            requester_id,
        )
        
        if can_query:
            self.privacy_budget.consume(
                self.privacy_config.epsilon,
                self.privacy_config.delta,
                topic_hash,
                requester_id,
            )
        
        return can_query


# =============================================================================
# MAIN AGGREGATION ENGINE
# =============================================================================


class FederationAggregator:
    """Main engine for cross-federation belief aggregation.
    
    Coordinates conflict detection, trust-weighted aggregation,
    and privacy preservation to produce aggregate beliefs.
    """
    
    def __init__(
        self,
        config: AggregationConfig | None = None,
        trust_lookup: Callable[[UUID], float] | None = None,
    ):
        """Initialize the aggregator.
        
        Args:
            config: Aggregation configuration
            trust_lookup: Function to look up federation trust by ID
        """
        self.config = config or AggregationConfig()
        self.trust_lookup = trust_lookup or (lambda _: 0.5)
        
        # Initialize components
        self.conflict_detector = ConflictDetector(
            semantic_threshold=self.config.conflict_semantic_threshold,
            confidence_divergence=self.config.conflict_confidence_divergence,
        )
        self.weighted_aggregator = TrustWeightedAggregator(
            trust_weight=self.config.trust_weight,
            recency_weight=self.config.recency_weight,
            corroboration_weight=self.config.corroboration_weight,
        )
        self.privacy_aggregator = PrivacyPreservingAggregator(
            privacy_config=self.config.privacy_config,
        )
        
        # Temporal smoother
        self.smoother: TemporalSmoother | None = None
        if self.config.apply_temporal_smoothing:
            self.smoother = TemporalSmoother(self.config.smoothing_hours)
    
    def aggregate(
        self,
        contributions: list[FederationContribution],
        domain_filter: list[str] | None = None,
        semantic_query: str | None = None,
        similarity_fn: Callable[[str, str], float] | None = None,
        requester_id: str | None = None,
    ) -> CrossFederationAggregateResult:
        """Perform cross-federation aggregation.
        
        Args:
            contributions: Contributions from each federation
            domain_filter: Domain path filter for query
            semantic_query: Optional semantic query
            similarity_fn: Function for computing belief similarity
            requester_id: ID of requester (for privacy budget)
            
        Returns:
            CrossFederationAggregateResult with aggregated beliefs
        """
        domain_filter = domain_filter or []
        
        # Compute query hash
        query_hash = compute_topic_hash(domain_filter, semantic_query)
        
        # Filter contributions by minimum trust
        valid_contributions = [
            c for c in contributions
            if c.trust_score >= self.config.min_federation_trust
        ]
        
        # Check minimum federation requirement
        if len(valid_contributions) < self.config.min_federations:
            return CrossFederationAggregateResult(
                query_hash=query_hash,
                domain_filter=domain_filter,
                semantic_query=semantic_query,
                k_anonymity_satisfied=False,
                config_used=self.config.to_dict(),
            )
        
        # Count total beliefs and contributors
        total_beliefs = sum(c.belief_count for c in valid_contributions)
        total_contributors = len(valid_contributions)
        
        # Check minimum beliefs
        if total_beliefs < self.config.min_total_beliefs:
            return CrossFederationAggregateResult(
                query_hash=query_hash,
                domain_filter=domain_filter,
                semantic_query=semantic_query,
                federation_count=len(valid_contributions),
                k_anonymity_satisfied=False,
                config_used=self.config.to_dict(),
            )
        
        # Detect conflicts
        conflicts = self.conflict_detector.detect_conflicts(
            valid_contributions, similarity_fn
        )
        
        # Handle conflicts based on resolution strategy
        resolved_contributions = self._resolve_conflicts(
            valid_contributions, conflicts
        )
        
        # Compute contribution weights
        weights = self.weighted_aggregator.compute_contribution_weights(
            resolved_contributions,
            self.smoother,
        )
        
        # Aggregate confidences
        aggregate_confidence = self.weighted_aggregator.aggregate_confidences(
            resolved_contributions, weights
        )
        
        # Compute agreement score
        agreement_score = self.weighted_aggregator.compute_agreement_score(
            resolved_contributions, weights
        )
        
        # Apply privacy
        (
            noisy_confidence,
            noisy_agreement,
            noisy_fed_count,
            noisy_belief_count,
            noisy_contributor_count,
            k_satisfied,
        ) = self.privacy_aggregator.apply_privacy(
            aggregate_confidence,
            agreement_score,
            len(resolved_contributions),
            total_beliefs,
            total_contributors,
            domain_filter,
        )
        
        # Consume privacy budget
        if requester_id:
            self.privacy_aggregator.consume_budget(
                domain_filter, semantic_query, requester_id
            )
        
        # Build result
        return CrossFederationAggregateResult(
            query_hash=query_hash,
            domain_filter=domain_filter,
            semantic_query=semantic_query,
            collective_confidence=noisy_confidence,
            agreement_score=noisy_agreement,
            federation_count=noisy_fed_count,
            total_belief_count=noisy_belief_count,
            total_contributor_count=noisy_contributor_count,
            conflicts_detected=conflicts,
            conflict_count=len(conflicts),
            conflicts_resolved=len([c for c in conflicts if c.conflict_type != ConflictType.NONE]),
            privacy_epsilon=self.config.privacy_config.epsilon,
            privacy_delta=self.config.privacy_config.delta,
            k_anonymity_satisfied=k_satisfied,
            config_used=self.config.to_dict(),
            valid_until=datetime.now(UTC) + timedelta(hours=1),
        )
    
    def _resolve_conflicts(
        self,
        contributions: list[FederationContribution],
        conflicts: list[DetectedConflict],
    ) -> list[FederationContribution]:
        """Resolve detected conflicts based on strategy.
        
        Args:
            contributions: Original contributions
            conflicts: Detected conflicts
            
        Returns:
            Contributions after conflict resolution
        """
        if not conflicts:
            return contributions
        
        resolution = self.config.conflict_resolution
        
        if resolution == ConflictResolution.TRUST_WEIGHTED:
            # Trust-weighted: keep all, let weights handle it
            return contributions
        
        elif resolution == ConflictResolution.RECENCY_WINS:
            # Remove beliefs from older conflicts
            beliefs_to_exclude: set[UUID] = set()
            
            for conflict in conflicts:
                if conflict.conflict_type == ConflictType.NONE:
                    continue
                
                # Find which belief is older
                belief_a = self._find_belief(contributions, conflict.belief_a_id)
                belief_b = self._find_belief(contributions, conflict.belief_b_id)
                
                if belief_a and belief_b and conflict.belief_a_id and conflict.belief_b_id:
                    if belief_a.signed_at < belief_b.signed_at:
                        beliefs_to_exclude.add(conflict.belief_a_id)
                    else:
                        beliefs_to_exclude.add(conflict.belief_b_id)
            
            return self._filter_beliefs(contributions, beliefs_to_exclude)
        
        elif resolution == ConflictResolution.EXCLUDE_CONFLICTING:
            # Exclude all conflicting beliefs
            exclude_set: set[UUID] = set()
            
            for conflict in conflicts:
                if conflict.conflict_type != ConflictType.NONE:
                    if conflict.belief_a_id:
                        exclude_set.add(conflict.belief_a_id)
                    if conflict.belief_b_id:
                        exclude_set.add(conflict.belief_b_id)
            
            return self._filter_beliefs(contributions, exclude_set)
        
        elif resolution == ConflictResolution.CORROBORATION:
            # Keep beliefs that have more corroboration (proxy: federation trust)
            corr_exclude: set[UUID] = set()
            
            for conflict in conflicts:
                if conflict.conflict_type == ConflictType.NONE:
                    continue
                
                contrib_a = self._find_contribution(contributions, conflict.federation_a_id)
                contrib_b = self._find_contribution(contributions, conflict.federation_b_id)
                
                if contrib_a and contrib_b and conflict.belief_a_id and conflict.belief_b_id:
                    if contrib_a.trust_score < contrib_b.trust_score:
                        corr_exclude.add(conflict.belief_a_id)
                    else:
                        corr_exclude.add(conflict.belief_b_id)
            
            return self._filter_beliefs(contributions, corr_exclude)
        
        # FLAG_FOR_REVIEW or default: return as-is
        return contributions
    
    def _find_belief(
        self,
        contributions: list[FederationContribution],
        belief_id: UUID | None,
    ) -> FederatedBelief | None:
        """Find a belief by ID across contributions."""
        if not belief_id:
            return None
        
        for contrib in contributions:
            for belief in contrib.beliefs:
                if belief.id == belief_id:
                    return belief
        return None
    
    def _find_contribution(
        self,
        contributions: list[FederationContribution],
        federation_id: UUID | None,
    ) -> FederationContribution | None:
        """Find a contribution by federation ID."""
        if not federation_id:
            return None
        
        for contrib in contributions:
            if contrib.federation_id == federation_id:
                return contrib
        return None
    
    def _filter_beliefs(
        self,
        contributions: list[FederationContribution],
        exclude_ids: set[UUID],
    ) -> list[FederationContribution]:
        """Filter out beliefs with given IDs."""
        result: list[FederationContribution] = []
        
        for contrib in contributions:
            filtered_beliefs = [
                b for b in contrib.beliefs
                if b.id not in exclude_ids
            ]
            
            if filtered_beliefs:
                result.append(FederationContribution(
                    federation_id=contrib.federation_id,
                    node_id=contrib.node_id,
                    federation_did=contrib.federation_did,
                    trust_score=contrib.trust_score,
                    is_anchor=contrib.is_anchor,
                    beliefs=filtered_beliefs,
                    local_confidence=contrib.local_confidence,
                    local_agreement=contrib.local_agreement,
                    joined_at=contrib.joined_at,
                    departed_at=contrib.departed_at,
                    contribution_weight=contrib.contribution_weight,
                ))
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def aggregate_cross_federation(
    contributions: list[FederationContribution],
    domain_filter: list[str] | None = None,
    semantic_query: str | None = None,
    config: AggregationConfig | None = None,
    trust_lookup: Callable[[UUID], float] | None = None,
) -> CrossFederationAggregateResult:
    """Convenience function for cross-federation aggregation.
    
    Args:
        contributions: List of federation contributions
        domain_filter: Domain filter for query
        semantic_query: Optional semantic query
        config: Aggregation configuration
        trust_lookup: Function to look up federation trust
        
    Returns:
        CrossFederationAggregateResult
    """
    aggregator = FederationAggregator(config=config, trust_lookup=trust_lookup)
    return aggregator.aggregate(
        contributions,
        domain_filter=domain_filter,
        semantic_query=semantic_query,
    )


def create_contribution(
    federation_id: UUID,
    node_id: UUID,
    federation_did: str,
    beliefs: list[FederatedBelief],
    trust_score: float = 0.5,
    is_anchor: bool = False,
) -> FederationContribution:
    """Helper to create a federation contribution.
    
    Args:
        federation_id: Federation UUID
        node_id: Node UUID
        federation_did: Federation DID string
        beliefs: List of beliefs from this federation
        trust_score: Trust score for federation
        is_anchor: Whether federation is an anchor
        
    Returns:
        FederationContribution
    """
    return FederationContribution(
        federation_id=federation_id,
        node_id=node_id,
        federation_did=federation_did,
        beliefs=beliefs,
        trust_score=trust_score,
        is_anchor=is_anchor,
    )
