"""Anti-Gaming Measures for Validator Selection.

Implements countermeasures against consensus capture attempts:
- Tenure penalties to prevent entrenchment
- Diversity scoring for validator sets
- Collusion detection patterns
- Stake manipulation detection
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Sequence
from uuid import UUID

from .models import (
    Validator,
    ValidatorSet,
    ValidatorTier,
    ValidatorPerformance,
    SlashingEvent,
    SlashingOffense,
    SlashingStatus,
    SlashingEvidence,
)


# =============================================================================
# ANTI-GAMING THRESHOLDS
# =============================================================================

# Tenure penalties
MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY = 4
TENURE_PENALTY_FACTOR = 0.9  # 10% reduction per epoch after threshold

# Collusion detection
VOTING_CORRELATION_THRESHOLD = 0.95  # Flag if >95% vote correlation
MIN_VOTES_FOR_CORRELATION = 20       # Need at least 20 votes to analyze
STAKE_TIMING_WINDOW_HOURS = 24       # Suspicious if stakes registered within window
MIN_CORRELATED_VALIDATORS = 3        # Flag if 3+ validators show correlation

# Diversity scoring
IDEAL_FEDERATION_DIVERSITY = 0.8     # Gini coefficient target
MIN_TIER_ENTROPY = 0.5               # Minimum tier distribution entropy


class CollusionIndicator(str, Enum):
    """Types of potential collusion indicators."""
    VOTING_CORRELATION = "voting_correlation"
    STAKE_TIMING = "stake_timing"
    FEDERATION_CLUSTERING = "federation_clustering"
    REWARD_SHARING = "reward_sharing"
    COORDINATED_UNAVAILABILITY = "coordinated_unavailability"


class SeverityLevel(str, Enum):
    """Severity levels for detected anomalies."""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# TENURE PENALTY
# =============================================================================


def compute_tenure_penalty(consecutive_epochs: int) -> float:
    """Compute the tenure penalty factor for anti-entrenchment.
    
    Per NODE-SELECTION.md:
    - No penalty for first 4 consecutive epochs
    - 10% reduction (0.9×) per epoch after that
    
    Examples:
        - 1 epoch: 1.0 (no penalty)
        - 4 epochs: 1.0 (no penalty)
        - 5 epochs: 0.9 (10% penalty)
        - 6 epochs: 0.81 (19% penalty)
        - 12 epochs: 0.43 (57% penalty)
    
    Args:
        consecutive_epochs: Number of consecutive epochs served
    
    Returns:
        Penalty multiplier (1.0 = no penalty, <1.0 = penalized)
    """
    if consecutive_epochs <= MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY:
        return 1.0
    
    excess = consecutive_epochs - MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY
    return TENURE_PENALTY_FACTOR ** excess


def tenure_epochs_until_disadvantage(consecutive_epochs: int) -> int | None:
    """Calculate epochs until a validator becomes disadvantaged.
    
    A validator becomes disadvantaged when their tenure penalty
    makes them less competitive than a new validator.
    
    Args:
        consecutive_epochs: Current consecutive epochs
    
    Returns:
        Number of additional epochs until disadvantage, or None if already disadvantaged
    """
    # A standard tier validator with penalty < 1.0 can be beaten by
    # a new standard tier validator with weight 1.0
    
    if consecutive_epochs > MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY:
        # Already in penalty zone
        penalty = compute_tenure_penalty(consecutive_epochs)
        if penalty < 1.0:
            return None  # Already disadvantaged
    
    return MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY - consecutive_epochs + 1


# =============================================================================
# DIVERSITY SCORING
# =============================================================================


def compute_diversity_score(validator_set: ValidatorSet) -> dict[str, Any]:
    """Compute comprehensive diversity scores for a validator set.
    
    Analyzes:
    - Federation distribution (Gini coefficient)
    - Tier distribution (entropy)
    - Tenure distribution
    - Geographic distribution (if available)
    
    Args:
        validator_set: The validator set to analyze
    
    Returns:
        Dictionary with diversity metrics
    """
    validators = validator_set.validators
    n = len(validators)
    
    if n == 0:
        return {
            "overall_score": 0.0,
            "federation_gini": 1.0,
            "tier_entropy": 0.0,
            "tenure_variance": 0.0,
            "new_validator_ratio": 0.0,
        }
    
    # Federation distribution (using Gini coefficient)
    federation_counts: Counter[str] = Counter()
    for v in validators:
        for fed in v.federation_membership:
            federation_counts[fed] += 1
    federation_gini = _compute_gini(list(federation_counts.values())) if federation_counts else 1.0
    
    # Tier distribution (using entropy)
    tier_counts = Counter(v.tier for v in validators)
    tier_entropy = _compute_entropy(list(tier_counts.values()))
    max_tier_entropy = _compute_entropy([n // 3] * 3)  # Maximum entropy if evenly split
    normalized_tier_entropy = tier_entropy / max_tier_entropy if max_tier_entropy > 0 else 0.0
    
    # Tenure distribution
    tenures = [v.tenure_epochs for v in validators]
    tenure_variance = _compute_variance(tenures)
    
    # New validator ratio
    new_count = sum(1 for v in validators if v.tenure_epochs <= 1)
    new_validator_ratio = new_count / n
    
    # Overall score (weighted average)
    # Lower Gini = more equal distribution = better
    # Higher entropy = more variety = better
    # Higher new ratio = more turnover = better
    overall_score = (
        (1.0 - federation_gini) * 0.4 +
        normalized_tier_entropy * 0.3 +
        new_validator_ratio * 0.3
    )
    
    return {
        "overall_score": overall_score,
        "federation_gini": federation_gini,
        "federation_counts": dict(federation_counts),
        "tier_entropy": tier_entropy,
        "tier_counts": {k.value: v for k, v in tier_counts.items()},
        "tenure_variance": tenure_variance,
        "tenure_distribution": Counter(tenures),
        "new_validator_ratio": new_validator_ratio,
        "validators_analyzed": n,
    }


def _compute_gini(values: list[int]) -> float:
    """Compute Gini coefficient for a distribution.
    
    0 = perfect equality
    1 = perfect inequality
    """
    if not values or sum(values) == 0:
        return 1.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)
    
    cumulative = 0
    gini_sum = 0
    for i, value in enumerate(sorted_values):
        cumulative += value
        gini_sum += (2 * (i + 1) - n - 1) * value
    
    return gini_sum / (n * total)


def _compute_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy for a distribution."""
    import math
    
    total = sum(counts)
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def _compute_variance(values: Sequence[int | float]) -> float:
    """Compute variance of a sequence of values."""
    if not values:
        return 0.0
    
    n = len(values)
    mean = sum(values) / n
    return sum((x - mean) ** 2 for x in values) / n


# =============================================================================
# COLLUSION DETECTION
# =============================================================================


@dataclass
class CollusionAlert:
    """Alert for potential collusion detection."""
    
    id: UUID
    indicator: CollusionIndicator
    severity: SeverityLevel
    
    # Involved validators
    validators: list[str]  # DIDs
    
    # Evidence
    description: str
    evidence_data: dict[str, Any] = field(default_factory=dict)
    
    # Timing
    detected_at: datetime = field(default_factory=datetime.now)
    epoch: int = 0
    
    # Status
    investigated: bool = False
    resolved: bool = False
    resolution_notes: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "indicator": self.indicator.value,
            "severity": self.severity.value,
            "validators": self.validators,
            "description": self.description,
            "evidence_data": self.evidence_data,
            "detected_at": self.detected_at.isoformat(),
            "epoch": self.epoch,
            "investigated": self.investigated,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class VotingRecord:
    """Record of a validator's vote for correlation analysis."""
    
    validator_id: str
    proposal_id: UUID
    vote: str  # 'approve', 'reject', 'abstain'
    voted_at: datetime


def detect_collusion_patterns(
    voting_records: list[VotingRecord],
    stake_registrations: list[tuple[str, datetime]],  # (validator_id, registered_at)
    validator_set: ValidatorSet,
) -> list[CollusionAlert]:
    """Detect potential collusion patterns in validator behavior.
    
    Checks for:
    1. Voting correlation: Validators voting identically on many proposals
    2. Stake timing: Validators registering stakes at suspiciously similar times
    3. Federation clustering: Unusual concentration from federated nodes
    
    Args:
        voting_records: Historical voting records
        stake_registrations: List of (validator_id, registration_time) tuples
        validator_set: Current validator set
    
    Returns:
        List of collusion alerts
    """
    from uuid import uuid4
    
    alerts: list[CollusionAlert] = []
    
    # 1. Voting correlation analysis
    correlation_alerts = _analyze_voting_correlation(voting_records, validator_set.epoch)
    alerts.extend(correlation_alerts)
    
    # 2. Stake timing analysis
    timing_alerts = _analyze_stake_timing(stake_registrations, validator_set.epoch)
    alerts.extend(timing_alerts)
    
    # 3. Federation clustering analysis
    clustering_alerts = _analyze_federation_clustering(validator_set)
    alerts.extend(clustering_alerts)
    
    return alerts


def _analyze_voting_correlation(
    records: list[VotingRecord],
    epoch: int,
) -> list[CollusionAlert]:
    """Analyze voting records for suspicious correlation."""
    from uuid import uuid4
    
    alerts: list[CollusionAlert] = []
    
    # Group votes by proposal
    proposals: dict[UUID, dict[str, str]] = {}  # proposal_id -> {validator_id -> vote}
    for record in records:
        if record.proposal_id not in proposals:
            proposals[record.proposal_id] = {}
        proposals[record.proposal_id][record.validator_id] = record.vote
    
    # Not enough proposals to analyze
    if len(proposals) < MIN_VOTES_FOR_CORRELATION:
        return alerts
    
    # Get all validators
    all_validators: set[str] = set()
    for votes in proposals.values():
        all_validators.update(votes.keys())
    
    # Compute pairwise correlation
    validator_list = list(all_validators)
    correlation_matrix: dict[tuple[str, str], float] = {}
    
    for i, v1 in enumerate(validator_list):
        for v2 in validator_list[i+1:]:
            # Count matching votes
            matching = 0
            total = 0
            for proposal_votes in proposals.values():
                if v1 in proposal_votes and v2 in proposal_votes:
                    total += 1
                    if proposal_votes[v1] == proposal_votes[v2]:
                        matching += 1
            
            if total >= MIN_VOTES_FOR_CORRELATION:
                correlation = matching / total
                correlation_matrix[(v1, v2)] = correlation
    
    # Find highly correlated groups
    high_correlation_pairs = [
        (v1, v2, corr) for (v1, v2), corr in correlation_matrix.items()
        if corr >= VOTING_CORRELATION_THRESHOLD
    ]
    
    if len(high_correlation_pairs) >= MIN_CORRELATED_VALIDATORS - 1:
        # Found suspicious correlation
        involved = set()
        for v1, v2, _ in high_correlation_pairs:
            involved.add(v1)
            involved.add(v2)
        
        avg_correlation = sum(c for _, _, c in high_correlation_pairs) / len(high_correlation_pairs)
        
        alerts.append(CollusionAlert(
            id=uuid4(),
            indicator=CollusionIndicator.VOTING_CORRELATION,
            severity=SeverityLevel.HIGH if avg_correlation > 0.98 else SeverityLevel.WARNING,
            validators=list(involved),
            description=f"High voting correlation ({avg_correlation:.2%}) detected among {len(involved)} validators",
            evidence_data={
                "correlation_pairs": [
                    {"v1": v1, "v2": v2, "correlation": corr}
                    for v1, v2, corr in high_correlation_pairs
                ],
                "proposals_analyzed": len(proposals),
            },
            epoch=epoch,
        ))
    
    return alerts


def _analyze_stake_timing(
    registrations: list[tuple[str, datetime]],
    epoch: int,
) -> list[CollusionAlert]:
    """Analyze stake registration timing for suspicious patterns."""
    from uuid import uuid4
    
    alerts = []
    window = timedelta(hours=STAKE_TIMING_WINDOW_HOURS)
    
    # Group registrations by time window
    registrations_sorted = sorted(registrations, key=lambda x: x[1])
    
    clusters: list[list[str]] = []
    current_cluster: list[str] = []
    cluster_start: datetime | None = None
    
    for validator_id, reg_time in registrations_sorted:
        if cluster_start is None:
            cluster_start = reg_time
            current_cluster = [validator_id]
        elif reg_time - cluster_start <= window:
            current_cluster.append(validator_id)
        else:
            if len(current_cluster) >= MIN_CORRELATED_VALIDATORS:
                clusters.append(current_cluster)
            cluster_start = reg_time
            current_cluster = [validator_id]
    
    # Check final cluster
    if len(current_cluster) >= MIN_CORRELATED_VALIDATORS:
        clusters.append(current_cluster)
    
    # Generate alerts for suspicious clusters
    for cluster in clusters:
        alerts.append(CollusionAlert(
            id=uuid4(),
            indicator=CollusionIndicator.STAKE_TIMING,
            severity=SeverityLevel.WARNING,
            validators=cluster,
            description=f"{len(cluster)} validators registered stakes within {STAKE_TIMING_WINDOW_HOURS}h window",
            evidence_data={
                "cluster_size": len(cluster),
                "window_hours": STAKE_TIMING_WINDOW_HOURS,
            },
            epoch=epoch,
        ))
    
    return alerts


def _analyze_federation_clustering(validator_set: ValidatorSet) -> list[CollusionAlert]:
    """Analyze federation membership for suspicious clustering."""
    from uuid import uuid4
    
    alerts: list[CollusionAlert] = []
    n = len(validator_set.validators)
    
    if n == 0:
        return alerts
    
    # Count federation memberships
    federation_counts: Counter[str] = Counter()
    for validator in validator_set.validators:
        for fed in validator.federation_membership:
            federation_counts[fed] += 1
    
    # Check for over-represented federations
    max_allowed = int(n * 0.25)  # 25% threshold (stricter than selection)
    
    for federation, count in federation_counts.items():
        if count > max_allowed:
            # Find validators from this federation
            validators_from_fed = [
                v.agent_id for v in validator_set.validators
                if federation in v.federation_membership
            ]
            
            alerts.append(CollusionAlert(
                id=uuid4(),
                indicator=CollusionIndicator.FEDERATION_CLUSTERING,
                severity=SeverityLevel.WARNING if count <= max_allowed * 1.5 else SeverityLevel.HIGH,
                validators=validators_from_fed,
                description=f"Federation '{federation}' over-represented: {count}/{n} ({count/n:.1%})",
                evidence_data={
                    "federation": federation,
                    "count": count,
                    "total_validators": n,
                    "threshold": max_allowed,
                    "percentage": count / n,
                },
                epoch=validator_set.epoch,
            ))
    
    return alerts


# =============================================================================
# ANTI-GAMING ENGINE
# =============================================================================


class AntiGamingEngine:
    """Comprehensive anti-gaming analysis engine.
    
    Provides a unified interface for:
    - Tenure penalty calculation
    - Diversity scoring
    - Collusion detection
    - Stake manipulation detection
    - Anomaly reporting
    
    Example:
        >>> engine = AntiGamingEngine()
        >>> analysis = engine.analyze_validator_set(validator_set, voting_records)
        >>> if analysis['alerts']:
        ...     print(f"Found {len(analysis['alerts'])} potential issues")
    """
    
    def __init__(
        self,
        voting_correlation_threshold: float = VOTING_CORRELATION_THRESHOLD,
        stake_timing_window_hours: int = STAKE_TIMING_WINDOW_HOURS,
    ):
        """Initialize the anti-gaming engine.
        
        Args:
            voting_correlation_threshold: Correlation threshold for collusion detection
            stake_timing_window_hours: Time window for stake timing analysis
        """
        self.voting_correlation_threshold = voting_correlation_threshold
        self.stake_timing_window_hours = stake_timing_window_hours
    
    def compute_tenure_penalty(self, consecutive_epochs: int) -> float:
        """Compute tenure penalty for a validator."""
        return compute_tenure_penalty(consecutive_epochs)
    
    def compute_diversity_score(self, validator_set: ValidatorSet) -> dict[str, Any]:
        """Compute diversity metrics for a validator set."""
        return compute_diversity_score(validator_set)
    
    def detect_collusion(
        self,
        voting_records: list[VotingRecord],
        stake_registrations: list[tuple[str, datetime]],
        validator_set: ValidatorSet,
    ) -> list[CollusionAlert]:
        """Run collusion detection analysis."""
        return detect_collusion_patterns(
            voting_records=voting_records,
            stake_registrations=stake_registrations,
            validator_set=validator_set,
        )
    
    def analyze_validator_set(
        self,
        validator_set: ValidatorSet,
        voting_records: list[VotingRecord] | None = None,
        stake_registrations: list[tuple[str, datetime]] | None = None,
    ) -> dict[str, Any]:
        """Run comprehensive analysis on a validator set.
        
        Args:
            validator_set: The validator set to analyze
            voting_records: Historical voting records (optional)
            stake_registrations: Stake registration timestamps (optional)
        
        Returns:
            Comprehensive analysis report
        """
        # Diversity analysis
        diversity = self.compute_diversity_score(validator_set)
        
        # Collusion detection
        alerts: list[CollusionAlert] = []
        if voting_records or stake_registrations:
            alerts = self.detect_collusion(
                voting_records=voting_records or [],
                stake_registrations=stake_registrations or [],
                validator_set=validator_set,
            )
        
        # Tenure analysis
        tenure_stats = {
            "validators_at_penalty": sum(
                1 for v in validator_set.validators
                if v.tenure_epochs > MAX_CONSECUTIVE_EPOCHS_BEFORE_PENALTY
            ),
            "max_tenure": max((v.tenure_epochs for v in validator_set.validators), default=0),
            "avg_tenure": (
                sum(v.tenure_epochs for v in validator_set.validators) / len(validator_set.validators)
                if validator_set.validators else 0
            ),
        }
        
        # Overall health score
        health_score = self._compute_health_score(diversity, alerts, tenure_stats)
        
        return {
            "validator_count": len(validator_set.validators),
            "epoch": validator_set.epoch,
            "diversity": diversity,
            "tenure_stats": tenure_stats,
            "alerts": [a.to_dict() for a in alerts],
            "alert_count": len(alerts),
            "health_score": health_score,
            "analysis_timestamp": datetime.now().isoformat(),
        }
    
    def _compute_health_score(
        self,
        diversity: dict[str, Any],
        alerts: list[CollusionAlert],
        tenure_stats: dict[str, Any],
    ) -> float:
        """Compute overall health score for validator set.
        
        Returns a score from 0.0 (critical issues) to 1.0 (healthy).
        """
        score = 1.0
        
        # Deduct for low diversity
        diversity_score = diversity.get("overall_score", 0.5)
        if diversity_score < 0.5:
            score -= (0.5 - diversity_score) * 0.5  # Up to -0.25
        
        # Deduct for alerts
        for alert in alerts:
            if alert.severity == SeverityLevel.CRITICAL:
                score -= 0.3
            elif alert.severity == SeverityLevel.HIGH:
                score -= 0.15
            elif alert.severity == SeverityLevel.WARNING:
                score -= 0.05
        
        # Deduct for tenure concentration
        penalty_ratio = tenure_stats["validators_at_penalty"] / max(1, len(alerts) + 1)
        if penalty_ratio > 0.3:
            score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def generate_slashing_evidence(
        self,
        alert: CollusionAlert,
    ) -> SlashingEvidence | None:
        """Generate slashing evidence from a collusion alert.
        
        Only generates evidence for HIGH or CRITICAL severity alerts.
        
        Args:
            alert: The collusion alert
        
        Returns:
            SlashingEvidence if evidence is sufficient, None otherwise
        """
        if alert.severity not in (SeverityLevel.HIGH, SeverityLevel.CRITICAL):
            return None
        
        import json
        
        evidence_data = json.dumps(alert.evidence_data).encode()
        evidence_hash = hashlib.sha256(evidence_data).digest()
        
        return SlashingEvidence(
            evidence_type=f"anti_gaming_{alert.indicator.value}",
            evidence_data=evidence_data,
            evidence_hash=evidence_hash,
            collected_at=alert.detected_at,
        )
