"""Dispute staking with quality scoring (#350).

Requires stake to file disputes, proportional to the belief's consensus level.
Failed disputes forfeit stake. Quality scoring tracks dispute accuracy per node,
and low-quality challengers are rate-limited.

Stake requirements:
- L1 belief (low consensus): 1x base stake
- L2 belief (medium consensus): 2x base stake
- L3+ belief (high consensus): 5x base stake

Quality scoring:
- score = wins / (filed + 1)  (Laplace smoothing)
- score < 0.2: require 2x stake multiplier
- score < 0.1: disputes rejected
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

BASE_STAKE = 1.0
QUALITY_REJECT_THRESHOLD = 0.1
QUALITY_PENALTY_THRESHOLD = 0.2
QUALITY_PENALTY_MULTIPLIER = 2.0


@dataclass
class StakeRequirement:
    """Calculated stake requirement for a dispute."""

    base_amount: float
    consensus_multiplier: float
    quality_multiplier: float
    total_required: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_amount": self.base_amount,
            "consensus_multiplier": self.consensus_multiplier,
            "quality_multiplier": self.quality_multiplier,
            "total_required": self.total_required,
            "reason": self.reason,
        }


@dataclass
class DisputeQuality:
    """Quality score for a dispute filer."""

    identity_id: str
    disputes_filed: int = 0
    disputes_won: int = 0
    disputes_lost: int = 0

    @property
    def score(self) -> float:
        """Laplace-smoothed quality score."""
        if self.disputes_filed == 0:
            return 0.5  # Neutral for new participants
        return self.disputes_won / (self.disputes_filed + 1)

    @property
    def can_file(self) -> bool:
        """Whether this identity can file disputes."""
        return self.score >= QUALITY_REJECT_THRESHOLD or self.disputes_filed < 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "disputes_filed": self.disputes_filed,
            "disputes_won": self.disputes_won,
            "disputes_lost": self.disputes_lost,
            "score": self.score,
            "can_file": self.can_file,
        }


def get_consensus_level(corroboration_count: int) -> int:
    """Map corroboration count to consensus level.

    Args:
        corroboration_count: Number of independent corroborations.

    Returns:
        Consensus level (1, 2, or 3).
    """
    if corroboration_count >= 5:
        return 3
    if corroboration_count >= 2:
        return 2
    return 1


def calculate_stake_requirement(
    corroboration_count: int,
    quality_score: float,
    base_stake: float = BASE_STAKE,
) -> StakeRequirement:
    """Calculate the stake required to file a dispute.

    Args:
        corroboration_count: Number of corroborations on the belief.
        quality_score: Filer's dispute quality score (0-1).
        base_stake: Base stake amount.

    Returns:
        StakeRequirement with breakdown.
    """
    level = get_consensus_level(corroboration_count)
    consensus_mult = {1: 1.0, 2: 2.0, 3: 5.0}[level]

    quality_mult = 1.0
    reason = f"L{level} belief"
    if quality_score < QUALITY_PENALTY_THRESHOLD:
        quality_mult = QUALITY_PENALTY_MULTIPLIER
        reason += f" + low quality penalty ({quality_score:.2f})"

    total = base_stake * consensus_mult * quality_mult

    return StakeRequirement(
        base_amount=base_stake,
        consensus_multiplier=consensus_mult,
        quality_multiplier=quality_mult,
        total_required=total,
        reason=reason,
    )


def get_dispute_quality(cur, identity_id: str) -> DisputeQuality:
    """Get the dispute quality score for an identity.

    Args:
        cur: Database cursor.
        identity_id: The identity to check.

    Returns:
        DisputeQuality with filed/won/lost counts.
    """
    cur.execute(
        """
        SELECT
            COUNT(*) as filed,
            COUNT(*) FILTER (WHERE outcome = 'upheld') as won,
            COUNT(*) FILTER (WHERE outcome IN ('overturned', 'dismissed')) as lost
        FROM disputes
        WHERE disputer_id = %s
        """,
        (identity_id,),
    )
    row = cur.fetchone()
    return DisputeQuality(
        identity_id=identity_id,
        disputes_filed=row["filed"],
        disputes_won=row["won"],
        disputes_lost=row["lost"],
    )


def validate_dispute_filing(
    cur,
    identity_id: str,
    belief_id: str,
    base_stake: float = BASE_STAKE,
) -> tuple[bool, StakeRequirement | str]:
    """Validate whether an identity can file a dispute on a belief.

    Args:
        cur: Database cursor.
        identity_id: The potential filer.
        belief_id: The belief to dispute.
        base_stake: Base stake amount.

    Returns:
        (allowed, StakeRequirement) if allowed, (False, reason_string) if not.
    """
    # Get quality score
    quality = get_dispute_quality(cur, identity_id)

    if not quality.can_file:
        return False, f"Dispute quality too low ({quality.score:.2f}). Must be >= {QUALITY_REJECT_THRESHOLD}"

    # Get belief corroboration count
    cur.execute(
        "SELECT COALESCE(corroboration_count, 0) as count FROM beliefs WHERE id = %s",
        (belief_id,),
    )
    row = cur.fetchone()
    if not row:
        return False, f"Belief not found: {belief_id}"

    requirement = calculate_stake_requirement(row["count"], quality.score, base_stake)

    return True, requirement
