"""Consensus mechanism — layer elevation, corroboration, challenges.

Implements the trust layer model from spec/components/consensus-mechanism/SPEC.md:
- L1 (Personal) → L2 (Federated) → L3 (Domain) → L4 (Communal)
- Corroboration tracking with independence scoring
- Challenge submission and resolution
- Finality computation
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from our_db import get_cursor

from ..exceptions import NotFoundError, ValidationException

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class TrustLayer(StrEnum):
    """Trust layers from L1 (personal) to L4 (communal consensus)."""

    L1_PERSONAL = "l1_personal"
    L2_FEDERATED = "l2_federated"
    L3_DOMAIN = "l3_domain"
    L4_COMMUNAL = "l4_communal"


class ChallengeStatus(StrEnum):
    PENDING = "pending"
    REVIEWING = "reviewing"
    UPHELD = "upheld"
    REJECTED = "rejected"
    EXPIRED = "expired"


class FinalityLevel(StrEnum):
    """How final is the belief's consensus status?"""

    TENTATIVE = "tentative"
    PROVISIONAL = "provisional"
    ESTABLISHED = "established"
    SETTLED = "settled"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class IndependenceScore:
    """How independent two corroborating beliefs are."""

    evidential: float = 0.0
    source: float = 0.0
    method: float = 0.0
    temporal: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "evidential": self.evidential,
            "source": self.source,
            "method": self.method,
            "temporal": self.temporal,
            "overall": self.overall,
        }


@dataclass
class Corroboration:
    """A corroboration event between two beliefs."""

    id: UUID
    primary_belief_id: UUID
    corroborating_belief_id: UUID
    primary_holder: str
    corroborator: str
    semantic_similarity: float
    independence: IndependenceScore
    effective_weight: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class BeliefConsensusStatus:
    """Current consensus status for a belief."""

    belief_id: UUID
    current_layer: TrustLayer = TrustLayer.L1_PERSONAL
    corroboration_count: int = 0
    total_corroboration_weight: float = 0.0
    finality: FinalityLevel = FinalityLevel.TENTATIVE
    last_challenge_at: datetime | None = None
    elevated_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Challenge:
    """A challenge to a belief's consensus layer."""

    id: UUID
    belief_id: UUID
    challenger_id: str
    target_layer: TrustLayer
    reasoning: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    stake_amount: float = 0.0
    status: ChallengeStatus = ChallengeStatus.PENDING
    resolution_reasoning: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: datetime | None = None


# ============================================================================
# Elevation Thresholds (from spec Section 3)
# ============================================================================

ELEVATION_THRESHOLDS = {
    TrustLayer.L2_FEDERATED: {
        "min_contributors": 5,
        "min_agreement_score": 0.6,
        "min_independence": 0.0,  # No independence required at L2
    },
    TrustLayer.L3_DOMAIN: {
        "min_contributors": 3,  # From different federations
        "min_agreement_score": 0.7,
        "min_independence": 0.5,
        "min_expert_count": 2,
        "min_expert_reputation": 0.7,
    },
    TrustLayer.L4_COMMUNAL: {
        "min_contributors": 10,
        "min_agreement_score": 0.8,
        "min_independence": 0.7,
        "min_stake_threshold": 0.67,  # 2/3 Byzantine threshold
    },
}


# ============================================================================
# Independence Calculation
# ============================================================================


def calculate_independence(
    belief_a_sources: list[str],
    belief_b_sources: list[str],
    belief_a_method: str | None = None,
    belief_b_method: str | None = None,
    time_gap_days: float = 0.0,
) -> IndependenceScore:
    """Calculate independence score between two corroborating beliefs.

    Uses Jaccard similarity for evidence/source overlap and temporal distance.
    """
    # Evidential independence: 1 - Jaccard similarity
    set_a = set(belief_a_sources)
    set_b = set(belief_b_sources)
    if set_a or set_b:
        jaccard = len(set_a & set_b) / len(set_a | set_b)
        evidential = 1.0 - jaccard
    else:
        evidential = 0.5  # No evidence to compare

    # Source independence (simplified: same as evidential for now)
    source = evidential

    # Method independence
    if belief_a_method and belief_b_method:
        method = 0.0 if belief_a_method == belief_b_method else 1.0
    else:
        method = 0.5

    # Temporal independence: max at 1 week apart
    temporal = min(1.0, time_gap_days / 7.0)

    # Weighted combination (from spec)
    overall = 0.4 * evidential + 0.3 * source + 0.2 * method + 0.1 * temporal

    return IndependenceScore(
        evidential=round(evidential, 4),
        source=round(source, 4),
        method=round(method, 4),
        temporal=round(temporal, 4),
        overall=round(overall, 4),
    )


# ============================================================================
# Consensus Status Operations
# ============================================================================


def get_consensus_status(belief_id: UUID) -> BeliefConsensusStatus | None:
    """Get the current consensus status for a belief."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM belief_consensus_status WHERE belief_id = %s", (str(belief_id),))
        row = cur.fetchone()
        if not row:
            return None

        return BeliefConsensusStatus(
            belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
            current_layer=TrustLayer(row["current_layer"]),
            corroboration_count=row["corroboration_count"],
            total_corroboration_weight=float(row["total_corroboration_weight"]),
            finality=FinalityLevel(row["finality"]),
            last_challenge_at=row.get("last_challenge_at"),
            elevated_at=row.get("elevated_at"),
            created_at=row["created_at"],
        )


def get_or_create_consensus_status(belief_id: UUID) -> BeliefConsensusStatus:
    """Get or create consensus status for a belief."""
    status = get_consensus_status(belief_id)
    if status:
        return status

    status = BeliefConsensusStatus(belief_id=belief_id)
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO belief_consensus_status (belief_id, current_layer, corroboration_count,
                total_corroboration_weight, finality, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (belief_id) DO NOTHING
            """,
            (
                str(belief_id), status.current_layer.value, 0, 0.0,
                status.finality.value, status.created_at,
            ),
        )
    return status


# ============================================================================
# Corroboration Operations
# ============================================================================


def submit_corroboration(
    primary_belief_id: UUID,
    corroborating_belief_id: UUID,
    primary_holder: str,
    corroborator: str,
    semantic_similarity: float,
    independence: IndependenceScore,
    corroborator_reputation: float = 0.5,
) -> Corroboration:
    """Submit a corroboration between two beliefs.

    Raises:
        ValidationException: If validation fails
    """
    if semantic_similarity < 0.85:
        raise ValidationException(f"Semantic similarity too low ({semantic_similarity:.2f}). Must be >= 0.85.")
    if primary_holder == corroborator:
        raise ValidationException("Cannot self-corroborate.")
    if primary_belief_id == corroborating_belief_id:
        raise ValidationException("Cannot corroborate a belief with itself.")

    effective_weight = independence.overall * corroborator_reputation

    corroboration = Corroboration(
        id=uuid4(),
        primary_belief_id=primary_belief_id,
        corroborating_belief_id=corroborating_belief_id,
        primary_holder=primary_holder,
        corroborator=corroborator,
        semantic_similarity=semantic_similarity,
        independence=independence,
        effective_weight=effective_weight,
    )

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO corroborations (
                id, primary_belief_id, corroborating_belief_id, primary_holder,
                corroborator, semantic_similarity, independence, effective_weight, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(corroboration.id), str(primary_belief_id), str(corroborating_belief_id),
                primary_holder, corroborator, semantic_similarity,
                json.dumps(independence.to_dict()), effective_weight, corroboration.created_at,
            ),
        )

        # Update consensus status
        cur.execute(
            """
            UPDATE belief_consensus_status
            SET corroboration_count = corroboration_count + 1,
                total_corroboration_weight = total_corroboration_weight + %s
            WHERE belief_id = %s
            """,
            (effective_weight, str(primary_belief_id)),
        )

    # Check if elevation is warranted
    _check_elevation(primary_belief_id)

    logger.info(f"Corroboration {corroboration.id}: {corroborating_belief_id} corroborates {primary_belief_id} (weight: {effective_weight:.4f})")
    return corroboration


def get_corroborations(belief_id: UUID) -> list[Corroboration]:
    """Get all corroborations for a belief."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM corroborations WHERE primary_belief_id = %s ORDER BY created_at DESC",
            (str(belief_id),),
        )
        result = []
        for row in cur.fetchall():
            ind = row.get("independence", {})
            if isinstance(ind, str):
                ind = json.loads(ind)

            result.append(Corroboration(
                id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
                primary_belief_id=row["primary_belief_id"] if isinstance(row["primary_belief_id"], UUID) else UUID(row["primary_belief_id"]),
                corroborating_belief_id=row["corroborating_belief_id"] if isinstance(row["corroborating_belief_id"], UUID) else UUID(row["corroborating_belief_id"]),
                primary_holder=row["primary_holder"],
                corroborator=row["corroborator"],
                semantic_similarity=float(row["semantic_similarity"]),
                independence=IndependenceScore(**ind) if ind else IndependenceScore(),
                effective_weight=float(row["effective_weight"]),
                created_at=row["created_at"],
            ))
        return result


# ============================================================================
# Elevation Logic
# ============================================================================


def _check_elevation(belief_id: UUID) -> TrustLayer | None:
    """Check if a belief should be elevated to a higher trust layer.

    Returns the new layer if elevated, None otherwise.
    """
    status = get_or_create_consensus_status(belief_id)
    corroborations = get_corroborations(belief_id)

    current = status.current_layer
    next_layer = _next_layer(current)
    if not next_layer:
        return None  # Already at L4

    thresholds = ELEVATION_THRESHOLDS[next_layer]

    # Check contributor count (unique corroborators)
    unique_corroborators = set(c.corroborator for c in corroborations)
    if len(unique_corroborators) < thresholds["min_contributors"]:
        return None

    # Check independence threshold
    if corroborations:
        avg_independence = sum(c.independence.overall for c in corroborations) / len(corroborations)
        if avg_independence < thresholds["min_independence"]:
            return None

    # Check L4-specific Byzantine threshold
    if next_layer == TrustLayer.L4_COMMUNAL:
        if status.total_corroboration_weight < thresholds["min_stake_threshold"]:
            return None

    # Elevation criteria met
    _elevate_belief(belief_id, next_layer)
    return next_layer


def _next_layer(current: TrustLayer) -> TrustLayer | None:
    """Get the next trust layer above the current one."""
    order = [TrustLayer.L1_PERSONAL, TrustLayer.L2_FEDERATED, TrustLayer.L3_DOMAIN, TrustLayer.L4_COMMUNAL]
    idx = order.index(current)
    return order[idx + 1] if idx < len(order) - 1 else None


def _elevate_belief(belief_id: UUID, new_layer: TrustLayer) -> None:
    """Elevate a belief to a higher trust layer."""
    now = datetime.now()
    finality = _compute_finality(new_layer, None)

    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE belief_consensus_status
            SET current_layer = %s, elevated_at = %s, finality = %s
            WHERE belief_id = %s
            """,
            (new_layer.value, now, finality.value, str(belief_id)),
        )

        # Record layer transition
        cur.execute(
            """
            INSERT INTO layer_transitions (id, belief_id, from_layer, to_layer, created_at)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (str(uuid4()), str(belief_id), None, new_layer.value, now),
        )

    logger.info(f"Belief {belief_id} elevated to {new_layer.value}")


def _compute_finality(layer: TrustLayer, last_challenge: datetime | None) -> FinalityLevel:
    """Compute finality level based on layer and challenge history."""
    if layer == TrustLayer.L1_PERSONAL:
        return FinalityLevel.TENTATIVE
    elif layer == TrustLayer.L2_FEDERATED:
        return FinalityLevel.PROVISIONAL
    elif layer == TrustLayer.L3_DOMAIN:
        if last_challenge and (datetime.now() - last_challenge).days < 30:
            return FinalityLevel.PROVISIONAL
        return FinalityLevel.ESTABLISHED
    elif layer == TrustLayer.L4_COMMUNAL:
        if last_challenge and (datetime.now() - last_challenge).days < 90:
            return FinalityLevel.ESTABLISHED
        return FinalityLevel.SETTLED
    return FinalityLevel.TENTATIVE


# ============================================================================
# Challenge Operations
# ============================================================================


def submit_challenge(
    belief_id: UUID,
    challenger_id: str,
    reasoning: str,
    evidence: list[dict[str, Any]] | None = None,
    stake_amount: float = 0.0,
) -> Challenge:
    """Submit a challenge to a belief's consensus status.

    Raises:
        NotFoundError: If belief has no consensus status
        ValidationException: If challenge is invalid
    """
    status = get_consensus_status(belief_id)
    if not status:
        raise NotFoundError("BeliefConsensusStatus", str(belief_id))

    if status.current_layer == TrustLayer.L1_PERSONAL:
        raise ValidationException("Cannot challenge L1 personal beliefs")

    challenge = Challenge(
        id=uuid4(),
        belief_id=belief_id,
        challenger_id=challenger_id,
        target_layer=status.current_layer,
        reasoning=reasoning,
        evidence=evidence or [],
        stake_amount=stake_amount,
    )

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO challenges (
                id, belief_id, challenger_id, target_layer, reasoning,
                evidence, stake_amount, status, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(challenge.id), str(belief_id), challenger_id,
                challenge.target_layer.value, reasoning,
                json.dumps(evidence or []), stake_amount,
                challenge.status.value, challenge.created_at,
            ),
        )

        # Update last challenge timestamp
        cur.execute(
            "UPDATE belief_consensus_status SET last_challenge_at = %s WHERE belief_id = %s",
            (challenge.created_at, str(belief_id)),
        )

    logger.info(f"Challenge {challenge.id} submitted against {belief_id} at layer {status.current_layer.value}")
    return challenge


def resolve_challenge(
    challenge_id: UUID,
    upheld: bool,
    resolution_reasoning: str,
) -> Challenge:
    """Resolve a pending challenge.

    If upheld, the belief may be demoted to a lower layer.
    If rejected, the challenger loses their stake.

    Raises:
        NotFoundError: If challenge doesn't exist
        ValidationException: If challenge is not pending
    """
    with get_cursor() as cur:
        cur.execute("SELECT * FROM challenges WHERE id = %s", (str(challenge_id),))
        row = cur.fetchone()
        if not row:
            raise NotFoundError("Challenge", str(challenge_id))

        if row["status"] not in (ChallengeStatus.PENDING.value, ChallengeStatus.REVIEWING.value):
            raise ValidationException(f"Challenge is not pending: {row['status']}")

    challenge = Challenge(
        id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
        belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
        challenger_id=row["challenger_id"],
        target_layer=TrustLayer(row["target_layer"]),
        reasoning=row["reasoning"],
        evidence=json.loads(row["evidence"]) if isinstance(row["evidence"], str) else (row["evidence"] or []),
        stake_amount=float(row.get("stake_amount", 0)),
        status=ChallengeStatus.UPHELD if upheld else ChallengeStatus.REJECTED,
        resolution_reasoning=resolution_reasoning,
        resolved_at=datetime.now(),
    )

    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE challenges
            SET status = %s, resolution_reasoning = %s, resolved_at = %s
            WHERE id = %s
            """,
            (challenge.status.value, resolution_reasoning, challenge.resolved_at, str(challenge_id)),
        )

        if upheld:
            # Demote belief to previous layer
            prev_layer = _prev_layer(challenge.target_layer)
            if prev_layer:
                cur.execute(
                    "UPDATE belief_consensus_status SET current_layer = %s, finality = %s WHERE belief_id = %s",
                    (prev_layer.value, FinalityLevel.TENTATIVE.value, str(challenge.belief_id)),
                )
                logger.info(f"Belief {challenge.belief_id} demoted to {prev_layer.value}")

    return challenge


def get_challenge(challenge_id: UUID) -> Challenge | None:
    """Get a challenge by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM challenges WHERE id = %s", (str(challenge_id),))
        row = cur.fetchone()
        if not row:
            return None

        return Challenge(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
            challenger_id=row["challenger_id"],
            target_layer=TrustLayer(row["target_layer"]),
            reasoning=row["reasoning"],
            evidence=json.loads(row["evidence"]) if isinstance(row["evidence"], str) else (row["evidence"] or []),
            stake_amount=float(row.get("stake_amount", 0)),
            status=ChallengeStatus(row["status"]),
            resolution_reasoning=row.get("resolution_reasoning"),
            created_at=row["created_at"],
            resolved_at=row.get("resolved_at"),
        )


def get_challenges_for_belief(belief_id: UUID) -> list[Challenge]:
    """Get all challenges for a belief."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM challenges WHERE belief_id = %s ORDER BY created_at DESC",
            (str(belief_id),),
        )
        return [
            Challenge(
                id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
                belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
                challenger_id=row["challenger_id"],
                target_layer=TrustLayer(row["target_layer"]),
                reasoning=row["reasoning"],
                evidence=json.loads(row["evidence"]) if isinstance(row["evidence"], str) else (row["evidence"] or []),
                stake_amount=float(row.get("stake_amount", 0)),
                status=ChallengeStatus(row["status"]),
                resolution_reasoning=row.get("resolution_reasoning"),
                created_at=row["created_at"],
                resolved_at=row.get("resolved_at"),
            )
            for row in cur.fetchall()
        ]


def _prev_layer(current: TrustLayer) -> TrustLayer | None:
    """Get the previous (lower) trust layer."""
    order = [TrustLayer.L1_PERSONAL, TrustLayer.L2_FEDERATED, TrustLayer.L3_DOMAIN, TrustLayer.L4_COMMUNAL]
    idx = order.index(current)
    return order[idx - 1] if idx > 0 else None
