"""Database persistence layer for the verification protocol.

Provides DB-backed operations for verifications, disputes, reputations,
and stake positions. Uses the same validation and calculation functions
as the in-memory VerificationService but persists to PostgreSQL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from our_db import get_cursor

from ..exceptions import NotFoundError, ValidationException
from .constants import ReputationConstants
from .enums import (
    DisputeOutcome,
    DisputeStatus,
    DisputeType,
    ResolutionMethod,
    StakeType,
    VerificationResult,
    VerificationStatus,
)
from .evidence import Evidence
from .results import ResultDetails, Stake
from .verification import (
    DiscrepancyBounty,
    Dispute,
    ReputationScore,
    ReputationUpdate,
    StakePosition,
    Verification,
    calculate_confirmation_reward,
    calculate_contradiction_reward,
    calculate_holder_confirmation_bonus,
    calculate_holder_contradiction_penalty,
    calculate_min_stake,
    calculate_partial_reward,
    validate_dispute_submission,
    validate_verification_submission,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Reputation Operations
# ============================================================================


def get_or_create_reputation(identity_id: str) -> ReputationScore:
    """Get or create reputation for an identity."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM reputations WHERE identity_id = %s", (identity_id,))
        row = cur.fetchone()
        if row:
            return ReputationScore.from_row(row)

        cur.execute(
            "INSERT INTO reputations (identity_id) VALUES (%s) RETURNING *",
            (identity_id,),
        )
        return ReputationScore.from_row(cur.fetchone())


def get_reputation(identity_id: str) -> ReputationScore | None:
    """Get reputation for an identity, or None if not found."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM reputations WHERE identity_id = %s", (identity_id,))
        row = cur.fetchone()
        return ReputationScore.from_row(row) if row else None


def update_reputation(reputation: ReputationScore) -> None:
    """Persist reputation changes to DB."""
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE reputations
            SET overall = %s, by_domain = %s, verification_count = %s,
                discrepancy_finds = %s, stake_at_risk = %s, modified_at = NOW()
            WHERE identity_id = %s
            """,
            (
                reputation.overall,
                json.dumps(reputation.by_domain),
                reputation.verification_count,
                reputation.discrepancy_finds,
                reputation.stake_at_risk,
                reputation.identity_id,
            ),
        )


# ============================================================================
# Verification Operations
# ============================================================================


def submit_verification(
    belief_id: UUID,
    belief_info: dict[str, Any],
    verifier_id: str,
    result: VerificationResult,
    evidence: list[Evidence],
    stake_amount: float,
    reasoning: str | None = None,
    result_details: ResultDetails | None = None,
) -> Verification:
    """Submit a new verification for a belief.

    Validates, locks stake, and persists to DB.

    Raises:
        ValidationException: If validation fails
    """
    holder_id = belief_info.get("holder_id", "unknown")

    stake = Stake(
        amount=stake_amount,
        type=StakeType.STANDARD,
        locked_until=datetime.now() + timedelta(days=ReputationConstants.STAKE_LOCKUP_DAYS),
        escrow_id=uuid4(),
    )

    verification = Verification(
        id=uuid4(),
        verifier_id=verifier_id,
        belief_id=belief_id,
        holder_id=holder_id,
        result=result,
        evidence=evidence,
        stake=stake,
        reasoning=reasoning,
        result_details=result_details,
        status=VerificationStatus.PENDING,
    )

    verifier_rep = get_or_create_reputation(verifier_id)
    existing = get_verifications_for_belief(belief_id)

    errors = validate_verification_submission(verification, belief_info, verifier_rep, existing)
    if errors:
        raise ValidationException("; ".join(errors))

    with get_cursor() as cur:
        # Lock stake
        _db_lock_stake(cur, verifier_rep, stake_amount, verification.id, None)

        # Insert verification
        cur.execute(
            """
            INSERT INTO verifications (
                id, verifier_id, belief_id, holder_id, result, evidence, stake,
                reasoning, result_details, status, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(verification.id),
                verification.verifier_id,
                str(verification.belief_id),
                verification.holder_id,
                verification.result.value,
                json.dumps([e.to_dict() for e in verification.evidence]),
                json.dumps(verification.stake.to_dict()),
                verification.reasoning,
                json.dumps(verification.result_details.to_dict()) if verification.result_details else None,
                verification.status.value,
                verification.created_at,
            ),
        )

    logger.info(f"Verification {verification.id} submitted by {verifier_id} for belief {belief_id}")
    return verification


def accept_verification(verification_id: UUID) -> Verification:
    """Accept a pending verification after validation window."""
    verification = get_verification(verification_id)
    if not verification:
        raise NotFoundError("Verification", str(verification_id))

    if verification.status != VerificationStatus.PENDING:
        raise ValidationException(f"Verification is not pending: {verification.status.value}")

    verification.status = VerificationStatus.ACCEPTED
    verification.accepted_at = datetime.now()

    with get_cursor() as cur:
        cur.execute(
            "UPDATE verifications SET status = %s, accepted_at = %s WHERE id = %s",
            (verification.status.value, verification.accepted_at, str(verification_id)),
        )

    _process_verification_reputation(verification)

    logger.info(f"Verification {verification_id} accepted")
    return verification


def get_verification(verification_id: UUID) -> Verification | None:
    """Get a verification by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM verifications WHERE id = %s", (str(verification_id),))
        row = cur.fetchone()
        return Verification.from_row(row) if row else None


def get_verifications_for_belief(belief_id: UUID) -> list[Verification]:
    """Get all verifications for a belief."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM verifications WHERE belief_id = %s ORDER BY created_at DESC",
            (str(belief_id),),
        )
        return [Verification.from_row(row) for row in cur.fetchall()]


def get_verification_summary(belief_id: UUID) -> dict[str, Any]:
    """Get summary of verifications for a belief."""
    verifications = get_verifications_for_belief(belief_id)

    by_result = {r.value: 0 for r in VerificationResult}
    by_status = {s.value: 0 for s in VerificationStatus}
    total_stake = 0.0

    for v in verifications:
        by_result[v.result.value] += 1
        by_status[v.status.value] += 1
        total_stake += v.stake.amount

    accepted = [v for v in verifications if v.status == VerificationStatus.ACCEPTED]
    consensus_result = None
    consensus_confidence = 0.0

    if accepted:
        result_scores: dict[str, float] = {}
        total_weight = 0.0

        for v in accepted:
            rep = get_or_create_reputation(v.verifier_id)
            weight = rep.overall * v.stake.amount
            result_scores[v.result.value] = result_scores.get(v.result.value, 0) + weight
            total_weight += weight

        if total_weight > 0:
            max_result = max(result_scores.items(), key=lambda x: x[1])
            consensus_result = max_result[0]
            consensus_confidence = max_result[1] / total_weight

    return {
        "total": len(verifications),
        "by_result": by_result,
        "by_status": by_status,
        "average_stake": total_stake / len(verifications) if verifications else 0,
        "total_stake": total_stake,
        "consensus_result": consensus_result,
        "consensus_confidence": consensus_confidence,
    }


# ============================================================================
# Dispute Operations
# ============================================================================


def submit_dispute(
    verification_id: UUID,
    disputer_id: str,
    counter_evidence: list[Evidence],
    stake_amount: float,
    dispute_type: DisputeType,
    reasoning: str,
    proposed_result: VerificationResult | None = None,
) -> Dispute:
    """Submit a dispute against a verification.

    Raises:
        NotFoundError: If verification doesn't exist
        ValidationException: If validation fails
    """
    verification = get_verification(verification_id)
    if not verification:
        raise NotFoundError("Verification", str(verification_id))

    stake = Stake(
        amount=stake_amount,
        type=StakeType.CHALLENGE,
        locked_until=datetime.now() + timedelta(days=ReputationConstants.RESOLUTION_TIMEOUT_DAYS),
        escrow_id=uuid4(),
    )

    dispute = Dispute(
        id=uuid4(),
        verification_id=verification_id,
        disputer_id=disputer_id,
        counter_evidence=counter_evidence,
        stake=stake,
        dispute_type=dispute_type,
        reasoning=reasoning,
        proposed_result=proposed_result,
    )

    disputer_rep = get_or_create_reputation(disputer_id)
    is_holder = disputer_id == verification.holder_id

    errors = validate_dispute_submission(dispute, verification, disputer_rep, is_holder)
    if errors:
        raise ValidationException("; ".join(errors))

    with get_cursor() as cur:
        # Lock stake
        _db_lock_stake(cur, disputer_rep, stake_amount, None, dispute.id)

        # Insert dispute
        cur.execute(
            """
            INSERT INTO disputes (
                id, verification_id, disputer_id, counter_evidence, stake,
                dispute_type, reasoning, proposed_result, status, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(dispute.id),
                str(dispute.verification_id),
                dispute.disputer_id,
                json.dumps([e.to_dict() for e in dispute.counter_evidence]),
                json.dumps(dispute.stake.to_dict()),
                dispute.dispute_type.value,
                dispute.reasoning,
                dispute.proposed_result.value if dispute.proposed_result else None,
                dispute.status.value,
                dispute.created_at,
            ),
        )

        # Update verification status
        cur.execute(
            "UPDATE verifications SET status = %s, dispute_id = %s WHERE id = %s",
            (VerificationStatus.DISPUTED.value, str(dispute.id), str(verification_id)),
        )

    logger.info(f"Dispute {dispute.id} submitted by {disputer_id} for verification {verification_id}")
    return dispute


def resolve_dispute(
    dispute_id: UUID,
    outcome: DisputeOutcome,
    resolution_reasoning: str,
    resolution_method: ResolutionMethod = ResolutionMethod.AUTOMATIC,
) -> Dispute:
    """Resolve a dispute.

    Raises:
        NotFoundError: If dispute doesn't exist
        ValidationException: If dispute is not pending
    """
    dispute = get_dispute(dispute_id)
    if not dispute:
        raise NotFoundError("Dispute", str(dispute_id))

    if dispute.status != DisputeStatus.PENDING:
        raise ValidationException(f"Dispute is not pending: {dispute.status.value}")

    verification = get_verification(dispute.verification_id)
    if not verification:
        raise NotFoundError("Verification", str(dispute.verification_id))

    dispute.outcome = outcome
    dispute.resolution_reasoning = resolution_reasoning
    dispute.resolution_method = resolution_method
    dispute.status = DisputeStatus.RESOLVED
    dispute.resolved_at = datetime.now()

    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE disputes
            SET outcome = %s, resolution_reasoning = %s, resolution_method = %s,
                status = %s, resolved_at = %s
            WHERE id = %s
            """,
            (
                dispute.outcome.value,
                dispute.resolution_reasoning,
                dispute.resolution_method.value,
                dispute.status.value,
                dispute.resolved_at,
                str(dispute_id),
            ),
        )

    _process_dispute_resolution(dispute, verification)

    logger.info(f"Dispute {dispute_id} resolved with outcome {outcome.value}")
    return dispute


def get_dispute(dispute_id: UUID) -> Dispute | None:
    """Get a dispute by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM disputes WHERE id = %s", (str(dispute_id),))
        row = cur.fetchone()
        return Dispute.from_row(row) if row else None


def get_disputes_for_verification(verification_id: UUID) -> list[Dispute]:
    """Get all disputes for a verification."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM disputes WHERE verification_id = %s ORDER BY created_at DESC",
            (str(verification_id),),
        )
        return [Dispute.from_row(row) for row in cur.fetchall()]


# ============================================================================
# Stake Operations
# ============================================================================


def _db_lock_stake(
    cur: Any,
    reputation: ReputationScore,
    amount: float,
    verification_id: UUID | None = None,
    dispute_id: UUID | None = None,
) -> StakePosition:
    """Lock reputation as stake (within existing cursor/transaction)."""
    stake_type = StakeType.CHALLENGE if dispute_id else StakeType.STANDARD
    lockup_days = ReputationConstants.STAKE_LOCKUP_DAYS

    position = StakePosition(
        id=uuid4(),
        identity_id=reputation.identity_id,
        amount=amount,
        type=stake_type,
        verification_id=verification_id,
        dispute_id=dispute_id,
        unlocks_at=datetime.now() + timedelta(days=lockup_days),
    )

    cur.execute(
        """
        INSERT INTO stake_positions (id, identity_id, amount, type, verification_id, dispute_id, locked_at, unlocks_at, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            str(position.id),
            position.identity_id,
            position.amount,
            position.type.value,
            str(position.verification_id) if position.verification_id else None,
            str(position.dispute_id) if position.dispute_id else None,
            position.locked_at,
            position.unlocks_at,
            position.status,
        ),
    )

    cur.execute(
        "UPDATE reputations SET stake_at_risk = stake_at_risk + %s, modified_at = NOW() WHERE identity_id = %s",
        (amount, reputation.identity_id),
    )

    reputation.stake_at_risk += amount
    return position


def _release_stake(
    identity_id: str,
    verification_id: UUID | None = None,
    dispute_id: UUID | None = None,
    forfeit: bool = False,
) -> float:
    """Release or forfeit staked reputation."""
    new_status = "forfeited" if forfeit else "returned"
    amount_released = 0.0

    with get_cursor() as cur:
        # Find matching positions
        conditions = ["identity_id = %s", "status = 'locked'"]
        params: list[Any] = [identity_id]

        if verification_id:
            conditions.append("verification_id = %s")
            params.append(str(verification_id))
        if dispute_id:
            conditions.append("dispute_id = %s")
            params.append(str(dispute_id))

        where = " AND ".join(conditions)
        cur.execute(f"SELECT id, amount FROM stake_positions WHERE {where}", params)
        positions = cur.fetchall()

        for pos in positions:
            cur.execute(
                "UPDATE stake_positions SET status = %s WHERE id = %s",
                (new_status, str(pos["id"])),
            )
            amount_released += float(pos["amount"])

        if amount_released > 0:
            cur.execute(
                "UPDATE reputations SET stake_at_risk = stake_at_risk - %s, modified_at = NOW() WHERE identity_id = %s",
                (amount_released, identity_id),
            )

    return amount_released


# ============================================================================
# Reputation Update Processing
# ============================================================================


def _apply_reputation_update(
    identity_id: str,
    delta: float,
    reason: str,
    verification_id: UUID | None = None,
    dispute_id: UUID | None = None,
    dimension: str = "overall",
) -> ReputationUpdate:
    """Apply a reputation update and log it."""
    rep = get_or_create_reputation(identity_id)
    old_value = rep.overall
    new_value = max(ReputationConstants.REPUTATION_FLOOR, min(1.0, old_value + delta))

    event = ReputationUpdate(
        id=uuid4(),
        identity_id=identity_id,
        delta=delta,
        old_value=old_value,
        new_value=new_value,
        reason=reason,
        dimension=dimension,
        verification_id=verification_id,
        dispute_id=dispute_id,
    )

    with get_cursor() as cur:
        cur.execute(
            "UPDATE reputations SET overall = %s, modified_at = NOW() WHERE identity_id = %s",
            (new_value, identity_id),
        )

        if dimension != "overall":
            by_domain = rep.by_domain
            if dimension in by_domain:
                by_domain[dimension] = max(0.1, min(1.0, by_domain[dimension] + delta))
                cur.execute(
                    "UPDATE reputations SET by_domain = %s WHERE identity_id = %s",
                    (json.dumps(by_domain), identity_id),
                )

        cur.execute(
            """
            INSERT INTO reputation_events (id, identity_id, delta, old_value, new_value, reason, dimension, verification_id, dispute_id, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(event.id),
                identity_id,
                delta,
                old_value,
                new_value,
                reason,
                dimension,
                str(verification_id) if verification_id else None,
                str(dispute_id) if dispute_id else None,
                event.created_at,
            ),
        )

    logger.info(f"Reputation update for {identity_id}: {old_value:.4f} -> {new_value:.4f} ({delta:+.4f}) - {reason}")
    return event


def _process_verification_reputation(verification: Verification) -> None:
    """Process reputation updates for an accepted verification."""
    verifier_rep = get_or_create_reputation(verification.verifier_id)

    # Get belief confidence from DB
    belief_confidence = 0.7
    with get_cursor() as cur:
        cur.execute("SELECT confidence FROM beliefs WHERE id = %s", (str(verification.belief_id),))
        row = cur.fetchone()
        if row and row.get("confidence"):
            conf = row["confidence"]
            if isinstance(conf, str):
                conf = json.loads(conf)
            belief_confidence = float(conf.get("overall", 0.7))

    existing = get_verifications_for_belief(verification.belief_id)
    existing_confirmations = len([v for v in existing if v.result == VerificationResult.CONFIRMED and v.id != verification.id])
    existing_contradictions = len([v for v in existing if v.result == VerificationResult.CONTRADICTED and v.id != verification.id])

    min_stake = calculate_min_stake(belief_confidence, 0.5)

    verifier_delta = 0.0
    holder_delta = 0.0

    if verification.result == VerificationResult.CONFIRMED:
        verifier_delta = calculate_confirmation_reward(verification.stake.amount, min_stake, belief_confidence, existing_confirmations)
        holder_delta = calculate_holder_confirmation_bonus(verifier_rep.overall, verification.stake.amount, min_stake)

    elif verification.result == VerificationResult.CONTRADICTED:
        is_first = existing_contradictions == 0
        verifier_delta = calculate_contradiction_reward(verification.stake.amount, min_stake, belief_confidence, is_first, existing_contradictions)
        holder_delta = -calculate_holder_contradiction_penalty(belief_confidence, verifier_rep.overall)

        with get_cursor() as cur:
            cur.execute(
                "UPDATE reputations SET discrepancy_finds = discrepancy_finds + 1 WHERE identity_id = %s",
                (verification.verifier_id,),
            )

    elif verification.result == VerificationResult.UNCERTAIN:
        verifier_delta = ReputationConstants.UNCERTAINTY_BASE

    elif verification.result == VerificationResult.PARTIAL:
        accuracy = 0.5
        if verification.result_details and verification.result_details.accuracy_estimate is not None:
            accuracy = verification.result_details.accuracy_estimate

        verifier_delta = calculate_partial_reward(accuracy, verification.stake.amount, min_stake, belief_confidence, existing_confirmations, existing_contradictions)

        confirm_bonus = calculate_holder_confirmation_bonus(verifier_rep.overall, verification.stake.amount, min_stake)
        contradict_penalty = calculate_holder_contradiction_penalty(belief_confidence, verifier_rep.overall)
        holder_delta = confirm_bonus * accuracy - contradict_penalty * (1 - accuracy)

    _apply_reputation_update(verification.verifier_id, verifier_delta, f"Verification {verification.result.value}", verification.id)
    if holder_delta != 0:
        _apply_reputation_update(verification.holder_id, holder_delta, f"Belief {verification.result.value}", verification.id)

    with get_cursor() as cur:
        cur.execute(
            "UPDATE reputations SET verification_count = verification_count + 1, modified_at = NOW() WHERE identity_id = %s",
            (verification.verifier_id,),
        )


def _process_dispute_resolution(dispute: Dispute, verification: Verification) -> None:
    """Process reputation updates after dispute resolution."""
    if dispute.outcome == DisputeOutcome.UPHELD:
        with get_cursor() as cur:
            cur.execute("UPDATE verifications SET status = %s WHERE id = %s", (VerificationStatus.ACCEPTED.value, str(verification.id)))

        bonus = dispute.stake.amount * 0.8
        _apply_reputation_update(verification.verifier_id, bonus, "Dispute upheld - defense bonus", dispute_id=dispute.id)
        _release_stake(dispute.disputer_id, dispute_id=dispute.id, forfeit=True)
        _apply_reputation_update(dispute.disputer_id, -dispute.stake.amount, "Dispute lost - stake forfeited", dispute_id=dispute.id)

    elif dispute.outcome == DisputeOutcome.OVERTURNED:
        with get_cursor() as cur:
            cur.execute("UPDATE verifications SET status = %s WHERE id = %s", (VerificationStatus.OVERTURNED.value, str(verification.id)))

        _release_stake(verification.verifier_id, verification_id=verification.id, forfeit=True)
        _apply_reputation_update(verification.verifier_id, -verification.stake.amount, "Verification overturned - stake forfeited", dispute_id=dispute.id)

        reward = verification.stake.amount * 0.8
        _apply_reputation_update(dispute.disputer_id, reward, "Dispute won - reward", dispute_id=dispute.id)
        _release_stake(dispute.disputer_id, dispute_id=dispute.id)

        if verification.result == VerificationResult.CONTRADICTED:
            verifier_rep = get_or_create_reputation(verification.verifier_id)
            restore_amount = calculate_holder_contradiction_penalty(0.7, verifier_rep.overall)
            _apply_reputation_update(verification.holder_id, restore_amount, "Contradiction overturned - reputation restored", dispute_id=dispute.id)

    elif dispute.outcome == DisputeOutcome.MODIFIED:
        with get_cursor() as cur:
            cur.execute("UPDATE verifications SET status = %s WHERE id = %s", (VerificationStatus.ACCEPTED.value, str(verification.id)))
        _release_stake(verification.verifier_id, verification_id=verification.id)
        _release_stake(dispute.disputer_id, dispute_id=dispute.id)

    elif dispute.outcome == DisputeOutcome.DISMISSED:
        with get_cursor() as cur:
            cur.execute("UPDATE verifications SET status = %s WHERE id = %s", (VerificationStatus.ACCEPTED.value, str(verification.id)))

        compensation = dispute.stake.amount * 0.5
        _apply_reputation_update(verification.verifier_id, compensation, "Frivolous dispute - compensation", dispute_id=dispute.id)

        _release_stake(dispute.disputer_id, dispute_id=dispute.id, forfeit=True)
        penalty = dispute.stake.amount * (1 + ReputationConstants.FRIVOLOUS_DISPUTE_PENALTY)
        _apply_reputation_update(dispute.disputer_id, -penalty, "Frivolous dispute - stake + penalty", dispute_id=dispute.id)


# ============================================================================
# Bounty Operations
# ============================================================================


def get_bounty(belief_id: UUID) -> DiscrepancyBounty | None:
    """Get the discrepancy bounty for a belief."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM discrepancy_bounties WHERE belief_id = %s", (str(belief_id),))
        row = cur.fetchone()
        if not row:
            return None

        return DiscrepancyBounty(
            belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
            holder_id=row["holder_id"],
            base_amount=float(row["base_amount"]),
            confidence_premium=float(row["confidence_premium"]),
            age_factor=float(row["age_factor"]),
            total_bounty=float(row["total_bounty"]),
            created_at=row["created_at"],
            expires_at=row.get("expires_at"),
            claimed=row["claimed"],
            claimed_by=row.get("claimed_by"),
            claimed_at=row.get("claimed_at"),
        )


def list_bounties(unclaimed_only: bool = True, limit: int = 20) -> list[DiscrepancyBounty]:
    """List discrepancy bounties."""
    with get_cursor() as cur:
        where = "WHERE NOT claimed" if unclaimed_only else ""
        cur.execute(
            f"SELECT * FROM discrepancy_bounties {where} ORDER BY total_bounty DESC LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
        return [
            DiscrepancyBounty(
                belief_id=row["belief_id"] if isinstance(row["belief_id"], UUID) else UUID(row["belief_id"]),
                holder_id=row["holder_id"],
                base_amount=float(row["base_amount"]),
                confidence_premium=float(row["confidence_premium"]),
                age_factor=float(row["age_factor"]),
                total_bounty=float(row["total_bounty"]),
                created_at=row["created_at"],
                expires_at=row.get("expires_at"),
                claimed=row["claimed"],
                claimed_by=row.get("claimed_by"),
                claimed_at=row.get("claimed_at"),
            )
            for row in rows
        ]


# ============================================================================
# Query Operations
# ============================================================================


def get_reputation_events(
    identity_id: str,
    limit: int = 50,
) -> list[ReputationUpdate]:
    """Get reputation event history for an identity."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM reputation_events
            WHERE identity_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (identity_id, limit),
        )
        rows = cur.fetchall()
        return [
            ReputationUpdate(
                id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
                identity_id=row["identity_id"],
                delta=float(row["delta"]),
                old_value=float(row["old_value"]),
                new_value=float(row["new_value"]),
                reason=row["reason"],
                dimension=row.get("dimension", "overall"),
                verification_id=row.get("verification_id"),
                dispute_id=row.get("dispute_id"),
                created_at=row["created_at"],
            )
            for row in rows
        ]
