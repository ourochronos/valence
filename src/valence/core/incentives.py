"""Incentive system — calibration, rewards, transfers, and velocity limits.

Builds on top of the verification protocol's reputation and stake infrastructure.
This module adds:
- Calibration scoring (Brier score) with monthly snapshots
- Reward creation and claiming
- System-initiated reputation transfers
- Velocity limit enforcement (daily/weekly gain caps)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from our_db import get_cursor

from .exceptions import NotFoundError, ValidationException
from .verification.constants import ReputationConstants
from .verification.enums import VerificationResult

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Reward:
    """An earned but potentially unclaimed reputation reward."""

    id: UUID
    identity_id: str
    amount: float
    reward_type: str
    source_id: UUID | None = None
    reason: str = ""
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    claimed_at: datetime | None = None
    expires_at: datetime | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Reward:
        return cls(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            identity_id=row["identity_id"],
            amount=float(row["amount"]),
            reward_type=row["reward_type"],
            source_id=row.get("source_id"),
            reason=row.get("reason", ""),
            status=row.get("status", "pending"),
            created_at=row.get("created_at", datetime.now()),
            claimed_at=row.get("claimed_at"),
            expires_at=row.get("expires_at"),
        )


@dataclass
class Transfer:
    """A system-initiated reputation movement between identities."""

    id: UUID
    from_identity_id: str
    to_identity_id: str
    amount: float
    transfer_type: str
    source_id: UUID | None = None
    reason: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class CalibrationSnapshot:
    """Monthly calibration score for an identity."""

    id: UUID
    identity_id: str
    period_start: date
    period_end: date
    brier_score: float
    sample_size: int = 0
    reward_earned: float = 0.0
    penalty_applied: float = 0.0
    consecutive_well_calibrated: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class VelocityStatus:
    """Current velocity tracking for gain caps."""

    identity_id: str
    daily_gain: float = 0.0
    daily_verifications: int = 0
    weekly_gain: float = 0.0
    daily_remaining: float = 0.0
    weekly_remaining: float = 0.0


# ============================================================================
# Calibration
# ============================================================================


def calculate_brier_score(identity_id: str, period_start: date, period_end: date) -> tuple[float, int]:
    """Calculate Brier score for an identity over a period.

    Brier score = 1 - mean(|claimed_confidence - actual_outcome|²)

    Where actual_outcome:
      1.0 if belief CONFIRMED
      0.0 if belief CONTRADICTED
      claimed_confidence if UNCERTAIN

    Returns:
        (brier_score, sample_size)
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT v.result, b.confidence
            FROM verifications v
            JOIN beliefs b ON b.id = v.belief_id
            WHERE v.verifier_id = %s
              AND v.status = 'accepted'
              AND v.accepted_at >= %s
              AND v.accepted_at < %s
            """,
            (identity_id, period_start, period_end),
        )
        rows = cur.fetchall()

    if not rows:
        return 0.0, 0

    squared_errors = []
    for row in rows:
        conf = row.get("confidence", {})
        if isinstance(conf, str):
            conf = json.loads(conf)
        claimed = float(conf.get("overall", 0.5))

        result = row["result"]
        if result == VerificationResult.CONFIRMED.value:
            actual = 1.0
        elif result == VerificationResult.CONTRADICTED.value:
            actual = 0.0
        else:
            actual = claimed

        squared_errors.append((claimed - actual) ** 2)

    mean_error = sum(squared_errors) / len(squared_errors)
    brier_score = max(0.0, min(1.0, 1.0 - mean_error))
    return brier_score, len(squared_errors)


def calculate_calibration_reward(
    brier_score: float,
    sample_size: int,
    consecutive_months: int,
) -> float:
    """Calculate calibration reward from spec formula.

    calibration_reward = BASE × score × volume_factor × consistency_bonus

    Requirements:
    - Minimum 50 verified beliefs in period
    - Score > 0.5 to earn reward
    - Score < 0.4 triggers penalty
    """
    if sample_size < 50:
        return 0.0

    if brier_score < ReputationConstants.CALIBRATION_PENALTY_THRESHOLD:
        return -(ReputationConstants.CALIBRATION_BONUS_BASE * 0.5)

    if brier_score <= 0.5:
        return 0.0

    volume_factor = min(1.0, sample_size / 50)
    consistency_bonus = min(1.5, 1.0 + consecutive_months * 0.05)

    return ReputationConstants.CALIBRATION_BONUS_BASE * brier_score * volume_factor * consistency_bonus


def run_calibration_snapshot(identity_id: str, period_start: date | None = None) -> CalibrationSnapshot | None:
    """Calculate and store a calibration snapshot for a monthly period.

    Returns None if insufficient data (< 50 samples).
    """
    if period_start is None:
        today = date.today()
        period_start = today.replace(day=1) - timedelta(days=1)
        period_start = period_start.replace(day=1)

    period_end = (period_start + timedelta(days=32)).replace(day=1)

    brier_score, sample_size = calculate_brier_score(identity_id, period_start, period_end)

    if sample_size == 0:
        return None

    # Get consecutive well-calibrated months
    consecutive = 0
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT consecutive_well_calibrated FROM calibration_snapshots
            WHERE identity_id = %s
            ORDER BY period_start DESC LIMIT 1
            """,
            (identity_id,),
        )
        row = cur.fetchone()
        if row:
            consecutive = row["consecutive_well_calibrated"]

    if brier_score > 0.5:
        consecutive += 1
    else:
        consecutive = 0

    reward = calculate_calibration_reward(brier_score, sample_size, consecutive)

    snapshot = CalibrationSnapshot(
        id=uuid4(),
        identity_id=identity_id,
        period_start=period_start,
        period_end=period_end,
        brier_score=brier_score,
        sample_size=sample_size,
        reward_earned=max(0.0, reward),
        penalty_applied=abs(min(0.0, reward)),
        consecutive_well_calibrated=consecutive,
    )

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO calibration_snapshots (
                id, identity_id, period_start, period_end, brier_score,
                sample_size, reward_earned, penalty_applied,
                consecutive_well_calibrated, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (identity_id, period_start) DO UPDATE SET
                brier_score = EXCLUDED.brier_score,
                sample_size = EXCLUDED.sample_size,
                reward_earned = EXCLUDED.reward_earned,
                penalty_applied = EXCLUDED.penalty_applied,
                consecutive_well_calibrated = EXCLUDED.consecutive_well_calibrated
            """,
            (
                str(snapshot.id), identity_id, period_start, period_end,
                snapshot.brier_score, snapshot.sample_size,
                snapshot.reward_earned, snapshot.penalty_applied,
                snapshot.consecutive_well_calibrated, snapshot.created_at,
            ),
        )

    # Create reward if positive
    if reward > 0:
        create_reward(identity_id, reward, "calibration", reason=f"Calibration bonus (Brier: {brier_score:.3f}, n={sample_size})")
    elif reward < 0:
        # Apply penalty via reputation update
        from .verification.db import _apply_reputation_update
        _apply_reputation_update(identity_id, reward, f"Calibration penalty (Brier: {brier_score:.3f})")

    return snapshot


def get_calibration_history(identity_id: str, limit: int = 12) -> list[CalibrationSnapshot]:
    """Get calibration snapshot history for an identity."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM calibration_snapshots
            WHERE identity_id = %s
            ORDER BY period_start DESC
            LIMIT %s
            """,
            (identity_id, limit),
        )
        return [
            CalibrationSnapshot(
                id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
                identity_id=row["identity_id"],
                period_start=row["period_start"],
                period_end=row["period_end"],
                brier_score=float(row["brier_score"]),
                sample_size=row["sample_size"],
                reward_earned=float(row["reward_earned"]),
                penalty_applied=float(row["penalty_applied"]),
                consecutive_well_calibrated=row["consecutive_well_calibrated"],
                created_at=row["created_at"],
            )
            for row in cur.fetchall()
        ]


# ============================================================================
# Rewards
# ============================================================================


def create_reward(
    identity_id: str,
    amount: float,
    reward_type: str,
    source_id: UUID | None = None,
    reason: str = "",
    expires_in_days: int | None = None,
) -> Reward:
    """Create a pending reward for an identity."""
    reward = Reward(
        id=uuid4(),
        identity_id=identity_id,
        amount=amount,
        reward_type=reward_type,
        source_id=source_id,
        reason=reason,
        expires_at=datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None,
    )

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO rewards (id, identity_id, amount, reward_type, source_id, reason, status, created_at, expires_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(reward.id), reward.identity_id, reward.amount,
                reward.reward_type, str(reward.source_id) if reward.source_id else None,
                reward.reason, reward.status, reward.created_at, reward.expires_at,
            ),
        )

    logger.info(f"Reward created: {reward.amount:.4f} for {identity_id} ({reward_type})")
    return reward


def get_pending_rewards(identity_id: str) -> list[Reward]:
    """Get all pending (unclaimed) rewards for an identity."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT * FROM rewards
            WHERE identity_id = %s AND status = 'pending'
              AND (expires_at IS NULL OR expires_at > NOW())
            ORDER BY created_at DESC
            """,
            (identity_id,),
        )
        return [Reward.from_row(row) for row in cur.fetchall()]


def claim_reward(reward_id: UUID) -> Reward:
    """Claim a pending reward, applying it to reputation.

    Raises:
        NotFoundError: If reward doesn't exist
        ValidationException: If reward is not claimable
    """
    with get_cursor() as cur:
        cur.execute("SELECT * FROM rewards WHERE id = %s", (str(reward_id),))
        row = cur.fetchone()
        if not row:
            raise NotFoundError("Reward", str(reward_id))

        reward = Reward.from_row(row)

        if reward.status != "pending":
            raise ValidationException(f"Reward is not pending: {reward.status}")
        if reward.expires_at and reward.expires_at < datetime.now():
            raise ValidationException("Reward has expired")

    # Check velocity limits before applying
    if not check_velocity_limit(reward.identity_id, reward.amount):
        raise ValidationException("Claiming this reward would exceed daily/weekly gain limits")

    # Apply to reputation
    from .verification.db import _apply_reputation_update
    _apply_reputation_update(
        reward.identity_id,
        reward.amount,
        f"Reward claimed: {reward.reason}",
    )

    # Update velocity tracking
    update_velocity(reward.identity_id, reward.amount)

    # Mark as claimed
    reward.status = "claimed"
    reward.claimed_at = datetime.now()

    with get_cursor() as cur:
        cur.execute(
            "UPDATE rewards SET status = %s, claimed_at = %s WHERE id = %s",
            (reward.status, reward.claimed_at, str(reward_id)),
        )

    logger.info(f"Reward {reward_id} claimed by {reward.identity_id}: {reward.amount:.4f}")
    return reward


def claim_all_rewards(identity_id: str) -> list[Reward]:
    """Claim all pending rewards for an identity (respecting velocity limits)."""
    pending = get_pending_rewards(identity_id)
    claimed = []

    for reward in pending:
        try:
            claimed_reward = claim_reward(reward.id)
            claimed.append(claimed_reward)
        except ValidationException:
            break  # Hit velocity limit, stop claiming

    return claimed


# ============================================================================
# Transfers
# ============================================================================


def record_transfer(
    from_id: str,
    to_id: str,
    amount: float,
    transfer_type: str,
    source_id: UUID | None = None,
    reason: str = "",
) -> Transfer:
    """Record a system-initiated reputation transfer."""
    transfer = Transfer(
        id=uuid4(),
        from_identity_id=from_id,
        to_identity_id=to_id,
        amount=amount,
        transfer_type=transfer_type,
        source_id=source_id,
        reason=reason,
    )

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO transfers (id, from_identity_id, to_identity_id, amount, transfer_type, source_id, reason, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(transfer.id), transfer.from_identity_id, transfer.to_identity_id,
                transfer.amount, transfer.transfer_type,
                str(transfer.source_id) if transfer.source_id else None,
                transfer.reason, transfer.created_at,
            ),
        )

    logger.info(f"Transfer {transfer.id}: {amount:.4f} from {from_id} to {to_id} ({transfer_type})")
    return transfer


def get_transfers(identity_id: str, direction: str = "both", limit: int = 50) -> list[Transfer]:
    """Get transfer history for an identity."""
    with get_cursor() as cur:
        if direction == "outgoing":
            where = "from_identity_id = %s"
        elif direction == "incoming":
            where = "to_identity_id = %s"
        else:
            where = "(from_identity_id = %s OR to_identity_id = %s)"

        params: list[Any] = [identity_id]
        if direction == "both":
            params.append(identity_id)
        params.append(limit)

        cur.execute(
            f"SELECT * FROM transfers WHERE {where} ORDER BY created_at DESC LIMIT %s",
            params,
        )
        return [
            Transfer(
                id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
                from_identity_id=row["from_identity_id"],
                to_identity_id=row["to_identity_id"],
                amount=float(row["amount"]),
                transfer_type=row["transfer_type"],
                source_id=row.get("source_id"),
                reason=row.get("reason", ""),
                created_at=row["created_at"],
            )
            for row in cur.fetchall()
        ]


# ============================================================================
# Velocity Limits
# ============================================================================


def check_velocity_limit(identity_id: str, proposed_gain: float) -> bool:
    """Check if a proposed gain would exceed velocity limits.

    Returns True if the gain is within limits, False otherwise.
    """
    today = date.today()
    week_start = today - timedelta(days=today.weekday())

    with get_cursor() as cur:
        # Check daily limit
        cur.execute(
            """
            SELECT COALESCE(total_gain, 0) as total_gain
            FROM velocity_tracking
            WHERE identity_id = %s AND period_type = 'daily' AND period_start = %s
            """,
            (identity_id, today),
        )
        row = cur.fetchone()
        daily_gain = float(row["total_gain"]) if row else 0.0

        if daily_gain + proposed_gain > ReputationConstants.MAX_DAILY_GAIN:
            return False

        # Check weekly limit
        cur.execute(
            """
            SELECT COALESCE(SUM(total_gain), 0) as total_gain
            FROM velocity_tracking
            WHERE identity_id = %s AND period_type = 'daily'
              AND period_start >= %s AND period_start <= %s
            """,
            (identity_id, week_start, today),
        )
        row = cur.fetchone()
        weekly_gain = float(row["total_gain"]) if row else 0.0

        if weekly_gain + proposed_gain > ReputationConstants.MAX_WEEKLY_GAIN:
            return False

    return True


def update_velocity(identity_id: str, gain: float) -> None:
    """Update velocity tracking after a gain is applied."""
    today = date.today()

    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO velocity_tracking (identity_id, period_type, period_start, total_gain, verification_count)
            VALUES (%s, 'daily', %s, %s, 0)
            ON CONFLICT (identity_id, period_type, period_start) DO UPDATE
            SET total_gain = velocity_tracking.total_gain + %s
            """,
            (identity_id, today, gain, gain),
        )


def get_velocity_status(identity_id: str) -> VelocityStatus:
    """Get current velocity status for an identity."""
    today = date.today()
    week_start = today - timedelta(days=today.weekday())

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(total_gain, 0) as total_gain, COALESCE(verification_count, 0) as verification_count
            FROM velocity_tracking
            WHERE identity_id = %s AND period_type = 'daily' AND period_start = %s
            """,
            (identity_id, today),
        )
        row = cur.fetchone()
        daily_gain = float(row["total_gain"]) if row else 0.0
        daily_verifications = int(row["verification_count"]) if row else 0

        cur.execute(
            """
            SELECT COALESCE(SUM(total_gain), 0) as total_gain
            FROM velocity_tracking
            WHERE identity_id = %s AND period_type = 'daily'
              AND period_start >= %s AND period_start <= %s
            """,
            (identity_id, week_start, today),
        )
        row = cur.fetchone()
        weekly_gain = float(row["total_gain"]) if row else 0.0

    return VelocityStatus(
        identity_id=identity_id,
        daily_gain=daily_gain,
        daily_verifications=daily_verifications,
        weekly_gain=weekly_gain,
        daily_remaining=max(0, ReputationConstants.MAX_DAILY_GAIN - daily_gain),
        weekly_remaining=max(0, ReputationConstants.MAX_WEEKLY_GAIN - weekly_gain),
    )
