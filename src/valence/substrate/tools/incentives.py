"""Incentive system MCP tool implementations.

Handlers for calibration, rewards, transfers, and velocity status.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any
from uuid import UUID

from valence.core.exceptions import NotFoundError, ValidationException
from valence.core.incentives import (
    claim_all_rewards,
    claim_reward,
    get_calibration_history,
    get_pending_rewards,
    get_transfers,
    get_velocity_status,
    run_calibration_snapshot,
)
from valence.core.verification.constants import ReputationConstants

logger = logging.getLogger(__name__)


def _parse_uuid(value: str, name: str) -> UUID | dict[str, Any]:
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return {"success": False, "error": f"Invalid UUID for {name}: {value}"}


# ============================================================================
# Calibration Tools
# ============================================================================


def calibration_run(identity_id: str, period_start: str | None = None, **_: Any) -> dict[str, Any]:
    """Run calibration scoring for an identity."""
    ps = None
    if period_start:
        try:
            ps = date.fromisoformat(period_start)
        except ValueError:
            return {"success": False, "error": f"Invalid date format: {period_start}. Use YYYY-MM-DD."}

    snapshot = run_calibration_snapshot(identity_id, ps)
    if not snapshot:
        return {"success": True, "message": "No verified beliefs found for this period", "snapshot": None}

    return {
        "success": True,
        "snapshot": {
            "identity_id": snapshot.identity_id,
            "period_start": snapshot.period_start.isoformat(),
            "period_end": snapshot.period_end.isoformat(),
            "brier_score": snapshot.brier_score,
            "sample_size": snapshot.sample_size,
            "reward_earned": snapshot.reward_earned,
            "penalty_applied": snapshot.penalty_applied,
            "consecutive_well_calibrated": snapshot.consecutive_well_calibrated,
        },
    }


def calibration_history(identity_id: str, limit: int = 12, **_: Any) -> dict[str, Any]:
    """Get calibration history for an identity."""
    snapshots = get_calibration_history(identity_id, limit=limit)
    return {
        "success": True,
        "snapshots": [
            {
                "period_start": s.period_start.isoformat(),
                "brier_score": s.brier_score,
                "sample_size": s.sample_size,
                "reward_earned": s.reward_earned,
                "penalty_applied": s.penalty_applied,
                "consecutive_well_calibrated": s.consecutive_well_calibrated,
            }
            for s in snapshots
        ],
        "total": len(snapshots),
    }


# ============================================================================
# Reward Tools
# ============================================================================


def rewards_pending(identity_id: str, **_: Any) -> dict[str, Any]:
    """Get pending rewards for an identity."""
    rewards = get_pending_rewards(identity_id)
    total_amount = sum(r.amount for r in rewards)
    return {
        "success": True,
        "rewards": [
            {
                "id": str(r.id),
                "amount": r.amount,
                "reward_type": r.reward_type,
                "reason": r.reason,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "expires_at": r.expires_at.isoformat() if r.expires_at else None,
            }
            for r in rewards
        ],
        "total_pending": total_amount,
        "count": len(rewards),
    }


def reward_claim(reward_id: str, **_: Any) -> dict[str, Any]:
    """Claim a single pending reward."""
    rid = _parse_uuid(reward_id, "reward_id")
    if isinstance(rid, dict):
        return rid

    try:
        reward = claim_reward(rid)
    except (NotFoundError, ValidationException) as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "reward": {
            "id": str(reward.id),
            "amount": reward.amount,
            "reward_type": reward.reward_type,
            "status": reward.status,
            "claimed_at": reward.claimed_at.isoformat() if reward.claimed_at else None,
        },
    }


def rewards_claim_all(identity_id: str, **_: Any) -> dict[str, Any]:
    """Claim all pending rewards for an identity."""
    claimed = claim_all_rewards(identity_id)
    total = sum(r.amount for r in claimed)
    return {
        "success": True,
        "claimed": [
            {
                "id": str(r.id),
                "amount": r.amount,
                "reward_type": r.reward_type,
            }
            for r in claimed
        ],
        "total_claimed": total,
        "count": len(claimed),
    }


# ============================================================================
# Transfer Tools
# ============================================================================


def transfer_history(identity_id: str, direction: str = "both", limit: int = 50, **_: Any) -> dict[str, Any]:
    """Get transfer history for an identity."""
    if direction not in ("both", "incoming", "outgoing"):
        return {"success": False, "error": f"Invalid direction '{direction}'. Must be: both, incoming, outgoing"}

    transfers = get_transfers(identity_id, direction=direction, limit=limit)
    return {
        "success": True,
        "transfers": [
            {
                "id": str(t.id),
                "from": t.from_identity_id,
                "to": t.to_identity_id,
                "amount": t.amount,
                "transfer_type": t.transfer_type,
                "reason": t.reason,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            }
            for t in transfers
        ],
        "total": len(transfers),
    }


# ============================================================================
# Velocity Tools
# ============================================================================


def velocity_status(identity_id: str, **_: Any) -> dict[str, Any]:
    """Get velocity status for an identity."""
    status = get_velocity_status(identity_id)
    return {
        "success": True,
        "velocity": {
            "identity_id": status.identity_id,
            "daily_gain": status.daily_gain,
            "daily_verifications": status.daily_verifications,
            "weekly_gain": status.weekly_gain,
            "daily_remaining": status.daily_remaining,
            "weekly_remaining": status.weekly_remaining,
            "daily_limit": ReputationConstants.MAX_DAILY_GAIN,
            "weekly_limit": ReputationConstants.MAX_WEEKLY_GAIN,
        },
    }
