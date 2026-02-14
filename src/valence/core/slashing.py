"""Slashing mechanism for validator stake forfeiture (#346).

Wires our-consensus anti_gaming detection to stake forfeiture in valence.
When collusion or other slashable offenses are detected, this module:
1. Creates a slashing event record
2. Locks the offender's stake
3. Provides a 24-hour appeal window
4. Executes the slash (forfeiture) after appeal expires

Slash percentages (from issue spec):
- CRITICAL severity: 50% max per event
- HIGH severity: 25% max per event
- Medium/Low: warning only, no slash
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# Maximum slash percentages by severity
SLASH_PERCENT_CRITICAL = 0.50
SLASH_PERCENT_HIGH = 0.25
APPEAL_WINDOW_HOURS = 24


class SlashingStatus(StrEnum):
    PENDING = "pending"
    APPEALED = "appealed"
    EXECUTED = "executed"
    REJECTED = "rejected"


class SlashableOffense(StrEnum):
    FALSE_BELIEFS = "false_beliefs"  # 3+ contradictions from same source
    POLICY_VIOLATION = "policy_violation"  # Sharing policy breach
    REPLAY_ATTACK = "replay_attack"  # Replaying old/superseded beliefs
    COLLUSION = "collusion"  # Coordinated manipulation (from anti_gaming)
    SYBIL_ATTACK = "sybil_attack"  # VDF bypass attempt


@dataclass
class SlashingEvent:
    """Record of a slashing action."""

    id: str
    validator_did: str
    offense: SlashableOffense
    severity: str  # "high" or "critical"
    evidence: dict[str, Any]
    stake_at_risk: float
    slash_amount: float
    status: SlashingStatus = SlashingStatus.PENDING
    reported_by: str = ""
    reported_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    appeal_deadline: datetime | None = None
    executed_at: datetime | None = None
    appeal_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "validator_did": self.validator_did,
            "offense": self.offense.value,
            "severity": self.severity,
            "evidence": self.evidence,
            "stake_at_risk": self.stake_at_risk,
            "slash_amount": self.slash_amount,
            "status": self.status.value,
            "reported_by": self.reported_by,
            "reported_at": self.reported_at.isoformat(),
            "appeal_deadline": self.appeal_deadline.isoformat() if self.appeal_deadline else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "appeal_reason": self.appeal_reason,
        }


def create_slashing_event(
    cur,
    validator_did: str,
    offense: SlashableOffense,
    severity: str,
    evidence: dict[str, Any],
    reported_by: str,
    stake_amount: float,
) -> SlashingEvent:
    """Create a new slashing event and lock the offender's stake.

    Args:
        cur: Database cursor.
        validator_did: DID of the offending validator.
        offense: Type of offense.
        severity: "high" or "critical".
        evidence: Evidence dict (from anti_gaming or other detection).
        reported_by: DID of the reporter.
        stake_amount: Current stake amount of the validator.

    Returns:
        SlashingEvent with calculated slash amount and appeal deadline.
    """
    # Calculate slash amount
    if severity == "critical":
        slash_pct = SLASH_PERCENT_CRITICAL
    elif severity == "high":
        slash_pct = SLASH_PERCENT_HIGH
    else:
        # Warning only for lower severities
        logger.info("Severity %s does not trigger slashing for %s", severity, validator_did)
        slash_pct = 0.0

    slash_amount = stake_amount * slash_pct
    now = datetime.now(UTC)
    appeal_deadline = now + timedelta(hours=APPEAL_WINDOW_HOURS)

    event = SlashingEvent(
        id=str(uuid.uuid4()),
        validator_did=validator_did,
        offense=offense,
        severity=severity,
        evidence=evidence,
        stake_at_risk=stake_amount,
        slash_amount=slash_amount,
        reported_by=reported_by,
        reported_at=now,
        appeal_deadline=appeal_deadline,
    )

    # Persist slashing event
    cur.execute(
        """
        INSERT INTO slashing_events (id, validator_did, offense, severity, evidence, stake_at_risk,
                                     slash_amount, status, reported_by, reported_at, appeal_deadline)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            event.id, event.validator_did, event.offense.value, event.severity,
            __import__("json").dumps(event.evidence), event.stake_at_risk,
            event.slash_amount, event.status.value, event.reported_by,
            event.reported_at, event.appeal_deadline,
        ),
    )

    # Lock the offender's active stake
    cur.execute(
        """
        UPDATE stake_positions SET status = 'locked'
        WHERE identity_id = %s AND status = 'active'
        """,
        (validator_did,),
    )

    logger.warning(
        "Slashing event created: %s for %s (%.1f%% of %.2f stake)",
        offense.value, validator_did, slash_pct * 100, stake_amount,
    )

    return event


def appeal_slashing_event(cur, event_id: str, appeal_reason: str) -> SlashingEvent | None:
    """File an appeal for a pending slashing event.

    Must be filed before the appeal deadline.

    Args:
        cur: Database cursor.
        event_id: UUID of the slashing event.
        appeal_reason: Explanation for the appeal.

    Returns:
        Updated SlashingEvent, or None if appeal is invalid.
    """
    cur.execute(
        "SELECT * FROM slashing_events WHERE id = %s",
        (event_id,),
    )
    row = cur.fetchone()
    if not row:
        return None

    if row["status"] != SlashingStatus.PENDING.value:
        logger.warning("Cannot appeal event %s: status is %s", event_id, row["status"])
        return None

    now = datetime.now(UTC)
    deadline = row["appeal_deadline"]
    if deadline and now > deadline:
        logger.warning("Appeal deadline expired for event %s", event_id)
        return None

    cur.execute(
        "UPDATE slashing_events SET status = %s, appeal_reason = %s WHERE id = %s",
        (SlashingStatus.APPEALED.value, appeal_reason, event_id),
    )

    return SlashingEvent(
        id=event_id,
        validator_did=row["validator_did"],
        offense=SlashableOffense(row["offense"]),
        severity=row["severity"],
        evidence=row["evidence"] if isinstance(row["evidence"], dict) else __import__("json").loads(row["evidence"]),
        stake_at_risk=float(row["stake_at_risk"]),
        slash_amount=float(row["slash_amount"]),
        status=SlashingStatus.APPEALED,
        reported_by=row["reported_by"],
        appeal_deadline=deadline,
        appeal_reason=appeal_reason,
    )


def execute_slashing(cur, event_id: str) -> SlashingEvent | None:
    """Execute a slashing event after appeal window expires.

    Forfeits the slash_amount from the offender's stake.

    Args:
        cur: Database cursor.
        event_id: UUID of the slashing event.

    Returns:
        Updated SlashingEvent, or None if execution is invalid.
    """
    cur.execute(
        "SELECT * FROM slashing_events WHERE id = %s",
        (event_id,),
    )
    row = cur.fetchone()
    if not row:
        return None

    if row["status"] not in (SlashingStatus.PENDING.value, SlashingStatus.APPEALED.value):
        logger.warning("Cannot execute event %s: status is %s", event_id, row["status"])
        return None

    now = datetime.now(UTC)

    # Forfeit stake
    cur.execute(
        """
        UPDATE stake_positions SET status = 'forfeited', amount = amount - %s
        WHERE identity_id = %s AND status = 'locked'
        """,
        (float(row["slash_amount"]), row["validator_did"]),
    )

    # Mark event as executed
    cur.execute(
        "UPDATE slashing_events SET status = %s, executed_at = %s WHERE id = %s",
        (SlashingStatus.EXECUTED.value, now, event_id),
    )

    return SlashingEvent(
        id=event_id,
        validator_did=row["validator_did"],
        offense=SlashableOffense(row["offense"]),
        severity=row["severity"],
        evidence=row["evidence"] if isinstance(row["evidence"], dict) else __import__("json").loads(row["evidence"]),
        stake_at_risk=float(row["stake_at_risk"]),
        slash_amount=float(row["slash_amount"]),
        status=SlashingStatus.EXECUTED,
        reported_by=row["reported_by"],
        executed_at=now,
    )


def reject_slashing(cur, event_id: str) -> bool:
    """Reject a slashing event (e.g., after successful appeal review).

    Unlocks the offender's stake.

    Args:
        cur: Database cursor.
        event_id: UUID of the slashing event.

    Returns:
        True if rejected successfully.
    """
    cur.execute("SELECT validator_did, status FROM slashing_events WHERE id = %s", (event_id,))
    row = cur.fetchone()
    if not row:
        return False

    # Unlock stake
    cur.execute(
        "UPDATE stake_positions SET status = 'active' WHERE identity_id = %s AND status = 'locked'",
        (row["validator_did"],),
    )

    cur.execute(
        "UPDATE slashing_events SET status = %s WHERE id = %s",
        (SlashingStatus.REJECTED.value, event_id),
    )

    return True
