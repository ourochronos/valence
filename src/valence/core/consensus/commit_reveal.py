"""Commit-reveal protocol for tamper-resistant corroboration (#353).

Two-phase protocol that prevents validators from seeing others'
corroboration votes before submitting their own:

1. Commit phase: Validator submits H(vote || nonce) — hidden commitment
2. Delay: Configurable wait period (VDF-enforced when available)
3. Reveal phase: Validator reveals vote + nonce, verified against commitment

Late reveals are penalized, no-reveals count as abstentions.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Protocol timing defaults
COMMIT_WINDOW_MINUTES = 60
REVEAL_WINDOW_MINUTES = 60
DELAY_SECONDS = 300  # 5 minutes between commit and reveal


@dataclass
class Commitment:
    """A hidden commitment to a corroboration vote."""

    id: str
    belief_id: str
    committer_did: str
    commitment_hash: str  # SHA256(vote_value || nonce)
    committed_at: datetime
    reveal_window_opens: datetime
    reveal_window_closes: datetime
    status: str = "committed"  # committed, revealed, no_reveal, penalty

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "belief_id": self.belief_id,
            "committer_did": self.committer_did,
            "commitment_hash": self.commitment_hash,
            "committed_at": self.committed_at.isoformat(),
            "reveal_window_opens": self.reveal_window_opens.isoformat(),
            "reveal_window_closes": self.reveal_window_closes.isoformat(),
            "status": self.status,
        }


@dataclass
class Reveal:
    """A revealed vote matching a prior commitment."""

    commitment_id: str
    vote_value: str
    nonce: str
    revealed_at: datetime
    is_valid: bool  # True if hash matches commitment
    is_late: bool  # True if revealed after window

    def to_dict(self) -> dict[str, Any]:
        return {
            "commitment_id": self.commitment_id,
            "vote_value": self.vote_value,
            "nonce": self.nonce,
            "revealed_at": self.revealed_at.isoformat(),
            "is_valid": self.is_valid,
            "is_late": self.is_late,
        }


def generate_nonce() -> str:
    """Generate a cryptographic nonce for commitment."""
    return secrets.token_hex(16)


def compute_commitment_hash(vote_value: str, nonce: str) -> str:
    """Compute the commitment hash H(vote || nonce).

    Args:
        vote_value: The vote (e.g., "corroborate", "dispute", confidence score).
        nonce: The secret nonce.

    Returns:
        Hex-encoded SHA256 hash.
    """
    data = f"{vote_value}:{nonce}".encode()
    return hashlib.sha256(data).hexdigest()


def submit_commitment(
    cur,
    belief_id: str,
    committer_did: str,
    commitment_hash: str,
    delay_seconds: int = DELAY_SECONDS,
    reveal_window_minutes: int = REVEAL_WINDOW_MINUTES,
) -> Commitment:
    """Submit a hidden commitment for a belief corroboration.

    Args:
        cur: Database cursor.
        belief_id: UUID of the belief being corroborated.
        committer_did: DID of the committer.
        commitment_hash: The hidden hash H(vote || nonce).
        delay_seconds: Minimum delay before reveal window opens.
        reveal_window_minutes: Duration of the reveal window.

    Returns:
        Commitment record.
    """
    now = datetime.now(UTC)
    reveal_opens = now + timedelta(seconds=delay_seconds)
    reveal_closes = reveal_opens + timedelta(minutes=reveal_window_minutes)

    commitment = Commitment(
        id=str(uuid4()),
        belief_id=belief_id,
        committer_did=committer_did,
        commitment_hash=commitment_hash,
        committed_at=now,
        reveal_window_opens=reveal_opens,
        reveal_window_closes=reveal_closes,
    )

    cur.execute(
        """
        INSERT INTO corroboration_commitments
            (id, belief_id, committer_did, commitment_hash, committed_at,
             reveal_window_opens, reveal_window_closes, status)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            commitment.id, commitment.belief_id, commitment.committer_did,
            commitment.commitment_hash, commitment.committed_at,
            commitment.reveal_window_opens, commitment.reveal_window_closes,
            commitment.status,
        ),
    )

    return commitment


def submit_reveal(
    cur,
    commitment_id: str,
    vote_value: str,
    nonce: str,
) -> Reveal:
    """Reveal a previously committed vote.

    Verifies the hash matches the original commitment.

    Args:
        cur: Database cursor.
        commitment_id: UUID of the commitment to reveal.
        vote_value: The actual vote.
        nonce: The secret nonce used in the commitment.

    Returns:
        Reveal with validation results.
    """
    cur.execute(
        "SELECT * FROM corroboration_commitments WHERE id = %s",
        (commitment_id,),
    )
    row = cur.fetchone()
    if not row:
        raise ValueError(f"Commitment not found: {commitment_id}")

    now = datetime.now(UTC)
    expected_hash = compute_commitment_hash(vote_value, nonce)
    is_valid = expected_hash == row["commitment_hash"]
    is_late = now > row["reveal_window_closes"]

    status = "revealed" if is_valid else "penalty"
    if is_late and is_valid:
        status = "penalty"  # Late but valid = penalty

    reveal = Reveal(
        commitment_id=commitment_id,
        vote_value=vote_value,
        nonce=nonce,
        revealed_at=now,
        is_valid=is_valid,
        is_late=is_late,
    )

    cur.execute(
        """
        UPDATE corroboration_commitments
        SET status = %s, revealed_at = %s, vote_value = %s, nonce = %s
        WHERE id = %s
        """,
        (status, now, vote_value, nonce, commitment_id),
    )

    return reveal


def expire_unrevealed(cur) -> int:
    """Mark expired unrevealed commitments as no_reveal.

    Args:
        cur: Database cursor.

    Returns:
        Number of commitments expired.
    """
    now = datetime.now(UTC)
    cur.execute(
        """
        UPDATE corroboration_commitments
        SET status = 'no_reveal'
        WHERE status = 'committed' AND reveal_window_closes < %s
        """,
        (now,),
    )
    return cur.rowcount
