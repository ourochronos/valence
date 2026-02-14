"""Merkle consistency proofs for belief set integrity (#351).

Builds Merkle trees over the active belief set, exchanges roots with
federation peers, and detects divergence (network partitions).

Uses our-storage MerkleTree for tree construction and proof generation.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class MerkleCheckpoint:
    """Snapshot of the belief set Merkle root at a point in time."""

    id: str
    root_hash: str
    belief_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    peer_roots: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "root_hash": self.root_hash,
            "belief_count": self.belief_count,
            "timestamp": self.timestamp.isoformat(),
            "peer_roots": self.peer_roots,
        }


@dataclass
class PartitionEvent:
    """A detected divergence between local and peer belief sets."""

    id: str
    peer_did: str
    local_root: str
    peer_root: str
    severity: str  # "info", "warning", "critical"
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    resolved_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "peer_did": self.peer_did,
            "local_root": self.local_root,
            "peer_root": self.peer_root,
            "severity": self.severity,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


def build_belief_merkle_root(cur) -> MerkleCheckpoint:
    """Build a Merkle tree over all active (non-superseded) beliefs.

    Args:
        cur: Database cursor.

    Returns:
        MerkleCheckpoint with the root hash and belief count.
    """
    from our_storage import MerkleTree

    # Get sorted belief IDs for deterministic tree
    cur.execute(
        "SELECT id FROM beliefs WHERE superseded_by_id IS NULL AND status != 'archived' ORDER BY id"
    )
    rows = cur.fetchall()

    if not rows:
        empty_hash = hashlib.sha256(b"empty").hexdigest()
        return MerkleCheckpoint(id=str(uuid4()), root_hash=empty_hash, belief_count=0)

    belief_hashes = [hashlib.sha256(str(row["id"]).encode()).hexdigest() for row in rows]
    tree = MerkleTree.from_hashes(belief_hashes)

    checkpoint = MerkleCheckpoint(
        id=str(uuid4()),
        root_hash=tree.root_hash,
        belief_count=len(rows),
    )

    # Persist checkpoint
    cur.execute(
        """
        INSERT INTO merkle_checkpoints (id, root_hash, belief_count, created_at)
        VALUES (%s, %s, %s, %s)
        """,
        (checkpoint.id, checkpoint.root_hash, checkpoint.belief_count, checkpoint.timestamp),
    )

    return checkpoint


def compare_with_peer(
    local_root: str,
    peer_did: str,
    peer_root: str,
    cur=None,
) -> PartitionEvent | None:
    """Compare local Merkle root with a peer's root.

    Args:
        local_root: Our current root hash.
        peer_did: The peer's DID.
        peer_root: The peer's root hash.
        cur: Optional database cursor for persisting events.

    Returns:
        PartitionEvent if roots diverge, None if consistent.
    """
    if local_root == peer_root:
        return None

    event = PartitionEvent(
        id=str(uuid4()),
        peer_did=peer_did,
        local_root=local_root,
        peer_root=peer_root,
        severity="warning",
    )

    logger.warning("Merkle divergence detected with peer %s: local=%s peer=%s", peer_did, local_root[:16], peer_root[:16])

    if cur is not None:
        cur.execute(
            """
            INSERT INTO partition_events (id, peer_did, local_root, peer_root, severity, detected_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (event.id, event.peer_did, event.local_root, event.peer_root, event.severity, event.detected_at),
        )

    return event


def get_recent_checkpoints(cur, limit: int = 10) -> list[MerkleCheckpoint]:
    """Get recent Merkle checkpoints.

    Args:
        cur: Database cursor.
        limit: Maximum checkpoints to return.

    Returns:
        List of MerkleCheckpoint objects, newest first.
    """
    cur.execute(
        "SELECT id, root_hash, belief_count, created_at FROM merkle_checkpoints ORDER BY created_at DESC LIMIT %s",
        (limit,),
    )
    rows = cur.fetchall()
    return [
        MerkleCheckpoint(
            id=str(row["id"]),
            root_hash=row["root_hash"],
            belief_count=row["belief_count"],
            timestamp=row["created_at"],
        )
        for row in rows
    ]
