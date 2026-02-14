"""Resilient storage — backup orchestration, encryption, integrity verification.

Wraps our-storage (erasure coding, Merkle trees, backends) with
Valence-specific backup logic: belief selection, encryption, and
DB-tracked backup sets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from our_db import get_cursor

from .exceptions import NotFoundError

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class BackupStatus(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"
    CORRUPTED = "corrupted"


class RedundancyLevel(StrEnum):
    """How many copies/shards to create."""

    MINIMAL = "minimal"  # 3-of-5
    PERSONAL = "personal"  # 5-of-8
    FEDERATION = "federation"  # 8-of-12
    PARANOID = "paranoid"  # 12-of-20


# Erasure coding parameters per redundancy level
REDUNDANCY_PARAMS = {
    RedundancyLevel.MINIMAL: {"data_shards": 3, "parity_shards": 2},
    RedundancyLevel.PERSONAL: {"data_shards": 5, "parity_shards": 3},
    RedundancyLevel.FEDERATION: {"data_shards": 8, "parity_shards": 4},
    RedundancyLevel.PARANOID: {"data_shards": 12, "parity_shards": 8},
}


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class BackupSet:
    """A completed backup operation."""

    id: UUID
    belief_count: int = 0
    total_size_bytes: int = 0
    content_hash: str = ""
    redundancy_level: RedundancyLevel = RedundancyLevel.PERSONAL
    encrypted: bool = False
    status: BackupStatus = BackupStatus.IN_PROGRESS
    shard_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    verified_at: datetime | None = None
    error_message: str | None = None


@dataclass
class BackupShard:
    """A single shard of a backup set."""

    id: UUID
    set_id: UUID
    shard_index: int
    is_parity: bool
    size_bytes: int
    checksum: str
    backend_id: str = "local"
    location: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    verified_at: datetime | None = None


@dataclass
class IntegrityReport:
    """Report from a backup integrity check."""

    set_id: UUID
    is_valid: bool
    shards_checked: int
    shards_valid: int
    shards_missing: int
    shards_corrupted: int
    can_recover: bool
    checked_at: datetime = field(default_factory=datetime.now)


# ============================================================================
# Backup Operations
# ============================================================================


def create_backup(
    redundancy: RedundancyLevel = RedundancyLevel.PERSONAL,
    domain_filter: list[str] | None = None,
    min_confidence: float | None = None,
    encrypt: bool = False,
) -> BackupSet:
    """Create a backup of beliefs from the database.

    Selects beliefs based on filters, serializes them, and creates
    an erasure-coded backup set tracked in the DB.
    """
    backup_set = BackupSet(
        id=uuid4(),
        redundancy_level=redundancy,
        encrypted=encrypt,
    )

    try:
        # Select beliefs
        beliefs = _select_beliefs(domain_filter, min_confidence)
        if not beliefs:
            backup_set.status = BackupStatus.COMPLETED
            backup_set.belief_count = 0
            _save_backup_set(backup_set)
            return backup_set

        # Serialize
        payload = json.dumps(beliefs, default=str).encode("utf-8")
        backup_set.total_size_bytes = len(payload)
        backup_set.belief_count = len(beliefs)
        backup_set.content_hash = hashlib.sha256(payload).hexdigest()

        # Encrypt if requested
        if encrypt:
            payload = _encrypt_payload(payload)

        # Create shards via erasure coding
        params = REDUNDANCY_PARAMS[redundancy]
        shards = _create_shards(payload, params["data_shards"], params["parity_shards"])
        backup_set.shard_count = len(shards)

        # Store shards
        for i, shard_data in enumerate(shards):
            shard = BackupShard(
                id=uuid4(),
                set_id=backup_set.id,
                shard_index=i,
                is_parity=i >= params["data_shards"],
                size_bytes=len(shard_data),
                checksum=hashlib.sha256(shard_data).hexdigest(),
                backend_id="local",
                location=f"backup/{backup_set.id}/{i}.shard",
            )
            _save_shard(shard, shard_data)

        backup_set.status = BackupStatus.COMPLETED
        _save_backup_set(backup_set)

        logger.info(f"Backup {backup_set.id} completed: {backup_set.belief_count} beliefs, {backup_set.shard_count} shards")

    except Exception as e:
        backup_set.status = BackupStatus.FAILED
        backup_set.error_message = str(e)
        _save_backup_set(backup_set)
        logger.error(f"Backup {backup_set.id} failed: {e}")

    return backup_set


def verify_backup(backup_set_id: UUID) -> IntegrityReport:
    """Verify integrity of a backup set.

    Checks each shard's checksum against stored value.
    """
    with get_cursor() as cur:
        cur.execute("SELECT * FROM backup_sets WHERE id = %s", (str(backup_set_id),))
        row = cur.fetchone()
        if not row:
            raise NotFoundError("BackupSet", str(backup_set_id))

        cur.execute(
            "SELECT * FROM backup_shards WHERE set_id = %s ORDER BY shard_index",
            (str(backup_set_id),),
        )
        shard_rows = cur.fetchall()

    shards_checked = 0
    shards_valid = 0
    shards_missing = 0
    shards_corrupted = 0

    data_shards = int(row.get("data_shards", 5))

    for shard_row in shard_rows:
        shards_checked += 1
        location = shard_row.get("location", "")
        stored_checksum = shard_row.get("checksum", "")

        if not os.path.exists(location):
            shards_missing += 1
            continue

        try:
            with open(location, "rb") as f:
                data = f.read()
            actual_checksum = hashlib.sha256(data).hexdigest()
            if actual_checksum == stored_checksum:
                shards_valid += 1
            else:
                shards_corrupted += 1
        except Exception:
            shards_corrupted += 1

    can_recover = shards_valid >= data_shards

    report = IntegrityReport(
        set_id=backup_set_id,
        is_valid=shards_corrupted == 0 and shards_missing == 0,
        shards_checked=shards_checked,
        shards_valid=shards_valid,
        shards_missing=shards_missing,
        shards_corrupted=shards_corrupted,
        can_recover=can_recover,
    )

    # Update backup set status
    new_status = BackupStatus.VERIFIED if report.is_valid else BackupStatus.CORRUPTED
    with get_cursor() as cur:
        cur.execute(
            "UPDATE backup_sets SET status = %s, verified_at = NOW() WHERE id = %s",
            (new_status.value, str(backup_set_id)),
        )

    return report


def list_backups(limit: int = 20) -> list[BackupSet]:
    """List backup sets ordered by creation date."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT * FROM backup_sets ORDER BY created_at DESC LIMIT %s",
            (limit,),
        )
        return [
            BackupSet(
                id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
                belief_count=row.get("belief_count", 0),
                total_size_bytes=row.get("total_size_bytes", 0),
                content_hash=row.get("content_hash", ""),
                redundancy_level=RedundancyLevel(row.get("redundancy_level", "personal")),
                encrypted=row.get("encrypted", False),
                status=BackupStatus(row.get("status", "completed")),
                shard_count=row.get("shard_count", 0),
                created_at=row["created_at"],
                verified_at=row.get("verified_at"),
                error_message=row.get("error_message"),
            )
            for row in cur.fetchall()
        ]


def get_backup(backup_set_id: UUID) -> BackupSet | None:
    """Get a backup set by ID."""
    with get_cursor() as cur:
        cur.execute("SELECT * FROM backup_sets WHERE id = %s", (str(backup_set_id),))
        row = cur.fetchone()
        if not row:
            return None

        return BackupSet(
            id=row["id"] if isinstance(row["id"], UUID) else UUID(row["id"]),
            belief_count=row.get("belief_count", 0),
            total_size_bytes=row.get("total_size_bytes", 0),
            content_hash=row.get("content_hash", ""),
            redundancy_level=RedundancyLevel(row.get("redundancy_level", "personal")),
            encrypted=row.get("encrypted", False),
            status=BackupStatus(row.get("status", "completed")),
            shard_count=row.get("shard_count", 0),
            created_at=row["created_at"],
            verified_at=row.get("verified_at"),
            error_message=row.get("error_message"),
        )


# ============================================================================
# Internal Helpers
# ============================================================================


def _select_beliefs(
    domain_filter: list[str] | None = None,
    min_confidence: float | None = None,
) -> list[dict[str, Any]]:
    """Select beliefs for backup."""
    with get_cursor() as cur:
        sql = "SELECT id, content, confidence, domain_path, source_id, created_at FROM beliefs WHERE status = 'active'"
        params: list[Any] = []

        if domain_filter:
            sql += " AND domain_path && %s"
            params.append(domain_filter)
        if min_confidence is not None:
            sql += " AND (confidence->>'overall')::float >= %s"
            params.append(min_confidence)

        sql += " ORDER BY created_at DESC"
        cur.execute(sql, params)
        return cur.fetchall()


def _create_shards(payload: bytes, data_shards: int, parity_shards: int) -> list[bytes]:
    """Create erasure-coded shards from payload.

    Uses simple splitting for now. In production, would use our_storage.ErasureCodec.
    """
    data_shards + parity_shards
    shard_size = (len(payload) + data_shards - 1) // data_shards
    shards = []

    for i in range(data_shards):
        start = i * shard_size
        end = min(start + shard_size, len(payload))
        shard = payload[start:end]
        if len(shard) < shard_size:
            shard += b"\x00" * (shard_size - len(shard))
        shards.append(shard)

    # Parity shards (simple XOR for now — production uses Reed-Solomon)
    for i in range(parity_shards):
        parity = bytearray(shard_size)
        for j in range(data_shards):
            for k in range(shard_size):
                parity[k] ^= shards[j][k]
        shards.append(bytes(parity))

    return shards


def _encrypt_payload(payload: bytes) -> bytes:
    """Encrypt payload. Placeholder — production uses our-crypto hybrid PQ."""
    # Simple XOR with random key for now (not cryptographically secure)
    # Production would use: our_crypto.X25519PREBackend + Kyber
    key = os.urandom(32)
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(payload))
    return key + encrypted  # Prefix key for demo purposes


def _save_backup_set(backup_set: BackupSet) -> None:
    """Persist backup set metadata to DB."""
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO backup_sets (
                id, belief_count, total_size_bytes, content_hash,
                redundancy_level, encrypted, status, shard_count,
                created_at, verified_at, error_message
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                status = EXCLUDED.status,
                verified_at = EXCLUDED.verified_at,
                error_message = EXCLUDED.error_message
            """,
            (
                str(backup_set.id), backup_set.belief_count,
                backup_set.total_size_bytes, backup_set.content_hash,
                backup_set.redundancy_level.value, backup_set.encrypted,
                backup_set.status.value, backup_set.shard_count,
                backup_set.created_at, backup_set.verified_at,
                backup_set.error_message,
            ),
        )


def _save_shard(shard: BackupShard, data: bytes) -> None:
    """Persist shard metadata and data."""
    # Save metadata to DB
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO backup_shards (
                id, set_id, shard_index, is_parity, size_bytes,
                checksum, backend_id, location, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(shard.id), str(shard.set_id), shard.shard_index,
                shard.is_parity, shard.size_bytes, shard.checksum,
                shard.backend_id, shard.location, shard.created_at,
            ),
        )

    # Write data to disk
    os.makedirs(os.path.dirname(shard.location), exist_ok=True)
    with open(shard.location, "wb") as f:
        f.write(data)
