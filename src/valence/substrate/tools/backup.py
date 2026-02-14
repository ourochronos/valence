"""Backup/resilient storage MCP tool implementations."""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from valence.core.backup import (
    RedundancyLevel,
)
from valence.core.backup import (
    create_backup as db_create_backup,
)
from valence.core.backup import (
    get_backup as db_get_backup,
)
from valence.core.backup import (
    list_backups as db_list_backups,
)
from valence.core.backup import (
    verify_backup as db_verify_backup,
)
from valence.core.exceptions import NotFoundError

logger = logging.getLogger(__name__)


def _parse_uuid(value: str, name: str) -> UUID | dict[str, Any]:
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return {"success": False, "error": f"Invalid UUID for {name}: {value}"}


def backup_create(
    redundancy: str = "personal",
    domain_filter: list[str] | None = None,
    min_confidence: float | None = None,
    encrypt: bool = False,
    **_: Any,
) -> dict[str, Any]:
    """Create a backup of beliefs."""
    try:
        level = RedundancyLevel(redundancy)
    except ValueError:
        return {"success": False, "error": f"Invalid redundancy '{redundancy}'. Must be: minimal, personal, federation, paranoid"}

    backup = db_create_backup(
        redundancy=level,
        domain_filter=domain_filter,
        min_confidence=min_confidence,
        encrypt=encrypt,
    )

    return {
        "success": True,
        "backup": {
            "id": str(backup.id),
            "belief_count": backup.belief_count,
            "total_size_bytes": backup.total_size_bytes,
            "shard_count": backup.shard_count,
            "redundancy_level": backup.redundancy_level.value,
            "encrypted": backup.encrypted,
            "status": backup.status.value,
            "content_hash": backup.content_hash,
            "created_at": backup.created_at.isoformat() if backup.created_at else None,
        },
    }


def backup_verify(backup_set_id: str, **_: Any) -> dict[str, Any]:
    """Verify integrity of a backup set."""
    bid = _parse_uuid(backup_set_id, "backup_set_id")
    if isinstance(bid, dict):
        return bid

    try:
        report = db_verify_backup(bid)
    except NotFoundError as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "report": {
            "is_valid": report.is_valid,
            "shards_checked": report.shards_checked,
            "shards_valid": report.shards_valid,
            "shards_missing": report.shards_missing,
            "shards_corrupted": report.shards_corrupted,
            "can_recover": report.can_recover,
            "checked_at": report.checked_at.isoformat() if report.checked_at else None,
        },
    }


def backup_list(limit: int = 20, **_: Any) -> dict[str, Any]:
    """List backup sets."""
    backups = db_list_backups(limit=limit)
    return {
        "success": True,
        "backups": [
            {
                "id": str(b.id),
                "belief_count": b.belief_count,
                "shard_count": b.shard_count,
                "status": b.status.value,
                "redundancy_level": b.redundancy_level.value,
                "created_at": b.created_at.isoformat() if b.created_at else None,
            }
            for b in backups
        ],
        "total": len(backups),
    }


def backup_get(backup_set_id: str, **_: Any) -> dict[str, Any]:
    """Get details of a backup set."""
    bid = _parse_uuid(backup_set_id, "backup_set_id")
    if isinstance(bid, dict):
        return bid

    backup = db_get_backup(bid)
    if not backup:
        return {"success": False, "error": f"Backup {backup_set_id} not found"}

    return {
        "success": True,
        "backup": {
            "id": str(backup.id),
            "belief_count": backup.belief_count,
            "total_size_bytes": backup.total_size_bytes,
            "shard_count": backup.shard_count,
            "content_hash": backup.content_hash,
            "redundancy_level": backup.redundancy_level.value,
            "encrypted": backup.encrypted,
            "status": backup.status.value,
            "created_at": backup.created_at.isoformat() if backup.created_at else None,
            "verified_at": backup.verified_at.isoformat() if backup.verified_at else None,
            "error_message": backup.error_message,
        },
    }
