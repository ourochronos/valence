"""Audit logging for compliance and security.

Every state-changing operation should create an audit record.
Audit logs are append-only and not deletable (even during GDPR erasure,
audit records are retained with PII redacted).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from our_db import get_cursor

logger = logging.getLogger(__name__)


class AuditAction(StrEnum):
    """Actions that create audit records."""

    # Belief operations
    BELIEF_CREATE = "belief_create"
    BELIEF_SUPERSEDE = "belief_supersede"
    BELIEF_ARCHIVE = "belief_archive"

    # Sharing operations
    BELIEF_SHARE = "belief_share"
    SHARE_REVOKE = "share_revoke"

    # Tension operations
    TENSION_RESOLVE = "tension_resolve"

    # Consent operations
    CONSENT_GRANT = "consent_grant"
    CONSENT_REVOKE = "consent_revoke"

    # Data access operations (GDPR)
    DATA_ACCESS = "data_access"
    DATA_EXPORT = "data_export"
    DATA_DELETE = "data_delete"

    # Session operations
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # Verification operations
    VERIFICATION_SUBMIT = "verification_submit"
    DISPUTE_SUBMIT = "dispute_submit"
    DISPUTE_RESOLVE = "dispute_resolve"


@dataclass
class AuditEntry:
    """A single audit log entry."""

    id: UUID
    timestamp: datetime
    actor_did: str | None
    action: AuditAction
    resource_type: str
    resource_id: str | None
    details: dict[str, Any]
    ip_address: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "actor_did": self.actor_did,
            "action": self.action.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "details": self.details,
            "ip_address": self.ip_address,
        }


class AuditLogger:
    """Append-only audit logger backed by PostgreSQL.

    Usage:
        audit = get_audit_logger()
        audit.log(
            action=AuditAction.BELIEF_CREATE,
            resource_type="belief",
            resource_id=str(belief_id),
            details={"content_preview": content[:100]},
            actor_did="did:valence:local",
        )
    """

    def log(
        self,
        action: AuditAction,
        resource_type: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        actor_did: str | None = None,
        ip_address: str | None = None,
    ) -> None:
        """Write an audit log entry. Non-fatal on error.

        Args:
            action: The action being audited
            resource_type: Type of resource (belief, share, consent, etc.)
            resource_id: ID of the affected resource
            details: Additional context about the action
            actor_did: DID of the actor performing the action
            ip_address: IP address of the request (if applicable)
        """
        try:
            with get_cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_log
                        (actor_did, action, resource_type, resource_id, details, ip_address)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        actor_did,
                        action.value,
                        resource_type,
                        resource_id,
                        json.dumps(details or {}),
                        ip_address,
                    ),
                )
        except Exception as e:
            # Audit logging should never break the main operation
            logger.warning(f"Failed to write audit log: {e}")

    def query(
        self,
        action: AuditAction | None = None,
        resource_type: str | None = None,
        resource_id: str | None = None,
        actor_did: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Query audit log entries.

        Args:
            action: Filter by action type
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            actor_did: Filter by actor
            since: Only entries after this timestamp
            limit: Maximum entries to return

        Returns:
            List of AuditEntry objects
        """
        with get_cursor() as cur:
            sql = "SELECT * FROM audit_log WHERE 1=1"
            params: list[Any] = []

            if action:
                sql += " AND action = %s"
                params.append(action.value)

            if resource_type:
                sql += " AND resource_type = %s"
                params.append(resource_type)

            if resource_id:
                sql += " AND resource_id = %s"
                params.append(resource_id)

            if actor_did:
                sql += " AND actor_did = %s"
                params.append(actor_did)

            if since:
                sql += " AND timestamp >= %s"
                params.append(since)

            sql += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)

            cur.execute(sql, params)
            rows = cur.fetchall()

            entries = []
            for row in rows:
                details = row.get("details", {})
                if isinstance(details, str):
                    details = json.loads(details)
                entries.append(
                    AuditEntry(
                        id=row["id"],
                        timestamp=row["timestamp"],
                        actor_did=row.get("actor_did"),
                        action=AuditAction(row["action"]),
                        resource_type=row["resource_type"],
                        resource_id=row.get("resource_id"),
                        details=details,
                        ip_address=row.get("ip_address"),
                    )
                )
            return entries


# Singleton instance
_audit_logger: AuditLogger | None = None


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
