"""Consent management for GDPR compliance.

Implements consent record lifecycle: grant, check, revoke, expiry.
Consent records have a 7-year minimum retention per GDPR requirements.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from our_db import get_cursor

logger = logging.getLogger(__name__)

# GDPR requires consent records be kept for at least 7 years
RETENTION_YEARS = 7


@dataclass
class ConsentRecord:
    """A record of consent granted by a data holder."""

    id: UUID
    holder_did: str
    purpose: str
    scope: str
    granted_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    retention_until: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if consent is currently active (not revoked, not expired)."""
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and self.expires_at < datetime.now():
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "holder_did": self.holder_did,
            "purpose": self.purpose,
            "scope": self.scope,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "retention_until": self.retention_until.isoformat() if self.retention_until else None,
            "is_active": self.is_active(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ConsentRecord:
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return cls(
            id=row["id"],
            holder_did=row["holder_did"],
            purpose=row["purpose"],
            scope=row["scope"],
            granted_at=row["granted_at"],
            expires_at=row.get("expires_at"),
            revoked_at=row.get("revoked_at"),
            retention_until=row.get("retention_until"),
            metadata=metadata,
        )


class ConsentManager:
    """Manages consent records for GDPR compliance.

    Consent purposes:
    - 'data_processing': General data processing consent
    - 'federation_sharing': Consent to share data via federation
    - 'analytics': Consent for pattern/insight extraction
    - 'backup': Consent for data backup and retention

    Scope values:
    - 'all': All data for the holder
    - 'beliefs': Only belief data
    - 'sessions': Only session/conversation data
    - 'patterns': Only behavioral pattern data
    """

    VALID_PURPOSES = {"data_processing", "federation_sharing", "analytics", "backup"}

    @staticmethod
    def record_consent(
        holder_did: str,
        purpose: str,
        scope: str = "all",
        expiry: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConsentRecord:
        """Record a new consent grant.

        Args:
            holder_did: DID of the consent holder
            purpose: Purpose of consent (data_processing, federation_sharing, etc.)
            scope: Scope of consent (all, beliefs, sessions, patterns)
            expiry: Optional expiration datetime
            metadata: Optional additional metadata

        Returns:
            The created ConsentRecord
        """
        retention_until = datetime.now() + timedelta(days=RETENTION_YEARS * 365)

        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO consent_records
                    (holder_did, purpose, scope, granted_at, expires_at, retention_until, metadata)
                VALUES (%s, %s, %s, NOW(), %s, %s, %s)
                RETURNING *
                """,
                (
                    holder_did,
                    purpose,
                    scope,
                    expiry,
                    retention_until,
                    json.dumps(metadata or {}),
                ),
            )
            row = cur.fetchone()
            record = ConsentRecord.from_row(dict(row))

            logger.info(f"Consent recorded: holder={holder_did}, purpose={purpose}, scope={scope}")
            return record

    @staticmethod
    def check_consent(holder_did: str, purpose: str) -> bool:
        """Check if active consent exists for a holder and purpose.

        Args:
            holder_did: DID of the consent holder
            purpose: Purpose to check consent for

        Returns:
            True if active consent exists
        """
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT id FROM consent_records
                WHERE holder_did = %s
                  AND purpose = %s
                  AND revoked_at IS NULL
                  AND (expires_at IS NULL OR expires_at > NOW())
                LIMIT 1
                """,
                (holder_did, purpose),
            )
            return cur.fetchone() is not None

    @staticmethod
    def revoke_consent(consent_id: str | UUID, reason: str | None = None) -> bool:
        """Revoke a consent record.

        The record is not deleted (retention requirement), only marked as revoked.

        Args:
            consent_id: UUID of the consent record
            reason: Optional reason for revocation

        Returns:
            True if revocation was successful
        """
        with get_cursor() as cur:
            cur.execute(
                """
                UPDATE consent_records
                SET revoked_at = NOW(), metadata = metadata || %s
                WHERE id = %s AND revoked_at IS NULL
                RETURNING id
                """,
                (
                    json.dumps({"revocation_reason": reason} if reason else {}),
                    str(consent_id),
                ),
            )
            row = cur.fetchone()
            if row:
                logger.info(f"Consent revoked: id={consent_id}, reason={reason}")
                return True
            return False

    @staticmethod
    def list_consents(holder_did: str, include_revoked: bool = False) -> list[ConsentRecord]:
        """List all consent records for a holder.

        Args:
            holder_did: DID of the consent holder
            include_revoked: Whether to include revoked consents

        Returns:
            List of ConsentRecord objects
        """
        with get_cursor() as cur:
            sql = "SELECT * FROM consent_records WHERE holder_did = %s"
            params: list[Any] = [holder_did]

            if not include_revoked:
                sql += " AND revoked_at IS NULL"

            sql += " ORDER BY granted_at DESC"

            cur.execute(sql, params)
            rows = cur.fetchall()
            return [ConsentRecord.from_row(dict(r)) for r in rows]

    @staticmethod
    def get_consent(consent_id: str | UUID) -> ConsentRecord | None:
        """Get a single consent record by ID."""
        with get_cursor() as cur:
            cur.execute("SELECT * FROM consent_records WHERE id = %s", (str(consent_id),))
            row = cur.fetchone()
            if row:
                return ConsentRecord.from_row(dict(row))
            return None
