"""Sharing service: business logic for belief sharing (#341).

Extracts core sharing operations from MCP tool handlers into a composable
service layer. Tool handlers become thin wrappers that validate input,
call the service, and format responses.

Composes:
- identity_service: DID resolution and Ed25519 signing
- crypto_service: PRE encryption/decryption
- DB operations via cursor
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class SharingError(Exception):
    """Base exception for sharing operations."""

    pass


class BeliefNotFoundError(SharingError):
    pass


class OwnershipError(SharingError):
    pass


class PolicyViolationError(SharingError):
    pass


class DuplicateShareError(SharingError):
    pass


@dataclass
class ShareResult:
    """Result of a share operation."""

    share_id: str
    consent_chain_id: str
    recipient_did: str
    intent: str
    policy: dict[str, Any]
    belief_content: str


@dataclass
class ShareInfo:
    """Information about an existing share."""

    share_id: str
    belief_id: str | None
    belief_content: str | None
    recipient_did: str
    origin_sharer: str
    intent: str
    created_at: str | None
    access_count: int
    revoked: bool
    revoked_at: str | None
    revocation_reason: str | None


@dataclass
class RevokeResult:
    """Result of a revoke operation."""

    share_id: str
    consent_chain_id: str
    reason: str | None


class SharingService:
    """Core sharing business logic.

    Stateless service that operates on a database cursor. Each method
    performs a complete operation within the caller's transaction.
    """

    def share_belief(
        self,
        cur,
        belief_id: str,
        recipient_did: str,
        intent: str,
        local_did: str,
        max_hops: int | None = None,
        expires_at: datetime | None = None,
        local_entity_id: str | None = None,
    ) -> ShareResult:
        """Create a share with consent chain and encrypted envelope.

        Args:
            cur: Database cursor (within a transaction).
            belief_id: UUID of the belief to share.
            recipient_did: DID of the recipient.
            intent: Sharing intent (know_me, work_with_me, etc.).
            local_did: The sharer's DID.
            max_hops: Maximum propagation hops (None = unlimited).
            expires_at: Optional expiration datetime.
            local_entity_id: Local entity ID for ownership check (if applicable).

        Returns:
            ShareResult with share details.

        Raises:
            BeliefNotFoundError: Belief does not exist.
            OwnershipError: Caller does not own the belief.
            PolicyViolationError: Share policy prevents sharing.
            DuplicateShareError: Already shared with this recipient.
        """
        from our_privacy.types import IntentConfig, SharePolicy, SharingIntent

        # Fetch belief
        cur.execute("SELECT id, content, share_policy, holder_id FROM beliefs WHERE id = %s", (belief_id,))
        belief_row = cur.fetchone()
        if not belief_row:
            raise BeliefNotFoundError(f"Belief not found: {belief_id}")

        # Ownership check
        holder_id = belief_row.get("holder_id")
        if holder_id is not None:
            if local_entity_id is None or str(holder_id) != local_entity_id:
                raise OwnershipError(f"Cannot share belief {belief_id}: you are not the holder of this belief")

        # Reshare policy check
        existing_policy_data = belief_row.get("share_policy")
        if existing_policy_data:
            if isinstance(existing_policy_data, str):
                existing_policy_data = json.loads(existing_policy_data)
            policy_dict = existing_policy_data.get("policy", existing_policy_data)
            existing_share_policy = SharePolicy.from_dict(policy_dict)
            if existing_share_policy.is_expired():
                raise PolicyViolationError("Belief's share policy has expired — sharing is no longer permitted")
            if not existing_share_policy.allows_sharing_to(recipient_did):
                raise PolicyViolationError(f"Belief's share policy does not allow sharing to {recipient_did}")

        # Build intent config and policy
        intent_config = IntentConfig(
            intent=SharingIntent(intent),
            recipients=[recipient_did],
            max_hops=max_hops,
            expires_at=expires_at,
        )
        share_policy = intent_config.to_share_policy()

        # Sign consent chain
        from .identity_service import build_consent_chain_message, compute_chain_hash, sign_data

        policy_json = json.dumps(intent_config.to_dict())
        message = build_consent_chain_message(belief_id, local_did, recipient_did, intent, policy_json)
        sig_result = sign_data(message)
        origin_signature = sig_result.signature
        chain_hash = compute_chain_hash(origin_signature)

        # Insert consent chain
        cur.execute(
            """
            INSERT INTO consent_chains (belief_id, origin_sharer, origin_timestamp, origin_policy, origin_signature, chain_hash)
            VALUES (%s, %s, NOW(), %s, %s, %s)
            RETURNING id
            """,
            (belief_id, local_did, policy_json, origin_signature, chain_hash),
        )
        consent_chain_id = cur.fetchone()["id"]

        # Encrypt content
        from .crypto_service import encrypt_for_sharing

        encrypted_envelope = encrypt_for_sharing(belief_row["content"], recipient_did)

        # Insert share
        try:
            cur.execute(
                """
                INSERT INTO shares (consent_chain_id, encrypted_envelope, recipient_did, intent, belief_id)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
                """,
                (str(consent_chain_id), json.dumps(encrypted_envelope), recipient_did, intent, belief_id),
            )
            share_id = cur.fetchone()["id"]
        except Exception as e:
            err_name = type(e).__name__
            if err_name == "UniqueViolation" or "unique" in str(e).lower() or "duplicate" in str(e).lower():
                raise DuplicateShareError("Belief already shared with this recipient") from e
            raise

        # Update belief's share_policy (only if currently NULL)
        cur.execute(
            "UPDATE beliefs SET share_policy = COALESCE(share_policy, %s), modified_at = NOW() WHERE id = %s",
            (json.dumps(intent_config.to_dict()), belief_id),
        )

        return ShareResult(
            share_id=str(share_id),
            consent_chain_id=str(consent_chain_id),
            recipient_did=recipient_did,
            intent=intent,
            policy=share_policy.to_dict(),
            belief_content=belief_row["content"],
        )

    def list_shares(
        self,
        cur,
        local_did: str,
        direction: str = "outgoing",
        include_revoked: bool = False,
        limit: int = 20,
        belief_id: str | None = None,
    ) -> list[ShareInfo]:
        """List shares for the local DID.

        Args:
            cur: Database cursor.
            local_did: The caller's DID.
            direction: "outgoing" or "incoming".
            include_revoked: Include revoked shares.
            limit: Maximum results.
            belief_id: Filter to specific belief.

        Returns:
            List of ShareInfo objects.
        """
        if direction == "outgoing":
            sql = """
                SELECT s.id as share_id, s.recipient_did, s.intent, s.belief_id,
                       s.created_at, s.access_count,
                       cc.origin_sharer, cc.revoked, cc.revoked_at, cc.revocation_reason,
                       b.content as belief_content
                FROM shares s
                JOIN consent_chains cc ON s.consent_chain_id = cc.id
                LEFT JOIN beliefs b ON s.belief_id = b.id
                WHERE cc.origin_sharer = %s
            """
        else:
            sql = """
                SELECT s.id as share_id, s.recipient_did, s.intent, s.belief_id,
                       s.created_at, s.access_count,
                       cc.origin_sharer, cc.revoked, cc.revoked_at, cc.revocation_reason,
                       b.content as belief_content
                FROM shares s
                JOIN consent_chains cc ON s.consent_chain_id = cc.id
                LEFT JOIN beliefs b ON s.belief_id = b.id
                WHERE s.recipient_did = %s
            """

        params: list[Any] = [local_did]

        if belief_id:
            sql += " AND s.belief_id = %s"
            params.append(belief_id)

        if not include_revoked:
            sql += " AND cc.revoked = false"

        sql += " ORDER BY s.created_at DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        return [
            ShareInfo(
                share_id=str(row["share_id"]),
                belief_id=str(row["belief_id"]) if row["belief_id"] else None,
                belief_content=row.get("belief_content"),
                recipient_did=row["recipient_did"],
                origin_sharer=row["origin_sharer"],
                intent=row["intent"],
                created_at=row["created_at"].isoformat() if row["created_at"] else None,
                access_count=row["access_count"],
                revoked=row["revoked"],
                revoked_at=row["revoked_at"].isoformat() if row.get("revoked_at") else None,
                revocation_reason=row.get("revocation_reason"),
            )
            for row in rows
        ]

    def revoke_share(
        self,
        cur,
        share_id: str,
        local_did: str,
        reason: str | None = None,
    ) -> RevokeResult:
        """Revoke a share by marking its consent chain.

        Args:
            cur: Database cursor.
            share_id: UUID of the share to revoke.
            local_did: The caller's DID (must be origin sharer).
            reason: Optional revocation reason.

        Returns:
            RevokeResult with revocation details.

        Raises:
            SharingError: Share not found, already revoked, or not owner.
        """
        cur.execute(
            """
            SELECT s.id, s.consent_chain_id, cc.origin_sharer, cc.revoked
            FROM shares s
            JOIN consent_chains cc ON s.consent_chain_id = cc.id
            WHERE s.id = %s
            """,
            (share_id,),
        )
        row = cur.fetchone()
        if not row:
            raise SharingError(f"Share not found: {share_id}")

        if row["revoked"]:
            raise SharingError("Share is already revoked")

        if row["origin_sharer"] != local_did:
            raise SharingError("Cannot revoke a share you did not create")

        cur.execute(
            """
            UPDATE consent_chains
            SET revoked = true, revoked_at = NOW(), revoked_by = %s, revocation_reason = %s
            WHERE id = %s
            """,
            (local_did, reason, row["consent_chain_id"]),
        )

        return RevokeResult(
            share_id=share_id,
            consent_chain_id=str(row["consent_chain_id"]),
            reason=reason,
        )


# Module-level singleton for convenience
_service = SharingService()


def get_sharing_service() -> SharingService:
    """Get the sharing service singleton."""
    return _service
