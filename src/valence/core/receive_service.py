"""Incoming share handling and reshare (#344).

Handles:
1. Receiving shared beliefs from federation peers
2. Validating consent chains and provenance
3. Storing received beliefs with provenance tracking
4. Resharing with policy validation and chain extension
5. Notifications for incoming shares
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass
class IncomingShare:
    """A share received from a federation peer."""

    id: str
    belief_id: str
    sender_did: str
    content: str
    confidence: dict[str, float]
    intent: str
    consent_chain: dict[str, Any]
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: str = "pending"  # pending, accepted, rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "belief_id": self.belief_id,
            "sender_did": self.sender_did,
            "content": self.content,
            "confidence": self.confidence,
            "intent": self.intent,
            "status": self.status,
            "received_at": self.received_at.isoformat(),
        }


@dataclass
class ReshareResult:
    """Result of a reshare operation."""

    success: bool
    share_id: str | None = None
    new_chain_hash: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"success": self.success}
        if self.share_id:
            d["share_id"] = self.share_id
        if self.new_chain_hash:
            d["new_chain_hash"] = self.new_chain_hash
        if self.error:
            d["error"] = self.error
        return d


@dataclass
class ShareNotification:
    """Notification about an incoming share."""

    id: str
    share_id: str
    recipient_did: str
    sender_did: str
    belief_id: str
    intent: str
    read: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "share_id": self.share_id,
            "sender_did": self.sender_did,
            "belief_id": self.belief_id,
            "intent": self.intent,
            "read": self.read,
            "created_at": self.created_at.isoformat(),
        }


def validate_consent_chain(chain: dict[str, Any]) -> tuple[bool, str]:
    """Validate a consent chain's integrity.

    Checks:
    - Chain has required fields (origin_sharer, origin_signature, chain_hash)
    - Chain is not revoked
    - Hop count within limits

    Args:
        chain: Consent chain dict.

    Returns:
        (valid, error_message) tuple.
    """
    required = ["origin_sharer", "chain_hash"]
    for field_name in required:
        if field_name not in chain:
            return False, f"Missing required field: {field_name}"

    if chain.get("revoked"):
        return False, "Consent chain has been revoked"

    max_hops = chain.get("max_hops")
    current_hops = chain.get("hop_count", 0)
    if max_hops is not None and current_hops >= max_hops:
        return False, f"Max hops exceeded: {current_hops} >= {max_hops}"

    return True, ""


def receive_share(
    cur,
    sender_did: str,
    belief_content: str,
    confidence: dict[str, float],
    intent: str,
    consent_chain: dict[str, Any],
    recipient_did: str,
    encrypted_envelope: dict[str, Any] | None = None,
) -> IncomingShare | str:
    """Process an incoming shared belief.

    Args:
        cur: Database cursor.
        sender_did: DID of the sender.
        belief_content: The belief content.
        confidence: Confidence dimensions.
        intent: Sharing intent.
        consent_chain: The provenance chain.
        recipient_did: DID of the recipient (local).
        encrypted_envelope: Optional encrypted envelope.

    Returns:
        IncomingShare if accepted, error string if rejected.
    """
    valid, error = validate_consent_chain(consent_chain)
    if not valid:
        return f"Invalid consent chain: {error}"

    share_id = str(uuid4())
    incoming = IncomingShare(
        id=share_id,
        belief_id=consent_chain.get("belief_id", str(uuid4())),
        sender_did=sender_did,
        content=belief_content,
        confidence=confidence,
        intent=intent,
        consent_chain=consent_chain,
        status="accepted",
    )

    # Store the received belief with provenance
    cur.execute(
        """
        INSERT INTO received_shares
            (id, belief_id, sender_did, recipient_did, content, confidence,
             intent, consent_chain, encrypted_envelope, status, received_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            incoming.id, incoming.belief_id, sender_did, recipient_did,
            belief_content, json.dumps(confidence), intent,
            json.dumps(consent_chain),
            json.dumps(encrypted_envelope) if encrypted_envelope else None,
            incoming.status, incoming.received_at,
        ),
    )

    # Create notification
    notification_id = str(uuid4())
    cur.execute(
        """
        INSERT INTO share_notifications
            (id, share_id, recipient_did, sender_did, belief_id, intent, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (notification_id, share_id, recipient_did, sender_did,
         incoming.belief_id, intent, incoming.received_at),
    )

    return incoming


def reshare_belief(
    cur,
    share_id: str,
    resharer_did: str,
    new_recipient_did: str,
) -> ReshareResult:
    """Reshare a received belief to another peer.

    Validates:
    - Original share exists and is accepted
    - Intent allows resharing (not know_me)
    - Hop count within limits
    - Not already shared to this recipient

    Args:
        cur: Database cursor.
        share_id: UUID of the received share to reshare.
        resharer_did: DID of the resharer.
        new_recipient_did: DID of the new recipient.

    Returns:
        ReshareResult.
    """
    cur.execute("SELECT * FROM received_shares WHERE id = %s", (share_id,))
    original = cur.fetchone()
    if not original:
        return ReshareResult(success=False, error="Share not found")

    if original["status"] != "accepted":
        return ReshareResult(success=False, error="Share not in accepted state")

    # know_me intent is direct-only
    if original["intent"] == "know_me":
        return ReshareResult(success=False, error="know_me intent does not allow resharing")

    # Check hop count
    chain = json.loads(original["consent_chain"]) if isinstance(original["consent_chain"], str) else original["consent_chain"]
    max_hops = chain.get("max_hops")
    current_hops = chain.get("hop_count", 0)
    if max_hops is not None and current_hops + 1 >= max_hops:
        return ReshareResult(success=False, error=f"Would exceed max hops: {current_hops + 1} >= {max_hops}")

    # Check not already shared to this recipient
    cur.execute(
        """
        SELECT 1 FROM received_shares
        WHERE belief_id = %s AND recipient_did = %s AND sender_did = %s
        """,
        (original["belief_id"], new_recipient_did, resharer_did),
    )
    if cur.fetchone():
        return ReshareResult(success=False, error="Already shared to this recipient")

    # Extend consent chain
    import hashlib
    new_chain = dict(chain)
    new_chain["hop_count"] = current_hops + 1
    new_chain_hash = hashlib.sha256(
        f"{chain.get('chain_hash', '')}:{resharer_did}:{new_recipient_did}".encode()
    ).hexdigest()
    new_chain["chain_hash"] = new_chain_hash

    new_share_id = str(uuid4())
    now = datetime.now(UTC)

    cur.execute(
        """
        INSERT INTO received_shares
            (id, belief_id, sender_did, recipient_did, content, confidence,
             intent, consent_chain, status, received_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            new_share_id, original["belief_id"], resharer_did, new_recipient_did,
            original["content"], original["confidence"], original["intent"],
            json.dumps(new_chain), "accepted", now,
        ),
    )

    # Notification for new recipient
    cur.execute(
        """
        INSERT INTO share_notifications
            (id, share_id, recipient_did, sender_did, belief_id, intent, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (str(uuid4()), new_share_id, new_recipient_did, resharer_did,
         original["belief_id"], original["intent"], now),
    )

    return ReshareResult(
        success=True,
        share_id=new_share_id,
        new_chain_hash=new_chain_hash,
    )


def get_notifications(
    cur,
    recipient_did: str,
    unread_only: bool = True,
    limit: int = 20,
) -> list[ShareNotification]:
    """Get share notifications for a recipient.

    Args:
        cur: Database cursor.
        recipient_did: DID to get notifications for.
        unread_only: Only return unread notifications.
        limit: Max results.

    Returns:
        List of ShareNotification objects.
    """
    where = "WHERE recipient_did = %s"
    params: list[Any] = [recipient_did]
    if unread_only:
        where += " AND read = FALSE"
    where += " ORDER BY created_at DESC LIMIT %s"
    params.append(limit)

    cur.execute(f"SELECT * FROM share_notifications {where}", tuple(params))
    rows = cur.fetchall()

    return [
        ShareNotification(
            id=row["id"],
            share_id=row["share_id"],
            recipient_did=row["recipient_did"],
            sender_did=row["sender_did"],
            belief_id=row["belief_id"],
            intent=row["intent"],
            read=row["read"],
            created_at=row["created_at"],
        )
        for row in rows
    ]


def mark_notification_read(cur, notification_id: str) -> bool:
    """Mark a notification as read.

    Args:
        cur: Database cursor.
        notification_id: UUID of the notification.

    Returns:
        True if updated, False if not found.
    """
    cur.execute(
        "UPDATE share_notifications SET read = TRUE WHERE id = %s",
        (notification_id,),
    )
    return cur.rowcount > 0
