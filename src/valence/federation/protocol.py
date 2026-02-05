"""Federation Protocol Handlers for Valence.

Implements the Valence Federation Protocol (VFP) message handling:
- Authentication (challenge/verify)
- Belief sharing (SHARE_BELIEF, REQUEST_BELIEFS)
- Sync operations (SYNC_REQUEST, SYNC_RESPONSE)
- Trust attestations
"""

from __future__ import annotations

import json
import logging
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from ..core.confidence import ConfidenceDimension, DimensionalConfidence
from ..core.db import get_cursor
from .identity import (
    sign_belief_content,
    verify_belief_signature,
)

logger = logging.getLogger(__name__)


# =============================================================================
# MESSAGE TYPES
# =============================================================================


class MessageType(StrEnum):
    """Federation protocol message types."""

    # Authentication
    AUTH_CHALLENGE = "AUTH_CHALLENGE"
    AUTH_CHALLENGE_RESPONSE = "AUTH_CHALLENGE_RESPONSE"
    AUTH_VERIFY = "AUTH_VERIFY"
    AUTH_VERIFY_RESPONSE = "AUTH_VERIFY_RESPONSE"

    # Beliefs
    SHARE_BELIEF = "SHARE_BELIEF"
    SHARE_BELIEF_RESPONSE = "SHARE_BELIEF_RESPONSE"
    REQUEST_BELIEFS = "REQUEST_BELIEFS"
    BELIEFS_RESPONSE = "BELIEFS_RESPONSE"

    # Sync
    SYNC_REQUEST = "SYNC_REQUEST"
    SYNC_RESPONSE = "SYNC_RESPONSE"

    # Trust
    TRUST_ATTESTATION = "TRUST_ATTESTATION"
    TRUST_ATTESTATION_RESPONSE = "TRUST_ATTESTATION_RESPONSE"
    ENDORSEMENT_REQUEST = "ENDORSEMENT_REQUEST"
    ENDORSEMENT_RESPONSE = "ENDORSEMENT_RESPONSE"

    # Errors
    ERROR = "ERROR"


class ErrorCode(StrEnum):
    """Federation protocol error codes."""

    AUTH_FAILED = "AUTH_FAILED"
    TRUST_INSUFFICIENT = "TRUST_INSUFFICIENT"
    VISIBILITY_DENIED = "VISIBILITY_DENIED"
    RATE_LIMITED = "RATE_LIMITED"
    SYNC_CURSOR_INVALID = "SYNC_CURSOR_INVALID"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    NODE_NOT_FOUND = "NODE_NOT_FOUND"


# =============================================================================
# MESSAGE DATACLASSES
# =============================================================================


@dataclass
class ProtocolMessage:
    """Base class for all protocol messages."""

    type: MessageType
    request_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.value,
            "request_id": str(self.request_id),
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ErrorMessage(ProtocolMessage):
    """Error response message."""

    type: MessageType = MessageType.ERROR
    error_code: ErrorCode = ErrorCode.INTERNAL_ERROR
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "error_code": self.error_code.value,
                "message": self.message,
                "details": self.details,
            }
        )
        return result


# Authentication Messages


@dataclass
class AuthChallengeRequest(ProtocolMessage):
    """Request an authentication challenge."""

    type: MessageType = MessageType.AUTH_CHALLENGE
    client_did: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["client_did"] = self.client_did
        return result


@dataclass
class AuthChallengeResponse(ProtocolMessage):
    """Response with authentication challenge."""

    type: MessageType = MessageType.AUTH_CHALLENGE_RESPONSE
    challenge: str = ""
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(minutes=5))

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "challenge": self.challenge,
                "expires_at": self.expires_at.isoformat(),
            }
        )
        return result


@dataclass
class AuthVerifyRequest(ProtocolMessage):
    """Verify authentication with signed challenge."""

    type: MessageType = MessageType.AUTH_VERIFY
    client_did: str = ""
    challenge: str = ""
    signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "client_did": self.client_did,
                "challenge": self.challenge,
                "signature": self.signature,
            }
        )
        return result


@dataclass
class AuthVerifyResponse(ProtocolMessage):
    """Response with session token."""

    type: MessageType = MessageType.AUTH_VERIFY_RESPONSE
    session_token: str = ""
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(hours=1))

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "session_token": self.session_token,
                "expires_at": self.expires_at.isoformat(),
            }
        )
        return result


# Belief Messages


@dataclass
class ShareBeliefRequest(ProtocolMessage):
    """Share beliefs with the federation."""

    type: MessageType = MessageType.SHARE_BELIEF
    beliefs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["beliefs"] = self.beliefs
        return result


@dataclass
class ShareBeliefResponse(ProtocolMessage):
    """Response to belief sharing."""

    type: MessageType = MessageType.SHARE_BELIEF_RESPONSE
    accepted: int = 0
    rejected: int = 0
    rejection_reasons: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "accepted": self.accepted,
                "rejected": self.rejected,
                "rejection_reasons": self.rejection_reasons,
            }
        )
        return result


@dataclass
class RequestBeliefsRequest(ProtocolMessage):
    """Request beliefs from the federation."""

    type: MessageType = MessageType.REQUEST_BELIEFS
    requester_did: str = ""
    domain_filter: list[str] = field(default_factory=list)
    semantic_query: str | None = None
    min_confidence: float = 0.0
    limit: int = 50
    cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "requester_did": self.requester_did,
                "query": {
                    "domain_filter": self.domain_filter,
                    "semantic_query": self.semantic_query,
                    "min_confidence": self.min_confidence,
                    "limit": self.limit,
                },
                "cursor": self.cursor,
            }
        )
        return result


@dataclass
class BeliefsResponse(ProtocolMessage):
    """Response with beliefs."""

    type: MessageType = MessageType.BELIEFS_RESPONSE
    beliefs: list[dict[str, Any]] = field(default_factory=list)
    total_available: int = 0
    cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "beliefs": self.beliefs,
                "total_available": self.total_available,
                "cursor": self.cursor,
            }
        )
        return result


# Sync Messages


@dataclass
class SyncRequest(ProtocolMessage):
    """Request sync from a peer."""

    type: MessageType = MessageType.SYNC_REQUEST
    since: datetime | None = None
    domains: list[str] = field(default_factory=list)
    cursor: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "since": self.since.isoformat() if self.since else None,
                "domains": self.domains,
                "cursor": self.cursor,
            }
        )
        return result


@dataclass
class SyncChange:
    """A single change in a sync response."""

    change_type: str  # belief_created, belief_superseded, belief_archived
    belief: dict[str, Any] | None = None
    old_belief_id: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": self.change_type,
            "timestamp": self.timestamp.isoformat(),
        }
        if self.belief:
            result["belief"] = self.belief
        if self.old_belief_id:
            result["old_belief_id"] = self.old_belief_id
        return result


@dataclass
class SyncResponse(ProtocolMessage):
    """Response with sync changes."""

    type: MessageType = MessageType.SYNC_RESPONSE
    changes: list[SyncChange] = field(default_factory=list)
    cursor: str | None = None
    has_more: bool = False
    vector_clock: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "changes": [c.to_dict() for c in self.changes],
                "cursor": self.cursor,
                "has_more": self.has_more,
                "vector_clock": self.vector_clock,
            }
        )
        return result


# Trust Messages


@dataclass
class TrustAttestationRequest(ProtocolMessage):
    """Submit a trust attestation."""

    type: MessageType = MessageType.TRUST_ATTESTATION
    attestation: dict[str, Any] = field(default_factory=dict)
    issuer_signature: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "attestation": self.attestation,
                "issuer_signature": self.issuer_signature,
            }
        )
        return result


@dataclass
class TrustAttestationResponse(ProtocolMessage):
    """Response to a trust attestation."""

    type: MessageType = MessageType.TRUST_ATTESTATION_RESPONSE
    accepted: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "accepted": self.accepted,
                "reason": self.reason,
            }
        )
        return result


# =============================================================================
# AUTHENTICATION STATE
# =============================================================================


# In-memory challenge store (should use Redis in production)
_pending_challenges: dict[str, tuple[str, datetime]] = {}


def create_auth_challenge(client_did: str) -> AuthChallengeResponse:
    """Create an authentication challenge for a client.

    Args:
        client_did: The DID of the client requesting authentication

    Returns:
        AuthChallengeResponse with challenge nonce
    """
    # Generate random challenge
    challenge = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(minutes=5)

    # Store challenge
    _pending_challenges[client_did] = (challenge, expires_at)

    return AuthChallengeResponse(
        challenge=challenge,
        expires_at=expires_at,
    )


def verify_auth_challenge(
    client_did: str,
    challenge: str,
    signature: str,
    public_key_multibase: str,
) -> AuthVerifyResponse | ErrorMessage:
    """Verify an authentication challenge response.

    Args:
        client_did: The client's DID
        challenge: The challenge nonce
        signature: Base64-encoded signature of the challenge
        public_key_multibase: Client's public key for verification

    Returns:
        AuthVerifyResponse with session token, or ErrorMessage on failure
    """
    # Check if challenge exists and is valid
    if client_did not in _pending_challenges:
        return ErrorMessage(
            error_code=ErrorCode.AUTH_FAILED,
            message="No pending challenge for this DID",
        )

    stored_challenge, expires_at = _pending_challenges[client_did]

    if datetime.now() > expires_at:
        del _pending_challenges[client_did]
        return ErrorMessage(
            error_code=ErrorCode.AUTH_FAILED,
            message="Challenge has expired",
        )

    if challenge != stored_challenge:
        return ErrorMessage(
            error_code=ErrorCode.AUTH_FAILED,
            message="Challenge mismatch",
        )

    # Verify signature
    try:
        import base64

        from .identity import verify_signature

        challenge_bytes = challenge.encode("utf-8")
        signature_bytes = base64.b64decode(signature)

        if not verify_signature(challenge_bytes, signature_bytes, public_key_multibase):
            return ErrorMessage(
                error_code=ErrorCode.AUTH_FAILED,
                message="Invalid signature",
            )
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        return ErrorMessage(
            error_code=ErrorCode.AUTH_FAILED,
            message="Signature verification failed",
        )

    # Clean up challenge
    del _pending_challenges[client_did]

    # Generate session token
    session_token = secrets.token_hex(32)
    expires_at = datetime.now() + timedelta(hours=1)

    # TODO: Store session token (use Redis in production)

    return AuthVerifyResponse(
        session_token=session_token,
        expires_at=expires_at,
    )


# =============================================================================
# BELIEF HANDLERS
# =============================================================================


def handle_share_belief(
    request: ShareBeliefRequest,
    sender_node_id: UUID,
    sender_trust: float,
) -> ShareBeliefResponse:
    """Handle incoming shared beliefs.

    Args:
        request: The share belief request
        sender_node_id: UUID of the sending node
        sender_trust: Current trust level of sender

    Returns:
        ShareBeliefResponse with acceptance/rejection counts
    """
    accepted = 0
    rejected = 0
    rejection_reasons: dict[str, str] = {}

    for belief_data in request.beliefs:
        try:
            result = _process_incoming_belief(belief_data, sender_node_id, sender_trust)
            if result is True:
                accepted += 1
            else:
                rejected += 1
                belief_id = belief_data.get("federation_id", "unknown")
                # result is str when not True (rejection reason)
                rejection_reasons[str(belief_id)] = str(result) if result else "Unknown error"
        except Exception as e:
            rejected += 1
            belief_id = belief_data.get("federation_id", "unknown")
            rejection_reasons[str(belief_id)] = f"Processing error: {str(e)}"
            logger.exception(f"Error processing belief {belief_id}")

    return ShareBeliefResponse(
        request_id=request.request_id,
        accepted=accepted,
        rejected=rejected,
        rejection_reasons=rejection_reasons,
    )


def _process_incoming_belief(
    belief_data: dict[str, Any],
    sender_node_id: UUID,
    sender_trust: float,
) -> bool | str:
    """Process a single incoming belief.

    Args:
        belief_data: The federated belief data
        sender_node_id: UUID of the sending node
        sender_trust: Trust level of sender

    Returns:
        True if accepted, or rejection reason string
    """
    # Validate required fields
    required_fields = [
        "federation_id",
        "origin_node_did",
        "content",
        "confidence",
        "origin_signature",
    ]
    for field_name in required_fields:
        if field_name not in belief_data:
            return f"Missing required field: {field_name}"

    federation_id = belief_data["federation_id"]
    origin_node_did = belief_data["origin_node_did"]
    content = belief_data["content"]

    # Check if we already have this belief
    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM belief_provenance WHERE federation_id = %s",
            (federation_id,),
        )
        if cur.fetchone():
            return "Belief already exists"

    # Verify signature
    signable_content = {
        "federation_id": str(federation_id),
        "origin_node_did": origin_node_did,
        "content": content,
        "confidence": belief_data["confidence"],
        "domain_path": belief_data.get("domain_path", []),
        "valid_from": belief_data.get("valid_from"),
        "valid_until": belief_data.get("valid_until"),
    }

    # Get origin node's public key
    with get_cursor() as cur:
        cur.execute(
            "SELECT public_key_multibase FROM federation_nodes WHERE did = %s",
            (origin_node_did,),
        )
        row = cur.fetchone()
        if not row:
            return f"Unknown origin node: {origin_node_did}"
        public_key = row["public_key_multibase"]

    # Verify signature
    if not verify_belief_signature(signable_content, belief_data["origin_signature"], public_key):
        return "Invalid signature"

    # Check hop count limit
    hop_count = belief_data.get("hop_count", 1)
    if hop_count > 3:  # Max hops from config
        return f"Hop count exceeded: {hop_count}"

    # Check visibility allows federation
    visibility = belief_data.get("visibility", "federated")
    if visibility == "private":
        return "Belief visibility is private"

    # Validate federation embedding if provided
    from ..core.federation_embedding import validate_incoming_belief_embedding

    is_valid, embed_error = validate_incoming_belief_embedding(belief_data)
    if not is_valid:
        return f"Invalid federation embedding: {embed_error}"

    # Check for corroboration with existing beliefs
    # If a highly similar belief exists, add corroboration instead of creating new
    try:
        from ..core.corroboration import process_incoming_belief_corroboration

        corroboration_result = process_incoming_belief_corroboration(
            content=content,
            source_did=origin_node_did,
        )
        if corroboration_result and corroboration_result.corroborated:
            if corroboration_result.is_new_source:
                logger.info(
                    f"Belief from {origin_node_did} corroborates existing belief "
                    f"{corroboration_result.existing_belief_id} "
                    f"(similarity={corroboration_result.similarity:.3f})"
                )
                return True  # Accepted as corroboration, not new belief
            else:
                return "Source already corroborated this belief"
    except Exception as e:
        # Log but don't fail - corroboration is optional enhancement
        logger.debug(f"Corroboration check skipped: {e}")

    # Create local belief
    confidence_data = belief_data["confidence"]
    if isinstance(confidence_data, dict):
        confidence = DimensionalConfidence.from_dict(confidence_data)
    else:
        confidence = DimensionalConfidence.simple(float(confidence_data))

    # Adjust confidence based on sender trust
    adjusted_overall = confidence.overall * sender_trust
    confidence = confidence.with_dimension(ConfidenceDimension.OVERALL, adjusted_overall, recalculate=False)

    with get_cursor() as cur:
        # Insert belief
        cur.execute(
            """
            INSERT INTO beliefs (
                content, confidence, domain_path, valid_from, valid_until,
                status, is_local, federation_id, visibility, share_level,
                origin_node_id
            ) VALUES (
                %s, %s, %s, %s, %s,
                'active', FALSE, %s, %s, %s,
                %s
            )
            RETURNING id
        """,
            (
                content,
                json.dumps(confidence.to_dict()),
                belief_data.get("domain_path", []),
                belief_data.get("valid_from"),
                belief_data.get("valid_until"),
                federation_id,
                visibility,
                belief_data.get("share_level", "belief_only"),
                sender_node_id,
            ),
        )
        belief_row = cur.fetchone()
        belief_id = belief_row["id"]

        # Insert provenance
        federation_path = belief_data.get("federation_path", [])
        cur.execute(
            """
            INSERT INTO belief_provenance (
                belief_id, federation_id, origin_node_id, origin_belief_id,
                origin_signature, signed_at, signature_verified,
                hop_count, federation_path, share_level
            ) VALUES (
                %s, %s,
                (SELECT id FROM federation_nodes WHERE did = %s),
                %s, %s, %s, TRUE, %s, %s, %s
            )
        """,
            (
                belief_id,
                federation_id,
                origin_node_did,
                belief_data.get("id", federation_id),
                belief_data["origin_signature"],
                belief_data.get("signed_at", datetime.now()),
                hop_count,
                federation_path,
                belief_data.get("share_level", "belief_only"),
            ),
        )

        # Update sender's trust metrics
        cur.execute(
            """
            UPDATE node_trust
            SET beliefs_received = beliefs_received + 1,
                last_interaction_at = NOW(),
                modified_at = NOW()
            WHERE node_id = %s
        """,
            (sender_node_id,),
        )

    logger.info(f"Accepted federated belief {federation_id} from {origin_node_did}")
    return True


def handle_request_beliefs(
    request: RequestBeliefsRequest,
    requester_node_id: UUID,
    requester_trust: float,
) -> BeliefsResponse | ErrorMessage:
    """Handle a request for beliefs.

    Args:
        request: The belief request
        requester_node_id: UUID of the requesting node
        requester_trust: Trust level of requester

    Returns:
        BeliefsResponse with matching beliefs, or ErrorMessage
    """
    # Check trust threshold
    min_trust_for_query = 0.1  # Observer level
    if requester_trust < min_trust_for_query:
        return ErrorMessage(
            request_id=request.request_id,
            error_code=ErrorCode.TRUST_INSUFFICIENT,
            message="Trust level too low for belief queries",
            details={"required": min_trust_for_query, "current": requester_trust},
        )

    # Build query
    conditions = ["status = 'active'", "is_local = TRUE"]
    params: list[Any] = []

    # Visibility filter based on trust
    if requester_trust >= 0.4:  # Participant
        conditions.append("visibility IN ('trusted', 'federated', 'public')")
    elif requester_trust >= 0.2:  # Contributor
        conditions.append("visibility IN ('federated', 'public')")
    else:  # Observer
        conditions.append("visibility = 'public'")

    # Domain filter
    if request.domain_filter:
        conditions.append("domain_path && %s")
        params.append(request.domain_filter)

    # Confidence filter
    if request.min_confidence > 0:
        conditions.append("(confidence->>'overall')::numeric >= %s")
        params.append(request.min_confidence)

    # Limit
    limit = min(request.limit, 100)  # Cap at 100

    # Execute query
    query = f"""  # nosec B608
        SELECT id, content, confidence, domain_path, valid_from, valid_until,
               visibility, share_level, created_at
        FROM beliefs
        WHERE {" AND ".join(conditions)}
        ORDER BY created_at DESC
        LIMIT %s
    """
    params.append(limit)

    beliefs_data = []
    with get_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

        for row in rows:
            belief_dict = _belief_row_to_federated(row, requester_trust)
            if belief_dict:
                beliefs_data.append(belief_dict)

    # Get total count for pagination
    count_query = f"""  # nosec B608
        SELECT COUNT(*) as total FROM beliefs
        WHERE {" AND ".join(conditions[:-1] if limit else conditions)}
    """
    with get_cursor() as cur:
        cur.execute(count_query, params[:-1] if limit else params)
        total = cur.fetchone()["total"]

    return BeliefsResponse(
        request_id=request.request_id,
        beliefs=beliefs_data,
        total_available=total,
        cursor=None,  # TODO: Implement cursor-based pagination
    )


def _belief_row_to_federated(
    row: dict[str, Any],
    requester_trust: float,
) -> dict[str, Any] | None:
    """Convert a belief row to federated format based on share level.

    Args:
        row: Database row
        requester_trust: Trust level of requester

    Returns:
        Federated belief dict, or None if not shareable
    """
    from ..core.config import get_federation_config

    settings = get_federation_config()

    share_level = row.get("share_level", "belief_only")

    # Basic belief data (always included)
    result = {
        "id": str(row["id"]),
        "federation_id": str(row.get("federation_id") or row["id"]),
        "origin_node_did": settings.federation_node_did or "did:vkb:web:localhost",
        "content": row["content"],
        "confidence": row["confidence"],
        "domain_path": row.get("domain_path", []),
        "visibility": row.get("visibility", "federated"),
        "share_level": share_level,
        "hop_count": 0,  # Originating from this node
        "federation_path": [],
    }

    # Add temporal validity
    if row.get("valid_from"):
        result["valid_from"] = row["valid_from"].isoformat()
    if row.get("valid_until"):
        result["valid_until"] = row["valid_until"].isoformat()

    # Sign the belief
    if settings.federation_private_key:
        signable = {
            "federation_id": result["federation_id"],
            "origin_node_did": result["origin_node_did"],
            "content": result["content"],
            "confidence": result["confidence"],
            "domain_path": result["domain_path"],
            "valid_from": result.get("valid_from"),
            "valid_until": result.get("valid_until"),
        }
        result["origin_signature"] = sign_belief_content(
            signable,
            bytes.fromhex(settings.federation_private_key),
        )
        result["signed_at"] = datetime.now().isoformat()

    return result


# =============================================================================
# SYNC HANDLERS
# =============================================================================


def handle_sync_request(
    request: SyncRequest,
    requester_node_id: UUID,
    requester_trust: float,
) -> SyncResponse | ErrorMessage:
    """Handle a sync request from a peer.

    Args:
        request: The sync request
        requester_node_id: UUID of the requesting node
        requester_trust: Trust level of requester

    Returns:
        SyncResponse with changes, or ErrorMessage
    """
    # Check trust threshold for sync
    min_trust_for_sync = 0.2  # Contributor level
    if requester_trust < min_trust_for_sync:
        return ErrorMessage(
            request_id=request.request_id,
            error_code=ErrorCode.TRUST_INSUFFICIENT,
            message="Trust level too low for sync",
            details={"required": min_trust_for_sync, "current": requester_trust},
        )

    # Build query for changes since last sync
    conditions = ["status = 'active'", "is_local = TRUE"]
    params: list[Any] = []

    # Time filter
    if request.since:
        conditions.append("modified_at > %s")
        params.append(request.since)

    # Domain filter
    if request.domains:
        conditions.append("domain_path && %s")
        params.append(request.domains)

    # Visibility filter
    if requester_trust >= 0.4:
        conditions.append("visibility IN ('trusted', 'federated', 'public')")
    else:
        conditions.append("visibility IN ('federated', 'public')")

    # Query changes
    query = f"""  # nosec B608
        SELECT id, content, confidence, domain_path, valid_from, valid_until,
               visibility, share_level, created_at, modified_at, status,
               supersedes_id, superseded_by_id
        FROM beliefs
        WHERE {" AND ".join(conditions)}
        ORDER BY modified_at ASC
        LIMIT 100
    """

    changes = []
    with get_cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

        for row in rows:
            belief_dict = _belief_row_to_federated(row, requester_trust)
            if belief_dict:
                # Determine change type
                if row.get("supersedes_id"):
                    change_type = "belief_superseded"
                elif row["status"] == "archived":
                    change_type = "belief_archived"
                else:
                    change_type = "belief_created"

                change = SyncChange(
                    change_type=change_type,
                    belief=belief_dict,
                    old_belief_id=(str(row["supersedes_id"]) if row.get("supersedes_id") else None),
                    timestamp=row["modified_at"],
                )
                changes.append(change)

    # Update sync metrics
    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE sync_state
            SET last_sync_at = NOW(),
                beliefs_sent = beliefs_sent + %s,
                modified_at = NOW()
            WHERE node_id = %s
        """,
            (len(changes), requester_node_id),
        )

        cur.execute(
            """
            UPDATE node_trust
            SET sync_requests_served = sync_requests_served + 1,
                last_interaction_at = NOW(),
                modified_at = NOW()
            WHERE node_id = %s
        """,
            (requester_node_id,),
        )

    # Determine if there are more changes
    has_more = len(changes) == 100

    # Fetch our vector clock for this peer relationship
    our_vector_clock: dict[str, int] = {}
    with get_cursor() as cur:
        cur.execute(
            "SELECT vector_clock FROM sync_state WHERE node_id = %s",
            (requester_node_id,),
        )
        row = cur.fetchone()
        if row:
            our_vector_clock = row.get("vector_clock", {}) or {}

    return SyncResponse(
        request_id=request.request_id,
        changes=changes,
        cursor=changes[-1].timestamp.isoformat() if changes else None,
        has_more=has_more,
        vector_clock=our_vector_clock,
    )


# =============================================================================
# MESSAGE PARSING
# =============================================================================


def parse_message(data: dict[str, Any]) -> ProtocolMessage | None:
    """Parse a raw message dict into a typed message.

    Args:
        data: Raw message dictionary

    Returns:
        Typed message object, or None if invalid
    """
    msg_type = data.get("type")
    if not msg_type:
        return None

    try:
        message_type = MessageType(msg_type)
    except ValueError:
        return None

    request_id = UUID(data.get("request_id", str(uuid4())))

    if message_type == MessageType.AUTH_CHALLENGE:
        return AuthChallengeRequest(
            request_id=request_id,
            client_did=data.get("client_did", ""),
        )

    elif message_type == MessageType.AUTH_VERIFY:
        return AuthVerifyRequest(
            request_id=request_id,
            client_did=data.get("client_did", ""),
            challenge=data.get("challenge", ""),
            signature=data.get("signature", ""),
        )

    elif message_type == MessageType.SHARE_BELIEF:
        return ShareBeliefRequest(
            request_id=request_id,
            beliefs=data.get("beliefs", []),
        )

    elif message_type == MessageType.REQUEST_BELIEFS:
        query = data.get("query", {})
        return RequestBeliefsRequest(
            request_id=request_id,
            requester_did=data.get("requester_did", ""),
            domain_filter=query.get("domain_filter", []),
            semantic_query=query.get("semantic_query"),
            min_confidence=query.get("min_confidence", 0.0),
            limit=query.get("limit", 50),
            cursor=data.get("cursor"),
        )

    elif message_type == MessageType.SYNC_REQUEST:
        since_str = data.get("since")
        since = datetime.fromisoformat(since_str) if since_str else None
        return SyncRequest(
            request_id=request_id,
            since=since,
            domains=data.get("domains", []),
            cursor=data.get("cursor"),
        )

    elif message_type == MessageType.TRUST_ATTESTATION:
        return TrustAttestationRequest(
            request_id=request_id,
            attestation=data.get("attestation", {}),
            issuer_signature=data.get("issuer_signature", ""),
        )

    return None


# =============================================================================
# MESSAGE DISPATCH
# =============================================================================


async def handle_message(
    message: ProtocolMessage,
    sender_did: str | None = None,
    sender_node_id: UUID | None = None,
) -> ProtocolMessage | dict[str, Any]:
    """Handle an incoming protocol message.

    Dispatches to the appropriate handler based on message type.

    Args:
        message: Parsed protocol message
        sender_did: DID of the sending node (for authenticated messages)
        sender_node_id: UUID of the sending node (if known)

    Returns:
        Response message or dict
    """
    try:
        msg_type = message.type

        # Authentication messages (no sender required)
        if msg_type == MessageType.AUTH_CHALLENGE:
            if isinstance(message, AuthChallengeRequest):
                response = create_auth_challenge(message.client_did)
                return response
            return ErrorMessage(
                error_code=ErrorCode.INVALID_REQUEST,
                message="Invalid AUTH_CHALLENGE message format",
            )

        elif msg_type == MessageType.AUTH_VERIFY:
            if not isinstance(message, AuthVerifyRequest):
                return ErrorMessage(
                    error_code=ErrorCode.INVALID_REQUEST,
                    message="Invalid AUTH_VERIFY message format",
                )
            # Get public key from database
            public_key = None
            if message.client_did:
                with get_cursor() as cur:
                    cur.execute(
                        "SELECT public_key_multibase FROM federation_nodes WHERE did = %s",
                        (message.client_did,),
                    )
                    row = cur.fetchone()
                    if row:
                        public_key = row["public_key_multibase"]

            if not public_key:
                return ErrorMessage(
                    error_code=ErrorCode.NODE_NOT_FOUND,
                    message=f"Unknown node: {message.client_did}",
                )

            verify_response = verify_auth_challenge(
                client_did=message.client_did,
                challenge=message.challenge,
                signature=message.signature,
                public_key_multibase=public_key,
            )
            return verify_response

        # Messages requiring sender context
        if not sender_node_id:
            # Try to look up sender by DID
            if sender_did:
                with get_cursor() as cur:
                    cur.execute("SELECT id FROM federation_nodes WHERE did = %s", (sender_did,))
                    row = cur.fetchone()
                    if row:
                        sender_node_id = row["id"]

        if not sender_node_id:
            return ErrorMessage(
                error_code=ErrorCode.AUTH_FAILED,
                message="Sender identification required",
            )

        # Get sender trust
        from .trust import get_effective_trust

        sender_trust = get_effective_trust(sender_node_id)

        # Handle authenticated messages
        if msg_type == MessageType.SHARE_BELIEF:
            if isinstance(message, ShareBeliefRequest):
                return handle_share_belief(message, sender_node_id, sender_trust)
            return ErrorMessage(
                error_code=ErrorCode.INVALID_REQUEST,
                message="Invalid SHARE_BELIEF format",
            )

        elif msg_type == MessageType.REQUEST_BELIEFS:
            if isinstance(message, RequestBeliefsRequest):
                return handle_request_beliefs(message, sender_node_id, sender_trust)
            return ErrorMessage(
                error_code=ErrorCode.INVALID_REQUEST,
                message="Invalid REQUEST_BELIEFS format",
            )

        elif msg_type == MessageType.SYNC_REQUEST:
            if isinstance(message, SyncRequest):
                return handle_sync_request(message, sender_node_id, sender_trust)
            return ErrorMessage(
                error_code=ErrorCode.INVALID_REQUEST,
                message="Invalid SYNC_REQUEST format",
            )

        elif msg_type == MessageType.TRUST_ATTESTATION:
            if isinstance(message, TrustAttestationRequest):
                return _handle_trust_attestation(message, sender_node_id, sender_trust)
            return ErrorMessage(
                error_code=ErrorCode.INVALID_REQUEST,
                message="Invalid TRUST_ATTESTATION format",
            )

        # Unknown message type
        return ErrorMessage(
            error_code=ErrorCode.INVALID_REQUEST,
            message=f"Unhandled message type: {msg_type}",
        )

    except Exception as e:
        logger.exception(f"Error handling message: {e}")
        return ErrorMessage(
            error_code=ErrorCode.INTERNAL_ERROR,
            message=str(e),
        )


def _handle_trust_attestation(
    request: TrustAttestationRequest,
    sender_node_id: UUID,
    sender_trust: float,
) -> TrustAttestationResponse:
    """Handle incoming trust attestation.

    Args:
        request: The attestation request
        sender_node_id: UUID of the sending node
        sender_trust: Trust level of sender

    Returns:
        TrustAttestationResponse
    """
    try:
        attestation = request.attestation
        subject_did = attestation.get("subject_did")

        if not subject_did:
            return TrustAttestationResponse(
                request_id=request.request_id,
                accepted=False,
                reason="Missing subject_did in attestation",
            )

        # Get subject node
        with get_cursor() as cur:
            cur.execute("SELECT id FROM federation_nodes WHERE did = %s", (subject_did,))
            row = cur.fetchone()
            if not row:
                return TrustAttestationResponse(
                    request_id=request.request_id,
                    accepted=False,
                    reason=f"Unknown subject node: {subject_did}",
                )

            subject_node_id = row["id"]

        # Process endorsement if attestation type is endorsement
        attestation_type = attestation.get("attestation_type", "endorsement")
        if attestation_type == "endorsement":
            from .models import TrustAttestation
            from .trust import get_trust_manager

            trust_attestation = TrustAttestation(
                issuer_did=attestation.get("issuer_did", ""),
                subject_did=subject_did,
                attestation_type=attestation_type,
                attested_dimensions=attestation.get("attested_dimensions", {}),
                domains=attestation.get("domains"),
            )

            manager = get_trust_manager()
            manager.process_endorsement(
                subject_node_id=subject_node_id,
                endorser_node_id=sender_node_id,
                attestation=trust_attestation,
            )

            return TrustAttestationResponse(
                request_id=request.request_id,
                accepted=True,
            )

        return TrustAttestationResponse(
            request_id=request.request_id,
            accepted=False,
            reason=f"Unknown attestation type: {attestation_type}",
        )

    except Exception as e:
        logger.exception("Error handling trust attestation")
        return TrustAttestationResponse(
            request_id=request.request_id,
            accepted=False,
            reason=str(e),
        )
