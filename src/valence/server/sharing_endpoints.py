"""Sharing API endpoints.

Implements:
- POST /api/v1/share - Share a belief with a recipient
- GET /api/v1/shares - List shares
- GET /api/v1/shares/{id} - Get share details
- POST /api/v1/shares/{id}/revoke - Revoke a share
"""

from __future__ import annotations

import json
import logging
from typing import Any

from oro_privacy.sharing import (
    RevokeRequest,
    Share,
    ShareRequest,
    SharingService,
)
from oro_privacy.types import SharePolicy
from starlette.requests import Request
from starlette.responses import JSONResponse

from .errors import (
    CONFLICT_ALREADY_REVOKED,
    FORBIDDEN_NOT_OWNER,
    NOT_FOUND_SHARE,
    VALIDATION_INVALID_VALUE,
    conflict_error,
    forbidden_error,
    internal_error,
    invalid_json_error,
    missing_field_error,
    not_found_error,
    service_unavailable_error,
    validation_error,
)

logger = logging.getLogger(__name__)


# Global sharing service instance (initialized in app startup)
_sharing_service: SharingService | None = None


def get_sharing_service() -> SharingService | None:
    """Get the sharing service instance."""
    return _sharing_service


def set_sharing_service(service: SharingService) -> None:
    """Set the sharing service instance."""
    global _sharing_service
    _sharing_service = service


async def share_belief_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/share - Share a belief with a specific recipient.

    Request Body (JSON):
        {
            "belief_id": "uuid-string",
            "recipient_did": "did:key:...",
            "policy": {  // optional, defaults to DIRECT
                "level": "direct",
                "enforcement": "cryptographic",
                "recipients": ["did:key:..."]
            }
        }

    Returns:
        200: Share result with share_id, consent_chain_id, etc.
        400: Invalid request (validation error)
        500: Server error
    """
    service = get_sharing_service()
    if service is None:
        return service_unavailable_error("Sharing service")

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return invalid_json_error()

    # Validate required fields
    belief_id = body.get("belief_id")
    if not belief_id:
        return missing_field_error("belief_id")

    recipient_did = body.get("recipient_did")
    if not recipient_did:
        return missing_field_error("recipient_did")

    # Parse optional policy
    policy = None
    if "policy" in body and body["policy"] is not None:
        try:
            policy = SharePolicy.from_dict(body["policy"])
        except (KeyError, ValueError) as e:
            return validation_error(f"Invalid policy: {e}", code=VALIDATION_INVALID_VALUE)

    # Create share request
    share_request = ShareRequest(
        belief_id=belief_id,
        recipient_did=recipient_did,
        policy=policy,
    )

    # Get sharer DID from authenticated context
    # For now, use identity service's local DID
    sharer_did = service.identity.get_did()

    try:
        result = await service.share(share_request, sharer_did)

        return JSONResponse(
            {
                "success": True,
                "share_id": result.share_id,
                "consent_chain_id": result.consent_chain_id,
                "encrypted_for": result.encrypted_for,
                "created_at": result.created_at,
            },
            status_code=200,
        )

    except ValueError as e:
        return validation_error(str(e), code=VALIDATION_INVALID_VALUE)
    except Exception as e:
        logger.exception(f"Error sharing belief: {e}")
        return internal_error()


async def list_shares_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/shares - List shares.

    Query Parameters:
        sharer_did: Filter by sharer DID (optional)
        recipient_did: Filter by recipient DID (optional)
        limit: Maximum results (optional, default 100)
        revoked: Include revoked shares (optional, default false)

    Returns:
        200: List of shares
        500: Server error
    """
    service = get_sharing_service()
    if service is None:
        return service_unavailable_error("Sharing service")

    sharer_did = request.query_params.get("sharer_did")
    recipient_did = request.query_params.get("recipient_did")

    # Parse include_revoked param
    revoked_param = request.query_params.get("revoked", "false").lower()
    include_revoked = revoked_param in ("true", "1", "yes")

    try:
        limit = int(request.query_params.get("limit", "100"))
        limit = min(limit, 1000)  # Cap at 1000
    except ValueError:
        limit = 100

    try:
        shares = await service.list_shares(
            sharer_did=sharer_did,
            recipient_did=recipient_did,
            limit=limit,
            include_revoked=include_revoked,
        )

        return JSONResponse(
            {
                "success": True,
                "shares": [_share_to_dict(s) for s in shares],
                "count": len(shares),
            },
            status_code=200,
        )

    except Exception as e:
        logger.exception(f"Error listing shares: {e}")
        return internal_error()


async def get_share_endpoint(request: Request) -> JSONResponse:
    """GET /api/v1/shares/{id} - Get share details.

    Path Parameters:
        id: Share UUID

    Returns:
        200: Share details
        404: Share not found
        500: Server error
    """
    service = get_sharing_service()
    if service is None:
        return service_unavailable_error("Sharing service")

    share_id = request.path_params.get("id")
    if not share_id:
        return missing_field_error("Share ID")

    try:
        share = await service.get_share(share_id)

        if share is None:
            return not_found_error("Share", code=NOT_FOUND_SHARE)

        return JSONResponse(
            {
                "success": True,
                "share": _share_to_dict(share),
            },
            status_code=200,
        )

    except Exception as e:
        logger.exception(f"Error getting share: {e}")
        return internal_error()


async def revoke_share_endpoint(request: Request) -> JSONResponse:
    """POST /api/v1/shares/{id}/revoke - Revoke a share.

    Path Parameters:
        id: Share UUID

    Request Body (JSON, optional):
        {
            "reason": "Optional reason for revocation"
        }

    Returns:
        200: Revocation result with share_id, consent_chain_id, revoked_at, affected_recipients
        400: Invalid request (validation error, already revoked)
        403: Permission denied (not the original sharer)
        404: Share not found
        500: Server error
    """
    service = get_sharing_service()
    if service is None:
        return service_unavailable_error("Sharing service")

    share_id = request.path_params.get("id")
    if not share_id:
        return missing_field_error("Share ID")

    # Parse optional reason from body
    reason = None
    try:
        body = await request.json()
        reason = body.get("reason")
    except json.JSONDecodeError:
        # Body is optional, so empty/invalid JSON is OK
        pass

    # Create revoke request
    revoke_request = RevokeRequest(
        share_id=share_id,
        reason=reason,
    )

    # Get revoker DID from authenticated context
    # For now, use identity service's local DID
    revoker_did = service.identity.get_did()

    try:
        result = await service.revoke_share(revoke_request, revoker_did)

        return JSONResponse(
            {
                "success": True,
                "share_id": result.share_id,
                "consent_chain_id": result.consent_chain_id,
                "revoked_at": result.revoked_at,
                "affected_recipients": result.affected_recipients,
            },
            status_code=200,
        )

    except ValueError as e:
        # Share not found, consent chain not found, or already revoked
        error_msg = str(e)
        if "not found" in error_msg.lower():
            return not_found_error("Share", code=NOT_FOUND_SHARE)
        if "already revoked" in error_msg.lower():
            return conflict_error(error_msg, code=CONFLICT_ALREADY_REVOKED)
        return validation_error(error_msg, code=VALIDATION_INVALID_VALUE)
    except PermissionError as e:
        return forbidden_error(str(e), code=FORBIDDEN_NOT_OWNER)
    except Exception as e:
        logger.exception(f"Error revoking share: {e}")
        return internal_error()


def _share_to_dict(share: Share) -> dict[str, Any]:
    """Convert a Share to a JSON-serializable dict."""
    return {
        "id": share.id,
        "consent_chain_id": share.consent_chain_id,
        "recipient_did": share.recipient_did,
        "created_at": share.created_at,
        "accessed_at": share.accessed_at,
        # Note: encrypted_envelope is not exposed in the listing
    }
