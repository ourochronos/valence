"""Sharing tool implementations (#341: thin handlers delegating to SharingService).

Functions:
    belief_share, belief_shares_list, belief_share_revoke,
    _get_local_did, _validate_enum (re-exported from _common)
"""

from __future__ import annotations

from typing import Any

from . import _common
from ._common import _validate_enum, datetime, os


def _get_local_did() -> str:
    """Get the local DID for signing via identity service."""
    from ...core.identity_service import get_local_did

    return get_local_did()


def belief_share(
    belief_id: str,
    recipient_did: str,
    intent: str = "know_me",
    max_hops: int | None = None,
    expires_at: str | None = None,
) -> dict[str, Any]:
    """Share a belief with a specific person.

    Thin handler: validates input, delegates to SharingService.
    """
    from ...core.sharing_service import (
        BeliefNotFoundError,
        DuplicateShareError,
        OwnershipError,
        PolicyViolationError,
        get_sharing_service,
    )

    # --- Input validation (stays in handler) ---
    valid_intents = ["know_me", "work_with_me", "learn_from_me", "use_this"]
    if err := _validate_enum(intent, valid_intents, "intent"):
        return err

    parsed_expires = None
    if expires_at:
        try:
            parsed_expires = datetime.fromisoformat(expires_at)
        except ValueError:
            return {"success": False, "error": f"Invalid expires_at format: '{expires_at}'. Must be ISO 8601."}

    local_did = _get_local_did()
    local_entity_id = os.environ.get("VALENCE_LOCAL_ENTITY_ID")
    service = get_sharing_service()

    with _common.get_cursor() as cur:
        try:
            result = service.share_belief(
                cur,
                belief_id=belief_id,
                recipient_did=recipient_did,
                intent=intent,
                local_did=local_did,
                max_hops=max_hops,
                expires_at=parsed_expires,
                local_entity_id=local_entity_id,
            )
        except BeliefNotFoundError as e:
            return {"success": False, "error": str(e)}
        except OwnershipError as e:
            return {"success": False, "error": str(e)}
        except PolicyViolationError as e:
            return {"success": False, "error": str(e)}
        except DuplicateShareError as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "share_id": result.share_id,
            "consent_chain_id": result.consent_chain_id,
            "recipient": result.recipient_did,
            "intent": result.intent,
            "policy": result.policy,
            "belief_content": result.belief_content,
        }


def belief_shares_list(
    direction: str = "outgoing",
    include_revoked: bool = False,
    limit: int = 20,
    belief_id: str | None = None,
) -> dict[str, Any]:
    """List shares -- outgoing or incoming.

    Thin handler: validates input, delegates to SharingService.
    """
    from ...core.sharing_service import get_sharing_service

    # --- Input validation ---
    if err := _validate_enum(direction, ["outgoing", "incoming"], "direction"):
        return err

    local_did = _get_local_did()
    service = get_sharing_service()

    with _common.get_cursor() as cur:
        shares = service.list_shares(
            cur,
            local_did=local_did,
            direction=direction,
            include_revoked=include_revoked,
            limit=limit,
            belief_id=belief_id,
        )

        return {
            "success": True,
            "direction": direction,
            "shares": [
                {
                    "share_id": s.share_id,
                    "belief_id": s.belief_id,
                    "belief_content": s.belief_content,
                    "recipient_did": s.recipient_did,
                    "origin_sharer": s.origin_sharer,
                    "intent": s.intent,
                    "created_at": s.created_at,
                    "access_count": s.access_count,
                    "revoked": s.revoked,
                    "revoked_at": s.revoked_at,
                    "revocation_reason": s.revocation_reason,
                }
                for s in shares
            ],
            "total_count": len(shares),
        }


def belief_share_revoke(
    share_id: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Revoke a share by marking its consent chain as revoked.

    Thin handler: delegates to SharingService.
    """
    from ...core.sharing_service import SharingError, get_sharing_service

    local_did = _get_local_did()
    service = get_sharing_service()

    with _common.get_cursor() as cur:
        try:
            result = service.revoke_share(cur, share_id=share_id, local_did=local_did, reason=reason)
        except SharingError as e:
            return {"success": False, "error": str(e)}

        return {
            "success": True,
            "share_id": result.share_id,
            "consent_chain_id": result.consent_chain_id,
            "revoked": True,
            "reason": result.reason,
        }
