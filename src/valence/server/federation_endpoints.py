"""Federation discovery endpoints for Valence.

Provides well-known endpoints for federation node discovery:
- /.well-known/vfp-node-metadata - Node DID document
- /.well-known/vfp-trust-anchors - Trusted nodes list (optional)

Security: Federation protocol endpoints require DID signature verification.
Discovery endpoints (/.well-known/*) are public by design.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from .config import get_settings
from .errors import (
    missing_field_error,
    invalid_json_error,
    auth_error,
    not_found_error,
    feature_not_enabled_error,
    internal_error,
    AUTH_SIGNATURE_FAILED,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DID SIGNATURE VERIFICATION MIDDLEWARE
# =============================================================================


class DIDSignatureError(Exception):
    """Error during DID signature verification."""
    pass


async def verify_did_signature(request: Request) -> dict[str, Any] | None:
    """Verify the DID signature on a federation request.

    Expects headers:
    - X-VFP-DID: The sender's DID
    - X-VFP-Signature: Base64-encoded Ed25519 signature
    - X-VFP-Timestamp: Unix timestamp (for replay protection)
    - X-VFP-Nonce: Random nonce (for replay protection)

    The signature covers: method + path + timestamp + nonce + body_hash

    Returns:
        Dict with verified DID info, or None if verification fails
    """
    did = request.headers.get("X-VFP-DID")
    signature_b64 = request.headers.get("X-VFP-Signature")
    timestamp_str = request.headers.get("X-VFP-Timestamp")
    nonce = request.headers.get("X-VFP-Nonce")

    if not all([did, signature_b64, timestamp_str, nonce]):
        return None

    try:
        timestamp = int(timestamp_str)
    except ValueError:
        return None

    # Check timestamp freshness (within 5 minutes)
    now = int(time.time())
    if abs(now - timestamp) > 300:
        logger.warning(f"DID signature timestamp too old/future: {timestamp} vs {now}")
        return None

    # Get the request body for hashing
    body = await request.body()
    body_hash = hashlib.sha256(body).hexdigest()

    # Construct the message that was signed
    # Format: METHOD PATH TIMESTAMP NONCE BODYHASH
    message = f"{request.method} {request.url.path} {timestamp} {nonce} {body_hash}"
    message_bytes = message.encode("utf-8")

    # Resolve the DID to get the public key
    try:
        from ..federation.identity import parse_did, resolve_did_sync, verify_signature

        parsed_did = parse_did(did)
        did_document = resolve_did_sync(parsed_did)

        if not did_document:
            logger.warning(f"Could not resolve DID: {did}")
            return None

        # Get the public key from the DID document
        public_key = did_document.get_public_key()
        if not public_key:
            logger.warning(f"No public key in DID document: {did}")
            return None

        # Verify the signature
        signature = base64.b64decode(signature_b64)
        if not verify_signature(message_bytes, signature, public_key):
            logger.warning(f"Invalid DID signature from: {did}")
            return None

        return {
            "did": did,
            "did_document": did_document,
            "timestamp": timestamp,
            "nonce": nonce,
        }

    except Exception as e:
        logger.warning(f"DID signature verification error: {e}")
        return None


def require_did_signature(handler: Callable) -> Callable:
    """Decorator that requires valid DID signature for federation endpoints.

    Usage:
        @require_did_signature
        async def my_handler(request: Request) -> JSONResponse:
            # request.state.did_info contains verified DID info
            ...
    """
    @wraps(handler)
    async def wrapper(request: Request) -> JSONResponse:
        settings = get_settings()

        # Skip verification if federation is disabled
        if not settings.federation_enabled:
            return feature_not_enabled_error("Federation")

        # Verify DID signature
        did_info = await verify_did_signature(request)
        if not did_info:
            return auth_error(
                "DID signature verification failed. Required headers: X-VFP-DID, X-VFP-Signature, X-VFP-Timestamp, X-VFP-Nonce",
                code=AUTH_SIGNATURE_FAILED
            )

        # Attach verified DID info to request state
        request.state.did_info = did_info

        return await handler(request)

    return wrapper


async def vfp_node_metadata(request: Request) -> JSONResponse:
    """Return the node's DID document.

    This is the primary discovery endpoint for federation.
    Returns a DID Document conforming to the W3C DID Core spec with
    VFP-specific extensions.

    Endpoint: GET /.well-known/vfp-node-metadata
    """
    settings = get_settings()

    # Check if federation is enabled
    if not settings.federation_enabled:
        return feature_not_enabled_error("Federation")

    # Build DID document from configuration
    did_document = _build_did_document(settings)

    return JSONResponse(
        did_document,
        headers={
            "Content-Type": "application/did+ld+json",
            "Cache-Control": "max-age=3600",  # Cache for 1 hour
        },
    )


async def vfp_trust_anchors(request: Request) -> JSONResponse:
    """Return the node's trust anchors (optional).

    Lists nodes that this node explicitly trusts.
    Only available if the node chooses to publish its trust anchors.

    Endpoint: GET /.well-known/vfp-trust-anchors
    """
    settings = get_settings()

    # Check if federation is enabled
    if not settings.federation_enabled:
        return feature_not_enabled_error("Federation")

    # Check if trust anchors are published
    if not settings.federation_publish_trust_anchors:
        return feature_not_enabled_error("Trust anchors publishing")

    # Get trust anchors from database
    trust_anchors = _get_trust_anchors()

    return JSONResponse(
        {
            "trust_anchors": trust_anchors,
            "updated_at": datetime.now().isoformat(),
        },
        headers={
            "Content-Type": "application/json",
            "Cache-Control": "max-age=3600",  # Cache for 1 hour
        },
    )


def _build_did_document(settings: Any) -> dict[str, Any]:
    """Build DID document from settings.

    Args:
        settings: Application settings

    Returns:
        DID Document as dictionary
    """
    from ..federation.identity import (
        DIDDocument,
        VerificationMethod,
        ServiceEndpoint,
        create_web_did,
        create_did_document,
    )

    # Determine the node's DID
    if settings.federation_node_did:
        did = settings.federation_node_did
    elif settings.external_url:
        # Derive from external URL
        from urllib.parse import urlparse
        parsed = urlparse(settings.external_url)
        did = f"did:vkb:web:{parsed.netloc}"
    else:
        # Fallback to localhost (not useful for federation)
        did = f"did:vkb:web:localhost:{settings.port}"

    # Build the DID document
    doc_dict = {
        "@context": [
            "https://www.w3.org/ns/did/v1",
            "https://valence.dev/ns/vfp/v1",
        ],
        "id": did,
    }

    # Add verification method if public key is configured
    if settings.federation_public_key:
        vm_id = f"{did}#keys-1"
        doc_dict["verificationMethod"] = [{
            "id": vm_id,
            "type": "Ed25519VerificationKey2020",
            "controller": did,
            "publicKeyMultibase": settings.federation_public_key,
        }]
        doc_dict["authentication"] = [vm_id]
        doc_dict["assertionMethod"] = [vm_id]

    # Add service endpoints
    services = []

    # Federation endpoint
    if settings.external_url:
        services.append({
            "id": f"{did}#vfp",
            "type": "ValenceFederationProtocol",
            "serviceEndpoint": f"{settings.external_url}/federation",
        })

        # MCP endpoint
        services.append({
            "id": f"{did}#mcp",
            "type": "ModelContextProtocol",
            "serviceEndpoint": f"{settings.external_url}/mcp",
        })

    if services:
        doc_dict["service"] = services

    # VFP-specific extensions
    doc_dict["vfp:capabilities"] = settings.federation_capabilities or ["belief_sync"]
    doc_dict["vfp:protocolVersion"] = "1.0"

    # Optional profile
    profile = {}
    if settings.federation_node_name:
        profile["name"] = settings.federation_node_name
    if settings.federation_domains:
        profile["domains"] = settings.federation_domains
    if profile:
        doc_dict["vfp:profile"] = profile

    return doc_dict


def _get_trust_anchors() -> list[dict[str, Any]]:
    """Get trust anchors from database.

    Returns:
        List of trust anchor entries
    """
    try:
        from ..core.db import get_cursor

        with get_cursor() as cur:
            cur.execute("""
                SELECT
                    fn.did,
                    fn.trust_phase,
                    fn.domains,
                    nt.trust->>'overall' as trust_overall,
                    unt.trust_preference,
                    nt.relationship_started_at
                FROM federation_nodes fn
                LEFT JOIN node_trust nt ON fn.id = nt.node_id
                LEFT JOIN user_node_trust unt ON fn.id = unt.node_id
                WHERE fn.status = 'active'
                  AND fn.trust_phase = 'anchor'
                  OR unt.trust_preference = 'anchor'
                ORDER BY COALESCE((nt.trust->>'overall')::numeric, 0) DESC
                LIMIT 100
            """)

            rows = cur.fetchall()

            return [
                {
                    "did": row["did"],
                    "trust_level": row["trust_phase"],
                    "domains": row["domains"] or [],
                    "trust_score": float(row["trust_overall"]) if row["trust_overall"] else None,
                    "relationship_started_at": row["relationship_started_at"].isoformat() if row["relationship_started_at"] else None,
                }
                for row in rows
            ]

    except Exception as e:
        logger.warning(f"Error fetching trust anchors: {e}")
        return []


# =============================================================================
# FEDERATION API ENDPOINTS (Authenticated)
# =============================================================================


async def federation_status(request: Request) -> JSONResponse:
    """Get federation status for this node.

    Returns information about:
    - Node identity
    - Connected peers
    - Sync status
    - Trust statistics

    Endpoint: GET /federation/status
    Requires authentication.
    """
    settings = get_settings()

    if not settings.federation_enabled:
        return feature_not_enabled_error("Federation")

    # Get federation statistics
    stats = _get_federation_stats()

    return JSONResponse({
        "node": {
            "did": settings.federation_node_did or _derive_did(settings),
            "name": settings.federation_node_name,
            "capabilities": settings.federation_capabilities or ["belief_sync"],
            "protocol_version": "1.0",
        },
        "federation": stats,
    })


def _derive_did(settings: Any) -> str:
    """Derive DID from settings."""
    if settings.external_url:
        from urllib.parse import urlparse
        parsed = urlparse(settings.external_url)
        return f"did:vkb:web:{parsed.netloc}"
    return f"did:vkb:web:localhost:{settings.port}"


def _get_federation_stats() -> dict[str, Any]:
    """Get federation statistics from database."""
    try:
        from ..core.db import get_cursor

        with get_cursor() as cur:
            # Get node counts by status
            cur.execute("""
                SELECT status, COUNT(*) as count
                FROM federation_nodes
                GROUP BY status
            """)
            nodes_by_status = {row["status"]: row["count"] for row in cur.fetchall()}

            # Get sync statistics
            cur.execute("""
                SELECT
                    COUNT(*) as total_syncing,
                    SUM(beliefs_sent) as total_beliefs_sent,
                    SUM(beliefs_received) as total_beliefs_received
                FROM sync_state
                WHERE status IN ('idle', 'syncing')
            """)
            sync_row = cur.fetchone()

            # Get belief counts
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE is_local = TRUE) as local_beliefs,
                    COUNT(*) FILTER (WHERE is_local = FALSE) as federated_beliefs
                FROM beliefs
                WHERE status = 'active'
            """)
            belief_row = cur.fetchone()

            return {
                "nodes": {
                    "total": sum(nodes_by_status.values()),
                    "by_status": nodes_by_status,
                },
                "sync": {
                    "active_peers": sync_row["total_syncing"] if sync_row else 0,
                    "beliefs_sent": sync_row["total_beliefs_sent"] if sync_row else 0,
                    "beliefs_received": sync_row["total_beliefs_received"] if sync_row else 0,
                },
                "beliefs": {
                    "local": belief_row["local_beliefs"] if belief_row else 0,
                    "federated": belief_row["federated_beliefs"] if belief_row else 0,
                },
            }

    except Exception as e:
        logger.warning(f"Error fetching federation stats: {e}")
        return {
            "nodes": {"total": 0, "by_status": {}},
            "sync": {"active_peers": 0, "beliefs_sent": 0, "beliefs_received": 0},
            "beliefs": {"local": 0, "federated": 0},
            "error": str(e),
        }


# =============================================================================
# FEDERATION PROTOCOL ENDPOINTS (Node-to-Node)
# =============================================================================


@require_did_signature
async def federation_protocol(request: Request) -> JSONResponse:
    """Main federation protocol endpoint.

    Handles all VFP protocol messages between nodes:
    - AUTH_CHALLENGE / AUTH_RESPONSE
    - SHARE_BELIEF / ACKNOWLEDGE_BELIEF
    - REQUEST_BELIEFS / BELIEFS_RESPONSE
    - SYNC_REQUEST / SYNC_RESPONSE
    - etc.

    Endpoint: POST /federation/protocol
    Requires: Valid DID signature (X-VFP-DID, X-VFP-Signature, X-VFP-Timestamp, X-VFP-Nonce headers)
    """
    settings = get_settings()

    # Note: federation_enabled check is done by @require_did_signature

    try:
        body = await request.json()

        # Validate message structure
        message_type = body.get("type")
        if not message_type:
            return missing_field_error("message type")

        # Import protocol handler
        from ..federation.protocol import parse_message, handle_message

        # Parse and handle the message
        message = parse_message(body)
        if not message:
            return not_found_error(f"Message type '{message_type}'")

        # Handle the message
        response = await handle_message(message)

        return JSONResponse(response.to_dict() if hasattr(response, 'to_dict') else response)

    except json.JSONDecodeError:
        return invalid_json_error()
    except Exception as e:
        logger.exception("Error handling federation protocol message")
        return internal_error(str(e))


# =============================================================================
# NODE MANAGEMENT ENDPOINTS
# =============================================================================


async def federation_nodes_list(request: Request) -> JSONResponse:
    """List federation nodes.

    Query parameters:
    - status: Filter by status (discovered, connecting, active, suspended, unreachable)
    - trust_phase: Filter by phase (observer, contributor, participant, anchor)
    - limit: Maximum results (default 50)

    Endpoint: GET /federation/nodes
    """
    settings = get_settings()

    if not settings.federation_enabled:
        return feature_not_enabled_error("Federation")

    from ..federation.tools import federation_node_list

    # Get query parameters
    status = request.query_params.get("status")
    trust_phase = request.query_params.get("trust_phase")
    limit = int(request.query_params.get("limit", 50))

    result = federation_node_list(
        status=status,
        trust_phase=trust_phase,
        include_trust=True,
        limit=limit,
    )

    return JSONResponse(result)


async def federation_nodes_get(request: Request) -> JSONResponse:
    """Get a specific federation node.

    Endpoint: GET /federation/nodes/{node_id}
    """
    settings = get_settings()

    if not settings.federation_enabled:
        return feature_not_enabled_error("Federation")

    from ..federation.tools import federation_node_get

    node_id = request.path_params.get("node_id")
    result = federation_node_get(node_id=node_id, include_sync_state=True)

    if not result.get("success"):
        return not_found_error("Federation node", code="NOT_FOUND_NODE")

    return JSONResponse(result)


@require_did_signature
async def federation_nodes_discover(request: Request) -> JSONResponse:
    """Discover and register a federation node.

    Body:
    - url_or_did: Node URL or DID to discover
    - auto_register: Whether to auto-register (default true)

    Endpoint: POST /federation/nodes/discover
    Requires: Valid DID signature
    """
    settings = get_settings()

    try:
        body = await request.json()
        url_or_did = body.get("url_or_did")
        if not url_or_did:
            return missing_field_error("url_or_did")

        from ..federation.tools import federation_node_discover

        result = federation_node_discover(
            url_or_did=url_or_did,
            auto_register=body.get("auto_register", True),
        )

        return JSONResponse(result)

    except json.JSONDecodeError:
        return invalid_json_error()


# =============================================================================
# TRUST MANAGEMENT ENDPOINTS
# =============================================================================


async def federation_trust_get(request: Request) -> JSONResponse:
    """Get trust information for a node.

    Endpoint: GET /federation/nodes/{node_id}/trust
    """
    settings = get_settings()

    if not settings.federation_enabled:
        return feature_not_enabled_error("Federation")

    from ..federation.tools import federation_trust_get as get_trust

    node_id = request.path_params.get("node_id")
    domain = request.query_params.get("domain")
    include_details = request.query_params.get("details", "false").lower() == "true"

    result = get_trust(
        node_id=node_id,
        domain=domain,
        include_details=include_details,
    )

    if not result.get("success"):
        return not_found_error("Federation node trust", code="NOT_FOUND_NODE")

    return JSONResponse(result)


@require_did_signature
async def federation_trust_set(request: Request) -> JSONResponse:
    """Set trust preference for a node.

    Body:
    - preference: blocked, reduced, automatic, elevated, anchor
    - manual_score: Optional manual override (0.0-1.0)
    - reason: Optional reason
    - domain: Optional domain-specific override

    Endpoint: POST /federation/nodes/{node_id}/trust
    Requires: Valid DID signature
    """
    settings = get_settings()

    try:
        body = await request.json()
        node_id = request.path_params.get("node_id")

        preference = body.get("preference")
        if not preference:
            return missing_field_error("preference")

        from ..federation.tools import federation_trust_set_preference

        result = federation_trust_set_preference(
            node_id=node_id,
            preference=preference,
            manual_score=body.get("manual_score"),
            reason=body.get("reason"),
            domain=body.get("domain"),
        )

        return JSONResponse(result)

    except json.JSONDecodeError:
        return invalid_json_error()


# =============================================================================
# SYNC MANAGEMENT ENDPOINTS
# =============================================================================


async def federation_sync_status(request: Request) -> JSONResponse:
    """Get sync status.

    Endpoint: GET /federation/sync
    """
    settings = get_settings()

    if not settings.federation_enabled:
        return feature_not_enabled_error("Federation")

    from ..federation.tools import federation_sync_status as get_sync_status

    node_id = request.query_params.get("node_id")
    include_history = request.query_params.get("history", "false").lower() == "true"

    result = get_sync_status(
        node_id=node_id,
        include_history=include_history,
    )

    return JSONResponse(result)


@require_did_signature
async def federation_sync_trigger(request: Request) -> JSONResponse:
    """Trigger synchronization.

    Body (optional):
    - node_id: Specific node to sync with

    Endpoint: POST /federation/sync
    Requires: Valid DID signature
    """
    settings = get_settings()

    try:
        body = await request.json() if await request.body() else {}
        node_id = body.get("node_id")

        from ..federation.tools import federation_sync_trigger as trigger_sync

        result = trigger_sync(node_id=node_id)

        return JSONResponse(result)

    except json.JSONDecodeError:
        return invalid_json_error()


# =============================================================================
# BELIEF FEDERATION ENDPOINTS
# =============================================================================


@require_did_signature
async def federation_belief_share(request: Request) -> JSONResponse:
    """Share a belief with the federation.

    Body:
    - belief_id: UUID of belief to share
    - visibility: trusted, federated, public (default: federated)
    - share_level: belief_only, with_provenance, full (default: belief_only)
    - target_nodes: Optional list of specific node IDs

    Endpoint: POST /federation/beliefs/share
    Requires: Valid DID signature
    """
    settings = get_settings()

    try:
        body = await request.json()
        belief_id = body.get("belief_id")
        if not belief_id:
            return missing_field_error("belief_id")

        from ..federation.tools import federation_belief_share as share_belief

        result = share_belief(
            belief_id=belief_id,
            visibility=body.get("visibility", "federated"),
            share_level=body.get("share_level", "belief_only"),
            target_nodes=body.get("target_nodes"),
        )

        return JSONResponse(result)

    except json.JSONDecodeError:
        return invalid_json_error()


@require_did_signature
async def federation_belief_query(request: Request) -> JSONResponse:
    """Query beliefs across the federation.

    Body:
    - query: Natural language search query
    - domain_filter: Optional domain filter
    - min_trust: Minimum trust for sources (default: 0.3)
    - include_local: Include local beliefs (default: true)
    - limit: Maximum results (default: 20)

    Endpoint: POST /federation/beliefs/query
    Requires: Valid DID signature
    """
    settings = get_settings()

    try:
        body = await request.json()
        query = body.get("query")
        if not query:
            return missing_field_error("query")

        from ..federation.tools import federation_belief_query as query_beliefs

        result = query_beliefs(
            query=query,
            domain_filter=body.get("domain_filter"),
            min_trust=body.get("min_trust", 0.3),
            include_local=body.get("include_local", True),
            limit=body.get("limit", 20),
        )

        return JSONResponse(result)

    except json.JSONDecodeError:
        return invalid_json_error()


@require_did_signature
async def federation_corroboration_check(request: Request) -> JSONResponse:
    """Check corroboration for a belief.

    Body:
    - belief_id: UUID of belief to check (or)
    - content: Belief content to check
    - domain: Optional domain context

    Endpoint: POST /federation/beliefs/corroboration
    Requires: Valid DID signature
    """
    settings = get_settings()

    try:
        body = await request.json()
        belief_id = body.get("belief_id")
        content = body.get("content")

        if not belief_id and not content:
            return missing_field_error("belief_id or content")

        from ..federation.tools import federation_corroboration_check as check_corroboration

        result = check_corroboration(
            belief_id=belief_id,
            content=content,
            domain=body.get("domain"),
        )

        return JSONResponse(result)

    except json.JSONDecodeError:
        return invalid_json_error()


# =============================================================================
# ROUTES FOR REGISTRATION
# =============================================================================

# These are exported for registration in the main app

FEDERATION_ROUTES = [
    # ==========================================================================
    # DISCOVERY ENDPOINTS (Public - no auth required)
    # These MUST remain public for federation discovery to work
    # ==========================================================================
    ("/.well-known/vfp-node-metadata", vfp_node_metadata, ["GET"]),
    ("/.well-known/vfp-trust-anchors", vfp_trust_anchors, ["GET"]),

    # ==========================================================================
    # FEDERATION API ENDPOINTS
    # Read endpoints require standard OAuth/bearer auth (handled by caller)
    # Write endpoints require DID signature verification (@require_did_signature)
    # ==========================================================================

    # Status (read-only, OAuth auth from caller)
    ("/federation/status", federation_status, ["GET"]),

    # Protocol endpoint (DID signature required via decorator)
    ("/federation/protocol", federation_protocol, ["POST"]),

    # Node management
    ("/federation/nodes", federation_nodes_list, ["GET"]),  # OAuth auth from caller
    ("/federation/nodes/discover", federation_nodes_discover, ["POST"]),  # DID signature required
    ("/federation/nodes/{node_id}", federation_nodes_get, ["GET"]),  # OAuth auth from caller

    # Trust management
    ("/federation/nodes/{node_id}/trust", federation_trust_get, ["GET"]),  # OAuth auth from caller
    ("/federation/nodes/{node_id}/trust", federation_trust_set, ["POST"]),  # DID signature required

    # Sync management
    ("/federation/sync", federation_sync_status, ["GET"]),  # OAuth auth from caller
    ("/federation/sync", federation_sync_trigger, ["POST"]),  # DID signature required

    # Belief federation (all POST endpoints require DID signature)
    ("/federation/beliefs/share", federation_belief_share, ["POST"]),  # DID signature required
    ("/federation/beliefs/query", federation_belief_query, ["POST"]),  # DID signature required
    ("/federation/beliefs/corroboration", federation_corroboration_check, ["POST"]),  # DID signature required
]
