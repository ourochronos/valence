"""Federation MCP tool definitions and implementations.

These tools enable Claude to interact with the federation layer:
- Node management (discover, register, list)
- Belief federation (share, query, pull)
- Trust management (set preference, get status)
- Sync control (trigger, status)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from mcp.types import Tool

from .models import (
    Visibility,
    ShareLevel,
    TrustPreference,
    TrustPhase,
    NodeStatus,
)
from .identity import create_web_did, generate_keypair, DIDDocument
from .discovery import (
    discover_node,
    discover_node_sync,
    register_node,
    get_node_by_did,
    get_node_by_id,
    list_nodes,
    list_nodes_with_trust,
    bootstrap_federation_sync,
)
from .sync import (
    get_sync_state,
    get_sync_status,
    trigger_sync,
    queue_belief_for_sync,
)
from .trust import (
    TrustManager,
    get_trust_manager,
    get_effective_trust,
    process_corroboration,
    process_dispute,
    assess_and_respond_to_threat,
)
from ..core.db import get_cursor

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================


FEDERATION_TOOLS = [
    # -------------------------------------------------------------------------
    # Node Discovery & Management
    # -------------------------------------------------------------------------
    Tool(
        name="federation_node_discover",
        description=(
            "Discover and register a federation node from a URL or DID.\n\n"
            "Use when:\n"
            "- User provides a node URL or DID to connect to\n"
            "- Setting up initial federation connections\n"
            "- Expanding the federation network\n\n"
            "Returns the node's identity information if discovery succeeds."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "url_or_did": {
                    "type": "string",
                    "description": "Node URL (e.g., https://valence.example.com) or DID (e.g., did:vkb:web:example.com)",
                },
                "auto_register": {
                    "type": "boolean",
                    "default": True,
                    "description": "Automatically register the node if discovery succeeds",
                },
            },
            "required": ["url_or_did"],
        },
    ),
    Tool(
        name="federation_node_list",
        description=(
            "List federation nodes with optional filters.\n\n"
            "Use when:\n"
            "- Checking connected nodes\n"
            "- Finding nodes by status or trust phase\n"
            "- Reviewing the federation network"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["discovered", "connecting", "active", "suspended", "unreachable"],
                    "description": "Filter by node status",
                },
                "trust_phase": {
                    "type": "string",
                    "enum": ["observer", "contributor", "participant", "anchor"],
                    "description": "Filter by trust phase",
                },
                "include_trust": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include trust information for each node",
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "description": "Maximum nodes to return",
                },
            },
        },
    ),
    Tool(
        name="federation_node_get",
        description=(
            "Get detailed information about a specific federation node.\n\n"
            "Use when:\n"
            "- Checking a specific node's status\n"
            "- Reviewing trust relationship with a node\n"
            "- Debugging federation issues"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Node UUID",
                },
                "did": {
                    "type": "string",
                    "description": "Node DID (alternative to node_id)",
                },
                "include_sync_state": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include sync state information",
                },
            },
        },
    ),
    Tool(
        name="federation_bootstrap",
        description=(
            "Bootstrap federation by connecting to initial nodes.\n\n"
            "Use when:\n"
            "- Setting up federation for the first time\n"
            "- Reconnecting after network changes\n\n"
            "Accepts a list of node URLs or DIDs to bootstrap from."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "bootstrap_nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of node URLs or DIDs to bootstrap from",
                },
            },
            "required": ["bootstrap_nodes"],
        },
    ),

    # -------------------------------------------------------------------------
    # Trust Management
    # -------------------------------------------------------------------------
    Tool(
        name="federation_trust_get",
        description=(
            "Get effective trust score for a federation node.\n\n"
            "Use when:\n"
            "- Checking how much to trust a node's beliefs\n"
            "- Making decisions about belief weighting\n"
            "- Reviewing trust relationships"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Node UUID",
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain for domain-specific trust",
                },
                "include_details": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include detailed trust dimensions",
                },
            },
            "required": ["node_id"],
        },
    ),
    Tool(
        name="federation_trust_set_preference",
        description=(
            "Set user preference for a federation node's trust.\n\n"
            "Use when:\n"
            "- User wants to block a node\n"
            "- User wants to elevate trust for a known-good node\n"
            "- Setting domain-specific trust preferences\n\n"
            "Preferences: blocked (0), reduced (0.5x), automatic (1x), elevated (1.2x), anchor (1.5x)"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Node UUID",
                },
                "preference": {
                    "type": "string",
                    "enum": ["blocked", "reduced", "automatic", "elevated", "anchor"],
                    "description": "Trust preference level",
                },
                "manual_score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Optional manual trust score override (0.0 to 1.0)",
                },
                "reason": {
                    "type": "string",
                    "description": "Reason for the preference setting",
                },
                "domain": {
                    "type": "string",
                    "description": "Optional domain for domain-specific override",
                },
            },
            "required": ["node_id", "preference"],
        },
    ),
    Tool(
        name="federation_trust_assess",
        description=(
            "Assess threat level for a node based on behavior signals.\n\n"
            "Use when:\n"
            "- Suspicious activity detected from a node\n"
            "- Periodic security review\n"
            "- High dispute rate with a node's beliefs"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Node UUID to assess",
                },
                "apply_response": {
                    "type": "boolean",
                    "default": False,
                    "description": "Automatically apply graduated response based on threat level",
                },
            },
            "required": ["node_id"],
        },
    ),

    # -------------------------------------------------------------------------
    # Belief Federation
    # -------------------------------------------------------------------------
    Tool(
        name="federation_belief_share",
        description=(
            "Mark a belief for sharing with the federation.\n\n"
            "Use when:\n"
            "- User wants to share knowledge with trusted nodes\n"
            "- Publishing research or findings to the network\n\n"
            "Beliefs are shared based on visibility level and will sync to appropriate nodes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "belief_id": {
                    "type": "string",
                    "description": "UUID of the belief to share",
                },
                "visibility": {
                    "type": "string",
                    "enum": ["trusted", "federated", "public"],
                    "default": "federated",
                    "description": "Who can see this belief",
                },
                "share_level": {
                    "type": "string",
                    "enum": ["belief_only", "with_provenance", "full"],
                    "default": "belief_only",
                    "description": "How much information to share",
                },
                "target_nodes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific node IDs to share with (optional)",
                },
            },
            "required": ["belief_id"],
        },
    ),
    Tool(
        name="federation_belief_query",
        description=(
            "Query beliefs across the federation network.\n\n"
            "Use when:\n"
            "- Searching for knowledge from trusted nodes\n"
            "- Checking if others have similar beliefs\n"
            "- Finding corroboration for local beliefs\n\n"
            "Results include trust-weighted relevance from federated nodes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "domain_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by domain path",
                },
                "min_trust": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.3,
                    "description": "Minimum trust score for source nodes",
                },
                "include_local": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include local beliefs in results",
                },
                "limit": {
                    "type": "integer",
                    "default": 20,
                    "description": "Maximum results",
                },
            },
            "required": ["query"],
        },
    ),

    # -------------------------------------------------------------------------
    # Sync Management
    # -------------------------------------------------------------------------
    Tool(
        name="federation_sync_trigger",
        description=(
            "Trigger synchronization with federation nodes.\n\n"
            "Use when:\n"
            "- User wants to force a sync\n"
            "- After sharing new beliefs\n"
            "- Checking for updates from peers"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Specific node to sync with (optional, syncs all active if omitted)",
                },
            },
        },
    ),
    Tool(
        name="federation_sync_status",
        description=(
            "Get synchronization status with federation nodes.\n\n"
            "Use when:\n"
            "- Checking if sync is working\n"
            "- Debugging connectivity issues\n"
            "- Reviewing sync history"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Specific node to check (optional)",
                },
                "include_history": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include recent sync events",
                },
            },
        },
    ),

    # -------------------------------------------------------------------------
    # Corroboration & Endorsement
    # -------------------------------------------------------------------------
    Tool(
        name="federation_corroboration_check",
        description=(
            "Check if a belief has been corroborated by federation nodes.\n\n"
            "Use when:\n"
            "- Verifying a belief's reliability\n"
            "- Checking multi-source agreement\n"
            "- Deciding how much to trust a claim"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "belief_id": {
                    "type": "string",
                    "description": "UUID of the belief to check",
                },
                "content": {
                    "type": "string",
                    "description": "Alternative: belief content to check for corroboration",
                },
                "domain": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Domain to search for corroboration",
                },
            },
        },
    ),
    Tool(
        name="federation_endorsement_give",
        description=(
            "Endorse another federation node.\n\n"
            "Use when:\n"
            "- Vouching for a trusted node\n"
            "- Helping a new node build trust\n"
            "- Acknowledging quality contributions"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "UUID of the node to endorse",
                },
                "dimensions": {
                    "type": "object",
                    "description": "Specific trust dimensions to endorse (e.g., {'belief_accuracy': 0.8})",
                },
                "domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Domains where this node excels",
                },
                "rationale": {
                    "type": "string",
                    "description": "Reason for endorsement",
                },
            },
            "required": ["node_id"],
        },
    ),
]


# =============================================================================
# TOOL HANDLERS
# =============================================================================


def federation_node_discover(
    url_or_did: str,
    auto_register: bool = True,
) -> dict[str, Any]:
    """Discover and optionally register a federation node."""
    try:
        # Discover the node
        did_doc = discover_node_sync(url_or_did)

        if not did_doc:
            return {
                "success": False,
                "error": f"Could not discover node at {url_or_did}",
            }

        result = {
            "success": True,
            "discovered": True,
            "did": did_doc.id,
            "public_key": did_doc.public_key_multibase,
            "services": [s.to_dict() for s in did_doc.services],
            "capabilities": did_doc.capabilities,
            "profile": did_doc.profile,
        }

        # Register if requested
        if auto_register:
            node = register_node(did_doc)
            if node:
                result["registered"] = True
                result["node_id"] = str(node.id)
                result["node_status"] = node.status.value
                result["trust_phase"] = node.trust_phase.value
            else:
                result["registered"] = False
                result["registration_error"] = "Failed to register node"

        return result

    except Exception as e:
        logger.exception(f"Error discovering node {url_or_did}")
        return {
            "success": False,
            "error": str(e),
        }


def federation_node_list(
    status: str | None = None,
    trust_phase: str | None = None,
    include_trust: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    """List federation nodes with optional filters."""
    try:
        node_status = NodeStatus(status) if status else None
        phase = TrustPhase(trust_phase) if trust_phase else None

        if include_trust:
            nodes_with_trust = list_nodes_with_trust()
            nodes = [
                {
                    "node": n.to_dict(),
                    "trust": t.to_dict() if t else None,
                }
                for n, t in nodes_with_trust
                if (node_status is None or n.status == node_status) and
                   (phase is None or n.trust_phase == phase)
            ][:limit]
        else:
            node_list = list_nodes(status=node_status, trust_phase=phase, limit=limit)
            nodes = [n.to_dict() for n in node_list]

        return {
            "success": True,
            "nodes": nodes,
            "count": len(nodes),
        }

    except Exception as e:
        logger.exception("Error listing federation nodes")
        return {
            "success": False,
            "error": str(e),
        }


def federation_node_get(
    node_id: str | None = None,
    did: str | None = None,
    include_sync_state: bool = True,
) -> dict[str, Any]:
    """Get detailed information about a federation node."""
    try:
        node = None
        if node_id:
            node = get_node_by_id(UUID(node_id))
        elif did:
            node = get_node_by_did(did)

        if not node:
            return {
                "success": False,
                "error": "Node not found",
            }

        result = {
            "success": True,
            "node": node.to_dict(),
        }

        # Get trust info
        trust_manager = get_trust_manager()
        node_trust = trust_manager.get_node_trust(node.id)
        if node_trust:
            result["trust"] = node_trust.to_dict()
            result["effective_trust"] = trust_manager.get_effective_trust(node.id)

        # Get user preference
        user_pref = trust_manager.get_user_trust_preference(node.id)
        if user_pref:
            result["user_preference"] = user_pref.to_dict()

        # Get sync state
        if include_sync_state:
            sync_state = get_sync_state(node.id)
            if sync_state:
                result["sync_state"] = sync_state.to_dict()

        return result

    except Exception as e:
        logger.exception("Error getting federation node")
        return {
            "success": False,
            "error": str(e),
        }


def federation_bootstrap(
    bootstrap_nodes: list[str],
) -> dict[str, Any]:
    """Bootstrap federation by connecting to initial nodes."""
    try:
        registered_nodes = bootstrap_federation_sync(bootstrap_nodes)

        return {
            "success": True,
            "registered_count": len(registered_nodes),
            "nodes": [
                {
                    "did": n.did,
                    "id": str(n.id),
                    "status": n.status.value,
                }
                for n in registered_nodes
            ],
        }

    except Exception as e:
        logger.exception("Error bootstrapping federation")
        return {
            "success": False,
            "error": str(e),
        }


def federation_trust_get(
    node_id: str,
    domain: str | None = None,
    include_details: bool = False,
) -> dict[str, Any]:
    """Get effective trust score for a federation node."""
    try:
        uuid = UUID(node_id)
        trust_manager = get_trust_manager()

        effective_trust = trust_manager.get_effective_trust(uuid, domain)

        result = {
            "success": True,
            "node_id": node_id,
            "effective_trust": effective_trust,
        }

        if domain:
            result["domain"] = domain

        if include_details:
            node_trust = trust_manager.get_node_trust(uuid)
            if node_trust:
                result["trust_details"] = node_trust.to_dict()

            user_pref = trust_manager.get_user_trust_preference(uuid)
            if user_pref:
                result["user_preference"] = user_pref.to_dict()

        return result

    except Exception as e:
        logger.exception(f"Error getting trust for node {node_id}")
        return {
            "success": False,
            "error": str(e),
        }


def federation_trust_set_preference(
    node_id: str,
    preference: str,
    manual_score: float | None = None,
    reason: str | None = None,
    domain: str | None = None,
) -> dict[str, Any]:
    """Set user preference for a federation node's trust."""
    try:
        uuid = UUID(node_id)
        pref = TrustPreference(preference)
        trust_manager = get_trust_manager()

        domain_overrides = None
        if domain:
            domain_overrides = {domain: preference}

        user_trust = trust_manager.set_user_preference(
            node_id=uuid,
            preference=pref,
            manual_score=manual_score,
            reason=reason,
            domain_overrides=domain_overrides,
        )

        if user_trust:
            return {
                "success": True,
                "preference": user_trust.to_dict(),
                "effective_trust": trust_manager.get_effective_trust(uuid, domain),
            }
        else:
            return {
                "success": False,
                "error": "Failed to set preference",
            }

    except Exception as e:
        logger.exception(f"Error setting trust preference for node {node_id}")
        return {
            "success": False,
            "error": str(e),
        }


def federation_trust_assess(
    node_id: str,
    apply_response: bool = False,
) -> dict[str, Any]:
    """Assess threat level for a node based on behavior signals."""
    try:
        uuid = UUID(node_id)
        trust_manager = get_trust_manager()

        level, assessment = trust_manager.assess_threat_level(uuid)

        result = {
            "success": True,
            "node_id": node_id,
            "threat_level": level.value,
            "threat_score": assessment["threat_score"],
            "signals": assessment["signals"],
        }

        if apply_response:
            trust_manager.apply_threat_response(uuid, level, assessment)
            result["response_applied"] = True
            result["effective_trust"] = trust_manager.get_effective_trust(uuid)

        return result

    except Exception as e:
        logger.exception(f"Error assessing threat for node {node_id}")
        return {
            "success": False,
            "error": str(e),
        }


def federation_belief_share(
    belief_id: str,
    visibility: str = "federated",
    share_level: str = "belief_only",
    target_nodes: list[str] | None = None,
) -> dict[str, Any]:
    """Mark a belief for sharing with the federation."""
    try:
        uuid = UUID(belief_id)
        vis = Visibility(visibility)
        level = ShareLevel(share_level)

        with get_cursor() as cur:
            # Update belief visibility
            cur.execute("""
                UPDATE beliefs
                SET visibility = %s,
                    share_level = %s,
                    modified_at = NOW()
                WHERE id = %s
                RETURNING id, content, visibility, share_level
            """, (vis.value, level.value, uuid))

            row = cur.fetchone()
            if not row:
                return {
                    "success": False,
                    "error": f"Belief not found: {belief_id}",
                }

            # Queue for sync
            target_ids = [UUID(t) for t in target_nodes] if target_nodes else None
            queue_belief_for_sync(uuid, target_ids)

        return {
            "success": True,
            "belief_id": belief_id,
            "visibility": vis.value,
            "share_level": level.value,
            "queued_for_sync": True,
        }

    except Exception as e:
        logger.exception(f"Error sharing belief {belief_id}")
        return {
            "success": False,
            "error": str(e),
        }


def federation_belief_query(
    query: str,
    domain_filter: list[str] | None = None,
    min_trust: float = 0.3,
    include_local: bool = True,
    limit: int = 20,
) -> dict[str, Any]:
    """Query beliefs across the federation network."""
    try:
        results = []

        with get_cursor() as cur:
            # Query federated beliefs with trust weighting
            if include_local:
                # Include local beliefs
                cur.execute("""
                    SELECT b.*, 1.0 as source_trust, NULL as origin_node_id, TRUE as is_local
                    FROM beliefs b
                    WHERE b.is_local = TRUE
                    AND b.status != 'superseded'
                    AND b.content ILIKE %s
                    ORDER BY b.modified_at DESC
                    LIMIT %s
                """, (f"%{query}%", limit // 2))
                local_rows = cur.fetchall()

                for row in local_rows:
                    results.append({
                        "belief_id": str(row["id"]),
                        "content": row["content"],
                        "confidence": row["confidence"],
                        "domain_path": row.get("domain_path", []),
                        "source_trust": 1.0,
                        "is_local": True,
                    })

            # Query federated beliefs
            cur.execute("""
                SELECT b.*, bp.origin_node_id, fn.did as origin_did,
                       (nt.trust->>'overall')::numeric as source_trust
                FROM beliefs b
                JOIN belief_provenance bp ON b.id = bp.belief_id
                JOIN federation_nodes fn ON bp.origin_node_id = fn.id
                LEFT JOIN node_trust nt ON fn.id = nt.node_id
                WHERE b.is_local = FALSE
                AND b.status != 'superseded'
                AND b.content ILIKE %s
                AND COALESCE((nt.trust->>'overall')::numeric, 0.1) >= %s
                ORDER BY COALESCE((nt.trust->>'overall')::numeric, 0.1) DESC, b.modified_at DESC
                LIMIT %s
            """, (f"%{query}%", min_trust, limit))
            federated_rows = cur.fetchall()

            for row in federated_rows:
                results.append({
                    "belief_id": str(row["id"]),
                    "content": row["content"],
                    "confidence": row["confidence"],
                    "domain_path": row.get("domain_path", []),
                    "source_trust": float(row["source_trust"] or 0.1),
                    "is_local": False,
                    "origin_did": row["origin_did"],
                })

        # Sort by trust-weighted relevance
        results.sort(key=lambda x: x["source_trust"], reverse=True)

        return {
            "success": True,
            "query": query,
            "results": results[:limit],
            "total_count": len(results),
            "local_count": sum(1 for r in results if r["is_local"]),
            "federated_count": sum(1 for r in results if not r["is_local"]),
        }

    except Exception as e:
        logger.exception(f"Error querying federation beliefs")
        return {
            "success": False,
            "error": str(e),
        }


def federation_sync_trigger(
    node_id: str | None = None,
) -> dict[str, Any]:
    """Trigger synchronization with federation nodes."""
    try:
        uuid = UUID(node_id) if node_id else None

        # Run the async trigger function
        result = asyncio.run(trigger_sync(uuid))

        return {
            "success": True,
            **result,
        }

    except Exception as e:
        logger.exception("Error triggering federation sync")
        return {
            "success": False,
            "error": str(e),
        }


def federation_sync_status(
    node_id: str | None = None,
    include_history: bool = False,
) -> dict[str, Any]:
    """Get synchronization status with federation nodes."""
    try:
        if node_id:
            uuid = UUID(node_id)
            sync_state = get_sync_state(uuid)

            if not sync_state:
                return {
                    "success": False,
                    "error": f"No sync state for node {node_id}",
                }

            result = {
                "success": True,
                "node_id": node_id,
                "sync_state": sync_state.to_dict(),
            }

            if include_history:
                with get_cursor() as cur:
                    cur.execute("""
                        SELECT * FROM sync_events
                        WHERE node_id = %s
                        ORDER BY occurred_at DESC
                        LIMIT 20
                    """, (uuid,))
                    events = cur.fetchall()
                    result["recent_events"] = [
                        {
                            "event_type": e["event_type"],
                            "direction": e["direction"],
                            "occurred_at": e["occurred_at"].isoformat(),
                            "details": e.get("details", {}),
                        }
                        for e in events
                    ]

            return result
        else:
            # Get overall sync status
            status = get_sync_status()
            return {
                "success": True,
                **status,
            }

    except Exception as e:
        logger.exception("Error getting sync status")
        return {
            "success": False,
            "error": str(e),
        }


def federation_corroboration_check(
    belief_id: str | None = None,
    content: str | None = None,
    domain: list[str] | None = None,
) -> dict[str, Any]:
    """Check if a belief has been corroborated by federation nodes."""
    try:
        if not belief_id and not content:
            return {
                "success": False,
                "error": "Either belief_id or content must be provided",
            }

        with get_cursor() as cur:
            if belief_id:
                # Get the belief content
                cur.execute("SELECT content, domain_path FROM beliefs WHERE id = %s", (UUID(belief_id),))
                row = cur.fetchone()
                if not row:
                    return {
                        "success": False,
                        "error": f"Belief not found: {belief_id}",
                    }
                content = row["content"]
                domain = domain or row.get("domain_path", [])

            # Search for similar beliefs from other nodes
            cur.execute("""
                SELECT b.id, b.content, bp.origin_node_id, fn.did as origin_did,
                       (nt.trust->>'overall')::numeric as source_trust
                FROM beliefs b
                JOIN belief_provenance bp ON b.id = bp.belief_id
                JOIN federation_nodes fn ON bp.origin_node_id = fn.id
                LEFT JOIN node_trust nt ON fn.id = nt.node_id
                WHERE b.is_local = FALSE
                AND b.status != 'superseded'
                AND b.content ILIKE %s
                LIMIT 20
            """, (f"%{content[:50]}%",))  # Use first 50 chars for matching

            similar_beliefs = cur.fetchall()

            if not similar_beliefs:
                return {
                    "success": True,
                    "corroborated": False,
                    "corroboration_level": 0.0,
                    "participating_nodes": 0,
                    "message": "No corroborating beliefs found",
                }

            # Calculate corroboration level based on number of sources and their trust
            unique_nodes = set(b["origin_did"] for b in similar_beliefs)
            trust_scores = [float(b["source_trust"] or 0.1) for b in similar_beliefs]
            avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0.0

            # Corroboration increases with more independent sources and their trust
            corroboration_level = min(1.0, (len(unique_nodes) * 0.15) + (avg_trust * 0.5))

            return {
                "success": True,
                "corroborated": len(unique_nodes) >= 2,
                "corroboration_level": corroboration_level,
                "participating_nodes": len(unique_nodes),
                "average_source_trust": avg_trust,
                "similar_beliefs": [
                    {
                        "belief_id": str(b["id"]),
                        "origin_did": b["origin_did"],
                        "source_trust": float(b["source_trust"] or 0.1),
                    }
                    for b in similar_beliefs[:5]
                ],
            }

    except Exception as e:
        logger.exception("Error checking corroboration")
        return {
            "success": False,
            "error": str(e),
        }


def federation_endorsement_give(
    node_id: str,
    dimensions: dict[str, float] | None = None,
    domains: list[str] | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    """Endorse another federation node."""
    try:
        from .models import TrustAttestation
        from .identity import get_node_did
        from ..server.config import get_settings

        uuid = UUID(node_id)
        trust_manager = get_trust_manager()

        # Get our node's DID
        settings = get_settings()
        our_did = settings.federation_node_did

        if not our_did:
            return {
                "success": False,
                "error": "Federation not configured - no local node DID",
            }

        # Get the target node
        target_node = get_node_by_id(uuid)
        if not target_node:
            return {
                "success": False,
                "error": f"Node not found: {node_id}",
            }

        # Create attestation
        attestation = TrustAttestation(
            issuer_did=our_did,
            subject_did=target_node.did,
            attestation_type="endorsement",
            attested_dimensions=dimensions or {},
            domains=domains,
        )

        # Process the endorsement
        # Note: In a full implementation, this would also be signed and
        # broadcast to the federation
        node_trust = trust_manager.process_endorsement(
            subject_node_id=uuid,
            endorser_node_id=uuid,  # Self-endorsement in local context
            attestation=attestation,
        )

        # Record endorsement in database
        with get_cursor() as cur:
            cur.execute("""
                UPDATE node_trust
                SET endorsements_received = endorsements_received + 1,
                    last_interaction_at = NOW()
                WHERE node_id = %s
            """, (uuid,))

        return {
            "success": True,
            "endorsed_node": target_node.did,
            "attestation_type": "endorsement",
            "dimensions": dimensions,
            "domains": domains,
            "rationale": rationale,
            "new_effective_trust": trust_manager.get_effective_trust(uuid),
        }

    except Exception as e:
        logger.exception(f"Error endorsing node {node_id}")
        return {
            "success": False,
            "error": str(e),
        }


# =============================================================================
# TOOL ROUTING
# =============================================================================


FEDERATION_TOOL_HANDLERS = {
    "federation_node_discover": federation_node_discover,
    "federation_node_list": federation_node_list,
    "federation_node_get": federation_node_get,
    "federation_bootstrap": federation_bootstrap,
    "federation_trust_get": federation_trust_get,
    "federation_trust_set_preference": federation_trust_set_preference,
    "federation_trust_assess": federation_trust_assess,
    "federation_belief_share": federation_belief_share,
    "federation_belief_query": federation_belief_query,
    "federation_sync_trigger": federation_sync_trigger,
    "federation_sync_status": federation_sync_status,
    "federation_corroboration_check": federation_corroboration_check,
    "federation_endorsement_give": federation_endorsement_give,
}


def handle_federation_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Route a tool call to its handler.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        Tool result dictionary
    """
    handler = FEDERATION_TOOL_HANDLERS.get(name)
    if not handler:
        return {
            "success": False,
            "error": f"Unknown federation tool: {name}",
        }

    return handler(**arguments)
