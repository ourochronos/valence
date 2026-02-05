"""Node Discovery and Registration for Valence Federation.

Handles:
- Discovering nodes from URLs or DIDs
- Registering nodes in the local database
- Bootstrap mechanism for initial network connection
- Periodic health checks for connected nodes
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlparse
from uuid import UUID, uuid4

import aiohttp
import psycopg2

from ..core.db import get_cursor
from .identity import (
    DID,
    DIDDocument,
    parse_did,
    resolve_did,
    WELL_KNOWN_NODE_METADATA,
)
from .models import (
    FederationNode,
    NodeTrust,
    NodeStatus,
    TrustPhase,
    SyncState,
    SyncStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# NODE DISCOVERY
# =============================================================================


async def discover_node(url_or_did: str) -> DIDDocument | None:
    """Discover a federation node from a URL or DID.

    Args:
        url_or_did: Either a DID string (did:vkb:...) or a base URL

    Returns:
        DIDDocument if discovered, None otherwise
    """
    if url_or_did.startswith("did:"):
        # It's a DID - resolve it
        return await resolve_did(url_or_did)
    else:
        # It's a URL - fetch the well-known endpoint
        return await _fetch_node_metadata(url_or_did)


async def _fetch_node_metadata(base_url: str) -> DIDDocument | None:
    """Fetch node metadata from a URL.

    Args:
        base_url: Base URL of the node (e.g., https://valence.example.com)

    Returns:
        DIDDocument if found, None otherwise
    """
    # Normalize URL
    base_url = base_url.rstrip("/")
    if not base_url.startswith("http"):
        base_url = f"https://{base_url}"

    url = f"{base_url}{WELL_KNOWN_NODE_METADATA}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return DIDDocument.from_dict(data)
                else:
                    logger.warning(f"Failed to fetch node metadata from {url}: {response.status}")
                    return None
    except aiohttp.ClientError as e:
        logger.warning(f"Network error fetching node metadata from {url}: {e}")
        return None
    except (ValueError, KeyError, TypeError) as e:
        # ValueError: invalid JSON, KeyError/TypeError: malformed response
        logger.warning(f"Error parsing node metadata from {url}: {e}")
        return None


def discover_node_sync(url_or_did: str) -> DIDDocument | None:
    """Synchronous version of discover_node.

    Uses asyncio.run() to execute the async version.
    """
    try:
        return asyncio.run(discover_node(url_or_did))
    except Exception as e:  # Intentionally broad: asyncio.run can raise various errors
        logger.warning(f"Error in sync discover: {e}")
        return None


# =============================================================================
# NODE REGISTRATION
# =============================================================================


def register_node(did_document: DIDDocument) -> FederationNode | None:
    """Register a discovered node in the local database.

    Args:
        did_document: The node's DID document

    Returns:
        FederationNode if registered, None on error
    """
    did = did_document.id
    public_key = did_document.public_key_multibase

    if not public_key:
        logger.warning(f"Cannot register node {did}: no public key")
        return None

    # Extract endpoints from services
    federation_endpoint = None
    mcp_endpoint = None
    for service in did_document.services:
        if service.type == "ValenceFederationProtocol":
            federation_endpoint = service.service_endpoint
        elif service.type == "ModelContextProtocol":
            mcp_endpoint = service.service_endpoint

    # Extract profile info
    profile = did_document.profile or {}
    name = profile.get("name")
    domains = profile.get("domains", [])

    # Get capabilities
    capabilities = did_document.capabilities or ["belief_sync"]

    try:
        with get_cursor() as cur:
            # Check if node already exists
            cur.execute("SELECT id, status FROM federation_nodes WHERE did = %s", (did,))
            existing = cur.fetchone()

            if existing:
                # Update existing node
                cur.execute("""
                    UPDATE federation_nodes SET
                        federation_endpoint = %s,
                        mcp_endpoint = %s,
                        public_key_multibase = %s,
                        name = %s,
                        domains = %s,
                        capabilities = %s,
                        protocol_version = %s,
                        last_seen_at = NOW(),
                        status = CASE
                            WHEN status = 'unreachable' THEN 'discovered'
                            ELSE status
                        END
                    WHERE did = %s
                    RETURNING *
                """, (
                    federation_endpoint,
                    mcp_endpoint,
                    public_key,
                    name,
                    domains,
                    capabilities,
                    did_document.protocol_version,
                    did,
                ))
                row = cur.fetchone()
                logger.info(f"Updated existing node: {did}")
            else:
                # Insert new node
                cur.execute("""
                    INSERT INTO federation_nodes (
                        did, federation_endpoint, mcp_endpoint,
                        public_key_multibase, name, domains,
                        capabilities, status, trust_phase,
                        protocol_version, discovered_at, last_seen_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    )
                    RETURNING *
                """, (
                    did,
                    federation_endpoint,
                    mcp_endpoint,
                    public_key,
                    name,
                    domains,
                    capabilities,
                    NodeStatus.DISCOVERED.value,
                    TrustPhase.OBSERVER.value,
                    did_document.protocol_version,
                ))
                row = cur.fetchone()
                node_id = row["id"]

                # Initialize trust record
                cur.execute("""
                    INSERT INTO node_trust (node_id, trust)
                    VALUES (%s, '{"overall": 0.1}')
                """, (node_id,))

                # Initialize sync state
                cur.execute("""
                    INSERT INTO sync_state (node_id, status)
                    VALUES (%s, 'idle')
                """, (node_id,))

                logger.info(f"Registered new node: {did}")

            return FederationNode.from_row(row)

    except psycopg2.Error as e:
        logger.exception(f"Database error registering node {did}")
        return None


def get_node_by_did(did: str) -> FederationNode | None:
    """Get a node by its DID.

    Args:
        did: The node's DID

    Returns:
        FederationNode if found, None otherwise
    """
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM federation_nodes WHERE did = %s", (did,))
            row = cur.fetchone()
            if row:
                return FederationNode.from_row(row)
            return None
    except psycopg2.Error as e:
        logger.warning(f"Database error getting node {did}: {e}")
        return None


def get_node_by_id(node_id: UUID) -> FederationNode | None:
    """Get a node by its ID.

    Args:
        node_id: The node's UUID

    Returns:
        FederationNode if found, None otherwise
    """
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM federation_nodes WHERE id = %s", (node_id,))
            row = cur.fetchone()
            if row:
                return FederationNode.from_row(row)
            return None
    except psycopg2.Error as e:
        logger.warning(f"Database error getting node {node_id}: {e}")
        return None


def get_node_trust(node_id: UUID) -> NodeTrust | None:
    """Get trust information for a node.

    Args:
        node_id: The node's UUID

    Returns:
        NodeTrust if found, None otherwise
    """
    try:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM node_trust WHERE node_id = %s", (node_id,))
            row = cur.fetchone()
            if row:
                return NodeTrust.from_row(row)
            return None
    except psycopg2.Error as e:
        logger.warning(f"Database error getting trust for node {node_id}: {e}")
        return None


# =============================================================================
# NODE STATUS MANAGEMENT
# =============================================================================


def update_node_status(node_id: UUID, status: NodeStatus) -> bool:
    """Update a node's status.

    Args:
        node_id: The node's UUID
        status: New status

    Returns:
        True if updated, False on error
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                UPDATE federation_nodes
                SET status = %s, last_seen_at = NOW()
                WHERE id = %s
            """, (status.value, node_id))
            return True
    except psycopg2.Error as e:
        logger.warning(f"Database error updating node status: {e}")
        return False


def mark_node_active(node_id: UUID) -> bool:
    """Mark a node as active after successful connection.

    Args:
        node_id: The node's UUID

    Returns:
        True if updated
    """
    return update_node_status(node_id, NodeStatus.ACTIVE)


def mark_node_unreachable(node_id: UUID) -> bool:
    """Mark a node as unreachable after failed connection.

    Args:
        node_id: The node's UUID

    Returns:
        True if updated
    """
    return update_node_status(node_id, NodeStatus.UNREACHABLE)


# =============================================================================
# BOOTSTRAP
# =============================================================================


async def bootstrap_federation(bootstrap_nodes: list[str]) -> list[FederationNode]:
    """Bootstrap federation by discovering and registering initial nodes.

    Args:
        bootstrap_nodes: List of DIDs or URLs to bootstrap from

    Returns:
        List of successfully registered nodes
    """
    registered = []

    for node_spec in bootstrap_nodes:
        try:
            logger.info(f"Bootstrapping from: {node_spec}")
            did_doc = await discover_node(node_spec)

            if did_doc:
                node = register_node(did_doc)
                if node:
                    registered.append(node)
                    logger.info(f"Bootstrapped node: {node.did}")
            else:
                logger.warning(f"Could not discover node: {node_spec}")

        except Exception as e:  # Intentionally broad: discovery can fail in many ways
            logger.warning(f"Error bootstrapping from {node_spec}: {e}")

    return registered


def bootstrap_federation_sync(bootstrap_nodes: list[str]) -> list[FederationNode]:
    """Synchronous version of bootstrap_federation."""
    try:
        return asyncio.run(bootstrap_federation(bootstrap_nodes))
    except Exception as e:  # Intentionally broad: asyncio.run can raise various errors
        logger.warning(f"Error in sync bootstrap: {e}")
        return []


# =============================================================================
# HEALTH CHECKS
# =============================================================================


async def check_node_health(node: FederationNode) -> bool:
    """Check if a node is healthy and reachable.

    Args:
        node: The node to check

    Returns:
        True if healthy, False otherwise
    """
    if not node.federation_endpoint:
        return False

    try:
        # Try to fetch the node's metadata
        did_doc = await _fetch_node_metadata(node.federation_endpoint.rsplit("/", 1)[0])

        if did_doc and did_doc.id == node.did:
            # Update last seen
            with get_cursor() as cur:
                cur.execute("""
                    UPDATE federation_nodes
                    SET last_seen_at = NOW(),
                        status = 'active'
                    WHERE id = %s
                """, (node.id,))
            return True
        else:
            mark_node_unreachable(node.id)
            return False

    except (aiohttp.ClientError, psycopg2.Error) as e:
        # Network error or database error during health check
        logger.warning(f"Health check failed for {node.did}: {e}")
        mark_node_unreachable(node.id)
        return False


async def check_all_nodes_health() -> dict[str, bool]:
    """Check health of all active nodes.

    Returns:
        Dict mapping node DIDs to health status
    """
    results = {}

    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT * FROM federation_nodes
                WHERE status IN ('active', 'connecting')
            """)
            rows = cur.fetchall()

        nodes = [FederationNode.from_row(row) for row in rows]

        # Check health concurrently
        tasks = [check_node_health(node) for node in nodes]
        healths = await asyncio.gather(*tasks, return_exceptions=True)

        for node, health in zip(nodes, healths):
            if isinstance(health, BaseException):
                results[node.did] = False
            else:
                results[node.did] = health

    except psycopg2.Error as e:
        logger.exception("Database error checking node health")

    return results


# =============================================================================
# NODE LISTING
# =============================================================================


def list_nodes(
    status: NodeStatus | None = None,
    trust_phase: TrustPhase | None = None,
    domains: list[str] | None = None,
    limit: int = 100,
) -> list[FederationNode]:
    """List federation nodes with optional filters.

    Args:
        status: Filter by status
        trust_phase: Filter by trust phase
        domains: Filter by domains (any match)
        limit: Maximum nodes to return

    Returns:
        List of matching nodes
    """
    conditions = []
    params: list[Any] = []

    if status:
        conditions.append("status = %s")
        params.append(status.value)

    if trust_phase:
        conditions.append("trust_phase = %s")
        params.append(trust_phase.value)

    if domains:
        conditions.append("domains && %s")
        params.append(domains)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    try:
        with get_cursor() as cur:
            cur.execute(f"""
                SELECT * FROM federation_nodes
                {where_clause}
                ORDER BY last_seen_at DESC NULLS LAST
                LIMIT %s
            """, params)
            rows = cur.fetchall()
            return [FederationNode.from_row(row) for row in rows]
    except psycopg2.Error as e:
        logger.warning(f"Database error listing nodes: {e}")
        return []


def list_active_nodes() -> list[FederationNode]:
    """List all active nodes."""
    return list_nodes(status=NodeStatus.ACTIVE)


def list_nodes_with_trust() -> list[tuple[FederationNode, NodeTrust | None]]:
    """List all nodes with their trust information.

    Returns:
        List of (node, trust) tuples
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT fn.*, nt.id as trust_id, nt.trust, nt.beliefs_received,
                       nt.beliefs_corroborated, nt.beliefs_disputed,
                       nt.relationship_started_at, nt.last_interaction_at
                FROM federation_nodes fn
                LEFT JOIN node_trust nt ON fn.id = nt.node_id
                ORDER BY (nt.trust->>'overall')::numeric DESC NULLS LAST
            """)
            rows = cur.fetchall()

            results = []
            for row in rows:
                node = FederationNode.from_row(row)

                if row.get("trust_id"):
                    trust = NodeTrust(
                        id=row["trust_id"],
                        node_id=node.id,
                        overall=float(row["trust"].get("overall", 0.1)),
                        beliefs_received=row.get("beliefs_received", 0),
                        beliefs_corroborated=row.get("beliefs_corroborated", 0),
                        beliefs_disputed=row.get("beliefs_disputed", 0),
                        relationship_started_at=row.get("relationship_started_at", datetime.now()),
                        last_interaction_at=row.get("last_interaction_at"),
                    )
                else:
                    trust = None

                results.append((node, trust))

            return results

    except psycopg2.Error as e:
        logger.warning(f"Database error listing nodes with trust: {e}")
        return []


# =============================================================================
# PEER EXCHANGE
# =============================================================================


def get_known_peers() -> list[dict[str, Any]]:
    """Get list of known peers for peer exchange.

    Returns a simplified list suitable for sharing with other nodes.
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT did, federation_endpoint, domains, trust_phase
                FROM federation_nodes
                WHERE status = 'active'
                ORDER BY last_seen_at DESC
                LIMIT 50
            """)
            rows = cur.fetchall()

            return [
                {
                    "did": row["did"],
                    "federation_endpoint": row["federation_endpoint"],
                    "domains": row["domains"],
                    "trust_phase": row["trust_phase"],
                }
                for row in rows
            ]

    except psycopg2.Error as e:
        logger.warning(f"Database error getting known peers: {e}")
        return []


async def exchange_peers(node: FederationNode) -> list[FederationNode]:
    """Exchange peer lists with another node.

    Args:
        node: The node to exchange with

    Returns:
        List of newly discovered nodes
    """
    if not node.federation_endpoint:
        return []

    # TODO: Implement peer exchange protocol
    # This would involve:
    # 1. Sending our known peers to the other node
    # 2. Receiving their known peers
    # 3. Discovering and registering new peers

    return []
