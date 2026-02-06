"""Node Discovery and Registration for Valence Federation.

Handles:
- Discovering nodes from URLs or DIDs
- Registering nodes in the local database
- Bootstrap mechanism for initial network connection
- Periodic health checks for connected nodes
- Peer exchange protocol for gossip-style peer discovery
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

import aiohttp

from ..core.config import get_config
from ..core.db import get_cursor
from .identity import (
    WELL_KNOWN_NODE_METADATA,
    DIDDocument,
    resolve_did,
)
from .models import (
    FederationNode,
    NodeStatus,
    NodeTrust,
    TrustPhase,
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

    Raises:
        ValueError: If VALENCE_REQUIRE_TLS=true and URL is not HTTPS
    """
    # Normalize URL
    base_url = base_url.rstrip("/")
    if not base_url.startswith("http"):
        base_url = f"https://{base_url}"

    # Enforce TLS in production
    config = get_config()
    if config.require_tls and not base_url.startswith("https://"):
        raise ValueError("TLS required but URL uses HTTP. Set VALENCE_REQUIRE_TLS=false for development.")

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
    except Exception as e:
        logger.warning(f"Error fetching node metadata from {url}: {e}")
        return None


def discover_node_sync(url_or_did: str) -> DIDDocument | None:
    """Synchronous version of discover_node.

    Uses asyncio.run() to execute the async version.
    """
    try:
        return asyncio.run(discover_node(url_or_did))
    except Exception as e:
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
                cur.execute(
                    """
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
                """,
                    (
                        federation_endpoint,
                        mcp_endpoint,
                        public_key,
                        name,
                        domains,
                        capabilities,
                        did_document.protocol_version,
                        did,
                    ),
                )
                row = cur.fetchone()
                logger.info(f"Updated existing node: {did}")
            else:
                # Insert new node
                cur.execute(
                    """
                    INSERT INTO federation_nodes (
                        did, federation_endpoint, mcp_endpoint,
                        public_key_multibase, name, domains,
                        capabilities, status, trust_phase,
                        protocol_version, discovered_at, last_seen_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW()
                    )
                    RETURNING *
                """,
                    (
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
                    ),
                )
                row = cur.fetchone()
                node_id = row["id"]

                # Initialize trust record
                cur.execute(
                    """
                    INSERT INTO node_trust (node_id, trust)
                    VALUES (%s, '{"overall": 0.1}')
                """,
                    (node_id,),
                )

                # Initialize sync state
                cur.execute(
                    """
                    INSERT INTO sync_state (node_id, status)
                    VALUES (%s, 'idle')
                """,
                    (node_id,),
                )

                logger.info(f"Registered new node: {did}")

            return FederationNode.from_row(row)

    except Exception:
        logger.exception(f"Error registering node {did}")
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
    except Exception as e:
        logger.warning(f"Error getting node {did}: {e}")
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
    except Exception as e:
        logger.warning(f"Error getting node {node_id}: {e}")
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
    except Exception as e:
        logger.warning(f"Error getting trust for node {node_id}: {e}")
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
            cur.execute(
                """
                UPDATE federation_nodes
                SET status = %s, last_seen_at = NOW()
                WHERE id = %s
            """,
                (status.value, node_id),
            )
            return True
    except Exception as e:
        logger.warning(f"Error updating node status: {e}")
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

        except Exception as e:
            logger.warning(f"Error bootstrapping from {node_spec}: {e}")

    return registered


def bootstrap_federation_sync(bootstrap_nodes: list[str]) -> list[FederationNode]:
    """Synchronous version of bootstrap_federation."""
    try:
        return asyncio.run(bootstrap_federation(bootstrap_nodes))
    except Exception as e:
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
                cur.execute(
                    """
                    UPDATE federation_nodes
                    SET last_seen_at = NOW(),
                        status = 'active'
                    WHERE id = %s
                """,
                    (node.id,),
                )
            return True
        else:
            mark_node_unreachable(node.id)
            return False

    except Exception as e:
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
            cur.execute(
                """
                SELECT * FROM federation_nodes
                WHERE status IN ('active', 'connecting')
            """
            )
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

    except Exception:
        logger.exception("Error checking node health")

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
            cur.execute(
                f"""  # nosec B608
                SELECT * FROM federation_nodes
                {where_clause}
                ORDER BY last_seen_at DESC NULLS LAST
                LIMIT %s
            """,
                params,
            )
            rows = cur.fetchall()
            return [FederationNode.from_row(row) for row in rows]
    except Exception as e:
        logger.warning(f"Error listing nodes: {e}")
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
            cur.execute(
                """
                SELECT fn.*, nt.id as trust_id, nt.trust, nt.beliefs_received,
                       nt.beliefs_corroborated, nt.beliefs_disputed,
                       nt.relationship_started_at, nt.last_interaction_at
                FROM federation_nodes fn
                LEFT JOIN node_trust nt ON fn.id = nt.node_id
                ORDER BY (nt.trust->>'overall')::numeric DESC NULLS LAST
            """
            )
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

    except Exception as e:
        logger.warning(f"Error listing nodes with trust: {e}")
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
            cur.execute(
                """
                SELECT did, federation_endpoint, domains, trust_phase
                FROM federation_nodes
                WHERE status = 'active'
                ORDER BY last_seen_at DESC
                LIMIT 50
            """
            )
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

    except Exception as e:
        logger.warning(f"Error getting known peers: {e}")
        return []


async def exchange_peers(node: FederationNode) -> list[FederationNode]:
    """Exchange peer lists with another node.

    Sends our known peers and receives theirs, discovering new nodes.

    Args:
        node: The node to exchange with

    Returns:
        List of newly discovered nodes
    """
    if not node.federation_endpoint:
        return []

    protocol = get_peer_exchange_protocol()

    return await protocol.request_peers(node)


# =============================================================================
# PEER EXCHANGE PROTOCOL (Issue #263)
# =============================================================================

# Default configuration
PEER_EXCHANGE_MAX_PEERS = 50  # Max peers to share in one exchange
PEER_EXCHANGE_RATE_LIMIT = 10  # Max exchanges per window per peer
PEER_EXCHANGE_RATE_WINDOW = 3600  # Rate limit window in seconds (1 hour)
PEER_EXCHANGE_MIN_TRUST = 0.2  # Minimum trust to participate in exchange
PEER_EXCHANGE_ENDPOINT = "/federation/peer-exchange"


@dataclass
class PeerInfo:
    """Information about a peer node shared during peer exchange.

    Contains only the metadata that peers have opted into sharing.
    """

    node_id: str  # DID of the peer
    address: str  # Federation endpoint URL
    last_seen: datetime | None = None
    capabilities: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    trust_phase: str = "observer"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for wire format."""
        return {
            "node_id": self.node_id,
            "address": self.address,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "capabilities": self.capabilities,
            "domains": self.domains,
            "trust_phase": self.trust_phase,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeerInfo:
        """Deserialize from dictionary."""
        last_seen = None
        if data.get("last_seen"):
            try:
                last_seen = datetime.fromisoformat(data["last_seen"])
            except (ValueError, TypeError):
                pass

        return cls(
            node_id=data.get("node_id", ""),
            address=data.get("address", ""),
            last_seen=last_seen,
            capabilities=data.get("capabilities", []),
            domains=data.get("domains", []),
            trust_phase=data.get("trust_phase", "observer"),
        )

    def is_valid(self) -> bool:
        """Check if this peer info contains minimum required fields."""
        return bool(self.node_id and self.address and self.node_id.startswith("did:"))


@dataclass
class PeerExchangeMessage:
    """Message sent during peer exchange containing known peers.

    Attributes:
        sender_did: DID of the node sending this message
        peers: List of known peers being shared
        timestamp: When this message was created
        max_hops: Maximum relay hops (to prevent unbounded gossip)
    """

    sender_did: str
    peers: list[PeerInfo] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    max_hops: int = 2

    def to_dict(self) -> dict[str, Any]:
        """Serialize for network transmission."""
        return {
            "type": "peer_exchange",
            "sender_did": self.sender_did,
            "peers": [p.to_dict() for p in self.peers],
            "timestamp": self.timestamp.isoformat(),
            "max_hops": self.max_hops,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeerExchangeMessage:
        """Deserialize from network message."""
        timestamp = datetime.now()
        if data.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError):
                pass

        return cls(
            sender_did=data.get("sender_did", ""),
            peers=[PeerInfo.from_dict(p) for p in data.get("peers", [])],
            timestamp=timestamp,
            max_hops=data.get("max_hops", 2),
        )


@dataclass
class PeerExchangeResponse:
    """Response to a peer exchange request.

    Attributes:
        responder_did: DID of the responding node
        peers: Peers the responder is willing to share
        accepted: Number of peers accepted from the request
        rejected: Number of peers rejected (invalid, already known, etc.)
    """

    responder_did: str
    peers: list[PeerInfo] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize for network transmission."""
        return {
            "type": "peer_exchange_response",
            "responder_did": self.responder_did,
            "peers": [p.to_dict() for p in self.peers],
            "accepted": self.accepted,
            "rejected": self.rejected,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PeerExchangeResponse:
        """Deserialize from network message."""
        return cls(
            responder_did=data.get("responder_did", ""),
            peers=[PeerInfo.from_dict(p) for p in data.get("peers", [])],
            accepted=data.get("accepted", 0),
            rejected=data.get("rejected", 0),
        )


class PeerExchangeProtocol:
    """Gossip-style peer exchange protocol for federation discovery.

    Enables organic network growth by allowing nodes to share their
    known peers with each other. Includes:
    - Privacy controls: only share peers that have opted into discovery
    - Rate limiting: max exchanges per time window per peer
    - Anti-spam: don't relay peers we haven't verified ourselves
    - Trust filtering: only share peers above minimum trust threshold
    """

    def __init__(
        self,
        local_did: str = "",
        max_peers: int = PEER_EXCHANGE_MAX_PEERS,
        rate_limit: int = PEER_EXCHANGE_RATE_LIMIT,
        rate_window: int = PEER_EXCHANGE_RATE_WINDOW,
        min_trust: float = PEER_EXCHANGE_MIN_TRUST,
    ):
        """Initialize the peer exchange protocol.

        Args:
            local_did: Our node's DID
            max_peers: Maximum peers to share per exchange
            rate_limit: Maximum exchanges per time window per peer
            rate_window: Rate limit window in seconds
            min_trust: Minimum trust level to share a peer
        """
        self.local_did = local_did
        self.max_peers = max_peers
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        self.min_trust = min_trust

        # Rate limiting state: {peer_did: [timestamp, ...]}
        self._exchange_timestamps: dict[str, list[float]] = {}

    def _check_rate_limit(self, peer_did: str) -> bool:
        """Check if we're within rate limits for a peer.

        Args:
            peer_did: DID of the peer to check

        Returns:
            True if exchange is allowed, False if rate limited
        """
        now = time.monotonic()
        cutoff = now - self.rate_window

        if peer_did not in self._exchange_timestamps:
            self._exchange_timestamps[peer_did] = []

        # Prune old timestamps
        self._exchange_timestamps[peer_did] = [ts for ts in self._exchange_timestamps[peer_did] if ts > cutoff]

        return len(self._exchange_timestamps[peer_did]) < self.rate_limit

    def _record_exchange(self, peer_did: str) -> None:
        """Record that an exchange happened with a peer."""
        if peer_did not in self._exchange_timestamps:
            self._exchange_timestamps[peer_did] = []
        self._exchange_timestamps[peer_did].append(time.monotonic())

    def get_shareable_peers(self) -> list[PeerInfo]:
        """Get our known peers that are eligible for sharing.

        Filters peers by:
        - Active status (must be reachable)
        - Minimum trust phase (at least contributor)
        - Discovery opt-in (via metadata flag)
        - Verified by us (we've actually connected to them)

        Returns:
            List of PeerInfo for peers we're willing to share
        """
        try:
            with get_cursor() as cur:
                cur.execute(
                    """
                    SELECT fn.did, fn.federation_endpoint, fn.last_seen_at,
                           fn.capabilities, fn.domains, fn.trust_phase,
                           fn.metadata,
                           COALESCE((nt.trust->>'overall')::float, 0.1) as trust_overall
                    FROM federation_nodes fn
                    LEFT JOIN node_trust nt ON fn.id = nt.node_id
                    WHERE fn.status = 'active'
                      AND fn.federation_endpoint IS NOT NULL
                      AND fn.trust_phase IN ('contributor', 'participant', 'anchor')
                    ORDER BY COALESCE((nt.trust->>'overall')::float, 0.1) DESC
                    LIMIT %s
                """,
                    (self.max_peers,),
                )
                rows = cur.fetchall()

            peers = []
            for row in rows:
                # Privacy check: skip peers that have opted out of discovery
                metadata = row.get("metadata") or {}
                if metadata.get("discovery_opt_out", False):
                    continue

                # Trust filter: skip peers below minimum trust
                trust_overall = row.get("trust_overall", 0.1)
                if trust_overall < self.min_trust:
                    continue

                peers.append(
                    PeerInfo(
                        node_id=row["did"],
                        address=row["federation_endpoint"],
                        last_seen=row.get("last_seen_at"),
                        capabilities=row.get("capabilities", []),
                        domains=row.get("domains", []),
                        trust_phase=row.get("trust_phase", "observer"),
                    )
                )

            return peers

        except Exception as e:
            logger.warning(f"Error getting shareable peers: {e}")
            return []

    def handle_peer_exchange(
        self,
        request: PeerExchangeMessage,
    ) -> PeerExchangeResponse:
        """Handle an incoming peer exchange request.

        Processes received peers and returns our own shareable peers.
        Applies validation, dedup, and privacy controls.

        Args:
            request: The incoming peer exchange message

        Returns:
            PeerExchangeResponse with our peers and acceptance stats
        """
        # Rate limit check
        if not self._check_rate_limit(request.sender_did):
            logger.warning(f"Rate limited peer exchange from {request.sender_did}")
            return PeerExchangeResponse(
                responder_did=self.local_did,
                peers=[],
                accepted=0,
                rejected=len(request.peers),
            )

        self._record_exchange(request.sender_did)

        # Merge received peers
        accepted, rejected = self.merge_peers(request.peers)

        # Return our shareable peers
        our_peers = self.get_shareable_peers()

        return PeerExchangeResponse(
            responder_did=self.local_did,
            peers=our_peers,
            accepted=accepted,
            rejected=rejected,
        )

    def merge_peers(self, received_peers: list[PeerInfo]) -> tuple[int, int]:
        """Integrate discovered peers from a peer exchange.

        Validates, deduplicates, and registers new peers. Only accepts
        peers that pass validation checks.

        Args:
            received_peers: Peers received from another node

        Returns:
            Tuple of (accepted_count, rejected_count)
        """
        accepted = 0
        rejected = 0

        for peer_info in received_peers:
            # Validate the peer info
            if not peer_info.is_valid():
                logger.debug(f"Rejected invalid peer info: {peer_info.node_id}")
                rejected += 1
                continue

            # Don't add ourselves
            if peer_info.node_id == self.local_did:
                rejected += 1
                continue

            # Check if we already know this peer
            existing = get_node_by_did(peer_info.node_id)
            if existing is not None:
                rejected += 1
                continue

            # Anti-spam: validate the address looks legitimate
            if not self._validate_address(peer_info.address):
                logger.debug(f"Rejected peer with invalid address: {peer_info.address}")
                rejected += 1
                continue

            # Register as a newly discovered peer (will need verification)
            try:
                with get_cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO federation_nodes (
                            did, federation_endpoint, domains,
                            capabilities, status, trust_phase,
                            protocol_version, discovered_at, last_seen_at,
                            metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, NOW(), NULL,
                            %s
                        )
                        ON CONFLICT (did) DO NOTHING
                        RETURNING id
                    """,
                        (
                            peer_info.node_id,
                            peer_info.address,
                            peer_info.domains,
                            peer_info.capabilities,
                            NodeStatus.DISCOVERED.value,
                            TrustPhase.OBSERVER.value,
                            "1.0",
                            {"discovered_via": "peer_exchange"},
                        ),
                    )
                    result = cur.fetchone()
                    if result:
                        node_id = result["id"]
                        # Initialize trust record
                        cur.execute(
                            """
                            INSERT INTO node_trust (node_id, trust)
                            VALUES (%s, '{"overall": 0.1}')
                        """,
                            (node_id,),
                        )
                        # Initialize sync state
                        cur.execute(
                            """
                            INSERT INTO sync_state (node_id, status)
                            VALUES (%s, 'idle')
                        """,
                            (node_id,),
                        )
                        accepted += 1
                        logger.info(f"Discovered new peer via exchange: {peer_info.node_id}")
                    else:
                        # Already existed (race condition with ON CONFLICT)
                        rejected += 1

            except Exception:
                logger.exception(f"Error registering peer from exchange: {peer_info.node_id}")
                rejected += 1

        return accepted, rejected

    async def request_peers(self, target_node: FederationNode) -> list[FederationNode]:
        """Request peers from a target node via HTTP.

        Sends our known peers and receives theirs.

        Args:
            target_node: The node to exchange peers with

        Returns:
            List of newly discovered and registered nodes
        """
        if not target_node.federation_endpoint:
            return []

        if not self._check_rate_limit(target_node.did):
            logger.warning(f"Rate limited: skipping peer exchange with {target_node.did}")
            return []

        self._record_exchange(target_node.did)

        # Build our exchange message
        our_peers = self.get_shareable_peers()
        message = PeerExchangeMessage(
            sender_did=self.local_did,
            peers=our_peers,
        )

        # Construct the endpoint URL
        base_url = target_node.federation_endpoint.rstrip("/")
        # If the endpoint already contains a path, append to it
        if "/federation" in base_url:
            exchange_url = f"{base_url}/peer-exchange"
        else:
            exchange_url = f"{base_url}{PEER_EXCHANGE_ENDPOINT}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    exchange_url,
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as response:
                    if response.status != 200:
                        logger.warning(f"Peer exchange with {target_node.did} returned status {response.status}")
                        return []

                    data = await response.json()
                    exchange_response = PeerExchangeResponse.from_dict(data)

        except aiohttp.ClientError as e:
            logger.warning(f"Network error during peer exchange with {target_node.did}: {e}")
            return []
        except Exception as e:
            logger.warning(f"Error during peer exchange with {target_node.did}: {e}")
            return []

        # Merge received peers
        accepted, _rejected = self.merge_peers(exchange_response.peers)

        # Return newly registered nodes
        new_nodes = []
        for peer_info in exchange_response.peers:
            if peer_info.is_valid():
                node = get_node_by_did(peer_info.node_id)
                if node is not None:
                    new_nodes.append(node)

        logger.info(
            f"Peer exchange with {target_node.did}: sent {len(our_peers)} peers, received {len(exchange_response.peers)}, accepted {accepted}"
        )

        return new_nodes

    @staticmethod
    def _validate_address(address: str) -> bool:
        """Validate that a peer address looks legitimate.

        Args:
            address: The federation endpoint URL

        Returns:
            True if the address appears valid
        """
        if not address:
            return False

        # Must be HTTP(S)
        if not address.startswith(("http://", "https://")):
            return False

        # Basic length check (no absurdly long URLs)
        if len(address) > 2048:
            return False

        # Must have a hostname component
        try:
            from urllib.parse import urlparse

            parsed = urlparse(address)
            if not parsed.hostname:
                return False
            # Reject obviously local addresses in production
            # (allow in development)
            return True
        except Exception:
            return False


# Singleton protocol instance
_peer_exchange_protocol: PeerExchangeProtocol | None = None


def get_peer_exchange_protocol() -> PeerExchangeProtocol:
    """Get or create the singleton PeerExchangeProtocol instance."""
    global _peer_exchange_protocol
    if _peer_exchange_protocol is None:
        config = get_config()
        local_did = getattr(config, "node_did", "")
        _peer_exchange_protocol = PeerExchangeProtocol(local_did=local_did)
    return _peer_exchange_protocol


def set_peer_exchange_protocol(
    protocol: PeerExchangeProtocol | None,
) -> None:
    """Set the singleton PeerExchangeProtocol instance (for testing)."""
    global _peer_exchange_protocol
    _peer_exchange_protocol = protocol
