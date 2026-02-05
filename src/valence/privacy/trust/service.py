"""High-level trust service API.

This module contains the TrustService class for managing trust relationships,
along with module-level convenience functions.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import datetime

from .computation import compute_delegated_trust
from .edges import TrustEdge, TrustEdge4D
from .graph_store import TrustGraphStore, get_trust_graph_store

logger = logging.getLogger(__name__)


class TrustService:
    """High-level API for managing trust relationships.

    Provides convenient methods for granting, revoking, and querying trust
    relationships. Supports both in-memory storage (for testing) and
    persistent storage via TrustGraphStore.

    Example:
        >>> service = TrustService()
        >>> service.grant_trust(
        ...     source_did="did:key:alice",
        ...     target_did="did:key:bob",
        ...     competence=0.9,
        ...     integrity=0.8,
        ...     confidentiality=0.7,
        ... )
        >>> edge = service.get_trust("did:key:alice", "did:key:bob")
        >>> print(f"Overall trust: {edge.overall_trust:.2f}")

        # List who Alice trusts
        >>> trusted = service.list_trusted("did:key:alice")

        # List who trusts Bob
        >>> trusters = service.list_trusters("did:key:bob")
    """

    def __init__(self, use_memory: bool = False):
        """Initialize the TrustService.

        Args:
            use_memory: If True, use in-memory storage (for testing).
                       If False, use database-backed TrustGraphStore.
        """
        self._use_memory = use_memory
        self._memory_store: dict[tuple[str, str, str], TrustEdge4D] = {}
        self._store: TrustGraphStore | None = None if use_memory else get_trust_graph_store()

    def _make_key(
        self,
        source_did: str,
        target_did: str,
        domain: str | None,
    ) -> tuple[str, str, str]:
        """Create a consistent key for in-memory storage."""
        return (source_did, target_did, domain or "")

    def grant_trust(
        self,
        source_did: str,
        target_did: str,
        competence: float,
        integrity: float,
        confidentiality: float,
        judgment: float = 0.1,
        domain: str | None = None,
        can_delegate: bool = False,
        delegation_depth: int = 0,
        expires_at: datetime | None = None,
    ) -> TrustEdge4D:
        """Grant trust from source to target.

        Creates a new trust edge or updates an existing one.

        Args:
            source_did: DID of the trusting agent
            target_did: DID of the agent being trusted
            competence: Trust in target's ability to perform correctly (0-1)
            integrity: Trust in target's honesty and reliability (0-1)
            confidentiality: Trust in target's ability to keep secrets (0-1)
            judgment: Trust in target's decision-making quality (0-1, default 0.1)
            domain: Optional scope/context for this trust relationship
            can_delegate: Whether trust can be transitively delegated (default False)
            delegation_depth: Maximum delegation chain length (0 = no limit when can_delegate=True)
            expires_at: Optional expiration time

        Returns:
            The created or updated TrustEdge4D

        Raises:
            ValueError: If source_did equals target_did, or scores are invalid
        """
        edge = TrustEdge4D(
            source_did=source_did,
            target_did=target_did,
            competence=competence,
            integrity=integrity,
            confidentiality=confidentiality,
            judgment=judgment,
            domain=domain,
            can_delegate=can_delegate,
            delegation_depth=delegation_depth,
            expires_at=expires_at,
        )

        if self._use_memory:
            key = self._make_key(source_did, target_did, domain)
            existing = self._memory_store.get(key)
            if existing:
                edge.created_at = existing.created_at
                edge.id = existing.id
            self._memory_store[key] = edge
            return edge
        else:
            assert self._store is not None
            return self._store.add_edge(edge)

    def revoke_trust(
        self,
        source_did: str,
        target_did: str,
        domain: str | None = None,
    ) -> bool:
        """Revoke trust from source to target.

        Removes the trust edge if it exists.

        Args:
            source_did: DID of the trusting agent
            target_did: DID whose trust is being revoked
            domain: Optional scope to revoke (None revokes general trust)

        Returns:
            True if an edge was revoked, False if not found
        """
        if self._use_memory:
            key = self._make_key(source_did, target_did, domain)
            if key in self._memory_store:
                del self._memory_store[key]
                return True
            return False
        else:
            assert self._store is not None
            return self._store.delete_edge(source_did, target_did, domain)

    def get_trust(
        self,
        source_did: str,
        target_did: str,
        domain: str | None = None,
    ) -> TrustEdge4D | None:
        """Get the trust edge from source to target.

        When a domain is specified, first checks for a domain-specific edge,
        then falls back to the global (domain=None) edge if not found.
        Domain-scoped trust overrides global trust for that domain.

        Args:
            source_did: DID of the trusting agent
            target_did: DID of the trusted agent
            domain: Optional scope to query. If specified, will check for
                   domain-specific edge first, then fall back to global.

        Returns:
            TrustEdge4D if found and not expired, None otherwise
        """
        if self._use_memory:
            # Try domain-specific edge first (if domain specified)
            if domain is not None:
                key = self._make_key(source_did, target_did, domain)
                edge = self._memory_store.get(key)
                if edge and not edge.is_expired():
                    return edge
                elif edge and edge.is_expired():
                    del self._memory_store[key]

                # Fall back to global edge
                global_key = self._make_key(source_did, target_did, None)
                global_edge = self._memory_store.get(global_key)
                if global_edge and not global_edge.is_expired():
                    return global_edge
                elif global_edge and global_edge.is_expired():
                    del self._memory_store[global_key]
                return None
            else:
                # Just look for global edge
                key = self._make_key(source_did, target_did, None)
                edge = self._memory_store.get(key)
                if edge and not edge.is_expired():
                    return edge
                elif edge and edge.is_expired():
                    del self._memory_store[key]
                return None
        else:
            assert self._store is not None
            # Try domain-specific edge first (if domain specified)
            if domain is not None:
                edge = self._store.get_edge(source_did, target_did, domain)
                if edge and not edge.is_expired():
                    return edge
                elif edge and edge.is_expired():
                    self._store.delete_edge(source_did, target_did, domain)

                # Fall back to global edge
                global_edge = self._store.get_edge(source_did, target_did, None)
                if global_edge and not global_edge.is_expired():
                    return global_edge
                elif global_edge and global_edge.is_expired():
                    self._store.delete_edge(source_did, target_did, None)
                return None
            else:
                # Just look for global edge
                edge = self._store.get_edge(source_did, target_did, None)
                if edge and edge.is_expired():
                    self._store.delete_edge(source_did, target_did, None)
                    return None
                return edge

    def list_trusted(
        self,
        source_did: str,
        domain: str | None = None,
    ) -> list[TrustEdge4D]:
        """List all agents trusted by the source.

        When a domain is specified, returns the effective trust edges for that
        domain context. For each target, returns the domain-specific edge if it
        exists, otherwise returns the global edge. Domain-scoped trust overrides
        global trust for that domain.

        Args:
            source_did: DID of the trusting agent
            domain: Optional scope filter. If specified, returns effective trust
                   for that domain (domain-specific edges override global).
                   If None, returns all edges (both global and domain-scoped).

        Returns:
            List of TrustEdge4D objects (excluding expired)
        """
        if self._use_memory:
            result = []
            expired_keys = []

            if domain is not None:
                # Collect all edges from this source, separating by target
                # For each target: domain-specific edge overrides global
                global_edges: dict[str, TrustEdge4D] = {}  # target -> edge
                domain_edges: dict[str, TrustEdge4D] = {}  # target -> edge

                for key, edge in self._memory_store.items():
                    src, tgt, dom = key
                    if src != source_did:
                        continue

                    if edge.is_expired():
                        expired_keys.append(key)
                        continue

                    if dom == domain:
                        domain_edges[tgt] = edge
                    elif dom == "":  # Global (domain is None stored as "")
                        global_edges[tgt] = edge

                # Domain-specific edges override global for each target
                all_targets = set(global_edges.keys()) | set(domain_edges.keys())
                for target in all_targets:
                    if target in domain_edges:
                        result.append(domain_edges[target])
                    elif target in global_edges:
                        result.append(global_edges[target])
            else:
                # Return all edges (no domain filter)
                for key, edge in self._memory_store.items():
                    src, tgt, dom = key
                    if src != source_did:
                        continue

                    if edge.is_expired():
                        expired_keys.append(key)
                        continue

                    result.append(edge)

            # Clean up expired
            for key in expired_keys:
                del self._memory_store[key]

            return result
        else:
            assert self._store is not None
            if domain is not None:
                # Get both domain-specific and global edges
                domain_edge_list = self._store.get_edges_from(source_did, domain, include_expired=False)
                global_edge_list = self._store.get_edges_from(source_did, None, include_expired=False)

                # Build result: domain-specific overrides global for each target
                domain_by_target: dict[str, TrustEdge4D] = {e.target_did: e for e in domain_edge_list}
                global_by_target: dict[str, TrustEdge4D] = {e.target_did: e for e in global_edge_list}

                db_result: list[TrustEdge4D] = []
                all_targets = set(domain_by_target.keys()) | set(global_by_target.keys())
                for target in all_targets:
                    if target in domain_by_target:
                        db_result.append(domain_by_target[target])
                    elif target in global_by_target:
                        db_result.append(global_by_target[target])

                return db_result
            else:
                # Return all edges (no domain filter)
                return self._store.get_edges_from(source_did, None, include_expired=False)

    def list_trusters(
        self,
        target_did: str,
        domain: str | None = None,
    ) -> list[TrustEdge4D]:
        """List all agents who trust the target.

        Args:
            target_did: DID of the trusted agent
            domain: Optional scope filter (None returns all)

        Returns:
            List of TrustEdge4D objects (excluding expired)
        """
        if self._use_memory:
            result = []
            expired_keys = []

            for key, edge in self._memory_store.items():
                src, tgt, dom = key
                if tgt != target_did:
                    continue

                # Filter by domain if specified
                if domain is not None and (dom or None) != domain:
                    continue

                if edge.is_expired():
                    expired_keys.append(key)
                    continue

                result.append(edge)

            # Clean up expired
            for key in expired_keys:
                del self._memory_store[key]

            return result
        else:
            assert self._store is not None
            return self._store.get_edges_to(target_did, domain, include_expired=False)

    def compute_delegated_trust(
        self,
        source: str,
        target: str,
        domain: str | None = None,
    ) -> TrustEdge4D | None:
        """Compute transitive trust through delegation chains with decay.

        Finds paths from source to target through edges where can_delegate=True,
        applies decay at each hop based on the intermediary's judgment score,
        and respects delegation_depth limits.

        The decay formula at each hop:
            delegated_trust[dim] = min(current[dim], next[dim]) * current.judgment

        This captures the intuition that:
        1. We can't trust the target more than we trust the intermediary
        2. The intermediary's judgment affects how much we weight their recommendation

        Args:
            source: Source DID (the entity seeking trust information)
            target: Target DID (the entity being evaluated)
            domain: Optional domain to scope the trust lookup

        Returns:
            TrustEdge representing delegated trust from source to target,
            or None if no valid delegation path exists.

        Example:
            >>> service = TrustService(use_memory=True)
            >>> # Alice trusts Bob with high judgment, allows delegation depth 2
            >>> edge = TrustEdge(
            ...     source_did="alice", target_did="bob",
            ...     competence=0.8, integrity=0.8, confidentiality=0.8,
            ...     judgment=0.9, can_delegate=True, delegation_depth=2
            ... )
            >>> service._memory_store[("alice", "bob", "")] = edge
            >>> # Bob trusts Carol
            >>> edge2 = TrustEdge(
            ...     source_did="bob", target_did="carol",
            ...     competence=0.9, integrity=0.9, confidentiality=0.9, judgment=0.8
            ... )
            >>> service._memory_store[("bob", "carol", "")] = edge2
            >>> # Compute Alice's delegated trust in Carol
            >>> delegated = service.compute_delegated_trust("alice", "carol")
            >>> # Decay: min(0.8, 0.9) * 0.9 = 0.72 for competence
        """
        # Check for direct trust first
        direct = self.get_trust(source, target, domain)
        if direct is not None:
            return direct

        def get_effective_edges(src_did: str) -> list[TrustEdge4D]:
            """Get edges for the given source, filtered by domain.

            For delegated trust computation:
            - If domain is None: only return global (domain-less) edges
            - If domain is specified: use list_trusted's override behavior
              (domain-specific edges take precedence, with global fallback)
            """
            if domain is None:
                # For global trust computation, only use global edges
                all_edges = self.list_trusted(src_did, None)
                return [e for e in all_edges if e.domain is None]
            else:
                # For domain-specific, use the override behavior
                return self.list_trusted(src_did, domain)

        # BFS to find delegation paths
        # Queue entries: (current_did, path_of_edges, remaining_depth)
        # remaining_depth tracks the minimum delegation_depth remaining in the chain
        # None means no limit (unlimited delegation)
        queue: deque[tuple[str, list[TrustEdge4D], int | None]] = deque()

        # Get initial edges from source that allow delegation
        source_edges = get_effective_edges(source)
        for edge in source_edges:
            if edge.can_delegate:
                # delegation_depth=0 means no limit, otherwise it's the max hops allowed
                initial_depth: int | None = edge.delegation_depth if edge.delegation_depth > 0 else None
                queue.append((edge.target_did, [edge], initial_depth))

        found_paths: list[list[TrustEdge4D]] = []
        # Track visited nodes with (depth, remaining_limit) to avoid inferior paths
        visited: dict[str, tuple[int, int | None]] = {}

        while queue:
            current_did, path, remaining_depth = queue.popleft()
            current_hops = len(path)

            # Found target - collect ALL paths to target (don't prune)
            if current_did == target:
                found_paths.append(path)
                continue

            # Check if we've found a better path to this intermediate node already
            # (only prune intermediate nodes, not the target)
            if current_did in visited:
                prev_hops, prev_remaining = visited[current_did]
                # Skip if we've reached this node in fewer hops
                if prev_hops < current_hops:
                    continue
                # Skip if same hops but better remaining depth
                if prev_hops == current_hops:
                    if prev_remaining is None:  # Unlimited is better
                        continue
                    if remaining_depth is not None and prev_remaining >= remaining_depth:
                        continue

            visited[current_did] = (current_hops, remaining_depth)

            # Can't go further if depth exhausted
            if remaining_depth is not None and remaining_depth <= 0:
                continue

            # Get outgoing edges from current node
            current_edges = get_effective_edges(current_did)
            for edge in current_edges:
                # For intermediate hops, edge must allow delegation
                # For final hop to target, we don't require can_delegate on the last edge
                if edge.target_did == target:
                    # Final hop - doesn't need can_delegate
                    new_remaining = remaining_depth - 1 if remaining_depth is not None else None
                    queue.append((edge.target_did, path + [edge], new_remaining))
                elif edge.can_delegate:
                    # Intermediate hop - must allow delegation
                    # Calculate new remaining depth
                    if remaining_depth is None:
                        # No limit from upstream; use this edge's limit if any
                        new_remaining = edge.delegation_depth if edge.delegation_depth > 0 else None
                    elif edge.delegation_depth == 0:
                        # This edge has no limit; decrement upstream limit
                        new_remaining = remaining_depth - 1
                    else:
                        # Both have limits; take the more restrictive and decrement
                        new_remaining = min(remaining_depth, edge.delegation_depth) - 1

                    # Only continue if we have depth left
                    if new_remaining is None or new_remaining >= 0:
                        queue.append((edge.target_did, path + [edge], new_remaining))

        if not found_paths:
            return None

        # Compute delegated trust for each path with decay
        path_trusts: list[TrustEdge4D] = []
        for path in found_paths:
            # Start with the first edge
            result: TrustEdge | None = path[0]

            # Chain through each subsequent edge, applying decay
            for next_edge in path[1:]:
                if result is None:
                    break
                result = compute_delegated_trust(result, next_edge)

            if result is not None:
                path_trusts.append(result)

        # Combine paths: take max of each dimension (optimistic combination)
        # This represents "trust via the best available path"
        return TrustEdge4D(
            source_did=source,
            target_did=target,
            competence=max(t.competence for t in path_trusts),
            integrity=max(t.integrity for t in path_trusts),
            confidentiality=max(t.confidentiality for t in path_trusts),
            judgment=max(t.judgment for t in path_trusts),
            domain=domain,
        )

    def clear(self) -> int:
        """Clear all trust edges (mainly for testing).

        Returns:
            Number of edges cleared
        """
        if self._use_memory:
            count = len(self._memory_store)
            self._memory_store.clear()
            return count
        else:
            # Note: This deletes ALL edges in the database - use with caution
            from valence.core.db import get_cursor

            with get_cursor() as cur:
                cur.execute("DELETE FROM trust_edges RETURNING id")
                count = len(cur.fetchall())
            logger.warning(f"Cleared all {count} trust edges from database")
            return count


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS (Issue #59)
# =============================================================================


# Default service instance (in-memory for easy testing)
_default_service: TrustService | None = None


def get_trust_service(use_memory: bool = True) -> TrustService:
    """Get the default TrustService instance.

    Args:
        use_memory: If True, use in-memory storage. If False, use database.
                   Only affects the first call (instance is cached).

    Returns:
        The TrustService singleton
    """
    global _default_service
    if _default_service is None:
        _default_service = TrustService(use_memory=use_memory)
    return _default_service


def grant_trust(
    source_did: str,
    target_did: str,
    competence: float,
    integrity: float,
    confidentiality: float,
    judgment: float = 0.1,
    domain: str | None = None,
    can_delegate: bool = False,
    delegation_depth: int = 0,
    expires_at: datetime | None = None,
) -> TrustEdge4D:
    """Grant trust (convenience function using default service)."""
    return get_trust_service().grant_trust(
        source_did=source_did,
        target_did=target_did,
        competence=competence,
        integrity=integrity,
        confidentiality=confidentiality,
        judgment=judgment,
        domain=domain,
        can_delegate=can_delegate,
        delegation_depth=delegation_depth,
        expires_at=expires_at,
    )


def revoke_trust(
    source_did: str,
    target_did: str,
    domain: str | None = None,
) -> bool:
    """Revoke trust (convenience function using default service)."""
    return get_trust_service().revoke_trust(
        source_did=source_did,
        target_did=target_did,
        domain=domain,
    )


def get_trust(
    source_did: str,
    target_did: str,
    domain: str | None = None,
) -> TrustEdge4D | None:
    """Get trust (convenience function using default service)."""
    return get_trust_service().get_trust(
        source_did=source_did,
        target_did=target_did,
        domain=domain,
    )


def list_trusted(
    source_did: str,
    domain: str | None = None,
) -> list[TrustEdge4D]:
    """List trusted agents (convenience function using default service)."""
    return get_trust_service().list_trusted(
        source_did=source_did,
        domain=domain,
    )


def list_trusters(
    target_did: str,
    domain: str | None = None,
) -> list[TrustEdge4D]:
    """List trusters (convenience function using default service)."""
    return get_trust_service().list_trusters(
        target_did=target_did,
        domain=domain,
    )


def compute_delegated_trust_from_service(
    source: str,
    target: str,
    domain: str | None = None,
) -> TrustEdge4D | None:
    """Compute delegated trust through the trust graph (convenience function).

    Finds paths from source to target through delegatable edges, applies
    decay at each hop based on the intermediary's judgment, and respects
    delegation_depth limits.

    Args:
        source: Source DID (the entity seeking trust information)
        target: Target DID (the entity being evaluated)
        domain: Optional domain to scope the trust lookup

    Returns:
        TrustEdge representing delegated trust, or None if no path exists.
    """
    return get_trust_service().compute_delegated_trust(
        source=source,
        target=target,
        domain=domain,
    )
