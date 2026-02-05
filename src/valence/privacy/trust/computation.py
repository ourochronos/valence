"""Trust computation functions for delegation and transitive trust.

This module contains algorithms for computing trust through delegation chains:
- compute_delegated_trust: Compute trust through a single delegation hop
- compute_transitive_trust: Compute trust through the entire graph
"""

from __future__ import annotations

from collections import deque

from .edges import TrustEdge


def compute_delegated_trust(
    direct_edge: TrustEdge,
    delegated_edge: TrustEdge,
    remaining_depth: int | None = None,
    respect_delegation: bool = True,
) -> TrustEdge | None:
    """Compute transitive trust through delegation.

    When A trusts B, and B trusts C, this computes A's delegated trust in C.
    The key insight: A's trust in B's judgment affects how much weight
    B's trust in C gets.

    Delegation policy is enforced when respect_delegation=True:
    - If direct_edge.can_delegate is False, returns None (no transitive trust)
    - If delegation_depth limits are exceeded, returns None

    Formula for each dimension:
        delegated_trust = direct_trust * (judgment_weight * delegated_value)

    Where judgment_weight scales B's opinions based on A's trust in B's judgment.

    Args:
        direct_edge: A's trust in B (the intermediary)
        delegated_edge: B's trust in C (what B thinks of C)
        remaining_depth: Remaining delegation hops allowed (None = use edge's depth)
        respect_delegation: If True, enforce can_delegate policy (default True)

    Returns:
        A new TrustEdge representing A's delegated trust in C, or None if
        delegation is not allowed by policy
    """
    # Check delegation policy on the direct edge (only if respecting delegation)
    if respect_delegation and not direct_edge.can_delegate:
        return None  # This edge does not allow transitive trust

    # Check depth limits
    if remaining_depth is not None and remaining_depth <= 0:
        return None  # Exceeded depth limit

    # A's trust in B's judgment determines how much we weight B's opinions
    judgment_weight = direct_edge.judgment

    # For each dimension, delegated trust is:
    # min(A's direct trust in B, B's trust in C) * judgment_weight
    # The min ensures we don't trust C more than we trust B
    # The judgment_weight scales down based on how much we trust B's judgment

    def delegate_dimension(direct_val: float, delegated_val: float) -> float:
        """Compute delegated trust for a single dimension."""
        # Can't trust C more than we trust B on this dimension
        base = min(direct_val, delegated_val)
        # Scale by how much we trust B's judgment
        return base * judgment_weight

    # Compute new delegation depth for the resulting edge
    # The result can delegate if both edges allow it
    result_can_delegate = direct_edge.can_delegate and delegated_edge.can_delegate

    # New depth is the minimum of both, decremented by 1 if either has a limit
    if direct_edge.delegation_depth == 0 and delegated_edge.delegation_depth == 0:
        result_depth = 0  # Both unlimited
    elif direct_edge.delegation_depth == 0:
        result_depth = max(0, delegated_edge.delegation_depth - 1)
    elif delegated_edge.delegation_depth == 0:
        result_depth = max(0, direct_edge.delegation_depth - 1)
    else:
        result_depth = max(0, min(direct_edge.delegation_depth, delegated_edge.delegation_depth) - 1)

    return TrustEdge(
        source_did=direct_edge.source_did,
        target_did=delegated_edge.target_did,
        competence=delegate_dimension(direct_edge.competence, delegated_edge.competence),
        integrity=delegate_dimension(direct_edge.integrity, delegated_edge.integrity),
        confidentiality=delegate_dimension(direct_edge.confidentiality, delegated_edge.confidentiality),
        # For judgment of C, we also apply A's trust in B's judgment
        judgment=delegate_dimension(direct_edge.judgment, delegated_edge.judgment),
        domain=delegated_edge.domain,  # Use the target edge's domain
        can_delegate=result_can_delegate,
        delegation_depth=result_depth,
    )


def compute_transitive_trust(
    source_did: str,
    target_did: str,
    trust_graph: dict[tuple[str, str], TrustEdge],
    max_hops: int = 3,
    respect_delegation: bool = True,
) -> TrustEdge | None:
    """Compute transitive trust through the graph.

    Uses breadth-first search to find trust paths and combines them.
    Judgment dimension affects how much each hop's trust recommendations
    are weighted.

    Delegation policy is enforced when respect_delegation=True:
    - Only edges with can_delegate=True participate in transitive trust
    - delegation_depth limits how far trust can propagate

    Args:
        source_did: Starting DID (the truster)
        target_did: Ending DID (the trustee)
        trust_graph: Dict mapping (source, target) to TrustEdge
        max_hops: Maximum path length to consider
        respect_delegation: If True, only traverse delegatable edges (default True)

    Returns:
        Combined TrustEdge if paths exist, None otherwise
    """
    # Direct trust check
    if (source_did, target_did) in trust_graph:
        return trust_graph[(source_did, target_did)]

    # BFS for paths
    # Queue entries: (current_did, path_of_edges, remaining_depth)
    # remaining_depth tracks the minimum delegation_depth limit along the path
    queue: deque[tuple[str, list[TrustEdge], int | None]] = deque()

    # Find all outgoing edges from source that can delegate (for transitive paths)
    for (src, tgt), edge in trust_graph.items():
        if src == source_did:
            if not respect_delegation or edge.can_delegate:
                # Determine initial remaining depth
                remaining = edge.delegation_depth if edge.delegation_depth > 0 else None
                queue.append((tgt, [edge], remaining))

    found_paths: list[list[TrustEdge]] = []
    visited_at_depth: dict[str, int] = {source_did: 0}

    while queue:
        current_did, path, remaining_depth = queue.popleft()
        current_depth = len(path)

        if current_depth > max_hops:
            continue

        # Found target - check BEFORE depth limit since reaching the target doesn't require another hop
        if current_did == target_did:
            found_paths.append(path)
            continue

        # Check delegation depth limit - only matters for continuing exploration
        if remaining_depth is not None and remaining_depth <= 0:
            continue

        # Explore neighbors
        for (src, tgt), edge in trust_graph.items():
            if src == current_did:
                # Only traverse delegatable edges for intermediate hops
                if respect_delegation and not edge.can_delegate:
                    # But we CAN reach the target if this is the final hop
                    if tgt == target_did:
                        # Final hop doesn't need can_delegate on the last edge
                        next_depth = current_depth + 1
                        if tgt not in visited_at_depth or visited_at_depth[tgt] >= next_depth:
                            visited_at_depth[tgt] = next_depth
                            queue.append((tgt, path + [edge], remaining_depth))
                    continue

                # Avoid cycles and redundant exploration
                next_depth = current_depth + 1
                if tgt not in visited_at_depth or visited_at_depth[tgt] >= next_depth:
                    visited_at_depth[tgt] = next_depth
                    # Update remaining depth
                    if remaining_depth is None:
                        # No limit so far
                        new_remaining = edge.delegation_depth if edge.delegation_depth > 0 else None
                    elif edge.delegation_depth == 0:
                        # This edge has no limit, keep existing limit
                        new_remaining = remaining_depth - 1 if remaining_depth else None
                    else:
                        # Both have limits, take the more restrictive
                        new_remaining = min(remaining_depth, edge.delegation_depth) - 1
                    queue.append((tgt, path + [edge], new_remaining))

    if not found_paths:
        return None

    # Compute delegated trust for each path
    path_trusts: list[TrustEdge] = []
    for path in found_paths:
        # Chain the edges
        result = path[0]
        for next_edge in path[1:]:
            delegated = compute_delegated_trust(result, next_edge, respect_delegation=respect_delegation)
            if delegated is None:
                # Delegation not allowed along this path
                break
            result = delegated
        else:
            # Only add if we successfully chained all edges
            path_trusts.append(result)

    if not path_trusts:
        return None

    # Combine paths: take max of each dimension (optimistic combination)
    # This represents "trust via the best available path"
    return TrustEdge(
        source_did=source_did,
        target_did=target_did,
        competence=max(t.competence for t in path_trusts),
        integrity=max(t.integrity for t in path_trusts),
        confidentiality=max(t.confidentiality for t in path_trusts),
        judgment=max(t.judgment for t in path_trusts),
    )
