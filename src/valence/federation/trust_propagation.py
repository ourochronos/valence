"""Transitive Trust Propagation for Valence Federation.

Implements PageRank-style trust propagation across the federation network.
If A trusts B, and B trusts C, then A has some (decayed) trust in C.

Key concepts:
- Direct trust: The trust A has directly assigned/computed for B
- Transitive trust: Trust propagated through the network
- Decay factor: Trust decreases with each hop (default 0.8)
- Max hops: Maximum propagation distance (default 4)
- Ring coefficient: Dampening applied when trust flows through cycles
  (per THREAT-MODEL.md §1.2.1 - Sybil Network Infiltration)
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeAlias
from uuid import UUID

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_DECAY_FACTOR = 0.8  # Trust retained per hop
DEFAULT_MAX_HOPS = 4        # Maximum propagation depth
DEFAULT_CACHE_TTL = 300     # 5 minutes cache TTL
DEFAULT_MIN_TRUST = 0.01    # Below this, don't propagate
DEFAULT_APPLY_RING_COEFFICIENT = True  # Apply ring dampening by default


# =============================================================================
# TRUST GRAPH EDGE
# =============================================================================


@dataclass
class TrustEdge:
    """An edge in the trust graph representing direct trust from A to B."""
    
    from_node_id: UUID
    to_node_id: UUID
    direct_trust: float  # 0.0 to 1.0
    domain: str | None = None  # Optional domain-specific trust
    
    def __hash__(self) -> int:
        return hash((self.from_node_id, self.to_node_id, self.domain))


@dataclass
class TransitiveTrustResult:
    """Result of transitive trust computation."""
    
    from_node_id: UUID
    to_node_id: UUID
    direct_trust: float | None  # None if no direct relationship
    transitive_trust: float     # Computed transitive trust
    path_count: int             # Number of paths found
    shortest_path_length: int | None  # Hops in shortest path
    computation_time_ms: float
    cached: bool = False
    computed_at: datetime = field(default_factory=datetime.now)
    ring_coefficient_applied: float = 1.0  # Ring dampening (1.0 = no dampening)
    rings_detected: int = 0  # Number of rings in paths
    
    @property
    def effective_trust(self) -> float:
        """Return the higher of direct or transitive trust."""
        if self.direct_trust is not None:
            return max(self.direct_trust, self.transitive_trust)
        return self.transitive_trust
    
    @property
    def undampened_trust(self) -> float:
        """Return what transitive trust would be without ring dampening."""
        if self.ring_coefficient_applied > 0:
            return self.transitive_trust / self.ring_coefficient_applied
        return self.transitive_trust
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "from_node_id": str(self.from_node_id),
            "to_node_id": str(self.to_node_id),
            "direct_trust": self.direct_trust,
            "transitive_trust": self.transitive_trust,
            "effective_trust": self.effective_trust,
            "path_count": self.path_count,
            "shortest_path_length": self.shortest_path_length,
            "computation_time_ms": self.computation_time_ms,
            "cached": self.cached,
            "computed_at": self.computed_at.isoformat(),
            "ring_coefficient_applied": self.ring_coefficient_applied,
            "rings_detected": self.rings_detected,
        }


# =============================================================================
# TRUST CACHE
# =============================================================================


class TrustCache:
    """Simple TTL cache for transitive trust computations."""
    
    def __init__(self, ttl_seconds: int = DEFAULT_CACHE_TTL):
        """Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries
        """
        self._cache: dict[tuple[UUID, UUID, str | None], tuple[TransitiveTrustResult, float]] = {}
        self._ttl = ttl_seconds
    
    def _make_key(self, from_id: UUID, to_id: UUID, domain: str | None) -> tuple[UUID, UUID, str | None]:
        return (from_id, to_id, domain)
    
    def get(
        self,
        from_node_id: UUID,
        to_node_id: UUID,
        domain: str | None = None,
    ) -> TransitiveTrustResult | None:
        """Get cached result if valid.
        
        Returns:
            Cached result or None if not found/expired
        """
        key = self._make_key(from_node_id, to_node_id, domain)
        if key not in self._cache:
            return None
        
        result, cached_at = self._cache[key]
        if time.time() - cached_at > self._ttl:
            # Expired
            del self._cache[key]
            return None
        
        # Return a copy with cached=True
        return TransitiveTrustResult(
            from_node_id=result.from_node_id,
            to_node_id=result.to_node_id,
            direct_trust=result.direct_trust,
            transitive_trust=result.transitive_trust,
            path_count=result.path_count,
            shortest_path_length=result.shortest_path_length,
            computation_time_ms=result.computation_time_ms,
            cached=True,
            computed_at=result.computed_at,
            ring_coefficient_applied=result.ring_coefficient_applied,
            rings_detected=result.rings_detected,
        )
    
    def set(
        self,
        result: TransitiveTrustResult,
        domain: str | None = None,
    ) -> None:
        """Cache a result."""
        key = self._make_key(result.from_node_id, result.to_node_id, domain)
        self._cache[key] = (result, time.time())
    
    def invalidate(self, node_id: UUID) -> int:
        """Invalidate all cache entries involving a node.
        
        Call this when a node's trust changes.
        
        Args:
            node_id: Node whose trust changed
            
        Returns:
            Number of entries invalidated
        """
        to_remove = [
            key for key in self._cache
            if key[0] == node_id or key[1] == node_id
        ]
        for key in to_remove:
            del self._cache[key]
        return len(to_remove)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    @property
    def size(self) -> int:
        """Number of entries in cache."""
        return len(self._cache)


# =============================================================================
# TRUST PROPAGATION ENGINE
# =============================================================================


class TrustPropagation:
    """Engine for computing transitive trust across the federation.
    
    Uses a simplified PageRank-style algorithm:
    1. Build trust graph from direct trust relationships
    2. For each (source, target) pair, find all paths up to max_hops
    3. For each path, compute decayed trust: direct_trust * decay^hops
    4. Apply ring coefficient dampening for paths through cycles
    5. Aggregate path contributions (max or weighted sum)
    
    Ring Coefficient (per THREAT-MODEL.md §1.2.1):
    When trust flows through cycles (A→B→C→A), the ring coefficient
    reduces the propagated trust to prevent Sybil networks from
    accumulating artificial transitive trust through coordinated rings.
    
    Example:
        >>> engine = TrustPropagation(trust_getter=my_trust_getter)
        >>> result = engine.compute_transitive_trust(node_a_id, node_c_id)
        >>> print(f"Transitive trust: {result.transitive_trust}")
        >>> print(f"Ring dampening: {result.ring_coefficient_applied}")
    """
    
    def __init__(
        self,
        trust_getter: TrustGetter | None = None,
        decay_factor: float = DEFAULT_DECAY_FACTOR,
        max_hops: int = DEFAULT_MAX_HOPS,
        min_trust_threshold: float = DEFAULT_MIN_TRUST,
        cache_ttl: int = DEFAULT_CACHE_TTL,
        apply_ring_coefficient: bool = DEFAULT_APPLY_RING_COEFFICIENT,
    ):
        """Initialize the propagation engine.
        
        Args:
            trust_getter: Callable to get direct trust relationships
                         Should return list of (node_id, trust_score) tuples
            decay_factor: Trust multiplier per hop (0-1)
            max_hops: Maximum path length
            min_trust_threshold: Don't propagate below this
            cache_ttl: Cache time-to-live in seconds
            apply_ring_coefficient: Whether to apply ring dampening (default True)
        """
        self.trust_getter = trust_getter or self._default_trust_getter
        self.decay_factor = decay_factor
        self.max_hops = max_hops
        self.min_trust_threshold = min_trust_threshold
        self.cache = TrustCache(ttl_seconds=cache_ttl)
        self.apply_ring_coefficient = apply_ring_coefficient
        
        # Ring coefficient calculator (lazy loaded)
        self._ring_calculator: Any = None
        
        # Stats
        self.stats = {
            "computations": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "rings_detected": 0,
        }
    
    @property
    def ring_calculator(self) -> Any:
        """Get or create the ring coefficient calculator."""
        if self._ring_calculator is None:
            from .ring_coefficient import RingCoefficientCalculator
            self._ring_calculator = RingCoefficientCalculator()
        return self._ring_calculator
    
    def _default_trust_getter(
        self,
        node_id: UUID,
        domain: str | None = None,
    ) -> list[tuple[UUID, float]]:
        """Default trust getter using the TrustManager.
        
        Override or inject a custom getter for testing.
        """
        # Import here to avoid circular imports
        from .trust import get_trust_manager
        from ..core.db import get_cursor
        
        try:
            with get_cursor() as cur:
                # Get all nodes this node has trust relationships with
                cur.execute("""
                    SELECT nt.node_id, nt.trust
                    FROM node_trust nt
                    JOIN federation_nodes fn ON fn.id = nt.node_id
                    WHERE fn.status != 'unreachable'
                """)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    trust_data = row["trust"]
                    if isinstance(trust_data, str):
                        import json
                        trust_data = json.loads(trust_data)
                    
                    # Get domain-specific or overall trust
                    if domain and "domain_expertise" in trust_data:
                        trust_score = trust_data["domain_expertise"].get(
                            domain, trust_data.get("overall", 0.1)
                        )
                    else:
                        trust_score = trust_data.get("overall", 0.1)
                    
                    results.append((row["node_id"], trust_score))
                
                return results
        except Exception as e:
            logger.warning(f"Error getting trust relationships: {e}")
            return []
    
    def compute_transitive_trust(
        self,
        from_node_id: UUID,
        to_node_id: UUID,
        domain: str | None = None,
        use_cache: bool = True,
    ) -> TransitiveTrustResult:
        """Compute transitive trust from one node to another.
        
        Args:
            from_node_id: The source node (observer)
            to_node_id: The target node (being trusted)
            domain: Optional domain for domain-specific trust
            use_cache: Whether to use cached results
            
        Returns:
            TransitiveTrustResult with computed trust
        """
        start_time = time.time()
        self.stats["computations"] += 1
        
        # Check cache
        if use_cache:
            cached = self.cache.get(from_node_id, to_node_id, domain)
            if cached:
                self.stats["cache_hits"] += 1
                return cached
        
        self.stats["cache_misses"] += 1
        
        # Build local trust graph (BFS from source)
        trust_graph = self._build_trust_graph(from_node_id, domain)
        
        # Get direct trust if it exists
        direct_trust = trust_graph.get(from_node_id, {}).get(to_node_id)
        
        # Compute transitive trust using modified Dijkstra/BFS
        transitive, paths, shortest, ring_coeff, rings = self._propagate_trust(
            trust_graph,
            from_node_id,
            to_node_id,
        )
        
        computation_time = (time.time() - start_time) * 1000
        
        result = TransitiveTrustResult(
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            direct_trust=direct_trust,
            transitive_trust=transitive,
            path_count=paths,
            shortest_path_length=shortest,
            computation_time_ms=computation_time,
            ring_coefficient_applied=ring_coeff,
            rings_detected=rings,
        )
        
        # Cache result
        if use_cache:
            self.cache.set(result, domain)
        
        return result
    
    def _build_trust_graph(
        self,
        source_node_id: UUID,
        domain: str | None = None,
    ) -> dict[UUID, dict[UUID, float]]:
        """Build a local trust graph using BFS from source.
        
        Returns:
            Dict mapping node_id -> {trusted_node_id -> trust_score}
        """
        graph: dict[UUID, dict[UUID, float]] = {}
        visited: set[UUID] = set()
        frontier: list[tuple[UUID, int]] = [(source_node_id, 0)]
        
        while frontier:
            current_id, depth = frontier.pop(0)
            
            if current_id in visited or depth > self.max_hops:
                continue
            
            visited.add(current_id)
            
            # Get trust relationships for this node
            trust_edges = self.trust_getter(current_id, domain)
            
            if trust_edges:
                graph[current_id] = {}
                for target_id, trust_score in trust_edges:
                    if trust_score >= self.min_trust_threshold:
                        graph[current_id][target_id] = trust_score
                        if target_id not in visited:
                            frontier.append((target_id, depth + 1))
        
        return graph
    
    def _propagate_trust(
        self,
        graph: dict[UUID, dict[UUID, float]],
        source: UUID,
        target: UUID,
    ) -> tuple[float, int, int | None, float, int]:
        """Compute transitive trust using path aggregation.
        
        Uses BFS to find all paths and aggregates trust contributions.
        Applies ring coefficient dampening when cycles are detected.
        
        Returns:
            Tuple of (transitive_trust, path_count, shortest_path_length, 
                      ring_coefficient, rings_detected)
        """
        if source == target:
            return 1.0, 1, 0, 1.0, 0
        
        if source not in graph:
            return 0.0, 0, None, 1.0, 0
        
        # Direct trust check
        if target in graph.get(source, {}):
            direct = graph[source][target]
            # Continue to find additional paths for boosting
        else:
            direct = 0.0
        
        # BFS to find all paths
        # State: (current_node, path_trust, hop_count, visited_set, path_list)
        paths_found: list[tuple[float, int, list[UUID]]] = []  # (trust, hops, path)
        rings_detected = 0
        
        queue: list[tuple[UUID, float, int, set[UUID], list[UUID]]] = [
            (source, 1.0, 0, {source}, [source])
        ]
        
        while queue:
            current, path_trust, hops, visited, path = queue.pop(0)
            
            if hops >= self.max_hops:
                continue
            
            for neighbor, edge_trust in graph.get(current, {}).items():
                if neighbor in visited:
                    # Ring detected - this is a back edge
                    if self.apply_ring_coefficient:
                        rings_detected += 1
                    continue
                
                # Decay trust for this hop
                new_trust = path_trust * edge_trust * self.decay_factor
                
                if new_trust < self.min_trust_threshold:
                    continue
                
                new_hops = hops + 1
                new_path = path + [neighbor]
                
                if neighbor == target:
                    paths_found.append((new_trust, new_hops, new_path))
                else:
                    new_visited = visited | {neighbor}
                    queue.append((neighbor, new_trust, new_hops, new_visited, new_path))
        
        if not paths_found:
            return direct, 0 if direct == 0 else 1, 1 if direct > 0 else None, 1.0, rings_detected
        
        # Calculate ring coefficient for paths
        ring_coefficient = 1.0
        if self.apply_ring_coefficient and rings_detected > 0:
            # Apply ring dampening based on graph structure
            ring_coefficient = self.ring_calculator.calculate_path_coefficient(
                [source, target], graph
            )
            self.stats["rings_detected"] += rings_detected
        
        # Aggregate paths using max (could also use weighted sum)
        # Using max is more conservative and interpretable
        max_trust = max(t for t, _, _ in paths_found)
        shortest = min(h for _, h, _ in paths_found)
        
        # Apply ring coefficient to transitive trust
        dampened_trust = max_trust * ring_coefficient
        
        # Combine direct and transitive (take max)
        # Note: Direct trust is NOT subject to ring dampening
        final_trust = max(direct, dampened_trust)
        
        return (
            final_trust, 
            len(paths_found) + (1 if direct > 0 else 0), 
            shortest,
            ring_coefficient,
            rings_detected,
        )
    
    def compute_trust_for_all_peers(
        self,
        from_node_id: UUID,
        domain: str | None = None,
    ) -> dict[UUID, TransitiveTrustResult]:
        """Compute transitive trust to all reachable peers.
        
        Useful for pre-computing trust before federation queries.
        
        Args:
            from_node_id: The source node
            domain: Optional domain filter
            
        Returns:
            Dict mapping target_node_id -> TransitiveTrustResult
        """
        # Build full graph first
        graph = self._build_trust_graph(from_node_id, domain)
        
        results = {}
        all_nodes = set()
        for node, edges in graph.items():
            all_nodes.add(node)
            all_nodes.update(edges.keys())
        
        for target_id in all_nodes:
            if target_id != from_node_id:
                results[target_id] = self.compute_transitive_trust(
                    from_node_id, target_id, domain
                )
        
        return results
    
    def get_stats(self) -> dict[str, Any]:
        """Get computation statistics."""
        return {
            **self.stats,
            "cache_size": self.cache.size,
            "cache_hit_rate": (
                self.stats["cache_hits"] / max(1, self.stats["computations"])
            ),
            "ring_coefficient_enabled": self.apply_ring_coefficient,
        }
    
    def record_trust_change(
        self,
        node_id: UUID,
        trust_delta: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a trust change for ring coefficient velocity tracking.
        
        Args:
            node_id: Node that received trust change
            trust_delta: Amount of trust change (positive or negative)
            timestamp: When the change occurred (defaults to now)
        """
        if self.apply_ring_coefficient:
            self.ring_calculator.record_trust_change(node_id, trust_delta, timestamp)
    
    def analyze_trust_graph(
        self,
        from_node_id: UUID,
        domain: str | None = None,
    ) -> Any:
        """Analyze the trust graph for Sybil patterns.
        
        Args:
            from_node_id: Root node to build graph from
            domain: Optional domain filter
            
        Returns:
            GraphAnalysisResult with ring detection and cluster analysis
        """
        graph = self._build_trust_graph(from_node_id, domain)
        return self.ring_calculator.analyze_graph(graph)
    
    def invalidate_node(self, node_id: UUID) -> int:
        """Invalidate cache entries for a node.
        
        Call when a node's trust changes.
        """
        return self.cache.invalidate(node_id)


# =============================================================================
# TYPE ALIAS
# =============================================================================


# Type for trust getter functions
TrustGetter: TypeAlias = Callable[[UUID, str | None], list[tuple[UUID, float]]]


# =============================================================================
# FEDERATION QUERY INTEGRATION
# =============================================================================


def weight_query_results_by_trust(
    results: list[dict[str, Any]],
    from_node_id: UUID,
    propagation: TrustPropagation | None = None,
    domain: str | None = None,
    trust_field: str = "origin_node_id",
    weight_field: str = "trust_weight",
) -> list[dict[str, Any]]:
    """Add trust weights to federation query results.
    
    Args:
        results: List of query result dicts
        from_node_id: The querying node's ID
        propagation: TrustPropagation engine (creates default if None)
        domain: Optional domain for domain-specific trust
        trust_field: Field name containing the source node ID
        weight_field: Field name to add with trust weight
        
    Returns:
        Results with trust weights added
    """
    if not results:
        return results
    
    if propagation is None:
        propagation = get_trust_propagation()
    
    weighted = []
    for result in results:
        result_copy = dict(result)
        
        source_node_id = result.get(trust_field)
        if source_node_id:
            if isinstance(source_node_id, str):
                source_node_id = UUID(source_node_id)
            
            trust_result = propagation.compute_transitive_trust(
                from_node_id, source_node_id, domain
            )
            result_copy[weight_field] = trust_result.effective_trust
            result_copy["_trust_details"] = {
                "direct": trust_result.direct_trust,
                "transitive": trust_result.transitive_trust,
                "path_count": trust_result.path_count,
            }
        else:
            result_copy[weight_field] = 0.0
        
        weighted.append(result_copy)
    
    # Sort by trust weight (highest first)
    weighted.sort(key=lambda x: x.get(weight_field, 0), reverse=True)
    
    return weighted


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================


# Default engine instance
_default_propagation: TrustPropagation | None = None
_default_propagation_lock = threading.Lock()


def get_trust_propagation(
    decay_factor: float = DEFAULT_DECAY_FACTOR,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> TrustPropagation:
    """Get the default TrustPropagation instance.
    
    Thread-safe initialization using double-checked locking pattern.
    """
    global _default_propagation
    if _default_propagation is None:
        with _default_propagation_lock:
            # Double-check after acquiring lock
            if _default_propagation is None:
                _default_propagation = TrustPropagation(
                    decay_factor=decay_factor,
                    max_hops=max_hops,
                )
    return _default_propagation


def compute_transitive_trust(
    from_node_id: UUID,
    to_node_id: UUID,
    domain: str | None = None,
) -> TransitiveTrustResult:
    """Compute transitive trust (convenience function)."""
    return get_trust_propagation().compute_transitive_trust(
        from_node_id, to_node_id, domain
    )


def invalidate_trust_cache(node_id: UUID | None = None) -> int:
    """Invalidate trust cache.
    
    Args:
        node_id: If provided, invalidate only entries for this node.
                 If None, clear entire cache.
                 
    Returns:
        Number of entries invalidated
    """
    engine = get_trust_propagation()
    if node_id:
        return engine.invalidate_node(node_id)
    else:
        size = engine.cache.size
        engine.cache.clear()
        return size
