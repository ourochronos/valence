# Trust Graph Algorithms

*Computational methods for trust propagation, decay, and graph operations*

---

## Overview

This document specifies the algorithms that power trust computation. Goals:
- **Correctness**: Trust values reflect actual relationships
- **Efficiency**: Scale to millions of agents, billions of edges
- **Resistance**: Robust against Sybil and manipulation attacks
- **Predictability**: Agents can reason about trust changes

---

## 1. Direct Trust Retrieval

The simplest case: looking up an explicit trust relationship.

### Algorithm: `get_direct_trust`

```python
def get_direct_trust(graph: TrustGraph, from_id: AgentId, to_id: AgentId, domain: str = None) -> float:
    """
    Retrieve direct trust with decay applied.
    Returns None if no direct edge exists.
    """
    if from_id == to_id:
        return 1.0  # Self-trust is always 1.0
    
    edge = graph.edges.get(to_id)
    if edge is None:
        return None
    
    # Get base trust level
    if domain is not None:
        base_trust = resolve_domain_trust(edge, domain)
    else:
        base_trust = edge.trust_level
    
    # Apply decay
    decayed_trust = apply_decay(base_trust, edge.last_used, graph.config)
    
    return decayed_trust


def resolve_domain_trust(edge: TrustEdge, domain: str) -> float:
    """
    Walk domain hierarchy to find most specific trust level.
    domain = "code:rust:async" checks:
      1. code:rust:async (exact)
      2. code:rust (parent)
      3. code (grandparent)
      4. trust_level (fallback)
    """
    parts = domain.split(":")
    
    for i in range(len(parts), 0, -1):
        prefix = ":".join(parts[:i])
        if prefix in edge.domain_trust:
            return edge.domain_trust[prefix]
    
    return edge.trust_level
```

**Complexity**: O(d) where d = domain depth (typically ≤5)

---

## 2. Trust Decay

Trust fades without reinforcement.

### Algorithm: `apply_decay`

```python
import math
from datetime import datetime

def apply_decay(trust: float, last_used: datetime, config: TrustConfig) -> float:
    """
    Apply exponential decay to a trust value.
    
    Formula: trust × exp(-decay_rate × days_since_use)
    
    Never goes below decay_floor.
    """
    now = datetime.utcnow()
    days_elapsed = (now - last_used).total_seconds() / 86400
    
    if days_elapsed <= 0:
        return trust
    
    # Exponential decay
    decay_factor = math.exp(-config.decay_rate * days_elapsed)
    decayed = trust * decay_factor
    
    # Floor
    return max(decayed, config.decay_floor)


def decay_half_life(config: TrustConfig) -> float:
    """
    Calculate how many days until trust drops to 50%.
    For planning and debugging.
    """
    # exp(-rate × t) = 0.5
    # -rate × t = ln(0.5)
    # t = -ln(0.5) / rate
    return math.log(2) / config.decay_rate
```

### Decay Curve Examples

With default `decay_rate = 0.001`:

| Days since use | Decay factor | Trust 0.8 becomes |
|----------------|--------------|-------------------|
| 0 | 1.000 | 0.800 |
| 30 | 0.970 | 0.776 |
| 90 | 0.914 | 0.731 |
| 180 | 0.835 | 0.668 |
| 365 | 0.697 | 0.558 |
| 730 | 0.486 | 0.389 |

Half-life ≈ 693 days.

### Variable Decay Rates

Some trust types decay differently:

```python
def get_decay_rate(edge: TrustEdge, config: TrustConfig) -> float:
    """
    Adjust decay rate based on trust properties.
    """
    base_rate = config.decay_rate
    
    # Manual trust decays slower (intentional relationships)
    if has_basis_type(edge, TrustBasisType.MANUAL):
        base_rate *= 0.1
    
    # High trust decays slower (valuable relationships)
    if edge.trust_level > 0.9:
        base_rate *= 0.5
    
    # Persistent flag (user marked as important)
    if edge.persistent:
        return 0.0
    
    return base_rate
```

---

## 3. Transitive Trust Computation

The core challenge: efficiently computing trust through the graph.

### Algorithm: `compute_transitive_trust`

We use a **modified Dijkstra's algorithm** optimized for trust multiplication (not sum).

```python
import heapq
from collections import defaultdict

def compute_transitive_trust(
    graph: TrustGraph,
    from_id: AgentId,
    to_id: AgentId,
    domain: str = None,
    max_hops: int = 3,
    require_distinct_sources: int = 2
) -> TransitiveTrustResult:
    """
    Find best trust path(s) from source to target.
    
    Returns the maximum trust achievable, subject to Sybil resistance rules.
    """
    if from_id == to_id:
        return TransitiveTrustResult(level=1.0, path_count=1, distinct_sources=1)
    
    # Direct trust check first
    direct = get_direct_trust(graph, from_id, to_id, domain)
    if direct is not None:
        return TransitiveTrustResult(
            level=direct, 
            source="direct", 
            path_count=1, 
            distinct_sources=1
        )
    
    # BFS/Dijkstra hybrid for trust paths
    # Priority queue: (-trust, hops, current_node, path, first_hop)
    # Negative trust because heapq is min-heap
    
    pq = []
    best_trust_to = defaultdict(float)  # Best trust reaching each node
    paths_by_first_hop = defaultdict(list)  # Group paths by first-hop agent
    
    damping = graph.config.transitive_damping
    
    # Initialize with direct neighbors
    for neighbor_id, edge in graph.edges.items():
        trust = get_direct_trust(graph, from_id, neighbor_id, domain)
        if trust and trust > 0:
            damped_trust = trust * damping
            heapq.heappush(pq, (-damped_trust, 1, neighbor_id, [from_id, neighbor_id], neighbor_id))
    
    found_paths = []
    
    while pq:
        neg_trust, hops, current, path, first_hop = heapq.heappop(pq)
        current_trust = -neg_trust
        
        # Skip if we've found a better path to this node
        if current_trust <= best_trust_to[current]:
            continue
        best_trust_to[current] = current_trust
        
        # Check if we reached target
        if current == to_id:
            found_paths.append({
                "trust": current_trust,
                "path": path,
                "first_hop": first_hop,
                "hops": hops
            })
            continue
        
        # Don't explore beyond max_hops
        if hops >= max_hops:
            continue
        
        # Explore neighbors (need to query their trust graph)
        # This requires federation or cached graph data
        for neighbor_id, neighbor_trust in get_outgoing_trust(current, domain):
            if neighbor_id in path:  # Cycle detection
                continue
            
            new_trust = current_trust * neighbor_trust * damping
            if new_trust > best_trust_to[neighbor_id]:
                new_path = path + [neighbor_id]
                heapq.heappush(pq, (-new_trust, hops + 1, neighbor_id, new_path, first_hop))
    
    if not found_paths:
        return TransitiveTrustResult(level=graph.config.default_trust, source="default")
    
    # Apply Sybil resistance: require distinct first-hop sources
    return apply_sybil_filter(found_paths, require_distinct_sources)


def apply_sybil_filter(paths: list, min_sources: int) -> TransitiveTrustResult:
    """
    Filter paths to ensure multiple distinct trust sources.
    """
    # Group by first-hop agent
    by_first_hop = defaultdict(list)
    for p in paths:
        by_first_hop[p["first_hop"]].append(p)
    
    distinct_sources = len(by_first_hop)
    
    if distinct_sources < min_sources:
        # Not enough independent sources - cap trust
        best = max(paths, key=lambda p: p["trust"])
        capped_trust = min(best["trust"], 0.3)  # Heavy penalty
        return TransitiveTrustResult(
            level=capped_trust,
            path_count=len(paths),
            distinct_sources=distinct_sources,
            capped=True,
            best_path=best["path"]
        )
    
    # Enough sources - use best path trust
    best = max(paths, key=lambda p: p["trust"])
    return TransitiveTrustResult(
        level=best["trust"],
        path_count=len(paths),
        distinct_sources=distinct_sources,
        capped=False,
        best_path=best["path"]
    )
```

**Complexity**: O(E × log V) where E = edges, V = vertices in explored subgraph

### Why Max, Not Average?

Consider: I trust Alice (0.8) and Bob (0.8). Both trust Carol.
- Alice→Carol: 0.7
- Bob→Carol: 0.3

**Average approach**: (0.8×0.7 + 0.8×0.3) / 2 = 0.4
**Max approach**: max(0.8×0.7, 0.8×0.3) = 0.56

Max is correct because:
1. One strong path should suffice
2. Average penalizes having weak secondary paths
3. Sybil resistance is handled separately (distinct sources)

---

## 4. Path-Based Trust Computation

### All Paths vs. Best Path

| Strategy | Pros | Cons |
|----------|------|------|
| **Best path** | Fast, simple | Ignores redundancy |
| **All paths** | More complete | Expensive, complex aggregation |
| **K-best paths** | Balance | Still needs aggregation |
| **Distinct-source best** | Sybil-resistant | Our choice |

We use **distinct-source best**: Find the best path through each first-hop neighbor, require at least N distinct neighbors to agree.

### Algorithm: `find_all_paths`

For debugging and analysis (not production queries):

```python
def find_all_paths(
    graph: TrustGraph,
    from_id: AgentId,
    to_id: AgentId,
    max_hops: int = 3,
    min_trust: float = 0.1
) -> list[TrustPath]:
    """
    Find all paths from source to target (expensive - use for analysis only).
    """
    paths = []
    
    def dfs(current: AgentId, path: list, trust: float, hops: int):
        if hops > max_hops:
            return
        if trust < min_trust:
            return
        if current == to_id:
            paths.append(TrustPath(path=path.copy(), cumulative_trust=trust))
            return
        
        for neighbor_id, edge in get_edges(current):
            if neighbor_id not in path:  # No cycles
                edge_trust = get_edge_trust(edge)
                path.append(neighbor_id)
                dfs(neighbor_id, path, trust * edge_trust * damping, hops + 1)
                path.pop()
    
    dfs(from_id, [from_id], 1.0, 0)
    return sorted(paths, key=lambda p: -p.cumulative_trust)
```

---

## 5. Cycle Handling

Trust graphs can have cycles: A trusts B, B trusts C, C trusts A.

### Problem

Naive algorithms might:
- Loop infinitely
- Double-count trust paths
- Create artificial trust amplification

### Solution: Path-Based Cycle Detection

```python
def is_cycle_free(path: list[AgentId]) -> bool:
    """Check if path contains no repeated nodes."""
    return len(path) == len(set(path))
```

In our algorithms:
1. **Track visited nodes per path** (not globally)
2. **Reject paths with repeated nodes**
3. **Max hop limit** prevents deep exploration

### Mutual Trust (A↔B)

Mutual trust is fine - it's two separate edges:
- A→B trust
- B→A trust

These are stored independently and don't create algorithmic issues.

---

## 6. Trust Propagation (Cache Invalidation)

When trust changes, what cached values need update?

### Algorithm: `propagate_update`

```python
def propagate_update(
    graph: TrustGraph,
    changed_edge: TrustEdge,
    old_level: float,
    new_level: float
) -> PropagationResult:
    """
    Determine which transitive trust values are affected by an edge change.
    """
    affected = []
    cache = graph.transitive_cache
    
    # Find all cached entries that include this edge
    for (source, target, domain, hops), cached_result in cache.items():
        if changed_edge.from_id not in cached_result.best_path:
            continue
        if changed_edge.to_id not in cached_result.best_path:
            continue
        
        # Check if edge is actually in the path
        path = cached_result.best_path
        for i in range(len(path) - 1):
            if path[i] == changed_edge.from_id and path[i+1] == changed_edge.to_id:
                # This cached value needs recomputation
                affected.append(CacheKey(source, target, domain, hops))
                break
    
    # Invalidate affected cache entries
    for key in affected:
        del cache[key]
    
    # Optionally: eagerly recompute high-value entries
    return PropagationResult(
        affected_count=len(affected),
        invalidated=affected
    )
```

### Lazy vs. Eager Recomputation

| Strategy | When to use |
|----------|-------------|
| **Lazy** | Invalidate cache, recompute on next query |
| **Eager** | Immediately recompute (for critical paths) |
| **Hybrid** | Eager for high-frequency queries, lazy otherwise |

Default: **Lazy**. Trust queries are not latency-critical enough to justify eager recomputation.

---

## 7. Sybil Resistance

### Attack Model

Attacker creates N Sybil identities that:
1. All trust each other maximally
2. All trust the attacker's "real" identity
3. Attempt to appear as N independent endorsements

### Defense: Distinct Source Requirement

```python
def sybil_resistant_trust(
    paths: list[TrustPath],
    min_distinct_sources: int = 2
) -> float:
    """
    Require paths through multiple distinct first-hop agents.
    """
    sources = set()
    for path in paths:
        if len(path.nodes) >= 2:
            sources.add(path.nodes[1])  # First hop from source
    
    if len(sources) < min_distinct_sources:
        # Insufficient independence - heavy penalty
        return min(max_path_trust(paths), 0.3)
    
    return max_path_trust(paths)
```

### Why This Works

If I trust Alice and Bob directly:
- Sybils created by Mallory only help if BOTH Alice and Bob trust them
- Alice and Bob are real agents with their own trust decisions
- Mallory can't control who Alice and Bob trust

### Additional Defenses

```python
def apply_new_agent_penalty(agent: AgentIdentity, trust: float) -> float:
    """New agents (< 30 days) have capped trust."""
    age_days = (now() - agent.created).days
    if age_days < 30:
        return min(trust, 0.3)
    return trust


def apply_verification_requirement(trust: float, edge: TrustEdge) -> float:
    """High trust requires verification history."""
    if trust > 0.7:
        if not has_basis_type(edge, VERIFICATION_HISTORY) and \
           not has_basis_type(edge, DIRECT_INTERACTION):
            return 0.6  # Cap at moderate trust
    return trust


def apply_velocity_limit(edge: TrustEdge, new_level: float) -> float:
    """Limit trust increase rate."""
    max_daily_increase = 0.1
    trust_yesterday = get_trust_at(edge, days_ago=1)
    if new_level > trust_yesterday + max_daily_increase:
        return trust_yesterday + max_daily_increase
    return new_level
```

---

## 8. Efficient Graph Queries

### Index Structures

```sql
-- B-tree for exact lookups
CREATE INDEX idx_trust_edges_lookup ON trust_edges (owner_id, to_id);

-- B-tree for range queries (find trusted agents above threshold)
CREATE INDEX idx_trust_edges_level ON trust_edges (owner_id, trust_level DESC);

-- B-tree for decay processing (find stale edges)
CREATE INDEX idx_trust_edges_stale ON trust_edges (last_used) 
  WHERE last_used < now() - interval '30 days';

-- GIN for domain queries
CREATE INDEX idx_trust_edges_domains ON trust_edges USING gin (domain_trust);
```

### Query Patterns

**Pattern 1: Direct Trust Lookup**
```sql
SELECT * FROM trust_edges 
WHERE owner_id = $1 AND to_id = $2;
-- O(1) with index
```

**Pattern 2: Find Trusted Agents**
```sql
SELECT to_id, trust_level FROM trust_edges
WHERE owner_id = $1 AND trust_level >= $2
ORDER BY trust_level DESC
LIMIT $3;
-- O(log n) with index
```

**Pattern 3: Decay Maintenance**
```sql
UPDATE trust_edges
SET trust_level = trust_level * exp(-0.001 * extract(days from now() - last_used))
WHERE last_used < now() - interval '7 days';
-- Batch process weekly
```

### Caching Strategy

```python
class TransitiveCache:
    """LRU cache for transitive trust computations."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl_seconds
    
    def get(self, key: CacheKey) -> Optional[TransitiveTrustResult]:
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if time.time() - entry.timestamp > self.ttl:
            del self.cache[key]
            return None
        
        # Move to end (LRU)
        self.cache.move_to_end(key)
        return entry.value
    
    def put(self, key: CacheKey, value: TransitiveTrustResult):
        if key in self.cache:
            del self.cache[key]
        
        self.cache[key] = CacheEntry(value, time.time())
        
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
    
    def invalidate_agent(self, agent_id: AgentId):
        """Invalidate all entries involving this agent."""
        to_delete = [k for k in self.cache 
                     if agent_id in (k.from_id, k.to_id)]
        for k in to_delete:
            del self.cache[k]
```

---

## 9. Batch Operations

For federation joins and imports.

### Algorithm: `batch_trust_import`

```python
def batch_trust_import(
    graph: TrustGraph,
    edges: list[TrustEdge],
    source: str,  # "federation:xyz", "import:file", etc.
    conflict_resolution: str = "max"  # "max", "newer", "keep"
) -> BatchResult:
    """
    Import multiple trust edges efficiently.
    """
    succeeded = 0
    failed = []
    
    # Pre-validate all edges
    valid_edges = []
    for edge in edges:
        try:
            validate_edge(edge)
            valid_edges.append(edge)
        except ValidationError as e:
            failed.append(FailedImport(edge, str(e)))
    
    # Group by target to detect conflicts
    by_target = defaultdict(list)
    for edge in valid_edges:
        by_target[edge.to_id].append(edge)
    
    # Resolve conflicts and insert
    for to_id, incoming_edges in by_target.items():
        existing = graph.edges.get(to_id)
        
        if not existing:
            # No conflict - use best incoming
            best = max(incoming_edges, key=lambda e: e.trust_level)
            graph.edges[to_id] = best
            succeeded += 1
        else:
            # Conflict resolution
            if conflict_resolution == "max":
                best = max([existing] + incoming_edges, key=lambda e: e.trust_level)
            elif conflict_resolution == "newer":
                best = max([existing] + incoming_edges, key=lambda e: e.updated)
            else:  # "keep"
                best = existing
            
            graph.edges[to_id] = best
            succeeded += 1
    
    # Invalidate cache
    graph.transitive_cache.clear()
    
    return BatchResult(succeeded=succeeded, failed=failed)
```

---

## 10. Trust Aggregation (for Federation)

When a federation wants to share aggregate trust.

### Algorithm: `aggregate_trust`

```python
def aggregate_trust(
    member_trusts: list[tuple[AgentId, float]],  # (member, their trust level)
    member_weights: dict[AgentId, float],         # Optional: weight per member
    method: str = "weighted_mean"
) -> float:
    """
    Aggregate multiple members' trust into a single score.
    """
    if not member_trusts:
        return 0.0
    
    if method == "weighted_mean":
        total_weight = 0
        weighted_sum = 0
        for member_id, trust in member_trusts:
            weight = member_weights.get(member_id, 1.0)
            weighted_sum += trust * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    elif method == "median":
        values = [t for _, t in member_trusts]
        return statistics.median(values)
    
    elif method == "min_threshold":
        # At least N members must trust above threshold
        threshold = 0.5
        min_count = max(2, len(member_trusts) // 3)
        above_threshold = sum(1 for _, t in member_trusts if t >= threshold)
        if above_threshold >= min_count:
            return statistics.mean([t for _, t in member_trusts if t >= threshold])
        return 0.0
    
    else:
        raise ValueError(f"Unknown method: {method}")
```

---

## Complexity Summary

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Direct trust lookup | O(d) | O(1) |
| Set trust | O(1) amortized | O(1) |
| Get trusted agents | O(log n + k) | O(k) |
| Transitive trust (best path) | O(E log V) | O(V) |
| Transitive trust (all paths) | O(b^h) | O(h × paths) |
| Cache invalidation | O(cache_size) | O(1) |
| Batch import | O(n log n) | O(n) |
| Decay update (batch) | O(n) | O(1) |

Where:
- d = domain hierarchy depth
- n = edges in graph
- k = result set size
- E = edges explored
- V = vertices explored
- b = average branching factor
- h = max hops

---

## Performance Targets

| Operation | Target Latency | At Scale (1M edges) |
|-----------|----------------|---------------------|
| Direct trust lookup | < 1ms | < 5ms |
| Get trusted agents (100) | < 10ms | < 50ms |
| Transitive trust (3 hops) | < 50ms | < 200ms |
| Cache hit | < 1ms | < 1ms |
| Batch import (1000 edges) | < 1s | < 5s |

---

*"Trust computation should feel instant. Math happens in the background."*
