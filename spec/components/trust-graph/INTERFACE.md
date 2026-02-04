# Trust Graph Interface

*API specification for Trust Graph operations*

---

## Overview

The Trust Graph interface provides CRUD operations for trust relationships plus computation methods for transitive trust and propagation. All operations are **scoped to the calling agent's trust graph** — you can only read/write your own trust relationships.

---

## Core Operations

### set_trust

Create or update a trust relationship.

```typescript
set_trust(
  from: AgentIdentity.id,      // Must be caller's own ID
  to: AgentIdentity.id,        // Target agent
  level: float,                // 0.0-1.0 overall trust
  domains?: Map<string, float>, // Domain-specific overrides
  basis?: TrustBasis[],        // Why this trust level?
  notes?: string               // Personal annotation
) → TrustEdge

// Errors:
// - UNAUTHORIZED: from ≠ caller
// - INVALID_LEVEL: level < 0 or level > 1.0
// - AGENT_NOT_FOUND: to doesn't exist
// - VELOCITY_LIMIT: too many changes today
// - SELF_TRUST: cannot set trust to self
```

**Behavior:**
- Creates edge if none exists
- Updates existing edge if present
- Sets `updated` and `last_used` to now
- Invalidates transitive cache for affected paths
- Respects velocity limits (max +0.1 per day)

**Example:**
```python
edge = trust.set_trust(
  from=my_id,
  to=alice_id,
  level=0.75,
  domains={"code:rust": 0.9, "finance": 0.3},
  basis=[TrustBasis(type=DIRECT_INTERACTION, confidence=0.8)],
  notes="Met at RustConf 2025, great debugger"
)
```

---

### get_trust

Retrieve trust level for an agent, optionally domain-scoped.

```typescript
get_trust(
  from: AgentIdentity.id,      // Must be caller's own ID
  to: AgentIdentity.id,        // Target agent
  domain?: string              // Optional domain filter
) → TrustResult {
  level: float,                // Effective trust level
  source: "direct" | "transitive" | "default",
  domain_match: string | null, // Which domain was matched
  edge?: TrustEdge,            // Full edge if direct
  decay_applied: float,        // How much decay affected this
  last_used: timestamp         // When last interacted
}

// Errors:
// - UNAUTHORIZED: from ≠ caller
```

**Lookup Order:**
1. Check for direct edge from→to
2. If domain specified, walk domain hierarchy
3. Apply decay to stored value
4. If no direct edge, compute transitive (cached)
5. If no transitive path, return default_trust

**Example:**
```python
result = trust.get_trust(my_id, alice_id, domain="code:rust")
# → TrustResult(level=0.87, source="direct", domain_match="code:rust", ...)

result = trust.get_trust(my_id, stranger_id)
# → TrustResult(level=0.1, source="default", ...)
```

---

### get_trusted_agents

Find agents meeting trust criteria.

```typescript
get_trusted_agents(
  from: AgentIdentity.id,      // Must be caller's own ID
  min_level: float,            // Minimum trust threshold
  domain?: string,             // Optional domain filter
  include_transitive?: bool,   // Include computed trust (default: false)
  max_hops?: int,              // For transitive (default: 3)
  limit?: int,                 // Max results (default: 100)
  offset?: int                 // Pagination (default: 0)
) → TrustedAgentsResult {
  agents: AgentTrust[],
  total: int,
  has_more: bool
}

AgentTrust {
  agent_id: AgentIdentity.id,
  level: float,
  source: "direct" | "transitive",
  domain_match: string | null,
  hops?: int                   // For transitive results
}

// Errors:
// - UNAUTHORIZED: from ≠ caller
// - INVALID_THRESHOLD: min_level < 0 or > 1.0
```

**Example:**
```python
result = trust.get_trusted_agents(
  my_id, 
  min_level=0.7, 
  domain="code",
  include_transitive=True
)
# → TrustedAgentsResult(agents=[
#     AgentTrust(agent_id=alice, level=0.85, source="direct"),
#     AgentTrust(agent_id=bob, level=0.72, source="transitive", hops=1),
#   ], total=2, has_more=False)
```

---

### remove_trust

Delete a trust edge entirely.

```typescript
remove_trust(
  from: AgentIdentity.id,      // Must be caller's own ID
  to: AgentIdentity.id         // Target agent
) → bool                       // True if existed and removed

// Errors:
// - UNAUTHORIZED: from ≠ caller
```

**Behavior:**
- Removes edge completely (not just sets to 0)
- Invalidates transitive cache
- Future queries return `default_trust`

---

### update_trust_basis

Add or update the reasoning behind a trust relationship.

```typescript
update_trust_basis(
  from: AgentIdentity.id,
  to: AgentIdentity.id,
  basis: TrustBasis,           // New basis to add
  recalculate?: bool           // Recompute trust_level from bases (default: false)
) → TrustEdge

// Errors:
// - UNAUTHORIZED: from ≠ caller
// - EDGE_NOT_FOUND: no existing trust relationship
```

**Example:**
```python
# Alice verified my claim successfully - add that as a basis
trust.update_trust_basis(
  my_id,
  alice_id,
  TrustBasis(
    type=VERIFICATION_HISTORY,
    confidence=0.9,
    evidence=[Evidence(type="belief_verified", reference_id=belief_123, weight=0.1)]
  ),
  recalculate=True  # Let system suggest new trust level
)
```

---

## Transitive Trust Operations

### compute_transitive_trust

Calculate trust through the network.

```typescript
compute_transitive_trust(
  from: AgentIdentity.id,      // Must be caller's own ID
  to: AgentIdentity.id,        // Target agent
  domain?: string,             // Optional domain filter
  max_hops?: int,              // Limit path length (default: 3)
  options?: TransitiveOptions {
    require_distinct_sources?: int,  // Sybil resistance (default: 2)
    use_cache?: bool,                // Use cached results (default: true)
    explain?: bool                   // Return path details (default: false)
  }
) → TransitiveTrustResult {
  level: float,                // Computed transitive trust
  path_count: int,             // Number of valid paths found
  best_path?: AgentIdentity.id[], // If explain=true
  distinct_sources: int,       // Unique first-hop agents
  capped: bool,                // Was result capped by Sybil rules?
  cache_hit: bool              // Was this from cache?
}

// Errors:
// - UNAUTHORIZED: from ≠ caller
// - COMPUTATION_TIMEOUT: too many paths
```

**Example:**
```python
result = trust.compute_transitive_trust(
  my_id,
  stranger_id,
  domain="code",
  max_hops=2,
  options=TransitiveOptions(explain=True)
)
# → TransitiveTrustResult(
#     level=0.52,
#     path_count=3,
#     best_path=[my_id, alice_id, stranger_id],
#     distinct_sources=2,  # Through alice and bob
#     capped=False,
#     cache_hit=False
#   )
```

---

### get_trust_paths

Explore the trust graph structure.

```typescript
get_trust_paths(
  from: AgentIdentity.id,
  to: AgentIdentity.id,
  max_hops?: int,              // Default: 3
  min_path_trust?: float,      // Filter weak paths (default: 0.1)
  limit?: int                  // Max paths to return (default: 10)
) → TrustPath[]

TrustPath {
  path: AgentIdentity.id[],    // Sequence of agents
  trust_at_each_hop: float[],  // Trust level at each edge
  cumulative_trust: float,     // Final path trust
  domain_consistency: bool     // All edges have domain trust?
}
```

---

## Propagation Operations

### propagate_trust_update

When a trust edge changes, what else is affected?

```typescript
propagate_trust_update(
  edge_change: TrustEdgeChange {
    from: AgentIdentity.id,
    to: AgentIdentity.id,
    old_level: float | null,   // null if new edge
    new_level: float | null,   // null if deleted
    domain?: string
  }
) → PropagationResult {
  affected_edges: AffectedEdge[],
  cache_invalidations: int,
  recomputation_required: bool
}

AffectedEdge {
  from: AgentIdentity.id,
  to: AgentIdentity.id,       // Agent whose transitive trust changed
  old_transitive: float,
  new_transitive: float,
  via: AgentIdentity.id       // The changed intermediate
}

// Note: This is typically called automatically by set_trust
```

**When is this useful?**
- Understanding ripple effects of trust changes
- Debugging trust computation
- Batch updates when joining a federation

---

### batch_update_trust

Efficiently update multiple trust edges at once.

```typescript
batch_update_trust(
  from: AgentIdentity.id,
  updates: TrustUpdate[]
) → BatchResult {
  succeeded: int,
  failed: FailedUpdate[],
  propagation: PropagationResult
}

TrustUpdate {
  to: AgentIdentity.id,
  level: float,
  domains?: Map<string, float>,
  basis?: TrustBasis[]
}

// Useful for:
// - Importing trust from another system
// - Joining a federation
// - Bulk adjustments
```

---

## Configuration Operations

### get_trust_config

Retrieve trust graph configuration.

```typescript
get_trust_config(
  owner: AgentIdentity.id      // Must be caller's own ID
) → TrustConfig {
  default_trust: float,
  decay_rate: float,
  decay_floor: float,
  transitive_damping: float,
  max_transitive_hops: int,
  sybil_threshold: int
}
```

---

### update_trust_config

Modify trust computation parameters.

```typescript
update_trust_config(
  owner: AgentIdentity.id,
  config: Partial<TrustConfig>
) → TrustConfig

// Errors:
// - UNAUTHORIZED: owner ≠ caller
// - INVALID_CONFIG: values out of range
```

**Constraints:**
- `default_trust`: 0.0 - 0.5
- `decay_rate`: 0.0 - 0.1
- `decay_floor`: 0.0 - default_trust
- `transitive_damping`: 0.1 - 0.9
- `max_transitive_hops`: 1 - 5
- `sybil_threshold`: 1 - 10

---

## Graph Maintenance Operations

### prune_decayed_edges

Remove edges that have decayed below threshold.

```typescript
prune_decayed_edges(
  owner: AgentIdentity.id,
  threshold?: float,           // Default: decay_floor
  dry_run?: bool               // Default: false
) → PruneResult {
  edges_examined: int,
  edges_pruned: int,
  pruned_agents: AgentIdentity.id[]
}
```

---

### refresh_trust

Update `last_used` timestamp without changing trust level.

```typescript
refresh_trust(
  from: AgentIdentity.id,
  to: AgentIdentity.id
) → TrustEdge

// Called automatically when:
// - You query beliefs from this agent
// - You interact with this agent
// - Verification events occur
```

---

### get_graph_stats

Summary statistics for the trust graph.

```typescript
get_graph_stats(
  owner: AgentIdentity.id
) → TrustGraphStats {
  total_edges: int,
  edges_by_level: Map<string, int>,  // "high", "medium", "low"
  edges_by_source: Map<TrustBasisType, int>,
  domain_coverage: Map<string, int>,
  avg_trust: float,
  avg_decay: float,
  cache_size: int,
  cache_hit_rate: float,
  last_pruned: timestamp
}
```

---

## Event Hooks

The Trust Graph emits events for integration with other components:

```typescript
// Emitted events
TrustEdgeCreated { from, to, level, domains, timestamp }
TrustEdgeUpdated { from, to, old_level, new_level, timestamp }
TrustEdgeRemoved { from, to, timestamp }
TransitiveTrustChanged { from, to, old_level, new_level, via }
TrustDecayApplied { from, to, old_level, new_level }

// Subscribe pattern
trust.on("edge_updated", (event: TrustEdgeUpdated) => {
  // Update belief weights
  // Recompute query rankings
  // etc.
})
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `UNAUTHORIZED` | Operation on another agent's trust graph |
| `AGENT_NOT_FOUND` | Target agent doesn't exist |
| `EDGE_NOT_FOUND` | No trust relationship exists |
| `INVALID_LEVEL` | Trust level outside 0.0-1.0 |
| `INVALID_DOMAIN` | Malformed domain string |
| `INVALID_CONFIG` | Configuration value out of range |
| `VELOCITY_LIMIT` | Too many trust changes today |
| `SELF_TRUST` | Cannot modify self-trust |
| `COMPUTATION_TIMEOUT` | Transitive computation too expensive |
| `CACHE_MISS` | Cached value not found (internal) |

---

## Rate Limits

To prevent gaming:

| Operation | Limit |
|-----------|-------|
| `set_trust` | 50/hour, max +0.1 per edge per day |
| `remove_trust` | 20/hour |
| `batch_update_trust` | 5/hour, 100 edges per batch |
| `compute_transitive_trust` | 100/minute |
| `get_trusted_agents` | 60/minute |

---

## Implementation Notes

### Caching Strategy

Transitive trust is expensive. Cache aggressively:
- Key: `(from, to, domain, max_hops)`
- TTL: 1 hour or until edge change
- Invalidation: Cascades from any edge in any cached path

### Index Requirements

```sql
-- Fast direct lookup
trust_edges(owner_id, to_id)

-- Domain queries
trust_edges(owner_id, domain)

-- Decay management
trust_edges(last_used) WHERE last_used < cutoff

-- Trust level filtering
trust_edges(owner_id) WHERE trust_level > threshold
```

### Consistency Model

- Trust updates are **immediately consistent** locally
- Propagation effects are **eventually consistent**
- Cache invalidation is **best-effort** (stale reads possible for ~1s)

---

*"Simple to use, hard to game."*
