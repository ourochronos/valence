# Query Protocol Specification

*Component 5 of Valence: Distributed Epistemic Infrastructure*

---

## Overview

The Query Protocol is how agents **ask questions of the knowledge network**. It translates natural language queries into trust-weighted, privacy-respecting searches across local, federated, and global belief stores.

Unlike traditional search:
- **Results are ranked by trust**, not just relevance
- **Privacy is enforced structurally**, not by policy
- **Rankings are explainable** — you can ask "why did this rank higher?"
- **Queries can be live** — subscribe to changes, not just snapshots

---

## Query Language

### 1. Query Structure

Every query has three components:

```
Query {
  // What to find
  semantic: string              // Natural language query text
  
  // How to filter
  filters: QueryFilters         // Hard constraints
  
  // How to rank
  context: TrustContext         // Trust weights and preferences
  
  // Execution options
  options: QueryOptions         // Pagination, scope, format
}
```

### 2. Semantic Expression

The `semantic` field accepts natural language:

```
"What do we know about quantum computing?"
"When was Anthropic founded?"
"What are the risks of autonomous vehicles?"
```

The query engine:
1. Embeds the query using the same model as beliefs (text-embedding-3-small)
2. Performs approximate nearest neighbor search against belief embeddings
3. Applies filters and trust weighting post-retrieval

**Advanced semantic operators:**

| Operator | Syntax | Meaning |
|----------|--------|---------|
| Must include | `+term` | Results must be semantically close to "term" |
| Must exclude | `-term` | Results must be semantically distant from "term" |
| Exact phrase | `"phrase"` | Content must contain exact phrase (post-filter) |
| Boost | `term^2.0` | Double weight on semantic similarity to "term" |

**Examples:**
```
"machine learning +medical -finance"     // ML in medicine, not finance
"\"San Francisco\" housing prices"       // Exact phrase required
"climate change policy^2.0 economics"    // Policy-heavy results
```

### 3. Query Filters

Hard constraints that exclude non-matching beliefs entirely:

```
QueryFilters {
  // Confidence thresholds (all optional)
  min_confidence: ConfidenceFilter {
    overall: float | null           // Min aggregated confidence
    source_reliability: float | null
    method_quality: float | null
    internal_consistency: float | null
    temporal_freshness: float | null
    corroboration: float | null
    domain_applicability: float | null
  }
  
  // Temporal bounds
  valid_at: timestamp | null         // Belief must be valid at this time
  created_after: timestamp | null    // Exclude beliefs older than this
  created_before: timestamp | null   // Exclude beliefs newer than this
  
  // Domains
  domains_include: string[]          // Must match at least one domain
  domains_exclude: string[]          // Must not match any domain
  
  // Source restrictions
  holder_ids: UUID[] | null          // Only from these agents
  exclude_holders: UUID[]            // Never from these agents
  min_holder_reputation: float | null // Min reputation of belief holder
  
  // Derivation
  derivation_types: DerivationType[] | null  // Only these derivation types
  max_derivation_depth: int | null   // Max steps from primary sources
  
  // Visibility (cannot exceed requester's access)
  visibility_levels: Visibility[]    // Which visibility levels to include
}
```

**Filter shorthand:**

For common patterns, use presets:

```
filters: "high_confidence"   // overall > 0.7
filters: "recent"           // created within last 30 days
filters: "verified"         // corroboration > 0.5
filters: "from_trusted"     // only from agents in trust graph with trust > 0.5
```

### 4. Trust Context

How to weight trust in ranking:

```
TrustContext {
  // The requester (for privacy and trust computation)
  requester_id: UUID
  
  // Trust weights
  trust_graph: TrustGraphRef         // Reference to requester's trust graph
  trust_weight: float                // How much trust affects ranking (0.0-1.0)
  
  // Domain emphasis
  query_domains: string[]            // Domains relevant to this query
  domain_trust_weight: float         // Weight domain-specific trust (0.0-1.0)
  
  // Transitive trust settings
  max_trust_hops: int                // Override default (usually 3)
  transitive_damping: float          // Override default (usually 0.7)
  
  // Confidence weighting
  confidence_weights: ConfidenceWeights {
    source_reliability: float        // Default: 0.25
    method_quality: float            // Default: 0.20
    internal_consistency: float      // Default: 0.15
    temporal_freshness: float        // Default: 0.15
    corroboration: float             // Default: 0.15
    domain_applicability: float      // Default: 0.10
  }
}
```

### 5. Query Options

Execution and response format:

```
QueryOptions {
  // Scope
  scope: QueryScope {
    local: boolean                   // Search requester's beliefs
    federated: boolean              // Search federations requester belongs to
    network: boolean                // Search public network beliefs
    nodes: UUID[] | null            // Specific nodes to query
  }
  
  // Pagination
  limit: int                        // Max results (default: 20, max: 1000)
  offset: int                       // Skip first N results
  cursor: string | null             // Opaque cursor for stable pagination
  
  // Response format
  include_derivation: boolean       // Include derivation chain (expensive)
  include_explanation: boolean      // Include ranking explanation
  max_derivation_depth: int         // How deep to trace derivations
  
  // Performance hints
  timeout_ms: int                   // Max query time (default: 5000)
  approximate: boolean              // Allow approximate results (faster)
  
  // Streaming
  stream: boolean                   // Return results as stream
  
  // Diversity
  diversity: DiversityConfig {
    enabled: boolean                // Ensure diverse results
    min_semantic_distance: float    // Min distance between results
    max_per_holder: int             // Max results from same agent
    max_per_domain: int             // Max results from same domain
  }
}
```

---

## Privacy Filters

Privacy is **enforced at the protocol level**, not optional.

### 1. Visibility Matrix

The query engine enforces this matrix:

| Belief Visibility | Requester Relationship | Can See? |
|-------------------|------------------------|----------|
| PRIVATE | Is holder | ✅ |
| PRIVATE | Not holder | ❌ |
| FEDERATED | Same federation | ✅ |
| FEDERATED | Different federation | ❌ |
| PUBLIC | Any | ✅ |

### 2. Privacy Enforcement Points

**Before embedding search:**
- Beliefs are partitioned by visibility
- Query only searches partitions requester can access

**After retrieval:**
- Double-check visibility before returning
- Never include content of inaccessible beliefs
- Aggregate stats may include counts without content

**In explanations:**
- Cannot reveal existence of private beliefs
- Federated beliefs show federation ID but not other members' details
- Trust graph edges are never exposed

### 3. Federation Privacy

When querying federated beliefs:

```
FederatedQueryResult {
  belief: Belief                    // The belief content
  federation_id: UUID               // Which federation
  member_count: int                 // How many members contributed
  // NOT included: individual member IDs or their trust relationships
}
```

### 4. Query Privacy (Future)

Optional protections for the query itself:

```
QueryPrivacy {
  // Encrypt query so intermediate nodes can't read it
  encrypted: boolean
  
  // Use differential privacy in aggregated results
  differential_privacy: {
    epsilon: float
    delta: float
  }
  
  // Route through onion-style relay
  anonymized: boolean
}
```

---

## Cross-Node Queries

Valence is distributed. Queries may span:
1. **Local node** — Requester's own beliefs
2. **Federation nodes** — Shared knowledge pools
3. **Network nodes** — Public consensus layer

### 1. Query Routing

```
┌─────────────────────────────────────────────────────────────────┐
│                        Query Dispatcher                          │
├─────────────────────────────────────────────────────────────────┤
│  Query → Parse → Route → [Local, Fed1, Fed2, Network] → Merge  │
└─────────────────────────────────────────────────────────────────┘
         │                      │                    │
         ▼                      ▼                    ▼
    ┌─────────┐          ┌─────────────┐      ┌──────────┐
    │  Local  │          │ Federations │      │ Network  │
    │  Store  │          │  (trusted)  │      │ (public) │
    └─────────┘          └─────────────┘      └──────────┘
```

### 2. Query Fanout

For federated/network queries:

```
CrossNodeQuery {
  // Original query (with privacy filters applied)
  query: Query
  
  // Routing
  target_nodes: NodeAddress[]       // Where to send
  
  // Constraints
  max_results_per_node: int         // Limit per source
  timeout_per_node_ms: int          // Per-node timeout
  
  // Aggregation
  merge_strategy: MergeStrategy {
    type: "interleave" | "waterfall" | "rerank"
    // interleave: Round-robin from each source
    // waterfall: Drain local first, then federated, then network
    // rerank: Collect all, globally rerank
  }
}
```

### 3. Result Merging

When results come from multiple nodes:

**Interleave** (fast, approximate):
1. Take top result from each node
2. Repeat until limit reached
3. Fast but may miss globally-optimal ranking

**Waterfall** (prioritized):
1. Local results first (highest trust)
2. Then federated (group trust)
3. Then network (public)
4. Good for "prefer my knowledge" scenarios

**Rerank** (accurate, slower):
1. Gather top-K from each node
2. Rerank globally with requester's trust context
3. Most accurate, but higher latency

### 4. Node Discovery

How to find nodes to query:

```
NodeDiscovery {
  // Known federations
  federations: Federation[]
  
  // Consensus nodes (for network queries)
  consensus_endpoints: Endpoint[]
  
  // DHT-style discovery (future)
  dht_bootstrap: Address[]
}
```

### 5. Caching

Reduce cross-node load:

```
QueryCache {
  // Cache remote results locally
  cache_duration: Duration          // How long to cache
  
  // Invalidation
  invalidate_on_update: boolean     // Clear cache when source updates
  
  // Staleness tolerance
  stale_while_revalidate: boolean   // Return stale + refresh async
}
```

---

## Result Pagination & Streaming

### 1. Offset-Based Pagination

Simple but unstable if data changes:

```
// First page
query(limit: 20, offset: 0) → results[0-19]

// Second page
query(limit: 20, offset: 20) → results[20-39]
```

**Problems:**
- New beliefs may shift results
- Not suitable for real-time scenarios

### 2. Cursor-Based Pagination

Stable pagination using opaque cursors:

```
// First page
query(limit: 20) → results[0-19] + cursor: "abc123"

// Next page
query(limit: 20, cursor: "abc123") → results[20-39] + cursor: "def456"
```

**Cursor contents (opaque to client):**
- Last seen belief ID
- Sort score at boundary
- Query hash (for validation)

### 3. Streaming Results

For large result sets or live queries:

```
stream(query) → AsyncIterator<RankedBelief>
```

**Backpressure:** Client controls consumption rate.

**Chunked delivery:**
```
StreamChunk {
  beliefs: RankedBelief[]           // Batch of results
  progress: float                   // 0.0-1.0 completion estimate
  done: boolean                     // No more results
}
```

### 4. Live Subscriptions

Subscribe to query results that update in real-time:

```
Subscription {
  query: Query
  callback: (event: SubscriptionEvent) → void
  
  // Configuration
  debounce_ms: int                  // Batch rapid updates
  include_removals: boolean         // Notify when beliefs no longer match
}

SubscriptionEvent {
  type: "added" | "updated" | "removed"
  belief: RankedBelief
  reason: string                    // Why this event occurred
}
```

---

## Query Examples

### Example 1: Simple Semantic Search

"What beliefs do I have about climate change?"

```json
{
  "semantic": "climate change",
  "filters": {},
  "context": {
    "requester_id": "my-agent-uuid",
    "trust_weight": 0.5
  },
  "options": {
    "scope": { "local": true },
    "limit": 10
  }
}
```

### Example 2: High-Confidence from Trusted Sources

"Beliefs about quantum computing with >80% confidence from sources I trust"

```json
{
  "semantic": "quantum computing",
  "filters": {
    "min_confidence": { "overall": 0.8 }
  },
  "context": {
    "requester_id": "my-agent-uuid",
    "trust_weight": 0.9,
    "trust_graph": "ref:my-trust-graph"
  },
  "options": {
    "scope": { "local": true, "federated": true },
    "limit": 20
  }
}
```

### Example 3: Recent Medical Knowledge from Network

"Latest medical research on mRNA vaccines, verified by multiple sources"

```json
{
  "semantic": "mRNA vaccines research",
  "filters": {
    "min_confidence": {
      "corroboration": 0.6,
      "temporal_freshness": 0.7
    },
    "domains_include": ["medical", "science/biology"],
    "created_after": "2024-01-01T00:00:00Z"
  },
  "context": {
    "requester_id": "my-agent-uuid",
    "query_domains": ["medical"],
    "domain_trust_weight": 0.8
  },
  "options": {
    "scope": { "local": true, "federated": true, "network": true },
    "limit": 50,
    "include_explanation": true
  }
}
```

### Example 4: Diverse Opinions

"What do different sources say about AI regulation?"

```json
{
  "semantic": "AI regulation policy opinions",
  "filters": {},
  "context": {
    "requester_id": "my-agent-uuid",
    "trust_weight": 0.3
  },
  "options": {
    "scope": { "network": true },
    "limit": 30,
    "diversity": {
      "enabled": true,
      "max_per_holder": 2,
      "min_semantic_distance": 0.3
    }
  }
}
```

---

## Performance Requirements

### Latency Targets

| Query Type | P50 | P99 | Max |
|------------|-----|-----|-----|
| Local only | 10ms | 50ms | 200ms |
| Local + federated | 50ms | 200ms | 500ms |
| Full network | 200ms | 1000ms | 5000ms |

### Throughput Targets

| Scale | Queries/sec |
|-------|-------------|
| Personal node | 100 |
| Federation node | 1,000 |
| Consensus node | 10,000 |

### Index Requirements

For billion-belief scale:

```
HNSW Index Parameters:
  - ef_construction: 200
  - M: 32
  - ef_search: 100 (adjustable per-query)
  
Memory estimate: ~4KB per belief → 4TB for 1B beliefs
Shard across nodes for horizontal scale
```

---

## Error Handling

### Query Errors

```
QueryError {
  code: ErrorCode
  message: string
  details: any
}

enum ErrorCode {
  INVALID_QUERY           // Malformed query structure
  INVALID_FILTERS         // Invalid filter values
  UNAUTHORIZED            // Requester cannot access scope
  TIMEOUT                 // Query exceeded timeout
  RATE_LIMITED            // Too many queries
  NODE_UNAVAILABLE        // Cross-node query failed
  INTERNAL_ERROR          // Unexpected error
}
```

### Partial Results

When some nodes fail:

```
QueryResult {
  beliefs: RankedBelief[]
  partial: boolean                  // True if some sources failed
  errors: NodeError[]               // Which nodes failed and why
  coverage: {
    local: boolean
    federated: UUID[]               // Which federations responded
    network: boolean
  }
}
```

---

## Security Considerations

### 1. Query Injection

- Semantic queries are embedded, not interpreted
- Filter values are parameterized
- No string interpolation in query processing

### 2. Denial of Service

- Rate limiting per requester
- Query complexity limits (max filters, max derivation depth)
- Timeout enforcement

### 3. Information Leakage

- Timing attacks: constant-time visibility checks
- Existence attacks: don't reveal count of filtered-out beliefs
- Side channels: careful with cache behavior

### 4. Trust Manipulation

- Cannot query with fake trust context
- Trust graph verified from requester's signed identity
- Transitive trust computed server-side, not client-provided

---

*"Ask and the network shall answer — with confidence, trust, and explanation."*
