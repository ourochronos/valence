# Query Protocol Interface

*Operations for querying the Valence epistemic network*

---

## Overview

This document defines the programmatic interface for querying beliefs in Valence. All operations are designed for:

- **Type safety** — Strong typing with clear contracts
- **Async-first** — All operations return promises/futures
- **Streaming** — Large results stream incrementally
- **Explainability** — Every result can be explained

---

## Core Types

### Request Types

```typescript
interface Query {
  semantic: string;
  filters?: QueryFilters;
  context: TrustContext;
  options?: QueryOptions;
}

interface QueryFilters {
  min_confidence?: ConfidenceFilter;
  valid_at?: Timestamp;
  created_after?: Timestamp;
  created_before?: Timestamp;
  domains_include?: string[];
  domains_exclude?: string[];
  holder_ids?: UUID[];
  exclude_holders?: UUID[];
  min_holder_reputation?: number;
  derivation_types?: DerivationType[];
  max_derivation_depth?: number;
  visibility_levels?: Visibility[];
}

interface ConfidenceFilter {
  overall?: number;
  source_reliability?: number;
  method_quality?: number;
  internal_consistency?: number;
  temporal_freshness?: number;
  corroboration?: number;
  domain_applicability?: number;
}

interface TrustContext {
  requester_id: UUID;
  trust_graph?: TrustGraphRef;
  trust_weight?: number;           // 0.0-1.0, default 0.5
  query_domains?: string[];
  domain_trust_weight?: number;    // 0.0-1.0, default 0.5
  max_trust_hops?: number;         // default 3
  transitive_damping?: number;     // default 0.7
  confidence_weights?: ConfidenceWeights;
}

interface ConfidenceWeights {
  source_reliability: number;
  method_quality: number;
  internal_consistency: number;
  temporal_freshness: number;
  corroboration: number;
  domain_applicability: number;
}

interface QueryOptions {
  scope?: QueryScope;
  limit?: number;                  // default 20, max 1000
  offset?: number;
  cursor?: string;
  include_derivation?: boolean;
  include_explanation?: boolean;
  max_derivation_depth?: number;
  timeout_ms?: number;             // default 5000
  approximate?: boolean;           // default false
  stream?: boolean;                // default false
  diversity?: DiversityConfig;
}

interface QueryScope {
  local?: boolean;                 // default true
  federated?: boolean;             // default false
  network?: boolean;               // default false
  nodes?: UUID[];
}

interface DiversityConfig {
  enabled: boolean;
  min_semantic_distance?: number;  // 0.0-1.0
  max_per_holder?: number;
  max_per_domain?: number;
}
```

### Response Types

```typescript
interface QueryResult {
  beliefs: RankedBelief[];
  total_count: number;             // Estimated total matches
  next_cursor?: string;            // For pagination
  query_time_ms: number;
  partial: boolean;                // True if some sources failed
  coverage: QueryCoverage;
  errors?: QueryError[];
}

interface RankedBelief {
  belief: Belief;
  score: RankingScore;
  explanation?: RankingExplanation;
  derivation_chain?: DerivationLink[];
}

interface RankingScore {
  final: number;                   // 0.0-1.0, the ranking score
  components: {
    semantic_similarity: number;
    confidence_score: number;
    trust_score: number;
    recency_score: number;
    diversity_penalty: number;     // Negative if penalized for similarity
  };
}

interface QueryCoverage {
  local: boolean;
  federations_queried: UUID[];
  federations_responded: UUID[];
  network_queried: boolean;
  network_responded: boolean;
}

interface QueryError {
  code: ErrorCode;
  message: string;
  source?: string;                 // Which node/federation failed
  recoverable: boolean;
}

enum ErrorCode {
  INVALID_QUERY = 'INVALID_QUERY',
  INVALID_FILTERS = 'INVALID_FILTERS',
  UNAUTHORIZED = 'UNAUTHORIZED',
  TIMEOUT = 'TIMEOUT',
  RATE_LIMITED = 'RATE_LIMITED',
  NODE_UNAVAILABLE = 'NODE_UNAVAILABLE',
  INTERNAL_ERROR = 'INTERNAL_ERROR'
}
```

---

## Primary Operations

### 1. query()

The main search operation.

```typescript
async function query(
  query: Query
): Promise<QueryResult>
```

**Parameters:**
- `query` — The complete query specification

**Returns:**
- `QueryResult` — Ranked beliefs matching the query

**Errors:**
- `INVALID_QUERY` — Malformed query structure
- `UNAUTHORIZED` — Requester cannot access requested scope
- `TIMEOUT` — Query exceeded timeout_ms
- `RATE_LIMITED` — Too many queries from this requester

**Example:**

```typescript
const result = await query({
  semantic: "machine learning in healthcare",
  filters: {
    min_confidence: { overall: 0.7 },
    domains_include: ["medical", "tech/ai"]
  },
  context: {
    requester_id: myAgentId,
    trust_weight: 0.8
  },
  options: {
    scope: { local: true, federated: true },
    limit: 20,
    include_explanation: true
  }
});

console.log(`Found ${result.total_count} beliefs`);
for (const ranked of result.beliefs) {
  console.log(`[${ranked.score.final.toFixed(2)}] ${ranked.belief.content}`);
}
```

---

### 2. queryStream()

Streaming variant for large result sets.

```typescript
async function* queryStream(
  query: Query
): AsyncGenerator<StreamChunk, void, unknown>

interface StreamChunk {
  beliefs: RankedBelief[];
  progress: number;                // 0.0-1.0
  done: boolean;
  partial_errors?: QueryError[];
}
```

**Parameters:**
- `query` — Query with `options.stream: true` (implicit)

**Yields:**
- `StreamChunk` — Batches of results as they become available

**Backpressure:** Consumer controls iteration speed.

**Example:**

```typescript
const stream = queryStream({
  semantic: "all beliefs about climate",
  context: { requester_id: myAgentId },
  options: { limit: 1000 }
});

for await (const chunk of stream) {
  console.log(`Progress: ${(chunk.progress * 100).toFixed(0)}%`);
  for (const ranked of chunk.beliefs) {
    await processResult(ranked);
  }
  if (chunk.done) break;
}
```

---

### 3. explain_ranking()

Get detailed explanation of why a belief ranked where it did.

```typescript
async function explain_ranking(
  belief_id: UUID,
  query: Query
): Promise<RankingExplanation>

interface RankingExplanation {
  belief_id: UUID;
  final_score: number;
  
  // Component breakdown
  components: ExplanationComponents;
  
  // Human-readable summary
  summary: string;
  
  // Comparative context
  rank: number;                    // Position in results
  percentile: number;              // 0-100, how this compares to all matches
  
  // What would change the ranking
  improvement_hints: ImprovementHint[];
}

interface ExplanationComponents {
  semantic_similarity: {
    score: number;
    embedding_distance: number;    // Raw cosine distance
    matched_terms: string[];       // Key terms that matched
  };
  
  confidence: {
    score: number;
    vector: ConfidenceVector;
    weights_used: ConfidenceWeights;
    weakest_dimension: string;     // Which dimension hurt most
  };
  
  trust: {
    score: number;
    direct_trust: number | null;   // If requester directly trusts holder
    transitive_trust: number | null; // If trust is transitive
    trust_path?: TrustPathSummary; // How trust was computed
    reputation_factor: number;     // Holder's network reputation
  };
  
  recency: {
    score: number;
    age_days: number;
    freshness_value: number;       // From confidence vector
    decay_applied: number;         // How much decay reduced score
  };
  
  diversity: {
    penalty: number;               // 0 if not penalized
    similar_higher_ranked: number; // Count of similar beliefs ranked higher
    reason?: string;               // Why penalized
  };
}

interface TrustPathSummary {
  hops: number;
  path: UUID[];                    // Agent IDs in path (anonymized if needed)
  damping_applied: number;
  bottleneck_trust: number;        // Lowest trust in path
}

interface ImprovementHint {
  dimension: string;               // Which component
  current: number;
  potential: number;               // If this improved
  suggestion: string;              // Human-readable suggestion
}
```

**Example:**

```typescript
const explanation = await explain_ranking(
  "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  originalQuery
);

console.log(explanation.summary);
// "This belief ranked #3 (top 5%) primarily due to high semantic match (0.92) 
//  and direct trust relationship (0.85). Confidence was moderate (0.68) with 
//  corroboration being the weakest factor (0.3)."

console.log(explanation.improvement_hints);
// [{ dimension: "corroboration", current: 0.3, potential: 0.7,
//    suggestion: "Independent verification would significantly boost ranking" }]
```

---

### 4. subscribe()

Subscribe to live updates for a query.

```typescript
async function subscribe(
  query: Query,
  callback: SubscriptionCallback
): Promise<Subscription>

type SubscriptionCallback = (event: SubscriptionEvent) => void | Promise<void>;

interface Subscription {
  id: UUID;
  query: Query;
  created_at: Timestamp;
  
  // Control methods
  pause(): void;
  resume(): void;
  cancel(): Promise<void>;
  
  // Status
  status: 'active' | 'paused' | 'cancelled';
  events_delivered: number;
  last_event_at: Timestamp | null;
}

interface SubscriptionEvent {
  subscription_id: UUID;
  type: 'added' | 'updated' | 'removed' | 'reranked';
  timestamp: Timestamp;
  
  // The belief
  belief: RankedBelief;
  
  // Change details
  change: ChangeDetails;
}

interface ChangeDetails {
  // For 'added'
  reason_added?: string;           // "New belief matched query"
  
  // For 'updated'
  previous_score?: number;
  score_delta?: number;
  fields_changed?: string[];       // Which belief fields changed
  
  // For 'removed'
  reason_removed?: string;         // "No longer matches filters" / "Visibility changed"
  
  // For 'reranked'
  previous_rank?: number;
  new_rank?: number;
  cause?: string;                  // What caused reranking
}
```

**Options in Query:**

```typescript
interface SubscriptionOptions {
  debounce_ms?: number;            // Batch rapid updates (default: 1000)
  include_removals?: boolean;      // Notify on removal (default: true)
  include_reranks?: boolean;       // Notify on rank changes (default: false)
  max_events_per_minute?: number;  // Rate limit (default: 60)
  notify_on_connect?: boolean;     // Send current results on subscribe (default: true)
}
```

**Example:**

```typescript
const subscription = await subscribe(
  {
    semantic: "Anthropic announcements",
    context: { requester_id: myAgentId },
    options: { scope: { network: true } }
  },
  async (event) => {
    switch (event.type) {
      case 'added':
        console.log(`New belief: ${event.belief.belief.content}`);
        await notifyUser(event.belief);
        break;
      case 'updated':
        console.log(`Belief updated, score: ${event.change.previous_score} → ${event.belief.score.final}`);
        break;
      case 'removed':
        console.log(`Belief no longer matches: ${event.change.reason_removed}`);
        break;
    }
  }
);

// Later...
subscription.pause();  // Temporarily stop events
subscription.resume(); // Continue receiving
await subscription.cancel(); // Clean up
```

---

## Utility Operations

### 5. count()

Get count without fetching results.

```typescript
async function count(
  query: Query
): Promise<CountResult>

interface CountResult {
  total: number;                   // Estimated total matches
  exact: boolean;                  // True if count is exact
  by_scope: {
    local: number;
    federated: number;
    network: number;
  };
}
```

---

### 6. facets()

Get aggregations for filtering UI.

```typescript
async function facets(
  query: Query,
  facet_fields: string[]
): Promise<FacetResult>

interface FacetResult {
  facets: Map<string, FacetValues>;
  query_time_ms: number;
}

interface FacetValues {
  field: string;
  values: FacetValue[];
  other_count: number;             // Count not in top values
}

interface FacetValue {
  value: string;
  count: number;
  selected: boolean;               // If currently filtered to this value
}
```

**Supported facet fields:**
- `domains` — Domain tag distribution
- `holder_id` — Belief holders
- `derivation_type` — How beliefs were derived
- `visibility` — Visibility levels
- `confidence_bucket` — Bucketed confidence ranges

**Example:**

```typescript
const facets = await facets(
  { semantic: "artificial intelligence", context },
  ["domains", "confidence_bucket"]
);

console.log(facets.facets.get("domains"));
// [
//   { value: "tech/ai", count: 1523, selected: false },
//   { value: "ethics", count: 342, selected: false },
//   { value: "business", count: 201, selected: false }
// ]
```

---

### 7. similar()

Find beliefs similar to a given belief.

```typescript
async function similar(
  belief_id: UUID,
  context: TrustContext,
  options?: SimilarOptions
): Promise<QueryResult>

interface SimilarOptions {
  limit?: number;
  include_same_holder?: boolean;   // default false
  min_similarity?: number;         // default 0.7
  scope?: QueryScope;
}
```

**Example:**

```typescript
const similar = await similar(
  "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  { requester_id: myAgentId },
  { limit: 10, min_similarity: 0.8 }
);
```

---

### 8. contradictions()

Find beliefs that contradict a given belief.

```typescript
async function contradictions(
  belief_id: UUID,
  context: TrustContext,
  options?: ContradictionOptions
): Promise<ContradictionResult>

interface ContradictionOptions {
  limit?: number;
  scope?: QueryScope;
  min_confidence?: number;         // Only consider confident contradictions
}

interface ContradictionResult {
  source_belief: Belief;
  contradictions: ContradictingBelief[];
  analysis_confidence: number;     // How confident in contradiction detection
}

interface ContradictingBelief {
  belief: RankedBelief;
  contradiction_type: 'direct' | 'implied' | 'tension';
  explanation: string;
  resolution_suggestion?: string;
}
```

**Example:**

```typescript
const contradictions = await contradictions(
  beliefId,
  { requester_id: myAgentId },
  { scope: { local: true, federated: true } }
);

for (const c of contradictions.contradictions) {
  console.log(`${c.contradiction_type}: ${c.explanation}`);
  console.log(`Suggestion: ${c.resolution_suggestion}`);
}
```

---

## Batch Operations

### 9. queryBatch()

Execute multiple queries efficiently.

```typescript
async function queryBatch(
  queries: Query[]
): Promise<BatchResult>

interface BatchResult {
  results: QueryResult[];          // Same order as input queries
  total_time_ms: number;
  queries_succeeded: number;
  queries_failed: number;
}
```

---

### 10. subscribeMultiple()

Manage multiple subscriptions as a group.

```typescript
async function subscribeMultiple(
  subscriptions: Array<{
    query: Query;
    callback: SubscriptionCallback;
  }>
): Promise<SubscriptionGroup>

interface SubscriptionGroup {
  id: UUID;
  subscriptions: Subscription[];
  
  pauseAll(): void;
  resumeAll(): void;
  cancelAll(): Promise<void>;
}
```

---

## Error Handling

### Error Types

```typescript
class QueryError extends Error {
  code: ErrorCode;
  recoverable: boolean;
  details?: any;
  
  static isRetryable(error: QueryError): boolean;
}

class ValidationError extends QueryError {
  field: string;
  constraint: string;
}

class AuthorizationError extends QueryError {
  required_permission: string;
  actual_permission: string;
}

class TimeoutError extends QueryError {
  elapsed_ms: number;
  partial_results?: QueryResult;
}

class RateLimitError extends QueryError {
  retry_after_ms: number;
  limit: number;
  window_ms: number;
}
```

### Error Handling Pattern

```typescript
try {
  const result = await query(myQuery);
  processResults(result);
} catch (error) {
  if (error instanceof TimeoutError && error.partial_results) {
    // Use partial results
    processResults(error.partial_results);
    console.warn("Query timed out, showing partial results");
  } else if (error instanceof RateLimitError) {
    // Back off and retry
    await sleep(error.retry_after_ms);
    return query(myQuery);
  } else if (QueryError.isRetryable(error)) {
    // Exponential backoff
    return retryWithBackoff(() => query(myQuery));
  } else {
    // Unrecoverable
    throw error;
  }
}
```

---

## Implementation Notes

### Thread Safety

All operations are thread-safe. Multiple concurrent queries are supported.

### Connection Pooling

For cross-node queries, connections are pooled and reused. Configure via:

```typescript
interface ConnectionConfig {
  max_connections_per_node: number;  // default 10
  connection_timeout_ms: number;     // default 5000
  idle_timeout_ms: number;           // default 60000
}
```

### Caching

Query results may be cached. Control via:

```typescript
interface CacheConfig {
  enabled: boolean;
  ttl_ms: number;                    // default 60000
  max_entries: number;               // default 10000
  cache_scope: 'local' | 'shared';
}
```

### Metrics

All operations emit metrics:

```typescript
interface QueryMetrics {
  queries_total: Counter;
  query_duration_ms: Histogram;
  results_returned: Histogram;
  cache_hits: Counter;
  cache_misses: Counter;
  errors_by_code: Counter<ErrorCode>;
  subscriptions_active: Gauge;
}
```

---

*"A well-designed interface makes the complex feel simple."*
