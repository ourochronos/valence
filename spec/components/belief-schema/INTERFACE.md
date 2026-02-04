# Belief Schema Interface

*CRUD operations and query patterns for the Belief store.*

---

## Overview

This document defines the programmatic interface for interacting with beliefs. All operations are designed for:
- **Consistency**: Atomic operations with clear semantics
- **Auditability**: Every change is traceable
- **Performance**: Sub-second queries at billion-belief scale
- **Privacy**: Visibility rules enforced at the interface level

---

## 1. Type Definitions

### 1.1 Input Types

```typescript
// For creating new beliefs
interface CreateBeliefInput {
  content: string;                        // Required: the claim
  confidence?: Partial<ConfidenceVector>; // Optional: custom confidence (defaults applied)
  domains?: string[];                     // Optional: categorical tags (defaults to [])
  visibility?: Visibility;                // Optional: defaults to PRIVATE
  valid_from?: timestamp;                 // Optional: defaults to now()
  valid_until?: timestamp;                // Optional: defaults to null (indefinite)
  derivation?: DerivationInput;           // Optional: how this was derived
}

interface DerivationInput {
  type: DerivationType;
  sources?: DerivationSourceInput[];
  method_description?: string;
  confidence_rationale?: string;
}

interface DerivationSourceInput {
  belief_id?: UUID;                       // Reference to source belief
  external_ref?: string;                  // External URL/citation
  contribution_type?: ContributionType;   // Defaults to PRIMARY
  weight?: number;                        // Defaults to 1.0
}

// For updating beliefs
interface UpdateBeliefInput {
  content?: string;                       // New content (triggers new version)
  confidence?: Partial<ConfidenceVector>; // Confidence adjustments
  domains?: string[];                     // Replace domains
  visibility?: Visibility;                // Change visibility
  valid_until?: timestamp;                // Set expiration
}

// For queries
interface QueryFilters {
  holder_id?: UUID;                       // Filter by holder
  domains?: string[];                     // Filter by domain (ANY match)
  domains_all?: string[];                 // Filter by domain (ALL match)
  visibility?: Visibility[];              // Filter by visibility levels
  min_confidence?: number;                // Minimum overall confidence
  valid_at?: timestamp;                   // Point-in-time query
  created_after?: timestamp;              // Created time filter
  created_before?: timestamp;             // Created time filter
  include_superseded?: boolean;           // Include old versions (default: false)
  derivation_type?: DerivationType[];     // Filter by derivation type
}

interface QueryOptions {
  limit?: number;                         // Max results (default: 20, max: 1000)
  offset?: number;                        // Pagination offset
  order_by?: 'relevance' | 'created_at' | 'confidence'; // Sort order
  order_dir?: 'asc' | 'desc';            // Sort direction
  include_derivation_chain?: boolean;     // Include full derivation (expensive)
  confidence_weights?: number[];          // Custom dimension weights for ranking
}
```

### 1.2 Output Types

```typescript
// Full belief record
interface Belief {
  id: UUID;
  version: number;
  content: string;
  content_hash: string;
  confidence: ConfidenceVector;
  confidence_overall: number;             // Computed aggregate
  valid_from: timestamp;
  valid_until: timestamp | null;
  derivation: Derivation;
  domains: string[];
  visibility: Visibility;
  holder_id: UUID;
  created_at: timestamp;
  supersedes: UUID | null;
  superseded_by: UUID | null;
}

// Query result with relevance
interface BeliefMatch {
  belief: Belief;
  relevance_score: number;                // 0.0-1.0 semantic similarity
  combined_score: number;                 // relevance * confidence_overall
}

interface QueryResult {
  matches: BeliefMatch[];
  total_count: number;                    // Total matching (before limit)
  query_time_ms: number;                  // Query execution time
  filters_applied: QueryFilters;          // Echo back applied filters
}

// Derivation chain result
interface DerivationChainResult {
  root: Belief;
  chain: DerivationLink[];
  max_depth: number;
  total_sources: number;
}
```

---

## 2. Core Operations

### 2.1 create_belief

Create a new belief in the store.

```typescript
function create_belief(
  input: CreateBeliefInput,
  holder_id: UUID                         // Authenticated holder
): Promise<Belief>
```

**Behavior:**
1. Validate input (content length, domain format, etc.)
2. Generate UUIDv7 for `id`
3. Compute `content_hash` (SHA-256)
4. Apply default confidence values for missing dimensions
5. Compute embedding from content + domains
6. Insert belief record
7. Return complete Belief object

**Validation Rules:**
- `content` must be non-empty, max 64KB
- `domains` must match pattern `^[a-z0-9_-]+(\/[a-z0-9_-]+)*$`
- `confidence` values must be in range [0.0, 1.0]
- `valid_from` cannot be more than 1 year in the future
- `derivation.sources[].belief_id` must exist if provided

**Example:**
```python
belief = await create_belief(
    input={
        "content": "Python 3.12 introduced the new 'type' statement for type aliases.",
        "confidence": {
            "source_reliability": 0.95,
            "method_quality": 0.9
        },
        "domains": ["tech/python", "programming/languages"],
        "visibility": "public",
        "derivation": {
            "type": "observation",
            "sources": [{
                "external_ref": "https://docs.python.org/3.12/whatsnew/3.12.html",
                "contribution_type": "primary"
            }],
            "method_description": "Read from official Python documentation"
        }
    },
    holder_id="agent-123"
)
# Returns complete Belief with id, embedding, timestamps, etc.
```

**Errors:**
- `ValidationError`: Invalid input format
- `DuplicateContentError`: Exact content already exists for this holder (optional enforcement)
- `InvalidReferenceError`: Referenced belief_id doesn't exist

---

### 2.2 get_belief

Retrieve a single belief by ID.

```typescript
function get_belief(
  id: UUID,
  options?: {
    include_derivation_chain?: boolean;   // Include full derivation tree
    max_chain_depth?: number;             // Limit derivation depth (default: 5)
  }
): Promise<Belief | null>
```

**Behavior:**
1. Look up belief by ID
2. Check visibility permissions (holder or appropriate access)
3. Optionally fetch derivation chain
4. Return Belief or null if not found/not accessible

**Example:**
```python
belief = await get_belief(
    id="01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
    options={"include_derivation_chain": True, "max_chain_depth": 3}
)
```

**Errors:**
- `NotFoundError`: Belief doesn't exist
- `AccessDeniedError`: Belief exists but requester lacks permission

---

### 2.3 update_belief

Update a belief, creating a new version.

```typescript
function update_belief(
  id: UUID,
  changes: UpdateBeliefInput,
  holder_id: UUID                         // Must match original holder
): Promise<Belief>
```

**Behavior:**
1. Fetch existing belief, verify ownership
2. Determine if content changed (requires new version) or metadata-only
3. If content changed:
   - Create new belief with `supersedes = old.id`
   - Set old belief's `superseded_by = new.id`
   - Increment version number
   - Recompute embedding
   - Add automatic derivation source referencing old belief
4. If metadata-only (confidence, domains, visibility):
   - Still creates new version (immutability)
   - Preserves content and embedding
5. Return new Belief

**Content Change Detection:**
```python
content_changed = (
    changes.content is not None and 
    changes.content != existing.content
)
```

**Example:**
```python
# Correct an error in a belief
updated = await update_belief(
    id="01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
    changes={
        "content": "Python 3.12 introduced the new 'type' statement for type aliases, released October 2023.",
        "confidence": {"source_reliability": 0.98}
    },
    holder_id="agent-123"
)
# Returns new Belief with version=2, supersedes=original_id
```

**Errors:**
- `NotFoundError`: Belief doesn't exist
- `AccessDeniedError`: Requester is not the holder
- `AlreadySupersededError`: Belief already has a newer version

---

### 2.4 supersede_belief

Explicitly replace one belief with another (different content).

```typescript
function supersede_belief(
  old_id: UUID,
  new_belief: CreateBeliefInput,
  holder_id: UUID
): Promise<Belief>
```

**Behavior:**
1. Fetch old belief, verify ownership
2. Create new belief with:
   - `supersedes = old_id`
   - `version = old.version + 1`
   - Automatic derivation source referencing old belief with type `CORRECTION`
3. Set old belief's `superseded_by = new.id`
4. Return new Belief

**Difference from update_belief:**
- `supersede_belief` is for explicit "this replaces that" relationships
- Automatically sets derivation type to `CORRECTION`
- Intended for corrections, not refinements

**Example:**
```python
corrected = await supersede_belief(
    old_id="01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
    new_belief={
        "content": "GPT-4 was released in March 2023 (not January as previously stated).",
        "confidence": {"source_reliability": 0.99},
        "domains": ["tech/ai/llm"],
        "derivation": {
            "type": "correction",
            "sources": [{"external_ref": "https://openai.com/blog/gpt-4"}],
            "method_description": "Verified against official announcement"
        }
    },
    holder_id="agent-123"
)
```

---

### 2.5 query_beliefs

Semantic search over beliefs with filters.

```typescript
function query_beliefs(
  semantic_query: string,                 // Natural language query
  filters?: QueryFilters,
  options?: QueryOptions,
  requester_id: UUID                      // For visibility filtering
): Promise<QueryResult>
```

**Behavior:**
1. Compute embedding for `semantic_query`
2. Apply visibility filters (requester can only see permitted beliefs)
3. Apply explicit filters (domains, time, confidence, etc.)
4. Perform vector similarity search
5. Compute `combined_score = relevance_score * confidence_overall`
6. Sort by specified order (default: combined_score desc)
7. Apply pagination (limit, offset)
8. Return QueryResult

**Visibility Enforcement:**
```sql
WHERE (
    visibility = 'public'
    OR (visibility = 'federated' AND holder_id IN (SELECT ... FROM federation_members WHERE ...))
    OR holder_id = :requester_id
)
```

**Example:**
```python
results = await query_beliefs(
    semantic_query="What programming languages support type aliases?",
    filters={
        "domains": ["tech", "programming"],
        "min_confidence": 0.6,
        "include_superseded": False
    },
    options={
        "limit": 10,
        "order_by": "combined_score",
        "confidence_weights": [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # Custom weights
    },
    requester_id="agent-123"
)

# Returns:
# {
#   "matches": [
#     {"belief": {...}, "relevance_score": 0.89, "combined_score": 0.78},
#     ...
#   ],
#   "total_count": 47,
#   "query_time_ms": 23,
#   "filters_applied": {...}
# }
```

**Performance Targets:**
- p50 latency: < 50ms
- p99 latency: < 200ms
- At billion-belief scale with proper indexing

---

## 3. Additional Operations

### 3.1 get_belief_chain

Retrieve the full version history of a belief.

```typescript
function get_belief_chain(
  belief_id: UUID,
  direction?: 'ancestors' | 'descendants' | 'both'
): Promise<Belief[]>
```

**Example:**
```python
chain = await get_belief_chain(
    belief_id="01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
    direction="both"
)
# Returns array of beliefs in chronological order
```

---

### 3.2 get_derivation_chain

Retrieve the full derivation tree (sources of sources).

```typescript
function get_derivation_chain(
  belief_id: UUID,
  max_depth?: number                      // Default: 5
): Promise<DerivationChainResult>
```

**Example:**
```python
chain = await get_derivation_chain(
    belief_id="01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
    max_depth=3
)
# Returns hierarchical derivation tree
```

---

### 3.3 list_beliefs

List beliefs with filters (no semantic query).

```typescript
function list_beliefs(
  filters?: QueryFilters,
  options?: QueryOptions,
  requester_id: UUID
): Promise<QueryResult>
```

**Same as query_beliefs but without semantic ranking.** Useful for browsing by domain, time, etc.

---

### 3.4 count_beliefs

Count beliefs matching filters.

```typescript
function count_beliefs(
  filters?: QueryFilters,
  requester_id: UUID
): Promise<{count: number}>
```

---

### 3.5 delete_belief (soft)

Mark a belief as deleted (tombstone).

```typescript
function delete_belief(
  id: UUID,
  reason?: string,
  holder_id: UUID
): Promise<Belief>
```

**Behavior:**
1. Create new version with empty content
2. Set `valid_until = now()`
3. Add deletion reason to derivation
4. Return tombstone belief

**Note:** Tombstones are excluded from queries by default but preserved for audit.

---

### 3.6 find_similar

Find beliefs similar to a given belief.

```typescript
function find_similar(
  belief_id: UUID,
  filters?: QueryFilters,
  options?: QueryOptions,
  requester_id: UUID
): Promise<QueryResult>
```

**Behavior:** Uses the belief's embedding directly instead of computing from query text.

---

### 3.7 bulk_create_beliefs

Create multiple beliefs in one transaction.

```typescript
function bulk_create_beliefs(
  inputs: CreateBeliefInput[],
  holder_id: UUID,
  options?: {
    stop_on_error?: boolean;              // Default: true
    return_failures?: boolean;            // Default: false
  }
): Promise<{
  created: Belief[];
  failed: Array<{index: number; error: string}>;
}>
```

**Performance:** Batches embedding computation and database inserts.

---

## 4. Query Patterns

### 4.1 Semantic Search (Primary Use Case)

```python
# "What do I know about X?"
results = await query_beliefs(
    semantic_query="Anthropic's approach to AI safety",
    filters={"min_confidence": 0.5},
    options={"limit": 10}
)
```

### 4.2 Domain Browsing

```python
# "Show me everything in this category"
results = await list_beliefs(
    filters={"domains": ["tech/ai/llm"], "include_superseded": False},
    options={"order_by": "created_at", "order_dir": "desc", "limit": 50}
)
```

### 4.3 Contradiction Detection

```python
# "Find beliefs that might conflict"
base_belief = await get_belief(id="...")
similar = await find_similar(
    belief_id=base_belief.id,
    filters={"min_confidence": 0.7},
    options={"limit": 20}
)
# Then check for semantic contradictions
```

### 4.4 Provenance Audit

```python
# "Where did this belief come from?"
chain = await get_derivation_chain(
    belief_id="...",
    max_depth=10
)
# Trace back through sources
```

### 4.5 Time-Travel Query

```python
# "What did I believe last month?"
results = await query_beliefs(
    semantic_query="project status",
    filters={
        "valid_at": "2024-01-01T00:00:00Z",
        "include_superseded": True
    }
)
```

### 4.6 High-Confidence Facts

```python
# "What am I most certain about?"
results = await list_beliefs(
    filters={"min_confidence": 0.9},
    options={"order_by": "confidence", "order_dir": "desc", "limit": 100}
)
```

---

## 5. Error Handling

### 5.1 Error Types

```typescript
class BeliefError extends Error {
  code: string;
  details?: object;
}

class ValidationError extends BeliefError { code = "VALIDATION_ERROR" }
class NotFoundError extends BeliefError { code = "NOT_FOUND" }
class AccessDeniedError extends BeliefError { code = "ACCESS_DENIED" }
class DuplicateContentError extends BeliefError { code = "DUPLICATE_CONTENT" }
class InvalidReferenceError extends BeliefError { code = "INVALID_REFERENCE" }
class AlreadySupersededError extends BeliefError { code = "ALREADY_SUPERSEDED" }
class RateLimitError extends BeliefError { code = "RATE_LIMIT" }
class EmbeddingError extends BeliefError { code = "EMBEDDING_FAILED" }
```

### 5.2 Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Content exceeds maximum length of 65536 bytes",
    "details": {
      "field": "content",
      "actual_length": 70000,
      "max_length": 65536
    }
  }
}
```

---

## 6. Rate Limits & Quotas

| Operation | Default Limit | Notes |
|-----------|---------------|-------|
| create_belief | 100/min | Per holder |
| query_beliefs | 1000/min | Per requester |
| bulk_create | 10/min | Max 100 beliefs per call |
| get_derivation_chain | 100/min | Expensive recursive query |

Limits are configurable per deployment.

---

## 7. Transactions & Consistency

### 7.1 Atomic Operations

All single operations (create, update, supersede) are atomic. Either fully succeed or fully rollback.

### 7.2 Bulk Operations

Bulk operations can be configured for:
- **All-or-nothing**: Rollback entire batch on any failure
- **Best-effort**: Continue on error, return partial results

### 7.3 Read Consistency

- Single-belief reads: Strong consistency
- Queries: Eventually consistent (may have slight delay after writes)
- Cross-shard queries: Eventual consistency guaranteed within 100ms

---

## 8. Implementation Notes

### 8.1 Embedding Pipeline

```python
async def compute_embedding(content: str, domains: list[str]) -> list[float]:
    text = f"BELIEF: {content}\nDOMAINS: {', '.join(domains)}"
    response = await openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
```

### 8.2 Confidence Aggregation

```python
def compute_overall_confidence(
    cv: ConfidenceVector,
    weights: list[float] = [0.25, 0.20, 0.15, 0.15, 0.15, 0.10]
) -> float:
    values = [
        cv.source_reliability,
        cv.method_quality,
        cv.internal_consistency,
        cv.temporal_freshness,
        cv.corroboration,
        cv.domain_applicability
    ]
    # Weighted geometric mean
    product = 1.0
    total_weight = sum(weights)
    for v, w in zip(values, weights):
        product *= v ** (w / total_weight)
    return product
```

### 8.3 Query Scoring

```python
def compute_combined_score(
    relevance: float,
    confidence: float,
    freshness_decay: float = 1.0
) -> float:
    return relevance * confidence * freshness_decay
```

---

*This interface is the canonical API contract for Belief operations in Valence v1.*
