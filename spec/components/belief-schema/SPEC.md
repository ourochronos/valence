# Belief Schema Specification

*The foundation of Valence's epistemic infrastructure.*

---

## Overview

A **Belief** is the atomic unit of knowledge in Valence. Unlike raw facts, beliefs carry rich metadata about confidence, provenance, validity, and derivation. This spec defines the complete data model for beliefs and their relationships.

---

## 1. Core Belief Structure

### 1.1 Belief

```typescript
interface Belief {
  // === Identity ===
  id: UUID;                      // Unique identifier (UUIDv7 for time-ordering)
  version: number;               // Monotonic version counter (starts at 1)
  
  // === Content ===
  content: string;               // The claim itself (UTF-8, max 64KB)
  content_hash: string;          // SHA-256 of content (for deduplication)
  
  // === Confidence ===
  confidence: ConfidenceVector;  // Multi-dimensional confidence scores
  
  // === Temporal Validity ===
  valid_from: timestamp;         // When this belief became/becomes valid
  valid_until: timestamp | null; // When this belief expires (null = indefinite)
  
  // === Derivation ===
  derivation: Derivation;        // How this belief was formed
  
  // === Organization ===
  domains: string[];             // Categorical tags (e.g., ["tech/ai", "projects/valence"])
  
  // === Privacy ===
  visibility: Visibility;        // Who can see this belief
  
  // === Provenance ===
  holder_id: UUID;               // Agent who holds this belief
  created_at: timestamp;         // When this version was created
  
  // === Versioning ===
  supersedes: UUID | null;       // Previous version this belief supersedes
  superseded_by: UUID | null;    // Newer version that supersedes this (back-link)
  
  // === Search ===
  embedding: float[];            // Vector embedding for semantic search
}
```

### 1.2 ConfidenceVector

Six orthogonal dimensions of confidence, each scored 0.0 to 1.0:

```typescript
interface ConfidenceVector {
  source_reliability: number;     // How trustworthy is the origin? (0.0-1.0)
  method_quality: number;         // How sound was the derivation method? (0.0-1.0)
  internal_consistency: number;   // Does this contradict other beliefs? (0.0-1.0)
  temporal_freshness: number;     // Is this current? (0.0-1.0, decays over time)
  corroboration: number;          // How many independent sources agree? (0.0-1.0)
  domain_applicability: number;   // How well-matched to claimed domains? (0.0-1.0)
}
```

**Aggregation Rules:**
- **overall_confidence** = weighted geometric mean of all dimensions
- Default weights: `[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]`
- Weights are configurable per-query based on context

**Default Values (new beliefs):**
```typescript
{
  source_reliability: 0.5,      // Unknown source
  method_quality: 0.5,          // Unknown method
  internal_consistency: 1.0,    // No known contradictions
  temporal_freshness: 1.0,      // Fresh at creation
  corroboration: 0.1,           // Single source
  domain_applicability: 0.8     // Self-assigned domains
}
```

### 1.3 Visibility

```typescript
enum Visibility {
  PRIVATE = 'private';           // Only holder can see
  FEDERATED = 'federated';       // Shared with holder's federation(s)
  PUBLIC = 'public';             // Visible to entire network
}
```

**Access Control Matrix:**

| Visibility | Holder | Same Federation | Network | Queries |
|------------|--------|-----------------|---------|---------|
| PRIVATE    | ✅     | ❌              | ❌      | Local only |
| FEDERATED  | ✅     | ✅              | ❌      | Federation scope |
| PUBLIC     | ✅     | ✅              | ✅      | Global scope |

---

## 2. Derivation Structure

### 2.1 Derivation

Every belief has exactly one derivation explaining its origin:

```typescript
interface Derivation {
  type: DerivationType;              // How was this belief formed?
  sources: DerivationSource[];       // What beliefs/data contributed?
  method_description: string | null; // Human-readable explanation
  confidence_rationale: string | null; // Why these confidence scores?
}

enum DerivationType {
  OBSERVATION = 'observation';       // Direct observation/measurement
  INFERENCE = 'inference';           // Logical derivation from other beliefs
  AGGREGATION = 'aggregation';       // Statistical aggregation of beliefs
  HEARSAY = 'hearsay';              // Reported by another agent
  ASSUMPTION = 'assumption';         // Asserted without direct evidence
  CORRECTION = 'correction';         // Supersedes a previous incorrect belief
  SYNTHESIS = 'synthesis';           // LLM/AI-generated combination
}
```

### 2.2 DerivationSource

```typescript
interface DerivationSource {
  belief_id: UUID | null;           // Reference to source belief (if any)
  external_ref: string | null;      // External reference (URL, citation)
  contribution_type: ContributionType;
  weight: number;                   // How much this source contributed (0.0-1.0)
}

enum ContributionType {
  PRIMARY = 'primary';              // Main evidence for this belief
  SUPPORTING = 'supporting';        // Additional supporting evidence
  CONTRADICTING = 'contradicting';  // Counter-evidence (for nuanced beliefs)
  CONTEXT = 'context';              // Background context
}
```

### 2.3 DerivationChain (Computed View)

For querying the full derivation history:

```typescript
interface DerivationChain {
  belief_id: UUID;
  depth: number;                    // 0 = target belief, 1 = immediate sources, etc.
  chain: DerivationLink[];
}

interface DerivationLink {
  belief_id: UUID;
  content_preview: string;          // First 200 chars of content
  derivation_type: DerivationType;
  confidence_overall: number;
  created_at: timestamp;
}
```

---

## 3. Embedding Strategy

### 3.1 Model Selection

**Primary Model:** `text-embedding-3-small` (OpenAI)
- Dimensions: 1536
- Cost: ~$0.00002 per 1K tokens
- Performance: Excellent for semantic similarity
- Availability: Widespread, stable API

**Future Consideration:** Local models (e.g., `nomic-embed-text`) for privacy-sensitive deployments.

### 3.2 What Gets Embedded

**Embedding Input Template:**
```
BELIEF: {content}
DOMAINS: {domains.join(", ")}
```

**Rationale:**
- Content is the primary semantic signal
- Domains provide categorical context for disambiguation
- Simple template enables consistent reconstruction

**Example:**
```
BELIEF: Claude is developed by Anthropic and released in March 2023
DOMAINS: tech/ai, companies/anthropic, products
```

### 3.3 Embedding Storage

- Store as `vector(1536)` in PGVector
- Index with HNSW for sub-linear query time
- Normalized to unit length for cosine similarity

### 3.4 Re-embedding Triggers

Embeddings are computed once and cached. Re-embed when:
1. Model version changes (manual migration)
2. Content is updated (creates new version anyway)
3. Batch recomputation for improved models

---

## 4. Versioning

### 4.1 Immutability Principle

**Beliefs are immutable.** Once created, a belief's content never changes. Updates create new versions with explicit supersession relationships.

### 4.2 Version Lifecycle

```
┌─────────────┐      supersedes      ┌─────────────┐
│  Belief v1  │ ◄───────────────────│  Belief v2  │
│  (active)   │                      │  (active)   │
└─────────────┘                      └─────────────┘
       │                                    │
       ▼                                    ▼
   superseded_by                       (current)
   back-linked
```

### 4.3 Version Chain Operations

**Create new version:**
1. Create new belief with `supersedes = old_belief.id`
2. Update old belief's `superseded_by = new_belief.id`
3. New belief inherits `version = old_belief.version + 1`
4. Derivation includes old belief as source with type `CORRECTION`

**Query semantics:**
- Default: Return only latest version (where `superseded_by IS NULL`)
- Historical: Return all versions in chain
- Point-in-time: Return version valid at specific timestamp

### 4.4 Deletion

Beliefs are **never deleted**. Instead:
1. Create tombstone version with `valid_until = now()`
2. Tombstone has empty content and special derivation type
3. Queries exclude tombstoned beliefs by default

---

## 5. Identifiers

### 5.1 Belief ID

**Format:** UUIDv7 (time-ordered UUID)

**Benefits:**
- Sortable by creation time
- No coordination required
- 128-bit collision resistance
- Lexicographically sortable for efficient indexes

**Generation:**
```python
import uuid
from datetime import datetime

def generate_belief_id():
    return uuid.uuid7()  # Python 3.12+ or uuid7 package
```

### 5.2 Content Hash

**Format:** SHA-256 hex string (64 characters)

**Purpose:**
- Deduplication detection
- Integrity verification
- Content-addressable lookups

**Computation:**
```python
import hashlib

def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
```

---

## 6. Domain Taxonomy

### 6.1 Structure

Domains are hierarchical paths using `/` separators:

```
tech/ai/llm
people/chris
projects/valence/schema
meta/preferences
```

### 6.2 Reserved Domains

| Domain | Purpose |
|--------|---------|
| `meta/*` | Beliefs about the system itself |
| `meta/preferences` | User preferences |
| `meta/corrections` | Error corrections |
| `self/*` | Self-referential beliefs (about the agent) |
| `system/*` | System-managed beliefs |

### 6.3 Domain Validation

- Max depth: 5 levels
- Max length per segment: 32 characters
- Allowed characters: `a-z`, `0-9`, `-`, `_`
- Case-insensitive (stored lowercase)

---

## 7. Constraints & Limits

### 7.1 Size Limits

| Field | Limit | Rationale |
|-------|-------|-----------|
| content | 64 KB | Generous for text, prevents abuse |
| domains[] | 20 items | Reasonable categorization |
| domain segment | 32 chars | Readable identifiers |
| derivation sources | 100 items | Complex beliefs allowed |
| method_description | 4 KB | Detailed explanations |
| confidence_rationale | 4 KB | Detailed explanations |

### 7.2 Cardinality

- One holder per belief
- One derivation per belief (but multiple sources)
- One embedding per belief
- Unlimited versions per belief chain

### 7.3 Indexing Requirements

For billion-scale operation:

| Query Pattern | Index Strategy |
|--------------|----------------|
| By ID | B-tree on `id` |
| By holder | B-tree on `holder_id` |
| By domain | GIN on `domains` |
| Semantic search | HNSW on `embedding` |
| By content hash | B-tree on `content_hash` |
| Active beliefs | Partial index where `superseded_by IS NULL` |
| By time | B-tree on `created_at` |

---

## 8. Temporal Semantics

### 8.1 Timestamps

All timestamps are **UTC** with microsecond precision:
- Format: ISO 8601 (`2024-01-15T10:30:00.123456Z`)
- Storage: `TIMESTAMP WITH TIME ZONE`

### 8.2 Validity Windows

```
                    valid_from              valid_until
                        │                       │
    ────────────────────┼───────────────────────┼────────────────────►
         (not yet       │      (valid)          │    (expired)
          valid)        │                       │
```

**Query logic:**
```sql
WHERE valid_from <= :query_time
  AND (valid_until IS NULL OR valid_until > :query_time)
```

### 8.3 Freshness Decay

`temporal_freshness` decays over time:

```python
def compute_freshness(created_at: datetime, half_life_days: int = 90) -> float:
    age_days = (now() - created_at).days
    return 0.5 ** (age_days / half_life_days)
```

- Half-life of 90 days by default
- Configurable per domain (e.g., news = 7 days, history = 3650 days)
- Recomputed at query time, not stored

---

## 9. Examples

### 9.1 Simple Factual Belief

```json
{
  "id": "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  "version": 1,
  "content": "Anthropic is headquartered in San Francisco, California.",
  "content_hash": "a3f2b8c1...",
  "confidence": {
    "source_reliability": 0.95,
    "method_quality": 0.9,
    "internal_consistency": 1.0,
    "temporal_freshness": 0.98,
    "corroboration": 0.85,
    "domain_applicability": 0.95
  },
  "valid_from": "2024-01-01T00:00:00Z",
  "valid_until": null,
  "derivation": {
    "type": "observation",
    "sources": [{
      "external_ref": "https://anthropic.com/about",
      "contribution_type": "primary",
      "weight": 1.0
    }],
    "method_description": "Verified from official company website",
    "confidence_rationale": "High confidence - authoritative source, easily verifiable"
  },
  "domains": ["companies/anthropic", "geography/us/california"],
  "visibility": "public",
  "holder_id": "agent-123",
  "created_at": "2024-01-15T10:30:00Z",
  "supersedes": null,
  "superseded_by": null
}
```

### 9.2 Derived Belief

```json
{
  "id": "01941d3b-8d6c-7c1b-9f3e-2b4c5d6e7f8a",
  "version": 1,
  "content": "Most major AI labs are located in the San Francisco Bay Area.",
  "content_hash": "b4g3c9d2...",
  "confidence": {
    "source_reliability": 0.75,
    "method_quality": 0.7,
    "internal_consistency": 0.95,
    "temporal_freshness": 0.95,
    "corroboration": 0.6,
    "domain_applicability": 0.9
  },
  "valid_from": "2024-01-15T00:00:00Z",
  "valid_until": null,
  "derivation": {
    "type": "inference",
    "sources": [
      {"belief_id": "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f", "contribution_type": "supporting", "weight": 0.3},
      {"belief_id": "01941d3a-8c6c-7c1b-8e3d-2a4b5c6d7e8f", "contribution_type": "supporting", "weight": 0.3},
      {"belief_id": "01941d3a-9d7d-7d2c-8f4e-3a5b6c7d8e9f", "contribution_type": "supporting", "weight": 0.3}
    ],
    "method_description": "Inferred from locations of Anthropic, OpenAI, and Google DeepMind offices",
    "confidence_rationale": "Moderate confidence - inference from limited sample, 'most' is imprecise"
  },
  "domains": ["tech/ai", "geography/us/california"],
  "visibility": "public",
  "holder_id": "agent-123",
  "created_at": "2024-01-15T11:00:00Z",
  "supersedes": null,
  "superseded_by": null
}
```

### 9.3 Superseded Belief

```json
{
  "id": "01941d3c-9e8e-7e3d-af5f-4b6c7d8e9f0a",
  "version": 2,
  "content": "GPT-4 was released in March 2023, not January 2023.",
  "content_hash": "c5h4d0e3...",
  "confidence": {
    "source_reliability": 0.99,
    "method_quality": 0.95,
    "internal_consistency": 1.0,
    "temporal_freshness": 0.99,
    "corroboration": 0.95,
    "domain_applicability": 1.0
  },
  "valid_from": "2023-03-14T00:00:00Z",
  "valid_until": null,
  "derivation": {
    "type": "correction",
    "sources": [
      {"belief_id": "01941d3a-0000-0000-0000-000000000000", "contribution_type": "contradicting", "weight": 0.5},
      {"external_ref": "https://openai.com/blog/gpt-4", "contribution_type": "primary", "weight": 1.0}
    ],
    "method_description": "Corrected based on official OpenAI announcement date",
    "confidence_rationale": "Very high confidence - official source contradicts previous belief"
  },
  "domains": ["tech/ai/llm", "products/openai"],
  "visibility": "public",
  "holder_id": "agent-123",
  "created_at": "2024-01-15T12:00:00Z",
  "supersedes": "01941d3a-0000-0000-0000-000000000000",
  "superseded_by": null
}
```

---

## 10. Design Rationale

### Why six confidence dimensions?

Binary true/false is too coarse. A single 0-1 score loses information. Six dimensions capture the key epistemological factors while remaining tractable.

### Why immutable beliefs?

Mutability creates provenance problems. With immutability, every historical state is preserved, audits are trivial, and distributed sync is simpler (no conflict resolution).

### Why UUIDv7?

Time-ordered UUIDs enable efficient time-range queries without separate indexes, maintain global uniqueness without coordination, and sort naturally in B-trees.

### Why geometric mean for aggregation?

Geometric mean ensures a single low score significantly impacts overall confidence. This matches intuition: a belief with 0.9 on five dimensions but 0.1 on one is weak overall.

### Why domains as arrays?

Beliefs often span multiple categories. Array storage with GIN indexes supports efficient multi-domain queries while keeping the model simple.

---

## 11. Future Considerations

### 11.1 Encryption at Rest

For private beliefs:
- Content encrypted with holder's key
- Embedding computed before encryption (or use encrypted embeddings)
- Metadata may remain searchable

### 11.2 Signatures

For verification:
- Holder signs belief hash with Ed25519 key
- Signatures enable trustless verification
- Signature field added when cryptographic identity is ready

### 11.3 Chunking

For long documents:
- Split into semantic chunks
- Each chunk is a separate belief
- Parent belief references chunks in derivation

---

*This spec is the canonical reference for Belief structure in Valence v1.*
