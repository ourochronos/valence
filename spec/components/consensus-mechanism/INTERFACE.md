# Consensus Mechanism — Interface

*API for querying and participating in Valence consensus.*

---

## Overview

This document defines the programmatic interface to the Consensus Mechanism. All operations are available via:
- **Local calls** (same-process)
- **MCP protocol** (cross-process)
- **Network RPC** (cross-node)

All operations that modify state require cryptographic signatures. Read operations may be anonymous.

---

## 1. Core Types

### 1.1 Enumerations

```typescript
enum Layer {
  L1_PERSONAL = 1
  L2_FEDERATED = 2
  L3_DOMAIN = 3
  L4_COMMUNAL = 4
}

enum FinalityStatus {
  PROVISIONAL = 'provisional'       // May change, few verifications
  STRENGTHENING = 'strengthening'   // Gaining corroboration
  FINAL = 'final'                   // High confidence, stable
  CONTESTED = 'contested'           // Active challenges
}

enum ChallengeType {
  FACTUAL_ERROR = 'factual_error'
  INDEPENDENCE_VIOLATION = 'independence'
  EVIDENCE_QUALITY = 'evidence_quality'
  OUTDATED = 'outdated'
  METHODOLOGY = 'methodology'
  SCOPE = 'scope'
}

enum ChallengeStatus {
  OPEN = 'open'
  UNDER_REVIEW = 'under_review'
  UPHELD = 'upheld'
  REJECTED = 'rejected'
  WITHDRAWN = 'withdrawn'
}

enum CorroborationResult {
  ACCEPTED = 'accepted'             // Valid corroboration recorded
  REJECTED_INSUFFICIENT = 'rejected_insufficient'  // Not independent enough
  REJECTED_DUPLICATE = 'rejected_duplicate'        // Already recorded
  REJECTED_SELF = 'rejected_self'                  // Can't corroborate own belief
  PENDING = 'pending'               // Needs verification
}
```

### 1.2 Core Data Types

```typescript
interface ConsensusStatus {
  belief_id: UUID
  current_layer: Layer
  
  corroboration: CorroborationSummary
  elevation_progress: ElevationProgress
  challenges: ChallengeSummary
  
  finality: FinalityStatus
  finality_confidence: float        // 0.0-1.0
  
  layer_history: LayerTransition[]
  last_updated: Timestamp
}

interface CorroborationSummary {
  total_corroborations: uint64
  independent_corroborations: uint64
  weighted_corroboration: float
  
  from_individuals: uint64
  from_federations: uint64
  from_domains: uint64
}

interface ElevationProgress {
  next_layer: Layer | null
  requirements_met: string[]
  requirements_pending: string[]
  blockers: string[]
  estimated_time_to_elevation?: Duration
}

interface ChallengeSummary {
  active_count: uint64
  total_challenge_weight: float
  strongest_challenge?: ChallengeRef
}

interface ChallengeRef {
  id: UUID
  type: ChallengeType
  weight: float
}

interface LayerTransition {
  from_layer: Layer
  to_layer: Layer
  timestamp: Timestamp
  reason: string
  triggered_by: DID | 'system'
}
```

### 1.3 Communal Knowledge Types

```typescript
interface CommunalBelief {
  id: UUID
  canonical_content: string
  communal_confidence: ConfidenceVector
  
  source_summary: SourceSummary
  domains: string[]
  
  established_at: Timestamp
  last_verified: Timestamp
  valid_until?: Timestamp
  
  status: CommunalStatus
  supersedes?: UUID[]
  superseded_by?: UUID
}

interface SourceSummary {
  domain_count: uint64
  federation_count: uint64
  verification_count: uint64
  unique_evidence_chains: uint64
}

enum CommunalStatus {
  ACTIVE = 'active'
  CONTESTED = 'contested'
  REVISED = 'revised'
  SUPERSEDED = 'superseded'
  EXPIRED = 'expired'
}
```

### 1.4 Evidence & Corroboration Types

```typescript
interface IndependentEvidence {
  belief_id?: UUID                  // If corroborating with an existing belief
  content?: string                  // If providing new evidence
  confidence?: ConfidenceVector
  
  derivation: DerivationChain       // How was this derived?
  sources: SourceReference[]        // Original sources
  
  independence_claim: IndependenceClaim
}

interface IndependenceClaim {
  // Declare why this is independent
  different_sources: boolean
  different_method: boolean
  independent_observer: boolean
  
  explanation: string               // How is this independent?
}

interface CorroborationResult {
  status: CorroborationResultStatus
  corroboration_id?: UUID           // If accepted
  
  independence_score?: IndependenceScore
  
  rejection_reason?: string
  suggestions?: string[]            // How to improve
  
  effects: {
    belief_layer_changed: boolean
    new_layer?: Layer
    confidence_delta: float
  }
}

interface IndependenceScore {
  evidential_independence: float
  source_independence: float
  method_independence: float
  temporal_independence: float
  overall: float
}
```

### 1.5 Challenge Types

```typescript
interface Challenge {
  id: UUID
  target_belief_id: UUID
  target_layer: Layer
  
  challenger_id: DID
  stake: Stake
  
  challenge_type: ChallengeType
  counter_evidence: Evidence[]
  reasoning: string
  
  status: ChallengeStatus
  
  resolution?: ChallengeResolution
  
  created_at: Timestamp
  review_deadline: Timestamp
  resolved_at?: Timestamp
}

interface ChallengeResolution {
  outcome: 'upheld' | 'rejected' | 'partial'
  resolved_by: DID[]
  reasoning: string
  
  effects: {
    belief_demoted: boolean
    belief_revised: boolean
    new_layer?: Layer
    challenger_reward: float
    holder_penalty: float
  }
}

interface ChallengeSubmission {
  target_belief_id: UUID
  challenge_type: ChallengeType
  
  counter_evidence: Evidence[]
  reasoning: string
  
  stake_amount: float               // Reputation to risk
}
```

---

## 2. Consensus Operations

### 2.1 check_consensus

Check the consensus status of a belief.

```typescript
function check_consensus(
  belief_id: UUID,
  options?: CheckConsensusOptions
): Promise<ConsensusStatus>

interface CheckConsensusOptions {
  include_history?: boolean         // Include layer transitions (default: true)
  include_challenges?: boolean      // Include challenge details (default: true)
  compute_projections?: boolean     // Estimate elevation timing (default: false)
}
```

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "consensus.check_consensus",
  "params": {
    "belief_id": "550e8400-e29b-41d4-a716-446655440000",
    "options": {
      "include_history": true,
      "compute_projections": true
    }
  },
  "id": 1
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "belief_id": "550e8400-e29b-41d4-a716-446655440000",
    "current_layer": 3,
    "corroboration": {
      "total_corroborations": 47,
      "independent_corroborations": 23,
      "weighted_corroboration": 18.7,
      "from_individuals": 12,
      "from_federations": 8,
      "from_domains": 3
    },
    "elevation_progress": {
      "next_layer": 4,
      "requirements_met": [
        "independent_domains >= 3",
        "independence_score > 0.7",
        "verification_count >= 10"
      ],
      "requirements_pending": [
        "minimum_age >= 7 days (current: 5 days)"
      ],
      "blockers": [],
      "estimated_time_to_elevation": "P2D"
    },
    "challenges": {
      "active_count": 1,
      "total_challenge_weight": 0.03,
      "strongest_challenge": {
        "id": "660e8400-e29b-41d4-a716-446655440001",
        "type": "scope",
        "weight": 0.03
      }
    },
    "finality": "strengthening",
    "finality_confidence": 0.78,
    "layer_history": [
      {
        "from_layer": 1,
        "to_layer": 2,
        "timestamp": "2026-01-15T10:30:00Z",
        "reason": "Federation threshold met",
        "triggered_by": "system"
      },
      {
        "from_layer": 2,
        "to_layer": 3,
        "timestamp": "2026-01-22T14:15:00Z",
        "reason": "Expert verification complete",
        "triggered_by": "did:valence:agent:expert123"
      }
    ],
    "last_updated": "2026-01-29T08:45:00Z"
  },
  "id": 1
}
```

**Errors:**
- `BELIEF_NOT_FOUND`: Belief ID doesn't exist
- `ACCESS_DENIED`: Caller lacks permission to view this belief's status

---

### 2.2 get_communal_knowledge

Retrieve communal knowledge (L4) on a topic.

```typescript
function get_communal_knowledge(
  query: CommunalQuery
): Promise<CommunalKnowledgeResponse>

interface CommunalQuery {
  // Query methods (at least one required)
  topic?: string                    // Natural language topic
  semantic_query?: string           // Semantic search
  domain?: string                   // Filter by domain
  
  // Filters
  min_confidence?: float            // Minimum overall confidence
  status?: CommunalStatus[]         // Filter by status (default: [ACTIVE])
  established_after?: Timestamp     // Only knowledge established after this time
  
  // Pagination
  limit?: uint32                    // Max results (default: 20, max: 100)
  offset?: uint32                   // Skip first N results
  
  // Ordering
  order_by?: 'confidence' | 'recency' | 'relevance'  // default: relevance
}

interface CommunalKnowledgeResponse {
  beliefs: CommunalBelief[]
  total_count: uint64
  has_more: boolean
  
  query_metadata: {
    domains_searched: string[]
    semantic_similarity_threshold: float
  }
}
```

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "consensus.get_communal_knowledge",
  "params": {
    "query": {
      "topic": "climate change effects on coral reefs",
      "domain": "marine_biology",
      "min_confidence": 0.7,
      "limit": 10,
      "order_by": "confidence"
    }
  },
  "id": 2
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "beliefs": [
      {
        "id": "770e8400-e29b-41d4-a716-446655440002",
        "canonical_content": "Ocean acidification caused by increased CO2 absorption reduces coral calcification rates by 15-30% under doubled atmospheric CO2 concentrations.",
        "communal_confidence": {
          "source_reliability": 0.91,
          "method_quality": 0.88,
          "internal_consistency": 0.94,
          "temporal_freshness": 0.82,
          "corroboration": 0.96,
          "domain_applicability": 0.95
        },
        "source_summary": {
          "domain_count": 4,
          "federation_count": 12,
          "verification_count": 67,
          "unique_evidence_chains": 23
        },
        "domains": ["marine_biology", "climate_science", "chemistry"],
        "established_at": "2025-09-15T00:00:00Z",
        "last_verified": "2026-01-28T12:00:00Z",
        "status": "active"
      }
    ],
    "total_count": 1,
    "has_more": false,
    "query_metadata": {
      "domains_searched": ["marine_biology", "climate_science"],
      "semantic_similarity_threshold": 0.85
    }
  },
  "id": 2
}
```

**Errors:**
- `INVALID_QUERY`: Query parameters invalid
- `DOMAIN_NOT_FOUND`: Specified domain doesn't exist

---

### 2.3 submit_corroboration

Submit independent evidence corroborating an existing belief.

```typescript
function submit_corroboration(
  belief_id: UUID,
  evidence: IndependentEvidence,
  options?: CorroborationOptions
): Promise<CorroborationResult>

interface CorroborationOptions {
  request_elevation_check?: boolean // Trigger elevation check if thresholds met
  notify_holder?: boolean           // Notify the belief holder
}
```

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "consensus.submit_corroboration",
  "params": {
    "belief_id": "550e8400-e29b-41d4-a716-446655440000",
    "evidence": {
      "content": "Coral bleaching events have increased 4x since 1980s based on NOAA monitoring data.",
      "confidence": {
        "source_reliability": 0.9,
        "method_quality": 0.85,
        "internal_consistency": 0.92,
        "temporal_freshness": 0.8,
        "corroboration": 0.7,
        "domain_applicability": 0.95
      },
      "derivation": {
        "type": "observation",
        "steps": ["Retrieved NOAA coral monitoring database", "Computed frequency trends"]
      },
      "sources": [
        {
          "type": "external",
          "url": "https://www.noaa.gov/coral-reef-watch",
          "accessed_at": "2026-01-28T10:00:00Z"
        }
      ],
      "independence_claim": {
        "different_sources": true,
        "different_method": true,
        "independent_observer": true,
        "explanation": "Original belief derived from academic literature review; this corroboration uses direct government monitoring data with different methodology."
      }
    },
    "options": {
      "request_elevation_check": true
    }
  },
  "id": 3
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "status": "accepted",
    "corroboration_id": "880e8400-e29b-41d4-a716-446655440003",
    "independence_score": {
      "evidential_independence": 0.85,
      "source_independence": 0.92,
      "method_independence": 0.78,
      "temporal_independence": 0.45,
      "overall": 0.79
    },
    "effects": {
      "belief_layer_changed": false,
      "confidence_delta": 0.03
    }
  },
  "id": 3
}
```

**Errors:**
- `BELIEF_NOT_FOUND`: Target belief doesn't exist
- `INSUFFICIENT_INDEPENDENCE`: Evidence not independent enough (< 0.5)
- `SELF_CORROBORATION`: Cannot corroborate your own belief
- `DUPLICATE_CORROBORATION`: This evidence chain already recorded
- `INSUFFICIENT_STAKE`: Caller lacks reputation to submit corroboration
- `BELIEF_LOCKED`: Belief is under active challenge, no new corroborations

---

### 2.4 challenge_consensus

Challenge an established belief with counter-evidence.

```typescript
function challenge_consensus(
  challenge: ChallengeSubmission
): Promise<ChallengeResponse>

interface ChallengeResponse {
  challenge_id: UUID
  status: ChallengeStatus
  
  stake_locked: Stake
  review_deadline: Timestamp
  
  estimated_resolution_time: Duration
  
  immediate_effects: {
    belief_status_changed: boolean
    new_status?: string
  }
}
```

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "consensus.challenge_consensus",
  "params": {
    "target_belief_id": "550e8400-e29b-41d4-a716-446655440000",
    "challenge_type": "outdated",
    "counter_evidence": [
      {
        "type": "external",
        "url": "https://doi.org/10.1234/recent-study",
        "content": "2026 study shows coral adaptation mechanisms previously unknown, revising bleaching projections downward.",
        "source_reputation": 0.88
      }
    ],
    "reasoning": "Recent research from January 2026 demonstrates coral thermal adaptation not accounted for in the original belief. The 15-30% calcification reduction estimate is based on 2020-era models that don't include these adaptation mechanisms.",
    "stake_amount": 0.05
  },
  "id": 4
}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "challenge_id": "990e8400-e29b-41d4-a716-446655440004",
    "status": "open",
    "stake_locked": {
      "amount": 0.05,
      "locked_until": "2026-02-17T12:00:00Z"
    },
    "review_deadline": "2026-02-10T12:00:00Z",
    "estimated_resolution_time": "P7D",
    "immediate_effects": {
      "belief_status_changed": true,
      "new_status": "contested"
    }
  },
  "id": 4
}
```

**Errors:**
- `BELIEF_NOT_FOUND`: Target belief doesn't exist
- `INSUFFICIENT_STAKE`: Caller lacks required reputation to stake
- `ALREADY_CHALLENGED`: Active challenge of same type exists
- `RATE_LIMITED`: Too many challenges from this caller
- `INVALID_EVIDENCE`: Counter-evidence doesn't meet requirements
- `LAYER_TOO_LOW`: Cannot challenge L1 beliefs (personal beliefs are not consensus)

---

## 3. Query Operations

### 3.1 get_elevation_requirements

Get the requirements for a belief to elevate to the next layer.

```typescript
function get_elevation_requirements(
  belief_id: UUID
): Promise<ElevationRequirements>

interface ElevationRequirements {
  current_layer: Layer
  next_layer: Layer | null
  
  requirements: Requirement[]
  
  estimated_effort: EffortEstimate
}

interface Requirement {
  name: string
  description: string
  current_value: any
  required_value: any
  met: boolean
  
  how_to_meet?: string             // Guidance on meeting this requirement
}

interface EffortEstimate {
  corroborations_needed: uint32
  verifications_needed: uint32
  time_remaining?: Duration
  
  difficulty: 'trivial' | 'easy' | 'moderate' | 'hard' | 'very_hard'
}
```

**Example Response:**
```json
{
  "current_layer": 2,
  "next_layer": 3,
  "requirements": [
    {
      "name": "federation_count",
      "description": "Corroboration from independent federations",
      "current_value": 2,
      "required_value": 3,
      "met": false,
      "how_to_meet": "Share belief with additional federations in related domains"
    },
    {
      "name": "independence_score",
      "description": "Average independence of corroborations",
      "current_value": 0.62,
      "required_value": 0.5,
      "met": true
    },
    {
      "name": "expert_verification",
      "description": "Verification by domain experts",
      "current_value": 1,
      "required_value": 2,
      "met": false,
      "how_to_meet": "Request verification from agents with domain reputation > 0.7"
    }
  ],
  "estimated_effort": {
    "corroborations_needed": 5,
    "verifications_needed": 1,
    "difficulty": "moderate"
  }
}
```

---

### 3.2 get_corroboration_graph

Get the network of corroborating evidence for a belief.

```typescript
function get_corroboration_graph(
  belief_id: UUID,
  options?: GraphOptions
): Promise<CorroborationGraph>

interface GraphOptions {
  max_depth?: uint8                 // How deep to trace (default: 2)
  include_derivations?: boolean     // Include derivation chains
  anonymize_contributors?: boolean  // Hide specific contributor IDs
}

interface CorroborationGraph {
  root: BeliefNode
  nodes: BeliefNode[]
  edges: CorroborationEdge[]
  
  summary: {
    total_nodes: uint64
    total_edges: uint64
    unique_sources: uint64
    independence_distribution: Distribution
  }
}

interface BeliefNode {
  id: UUID
  content_summary: string           // Truncated content
  layer: Layer
  holder_anonymized?: string        // e.g., "Contributor A"
  holder_id?: DID                   // If not anonymized
  confidence: float                 // Overall confidence
}

interface CorroborationEdge {
  from_id: UUID
  to_id: UUID
  independence_score: float
  edge_type: 'corroboration' | 'derivation' | 'citation'
}
```

---

### 3.3 list_challenges

List challenges against a belief or by a challenger.

```typescript
function list_challenges(
  filter: ChallengeFilter
): Promise<ChallengeList>

interface ChallengeFilter {
  belief_id?: UUID                  // Challenges against this belief
  challenger_id?: DID               // Challenges by this agent
  status?: ChallengeStatus[]        // Filter by status
  type?: ChallengeType[]            // Filter by type
  
  created_after?: Timestamp
  created_before?: Timestamp
  
  limit?: uint32
  offset?: uint32
}

interface ChallengeList {
  challenges: ChallengeSummary[]
  total_count: uint64
  has_more: boolean
}

interface ChallengeSummary {
  id: UUID
  target_belief_id: UUID
  target_belief_content: string     // Truncated
  challenger_id: DID
  
  type: ChallengeType
  status: ChallengeStatus
  stake: float
  
  created_at: Timestamp
  review_deadline: Timestamp
  resolution_summary?: string
}
```

---

### 3.4 get_consensus_history

Get the consensus history for a belief.

```typescript
function get_consensus_history(
  belief_id: UUID,
  options?: HistoryOptions
): Promise<ConsensusHistory>

interface HistoryOptions {
  include_corroborations?: boolean
  include_challenges?: boolean
  include_revisions?: boolean
  since?: Timestamp
}

interface ConsensusHistory {
  belief_id: UUID
  
  timeline: HistoryEvent[]
  
  statistics: {
    total_corroborations: uint64
    total_challenges: uint64
    total_revisions: uint64
    time_at_each_layer: Map<Layer, Duration>
  }
}

interface HistoryEvent {
  timestamp: Timestamp
  event_type: 'elevation' | 'demotion' | 'corroboration' | 'challenge' | 'revision' | 'verification'
  
  details: any                      // Event-specific
  triggered_by: DID | 'system'
}
```

---

## 4. Subscription Operations

### 4.1 subscribe_consensus_changes

Subscribe to consensus changes for specific beliefs or topics.

```typescript
function subscribe_consensus_changes(
  subscription: ConsensusSubscription
): AsyncIterator<ConsensusEvent>

interface ConsensusSubscription {
  // What to subscribe to (at least one required)
  belief_ids?: UUID[]               // Specific beliefs
  domains?: string[]                // All beliefs in domain
  min_layer?: Layer                 // Beliefs at or above layer
  
  // What events to receive
  event_types?: ConsensusEventType[]
}

enum ConsensusEventType {
  LAYER_CHANGED = 'layer_changed'
  CHALLENGE_OPENED = 'challenge_opened'
  CHALLENGE_RESOLVED = 'challenge_resolved'
  CORROBORATION_ADDED = 'corroboration_added'
  CONFIDENCE_CHANGED = 'confidence_changed'
  BELIEF_REVISED = 'belief_revised'
  BELIEF_SUPERSEDED = 'belief_superseded'
}

interface ConsensusEvent {
  event_id: UUID
  event_type: ConsensusEventType
  belief_id: UUID
  timestamp: Timestamp
  
  payload: any                      // Event-specific data
}
```

**WebSocket Example:**
```javascript
const ws = new WebSocket('wss://node.valence.network/consensus/subscribe');

ws.send(JSON.stringify({
  action: 'subscribe',
  subscription: {
    domains: ['marine_biology'],
    min_layer: 3,
    event_types: ['layer_changed', 'challenge_opened']
  }
}));

ws.onmessage = (event) => {
  const consensusEvent = JSON.parse(event.data);
  console.log(`Event: ${consensusEvent.event_type} for belief ${consensusEvent.belief_id}`);
};
```

---

## 5. Administrative Operations

### 5.1 request_elevation_review

Request manual review for elevation (when automated elevation is blocked).

```typescript
function request_elevation_review(
  belief_id: UUID,
  request: ElevationReviewRequest
): Promise<ReviewRequestResult>

interface ElevationReviewRequest {
  target_layer: Layer
  justification: string
  
  // Why automated elevation isn't working
  blockers_explanation: string
  
  // Supporting evidence
  additional_evidence?: Evidence[]
}

interface ReviewRequestResult {
  request_id: UUID
  status: 'queued' | 'rejected'
  
  estimated_review_time?: Duration
  rejection_reason?: string
  
  reviewer_assigned?: DID
}
```

---

### 5.2 report_independence_violation

Report suspected independence violation in corroborations.

```typescript
function report_independence_violation(
  report: IndependenceViolationReport
): Promise<ReportResult>

interface IndependenceViolationReport {
  // The corroboration being challenged
  corroboration_id: UUID
  
  // Evidence of dependence
  shared_source_evidence: Evidence[]
  coordination_evidence?: Evidence[]
  
  explanation: string
}

interface ReportResult {
  report_id: UUID
  status: 'accepted' | 'rejected'
  
  investigation_opened: boolean
  
  // If accepted
  corroboration_suspended?: boolean
  belief_flagged?: boolean
}
```

---

## 6. Batch Operations

### 6.1 batch_check_consensus

Check consensus status for multiple beliefs efficiently.

```typescript
function batch_check_consensus(
  belief_ids: UUID[],
  options?: CheckConsensusOptions
): Promise<Map<UUID, ConsensusStatus>>
```

**Limits:**
- Maximum 100 belief IDs per request

---

### 6.2 batch_get_communal_knowledge

Query multiple topics in a single request.

```typescript
function batch_get_communal_knowledge(
  queries: CommunalQuery[]
): Promise<Map<string, CommunalKnowledgeResponse>>
```

**Limits:**
- Maximum 10 queries per request
- Total result limit of 200 beliefs

---

## 7. Error Codes

| Code | Name | Description |
|------|------|-------------|
| `1001` | `BELIEF_NOT_FOUND` | Specified belief doesn't exist |
| `1002` | `ACCESS_DENIED` | Caller lacks permission |
| `1003` | `INVALID_LAYER` | Invalid layer specified |
| `2001` | `INSUFFICIENT_INDEPENDENCE` | Corroboration not independent enough |
| `2002` | `SELF_CORROBORATION` | Cannot corroborate own belief |
| `2003` | `DUPLICATE_CORROBORATION` | Evidence already recorded |
| `2004` | `BELIEF_LOCKED` | Belief locked due to active challenge |
| `3001` | `INSUFFICIENT_STAKE` | Caller lacks required reputation |
| `3002` | `ALREADY_CHALLENGED` | Active challenge exists |
| `3003` | `RATE_LIMITED` | Too many operations |
| `3004` | `INVALID_EVIDENCE` | Evidence doesn't meet requirements |
| `4001` | `INVALID_QUERY` | Query parameters invalid |
| `4002` | `DOMAIN_NOT_FOUND` | Specified domain doesn't exist |
| `5001` | `SUBSCRIPTION_LIMIT` | Too many active subscriptions |
| `5002` | `INVALID_SUBSCRIPTION` | Subscription parameters invalid |

---

## 8. Rate Limits

| Operation | Limit | Window |
|-----------|-------|--------|
| `check_consensus` | 100 | per minute |
| `get_communal_knowledge` | 50 | per minute |
| `submit_corroboration` | 20 | per hour |
| `challenge_consensus` | 5 | per day |
| `subscribe_consensus_changes` | 10 | concurrent |
| `batch_*` | 10 | per minute |

---

## 9. Authentication

All write operations require authentication via:

1. **Bearer token**: `Authorization: Bearer <jwt>`
2. **Signature**: Request body signed with agent's Ed25519 key

```typescript
interface AuthenticatedRequest {
  // Standard request fields...
  
  auth: {
    agent_id: DID
    timestamp: Timestamp
    signature: bytes              // Sign(private_key, sha256(request_body + timestamp))
  }
}
```

---

*"The API to collective truth."*
