# Valence API Specification

*REST, GraphQL, and WebSocket interfaces for the distributed epistemic network.*

---

## Overview

Valence exposes three complementary API surfaces:

| Protocol | Use Case | Characteristics |
|----------|----------|-----------------|
| **REST** | Simple CRUD, mobile apps, integrations | Stateless, cacheable, widely supported |
| **GraphQL** | Complex queries, flexible data fetching | Single endpoint, client-specified fields |
| **WebSocket** | Real-time updates, subscriptions | Persistent connection, bidirectional |

All APIs share:
- Common authentication
- Consistent error formats
- Unified rate limiting
- Same underlying data model

---

## 1. REST API

### 1.1 Base URL & Versioning

```
https://api.valence.network/v1/
```

**Version Strategy:**
- URL path versioning (`/v1/`, `/v2/`)
- Major versions for breaking changes
- Minor versions via headers: `Valence-API-Version: 2024-01-15`
- Deprecation notices in response headers
- Minimum 12-month support after deprecation

**Example:**
```http
GET /v1/beliefs/01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f
Host: api.valence.network
Valence-API-Version: 2024-01-15
```

### 1.2 Authentication

Three authentication methods, progressively more powerful:

#### API Keys (Simple)

For server-to-server or personal use.

```http
Authorization: Bearer val_sk_1234567890abcdef
```

**Key Types:**
| Prefix | Scope | Use Case |
|--------|-------|----------|
| `val_pk_` | Public (read-only) | Public queries, embedding |
| `val_sk_` | Secret (read/write) | Server apps, personal tools |
| `val_fk_` | Federation | Federation node operations |

**Key Management Endpoints:**
```
POST   /v1/auth/keys              # Create new API key
GET    /v1/auth/keys              # List your keys
DELETE /v1/auth/keys/{key_id}     # Revoke key
POST   /v1/auth/keys/{key_id}/rotate  # Rotate key
```

#### DID Authentication (Decentralized)

For agents with Valence identities.

```http
Authorization: DID-Auth did:valence:z6Mk...
X-Valence-Signature: <base64-signature>
X-Valence-Timestamp: 1704067200
X-Valence-Nonce: <random-32-bytes-hex>
```

**Signature Construction:**
```
message = HTTP_METHOD + "\n" +
          PATH + "\n" +
          TIMESTAMP + "\n" +
          NONCE + "\n" +
          SHA256(BODY)

signature = Ed25519_Sign(message, private_key)
```

**Benefits:**
- No API key to manage
- Portable across Valence nodes
- Reputation travels with identity

#### OAuth 2.0 (Third-Party Apps)

For applications acting on behalf of users.

```
Authorization endpoint: https://auth.valence.network/oauth/authorize
Token endpoint: https://auth.valence.network/oauth/token
```

**Scopes:**
| Scope | Description |
|-------|-------------|
| `beliefs:read` | Query beliefs |
| `beliefs:write` | Create/update beliefs |
| `trust:read` | Read trust graph |
| `trust:write` | Modify trust relationships |
| `federation:read` | Query federation data |
| `federation:write` | Share to federations |
| `identity:read` | Read identity info |
| `identity:write` | Update identity metadata |
| `verify` | Submit verifications |

### 1.3 Core Endpoints

#### Beliefs

```http
# Create belief
POST /v1/beliefs
Content-Type: application/json

{
  "content": "Python 3.12 introduced the new 'type' statement",
  "confidence": {
    "source_reliability": 0.95,
    "method_quality": 0.9
  },
  "domains": ["tech/python", "programming"],
  "visibility": "public",
  "derivation": {
    "type": "observation",
    "sources": [{
      "external_ref": "https://docs.python.org/3.12/whatsnew/"
    }]
  }
}

Response: 201 Created
{
  "id": "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  "version": 1,
  "content": "...",
  "confidence": { ... },
  "confidence_overall": 0.72,
  "holder_id": "did:valence:z6Mk...",
  "created_at": "2024-01-15T10:30:00Z",
  ...
}
```

```http
# Get belief by ID
GET /v1/beliefs/{belief_id}
GET /v1/beliefs/{belief_id}?include=derivation_chain,verifications

# Update belief (creates new version)
PATCH /v1/beliefs/{belief_id}
{
  "content": "Updated content",
  "confidence": { "source_reliability": 0.98 }
}

# Delete belief (soft delete, creates tombstone)
DELETE /v1/beliefs/{belief_id}

# Get belief version history
GET /v1/beliefs/{belief_id}/versions

# Get derivation chain
GET /v1/beliefs/{belief_id}/derivation
```

#### Semantic Search

```http
# Query beliefs
POST /v1/beliefs/query
{
  "semantic": "machine learning in healthcare",
  "filters": {
    "min_confidence": 0.6,
    "domains_include": ["medical", "tech/ai"],
    "valid_at": "2024-01-15T00:00:00Z"
  },
  "options": {
    "limit": 20,
    "include_explanation": true,
    "scope": {
      "local": true,
      "federated": true
    }
  }
}

Response: 200 OK
{
  "beliefs": [
    {
      "belief": { ... },
      "score": {
        "final": 0.87,
        "components": {
          "semantic_similarity": 0.92,
          "confidence_score": 0.78,
          "trust_score": 0.85,
          "recency_score": 0.95
        }
      },
      "explanation": { ... }
    }
  ],
  "total_count": 234,
  "next_cursor": "eyJvZmZzZXQiOjIwfQ==",
  "query_time_ms": 47
}
```

```http
# Find similar beliefs
GET /v1/beliefs/{belief_id}/similar?limit=10&min_similarity=0.8

# Find contradictions
GET /v1/beliefs/{belief_id}/contradictions

# Get facets for filtering
POST /v1/beliefs/facets
{
  "semantic": "artificial intelligence",
  "facet_fields": ["domains", "confidence_bucket", "derivation_type"]
}
```

#### Trust Graph

```http
# Set trust relationship
PUT /v1/trust/{target_did}
{
  "level": 0.75,
  "domains": {
    "code:rust": 0.9,
    "finance": 0.3
  },
  "basis": [{
    "type": "direct_interaction",
    "confidence": 0.8
  }],
  "notes": "Met at RustConf 2025"
}

# Get trust for agent
GET /v1/trust/{target_did}
GET /v1/trust/{target_did}?domain=code:rust

# Get trusted agents
GET /v1/trust?min_level=0.7&domain=code&include_transitive=true

# Remove trust
DELETE /v1/trust/{target_did}

# Compute transitive trust
POST /v1/trust/compute
{
  "target": "did:valence:z6Mk...",
  "domain": "ai/safety",
  "max_hops": 3,
  "explain": true
}

# Get trust graph statistics
GET /v1/trust/stats
```

#### Verifications

```http
# Submit verification
POST /v1/verifications
{
  "belief_id": "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  "result": "confirmed",
  "evidence": [{
    "type": "external",
    "external_source": {
      "url": "https://authoritative-source.com/fact",
      "archive_hash": "a3f2b8c1..."
    },
    "relevance": 0.95,
    "contribution": "supports"
  }],
  "stake": 0.05,
  "reasoning": "Verified against primary source"
}

# Get verifications for belief
GET /v1/beliefs/{belief_id}/verifications
GET /v1/beliefs/{belief_id}/verifications?status=accepted&order_by=stake

# Dispute verification
POST /v1/verifications/{verification_id}/dispute
{
  "counter_evidence": [...],
  "dispute_stake": 0.075,
  "dispute_type": "new_evidence",
  "reasoning": "New evidence contradicts this"
}

# Get verification history (for verifier)
GET /v1/verifications?verifier=me
```

#### Federations

```http
# Create federation
POST /v1/federations
{
  "name": "AI Safety Researchers",
  "description": "Sharing research insights",
  "domains": ["ai/safety", "ai/alignment"],
  "visibility": "discoverable",
  "governance": {
    "type": "council",
    "join_policy": { "type": "approval_required" }
  }
}

# List federations
GET /v1/federations?member_of=true
GET /v1/federations?visibility=discoverable&domains=ai

# Get federation details
GET /v1/federations/{federation_id}
GET /v1/federations/{federation_id}?include=stats,members

# Join federation
POST /v1/federations/{federation_id}/join
{
  "application_message": "I research interpretability at Anthropic"
}

# Leave federation
POST /v1/federations/{federation_id}/leave

# Share belief to federation
POST /v1/federations/{federation_id}/share
{
  "belief_id": "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  "anonymous": true
}

# Query federation knowledge
POST /v1/federations/{federation_id}/query
{
  "semantic": "interpretability techniques",
  "min_agreement": 0.7
}

# Federation governance
POST /v1/federations/{federation_id}/proposals
GET /v1/federations/{federation_id}/proposals?status=active
POST /v1/federations/{federation_id}/proposals/{id}/vote
```

#### Identity

```http
# Get own identity
GET /v1/identity

# Update identity metadata
PATCH /v1/identity
{
  "metadata": { "display_name": "New Name" }
}

# Resolve DID
GET /v1/identities/{did}

# Get reputation
GET /v1/identities/{did}/reputation
GET /v1/identities/{did}/reputation?domain=code

# Rotate keys
POST /v1/identity/rotate-keys
{
  "reason": "scheduled rotation"
}
```

### 1.4 Pagination

**Cursor-based** (preferred):
```http
GET /v1/beliefs?limit=20
# Response includes: "next_cursor": "eyJvZmZzZXQiOjIwfQ=="

GET /v1/beliefs?limit=20&cursor=eyJvZmZzZXQiOjIwfQ==
```

**Offset-based** (for random access):
```http
GET /v1/beliefs?limit=20&offset=40
```

**Response format:**
```json
{
  "data": [...],
  "pagination": {
    "total_count": 1234,
    "limit": 20,
    "offset": 40,
    "next_cursor": "...",
    "has_more": true
  }
}
```

### 1.5 Error Responses

**Standard format:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Content exceeds maximum length",
    "details": {
      "field": "content",
      "actual_length": 70000,
      "max_length": 65536
    },
    "request_id": "req_1234567890",
    "documentation_url": "https://docs.valence.network/errors/validation"
  }
}
```

**HTTP Status Codes:**
| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 204 | No Content (successful delete) |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized (missing/invalid auth) |
| 403 | Forbidden (insufficient permissions) |
| 404 | Not Found |
| 409 | Conflict (duplicate, already exists) |
| 422 | Unprocessable Entity (semantic error) |
| 429 | Too Many Requests (rate limited) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (maintenance) |

**Error Codes:**
```
VALIDATION_ERROR, AUTHENTICATION_REQUIRED, INVALID_SIGNATURE,
PERMISSION_DENIED, NOT_FOUND, DUPLICATE, CONFLICT, RATE_LIMITED,
INSUFFICIENT_REPUTATION, STAKE_INSUFFICIENT, INTERNAL_ERROR
```

---

## 2. GraphQL API

### 2.1 Endpoint

```
POST https://api.valence.network/graphql
```

Single endpoint for all operations. Use query complexity analysis to prevent abuse.

### 2.2 Schema

```graphql
# Core Types

scalar UUID
scalar DID
scalar Timestamp
scalar JSON

type Belief {
  id: UUID!
  version: Int!
  content: String!
  contentHash: String!
  
  confidence: ConfidenceVector!
  confidenceOverall: Float!
  
  validFrom: Timestamp!
  validUntil: Timestamp
  
  derivation: Derivation
  derivationChain(maxDepth: Int = 5): [DerivationLink!]
  
  domains: [String!]!
  visibility: Visibility!
  
  holder: AgentIdentity!
  holderId: DID!
  
  createdAt: Timestamp!
  supersedes: Belief
  supersededBy: Belief
  
  # Relationships
  verifications(
    status: [VerificationStatus!]
    limit: Int = 20
  ): VerificationConnection!
  
  similar(
    limit: Int = 10
    minSimilarity: Float = 0.7
  ): [RankedBelief!]!
  
  contradictions: [ContradictingBelief!]!
}

type ConfidenceVector {
  sourceReliability: Float
  methodQuality: Float
  internalConsistency: Float
  temporalFreshness: Float
  corroboration: Float
  domainApplicability: Float
  overall: Float!
}

type RankedBelief {
  belief: Belief!
  score: RankingScore!
  explanation: RankingExplanation
}

type RankingScore {
  final: Float!
  components: ScoreComponents!
}

type ScoreComponents {
  semanticSimilarity: Float!
  confidenceScore: Float!
  trustScore: Float!
  recencyScore: Float!
  diversityPenalty: Float!
}

type RankingExplanation {
  summary: String!
  rank: Int!
  percentile: Float!
  improvementHints: [ImprovementHint!]!
}

type AgentIdentity {
  id: DID!
  displayName: String
  reputation: ReputationScore!
  createdAt: Timestamp!
  
  # Only if trusted/permitted
  trustLevel(domain: String): Float
  beliefs(
    limit: Int = 20
    domains: [String!]
  ): BeliefConnection!
}

type ReputationScore {
  overall: Float!
  byDomain(domain: String!): Float
  verificationCount: Int!
  discrepancyFinds: Int!
}

type TrustEdge {
  from: DID!
  to: AgentIdentity!
  level: Float!
  domains: JSON
  basis: [TrustBasis!]
  notes: String
  createdAt: Timestamp!
  updatedAt: Timestamp!
  lastUsed: Timestamp!
}

type Verification {
  id: UUID!
  belief: Belief!
  verifier: AgentIdentity!
  result: VerificationResult!
  evidence: [Evidence!]!
  stake: Float!
  reasoning: String
  status: VerificationStatus!
  createdAt: Timestamp!
  dispute: Dispute
}

type Federation {
  id: UUID!
  name: String!
  description: String
  domains: [String!]!
  visibility: FederationVisibility!
  memberCount: Int!
  
  # Only if member
  members(
    role: [MemberRole!]
    limit: Int = 50
  ): MemberConnection
  
  aggregatedBeliefs(
    semantic: String
    minAgreement: Float
    limit: Int = 20
  ): [AggregatedBelief!]!
  
  stats: FederationStats
}

type AggregatedBelief {
  topicHash: String!
  contentSummary: String!
  aggregateConfidence: ConfidenceVector!
  contributorCount: Int!
  agreementScore: Float!
  dominantDerivationTypes: [DerivationType!]!
  firstContributed: Timestamp!
  lastUpdated: Timestamp!
}

# Enums

enum Visibility {
  PRIVATE
  FEDERATED
  PUBLIC
}

enum VerificationResult {
  CONFIRMED
  CONTRADICTED
  UNCERTAIN
  PARTIAL
}

enum VerificationStatus {
  PENDING
  ACCEPTED
  DISPUTED
  OVERTURNED
}

enum FederationVisibility {
  HIDDEN
  UNLISTED
  DISCOVERABLE
  PUBLIC
}

enum DerivationType {
  OBSERVATION
  INFERENCE
  HEARSAY
  SYNTHESIS
  CORRECTION
  PREDICTION
}

# Connections (Relay-style pagination)

type BeliefConnection {
  edges: [BeliefEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type BeliefEdge {
  node: Belief!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# Inputs

input BeliefInput {
  content: String!
  confidence: ConfidenceInput
  domains: [String!]
  visibility: Visibility
  validFrom: Timestamp
  validUntil: Timestamp
  derivation: DerivationInput
}

input ConfidenceInput {
  sourceReliability: Float
  methodQuality: Float
  internalConsistency: Float
  temporalFreshness: Float
  corroboration: Float
  domainApplicability: Float
}

input QueryFiltersInput {
  minConfidence: ConfidenceInput
  validAt: Timestamp
  createdAfter: Timestamp
  createdBefore: Timestamp
  domainsInclude: [String!]
  domainsExclude: [String!]
  holderIds: [DID!]
  minHolderReputation: Float
  derivationTypes: [DerivationType!]
}

input QueryOptionsInput {
  limit: Int
  cursor: String
  includeDerivation: Boolean
  includeExplanation: Boolean
  scope: QueryScopeInput
  diversity: DiversityInput
}

input QueryScopeInput {
  local: Boolean
  federated: Boolean
  network: Boolean
}

input DiversityInput {
  enabled: Boolean!
  minSemanticDistance: Float
  maxPerHolder: Int
  maxPerDomain: Int
}

input TrustInput {
  level: Float!
  domains: JSON
  basis: [TrustBasisInput!]
  notes: String
}

# Queries

type Query {
  # Beliefs
  belief(id: UUID!): Belief
  beliefs(
    ids: [UUID!]
  ): [Belief!]!
  
  queryBeliefs(
    semantic: String!
    filters: QueryFiltersInput
    options: QueryOptionsInput
  ): BeliefConnection!
  
  # Trust
  trust(targetDid: DID!, domain: String): TrustEdge
  trustedAgents(
    minLevel: Float!
    domain: String
    includeTransitive: Boolean
    limit: Int
  ): [TrustEdge!]!
  
  transitivesTrust(
    targetDid: DID!
    domain: String
    maxHops: Int
    explain: Boolean
  ): TransitiveTrustResult!
  
  # Identity
  me: AgentIdentity!
  identity(did: DID!): AgentIdentity
  
  # Federations
  federation(id: UUID!): Federation
  federations(
    memberOf: Boolean
    visibility: [FederationVisibility!]
    domains: [String!]
    limit: Int
  ): [Federation!]!
  
  # Verifications
  verification(id: UUID!): Verification
  pendingVerifications: [Verification!]!
}

# Mutations

type Mutation {
  # Beliefs
  createBelief(input: BeliefInput!): Belief!
  updateBelief(id: UUID!, input: BeliefInput!): Belief!
  supersedeBelief(id: UUID!, newBelief: BeliefInput!): Belief!
  deleteBelief(id: UUID!, reason: String): Boolean!
  
  # Trust
  setTrust(targetDid: DID!, input: TrustInput!): TrustEdge!
  removeTrust(targetDid: DID!): Boolean!
  
  # Verifications
  submitVerification(
    beliefId: UUID!
    result: VerificationResult!
    evidence: [EvidenceInput!]!
    stake: Float!
    reasoning: String
  ): Verification!
  
  disputeVerification(
    verificationId: UUID!
    counterEvidence: [EvidenceInput!]!
    disputeStake: Float!
    disputeType: DisputeType!
    reasoning: String!
  ): Dispute!
  
  # Federations
  createFederation(input: FederationInput!): Federation!
  joinFederation(federationId: UUID!, input: JoinInput): JoinResult!
  leaveFederation(federationId: UUID!): Boolean!
  shareToFederation(
    federationId: UUID!
    beliefId: UUID!
    anonymous: Boolean
  ): SharedBelief!
  
  # Identity
  updateIdentity(metadata: JSON!): AgentIdentity!
  rotateKeys(reason: String): KeyRotationResult!
}

# Subscriptions

type Subscription {
  # Belief updates
  beliefCreated(
    domains: [String!]
    minConfidence: Float
  ): Belief!
  
  beliefUpdated(beliefIds: [UUID!]): Belief!
  
  # Query subscription (live results)
  queryResults(
    semantic: String!
    filters: QueryFiltersInput
  ): QuerySubscriptionEvent!
  
  # Trust changes
  trustChanged(watchDids: [DID!]): TrustChangeEvent!
  
  # Verifications
  verificationSubmitted(beliefIds: [UUID!]): Verification!
  verificationDisputed(verificationIds: [UUID!]): Dispute!
  
  # Federation events
  federationEvent(federationId: UUID!): FederationEvent!
}

type QuerySubscriptionEvent {
  type: QueryEventType!
  belief: RankedBelief!
  previousRank: Int
  newRank: Int
}

enum QueryEventType {
  ADDED
  UPDATED
  REMOVED
  RERANKED
}
```

### 2.3 Example Queries

**Complex belief query with nested data:**
```graphql
query SearchBeliefs($query: String!, $domain: String) {
  queryBeliefs(
    semantic: $query
    filters: {
      domainsInclude: [$domain]
      minConfidence: { overall: 0.6 }
    }
    options: {
      limit: 10
      includeExplanation: true
      scope: { local: true, federated: true }
    }
  ) {
    totalCount
    edges {
      node {
        id
        content
        confidence {
          overall
          sourceReliability
          corroboration
        }
        holder {
          displayName
          reputation {
            overall
            byDomain(domain: $domain)
          }
        }
        verifications(status: [ACCEPTED], limit: 3) {
          edges {
            node {
              result
              stake
              verifier {
                displayName
              }
            }
          }
        }
      }
      cursor
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
```

**Federation aggregate query:**
```graphql
query FederationKnowledge($fedId: UUID!) {
  federation(id: $fedId) {
    name
    memberCount
    aggregatedBeliefs(
      semantic: "best practices"
      minAgreement: 0.7
      limit: 20
    ) {
      contentSummary
      aggregateConfidence {
        overall
      }
      contributorCount
      agreementScore
      lastUpdated
    }
  }
}
```

### 2.4 Query Complexity

Prevent abuse with complexity scoring:

```
complexity = base_cost + (field_cost * nesting_depth) + (list_cost * limit)
```

| Field Type | Cost |
|------------|------|
| Scalar | 0 |
| Object | 1 |
| List | 2 × limit |
| Connection | 3 × limit |
| derivationChain | 5 × maxDepth |

**Max complexity per query:** 1000 (configurable per auth level)

---

## 3. WebSocket API

### 3.1 Connection

```
wss://api.valence.network/ws
```

**Connection flow:**
1. Open WebSocket connection
2. Send authentication message
3. Server responds with session info
4. Subscribe to topics
5. Receive real-time events

### 3.2 Protocol

**Message format:**
```json
{
  "type": "subscribe" | "unsubscribe" | "message" | "error" | "ack",
  "id": "msg_123",           // Client-generated, for correlation
  "topic": "beliefs",        // Topic name
  "payload": { ... }         // Type-specific payload
}
```

**Authentication:**
```json
// Client sends
{
  "type": "auth",
  "id": "auth_1",
  "payload": {
    "method": "api_key",
    "token": "val_sk_..."
  }
}

// Server responds
{
  "type": "ack",
  "id": "auth_1",
  "payload": {
    "session_id": "sess_abc123",
    "identity": "did:valence:z6Mk...",
    "expires_at": "2024-01-15T11:30:00Z"
  }
}
```

### 3.3 Subscription Topics

#### Belief Query Subscription

Subscribe to live query results:

```json
// Subscribe
{
  "type": "subscribe",
  "id": "sub_1",
  "topic": "query",
  "payload": {
    "semantic": "AI safety research",
    "filters": {
      "min_confidence": 0.6,
      "domains_include": ["ai/safety"]
    },
    "options": {
      "notify_on_connect": true,
      "include_removals": true,
      "debounce_ms": 1000
    }
  }
}

// Events
{
  "type": "message",
  "topic": "query",
  "subscription_id": "sub_1",
  "payload": {
    "event": "added",
    "belief": { ... },
    "score": { "final": 0.87, ... }
  }
}
```

#### Belief Updates

Watch specific beliefs:

```json
{
  "type": "subscribe",
  "id": "sub_2",
  "topic": "beliefs",
  "payload": {
    "belief_ids": ["01941d3a-...", "01941d3b-..."]
  }
}

// Events: updated, superseded, verified, deleted
```

#### Trust Graph Changes

```json
{
  "type": "subscribe",
  "id": "sub_3",
  "topic": "trust",
  "payload": {
    "watch_dids": ["did:valence:z6Mk..."],
    "include_transitive": true
  }
}

// Events: trust_set, trust_removed, transitive_changed
```

#### Verification Events

```json
{
  "type": "subscribe",
  "id": "sub_4",
  "topic": "verifications",
  "payload": {
    "belief_ids": ["..."],         // OR
    "for_holder": true,            // Beliefs I hold
    "for_verifier": true           // My verifications
  }
}

// Events: submitted, accepted, disputed, resolved
```

#### Federation Events

```json
{
  "type": "subscribe",
  "id": "sub_5",
  "topic": "federation",
  "payload": {
    "federation_id": "..."
  }
}

// Events: member_joined, member_left, belief_shared, aggregate_updated, proposal_created, vote_cast
```

### 3.4 Presence & Heartbeat

```json
// Server sends every 30 seconds
{ "type": "ping" }

// Client must respond
{ "type": "pong" }

// Miss 3 pongs = disconnect
```

### 3.5 Error Handling

```json
{
  "type": "error",
  "id": "sub_1",                   // Correlates to request
  "payload": {
    "code": "INVALID_SUBSCRIPTION",
    "message": "Invalid filter syntax",
    "recoverable": true
  }
}
```

**Error codes:**
| Code | Meaning | Recoverable |
|------|---------|-------------|
| `AUTH_FAILED` | Authentication failed | Yes (retry) |
| `AUTH_EXPIRED` | Session expired | Yes (reauth) |
| `INVALID_MESSAGE` | Malformed message | Yes |
| `INVALID_SUBSCRIPTION` | Bad subscription params | Yes |
| `SUBSCRIPTION_LIMIT` | Too many subscriptions | Yes |
| `RATE_LIMITED` | Too many messages | Yes |
| `INTERNAL_ERROR` | Server error | Sometimes |

---

## 4. Rate Limiting & Quotas

### 4.1 Rate Limits

**Per-endpoint limits** (requests per minute):

| Endpoint Category | Public Key | Secret Key | DID Auth |
|-------------------|------------|------------|----------|
| Read (GET) | 60 | 600 | 1000 |
| Write (POST/PUT/PATCH) | - | 100 | 200 |
| Query (semantic search) | 20 | 200 | 500 |
| Delete | - | 20 | 50 |
| WebSocket subscriptions | 5 | 50 | 100 |

**Headers:**
```http
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 542
X-RateLimit-Reset: 1704067260
Retry-After: 30          # Only on 429
```

### 4.2 Quotas

**Resource quotas** (monthly):

| Resource | Free | Pro | Enterprise |
|----------|------|-----|------------|
| Beliefs stored | 1,000 | 100,000 | Unlimited |
| Queries | 10,000 | 1,000,000 | Unlimited |
| Federation memberships | 3 | 20 | Unlimited |
| API keys | 2 | 10 | 100 |
| WebSocket connections | 1 | 10 | 100 |

**Quota headers:**
```http
X-Quota-Beliefs-Limit: 100000
X-Quota-Beliefs-Used: 42350
X-Quota-Beliefs-Reset: 2024-02-01
```

### 4.3 Burst Handling

- Short bursts allowed (up to 2x limit for 10 seconds)
- Sustained over-limit triggers 429
- Backoff multiplier for repeated violations

### 4.4 Priority Queue

During high load:
1. Authenticated requests prioritized over anonymous
2. DID Auth prioritized over API keys
3. Higher reputation = higher priority
4. Federation queries from members prioritized

---

## 5. Common Patterns

### 5.1 Idempotency

Write operations accept `Idempotency-Key` header:

```http
POST /v1/beliefs
Idempotency-Key: idem_abc123def456

# Replays within 24h return cached response
```

### 5.2 Conditional Requests

```http
# ETag for caching
GET /v1/beliefs/123
ETag: "a1b2c3d4"

GET /v1/beliefs/123
If-None-Match: "a1b2c3d4"
# Returns 304 Not Modified if unchanged

# Optimistic locking for updates
PATCH /v1/beliefs/123
If-Match: "a1b2c3d4"
# Returns 412 Precondition Failed if changed
```

### 5.3 Batch Operations

```http
POST /v1/batch
{
  "operations": [
    { "method": "POST", "path": "/v1/beliefs", "body": {...} },
    { "method": "PUT", "path": "/v1/trust/did:...", "body": {...} }
  ]
}

# Returns array of responses
```

### 5.4 Webhooks

Register webhooks for server-to-server notifications:

```http
POST /v1/webhooks
{
  "url": "https://my-app.com/valence-webhook",
  "events": ["belief.verified", "trust.changed"],
  "secret": "whsec_..."
}
```

Webhook payload:
```json
{
  "id": "evt_123",
  "type": "belief.verified",
  "created_at": "2024-01-15T10:30:00Z",
  "data": { ... }
}

// Verify with: HMAC-SHA256(payload, secret)
```

---

## 6. SDK Considerations

See SDK.md for detailed client library specifications. Key points:

- SDKs should handle auth, pagination, rate limiting automatically
- Provide both sync and async interfaces
- Offline-first with sync queue
- Type-safe with generated types from OpenAPI/GraphQL schema

---

*"Simple for simple things, powerful for complex ones."*
