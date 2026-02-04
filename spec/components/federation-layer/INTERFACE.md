# Federation Layer Interface

*API specification for Valence federation operations.*

---

## Overview

This document defines the programmatic interface for federation operations. All functions are designed to be:
- **Idempotent** where possible
- **Privacy-preserving** by default
- **Cryptographically verified**
- **Auditable**

---

## Core Operations

### create_federation

Create a new federation with the caller as founder.

```typescript
function create_federation(config: CreateFederationConfig): Promise<CreateFederationResult>

interface CreateFederationConfig {
  // Required
  name: string                       // Human-readable name (3-100 chars)
  
  // Optional (sensible defaults)
  description?: string               // Purpose and scope (max 2000 chars)
  domains?: string[]                 // Focus areas (max 10)
  visibility?: FederationVisibility  // Default: 'unlisted'
  
  // Governance (defaults to founder-controlled)
  governance?: {
    type?: GovernanceType            // Default: 'autocratic'
    join_policy?: JoinPolicy         // Default: 'invite_only'
    decision_policy?: DecisionPolicy // Default: 'founder_decides'
    moderation_policy?: ModerationPolicy // Default: 'none'
  }
  
  // Privacy (defaults to maximum privacy)
  privacy?: {
    hide_member_list?: boolean       // Default: true
    hide_contribution_source?: boolean // Default: true
    aggregation_noise?: float        // Default: 0.1 (epsilon)
    enable_plausible_deniability?: boolean // Default: false
  }
  
  // Limits
  max_members?: uint64               // Default: 0 (unlimited)
  min_members_for_aggregation?: uint8 // Default: 5 (k-anonymity)
  
  // Key management
  key_rotation_period?: duration     // Default: '30d'
}

interface CreateFederationResult {
  success: boolean
  federation?: Federation
  membership?: Membership            // Founder's membership
  error?: FederationError
}
```

**Example:**
```typescript
const result = await create_federation({
  name: "AI Safety Researchers",
  description: "Sharing research insights on AI alignment and safety",
  domains: ["ai/safety", "ai/alignment", "research"],
  visibility: 'discoverable',
  governance: {
    type: 'council',
    join_policy: { type: 'approval_required', required_vouches: 2 },
    decision_policy: { type: 'majority_vote', quorum: 0.5, threshold: 0.6 }
  },
  min_members_for_aggregation: 3
});
```

**Errors:**
| Code | Meaning |
|------|---------|
| `NAME_TAKEN` | Federation name already exists |
| `INVALID_CONFIG` | Configuration validation failed |
| `INSUFFICIENT_REPUTATION` | Creator lacks minimum reputation |
| `RATE_LIMITED` | Too many recent federation creations |

---

### join_federation

Request to join an existing federation.

```typescript
function join_federation(
  federation_id: UUID,
  options?: JoinOptions
): Promise<JoinFederationResult>

interface JoinOptions {
  invite_code?: string               // Required for invite_only federations
  invite_token?: InviteToken         // Cryptographic invite from member
  application_message?: string       // For approval_required federations
  vouchers?: DID[]                   // Members vouching for applicant
  token_proof?: TokenProof           // For token_gated federations
}

interface JoinFederationResult {
  success: boolean
  membership?: Membership
  status: 'joined' | 'pending_approval' | 'rejected'
  pending_info?: {
    application_id: UUID
    required_approvals: uint8
    current_approvals: uint8
    expires_at: timestamp
  }
  error?: FederationError
}
```

**Join Flow by Policy:**

| Policy | Invite Required | Approval Required | Immediate Join |
|--------|-----------------|-------------------|----------------|
| `open` | ❌ | ❌ | ✅ |
| `approval_required` | ❌ | ✅ | ❌ |
| `invite_only` | ✅ | ❌ | ✅ |
| `invite_only` + approval | ✅ | ✅ | ❌ |
| `token_gated` | Token proof | Optional | Depends |

**Example:**
```typescript
// Open federation
await join_federation(federationId);

// Invite-only
await join_federation(federationId, {
  invite_token: inviteFromAlice
});

// Approval required
await join_federation(federationId, {
  application_message: "I research interpretability at Anthropic",
  vouchers: [aliceDid, bobDid]
});
```

**Errors:**
| Code | Meaning |
|------|---------|
| `FEDERATION_NOT_FOUND` | No federation with this ID |
| `ALREADY_MEMBER` | Caller is already a member |
| `INVITE_REQUIRED` | Federation requires invite |
| `INVITE_INVALID` | Invite code/token invalid or expired |
| `INSUFFICIENT_VOUCHES` | Not enough members vouching |
| `MEMBERSHIP_CLOSED` | Federation not accepting members |
| `TOKEN_REQUIRED` | Token gate not satisfied |

---

### leave_federation

Voluntarily leave a federation.

```typescript
function leave_federation(
  federation_id: UUID,
  options?: LeaveOptions
): Promise<LeaveFederationResult>

interface LeaveOptions {
  export_contributions?: boolean     // Get copy of own contributions (default: false)
  reason?: string                    // Optional departure reason
  effective_at?: timestamp           // Delayed departure (default: immediate)
}

interface LeaveFederationResult {
  success: boolean
  exported_beliefs?: EncryptedExport // If requested
  key_rotation_triggered: boolean
  error?: FederationError
}
```

**Post-Departure:**
- Membership marked as `departed`
- Key rotation triggered (forward secrecy)
- Historical contributions remain (anonymized)
- Cannot re-join immediately (cooldown period)

**Example:**
```typescript
await leave_federation(federationId, {
  export_contributions: true,
  reason: "Moving to new research focus"
});
```

**Errors:**
| Code | Meaning |
|------|---------|
| `NOT_A_MEMBER` | Caller is not a member |
| `FOUNDER_CANNOT_LEAVE` | Founders must transfer or dissolve |
| `PENDING_OBLIGATIONS` | Outstanding governance votes |

---

### share_to_federation

Share a belief with a federation.

```typescript
function share_to_federation(
  belief_id: UUID,
  federation_id: UUID,
  options?: ShareOptions
): Promise<ShareResult>

interface ShareOptions {
  // Privacy controls
  anonymous?: boolean                // Don't link to identity (default: true)
  use_decoy?: boolean               // Add plausible deniability (default: false)
  
  // Contribution metadata
  domains?: string[]                 // Override belief domains
  context?: string                   // Additional context for aggregation
  
  // Temporal
  expires_at?: timestamp             // Auto-expire sharing
}

interface ShareResult {
  success: boolean
  shared_belief?: SharedBelief
  error?: FederationError
}

interface SharedBelief {
  id: UUID                           // Unique sharing ID (not belief ID)
  federation_id: UUID
  
  // The belief content (encrypted for federation)
  encrypted_content: EncryptedPayload
  
  // Metadata (may be partially visible)
  topic_hash: bytes                  // For aggregation clustering
  confidence_bucket: 'low' | 'medium' | 'high'  // Bucketed, not exact
  derivation_type: DerivationType    // How belief was formed
  domains: string[]
  
  // Temporal
  shared_at: timestamp
  expires_at?: timestamp
  
  // Contributor identity (hidden by default)
  contributor_commitment: bytes      // Cryptographic commitment, not identity
}
```

**Sharing Flow:**
```
1. Verify caller is member with SHARE_BELIEFS permission
2. Validate belief exists and caller holds it
3. Check belief meets federation requirements (min_confidence, max_age, etc.)
4. Encrypt belief content with federation group key
5. Generate topic hash for aggregation
6. Create contributor commitment (zero-knowledge proof of membership)
7. If decoy requested: also share decoy beliefs
8. Store SharedBelief
9. Trigger aggregation update
```

**Example:**
```typescript
// Standard anonymous share
await share_to_federation(myBeliefId, federationId);

// With plausible deniability
await share_to_federation(myBeliefId, federationId, {
  anonymous: true,
  use_decoy: true,
  context: "Based on direct experimental results"
});

// Time-limited share
await share_to_federation(myBeliefId, federationId, {
  expires_at: Date.now() + 30 * 24 * 60 * 60 * 1000  // 30 days
});
```

**Errors:**
| Code | Meaning |
|------|---------|
| `NOT_A_MEMBER` | Caller not a federation member |
| `PERMISSION_DENIED` | Lacks SHARE_BELIEFS permission |
| `BELIEF_NOT_FOUND` | Belief doesn't exist |
| `NOT_BELIEF_HOLDER` | Caller doesn't hold this belief |
| `BELOW_MIN_CONFIDENCE` | Belief confidence too low |
| `BELIEF_TOO_OLD` | Belief exceeds max_age |
| `DOMAIN_MISMATCH` | Belief domains don't match federation |
| `DUPLICATE_SHARE` | Already shared this belief |

---

### query_federation

Query beliefs shared within a federation.

```typescript
function query_federation(
  federation_id: UUID,
  query: FederationQuery
): Promise<QueryResult>

interface FederationQuery {
  // Content search
  semantic_query?: string            // Natural language query
  topic_hash?: bytes                 // Exact topic match
  
  // Filters
  domains?: string[]                 // Filter by domain
  min_confidence?: float             // Minimum aggregate confidence
  min_agreement?: float              // Minimum agreement score
  min_contributors?: uint8           // Minimum contributor count
  
  // Temporal
  since?: timestamp                  // Beliefs shared after this time
  until?: timestamp                  // Beliefs shared before this time
  
  // Result control
  limit?: uint16                     // Max results (default: 20, max: 100)
  offset?: uint16                    // Pagination offset
  order_by?: 'relevance' | 'confidence' | 'recency' | 'agreement'
  
  // Privacy options
  include_contested?: boolean        // Include low-agreement topics (default: false)
}

interface QueryResult {
  success: boolean
  aggregates: AggregatedBelief[]
  
  // Query metadata
  total_matches: uint64              // Total matching (may be noised)
  query_coverage: float              // How much of federation knowledge was searchable
  
  // Privacy info
  suppressed_count?: uint64          // Results hidden due to k-anonymity
  
  pagination: {
    offset: uint16
    limit: uint16
    has_more: boolean
  }
  
  error?: FederationError
}
```

**Query Semantics:**

Queries return **aggregated beliefs**, never individual contributions. The aggregation ensures:
- k-anonymity: No aggregate from fewer than k contributors
- Differential privacy: Counts are noised
- No linkability: Can't determine who contributed what

**Example:**
```typescript
// Semantic search
const results = await query_federation(federationId, {
  semantic_query: "interpretability techniques for large language models",
  domains: ["ai/interpretability"],
  min_confidence: 0.6,
  min_contributors: 3,
  order_by: 'relevance'
});

// Recent high-agreement knowledge
const consensus = await query_federation(federationId, {
  min_agreement: 0.8,
  since: Date.now() - 7 * 24 * 60 * 60 * 1000,  // Last week
  order_by: 'agreement'
});
```

**Errors:**
| Code | Meaning |
|------|---------|
| `NOT_A_MEMBER` | Caller not a federation member |
| `PERMISSION_DENIED` | Lacks QUERY_BELIEFS permission |
| `INVALID_QUERY` | Query validation failed |
| `FEDERATION_INACTIVE` | Federation dissolved or suspended |

---

### get_aggregated_beliefs

Get aggregated beliefs on a specific topic.

```typescript
function get_aggregated_beliefs(
  federation_id: UUID,
  topic: TopicSpecifier
): Promise<AggregatedBeliefsResult>

type TopicSpecifier = 
  | { type: 'semantic', query: string }
  | { type: 'hash', topic_hash: bytes }
  | { type: 'domain', domain: string }
  | { type: 'cluster_id', cluster: UUID }

interface AggregatedBeliefsResult {
  success: boolean
  
  // Primary aggregate
  aggregate?: AggregatedBelief
  
  // Related aggregates (semantically similar topics)
  related?: AggregatedBelief[]
  
  // If topic is contested, may have multiple perspectives
  perspectives?: {
    aggregate: AggregatedBelief
    stance: string                   // Brief characterization
    contributor_count: uint64        // Noised
  }[]
  
  // Temporal evolution
  history?: {
    timestamp: timestamp
    confidence: ConfidenceVector
    contributor_count: uint64
    agreement_score: float
  }[]
  
  error?: FederationError
}
```

**Aggregate Structure (from SPEC.md):**

```typescript
interface AggregatedBelief {
  federation_id: UUID
  topic_hash: bytes
  
  content_summary: string            // Aggregated content
  aggregate_confidence: ConfidenceVector
  
  contributor_count: uint64          // Noised if below k
  confidence_distribution: {
    low: uint64                      // < 0.4
    medium: uint64                   // 0.4 - 0.7  
    high: uint64                     // > 0.7
  }
  agreement_score: float
  
  first_contributed: timestamp
  last_updated: timestamp
  
  dominant_derivation_types: DerivationType[]
  external_ref_count: uint64
}
```

**Example:**
```typescript
// Get aggregate on specific topic
const result = await get_aggregated_beliefs(federationId, {
  type: 'semantic',
  query: "effectiveness of RLHF for alignment"
});

if (result.perspectives) {
  // Topic is contested - multiple viewpoints
  for (const p of result.perspectives) {
    console.log(`${p.stance}: ${p.aggregate.content_summary}`);
    console.log(`  Supporters: ~${p.contributor_count}`);
  }
}
```

**Errors:**
| Code | Meaning |
|------|---------|
| `TOPIC_NOT_FOUND` | No beliefs match this topic |
| `INSUFFICIENT_CONTRIBUTORS` | Topic exists but below k-threshold |
| `NOT_A_MEMBER` | Caller not a federation member |

---

## Membership Management

### invite_member

Generate an invitation for a new member.

```typescript
function invite_member(
  federation_id: UUID,
  options?: InviteOptions
): Promise<InviteResult>

interface InviteOptions {
  // Target (optional - can be open invite)
  target_did?: DID                   // Specific agent to invite
  
  // Invite properties
  role?: MemberRole                  // Role to grant (default: 'member')
  permissions?: Permission[]         // Additional permissions
  
  // Limits
  uses?: uint8                       // Max uses (default: 1)
  expires_at?: timestamp             // Expiration (default: 7 days)
  
  // Message
  message?: string                   // Personal message to invitee
}

interface InviteResult {
  success: boolean
  invite?: {
    id: UUID
    token: InviteToken               // Cryptographic token
    code: string                     // Human-shareable code (e.g., "VALNC-ABCD-EFGH")
    url: string                      // Direct join URL
    expires_at: timestamp
    remaining_uses: uint8
  }
  error?: FederationError
}
```

**Example:**
```typescript
// Invite specific person
const invite = await invite_member(federationId, {
  target_did: aliceDid,
  role: 'member',
  message: "Would love to have you in our research group!"
});

// Open invite link (limited uses)
const openInvite = await invite_member(federationId, {
  uses: 10,
  expires_at: Date.now() + 24 * 60 * 60 * 1000,  // 24 hours
});

// Share the code or URL
console.log(`Join code: ${openInvite.invite.code}`);
```

---

### approve_member

Approve a pending membership application.

```typescript
function approve_member(
  federation_id: UUID,
  application_id: UUID,
  decision: ApprovalDecision
): Promise<ApprovalResult>

interface ApprovalDecision {
  approved: boolean
  role?: MemberRole                  // Override default role
  permissions?: Permission[]         // Additional permissions
  message?: string                   // Message to applicant
}

interface ApprovalResult {
  success: boolean
  membership?: Membership            // If approved
  application_status: 'approved' | 'rejected' | 'pending'
  remaining_approvals?: uint8        // If still pending
  error?: FederationError
}
```

---

### update_member

Update a member's role or permissions.

```typescript
function update_member(
  federation_id: UUID,
  member_id: DID,
  updates: MemberUpdates
): Promise<UpdateMemberResult>

interface MemberUpdates {
  role?: MemberRole
  add_permissions?: Permission[]
  remove_permissions?: Permission[]
  suspended?: boolean
  reason?: string
}
```

---

### remove_member

Remove a member from the federation.

```typescript
function remove_member(
  federation_id: UUID,
  member_id: DID,
  options?: RemoveOptions
): Promise<RemoveMemberResult>

interface RemoveOptions {
  reason: string                     // Required for audit
  ban?: boolean                      // Prevent re-joining (default: false)
  quarantine_contributions?: boolean // Flag their contributions (default: false)
}
```

---

## Governance Operations

### create_proposal

Create a governance proposal.

```typescript
function create_proposal(
  federation_id: UUID,
  proposal: ProposalConfig
): Promise<CreateProposalResult>

interface ProposalConfig {
  type: ProposalType
  title: string
  description: string
  
  // Type-specific payload
  payload: ProposalPayload
  
  // Voting
  voting_period?: duration           // Override default
  quorum_override?: float            // Require higher quorum
}

enum ProposalType {
  CONFIG_CHANGE = 'config_change'
  MEMBER_ACTION = 'member_action'
  GOVERNANCE_CHANGE = 'governance_change'
  KEY_ROTATION = 'key_rotation'
  DISSOLUTION = 'dissolution'
  CUSTOM = 'custom'
}

type ProposalPayload =
  | { type: 'config_change', changes: Partial<FederationConfig> }
  | { type: 'member_action', action: 'promote' | 'demote' | 'remove', target: DID }
  | { type: 'governance_change', new_governance: Partial<GovernanceModel> }
  | { type: 'dissolution', successor?: DID, archive_policy: string }
  | { type: 'custom', data: any }
```

---

### cast_vote

Vote on a proposal.

```typescript
function cast_vote(
  federation_id: UUID,
  proposal_id: UUID,
  vote: Vote
): Promise<CastVoteResult>

interface Vote {
  choice: 'approve' | 'reject' | 'abstain'
  weight?: float                     // For meritocratic voting (auto if not specified)
  reason?: string                    // Optional public rationale
  anonymous?: boolean                // Hide voter identity (if allowed)
}
```

---

### get_proposals

List active and historical proposals.

```typescript
function get_proposals(
  federation_id: UUID,
  filters?: ProposalFilters
): Promise<GetProposalsResult>

interface ProposalFilters {
  status?: 'active' | 'passed' | 'failed' | 'expired'
  type?: ProposalType
  since?: timestamp
  limit?: uint16
}
```

---

## Administrative Operations

### update_federation

Update federation configuration.

```typescript
function update_federation(
  federation_id: UUID,
  updates: FederationUpdates
): Promise<UpdateFederationResult>

interface FederationUpdates {
  name?: string
  description?: string
  domains?: string[]
  visibility?: FederationVisibility
  config?: Partial<FederationConfig>
  // Governance changes may require proposal
}
```

---

### rotate_keys

Trigger key rotation.

```typescript
function rotate_keys(
  federation_id: UUID,
  options?: KeyRotationOptions
): Promise<KeyRotationResult>

interface KeyRotationOptions {
  reason: 'scheduled' | 'member_departure' | 'compromise' | 'admin_request'
  urgent?: boolean                   // Skip grace period
  notify_members?: boolean           // Send notifications (default: true)
}

interface KeyRotationResult {
  success: boolean
  new_epoch: uint64
  members_updated: uint64
  members_pending: uint64            // Still need to fetch new key
  error?: FederationError
}
```

---

### dissolve_federation

Dissolve the federation.

```typescript
function dissolve_federation(
  federation_id: UUID,
  options: DissolutionOptions
): Promise<DissolutionResult>

interface DissolutionOptions {
  reason: string
  grace_period?: duration            // Time before keys destroyed (default: 30d)
  archive_beliefs?: boolean          // Export beliefs before dissolution
  successor_federation?: UUID        // Recommend members join here
  notify_members?: boolean           // Send notifications (default: true)
}
```

---

## Query Operations

### get_federation

Get federation details.

```typescript
function get_federation(
  federation_id: UUID,
  options?: GetFederationOptions
): Promise<GetFederationResult>

interface GetFederationOptions {
  include_members?: boolean          // Include member list (if visible)
  include_stats?: boolean            // Include activity statistics
  include_governance?: boolean       // Include governance details
}

interface GetFederationResult {
  success: boolean
  federation?: Federation
  membership?: Membership            // Caller's membership (if member)
  stats?: FederationStats
  error?: FederationError
}

interface FederationStats {
  member_count: uint64               // May be noised
  belief_count: uint64               // Approximate
  aggregate_count: uint64
  active_members_30d: uint64
  avg_beliefs_per_member: float
  top_domains: { domain: string, count: uint64 }[]
  created_at: timestamp
  last_activity: timestamp
}
```

---

### list_federations

List federations the caller can see.

```typescript
function list_federations(
  filters?: FederationFilters
): Promise<ListFederationsResult>

interface FederationFilters {
  // Membership
  member_of?: boolean                // Only federations I'm a member of
  invited_to?: boolean               // Include federations I'm invited to
  
  // Discovery
  visibility?: FederationVisibility[]
  domains?: string[]                 // Filter by domain overlap
  min_members?: uint64
  max_members?: uint64
  
  // Search
  name_query?: string                // Search by name
  
  // Pagination
  limit?: uint16
  offset?: uint16
  order_by?: 'name' | 'members' | 'created' | 'activity'
}
```

---

### get_members

Get federation member list (if permitted).

```typescript
function get_members(
  federation_id: UUID,
  filters?: MemberFilters
): Promise<GetMembersResult>

interface MemberFilters {
  role?: MemberRole[]
  status?: MemberStatus[]
  include_contribution_scores?: boolean
  limit?: uint16
  offset?: uint16
}

interface GetMembersResult {
  success: boolean
  
  // Full list (if permitted)
  members?: Membership[]
  
  // Or just summary (if member list hidden)
  summary?: {
    total: uint64
    by_role: Map<MemberRole, uint64>
    by_status: Map<MemberStatus, uint64>
  }
  
  error?: FederationError
}
```

---

## Events & Subscriptions

### subscribe_federation

Subscribe to federation events.

```typescript
function subscribe_federation(
  federation_id: UUID,
  options?: SubscriptionOptions
): AsyncIterable<FederationEvent>

interface SubscriptionOptions {
  event_types?: FederationEventType[]  // Filter event types
  since?: timestamp                    // Start from this time
  include_historical?: boolean         // Replay past events first
}
```

---

### get_event_log

Get historical events.

```typescript
function get_event_log(
  federation_id: UUID,
  filters?: EventFilters
): Promise<GetEventLogResult>

interface EventFilters {
  event_types?: FederationEventType[]
  actors?: DID[]
  since?: timestamp
  until?: timestamp
  limit?: uint16
}
```

---

## Error Types

```typescript
interface FederationError {
  code: FederationErrorCode
  message: string
  details?: any
}

enum FederationErrorCode {
  // General
  FEDERATION_NOT_FOUND = 'federation_not_found'
  FEDERATION_INACTIVE = 'federation_inactive'
  INVALID_REQUEST = 'invalid_request'
  RATE_LIMITED = 'rate_limited'
  
  // Membership
  NOT_A_MEMBER = 'not_a_member'
  ALREADY_MEMBER = 'already_member'
  PERMISSION_DENIED = 'permission_denied'
  INVITE_REQUIRED = 'invite_required'
  INVITE_INVALID = 'invite_invalid'
  MEMBERSHIP_CLOSED = 'membership_closed'
  
  // Beliefs
  BELIEF_NOT_FOUND = 'belief_not_found'
  NOT_BELIEF_HOLDER = 'not_belief_holder'
  BELOW_MIN_CONFIDENCE = 'below_min_confidence'
  BELIEF_TOO_OLD = 'belief_too_old'
  DOMAIN_MISMATCH = 'domain_mismatch'
  DUPLICATE_SHARE = 'duplicate_share'
  INSUFFICIENT_CONTRIBUTORS = 'insufficient_contributors'
  
  // Governance
  PROPOSAL_NOT_FOUND = 'proposal_not_found'
  VOTING_CLOSED = 'voting_closed'
  ALREADY_VOTED = 'already_voted'
  QUORUM_NOT_MET = 'quorum_not_met'
  
  // Cryptographic
  KEY_ROTATION_FAILED = 'key_rotation_failed'
  DECRYPTION_FAILED = 'decryption_failed'
  SIGNATURE_INVALID = 'signature_invalid'
}
```

---

## Rate Limits

| Operation | Default Limit | Scope |
|-----------|--------------|-------|
| create_federation | 5/day | Per agent |
| join_federation | 20/hour | Per agent |
| share_to_federation | 100/hour | Per federation |
| query_federation | 1000/hour | Per federation |
| invite_member | 50/day | Per federation |

---

## Implementation Notes

### Idempotency

Operations that should be idempotent:
- `join_federation` with same invite
- `share_to_federation` with same belief
- `cast_vote` (last vote wins, not error)

### Pagination

All list operations support cursor-based pagination:
```typescript
// First page
const page1 = await query_federation(fedId, { limit: 20 });

// Next page
const page2 = await query_federation(fedId, { 
  limit: 20, 
  offset: 20  // Or use cursor from page1 if implemented
});
```

### Async Operations

Long-running operations return immediately with status:
```typescript
const result = await rotate_keys(fedId, { reason: 'scheduled' });
// result.members_pending > 0 means rotation still propagating
```

### Encryption Context

All `share_to_federation` and `query_federation` operations automatically handle encryption/decryption using the caller's key share. The interface hides cryptographic complexity.

---

*"Simple interface, strong guarantees, privacy by default."*
