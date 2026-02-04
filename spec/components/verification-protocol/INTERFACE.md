# Verification Protocol — Interface

*API specification for verification operations.*

---

## Overview

This document defines the programmatic interface for the Verification Protocol. All operations are designed to be:

- **Atomic**: Operations complete fully or fail cleanly
- **Authenticated**: Require valid DID signatures
- **Auditable**: All state changes logged
- **Idempotent**: Where possible, safe to retry

---

## 1. Core Operations

### 1.1 submit_verification

Submit a new verification for a belief.

```typescript
function submit_verification(
  belief_id: UUID,
  result: VerificationResult,
  evidence: Evidence[],
  stake: float,
  options?: SubmitOptions
): Promise<VerificationSubmission>

interface SubmitOptions {
  reasoning?: string              // Explanation (max 16KB)
  result_details?: ResultDetails  // Structured breakdown
  stake_type?: StakeType          // default: STANDARD
  notify_holder?: boolean         // default: true
}

interface VerificationSubmission {
  verification_id: UUID
  status: 'pending' | 'rejected'
  stake_locked: float
  escrow_id: UUID
  estimated_acceptance: timestamp  // When it will be accepted (if valid)
  rejection_reason?: string        // If rejected
}
```

**Authentication**: Requires valid signature from verifier's identity key.

**Request signing**:
```typescript
// Payload to sign
const payload = {
  belief_id,
  result,
  evidence_hashes: evidence.map(e => sha256(canonicalize(e))),
  stake,
  timestamp: now(),
  nonce: random_bytes(16)
}
const signature = ed25519_sign(payload, verifier_private_key)
```

**Validation checks** (in order):
1. Verifier DID is valid and not revoked
2. Belief exists and is verifiable (public/federated, not expired)
3. Verifier ≠ belief holder
4. No existing verification by this verifier for this belief
5. Evidence meets minimum requirements for result type
6. Stake meets minimum requirement
7. Verifier has sufficient reputation for stake
8. Rate limits not exceeded

**Errors**:
| Code | Meaning |
|------|---------|
| `INVALID_IDENTITY` | Verifier DID invalid or revoked |
| `BELIEF_NOT_FOUND` | Belief doesn't exist |
| `BELIEF_NOT_VERIFIABLE` | Private, expired, or tombstoned |
| `SELF_VERIFICATION` | Cannot verify own belief |
| `DUPLICATE_VERIFICATION` | Already verified this belief |
| `INSUFFICIENT_EVIDENCE` | Evidence doesn't meet requirements |
| `INSUFFICIENT_STAKE` | Stake below minimum |
| `INSUFFICIENT_REPUTATION` | Not enough reputation to stake |
| `RATE_LIMITED` | Too many verifications |

**Example**:
```typescript
const submission = await submit_verification(
  "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  VerificationResult.CONTRADICTED,
  [{
    type: EvidenceType.EXTERNAL,
    external_source: {
      url: "https://authoritative-source.com/correction",
      archive_hash: "a3f2b8c1...",
      source_reputation: 0.9
    },
    relevance: 0.95,
    contribution: EvidenceContribution.CONTRADICTS,
    verifier_notes: "Original claim stated 2022, actual date was 2023"
  }],
  0.05,  // 5% stake
  {
    reasoning: "The belief claims event X happened in 2022, but authoritative records show it was 2023.",
    result_details: {
      contradiction_type: ContradictionType.FACTUALLY_FALSE,
      corrected_belief: "Event X occurred in March 2023, not 2022.",
      severity: 'moderate'
    }
  }
)

// Returns:
{
  verification_id: "01941d3c-9e8e-7e3d-af5f-4b6c7d8e9f0a",
  status: "pending",
  stake_locked: 0.05,
  escrow_id: "escrow-123",
  estimated_acceptance: "2024-01-16T10:30:00Z"
}
```

---

### 1.2 get_verifications

Retrieve verifications for a belief.

```typescript
function get_verifications(
  belief_id: UUID,
  options?: GetVerificationsOptions
): Promise<VerificationList>

interface GetVerificationsOptions {
  status?: VerificationStatus[]   // Filter by status (default: all)
  result?: VerificationResult[]   // Filter by result type
  verifier_id?: DID               // Filter by verifier
  min_stake?: float               // Minimum stake threshold
  limit?: number                  // Max results (default: 50, max: 200)
  offset?: number                 // Pagination offset
  order_by?: 'created_at' | 'stake' | 'verifier_reputation'
  order_dir?: 'asc' | 'desc'
  include_evidence?: boolean      // Include full evidence (default: false)
}

interface VerificationList {
  verifications: Verification[]
  total_count: number
  has_more: boolean
  summary: VerificationSummary
}

interface VerificationSummary {
  total: number
  by_result: {
    confirmed: number
    contradicted: number
    uncertain: number
    partial: number
  }
  by_status: {
    pending: number
    accepted: number
    disputed: number
    overturned: number
  }
  average_stake: float
  total_stake: float
  consensus_result?: VerificationResult  // Weighted consensus if clear
  consensus_confidence?: float           // How confident in consensus
}
```

**Authentication**: None required for public beliefs. Verifier DID required for federated beliefs.

**Example**:
```typescript
const result = await get_verifications(
  "01941d3a-7c5b-7b0a-8e2d-1a3b4c5d6e7f",
  {
    status: [VerificationStatus.ACCEPTED],
    order_by: 'stake',
    order_dir: 'desc',
    limit: 10
  }
)

// Returns:
{
  verifications: [...],
  total_count: 47,
  has_more: true,
  summary: {
    total: 47,
    by_result: { confirmed: 38, contradicted: 5, uncertain: 3, partial: 1 },
    by_status: { pending: 2, accepted: 42, disputed: 2, overturned: 1 },
    average_stake: 0.032,
    total_stake: 1.504,
    consensus_result: "confirmed",
    consensus_confidence: 0.87
  }
}
```

---

### 1.3 dispute_verification

Challenge an accepted verification with counter-evidence.

```typescript
function dispute_verification(
  verification_id: UUID,
  counter_evidence: Evidence[],
  dispute_stake: float,
  options?: DisputeOptions
): Promise<DisputeSubmission>

interface DisputeOptions {
  reasoning: string               // Required: why is the verification wrong?
  dispute_type: DisputeType       // What kind of dispute
  proposed_result?: VerificationResult  // What result should it be?
  notify_verifier?: boolean       // default: true
}

enum DisputeType {
  EVIDENCE_INVALID = 'evidence_invalid'      // Evidence doesn't support result
  EVIDENCE_FABRICATED = 'evidence_fabricated' // Evidence is fake/manipulated
  EVIDENCE_INSUFFICIENT = 'evidence_insufficient'  // Not enough evidence
  REASONING_FLAWED = 'reasoning_flawed'      // Logic doesn't follow
  CONFLICT_OF_INTEREST = 'conflict_of_interest'  // Verifier biased
  NEW_EVIDENCE = 'new_evidence'              // New contradicting evidence found
}

interface DisputeSubmission {
  dispute_id: UUID
  status: 'pending' | 'rejected'
  stake_locked: float
  escrow_id: UUID
  resolution_deadline: timestamp   // When dispute must be resolved
  rejection_reason?: string
}
```

**Authentication**: Requires valid signature from disputer's identity key.

**Who can dispute**:
- Belief holder (always)
- Any agent (if dispute_stake ≥ original verification stake)

**Dispute stake minimum**:
```
min_dispute_stake = verification.stake × dispute_multiplier

where:
  dispute_multiplier = 1.0 (belief holder)
                     = 1.5 (third party)
```

**Validation checks**:
1. Verification exists and status is ACCEPTED
2. Dispute window not expired (7 days from acceptance)
3. Disputer has sufficient reputation for stake
4. Counter-evidence provided
5. Not a duplicate dispute (same disputer, same grounds)

**Errors**:
| Code | Meaning |
|------|---------|
| `VERIFICATION_NOT_FOUND` | Verification doesn't exist |
| `NOT_ACCEPTED` | Verification not in ACCEPTED status |
| `WINDOW_EXPIRED` | Dispute window has passed |
| `INSUFFICIENT_STAKE` | Stake below minimum |
| `INSUFFICIENT_REPUTATION` | Not enough reputation |
| `DUPLICATE_DISPUTE` | Already disputed on these grounds |
| `NO_COUNTER_EVIDENCE` | Must provide counter-evidence |

**Example**:
```typescript
const dispute = await dispute_verification(
  "01941d3c-9e8e-7e3d-af5f-4b6c7d8e9f0a",
  [{
    type: EvidenceType.EXTERNAL,
    external_source: {
      url: "https://more-authoritative-source.com/fact",
      archive_hash: "b4h3d1f2...",
    },
    relevance: 0.98,
    contribution: EvidenceContribution.SUPPORTS
  }],
  0.075,  // 7.5% stake
  {
    reasoning: "The verification claimed the original belief was false, but more authoritative sources confirm the original belief was correct.",
    dispute_type: DisputeType.NEW_EVIDENCE,
    proposed_result: VerificationResult.CONFIRMED
  }
)
```

---

### 1.4 resolve_dispute

Resolve a pending dispute. Called by resolution mechanism.

```typescript
function resolve_dispute(
  dispute_id: UUID,
  resolution: Resolution
): Promise<DisputeResolution>

interface Resolution {
  outcome: DisputeOutcome
  reasoning: string                // Explanation of resolution
  evidence_assessment: {           // How each piece of evidence was weighed
    evidence_id: UUID
    credibility: float             // 0.0-1.0
    relevance: float               // 0.0-1.0
    weight_in_decision: float      // 0.0-1.0
    notes?: string
  }[]
  arbiters?: DID[]                 // Who participated in resolution
  confidence: float                // How confident in this resolution
}

enum DisputeOutcome {
  UPHELD = 'upheld'                // Original verification stands
  OVERTURNED = 'overturned'        // Verification was wrong
  MODIFIED = 'modified'            // Result changed (e.g., CONFIRMED → PARTIAL)
  DISMISSED = 'dismissed'          // Dispute was frivolous/invalid
}

interface DisputeResolution {
  dispute_id: UUID
  outcome: DisputeOutcome
  verification_new_status: VerificationStatus
  stake_transfers: StakeTransfer[]
  reputation_updates: ReputationUpdate[]
  resolved_at: timestamp
}

interface StakeTransfer {
  from: DID
  to: DID
  amount: float
  reason: string
}

interface ReputationUpdate {
  identity: DID
  dimension: string   // 'overall' | domain
  delta: float
  reason: string
}
```

**Authentication**: Resolution mechanism only (see Resolution Mechanisms below).

**Outcome effects**:

| Outcome | Verification | Verifier | Disputer | Holder |
|---------|--------------|----------|----------|--------|
| UPHELD | Stays ACCEPTED | Keeps stake + bonus | Loses stake | No change |
| OVERTURNED | → OVERTURNED | Loses stake | Gets stake + bonus | Restored |
| MODIFIED | Updated | Partial stake | Partial return | Proportional |
| DISMISSED | Stays ACCEPTED | Keeps stake | Loses stake, penalty | No change |

---

## 2. Query Operations

### 2.1 get_verification

Get a single verification by ID.

```typescript
function get_verification(
  verification_id: UUID,
  options?: GetVerificationOptions
): Promise<Verification | null>

interface GetVerificationOptions {
  include_evidence?: boolean      // Include full evidence objects
  include_dispute?: boolean       // Include dispute if exists
  include_holder?: boolean        // Include belief holder info
}
```

---

### 2.2 get_verifier_history

Get verification history for a verifier.

```typescript
function get_verifier_history(
  verifier_id: DID,
  options?: HistoryOptions
): Promise<VerifierHistory>

interface HistoryOptions {
  domain?: string                 // Filter by domain
  result?: VerificationResult[]   // Filter by result
  status?: VerificationStatus[]   // Filter by status
  from_date?: timestamp
  to_date?: timestamp
  limit?: number
  offset?: number
}

interface VerifierHistory {
  verifications: Verification[]
  total_count: number
  stats: {
    total_verifications: number
    by_result: Record<VerificationResult, number>
    by_status: Record<VerificationStatus, number>
    accuracy_rate: float          // Accepted / (Accepted + Overturned)
    discrepancy_rate: float       // Contradicted / Total
    avg_stake: float
    total_stake_earned: float
    total_stake_lost: float
  }
}
```

---

### 2.3 get_disputes

Get disputes for a verification or by participant.

```typescript
function get_disputes(
  filter: DisputeFilter
): Promise<DisputeList>

interface DisputeFilter {
  verification_id?: UUID          // Disputes for this verification
  disputer_id?: DID               // Disputes by this agent
  verifier_id?: DID               // Disputes against this verifier
  status?: DisputeStatus[]        // Filter by status
  type?: DisputeType[]            // Filter by dispute type
  limit?: number
  offset?: number
}

interface DisputeList {
  disputes: Dispute[]
  total_count: number
}
```

---

### 2.4 get_pending_verifications

Get verifications awaiting action (for verifiers or holders).

```typescript
function get_pending_verifications(
  options: PendingOptions
): Promise<PendingList>

interface PendingOptions {
  for_holder?: DID                // Beliefs I hold being verified
  for_verifier?: DID              // My verifications pending
  type: 'awaiting_acceptance' | 'awaiting_dispute' | 'disputed' | 'all'
}

interface PendingList {
  verifications: Verification[]
  deadlines: Map<UUID, timestamp> // Important dates per verification
}
```

---

## 3. Stake Management

### 3.1 get_stake_balance

Get current stake positions.

```typescript
function get_stake_balance(
  identity: DID
): Promise<StakeBalance>

interface StakeBalance {
  available_reputation: float     // Can be staked
  total_staked: float             // Currently locked
  by_verification: StakePosition[]
  by_dispute: StakePosition[]
  pending_returns: {
    amount: float
    unlock_at: timestamp
  }[]
}

interface StakePosition {
  id: UUID                        // Verification or dispute ID
  amount: float
  type: StakeType
  locked_at: timestamp
  unlocks_at: timestamp
  status: 'locked' | 'pending_return' | 'forfeited' | 'returned'
}
```

---

### 3.2 withdraw_stake

Withdraw stake after lockup period (if verification not disputed).

```typescript
function withdraw_stake(
  verification_id: UUID
): Promise<StakeWithdrawal>

interface StakeWithdrawal {
  amount: float
  bonus: float                    // Earned from successful verification
  returned_to: DID
  withdrawn_at: timestamp
}
```

**Conditions**:
- Verification status is ACCEPTED
- Lockup period has passed
- No pending dispute

---

## 4. Resolution Mechanisms

### 4.1 Resolution Methods

Disputes can be resolved through multiple mechanisms:

```typescript
enum ResolutionMethod {
  AUTOMATIC = 'automatic'         // Algorithm decides based on evidence
  JURY = 'jury'                   // Random selection of qualified jurors
  EXPERT = 'expert'               // Domain experts decide
  APPEAL = 'appeal'               // Higher-level review
}
```

### 4.2 Automatic Resolution

For clear-cut cases:

```typescript
interface AutomaticResolution {
  method: ResolutionMethod.AUTOMATIC
  criteria: {
    evidence_quality_threshold: float  // Minimum quality for decision
    confidence_threshold: float        // Minimum confidence to auto-resolve
    agreement_required: float          // Fraction of evidence agreeing
  }
  result: Resolution
}
```

**Triggers automatic resolution**:
- Overwhelming evidence one direction (>90% quality-weighted)
- Dispute filed without valid counter-evidence
- Technical violations (forged evidence, broken references)

### 4.3 Jury Resolution

For contested cases:

```typescript
interface JuryResolution {
  method: ResolutionMethod.JURY
  jury_size: number               // 5, 11, or 21
  jury_selection: {
    domain_required?: string      // Must have domain reputation
    min_reputation: float         // Minimum overall reputation
    max_disputes_with_parties: number  // Conflict of interest limit
    random_seed: bytes            // Verifiable randomness
  }
  votes: JuryVote[]
  result: Resolution
}

interface JuryVote {
  juror: DID
  vote: DisputeOutcome
  confidence: float
  reasoning?: string
}
```

**Jury selection process**:
1. Filter eligible jurors (reputation, domain, no conflicts)
2. Use verifiable random selection (VRF or commit-reveal)
3. Jurors have 48 hours to vote
4. Majority determines outcome
5. Jurors earn small reputation bonus for participation

### 4.4 Expert Resolution

For domain-specific disputes:

```typescript
interface ExpertResolution {
  method: ResolutionMethod.EXPERT
  required_domain: string
  experts_selected: DID[]
  min_domain_reputation: float
  expert_opinions: ExpertOpinion[]
  result: Resolution
}

interface ExpertOpinion {
  expert: DID
  opinion: DisputeOutcome
  confidence: float
  detailed_analysis: string
  evidence_citations: UUID[]
}
```

---

## 5. Events & Notifications

### 5.1 Verification Events

```typescript
type VerificationEvent = 
  | { type: 'verification_submitted', verification_id: UUID, belief_id: UUID, verifier: DID }
  | { type: 'verification_accepted', verification_id: UUID, result: VerificationResult }
  | { type: 'verification_rejected', verification_id: UUID, reason: string }
  | { type: 'verification_disputed', verification_id: UUID, dispute_id: UUID, disputer: DID }
  | { type: 'dispute_resolved', dispute_id: UUID, outcome: DisputeOutcome }
  | { type: 'stake_returned', verification_id: UUID, amount: float, recipient: DID }
  | { type: 'reputation_updated', identity: DID, delta: float, reason: string }
```

### 5.2 Subscribe to Events

```typescript
function subscribe_verification_events(
  filter: EventFilter,
  callback: (event: VerificationEvent) => void
): Subscription

interface EventFilter {
  belief_ids?: UUID[]             // Events for specific beliefs
  verifier_ids?: DID[]            // Events for specific verifiers
  holder_ids?: DID[]              // Events for specific holders
  event_types?: string[]          // Specific event types
}

interface Subscription {
  unsubscribe(): void
  id: string
}
```

---

## 6. Batch Operations

### 6.1 submit_batch_verification

Verify multiple related beliefs at once.

```typescript
function submit_batch_verification(
  belief_ids: UUID[],
  common_result: VerificationResult,
  common_evidence: Evidence[],
  stake: float,
  options?: BatchOptions
): Promise<BatchSubmission>

interface BatchOptions {
  per_belief_notes?: Map<UUID, string>
  per_belief_result?: Map<UUID, VerificationResult>  // Override common result
  reasoning?: string
}

interface BatchSubmission {
  batch_id: UUID
  verification_ids: UUID[]
  total_stake_locked: float
  status: 'pending' | 'partial' | 'rejected'
  failures?: { belief_id: UUID, reason: string }[]
}
```

**Constraints**:
- Max 20 beliefs per batch
- All beliefs must share at least one domain
- Single stake covers all (distributed proportionally)

---

## 7. Error Handling

### 7.1 Error Response Format

```typescript
interface VerificationError {
  code: string
  message: string
  details?: Record<string, any>
  retry_after?: number            // Seconds, for rate limits
  help_url?: string               // Documentation link
}
```

### 7.2 Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `INVALID_IDENTITY` | 401 | Identity invalid or revoked |
| `INVALID_SIGNATURE` | 401 | Signature doesn't verify |
| `BELIEF_NOT_FOUND` | 404 | Belief doesn't exist |
| `VERIFICATION_NOT_FOUND` | 404 | Verification doesn't exist |
| `DISPUTE_NOT_FOUND` | 404 | Dispute doesn't exist |
| `NOT_AUTHORIZED` | 403 | Cannot perform this action |
| `SELF_VERIFICATION` | 400 | Cannot verify own belief |
| `DUPLICATE_VERIFICATION` | 409 | Already verified |
| `INSUFFICIENT_STAKE` | 400 | Stake too low |
| `INSUFFICIENT_REPUTATION` | 400 | Not enough reputation |
| `INSUFFICIENT_EVIDENCE` | 400 | Evidence doesn't meet requirements |
| `WINDOW_EXPIRED` | 400 | Action window has passed |
| `RATE_LIMITED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |

---

## 8. Implementation Notes

### 8.1 Idempotency

Operations that modify state accept an `idempotency_key`:

```typescript
function submit_verification(
  ...,
  options?: {
    ...,
    idempotency_key?: string   // UUID, prevents double-submission
  }
)
```

Same key within 24 hours returns cached response.

### 8.2 Rate Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| submit_verification | 50 | per day |
| dispute_verification | 10 | per day |
| get_verifications | 1000 | per hour |
| get_verification | 5000 | per hour |
| subscribe_events | 10 | concurrent |

### 8.3 Pagination

List endpoints use cursor-based pagination:

```typescript
interface Paginated<T> {
  items: T[]
  cursor?: string       // Pass to next request
  has_more: boolean
}
```

---

*"Clear interfaces enable trustworthy systems."*
