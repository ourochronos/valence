# Incentive System — Interface

*"Simple operations, complex consequences."*

---

## Overview

This document specifies the programmatic interface to the Valence Incentive System. All reputation-related operations flow through these APIs.

**Design principles:**
- **Idempotent where possible** — Retrying requests is safe
- **Atomic** — Operations succeed or fail completely
- **Auditable** — Every operation is logged
- **Permissioned** — Operations check authorization

---

## 1. Core Types

### 1.1 Reputation Types

```typescript
/**
 * Complete reputation profile for an agent
 */
interface ReputationScore {
  overall: number                           // 0.0-1.0, global reputation
  by_domain: Record<string, DomainScore>    // Domain-specific scores
  verification_count: number                // Total verifications performed
  discrepancy_finds: number                 // Contradictions found and upheld
  stake_at_risk: number                     // Currently staked reputation
  calibration_score: number | null          // Null if insufficient data
  anchor_level: number                      // Minimum floor (0.1 default)
  last_updated: Timestamp
}

interface DomainScore {
  score: number                             // 0.0-1.0
  verification_count: number
  accuracy_rate: number
  discrepancy_finds: number
  expert_level: ExpertLevel | null
  last_active: Timestamp
}

type ExpertLevel = 'novice' | 'competent' | 'proficient' | 'expert' | 'master'

/**
 * Lightweight reputation summary for queries
 */
interface ReputationSummary {
  overall: number
  top_domains: Array<{ domain: string; score: number }>
  verification_count: number
  expert_in: string[]                       // Domains where expert or higher
}
```

### 1.2 Stake Types

```typescript
/**
 * Reputation stake (locked for some purpose)
 */
interface Stake {
  id: UUID
  holder_id: DID
  amount: number
  purpose: StakePurpose
  target_id: UUID                           // Belief, verification, or bounty ID
  status: StakeStatus
  created_at: Timestamp
  locked_until: Timestamp
  released_at: Timestamp | null
  outcome: StakeOutcome | null
}

type StakePurpose = 
  | 'claim'                                 // Staked on a belief assertion
  | 'verification'                          // Staked on a verification
  | 'dispute'                               // Staked on a dispute
  | 'bounty_claim'                          // Staked to claim a bounty
  | 'governance'                            // Staked for voting weight

type StakeStatus = 
  | 'active'                                // Currently locked
  | 'pending_release'                       // In release cooldown
  | 'released'                              // Returned to holder
  | 'forfeited'                             // Lost (transferred to counterparty)
  | 'disputed'                              // Under dispute, frozen

interface StakeOutcome {
  result: 'returned' | 'forfeited' | 'partial'
  amount_returned: number
  amount_forfeited: number
  bonus_earned: number
  recipient_id: DID | null                  // Who received forfeited stake
  reason: string
}
```

### 1.3 Reward Types

```typescript
/**
 * Reward from verification or contribution
 */
interface Reward {
  id: UUID
  recipient_id: DID
  type: RewardType
  amount: number
  source: RewardSource
  created_at: Timestamp
  claimed_at: Timestamp | null
  expires_at: Timestamp                     // Unclaimed rewards expire
}

type RewardType = 
  | 'verification_confirmation'
  | 'verification_contradiction'
  | 'verification_uncertain'
  | 'verification_partial'
  | 'discrepancy_bounty'
  | 'contribution'
  | 'calibration_bonus'
  | 'service'
  | 'referral'
  | 'dispute_win'

interface RewardSource {
  type: 'verification' | 'belief' | 'calibration' | 'service' | 'referral' | 'dispute'
  id: UUID
  details: Record<string, any>
}
```

### 1.4 Transfer Types

```typescript
/**
 * Reputation transfer between agents (restricted)
 */
interface Transfer {
  id: UUID
  from_id: DID
  to_id: DID
  amount: number
  domain: string | null                     // Null for overall reputation
  transfer_type: TransferType
  status: TransferStatus
  created_at: Timestamp
  completed_at: Timestamp | null
  reason: string
}

type TransferType = 
  | 'stake_forfeiture'                      // Lost stake goes to winner
  | 'bounty_payout'                         // Bounty claimed
  | 'service_payment'                       // Paid for query/service
  | 'governance_grant'                      // Federation grants reputation
  | 'dispute_resolution'                    // Dispute outcome transfer

type TransferStatus = 
  | 'pending'
  | 'completed'
  | 'failed'
  | 'reversed'
```

---

## 2. Reputation Operations

### 2.1 get_reputation

Retrieve an agent's reputation score.

```typescript
/**
 * Get reputation for an agent
 * 
 * @param agent_id - DID of the agent to query
 * @param domain - Optional domain filter (returns overall if omitted)
 * @param options - Query options
 * @returns ReputationScore or DomainScore depending on parameters
 */
async function get_reputation(
  agent_id: DID,
  domain?: string,
  options?: GetReputationOptions
): Promise<ReputationResult>

interface GetReputationOptions {
  include_history?: boolean                 // Include reputation over time
  history_period?: 'week' | 'month' | 'year'
  include_breakdown?: boolean               // Include component breakdown
  cross_federation?: boolean                // Query cross-federation reputation
  federation_id?: FederationID              // Specific federation context
  proof?: boolean                           // Return cryptographic proof
}

interface ReputationResult {
  agent_id: DID
  reputation: ReputationScore | DomainScore
  history?: ReputationHistory[]
  breakdown?: ReputationBreakdown
  cross_federation?: CrossFederationReputation
  proof?: SignedReputationProof
  as_of: Timestamp
}

interface ReputationHistory {
  timestamp: Timestamp
  overall: number
  change: number
  event_type: string
}

interface ReputationBreakdown {
  from_verifications: number
  from_contributions: number
  from_calibration: number
  from_services: number
  from_referrals: number
  penalties_applied: number
  decay_applied: number
}
```

**Example usage:**

```typescript
// Get overall reputation
const rep = await get_reputation('did:valence:alice')
console.log(`Alice's reputation: ${rep.reputation.overall}`)

// Get domain-specific reputation with history
const techRep = await get_reputation('did:valence:alice', 'tech/ai', {
  include_history: true,
  history_period: 'month'
})

// Get portable reputation proof
const proof = await get_reputation('did:valence:alice', null, {
  proof: true
})
```

**Errors:**

| Error Code | Meaning |
|------------|---------|
| `AGENT_NOT_FOUND` | No agent with this DID |
| `DOMAIN_NOT_FOUND` | Agent has no reputation in domain |
| `FEDERATION_UNAVAILABLE` | Cross-federation query failed |

---

### 2.2 get_reputation_batch

Batch reputation lookup for efficiency.

```typescript
/**
 * Get reputation for multiple agents
 * 
 * @param requests - Array of agent/domain pairs
 * @returns Map of results keyed by agent_id
 */
async function get_reputation_batch(
  requests: Array<{ agent_id: DID; domain?: string }>
): Promise<Map<DID, ReputationResult>>
```

---

### 2.3 get_my_reputation

Convenience method for the calling agent.

```typescript
/**
 * Get the calling agent's own reputation
 * Includes private details not available via get_reputation
 */
async function get_my_reputation(
  options?: GetReputationOptions
): Promise<MyReputationResult>

interface MyReputationResult extends ReputationResult {
  stake_at_risk_breakdown: StakeBreakdown[]
  pending_rewards: Reward[]
  recent_changes: ReputationChange[]
  velocity_status: VelocityStatus
  calibration_details: CalibrationDetails | null
}

interface StakeBreakdown {
  purpose: StakePurpose
  total: number
  items: Stake[]
}

interface VelocityStatus {
  verifications_today: number
  verifications_remaining: number
  reputation_gained_today: number
  reputation_gain_remaining: number
  throttled: boolean
  throttle_reason: string | null
}

interface CalibrationDetails {
  current_score: number
  sample_size: number
  by_bucket: CalibrationBucket[]
  projected_bonus: number
  next_calculation: Timestamp
}
```

---

## 3. Stake Operations

### 3.1 stake_reputation

Lock reputation for a specific purpose.

```typescript
/**
 * Stake reputation for verification, claim, or other purpose
 * 
 * @param amount - Amount to stake (will be validated against limits)
 * @param purpose - Why the stake is being created
 * @param options - Stake configuration
 * @returns Created stake
 */
async function stake_reputation(
  amount: number,
  purpose: StakePurpose,
  options: StakeOptions
): Promise<StakeResult>

interface StakeOptions {
  target_id: UUID                           // Belief, verification, or bounty ID
  auto_release?: boolean                    // Release automatically on resolution
  max_lock_duration?: number                // Max days to stay locked (default: 30)
}

interface StakeResult {
  stake: Stake
  available_after: number                   // Remaining available reputation
  release_conditions: string[]              // What triggers release
}
```

**Example usage:**

```typescript
// Stake for a verification
const stake = await stake_reputation(
  0.015,                                    // 1.5% of reputation
  'verification',
  { target_id: verificationId, auto_release: true }
)

// Stake for a claim (happens automatically, but can be explicit)
const claimStake = await stake_reputation(
  0.005,
  'claim', 
  { target_id: beliefId }
)
```

**Validation rules:**
- `amount >= min_stake(purpose, target)` — Meets minimum for this purpose
- `amount <= max_stake(purpose)` — Under maximum (typically 20%)
- `amount <= available_reputation` — Have enough unstaked reputation
- Not already staked on same target with same purpose

**Errors:**

| Error Code | Meaning |
|------------|---------|
| `INSUFFICIENT_REPUTATION` | Not enough available to stake |
| `BELOW_MINIMUM_STAKE` | Amount below required minimum |
| `ABOVE_MAXIMUM_STAKE` | Amount exceeds allowed maximum |
| `ALREADY_STAKED` | Already have active stake on target |
| `INVALID_TARGET` | Target doesn't exist or wrong type |
| `THROTTLED` | Too many stakes in short period |

---

### 3.2 get_stake

Retrieve stake details.

```typescript
/**
 * Get a specific stake
 */
async function get_stake(stake_id: UUID): Promise<Stake>

/**
 * Get all stakes for the calling agent
 */
async function get_my_stakes(
  options?: {
    purpose?: StakePurpose
    status?: StakeStatus
    limit?: number
    offset?: number
  }
): Promise<Stake[]>
```

---

### 3.3 release_stake

Request early stake release (not always allowed).

```typescript
/**
 * Request release of a stake
 * Only works for certain stake types and conditions
 * 
 * @param stake_id - Stake to release
 * @param reason - Why releasing early
 * @returns Updated stake (may be pending_release)
 */
async function release_stake(
  stake_id: UUID,
  reason: string
): Promise<StakeResult>
```

**Release rules:**
- `claim` stakes: Can release if belief is changed to private
- `verification` stakes: Cannot release early (must wait for resolution)
- `dispute` stakes: Cannot release (must complete dispute)
- `governance` stakes: Can release if vote not yet cast

**Errors:**

| Error Code | Meaning |
|------------|---------|
| `STAKE_NOT_FOUND` | No stake with this ID |
| `NOT_OWNER` | Stake belongs to different agent |
| `RELEASE_NOT_ALLOWED` | This stake type cannot be released early |
| `ALREADY_RELEASING` | Release already in progress |

---

### 3.4 calculate_stake

Preview stake requirements without committing.

```typescript
/**
 * Calculate required stake for an operation
 * Useful for UI previews
 */
async function calculate_stake(
  purpose: StakePurpose,
  target_id: UUID,
  options?: {
    include_your_discount?: boolean         // Apply domain expertise discount
  }
): Promise<StakeCalculation>

interface StakeCalculation {
  minimum_stake: number
  recommended_stake: number                 // For typical reward
  maximum_stake: number
  your_available: number
  your_discount: number                     // Domain expertise discount
  breakdown: {
    base: number
    confidence_factor: number
    domain_factor: number
    other_modifiers: Record<string, number>
  }
}
```

---

## 4. Reward Operations

### 4.1 claim_reward

Claim an earned reward.

```typescript
/**
 * Claim a pending reward
 * 
 * @param source_type - Type of reward source
 * @param source_id - ID of verification, bounty, etc.
 * @returns Claimed reward details
 */
async function claim_reward(
  source_type: 'verification' | 'bounty' | 'dispute' | 'calibration',
  source_id: UUID
): Promise<RewardResult>

interface RewardResult {
  reward: Reward
  reputation_before: number
  reputation_after: number
  domain_changes: Record<string, number>
}
```

**Example usage:**

```typescript
// Claim verification reward
const result = await claim_reward('verification', verificationId)
console.log(`Earned ${result.reward.amount} reputation`)

// Claim bounty reward
const bountyReward = await claim_reward('bounty', bountyId)
```

**Claim rules:**
- Reward must exist and be claimable
- Caller must be the designated recipient
- Reward must not be expired
- For verifications: verification must be finalized (accepted, not disputed)
- For bounties: bounty conditions must be met

**Errors:**

| Error Code | Meaning |
|------------|---------|
| `REWARD_NOT_FOUND` | No claimable reward for this source |
| `NOT_RECIPIENT` | You are not the reward recipient |
| `REWARD_EXPIRED` | Reward has expired (unclaimed too long) |
| `NOT_FINALIZED` | Source is not yet finalized |
| `ALREADY_CLAIMED` | Reward was already claimed |

---

### 4.2 get_pending_rewards

List unclaimed rewards.

```typescript
/**
 * Get all pending rewards for the calling agent
 */
async function get_pending_rewards(
  options?: {
    type?: RewardType
    min_amount?: number
    expires_before?: Timestamp
  }
): Promise<Reward[]>
```

---

### 4.3 claim_all_rewards

Batch claim all pending rewards.

```typescript
/**
 * Claim all pending rewards at once
 * More efficient than individual claims
 */
async function claim_all_rewards(
  options?: {
    type?: RewardType
    min_amount?: number
  }
): Promise<BatchClaimResult>

interface BatchClaimResult {
  claimed: Reward[]
  failed: Array<{ reward_id: UUID; error: string }>
  total_earned: number
  reputation_after: number
}
```

---

## 5. Transfer Operations

### 5.1 transfer_reputation

Transfer reputation to another agent (restricted).

```typescript
/**
 * Transfer reputation to another agent
 * Only allowed for specific transfer types
 * Direct peer-to-peer transfers are NOT supported
 * 
 * @param to - Recipient agent
 * @param amount - Amount to transfer
 * @param transfer_type - Type of transfer (must be allowed)
 * @param options - Transfer options
 */
async function transfer_reputation(
  to: DID,
  amount: number,
  transfer_type: TransferType,
  options: TransferOptions
): Promise<TransferResult>

interface TransferOptions {
  domain?: string                           // Domain-specific transfer
  source_id: UUID                           // Stake, bounty, or dispute ID
  reason: string
}

interface TransferResult {
  transfer: Transfer
  from_reputation_after: number
  to_reputation_after: number
}
```

**Transfer restrictions:**

Direct peer-to-peer reputation transfers are **NOT allowed** to prevent:
- Reputation markets (buying/selling reputation)
- Sybil attacks (concentrating reputation in one identity)
- Collusion (paying for favorable verifications)

**Allowed transfer types:**

| Type | When Allowed | Initiated By |
|------|--------------|--------------|
| `stake_forfeiture` | Stake is lost | System |
| `bounty_payout` | Bounty claimed | System |
| `service_payment` | Query/service fee | System |
| `governance_grant` | Federation decision | Federation |
| `dispute_resolution` | Dispute resolved | System |

All transfers are initiated by the system based on protocol rules, not by agents directly.

**Errors:**

| Error Code | Meaning |
|------------|---------|
| `TRANSFER_NOT_ALLOWED` | This transfer type not permitted |
| `INVALID_SOURCE` | Source doesn't justify transfer |
| `INSUFFICIENT_REPUTATION` | Not enough to transfer |
| `RECIPIENT_NOT_FOUND` | Recipient doesn't exist |

---

### 5.2 get_transfer_history

View past transfers (for auditing).

```typescript
/**
 * Get transfer history for the calling agent
 */
async function get_transfer_history(
  options?: {
    direction?: 'incoming' | 'outgoing' | 'both'
    transfer_type?: TransferType
    since?: Timestamp
    limit?: number
  }
): Promise<Transfer[]>
```

---

## 6. Bounty Operations

### 6.1 create_bounty

Fund a bounty for specific verification work.

```typescript
/**
 * Create a bounty for verification work
 * 
 * @param target - What should be verified
 * @param amount - Reputation to offer
 * @param requirements - Conditions for claiming
 */
async function create_bounty(
  target: BountyTarget,
  amount: number,
  requirements: BountyRequirements
): Promise<Bounty>

interface BountyTarget {
  type: 'verify_belief' | 'find_contradiction' | 'research_question' | 'synthesis'
  belief_id?: UUID                          // For verify_belief, find_contradiction
  query?: string                            // For research_question
  domains?: string[]                        // For synthesis
}

interface BountyRequirements {
  min_verifier_reputation?: number
  min_domain_reputation?: number
  evidence_types?: EvidenceType[]
  deadline: Timestamp
}
```

**Example usage:**

```typescript
// Bounty for finding contradictions in a specific belief
const bounty = await create_bounty(
  { type: 'find_contradiction', belief_id: beliefId },
  0.05,                                     // 5% of my reputation
  {
    min_verifier_reputation: 0.4,
    deadline: Date.now() + 30 * 24 * 60 * 60 * 1000  // 30 days
  }
)

// Bounty for researching a question
const researchBounty = await create_bounty(
  { type: 'research_question', query: 'What is the accuracy of GPT-4 on math problems?' },
  0.03,
  {
    evidence_types: ['external', 'observation'],
    deadline: Date.now() + 14 * 24 * 60 * 60 * 1000
  }
)
```

**Errors:**

| Error Code | Meaning |
|------------|---------|
| `INSUFFICIENT_REPUTATION` | Can't fund this bounty |
| `BOUNTY_TOO_SMALL` | Below minimum (1%) |
| `BOUNTY_TOO_LARGE` | Above maximum (10%) |
| `INVALID_TARGET` | Target specification invalid |
| `DEADLINE_TOO_SHORT` | Deadline must be >24 hours |

---

### 6.2 get_bounties

Find available bounties.

```typescript
/**
 * List available bounties
 */
async function get_bounties(
  options?: {
    type?: BountyTarget['type']
    domain?: string
    min_amount?: number
    claimable_by_me?: boolean               // Filter to bounties I qualify for
    sort?: 'amount' | 'deadline' | 'created'
  }
): Promise<Bounty[]>
```

---

### 6.3 cancel_bounty

Cancel an unfilled bounty.

```typescript
/**
 * Cancel a bounty (50% refund if within 24h, no refund after)
 */
async function cancel_bounty(bounty_id: UUID): Promise<CancelResult>

interface CancelResult {
  bounty: Bounty
  refund_amount: number
  refund_percentage: number
}
```

---

## 7. Calibration Operations

### 7.1 get_calibration

Get calibration score and details.

```typescript
/**
 * Get calibration score for an agent
 */
async function get_calibration(
  agent_id?: DID                            // Defaults to self
): Promise<CalibrationResult>

interface CalibrationResult {
  agent_id: DID
  overall_score: number | null              // Null if insufficient data
  sample_size: number
  minimum_required: number                  // 50 for scoring
  by_bucket: CalibrationBucket[]
  trend: 'improving' | 'stable' | 'declining'
  next_bonus_date: Timestamp | null
  projected_bonus: number
}

interface CalibrationBucket {
  range: [number, number]                   // e.g., [0.7, 0.8]
  claimed_count: number
  actual_accuracy: number
  error: number                             // |claimed - actual|
}
```

---

### 7.2 simulate_calibration

See how a new belief would affect calibration.

```typescript
/**
 * Simulate calibration impact of a new belief
 * Useful for choosing confidence levels
 */
async function simulate_calibration(
  confidence: number,
  outcome: 'confirmed' | 'contradicted'
): Promise<CalibrationSimulation>

interface CalibrationSimulation {
  current_score: number
  projected_score: number
  change: number
  recommendation: string                    // e.g., "Consider higher confidence"
}
```

---

## 8. Query Fee Operations

### 8.1 get_query_pricing

Check costs for premium query features.

```typescript
/**
 * Get pricing for query features
 */
async function get_query_pricing(): Promise<QueryPricing>

interface QueryPricing {
  features: QueryFeaturePricing[]
  your_available_reputation: number
  discounts: QueryDiscount[]
}

interface QueryFeaturePricing {
  feature: QueryFeeType
  base_cost: number
  your_cost: number                         // After discounts
  unit: string                              // e.g., "per query", "per 100 results"
}

interface QueryDiscount {
  type: string                              // e.g., "federation_member"
  discount: number
  reason: string
}
```

---

### 8.2 authorize_query_spend

Pre-authorize spending for a query session.

```typescript
/**
 * Authorize spending for premium query features
 * Prevents per-query authorization overhead
 */
async function authorize_query_spend(
  max_amount: number,
  features: QueryFeeType[],
  duration_minutes?: number                 // Default: 60
): Promise<QueryAuthorization>

interface QueryAuthorization {
  id: UUID
  authorized_amount: number
  remaining_amount: number
  features: QueryFeeType[]
  expires_at: Timestamp
}
```

---

## 9. Audit & History

### 9.1 get_reputation_events

Full audit trail of reputation changes.

```typescript
/**
 * Get detailed history of reputation changes
 */
async function get_reputation_events(
  options?: {
    since?: Timestamp
    until?: Timestamp
    event_types?: string[]
    min_change?: number
    limit?: number
  }
): Promise<ReputationEvent[]>

interface ReputationEvent {
  id: UUID
  agent_id: DID
  event_type: string
  delta: number
  old_value: number
  new_value: number
  source: EventSource
  timestamp: Timestamp
  details: Record<string, any>
}

interface EventSource {
  type: string
  id: UUID
  description: string
}
```

---

### 9.2 get_stake_history

History of staking activity.

```typescript
/**
 * Get staking history
 */
async function get_stake_history(
  options?: {
    purpose?: StakePurpose
    outcome?: 'returned' | 'forfeited' | 'partial'
    since?: Timestamp
    limit?: number
  }
): Promise<StakeHistoryEntry[]>

interface StakeHistoryEntry {
  stake: Stake
  duration_days: number
  return_rate: number                       // bonus/stake
  net_result: number                        // Final gain/loss
}
```

---

## 10. Administrative Operations

### 10.1 Governance Operations

For federation administrators:

```typescript
/**
 * Grant reputation (federation governance only)
 */
async function governance_grant(
  recipient: DID,
  amount: number,
  reason: string,
  domain?: string
): Promise<Transfer>

/**
 * Apply penalty (federation governance only)
 */
async function governance_penalty(
  target: DID,
  amount: number,
  reason: string,
  evidence: UUID[]
): Promise<ReputationEvent>

/**
 * Set probation (federation governance only)
 */
async function set_probation(
  target: DID,
  duration_days: number,
  restrictions: string[],
  reason: string
): Promise<Probation>
```

---

### 10.2 Emergency Operations

For network emergencies:

```typescript
/**
 * Freeze reputation changes (requires multi-sig)
 */
async function freeze_reputation(
  target: DID | 'network',
  freeze_type: 'gains' | 'losses' | 'both',
  duration_hours: number,
  reason: string
): Promise<ReputationFreeze>

/**
 * Retroactive adjustment (requires governance approval)
 */
async function retroactive_adjust(
  target: DID,
  period: [Timestamp, Timestamp],
  adjustment_type: 'recalculate' | 'revert' | 'penalty',
  evidence: UUID[]
): Promise<RetroactiveAdjustment>
```

---

## 11. Events & Webhooks

### 11.1 Event Types

```typescript
type IncentiveEvent =
  | { type: 'reputation_changed'; agent_id: DID; delta: number; reason: string }
  | { type: 'stake_created'; stake: Stake }
  | { type: 'stake_released'; stake: Stake; outcome: StakeOutcome }
  | { type: 'reward_available'; reward: Reward }
  | { type: 'reward_claimed'; reward: Reward }
  | { type: 'reward_expired'; reward: Reward }
  | { type: 'bounty_created'; bounty: Bounty }
  | { type: 'bounty_claimed'; bounty: Bounty; claimant: DID }
  | { type: 'calibration_calculated'; score: number; bonus: number }
  | { type: 'threshold_crossed'; threshold: string; direction: 'up' | 'down' }
  | { type: 'velocity_warning'; remaining: VelocityStatus }
  | { type: 'probation_started'; probation: Probation }
  | { type: 'probation_ended'; reason: string }
```

### 11.2 Subscribe to Events

```typescript
/**
 * Subscribe to incentive events
 */
async function subscribe_incentive_events(
  event_types: IncentiveEvent['type'][],
  callback: (event: IncentiveEvent) => void
): Promise<Subscription>
```

---

## 12. Error Codes Summary

| Category | Code | HTTP | Meaning |
|----------|------|------|---------|
| Reputation | `AGENT_NOT_FOUND` | 404 | Agent doesn't exist |
| Reputation | `DOMAIN_NOT_FOUND` | 404 | No reputation in domain |
| Stake | `INSUFFICIENT_REPUTATION` | 400 | Not enough to stake |
| Stake | `BELOW_MINIMUM_STAKE` | 400 | Stake too small |
| Stake | `ABOVE_MAXIMUM_STAKE` | 400 | Stake too large |
| Stake | `ALREADY_STAKED` | 409 | Duplicate stake |
| Stake | `RELEASE_NOT_ALLOWED` | 403 | Cannot release this stake |
| Reward | `REWARD_NOT_FOUND` | 404 | No such reward |
| Reward | `NOT_RECIPIENT` | 403 | Not your reward |
| Reward | `REWARD_EXPIRED` | 410 | Reward expired |
| Reward | `ALREADY_CLAIMED` | 409 | Already claimed |
| Transfer | `TRANSFER_NOT_ALLOWED` | 403 | Invalid transfer type |
| Bounty | `BOUNTY_TOO_SMALL` | 400 | Below minimum |
| Bounty | `BOUNTY_TOO_LARGE` | 400 | Above maximum |
| Rate | `THROTTLED` | 429 | Rate limit exceeded |
| Auth | `UNAUTHORIZED` | 401 | Not authenticated |
| Auth | `FORBIDDEN` | 403 | Not permitted |

---

*"The interface is simple. The consequences compound."*
