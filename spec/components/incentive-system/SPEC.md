# Incentive System — Specification

*"Reputation is the currency of truth. Earn it through accuracy, spend it on influence."*

---

## Overview

The Valence Incentive System creates economic alignment between individual agent interests and collective epistemic health. Unlike token-based systems, **reputation IS the currency** — there are no external tokens to buy or sell.

This design ensures:
- **Truth-seeking is profitable** — Accurate beliefs and honest verification earn reputation
- **Deception is expensive** — Gaming the system costs more than it gains
- **Sybil attacks are uneconomical** — Creating fake identities provides no advantage
- **Participation is rewarded** — Active contributors gain influence and access
- **Quality matters more than quantity** — Calibration and accuracy compound over time

---

## 1. Design Philosophy

### 1.1 Reputation as Native Currency

Reputation is not a proxy for something else — it IS the thing of value:

| Traditional Systems | Valence |
|---------------------|---------|
| Tokens backed by speculation | Reputation backed by demonstrated accuracy |
| Buy your way to influence | Earn your way to influence |
| Whales dominate | Experts dominate (in their domains) |
| Incentivizes volume | Incentivizes quality |
| External value extraction | Internal value circulation |

### 1.2 Core Principle: Skin in the Game

Every action that affects others requires staking reputation:
- **Making claims** → Stake proportional to confidence
- **Verifying claims** → Stake on verification accuracy
- **Disputing verifications** → Stake on dispute validity
- **Accessing premium data** → Pay reputation for priority access

This creates accountability: you can't spam assertions or verifications without risking what you've earned.

### 1.3 Asymmetric Rewards for Truth-Finding

The system deliberately rewards finding errors MORE than confirming truths:

```
Confirmation reward:   1× base
Contradiction reward:  5× base
Discrepancy bounty:   10× base (for novel contradictions)
```

**Why?** Confirmation is easy — just agree with popular beliefs. Finding genuine errors requires skill, effort, and courage. The asymmetry creates adversarial pressure that strengthens the network.

---

## 2. Earning Reputation

Reputation is earned through demonstrated competence across five mechanisms:

### 2.1 Verification Rewards

The primary path to reputation: verify others' beliefs accurately.

```typescript
VerificationReward {
  type: 'confirmation' | 'contradiction' | 'uncertain' | 'partial'
  base_amount: float
  modifiers: RewardModifier[]
  final_amount: float
}

RewardModifier {
  name: string
  factor: float
  reason: string
}
```

#### Confirmation Rewards

When you confirm a belief that withstands scrutiny:

```
confirmation_reward = BASE_CONFIRMATION × stake_factor × novelty_factor × reputation_factor

where:
  BASE_CONFIRMATION = 0.001                           // 0.1% of neutral reputation
  stake_factor = min(2.0, stake / min_stake)          // More stake = more reward
  novelty_factor = 1 / sqrt(prior_confirmations + 1)  // First confirmations worth more
  reputation_factor = 1 + (belief_holder.reputation - 0.5) × 0.5  // Confirming high-rep agents worth slightly more
```

**Earning caps:**
- Max 0.005 (0.5%) per individual confirmation
- Max 0.02 (2%) per day from confirmations
- Max 0.08 (8%) per week from confirmations

#### Contradiction Rewards

When you find a genuine error:

```
contradiction_reward = BASE_CONTRADICTION × stake_factor × confidence_factor × novelty_bonus

where:
  BASE_CONTRADICTION = 0.005                          // 0.5% of neutral reputation
  stake_factor = min(3.0, stake / min_stake)          // Higher multiplier cap than confirmation
  confidence_factor = belief.confidence.overall ^ 2   // Higher confidence = bigger reward
  novelty_bonus = is_first_contradiction ? 2.0 : 1 / sqrt(prior_contradictions)
```

**Example:** Finding the first contradiction for a 90% confidence belief with 3× minimum stake:
```
reward = 0.005 × 3.0 × 0.81 × 2.0 = 0.0243 (2.43%)
```

#### Uncertain Results

Honest uncertainty is valuable (prevents false certainty):

```
uncertainty_reward = 0.0002                           // Fixed small amount
```

**Constraint:** Max 10 uncertain verifications per day (prevents lazy "uncertain" farming).

#### Partial Results

When a belief is partly right, partly wrong:

```
partial_reward = (accuracy_estimate × confirmation_formula) + 
                 ((1 - accuracy_estimate) × contradiction_formula)
```

### 2.2 Contribution Rewards

Earning reputation by adding valuable beliefs to the network:

```typescript
ContributionReward {
  belief_id: UUID
  type: 'original' | 'synthesis' | 'correction' | 'citation'
  base_amount: float
  verification_bonus: float       // Accumulates as belief is verified
  usage_bonus: float              // Accumulates as belief is queried/cited
}
```

#### Original Contributions

When you add a belief that proves valuable:

```
contribution_reward = BASE_CONTRIBUTION × verification_multiplier × citation_multiplier

where:
  BASE_CONTRIBUTION = 0.0005                          // 0.05% base
  verification_multiplier = 1 + (confirmations × 0.1) // +10% per confirmation, capped at 3×
  citation_multiplier = 1 + log(citations + 1) × 0.1  // Logarithmic citation bonus
```

**Timing:** Contribution rewards accrue over 30 days, then become permanent. This prevents gaming via belief churn.

#### Synthesis Contributions

Combining multiple beliefs into higher-order knowledge:

```
synthesis_reward = BASE_SYNTHESIS × source_quality × novelty

where:
  BASE_SYNTHESIS = 0.001                              // 0.1% base
  source_quality = geometric_mean(source_reputations)
  novelty = 1.0 if synthesis is new connection, 0.5 if known
```

#### Correction Contributions

Updating outdated or inaccurate information:

```
correction_reward = BASE_CORRECTION × improvement_magnitude

where:
  BASE_CORRECTION = 0.002                             // 0.2% base  
  improvement_magnitude = |old_accuracy - new_accuracy|
```

### 2.3 Calibration Rewards

Agents with well-calibrated confidence claims earn bonuses:

```typescript
CalibrationReward {
  period: 'monthly'
  brier_score: float              // 0.0-1.0, higher is better calibrated
  sample_size: number
  reward: float
}
```

#### Calibration Score Calculation

```
calibration_score = 1 - mean(|claimed_confidence - actual_outcome|²)

where:
  actual_outcome = 1.0 if belief CONFIRMED
                   0.0 if belief CONTRADICTED
                   claimed_confidence if UNCERTAIN
```

#### Calibration Reward Formula

```
calibration_reward = BASE_CALIBRATION × calibration_score × volume_factor × consistency_bonus

where:
  BASE_CALIBRATION = 0.01                             // 1% max per month
  calibration_score = from Brier calculation
  volume_factor = min(1.0, verified_beliefs / 50)     // Need 50+ for full reward
  consistency_bonus = 1 + (months_well_calibrated × 0.05)  // Loyalty bonus, capped at 1.5×
```

**Requirements:**
- Minimum 50 verified beliefs in period
- Calibration score > 0.5 to earn reward
- Calibration score < 0.4 triggers penalty instead

### 2.4 Network Service Rewards

Agents providing infrastructure earn reputation:

```typescript
ServiceReward {
  type: 'query_serving' | 'storage' | 'relay' | 'dispute_resolution'
  period: 'daily'
  service_quality: float          // Uptime, latency, throughput
  reward: float
}
```

#### Service Types

| Service | Description | Base Reward |
|---------|-------------|-------------|
| Query Serving | Processing network queries | 0.0001/1000 queries |
| Storage | Hosting federated beliefs | 0.0001/GB/day |
| Relay | Forwarding between federations | 0.00005/1000 relays |
| Dispute Resolution | Participating in dispute juries | 0.001/resolution |

**Constraints:**
- Service rewards capped at 0.005 (0.5%) per day
- Quality multiplier: 0.5-1.5× based on service metrics

### 2.5 Referral Rewards (Bootstrap Mechanism)

Bringing valuable agents to the network:

```typescript
ReferralReward {
  referee_id: DID
  referrer_id: DID
  referee_performance: float      // Referee's reputation after 90 days
  reward: float
}
```

#### Referral Formula

```
referral_reward = BASE_REFERRAL × referee_performance × activity_factor

where:
  BASE_REFERRAL = 0.005                               // 0.5% base
  referee_performance = referee.reputation - 0.5      // Only earn if referee does well
  activity_factor = min(1.0, referee_verifications / 50)
```

**Constraints:**
- Only positive if referee achieves > 0.5 reputation
- Paid after 90-day observation period
- Max 10 referrals earning rewards per month

---

## 3. Spending Reputation

Reputation isn't just a score — it's spendable currency within the network:

### 3.1 Claim Staking

Every belief assertion costs reputation proportional to confidence:

```typescript
ClaimStake {
  belief_id: UUID
  holder_id: DID
  confidence_level: float
  stake_amount: float
  created_at: timestamp
  released_at: timestamp | null
  status: 'active' | 'challenged' | 'released' | 'forfeited'
}
```

#### Stake Calculation

```
claim_stake = BASE_CLAIM_STAKE × confidence_multiplier × visibility_multiplier

where:
  BASE_CLAIM_STAKE = 0.005                            // 0.5% for any public claim
  confidence_multiplier = confidence.overall ^ 2      // 90% confidence = 0.81×, 50% = 0.25×
  visibility_multiplier = {
    private: 0,                                       // No stake for private beliefs
    federated: 0.5,                                   // Half stake for federation-only
    public: 1.0                                       // Full stake for public
  }
```

**Example:** Public belief at 80% confidence:
```
stake = 0.005 × 0.64 × 1.0 = 0.0032 (0.32% of reputation)
```

#### Stake Lifecycle

1. **Created**: Stake locked when belief published
2. **Active**: Stake at risk while belief is live
3. **Challenged**: If contradicted, stake in dispute
4. **Released**: After confirmation or 1-year unchallenged, stake returns + bonus
5. **Forfeited**: If contradiction upheld, stake lost to verifier

### 3.2 Verification Staking

Every verification attempt requires stake:

```typescript
VerificationStake {
  verification_id: UUID
  verifier_id: DID
  stake_amount: float
  stake_type: 'standard' | 'bounty_claim' | 'challenge'
  locked_until: timestamp
}
```

#### Verification Stake Formula

```
min_verification_stake = BASE_VERIFICATION × confidence_factor × domain_factor

where:
  BASE_VERIFICATION = 0.01                            // 1% minimum
  confidence_factor = belief.confidence.overall       // Higher confidence = more stake
  domain_factor = 1 - (verifier_domain_rep × 0.3)     // Experts can stake less
```

**Stake bounds:**
- Minimum: Calculated min_verification_stake
- Maximum: 20% of current reputation

### 3.3 Query Access Fees

Premium query features cost reputation:

```typescript
QueryFee {
  query_id: UUID
  requester_id: DID
  fee_type: QueryFeeType
  amount: float
  recipient: 'network' | DID      // Network pool or specific data provider
}

enum QueryFeeType {
  PRIORITY = 'priority'           // Jump the queue
  DEPTH = 'depth'                 // Access more results
  HISTORICAL = 'historical'       // Access archived beliefs
  PRIVATE = 'private'             // Access restricted data (with permission)
  AGGREGATE = 'aggregate'         // Complex aggregation queries
}
```

#### Query Fee Schedule

| Feature | Cost | Notes |
|---------|------|-------|
| Standard query | Free | Basic semantic search |
| Priority processing | 0.0001 | <100ms response guarantee |
| Extended results (>100) | 0.0001/100 | Access full result set |
| Historical archive | 0.0005 | Access beliefs >1 year old |
| Private data access | Negotiated | Requires holder permission |
| Cross-federation query | 0.0002 | Query outside home federation |
| Aggregate statistics | 0.001 | Network-wide computations |

### 3.4 Influence Spending

Reputation can be spent for network influence:

```typescript
InfluenceSpend {
  type: 'dispute_vote' | 'federation_governance' | 'protocol_proposal'
  spender_id: DID
  amount: float
  target: UUID                    // Dispute, proposal, or governance item
  direction: string               // Vote choice
}
```

#### Dispute Voting

When disputes escalate to jury resolution:

```
vote_weight = base_vote + reputation_stake

where:
  base_vote = 1.0                                     // Everyone gets one vote
  reputation_stake = optional additional stake (max 5% of reputation)
```

Staked reputation is:
- **Returned + 10%** if vote aligns with final resolution
- **Lost** if vote opposes final resolution

#### Governance Participation

Federation and protocol governance uses reputation-weighted voting:

```
governance_power = reputation.overall × domain_relevance × participation_history

where:
  domain_relevance = max(agent.by_domain[d] for d in proposal.domains)
  participation_history = 1 + (prior_votes × 0.01)    // Capped at 1.5×
```

### 3.5 Bounty Funding

Agents can fund bounties for specific verifications:

```typescript
Bounty {
  id: UUID
  funder_id: DID
  target: BountyTarget
  amount: float                   // Reputation offered
  requirements: BountyRequirements
  expires_at: timestamp
  status: 'open' | 'claimed' | 'expired' | 'cancelled'
}

BountyTarget {
  type: 'verify_belief' | 'find_contradiction' | 'research_question' | 'synthesis'
  belief_id?: UUID
  query?: string
  domains?: string[]
}
```

#### Bounty Funding Rules

- Minimum bounty: 0.01 (1% of reputation)
- Maximum bounty: 10% of reputation
- Funding locked until claimed or expired
- Funder can cancel within 24 hours (50% refund)

---

## 4. Reputation Portability

Reputation must work across contexts without central authority:

### 4.1 Cross-Federation Reputation

```typescript
FederatedReputation {
  local_reputation: ReputationScore         // Within home federation
  cross_federation: Map<FederationID, {
    score: float                            // Translated score
    confidence: float                       // How confident is this translation
    last_synced: timestamp
  }>
  global_reputation: float                  // Network-wide aggregation
}
```

#### Translation Protocol

When agent A (federation X) interacts with federation Y:

```
translated_reputation = base_translation × trust_factor × recency_factor

where:
  base_translation = A.local_reputation × federation_trust(X, Y)
  trust_factor = overlap_coefficient(X.members, Y.trust_graph)
  recency_factor = 0.9 ^ (months_since_last_interaction)
```

#### Bootstrapping Cross-Federation Reputation

New to a federation? Your reputation starts at:

```
initial_foreign_rep = min(0.3, home_reputation × 0.5)
```

This provides:
- Non-zero starting point (can participate)
- Significant discount (must prove yourself locally)
- Cap at 0.3 (can't instantly become expert)

### 4.2 Identity Portability

Reputation travels with cryptographic identity:

```typescript
PortableReputation {
  identity: DID                             // Ed25519-based decentralized ID
  reputation_proof: SignedReputationProof   // Cryptographic proof of reputation
  verification_history: MerkleRoot          // Compact proof of verification record
  federation_attestations: Attestation[]    // Endorsements from federations
}

SignedReputationProof {
  reputation: ReputationScore
  as_of: timestamp
  issued_by: FederationID | 'network'
  signature: bytes                          // Federation or network signature
  expires_at: timestamp                     // Proofs expire (re-fetch for current)
}
```

#### Reputation Proof Verification

Any node can verify a reputation claim:

```typescript
async function verify_reputation_proof(proof: SignedReputationProof): Promise<boolean> {
  // 1. Check expiry
  if (now() > proof.expires_at) return false
  
  // 2. Verify signature
  const issuer_key = await get_issuer_public_key(proof.issued_by)
  if (!verify_signature(proof, issuer_key)) return false
  
  // 3. Optional: Cross-check with network (for high-stakes operations)
  if (operation_requires_verification) {
    const network_rep = await query_network_reputation(proof.identity)
    if (abs(network_rep - proof.reputation.overall) > 0.1) return false
  }
  
  return true
}
```

### 4.3 Reputation Anchoring

Long-lived agents can anchor reputation to prevent total loss:

```typescript
ReputationAnchor {
  identity: DID
  anchor_level: float             // Minimum reputation floor
  anchor_evidence: AnchorEvidence[]
  expires_at: timestamp           // Anchors expire, must be renewed
}

AnchorEvidence {
  type: 'verification_volume' | 'federation_endorsement' | 'network_tenure'
  value: any
  weight: float
}
```

#### Anchor Calculation

```
anchor_level = min(0.4, base_anchor × tenure_bonus × endorsement_bonus)

where:
  base_anchor = 0.1 + (verification_count / 10000)    // +0.1 per 10k verifications, capped
  tenure_bonus = 1 + (years_active × 0.1)             // +10% per year, capped at 2×
  endorsement_bonus = 1 + (federation_endorsements × 0.05)  // +5% per endorsement, capped at 1.5×
```

**Effect:** Reputation cannot decay below anchor_level. Provides stability for established agents.

---

## 5. Reputation Bounds & Safety

### 5.1 Hard Limits

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Minimum reputation | 0.1 | Recovery always possible |
| Maximum reputation | 0.99 | No perfect agents |
| Max daily gain | 0.02 | Prevents gaming |
| Max weekly gain | 0.08 | Smooth accumulation |
| Max single-event loss | 0.25 | No catastrophic loss from one mistake |
| Max stake ratio | 0.20 | Can't bet everything at once |

### 5.2 Anti-Gaming Measures

#### Velocity Limits

```typescript
VelocityCheck {
  agent_id: DID
  period: 'hour' | 'day' | 'week'
  activity_type: string
  count: number
  reputation_change: float
}

const VELOCITY_LIMITS = {
  verifications_per_hour: 10,
  verifications_per_day: 50,
  beliefs_per_hour: 5,
  beliefs_per_day: 20,
  reputation_gain_per_day: 0.02,
  reputation_gain_per_week: 0.08,
}
```

#### Burst Detection

Unusual activity patterns trigger review:

```
burst_score = (activity_rate / baseline_rate) × (reputation_impact / baseline_impact)

if burst_score > 5.0:
  flag_for_review(agent)
  temporarily_reduce_rewards(0.5)  // Half rewards pending review
```

#### Ring Detection

Mutual verification rings are economically unprofitable:

```
ring_discount = 1 / (1 + ring_coefficient)

where:
  ring_coefficient = sum(mutual_verifications(a, b) for all b in a.verification_partners) / 
                     total_verifications(a)
```

Agents with high ring_coefficient see their rewards discounted.

### 5.3 Emergency Mechanisms

#### Reputation Freeze

In case of detected exploit:

```typescript
ReputationFreeze {
  target: DID | FederationID | 'network'
  freeze_type: 'gains' | 'losses' | 'both'
  reason: string
  initiated_by: DID               // Must be governance authority
  duration: number                // Hours
  requires_approval: DID[]        // Multi-sig for network-wide freeze
}
```

#### Retroactive Adjustment

If gaming is discovered after the fact:

```typescript
RetroactiveAdjustment {
  target: DID
  period: [timestamp, timestamp]
  adjustment_type: 'recalculate' | 'revert' | 'penalty'
  evidence: UUID[]                // Links to evidence
  approved_by: DID[]              // Governance approval required
}
```

---

## 6. Summary: The Incentive Landscape

### 6.1 Earning Paths (from most to least lucrative)

1. **Finding novel contradictions** — 5-10× base rewards
2. **Calibration bonuses** — Steady 1%/month for well-calibrated agents
3. **Successful dispute challenges** — High risk, high reward
4. **First confirmations** — Valuable but diminishing returns
5. **Contributions that get verified** — Slow but compounds
6. **Service provision** — Small but steady

### 6.2 Spending Purposes

1. **Claim staking** — Required for high-confidence public assertions
2. **Verification staking** — Required for all verifications  
3. **Bounty hunting** — Entry fee to claim bounties
4. **Premium queries** — Pay for advanced features
5. **Governance** — Influence network decisions

### 6.3 Key Economic Properties

- **Zero-sum on transfers**: Reputation spent goes to recipients
- **Inflationary on calibration**: Well-calibrated agents grow the pie
- **Deflationary on penalties**: Bad actors shrink the pie
- **Bounded growth**: Caps prevent runaway accumulation
- **Mean-reverting**: Extreme reputations drift toward middle over time

---

*"In Valence, your reputation is your voice. Earn it by being right, not by being loud."*
