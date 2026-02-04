# Verification Protocol — Reputation Updates

*"Reputation is the accumulated trust from demonstrated competence."*

---

## Overview

This document specifies how verification activities affect reputation for all participants. The reputation system is designed to:

- **Reward accurate verification** — Good verifiers gain reputation
- **Penalize inaccurate verification** — Bad verifiers lose reputation
- **Incentivize truth-finding** — Discrepancies earn more than confirmations
- **Encourage calibration** — Match confidence claims to reality
- **Resist manipulation** — Sybil attacks and collusion are economically unprofitable

---

## 1. Reputation Model Recap

From the Identity spec, each agent has:

```typescript
ReputationScore {
  overall: float                  // 0.0-1.0, global reputation
  by_domain: Map<string, float>   // Domain-specific scores
  verification_count: uint64      // Total verifications performed
  discrepancy_finds: uint64       // Contradictions found and upheld
  stake_at_risk: float            // Currently staked reputation
}
```

**Initial state**: `overall: 0.5` (neutral starting point)

---

## 2. Verification Outcome Effects

### 2.1 Effect Matrix

When a verification is finalized (accepted and undisputed, or dispute resolved):

| Result | Verifier | Holder | Notes |
|--------|----------|--------|-------|
| CONFIRMED | ↑ small | ↑ small | Both rewarded for accurate belief |
| CONTRADICTED | ↑↑ large | ↓ proportional | Verifier rewarded, holder penalized |
| UNCERTAIN | ↑ tiny | — neutral | Honest uncertainty is valuable |
| PARTIAL | ↑ moderate | ↑↓ mixed | Proportional to accuracy split |

### 2.2 Verifier Reputation Updates

#### CONFIRMED Verification

```
verifier.overall += confirmation_reward(stake, belief_confidence, verifier_reputation)

confirmation_reward = base_reward × stake_multiplier × confidence_factor × diminishing_factor

where:
  base_reward = 0.001                    // 0.1% base
  stake_multiplier = stake / min_stake   // More stake = more reward (capped at 2×)
  confidence_factor = belief.confidence.overall  // Confirming high-confidence earns less
  diminishing_factor = 1 / sqrt(existing_confirmations + 1)  // First confirmations worth more
```

**Rationale**: Confirming already-confirmed beliefs provides decreasing marginal value. First verifiers take more risk, earn more reward.

#### CONTRADICTED Verification

```
verifier.overall += contradiction_reward(stake, belief_confidence, is_first_contradiction)

contradiction_reward = base_bounty × stake_multiplier × confidence_premium × novelty_bonus

where:
  base_bounty = 0.005                    // 0.5% base (5× confirmation)
  stake_multiplier = stake / min_stake   // Capped at 3×
  confidence_premium = belief.confidence.overall ^ 2  // High-confidence contradictions earn more
  novelty_bonus = is_first_contradiction ? 2.0 : 1.0 / sqrt(existing_contradictions)
```

**Rationale**: Finding errors is harder and more valuable. High-confidence false beliefs are dangerous; finding them is rewarded proportionally.

#### UNCERTAIN Verification

```
verifier.overall += uncertainty_reward

uncertainty_reward = 0.0002              // Tiny fixed reward
```

**Rationale**: Honest uncertainty prevents false certainty. Small reward acknowledges work without incentivizing "lazy uncertain."

#### PARTIAL Verification

```
verifier.overall += partial_reward(accuracy_estimate, stake)

partial_reward = confirmation_reward × accuracy_estimate + 
                 contradiction_reward × (1 - accuracy_estimate)
```

**Rationale**: Proportional credit based on what was confirmed vs contradicted.

### 2.3 Holder Reputation Updates

#### When CONFIRMED

```
holder.overall += holder_confirmation_bonus(verification_quality)

holder_confirmation_bonus = 0.0005 × verifier.overall × sqrt(stake / min_stake)
```

**Rationale**: Confirmation from reputable verifiers with high stake means more. Your beliefs being validated builds reputation.

#### When CONTRADICTED

```
holder.overall -= holder_contradiction_penalty(belief_confidence, verifier_reputation)

holder_contradiction_penalty = base_penalty × overconfidence_multiplier × verifier_weight

where:
  base_penalty = 0.003                   // 0.3% base
  overconfidence_multiplier = belief.confidence.overall ^ 2  // Claimed high confidence = bigger penalty
  verifier_weight = verifier.overall     // High-rep verifier = more credible contradiction
```

**Rationale**: The penalty scales with how wrong you were. Claiming 90% confidence on something false hurts more than 50% confidence.

#### When UNCERTAIN

No change. Unverifiable beliefs aren't penalized until contradicted.

#### When PARTIAL

```
holder.overall += holder_partial_adjustment(accuracy_estimate)

holder_partial_adjustment = 
  confirmation_bonus × accuracy_estimate - 
  contradiction_penalty × (1 - accuracy_estimate)
```

---

## 3. Discrepancy Bounties

### 3.1 Bounty Structure

Every public belief with confidence > 0.5 automatically offers a bounty for finding contradictions:

```typescript
DiscrepancyBounty {
  belief_id: UUID
  base_amount: float              // From holder's staked reputation
  confidence_premium: float       // Bonus for high-confidence claims
  age_factor: float               // Increases over time (up to cap)
  total_bounty: float             // Sum available
  funded_by: DID                  // Belief holder
  expires_at: timestamp | null    // When bounty expires
}
```

### 3.2 Bounty Calculation

```
total_bounty = holder_stake × confidence_premium × age_factor × domain_multiplier

where:
  holder_stake = belief.holder.stake_at_creation (or min_stake if unstaked)
  confidence_premium = belief.confidence.overall ^ 2
  age_factor = min(2.0, 1.0 + days_since_creation / 30)
  domain_multiplier = importance_weight(belief.domains)
```

**Example**:
- Belief with 80% confidence, created 60 days ago, holder staked 2%
- `bounty = 0.02 × 0.64 × 2.0 × 1.0 = 0.0256` (2.56% reputation reward)

### 3.3 Bounty Claiming

To claim a discrepancy bounty:

1. Submit CONTRADICTED verification with valid evidence
2. Pass validation and acceptance period
3. Survive dispute window (or win dispute)
4. Bounty transfers from holder's escrowed stake to verifier

**First finder bonus**: First valid contradiction gets 2× the bounty. Subsequent verifiers split remaining pool.

### 3.4 Bounty Funding

Holders implicitly fund bounties through confidence claims:

```
belief_creation →
  stake_required = base_stake × confidence.overall
  holder.stake_at_risk += stake_required
  bounty_pool += stake_required × 0.5  // Half of stake backs bounties
```

**Rationale**: High-confidence claims are "bets" on correctness. If wrong, the bounty incentivizes someone to prove it.

---

## 4. Calibration Rewards

### 4.1 What is Calibration?

Calibration measures how well confidence claims match reality:
- A well-calibrated agent's 80% confidence beliefs are true ~80% of the time
- Overconfidence: claiming 90%, being right 60%
- Underconfidence: claiming 50%, being right 90%

### 4.2 Calibration Score Calculation

```typescript
CalibrationScore {
  overall: float                  // 0.0-1.0, Brier score derived
  by_confidence_bucket: {
    bucket: [float, float]        // e.g., [0.8, 0.9]
    claims: number                // Beliefs in this range
    accuracy: float               // Fraction confirmed
    calibration_error: float      // |claimed - actual|
  }[]
  sample_size: number
  last_updated: timestamp
}
```

**Brier-derived score**:
```
calibration_score = 1 - mean(|claimed_confidence - actual_outcome|²)

where:
  actual_outcome = 1.0 if CONFIRMED, 0.0 if CONTRADICTED, 0.5 if UNCERTAIN
```

### 4.3 Calibration Rewards

Agents with good calibration receive bonuses:

```
calibration_bonus = base_bonus × calibration_score × verification_volume

where:
  base_bonus = 0.01               // 1% per period
  calibration_score = from CalibrationScore
  verification_volume = min(1.0, verified_beliefs / 100)  // Need volume for stats
```

**Frequency**: Calculated and distributed monthly.

**Minimum sample**: 50 verified beliefs required for calibration bonus eligibility.

### 4.4 Calibration Penalties

Severely miscalibrated agents face penalties:

```
if calibration_score < 0.4:
  holder.overall -= miscalibration_penalty
  
miscalibration_penalty = 0.005 × (0.4 - calibration_score) × belief_count / 100
```

**Rationale**: Systematic overconfidence or underconfidence degrades network quality. Small penalty encourages improvement.

---

## 5. Dispute Resolution Effects

### 5.1 Dispute Upheld (Original Verification Stands)

| Party | Effect |
|-------|--------|
| Original Verifier | Keeps stake + dispute_defense_bonus |
| Disputer | Loses stake (transferred to verifier) |
| Holder | No change (already processed) |

```
dispute_defense_bonus = disputer_stake × 0.8  // 80% of disputer's stake
```

### 5.2 Dispute Overturned (Original Verification Wrong)

| Party | Effect |
|-------|--------|
| Original Verifier | Loses stake + penalty |
| Disputer | Gets verifier's stake + bonus |
| Holder | Reputation restored (if CONTRADICTED → CONFIRMED) |

```
verifier_penalty = stake + malicious_bonus_if_applicable

malicious_bonus_if_applicable = 
  fabricated_evidence ? stake × 2.0 :
  gross_negligence ? stake × 1.0 :
  0.0

disputer_reward = verifier_stake × 0.8 + bounty_if_applicable
```

### 5.3 Dispute Modified (Partial Correction)

| Party | Effect |
|-------|--------|
| Original Verifier | Keeps partial stake, partial penalty |
| Disputer | Gets partial reward |
| Holder | Proportional adjustment |

```
modification_ratio = |old_result - new_result| / |old_result - extreme|

verifier_keeps = stake × (1 - modification_ratio)
disputer_gets = verifier_stake × modification_ratio × 0.8
```

### 5.4 Dispute Dismissed (Frivolous)

| Party | Effect |
|-------|--------|
| Original Verifier | Keeps stake + harassment_compensation |
| Disputer | Loses stake + frivolous_penalty |
| Holder | No change |

```
harassment_compensation = disputer_stake × 0.5
frivolous_penalty = disputer_stake × 0.2  // Additional reputation hit
```

**Rationale**: Frivolous disputes waste resources and should be discouraged.

---

## 6. Domain Reputation

### 6.1 Domain-Specific Tracking

Reputation is tracked per domain:

```typescript
by_domain: Map<string, DomainReputation>

DomainReputation {
  score: float                    // 0.0-1.0
  verification_count: number
  accuracy_rate: float
  discrepancy_finds: number
  last_active: timestamp
}
```

### 6.2 Domain Reputation Updates

Domain reputation updates mirror overall reputation but scoped:

```
domain_update = overall_update × domain_relevance

where:
  domain_relevance = 1.0 if belief.domains includes domain
                   = 0.5 if belief.domains includes parent domain
                   = 0.0 otherwise
```

**Example**: Verifying a belief tagged `tech/ai/llm` updates:
- `tech/ai/llm` at 1.0×
- `tech/ai` at 0.5×
- `tech` at 0.25×

### 6.3 Expert Status

Agents with high domain reputation gain expert status:

```typescript
ExpertStatus {
  domain: string
  level: 'novice' | 'competent' | 'proficient' | 'expert' | 'master'
  threshold: {
    score: float
    verification_count: number
    accuracy_rate: float
  }
}
```

| Level | Score | Verifications | Accuracy |
|-------|-------|---------------|----------|
| Novice | 0.3 | 5 | 0.5 |
| Competent | 0.5 | 20 | 0.6 |
| Proficient | 0.65 | 50 | 0.7 |
| Expert | 0.8 | 100 | 0.8 |
| Master | 0.9 | 250 | 0.9 |

**Expert benefits**:
- Lower minimum stake requirements
- Higher weight in dispute resolution
- Faster verification acceptance
- Expert verification badge

---

## 7. Collusion Protection

### 7.1 Threat Model

**Collusion attacks**:
1. **Mutual confirmation**: Agents confirm each other's false beliefs
2. **Coordinated contradictions**: Gang up to contradict truthful beliefs
3. **Verification rings**: Circulate reputation through fake verifications
4. **Bounty farming**: Create false beliefs then "find" contradictions

### 7.2 Detection Mechanisms

#### Graph Analysis

```typescript
CollusionSignal {
  agents: DID[]
  signal_type: 'mutual_verification' | 'timing_correlation' | 'evidence_sharing' | 'dispute_coordination'
  strength: float                 // 0.0-1.0, how suspicious
  evidence: CollusionEvidence[]
}
```

**Signals**:
- Agents who exclusively verify each other
- Verifications submitted within short time windows
- Same evidence reused across multiple verifications
- Coordinated disputes always against same targets

#### Statistical Outliers

```
outlier_score = |actual_confirmation_rate - expected_confirmation_rate| / σ

if outlier_score > 3.0:
  flag_for_review(agent)
```

**Expected rates** (from network baseline):
- Confirmation rate: ~70% for random beliefs
- Contradiction rate: ~10% for random beliefs
- Dispute rate: ~5% for random verifications

### 7.3 Mitigation Mechanisms

#### Independence Requirements

Verifications only count toward corroboration if:

```
is_independent(v1, v2) = 
  v1.verifier ≠ v2.verifier AND
  trust_distance(v1.verifier, v2.verifier) > 2 AND
  evidence_overlap(v1.evidence, v2.evidence) < 0.5 AND
  time_gap(v1.created_at, v2.created_at) > 1 hour
```

#### Reputation Weighting

Consensus calculations weight by diversity:

```
effective_weight(verification) = 
  verifier.reputation × diversity_factor

diversity_factor = 1 / (1 + sum(colluder_overlap for prior verifications))
```

#### Velocity Limits

Prevent rapid reputation accumulation:

```
max_daily_reputation_gain = 0.02  // 2% per day
max_weekly_reputation_gain = 0.08 // 8% per week
```

Excess gains are escrowed and released slowly.

### 7.4 Penalties for Detected Collusion

| Severity | Evidence | Penalty |
|----------|----------|---------|
| Suspected | Statistical anomaly | Verifications weighted 0.5× |
| Probable | Pattern + circumstantial | Verifications frozen pending review |
| Confirmed | Direct evidence | All participants lose 50% reputation |
| Egregious | Organized ring | Permanent reputation reset to 0.1 |

**Appeal process**: Flagged agents can dispute collusion findings through normal dispute mechanism.

---

## 8. Reputation Decay

### 8.1 Inactivity Decay

Reputation decays without ongoing participation:

```
decay_per_month = 0.02 × (1 - activity_score)

activity_score = min(1.0, verifications_this_month / 10)
```

**Rationale**: Reputation represents current ability, not historical achievement. Inactive agents may have stale knowledge.

### 8.2 Time-Based Decay

Old verifications contribute less to current reputation:

```
verification_weight = base_weight × time_decay

time_decay = 0.5 ^ (months_since_verification / 12)  // Half-life of 1 year
```

**Rationale**: Recent performance is more predictive than ancient history.

### 8.3 Decay Floor

Reputation never decays below 0.1 (minimum floor):

```
reputation = max(0.1, reputation - decay)
```

**Rationale**: Complete loss makes recovery impossible. Floor allows comeback.

---

## 9. Reputation Recovery

### 9.1 After Penalties

Agents who lost reputation can recover through:

1. **High-quality verifications**: Successful verifications rebuild reputation
2. **Time**: Some penalties fade over time
3. **Dispute wins**: Successfully defending against false disputes
4. **Calibration improvement**: Better confidence claims

### 9.2 Recovery Rates

```
max_recovery_per_month = 0.03  // 3%
recovery_rate = base_rate × quality_multiplier

base_rate = 0.01
quality_multiplier = accuracy_rate ^ 2  // High accuracy = faster recovery
```

### 9.3 Probation Period

After major penalty (>20% loss):

```typescript
Probation {
  started_at: timestamp
  duration: 90 days
  restrictions: [
    'max_stake_0.01',
    'no_expert_status',
    'no_dispute_filing',
    'verifications_weighted_0.5'
  ]
}
```

---

## 10. Reputation Formulas Summary

### 10.1 Quick Reference

| Event | Verifier Δ | Holder Δ |
|-------|------------|----------|
| Confirmation | +0.001 × modifiers | +0.0005 × modifiers |
| Contradiction | +0.005 × modifiers | -0.003 × modifiers |
| Uncertain | +0.0002 | — |
| Dispute won (verifier) | +0.8 × disputer_stake | — |
| Dispute lost (verifier) | -stake - penalty | restored |
| Collusion detected | -50% | -50% |
| Monthly calibration | +0.01 × score | — |
| Monthly decay | -0.02 × inactivity | -0.02 × inactivity |

### 10.2 Constants

```typescript
const REPUTATION_CONSTANTS = {
  // Base rewards
  CONFIRMATION_BASE: 0.001,
  CONTRADICTION_BASE: 0.005,
  UNCERTAINTY_BASE: 0.0002,
  
  // Bounties
  BOUNTY_MULTIPLIER: 0.5,
  FIRST_FINDER_BONUS: 2.0,
  
  // Penalties
  CONTRADICTION_PENALTY_BASE: 0.003,
  FRIVOLOUS_DISPUTE_PENALTY: 0.2,
  COLLUSION_PENALTY: 0.5,
  
  // Calibration
  CALIBRATION_BONUS_BASE: 0.01,
  CALIBRATION_PENALTY_THRESHOLD: 0.4,
  
  // Decay
  MONTHLY_INACTIVITY_DECAY: 0.02,
  VERIFICATION_HALF_LIFE_MONTHS: 12,
  
  // Limits
  MAX_DAILY_GAIN: 0.02,
  MAX_WEEKLY_GAIN: 0.08,
  REPUTATION_FLOOR: 0.1,
  MAX_STAKE_RATIO: 0.2,
  
  // Recovery
  MAX_MONTHLY_RECOVERY: 0.03,
  PROBATION_DURATION_DAYS: 90,
}
```

---

## 11. Implementation Notes

### 11.1 Atomic Updates

Reputation updates must be atomic:

```typescript
async function update_reputation(updates: ReputationUpdate[]): Promise<void> {
  await transaction(async (tx) => {
    for (const update of updates) {
      const current = await tx.get_reputation(update.identity)
      const new_score = apply_update(current, update)
      const bounded = apply_limits(new_score)
      await tx.set_reputation(update.identity, bounded)
      await tx.log_reputation_event(update)
    }
  })
}
```

### 11.2 Audit Trail

All reputation changes logged:

```typescript
ReputationEvent {
  id: UUID
  identity: DID
  event_type: string
  delta: float
  old_value: float
  new_value: float
  source: {
    verification_id?: UUID
    dispute_id?: UUID
    system_event?: string
  }
  timestamp: timestamp
}
```

### 11.3 Recalculation

Periodic full recalculation (monthly) to correct drift:

```typescript
async function recalculate_all_reputations(): Promise<void> {
  for (const agent of await get_all_agents()) {
    const events = await get_reputation_events(agent.id, { since: epoch })
    const calculated = replay_events(events)
    await update_reputation(agent.id, calculated)
  }
}
```

---

*"Your reputation is the shadow cast by your actions. It follows you everywhere."*
