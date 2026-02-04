# Verification Protocol — Specification

*"Truth emerges from adversarial verification, not authoritative declaration."*

---

## Overview

The Verification Protocol enables agents to **validate, challenge, or refine** beliefs in the Valence network. Unlike centralized fact-checking, verification is:

- **Decentralized**: Anyone can verify any public belief
- **Incentive-aligned**: Verifiers stake reputation, earn rewards for accurate work
- **Adversarial**: Finding contradictions is MORE valuable than confirming
- **Composable**: Verifications themselves become evidence for future verifications

This protocol is the engine that transforms individual beliefs into collective knowledge.

---

## 1. Core Data Structures

### 1.1 Verification

```typescript
Verification {
  // === Identity ===
  id: UUID                        // UUIDv7 for time-ordering
  
  // === Participants ===
  verifier_id: DID                // Who performed this verification
  belief_id: UUID                 // What belief was verified
  holder_id: DID                  // Who holds the belief (cached for efficiency)
  
  // === Result ===
  result: VerificationResult      // CONFIRMED | CONTRADICTED | UNCERTAIN | PARTIAL
  result_details: ResultDetails   // Structured breakdown
  
  // === Evidence ===
  evidence: Evidence[]            // Supporting material
  reasoning: string               // Verifier's explanation (max 16KB)
  
  // === Economics ===
  stake: Stake                    // Reputation risked
  
  // === Metadata ===
  created_at: timestamp
  signature: bytes                // Verifier's Ed25519 signature over (id, belief_id, result, evidence_hashes, stake)
  
  // === Status ===
  status: VerificationStatus      // PENDING | ACCEPTED | DISPUTED | OVERTURNED
  dispute_id: UUID | null         // If disputed, link to dispute
}
```

### 1.2 VerificationResult

```typescript
enum VerificationResult {
  CONFIRMED = 'confirmed'         // Evidence supports the belief
  CONTRADICTED = 'contradicted'   // Evidence refutes the belief
  UNCERTAIN = 'uncertain'         // Insufficient evidence either way
  PARTIAL = 'partial'             // Partially correct, with qualifications
}
```

**Result semantics:**

| Result | Meaning | Reputation Effect |
|--------|---------|-------------------|
| CONFIRMED | "I verify this belief is accurate" | Holder ↑, Verifier ↑ |
| CONTRADICTED | "I have evidence this belief is false" | Holder ↓, Verifier ↑↑ |
| UNCERTAIN | "I cannot verify this either way" | Neutral, small Verifier ↑ |
| PARTIAL | "This is partly right, partly wrong" | Depends on details |

### 1.3 ResultDetails

For nuanced verification outcomes:

```typescript
ResultDetails {
  // For CONFIRMED
  confirmation_strength?: 'strong' | 'moderate' | 'weak'
  confirmed_aspects?: string[]    // Which parts are confirmed
  
  // For CONTRADICTED
  contradiction_type?: ContradictionType
  corrected_belief?: string       // Suggested correction
  severity?: 'minor' | 'moderate' | 'major' | 'critical'
  
  // For PARTIAL
  accurate_portions?: string[]    // What's right
  inaccurate_portions?: string[]  // What's wrong
  accuracy_estimate?: float       // 0.0-1.0, how much is correct
  
  // For UNCERTAIN
  uncertainty_reason?: UncertaintyReason
  additional_evidence_needed?: string[]
  
  // Universal
  confidence_assessment?: {       // Verifier's assessment of belief's confidence vector
    dimension: string
    claimed: float
    actual: float
    justification: string
  }[]
}

enum ContradictionType {
  FACTUALLY_FALSE = 'factually_false'       // Claim is objectively wrong
  OUTDATED = 'outdated'                      // Was true, no longer is
  MISATTRIBUTED = 'misattributed'           // Wrong source/origin
  OVERSTATED = 'overstated'                 // True but exaggerated
  MISSING_CONTEXT = 'missing_context'       // Misleading without context
  LOGICAL_ERROR = 'logical_error'           // Derivation is flawed
}

enum UncertaintyReason {
  INSUFFICIENT_EVIDENCE = 'insufficient_evidence'
  CONFLICTING_SOURCES = 'conflicting_sources'
  OUTSIDE_EXPERTISE = 'outside_expertise'
  UNFALSIFIABLE = 'unfalsifiable'
  REQUIRES_EXPERIMENT = 'requires_experiment'
}
```

---

## 2. Evidence Model

### 2.1 Evidence Structure

```typescript
Evidence {
  id: UUID
  type: EvidenceType
  
  // Content (exactly one of these populated)
  belief_reference?: BeliefReference   // Another belief as evidence
  external_source?: ExternalSource     // URL, citation, etc.
  observation?: Observation            // Direct observation claim
  derivation?: DerivationProof         // Logical/mathematical proof
  
  // Metadata
  relevance: float                     // 0.0-1.0, how relevant to the claim
  contribution: EvidenceContribution   // How this evidence contributes
  verifier_notes?: string              // Explanation of this evidence
}

enum EvidenceType {
  BELIEF = 'belief'                    // Reference to another Valence belief
  EXTERNAL = 'external'                // External source (URL, paper, etc.)
  OBSERVATION = 'observation'          // Verifier's direct observation
  DERIVATION = 'derivation'            // Logical/mathematical proof
  TESTIMONY = 'testimony'              // Statement from another agent
}

enum EvidenceContribution {
  SUPPORTS = 'supports'                // Evidence for the belief
  CONTRADICTS = 'contradicts'          // Evidence against the belief
  CONTEXT = 'context'                  // Adds relevant context
  QUALIFIES = 'qualifies'              // Adds conditions/limitations
}
```

### 2.2 Evidence Type Details

#### Belief Reference
```typescript
BeliefReference {
  belief_id: UUID
  holder_id: DID                       // For cross-network lookups
  content_hash: string                 // SHA-256, for integrity
  confidence_at_time: ConfidenceVector // Snapshot when referenced
}
```

#### External Source
```typescript
ExternalSource {
  url?: string                         // Web resource
  doi?: string                         // Academic paper
  isbn?: string                        // Book
  citation?: string                    // Formatted citation
  archive_hash?: string                // Content hash of archived copy
  archived_at?: timestamp              // When archival snapshot was taken
  source_reputation?: float            // Verifier's assessment of source quality
}
```

#### Observation
```typescript
Observation {
  description: string                  // What was observed
  timestamp: timestamp                 // When observation occurred
  method: string                       // How observation was made
  reproducible: boolean                // Can others replicate?
  reproduction_instructions?: string   // How to reproduce
}
```

#### Derivation Proof
```typescript
DerivationProof {
  premises: UUID[]                     // Belief IDs of premises
  logic_type: 'deductive' | 'inductive' | 'abductive'
  proof_steps: string[]                // Step-by-step reasoning
  formal_notation?: string             // Optional formal logic representation
}
```

### 2.3 Evidence Requirements

**Minimum evidence by result type:**

| Result | Minimum Evidence | Quality Bar |
|--------|------------------|-------------|
| CONFIRMED | 1 supporting | Must strengthen confidence |
| CONTRADICTED | 1 contradicting | Must provide counter-evidence |
| UNCERTAIN | 0 | Must explain why uncertain |
| PARTIAL | 1 supporting + 1 contradicting | Must show both sides |

**Evidence quality factors:**
- Independence from the original belief's derivation
- Source reputation (for external sources)
- Confidence level (for belief references)
- Recency (fresher evidence weighted higher)
- Relevance to the specific claim

**Invalid evidence (rejected):**
- Circular reference (belief citing itself)
- Self-corroboration (verifier's own belief as sole evidence for CONFIRMED)
- Unverifiable claims (observation without reproduction path for major claims)
- Broken references (belief IDs that don't exist)

---

## 3. Stake Mechanics

### 3.1 Stake Structure

```typescript
Stake {
  amount: float                        // Reputation points at risk
  type: StakeType
  locked_until: timestamp              // When stake can be released
  escrow_id: UUID                      // Reference to escrow record
}

enum StakeType {
  STANDARD = 'standard'                // Normal verification stake
  BOUNTY = 'bounty'                    // Claiming a discrepancy bounty
  CHALLENGE = 'challenge'              // Challenging existing verification
}
```

### 3.2 Stake Calculation

**Minimum stake** (required to submit verification):

```
min_stake = base_stake × confidence_multiplier × domain_multiplier

where:
  base_stake = 0.01                    (1% of neutral reputation)
  confidence_multiplier = belief.confidence.overall
  domain_multiplier = 1.0 + (verifier_domain_reputation × 0.5)
```

**Rationale:**
- Higher-confidence beliefs require more stake to challenge
- Domain experts can verify with relatively less risk (they know what they're doing)
- Prevents spam verifications

**Maximum stake:**
- Capped at 20% of verifier's current overall reputation
- Prevents catastrophic loss from single verification

**Stake lockup period:**
- Standard: 7 days
- Bounty claims: 14 days (allows dispute period)
- Challenge: Until dispute resolution

### 3.3 Stake Outcomes

**Verification ACCEPTED (no dispute):**
| Result | Verifier Stake | Holder Effect |
|--------|----------------|---------------|
| CONFIRMED | Returned + bonus | Reputation boost |
| CONTRADICTED | Returned + 2× bonus | Reputation penalty |
| UNCERTAIN | Returned (no bonus) | No effect |
| PARTIAL | Returned + partial bonus | Proportional effect |

**Verification DISPUTED and OVERTURNED:**
- Verifier loses staked amount
- Stake transferred to disputer
- Verifier reputation penalized (beyond stake)

**Verification DISPUTED and UPHELD:**
- Verifier keeps stake + dispute bonus
- Disputer loses their challenge stake

### 3.4 Discrepancy Bounties

High-confidence beliefs automatically offer bounties for finding contradictions:

```
bounty = base_bounty × confidence_premium × age_factor

where:
  base_bounty = belief.stake_at_creation × 0.5
  confidence_premium = belief.confidence.overall ^ 2
  age_factor = min(1.0, days_since_creation / 30)  // Max at 30 days
```

**Why bounties?**
- Incentivizes adversarial verification (truth-seeking)
- High-confidence beliefs are "putting their money where their mouth is"
- Natural market for fact-checking

---

## 4. Verification Lifecycle

### 4.1 State Machine

```
                    ┌─────────────┐
                    │   PENDING   │ ◄── Verification submitted
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        [valid evidence]  [invalid]  [timeout]
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ ACCEPTED │  │ REJECTED │  │ EXPIRED  │
        └────┬─────┘  └──────────┘  └──────────┘
             │
        [disputed]
             │
             ▼
        ┌──────────┐
        │ DISPUTED │
        └────┬─────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
[resolved: upheld] [resolved: overturned]
    │                 │
    ▼                 ▼
┌──────────┐     ┌───────────┐
│ ACCEPTED │     │ OVERTURNED│
└──────────┘     └───────────┘
```

### 4.2 Timing Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Validation window | 1 hour | Time to check evidence validity |
| Acceptance delay | 24 hours | Allows holder to notice |
| Dispute window | 7 days | Time to challenge accepted verification |
| Resolution timeout | 14 days | Max time for dispute resolution |
| Stake lockup | 7-14 days | Prevents hit-and-run verifications |

### 4.3 Validation Checks

Before a verification is accepted:

1. **Identity check**: Verifier DID is valid and not revoked
2. **Signature check**: Verification is properly signed
3. **Stake check**: Verifier has sufficient reputation, stake is locked
4. **Evidence check**: Minimum evidence requirements met
5. **Conflict check**: No existing verification by same verifier for same belief
6. **Self-verification check**: Verifier ≠ holder
7. **Freshness check**: Belief is still active (not superseded)

---

## 5. Verification Constraints

### 5.1 Who Can Verify

**Anyone can verify public beliefs** — this is fundamental to decentralization.

**Restrictions:**
- Cannot verify your own beliefs
- Cannot verify beliefs you derived (too close to source)
- Cannot verify while under verification suspension (after multiple overturned verifications)
- Cannot verify the same belief twice (but can submit new evidence via dispute)

### 5.2 What Can Be Verified

**Verifiable:**
- PUBLIC visibility beliefs
- FEDERATED visibility beliefs (by federation members)

**Not verifiable:**
- PRIVATE beliefs (owner only)
- Expired beliefs (past valid_until)
- Tombstoned beliefs

### 5.3 Rate Limits

To prevent gaming and spam:

| Limit | Value | Scope |
|-------|-------|-------|
| Verifications per day | 50 | Per verifier |
| Pending verifications | 10 | Per verifier |
| Verifications per belief | 100 | Total |
| Verification velocity | 5 per hour | Per belief |

### 5.4 Conflict Resolution

When verifications conflict (e.g., one CONFIRMED, another CONTRADICTED):

1. Both remain valid records
2. Net effect computed from reputation-weighted consensus:
   ```
   net_result = Σ(verifier_reputation × verification_weight) / Σ(verifier_reputation)
   ```
3. Conflicting verifications trigger automatic review queue
4. High-value conflicts may spawn formal disputes

---

## 6. Special Verification Types

### 6.1 Expert Verification

Verifiers with high domain reputation can perform **expert verifications**:

```typescript
ExpertVerification extends Verification {
  expert_domains: string[]           // Domains of expertise claimed
  expert_confidence: float           // Additional confidence from expertise
  peer_review_required: boolean      // For very high-stakes claims
}
```

**Benefits:**
- Higher weight in consensus calculations
- Lower minimum stake requirements
- Can verify faster (shorter acceptance delay)

**Requirements:**
- Domain reputation > 0.7 in claimed domains
- Verification count > 100 in domain
- No recent overturned verifications in domain

### 6.2 Batch Verification

For verifying multiple related beliefs efficiently:

```typescript
BatchVerification {
  id: UUID
  verifier_id: DID
  belief_ids: UUID[]                 // Related beliefs
  common_result: VerificationResult
  common_evidence: Evidence[]        // Shared evidence
  per_belief_notes?: Map<UUID, string>
  stake: Stake                       // Single stake for batch
}
```

**Constraints:**
- Max 20 beliefs per batch
- All beliefs must share at least one domain
- Common evidence must apply to all beliefs

### 6.3 Cascade Verification

When a belief's sources are verified, effects cascade:

```
Belief A (derived from B, C)
    │
    ├── B verified as CONFIRMED
    │   └── A's confidence.source_reliability ↑
    │
    └── C verified as CONTRADICTED
        └── A flagged for re-evaluation
```

**Cascade rules:**
- CONFIRMED sources boost derived belief confidence
- CONTRADICTED sources flag derived beliefs
- Multiple contradicted sources may auto-invalidate derivatives
- Cascade depth limited to 3 levels (prevent runaway effects)

---

## 7. Anti-Gaming Measures

### 7.1 Sybil Resistance

Multiple fake identities gaming verifications:

**Mitigations:**
- Stake requirements (creating identities with reputation is expensive)
- Independence detection (similar evidence patterns flagged)
- Velocity limits (many verifications from new identities trigger review)
- Trust graph analysis (isolated nodes weighted less)

### 7.2 Collusion Resistance

Groups coordinating to manipulate verifications:

**Mitigations:**
- Independence requirements (evidence can't trace to same source)
- Statistical outlier detection (unusual agreement patterns)
- Temporal analysis (coordinated timing flagged)
- Reputation diversity requirements (consensus needs diverse verifiers)

### 7.3 Griefing Resistance

Malicious verifications to harm reputations:

**Mitigations:**
- Stake requirements (griefing is expensive)
- Dispute mechanism (targets can challenge)
- Reputation recovery (single bad verification doesn't destroy reputation)
- Pattern detection (serial contradictors flagged for review)

---

## 8. Integration Points

### 8.1 Belief Schema Integration

Verifications update belief confidence vectors:

```
verification(CONFIRMED) →
  belief.confidence.corroboration ↑
  belief.confidence.source_reliability ↑ (slightly)

verification(CONTRADICTED) →
  belief.confidence.internal_consistency ↓
  belief.confidence.corroboration ↓
  (may trigger supersession)
```

### 8.2 Identity Integration

Verifications require and update identity:

```
submission →
  verifier.stake_at_risk ↑
  
resolution →
  verifier.verification_count ↑
  verifier.reputation updated
  verifier.by_domain[domain] updated
  
discrepancy found →
  verifier.discrepancy_finds ↑
```

### 8.3 Reputation Integration

See REPUTATION.md for detailed reputation update mechanics.

### 8.4 Trust Graph Integration

Verifications can influence trust:

```
A verifies B's belief (CONFIRMED, high quality) →
  B may increase trust in A (optional, user-controlled)

A verifies B's belief (CONTRADICTED, upheld) →
  B may decrease trust in A
  Network may decrease trust in B
```

---

## 9. Design Rationale

### Why stake?
Without skin in the game, verifications are cheap talk. Stake creates accountability and filters for serious verifiers.

### Why bounties for contradictions?
Confirming is easy (just agree). Finding errors is hard and valuable. Economic incentives should match social value.

### Why not voting?
Voting is popularity, not truth. Verification with evidence and stake creates accountability that voting lacks.

### Why allow conflicting verifications?
Reality is messy. Multiple viewpoints with evidence are more valuable than forced consensus. Let consumers weight by reputation.

### Why time delays?
Immediate acceptance enables hit-and-run attacks. Delays allow observation and dispute while keeping the system responsive.

---

## 10. Future Extensions

### 10.1 Prediction Markets Integration
Link verifications to prediction market outcomes for beliefs about future events.

### 10.2 Formal Verification
For mathematical/logical claims, integrate automated proof checkers.

### 10.3 Cross-Network Verification
Verify beliefs from other epistemic networks (Wikipedia, academic databases) with appropriate trust mapping.

### 10.4 AI-Assisted Verification
LLM agents as verifiers with appropriate stake and accountability mechanisms.

---

*"Verification is not about authority. It's about evidence, stake, and accountability."*
