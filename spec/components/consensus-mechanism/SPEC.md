# Consensus Mechanism — Specification

*"Truth emerges from independent convergence, not coordinated agreement."*

---

## Overview

The Consensus Mechanism governs how beliefs **elevate through trust layers** in the Valence network—from personal conviction to communal knowledge. Unlike traditional consensus protocols (which agree on transaction ordering), Valence consensus determines **epistemic status**: when is something "known" vs. merely "believed"?

**Core principle**: Communal knowledge requires **independent corroboration**—multiple agents reaching the same conclusion through different paths. This is how science works, how courts work, and how trustworthy knowledge has always emerged.

---

## 1. Trust Layer Model

### 1.1 The Four Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│  L4: COMMUNAL CONSENSUS                                             │
│      "The network has independently verified X"                     │
│      Requirements: Cross-federation corroboration, byzantine quorum │
│      Trust: Highest (default accept unless contradicted)            │
├─────────────────────────────────────────────────────────────────────┤
│  L3: DOMAIN KNOWLEDGE                                               │
│      "Experts in domain Y agree on X"                               │
│      Requirements: Expert verification, domain reputation threshold │
│      Trust: High within domain context                              │
├─────────────────────────────────────────────────────────────────────┤
│  L2: FEDERATED KNOWLEDGE                                            │
│      "My trusted federation believes X"                             │
│      Requirements: Federation aggregation threshold                 │
│      Trust: Medium (within federation trust boundary)               │
├─────────────────────────────────────────────────────────────────────┤
│  L1: PERSONAL BELIEF                                                │
│      "I believe X with confidence C"                                │
│      Requirements: None (self-attested)                             │
│      Trust: Per-relationship (trust graph dependent)                │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Layer Semantics

| Layer | Who Believes | Evidence Required | Can Be Wrong? | Revision Cost |
|-------|--------------|-------------------|---------------|---------------|
| L1 | Individual agent | Self-assessment | Yes (personal risk) | Low |
| L2 | Federation aggregate | k contributors | Yes (federation risk) | Medium |
| L3 | Domain experts | Expert verification | Rarely (domain reputation) | High |
| L4 | Network consensus | Independent corroboration | Possible but costly | Very High |

### 1.3 Layer Visibility

- **L1**: Visible only to holder + explicit shares
- **L2**: Visible to federation members
- **L3**: Visible to anyone (public domain knowledge)
- **L4**: Visible to everyone (communal truth)

---

## 2. Corroboration Model

### 2.1 What Is Corroboration?

Corroboration is **independent verification**—reaching the same conclusion through different evidentiary paths. This is distinct from:

- **Agreement**: Same conclusion (may be from same source)
- **Voting**: Popularity contest (no evidence required)
- **Replication**: Same method (catches method errors, not source errors)

**True corroboration requires:**
1. **Semantic equivalence**: Claims must mean the same thing
2. **Evidential independence**: Different evidence chains
3. **Source diversity**: Not traceable to common origin
4. **Temporal distribution**: Not coordinated in time

### 2.2 Independence Measurement

```typescript
IndependenceScore {
  // How different are the evidence chains?
  evidential_independence: float    // 0.0 = same evidence, 1.0 = fully independent
  
  // How different are the sources?
  source_independence: float        // 0.0 = same source chain, 1.0 = fully independent
  
  // How different are the derivation methods?
  method_independence: float        // 0.0 = same method, 1.0 = different approaches
  
  // How distributed in time?
  temporal_independence: float      // 0.0 = simultaneous, 1.0 = well-distributed
  
  // Composite score
  overall: float                    // Weighted combination
}
```

**Independence calculation:**

```python
def calculate_independence(belief_a: Belief, belief_b: Belief) -> IndependenceScore:
    """Calculate how independent two corroborating beliefs are."""
    
    # Evidential independence: trace derivation chains
    evidence_overlap = jaccard_similarity(
        extract_evidence_sources(belief_a),
        extract_evidence_sources(belief_b)
    )
    evidential = 1.0 - evidence_overlap
    
    # Source independence: trace to original sources
    source_chains_a = trace_source_chain(belief_a)
    source_chains_b = trace_source_chain(belief_b)
    source = 1.0 - chain_overlap(source_chains_a, source_chains_b)
    
    # Method independence: different derivation types?
    method = method_diversity_score(
        belief_a.derivation,
        belief_b.derivation
    )
    
    # Temporal independence: when were they created?
    time_gap = abs(belief_a.created_at - belief_b.created_at)
    temporal = min(1.0, time_gap.days / 7)  # Max at 1 week apart
    
    # Weighted combination (evidential matters most)
    overall = (
        0.4 * evidential +
        0.3 * source +
        0.2 * method +
        0.1 * temporal
    )
    
    return IndependenceScore(evidential, source, method, temporal, overall)
```

### 2.3 Corroboration Events

```typescript
Corroboration {
  id: UUID
  
  // The beliefs involved
  primary_belief_id: UUID          // The belief being corroborated
  corroborating_belief_id: UUID    // The supporting belief
  
  // Participants
  primary_holder: DID
  corroborator: DID
  
  // Analysis
  semantic_similarity: float       // How similar are the claims? (must be > 0.85)
  independence: IndependenceScore  // How independent is this corroboration?
  
  // Strength
  effective_weight: float          // independence.overall × corroborator_reputation
  
  // Metadata
  created_at: timestamp
  verified_by: DID[]               // Third parties who verified the independence claim
}
```

---

## 3. Elevation Thresholds

### 3.1 L1 → L2 (Personal → Federated)

**Requirements:**
- Belief shared to federation
- Federation's aggregation threshold met (default k=5 contributors)
- Contributors agree (agreement_score > 0.6)

**Process:**
```
Agent shares belief to federation
  │
  ├── Belief enters federation pool
  │
  ├── Aggregation job runs:
  │   ├── Find semantically similar beliefs
  │   ├── Check k-anonymity threshold
  │   ├── Compute agreement score
  │   └── If thresholds met → publish aggregate
  │
  └── Aggregate becomes L2 knowledge
```

**No independence requirement** at L2—federation members are assumed to share context.

### 3.2 L2 → L3 (Federated → Domain)

**Requirements:**
- At least 3 federations with relevant domain expertise
- Aggregate beliefs from different federations corroborate
- Independence score > 0.5 between federation sources
- Domain expert verification (at least 2 experts with domain reputation > 0.7)

**Process:**
```
Multiple federations have similar aggregates
  │
  ├── Cross-federation matching:
  │   ├── Semantic similarity > 0.85
  │   ├── Independence score > 0.5
  │   └── At least 3 federations
  │
  ├── Expert verification:
  │   ├── Domain experts review
  │   ├── Submit expert verifications
  │   └── Verification stakes apply
  │
  ├── If all requirements met:
  │   ├── Create DomainKnowledge record
  │   ├── Link source federations
  │   └── Publish to domain knowledge layer
  │
  └── Belief is now L3 domain knowledge
```

### 3.3 L3 → L4 (Domain → Communal)

**Requirements:**
- Corroboration from multiple independent domains
- Byzantine quorum (see Section 4)
- No unresolved challenges above threshold
- Minimum age (anti-rush protection)

**Thresholds (configurable per domain):**

| Requirement | Default | Rationale |
|-------------|---------|-----------|
| Independent domains | ≥ 3 | Multiple perspectives |
| Independence score | > 0.7 | Strong evidential independence |
| Byzantine quorum | 2f+1 of 3f+1 | Tolerate f byzantine nodes |
| Active challenges | < 3 | No major disputes |
| Challenge weight | < 10% | Challenges are weak |
| Minimum age | 7 days | Resist coordinated rushes |
| Verification count | ≥ 10 | Multiple independent verifications |

**Process:**
```
Domain knowledge gains cross-domain corroboration
  │
  ├── Independence verification:
  │   ├── Trace evidence to independent sources
  │   ├── Verify no common derivation chain
  │   └── Compute independence score
  │
  ├── Byzantine consensus:
  │   ├── Consensus nodes validate
  │   ├── Each node independently verifies
  │   └── 2f+1 agreement required
  │
  ├── Challenge check:
  │   ├── Active challenges < threshold
  │   ├── Challenge weight < 10%
  │   └── No pending expert disputes
  │
  ├── Age check:
  │   └── Domain knowledge age > 7 days
  │
  ├── If all pass:
  │   ├── Create CommunalKnowledge record
  │   ├── Emit ElevatedToCommunal event
  │   ├── Update all source beliefs
  │   └── Notify interested parties
  │
  └── Belief is now L4 communal knowledge
```

### 3.4 Threshold Summary

| Elevation | Corroborators | Independence | Expert Review | Time |
|-----------|---------------|--------------|---------------|------|
| L1 → L2 | k (default 5) | Not required | Not required | Immediate |
| L2 → L3 | 3+ federations | > 0.5 | 2+ domain experts | After aggregation |
| L3 → L4 | 3+ domains | > 0.7 | Byzantine quorum | 7+ days |

---

## 4. Consensus Data Structures

### 4.1 ConsensusStatus

```typescript
ConsensusStatus {
  belief_id: UUID
  current_layer: Layer              // L1 | L2 | L3 | L4
  
  // Corroboration state
  corroboration: {
    total_corroborations: uint64
    independent_corroborations: uint64  // Independence > 0.5
    weighted_corroboration: float       // Sum of effective weights
    
    // By source type
    from_individuals: uint64
    from_federations: uint64
    from_domains: uint64
  }
  
  // Elevation progress
  elevation_progress: {
    next_layer: Layer | null
    requirements_met: string[]
    requirements_pending: string[]
    blockers: string[]              // Why elevation isn't happening
    estimated_time?: duration       // If predictable
  }
  
  // Challenge state
  challenges: {
    active_count: uint64
    total_challenge_weight: float
    strongest_challenge_id?: UUID
  }
  
  // Finality
  finality: FinalityStatus          // PROVISIONAL | STRENGTHENING | FINAL | CONTESTED
  finality_confidence: float        // 0.0-1.0
  
  // History
  layer_history: LayerTransition[]
  last_updated: timestamp
}

interface LayerTransition {
  from_layer: Layer
  to_layer: Layer
  timestamp: timestamp
  reason: string
  triggered_by: DID | 'system'
}
```

### 4.2 CommunalKnowledge

```typescript
CommunalKnowledge {
  id: UUID
  
  // The claim
  canonical_content: string         // Normalized statement of the belief
  content_hash: bytes               // For integrity
  
  // Confidence
  communal_confidence: ConfidenceVector {
    // Computed from aggregating all sources
    source_reliability: float       // Weighted by source layer
    method_quality: float           // Diversity of methods
    internal_consistency: float     // Agreement across sources
    temporal_freshness: float       // Most recent update
    corroboration: float            // Depth of independent verification
    domain_applicability: float     // Relevance scores
  }
  
  // Sources (traceable but privacy-preserving)
  source_summary: {
    domain_count: uint64
    federation_count: uint64
    verification_count: uint64
    unique_evidence_chains: uint64
  }
  
  // Independence proof
  independence_certificate: IndependenceCertificate
  
  // Status
  status: CommunalStatus            // ACTIVE | CONTESTED | REVISED | SUPERSEDED
  
  // Temporal
  established_at: timestamp
  last_verified: timestamp
  valid_until?: timestamp           // For time-bound knowledge
  
  // Links
  domains: string[]
  supersedes?: UUID[]               // Previous communal knowledge this replaces
  superseded_by?: UUID              // If this has been revised
}

interface IndependenceCertificate {
  // Cryptographic proof of independence
  verifier_signatures: Signature[]  // Consensus nodes who verified
  independence_proofs: {
    source_pair: [DID, DID]
    independence_score: float
    proof_hash: bytes               // Hash of independence calculation
  }[]
  created_at: timestamp
}
```

### 4.3 Challenge

```typescript
Challenge {
  id: UUID
  
  // Target
  target_belief_id: UUID
  target_layer: Layer
  
  // Challenger
  challenger_id: DID
  stake: Stake                      // Reputation at risk
  
  // Challenge content
  challenge_type: ChallengeType
  counter_evidence: Evidence[]
  reasoning: string
  
  // Status
  status: ChallengeStatus           // OPEN | UNDER_REVIEW | UPHELD | REJECTED | WITHDRAWN
  
  // Resolution
  resolution?: ChallengeResolution
  
  // Timing
  created_at: timestamp
  review_deadline: timestamp
  resolved_at?: timestamp
}

enum ChallengeType {
  FACTUAL_ERROR = 'factual_error'           // Claim is wrong
  INDEPENDENCE_VIOLATION = 'independence'    // Sources aren't independent
  EVIDENCE_QUALITY = 'evidence_quality'      // Evidence is weak
  OUTDATED = 'outdated'                      // Was true, no longer
  METHODOLOGY = 'methodology'                // Derivation is flawed
  SCOPE = 'scope'                            // Overgeneralized
}

interface ChallengeResolution {
  outcome: 'upheld' | 'rejected' | 'partial'
  resolved_by: DID[]                // Reviewers
  reasoning: string
  effects: {
    belief_demoted: boolean
    belief_revised: boolean
    challenger_rewarded: float
    holder_penalized: float
  }
}
```

---

## 5. Revision Handling

### 5.1 When Consensus Changes

Consensus is not immutable. Knowledge evolves. Revision happens when:

1. **Contradiction discovered**: New evidence contradicts established belief
2. **Source invalidated**: Key source loses reputation or is retracted
3. **Challenge upheld**: Formal challenge succeeds
4. **Supersession**: Better/more complete knowledge replaces old
5. **Temporal expiry**: Time-bound knowledge reaches end of validity

### 5.2 Revision Process

```
Challenge or contradiction detected
  │
  ├── Challenge enters review queue
  │
  ├── Review process:
  │   ├── Independent reviewers assigned (minimum 3)
  │   ├── Reviewers examine evidence
  │   ├── Reviewers stake reputation on assessment
  │   └── 2/3 agreement required for resolution
  │
  ├── If challenge UPHELD:
  │   │
  │   ├── Immediate effects:
  │   │   ├── Belief marked CONTESTED
  │   │   ├── Alert all consumers
  │   │   └── Freeze further elevation
  │   │
  │   ├── Resolution path determined:
  │   │   ├── DEMOTE: Move to lower layer
  │   │   ├── REVISE: Update with corrections
  │   │   ├── SUPERSEDE: Replace with new belief
  │   │   └── TOMBSTONE: Mark as retracted
  │   │
  │   └── Reputation effects:
  │       ├── Challenger gains (bonus for finding error)
  │       ├── Original sources penalized
  │       └── Previous verifiers penalized (proportional to confidence)
  │
  └── If challenge REJECTED:
      ├── Belief status restored
      ├── Challenger loses stake
      └── Belief gains "contested and upheld" mark (strengthens confidence)
```

### 5.3 Revision Types

| Type | Trigger | Effect | Reversible? |
|------|---------|--------|-------------|
| **Demotion** | Lost corroboration | Drop to lower layer | Yes (re-earn) |
| **Correction** | Partial error | Update content, keep layer | Yes |
| **Supersession** | Better knowledge | Old → superseded, new → active | Rarely |
| **Retraction** | Fundamental error | Tombstone, no layer | Rarely |
| **Expiry** | Time-bound | Status → EXPIRED | No |

### 5.4 Cascade Effects

When communal knowledge is revised, effects cascade:

```
Communal belief X revised
  │
  ├── All beliefs derived from X:
  │   ├── Flag for re-evaluation
  │   ├── Reduce confidence.source_reliability
  │   └── If heavily dependent → automatic demotion
  │
  ├── All verifications citing X:
  │   └── Mark as "evidence revised"
  │
  ├── All federations citing X:
  │   └── Trigger aggregate recomputation
  │
  └── Notification to:
      ├── All agents who queried X recently
      ├── All agents who derived from X
      └── All domain experts in X's domains
```

### 5.5 Revision Safeguards

**Anti-oscillation:**
- Minimum 30 days between demotion and re-elevation
- Revised beliefs need higher thresholds (1.5× normal)

**Anti-griefing:**
- Challenges require substantial stake
- Failed challenges penalize challenger
- Rate limits on challenges per belief

**Transparency:**
- Full revision history is public
- All actors are recorded
- Reputation changes are traceable

---

## 6. Confidence Aggregation

### 6.1 Aggregation by Layer

Each layer aggregates confidence differently:

**L1 (Personal):** No aggregation—individual assessment.

**L2 (Federated):**
```python
def aggregate_federation_confidence(beliefs: List[Belief], member_reps: Dict) -> ConfidenceVector:
    """Reputation-weighted average within federation."""
    weights = [member_reps[b.holder_id] for b in beliefs]
    return weighted_mean(beliefs, weights)
```

**L3 (Domain):**
```python
def aggregate_domain_confidence(aggregates: List[FederationAggregate], fed_reps: Dict) -> ConfidenceVector:
    """Federation-reputation-weighted, with independence bonus."""
    weights = [fed_reps[a.federation_id] * a.independence_score for a in aggregates]
    base = weighted_mean(aggregates, weights)
    
    # Bonus for independent corroboration
    base.corroboration = min(1.0, base.corroboration * (1 + 0.1 * len(aggregates)))
    return base
```

**L4 (Communal):**
```python
def aggregate_communal_confidence(domain_knowledge: List[DomainKnowledge]) -> ConfidenceVector:
    """Cross-domain aggregation with strong independence weighting."""
    
    # Verify independence
    independence_matrix = compute_pairwise_independence(domain_knowledge)
    avg_independence = matrix_mean(independence_matrix)
    
    # Base aggregation
    base = weighted_mean(domain_knowledge, [dk.reputation for dk in domain_knowledge])
    
    # Independence strongly affects corroboration
    base.corroboration = min(1.0, avg_independence * len(domain_knowledge) / 3)
    
    # Cross-domain agreement affects internal_consistency
    base.internal_consistency = compute_cross_domain_agreement(domain_knowledge)
    
    return base
```

### 6.2 Confidence Decay

Confidence decays without fresh corroboration:

```python
def apply_decay(confidence: ConfidenceVector, age: Duration, last_verification: Duration) -> ConfidenceVector:
    """Confidence decays over time without fresh verification."""
    
    # Freshness decays with age
    confidence.temporal_freshness *= exp(-age.days / 365)
    
    # Corroboration decays without re-verification
    verification_gap = last_verification.days
    if verification_gap > 90:
        decay_factor = exp(-(verification_gap - 90) / 180)
        confidence.corroboration *= decay_factor
    
    return confidence
```

---

## 7. Monitoring & Metrics

### 7.1 Consensus Health Metrics

```typescript
ConsensusMetrics {
  // Layer distribution
  beliefs_by_layer: Map<Layer, uint64>
  
  // Elevation flow
  elevations_last_24h: {
    l1_to_l2: uint64
    l2_to_l3: uint64
    l3_to_l4: uint64
  }
  
  // Revision health
  revisions_last_24h: {
    demotions: uint64
    corrections: uint64
    supersessions: uint64
    retractions: uint64
  }
  
  // Challenge health
  challenges: {
    open: uint64
    upheld_rate_30d: float
    avg_resolution_time: duration
  }
  
  // Independence
  avg_independence_score: float
  independence_violations_detected: uint64
  
  // Network health
  consensus_node_participation: float
  byzantine_fault_events: uint64
}
```

### 7.2 Alerts

System generates alerts for:

- **Elevation stall**: Belief meets thresholds but isn't elevating
- **Challenge spike**: Unusual number of challenges
- **Independence degradation**: Average independence scores dropping
- **Cascade event**: Major revision affecting many derived beliefs
- **Byzantine behavior**: Potential coordinated attack detected

---

## 8. Design Rationale

### Why independence-based consensus?

**Traditional consensus** (Byzantine Fault Tolerance) answers: "Did these nodes agree on this value?"  
**Epistemic consensus** must answer: "Did independent investigation converge on this truth?"

Voting and agreement can be gamed. Independent convergence is robust because it requires actually finding evidence through different paths.

### Why gradual elevation?

Knowledge builds trust over time. Rushed consensus is fragile consensus. The layer system lets beliefs prove themselves gradually while providing appropriate trust at each level.

### Why allow revision?

Immutable consensus would calcify errors. Science progresses by revision. The goal isn't permanent truth but currently-best-supported knowledge with clear provenance.

### Why separate domain knowledge (L3)?

Domain expertise matters. A claim about quantum physics should be evaluated by physicists before becoming communal knowledge. L3 provides expert gatekeeping without central authority.

### Why high thresholds for L4?

Communal knowledge is the network's strongest claim. False positives here damage network credibility. Better to have less L4 knowledge that's reliable than more that's uncertain.

---

## 9. Integration Points

| Component | Integration |
|-----------|-------------|
| **Verification Protocol** | Verifications contribute to corroboration |
| **Federation Layer** | L2 aggregates feed L3 |
| **Trust Graph** | Trust weights affect corroboration strength |
| **Reputation System** | Elevation/revision affects reputation |
| **Query Protocol** | Layer filters query results |
| **Identity** | All participants cryptographically identified |

---

## 10. Future Considerations

### 10.1 Predictive Consensus

ML models to predict which beliefs will likely elevate, enabling proactive verification focus.

### 10.2 Conditional Consensus

"X is true IF Y holds"—consensus with explicit preconditions.

### 10.3 Probabilistic Layers

Replace discrete layers with continuous consensus strength, preserving layer semantics as thresholds.

### 10.4 Cross-Network Consensus

Bridges to external knowledge networks (Wikipedia, academic databases) with appropriate trust mapping.

---

*"Consensus is not agreement. Consensus is independent convergence on truth."*
