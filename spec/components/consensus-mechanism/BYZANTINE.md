# Consensus Mechanism — Byzantine Fault Tolerance

*"Design for adversaries, optimize for friends."*

---

## Overview

The Valence consensus mechanism must remain trustworthy even when some participants are malicious. This document specifies:

1. **Byzantine fault tolerance model** — How many bad actors can we tolerate?
2. **Network partition handling** — What if the network splits?
3. **Consistency vs availability tradeoffs** — CAP theorem decisions
4. **Finality** — When is consensus "final" vs "provisional"?

**Key insight**: Epistemic consensus differs from transactional consensus. We're not ordering transactions—we're aggregating evidence. This allows for probabilistic, eventually-consistent approaches that traditional BFT systems can't use.

---

## 1. Threat Model

### 1.1 Adversary Types

| Adversary | Capability | Goal | Defense |
|-----------|------------|------|---------|
| **Sybil** | Create many identities | Inflate corroboration | Stake requirements, trust graph |
| **Collusion Ring** | Coordinate privately | Fake independence | Independence verification |
| **Eclipse** | Isolate honest nodes | Forge local consensus | Diverse peer connections |
| **Griefing** | Spam challenges | Degrade service | Stake costs, rate limits |
| **Long-range** | Control old keys | Rewrite history | Checkpointing, finality |
| **Influence** | Legitimate reputation | Bias consensus | Diversity requirements |

### 1.2 Attack Scenarios

**Scenario A: Corroboration Stuffing**
```
Attacker creates 50 identities
Each provides "independent" corroboration
Belief rushes to L4 without real verification
```

**Scenario B: Challenge Flooding**
```
Attacker challenges every high-value belief
Real contributors spend resources defending
Network becomes unusable
```

**Scenario C: Consensus Splitting**
```
Attacker runs nodes in multiple partitions
Provides different "evidence" to each
Creates conflicting communal knowledge
```

**Scenario D: Reputation Laundering**
```
Attacker builds reputation legitimately
Uses reputation for single high-value attack
Extracts more value than reputation cost
```

---

## 2. Byzantine Fault Tolerance Model

### 2.1 The Basic Bound

For traditional BFT: **N ≥ 3f + 1** nodes to tolerate **f** Byzantine faults.

**Valence adaptation**: We don't have a fixed validator set. Instead, we use **weighted stake** where Byzantine tolerance depends on reputation distribution:

```
Honest stake (Sh) > Byzantine stake (Sb)

For safety: Sh > 2 × Sb
Therefore: Sb < 1/3 of total stake
```

**Implication**: As long as honest participants control > 2/3 of total reputation, consensus is safe.

### 2.2 Layer-Specific Tolerance

Different layers have different fault tolerance requirements:

| Layer | Tolerance Model | Byzantine Threshold | Rationale |
|-------|-----------------|---------------------|-----------|
| L1 | N/A | N/A | Personal beliefs, no consensus needed |
| L2 | Federation governance | Per-federation | Trust federation's internal controls |
| L3 | Domain expert weighted | Sb < 20% domain rep | Smaller expert pools need tighter bounds |
| L4 | Network-wide weighted | Sb < 33% total rep | Classic 1/3 threshold |

### 2.3 Weighted Consensus for L4

Communal knowledge elevation requires weighted agreement:

```python
def byzantine_safe_elevation(
    corroborations: List[Corroboration],
    verifications: List[Verification]
) -> bool:
    """
    Check if elevation is safe under Byzantine assumptions.
    Assumes up to 1/3 of stake may be Byzantine.
    """
    
    # Calculate total supporting stake
    supporting_stake = sum(
        c.corroborator_reputation * c.independence_score
        for c in corroborations
        if c.result == 'supporting'
    )
    
    # Calculate total opposing stake
    opposing_stake = sum(
        v.verifier_reputation
        for v in verifications
        if v.result == 'contradicted'
    )
    
    # Total stake in play
    total_stake = supporting_stake + opposing_stake
    
    # Byzantine safety: supporting must exceed 2/3 of total
    # This guarantees honest majority even if 1/3 of supporters are Byzantine
    if total_stake > 0:
        supporting_ratio = supporting_stake / total_stake
        if supporting_ratio < 0.67:
            return False
    
    # Minimum absolute stake (prevent trivial consensus)
    MIN_STAKE_THRESHOLD = 10.0  # Configurable
    if supporting_stake < MIN_STAKE_THRESHOLD:
        return False
    
    # Diversity check: can't be dominated by single actor
    stake_by_actor = group_by_actor(corroborations)
    max_single_actor = max(stake_by_actor.values())
    if max_single_actor > supporting_stake * 0.25:
        return False  # No single actor > 25% of support
    
    return True
```

### 2.4 Independence as Defense

Traditional BFT assumes arbitrary Byzantine behavior. Valence adds **independence verification** which makes certain attacks harder:

**Why independence helps:**
- Sybils can create identities but struggle to create *independent evidence chains*
- Collusion can agree on claims but must fake *different sources*
- Independence is verifiable and forgery-detectable

**Independence-augmented threshold:**

```python
def independence_adjusted_threshold(base_threshold: float, avg_independence: float) -> float:
    """
    Higher independence scores allow lower Byzantine thresholds.
    Intuition: if evidence is truly independent, fewer corroborators needed.
    """
    # Independence from 0.5 to 1.0 can reduce threshold by up to 20%
    independence_bonus = max(0, avg_independence - 0.5) * 0.4
    return base_threshold * (1 - independence_bonus)

# Example:
# base_threshold = 0.67 (need 2/3 support)
# avg_independence = 0.9 (very independent)
# adjusted = 0.67 * (1 - 0.4 * 0.4) = 0.67 * 0.84 = 0.56
# Still > 50%, but less than pure BFT requires
```

---

## 3. Sybil Resistance

### 3.1 The Sybil Problem

Anyone can create identities. Without cost, an attacker creates millions of identities to overwhelm honest participants.

### 3.2 Defense: Reputation as Stake

**Core mechanism**: Actions require reputation stake. Creating influence costs reputation. Reputation is earned slowly and lost quickly.

```typescript
ReputationCost {
  // Cost to perform consensus-relevant actions
  submit_corroboration: 0.001 base + 0.01 × confidence_claimed
  challenge_consensus: 0.02 base + 0.05 × target_layer
  vote_on_challenge: 0.005
  
  // Cost is locked, returned on honest behavior
  // Lost on detected bad behavior
}
```

**Economics of Sybil attack:**

```
Cost to create n Sybil identities with reputation R each:
  = n × (time to earn R) × (opportunity cost)
  
Value of Sybil attack:
  = (reward from fake consensus) - (risk of detection × penalty)
  
For attack to be profitable:
  Value > Cost
  
Defense: Make cost > value for all attacks
```

### 3.3 Defense: Trust Graph Analysis

The trust graph reveals Sybil clusters:

```python
def detect_sybil_cluster(corroborations: List[Corroboration]) -> SybilRisk:
    """
    Analyze corroboration pattern for Sybil indicators.
    """
    contributors = [c.corroborator for c in corroborations]
    
    # Build contributor trust subgraph
    trust_subgraph = extract_trust_subgraph(contributors)
    
    # Sybil indicator 1: Dense internal connections, sparse external
    internal_density = trust_subgraph.internal_edge_density()
    external_density = trust_subgraph.external_edge_density()
    insularity_score = internal_density / max(external_density, 0.01)
    
    # Sybil indicator 2: Similar creation times
    creation_times = [get_identity_age(c) for c in contributors]
    temporal_clustering = coefficient_of_variation(creation_times)
    
    # Sybil indicator 3: Similar activity patterns
    activity_patterns = [get_activity_vector(c) for c in contributors]
    pattern_similarity = avg_pairwise_cosine(activity_patterns)
    
    # Combine indicators
    risk_score = (
        0.4 * sigmoid(insularity_score - 5) +   # Insular clusters suspicious
        0.3 * (1 - temporal_clustering) +        # Simultaneous creation suspicious
        0.3 * pattern_similarity                  # Similar behavior suspicious
    )
    
    return SybilRisk(
        score=risk_score,
        indicators={
            'insularity': insularity_score,
            'temporal_clustering': temporal_clustering,
            'pattern_similarity': pattern_similarity
        }
    )
```

### 3.4 Defense: Progressive Stake Requirements

New identities face higher barriers:

```python
def stake_requirement(agent: Agent, action: Action) -> float:
    """
    New accounts and accounts with thin history face higher stakes.
    """
    base_stake = action.base_stake_requirement
    
    # Age factor: new accounts need 3x stake, decreasing over 1 year
    age_days = agent.age.days
    age_factor = max(1.0, 3.0 - (age_days / 365) * 2)
    
    # History factor: thin history needs more stake
    history_factor = max(1.0, 2.0 - agent.verification_count / 100)
    
    # Trust factor: well-trusted agents need less
    trust_factor = max(0.5, 1.0 - agent.avg_trust_received * 0.5)
    
    return base_stake * age_factor * history_factor * trust_factor
```

---

## 4. Network Partition Handling

### 4.1 Partition Scenarios

**Scenario 1: Clean Split**
```
Network splits into partitions A and B
Each continues operating independently
Later merges back together
```

**Scenario 2: Degraded Connectivity**
```
Some nodes can reach A, some can reach B, some can reach both
Messages delay or drop between partitions
```

**Scenario 3: Adversarial Partition**
```
Attacker deliberately isolates victims
Feeds them false information
Tries to create conflicting consensus
```

### 4.2 Design Choice: Availability over Consistency

Valence chooses **AP** (Availability + Partition tolerance) from the CAP theorem:

- **During partition**: Both sides continue operating
- **After healing**: Reconciliation with conflict detection
- **Rationale**: Epistemic knowledge isn't transactional—conflicting beliefs can coexist and be resolved

### 4.3 Partition Detection

```python
def detect_partition(node: Node) -> PartitionStatus:
    """
    Detect if we might be in a partition.
    """
    # Check peer connectivity
    reachable_peers = node.get_reachable_peers()
    expected_peers = node.get_expected_peers()
    connectivity_ratio = len(reachable_peers) / len(expected_peers)
    
    # Check consensus node reachability
    reachable_consensus = node.get_reachable_consensus_nodes()
    consensus_ratio = len(reachable_consensus) / EXPECTED_CONSENSUS_NODES
    
    # Check recent message latency
    avg_latency = node.get_avg_peer_latency()
    latency_normal = avg_latency < NORMAL_LATENCY_THRESHOLD
    
    if connectivity_ratio < 0.3 or consensus_ratio < 0.5:
        return PartitionStatus.LIKELY_PARTITIONED
    elif connectivity_ratio < 0.7 or not latency_normal:
        return PartitionStatus.POSSIBLY_PARTITIONED
    else:
        return PartitionStatus.HEALTHY
```

### 4.4 Behavior During Partition

**L1-L2**: Continue normally (local/federation scope)

**L3**: 
- Continue domain knowledge operations
- Mark new L3 knowledge as "partition-provisional"
- Higher threshold for elevation (compensate for reduced visibility)

**L4**:
- **Do not create new L4 knowledge during detected partition**
- Existing L4 remains valid
- Queue elevation candidates for post-partition review

```python
def elevation_allowed(node: Node, target_layer: Layer) -> bool:
    partition_status = detect_partition(node)
    
    if target_layer == Layer.L4:
        return partition_status == PartitionStatus.HEALTHY
    
    if target_layer == Layer.L3:
        # Allow but mark provisional
        return True  # with provisional flag
    
    return True
```

### 4.5 Partition Reconciliation

When partitions heal:

```python
def reconcile_partitions(local_state: State, remote_state: State) -> ReconciliationResult:
    """
    Merge states after partition heals.
    """
    conflicts = []
    
    # Compare L4 beliefs created during partition
    local_l4 = local_state.get_l4_since(partition_start)
    remote_l4 = remote_state.get_l4_since(partition_start)
    
    for belief in local_l4 + remote_l4:
        # Check for contradictions
        contradictions = find_contradictions(belief, remote_state if belief in local_l4 else local_state)
        
        if contradictions:
            conflicts.append(ConflictRecord(
                belief=belief,
                contradictions=contradictions,
                resolution_needed=True
            ))
    
    # For non-conflicting beliefs: merge (union)
    merged_l4 = merge_non_conflicting(local_l4, remote_l4)
    
    # For conflicting beliefs: demote to L3, trigger review
    for conflict in conflicts:
        demote_to_l3(conflict.belief)
        demote_to_l3(conflict.contradictions)
        create_conflict_review(conflict)
    
    return ReconciliationResult(
        merged_beliefs=len(merged_l4),
        conflicts_found=len(conflicts),
        reviews_created=len(conflicts)
    )
```

### 4.6 Split-Brain Prevention for L4

To prevent both partitions from creating conflicting L4:

**Mechanism: Quorum requirement**

```python
CONSENSUS_NODE_COUNT = 21  # Example: 21 consensus nodes globally
QUORUM_SIZE = 14           # 2/3 + 1

def can_create_l4(reachable_consensus_nodes: int) -> bool:
    """
    Only create L4 if we can reach a quorum of consensus nodes.
    """
    return reachable_consensus_nodes >= QUORUM_SIZE
```

**If neither partition has quorum**: Neither creates L4. This is the safe choice—no L4 knowledge is created during partition, preventing conflicts.

---

## 5. Finality Model

### 5.1 Finality Levels

Unlike blockchain finality (transaction irreversible), epistemic finality is probabilistic:

```typescript
enum FinalityLevel {
  PROVISIONAL = 'provisional'       // Newly elevated, may change
  STRENGTHENING = 'strengthening'   // Gaining corroboration, increasingly stable
  PRACTICAL = 'practical'           // Very unlikely to change
  MAXIMUM = 'maximum'               // Would require extraordinary evidence to revise
}

interface FinalityAssessment {
  level: FinalityLevel
  confidence: float                 // 0.0-1.0, probability of revision
  
  factors: {
    corroboration_depth: float      // How deep is the evidence tree?
    time_stable: Duration           // How long unchanged?
    challenge_history: float        // Survived challenges? (-1 to +1)
    independence_strength: float    // How independent are sources?
  }
}
```

### 5.2 Finality Calculation

```python
def calculate_finality(belief: CommunalKnowledge) -> FinalityAssessment:
    """
    Calculate finality level for communal knowledge.
    """
    # Factor 1: Corroboration depth
    corroboration_depth = compute_evidence_tree_depth(belief)
    depth_score = min(1.0, corroboration_depth / 5)  # Max at depth 5
    
    # Factor 2: Time stability
    time_since_last_change = now() - belief.last_modified
    time_score = min(1.0, time_since_last_change.days / 365)  # Max at 1 year
    
    # Factor 3: Challenge survival
    challenges = get_challenges(belief.id)
    upheld_challenges = [c for c in challenges if c.status == 'upheld']
    rejected_challenges = [c for c in challenges if c.status == 'rejected']
    
    if challenges:
        challenge_score = (len(rejected_challenges) - len(upheld_challenges)) / len(challenges)
    else:
        challenge_score = 0  # Neutral if never challenged
    
    # Factor 4: Independence strength
    independence_score = belief.independence_certificate.avg_independence
    
    # Combine factors
    finality_score = (
        0.25 * depth_score +
        0.25 * time_score +
        0.25 * normalize(challenge_score) +
        0.25 * independence_score
    )
    
    # Map to level
    if finality_score < 0.3:
        level = FinalityLevel.PROVISIONAL
    elif finality_score < 0.6:
        level = FinalityLevel.STRENGTHENING
    elif finality_score < 0.85:
        level = FinalityLevel.PRACTICAL
    else:
        level = FinalityLevel.MAXIMUM
    
    # Confidence = probability belief won't be revised
    confidence = 0.5 + 0.5 * finality_score  # Range: 0.5 to 1.0
    
    return FinalityAssessment(
        level=level,
        confidence=confidence,
        factors={
            'corroboration_depth': depth_score,
            'time_stable': time_score,
            'challenge_history': challenge_score,
            'independence_strength': independence_score
        }
    )
```

### 5.3 Finality Thresholds

| Level | Confidence | Revision Requires | Typical Time |
|-------|------------|-------------------|--------------|
| PROVISIONAL | 0.50-0.65 | New evidence | 0-30 days |
| STRENGTHENING | 0.65-0.80 | Strong counter-evidence | 1-6 months |
| PRACTICAL | 0.80-0.92 | Paradigm-shifting evidence | 6-24 months |
| MAXIMUM | 0.92-1.00 | Extraordinary evidence | 2+ years |

### 5.4 Finality and Consumers

How should consumers treat finality?

```typescript
interface ConsumerGuidance {
  level: FinalityLevel
  
  // Recommended behavior
  recommended_action: string
  confidence_in_decisions: string
  revision_monitoring: string
}

const CONSUMER_GUIDANCE: Map<FinalityLevel, ConsumerGuidance> = {
  PROVISIONAL: {
    recommended_action: 'Treat as likely but verify independently for important decisions',
    confidence_in_decisions: 'Low-stakes decisions only',
    revision_monitoring: 'Subscribe to changes'
  },
  STRENGTHENING: {
    recommended_action: 'Reasonable to act on for most purposes',
    confidence_in_decisions: 'Medium-stakes decisions acceptable',
    revision_monitoring: 'Periodic check recommended'
  },
  PRACTICAL: {
    recommended_action: 'Safe to treat as established knowledge',
    confidence_in_decisions: 'High-stakes decisions acceptable',
    revision_monitoring: 'Annual review sufficient'
  },
  MAXIMUM: {
    recommended_action: 'Treat as reliable baseline fact',
    confidence_in_decisions: 'Foundation for other reasoning',
    revision_monitoring: 'Only if paradigm changes'
  }
}
```

### 5.5 Finality Is Not Immutability

**Critical distinction**: Even MAXIMUM finality knowledge can be revised. "Earth orbits the Sun" is MAXIMUM finality but could theoretically be revised with extraordinary evidence.

Finality indicates:
- How much verification has occurred
- How stable the knowledge has been
- How much evidence would be needed to revise

**Not**:
- That revision is impossible
- That challenges are rejected automatically
- That the knowledge is certainly true

---

## 6. Attack-Specific Defenses

### 6.1 Long-Range Attack Defense

**Attack**: Attacker with old keys creates alternative history.

**Defense**: Checkpointing + social consensus

```python
CHECKPOINT_INTERVAL = 1000  # blocks/epochs
CHECKPOINT_SIGNERS = 100    # Random selection of high-reputation agents

def create_checkpoint(state: State) -> Checkpoint:
    """
    Create a checkpoint that's hard to forge retroactively.
    """
    checkpoint = Checkpoint(
        epoch=state.current_epoch,
        state_root=state.merkle_root(),
        l4_beliefs_hash=hash_l4_beliefs(state),
        timestamp=now()
    )
    
    # Get signatures from random high-reputation agents
    signers = select_checkpoint_signers(CHECKPOINT_SIGNERS)
    signatures = collect_signatures(checkpoint, signers)
    
    # Also anchor to external timestamping (e.g., Bitcoin)
    external_anchor = create_external_anchor(checkpoint)
    
    checkpoint.signatures = signatures
    checkpoint.external_anchor = external_anchor
    
    return checkpoint

def validate_history(claimed_state: State, checkpoints: List[Checkpoint]) -> bool:
    """
    Validate that claimed state is consistent with checkpoints.
    """
    for checkpoint in checkpoints:
        # Verify checkpoint signatures (would need to compromise many agents)
        if not verify_checkpoint_signatures(checkpoint):
            return False
        
        # Verify state matches checkpoint at that epoch
        historical_state = claimed_state.at_epoch(checkpoint.epoch)
        if historical_state.merkle_root() != checkpoint.state_root:
            return False
        
        # Verify external anchor (would need to rewrite Bitcoin)
        if not verify_external_anchor(checkpoint):
            return False
    
    return True
```

### 6.2 Grinding Attack Defense

**Attack**: Attacker tries many variations to find one that passes validation.

**Defense**: Commit-reveal + economic cost

```python
def submit_corroboration_two_phase(evidence: Evidence) -> Commitment:
    """
    Two-phase commit prevents grinding.
    """
    # Phase 1: Commit hash of evidence (can't see if it will pass)
    commitment = Commitment(
        evidence_hash=hash(evidence),
        submitter=current_agent(),
        stake_locked=compute_stake(evidence),
        timestamp=now()
    )
    
    # Commitment is binding—stake locked regardless of outcome
    lock_stake(commitment.stake_locked)
    
    return commitment

def reveal_corroboration(commitment: Commitment, evidence: Evidence) -> CorroborationResult:
    """
    Phase 2: Reveal actual evidence.
    """
    # Verify commitment matches
    if hash(evidence) != commitment.evidence_hash:
        slash_stake(commitment.stake_locked, reason='commitment_mismatch')
        return CorroborationResult.REJECTED
    
    # Now process normally
    # Attacker already committed—can't grind for better outcome
    return process_corroboration(evidence)
```

### 6.3 Reputation Manipulation Defense

**Attack**: Build reputation legitimately, then use it for one big attack.

**Defense**: Stake velocity limits + reputation decay

```python
MAX_STAKE_VELOCITY = 0.1  # Can't stake more than 10% of reputation per day

def check_stake_velocity(agent: Agent, new_stake: float) -> bool:
    """
    Prevent rapid reputation deployment.
    """
    staked_last_24h = agent.get_stake_last_24h()
    total_reputation = agent.reputation.overall
    
    if (staked_last_24h + new_stake) > total_reputation * MAX_STAKE_VELOCITY:
        return False
    
    return True

def apply_reputation_decay(agent: Agent):
    """
    Reputation decays without ongoing positive participation.
    This prevents "reputation savings accounts" for future attacks.
    """
    days_since_positive_contribution = agent.days_since_last_positive()
    
    if days_since_positive_contribution > 30:
        decay_rate = 0.01 * (days_since_positive_contribution - 30)  # 1% per day after 30 days
        agent.reputation.overall *= (1 - min(decay_rate, 0.1))  # Cap at 10% per application
```

### 6.4 Eclipse Attack Defense

**Attack**: Surround victim node with attacker-controlled peers.

**Defense**: Diverse peer selection + anchor nodes

```python
MIN_PEER_DIVERSITY = 0.5  # At least 50% of peers from different "regions"

def select_peers(node: Node) -> List[Peer]:
    """
    Select diverse peers to resist eclipse attacks.
    """
    candidate_peers = node.discover_peers()
    
    selected = []
    regions_represented = set()
    
    for peer in sorted(candidate_peers, key=lambda p: -p.reputation):
        peer_region = determine_region(peer)  # Network/geographic/trust-graph region
        
        # Ensure diversity
        if len(selected) > 8 and peer_region in regions_represented:
            if len(regions_represented) / len(selected) < MIN_PEER_DIVERSITY:
                continue  # Skip to get more diversity
        
        selected.append(peer)
        regions_represented.add(peer_region)
        
        if len(selected) >= MAX_PEERS:
            break
    
    # Add hardcoded anchor nodes (known good, diverse)
    anchor_nodes = get_anchor_nodes()
    for anchor in anchor_nodes:
        if anchor not in selected:
            selected.append(anchor)
    
    return selected
```

---

## 7. Graceful Degradation

### 7.1 Degradation Levels

The system should degrade gracefully, not fail catastrophically:

| Condition | Degradation | Effect |
|-----------|-------------|--------|
| 10-30% nodes offline | None | Normal operation |
| 30-50% nodes offline | Minor | Slower consensus, higher thresholds |
| 50-67% nodes offline | Moderate | L4 elevation paused, L1-L3 continue |
| 67%+ nodes offline | Major | Read-only mode, no new consensus |
| Network partition | Significant | Per-partition degradation + reconciliation queue |

### 7.2 Adaptive Thresholds

```python
def compute_adaptive_thresholds(network_health: NetworkHealth) -> Thresholds:
    """
    Adjust thresholds based on network conditions.
    """
    base_thresholds = get_base_thresholds()
    
    # Availability factor: how much of network is reachable?
    availability = network_health.reachable_stake / network_health.total_stake
    
    if availability > 0.9:
        # Healthy network: normal thresholds
        return base_thresholds
    
    elif availability > 0.67:
        # Degraded: increase thresholds for safety
        return Thresholds(
            l4_support_ratio=min(0.85, base_thresholds.l4_support_ratio * 1.2),
            l4_min_stake=base_thresholds.l4_min_stake * 1.5,
            challenge_review_size=base_thresholds.challenge_review_size + 2
        )
    
    elif availability > 0.5:
        # Significantly degraded: pause L4, raise L3 thresholds
        return Thresholds(
            l4_elevation_enabled=False,
            l3_support_ratio=min(0.9, base_thresholds.l3_support_ratio * 1.3),
            new_belief_rate_limit=base_thresholds.new_belief_rate_limit * 0.5
        )
    
    else:
        # Severely degraded: read-only mode
        return Thresholds(
            l4_elevation_enabled=False,
            l3_elevation_enabled=False,
            write_operations_enabled=False
        )
```

### 7.3 Recovery Protocol

When network recovers from degradation:

```python
def execute_recovery_protocol(previous_health: NetworkHealth, current_health: NetworkHealth):
    """
    Safely transition from degraded to healthy state.
    """
    # Phase 1: Validation (don't trust immediately)
    validation_period = Duration(hours=24)
    
    if not sustained_health(current_health, validation_period):
        return  # Wait for sustained recovery
    
    # Phase 2: Reconciliation
    if was_partitioned(previous_health):
        reconciliation_result = reconcile_partitions()
        if reconciliation_result.conflicts_found > 0:
            notify_admins(reconciliation_result)
            # Don't fully restore until conflicts reviewed
            return
    
    # Phase 3: Process backlog
    process_queued_elevations()
    process_queued_challenges()
    
    # Phase 4: Restore normal operation
    restore_normal_thresholds()
    emit_event(NetworkRecovered())
```

---

## 8. Monitoring & Alerting

### 8.1 Byzantine Indicators

```typescript
interface ByzantineIndicators {
  // Sybil indicators
  sybil_risk_score: float           // 0.0-1.0
  suspicious_corroboration_patterns: uint64
  new_identity_velocity: float       // New IDs per hour
  
  // Collusion indicators
  unusual_agreement_clusters: uint64
  independence_score_anomalies: uint64
  coordinated_timing_events: uint64
  
  // Partition indicators
  connectivity_ratio: float
  consensus_node_reachability: float
  message_latency_anomalies: uint64
  
  // Attack indicators
  challenge_spike: boolean
  reputation_drain_rate: float       // Unusual reputation losses
  stake_velocity_violations: uint64
}
```

### 8.2 Alert Conditions

```python
ALERT_CONDITIONS = [
    AlertCondition(
        name='potential_sybil_attack',
        condition=lambda i: i.sybil_risk_score > 0.7,
        severity='high',
        action='pause_l4_elevation'
    ),
    AlertCondition(
        name='partition_detected',
        condition=lambda i: i.connectivity_ratio < 0.5,
        severity='critical',
        action='enter_partition_mode'
    ),
    AlertCondition(
        name='challenge_flooding',
        condition=lambda i: i.challenge_spike and recent_challenges() > normal_rate * 10,
        severity='medium',
        action='increase_challenge_stake'
    ),
    AlertCondition(
        name='collusion_suspected',
        condition=lambda i: i.unusual_agreement_clusters > 5,
        severity='high',
        action='flag_for_review'
    )
]
```

---

## 9. Design Principles

### 9.1 Defense in Depth

No single defense is sufficient. Layer defenses:

1. **Economic**: Stake requirements make attacks expensive
2. **Cryptographic**: Signatures prevent forgery
3. **Statistical**: Pattern detection finds anomalies
4. **Social**: Trust graph isolates bad actors
5. **Temporal**: Time delays prevent rush attacks

### 9.2 Fail-Safe Defaults

When uncertain:
- **Don't elevate** (false negatives are better than false positives for L4)
- **Don't slash** (innocent-until-proven-guilty for reputation)
- **Do preserve** (keep records for later analysis)
- **Do alert** (human review for edge cases)

### 9.3 Incentive Alignment

Make honest behavior profitable and dishonest behavior costly:

```
Honest participation reward >> Attack profit potential
Detection probability × Penalty > Attack expected value
```

### 9.4 Transparency

All consensus operations are logged and auditable. Bad actors can be identified retroactively. This creates deterrence even when real-time detection fails.

---

## 10. Future Work

### 10.1 Formal Verification

Model the consensus mechanism in TLA+ or similar to prove safety properties.

### 10.2 Economic Analysis

Game-theoretic analysis of attack economics under various parameter settings.

### 10.3 Privacy-Preserving BFT

Explore zero-knowledge proofs for Byzantine agreement without revealing identities.

### 10.4 AI-Specific Threats

As AI agents become participants, new attack vectors emerge (e.g., prompt injection to manipulate agent beliefs). Design defenses specific to AI participants.

---

*"The network must be trustworthy even when participants are not."*
