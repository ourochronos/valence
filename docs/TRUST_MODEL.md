# Valence Federation Trust Model

*How trust is earned, computed, and applied in the federation*

This document specifies the trust model for Valence federation, where trust is earned through epistemic behavior rather than credentials or stake.

---

## 1. Core Philosophy

### 1.1 Trust Principles

From [PRINCIPLES.md](./PRINCIPLES.md):

- **Structurally incapable of betrayal**: Trust is enforced by architecture, not promises
- **Reduced access rather than exile**: Graceful degradation, never binary banning
- **Aggregation serves users**: Trust mechanisms exist to protect user value

### 1.2 Trust vs. Reputation

| Aspect | Reputation Systems | Valence Trust |
|--------|-------------------|---------------|
| Basis | Social endorsement | Epistemic accuracy |
| Measurement | Votes/ratings | Corroboration rate |
| Gaming | Vote manipulation | Must actually produce quality beliefs |
| Recovery | Hard (permanent record) | Gradual through behavior |
| Failure mode | Popularity contests | Information quality |

---

## 2. Multi-Dimensional Trust

### 2.1 NodeTrust Model

Trust mirrors the `DimensionalConfidence` model for epistemic consistency:

```python
@dataclass
class NodeTrust:
    """Trust dimensions for a federation node."""

    # Combined score
    overall: float = 0.1  # Default low trust for new nodes

    # Epistemic dimensions
    belief_accuracy: float | None = None      # How often beliefs are corroborated
    extraction_quality: float | None = None   # Quality of knowledge extraction
    curation_accuracy: float | None = None    # Handling of contradictions

    # Operational dimensions
    uptime_reliability: float | None = None       # Consistent availability
    contribution_consistency: float | None = None # Regular, quality participation

    # Social dimensions
    endorsement_strength: float | None = None     # Trust from nodes we trust

    # Domain-specific
    domain_expertise: dict[str, float] = {}  # Per-domain competence

    # Temporal
    relationship_age_days: int = 0
    last_interaction: datetime | None = None
```

### 2.2 Dimension Weights

Default weights for computing overall trust:

```python
TRUST_WEIGHTS = {
    "belief_accuracy": 0.30,        # Most important: actual quality
    "extraction_quality": 0.15,     # Process quality
    "curation_accuracy": 0.10,      # Contradiction handling
    "uptime_reliability": 0.10,     # Operational reliability
    "contribution_consistency": 0.15, # Regular participation
    "endorsement_strength": 0.15,   # Social trust
    "relationship_age": 0.05,       # Time-based trust (small)
}
```

### 2.3 Trust Computation

```python
def compute_overall_trust(node_trust: NodeTrust) -> float:
    """Compute overall trust from dimensions."""
    weighted_sum = 0.0
    total_weight = 0.0

    for dim, weight in TRUST_WEIGHTS.items():
        value = getattr(node_trust, dim, None)
        if value is not None:
            weighted_sum += value * weight
            total_weight += weight

    # Age bonus (capped at 0.05 after 90 days)
    age_bonus = min(0.05, node_trust.relationship_age_days / 90 * 0.05)

    if total_weight > 0:
        return min(1.0, (weighted_sum / total_weight) + age_bonus)

    # Fallback for nodes with no measurements
    return 0.1 + age_bonus
```

---

## 3. Trust Establishment Phases

### 3.1 Phase Overview

```
Observer → Contributor → Participant → Anchor
  0.1         0.1-0.4       0.4-0.8      0.8-1.0
 Days 1-7     Days 7-30     Day 30+      Earned
```

### 3.2 Observer Phase (Days 1-7)

**Capabilities:**
- Query federated beliefs (read-only)
- Receive sync updates (passive)

**Cannot:**
- Contribute to aggregation queries
- Share beliefs to federation
- Vote on tension resolutions

**Trust Range:** 0.1 (fixed baseline)

**Exit Criteria:**
- 7 days active
- Successfully received 10+ syncs
- No protocol violations

### 3.3 Contributor Phase (Days 7-30)

**Capabilities:**
- Share beliefs with LOW influence weight (0.25x)
- Contribute to aggregation (reduced weight)
- Receive corroboration feedback

**Cannot:**
- Full aggregation influence
- Vote on critical tensions
- Endorse other nodes

**Trust Range:** 0.1 → 0.4

**Trust Factors:**
- Belief accuracy: % of shared beliefs that get corroborated
- Response reliability: % of aggregation queries responded to

**Exit Criteria:**
- 30 days in contributor phase
- Shared 50+ beliefs
- Belief accuracy > 60%
- No significant violations

### 3.4 Participant Phase (Day 30+)

**Capabilities:**
- Full aggregation participation (normal weight)
- Vote on tension resolutions
- Endorse other nodes (contributor+ only)

**Trust Range:** 0.4 → 0.8

**Trust Factors:**
- All contributor factors
- Curation accuracy (tension resolution quality)
- Endorsement validity (do endorsed nodes perform well?)
- Domain expertise development

### 3.5 Anchor Phase (Earned)

**Requirements:**
- 90+ days as participant
- Trust score > 0.8 sustained
- Endorsed by 2+ existing anchors
- Exceptional track record

**Capabilities:**
- Help validate new nodes faster (vouching)
- Higher aggregation weight
- Authority in tension resolution
- Can establish new anchors (with other anchors)

**Trust Range:** 0.8 → 1.0

---

## 4. Trust Signals

### 4.1 Positive Signals

| Signal | Trust Impact | Dimension Affected |
|--------|-------------|-------------------|
| Belief corroborated by others | +0.02 | belief_accuracy |
| Clean sync completed | +0.01 | uptime_reliability |
| Aggregation response on time | +0.01 | contribution_consistency |
| Endorsed node performs well | +0.02 | endorsement_strength |
| Tension resolution accepted | +0.02 | curation_accuracy |

### 4.2 Negative Signals

| Signal | Trust Impact | Dimension Affected |
|--------|-------------|-------------------|
| Belief disputed by multiple nodes | -0.05 | belief_accuracy |
| Sync timeout/failure | -0.02 | uptime_reliability |
| Aggregation non-response | -0.02 | contribution_consistency |
| Endorsed node behaves badly | -0.03 | endorsement_strength |
| Invalid signature detected | -0.10 | belief_accuracy |
| Protocol violation | -0.10 | overall |

### 4.3 Signal Decay

Trust signals decay over time to reflect recent behavior:

```python
def apply_signal_decay(trust: NodeTrust, days_since: int) -> NodeTrust:
    """Apply decay to trust dimensions based on time since last interaction."""
    decay_rate = 0.99  # 1% decay per day

    for dim in trust_dimensions:
        current = getattr(trust, dim)
        if current is not None:
            decayed = current * (decay_rate ** days_since)
            # Floor at phase minimum
            setattr(trust, dim, max(phase_floor(trust), decayed))

    return trust.recalculate_overall()
```

---

## 5. Sybil Resistance

### 5.1 Attack Vector

Sybil attack: Creating many fake nodes to gain disproportionate influence.

### 5.2 Mitigations

**1. Proof of Contribution**
```python
def contribution_required_for_trust(target_trust: float) -> int:
    """Beliefs that must be corroborated to reach trust level."""
    # Exponentially increasing cost
    return int(10 * (target_trust ** 2) * 100)

# Example: Trust 0.4 requires 160 corroborated beliefs
# Example: Trust 0.8 requires 640 corroborated beliefs
```

**2. Social Verification**
- Participant phase requires endorsement
- Anchor status requires multiple anchor endorsements
- Endorsement from compromised nodes doesn't count

**3. Diminishing Returns**
```python
def sybil_detection(nodes: list[FederationNode]) -> list[NodeCluster]:
    """Detect nodes with suspiciously similar behavior."""
    clusters = cluster_by_belief_similarity(nodes)

    for cluster in clusters:
        if len(cluster) > 1:
            # Apply sublinear influence to cluster members
            combined_influence = sum(n.trust for n in cluster) ** 0.5
            for node in cluster:
                node.effective_trust = node.trust * (combined_influence / len(cluster))
```

**4. Rate Limiting**
- New nodes: 10 beliefs/day shareable
- Contributor nodes: 50 beliefs/day
- Participant nodes: 200 beliefs/day
- Anchor nodes: 500 beliefs/day

---

## 6. Graceful Degradation

### 6.1 Philosophy

Per PRINCIPLES: "Reduced access rather than exile"

Bad actors are **attenuated**, not banned. They can always:
- Query beliefs (read)
- Receive sync updates
- Attempt to rebuild trust

### 6.2 Threat Response Levels

```python
class ThreatResponse:
    """Proportional response to detected threats."""

    def respond(self, node: FederationNode, threat_score: float) -> list[Action]:
        actions = []

        if threat_score >= 0.2:
            # Level 1: Increased scrutiny
            actions.append(IncreasedValidation(node))
            # All shared beliefs require signature re-verification

        if threat_score >= 0.4:
            # Level 2: Reduced influence
            actions.append(TrustPenalty(node, reduction=0.3))
            # Trust score reduced by 30%

        if threat_score >= 0.6:
            # Level 3: Quarantine from sensitive operations
            actions.append(ExcludeFromAggregation(node, duration=days(7)))
            actions.append(ExcludeFromVoting(node, duration=days(7)))

        if threat_score >= 0.8:
            # Level 4: Functional isolation
            actions.append(ReadOnlyMode(node))
            # Can query and receive, cannot contribute
            # NOT exiled - can still benefit from federation

        # NEVER: permanent ban, data deletion, identity revocation
        return actions
```

### 6.3 Trust Recovery

```python
def trust_recovery_path(node: FederationNode, current_trust: float) -> RecoveryPath:
    """Calculate path back to good standing."""

    return RecoveryPath(
        target_trust=0.4,  # Participant threshold

        requirements=[
            Requirement(
                action="Share quality beliefs",
                count=50,
                metric="corroboration_rate > 0.7"
            ),
            Requirement(
                action="Maintain uptime",
                duration=days(14),
                metric="availability > 0.95"
            ),
            Requirement(
                action="No protocol violations",
                duration=days(30)
            )
        ],

        estimated_duration=days(30 + (0.4 - current_trust) * 60)
    )
```

---

## 7. Trust Attestations

### 7.1 Attestation Format

```python
@dataclass
class TrustAttestation:
    """Signed statement of trust from one node to another."""

    issuer_did: str
    subject_did: str

    # What is being attested
    attestation_type: str  # "endorsement", "domain_expertise", "vouching"

    # Trust dimensions being attested
    attested_dimensions: dict[str, float]

    # Scope
    domains: list[str] | None  # None = general

    # Validity
    issued_at: datetime
    expires_at: datetime

    # Cryptographic proof
    signature: str

    def verify(self) -> bool:
        """Verify attestation signature."""
        issuer_key = resolve_did(self.issuer_did).verification_key
        message = canonical_json(self.attestation_content())
        return ed25519_verify(issuer_key, message, self.signature)
```

### 7.2 Attestation Types

| Type | Meaning | Requirements |
|------|---------|-------------|
| `endorsement` | General recommendation | Issuer must be Participant+ |
| `domain_expertise` | Expertise in specific domain | Issuer has expertise in domain |
| `vouching` | Accelerate new node onboarding | Issuer must be Anchor |
| `corroboration` | Belief content agreement | Automatic from aggregation |

### 7.3 Attestation Weight

```python
def attestation_weight(attestation: TrustAttestation) -> float:
    """Calculate weight of an attestation based on issuer's trust."""
    issuer_trust = get_node_trust(attestation.issuer_did)

    # Base weight from issuer trust
    base_weight = issuer_trust.overall

    # Recency bonus (recent attestations worth more)
    age_days = (now() - attestation.issued_at).days
    recency_factor = 0.95 ** age_days  # 5% decay per day

    # Domain relevance
    if attestation.domains:
        domain_factor = max(
            issuer_trust.domain_expertise.get(d, 0.5)
            for d in attestation.domains
        )
    else:
        domain_factor = 1.0

    return base_weight * recency_factor * domain_factor
```

---

## 8. Domain-Specific Trust

### 8.1 Domain Expertise

Trust varies by knowledge domain:

```python
class DomainTrust:
    """Track trust per knowledge domain."""

    def __init__(self, node_id: UUID):
        self.node_id = node_id
        self.domains: dict[str, DomainExpertise] = {}

    def record_belief_outcome(self, belief: Belief, outcome: str):
        """Update domain trust based on belief outcome."""
        for domain in belief.domain_path:
            if domain not in self.domains:
                self.domains[domain] = DomainExpertise(domain)

            self.domains[domain].update(
                corroborated=(outcome == "corroborated"),
                disputed=(outcome == "disputed")
            )

    def get_domain_trust(self, domain: str) -> float:
        """Get trust for specific domain."""
        expertise = self.domains.get(domain)
        if expertise is None:
            return 0.5  # Neutral for unknown domains
        return expertise.compute_trust()
```

### 8.2 Cross-Domain Trust

When querying cross-domain beliefs:

```python
def compute_cross_domain_trust(
    node: FederationNode,
    query_domains: list[str]
) -> float:
    """Compute trust for a multi-domain query."""
    domain_trusts = [
        node.trust.domain_expertise.get(d, 0.5)
        for d in query_domains
    ]

    # Use geometric mean (penalizes low scores in any domain)
    return geometric_mean(domain_trusts)
```

---

## 9. Trust Storage

### 9.1 Database Schema

See [FEDERATION_SCHEMA.md](./FEDERATION_SCHEMA.md) for:
- `node_trust` - Node-to-node trust relationships
- `user_node_trust` - User preference overrides
- `belief_trust_annotations` - Per-belief trust adjustments

### 9.2 Trust Cache

```python
class TrustCache:
    """In-memory cache for frequently accessed trust data."""

    def __init__(self, ttl_seconds: int = 300):
        self.cache: dict[UUID, CachedTrust] = {}
        self.ttl = ttl_seconds

    def get(self, node_id: UUID) -> NodeTrust | None:
        cached = self.cache.get(node_id)
        if cached and not cached.is_expired(self.ttl):
            return cached.trust
        return None

    def set(self, node_id: UUID, trust: NodeTrust):
        self.cache[node_id] = CachedTrust(trust, timestamp=now())

    def invalidate(self, node_id: UUID):
        self.cache.pop(node_id, None)
```

---

## 10. User Control

### 10.1 Trust Preferences

Users can override automatic trust:

| Preference | Effect | Use Case |
|------------|--------|----------|
| `blocked` | Node's beliefs never shown | Known bad actor |
| `reduced` | Trust score halved | Skepticism |
| `automatic` | Use computed trust | Default |
| `elevated` | Trust score × 1.5 | Trusted source |
| `anchor` | Treat as anchor | Personal trust |

### 10.2 Domain Overrides

```python
# Example: User trusts Alice for tech but not politics
user_preferences = {
    "did:vkb:web:alice.example.com": {
        "default": "automatic",
        "domains": {
            "tech": "elevated",
            "politics": "blocked"
        }
    }
}
```

### 10.3 Transparency

Users can always see:
- Trust scores for all known nodes
- Factors contributing to trust
- Their own override settings
- Attestations issued and received

---

## 11. Implementation Checklist

### Phase 2 (Basic Federation)
- [ ] NodeTrust dataclass
- [ ] Basic trust computation
- [ ] Trust storage (database)
- [ ] Phase transitions (observer → contributor)

### Phase 4 (Trust Networks)
- [ ] Trust attestations
- [ ] Sybil detection
- [ ] Graceful degradation
- [ ] Domain-specific trust
- [ ] Trust recovery paths
- [ ] User preference overrides

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0-draft | 2025-01-20 | Initial trust model specification |
