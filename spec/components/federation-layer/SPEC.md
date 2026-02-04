# Federation Layer Specification

*Component 7 of Valence: Distributed Epistemic Infrastructure*

---

## Overview

A **Federation** is a group of agents who share beliefs privately within a bounded trust context. Federations implement the **L2: Shared Context** layer from the Valence architecture—"My trusted peers believe X."

Unlike public network beliefs or private personal beliefs, federated beliefs exist in a middle ground:
- **More trusted than public**: All sharers are known members
- **More private than public**: Content encrypted for members only
- **Aggregated**: Individual contributions blend into collective knowledge
- **Privacy-preserving**: Aggregates reveal knowledge without exposing contributors

---

## Core Concepts

### What Is a Federation?

A federation is:
- A **cryptographic group** with a shared encryption key
- A **membership roster** with roles and permissions
- A **belief pool** of shared knowledge from members
- A **governance model** for how decisions are made
- A **privacy boundary** that protects individual contributions

### Why Federations?

1. **Trusted Context**: Query beliefs from people you actually trust
2. **Privacy**: Share sensitive knowledge within boundaries
3. **Aggregation**: "5 doctors in our network believe X" without knowing which 5
4. **Domain Expertise**: Form groups around shared expertise
5. **Scaling Trust**: Can't personally know 1M agents; can trust a vetted group

---

## Data Structures

### Federation

```typescript
Federation {
  // === Identity ===
  id: UUID                           // Unique federation identifier
  did: DID                           // did:valence:fed:<fingerprint>
  name: string                       // Human-readable name
  description: string                // Purpose and scope
  
  // === Cryptographic Material ===
  group_key: GroupEncryptionKey {
    current_epoch: uint64            // Key rotation counter
    public_key: X25519PublicKey      // For encrypting to federation
    // Private key shares distributed to members
  }
  signing_key: Ed25519PublicKey      // Federation's collective identity
  
  // === Membership ===
  members: Map<DID, Membership>
  member_count: uint64               // May differ from members.size() for privacy
  
  // === Governance ===
  governance: GovernanceModel
  
  // === Domains ===
  domains: string[]                  // Areas of expertise/focus
  
  // === Configuration ===
  config: FederationConfig
  
  // === Metadata ===
  created_at: timestamp
  created_by: DID
  visibility: FederationVisibility   // discoverable | unlisted | secret
}
```

### Membership

```typescript
Membership {
  agent_id: DID                      // The member
  federation_id: UUID                // The federation
  
  // === Role & Permissions ===
  role: MemberRole
  permissions: Permission[]          // Explicit grants beyond role
  
  // === Cryptographic State ===
  key_share: EncryptedKeyShare       // Their portion of group key
  epoch_joined: uint64               // Which key epoch they joined at
  
  // === Status ===
  status: MemberStatus               // active | suspended | departed
  joined_at: timestamp
  invited_by: DID | null             // Who sponsored them
  
  // === Contribution Stats (privacy-preserving) ===
  contribution_score: float          // Relative activity (0.0-1.0)
  // Actual belief count hidden to preserve privacy
}

enum MemberRole {
  FOUNDER = 'founder'        // Created the federation, full control
  ADMIN = 'admin'            // Can manage members and settings
  MODERATOR = 'moderator'    // Can moderate shared beliefs
  MEMBER = 'member'          // Standard participation rights
  OBSERVER = 'observer'      // Read-only access
}

enum Permission {
  // Membership
  INVITE_MEMBERS = 'invite_members'
  APPROVE_MEMBERS = 'approve_members'
  REMOVE_MEMBERS = 'remove_members'
  
  // Beliefs
  SHARE_BELIEFS = 'share_beliefs'
  QUERY_BELIEFS = 'query_beliefs'
  MODERATE_BELIEFS = 'moderate_beliefs'
  VIEW_AGGREGATES = 'view_aggregates'
  
  // Governance
  PROPOSE_CHANGES = 'propose_changes'
  VOTE_ON_PROPOSALS = 'vote_on_proposals'
  EXECUTE_PROPOSALS = 'execute_proposals'
  
  // Administration
  MODIFY_CONFIG = 'modify_config'
  ROTATE_KEYS = 'rotate_keys'
  DISSOLVE_FEDERATION = 'dissolve_federation'
}
```

### GovernanceModel

```typescript
GovernanceModel {
  type: GovernanceType
  
  // Membership approval
  join_policy: JoinPolicy {
    type: 'open' | 'invite_only' | 'approval_required' | 'token_gated'
    min_reputation?: float           // Minimum network reputation to join
    required_vouches?: uint8         // Members who must vouch
    token_requirement?: TokenGate    // For token-gated communities
  }
  
  // Decision making
  decision_policy: DecisionPolicy {
    type: 'founder_decides' | 'admin_decides' | 'majority_vote' | 'supermajority' | 'consensus'
    quorum?: float                   // Minimum participation (0.0-1.0)
    threshold?: float                // Approval threshold (0.0-1.0)
    voting_period?: duration         // How long votes stay open
  }
  
  // Belief moderation
  moderation_policy: ModerationPolicy {
    type: 'none' | 'post_hoc' | 'pre_approval' | 'reputation_weighted'
    min_confidence?: float           // Beliefs below this get flagged
    dispute_threshold?: uint8        // Reports before review
  }
}

enum GovernanceType {
  AUTOCRATIC = 'autocratic'          // Single leader decides
  COUNCIL = 'council'                // Admin group decides
  DEMOCRATIC = 'democratic'          // All members vote
  MERITOCRATIC = 'meritocratic'      // Votes weighted by reputation
  FEDERATED = 'federated'            // Sub-groups with delegates
}
```

### FederationConfig

```typescript
FederationConfig {
  // Size limits
  max_members: uint64                // 0 = unlimited
  min_members_for_aggregation: uint8 // k-anonymity threshold (default: 5)
  
  // Privacy settings
  privacy: PrivacyConfig {
    hide_member_list: boolean        // Only show count, not identities
    hide_contribution_source: boolean // Don't link beliefs to sharers
    aggregation_noise: float         // Differential privacy epsilon
    enable_plausible_deniability: boolean
  }
  
  // Belief settings
  beliefs: BeliefConfig {
    allowed_visibility: Visibility[] // Which belief visibilities can be shared
    require_derivation: boolean      // Must explain where belief came from
    min_confidence: float            // Floor for shared beliefs
    max_age_days: uint32             // Reject beliefs older than this
  }
  
  // Key management
  key_rotation_period: duration      // How often to rotate group key
  
  // Reputation
  reputation_weight: float           // How much federation membership affects agent reputation
}
```

---

## Federation Lifecycle

### 1. Creation

```
Agent calls create_federation(config)
  │
  ├── Generate federation keypairs
  │   ├── Group encryption key (X25519)
  │   └── Signing key (Ed25519)
  │
  ├── Initialize membership with founder
  │   └── Founder gets full key, max permissions
  │
  ├── Configure governance model
  │
  ├── Register DID: did:valence:fed:<fingerprint>
  │
  └── Return Federation object
```

### 2. Membership Changes

**Join Flow (Approval Required):**
```
Applicant calls join_federation(federation_id, invite?)
  │
  ├── If invite_only && no valid invite → REJECT
  │
  ├── If approval_required:
  │   ├── Create pending membership
  │   ├── Notify approvers
  │   └── Wait for approval
  │
  ├── On approval:
  │   ├── Generate key share for new member
  │   ├── Encrypt and deliver key share
  │   ├── Add to member roster
  │   └── Emit MemberJoined event
  │
  └── Return Membership object
```

**Leave Flow:**
```
Member calls leave_federation(federation_id)
  │
  ├── Mark membership as departed
  │
  ├── Trigger key rotation
  │   └── New key share excludes departed member
  │
  ├── Retain historical contributions
  │   └── But unlinked from departed identity
  │
  └── Emit MemberDeparted event
```

**Removal Flow:**
```
Admin calls remove_member(federation_id, member_id, reason)
  │
  ├── Verify admin has REMOVE_MEMBERS permission
  │
  ├── Mark membership as removed
  │
  ├── Trigger immediate key rotation
  │
  ├── Optionally: quarantine their contributions
  │
  └── Emit MemberRemoved event
```

### 3. Key Rotation

Group keys rotate:
- **Periodically** (per config, e.g., monthly)
- **On member departure** (forward secrecy)
- **On suspected compromise**
- **On admin command**

```
Key Rotation Process:
  │
  ├── Generate new epoch keypair
  │
  ├── For each active member:
  │   ├── Encrypt new key share to their public key
  │   └── Deliver encrypted share
  │
  ├── Update group_key.current_epoch
  │
  ├── Re-encrypt active belief pool with new key
  │   └── (Or maintain multi-epoch decryption capability)
  │
  └── Archive old epoch key for historical reads
```

### 4. Dissolution

```
Founder/Admin calls dissolve_federation(federation_id, reason)
  │
  ├── Verify DISSOLVE_FEDERATION permission
  │
  ├── If governance requires vote:
  │   └── Run dissolution vote per decision_policy
  │
  ├── On approval:
  │   ├── Mark federation as dissolved
  │   ├── Export member list (to members only)
  │   ├── Optionally export belief archive
  │   ├── Destroy group keys after grace period
  │   └── Emit FederationDissolved event
  │
  └── Historical reference retained (DID → "dissolved")
```

---

## Belief Aggregation

The core value proposition: turning individual beliefs into collective knowledge while preserving privacy.

### Aggregation Model

```typescript
AggregatedBelief {
  federation_id: UUID
  topic_hash: bytes                  // Privacy-preserving topic identifier
  
  // Aggregate content (not individual beliefs)
  content_summary: string            // "Members believe X" or semantic cluster
  
  // Aggregate confidence
  aggregate_confidence: ConfidenceVector {
    // Each dimension is aggregated from contributors
    source_reliability: float        // Weighted by contributor reputation
    method_quality: float            // Average of contributed beliefs
    internal_consistency: float      // Cross-contributor agreement
    temporal_freshness: float        // Recency-weighted
    corroboration: float             // f(contributor_count, independence)
    domain_applicability: float      // Domain match score
  }
  
  // Privacy-preserving statistics
  contributor_count: uint64          // Noised if below k-threshold
  confidence_distribution: {         // Histogram, not individual values
    low: uint64                      // Contributors with confidence < 0.4
    medium: uint64                   // 0.4 - 0.7
    high: uint64                     // > 0.7
  }
  agreement_score: float             // 0.0 = total disagreement, 1.0 = consensus
  
  // Temporal
  first_contributed: timestamp
  last_updated: timestamp
  
  // Derivation (aggregated)
  dominant_derivation_types: DerivationType[]  // Most common sources
  external_ref_count: uint64         // How many cite external sources
}
```

### Aggregation Rules

**1. Minimum Contributors (k-anonymity)**
- Aggregates only published when `contributor_count >= min_members_for_aggregation`
- Below threshold: exists but returns "insufficient data"
- Default k = 5

**2. Confidence Aggregation**
```python
def aggregate_confidence(beliefs: List[Belief], member_reputations: Dict[DID, float]) -> ConfidenceVector:
    """Weight by member reputation within federation."""
    
    weights = [member_reputations.get(b.holder_id, 0.5) for b in beliefs]
    total_weight = sum(weights)
    
    return ConfidenceVector(
        source_reliability = sum(w * b.confidence.source_reliability for w, b in zip(weights, beliefs)) / total_weight,
        method_quality = sum(w * b.confidence.method_quality for w, b in zip(weights, beliefs)) / total_weight,
        internal_consistency = compute_agreement(beliefs),  # Special: measures cross-belief consistency
        temporal_freshness = max(b.confidence.temporal_freshness for b in beliefs),  # Most recent
        corroboration = min(1.0, len(beliefs) / 10),  # Scales with count
        domain_applicability = harmonic_mean([b.confidence.domain_applicability for b in beliefs])
    )
```

**3. Content Clustering**
When multiple beliefs address similar topics:
- Cluster by semantic similarity (embedding cosine > 0.85)
- Generate summary content via LLM or template
- Preserve strongest supporting derivations

**4. Disagreement Handling**
When contributors disagree:
- `agreement_score` reflects variance
- If agreement < 0.5: flag as "contested"
- Optionally: expose opposing clusters as separate aggregates
- Never expose which individuals disagree

### Aggregation Pipeline

```
Individual belief shared to federation
  │
  ├── Encrypt with group key
  │
  ├── Store in federation belief pool
  │
  ├── Periodic aggregation job:
  │   │
  │   ├── Cluster beliefs by topic (semantic similarity)
  │   │
  │   ├── For each cluster with >= k contributors:
  │   │   ├── Compute aggregate confidence
  │   │   ├── Generate content summary
  │   │   ├── Apply differential privacy noise
  │   │   └── Publish AggregatedBelief
  │   │
  │   └── Update federation knowledge index
  │
  └── Members query aggregated beliefs
```

---

## Privacy Preservation

See [PRIVACY.md](./PRIVACY.md) for detailed mechanisms. Summary:

### Privacy Guarantees

| Guarantee | Mechanism | Protection Against |
|-----------|-----------|-------------------|
| **k-anonymity** | Min contributor threshold | Re-identification |
| **Differential privacy** | Noise injection | Statistical inference |
| **Encrypted storage** | Group encryption | External observation |
| **Unlinkability** | Blinded contributions | Contribution tracing |
| **Plausible deniability** | Decoy beliefs | Compelled disclosure |
| **Forward secrecy** | Key rotation on departure | Past exposure |

### What's Hidden vs. Visible

| Data | Members See | Outsiders See |
|------|-------------|---------------|
| Federation exists | ✅ | Depends on visibility |
| Member list | Config-dependent | ❌ |
| Member count | ✅ (possibly noised) | ❌ |
| Individual shared beliefs | ❌ (only aggregates) | ❌ |
| Who shared what | ❌ | ❌ |
| Aggregate beliefs | ✅ | ❌ |
| Agreement scores | ✅ | ❌ |

---

## Trust Integration

Federations interact with the Trust Graph:

### Federation-Based Trust

```typescript
// TrustBasis type for federation membership
TrustBasis {
  type: FEDERATION_MEMBER
  federation_id: UUID
  confidence: float          // Based on federation's reputation
  evidence: [{
    type: 'shared_federation',
    federation_id: UUID,
    federation_reputation: float
  }]
}
```

### Trust Flow

1. **Joining boosts trust**: Mutual federation membership adds trust basis
2. **Federation reputation**: Federations themselves have reputation scores
3. **Transitive through federation**: "I trust the federation → I partially trust members"
4. **Contribution affects trust**: Active contributors in good standing gain trust

### Trust Decay in Federations

- Inactive members: trust from federation membership decays
- Departed members: trust basis removed (but history preserved)
- Dissolved federations: trust basis frozen at dissolution value

---

## Federation Types

### By Purpose

**Knowledge Federations**
- Focus: Sharing domain expertise
- Example: "AI Safety Researchers Federation"
- Config: High min_confidence, require_derivation = true

**Working Groups**
- Focus: Coordinating on shared projects
- Example: "Valence Core Contributors"
- Config: Lower barriers, higher trust baseline

**Communities of Practice**
- Focus: Ongoing learning and discussion
- Example: "Rust Async Working Group"
- Config: Democratic governance, open-ish membership

**Validation Networks**
- Focus: Cross-validating claims
- Example: "Medical Literature Reviewers"
- Config: Meritocratic, high reputation requirements

### By Size

| Size | Members | Governance | Key Management |
|------|---------|------------|----------------|
| Micro | 2-10 | Informal | Direct key sharing |
| Small | 10-50 | Council | Threshold scheme |
| Medium | 50-500 | Democratic | Hierarchical keys |
| Large | 500-5000 | Delegated | Tree-based distribution |
| Massive | 5000+ | Federated | Multi-tier federation |

---

## Inter-Federation Relations

Federations can relate to each other:

### Federation Bridges

```typescript
FederationBridge {
  federation_a: UUID
  federation_b: UUID
  
  bridge_type: 'symmetric' | 'a_to_b' | 'b_to_a'
  
  // What flows across the bridge
  shared_domains: string[]           // Only beliefs in these domains
  min_aggregate_agreement: float     // Only high-agreement aggregates
  
  // Bridge-specific key for cross-federation encryption
  bridge_key: GroupEncryptionKey
  
  governance: {
    requires_both_approval: boolean
    review_period: duration
  }
}
```

### Hierarchical Federations

Federations can contain sub-federations:
```
Parent Federation
  ├── Sub-federation A
  │   └── Members
  ├── Sub-federation B
  │   └── Members
  └── Direct members
```

Sub-federations:
- Inherit parent policies (can be stricter)
- Can share aggregates upward
- Members of sub are implicit members of parent

---

## Events

Federation operations emit events for audit and synchronization:

```typescript
FederationEvent {
  id: UUID
  federation_id: UUID
  timestamp: timestamp
  type: FederationEventType
  actor: DID                        // Who caused this (may be null for system)
  payload: any                      // Event-specific data
  signature: bytes                  // Signed by actor or federation key
}

enum FederationEventType {
  // Lifecycle
  FEDERATION_CREATED
  FEDERATION_DISSOLVED
  CONFIG_CHANGED
  
  // Membership
  MEMBER_JOINED
  MEMBER_DEPARTED
  MEMBER_REMOVED
  MEMBER_ROLE_CHANGED
  
  // Keys
  KEY_ROTATED
  KEY_COMPROMISED
  
  // Beliefs
  BELIEF_SHARED             // (payload is encrypted/hashed)
  AGGREGATE_PUBLISHED
  BELIEF_MODERATED
  
  // Governance
  PROPOSAL_CREATED
  VOTE_CAST                 // (anonymous if configured)
  PROPOSAL_RESOLVED
}
```

---

## Invariants

These must always hold:

1. **Key access = membership**: Only active members can decrypt current-epoch content
2. **k-anonymity**: Aggregates never published below contributor threshold
3. **Forward secrecy**: Departed members cannot read post-departure content
4. **Audit trail**: All membership changes are logged and signed
5. **Governance compliance**: Actions requiring votes cannot bypass governance
6. **Privacy bounds**: Individual contributions never exposed via aggregates

---

## Storage Requirements

### Per-Federation

| Data | Size Estimate |
|------|---------------|
| Federation metadata | ~2 KB |
| Per member | ~500 bytes |
| Per shared belief | ~5 KB (encrypted) |
| Per aggregate | ~2 KB |
| Event log | ~200 bytes/event |

### Scaling

- Small federation (50 members, 1000 beliefs): ~10 MB
- Medium federation (500 members, 50000 beliefs): ~500 MB
- Large federation (5000 members, 1M beliefs): ~10 GB

### Indices

```sql
-- Federation lookups
CREATE INDEX ON federations (id);
CREATE INDEX ON federations (visibility) WHERE visibility = 'discoverable';

-- Membership
CREATE INDEX ON memberships (federation_id, status);
CREATE INDEX ON memberships (agent_id, status);

-- Beliefs
CREATE INDEX ON shared_beliefs (federation_id);
CREATE INDEX ON shared_beliefs (topic_hash);

-- Aggregates
CREATE INDEX ON aggregated_beliefs (federation_id, topic_hash);
CREATE INDEX ON aggregated_beliefs (last_updated);

-- Events
CREATE INDEX ON federation_events (federation_id, timestamp);
CREATE INDEX ON federation_events (actor, timestamp);
```

---

## Relationship to Other Components

| Component | Relationship |
|-----------|--------------|
| **Belief Schema** | Shared beliefs follow Belief structure |
| **Identity** | Members identified by DID, federation has DID |
| **Trust Graph** | Federation membership is a trust basis |
| **Query Protocol** | Federation scope is a query filter |
| **Verification** | Cross-federation verification possible |
| **Consensus** | Federation aggregates can elevate to L3/L4 |

---

*"Federations: where private meets trusted, and individual becomes collective."*
