# Federation Privacy Mechanisms

*Privacy-preserving group knowledge sharing in Valence.*

---

## Overview

Federation privacy addresses a fundamental tension: **sharing knowledge within a group while protecting individual contributions**. This document specifies the cryptographic and statistical mechanisms that enable:

1. **Aggregation without attribution** — Learn what the group knows, not who knows it
2. **Encrypted sharing** — Only members can read federated content
3. **Plausible deniability** — Cannot prove an individual shared specific content
4. **Audit without exposure** — Verify rules are followed without seeing data
5. **Forward secrecy** — Past content protected when members leave

---

## Threat Model

### Adversaries

| Adversary | Capabilities | Goals |
|-----------|-------------|-------|
| **External observer** | Network traffic, public metadata | Learn federation content or membership |
| **Curious member** | All member access | Identify who shared what |
| **Compromised member** | Key material, collude with others | De-anonymize contributions |
| **Malicious admin** | Admin privileges | Access beyond legitimate scope |
| **Compelled disclosure** | Legal/coercive access to member | Prove membership or contributions |
| **Future attacker** | Access after key rotation | Decrypt historical content |

### Privacy Goals

| Goal | Protection Against |
|------|-------------------|
| **Content confidentiality** | External observer |
| **Membership privacy** | External observer, compelled disclosure |
| **Contribution unlinkability** | Curious member, compromised member |
| **Plausible deniability** | Compelled disclosure |
| **Forward secrecy** | Future attacker, departed member |
| **Audit integrity** | Malicious admin |

---

## 1. Differential Privacy for Aggregations

### The Problem

Aggregated statistics can leak individual information:
- "3 people believe X" → if you know 2 of them don't, the third is identified
- Change in aggregate after your share → reveals your contribution
- Aggregate on niche topic → small anonymity set

### The Solution: (ε, δ)-Differential Privacy

Add calibrated noise to all aggregated outputs so that any individual's contribution is statistically masked.

**Definition:**
A mechanism M is (ε, δ)-differentially private if for any two datasets D and D' differing in one individual, and any output S:

```
Pr[M(D) ∈ S] ≤ exp(ε) × Pr[M(D') ∈ S] + δ
```

**Practical meaning:** An adversary cannot reliably distinguish whether any specific individual contributed.

### Implementation

#### Noise Mechanism: Gaussian

For numeric aggregates (counts, averages, sums):

```python
def add_gaussian_noise(true_value: float, sensitivity: float, epsilon: float, delta: float) -> float:
    """Add Gaussian noise calibrated for (ε, δ)-DP."""
    # σ = sensitivity × √(2 ln(1.25/δ)) / ε
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    noise = random.gauss(0, sigma)
    return true_value + noise
```

#### Per-Query Privacy Budget

Each query consumes privacy budget. Track cumulative spend:

```python
class PrivacyAccountant:
    def __init__(self, total_epsilon: float, total_delta: float):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
    
    def can_query(self, query_epsilon: float, query_delta: float) -> bool:
        # Simple composition (could use advanced composition)
        return (self.spent_epsilon + query_epsilon <= self.total_epsilon and
                self.spent_delta + query_delta <= self.total_delta)
    
    def record_query(self, query_epsilon: float, query_delta: float):
        self.spent_epsilon += query_epsilon
        self.spent_delta += query_delta
```

#### Aggregation Functions

| Aggregate | Sensitivity | Noise Formula |
|-----------|-------------|---------------|
| Count | 1 | Gaussian(σ = √(2ln(1.25/δ))/ε) |
| Sum (bounded) | max_value | Gaussian scaled by max_value |
| Average | max_value / n | Gaussian scaled by range/n |
| Histogram | 1 per bin | Gaussian per bin |

#### Configuration Defaults

```typescript
interface DifferentialPrivacyConfig {
  epsilon: float          // Default: 0.1 (strong privacy)
  delta: float            // Default: 1e-6
  
  // Per-query limits
  max_epsilon_per_query: float  // Default: 0.01
  max_queries_per_day: uint32   // Default: 1000
  
  // Budget reset
  budget_reset_period: duration // Default: '1d' (daily reset)
}
```

#### Privacy Levels

| Level | ε | δ | Use Case |
|-------|---|---|----------|
| **High privacy** | 0.1 | 10⁻⁷ | Sensitive topics, small federations |
| **Standard** | 1.0 | 10⁻⁶ | General use |
| **Low privacy** | 5.0 | 10⁻⁵ | Large federations, non-sensitive |

### Example: Noisy Contributor Count

```python
def get_noised_contributor_count(
    true_count: int,
    config: DifferentialPrivacyConfig
) -> int:
    """Return contributor count with DP noise."""
    noised = add_gaussian_noise(
        true_value=true_count,
        sensitivity=1,  # Adding/removing one person changes count by 1
        epsilon=config.epsilon,
        delta=config.delta
    )
    # Clamp to reasonable range
    return max(0, round(noised))
```

---

## 2. k-Anonymity for Shared Beliefs

### The Problem

Even with aggregate noise, topics with few contributors risk identification:
- Topic has 2 contributors → 50% chance of identification
- Unique derivation type → fingerprints the contributor
- Timing correlation → reveals who shared when

### The Solution: k-Anonymity Threshold

**Never publish aggregates with fewer than k contributors**, where k is the minimum anonymity set size.

```typescript
interface KAnonymityConfig {
  k: uint8                      // Default: 5
  suppress_below_k: boolean     // Default: true (don't publish)
  noise_below_threshold: uint8  // If not suppressing, add extra noise below this
}
```

### Enforcement Points

#### 1. Aggregate Publication Gate

```python
def should_publish_aggregate(
    contributor_count: int,
    k: int
) -> tuple[bool, str]:
    """Determine if aggregate can be published."""
    if contributor_count < k:
        return False, f"insufficient_contributors (need {k}, have {contributor_count})"
    return True, "ok"
```

#### 2. Query Response Filtering

```python
def filter_query_results(
    aggregates: List[AggregatedBelief],
    k: int
) -> List[AggregatedBelief]:
    """Filter out aggregates below k-threshold."""
    return [
        agg for agg in aggregates
        if agg.contributor_count >= k
    ]
```

#### 3. Topic Clustering

When a topic is below k, consider clustering with related topics:

```python
def cluster_for_anonymity(
    topic: TopicCluster,
    all_topics: List[TopicCluster],
    k: int,
    similarity_threshold: float = 0.8
) -> Optional[TopicCluster]:
    """Merge topic with similar topics to achieve k-anonymity."""
    if topic.contributor_count >= k:
        return topic
    
    # Find similar topics to merge
    candidates = [
        t for t in all_topics
        if t.id != topic.id and 
           semantic_similarity(t, topic) >= similarity_threshold
    ]
    
    merged = merge_topics([topic] + candidates)
    if merged.contributor_count >= k:
        return merged
    
    return None  # Cannot achieve k-anonymity
```

### l-Diversity Extension

k-anonymity alone doesn't prevent attribute disclosure. Add l-diversity: each anonymity set must have at least l different values for sensitive attributes.

```python
def check_l_diversity(
    aggregate: AggregatedBelief,
    l: int
) -> bool:
    """Ensure aggregate has sufficient diversity in contributor attributes."""
    # For federations: ensure l different confidence levels represented
    unique_confidence_buckets = len(set(
        bucket_confidence(c) for c in aggregate.contributor_confidences
    ))
    return unique_confidence_buckets >= l
```

---

## 3. Encrypted Belief Sharing

### The Problem

Beliefs shared to a federation should only be readable by current members. Requirements:
- Encrypt once, readable by all members
- New members can read (after joining)
- Departed members cannot read (future content)
- Efficient for many members

### The Solution: Group Encryption with Key Rotation

#### Architecture

```
Sharer                          Federation                        Members
   │                                │                                │
   │  Encrypt(belief, group_key)    │                                │
   │ ─────────────────────────────► │                                │
   │                                │  Store encrypted               │
   │                                │                                │
   │                                │  On query: return encrypted    │
   │                                │ ◄────────────────────────────► │
   │                                │                                │
   │                                │  Member decrypts with their    │
   │                                │  key share                     │
```

#### Key Structure

```typescript
interface GroupKey {
  epoch: uint64                      // Monotonic key version
  
  // For encryption (anyone can encrypt to group)
  public_key: X25519PublicKey
  
  // Key shares (each member holds one)
  // Threshold scheme: any t-of-n can reconstruct
  threshold: uint8
  total_shares: uint16
}

interface MemberKeyShare {
  member_id: DID
  epoch: uint64
  share: EncryptedBlob              // Encrypted to member's personal key
  share_index: uint16               // For reconstruction
}
```

#### Encryption Scheme: Hybrid AEAD

1. **Symmetric content encryption** (fast)
2. **Asymmetric key wrapping** (for group key)

```python
def encrypt_for_federation(
    plaintext: bytes,
    federation_public_key: X25519PublicKey,
    epoch: int
) -> EncryptedPayload:
    """Encrypt content for federation members."""
    
    # 1. Generate ephemeral keypair
    ephemeral_private, ephemeral_public = generate_x25519_keypair()
    
    # 2. Derive shared secret
    shared_secret = x25519_dh(ephemeral_private, federation_public_key)
    
    # 3. Derive symmetric key via HKDF
    symmetric_key = hkdf_sha256(
        shared_secret,
        salt=f"valence-fed-{epoch}".encode(),
        info=b"content-encryption",
        length=32
    )
    
    # 4. Encrypt with XChaCha20-Poly1305
    nonce = random_bytes(24)
    ciphertext, tag = xchacha20poly1305_encrypt(
        symmetric_key,
        nonce,
        plaintext,
        associated_data=epoch.to_bytes(8, 'big')
    )
    
    return EncryptedPayload(
        epoch=epoch,
        ephemeral_public=ephemeral_public,
        nonce=nonce,
        ciphertext=ciphertext,
        tag=tag
    )
```

```python
def decrypt_federation_content(
    payload: EncryptedPayload,
    member_key_share: bytes,
    other_shares: List[bytes],  # From threshold reconstruction
    federation_id: UUID
) -> bytes:
    """Decrypt content using member's key share."""
    
    # 1. Reconstruct federation private key from shares
    federation_private_key = shamir_reconstruct(
        [member_key_share] + other_shares
    )
    
    # 2. Derive shared secret
    shared_secret = x25519_dh(federation_private_key, payload.ephemeral_public)
    
    # 3. Derive symmetric key
    symmetric_key = hkdf_sha256(
        shared_secret,
        salt=f"valence-fed-{payload.epoch}".encode(),
        info=b"content-encryption",
        length=32
    )
    
    # 4. Decrypt
    plaintext = xchacha20poly1305_decrypt(
        symmetric_key,
        payload.nonce,
        payload.ciphertext,
        payload.tag,
        associated_data=payload.epoch.to_bytes(8, 'big')
    )
    
    return plaintext
```

#### Key Rotation Process

```python
def rotate_federation_key(
    federation: Federation,
    reason: str
) -> KeyRotationResult:
    """Rotate federation group key."""
    
    new_epoch = federation.group_key.epoch + 1
    
    # 1. Generate new keypair
    new_private, new_public = generate_x25519_keypair()
    
    # 2. Split into shares (Shamir's Secret Sharing)
    active_members = [m for m in federation.members if m.status == 'active']
    threshold = compute_threshold(len(active_members))  # e.g., majority
    
    shares = shamir_split(
        secret=new_private,
        threshold=threshold,
        num_shares=len(active_members)
    )
    
    # 3. Encrypt each share to the member's personal key
    member_shares = []
    for member, share in zip(active_members, shares):
        encrypted_share = encrypt_to_member(share, member.encryption_key)
        member_shares.append(MemberKeyShare(
            member_id=member.id,
            epoch=new_epoch,
            share=encrypted_share
        ))
    
    # 4. Update federation key
    federation.group_key = GroupKey(
        epoch=new_epoch,
        public_key=new_public,
        threshold=threshold,
        total_shares=len(active_members)
    )
    
    # 5. Distribute shares (members fetch on next access)
    for ms in member_shares:
        store_key_share(ms)
    
    return KeyRotationResult(
        success=True,
        new_epoch=new_epoch,
        members_updated=len(member_shares)
    )
```

#### Forward Secrecy

When a member departs:
1. Immediate key rotation
2. New epoch key created without departed member's share
3. Content encrypted with new epoch is inaccessible to departed
4. Historical content (old epochs) may still be readable if member cached shares

**Mitigation for historical access:**
- Option A: Re-encrypt historical content with new key (expensive)
- Option B: Accept that departed members may retain historical access
- Option C: Time-limited key caching with forced refresh

---

## 4. Plausible Deniability

### The Problem

A member may be compelled to reveal their contributions:
- Legal subpoena
- Coercive threat
- Social pressure

We want members to credibly deny having shared specific content.

### The Solution: Decoy Beliefs

Members can generate **decoy beliefs** that are cryptographically indistinguishable from real beliefs but contain false or meaningless content.

#### Decoy Generation

```python
def share_with_deniability(
    real_belief: Belief,
    federation_id: UUID,
    decoy_count: int = 3
) -> List[SharedBelief]:
    """Share a belief with plausible deniability via decoys."""
    
    shared = []
    
    # 1. Share the real belief
    real_shared = share_to_federation(real_belief, federation_id)
    shared.append(real_shared)
    
    # 2. Generate and share decoys
    for _ in range(decoy_count):
        decoy = generate_decoy_belief(real_belief)
        decoy_shared = share_to_federation(decoy, federation_id, is_decoy=True)
        shared.append(decoy_shared)
    
    # 3. Shuffle so order doesn't reveal which is real
    random.shuffle(shared)
    
    return shared

def generate_decoy_belief(template: Belief) -> Belief:
    """Generate a decoy belief similar to template but with different content."""
    return Belief(
        id=generate_uuid(),
        content=generate_plausible_content(template.domains),  # LLM-generated
        confidence=perturb_confidence(template.confidence),
        derivation=generate_plausible_derivation(template.derivation.type),
        domains=template.domains,
        # ... other fields
    )
```

#### Decoy Detection Prevention

Decoys must be indistinguishable:

1. **Timing:** All beliefs (real + decoys) shared in single batch
2. **Size:** Pad to uniform size before encryption
3. **Metadata:** Decoys have realistic metadata
4. **Aggregation:** Decoys excluded from aggregation via secret flag

```python
# Decoy exclusion from aggregation (known only to system)
def aggregate_topic(
    shared_beliefs: List[SharedBelief],
    decoy_markers: Set[UUID]  # Stored separately, never exposed
) -> AggregatedBelief:
    real_beliefs = [b for b in shared_beliefs if b.id not in decoy_markers]
    return compute_aggregate(real_beliefs)
```

#### Denial Protocol

When compelled to reveal:

```
Compeller: "Did you share belief X?"

Member: "I cannot confirm or deny. My sharing protocol uses decoy 
beliefs. Any belief you observe may be a decoy that I generated 
but do not actually hold. The system is designed so that I cannot 
prove which of my shares are genuine, even to myself after time 
passes."

[Technical truth: member's client doesn't store which were decoys]
```

**Key property:** Member genuinely cannot prove which shares are real because:
- Decoy flag is stored server-side, encrypted
- Member's client discards the real/decoy distinction after sharing
- Both real and decoy shares have valid cryptographic proofs of membership

---

## 5. Audit Without Exposure

### The Problem

Need to verify:
- Federation rules are being followed
- No unauthorized access occurring
- Aggregation is computed correctly

Without exposing:
- Individual contributions
- Member identities
- Belief contents

### The Solution: Zero-Knowledge Audit Proofs

#### Audit Properties

| Audit Question | ZK Proof Type |
|----------------|--------------|
| "Did k contributors produce this aggregate?" | Range proof on count |
| "Are all contributors valid members?" | Set membership proof |
| "Was noise correctly applied?" | Commitment verification |
| "Is the aggregate computation correct?" | Computation proof |

#### Implementation: Commitment Schemes

**Pedersen Commitments for Contributions:**

```python
def commit_contribution(
    contribution: Belief,
    blinding_factor: bytes
) -> Commitment:
    """Create a hiding commitment to a contribution."""
    # Pedersen commitment: C = g^m × h^r
    m = hash_belief(contribution)
    r = blinding_factor
    return pedersen_commit(m, r)

def verify_aggregate_commitment(
    aggregate: AggregatedBelief,
    contribution_commitments: List[Commitment]
) -> bool:
    """Verify aggregate was correctly computed from commitments."""
    # Homomorphic property: sum of commitments = commitment to sum
    expected_commitment = sum_commitments(contribution_commitments)
    actual_commitment = aggregate.commitment
    return expected_commitment == actual_commitment
```

#### Audit Log Structure

```typescript
interface AuditEntry {
  id: UUID
  timestamp: timestamp
  
  // What happened (encrypted description)
  action: AuditAction
  
  // Who did it (commitment, not identity)
  actor_commitment: Commitment
  
  // Proof it was authorized
  authorization_proof: ZKProof
  
  // Proof of correct execution
  execution_proof: ZKProof
  
  // Link to previous entry (hash chain)
  prev_hash: bytes
}

enum AuditAction {
  MEMBER_JOINED
  MEMBER_DEPARTED
  BELIEF_SHARED
  AGGREGATE_PUBLISHED
  QUERY_EXECUTED
  KEY_ROTATED
  CONFIG_CHANGED
}
```

#### Audit Queries

```python
def audit_aggregate(
    aggregate_id: UUID,
    auditor_key: AuditorKey
) -> AuditReport:
    """Audit an aggregate without seeing individual contributions."""
    
    aggregate = get_aggregate(aggregate_id)
    audit_entries = get_audit_entries(aggregate_id)
    
    report = AuditReport()
    
    # 1. Verify contributor count
    count_proof = aggregate.contributor_count_proof
    report.contributor_count_valid = verify_range_proof(
        count_proof,
        min_value=aggregate.min_k,
        max_value=None
    )
    
    # 2. Verify all contributors were members
    membership_proofs = [e.authorization_proof for e in audit_entries 
                         if e.action == BELIEF_SHARED]
    report.all_members_valid = all(
        verify_membership_proof(p, aggregate.federation_id)
        for p in membership_proofs
    )
    
    # 3. Verify aggregation computation
    report.aggregation_valid = verify_computation_proof(
        aggregate.computation_proof,
        aggregate.commitment
    )
    
    # 4. Verify differential privacy noise
    report.privacy_valid = verify_noise_proof(
        aggregate.noise_proof,
        aggregate.privacy_config
    )
    
    return report
```

#### Auditor Roles

```typescript
interface Auditor {
  id: DID
  role: AuditorRole
  
  // What they can see
  access_level: AuditAccessLevel
  
  // Cryptographic capability
  audit_key: AuditorKey
}

enum AuditorRole {
  INTERNAL_AUDITOR      // Federation-appointed
  EXTERNAL_AUDITOR      // Third-party verifier
  REGULATORY_AUDITOR    // Compliance verification
  MEMBER_AUDITOR        // Any member can audit
}

enum AuditAccessLevel {
  PROOFS_ONLY           // Can verify proofs, see nothing else
  METADATA              // Proofs + non-sensitive metadata
  AGGREGATES            // Proofs + metadata + aggregate values
  FULL                  // Everything except individual contributions
}
```

---

## Privacy Configuration

### Federation-Level Settings

```typescript
interface FederationPrivacyConfig {
  // Differential privacy
  differential_privacy: {
    enabled: boolean                 // Default: true
    epsilon: float                   // Default: 1.0
    delta: float                     // Default: 1e-6
    budget_reset_period: duration    // Default: '1d'
  }
  
  // k-Anonymity
  k_anonymity: {
    k: uint8                         // Default: 5
    suppress_below_k: boolean        // Default: true
    cluster_for_anonymity: boolean   // Default: true
  }
  
  // Encryption
  encryption: {
    algorithm: 'xchacha20-poly1305' | 'aes-256-gcm'
    key_rotation_period: duration    // Default: '30d'
    key_rotation_on_departure: boolean // Default: true
  }
  
  // Plausible deniability
  deniability: {
    enabled: boolean                 // Default: false (opt-in)
    decoy_ratio: float               // Default: 3.0 (3 decoys per real)
    auto_decoy: boolean              // Default: false
  }
  
  // Audit
  audit: {
    enabled: boolean                 // Default: true
    proof_type: 'pedersen' | 'bulletproof' | 'snark'
    retention_period: duration       // Default: '365d'
  }
  
  // Membership privacy
  membership: {
    hide_member_list: boolean        // Default: true
    hide_member_count: boolean       // Default: false
    noise_member_count: boolean      // Default: true
  }
}
```

### Per-Share Settings

```typescript
interface SharePrivacyOptions {
  anonymous: boolean                 // Default: true
  use_decoy: boolean                 // Default: per-federation config
  decoy_count: uint8                 // Default: per-federation config
  
  // Additional protections
  delay_sharing: duration            // Add random delay
  batch_with_others: boolean         // Wait to batch (timing attack mitigation)
}
```

---

## Privacy Guarantees Matrix

| Threat | Mechanism | Guarantee |
|--------|-----------|-----------|
| Identify contributor from aggregate | Differential privacy | ε-indistinguishability |
| Re-identify from small group | k-anonymity | Cannot identify among k |
| External reads content | Group encryption | Computational security |
| Departed member reads future | Key rotation | Forward secrecy |
| Compelled to prove contribution | Plausible deniability | Cannot prove (decoys) |
| Admin sees individual shares | ZK audit | Verification without exposure |
| Timing correlation | Random delays + batching | Timing independence |
| Traffic analysis | Uniform message sizes | Size independence |

---

## Privacy Limitations

### What We Cannot Protect Against

1. **Global adversary with all keys**: If adversary obtains all member key shares, content is decryptable
2. **Malicious majority**: If >50% of members collude, they can potentially de-anonymize
3. **Side channels**: Timing attacks, power analysis, etc. at implementation level
4. **Metadata leakage**: Federation membership may be partially observable
5. **Long-term cryptographic breaks**: Future quantum computers or algorithmic advances

### Honest-but-Curious Model

Our privacy guarantees assume:
- Federation infrastructure executes protocols correctly
- Individual members may try to learn about others
- No majority collusion among members
- Cryptographic primitives remain secure

### Active Adversary Considerations

For stronger security against active adversaries:
- Use verifiable encryption (prove ciphertext is well-formed)
- Implement threshold decryption (multiple parties must cooperate)
- Add redundancy for Byzantine fault tolerance

---

## Implementation Checklist

- [ ] Implement Gaussian noise mechanism with proper σ calibration
- [ ] Privacy accountant with budget tracking and reset
- [ ] k-anonymity enforcement at publication and query time
- [ ] Topic clustering for below-threshold anonymity
- [ ] X25519 group key generation and distribution
- [ ] Shamir secret sharing for threshold reconstruction
- [ ] Key rotation protocol with forward secrecy
- [ ] Decoy belief generation (LLM-assisted)
- [ ] Decoy storage with separation from real beliefs
- [ ] Pedersen commitments for contributions
- [ ] ZK proofs for audit queries
- [ ] Audit log with hash chain integrity
- [ ] Random delay and batching for timing protection
- [ ] Uniform padding for message sizes

---

*"Privacy is not just a feature—it's the foundation of trust in collective knowledge."*
