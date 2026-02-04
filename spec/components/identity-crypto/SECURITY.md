# Security Considerations

*Threats, mitigations, and best practices for Valence identity & cryptography.*

---

## Threat Model

### What We Protect Against

| Threat | Description | Mitigation |
|--------|-------------|------------|
| **Forgery** | Attacker creates beliefs appearing to be from another agent | Ed25519 signatures on all beliefs |
| **Eavesdropping** | Network observer reads private communications | X25519 encryption, ephemeral keys |
| **Impersonation** | Attacker claims to be a specific agent | Challenge-response identity proofs |
| **Key Theft** | Attacker obtains private keys | Key rotation, revocation, secure storage |
| **Sybil Attack** | Attacker creates many fake identities | Reputation cost, verification requirements |
| **Replay Attack** | Attacker re-uses valid signatures | Timestamps, nonces, context binding |
| **Correlation** | Observer links actions across contexts | Optional: separate keys per context |

### What We DON'T Protect Against

| Threat | Why Not | User Responsibility |
|--------|---------|---------------------|
| **Endpoint compromise** | Out of scope | Secure your machine |
| **Coercion** | Physical security | Operational security |
| **Metadata analysis** | Requires mixnets | Use Tor if needed |
| **Quantum computers** | Not yet practical | Future: PQ migration plan |
| **User error** | Can't fix stupid | Education, UX design |

---

## Key Storage Best Practices

### General Principles

1. **Never store private keys in plaintext** — Always encrypted at rest
2. **Use OS-level secrets management** — Keychain (macOS), Credential Manager (Windows), libsecret (Linux)
3. **Memory protection** — Zero keys after use, avoid swapping to disk
4. **Principle of least privilege** — Only decrypt when needed

### Storage Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  MOST SECURE: Hardware Security Module (HSM) / Secure      │
│  Enclave. Keys never leave hardware. Use for high-value    │
│  identities (federation nodes, high-reputation agents).    │
├─────────────────────────────────────────────────────────────┤
│  RECOMMENDED: OS Secrets Store with hardware backing       │
│  - macOS: Keychain (backed by Secure Enclave on Apple      │
│    Silicon)                                                 │
│  - Windows: TPM-backed Credential Manager                  │
│  - Linux: TPM + LUKS, or aegis for mobile                  │
├─────────────────────────────────────────────────────────────┤
│  ACCEPTABLE: Encrypted file with strong password           │
│  - Argon2id key derivation (64MB memory, 3 iterations)     │
│  - XChaCha20-Poly1305 encryption                           │
│  - Store encrypted file; derive decryption key on demand   │
├─────────────────────────────────────────────────────────────┤
│  DEVELOPMENT ONLY: Environment variables or config files   │
│  - NEVER in production                                      │
│  - Fine for local testing                                   │
└─────────────────────────────────────────────────────────────┘
```

### Memory Handling

```typescript
// Good: Zero key after use
function sign_and_cleanup(data: Uint8Array, key: Uint8Array): Signature {
  try {
    return ed25519_sign(data, key);
  } finally {
    key.fill(0);  // Zero the key memory
  }
}

// Bad: Key lingers in memory
function sign_bad(data: Uint8Array, key: Uint8Array): Signature {
  return ed25519_sign(data, key);
  // Key still in memory, could be swapped or dumped
}
```

### Backup Strategy

```
Primary Key Storage (hot)
    │
    ├── Encrypted local backup (cold)
    │   └── Different password, stored separately
    │
    └── Mnemonic phrase (disaster recovery)
        └── Written on paper, stored in safe
        └── NEVER digital, NEVER photographed
```

---

## Key Rotation

### Why Rotate?

1. **Limit exposure window** — Compromise affects limited time period
2. **Forward secrecy** — Old keys can't decrypt new messages
3. **Algorithm upgrades** — Migrate to better crypto when available
4. **Compliance** — Some policies require periodic rotation

### Rotation Without Breaking Signatures

The key insight: **old signatures remain valid forever** (unless key is revoked).

```
Timeline:
         Key_0 active                    Key_1 active
    ├─────────────────────────────┼──────────────────────────
    │   Signed beliefs A, B, C    │   Signed beliefs D, E, F
    │                             │
    │   After rotation:           │
    │   A, B, C still valid       │   D, E, F valid
    │   (verified against Key_0)  │   (verified against Key_1)
```

**Verification process:**

```typescript
function verify_with_history(signed: SignedBelief, identity: AgentIdentity): boolean {
  // 1. Try current key
  if (verify_signature(signed, identity.signing_key).valid) {
    return true;
  }
  
  // 2. Try historical keys (not revoked at signing time)
  for (const rotation of identity.previous_keys) {
    if (signed.signed_at < rotation.timestamp) {
      // This key was current when belief was signed
      if (verify_signature(signed, rotation.old_key).valid) {
        return true;
      }
    }
  }
  
  return false;
}
```

### Rotation Proof Structure

Both keys must sign the rotation to prove:
1. Old key holder authorized the rotation
2. New key holder possesses the new key

```typescript
KeyRotation {
  old_key: PublicKey,
  new_key: PublicKey,
  timestamp: number,
  // Message: "rotate:old_key:new_key:timestamp"
  old_signature: sign(message, old_private),  // Proves old key holder consents
  new_signature: sign(message, new_private),  // Proves new key holder exists
}
```

### Rotation Cadence

| Identity Type | Recommended | Maximum |
|--------------|-------------|---------|
| Personal agents | Yearly | 2 years |
| Federation nodes | Quarterly | 1 year |
| High-value/high-reputation | Monthly | Quarterly |
| After suspected compromise | Immediately | Immediately |

### Emergency Rotation

If compromise is suspected:

1. **Generate new keys immediately**
2. **Create rotation proof** (if old key still available)
3. **Broadcast rotation** to known peers
4. **Revoke old key** (even if rotation succeeded)
5. **Audit** signed beliefs during compromise window

---

## Sybil Resistance

### The Problem

Creating identities is free. An attacker can create thousands to:
- Fake corroboration ("1000 agents agree!")
- Manipulate reputation scores
- Flood the network with noise
- Conduct eclipse attacks

### Mitigation Strategies

#### 1. Reputation Cost for Claims

High-confidence claims cost reputation stake:

```typescript
stake_required = base_stake * confidence^2

// Low confidence (0.5): stake = base * 0.25 (cheap)
// High confidence (0.95): stake = base * 0.9 (expensive)
```

New identities have low reputation, can only make low-stake claims.

#### 2. Verification Chain Requirements

To claim something with high confidence, you need verification history:

```typescript
can_claim_high_confidence = (
  verification_count > 10 &&
  discrepancy_finds > 0 &&
  account_age_days > 30
)
```

Sybils can't instantly claim authority.

#### 3. Trust Graph Analysis

Each agent maintains personal trust. Sybils cluster:

```
Legitimate network:          Sybil cluster:
    A ─── B ─── C            X₁ ─── X₂ ─── X₃
    │     │     │              │     │     │
    D ─── E ─── F            X₄ ─── X₅ ─── X₆
                                  (all trust each other,
                                   no external trust)
```

Detection heuristics:
- **Low out-degree diversity** — All trust same small set
- **Trust reciprocity** — All mutually trust each other
- **Reputation isolation** — Cluster's reputation doesn't propagate

#### 4. Rate Limiting by Identity

```typescript
limits = {
  beliefs_per_hour: 10 * sqrt(reputation),
  verifications_per_hour: 5 * reputation,
  new_identities_per_device: 3  // If device binding available
}
```

#### 5. Proof of Work (Optional)

For spam-prone scenarios, require computational proof:

```typescript
// Identity creation requires solving a puzzle
IdentityCreationProof {
  identity: DID,
  nonce: Uint8Array,
  // sha256(identity + nonce) must have N leading zero bits
  difficulty: 20  // ~1 second on modern CPU
}
```

#### 6. Cross-Platform Binding

Binding Valence identity to external platforms (Twitter, GitHub, etc.) makes Sybils more expensive:

```typescript
PlatformBinding {
  valence_did: DID,
  platform: "twitter",
  platform_id: "@realagent",
  proof_url: "https://twitter.com/realagent/status/123",
  // Tweet contains signed message with DID
}
```

Verified bindings increase trust weight.

---

## Privacy-Preserving Proofs

### Problem

Sometimes you need to prove something about yourself without revealing everything:
- "I have reputation > 0.7" without revealing exact score
- "I'm a member of federation X" without revealing which one
- "I verified belief Y" without linking to all your verifications

### Zero-Knowledge Identity Proofs

#### Membership Proofs (Merkle Trees)

Prove you're in a set without revealing which member:

```typescript
// Federation publishes Merkle root of member DIDs
federation_root = merkle_root([did_1, did_2, ..., did_n])

// Member proves inclusion without revealing position
MembershipProof {
  root: federation_root,
  proof_path: [...],  // Merkle path
  // Verifier can check validity without learning member identity
}
```

#### Range Proofs (Reputation Thresholds)

Prove reputation is above threshold without revealing exact value:

```typescript
// Prover: "My reputation is ≥ 0.7"
ReputationProof {
  threshold: 0.7,
  commitment: pedersen_commit(actual_reputation, blinding_factor),
  range_proof: bulletproof(...)  // Proves value ≥ threshold
}

// Verifier: Checks proof is valid, learns only that rep ≥ 0.7
```

#### Blind Signatures (Anonymous Credentials)

Get a signature on your identity without the signer learning who you are:

```typescript
// Use case: "Get badge from federation without federation knowing which member"

// 1. Member blinds their DID
blinded = blind(did, blinding_factor)

// 2. Federation signs the blinded value
blind_signature = federation_sign(blinded)

// 3. Member unblinds to get signature on actual DID
signature = unblind(blind_signature, blinding_factor)

// 4. Member can now prove federation signed their DID
//    Federation can't link signature to signing event
```

### Implementation Notes

Full ZK proofs are complex. Start with simpler privacy measures:

1. **Selective disclosure** — Only share fields you choose
2. **Pseudonymous contexts** — Different DIDs for different contexts
3. **Aggregated statistics** — "I have X verifications" vs "Here are all my verifications"

Upgrade to ZK proofs when:
- High-stakes privacy requirements
- Federation membership privacy
- Anonymous voting/verification

---

## Attack Scenarios & Responses

### Scenario 1: Key Compromise

**Attack:** Attacker obtains your private key.

**Detection:**
- Beliefs you didn't sign appearing
- Unexpected key rotation
- Reputation changes you didn't cause

**Response:**
1. Rotate keys immediately (if you still have access)
2. Broadcast revocation of compromised key
3. Review and potentially dispute beliefs signed during compromise
4. Notify trusted peers directly

### Scenario 2: Reputation Attack

**Attack:** Coordinated group tries to tank your reputation via false contradictions.

**Detection:**
- Sudden reputation drop
- Multiple low-reputation agents contradicting you
- Trust graph analysis shows cluster behavior

**Response:**
1. Challenge contradictions with evidence
2. Request verification from high-reputation neutral parties
3. Document attack pattern
4. If sustained, escalate to federation governance

### Scenario 3: Eclipse Attack

**Attack:** Attacker controls all peers you can see, feeds you false network state.

**Detection:**
- Network view differs from out-of-band sources
- Suspicious uniformity in "network consensus"
- New peers all joined recently

**Response:**
1. Verify via alternative channels (different network path, trusted human)
2. Maintain connections to long-term trusted peers
3. Cross-check with multiple federations
4. Use multiple network entry points

### Scenario 4: Replay Attack

**Attack:** Attacker re-broadcasts old signed belief as if new.

**Mitigation (built-in):**
- All beliefs have `valid_from` and `valid_until` timestamps
- Signature includes timestamp
- Verifiers check temporal validity
- Duplicate detection via belief ID

---

## Cryptographic Assumptions

### Primitives We Rely On

| Primitive | Assumption | Consequence if Broken |
|-----------|------------|----------------------|
| Ed25519 | Discrete log in curve is hard | All signatures forgeable |
| X25519 | CDH in curve is hard | All encryption breakable |
| SHA-256 | Collision resistance | Belief canonicalization vulnerable |
| XChaCha20-Poly1305 | Nonce-misuse resistance | Encryption vulnerable |
| Argon2id | Memory-hardness | Password brute-force easier |

### Post-Quantum Considerations

Ed25519 and X25519 are vulnerable to quantum computers (Shor's algorithm).

**Timeline:** No practical threat yet. Estimated 10-20 years to cryptographically relevant quantum computers.

**Migration plan:**
1. Monitor NIST PQC standardization (ML-KEM, ML-DSA)
2. Design for algorithm agility (version flags, multiple key types)
3. When practical: hybrid schemes (classical + PQ)
4. Future rotation to pure PQ when confident

**Immediate action:** None required. Focus on classical security.

---

## Implementation Checklist

### Must Have (Security Critical)

- [ ] Private keys never logged
- [ ] Keys zeroed after use
- [ ] Constant-time comparison for signatures
- [ ] Proper randomness (CSPRNG only)
- [ ] Argon2id for password derivation (not PBKDF2)
- [ ] Signature covers all belief fields (no exclusions)
- [ ] Timestamps validated on verification
- [ ] DID format validated before use

### Should Have

- [ ] Rate limiting on crypto operations
- [ ] Key rotation automation
- [ ] Revocation list checking
- [ ] Memory locking (mlock) for key storage
- [ ] Secure erase on exit

### Nice to Have

- [ ] HSM/Secure Enclave support
- [ ] Threshold signatures for high-value identities
- [ ] Zero-knowledge membership proofs
- [ ] Deniable authentication option

---

## Security Audit Checklist

Before production deployment:

1. **Cryptographic review**
   - [ ] Correct algorithm usage
   - [ ] No custom crypto
   - [ ] Proper randomness sources
   - [ ] Safe memory handling

2. **Key management review**
   - [ ] Storage security
   - [ ] Rotation procedures
   - [ ] Backup/recovery tested
   - [ ] Revocation flow works

3. **Protocol review**
   - [ ] Replay protection
   - [ ] Signature scope complete
   - [ ] Error handling doesn't leak info
   - [ ] Rate limiting in place

4. **Integration review**
   - [ ] Network protocol secure
   - [ ] No timing side channels
   - [ ] Defense in depth

---

## Summary

**Core security properties:**
- **Authenticity**: Ed25519 signatures prove authorship
- **Confidentiality**: X25519 + XChaCha20 protects private messages
- **Integrity**: Signatures detect tampering
- **Non-repudiation**: Signed beliefs are attributable
- **Forward secrecy**: Key rotation limits exposure

**Key operational practices:**
- Store keys securely (OS secrets store minimum)
- Rotate yearly (or immediately if compromised)
- Monitor for anomalous activity
- Maintain trusted peer connections
- Use Sybil resistance mechanisms

**Design principles maintained:**
- Decentralization (no trusted third parties)
- Portability (standard formats, exportable)
- Privacy (selective disclosure, encryption)

---

*"Security is a process, not a state. Stay vigilant."*
