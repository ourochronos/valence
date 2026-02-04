# Identity & Cryptography Specification

*Cryptographic foundation for decentralized epistemic identity.*

---

## Overview

Every agent in Valence has a cryptographic identity that:
- **Authenticates** beliefs and actions (signatures)
- **Enables** private communication (encryption)
- **Carries** portable reputation across contexts
- **Works** without any central authority

---

## Core Data Structures

### AgentIdentity

```typescript
AgentIdentity {
  // Immutable core
  id: DID                           // did:valence:<fingerprint>
  signing_key: Ed25519PublicKey     // 32 bytes, for signatures
  encryption_key: X25519PublicKey   // 32 bytes, for encryption (derived or separate)
  created_at: timestamp
  
  // Mutable metadata
  reputation: ReputationScore {
    overall: 0.0-1.0
    by_domain: Map<string, float>
    verification_count: uint64
    discrepancy_finds: uint64
    stake_at_risk: float
  }
  
  // Optional profile
  metadata: {
    display_name?: string           // Human-readable name
    avatar_hash?: bytes             // Content-addressed avatar
    domains?: string[]              // Claimed expertise areas
    endpoints?: string[]            // Where to reach this agent
    extensions?: Map<string, any>   // Platform-specific data
  }
  
  // Key history (for rotation)
  previous_keys: KeyRotation[]
}
```

### Key Types

#### Ed25519 — Signing
- **Purpose**: Sign beliefs, verify authorship, prove identity
- **Why**: Fast, small signatures (64 bytes), widely supported, deterministic
- **Key size**: 32-byte secret, 32-byte public

#### X25519 — Encryption
- **Purpose**: Encrypt data for specific recipients, key agreement
- **Why**: Efficient Diffie-Hellman for establishing shared secrets
- **Key size**: 32-byte secret, 32-byte public
- **Derivation**: Can derive from Ed25519 (same curve) or generate separately

### Key Hierarchy

```
Master Seed (32 bytes, BIP39 mnemonic optional)
    │
    ├── Identity Key (Ed25519)
    │   └── Derives: Agent DID fingerprint
    │
    ├── Encryption Key (X25519)
    │   └── Can derive from Identity Key or separate
    │
    └── Rotation Keys (sequence)
        └── Each rotation: new signing key, linked to previous
```

---

## Decentralized Identifiers (DIDs)

Valence uses a custom DID method for maximum portability:

```
did:valence:<multibase-encoded-fingerprint>

Example:
did:valence:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
```

### DID Document Structure

```json
{
  "@context": ["https://www.w3.org/ns/did/v1", "https://valence.network/v1"],
  "id": "did:valence:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK",
  "verificationMethod": [
    {
      "id": "did:valence:z6Mk...#signing-key",
      "type": "Ed25519VerificationKey2020",
      "controller": "did:valence:z6Mk...",
      "publicKeyMultibase": "z6Mk..."
    },
    {
      "id": "did:valence:z6Mk...#encryption-key",
      "type": "X25519KeyAgreementKey2020",
      "controller": "did:valence:z6Mk...",
      "publicKeyMultibase": "z6LS..."
    }
  ],
  "authentication": ["did:valence:z6Mk...#signing-key"],
  "keyAgreement": ["did:valence:z6Mk...#encryption-key"],
  "service": [
    {
      "id": "did:valence:z6Mk...#valence-node",
      "type": "ValenceNode",
      "serviceEndpoint": "https://node.example.com/valence"
    }
  ]
}
```

### Why DIDs?

1. **No registry required** — Self-certifying, identity = public key
2. **Portable** — Works anywhere, no platform lock-in
3. **Resolvable** — Standard mechanism for discovering keys and endpoints
4. **Extensible** — Add services, keys, metadata as needed

---

## Identity Lifecycle

### 1. Creation

```
generate_seed() → 32 random bytes
    │
    ├── derive_identity_key(seed, path=0) → Ed25519 keypair
    │
    ├── derive_encryption_key(seed, path=1) → X25519 keypair
    │   OR derive_from_ed25519(identity_key) → X25519 keypair
    │
    └── construct_did(identity_public_key) → did:valence:...

Output: AgentIdentity + PrivateKeyBundle
```

**Initialization state:**
- Reputation: `overall: 0.5` (neutral starting point)
- No domain reputation
- No trust relationships
- Empty key history

### 2. Active Use

During normal operation:
- Sign beliefs with identity key
- Encrypt messages with recipient's encryption key
- Build reputation through verification
- Accumulate trust relationships

### 3. Key Rotation

Keys should rotate:
- **Periodically** (recommended: yearly)
- **On compromise suspicion**
- **When upgrading algorithms**

```typescript
KeyRotation {
  old_key: Ed25519PublicKey
  new_key: Ed25519PublicKey
  timestamp: uint64
  proof: {
    // Signed by OLD key, proving control
    old_signature: bytes
    // Signed by NEW key, proving possession
    new_signature: bytes
  }
  reason?: string  // "scheduled" | "compromise" | "upgrade"
}
```

**Rotation preserves:**
- DID (derived from first key, but document updated)
- Reputation
- Trust relationships
- Belief authorship (old signatures remain valid)

**Rotation chain:**
```
Key_0 → signs rotation to → Key_1 → signs rotation to → Key_2
  ↑                           ↑                           ↑
 (original)              (still valid)              (current)
```

### 4. Revocation

For compromised keys or identity retirement:

```typescript
Revocation {
  identity: DID
  revoked_key: Ed25519PublicKey
  timestamp: uint64
  signature: bytes  // Signed by a DIFFERENT valid key, or by revoked key itself
  successor?: DID   // Optional: "this identity continues as..."
}
```

**Revocation publishing:**
- Include in DID document
- Gossip to known peers
- Federation nodes should propagate

**Post-revocation:**
- Old signatures before revocation: **still valid** (they were valid when made)
- New signatures with revoked key: **invalid**
- Reputation: frozen (not transferred unless explicit successor)

---

## Portable Identity Format

For moving identities across platforms:

### Export Format (encrypted)

```typescript
IdentityExport {
  version: 1
  encrypted_bundle: {
    // Encrypted with password-derived key (Argon2id + XChaCha20-Poly1305)
    ciphertext: bytes
    nonce: bytes
    salt: bytes       // For Argon2id
    params: {
      memory: 65536   // 64 MB
      iterations: 3
      parallelism: 4
    }
  }
  public_hint: {
    did: string
    created_at: timestamp
    key_count: number  // How many keys in rotation history
  }
}

// Decrypted bundle contains:
PrivateKeyBundle {
  seed?: bytes                    // If seed-based derivation used
  identity_key: Ed25519PrivateKey
  encryption_key: X25519PrivateKey
  rotation_history: KeyRotation[]
  metadata: any
}
```

### Import Process

1. User provides export file + password
2. Derive decryption key via Argon2id
3. Decrypt bundle
4. Verify keys match public hint
5. Reconstruct AgentIdentity locally

### Platform Bridging

When an identity exists on multiple platforms:

```typescript
PlatformBinding {
  valence_did: DID
  platform: string              // "moltbook" | "discord" | "twitter" | ...
  platform_id: string           // Platform-specific identifier
  proof: {
    // Signed by Valence identity key
    valence_signature: bytes
    // Platform-specific verification (e.g., signed message posted publicly)
    platform_verification?: string
  }
  created_at: timestamp
}
```

This allows:
- Reputation aggregation across platforms
- Identity verification without central authority
- Selective disclosure (share bindings you want known)

---

## Reputation Integration

Reputation is **attached to identity, not stored in it**:

```typescript
// Identity stores current snapshot
AgentIdentity.reputation: ReputationScore

// Full history lives in reputation layer
ReputationHistory {
  identity: DID
  events: ReputationEvent[]  // Verifications, discrepancies, etc.
  checkpoints: {             // Periodic snapshots for efficiency
    timestamp: uint64
    score: ReputationScore
    merkle_root: bytes       // For verification
  }[]
}
```

**Key principle**: Identity layer provides authentication. Reputation layer provides the scores. They're coupled by DID but managed separately.

---

## Trust Graph

Each agent maintains a personal trust graph:

```typescript
TrustGraph {
  owner: DID
  edges: Map<DID, TrustEdge>
}

TrustEdge {
  target: DID
  level: float           // 0.0 = no trust, 1.0 = full trust
  domains?: string[]     // "I trust them on X" vs general trust
  evidence?: string      // Why this trust level
  updated_at: timestamp
}
```

**Trust is:**
- Personal (each agent has their own)
- Asymmetric (A trusts B ≠ B trusts A)
- Domain-specific (optional)
- Exportable (can share your trust graph)

---

## Encoding Standards

### Keys
- **Public keys**: Multibase-encoded (base58btc, prefix `z`)
- **Signatures**: Raw 64-byte Ed25519, base64url when serialized
- **Encrypted data**: Base64url or raw bytes depending on context

### DIDs
- **Format**: `did:valence:<multibase-fingerprint>`
- **Fingerprint**: SHA-256 of public key, first 16 bytes, multibase-encoded

### Timestamps
- **Format**: Unix epoch milliseconds (uint64)
- **Timezone**: Always UTC

---

## Implementation Notes

### Recommended Libraries

| Language | Signing | Encryption | DIDs |
|----------|---------|------------|------|
| Python | `pynacl` | `pynacl` | custom |
| Rust | `ed25519-dalek` | `x25519-dalek` | `did-key` |
| TypeScript | `@noble/ed25519` | `@noble/curves` | `did-resolver` |
| Go | `crypto/ed25519` | `x/crypto/curve25519` | custom |

### Storage Requirements

Per identity:
- Public data: ~500 bytes (keys + DID + metadata)
- Private data: ~200 bytes (private keys + seed)
- Rotation history: ~150 bytes per rotation

### Performance Targets

| Operation | Target |
|-----------|--------|
| Key generation | < 1ms |
| Signing | < 0.1ms |
| Verification | < 0.2ms |
| Encryption (1KB) | < 0.5ms |
| Decryption (1KB) | < 0.5ms |

---

## Summary

The identity model provides:

1. **Self-sovereign identity** via Ed25519 keypairs
2. **Portable DIDs** that work without registries
3. **Clean key rotation** that preserves history
4. **Flexible encryption** via X25519
5. **Platform bridging** for cross-context reputation
6. **Privacy by design** with selective disclosure

All without any central authority.

---

*"Your identity is your key. Your reputation is your history. Both travel with you."*
