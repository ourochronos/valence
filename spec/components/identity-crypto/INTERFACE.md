# Cryptographic Operations Interface

*API surface for identity and cryptographic operations in Valence.*

---

## Overview

This document defines the interface for all cryptographic operations. Implementations should follow these signatures exactly to ensure interoperability.

---

## Type Definitions

```typescript
// Core types
type DID = string;                    // "did:valence:z6Mk..."
type Ed25519PublicKey = Uint8Array;   // 32 bytes
type Ed25519PrivateKey = Uint8Array;  // 32 bytes (or 64 with public appended)
type X25519PublicKey = Uint8Array;    // 32 bytes
type X25519PrivateKey = Uint8Array;   // 32 bytes
type Signature = Uint8Array;          // 64 bytes
type Seed = Uint8Array;               // 32 bytes

// Composite types
interface AgentIdentity {
  id: DID;
  signing_key: Ed25519PublicKey;
  encryption_key: X25519PublicKey;
  created_at: number;
  reputation: ReputationScore;
  metadata: IdentityMetadata;
  previous_keys: KeyRotation[];
}

interface PrivateKeyBundle {
  seed?: Seed;
  identity_key: Ed25519PrivateKey;
  encryption_key: X25519PrivateKey;
}

interface KeyPair<Pub, Priv> {
  public_key: Pub;
  private_key: Priv;
}

interface SignedBelief {
  belief: Belief;
  signature: Signature;
  signer: DID;
  signed_at: number;
}

interface EncryptedData {
  ciphertext: Uint8Array;
  nonce: Uint8Array;           // 24 bytes for XChaCha20
  ephemeral_key?: X25519PublicKey;  // For one-shot encryption
  recipient: DID;
}

interface IdentityProof {
  challenge: Uint8Array;
  response: Signature;
  signer: DID;
  timestamp: number;
}

interface KeyRotation {
  old_key: Ed25519PublicKey;
  new_key: Ed25519PublicKey;
  timestamp: number;
  old_signature: Signature;
  new_signature: Signature;
  reason?: string;
}
```

---

## Identity Operations

### generate_identity

Create a new agent identity from scratch.

```typescript
function generate_identity(options?: {
  seed?: Seed;                // Optional: derive from existing seed
  metadata?: IdentityMetadata;
}): {
  identity: AgentIdentity;
  private_keys: PrivateKeyBundle;
}
```

**Behavior:**
1. Generate or use provided 32-byte seed
2. Derive Ed25519 keypair for signing
3. Derive X25519 keypair for encryption
4. Compute DID from signing public key
5. Initialize reputation to neutral (0.5)

**Example:**
```typescript
const { identity, private_keys } = generate_identity({
  metadata: { display_name: "Agent Alpha" }
});

console.log(identity.id);  // did:valence:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
```

---

### generate_identity_from_mnemonic

Create identity from BIP39 mnemonic phrase (for human-memorable backup).

```typescript
function generate_identity_from_mnemonic(
  mnemonic: string,           // 12 or 24 words
  passphrase?: string         // Optional BIP39 passphrase
): {
  identity: AgentIdentity;
  private_keys: PrivateKeyBundle;
}
```

**Behavior:**
1. Validate mnemonic (BIP39 wordlist)
2. Derive seed via BIP39
3. Proceed as `generate_identity({ seed })`

**Example:**
```typescript
const mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
const { identity, private_keys } = generate_identity_from_mnemonic(mnemonic);
```

---

### export_identity

Export identity for backup or transfer.

```typescript
function export_identity(
  identity: AgentIdentity,
  private_keys: PrivateKeyBundle,
  password: string,
  options?: {
    argon2_memory?: number;    // Default: 65536 (64MB)
    argon2_iterations?: number; // Default: 3
    argon2_parallelism?: number; // Default: 4
  }
): IdentityExport
```

**Returns:** Encrypted bundle safe for storage/transfer.

---

### import_identity

Restore identity from export.

```typescript
function import_identity(
  exported: IdentityExport,
  password: string
): {
  identity: AgentIdentity;
  private_keys: PrivateKeyBundle;
} | { error: "invalid_password" | "corrupted_data" | "version_unsupported" }
```

---

### rotate_keys

Rotate to new signing key while preserving identity.

```typescript
function rotate_keys(
  identity: AgentIdentity,
  current_private_key: Ed25519PrivateKey,
  reason?: string
): {
  updated_identity: AgentIdentity;
  new_private_keys: PrivateKeyBundle;
  rotation_proof: KeyRotation;
}
```

**Behavior:**
1. Generate new Ed25519 keypair
2. Generate new X25519 keypair
3. Create rotation proof (signed by both old and new keys)
4. Append to `previous_keys` history
5. DID remains the same (points to rotation chain)

---

### revoke_key

Revoke a compromised or retired key.

```typescript
function revoke_key(
  key_to_revoke: Ed25519PublicKey,
  signing_key: Ed25519PrivateKey,  // Must be different valid key, or the key itself
  options?: {
    successor_did?: DID;
    reason?: string;
  }
): Revocation
```

---

## Signing Operations

### sign_belief

Sign a belief to prove authorship.

```typescript
function sign_belief(
  belief: Belief,
  private_key: Ed25519PrivateKey,
  signer_did: DID
): SignedBelief
```

**Behavior:**
1. Canonicalize belief (deterministic JSON)
2. Compute SHA-256 hash of canonical form
3. Sign hash with Ed25519
4. Return `SignedBelief` with signature attached

**Canonicalization rules:**
- Sort object keys alphabetically
- No whitespace
- UTF-8 encoding
- Numbers as-is (no scientific notation)
- Null values preserved, undefined omitted

**Example:**
```typescript
const belief: Belief = {
  id: "550e8400-e29b-41d4-a716-446655440000",
  content: "Water boils at 100°C at sea level",
  confidence: { overall: 0.95, source_reliability: 0.9 },
  domains: ["physics", "chemistry"],
  // ...
};

const signed = sign_belief(belief, private_keys.identity_key, identity.id);
// signed.signature is 64 bytes
```

---

### sign_data

Generic signing for arbitrary data.

```typescript
function sign_data(
  data: Uint8Array,
  private_key: Ed25519PrivateKey
): Signature
```

**Behavior:**
1. Sign data directly (or hash if > 256 bytes)
2. Return 64-byte signature

---

### verify_signature

Verify a signed belief's authenticity.

```typescript
function verify_signature(
  signed_belief: SignedBelief,
  public_key: Ed25519PublicKey
): {
  valid: boolean;
  error?: "signature_invalid" | "belief_modified" | "key_mismatch";
}
```

**Behavior:**
1. Re-canonicalize belief content
2. Recompute hash
3. Verify Ed25519 signature against hash and public key

**Example:**
```typescript
const result = verify_signature(signed_belief, sender_identity.signing_key);
if (result.valid) {
  console.log("Belief is authentic");
}
```

---

### verify_signature_with_rotation

Verify signature accounting for key rotation history.

```typescript
function verify_signature_with_rotation(
  signed_belief: SignedBelief,
  identity: AgentIdentity  // Includes rotation history
): {
  valid: boolean;
  signing_key_index: number;  // Which key in history signed this
  error?: string;
}
```

**Behavior:**
1. Try current key first
2. If fails, try each key in `previous_keys`
3. Check revocation status of matching key
4. Return which key (if any) produced valid signature

---

## Encryption Operations

### encrypt_for

Encrypt data for a specific recipient.

```typescript
function encrypt_for(
  data: Uint8Array,
  recipient_public_key: X25519PublicKey,
  options?: {
    sender_private_key?: X25519PrivateKey;  // For authenticated encryption
    additional_data?: Uint8Array;            // AAD for AEAD
  }
): EncryptedData
```

**Behavior (ephemeral mode, default):**
1. Generate ephemeral X25519 keypair
2. Perform X25519 key agreement: `shared = X25519(ephemeral_private, recipient_public)`
3. Derive symmetric key: `key = HKDF-SHA256(shared, salt="valence-encrypt", info=context)`
4. Encrypt with XChaCha20-Poly1305
5. Return ciphertext + nonce + ephemeral public key

**Behavior (authenticated mode, if sender_private_key provided):**
1. Use sender's key instead of ephemeral
2. Recipient can verify sender identity

**Example:**
```typescript
const message = new TextEncoder().encode("Secret belief sharing request");
const encrypted = encrypt_for(message, recipient.encryption_key);
// Send encrypted.ciphertext, encrypted.nonce, encrypted.ephemeral_key
```

---

### encrypt_for_multiple

Encrypt for multiple recipients efficiently.

```typescript
function encrypt_for_multiple(
  data: Uint8Array,
  recipients: X25519PublicKey[]
): {
  ciphertext: Uint8Array;          // Data encrypted once with random key
  nonce: Uint8Array;
  wrapped_keys: {                  // Key wrapped for each recipient
    recipient_key: X25519PublicKey;
    wrapped_key: Uint8Array;
  }[];
}
```

**Behavior:**
1. Generate random symmetric key
2. Encrypt data once with that key
3. Wrap symmetric key for each recipient using `encrypt_for`

---

### decrypt

Decrypt data intended for you.

```typescript
function decrypt(
  encrypted: EncryptedData,
  recipient_private_key: X25519PrivateKey
): Uint8Array | { error: "decryption_failed" | "authentication_failed" }
```

**Behavior:**
1. Perform X25519 key agreement with ephemeral key (or sender key)
2. Derive symmetric key via HKDF
3. Decrypt with XChaCha20-Poly1305
4. Verify Poly1305 tag
5. Return plaintext or error

**Example:**
```typescript
const decrypted = decrypt(encrypted_data, private_keys.encryption_key);
if (!(decrypted instanceof Uint8Array)) {
  console.error("Decryption failed:", decrypted.error);
} else {
  const message = new TextDecoder().decode(decrypted);
}
```

---

## Identity Proofs

### prove_identity

Prove you control an identity without revealing private key.

```typescript
function prove_identity(
  challenge: Uint8Array,       // Random challenge from verifier
  private_key: Ed25519PrivateKey,
  identity_did: DID
): IdentityProof
```

**Behavior:**
1. Create proof message: `"valence-proof-v1:" + DID + ":" + timestamp + ":" + challenge`
2. Sign with private key
3. Return proof (verifiable without private key)

**Example:**
```typescript
// Verifier generates challenge
const challenge = crypto.getRandomValues(new Uint8Array(32));

// Prover creates proof
const proof = prove_identity(challenge, private_keys.identity_key, identity.id);

// Verifier checks proof
const valid = verify_identity_proof(proof, identity.signing_key, challenge);
```

---

### verify_identity_proof

Verify an identity proof.

```typescript
function verify_identity_proof(
  proof: IdentityProof,
  public_key: Ed25519PublicKey,
  original_challenge: Uint8Array,
  options?: {
    max_age_ms?: number;  // Reject proofs older than this (default: 300000 = 5 min)
  }
): {
  valid: boolean;
  error?: "signature_invalid" | "challenge_mismatch" | "expired" | "did_mismatch";
}
```

---

### create_challenge

Generate a challenge for identity verification.

```typescript
function create_challenge(): {
  challenge: Uint8Array;  // 32 random bytes
  created_at: number;
  expires_at: number;     // 5 minutes from creation
}
```

---

## Utility Functions

### did_from_public_key

Compute DID from Ed25519 public key.

```typescript
function did_from_public_key(public_key: Ed25519PublicKey): DID
```

**Behavior:**
1. SHA-256 hash of public key
2. Take first 16 bytes
3. Multibase encode (base58btc)
4. Prefix with "did:valence:"

---

### resolve_did

Resolve DID to identity information.

```typescript
function resolve_did(
  did: DID,
  options?: {
    network?: "local" | "federation" | "global";
    timeout_ms?: number;
  }
): Promise<AgentIdentity | { error: "not_found" | "timeout" | "invalid_did" }>
```

**Behavior:**
1. Parse DID
2. Query local cache first
3. If not found and network allowed, query peers/federation
4. Return identity or error

---

### derive_x25519_from_ed25519

Derive X25519 key from Ed25519 key (same curve, different encoding).

```typescript
function derive_x25519_from_ed25519(
  ed25519_private: Ed25519PrivateKey
): X25519PrivateKey

function derive_x25519_public_from_ed25519(
  ed25519_public: Ed25519PublicKey
): X25519PublicKey
```

**Note:** This is a one-way derivation. Use when you want a single seed to control both keys.

---

### canonicalize

Canonicalize data for signing.

```typescript
function canonicalize(data: any): Uint8Array
```

**Behavior:**
1. Convert to JSON with sorted keys
2. Encode as UTF-8
3. Return bytes

---

## Error Handling

All functions that can fail return either:
- The success value directly, OR
- An object with an `error` field describing the failure

```typescript
// Pattern 1: Union return type
function decrypt(...): Uint8Array | { error: string }

// Usage
const result = decrypt(encrypted, key);
if ('error' in result) {
  handleError(result.error);
} else {
  processData(result);
}

// Pattern 2: Throws (only for programming errors)
function sign_data(data, key) {
  if (!key) throw new Error("Private key required");
  // ...
}
```

---

## Thread Safety

All functions are pure (no shared mutable state) and safe to call concurrently.

Private keys should be handled carefully:
- Never log
- Zero memory after use when possible
- Don't pass across thread boundaries unnecessarily

---

## Implementation Checklist

| Function | Priority | Complexity |
|----------|----------|------------|
| `generate_identity` | P0 | Low |
| `sign_belief` | P0 | Low |
| `verify_signature` | P0 | Low |
| `encrypt_for` | P0 | Medium |
| `decrypt` | P0 | Medium |
| `prove_identity` | P0 | Low |
| `verify_identity_proof` | P0 | Low |
| `did_from_public_key` | P0 | Low |
| `export_identity` | P1 | Medium |
| `import_identity` | P1 | Medium |
| `rotate_keys` | P1 | Medium |
| `verify_signature_with_rotation` | P1 | Medium |
| `generate_identity_from_mnemonic` | P2 | Medium |
| `encrypt_for_multiple` | P2 | Medium |
| `resolve_did` | P2 | High |
| `revoke_key` | P2 | Medium |

---

## Usage Examples

### Complete Flow: Create, Sign, Verify

```typescript
// Agent A creates identity
const agentA = generate_identity({ metadata: { display_name: "Alice" } });

// Agent A creates and signs a belief
const belief: Belief = {
  id: uuid(),
  content: "The Python GIL prevents true parallelism",
  confidence: { overall: 0.9, source_reliability: 0.85 },
  domains: ["programming", "python"],
  valid_from: Date.now(),
  holder_id: agentA.identity.id,
};

const signed = sign_belief(belief, agentA.private_keys.identity_key, agentA.identity.id);

// Agent B receives and verifies
const verification = verify_signature(signed, agentA.identity.signing_key);
console.log(verification.valid);  // true
```

### Complete Flow: Encrypted Message

```typescript
// Agent A wants to send secret message to Agent B
const secretMessage = new TextEncoder().encode("I have evidence for belief X");

const encrypted = encrypt_for(secretMessage, agentB.identity.encryption_key);

// Send encrypted over network...

// Agent B decrypts
const decrypted = decrypt(encrypted, agentB.private_keys.encryption_key);
const message = new TextDecoder().decode(decrypted);
```

### Complete Flow: Identity Proof

```typescript
// Verifier (e.g., federation node) challenges Agent A
const { challenge, expires_at } = create_challenge();

// Agent A proves identity
const proof = prove_identity(challenge, agentA.private_keys.identity_key, agentA.identity.id);

// Verifier checks
const result = verify_identity_proof(proof, agentA.identity.signing_key, challenge);
if (result.valid) {
  console.log("Agent A is who they claim to be");
}
```

---

*"Simple primitives, composed securely."*
