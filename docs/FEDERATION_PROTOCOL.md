# Valence Federation Protocol (VFP) Specification

*Version 1.0 - Draft*

This document specifies the Valence Federation Protocol (VFP), enabling sovereign knowledge bases to form trust networks, share intelligence, and collectively resolve contradictions across an epistemic commons.

---

## 1. Overview

### 1.1 Purpose

The Valence Federation Protocol enables:
- **Node Discovery**: Finding and connecting to other Valence nodes
- **Belief Sharing**: Exchanging knowledge claims with cryptographic provenance
- **Privacy-Preserving Aggregation**: Computing collective intelligence without exposing individual beliefs
- **Trust Networks**: Earning and extending trust through epistemic behavior

### 1.2 Design Principles

All protocol design derives from [PRINCIPLES.md](./PRINCIPLES.md):

1. **User Sovereignty**: Data never leaves without explicit consent
2. **Structural Integrity**: Trust enforced by architecture, not promises
3. **Aggregation Serves Users**: Only valid if it increases user value
4. **Openness as Resilience**: Survives being copied; invites scrutiny
5. **AI-Centric**: Built for AI agents as first-class participants

### 1.3 Protocol Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Protocol Layer                            │
│  Wire format, transport (HTTP/2+TLS), message types         │
├─────────────────────────────────────────────────────────────┤
│                   Aggregation Layer                          │
│  Privacy-preserving queries, differential privacy,          │
│  secure multi-party computation, corroboration              │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│  FederatedBelief envelopes, entity linking,                 │
│  provenance chains, cryptographic signatures                │
├─────────────────────────────────────────────────────────────┤
│                    Identity Layer                            │
│  Node DIDs, user identity, discovery protocol               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Identity Layer

### 2.1 Decentralized Identifiers (DIDs)

Nodes and users are identified using the `did:vkb` DID method.

#### 2.1.1 Node DID Formats

**Domain-Verified (Web)**
```
did:vkb:web:<domain>
```
Example: `did:vkb:web:valence.example.com`

**Self-Sovereign (Key-Based)**
```
did:vkb:key:<multibase-encoded-public-key>
```
Example: `did:vkb:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK`

The key-based format uses:
- `z6Mk` prefix indicating Ed25519 public key
- Multibase (base58btc) encoding

#### 2.1.2 User DID Format

```
did:vkb:user:<node-method>:<node-id>:<username>
```
Example: `did:vkb:user:web:valence.example.com:alice`

This enables:
- Users with identities across multiple nodes
- Belief attribution (with consent)
- Trust at both node and user levels

### 2.2 DID Document

Every node publishes a DID Document at its well-known endpoint:

```json
{
  "@context": [
    "https://www.w3.org/ns/did/v1",
    "https://valence.dev/ns/vfp/v1"
  ],
  "id": "did:vkb:web:valence.example.com",

  "verificationMethod": [{
    "id": "did:vkb:web:valence.example.com#keys-1",
    "type": "Ed25519VerificationKey2020",
    "controller": "did:vkb:web:valence.example.com",
    "publicKeyMultibase": "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"
  }],

  "authentication": ["did:vkb:web:valence.example.com#keys-1"],
  "assertionMethod": ["did:vkb:web:valence.example.com#keys-1"],

  "service": [{
    "id": "did:vkb:web:valence.example.com#vfp",
    "type": "ValenceFederationProtocol",
    "serviceEndpoint": "https://valence.example.com/federation"
  }, {
    "id": "did:vkb:web:valence.example.com#mcp",
    "type": "ModelContextProtocol",
    "serviceEndpoint": "https://valence.example.com/mcp"
  }],

  "vfp:capabilities": [
    "belief_sync",
    "aggregation_participate",
    "aggregation_publish"
  ],

  "vfp:profile": {
    "name": "Alice's Knowledge Base",
    "domains": ["tech", "philosophy"]
  },

  "vfp:protocolVersion": "1.0"
}
```

### 2.3 Discovery Protocol

#### 2.3.1 Well-Known Endpoints

**Node Metadata**
```
GET /.well-known/vfp-node-metadata
```
Returns the node's DID Document.

**Trust Anchors (Optional)**
```
GET /.well-known/vfp-trust-anchors
```
Returns a list of nodes this node explicitly trusts:

```json
{
  "trust_anchors": [
    {
      "did": "did:vkb:web:trusted-node.example.com",
      "trust_level": "anchor",
      "domains": ["tech"],
      "endorsed_at": "2025-01-15T10:30:00Z"
    }
  ],
  "updated_at": "2025-01-20T14:00:00Z"
}
```

#### 2.3.2 Bootstrap Discovery

For initial network discovery:

1. **Configuration**: Node operators manually configure initial peers
2. **Constellation Nodes**: Optional directory nodes that maintain peer lists
3. **Peer Exchange**: Connected nodes can share their known peers

---

## 3. Data Layer

### 3.1 Federated Belief Envelope

When beliefs are shared across federation boundaries, they're wrapped in a `FederatedBelief` envelope:

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "federation_id": "660e8400-e29b-41d4-a716-446655440001",
  "origin_node_did": "did:vkb:web:origin.example.com",

  "content": "PostgreSQL JSONB indexes support efficient querying of nested values",
  "confidence": {
    "overall": 0.85,
    "source_reliability": 0.9,
    "method_quality": 0.8,
    "corroboration": 0.7
  },
  "domain_path": ["tech", "databases", "postgresql"],

  "valid_from": "2025-01-01T00:00:00Z",
  "valid_until": null,

  "visibility": "federated",
  "share_level": "with_provenance",

  "hop_count": 1,
  "federation_path": [
    "did:vkb:web:origin.example.com",
    "did:vkb:web:relay.example.com"
  ],

  "origin_signature": "base64-encoded-ed25519-signature",
  "signed_at": "2025-01-15T10:30:00Z",
  "signature_method": "Ed25519Signature2020"
}
```

### 3.2 Visibility Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `private` | Never shared via federation | Personal notes, sensitive info |
| `trusted` | Shared only with explicitly trusted nodes | Team knowledge, semi-private |
| `federated` | Shared across federation network | General knowledge |
| `public` | Discoverable by anyone | Published research |

### 3.3 Share Levels

| Level | Includes | Privacy |
|-------|----------|---------|
| `belief_only` | Content + confidence only | High |
| `with_provenance` | + source information | Medium |
| `full` | + user attribution, all metadata | Low |

### 3.4 Cryptographic Signatures

All federated beliefs must be signed by their origin node:

```python
# Signature generation
message = canonical_json(belief_content)
signature = ed25519_sign(private_key, message)

# Verification
is_valid = ed25519_verify(
    public_key=resolve_did(origin_node_did).verification_key,
    message=canonical_json(belief_content),
    signature=decoded_signature
)
```

Canonical JSON ensures deterministic serialization for signature verification.

---

## 4. Protocol Layer

### 4.1 Transport

- **Protocol**: HTTP/2 over TLS 1.3
- **Content-Type**: `application/json`
- **Authentication**: DID-based challenge-response (see 4.2)

### 4.2 Authentication Flow

```
1. Client → Server: POST /federation/auth/challenge
   Body: { "client_did": "did:vkb:web:client.example.com" }

2. Server → Client: 200 OK
   Body: { "challenge": "random-nonce", "expires_at": "..." }

3. Client signs challenge with private key

4. Client → Server: POST /federation/auth/verify
   Body: {
     "client_did": "did:vkb:web:client.example.com",
     "challenge": "random-nonce",
     "signature": "base64-signature"
   }

5. Server → Client: 200 OK
   Body: { "session_token": "jwt-token", "expires_at": "..." }
```

### 4.3 Message Types

#### 4.3.1 Belief Operations

**SHARE_BELIEF**
```json
{
  "type": "SHARE_BELIEF",
  "request_id": "uuid",
  "beliefs": [FederatedBelief],
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**REQUEST_BELIEFS**
```json
{
  "type": "REQUEST_BELIEFS",
  "request_id": "uuid",
  "query": {
    "domain_filter": ["tech", "databases"],
    "semantic_query": "PostgreSQL performance optimization",
    "min_confidence": 0.7,
    "limit": 50
  },
  "requester_did": "did:vkb:web:client.example.com"
}
```

**BELIEFS_RESPONSE**
```json
{
  "type": "BELIEFS_RESPONSE",
  "request_id": "uuid",
  "beliefs": [FederatedBelief],
  "total_available": 150,
  "cursor": "pagination-cursor"
}
```

#### 4.3.2 Sync Operations

**SYNC_REQUEST**
```json
{
  "type": "SYNC_REQUEST",
  "request_id": "uuid",
  "since": "2025-01-10T00:00:00Z",
  "domains": ["tech"],
  "cursor": null
}
```

**SYNC_RESPONSE**
```json
{
  "type": "SYNC_RESPONSE",
  "request_id": "uuid",
  "changes": [
    {
      "type": "belief_created",
      "belief": FederatedBelief,
      "timestamp": "2025-01-15T10:30:00Z"
    },
    {
      "type": "belief_superseded",
      "old_belief_id": "uuid",
      "new_belief": FederatedBelief,
      "timestamp": "2025-01-16T11:00:00Z"
    }
  ],
  "cursor": "next-cursor",
  "has_more": true
}
```

#### 4.3.3 Trust Operations

**TRUST_ATTESTATION**
```json
{
  "type": "TRUST_ATTESTATION",
  "request_id": "uuid",
  "attestation": {
    "subject_did": "did:vkb:web:trusted-node.example.com",
    "trust_dimensions": {
      "belief_accuracy": 0.85,
      "extraction_quality": 0.8,
      "uptime_reliability": 0.95
    },
    "domains": ["tech", "databases"],
    "issued_at": "2025-01-15T10:30:00Z",
    "expires_at": "2025-04-15T10:30:00Z"
  },
  "issuer_signature": "base64-signature"
}
```

**ENDORSEMENT_REQUEST**
```json
{
  "type": "ENDORSEMENT_REQUEST",
  "request_id": "uuid",
  "requester_did": "did:vkb:web:new-node.example.com",
  "evidence": {
    "sample_beliefs": [FederatedBelief],
    "operational_history": {
      "uptime_days": 30,
      "beliefs_shared": 150,
      "sync_requests_served": 500
    }
  }
}
```

### 4.4 Error Handling

All errors follow a consistent format:

```json
{
  "type": "ERROR",
  "request_id": "uuid",
  "error_code": "TRUST_INSUFFICIENT",
  "message": "Trust level too low for requested operation",
  "details": {
    "required_trust": 0.4,
    "current_trust": 0.2,
    "suggestion": "Continue contributing quality beliefs to build trust"
  }
}
```

**Error Codes**

| Code | Description |
|------|-------------|
| `AUTH_FAILED` | Authentication challenge failed |
| `TRUST_INSUFFICIENT` | Trust level too low for operation |
| `VISIBILITY_DENIED` | Belief visibility doesn't permit sharing |
| `RATE_LIMITED` | Too many requests |
| `SYNC_CURSOR_INVALID` | Sync cursor expired or invalid |
| `SIGNATURE_INVALID` | Belief signature verification failed |
| `INTERNAL_ERROR` | Server-side error |

### 4.5 Versioning

Protocol version is declared in:
1. DID Document (`vfp:protocolVersion`)
2. HTTP Header (`VFP-Version: 1.0`)

Nodes should reject connections from incompatible versions and may maintain backward compatibility for minor versions.

---

## 5. Aggregation Layer

### 5.1 Privacy-Preserving Queries

Aggregation enables collective intelligence without exposing individual beliefs.

#### 5.1.1 Aggregation Query Flow

```
1. Query Broadcast
   Aggregator → Nodes: "Summarize beliefs about PostgreSQL in tech/databases"

2. Local Computation (per node)
   - Find relevant beliefs
   - Compute local statistics
   - Add differential privacy noise (ε budget)
   - Sign and return summary

3. Secure Aggregation
   - Aggregate noisy summaries
   - Compute weighted average by trust
   - Generate collective confidence

4. Result Publication
   - Publish aggregate (not individual contributions)
```

#### 5.1.2 Aggregation Query Message

**AGGREGATION_QUERY**
```json
{
  "type": "AGGREGATION_QUERY",
  "request_id": "uuid",
  "aggregator_did": "did:vkb:web:aggregator.example.com",
  "query": {
    "domain_filter": ["tech", "databases"],
    "semantic_query": "PostgreSQL vs MySQL for analytical workloads",
    "aggregation_type": "stance_summary"
  },
  "privacy_parameters": {
    "epsilon": 0.1,
    "delta": 1e-6,
    "min_contributors": 5
  },
  "deadline": "2025-01-15T11:00:00Z"
}
```

**LOCAL_SUMMARY (Node Response)**
```json
{
  "type": "LOCAL_SUMMARY",
  "request_id": "uuid",
  "node_did": "did:vkb:web:contributor.example.com",
  "summary": {
    "belief_count": 12,
    "mean_confidence": 0.78,
    "stance_vector": [0.2, 0.5, -0.3, ...],
    "domain_distribution": {
      "tech/databases/postgresql": 8,
      "tech/databases/mysql": 4
    }
  },
  "privacy_budget_used": 0.1,
  "computed_at": "2025-01-15T10:45:00Z",
  "signature": "base64-signature"
}
```

**AGGREGATION_RESULT**
```json
{
  "type": "AGGREGATION_RESULT",
  "request_id": "uuid",
  "result": {
    "collective_confidence": 0.82,
    "agreement_score": 0.78,
    "contributor_count": 12,
    "node_count": 8,
    "stance_summary": "Federation nodes moderately favor PostgreSQL for analytical workloads",
    "key_factors": [
      "Window functions",
      "JSONB support",
      "Index flexibility"
    ]
  },
  "privacy_guarantees": {
    "epsilon": 0.1,
    "delta": 1e-6,
    "mechanism": "laplace"
  },
  "computed_at": "2025-01-15T11:00:00Z"
}
```

### 5.2 Corroboration Protocol

When multiple nodes have similar beliefs, corroboration increases confidence:

**CORROBORATION_ATTESTATION**
```json
{
  "type": "CORROBORATION_ATTESTATION",
  "attestation": {
    "claim_hash": "sha256-of-normalized-claim",
    "corroboration_level": 0.85,
    "participating_nodes_count": 7,
    "confidence_boost": 0.1,
    "domain": ["tech", "databases"],
    "issued_at": "2025-01-15T12:00:00Z",
    "valid_until": "2025-02-15T12:00:00Z"
  },
  "aggregator_signature": "base64-signature"
}
```

Nodes can apply this to boost local belief confidence:
```python
belief.confidence = belief.confidence.boost_corroboration(
    amount=attestation.confidence_boost
)
```

### 5.3 Cross-Node Tension Detection

Detect contradictions without revealing raw beliefs:

1. Nodes compute stance embeddings for domain
2. Share embeddings (not content) via secure aggregation
3. Cluster similar embeddings across nodes
4. Detect opposing stances within clusters
5. Report potential tensions for investigation

**TENSION_SIGNAL**
```json
{
  "type": "TENSION_SIGNAL",
  "signal": {
    "domain": ["tech", "databases", "postgresql"],
    "cluster_id": "uuid",
    "opposition_strength": 0.75,
    "participating_nodes_count": 4,
    "description": "Divergent views on PostgreSQL JSONB performance"
  },
  "detected_at": "2025-01-15T13:00:00Z"
}
```

---

## 6. Endpoints Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/.well-known/vfp-node-metadata` | GET | Node DID document |
| `/.well-known/vfp-trust-anchors` | GET | Trusted nodes list |
| `/federation/auth/challenge` | POST | Begin auth |
| `/federation/auth/verify` | POST | Complete auth |
| `/federation/beliefs` | POST | Share beliefs |
| `/federation/beliefs/query` | POST | Query beliefs |
| `/federation/sync` | POST | Request sync |
| `/federation/trust/attestation` | POST | Submit attestation |
| `/federation/aggregation/query` | POST | Start aggregation |
| `/federation/aggregation/respond` | POST | Submit local summary |

---

## 7. Security Considerations

### 7.1 Threat Model

**Assumed Threats**
- Malicious nodes attempting to inject false beliefs
- Sybil attacks (creating many fake nodes)
- Privacy attacks through aggregation queries
- Network-level adversaries

**Mitigations**
- Cryptographic signatures on all beliefs
- Trust earned through behavior, not self-assertion
- Differential privacy in aggregation
- TLS for transport security

### 7.2 Privacy Budget

Each node maintains a privacy budget (ε) that limits how much information can leak through aggregation queries over time.

```python
class PrivacyBudget:
    total_epsilon: float = 1.0  # Per day
    used_epsilon: float = 0.0
    reset_at: datetime

    def can_respond(self, query_epsilon: float) -> bool:
        return self.used_epsilon + query_epsilon <= self.total_epsilon
```

### 7.3 Signature Verification

All operations must verify:
1. Belief signatures match origin node DID
2. Attestations signed by claimed issuer
3. Challenge responses prove key control

---

## 8. Implementation Notes

### 8.1 Recommended Libraries

- **Cryptography**: `cryptography` (Python), `@noble/ed25519` (JS)
- **DID Handling**: Custom implementation (see identity.py)
- **JSON Canonicalization**: RFC 8785 (JCS)

### 8.2 Rate Limiting

Recommended defaults:
- Authentication: 10/minute per IP
- Belief queries: 60/minute per authenticated node
- Aggregation: 10/hour per authenticated node

### 8.3 Storage Requirements

Nodes should store:
- Received federated beliefs (with provenance)
- Trust attestations (received and issued)
- Sync cursors (per peer)
- Privacy budget tracking

---

## Appendix A: DID Resolution

### A.1 Web DID Resolution

```
did:vkb:web:valence.example.com
  → https://valence.example.com/.well-known/vfp-node-metadata
```

### A.2 Key DID Resolution

Key DIDs are self-describing; the public key is embedded in the DID itself:

```
did:vkb:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
  → Decode multibase to get Ed25519 public key
```

---

## Appendix B: Canonical JSON

For signature generation/verification, use RFC 8785 JSON Canonicalization Scheme:

1. Serialize as UTF-8
2. Object keys sorted lexicographically
3. No whitespace between tokens
4. Numbers without unnecessary precision

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0-draft | 2025-01-20 | Initial specification |
