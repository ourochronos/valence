# Resilient Storage Component

## Overview

A personal knowledge substrate must survive hardware failures, provider shutdowns, and cryptographic breaks. This component addresses backup, disaster recovery, and long-term data resilience with an eye toward post-quantum security.

## Design Principles

### 1. No Single Point of Failure
- Local primary + distributed backup
- Erasure coding across multiple locations
- Recovery from partial data loss

### 2. Encrypt Before Distribution
- Beliefs encrypted locally before any sync
- No external party sees plaintext
- Post-quantum algorithms for longevity

### 3. Verify Without Trust
- Merkle proofs for integrity
- Challenge-response for availability
- Cryptographic receipts for storage

### 4. Graceful Degradation
- Works fully offline (local primary)
- Sync when connected
- Rebuild from fragments if needed

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Local Primary                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │  Postgres   │  │  PGVector   │  │   Graph     │               │
│  │  (beliefs)  │  │ (embeddings)│  │  (edges)    │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Encryption Layer                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │ Post-Quantum    │  │ Content-Addressed│  │  Merkle Tree     │  │
│  │ Encryption      │  │ Chunking         │  │  (integrity)     │  │
│  └─────────────────┘  └─────────────────┘  └──────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│                      Erasure Coding Layer                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ Reed-Solomon (k of n) — reconstruct from any k fragments    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │ Storage  │    │ Storage  │    │ Storage  │
        │ Node 1   │    │ Node 2   │    │ Node n   │
        │ (shard)  │    │ (shard)  │    │ (shard)  │
        └──────────┘    └──────────┘    └──────────┘
```

## Post-Quantum Cryptography

### Why Now?
- "Harvest now, decrypt later" attacks
- Knowledge substrate is long-lived
- Transition takes years; start early

### Algorithm Selection

| Purpose | Algorithm | Standard | Notes |
|---------|-----------|----------|-------|
| Signatures | CRYSTALS-Dilithium | FIPS 204 | Primary recommendation |
| Signatures (stateless) | SPHINCS+ | FIPS 205 | Larger but hash-based |
| Key encapsulation | CRYSTALS-Kyber (ML-KEM) | FIPS 203 | For symmetric key exchange |
| Symmetric encryption | AES-256-GCM | Existing | Quantum-resistant at 256-bit |

### Hybrid Approach (Transition Period)
```
encrypted_belief = AES-256-GCM(
  key = HKDF(
    Kyber.decapsulate(kyber_ciphertext) ||
    X25519.derive(x25519_pubkey)
  ),
  plaintext = belief_data
)
```

Combines post-quantum (Kyber) with classical (X25519). Secure if either holds.

## Erasure Coding

### Reed-Solomon Parameters

| Scenario | Data (k) | Total (n) | Overhead | Survives |
|----------|----------|-----------|----------|----------|
| Personal backup | 3 | 5 | 67% | Any 2 failures |
| Federation | 5 | 9 | 80% | Any 4 failures |
| Paranoid | 7 | 15 | 114% | Any 8 failures |

### Implementation
```python
from reedsolo import RSCodec

# 5 of 9 scheme
rs = RSCodec(nsym=4)  # 4 parity symbols

# Encode
shards = rs.encode(belief_bytes)

# Decode (works with up to 2 erasures)
original = rs.decode(partial_shards)
```

### Content-Addressed Chunking

Beliefs chunked by content hash (like IPFS):
- Deduplication across beliefs
- Efficient incremental backup
- Merkle tree for integrity verification

```
belief_id: abc123
├── chunk_0: QmX7y2...
├── chunk_1: QmA3b4...
└── chunk_2: QmZ9c8...

merkle_root: QmR00t...
```

## Storage Backends

### Tier 1: Self-Controlled
| Backend | Pros | Cons |
|---------|------|------|
| Local disk | Fast, free, private | Single point of failure |
| NAS/home server | Redundant, private | Requires hardware |
| Self-hosted S3 (MinIO) | S3 API, RAID | Complexity |

### Tier 2: Trusted Providers
| Backend | Pros | Cons |
|---------|------|------|
| Backblaze B2 | Cheap, reliable | Centralized |
| Wasabi | No egress fees | Centralized |
| AWS S3 Glacier | Very cheap archive | Slow retrieval |

### Tier 3: Decentralized
| Backend | Pros | Cons |
|---------|------|------|
| IPFS + Pinning | Content-addressed | Pinning costs |
| Filecoin | Incentivized storage | Complexity |
| Sia | Erasure coded, encrypted | Requires Siacoin |
| Arweave | Permanent storage | Pay once model |

### Recommended Multi-Tier Strategy
```yaml
backup_strategy:
  primary:
    backend: local_postgres
    frequency: continuous
    
  secondary:
    backend: nas_replica
    frequency: hourly
    retention: 30_days
    
  offsite:
    backend: backblaze_b2
    frequency: daily
    encryption: hybrid_pq
    retention: 1_year
    
  archive:
    backend: sia_or_filecoin
    frequency: weekly
    erasure_coding: 5_of_9
    encryption: hybrid_pq
    retention: permanent
```

## Disaster Recovery Scenarios

### Scenario 1: Disk Failure
- **Detection**: Postgres won't start
- **Recovery**: Restore from NAS replica
- **RTO**: < 1 hour
- **Data loss**: < 1 hour (last sync)

### Scenario 2: Home Disaster
- **Detection**: All local systems unavailable
- **Recovery**: Restore from B2 to new hardware
- **RTO**: < 24 hours
- **Data loss**: < 24 hours

### Scenario 3: Provider Shutdown
- **Detection**: B2 unavailable
- **Recovery**: Reconstruct from Sia shards
- **RTO**: Days (retrieval time)
- **Data loss**: None (erasure coded)

### Scenario 4: Cryptographic Break
- **Detection**: Algorithm deprecated
- **Recovery**: Re-encrypt with new algorithm
- **RTO**: Depends on data size
- **Data loss**: None (if caught early)

### Scenario 5: Key Loss
- **Detection**: Can't decrypt backups
- **Recovery**: Shamir's Secret Sharing recovery
- **RTO**: Depends on shard holders
- **Data loss**: None (if recovery succeeds)

## Key Management

### Master Key Derivation
```
master_seed (256 bits, from secure random)
    │
    ├── encryption_key = HKDF(master_seed, "encryption")
    ├── signing_key = HKDF(master_seed, "signing")
    └── backup_key = HKDF(master_seed, "backup")
```

### Shamir's Secret Sharing for Recovery
```python
from secretsharing import PlaintextToHexSecretSharer

# Split master seed into 5 shares, need 3 to recover
shares = PlaintextToHexSecretSharer.split_secret(
    master_seed_hex,
    share_threshold=3,
    num_shares=5
)

# Distribute shares to trusted parties
# - Hardware security module
# - Trusted friend 1
# - Trusted friend 2
# - Safe deposit box
# - Lawyer

# Recovery
recovered = PlaintextToHexSecretSharer.recover_secret(
    [share_1, share_3, share_5]  # Any 3
)
```

## Integrity Verification

### Merkle Tree Structure
```
                    root_hash
                   /         \
            hash_01           hash_23
           /      \          /      \
      hash_0   hash_1   hash_2   hash_3
         │        │        │        │
      shard_0  shard_1  shard_2  shard_3
```

### Periodic Verification
```python
async def verify_backup_integrity(storage_nodes):
    """Challenge-response integrity check."""
    for node in storage_nodes:
        # Request random chunks
        challenges = generate_random_challenges(n=10)
        
        for chunk_id in challenges:
            # Node must prove possession
            proof = await node.prove_possession(chunk_id)
            
            if not verify_merkle_proof(proof, root_hash):
                alert(f"Integrity failure: {node.id} chunk {chunk_id}")
                trigger_repair(chunk_id)
```

### Repair Protocol
When verification fails:
1. Identify missing/corrupt shards
2. Retrieve k valid shards
3. Reconstruct via Reed-Solomon
4. Re-distribute to healthy nodes
5. Update shard locations

## Graph-Aware Backup

### The Problem
PGVector stores embeddings but not relationships. Beliefs have explicit edges:
- `supports(belief_a, belief_b)`
- `contradicts(belief_a, belief_b)`
- `supersedes(belief_a, belief_b)`
- `derives_from(belief_a, source_c)`

### Graph Storage Options

#### Option 1: Apache AGE (Recommended)
```sql
-- Enable graph extension
CREATE EXTENSION age;

-- Create graph
SELECT create_graph('belief_graph');

-- Add edge
SELECT * FROM cypher('belief_graph', $$
  MATCH (a:Belief {id: 'abc123'})
  MATCH (b:Belief {id: 'def456'})
  CREATE (a)-[:SUPPORTS {weight: 0.8}]->(b)
$$) as (e agtype);

-- Query relationships
SELECT * FROM cypher('belief_graph', $$
  MATCH (a:Belief)-[r:SUPPORTS*1..3]->(b:Belief)
  WHERE a.id = 'abc123'
  RETURN b.id, length(r) as distance
$$) as (belief_id agtype, distance agtype);
```

#### Option 2: Adjacency Table (Simpler)
```sql
CREATE TABLE belief_edges (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_belief_id UUID REFERENCES beliefs(id),
  target_belief_id UUID REFERENCES beliefs(id),
  edge_type TEXT NOT NULL,  -- supports, contradicts, supersedes, derives_from
  weight FLOAT DEFAULT 1.0,
  created_at TIMESTAMPTZ DEFAULT now(),
  metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_belief_edges_source ON belief_edges(source_belief_id);
CREATE INDEX idx_belief_edges_target ON belief_edges(target_belief_id);
CREATE INDEX idx_belief_edges_type ON belief_edges(edge_type);
```

### Backup Considerations
Graph edges must be backed up with beliefs:
- Include edges in Merkle tree
- Maintain referential integrity during restore
- Handle orphaned edges gracefully

## Migration from Existing Postgres

### For PGVector-based KBs
```python
async def migrate_pgvector_kb(source_conn, target_conn):
    """Migrate existing PGVector data to Valence schema."""
    
    # 1. Extract existing data
    rows = await source_conn.fetch("""
        SELECT id, content, embedding, metadata, created_at
        FROM your_existing_table
    """)
    
    # 2. Transform to Valence beliefs
    for row in rows:
        belief = Belief(
            content=row['content'],
            embedding=row['embedding'],
            confidence=infer_confidence(row),  # From metadata/heuristics
            domains=extract_domains(row),
            provenance=Provenance(
                source_type='migration',
                source_id=str(row['id']),
                created_at=row['created_at']
            )
        )
        await target_conn.insert_belief(belief)
    
    # 3. Infer relationships (optional)
    await infer_belief_edges(target_conn)
```

### For Custom Extension KBs
Need to understand extension semantics:
- What additional columns/types?
- What constraints?
- What functions?

Then build custom transformer.

## CLI Interface

```bash
# Configure backup
valence backup config --add-backend b2 \
  --bucket my-valence-backup \
  --key-id xxx --key xxx

# Run backup
valence backup run

# Verify integrity
valence backup verify

# Restore
valence backup restore --from b2 --to ./restore

# Key management
valence keys init              # Generate master key
valence keys split --shares 5 --threshold 3
valence keys recover           # Interactive recovery

# Graph operations
valence graph show belief_id   # Show relationships
valence graph add-edge a b --type supports
valence graph path a b         # Find relationship path
```

## Security Considerations

### Threat Model
| Threat | Mitigation |
|--------|------------|
| Storage provider reads data | Encrypt before upload |
| Quantum computer | Post-quantum algorithms |
| Key compromise | Shamir recovery, rotation |
| Data corruption | Merkle verification |
| Provider shutdown | Multi-backend, erasure coding |
| Ransomware | Offline/immutable backups |

### Defense in Depth
1. **Encryption**: Data unreadable without keys
2. **Distribution**: No single point of failure
3. **Verification**: Detect tampering
4. **Redundancy**: Survive partial loss
5. **Recovery**: Multiple paths to restore

## Future Considerations

### Homomorphic Operations (Research)
Query encrypted beliefs without decryption:
- Semantic search on encrypted embeddings
- Privacy-preserving federation
- Currently impractical, watch for advances

### Zero-Knowledge Proofs
Prove belief properties without revealing content:
- "I have a belief about X with confidence > 0.8"
- Useful for trust establishment
- SNARK/STARK based

### Decentralized Key Recovery
Replace trusted parties with smart contracts:
- Time-locked recovery
- Social recovery (M of N guardians)
- Dead man's switch

---

## Summary

Resilient storage is not optional for a knowledge substrate. This component provides:

1. **Post-quantum encryption** for longevity
2. **Erasure coding** for redundancy
3. **Multi-backend** for no single point of failure
4. **Graph-aware** backup including relationships
5. **Key recovery** via Shamir's Secret Sharing
6. **Integrity verification** via Merkle proofs

The goal: Your knowledge survives you, your hardware, your providers, and current cryptography.
