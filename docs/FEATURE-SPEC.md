# Valence Feature Specification

**Version**: 0.2.x
**Last Updated**: 2026-02-05

This document specifies all features available in the current Valence release.

## Core Concepts

### Beliefs
The fundamental unit of knowledge in Valence. Unlike simple facts, beliefs carry:
- **Content**: The actual information (text, structured data)
- **6-Dimensional Confidence**: Source reliability, method quality, internal consistency, temporal freshness, corroboration, domain applicability
- **Temporal Validity**: When the belief is/was valid
- **Provenance**: Full chain of custody from origin
- **Domain Tags**: Categorical classification

### Trust Graph
Relationships between entities with multi-dimensional trust:
- **Competence**: Ability to provide accurate information in a domain
- **Integrity**: Honesty and consistency
- **Confidentiality**: Ability to keep information private
- **Judgment**: Quality of assessments and recommendations

## API Features

### Belief Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/beliefs` | POST | Create a new belief |
| `/api/v1/beliefs` | GET | List beliefs with filtering |
| `/api/v1/beliefs/{id}` | GET | Get specific belief |
| `/api/v1/beliefs/{id}` | PUT | Update belief |
| `/api/v1/beliefs/{id}` | DELETE | Delete belief |
| `/api/v1/beliefs/search` | POST | Semantic search with multi-signal ranking |

#### Semantic Search
Multi-signal ranking formula:
```
final_score = w_semantic × similarity + w_confidence × confidence + w_recency × recency
```

Default weights: semantic=0.50, confidence=0.35, recency=0.15

Query parameters:
- `recency_weight`: Boost newer beliefs (0.0-1.0)
- `min_confidence`: Filter by confidence threshold
- `domain`: Filter by domain tag
- `include_revoked`: Include revoked beliefs (default: false)

### Trust Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/trust` | POST | Grant trust to entity |
| `/api/v1/trust` | GET | List trust relationships |
| `/api/v1/trust/{entity}` | GET | Get trust for specific entity |
| `/api/v1/trust/{entity}` | DELETE | Revoke trust |
| `/api/v1/trust/propagate` | POST | Compute transitive trust |

#### Trust Features
- 4-dimensional trust (competence, integrity, confidentiality, judgment)
- Domain-scoped trust overrides
- Trust decay over time
- Delegated trust computation
- Max hops propagation limits

### Privacy & Sharing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/share` | POST | Share belief with entity |
| `/api/v1/share/pending` | GET | List pending incoming shares |
| `/api/v1/share/{id}/accept` | POST | Accept shared belief |
| `/api/v1/share/{id}/revoke` | POST | Revoke previously shared belief |

#### Share Levels
1. **PRIVATE**: Only visible to owner
2. **DIRECT**: Shared with specific entities, no resharing
3. **BOUNDED**: Can be reshared up to N hops
4. **CASCADING**: Can be reshared indefinitely
5. **PUBLIC**: Visible to anyone

#### Privacy Features
- Consent chains with cryptographic signatures
- Strip-on-forward field redaction
- Revocation propagation to all recipients
- Differential privacy for aggregate queries
- k-anonymity for sensitive queries

### Authentication

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/oauth/register` | POST | Register OAuth client (dynamic) |
| `/oauth/authorize` | GET | Authorization endpoint |
| `/oauth/token` | POST | Token endpoint |
| `/oauth/userinfo` | GET | Get authenticated user info |

#### OAuth Features
- RFC 7591 Dynamic Client Registration
- PKCE support (required for public clients)
- Rate limiting on auth endpoints
- Constant-time credential comparison

### Health & Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health status |
| `/` | GET | API info and version |
| `/metrics` | GET | Prometheus metrics |

## Federation Features

### Node Discovery & Registration
- 3-tier topology: Seeds → Routers → Nodes
- Ed25519 signature verification
- Proof-of-work for Sybil resistance
- Heartbeat protocol (5-minute intervals)

### Federation Protocol
- Cross-node belief synchronization
- Trust edge federation
- Conflict detection and resolution
- Privacy-preserving sync (bucketed logging)

### Network Resilience
- Multi-router connections with failover
- Exponential backoff on failures
- Offline message queueing
- Router health monitoring
- Malicious router detection

### Security
- End-to-end encryption (X25519 + AES-256-GCM)
- MLS group encryption for multi-party
- Gateway nodes for cross-federation
- Eclipse attack mitigation
- Traffic analysis mitigation

## Storage Features

### Embeddings
- Default: Local bge-small-en-v1.5 (384 dimensions)
- Optional: OpenAI text-embedding-3-small (1536 dimensions)
- Offline mode support
- Batch processing

### Resilient Storage
- Reed-Solomon erasure coding
- Redundancy levels: MINIMAL (2/3), PERSONAL (3/5), FEDERATION (5/9), PARANOID (7/15)
- Merkle tree integrity verification
- Multi-backend distribution

### Database
- PostgreSQL with pgvector
- Connection pooling
- HNSW indexes for vector search

## CLI Commands

```bash
# Belief management
valence add "belief content" --domain domain1
valence query "search query" --min-confidence 0.7
valence list --limit 10
valence get <belief-id>

# Trust management
valence trust grant <entity> --competence 0.8
valence trust list
valence trust propagate <entity>

# Federation
valence discover              # Find federation peers
valence federate list         # List known peers
valence federate status       # Federation status
valence federate sync <peer>  # Sync with peer

# Network (router/seed)
valence-router                # Run as router node
valence-seed                  # Run as seed node

# Administration
valence migrate               # Run database migrations
valence export                # Export beliefs
valence import                # Import beliefs
```

## MCP Integration

Valence exposes an MCP (Model Context Protocol) interface for AI agent integration.

### Tools
- `valence_add_belief`: Create beliefs
- `valence_query_beliefs`: Semantic search
- `valence_get_belief`: Retrieve specific belief

### Resources
- `valence://beliefs`: List all beliefs
- `valence://beliefs/{id}`: Specific belief
- `valence://stats`: System statistics

## Configuration

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `VALENCE_DATABASE_URL` | PostgreSQL connection string | Required |
| `VALENCE_JWT_SECRET` | JWT signing secret | Auto-generated (warning logged) |
| `VALENCE_ENV` | Environment (development/production) | development |
| `VALENCE_EMBEDDING_PROVIDER` | local or openai | local |
| `VALENCE_EMBEDDING_MODEL_PATH` | Path to local model | Auto-download |
| `OPENAI_API_KEY` | For OpenAI embeddings | Optional |

### Production Recommendations
- Set explicit `VALENCE_JWT_SECRET`
- Use `VALENCE_ENV=production`
- Enable Prometheus metrics scraping
- Configure connection pool size for load
- Use structured JSON logging

## Deployment Validation

See `tests/integration/test_live_nodes.py` for the full validation suite.

### Required Checks
1. Health endpoint returns healthy
2. Database connection verified
3. OAuth flow completes
4. Belief CRUD operations work
5. Semantic search returns results
6. Federation discovery succeeds
7. Federation sync completes
8. MCP endpoints respond
9. Privacy controls enforced

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.2.2 | 2026-02-05 | Zero mypy errors, OAuth compliance |
| 0.2.1 | 2026-02-05 | Security fixes (6 HIGH), privacy hardening |
| 0.2.0 | 2026-02-04 | Privacy architecture, federation, network layer |
| 0.1.0 | 2026-02-03 | Initial alpha release |
