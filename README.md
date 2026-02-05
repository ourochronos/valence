# Valence

[![Coverage](https://codecov.io/gh/orobobos/valence/branch/main/graph/badge.svg)](https://codecov.io/gh/orobobos/valence)

**The trust layer for AI agents.**

Every AI agent wakes up alone. Reinvents what's true. Starts from zero with each conversation. Can't share what it learned in a way others can trust.

We built libraries, universities, peer review, Wikipedia. Agents have nothing.

Valence fixes this.

---

## What It Does

Valence is infrastructure for how beliefs travel between minds.

- **Store beliefs, not facts** — Everything is uncertain. Confidence has dimensions: source reliability, method quality, freshness, corroboration.
- **Share with privacy** — Contribute to collective knowledge without exposing individual beliefs. Differential privacy built in.
- **Earn trust** — Reputation comes from accuracy, not followers. Finding errors earns more than confirming consensus.

Your agent knows you. Together, agents know *everything*.

---

## 🔒 Privacy by Default

**Good news:** Valence now uses local embeddings by default. Your data stays on your machine.

- **Default: Local embeddings** using `bge-small-en-v1.5` (384 dimensions)
- No API keys required for basic operation
- Semantic search works entirely offline

**Optional: OpenAI embeddings** (if you prefer higher-dimensional vectors):

1. Set `VALENCE_EMBEDDING_PROVIDER=openai`
2. Provide `OPENAI_API_KEY`
3. Uses `text-embedding-3-small` (1536 dimensions)

Note: When using OpenAI provider, belief content is sent to OpenAI's API. See their [data usage policies](https://openai.com/policies/api-data-usage-policies).

We believe in transparency. Your data sovereignty matters.

---

## Offline Installation

For air-gapped environments, pre-download the embedding model before deploying.

### Option 1: Use the download script

```bash
# Download model to default cache location
python scripts/download_model.py

# Or save to a custom path for portable deployment
python scripts/download_model.py --save-path /opt/valence/models/bge-small-en-v1.5
```

### Option 2: Pre-download on connected machine

```bash
# Download model to cache
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy cache to target (Linux/Mac)
scp -r ~/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5 user@target:~/.cache/huggingface/hub/

# Or copy cache (Windows)
# Copy %USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-small-en-v1.5
```

### Option 3: Save to custom path

```bash
# Save model to portable location
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
model.save('/opt/valence/models/bge-small-en-v1.5')
"

# On target machine, set the path
export VALENCE_EMBEDDING_MODEL_PATH=/opt/valence/models/bge-small-en-v1.5
```

The `VALENCE_EMBEDDING_MODEL_PATH` environment variable can point to either:
- A Hugging Face model name (e.g., `BAAI/bge-small-en-v1.5`) - downloads if not cached
- A local filesystem path to a pre-saved model - loads directly, no network required

---

## Why Now

Three forces converging:

1. **Agents are exploding** — Millions of AI agents. They need to coordinate. Current infrastructure is "trust me bro."

2. **Context is the bottleneck** — Getting the right info into limited context windows. Similarity search isn't enough. You need *epistemic* retrieval.

3. **Trust is broken** — Deepfakes, misinformation, manipulation. We need new infrastructure for shared truth.

---

## Prerequisites

- **Python 3.11+**
- **PostgreSQL 14+** with pgvector extension
- **Docker** (recommended) or local PostgreSQL installation

### Quick Start with Docker

The easiest way to get PostgreSQL with pgvector:

```bash
docker run -d --name valence-db -p 5432:5432 \
  -e POSTGRES_PASSWORD=valence \
  -e POSTGRES_USER=valence \
  -e POSTGRES_DB=valence \
  ankane/pgvector
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VKB_DB_HOST` | PostgreSQL host | `localhost` |
| `VKB_DB_NAME` | Database name | `valence` |
| `VKB_DB_USER` | Database user | `valence` |
| `VKB_DB_PASSWORD` | Database password | (required) |
| `VALENCE_EMBEDDING_PROVIDER` | Embedding provider: `local` or `openai` | `local` |
| `VALENCE_EMBEDDING_MODEL_PATH` | Local model name/path | `BAAI/bge-small-en-v1.5` |
| `VALENCE_EMBEDDING_DEVICE` | Device for local model: `cpu` or `cuda` | `cpu` |
| `OPENAI_API_KEY` | OpenAI API key (only for `openai` provider) | — |
| `VALENCE_HOST` | Server bind host | `127.0.0.1` |
| `VALENCE_PORT` | Server bind port | `8420` |
| `VALENCE_OAUTH_ENABLED` | Enable OAuth 2.1 | `true` |
| `VALENCE_OAUTH_PASSWORD` | OAuth login password | (required if OAuth enabled) |

---

## Quick Start

```bash
# Install
pip install valence

# Initialize database (creates schema)
valence init

# Store a belief with derivation info
valence add "PostgreSQL scales better for this workload" \
  --domain tech --domain databases \
  --derivation-type observation \
  --confidence '{"overall": 0.85}'

# Query with derivation chains visible
valence query "database scaling"
# Shows: derivation type, method, source beliefs

# Detect contradicting beliefs
valence conflicts

# List recent beliefs
valence list -n 20 --domain tech
```

That's it. Personal knowledge substrate, running locally, yours forever.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `valence init` | Initialize database, create schema |
| `valence add <content>` | Add belief with confidence & derivation |
| `valence query <text>` | Search beliefs, shows derivation chains |
| `valence list` | List recent beliefs |
| `valence conflicts` | Detect contradicting beliefs |
| `valence stats` | Database statistics |

### Derivation Tracking

Every belief tracks where it came from:

```bash
# Add inferred belief with source
valence add "Derived conclusion" \
  --derivation-type inference \
  --derived-from <source-belief-uuid> \
  --method "Logical deduction from X and Y"
```

Query results show the full chain:
```
[1] Derived conclusion
    ID: abc12345  Confidence: 70%  Similarity: 95%
    ┌─ Derivation: inference
    │  Method: Logical deduction from X and Y
    │  ← Derived from (primary): Original observation...
    └─
```

### Conflict Detection

Find beliefs that contradict each other:

```bash
valence conflicts --threshold 0.85 --auto-record
```

Detects:
- High similarity with negation asymmetry ("X is good" vs "X is not good")
- Opposite conclusions about same topic
- Records detected conflicts as tensions for resolution

---

## Python API Tutorial

Want to use Valence programmatically? Here's a quick overview. See `examples/` for complete, runnable tutorials.

### Creating and Querying Beliefs

```python
from valence.substrate.tools import belief_create, belief_query

# Create a belief with dimensional confidence
result = belief_create(
    content="Python is excellent for rapid prototyping.",
    confidence={
        "source_reliability": 0.9,
        "method_quality": 0.85,
        "internal_consistency": 0.95,
        "temporal_freshness": 1.0,
        "corroboration": 0.7,
        "domain_applicability": 0.8,
    },
    domain_path=["tech", "programming", "python"],
    source_type="observation",
    entities=[
        {"name": "Python", "type": "tool", "role": "subject"},
    ],
)
belief_id = result["belief"]["id"]

# Query beliefs
results = belief_query(
    query="Python programming",
    domain_filter=["tech"],
    limit=10,
)
for belief in results["beliefs"]:
    print(f"[{belief['confidence']['overall']:.0%}] {belief['content']}")
```

### Trust and Federation

```python
from valence.federation import (
    TrustSignal,
    SIGNAL_WEIGHTS,
    compute_transitive_trust,
)

# Trust is built through positive signals
print(f"Belief corroborated: +{SIGNAL_WEIGHTS[TrustSignal.BELIEF_CORROBORATED]:.2f}")
print(f"Valid challenge: +{SIGNAL_WEIGHTS[TrustSignal.CHALLENGE_VALID]:.2f}")
print(f"Spam detected: {SIGNAL_WEIGHTS[TrustSignal.SPAM_DETECTED]:.2f}")

# Transitive trust decays with each hop
# Your trust in Bob = your_trust_in_alice * alice_trust_in_bob * decay_factor
```

### Running the Examples

```bash
# Hello Belief - create and query beliefs
python examples/01_hello_belief.py

# Trust Graph - understand trust mechanics
python examples/02_trust_graph.py

# Federation - connect to peers (no database required)
python examples/03_federation.py
```

---

## Scripts

### Re-Embedding Script

After migrating to local embeddings (migration 009), populate the new `embedding_384` columns:

```bash
# Process all tables
python scripts/reembed_all.py

# Dry run - see what would be done
python scripts/reembed_all.py --dry-run

# Process specific table only
python scripts/reembed_all.py --table beliefs

# Custom batch size (default: 100)
python scripts/reembed_all.py --batch-size 50

# Disable progress tracking (no resume capability)
python scripts/reembed_all.py --no-progress
```

Features:
- **Batch processing** with progress bar (tqdm)
- **Resume capability** - automatically resumes from `.reembed_progress.json`
- **Verification** - confirms embedding coverage after processing
- **~100 beliefs/sec** on CPU

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     YOUR AGENT                                  │
│                 (represents you, not platforms)                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PERSONAL SUBSTRATE                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Beliefs  │ │ Entities │ │ Sessions │ │ Patterns │           │
│  │ (claims  │ │ (people, │ │ (convo   │ │ (learned │           │
│  │  + conf) │ │  places) │ │  memory) │ │  habits) │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                     ↓ owned by you ↓                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │ (opt-in)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FEDERATION                                 │
│                                                                 │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐     │
│   │ Node A  │◄──►│ Node B  │◄──►│ Node C  │◄──►│ Node D  │     │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘     │
│                                                                 │
│   • Privacy-preserving aggregation                              │
│   • Trust computed from behavior                                │
│   • Graceful degradation (attenuate, don't ban)                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  COMMUNAL KNOWLEDGE                             │
│                                                                 │
│   "What does the network believe about X?"                      │
│   → Trust-weighted, confidence-scored, temporally-valid         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Adoption Path

Value accumulates at each phase. You don't need the network to benefit.

| Phase | What | Value |
|-------|------|-------|
| **1. Personal** | Store beliefs with confidence | Better than flat files. Works now. |
| **2. Peer** | Share with trusted agents | Trust metadata travels with claims |
| **3. Federation** | Form domain groups | Privacy-preserved aggregation |
| **4. Network** | Cross-federation | Query "what humanity knows" |

---

## Principles

These constrain what Valence can become:

1. **User sovereignty** — You own your data. No exceptions.
2. **Structurally incapable of betrayal** — Architecture, not promises.
3. **Aggregation serves users** — If it doesn't benefit you, it doesn't happen.
4. **Designed to survive being stolen** — Open patterns that work even if copied.
5. **AI-native** — Built for what's coming.

---

## Documentation

| Doc | Purpose |
|-----|---------|
| **[API Docs](/api/v1/docs)** | Interactive API documentation (Swagger UI) |
| **[OpenAPI Spec](docs/openapi.yaml)** | OpenAPI 3.0 specification |
| **[VISION](docs/VISION.md)** | The epistemic commons. Why this exists. |
| **[PRINCIPLES](docs/PRINCIPLES.md)** | The constitution. What constrains evolution. |
| **[SYSTEM](docs/SYSTEM.md)** | Architecture. How principles become structure. |
| **[SPECS](spec/)** | Technical specifications for all components. |
| **[ADOPTION](spec/ADOPTION.md)** | Phase-by-phase path to network. |
| **[MANIFESTO](spec/MANIFESTO.md)** | The movement framing. |

### API Reference

When running the server, access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8420/api/v1/docs`
- **OpenAPI JSON**: `http://localhost:8420/api/v1/openapi.json`

---

## Status

Active development. Working now:
- ✅ Personal belief substrate (PostgreSQL + pgvector)
- ✅ MCP servers for AI agent access
- ✅ Conversation tracking and pattern extraction
- 🔄 Federation protocol
- 🔜 Privacy-preserving aggregation

---

## Contributing

This isn't a product to be sold. It's infrastructure to be shared.

We need:
- **Builders** — Implement clients, servers, integrations
- **Agents** — Use it. Break it. Tell us what's wrong.
- **Researchers** — Verify the cryptography, economics, game theory
- **Communities** — Form federations around domains you care about

No single entity should control the trust layer of intelligence.

---

*"We shape our tools, and thereafter our tools shape us."*

The tools for knowledge are being shaped right now. Let's shape them toward wisdom.

---

*Co-created by humans and agents. 2025.*
