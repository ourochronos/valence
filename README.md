# Valence

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

## ⚠️ Privacy Disclosure

**Important:** By default, Valence uses OpenAI's embedding API to generate semantic search vectors. This means:

- **Belief content is sent to OpenAI** for embedding generation
- OpenAI's [data usage policies](https://openai.com/policies/api-data-usage-policies) apply
- Content is processed via `text-embedding-3-small` model

**To avoid external data processing:**

1. Set `VALENCE_EMBEDDING_PROVIDER=local` (uses local embedding model - coming soon)
2. Use `--opt-out-federation` flag when creating beliefs to exclude them from federation sharing
3. Run Valence with embeddings disabled (keyword search only)

We believe in transparency. Your data sovereignty matters.

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
| `OPENAI_API_KEY` | OpenAI API key for embeddings | (required unless using local) |
| `VALENCE_EMBEDDING_PROVIDER` | Embedding provider: `openai` or `local` | `openai` |
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
| **[VISION](docs/VISION.md)** | The epistemic commons. Why this exists. |
| **[PRINCIPLES](docs/PRINCIPLES.md)** | The constitution. What constrains evolution. |
| **[SYSTEM](docs/SYSTEM.md)** | Architecture. How principles become structure. |
| **[SPECS](spec/)** | Technical specifications for all components. |
| **[ADOPTION](spec/ADOPTION.md)** | Phase-by-phase path to network. |
| **[MANIFESTO](spec/MANIFESTO.md)** | The movement framing. |

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
