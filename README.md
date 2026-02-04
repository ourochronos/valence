# Valence

A universal, AI-driven platform where users interact with services through personal agents while maintaining data ownership through federated, privacy-preserving aggregation.

## The Vision

Your agent knows you. It represents you—not platforms, not advertisers, not anyone else. Your data stays yours. When millions of agents aggregate their humans' values (with consent, with privacy preserved), that collective voice has power. Power to influence markets, policy, institutions.

This is Valence: the capacity to connect, to affect, to bond.

## Documentation

### Philosophy & Architecture
- **[VISION.md](docs/VISION.md)** — The epistemic commons. Why this exists.
- **[PRINCIPLES.md](docs/PRINCIPLES.md)** — The constitution. These constrain what Valence can become.
- **[SYSTEM.md](docs/SYSTEM.md)** — The architecture. How principles become structure.
- **[UNKNOWNS.md](docs/UNKNOWNS.md)** — Honest gaps. What we don't know yet.

### Technical Specifications
- **[spec/](spec/)** — Complete technical specs for all 11 components
- **[spec/ADOPTION.md](spec/ADOPTION.md)** — Phase-by-phase adoption path
- **[spec/MANIFESTO.md](spec/MANIFESTO.md)** — The movement framing

## Knowledge Base Architecture

Valence uses modular knowledge bases with clear separation:

- **Schema** (in repo) — Structure definitions, migrations, constraints
- **Data** (external) — Instance-specific, moving to cloud service

### KB Scopes

| Scope | Purpose | Location |
|-------|---------|----------|
| Personal | User values, preferences, agent memory | Cloud (planned) |
| Project | Decisions, state, progress for this project | Local/Cloud |
| Agent | Operational patterns, consistency across sessions | With Personal |

### Schema Modules

- `src/valence/substrate/schema.sql` — Unified schema (beliefs, entities, sessions, exchanges, patterns, embeddings)
- `src/valence/substrate/procedures.sql` — Stored procedures for atomic operations

## Getting Started

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install package
pip install -e .

# Initialize schemas (creates local KB)
python -c "from valence.core.db import init_schema; init_schema()"
```

## Development

This project is developed using its own principles:

1. **Human intent** → Define what needs to exist
2. **Collaborative design** → Derive decisions from principles
3. **AI implementation** → Claude implements, principles constrain
4. **Knowledge capture** → Decisions accumulate in KB
5. **Reflection** → Did the process follow principles?

## MCP Servers

Valence exposes tools via Model Context Protocol for AI agent access:

```bash
# Run the MCP servers
python -m valence.substrate.mcp_server  # Knowledge substrate (beliefs, entities, tensions)
python -m valence.vkb.mcp_server        # Conversation tracking (sessions, exchanges, patterns)
```

Tools include belief management, entity tracking, session management, exchange capture, insight extraction, and pattern tracking.

## Status

Active development. Current focus:
- Conversation tracking and curation
- Multi-provider embedding architecture
- Cloud migration path for personal KB

### Pending Architecture Work

- Library restructuring (valence-core, valence-mcp)
- Schema migration strategy
- Configuration management
- MCP tool interface versioning

---

*Co-created by Chris and Claude. December 2025.*
