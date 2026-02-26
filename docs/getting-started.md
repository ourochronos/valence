# Getting Started with Valence

Valence is a personal knowledge substrate that gives Claude persistent memory across sessions. It stores beliefs, tracks conversations, learns patterns, and grows smarter over time.

## Prerequisites

- Python 3.11+
- PostgreSQL 16 with pgvector extension
- Claude Code CLI
- (Optional) OpenAI API key for semantic search embeddings

## Quick Start (Docker)

The fastest way to get running:

```bash
# Clone the repo
git clone https://github.com/ourochronos/valence.git
cd valence

# Copy environment template
cp .env.example .env
# Edit .env if you want to customize (defaults work fine)

# Start PostgreSQL with pgvector
docker compose up -d

# Wait for healthy database
docker compose ps  # Should show "healthy"

# Install valence
pip install -e .

# Verify
valence-mcp --health-check
```

## Quick Start (Manual PostgreSQL)

If you already have PostgreSQL running:

```bash
# Create database and user
createdb valence
psql -d valence -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Initialize schema
psql -d valence -f migrations/schema.sql
psql -d valence -f src/valence/substrate/procedures.sql

# Install
pip install -e .

# Set connection (if not using defaults)
export VALENCE_DB_HOST=localhost
export VALENCE_DB_NAME=valence
export VALENCE_DB_USER=valence
export VALENCE_DB_PASSWORD=your_password

# Verify
valence-mcp --health-check
```

## Configure Claude Code

### Plugin Installation

Copy the plugin to your Claude Code plugins directory:

```bash
# Create plugins dir if needed
mkdir -p ~/.claude/plugins

# Symlink the plugin
ln -s /path/to/valence/plugin ~/.claude/plugins/valence
```

Or use Claude Code directly with the plugin directory:

```bash
claude --plugin-dir /path/to/valence/plugin
```

### MCP Server

The plugin's `.mcp.json` configures a unified MCP server that provides all Valence tools. Claude Code will start it automatically when you launch with the plugin.

The server connects to PostgreSQL using environment variables:
- `VALENCE_DB_HOST` (default: localhost)
- `VALENCE_DB_PORT` (default: 5433)
- `VALENCE_DB_NAME` (default: valence)
- `VALENCE_DB_USER` (default: valence)
- `VALENCE_DB_PASSWORD` (default: empty)

### Embeddings (Optional)

For semantic search (finding beliefs by meaning, not just keywords):

```bash
export OPENAI_API_KEY=sk-...
```

Without this, `belief_query` (keyword search) works fine. `belief_search` (semantic search) requires embeddings.

## First Session

Start Claude Code with the Valence plugin. You'll see context injected at session start showing:
- Recent beliefs from the knowledge base
- Established patterns from past sessions
- Available skills

Just have a natural conversation. When the session ends:
1. The session-end hook fires automatically
2. If Claude called `session_end` with a summary/themes, those are auto-captured as beliefs
3. The session is closed in the database

## What Happens Automatically

### Session Start
- Creates a session record in the database
- Queries recent beliefs relevant to your project
- Queries established behavioral patterns
- Injects this context so Claude knows what you've discussed before

### During the Session
- Claude has MCP tools to query, create, and manage beliefs
- Beliefs are created explicitly when decisions are made
- Sessions, exchanges, and patterns are tracked

### Session End
- Session summary and themes are captured as beliefs (auto-capture)
- Each theme becomes its own belief for precise retrieval
- Beliefs get embeddings for semantic search (if OpenAI key is set)
- Max 10 auto-captured beliefs per session (prevents spam)

### Over Time
- Patterns emerge from repeated topics across sessions
- Beliefs accumulate, creating a knowledge base unique to you
- Confidence scores reflect reliability (dimensional: source, method, consistency, freshness, corroboration, applicability)
- Supersession chains track how knowledge evolves

## Skills Reference

Skills are invoked with `/valence:<skill-name>`:

| Skill | Description |
|-------|-------------|
| `/valence:using-valence` | Learn about the knowledge substrate |
| `/valence:query-knowledge` | Search the knowledge base |
| `/valence:capture-insight` | Store important information |
| `/valence:ingest-document` | Add documents to the substrate |
| `/valence:review-tensions` | Review and resolve contradictions |
| `/valence:status` | View knowledge base dashboard |

## MCP Tools (58 total)

### Knowledge Substrate (45 tools)
| Category | Tools |
|----------|-------|
| Beliefs | `belief_query`, `belief_search`, `belief_create`, `belief_supersede`, `belief_get`, `belief_share`, `belief_shares_list`, `belief_share_revoke`, `belief_corroboration` |
| Entities | `entity_get`, `entity_search` |
| Tensions | `tension_list`, `tension_resolve` |
| Confidence & Trust | `confidence_explain`, `trust_check` |
| Verification | `verification_submit`, `verification_accept`, `verification_get`, `verification_list`, `verification_summary` |
| Disputes | `dispute_submit`, `dispute_resolve`, `dispute_get` |
| Reputation | `reputation_get`, `reputation_events` |
| Bounties | `bounty_get`, `bounty_list` |
| Calibration & Rewards | `calibration_run`, `calibration_history`, `rewards_pending`, `reward_claim`, `rewards_claim_all`, `transfer_history`, `velocity_status` |
| Consensus | `consensus_status`, `corroboration_submit`, `corroboration_list`, `challenge_submit`, `challenge_resolve`, `challenge_get`, `challenges_list` |
| Backup | `backup_create`, `backup_verify`, `backup_list`, `backup_get` |

### Conversation Tracking (13 tools)
| Category | Tools |
|----------|-------|
| Sessions | `session_start`, `session_end`, `session_get`, `session_list`, `session_find_by_room` |
| Exchanges | `exchange_add`, `exchange_list` |
| Patterns | `pattern_record`, `pattern_reinforce`, `pattern_list`, `pattern_search` |
| Insights | `insight_extract`, `insight_list` |

See [API.md](API.md) for complete tool documentation with parameters and descriptions.

## Demo

Run the interactive demo to see the core value loop:

```bash
python examples/demo.py
```

This walks through: create beliefs, query, supersede, search entities, check trust, extract insights.

## Going Deeper

- **[API Reference](API.md)** — All 58 MCP tools and REST endpoints
- **[Implementation Status](IMPLEMENTATION-STATUS.md)** — Component-by-component status
- **[Error Codes](ERROR-CODES.md)** — All error codes and response formats
- **[Rate Limits](RATE-LIMITS.md)** — Rate limiting configuration
- **Architecture**: See `CLAUDE.md` in the repo root for full architecture documentation
- **Schema**: `migrations/schema.sql` defines all tables
- **Confidence Model**: Beliefs have 6 confidence dimensions, not just one score
- **Federation**: Valence supports peer-to-peer knowledge sharing (advanced)
- **HTTP Server**: For remote access: `valence-server` starts on port 8420
- **Deployment**: See `infra/` for production deployment with Ansible
