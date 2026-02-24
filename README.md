# Valence

**A knowledge system for AI agents.**

Valence ingests information from diverse sources, compiles it into right-sized articles through use-driven promotion, and maintains those articles as living documents — always current, always traceable to their origins, always queryable via hybrid semantic + keyword search.

Sources are append-only and immutable with supersession tracking. Articles are compiled on demand, not eagerly. Every article tracks which sources built it, how they contributed, and whether any of them disagree.

---

## Why Valence?

Most agent memory systems are key-value stores with embeddings bolted on. Valence is different:

- **Epistemics-native** — provenance tracking, contention detection, confidence scoring. Not just "what do I know?" but "why do I believe it, and does anything disagree?"
- **Local-first** — runs on your machine. PostgreSQL + pgvector. No cloud dependency.
- **Hybrid retrieval** — Reciprocal Rank Fusion combines vector similarity (semantic) with full-text search (keyword). Finds what you mean, not just what you said.
- **Right-sized for context windows** — articles are automatically sized to 300–800 tokens for optimal embedding retrieval quality.
- **Stigmergic refinement** — usage signals drive self-organization. Frequently-used articles stay well-maintained; unused ones decay organically.
- **Graph-vector duality** — knowledge exists in both graph space (provenance links, relationships) and vector space (embeddings, similarity). Two views of the same knowledge.
- **Supersession tracking** — sources can supersede previous versions with full provenance chains. Append-only architecture preserves history while surfacing current knowledge.

### Compared to alternatives

| | Valence | Mem0 | Zep | LangMem |
|---|---|---|---|---|
| **Local-first** | ✅ | ❌ ($249/mo for graph) | ❌ (cloud for best features) | ❌ (LangGraph-only) |
| **Provenance tracking** | ✅ | ❌ | ❌ | ❌ |
| **Hybrid retrieval (RRF)** | ✅ | Partial | Partial | ❌ |
| **Contention detection** | ✅ | ❌ | ❌ | ❌ |
| **Supersession chains** | ✅ | ❌ | ❌ | ❌ |
| **MCP protocol** | ✅ (16 tools) | ❌ | ❌ | ❌ |
| **Framework-agnostic** | ✅ | ✅ | ❌ | ❌ |
| **Cost** | Free (self-hosted) | $249/mo+ | $99/mo+ | Free (LangGraph lock-in) |

---

## Architecture

```
Sources (immutable, typed)
    │  ingest → store → embed (deferred)
    ▼
Article compilation (use-driven)
    │  query → surface sources → compile via inference
    ▼
Articles (versioned, right-sized 300–800 tokens)
    │  provenance links, contention flags, freshness scores
    ▼
Hybrid Retrieval (RRF: vector KNN + full-text)
    │  ranked by relevance × confidence × freshness
    ▼
Agent / CLI / REST / MCP consumer
```

**Four layers:**

- **Sources** — raw inputs (conversations, documents, web, code, observations, tool outputs). Ingested cheaply, stored immutably. Embedding deferred until needed.
- **Articles** — compiled knowledge units. Created when a query surfaces ungrouped sources; updated when new source material arrives. Each article is right-sized and carries full provenance.
- **Provenance** — typed relationships from sources to articles: `originates`, `confirms`, `supersedes`, `contradicts`, `contends`. Contention is surfaced at retrieval time, not silently resolved.
- **Traces** — usage signals that drive self-organization. Articles used frequently stay well-maintained; unused articles deprioritize and are candidates for organic forgetting.

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/ourochronos/valence.git
cd valence
docker compose up -d
```

PostgreSQL + pgvector starts on `localhost:5433`. The Valence API server starts on `http://localhost:8420`. Schema is applied automatically.

### Database only (for development)

```bash
docker compose up -d postgres
```

Then install and run locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
valence init          # apply schema migrations
```

### Configure

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key settings:

```bash
# Database (all have sensible defaults for Docker)
VALENCE_DB_HOST=127.0.0.1
VALENCE_DB_PORT=5433
VALENCE_DB_NAME=valence
VALENCE_DB_USER=valence
VALENCE_DB_PASSWORD=valence

# Embeddings (OpenAI recommended; local fallback available)
VALENCE_EMBEDDING_PROVIDER=openai
VALENCE_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-your-key-here
```

### Configure inference

By default, the system runs in degraded mode (concatenation fallback). Configure a real backend:

```bash
# Gemini 2.5 Flash via local gemini CLI (no API key needed)
valence config inference gemini

# Local Ollama (fully offline)
valence config inference ollama --model qwen3:30b

# View current config
valence config inference show
```

### Smoke test

```bash
# Ingest a source
valence ingest "Python's GIL was replaced in 3.13 by per-interpreter locks." \
  --type document --title "Python 3.13 release notes"

# Search (hybrid: semantic + keyword)
valence search "Python concurrency"

# System status
valence status
```

---

## CLI Usage

### Unified search

```bash
# Search everything (articles + sources)
valence search "query"
valence search "Python GIL" --limit 10

# Search only articles or sources
valence search "query" --articles-only
valence search "query" --sources-only
```

### Ingest

```bash
# From argument, file, URL, or stdin
valence ingest "Content here" --type observation --title "My note"
valence ingest path/to/file.md --type document
valence ingest https://example.com/page --type web
cat notes.txt | valence ingest - --type document
```

**Source types:** `document`, `conversation`, `web`, `code`, `observation`, `tool_output`, `user_input`

### Compile

```bash
# Auto-compile all unlinked sources
valence compile --auto

# Compile specific sources into an article
valence compile <source-id-1> <source-id-2> --title "Topic hint"
```

### Articles

```bash
valence articles list
valence articles get <article-id>
valence articles search "query"
```

### Sources

```bash
valence sources list
valence sources get <source-id>
valence sources search "query"
```

### Provenance

```bash
valence provenance get <article-id>
valence provenance trace <article-id> "claim text to trace"
valence provenance link <article-id> <source-id> --relationship confirms
```

**Relationship types:** `originates`, `confirms`, `supersedes`, `contradicts`, `contends`

### Configuration

```bash
valence config show                              # all current settings
valence config set <key> <value>                 # set a config value
valence config inference show                    # inference backend
valence config inference gemini --model gemini-2.5-flash
```

### Status

```bash
valence status    # article count, source count, embedding coverage, DB info
```

### Global flags

```bash
valence --json articles search "query"          # JSON output
valence --output table sources list             # table output
valence --server http://remote:8420 stats       # remote server
```

---

## Interfaces

### CLI (primary)
The `valence` command covers all operations. See above.

### REST API
OpenAPI 3.1 spec at `docs/openapi.yaml`. Server runs on port 8420:

```bash
valence serve                    # start the API server
curl http://localhost:8420/health
```

### MCP (Model Context Protocol)

Valence exposes **16 MCP tools** for AI agent integration. Perfect for Claude Desktop, OpenClaw, or any MCP-aware client.

```bash
valence mcp    # start MCP server on stdio
```

**Available tools:**

**Source Management**
- `source_ingest` — Ingest a new source (document, conversation, web, code, observation, tool_output, user_input)
- `source_get` — Get source by ID with full details
- `source_search` — Full-text search over source content

**Knowledge Retrieval**
- `knowledge_search` — Unified retrieval across articles and sources with hybrid ranking (relevance × confidence × freshness)

**Article Management**
- `article_get` — Get article by ID, optionally with full provenance
- `article_create` — Manually create a new article
- `article_compile` — Compile sources into an article via LLM summarization
- `article_update` — Update article content with new material

**Article Right-Sizing**
- `article_split` — Split an oversized article into two
- `article_merge` — Merge two related articles into one

**Provenance**
- `provenance_trace` — Trace which sources contributed a specific claim

**Contention Resolution**
- `contention_list` — List active contradictions and disagreements
- `contention_resolve` — Resolve a contention (supersede_a, supersede_b, accept_both, dismiss)

**Administration**
- `admin_forget` — Permanently remove a source or article (irreversible)
- `admin_stats` — Health and capacity statistics
- `admin_maintenance` — Trigger maintenance operations (recompute scores, process queue, evict if over capacity)

Source types carry different reliability scores: `document`/`code` (0.8), `tool_output` (0.7), `user_input` (0.75), `web` (0.6), `conversation` (0.5), `observation` (0.4).

Provenance relationships: `originates`, `confirms`, `supersedes`, `contradicts`, `contends`.

### OpenClaw
Valence integrates with OpenClaw via CLI skill wrapping:

```bash
# OpenClaw skill calls valence CLI directly
valence ingest "$CONTENT" --type observation
valence search "$QUERY"
```

---

## Costs

Valence is free to self-host. Optional costs:

- **Embeddings**: OpenAI text-embedding-3-small costs ~$0.02 per million tokens. A typical knowledge base of 100 articles costs <$0.01 to embed.
- **Compilation**: Uses your configured inference backend. Gemini CLI and Ollama are free. Cloud providers charge per-token.
- **Database**: PostgreSQL + pgvector. Runs locally or on any Postgres host.

---

## Known Limitations

- **Article splitting and merging**: The `split_article` and `merge_articles` operations referenced in the work unit system are not yet implemented. These features are planned for a future release to support automated article reorganization based on usage patterns.

---

## Development

```bash
git clone https://github.com/ourochronos/valence.git
cd valence

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Start the database
docker compose up -d postgres

# Apply schema
valence init

# Run tests
pytest tests/ -x -q

# Lint
ruff check src/
```

---

## Data Sovereignty

All data stays local by default. Embeddings (if using OpenAI) are one-way — content cannot be reconstructed from them. Inference can run fully local via Ollama or Gemini CLI. No telemetry, no cloud dependencies.

---

## License

MIT
