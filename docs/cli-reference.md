# Valence CLI Reference

Complete reference for `valence` v2 commands.

For architecture context, see [architecture.md](architecture.md).

---

## Global Options

All commands accept these global flags before the subcommand:

| Flag | Default | Description |
|---|---|---|
| `--server URL` | `http://127.0.0.1:8420` | Valence server URL. Env: `VALENCE_SERVER_URL` |
| `--token TOKEN` | — | Auth token. Env: `VALENCE_TOKEN` |
| `--output {json,text,table}` | `text` | Output format. Env: `VALENCE_OUTPUT` |
| `--json` | — | Shorthand for `--output json` |
| `--timeout SECS` | `30` | Request timeout in seconds |

```bash
valence --json articles search "query"
valence --server http://remote:8420 --output table sources list
```

---

## `valence sources`

Manage knowledge sources (C1). Sources are immutable after ingestion.

### `valence sources list`

List recent sources.

```
valence sources list [--type TYPE] [--limit N] [--offset N]
```

| Option | Default | Description |
|---|---|---|
| `--type TYPE` | — | Filter by source type |
| `--limit N`, `-n N` | `20` | Maximum results |
| `--offset N` | `0` | Pagination offset |

**Source types:** `document`, `conversation`, `web`, `code`, `observation`, `tool_output`, `user_input`

**Examples:**

```bash
# List 20 most recent sources
valence sources list

# List only web sources
valence sources list --type web

# Paginate
valence sources list --limit 50 --offset 50

# JSON output
valence --json sources list --type document
```

---

### `valence sources get`

Retrieve a single source by UUID.

```
valence sources get <source-id>
```

**Examples:**

```bash
valence sources get 550e8400-e29b-41d4-a716-446655440000
valence --json sources get 550e8400-e29b-41d4-a716-446655440000
```

---

### `valence sources ingest`

Ingest new source material. Ingestion is cheap — embedding and entity extraction are deferred.

```
valence sources ingest <content> [--type TYPE] [--title TITLE] [--url URL]
```

| Option | Default | Description |
|---|---|---|
| `--type TYPE`, `-t TYPE` | `document` | Source type |
| `--title TITLE` | — | Human-readable title |
| `--url URL` | — | Canonical URL |

**Examples:**

```bash
# Ingest a document snippet
valence sources ingest "Python 3.13 replaced the GIL with per-interpreter locks." \
  --type document --title "Python 3.13 release notes"

# Ingest a web page excerpt
valence sources ingest "$(curl -s https://example.com/article | extract-text)" \
  --type web --url https://example.com/article --title "Example Article"

# Ingest from a file
valence sources ingest "$(cat meeting-notes.txt)" \
  --type conversation --title "Team meeting 2026-02-21"

# Ingest a code snippet
valence sources ingest "$(cat src/module.py)" \
  --type code --title "module.py"

# Ingest a tool output
valence sources ingest "$(valence stats)" \
  --type tool_output --title "valence stats output"
```

**Returns:** Source ID, fingerprint, reliability score, and creation timestamp.

---

### `valence sources search`

Full-text search across all ingested sources.

```
valence sources search <query> [--limit N]
```

| Option | Default | Description |
|---|---|---|
| `--limit N`, `-n N` | `20` | Maximum results |

**Examples:**

```bash
valence sources search "Python concurrency"
valence sources search "GIL per-interpreter" --limit 5
valence --json sources search "thread safety"
```

---

## `valence articles`

Manage compiled knowledge articles (C2, C3). Articles are created and updated on demand through use.

### `valence articles list`

List recent articles.

```
valence articles list [--limit N]
```

| Option | Default | Description |
|---|---|---|
| `--limit N`, `-n N` | `10` | Maximum results |

**Examples:**

```bash
valence articles list
valence articles list --limit 50
valence --output table articles list
```

---

### `valence articles get`

Retrieve a single article by UUID, optionally including its full provenance chain.

```
valence articles get <article-id> [--provenance]
```

| Option | Default | Description |
|---|---|---|
| `--provenance`, `-p` | off | Include full source provenance |

**Examples:**

```bash
# Get article content
valence articles get 550e8400-e29b-41d4-a716-446655440000

# Get article with provenance (shows all contributing sources and relationship types)
valence articles get 550e8400-e29b-41d4-a716-446655440000 --provenance

# JSON output with provenance — useful for scripting
valence --json articles get 550e8400-e29b-41d4-a716-446655440000 --provenance
```

**Provenance fields** (when `--provenance` is set):
- `sources` — list of contributing sources
- `relationship` — how each source contributed (`originates`, `confirms`, `supersedes`, `contradicts`, `contends`)
- `contentions` — any active disagreements flagged on this article

---

### `valence articles create`

Create a new article manually. Use this for knowledge that doesn't derive from a traditional source (e.g., operator-authored summaries, agent-synthesized insights).

```
valence articles create <content> [--title TITLE] [--domain DOMAIN] [--author-type TYPE]
```

| Option | Default | Description |
|---|---|---|
| `--title TITLE`, `-t TITLE` | — | Article title |
| `--domain DOMAIN`, `-d DOMAIN` | — | Domain path tag (repeatable) |
| `--author-type TYPE` | `agent` | Author type: `system`, `operator`, or `agent` |

**Examples:**

```bash
# Create an operator-authored summary
valence articles create "Python's threading model changed significantly in 3.13." \
  --title "Python 3.13 threading summary" \
  --author-type operator \
  --domain engineering --domain python

# Agent-authored synthesis
valence articles create "Based on three sources, the consensus is that GIL removal improves I/O throughput." \
  --title "GIL removal impact" \
  --author-type agent

# JSON output (returns new article ID)
valence --json articles create "Content here" --title "My Article"
```

---

### `valence articles search`

Search compiled articles by query. This is the primary retrieval interface. When a query surfaces relevant ungrouped sources that haven't been compiled yet, compilation is triggered on demand.

```
valence articles search <query> [--limit N] [--domain DOMAIN]
```

| Option | Default | Description |
|---|---|---|
| `--limit N`, `-n N` | `10` | Maximum results |
| `--domain DOMAIN`, `-d DOMAIN` | — | Domain path filter (repeatable) |

Results are ranked by relevance × confidence × freshness. Each result includes provenance summary, freshness indicators, and any active contentions.

**Examples:**

```bash
# Basic semantic search
valence articles search "Python concurrency and threading"

# Narrow to a domain
valence articles search "thread safety" --domain engineering

# Multiple domain filters
valence articles search "performance" --domain engineering --domain python

# Increase result count
valence articles search "database design" --limit 20

# JSON (for piping/scripting)
valence --json articles search "query" | jq '.data.results[0].content'
```

---

## `valence provenance`

Manage source provenance for articles (C5). Provenance answers: "where did this come from?"

### `valence provenance get`

List all sources that contributed to an article, with their relationship types.

```
valence provenance get <article-id>
```

**Examples:**

```bash
valence provenance get 550e8400-e29b-41d4-a716-446655440000

# JSON for scripting
valence --json provenance get 550e8400-e29b-41d4-a716-446655440000
```

**Returns:**
- Full list of contributing sources
- Relationship type for each (`originates`, `confirms`, `supersedes`, `contradicts`, `contends`)
- Confidence score and corroboration depth
- Any active contentions

---

### `valence provenance trace`

Trace a specific claim text back to the sources that likely contributed it. Uses inference to match the claim against contributing sources.

```
valence provenance trace <article-id> <claim-text>
```

**Examples:**

```bash
# Trace a specific statement
valence provenance trace 550e8400-e29b-41d4-a716-446655440000 \
  "per-interpreter locks replace the global lock"

# Quote the claim if it contains spaces
valence provenance trace "$ARTICLE_ID" "the GIL was removed in Python 3.13"

# JSON output
valence --json provenance trace "$ARTICLE_ID" "$CLAIM"
```

**Returns:** Ranked list of sources most likely responsible for the claim, with match confidence and excerpts from each source.

> **Note:** On-demand tracing uses inference. In degraded mode, this falls back to keyword matching against contributing sources.

---

### `valence provenance link`

Manually link a source to an article with an explicit typed relationship. Use this to correct or supplement automatically detected relationships.

```
valence provenance link <article-id> <source-id> [--relationship TYPE] [--notes TEXT]
```

| Option | Default | Description |
|---|---|---|
| `--relationship TYPE`, `-r TYPE` | `confirms` | Relationship type |
| `--notes TEXT` | — | Optional notes about the relationship |

**Relationship types:**
- `originates` — source introduced the information first captured in this article
- `confirms` — independent corroboration of information in the article
- `supersedes` — newer source authoritatively replacing older information
- `contradicts` — source actively disagreeing with the article
- `contends` — source differing without direct contradiction (tension preserved)

**Examples:**

```bash
# Mark a source as confirming (corroborating) an article
valence provenance link $ARTICLE_ID $SOURCE_ID --relationship confirms

# Mark a source as contradicting an article
valence provenance link $ARTICLE_ID $SOURCE_ID \
  --relationship contradicts \
  --notes "2026-02 benchmarks contradict the 2025 performance claims"

# Mark a source as superseding older content
valence provenance link $ARTICLE_ID $SOURCE_ID \
  --relationship supersedes \
  --notes "Python 3.13 release notes supersede 3.12 threading docs"

# Originating source (source introduced the core information)
valence provenance link $ARTICLE_ID $SOURCE_ID --relationship originates
```

---

## `valence config`

Configure system settings.

### `valence config inference show`

Display the current inference backend configuration.

```
valence config inference show
```

```bash
valence config inference show
valence --json config inference show
```

**Output:**
```
Inference config (last updated: 2026-02-21 16:00:00):
{
  "provider": "gemini",
  "model": "gemini-2.5-flash"
}
```

---

### `valence config inference gemini`

Configure Gemini 2.5 Flash via the local `gemini` CLI. No API key required.

```
valence config inference gemini [--model MODEL]
```

| Option | Default | Description |
|---|---|---|
| `--model MODEL` | `gemini-2.5-flash` | Gemini model name |

```bash
# Use default (gemini-2.5-flash)
valence config inference gemini

# Specify a different model
valence config inference gemini --model gemini-2.0-flash
```

**Requires:** `gemini` CLI installed and authenticated (`gemini auth`).

---

### `valence config inference cerebras`

Configure Cerebras Cloud as the inference backend. Ultra-low-latency; well-suited for classification tasks.

```
valence config inference cerebras --api-key KEY [--model MODEL]
```

| Option | Default | Description |
|---|---|---|
| `--api-key KEY` | (required) | Cerebras API key |
| `--model MODEL` | `llama-4-scout-17b-16e-instruct` | Cerebras model name |

```bash
valence config inference cerebras --api-key csk-xxxxx

valence config inference cerebras \
  --api-key csk-xxxxx \
  --model llama-4-scout-17b-16e-instruct
```

**API keys:** https://cloud.cerebras.ai/

> The API key is stored in the `system_config` table. `config inference show` masks it as `***`.

---

### `valence config inference ollama`

Configure a local Ollama instance for fully offline inference.

```
valence config inference ollama [--host HOST] [--model MODEL]
```

| Option | Default | Description |
|---|---|---|
| `--host HOST` | `http://localhost:11434` | Ollama server URL |
| `--model MODEL` | `qwen3:30b` | Ollama model name |

```bash
# Use defaults (local Ollama, qwen3:30b)
valence config inference ollama

# Remote Ollama instance
valence config inference ollama --host http://192.168.1.50:11434

# Specify model
valence config inference ollama --model llama3.3:70b

# Remote host + model
valence config inference ollama \
  --host http://derptop.local:11434 \
  --model qwen3:30b
```

**Requires:** `ollama pull <model>` run first.

---

## Legacy Commands

These commands are preserved from v1 for compatibility. New code should prefer `sources`, `articles`, and `provenance`.

| Command | Description |
|---|---|
| `valence init` | Initialize database schema (apply all migrations) |
| `valence add <content>` | Add a belief (legacy) |
| `valence query <text>` | Search beliefs (legacy) |
| `valence list` | List recent beliefs (legacy) |
| `valence conflicts` | Detect contradicting beliefs |
| `valence stats` | Show database statistics |
| `valence export` | Export beliefs |
| `valence import` | Import beliefs |
| `valence embeddings` | Embedding management |
| `valence migrate` | Database migration management |
| `valence qos` | QoS management |
| `valence maintenance` | Run database maintenance |
| `valence verify-chains` | Verify supersession chain integrity |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VALENCE_SERVER_URL` | `http://127.0.0.1:8420` | Server URL |
| `VALENCE_TOKEN` | — | Auth token |
| `VALENCE_OUTPUT` | `text` | Output format (`json`, `text`, `table`) |
| `VALENCE_DB_HOST` | `localhost` | PostgreSQL host |
| `VALENCE_DB_PORT` | `5433` | PostgreSQL port |
| `VALENCE_DB_NAME` | `valence` | Database name |
| `VALENCE_DB_USER` | `valence` | Database user |
| `VALENCE_DB_PASSWORD` | `valence` | Database password |
| `VALENCE_HOST` | `127.0.0.1` | Server bind host |
| `VALENCE_PORT` | `8420` | Server bind port |

Variables can be set in a `.env` file in the current directory or `~/.valence/.env`.
