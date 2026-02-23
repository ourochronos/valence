# Valence v2.0.0 Refactor Spec

## Motivation

The codebase carries v1 architecture (federation, bricks, VKB session tracking, 27 migrations, three env var prefixes) that doesn't serve the v2 knowledge system. Clean slate.

## Principles

1. **Append-only sources** — sources are evidence. No deletion except privacy redaction (`valence redact`). Supersession and contradiction handled via provenance relationships.
2. **One codebase, flat structure** — no vendored libs, no substrate/vkb split.
3. **One schema** — single `schema.sql`, no migration history.
4. **One config prefix** — `VALENCE_*` only, no shims.

---

## Phase 1: Schema consolidation

**Branch:** `refactor/schema`

Drop the 27-file migrations directory. Create one `schema.sql` that represents the actual v2 tables.

### Tables to keep (v2 knowledge system)
```
sources          — immutable evidence (append-only)
articles         — compiled knowledge units
article_sources  — provenance links (many-to-many with relationship type)
usage_traces     — retrieval signals for decay/prioritization
contentions      — flagged disagreements between sources
entities         — extracted entities from sources/articles
article_entities — entity-article links
system_config    — runtime config (inference backend, etc.)
```

### Tables to drop (v1/unused)
```
belief_corroborations  — v1 belief system (0 rows)
consent_records        — v1 federation (0 rows)
embedding_coverage     — tracking table (0 rows, can derive)
embedding_types        — single row, fold into system_config
audit_log              — 0 rows, not wired up
tombstones             — 0 rows, replaced by append-only + redact
vkb_sessions           — v1 conversation tracking (0 rows)
vkb_exchanges          — v1 conversation tracking (0 rows)
vkb_patterns           — v1 conversation tracking (0 rows)
vkb_session_insights   — v1 conversation tracking (0 rows)
```

### Tables to evaluate
```
article_mutations  — 117 rows. Tracks article edit history. Keep if useful for provenance audit trail, otherwise fold mutation info into articles table.
mutation_queue     — 39 rows. Pending async mutations. Keep if background compilation uses it.
```

### Add
```
-- Redaction support (append-only with privacy escape hatch)
ALTER TABLE sources ADD COLUMN redacted_at TIMESTAMPTZ;
ALTER TABLE sources ADD COLUMN redacted_by TEXT;
-- When redacted: content=NULL, embedding=NULL, metadata preserved
```

### Deliverables
- `schema.sql` — complete DDL, extensions, indexes, constraints
- `scripts/reset_db.sh` — drop and recreate from schema.sql
- Delete `migrations/` directory entirely
- Delete `src/valence/core/migrations.py`
- Delete `src/valence/lib/our_db/migrations.py`
- Update `valence init` to apply `schema.sql` directly (not migrations)
- Remove `valence migrate` CLI command

---

## Phase 2: Unvendor libs

**Branch:** `refactor/unvendor`

Delete `src/valence/lib/` entirely. Fold essential functionality into `src/valence/core/`.

### our_db (754 lines) → `core/db.py` (~150 lines)
Keep:
- `get_cursor()` context manager (sync, psycopg2, RealDictCursor)
- Connection config from `VALENCE_DB_*` env vars
- Connection pooling (ThreadedConnectionPool)

Drop:
- asyncpg support (not used in practice — server is sync)
- `ORO_DB_*` env var support
- Migration runner
- All the interface/abstract classes
- UUID generation utilities (use stdlib)

### our_embeddings → `core/embeddings.py` (~100 lines)
Keep:
- `generate_embedding(text) -> list[float]` (calls OpenAI API)
- Config from `VALENCE_EMBEDDING_MODEL`, `VALENCE_EMBEDDING_DIMS`

Drop:
- Local model support (bge-small, sentence-transformers) — we committed to OpenAI
- Federation layer, DID-based identity
- Provider registry pattern
- `our_embeddings/providers/`, `federation.py`, `registry.py`

### our_confidence → `core/confidence.py` (~50 lines)
Keep:
- Basic confidence scoring (0-1 float, dimension weighting)

Drop:
- Dimension registry (overengineered for current use)
- Abstract interfaces

### our_models → inline or delete
Check what's actually imported. If it's just dataclass definitions, inline them where used.

### Import updates
Every file that imports from `valence.lib.our_*` gets updated to `valence.core.*`.

---

## Phase 3: Flatten modules

**Branch:** `refactor/flatten`

### Delete substrate/
`src/valence/substrate/` contains MCP tool handlers for the knowledge system. These should live in the MCP server file or in `core/`.

- `substrate/tools/retrieval.py` → functionality already in `core/retrieval.py`
- `substrate/tools/articles.py` → functionality already in `core/articles.py`
- `substrate/tools/sources.py` → functionality already in `core/sources.py`
- `substrate/tools/admin.py` → functionality already in `server/admin_endpoints.py`
- `substrate/tools/contention.py` → functionality already in `core/contention.py`
- `substrate/tools/entities.py` → functionality already in... check
- `substrate/tools/definitions.py` → fold into `mcp_server.py`
- `substrate/tools/_common.py` → fold into `mcp_server.py`
- `substrate/mcp_server.py` → fold into top-level `mcp_server.py`

### Delete vkb/
`src/valence/vkb/` is the v1 conversation tracking system. All 4 tables are empty. Delete entirely.

- `vkb/tools/sessions.py` — session_start, session_end (v1)
- `vkb/tools/exchanges.py` — exchange_add (v1)
- `vkb/tools/patterns.py` — pattern_record (v1)
- `vkb/tools/insights.py` — insight extraction (v1)
- `vkb/mcp_server.py` — VKB MCP server (v1)

If any VKB concepts are useful for v2 (session tracking for usage traces?), redesign them in core/ rather than carrying the v1 implementation.

### Delete dead CLI commands
- `cli/commands/beliefs.py` — v1
- `cli/commands/conflicts.py` — v1 (contentions replaced this)
- `cli/commands/migration.py` — replaced by `valence init`

### Result: target directory structure
```
src/valence/
  __init__.py
  mcp_server.py          # single MCP entry point
  core/
    __init__.py
    db.py                # from our_db (simplified)
    embeddings.py        # from our_embeddings (OpenAI only)
    confidence.py        # from our_confidence (simplified)
    config.py            # VALENCE_* only
    articles.py
    sources.py
    retrieval.py         # hybrid RRF
    compilation.py
    contention.py
    provenance.py
    usage.py
    forgetting.py
    inference.py
    backends/            # inference backends (keep as-is)
      gemini_cli.py
      ollama.py
      openai_compat.py
      cerebras.py
    health.py
    defaults.py
    exceptions.py
    ...                  # other core modules as needed
  cli/
    __init__.py
    main.py
    config.py
    http_client.py
    output.py
    utils.py
    commands/
      articles.py
      sources.py
      compile.py
      config_cmd.py
      embeddings.py
      ingest.py
      maintenance.py
      provenance.py
      stats.py
      status.py
      unified_search.py
  server/
    __init__.py
    app.py
    config.py
    auth.py
    endpoints/
      articles.py
      sources.py
    admin_endpoints.py
    errors.py
    formatters.py
    metrics.py
    ...
```

---

## Phase 4: Env cleanup

**Branch:** can fold into Phase 2 since it's coupled to unvendoring our_db

- `VALENCE_DB_HOST`, `VALENCE_DB_PORT`, `VALENCE_DB_NAME`, `VALENCE_DB_USER`, `VALENCE_DB_PASSWORD`
- `VALENCE_EMBEDDING_PROVIDER`, `VALENCE_EMBEDDING_MODEL`, `VALENCE_EMBEDDING_DIMS`
- `VALENCE_SERVER_URL`, `VALENCE_TOKEN`, `VALENCE_OUTPUT`
- `VALENCE_MIN_CAPTURE_CONFIDENCE`, `VALENCE_USAGE_DECAY_RATE`, `VALENCE_BACKFILL_INTERVAL`
- `OPENAI_API_KEY` (external, not prefixed)

No `VKB_*`. No `ORO_*`. No shims. No backward compat.

Update `.env.example` to be the single source of truth.

---

## Phase 5: Append-only + redact

**Branch:** `refactor/append-only`

### Remove any source deletion paths
Audit all code paths that could delete or modify source content. Remove them.

### Add `valence redact`
```bash
valence redact <source-id> --reason "PII removal per user request"
```

Implementation:
- Sets `sources.content = NULL`, `sources.embedding = NULL`
- Sets `sources.redacted_at = NOW()`, `sources.redacted_by = <reason>`
- Preserves: id, type, title, metadata, created_at, all provenance links
- Articles that referenced this source get a flag: `has_redacted_sources = true`
- CLI confirms before executing (unless `--yes`)

### Supersession via API
Sources don't get deleted when better information arrives. Instead:
```bash
valence ingest "Updated info..." --supersedes <old-source-id>
```
This creates a provenance link of type `supersedes`. Retrieval ranking naturally deprioritizes the old source via freshness scoring. The old source remains as historical evidence.

---

## Execution plan

### Ordering (dependency chain)
1. **Phase 1** (schema) — standalone, no code deps
2. **Phase 2** (unvendor + env) — depends on nothing, but Phase 1 removes migration code so do after
3. **Phase 3** (flatten) — depends on Phase 2 (import paths change)
4. **Phase 5** (append-only) — depends on Phase 1 (schema changes) and Phase 3 (clean codebase)

### Parallelism
- Phase 1 and Phase 2 can run in parallel (different files)
- Phase 3 must wait for Phase 2
- Phase 5 can start after Phase 1

### Worker assignment
- Phase 1: one worker (schema + cleanup)
- Phase 2: one worker (unvendor + env + import updates)  
- Phase 3: one worker (flatten + delete — needs Phase 2 done first)
- Phase 5: one worker (redact endpoint + CLI + tests)

### Test strategy
Each phase must maintain passing tests. Tests that reference deleted modules get updated or removed. New tests for new modules.

### Data migration
Before running `schema.sql` on the real DB:
```bash
# Dump current data
pg_dump -h 127.0.0.1 -p 5433 -U valence valence > backup.sql

# Apply new schema (drops and recreates)
psql -h 127.0.0.1 -p 5433 -U valence valence < schema.sql

# Re-import data for kept tables
# (script to be written — copies from backup into new schema)
```

Since we're the only users, acceptable to just re-ingest if needed.
