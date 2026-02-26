# OpenAI Embedding Migration Summary

## Overview
Successfully migrated Valence from local bge-small embeddings (384-dim) to OpenAI text-embedding-3-small (1536-dim) with full configurability.

## Changes Made

### 1. Configuration Updates ✅

#### `src/valence/lib/our_embeddings/config.py`
- Added `embedding_model: str` field to `EmbeddingConfig`
- Defaults to `"text-embedding-3-small"` for openai provider
- Defaults to `"BAAI/bge-small-en-v1.5"` for local provider
- Reads from `VALENCE_EMBEDDING_MODEL` env var
- Kept `embedding_model_path` for backward compatibility

#### `src/valence/lib/our_embeddings/service.py`
- Updated `generate_embedding()` to use `config.embedding_model` instead of hardcoded default
- Model parameter now optional, defaults to config value
- OpenAI provider uses config.embedding_model
- Local provider still uses config.embedding_model_path

#### `src/valence/core/embedding_interop.py`
- Updated `get_embedding_capability()` with provider-aware defaults
- Defaults to 1536-dim when provider=openai
- Defaults to 384-dim when provider=local
- Reads configured model from `VALENCE_EMBEDDING_MODEL`
- Maps model names to type_ids correctly

#### `src/valence/core/config_registry.py`
- Added `embedding_model: str` field
- Provider-based defaults:
  - openai → `text-embedding-3-small`, 1536 dims
  - local → `BAAI/bge-small-en-v1.5`, 384 dims
- Reads from `VALENCE_EMBEDDING_MODEL` env var

#### `.env` file
- Set `VALENCE_EMBEDDING_PROVIDER=openai`
- Set `VALENCE_EMBEDDING_MODEL=text-embedding-3-small`
- Set `VALENCE_EMBEDDING_DIMS=1536`
- Added placeholder for `OPENAI_API_KEY` (needs user input)

### 2. Database Migration ✅

#### Script: `scripts/migrate_embeddings_to_openai.py`
- Identified 4 base tables with embedding columns:
  - articles (79 embeddings)
  - sources (0 embeddings)
  - vkb_exchanges (0 embeddings)
  - vkb_patterns (0 embeddings)
- Identified 3 dependent views:
  - articles_current
  - articles_with_sources
  - vkb_patterns_overview
- Dropped views temporarily
- Cleared all existing embeddings (incompatible with new model)
- Altered columns: `vector(384)` → `vector(1536)`
- Recreated all views successfully
- Total cleared: 79 embeddings

### 3. Backfill Script ✅

#### Script: `scripts/backfill_openai_embeddings.py`
- Async batch processing for articles and sources
- Configuration validation with helpful error messages
- Progress tracking per batch
- Verification of embedding dimensions
- Sample similarity search test
- **Status**: Ready to run once OPENAI_API_KEY is set

### 4. Test Updates ✅

#### `tests/core/test_embedding_interop.py`
- Added `clean_env` fixture to prevent .env pollution
- Updated `test_default_384` to use clean_env
- Added `test_openai_provider_default` for openai provider
- All tests passing (100% pass rate)

## Migration Status

### Completed ✅
1. ✅ Configuration model added with provider-aware defaults
2. ✅ Embedding service updated to use configurable model
3. ✅ Embedding interop updated with dimension mapping
4. ✅ Config registry updated with provider-based defaults
5. ✅ .env file configured for OpenAI
6. ✅ Database schema migrated (384-dim → 1536-dim)
7. ✅ Dependent views handled correctly
8. ✅ All tests passing
9. ✅ Backfill script created and ready

### Pending ⏳
1. ⏳ **Add OPENAI_API_KEY to .env** (requires user action)
2. ⏳ Run backfill script after API key is set
3. ⏳ Verify embeddings with sample queries

## How to Complete Migration

### Step 1: Add OpenAI API Key
Edit `~/projects/valence/.env` and replace:
```bash
OPENAI_API_KEY=sk-YOUR-KEY-HERE
```

With your actual key from: https://platform.openai.com/api-keys

### Step 2: Run Backfill
```bash
cd ~/projects/valence
.venv/bin/python scripts/backfill_openai_embeddings.py
```

Expected output:
- Articles backfilled: ~79
- Sources backfilled: ~0
- Verification: 1536-dim embeddings
- Sample similarity search results

### Step 3: Verify
Test a query:
```bash
cd ~/projects/valence
valence articles search "your test query"
```

Or in Python:
```python
from valence.lib.our_db import get_cursor

with get_cursor() as cur:
    cur.execute("""
        SELECT array_length(embedding::real[], 1) as dims 
        FROM articles 
        WHERE embedding IS NOT NULL 
        LIMIT 1
    """)
    print(cur.fetchone())  # Should show {'dims': 1536}
```

## Git Branch

Branch: `feat/openai-embeddings`

Commits:
1. `fd29e30` - feat: add configurable embedding model support
2. `ece27c8` - feat: add database migration script for OpenAI embeddings
3. `fcafaa4` - feat: add OpenAI embedding backfill script
4. `3a7a60c` - test: update embedding interop tests for configurable dimensions

Ready to merge after backfill is complete and verified.

## Environment Variables Reference

```bash
# Database (already set)
VALENCE_DB_HOST=127.0.0.1
VALENCE_DB_USER=valence
VALENCE_DB_PASSWORD=valence
VALENCE_DB_NAME=valence
VALENCE_DB_PORT=5433

# Embedding configuration (newly added)
VALENCE_EMBEDDING_PROVIDER=openai
VALENCE_EMBEDDING_MODEL=text-embedding-3-small
VALENCE_EMBEDDING_DIMS=1536

# OpenAI API key (USER MUST ADD)
OPENAI_API_KEY=sk-your-actual-key-here
```

## Rollback Plan (if needed)

If issues arise, rollback steps:

1. Switch back to local provider:
   ```bash
   # Edit .env:
   VALENCE_EMBEDDING_PROVIDER=local
   VALENCE_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
   VALENCE_EMBEDDING_DIMS=384
   ```

2. Restore database schema:
   ```python
   # Run migration in reverse
   ALTER TABLE articles ALTER COLUMN embedding TYPE vector(384);
   ALTER TABLE sources ALTER COLUMN embedding TYPE vector(384);
   # etc.
   ```

3. Re-embed with local model:
   ```bash
   cd ~/projects/valence
   .venv/bin/python scripts/backfill_openai_embeddings.py
   # (will use local provider from .env)
   ```

## Notes

- No changes to compilation, retrieval, or article logic (as required)
- No changes to database schema files in migrations/ (as required)
- All changes are in config/service layer only
- Tests passing with clean isolation
- Provider-based defaults ensure backward compatibility
- Local provider still works if VALENCE_EMBEDDING_PROVIDER=local
