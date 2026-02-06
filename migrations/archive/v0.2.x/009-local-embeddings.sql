-- ============================================================================
-- Migration 009: Add 384-dimensional embeddings (bge-small-en-v1.5)
-- ============================================================================
-- This reduces storage by 75% and enables local-first operation
-- 
-- STRATEGY: Additive migration - add new 384-dim columns alongside existing
-- 1536-dim columns. Old columns are preserved for rollback. After verification,
-- drop old columns with migration 010.
-- ============================================================================

-- ============================================================================
-- STEP 1: Drop existing vector indexes on 1536-dim columns
-- ============================================================================
-- These use IVFFlat which isn't optimal for updates. We'll create HNSW on new columns.

DROP INDEX IF EXISTS idx_beliefs_embedding;
DROP INDEX IF EXISTS idx_vkb_exchanges_embedding;
DROP INDEX IF EXISTS idx_vkb_patterns_embedding;

-- ============================================================================
-- STEP 2: Add new 384-dimensional embedding columns
-- ============================================================================
-- Keep old 'embedding' columns for rollback

ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS embedding_384 VECTOR(384);
ALTER TABLE vkb_exchanges ADD COLUMN IF NOT EXISTS embedding_384 VECTOR(384);
ALTER TABLE vkb_patterns ADD COLUMN IF NOT EXISTS embedding_384 VECTOR(384);

-- ============================================================================
-- STEP 3: Create HNSW indexes on new columns
-- ============================================================================
-- HNSW is better than IVFFlat for:
-- - Incremental updates (no need to rebuild after inserts)
-- - Better recall at low latency
-- - Smaller datasets (<1M rows)
--
-- Parameters:
--   m = 16: Max connections per node (higher = better recall, more memory)
--   ef_construction = 64: Search width during build (higher = better quality, slower build)

CREATE INDEX IF NOT EXISTS idx_beliefs_embedding_384 
ON beliefs USING hnsw (embedding_384 vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_vkb_exchanges_embedding_384 
ON vkb_exchanges USING hnsw (embedding_384 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_vkb_patterns_embedding_384 
ON vkb_patterns USING hnsw (embedding_384 vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- ============================================================================
-- STEP 4: Register new embedding type
-- ============================================================================
-- BGE-small-en-v1.5 from BAAI:
-- - 384 dimensions (75% smaller than OpenAI's 1536)
-- - Runs locally (no API costs)
-- - English-focused, excellent for general text
-- - Apache 2.0 license

INSERT INTO embedding_types (id, provider, model, dimensions, is_default, status)
VALUES ('bge_small_en_v15', 'local', 'BAAI/bge-small-en-v1.5', 384, TRUE, 'active')
ON CONFLICT (id) DO UPDATE SET 
    is_default = TRUE,
    status = 'active';

-- Mark old OpenAI type as non-default
UPDATE embedding_types 
SET is_default = FALSE 
WHERE id != 'bge_small_en_v15';

-- ============================================================================
-- STEP 5: Update stats function to track 384-dim coverage
-- ============================================================================

CREATE OR REPLACE FUNCTION stats_beliefs()
RETURNS TABLE(
    total_beliefs BIGINT,
    active_beliefs BIGINT,
    beliefs_with_embeddings BIGINT,
    beliefs_with_embeddings_384 BIGINT,
    beliefs_with_derivations BIGINT,
    tombstoned_beliefs BIGINT,
    unique_holders BIGINT,
    avg_confidence REAL
) AS $$
SELECT
    COUNT(*),
    COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL),
    COUNT(*) FILTER (WHERE embedding IS NOT NULL),
    COUNT(*) FILTER (WHERE embedding_384 IS NOT NULL),
    (SELECT COUNT(*) FROM belief_derivations),
    COUNT(*) FILTER (WHERE deleted_at IS NOT NULL),
    COUNT(DISTINCT holder_id),
    AVG(compute_confidence_overall(
        confidence_source, confidence_method, confidence_consistency,
        confidence_freshness, confidence_corroboration, confidence_applicability
    ))::REAL
FROM beliefs;
$$ LANGUAGE SQL STABLE;

-- ============================================================================
-- MIGRATION NOTES
-- ============================================================================
-- After running this migration:
-- 1. Run reembed_all.py to populate embedding_384 columns with new embeddings
-- 2. Verify queries return expected results with 384-dim embeddings
-- 3. Monitor performance (HNSW should be faster for small datasets)
-- 4. Once verified, run migration 010 to drop old columns and save space
--
-- Old columns (embedding VECTOR(1536)) are kept for:
-- - Rollback capability
-- - Parallel testing during transition
-- - Gradual migration of dependent services
-- ============================================================================

-- Verify migration
DO $$
DECLARE
    new_cols INTEGER;
    new_indexes INTEGER;
    default_embedding TEXT;
BEGIN
    -- Count new columns
    SELECT COUNT(*) INTO new_cols 
    FROM information_schema.columns 
    WHERE table_name IN ('beliefs', 'vkb_exchanges', 'vkb_patterns')
      AND column_name = 'embedding_384';
    
    -- Count new indexes
    SELECT COUNT(*) INTO new_indexes
    FROM pg_indexes
    WHERE indexname LIKE '%embedding_384%';
    
    -- Check default embedding type
    SELECT id INTO default_embedding
    FROM embedding_types
    WHERE is_default = TRUE;
    
    IF new_cols != 3 THEN
        RAISE EXCEPTION 'Migration 009 failed: Expected 3 new columns, got %', new_cols;
    END IF;
    
    IF new_indexes != 3 THEN
        RAISE EXCEPTION 'Migration 009 failed: Expected 3 new indexes, got %', new_indexes;
    END IF;
    
    RAISE NOTICE 'Migration 009 complete: % new embedding_384 columns, % HNSW indexes, default embedding type: %', 
                 new_cols, new_indexes, default_embedding;
END $$;
