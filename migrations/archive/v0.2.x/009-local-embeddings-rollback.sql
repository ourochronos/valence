-- ============================================================================
-- Rollback Migration 009: Revert 384-dimensional embeddings
-- ============================================================================
-- This script reverses migration 009, restoring the 1536-dim embedding setup.
-- 
-- USE THIS WHEN:
-- - 384-dim embeddings aren't working as expected
-- - Need to return to OpenAI embeddings
-- - Issues with local embedding model
-- ============================================================================

-- ============================================================================
-- STEP 1: Drop HNSW indexes on 384-dim columns
-- ============================================================================

DROP INDEX IF EXISTS idx_beliefs_embedding_384;
DROP INDEX IF EXISTS idx_vkb_exchanges_embedding_384;
DROP INDEX IF EXISTS idx_vkb_patterns_embedding_384;

-- ============================================================================
-- STEP 2: Drop 384-dimensional embedding columns
-- ============================================================================
-- WARNING: This will delete all 384-dim embeddings! Ensure backups exist.

ALTER TABLE beliefs DROP COLUMN IF EXISTS embedding_384;
ALTER TABLE vkb_exchanges DROP COLUMN IF EXISTS embedding_384;
ALTER TABLE vkb_patterns DROP COLUMN IF EXISTS embedding_384;

-- ============================================================================
-- STEP 3: Recreate IVFFlat indexes on original 1536-dim columns
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_beliefs_embedding 
ON beliefs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vkb_exchanges_embedding 
ON vkb_exchanges USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_vkb_patterns_embedding 
ON vkb_patterns USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- ============================================================================
-- STEP 4: Restore OpenAI as default embedding type
-- ============================================================================

-- Mark BGE as non-default (if it exists)
UPDATE embedding_types 
SET is_default = FALSE 
WHERE id = 'bge_small_en_v15';

-- Restore OpenAI as default
UPDATE embedding_types 
SET is_default = TRUE 
WHERE id = 'openai_text3_small';

-- Optionally mark BGE as deprecated
UPDATE embedding_types 
SET status = 'deprecated' 
WHERE id = 'bge_small_en_v15';

-- ============================================================================
-- STEP 5: Restore original stats function
-- ============================================================================

CREATE OR REPLACE FUNCTION stats_beliefs()
RETURNS TABLE(
    total_beliefs BIGINT,
    active_beliefs BIGINT,
    beliefs_with_embeddings BIGINT,
    beliefs_with_derivations BIGINT,
    tombstoned_beliefs BIGINT,
    unique_holders BIGINT,
    avg_confidence REAL
) AS $$
SELECT
    COUNT(*),
    COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL),
    COUNT(*) FILTER (WHERE embedding IS NOT NULL),
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
-- VERIFY ROLLBACK
-- ============================================================================

DO $$
DECLARE
    old_cols_exist INTEGER;
    old_indexes INTEGER;
    default_embedding TEXT;
BEGIN
    -- Verify old columns still exist
    SELECT COUNT(*) INTO old_cols_exist 
    FROM information_schema.columns 
    WHERE table_name IN ('beliefs', 'vkb_exchanges', 'vkb_patterns')
      AND column_name = 'embedding';
    
    -- Verify old indexes recreated
    SELECT COUNT(*) INTO old_indexes
    FROM pg_indexes
    WHERE indexname IN ('idx_beliefs_embedding', 'idx_vkb_exchanges_embedding', 'idx_vkb_patterns_embedding');
    
    -- Check default is back to OpenAI
    SELECT id INTO default_embedding
    FROM embedding_types
    WHERE is_default = TRUE;
    
    IF old_cols_exist != 3 THEN
        RAISE WARNING 'Rollback issue: Expected 3 embedding columns, got %', old_cols_exist;
    END IF;
    
    RAISE NOTICE 'Rollback 009 complete: % original embedding columns, % IVFFlat indexes, default embedding: %', 
                 old_cols_exist, old_indexes, default_embedding;
END $$;
