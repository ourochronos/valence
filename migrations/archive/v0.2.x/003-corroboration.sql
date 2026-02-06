-- ============================================================================
-- Migration 003: Corroboration Tracking
-- ============================================================================
-- 
-- Purpose: Track how many independent sources confirm each belief.
-- 
-- Gap: Beliefs don't track corroboration from multiple sources.
-- 
-- New columns:
--   - corroboration_count: Number of independent sources confirming this belief
--   - corroborating_sources: JSONB array of source metadata
-- ============================================================================

-- ============================================================================
-- PHASE 1: ADD CORROBORATION COLUMNS
-- ============================================================================

-- Count of independent corroborating sources
ALTER TABLE beliefs 
ADD COLUMN IF NOT EXISTS corroboration_count INTEGER NOT NULL DEFAULT 0;

COMMENT ON COLUMN beliefs.corroboration_count IS 
    'Number of independent sources that corroborate this belief. '
    'Used to boost confidence_corroboration dimension.';

-- Detailed source tracking as JSONB array
-- Format: [{"source_did": "did:...", "similarity": 0.95, "corroborated_at": "2025-..."}]
ALTER TABLE beliefs 
ADD COLUMN IF NOT EXISTS corroborating_sources JSONB NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN beliefs.corroborating_sources IS 
    'Array of corroborating source metadata: source_did, similarity score, timestamp. '
    'Same source (by DID) counts only once.';

-- ============================================================================
-- PHASE 2: CONSTRAINTS
-- ============================================================================

-- Ensure count is non-negative
DO $$ 
BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT corroboration_count_non_negative 
        CHECK (corroboration_count >= 0);
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'corroboration_count_non_negative constraint already exists';
END $$;

-- Ensure corroborating_sources is an array
DO $$ 
BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT corroborating_sources_is_array 
        CHECK (jsonb_typeof(corroborating_sources) = 'array');
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'corroborating_sources_is_array constraint already exists';
END $$;

-- ============================================================================
-- PHASE 3: INDEXES
-- ============================================================================

-- Index for finding highly corroborated beliefs
CREATE INDEX IF NOT EXISTS idx_beliefs_corroboration_count 
    ON beliefs(corroboration_count DESC) 
    WHERE corroboration_count > 0 AND superseded_by_id IS NULL;

-- GIN index for querying corroborating sources
CREATE INDEX IF NOT EXISTS idx_beliefs_corroborating_sources 
    ON beliefs USING GIN (corroborating_sources);

-- ============================================================================
-- PHASE 4: HELPER FUNCTIONS
-- ============================================================================

-- Function to add a corroborating source (deduplicates by source_did)
CREATE OR REPLACE FUNCTION add_corroborating_source(
    p_belief_id UUID,
    p_source_did TEXT,
    p_similarity REAL,
    p_boost_confidence BOOLEAN DEFAULT TRUE
) RETURNS BOOLEAN AS $$
DECLARE
    v_sources JSONB;
    v_existing JSONB;
    v_new_source JSONB;
    v_count INTEGER;
    v_new_corroboration REAL;
BEGIN
    -- Get current sources
    SELECT corroborating_sources INTO v_sources 
    FROM beliefs WHERE id = p_belief_id;
    
    IF v_sources IS NULL THEN
        RETURN FALSE;
    END IF;
    
    -- Check if source already exists
    SELECT elem INTO v_existing
    FROM jsonb_array_elements(v_sources) AS elem
    WHERE elem->>'source_did' = p_source_did
    LIMIT 1;
    
    IF v_existing IS NOT NULL THEN
        -- Source already counted
        RETURN FALSE;
    END IF;
    
    -- Add new source
    v_new_source := jsonb_build_object(
        'source_did', p_source_did,
        'similarity', p_similarity,
        'corroborated_at', NOW()::text
    );
    
    v_sources := v_sources || v_new_source;
    v_count := jsonb_array_length(v_sources);
    
    -- Calculate new corroboration confidence (asymptotic approach to 1.0)
    -- Formula: 1 - (1 / (1 + count * 0.3))
    -- 0 sources: 0.0, 1 source: 0.23, 2 sources: 0.38, 5 sources: 0.60, 10 sources: 0.75
    v_new_corroboration := 1.0 - (1.0 / (1.0 + v_count * 0.3));
    
    -- Update belief
    IF p_boost_confidence THEN
        UPDATE beliefs
        SET corroborating_sources = v_sources,
            corroboration_count = v_count,
            confidence_corroboration = GREATEST(confidence_corroboration, v_new_corroboration),
            modified_at = NOW()
        WHERE id = p_belief_id;
    ELSE
        UPDATE beliefs
        SET corroborating_sources = v_sources,
            corroboration_count = v_count,
            modified_at = NOW()
        WHERE id = p_belief_id;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Function to get corroboration details for a belief
CREATE OR REPLACE FUNCTION get_belief_corroboration(p_belief_id UUID)
RETURNS TABLE(
    belief_id UUID,
    corroboration_count INTEGER,
    confidence_corroboration REAL,
    sources JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        b.id,
        b.corroboration_count,
        b.confidence_corroboration,
        b.corroborating_sources
    FROM beliefs b
    WHERE b.id = p_belief_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PHASE 5: VERIFICATION
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
BEGIN
    -- Verify columns exist
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns 
    WHERE table_name = 'beliefs' 
      AND column_name IN ('corroboration_count', 'corroborating_sources');
    
    IF col_count != 2 THEN
        RAISE EXCEPTION 'Migration failed: expected 2 corroboration columns, found %', col_count;
    END IF;
    
    RAISE NOTICE 'Migration 003 (Corroboration Tracking) complete: 2 columns added';
END $$;
