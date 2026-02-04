-- ============================================================================
-- Migration 004: Peer Sync Support
-- ============================================================================
-- 
-- Purpose: Track source node for federated beliefs.
-- 
-- Week 2 MVP: Two nodes can share beliefs with trust weighting.
-- 
-- New columns:
--   - origin_node_did: DID of the peer that sent this belief
--   - origin_node_trust: Trust level at time of import
--   - federation_id: Stable ID across federation
-- ============================================================================

-- ============================================================================
-- PHASE 1: ADD PEER TRACKING COLUMNS
-- ============================================================================

-- DID of the origin node (null for local beliefs)
ALTER TABLE beliefs 
ADD COLUMN IF NOT EXISTS origin_node_did TEXT;

COMMENT ON COLUMN beliefs.origin_node_did IS 
    'DID of the peer node that sent this belief. NULL for locally created beliefs.';

-- Trust level at import time (for provenance)
ALTER TABLE beliefs 
ADD COLUMN IF NOT EXISTS origin_node_trust REAL;

COMMENT ON COLUMN beliefs.origin_node_trust IS 
    'Trust level we had in the origin node at import time. '
    'Confidence was weighted by this value.';

-- Federation ID for deduplication
ALTER TABLE beliefs 
ADD COLUMN IF NOT EXISTS federation_id TEXT;

COMMENT ON COLUMN beliefs.federation_id IS 
    'Stable identifier for this belief across federation. '
    'Used for deduplication on import.';

-- Content hash for duplicate detection
ALTER TABLE beliefs 
ADD COLUMN IF NOT EXISTS content_hash TEXT;

COMMENT ON COLUMN beliefs.content_hash IS 
    'SHA256 hash of content (first 16 chars). For duplicate detection.';

-- ============================================================================
-- PHASE 2: INDEXES
-- ============================================================================

-- Index for finding beliefs by origin node
CREATE INDEX IF NOT EXISTS idx_beliefs_origin_node_did 
    ON beliefs(origin_node_did) 
    WHERE origin_node_did IS NOT NULL;

-- Index for federation ID lookups (deduplication)
CREATE INDEX IF NOT EXISTS idx_beliefs_federation_id 
    ON beliefs(federation_id) 
    WHERE federation_id IS NOT NULL;

-- Index for content hash lookups (deduplication)
CREATE INDEX IF NOT EXISTS idx_beliefs_content_hash 
    ON beliefs(content_hash) 
    WHERE content_hash IS NOT NULL;

-- Index for querying local vs federated beliefs
CREATE INDEX IF NOT EXISTS idx_beliefs_is_local 
    ON beliefs(is_local);

-- ============================================================================
-- PHASE 3: BACKFILL CONTENT HASHES
-- ============================================================================

-- Generate content hashes for existing beliefs that don't have them
UPDATE beliefs 
SET content_hash = substring(encode(sha256(content::bytea), 'hex') for 16)
WHERE content_hash IS NULL;

-- ============================================================================
-- PHASE 4: VERIFICATION
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
BEGIN
    -- Verify columns exist
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns 
    WHERE table_name = 'beliefs' 
      AND column_name IN ('origin_node_did', 'origin_node_trust', 'federation_id', 'content_hash');
    
    IF col_count < 4 THEN
        RAISE WARNING 'Migration 004: expected 4 peer sync columns, found %. Some may have existed.', col_count;
    END IF;
    
    RAISE NOTICE 'Migration 004 (Peer Sync Support) complete';
END $$;
