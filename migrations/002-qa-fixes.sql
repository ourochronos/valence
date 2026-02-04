-- ============================================================================
-- Migration 002: QA Schema Compliance Fixes
-- ============================================================================
-- 
-- Purpose: Address 4 gaps identified by QA between spec and implementation.
-- 
-- Gaps Addressed:
--   1. holder_id column - ownership model (spec: holder_id, not just source_id)
--   2. version column - explicit versioning (spec: version field, not just supersession)
--   3. content_hash column - deduplication (spec: SHA-256 hash of content)
--   4. visibility enum/column - federation privacy (spec: private/federated/public)
--
-- Constraints:
--   - Backward compatible: existing data preserved
--   - Idempotent: safe to run multiple times
--   - No data loss: nullable columns where existing rows lack data
--
-- Note: Migration 001 may have already added these. This migration ensures
-- they exist and are properly configured regardless of starting state.
-- ============================================================================

-- ============================================================================
-- PHASE 1: VISIBILITY ENUM
-- ============================================================================
-- Required for federation privacy levels per FEDERATION_SCHEMA.md

DO $$ 
BEGIN
    CREATE TYPE visibility_level AS ENUM ('private', 'federated', 'public');
    RAISE NOTICE 'Created visibility_level enum';
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'visibility_level enum already exists';
END $$;

-- ============================================================================
-- PHASE 2: ADD MISSING COLUMNS
-- ============================================================================

-- holder_id: WHO owns this belief (distinct from source_id which is WHERE it came from)
-- Spec: Required for multi-holder federation model
-- Nullable for backward compatibility - existing beliefs get system holder
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS holder_id UUID;

COMMENT ON COLUMN beliefs.holder_id IS 
    'UUID of the holder (agent/node) who owns this belief. '
    'Distinct from source_id (provenance). Required for federation.';

-- version: Explicit version number for efficient sync and conflict resolution
-- Spec: Required alongside supersession chain for federation
-- Default 1 for existing beliefs
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

COMMENT ON COLUMN beliefs.version IS 
    'Explicit version number for this belief. Increments on update. '
    'Complements supersession chain for efficient federation sync.';

-- content_hash: SHA-256 hash for deduplication and integrity
-- Spec: Required for efficient federation dedup
-- Will be computed for existing rows
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS content_hash CHAR(64);

COMMENT ON COLUMN beliefs.content_hash IS 
    'SHA-256 hex hash of content for deduplication and integrity verification.';

-- visibility: Privacy level for federation sharing
-- Spec: Required for graduated federation privacy
-- Default private (privacy-first)
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS visibility visibility_level DEFAULT 'private';

COMMENT ON COLUMN beliefs.visibility IS 
    'Federation visibility: private (holder only), federated (trusted nodes), public (anyone).';

-- ============================================================================
-- PHASE 3: CONSTRAINTS
-- ============================================================================

-- version must be positive
DO $$ 
BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT qa_version_positive CHECK (version > 0);
    RAISE NOTICE 'Added version_positive constraint';
EXCEPTION
    WHEN duplicate_object THEN 
        RAISE NOTICE 'version_positive constraint already exists';
END $$;

-- ============================================================================
-- PHASE 4: COMPUTE CONTENT HASH FOR EXISTING ROWS
-- ============================================================================

-- Enable pgcrypto for SHA-256 if not present
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Compute hash for any rows missing it
UPDATE beliefs 
SET content_hash = encode(digest(content, 'sha256'), 'hex')
WHERE content_hash IS NULL;

-- Now make content_hash NOT NULL (after backfill)
DO $$
BEGIN
    ALTER TABLE beliefs ALTER COLUMN content_hash SET NOT NULL;
    RAISE NOTICE 'Set content_hash to NOT NULL';
EXCEPTION
    WHEN others THEN
        RAISE NOTICE 'Could not set content_hash NOT NULL: %', SQLERRM;
END $$;

-- ============================================================================
-- PHASE 5: AUTO-COMPUTE HASH ON INSERT/UPDATE
-- ============================================================================

-- Function to compute content hash
CREATE OR REPLACE FUNCTION qa_compute_content_hash()
RETURNS TRIGGER AS $$
BEGIN
    -- Only compute if content changed or hash is null
    IF NEW.content_hash IS NULL OR 
       (TG_OP = 'UPDATE' AND NEW.content IS DISTINCT FROM OLD.content) THEN
        NEW.content_hash := encode(digest(NEW.content, 'sha256'), 'hex');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for auto-hash
DROP TRIGGER IF EXISTS trg_qa_beliefs_content_hash ON beliefs;
CREATE TRIGGER trg_qa_beliefs_content_hash
    BEFORE INSERT OR UPDATE OF content ON beliefs
    FOR EACH ROW
    EXECUTE FUNCTION qa_compute_content_hash();

-- ============================================================================
-- PHASE 6: INDEXES
-- ============================================================================

-- Index for holder-based queries (federation common pattern)
CREATE INDEX IF NOT EXISTS idx_qa_beliefs_holder 
    ON beliefs(holder_id) 
    WHERE holder_id IS NOT NULL;

-- Index for active beliefs per holder
CREATE INDEX IF NOT EXISTS idx_qa_beliefs_holder_active 
    ON beliefs(holder_id, created_at DESC) 
    WHERE superseded_by_id IS NULL;

-- Index for content hash lookups (deduplication)
CREATE INDEX IF NOT EXISTS idx_qa_beliefs_content_hash 
    ON beliefs(content_hash);

-- Index for visibility-based federation queries
CREATE INDEX IF NOT EXISTS idx_qa_beliefs_visibility 
    ON beliefs(visibility) 
    WHERE visibility != 'private';

-- Composite index for public beliefs (federation discovery)
CREATE INDEX IF NOT EXISTS idx_qa_beliefs_public 
    ON beliefs(created_at DESC) 
    WHERE visibility = 'public' 
      AND superseded_by_id IS NULL;

-- ============================================================================
-- PHASE 7: SET DEFAULT HOLDER FOR EXISTING ROWS
-- ============================================================================

-- System holder UUID for pre-federation beliefs
-- Format: 00000000-0000-0000-0000-000000000001 (reserved system holder)
UPDATE beliefs 
SET holder_id = '00000000-0000-0000-0000-000000000001'::uuid
WHERE holder_id IS NULL;

-- ============================================================================
-- PHASE 8: VERIFICATION
-- ============================================================================

DO $$
DECLARE
    col_count INTEGER;
    null_hash_count INTEGER;
    null_holder_count INTEGER;
BEGIN
    -- Verify columns exist
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns 
    WHERE table_name = 'beliefs' 
      AND column_name IN ('holder_id', 'version', 'content_hash', 'visibility');
    
    IF col_count != 4 THEN
        RAISE EXCEPTION 'Migration failed: expected 4 QA columns, found %', col_count;
    END IF;
    
    -- Verify no null content hashes
    SELECT COUNT(*) INTO null_hash_count
    FROM beliefs WHERE content_hash IS NULL;
    
    IF null_hash_count > 0 THEN
        RAISE WARNING 'Found % beliefs with NULL content_hash', null_hash_count;
    END IF;
    
    -- Verify no null holders
    SELECT COUNT(*) INTO null_holder_count
    FROM beliefs WHERE holder_id IS NULL;
    
    IF null_holder_count > 0 THEN
        RAISE WARNING 'Found % beliefs with NULL holder_id', null_holder_count;
    END IF;
    
    RAISE NOTICE 'Migration 002 (QA Fixes) complete: 4 columns verified';
END $$;

-- ============================================================================
-- MIGRATION METADATA
-- ============================================================================

-- Record migration (if migrations table exists)
DO $$
BEGIN
    INSERT INTO schema_migrations (version, name, applied_at)
    VALUES ('002', 'qa-fixes', NOW())
    ON CONFLICT (version) DO UPDATE SET applied_at = NOW();
EXCEPTION
    WHEN undefined_table THEN
        RAISE NOTICE 'schema_migrations table not found, skipping metadata';
END $$;
