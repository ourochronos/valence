-- ============================================================================
-- Migration 001: Upgrade beliefs table to full Valence schema
-- ============================================================================
-- Current: 642 beliefs, 373 with embeddings, 269 with source links
-- Target: Full Valence schema with 6D confidence, derivations, visibility
-- 
-- STRATEGY: Additive migration - add new columns, preserve old ones during
-- transition, then drop deprecated columns in a later migration.
-- ============================================================================

-- ============================================================================
-- PHASE 1: CREATE NEW ENUMS
-- ============================================================================

-- Check if enum exists before creating (for idempotency)
DO $$ BEGIN
    CREATE TYPE visibility_level AS ENUM ('private', 'federated', 'public');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE derivation_type AS ENUM (
        'observation',   -- Direct observation/measurement
        'inference',     -- Logical derivation from other beliefs
        'aggregation',   -- Statistical aggregation
        'hearsay',       -- Reported by another agent
        'assumption',    -- Asserted without direct evidence
        'correction',    -- Supersedes a previous incorrect belief
        'synthesis'      -- AI-generated combination
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE contribution_type AS ENUM ('primary', 'supporting', 'contradicting', 'context');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ============================================================================
-- PHASE 2: ADD NEW COLUMNS TO BELIEFS (with defaults for existing rows)
-- ============================================================================

-- Enable pg_trgm if not present
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Add version column
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1;

-- Add content_hash column (will compute for existing rows)
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS content_hash CHAR(64);

-- Add 6-dimensional confidence columns
-- Migrate from existing JSONB confidence.overall to new columns
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS confidence_source REAL DEFAULT 0.5;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS confidence_method REAL DEFAULT 0.5;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS confidence_consistency REAL DEFAULT 1.0;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS confidence_freshness REAL DEFAULT 1.0;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS confidence_corroboration REAL DEFAULT 0.1;
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS confidence_applicability REAL DEFAULT 0.8;

-- Add visibility column
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS visibility visibility_level DEFAULT 'private';

-- Add holder_id (required in target; default to a system UUID for existing)
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS holder_id UUID DEFAULT '00000000-0000-0000-0000-000000000001';

-- Add soft delete column
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

-- Rename domain_path to domains (if domain_path exists and domains doesn't)
DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'domain_path'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'domains'
    ) THEN
        ALTER TABLE beliefs RENAME COLUMN domain_path TO domains;
    END IF;
END $$;

-- Add domains column if neither exists
ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS domains TEXT[] NOT NULL DEFAULT '{}';

-- Rename supersedes_id to supersedes (if needed)
DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'supersedes_id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'supersedes'
    ) THEN
        ALTER TABLE beliefs RENAME COLUMN supersedes_id TO supersedes;
    END IF;
END $$;

-- Rename superseded_by_id to superseded_by (if needed)
DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'superseded_by_id'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'superseded_by'
    ) THEN
        ALTER TABLE beliefs RENAME COLUMN superseded_by_id TO superseded_by;
    END IF;
END $$;

-- ============================================================================
-- PHASE 3: MIGRATE EXISTING DATA
-- ============================================================================

-- Compute content_hash for existing rows (SHA-256)
UPDATE beliefs 
SET content_hash = encode(digest(content, 'sha256'), 'hex')
WHERE content_hash IS NULL;

-- Make content_hash NOT NULL after migration
ALTER TABLE beliefs ALTER COLUMN content_hash SET NOT NULL;

-- Migrate JSONB confidence to new columns
-- Use overall value for source & method, keep others at defaults
UPDATE beliefs
SET 
    confidence_source = COALESCE((confidence->>'overall')::REAL, 0.7),
    confidence_method = COALESCE((confidence->>'method')::REAL, 
                                 (confidence->>'overall')::REAL, 0.5)
WHERE confidence IS NOT NULL
  AND confidence_source = 0.5;  -- Only update if still at default

-- Map old 'status' to visibility where sensible
-- active -> private, archived -> private, disputed -> private
-- (all private since this is single-holder system)
-- No action needed, default is 'private'

-- ============================================================================
-- PHASE 4: ADD CONSTRAINTS
-- ============================================================================

-- Add check constraints (only if they don't exist)
DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT beliefs_confidence_source_check 
        CHECK (confidence_source BETWEEN 0.0 AND 1.0);
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT beliefs_confidence_method_check 
        CHECK (confidence_method BETWEEN 0.0 AND 1.0);
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT beliefs_confidence_consistency_check 
        CHECK (confidence_consistency BETWEEN 0.0 AND 1.0);
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT beliefs_confidence_freshness_check 
        CHECK (confidence_freshness BETWEEN 0.0 AND 1.0);
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT beliefs_confidence_corroboration_check 
        CHECK (confidence_corroboration BETWEEN 0.0 AND 1.0);
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT beliefs_confidence_applicability_check 
        CHECK (confidence_applicability BETWEEN 0.0 AND 1.0);
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT version_positive CHECK (version > 0);
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    ALTER TABLE beliefs ADD CONSTRAINT domains_limit 
        CHECK (array_length(domains, 1) IS NULL OR array_length(domains, 1) <= 20);
EXCEPTION WHEN duplicate_object THEN null; END $$;

-- ============================================================================
-- PHASE 5: CREATE BELIEF_DERIVATIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS belief_derivations (
    belief_id               UUID PRIMARY KEY REFERENCES beliefs(id) ON DELETE CASCADE,
    derivation_type         derivation_type NOT NULL DEFAULT 'assumption',
    method_description      TEXT CHECK (length(method_description) <= 4096),
    confidence_rationale    TEXT CHECK (length(confidence_rationale) <= 4096),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Migrate existing extraction_method to derivations
INSERT INTO belief_derivations (belief_id, derivation_type, method_description)
SELECT 
    id,
    CASE extraction_method
        WHEN 'explicit_statement' THEN 'observation'::derivation_type
        WHEN 'summarization' THEN 'synthesis'::derivation_type
        WHEN 'inference' THEN 'inference'::derivation_type
        WHEN 'llm_extraction' THEN 'synthesis'::derivation_type
        ELSE 'assumption'::derivation_type
    END,
    extraction_method
FROM beliefs
WHERE extraction_method IS NOT NULL
  AND NOT EXISTS (SELECT 1 FROM belief_derivations d WHERE d.belief_id = beliefs.id)
ON CONFLICT (belief_id) DO NOTHING;

-- ============================================================================
-- PHASE 6: CREATE DERIVATION_SOURCES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS derivation_sources (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id               UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    source_belief_id        UUID REFERENCES beliefs(id),
    external_ref            TEXT CHECK (length(external_ref) <= 2048),
    contribution_type       contribution_type NOT NULL DEFAULT 'primary',
    weight                  REAL NOT NULL DEFAULT 1.0 CHECK (weight BETWEEN 0.0 AND 1.0),
    
    CONSTRAINT has_source CHECK (source_belief_id IS NOT NULL OR external_ref IS NOT NULL)
);

-- Migrate existing source_id references to derivation_sources
INSERT INTO derivation_sources (belief_id, external_ref, contribution_type)
SELECT 
    b.id,
    'source:' || s.type || ':' || COALESCE(s.title, s.url, s.id::text),
    'primary'::contribution_type
FROM beliefs b
JOIN sources s ON b.source_id = s.id
WHERE NOT EXISTS (
    SELECT 1 FROM derivation_sources ds WHERE ds.belief_id = b.id
)
ON CONFLICT DO NOTHING;

-- Create indexes for derivation tables
CREATE INDEX IF NOT EXISTS idx_derivation_sources_belief ON derivation_sources(belief_id);
CREATE INDEX IF NOT EXISTS idx_derivation_sources_source ON derivation_sources(source_belief_id) 
    WHERE source_belief_id IS NOT NULL;

-- ============================================================================
-- PHASE 7: CREATE NEW INDEXES
-- ============================================================================

-- Holder lookup
CREATE INDEX IF NOT EXISTS idx_beliefs_holder ON beliefs(holder_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_holder_created ON beliefs(holder_id, created_at DESC);

-- Content hash lookup
CREATE INDEX IF NOT EXISTS idx_beliefs_content_hash ON beliefs(content_hash);

-- Active beliefs filter
CREATE INDEX IF NOT EXISTS idx_beliefs_active ON beliefs(holder_id, created_at DESC) 
    WHERE superseded_by IS NULL AND deleted_at IS NULL;

-- Visibility-based queries
CREATE INDEX IF NOT EXISTS idx_beliefs_public ON beliefs(created_at DESC) 
    WHERE visibility = 'public' AND superseded_by IS NULL AND deleted_at IS NULL;

-- Trigram index for fuzzy text search
CREATE INDEX IF NOT EXISTS idx_beliefs_content_trgm ON beliefs USING GIN(content gin_trgm_ops);

-- ============================================================================
-- PHASE 8: CREATE FUNCTIONS
-- ============================================================================

-- Function to compute content hash
CREATE OR REPLACE FUNCTION compute_content_hash(content TEXT)
RETURNS CHAR(64) AS $$
BEGIN
    RETURN encode(digest(content, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Trigger to auto-compute content hash on insert
CREATE OR REPLACE FUNCTION auto_content_hash()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.content_hash IS NULL THEN
        NEW.content_hash := compute_content_hash(NEW.content);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_beliefs_content_hash ON beliefs;
CREATE TRIGGER trg_beliefs_content_hash
    BEFORE INSERT ON beliefs
    FOR EACH ROW
    EXECUTE FUNCTION auto_content_hash();

-- Supersession linking trigger
CREATE OR REPLACE FUNCTION link_supersession()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.supersedes IS NOT NULL THEN
        UPDATE beliefs 
        SET superseded_by = NEW.id
        WHERE id = NEW.supersedes
          AND superseded_by IS NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_beliefs_link_supersession ON beliefs;
CREATE TRIGGER trg_beliefs_link_supersession
    AFTER INSERT ON beliefs
    FOR EACH ROW
    WHEN (NEW.supersedes IS NOT NULL)
    EXECUTE FUNCTION link_supersession();

-- Computed confidence overall function (since we can't add generated column easily)
CREATE OR REPLACE FUNCTION compute_confidence_overall(
    src REAL, meth REAL, cons REAL, fresh REAL, corr REAL, app REAL
) RETURNS REAL AS $$
BEGIN
    RETURN POWER(
        POWER(src, 0.25) *
        POWER(meth, 0.20) *
        POWER(cons, 0.15) *
        POWER(fresh, 0.15) *
        POWER(corr, 0.15) *
        POWER(app, 0.10),
        1.0
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ============================================================================
-- PHASE 9: CREATE VIEWS
-- ============================================================================

-- Active beliefs view
CREATE OR REPLACE VIEW v_active_beliefs AS
SELECT *
FROM beliefs
WHERE superseded_by IS NULL
  AND deleted_at IS NULL;

-- Beliefs with derivation info
CREATE OR REPLACE VIEW v_beliefs_with_derivation AS
SELECT 
    b.*,
    d.derivation_type,
    d.method_description,
    d.confidence_rationale,
    (
        SELECT json_agg(json_build_object(
            'source_belief_id', ds.source_belief_id,
            'external_ref', ds.external_ref,
            'contribution_type', ds.contribution_type,
            'weight', ds.weight
        ))
        FROM derivation_sources ds
        WHERE ds.belief_id = b.id
    ) AS derivation_sources_json
FROM beliefs b
LEFT JOIN belief_derivations d ON d.belief_id = b.id;

-- Full belief view with computed confidence
CREATE OR REPLACE VIEW v_beliefs_full AS
SELECT 
    b.id,
    b.version,
    b.content,
    b.content_hash,
    json_build_object(
        'source_reliability', b.confidence_source,
        'method_quality', b.confidence_method,
        'internal_consistency', b.confidence_consistency,
        'temporal_freshness', b.confidence_freshness,
        'corroboration', b.confidence_corroboration,
        'domain_applicability', b.confidence_applicability
    ) AS confidence_vector,
    compute_confidence_overall(
        b.confidence_source, b.confidence_method, b.confidence_consistency,
        b.confidence_freshness, b.confidence_corroboration, b.confidence_applicability
    ) AS confidence_overall,
    b.valid_from,
    b.valid_until,
    b.domains,
    b.visibility,
    b.holder_id,
    b.created_at,
    b.supersedes,
    b.superseded_by,
    b.deleted_at,
    d.derivation_type,
    d.method_description
FROM beliefs b
LEFT JOIN belief_derivations d ON d.belief_id = b.id;

-- ============================================================================
-- PHASE 10: UPDATE STATISTICS FUNCTION
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
-- NOTES FOR FOLLOW-UP MIGRATION (002-cleanup.sql)
-- ============================================================================
-- After vmem CLI is updated to use new schema:
-- 1. DROP old JSONB confidence column
-- 2. DROP old source_id column  
-- 3. DROP old extraction_method column
-- 4. DROP old modified_at column (or keep for compatibility)
-- 5. DROP old status column (replaced by deleted_at + superseded_by)
-- 6. Make holder_id NOT NULL (after setting proper values)
-- 7. Add confidence_overall as GENERATED ALWAYS column
-- ============================================================================

-- Verify migration
DO $$
DECLARE
    new_cols INTEGER;
    derivation_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO new_cols 
    FROM information_schema.columns 
    WHERE table_name = 'beliefs' 
      AND column_name IN ('version', 'content_hash', 'visibility', 'holder_id', 
                          'confidence_source', 'confidence_method');
    
    SELECT COUNT(*) INTO derivation_count FROM belief_derivations;
    
    RAISE NOTICE 'Migration 001 complete: % new columns added, % derivations created', 
                 new_cols, derivation_count;
END $$;
