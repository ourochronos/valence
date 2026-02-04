-- ============================================================================
-- Valence Belief Schema
-- PGVector-compatible PostgreSQL schema for distributed epistemic storage
-- ============================================================================
-- Designed for: billions of beliefs, sub-second queries, full auditability
-- Requires: PostgreSQL 15+, pgvector extension
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search optimization

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE visibility_level AS ENUM ('private', 'federated', 'public');
CREATE TYPE derivation_type AS ENUM (
    'observation',   -- Direct observation/measurement
    'inference',     -- Logical derivation from other beliefs
    'aggregation',   -- Statistical aggregation
    'hearsay',       -- Reported by another agent
    'assumption',    -- Asserted without direct evidence
    'correction',    -- Supersedes a previous incorrect belief
    'synthesis'      -- AI-generated combination
);
CREATE TYPE contribution_type AS ENUM ('primary', 'supporting', 'contradicting', 'context');

-- ============================================================================
-- CORE TABLES
-- ============================================================================

-- ---------------------------------------------------------------------------
-- beliefs: The primary belief store
-- ---------------------------------------------------------------------------
CREATE TABLE beliefs (
    -- Identity
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    version                 INTEGER NOT NULL DEFAULT 1,
    
    -- Content
    content                 TEXT NOT NULL CHECK (length(content) <= 65536),
    content_hash            CHAR(64) NOT NULL,  -- SHA-256 hex
    
    -- Confidence Vector (6 dimensions, each 0.0-1.0)
    confidence_source       REAL NOT NULL DEFAULT 0.5 CHECK (confidence_source BETWEEN 0.0 AND 1.0),
    confidence_method       REAL NOT NULL DEFAULT 0.5 CHECK (confidence_method BETWEEN 0.0 AND 1.0),
    confidence_consistency  REAL NOT NULL DEFAULT 1.0 CHECK (confidence_consistency BETWEEN 0.0 AND 1.0),
    confidence_freshness    REAL NOT NULL DEFAULT 1.0 CHECK (confidence_freshness BETWEEN 0.0 AND 1.0),
    confidence_corroboration REAL NOT NULL DEFAULT 0.1 CHECK (confidence_corroboration BETWEEN 0.0 AND 1.0),
    confidence_applicability REAL NOT NULL DEFAULT 0.8 CHECK (confidence_applicability BETWEEN 0.0 AND 1.0),
    
    -- Computed overall confidence (materialized for query performance)
    confidence_overall      REAL GENERATED ALWAYS AS (
        -- Weighted geometric mean: source=0.25, method=0.20, consistency=0.15, 
        -- freshness=0.15, corroboration=0.15, applicability=0.10
        POWER(
            POWER(confidence_source, 0.25) *
            POWER(confidence_method, 0.20) *
            POWER(confidence_consistency, 0.15) *
            POWER(confidence_freshness, 0.15) *
            POWER(confidence_corroboration, 0.15) *
            POWER(confidence_applicability, 0.10),
            1.0  -- Normalize (already normalized by weights summing to 1)
        )
    ) STORED,
    
    -- Temporal validity
    valid_from              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until             TIMESTAMPTZ,  -- NULL = indefinite
    
    -- Organization
    domains                 TEXT[] NOT NULL DEFAULT '{}',
    
    -- Privacy
    visibility              visibility_level NOT NULL DEFAULT 'private',
    
    -- Provenance
    holder_id               UUID NOT NULL,  -- References agents table (external)
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Versioning
    supersedes              UUID REFERENCES beliefs(id),
    superseded_by           UUID REFERENCES beliefs(id),
    
    -- Embedding for semantic search (1536 dimensions for text-embedding-3-small)
    embedding               vector(1536),
    
    -- Soft delete
    deleted_at              TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT valid_temporal_range CHECK (valid_until IS NULL OR valid_until > valid_from),
    CONSTRAINT version_positive CHECK (version > 0),
    CONSTRAINT domains_limit CHECK (array_length(domains, 1) IS NULL OR array_length(domains, 1) <= 20)
);

-- ---------------------------------------------------------------------------
-- belief_derivations: How beliefs are derived (one per belief)
-- ---------------------------------------------------------------------------
CREATE TABLE belief_derivations (
    belief_id               UUID PRIMARY KEY REFERENCES beliefs(id) ON DELETE CASCADE,
    derivation_type         derivation_type NOT NULL DEFAULT 'assumption',
    method_description      TEXT CHECK (length(method_description) <= 4096),
    confidence_rationale    TEXT CHECK (length(confidence_rationale) <= 4096),
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- derivation_sources: Sources that contributed to a belief's derivation
-- ---------------------------------------------------------------------------
CREATE TABLE derivation_sources (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id               UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    source_belief_id        UUID REFERENCES beliefs(id),  -- NULL if external source
    external_ref            TEXT CHECK (length(external_ref) <= 2048),
    contribution_type       contribution_type NOT NULL DEFAULT 'primary',
    weight                  REAL NOT NULL DEFAULT 1.0 CHECK (weight BETWEEN 0.0 AND 1.0),
    
    -- At least one source type must be specified
    CONSTRAINT has_source CHECK (source_belief_id IS NOT NULL OR external_ref IS NOT NULL)
);

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Primary lookup indexes
CREATE INDEX idx_beliefs_holder ON beliefs(holder_id);
CREATE INDEX idx_beliefs_holder_created ON beliefs(holder_id, created_at DESC);
CREATE INDEX idx_beliefs_content_hash ON beliefs(content_hash);

-- Domain search (GIN for array contains)
CREATE INDEX idx_beliefs_domains ON beliefs USING GIN(domains);

-- Temporal queries
CREATE INDEX idx_beliefs_valid_range ON beliefs(valid_from, valid_until);
CREATE INDEX idx_beliefs_created_at ON beliefs(created_at DESC);

-- Version chain navigation
CREATE INDEX idx_beliefs_supersedes ON beliefs(supersedes) WHERE supersedes IS NOT NULL;
CREATE INDEX idx_beliefs_superseded_by ON beliefs(superseded_by) WHERE superseded_by IS NOT NULL;

-- Active beliefs (not superseded, not deleted)
CREATE INDEX idx_beliefs_active ON beliefs(holder_id, created_at DESC) 
    WHERE superseded_by IS NULL AND deleted_at IS NULL;

-- Visibility-based queries
CREATE INDEX idx_beliefs_public ON beliefs(created_at DESC) 
    WHERE visibility = 'public' AND superseded_by IS NULL AND deleted_at IS NULL;
CREATE INDEX idx_beliefs_federated ON beliefs(holder_id, created_at DESC)
    WHERE visibility IN ('federated', 'public') AND superseded_by IS NULL AND deleted_at IS NULL;

-- Confidence filtering
CREATE INDEX idx_beliefs_high_confidence ON beliefs(confidence_overall DESC)
    WHERE confidence_overall >= 0.7 AND superseded_by IS NULL AND deleted_at IS NULL;

-- Vector similarity search (HNSW for approximate nearest neighbor)
-- m=16: connections per node, ef_construction=64: build-time quality
CREATE INDEX idx_beliefs_embedding ON beliefs 
    USING hnsw(embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Derivation lookups
CREATE INDEX idx_derivation_sources_belief ON derivation_sources(belief_id);
CREATE INDEX idx_derivation_sources_source ON derivation_sources(source_belief_id) 
    WHERE source_belief_id IS NOT NULL;

-- Full-text search on content (optional, for keyword fallback)
CREATE INDEX idx_beliefs_content_trgm ON beliefs USING GIN(content gin_trgm_ops);

-- ============================================================================
-- FUNCTIONS
-- ============================================================================

-- ---------------------------------------------------------------------------
-- compute_content_hash: Generate SHA-256 hash of content
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION compute_content_hash(content TEXT)
RETURNS CHAR(64) AS $$
BEGIN
    RETURN encode(digest(content, 'sha256'), 'hex');
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ---------------------------------------------------------------------------
-- auto_content_hash: Trigger to auto-compute content hash on insert
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION auto_content_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash := compute_content_hash(NEW.content);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_beliefs_content_hash
    BEFORE INSERT ON beliefs
    FOR EACH ROW
    EXECUTE FUNCTION auto_content_hash();

-- ---------------------------------------------------------------------------
-- link_supersession: Update superseded_by back-link when supersedes is set
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION link_supersession()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.supersedes IS NOT NULL THEN
        UPDATE beliefs 
        SET superseded_by = NEW.id
        WHERE id = NEW.supersedes
          AND superseded_by IS NULL;  -- Only if not already superseded
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_beliefs_link_supersession
    AFTER INSERT ON beliefs
    FOR EACH ROW
    WHEN (NEW.supersedes IS NOT NULL)
    EXECUTE FUNCTION link_supersession();

-- ---------------------------------------------------------------------------
-- semantic_search: Vector similarity search with filters
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding vector(1536),
    p_holder_id UUID DEFAULT NULL,
    p_domains TEXT[] DEFAULT NULL,
    p_visibility visibility_level[] DEFAULT NULL,
    p_min_confidence REAL DEFAULT 0.0,
    p_valid_at TIMESTAMPTZ DEFAULT NOW(),
    p_include_superseded BOOLEAN DEFAULT FALSE,
    p_limit INTEGER DEFAULT 20,
    p_offset INTEGER DEFAULT 0
)
RETURNS TABLE(
    belief_id UUID,
    relevance_score REAL,
    combined_score REAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        b.id,
        (1 - (b.embedding <=> query_embedding))::REAL AS relevance,
        ((1 - (b.embedding <=> query_embedding)) * b.confidence_overall)::REAL AS combined
    FROM beliefs b
    WHERE 
        -- Embedding must exist
        b.embedding IS NOT NULL
        -- Visibility filter
        AND (p_visibility IS NULL OR b.visibility = ANY(p_visibility))
        -- Holder filter
        AND (p_holder_id IS NULL OR b.holder_id = p_holder_id)
        -- Domain filter (any match)
        AND (p_domains IS NULL OR b.domains && p_domains)
        -- Confidence filter
        AND b.confidence_overall >= p_min_confidence
        -- Temporal validity
        AND b.valid_from <= p_valid_at
        AND (b.valid_until IS NULL OR b.valid_until > p_valid_at)
        -- Version filter
        AND (p_include_superseded OR b.superseded_by IS NULL)
        -- Not deleted
        AND b.deleted_at IS NULL
    ORDER BY combined DESC
    LIMIT p_limit
    OFFSET p_offset;
END;
$$ LANGUAGE plpgsql STABLE;

-- ---------------------------------------------------------------------------
-- get_derivation_chain: Recursive derivation tree
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_derivation_chain(
    p_belief_id UUID,
    p_max_depth INTEGER DEFAULT 5
)
RETURNS TABLE(
    belief_id UUID,
    content_preview TEXT,
    derivation_type derivation_type,
    confidence_overall REAL,
    depth INTEGER
) AS $$
WITH RECURSIVE chain AS (
    -- Base case: the target belief
    SELECT 
        b.id,
        LEFT(b.content, 200) AS content_preview,
        COALESCE(d.derivation_type, 'assumption'::derivation_type) AS derivation_type,
        b.confidence_overall,
        0 AS depth
    FROM beliefs b
    LEFT JOIN belief_derivations d ON d.belief_id = b.id
    WHERE b.id = p_belief_id
    
    UNION ALL
    
    -- Recursive case: source beliefs
    SELECT 
        b.id,
        LEFT(b.content, 200),
        COALESCE(d.derivation_type, 'assumption'::derivation_type),
        b.confidence_overall,
        c.depth + 1
    FROM chain c
    JOIN derivation_sources ds ON ds.belief_id = c.belief_id
    JOIN beliefs b ON b.id = ds.source_belief_id
    LEFT JOIN belief_derivations d ON d.belief_id = b.id
    WHERE 
        c.depth < p_max_depth
        AND ds.source_belief_id IS NOT NULL
)
SELECT * FROM chain ORDER BY depth, belief_id;
$$ LANGUAGE SQL STABLE;

-- ---------------------------------------------------------------------------
-- get_version_chain: Get all versions of a belief
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION get_version_chain(
    p_belief_id UUID,
    p_direction TEXT DEFAULT 'both'  -- 'ancestors', 'descendants', 'both'
)
RETURNS TABLE(
    belief_id UUID,
    version INTEGER,
    created_at TIMESTAMPTZ,
    direction TEXT
) AS $$
WITH RECURSIVE 
ancestors AS (
    SELECT id, version, created_at, 'self'::TEXT AS dir
    FROM beliefs WHERE id = p_belief_id
    
    UNION ALL
    
    SELECT b.id, b.version, b.created_at, 'ancestor'
    FROM ancestors a
    JOIN beliefs b ON b.id = a.id
    JOIN beliefs prev ON prev.superseded_by = b.id
    WHERE p_direction IN ('ancestors', 'both')
),
descendants AS (
    SELECT id, version, created_at, 'self'::TEXT AS dir
    FROM beliefs WHERE id = p_belief_id
    
    UNION ALL
    
    SELECT b.id, b.version, b.created_at, 'descendant'
    FROM descendants d
    JOIN beliefs b ON b.supersedes = d.id
    WHERE p_direction IN ('descendants', 'both')
)
SELECT DISTINCT belief_id, version, created_at, direction
FROM (
    SELECT id AS belief_id, version, created_at, dir AS direction FROM ancestors
    UNION
    SELECT id, version, created_at, dir FROM descendants
) combined
ORDER BY version;
$$ LANGUAGE SQL STABLE;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- ---------------------------------------------------------------------------
-- v_active_beliefs: Only current, non-deleted beliefs
-- ---------------------------------------------------------------------------
CREATE VIEW v_active_beliefs AS
SELECT *
FROM beliefs
WHERE superseded_by IS NULL
  AND deleted_at IS NULL;

-- ---------------------------------------------------------------------------
-- v_beliefs_with_derivation: Beliefs with derivation info joined
-- ---------------------------------------------------------------------------
CREATE VIEW v_beliefs_with_derivation AS
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
    ) AS derivation_sources
FROM beliefs b
LEFT JOIN belief_derivations d ON d.belief_id = b.id;

-- ---------------------------------------------------------------------------
-- v_beliefs_full: Complete belief view with computed fields
-- ---------------------------------------------------------------------------
CREATE VIEW v_beliefs_full AS
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
    ) AS confidence,
    b.confidence_overall,
    b.valid_from,
    b.valid_until,
    b.domains,
    b.visibility,
    b.holder_id,
    b.created_at,
    b.supersedes,
    b.superseded_by,
    json_build_object(
        'type', COALESCE(d.derivation_type, 'assumption'),
        'method_description', d.method_description,
        'confidence_rationale', d.confidence_rationale,
        'sources', COALESCE(
            (SELECT json_agg(json_build_object(
                'belief_id', ds.source_belief_id,
                'external_ref', ds.external_ref,
                'contribution_type', ds.contribution_type,
                'weight', ds.weight
            ))
            FROM derivation_sources ds
            WHERE ds.belief_id = b.id),
            '[]'::json
        )
    ) AS derivation
FROM beliefs b
LEFT JOIN belief_derivations d ON d.belief_id = b.id;

-- ============================================================================
-- PARTITIONING SUPPORT (for billion-scale deployments)
-- ============================================================================

-- Example: Partition by holder_id hash for distributed deployments
-- Uncomment and modify based on deployment needs

-- CREATE TABLE beliefs_partitioned (
--     LIKE beliefs INCLUDING ALL
-- ) PARTITION BY HASH(holder_id);
-- 
-- CREATE TABLE beliefs_p0 PARTITION OF beliefs_partitioned
--     FOR VALUES WITH (MODULUS 16, REMAINDER 0);
-- CREATE TABLE beliefs_p1 PARTITION OF beliefs_partitioned
--     FOR VALUES WITH (MODULUS 16, REMAINDER 1);
-- ... etc for 16 partitions

-- ============================================================================
-- MAINTENANCE
-- ============================================================================

-- ---------------------------------------------------------------------------
-- Vacuum and analyze schedule (run via pg_cron or external scheduler)
-- ---------------------------------------------------------------------------
-- VACUUM ANALYZE beliefs;
-- VACUUM ANALYZE derivation_sources;

-- ---------------------------------------------------------------------------
-- Reindex embedding index (if performance degrades)
-- ---------------------------------------------------------------------------
-- REINDEX INDEX CONCURRENTLY idx_beliefs_embedding;

-- ============================================================================
-- MIGRATION HELPERS
-- ============================================================================

-- ---------------------------------------------------------------------------
-- migrate_add_embedding: Batch update embeddings for beliefs without them
-- ---------------------------------------------------------------------------
-- This is a placeholder - actual embedding computation happens in application
CREATE OR REPLACE FUNCTION count_missing_embeddings()
RETURNS BIGINT AS $$
    SELECT COUNT(*) FROM beliefs WHERE embedding IS NULL AND deleted_at IS NULL;
$$ LANGUAGE SQL STABLE;

-- ---------------------------------------------------------------------------
-- cleanup_old_tombstones: Remove very old soft-deleted beliefs
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION cleanup_old_tombstones(
    older_than INTERVAL DEFAULT INTERVAL '1 year'
)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    -- First, remove derivation sources pointing to beliefs we'll delete
    DELETE FROM derivation_sources
    WHERE source_belief_id IN (
        SELECT id FROM beliefs 
        WHERE deleted_at IS NOT NULL 
          AND deleted_at < NOW() - older_than
    );
    
    -- Then delete the tombstoned beliefs
    WITH deleted AS (
        DELETE FROM beliefs
        WHERE deleted_at IS NOT NULL
          AND deleted_at < NOW() - older_than
        RETURNING id
    )
    SELECT COUNT(*) INTO deleted_count FROM deleted;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERMISSIONS (example, adjust for your deployment)
-- ============================================================================

-- Application role
-- CREATE ROLE valence_app WITH LOGIN PASSWORD 'xxx';
-- GRANT SELECT, INSERT, UPDATE ON beliefs TO valence_app;
-- GRANT SELECT, INSERT, UPDATE ON belief_derivations TO valence_app;
-- GRANT SELECT, INSERT, UPDATE ON derivation_sources TO valence_app;
-- GRANT SELECT ON v_active_beliefs TO valence_app;
-- GRANT SELECT ON v_beliefs_full TO valence_app;
-- GRANT EXECUTE ON FUNCTION semantic_search TO valence_app;

-- Read-only role for analytics
-- CREATE ROLE valence_readonly WITH LOGIN PASSWORD 'xxx';
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO valence_readonly;

-- ============================================================================
-- STATISTICS
-- ============================================================================

-- ---------------------------------------------------------------------------
-- stats_beliefs: Quick statistics about the belief store
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION stats_beliefs()
RETURNS TABLE(
    total_beliefs BIGINT,
    active_beliefs BIGINT,
    beliefs_with_embeddings BIGINT,
    tombstoned_beliefs BIGINT,
    unique_holders BIGINT,
    unique_domains BIGINT,
    avg_confidence REAL
) AS $$
SELECT
    COUNT(*),
    COUNT(*) FILTER (WHERE superseded_by IS NULL AND deleted_at IS NULL),
    COUNT(*) FILTER (WHERE embedding IS NOT NULL),
    COUNT(*) FILTER (WHERE deleted_at IS NOT NULL),
    COUNT(DISTINCT holder_id),
    (SELECT COUNT(DISTINCT unnest) FROM (SELECT unnest(domains) FROM beliefs) d),
    AVG(confidence_overall)::REAL
FROM beliefs;
$$ LANGUAGE SQL STABLE;

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Uncomment to insert test data
/*
INSERT INTO beliefs (content, holder_id, visibility, domains, confidence_source, confidence_method)
VALUES 
    ('PostgreSQL is a powerful open-source relational database.', 
     'a0000000-0000-0000-0000-000000000001', 'public', 
     ARRAY['tech/databases', 'open-source'], 0.95, 0.9),
    ('The pgvector extension enables vector similarity search in PostgreSQL.',
     'a0000000-0000-0000-0000-000000000001', 'public',
     ARRAY['tech/databases', 'tech/ai'], 0.9, 0.85);
*/

-- ============================================================================
-- END OF SCHEMA
-- ============================================================================
