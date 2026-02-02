-- Valence Unified Schema
-- PostgreSQL with pgvector for similarity search
-- Merges EKB (Epistemic Knowledge Base) and VKB (Conversation Tracking)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- ============================================================================
-- KNOWLEDGE SUBSTRATE TABLES
-- ============================================================================

-- Sources: Provenance tracking for all knowledge
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    type TEXT NOT NULL,
    -- Types: document, conversation, inference, observation, api, user_input

    title TEXT,
    url TEXT,

    -- For documents: content hash for deduplication
    content_hash TEXT,

    -- For conversations: link to session
    session_id UUID,  -- FK added after sessions table

    -- Flexible metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(type);
CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash) WHERE content_hash IS NOT NULL;

-- Beliefs: Core knowledge claims
CREATE TABLE IF NOT EXISTS beliefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    content TEXT NOT NULL,

    -- Dimensional confidence (JSONB for flexibility)
    -- Dimensions: overall, source_reliability, method_quality,
    -- internal_consistency, temporal_freshness, corroboration, domain_applicability
    confidence JSONB NOT NULL DEFAULT '{"overall": 0.7}',

    -- Domain classification (hierarchical path)
    domain_path TEXT[] NOT NULL DEFAULT '{}',

    -- Temporal validity
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Provenance
    source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
    extraction_method TEXT,

    -- Supersession chain
    supersedes_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,
    superseded_by_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,

    -- Status
    status TEXT NOT NULL DEFAULT 'active',
    -- Values: active, superseded, disputed, archived

    -- Vector embedding for similarity search
    embedding VECTOR(1536),

    -- Full-text search
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED,

    CONSTRAINT beliefs_valid_status CHECK (status IN ('active', 'superseded', 'disputed', 'archived')),
    CONSTRAINT beliefs_valid_confidence CHECK (
        (confidence->>'overall')::numeric >= 0 AND
        (confidence->>'overall')::numeric <= 1
    )
);

CREATE INDEX IF NOT EXISTS idx_beliefs_domain ON beliefs USING GIN (domain_path);
CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status);
CREATE INDEX IF NOT EXISTS idx_beliefs_created ON beliefs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_beliefs_tsv ON beliefs USING GIN (content_tsv);
CREATE INDEX IF NOT EXISTS idx_beliefs_source ON beliefs(source_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_embedding ON beliefs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Entities: People, organizations, tools, concepts
CREATE TABLE IF NOT EXISTS entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    name TEXT NOT NULL,
    type TEXT NOT NULL,
    -- Types: person, organization, tool, concept, project, location, service

    description TEXT,
    aliases TEXT[] DEFAULT '{}',

    -- For entity resolution (merging duplicates)
    canonical_id UUID REFERENCES entities(id) ON DELETE SET NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT entities_valid_type CHECK (type IN (
        'person', 'organization', 'tool', 'concept', 'project', 'location', 'service'
    ))
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_aliases ON entities USING GIN (aliases);
CREATE INDEX IF NOT EXISTS idx_entities_canonical ON entities(canonical_id) WHERE canonical_id IS NOT NULL;

-- Unique constraint on canonical entities only
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_unique_canonical
    ON entities(name, type)
    WHERE canonical_id IS NULL;

-- Belief-Entity junction
CREATE TABLE IF NOT EXISTS belief_entities (
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,

    role TEXT NOT NULL DEFAULT 'subject',
    -- Roles: subject, object, context, source

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    PRIMARY KEY (belief_id, entity_id, role),
    CONSTRAINT belief_entities_valid_role CHECK (role IN ('subject', 'object', 'context', 'source'))
);

CREATE INDEX IF NOT EXISTS idx_belief_entities_entity ON belief_entities(entity_id);

-- Tensions: Contradictions between beliefs
CREATE TABLE IF NOT EXISTS tensions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    belief_a_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    belief_b_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,

    type TEXT NOT NULL DEFAULT 'contradiction',
    -- Types: contradiction, temporal_conflict, scope_conflict, partial_overlap

    description TEXT,

    severity TEXT NOT NULL DEFAULT 'medium',
    -- Severity: low, medium, high, critical

    status TEXT NOT NULL DEFAULT 'detected',
    -- Status: detected, investigating, resolved, accepted

    resolution TEXT,
    resolved_at TIMESTAMPTZ,

    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT tensions_different_beliefs CHECK (belief_a_id != belief_b_id),
    CONSTRAINT tensions_valid_type CHECK (type IN (
        'contradiction', 'temporal_conflict', 'scope_conflict', 'partial_overlap'
    )),
    CONSTRAINT tensions_valid_severity CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    CONSTRAINT tensions_valid_status CHECK (status IN ('detected', 'investigating', 'resolved', 'accepted'))
);

CREATE INDEX IF NOT EXISTS idx_tensions_status ON tensions(status);
CREATE INDEX IF NOT EXISTS idx_tensions_severity ON tensions(severity);
CREATE INDEX IF NOT EXISTS idx_tensions_belief_a ON tensions(belief_a_id);
CREATE INDEX IF NOT EXISTS idx_tensions_belief_b ON tensions(belief_b_id);

-- ============================================================================
-- CONVERSATION TRACKING TABLES (VKB)
-- Note: Uses vkb_ prefix to avoid conflicts with Synapse tables
-- ============================================================================

-- Sessions: Meso-scale conversation tracking
CREATE TABLE IF NOT EXISTS vkb_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    platform TEXT NOT NULL,
    -- Platforms: claude-code, matrix, api, slack

    project_context TEXT,

    status TEXT NOT NULL DEFAULT 'active',
    -- Status: active, completed, abandoned

    summary TEXT,
    themes TEXT[] DEFAULT '{}',

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,

    -- Claude Code session ID for resume
    claude_session_id TEXT,

    -- Room/channel reference for chat platforms
    external_room_id TEXT,

    -- Flexible metadata
    metadata JSONB DEFAULT '{}',

    CONSTRAINT vkb_sessions_valid_status CHECK (status IN ('active', 'completed', 'abandoned')),
    CONSTRAINT vkb_sessions_valid_platform CHECK (platform IN ('claude-code', 'matrix', 'api', 'slack'))
);

CREATE INDEX IF NOT EXISTS idx_vkb_sessions_platform ON vkb_sessions(platform);
CREATE INDEX IF NOT EXISTS idx_vkb_sessions_status ON vkb_sessions(status);
CREATE INDEX IF NOT EXISTS idx_vkb_sessions_started ON vkb_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_vkb_sessions_project ON vkb_sessions(project_context);
CREATE INDEX IF NOT EXISTS idx_vkb_sessions_external_room ON vkb_sessions(external_room_id);
CREATE INDEX IF NOT EXISTS idx_vkb_sessions_claude_id ON vkb_sessions(claude_session_id) WHERE claude_session_id IS NOT NULL;

-- Add FK from sources to vkb_sessions (now that vkb_sessions exists)
ALTER TABLE sources
    ADD CONSTRAINT fk_sources_vkb_session
    FOREIGN KEY (session_id) REFERENCES vkb_sessions(id) ON DELETE SET NULL;

-- Exchanges: Individual conversation turns
CREATE TABLE IF NOT EXISTS vkb_exchanges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES vkb_sessions(id) ON DELETE CASCADE,

    sequence INTEGER NOT NULL,

    role TEXT NOT NULL,
    -- Roles: user, assistant, system

    content TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Approximate token count for context management
    tokens_approx INTEGER,

    -- Tools used in this turn
    tool_uses JSONB DEFAULT '[]',

    -- Vector embedding
    embedding VECTOR(1536),

    UNIQUE (session_id, sequence),
    CONSTRAINT vkb_exchanges_valid_role CHECK (role IN ('user', 'assistant', 'system'))
);

CREATE INDEX IF NOT EXISTS idx_vkb_exchanges_session ON vkb_exchanges(session_id);
CREATE INDEX IF NOT EXISTS idx_vkb_exchanges_created ON vkb_exchanges(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vkb_exchanges_embedding ON vkb_exchanges USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Patterns: Macro-scale behavioral patterns
CREATE TABLE IF NOT EXISTS vkb_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    type TEXT NOT NULL,
    -- Types: topic_recurrence, preference, working_style, communication_pattern, value_expression

    description TEXT NOT NULL,

    evidence UUID[] NOT NULL DEFAULT '{}',
    -- Array of session IDs that support this pattern

    occurrence_count INTEGER DEFAULT 1,
    confidence NUMERIC(3,2) DEFAULT 0.5,

    status TEXT NOT NULL DEFAULT 'emerging',
    -- Status: emerging, established, fading, archived

    first_observed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_observed TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Vector embedding
    embedding VECTOR(1536),

    CONSTRAINT vkb_patterns_valid_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT vkb_patterns_valid_status CHECK (status IN ('emerging', 'established', 'fading', 'archived'))
);

CREATE INDEX IF NOT EXISTS idx_vkb_patterns_type ON vkb_patterns(type);
CREATE INDEX IF NOT EXISTS idx_vkb_patterns_status ON vkb_patterns(status);
CREATE INDEX IF NOT EXISTS idx_vkb_patterns_embedding ON vkb_patterns USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Session Insights: Links sessions to extracted beliefs
CREATE TABLE IF NOT EXISTS vkb_session_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES vkb_sessions(id) ON DELETE CASCADE,
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,

    extraction_method TEXT NOT NULL DEFAULT 'manual',
    -- Methods: manual, auto, hybrid

    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (session_id, belief_id),
    CONSTRAINT vkb_session_insights_valid_method CHECK (extraction_method IN ('manual', 'auto', 'hybrid'))
);

CREATE INDEX IF NOT EXISTS idx_vkb_session_insights_session ON vkb_session_insights(session_id);
CREATE INDEX IF NOT EXISTS idx_vkb_session_insights_belief ON vkb_session_insights(belief_id);

-- ============================================================================
-- EMBEDDING SUPPORT TABLES
-- ============================================================================

-- Embedding type registry (for multi-model support)
CREATE TABLE IF NOT EXISTS embedding_types (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT embedding_types_valid_status CHECK (status IN ('active', 'deprecated', 'backfilling'))
);

-- Only one default embedding type
CREATE UNIQUE INDEX IF NOT EXISTS idx_embedding_types_default
    ON embedding_types(is_default) WHERE is_default = TRUE;

-- Track embedding coverage across tables
CREATE TABLE IF NOT EXISTS embedding_coverage (
    content_type TEXT NOT NULL,
    content_id UUID NOT NULL,
    embedding_type_id TEXT NOT NULL REFERENCES embedding_types(id),
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_type, content_id, embedding_type_id)
);

CREATE INDEX IF NOT EXISTS idx_embedding_coverage_type ON embedding_coverage(embedding_type_id);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Current active beliefs (not superseded)
CREATE OR REPLACE VIEW beliefs_current AS
SELECT * FROM beliefs
WHERE status = 'active'
  AND superseded_by_id IS NULL;

-- Beliefs with entity names
CREATE OR REPLACE VIEW beliefs_with_entities AS
SELECT
    b.*,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'subject') as subjects,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'object') as objects,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'context') as contexts
FROM beliefs b
LEFT JOIN belief_entities be ON b.id = be.belief_id
LEFT JOIN entities e ON be.entity_id = e.id
GROUP BY b.id;

-- Sessions with exchange and insight counts
CREATE OR REPLACE VIEW vkb_sessions_overview AS
SELECT
    s.*,
    COUNT(DISTINCT e.id) as exchange_count,
    COUNT(DISTINCT si.id) as insight_count
FROM vkb_sessions s
LEFT JOIN vkb_exchanges e ON s.id = e.session_id
LEFT JOIN vkb_session_insights si ON s.id = si.session_id
GROUP BY s.id;

-- Patterns with readable status
CREATE OR REPLACE VIEW vkb_patterns_overview AS
SELECT
    p.*,
    array_length(p.evidence, 1) as evidence_count
FROM vkb_patterns p;

-- ============================================================================
-- DEFAULT DATA
-- ============================================================================

-- Insert default embedding type
INSERT INTO embedding_types (id, provider, model, dimensions, is_default, status)
VALUES ('openai_text3_small', 'openai', 'text-embedding-3-small', 1536, TRUE, 'active')
ON CONFLICT (id) DO NOTHING;
