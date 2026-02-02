# Valence Database Schema Design

## Overview

The unified schema merges the EKB (Epistemic Knowledge Base) and VKB (Valence Knowledge Base) into a single PostgreSQL database with pgvector for similarity search.

## Core Tables

### beliefs
The central table for all knowledge claims.

```sql
CREATE TABLE beliefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT NOT NULL,

    -- Dimensional confidence (JSONB for flexibility)
    confidence JSONB NOT NULL DEFAULT '{"overall": 0.7}',
    -- Dimensions: source_reliability, method_quality, internal_consistency,
    -- temporal_freshness, corroboration, domain_applicability

    -- Domain classification
    domain_path TEXT[] NOT NULL DEFAULT '{}',

    -- Temporal validity
    valid_from TIMESTAMPTZ,
    valid_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Provenance
    source_id UUID REFERENCES sources(id),
    extraction_method TEXT,

    -- Supersession chain
    supersedes_id UUID REFERENCES beliefs(id),
    superseded_by_id UUID REFERENCES beliefs(id),

    -- Status
    status TEXT NOT NULL DEFAULT 'active',
    -- Values: active, superseded, disputed, archived

    -- Full-text search
    content_tsv TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('english', content)
    ) STORED
);
```

### entities
People, organizations, tools, concepts that beliefs are about.

```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    -- Types: person, organization, tool, concept, project, location

    description TEXT,
    aliases TEXT[] DEFAULT '{}',

    -- For entity resolution
    canonical_id UUID REFERENCES entities(id),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_canonical_name UNIQUE (name, type)
        WHERE canonical_id IS NULL
);
```

### belief_entities
Junction table linking beliefs to entities.

```sql
CREATE TABLE belief_entities (
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    role TEXT NOT NULL DEFAULT 'subject',
    -- Roles: subject, object, context, source

    PRIMARY KEY (belief_id, entity_id, role)
);
```

### sources
Provenance tracking for all knowledge.

```sql
CREATE TABLE sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    type TEXT NOT NULL,
    -- Types: document, conversation, inference, observation, api

    title TEXT,
    url TEXT,

    -- For documents
    content_hash TEXT,

    -- For conversations
    session_id UUID REFERENCES sessions(id),

    -- Metadata
    metadata JSONB DEFAULT '{}',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### tensions
Contradictions or conflicts between beliefs.

```sql
CREATE TABLE tensions (
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

    CONSTRAINT different_beliefs CHECK (belief_a_id != belief_b_id)
);
```

## Conversation Tables (VKB)

### sessions
Meso-scale conversation tracking.

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    platform TEXT NOT NULL,
    -- Platforms: claude-code, matrix, api, slack

    project_context TEXT,

    status TEXT NOT NULL DEFAULT 'active',
    -- Status: active, completed, abandoned

    summary TEXT,
    themes TEXT[],

    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,

    -- Claude Code session ID for resume
    claude_session_id TEXT,

    -- Room/channel reference for chat platforms
    external_room_id TEXT,

    metadata JSONB DEFAULT '{}'
);
```

### exchanges
Individual conversation turns.

```sql
CREATE TABLE exchanges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

    sequence INTEGER NOT NULL,
    role TEXT NOT NULL,
    -- Roles: user, assistant, system

    content TEXT NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Approximate token count for context management
    tokens_approx INTEGER,

    -- Link to any tools used
    tool_uses JSONB DEFAULT '[]',

    UNIQUE (session_id, sequence)
);
```

### patterns
Macro-scale behavioral patterns.

```sql
CREATE TABLE patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    type TEXT NOT NULL,
    -- Types: topic_recurrence, preference, working_style,
    -- communication_pattern, value_expression

    description TEXT NOT NULL,

    evidence UUID[] NOT NULL DEFAULT '{}',
    -- Array of session IDs

    occurrence_count INTEGER DEFAULT 1,
    confidence NUMERIC(3,2) DEFAULT 0.5,

    status TEXT NOT NULL DEFAULT 'emerging',
    -- Status: emerging, established, fading, archived

    first_observed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_observed TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);
```

### session_insights
Links sessions to extracted beliefs.

```sql
CREATE TABLE session_insights (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,

    extraction_method TEXT NOT NULL DEFAULT 'manual',
    -- Methods: manual, auto, hybrid

    extracted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (session_id, belief_id)
);
```

## Embedding Support

### Using pgvector for similarity search

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Add embedding columns to key tables
ALTER TABLE beliefs ADD COLUMN embedding VECTOR(1536);
ALTER TABLE exchanges ADD COLUMN embedding VECTOR(1536);
ALTER TABLE patterns ADD COLUMN embedding VECTOR(1536);

-- Create indexes for similarity search
CREATE INDEX ON beliefs USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
CREATE INDEX ON exchanges USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
```

### Embedding registry (for multi-model support)

```sql
CREATE TABLE embedding_types (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE embedding_coverage (
    content_type TEXT NOT NULL,
    content_id UUID NOT NULL,
    embedding_type_id TEXT NOT NULL REFERENCES embedding_types(id),
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (content_type, content_id, embedding_type_id)
);
```

## Indexes

```sql
-- Beliefs
CREATE INDEX idx_beliefs_domain ON beliefs USING GIN (domain_path);
CREATE INDEX idx_beliefs_status ON beliefs (status);
CREATE INDEX idx_beliefs_created ON beliefs (created_at DESC);
CREATE INDEX idx_beliefs_tsv ON beliefs USING GIN (content_tsv);

-- Entities
CREATE INDEX idx_entities_type ON entities (type);
CREATE INDEX idx_entities_name ON entities (name);
CREATE INDEX idx_entities_aliases ON entities USING GIN (aliases);

-- Sessions
CREATE INDEX idx_sessions_platform ON sessions (platform);
CREATE INDEX idx_sessions_status ON sessions (status);
CREATE INDEX idx_sessions_started ON sessions (started_at DESC);
CREATE INDEX idx_sessions_project ON sessions (project_context);
CREATE INDEX idx_sessions_external_room ON sessions (external_room_id);

-- Exchanges
CREATE INDEX idx_exchanges_session ON exchanges (session_id);
CREATE INDEX idx_exchanges_created ON exchanges (created_at DESC);

-- Patterns
CREATE INDEX idx_patterns_type ON patterns (type);
CREATE INDEX idx_patterns_status ON patterns (status);
```

## Views

### beliefs_current
Only active, non-superseded beliefs.

```sql
CREATE VIEW beliefs_current AS
SELECT * FROM beliefs
WHERE status = 'active'
  AND superseded_by_id IS NULL;
```

### beliefs_with_entities
Beliefs joined with their entities.

```sql
CREATE VIEW beliefs_with_entities AS
SELECT
    b.*,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'subject') as subjects,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'object') as objects
FROM beliefs b
LEFT JOIN belief_entities be ON b.id = be.belief_id
LEFT JOIN entities e ON be.entity_id = e.id
GROUP BY b.id;
```

### sessions_overview
Session summary with counts.

```sql
CREATE VIEW sessions_overview AS
SELECT
    s.*,
    COUNT(DISTINCT e.id) as exchange_count,
    COUNT(DISTINCT si.id) as insight_count
FROM sessions s
LEFT JOIN exchanges e ON s.id = e.session_id
LEFT JOIN session_insights si ON s.id = si.session_id
GROUP BY s.id;
```

## Stored Procedures

See `procedures.sql` for:
- `belief_create_with_entities()` - Create belief and link entities atomically
- `belief_supersede()` - Supersede a belief with proper chain maintenance
- `session_summarize()` - Auto-generate session summary from exchanges
- `pattern_reinforce()` - Update pattern confidence and evidence
- `tension_detect()` - Find potential contradictions in recent beliefs
- `embedding_backfill()` - Queue embeddings for content missing them

## Migration Strategy

1. Create new schema alongside existing
2. Migrate entries → beliefs (preserving IDs where possible)
3. Migrate vkb_sessions → sessions
4. Migrate vkb_exchanges → exchanges
5. Migrate vkb_patterns → patterns
6. Create entity records from belief content
7. Backfill embeddings using new pgvector columns
8. Validate data integrity
9. Switch MCP servers to new schema
10. Archive old tables
