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

    -- Ownership (distinct from source provenance)
    holder_id UUID,

    -- Belief versioning
    version INTEGER NOT NULL DEFAULT 1,

    -- Content integrity / deduplication
    content_hash CHAR(64),

    -- Federation privacy level
    visibility TEXT NOT NULL DEFAULT 'private',
    -- Values: private, federated, public

    -- Federation privacy
    opt_out_federation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Share policy
    share_policy JSONB,

    -- Extraction metadata
    extraction_metadata JSONB,

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
    CONSTRAINT beliefs_valid_visibility CHECK (visibility IN ('private', 'federated', 'public')),
    CONSTRAINT beliefs_version_positive CHECK (version > 0),
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
CREATE INDEX IF NOT EXISTS idx_beliefs_holder ON beliefs(holder_id) WHERE holder_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_beliefs_content_hash ON beliefs(content_hash) WHERE content_hash IS NOT NULL;

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
    -- Federation privacy
    opt_out_federation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Share policy
    share_policy JSONB,

    -- Extraction metadata
    extraction_metadata JSONB,

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
    -- Federation privacy
    opt_out_federation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Share policy
    share_policy JSONB,

    -- Extraction metadata
    extraction_metadata JSONB,

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
    -- Federation privacy
    opt_out_federation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Share policy
    share_policy JSONB,

    -- Extraction metadata
    extraction_metadata JSONB,

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
-- BROWSER AUTOMATION TABLES
-- ============================================================================

-- Extractors: Learned content extractors for web pages
CREATE TABLE IF NOT EXISTS extractors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- URL pattern (regex) that this extractor matches
    url_pattern TEXT NOT NULL,

    -- Human-readable name
    name TEXT NOT NULL,

    -- JavaScript extraction script
    -- Called with (selector) argument, returns extracted content object
    script TEXT NOT NULL,

    -- Effectiveness score (0-1), updated based on usage
    effectiveness FLOAT DEFAULT 0.5,

    -- Usage tracking
    usage_count INTEGER DEFAULT 0,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    modified_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(url_pattern),
    CONSTRAINT extractors_valid_effectiveness CHECK (effectiveness >= 0 AND effectiveness <= 1)
);

CREATE INDEX IF NOT EXISTS idx_extractors_pattern ON extractors(url_pattern);
CREATE INDEX IF NOT EXISTS idx_extractors_effectiveness ON extractors(effectiveness DESC);

-- ============================================================================
-- FEDERATION TABLES
-- ============================================================================

-- Federation Nodes: Identity and metadata for federation nodes
CREATE TABLE IF NOT EXISTS federation_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- DID identifier (unique)
    did TEXT NOT NULL UNIQUE,
    -- Format: did:vkb:web:domain or did:vkb:key:z6Mk...

    -- Connection information
    federation_endpoint TEXT,
    mcp_endpoint TEXT,

    -- Cryptographic identity
    public_key_multibase TEXT NOT NULL,
    -- Ed25519 public key in multibase format

    -- Node profile (optional)
    name TEXT,
    domains TEXT[] DEFAULT '{}',

    -- Capabilities
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    -- Values: belief_sync, aggregation_participate, aggregation_publish

    -- Federation privacy
    opt_out_federation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Share policy
    share_policy JSONB,

    -- Extraction metadata
    extraction_metadata JSONB,

    -- Status
    status TEXT NOT NULL DEFAULT 'discovered',
    -- Values: discovered, connecting, active, suspended, unreachable

    -- Trust phase (per TRUST_MODEL.md)
    trust_phase TEXT NOT NULL DEFAULT 'observer',
    -- Values: observer, contributor, participant, anchor

    phase_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Protocol version
    protocol_version TEXT NOT NULL DEFAULT '1.0',

    -- Timestamps
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ,

    -- Metadata
    metadata JSONB DEFAULT '{}',

    CONSTRAINT federation_nodes_valid_status CHECK (
        status IN ('discovered', 'connecting', 'active', 'suspended', 'unreachable')
    ),
    CONSTRAINT federation_nodes_valid_phase CHECK (
        trust_phase IN ('observer', 'contributor', 'participant', 'anchor')
    )
);

CREATE INDEX IF NOT EXISTS idx_federation_nodes_did ON federation_nodes(did);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_status ON federation_nodes(status);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_phase ON federation_nodes(trust_phase);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_domains ON federation_nodes USING GIN (domains);

-- Belief Provenance: Federation path and origin tracking
CREATE TABLE IF NOT EXISTS belief_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to local belief
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,

    -- Federation identity
    federation_id UUID NOT NULL,
    -- Stable ID across federation (same belief on different nodes)

    -- Origin information
    origin_node_id UUID NOT NULL REFERENCES federation_nodes(id),
    origin_belief_id UUID NOT NULL,
    -- Original belief ID on origin node

    -- Cryptographic proof
    origin_signature TEXT NOT NULL,
    signed_at TIMESTAMPTZ NOT NULL,
    signature_verified BOOLEAN DEFAULT FALSE,

    -- Federation path
    hop_count INTEGER NOT NULL DEFAULT 1,
    federation_path TEXT[] NOT NULL DEFAULT '{}',
    -- Array of DIDs showing path from origin

    -- Share level at which belief was shared
    share_level TEXT NOT NULL DEFAULT 'belief_only',
    -- Values: belief_only, with_provenance, full

    -- Reception timestamps
    received_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT belief_provenance_valid_share_level CHECK (
        share_level IN ('belief_only', 'with_provenance', 'full')
    )
);

CREATE INDEX IF NOT EXISTS idx_belief_provenance_belief ON belief_provenance(belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_provenance_federation ON belief_provenance(federation_id);
CREATE INDEX IF NOT EXISTS idx_belief_provenance_origin ON belief_provenance(origin_node_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_belief_provenance_unique
    ON belief_provenance(origin_node_id, origin_belief_id);

-- Node Trust: Node-to-node trust relationships
CREATE TABLE IF NOT EXISTS node_trust (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The node being trusted
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Trust dimensions (mirrors DimensionalConfidence)
    trust JSONB NOT NULL DEFAULT '{"overall": 0.1}',
    -- Dimensions:
    --   overall: Combined score (0-1)
    --   belief_accuracy: How often their beliefs are corroborated
    --   extraction_quality: Quality of their knowledge extraction
    --   curation_accuracy: How well they handle contradictions
    --   uptime_reliability: Consistent availability
    --   contribution_consistency: Regular, quality participation
    --   endorsement_strength: Trust from others we trust
    --   domain_expertise: Per-domain scores (JSONB nested)

    -- Trust factors
    beliefs_received INTEGER DEFAULT 0,
    beliefs_corroborated INTEGER DEFAULT 0,
    beliefs_disputed INTEGER DEFAULT 0,
    sync_requests_served INTEGER DEFAULT 0,
    aggregation_participations INTEGER DEFAULT 0,

    -- Social trust
    endorsements_received INTEGER DEFAULT 0,
    endorsements_given INTEGER DEFAULT 0,

    -- Relationship timeline
    relationship_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_interaction_at TIMESTAMPTZ,

    -- Manual adjustments (user override)
    manual_trust_adjustment NUMERIC(3,2) DEFAULT 0,
    adjustment_reason TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(node_id)
);

CREATE INDEX IF NOT EXISTS idx_node_trust_node ON node_trust(node_id);
CREATE INDEX IF NOT EXISTS idx_node_trust_overall ON node_trust(
    ((trust->>'overall')::numeric) DESC
);

-- User Node Trust: User preference overrides for node trust
CREATE TABLE IF NOT EXISTS user_node_trust (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- The node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- User preference
    trust_preference TEXT NOT NULL DEFAULT 'automatic',
    -- Values: blocked, reduced, automatic, elevated, anchor

    -- Manual trust score (if not automatic)
    manual_trust_score NUMERIC(3,2),

    -- Reason for override
    reason TEXT,

    -- Domain-specific overrides
    domain_overrides JSONB DEFAULT '{}',
    -- Format: { "tech": "elevated", "politics": "blocked" }

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(node_id),
    CONSTRAINT user_node_trust_valid_preference CHECK (
        trust_preference IN ('blocked', 'reduced', 'automatic', 'elevated', 'anchor')
    )
);

CREATE INDEX IF NOT EXISTS idx_user_node_trust_node ON user_node_trust(node_id);
CREATE INDEX IF NOT EXISTS idx_user_node_trust_pref ON user_node_trust(trust_preference);

-- Belief Trust Annotations: Per-belief trust adjustments from federation
CREATE TABLE IF NOT EXISTS belief_trust_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,

    -- Annotation type
    type TEXT NOT NULL,
    -- Values: corroboration, dispute, endorsement, flag

    -- Source of annotation
    source_node_id UUID REFERENCES federation_nodes(id) ON DELETE SET NULL,

    -- Corroboration details
    corroboration_attestation JSONB,
    -- Contains: claim_hash, corroboration_level, confidence_boost, etc.

    -- Confidence adjustment
    confidence_delta NUMERIC(3,2) DEFAULT 0,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    CONSTRAINT belief_trust_annotations_valid_type CHECK (
        type IN ('corroboration', 'dispute', 'endorsement', 'flag')
    )
);

CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_belief ON belief_trust_annotations(belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_type ON belief_trust_annotations(type);
CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_source ON belief_trust_annotations(source_node_id);

-- Aggregated Beliefs: Privacy-preserving aggregates from federation
CREATE TABLE IF NOT EXISTS aggregated_beliefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query that produced this aggregate
    query_hash TEXT NOT NULL,
    -- SHA256 of normalized query for deduplication

    query_domain TEXT[] NOT NULL DEFAULT '{}',
    query_semantic TEXT,

    -- Aggregate results
    collective_confidence NUMERIC(3,2) NOT NULL,
    agreement_score NUMERIC(3,2),

    -- Participation stats (not which nodes, just counts)
    contributor_count INTEGER NOT NULL,
    node_count INTEGER NOT NULL,
    total_belief_count INTEGER,

    -- Stance summary (AI-generated)
    stance_summary TEXT,
    key_factors TEXT[],

    -- Privacy guarantees
    privacy_epsilon NUMERIC(5,4) NOT NULL,
    privacy_delta NUMERIC(10,9) NOT NULL,
    privacy_mechanism TEXT NOT NULL DEFAULT 'laplace',

    -- Aggregator info
    aggregator_node_id UUID REFERENCES federation_nodes(id),

    -- Validity
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,

    -- Metadata
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_query ON aggregated_beliefs(query_hash);
CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_domain ON aggregated_beliefs USING GIN (query_domain);
CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_computed ON aggregated_beliefs(computed_at DESC);

-- Aggregation Sources: Links aggregates to anonymized sources
CREATE TABLE IF NOT EXISTS aggregation_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    aggregated_belief_id UUID NOT NULL REFERENCES aggregated_beliefs(id) ON DELETE CASCADE,

    -- Anonymous source tracking (hashed for privacy)
    source_hash TEXT NOT NULL,
    -- SHA256(node_did + salt), not reversible

    -- Contribution metadata (no identifying info)
    contribution_weight NUMERIC(3,2) NOT NULL,
    local_confidence NUMERIC(3,2),
    local_belief_count INTEGER,

    -- Timestamps
    contributed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_aggregation_sources_aggregate ON aggregation_sources(aggregated_belief_id);

-- Tension Resolutions: Cross-node conflict resolution
CREATE TABLE IF NOT EXISTS tension_resolutions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Link to local tension (if exists)
    tension_id UUID REFERENCES tensions(id) ON DELETE SET NULL,

    -- Cross-node tension identification
    is_cross_node BOOLEAN NOT NULL DEFAULT FALSE,

    -- Participating nodes
    node_a_id UUID REFERENCES federation_nodes(id),
    node_b_id UUID REFERENCES federation_nodes(id),

    -- Resolution proposal
    proposed_resolution TEXT NOT NULL,
    -- Values: supersede_a, supersede_b, accept_both, merge, refer_to_authority

    resolution_rationale TEXT,

    -- Proposer
    proposed_by_node_id UUID REFERENCES federation_nodes(id),
    proposed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Consensus tracking
    consensus_method TEXT NOT NULL DEFAULT 'trust_weighted',
    -- Values: trust_weighted, unanimous, majority, authority

    consensus_threshold NUMERIC(3,2) DEFAULT 0.6,
    current_support NUMERIC(3,2) DEFAULT 0,

    -- Federation privacy
    opt_out_federation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Share policy
    share_policy JSONB,

    -- Extraction metadata
    extraction_metadata JSONB,

    -- Status
    status TEXT NOT NULL DEFAULT 'proposed',
    -- Values: proposed, voting, accepted, rejected, implemented

    resolved_at TIMESTAMPTZ,

    -- Result
    winning_belief_id UUID REFERENCES beliefs(id),
    superseded_belief_id UUID REFERENCES beliefs(id),

    CONSTRAINT tension_resolutions_valid_proposal CHECK (
        proposed_resolution IN ('supersede_a', 'supersede_b', 'accept_both', 'merge', 'refer_to_authority')
    ),
    CONSTRAINT tension_resolutions_valid_status CHECK (
        status IN ('proposed', 'voting', 'accepted', 'rejected', 'implemented')
    )
);

CREATE INDEX IF NOT EXISTS idx_tension_resolutions_tension ON tension_resolutions(tension_id);
CREATE INDEX IF NOT EXISTS idx_tension_resolutions_status ON tension_resolutions(status);
CREATE INDEX IF NOT EXISTS idx_tension_resolutions_nodes ON tension_resolutions(node_a_id, node_b_id);

-- Consensus Votes: Individual votes on tension resolutions
CREATE TABLE IF NOT EXISTS consensus_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    resolution_id UUID NOT NULL REFERENCES tension_resolutions(id) ON DELETE CASCADE,

    -- Voter
    voter_node_id UUID NOT NULL REFERENCES federation_nodes(id),

    -- Vote
    vote TEXT NOT NULL,
    -- Values: support, oppose, abstain

    -- Weight (based on trust at vote time)
    vote_weight NUMERIC(3,2) NOT NULL,

    -- Rationale (optional)
    rationale TEXT,

    -- Signature
    signature TEXT NOT NULL,

    voted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(resolution_id, voter_node_id),
    CONSTRAINT consensus_votes_valid_vote CHECK (
        vote IN ('support', 'oppose', 'abstain')
    )
);

CREATE INDEX IF NOT EXISTS idx_consensus_votes_resolution ON consensus_votes(resolution_id);
CREATE INDEX IF NOT EXISTS idx_consensus_votes_voter ON consensus_votes(voter_node_id);

-- Sync State: Track synchronization state with peer nodes
CREATE TABLE IF NOT EXISTS sync_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Peer node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Sync cursors
    last_received_cursor TEXT,
    last_sent_cursor TEXT,

    -- Vector clock (for conflict resolution)
    vector_clock JSONB NOT NULL DEFAULT '{}',
    -- Format: { "node_did": sequence_number, ... }

    -- Sync status
    status TEXT NOT NULL DEFAULT 'idle',
    -- Values: idle, syncing, error, paused

    -- Statistics
    beliefs_sent INTEGER DEFAULT 0,
    beliefs_received INTEGER DEFAULT 0,
    last_sync_duration_ms INTEGER,

    -- Error tracking
    last_error TEXT,
    error_count INTEGER DEFAULT 0,

    -- Timestamps
    last_sync_at TIMESTAMPTZ,
    next_sync_scheduled TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE(node_id),
    CONSTRAINT sync_state_valid_status CHECK (
        status IN ('idle', 'syncing', 'error', 'paused')
    )
);

CREATE INDEX IF NOT EXISTS idx_sync_state_node ON sync_state(node_id);
CREATE INDEX IF NOT EXISTS idx_sync_state_status ON sync_state(status);
CREATE INDEX IF NOT EXISTS idx_sync_state_next ON sync_state(next_sync_scheduled);

-- Sync Events: Audit log for sync operations
CREATE TABLE IF NOT EXISTS sync_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Peer node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Event type
    event_type TEXT NOT NULL,
    -- Values: sync_started, sync_completed, sync_failed, belief_sent, belief_received,
    --         cursor_updated, conflict_detected, conflict_resolved

    -- Event details
    details JSONB DEFAULT '{}',

    -- Direction
    direction TEXT NOT NULL DEFAULT 'inbound',
    -- Values: inbound, outbound

    -- Related belief (if applicable)
    belief_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,

    -- Timestamp
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sync_events_node ON sync_events(node_id);
CREATE INDEX IF NOT EXISTS idx_sync_events_type ON sync_events(event_type);
CREATE INDEX IF NOT EXISTS idx_sync_events_time ON sync_events(occurred_at DESC);

-- Sync Outbound Queue: Pending outbound sync operations
CREATE TABLE IF NOT EXISTS sync_outbound_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Target node (NULL = broadcast to all syncing nodes)
    target_node_id UUID REFERENCES federation_nodes(id) ON DELETE CASCADE,

    -- Operation type
    operation TEXT NOT NULL,
    -- Values: share_belief, update_belief, supersede_belief, share_tension

    -- Payload
    belief_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
    payload JSONB,

    -- Priority
    priority INTEGER NOT NULL DEFAULT 5,
    -- 1 = highest, 10 = lowest

    -- Federation privacy
    opt_out_federation BOOLEAN NOT NULL DEFAULT FALSE,

    -- Share policy
    share_policy JSONB,

    -- Extraction metadata
    extraction_metadata JSONB,

    -- Status
    status TEXT NOT NULL DEFAULT 'pending',
    -- Values: pending, processing, sent, failed, cancelled

    -- Retry tracking
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_attempt_at TIMESTAMPTZ,
    last_error TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    scheduled_for TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT sync_outbound_queue_valid_status CHECK (
        status IN ('pending', 'processing', 'sent', 'failed', 'cancelled')
    )
);

CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_target ON sync_outbound_queue(target_node_id);
CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_status ON sync_outbound_queue(status);
CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_scheduled ON sync_outbound_queue(scheduled_for)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_sync_outbound_queue_belief ON sync_outbound_queue(belief_id);

-- Peer Nodes: Simplified peer tracking for federation sync
-- This table provides a lightweight peer registry used by federation sync tests.
-- For full node identity and metadata, use federation_nodes.
CREATE TABLE IF NOT EXISTS peer_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Node identifier (unique string ID for the peer)
    node_id TEXT NOT NULL UNIQUE,

    -- Connection endpoint
    endpoint TEXT NOT NULL,

    -- Trust level (0.0 to 1.0)
    trust_level NUMERIC(3,2) NOT NULL DEFAULT 0.5,

    -- Peer status
    status TEXT NOT NULL DEFAULT 'discovered',
    -- Values: discovered, active, suspended, unreachable

    -- Timestamps
    last_seen TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT peer_nodes_valid_trust CHECK (trust_level >= 0 AND trust_level <= 1),
    CONSTRAINT peer_nodes_valid_status CHECK (
        status IN ('discovered', 'active', 'suspended', 'unreachable')
    )
);

CREATE INDEX IF NOT EXISTS idx_peer_nodes_node_id ON peer_nodes(node_id);
CREATE INDEX IF NOT EXISTS idx_peer_nodes_status ON peer_nodes(status);
CREATE INDEX IF NOT EXISTS idx_peer_nodes_trust ON peer_nodes(trust_level DESC);
CREATE INDEX IF NOT EXISTS idx_peer_nodes_last_seen ON peer_nodes(last_seen DESC);

-- Trust Edges: 4D trust relationships between DIDs
-- Implements the multi-dimensional trust model with competence, integrity,
-- confidentiality, and judgment dimensions. Used by the privacy module
-- for DID-to-DID trust management.
CREATE TABLE IF NOT EXISTS trust_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Trust relationship: source trusts target
    source_did TEXT NOT NULL,
    target_did TEXT NOT NULL,

    -- 4D trust dimensions (all 0.0 to 1.0)
    competence NUMERIC(3,2) NOT NULL DEFAULT 0.5,
    -- Ability to perform tasks correctly

    integrity NUMERIC(3,2) NOT NULL DEFAULT 0.5,
    -- Honesty and consistency

    confidentiality NUMERIC(3,2) NOT NULL DEFAULT 0.5,
    -- Ability to keep secrets

    judgment NUMERIC(3,2) NOT NULL DEFAULT 0.1,
    -- Ability to evaluate others (affects delegated trust)
    -- Very low default: trust in judgment must be earned

    -- Domain-specific trust (NULL = global trust)
    domain TEXT,

    -- Delegation policy
    can_delegate BOOLEAN NOT NULL DEFAULT FALSE,
    -- If true, this trust edge can be used for transitive trust

    delegation_depth INTEGER NOT NULL DEFAULT 0,
    -- Maximum hops for delegation (0 = unlimited when can_delegate=true)

    -- Trust decay settings
    decay_rate NUMERIC(5,4) NOT NULL DEFAULT 0.0,
    -- Rate of trust decay per day (0.0 = no decay)

    decay_model TEXT NOT NULL DEFAULT 'exponential',
    -- Values: none, linear, exponential

    last_refreshed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- When trust was last confirmed/refreshed

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT trust_edges_valid_competence CHECK (competence >= 0 AND competence <= 1),
    CONSTRAINT trust_edges_valid_integrity CHECK (integrity >= 0 AND integrity <= 1),
    CONSTRAINT trust_edges_valid_confidentiality CHECK (confidentiality >= 0 AND confidentiality <= 1),
    CONSTRAINT trust_edges_valid_judgment CHECK (judgment >= 0 AND judgment <= 1),
    CONSTRAINT trust_edges_valid_decay_rate CHECK (decay_rate >= 0 AND decay_rate <= 1),
    CONSTRAINT trust_edges_valid_decay_model CHECK (
        decay_model IN ('none', 'linear', 'exponential')
    ),
    CONSTRAINT trust_edges_valid_delegation_depth CHECK (delegation_depth >= 0),
    CONSTRAINT trust_edges_no_self_trust CHECK (source_did != target_did)
);

-- Unique constraint for upsert: one edge per (source, target, domain) tuple
-- COALESCE handles NULL domain for uniqueness
CREATE UNIQUE INDEX IF NOT EXISTS idx_trust_edges_unique
    ON trust_edges(source_did, target_did, COALESCE(domain, ''));

CREATE INDEX IF NOT EXISTS idx_trust_edges_source ON trust_edges(source_did);
CREATE INDEX IF NOT EXISTS idx_trust_edges_target ON trust_edges(target_did);
CREATE INDEX IF NOT EXISTS idx_trust_edges_domain ON trust_edges(domain) WHERE domain IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trust_edges_expires ON trust_edges(expires_at)
    WHERE expires_at IS NOT NULL;

-- ============================================================================
-- BELIEFS TABLE EXTENSIONS FOR FEDERATION
-- ============================================================================

-- Add federation columns to beliefs (backward compatible)
-- NOTE: Run as ALTER TABLE in migration, not in initial schema
-- ALTER TABLE beliefs
--     ADD COLUMN IF NOT EXISTS origin_node_id UUID REFERENCES federation_nodes(id),
--     ADD COLUMN IF NOT EXISTS is_local BOOLEAN NOT NULL DEFAULT TRUE,
--     ADD COLUMN IF NOT EXISTS federation_id UUID,
--     ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'private',
--     ADD COLUMN IF NOT EXISTS share_level TEXT NOT NULL DEFAULT 'belief_only';

-- For new installations, include federation columns in beliefs table definition
-- These columns are added via the ALTER statements in the migration file

-- ============================================================================
-- FEDERATION VIEWS
-- ============================================================================

-- Active federation nodes with trust
CREATE OR REPLACE VIEW federation_nodes_with_trust AS
SELECT
    fn.*,
    nt.trust,
    (nt.trust->>'overall')::numeric as trust_overall,
    nt.beliefs_received,
    nt.beliefs_corroborated,
    unt.trust_preference as user_preference
FROM federation_nodes fn
LEFT JOIN node_trust nt ON fn.id = nt.node_id
LEFT JOIN user_node_trust unt ON fn.id = unt.node_id;

-- Federated beliefs with provenance
CREATE OR REPLACE VIEW federated_beliefs AS
SELECT
    b.*,
    bp.federation_id,
    bp.origin_node_id,
    bp.hop_count,
    bp.federation_path,
    bp.share_level as received_share_level,
    fn.did as origin_did,
    fn.name as origin_name
FROM beliefs b
JOIN belief_provenance bp ON b.id = bp.belief_id
JOIN federation_nodes fn ON bp.origin_node_id = fn.id;

-- Sync status overview
CREATE OR REPLACE VIEW sync_status_overview AS
SELECT
    fn.id as node_id,
    fn.did,
    fn.name,
    fn.status as node_status,
    ss.status as sync_status,
    ss.last_sync_at,
    ss.beliefs_sent,
    ss.beliefs_received,
    ss.error_count
FROM federation_nodes fn
LEFT JOIN sync_state ss ON fn.id = ss.node_id
WHERE fn.status = 'active';

-- ============================================================================
-- DEFAULT DATA
-- ============================================================================

-- Insert default embedding type
INSERT INTO embedding_types (id, provider, model, dimensions, is_default, status)
VALUES ('openai_text3_small', 'openai', 'text-embedding-3-small', 1536, TRUE, 'active')
ON CONFLICT (id) DO NOTHING;
-- Migration 002: Add sharing tables (Issue #50)
-- Implements consent chains and encrypted shares for DIRECT sharing level

-- ============================================================================
-- CONSENT CHAINS
-- ============================================================================

-- Consent chains track the provenance and permissions for shared beliefs
CREATE TABLE IF NOT EXISTS consent_chains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to the shared belief
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    
    -- Origin information (first sharer in the chain)
    origin_sharer TEXT NOT NULL,  -- DID of original sharer
    origin_timestamp TIMESTAMPTZ NOT NULL,
    origin_policy JSONB NOT NULL,  -- SharePolicy as JSON
    origin_signature BYTEA NOT NULL,  -- Ed25519 signature
    
    -- Chain integrity
    chain_hash BYTEA NOT NULL,  -- SHA256(consent_origin + signature)
    hops JSONB NOT NULL DEFAULT '[]',  -- Array of hop records for BOUNDED/CASCADING
    
    -- Status
    revoked BOOLEAN NOT NULL DEFAULT FALSE,
    revoked_at TIMESTAMPTZ,
    revoked_by TEXT,  -- DID
    revocation_reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_consent_chains_belief ON consent_chains(belief_id);
CREATE INDEX IF NOT EXISTS idx_consent_chains_sharer ON consent_chains(origin_sharer);
CREATE INDEX IF NOT EXISTS idx_consent_chains_revoked ON consent_chains(revoked);
CREATE INDEX IF NOT EXISTS idx_consent_chains_created ON consent_chains(created_at DESC);

-- ============================================================================
-- SHARES
-- ============================================================================

-- Shares link encrypted content to consent chains and recipients
CREATE TABLE IF NOT EXISTS shares (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Link to consent chain
    consent_chain_id UUID NOT NULL REFERENCES consent_chains(id) ON DELETE CASCADE,
    
    -- Encrypted content for recipient
    encrypted_envelope JSONB NOT NULL,  -- EncryptionEnvelope as JSON
    
    -- Recipient
    recipient_did TEXT NOT NULL,
    
    -- Access tracking
    accessed_at TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shares_consent_chain ON shares(consent_chain_id);
CREATE INDEX IF NOT EXISTS idx_shares_recipient ON shares(recipient_did);
CREATE INDEX IF NOT EXISTS idx_shares_created ON shares(created_at DESC);

-- Unique constraint: one share per consent chain per recipient
CREATE UNIQUE INDEX IF NOT EXISTS idx_shares_unique_recipient 
    ON shares(consent_chain_id, recipient_did);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- View of shares with consent chain info
CREATE OR REPLACE VIEW shares_with_consent AS
SELECT 
    s.*,
    cc.belief_id,
    cc.origin_sharer,
    cc.origin_timestamp,
    cc.origin_policy,
    cc.revoked as consent_revoked
FROM shares s
JOIN consent_chains cc ON s.consent_chain_id = cc.id;

-- Active shares (not revoked)
CREATE OR REPLACE VIEW active_shares AS
SELECT * FROM shares_with_consent
WHERE NOT consent_revoked;
