-- Migration: Add federation support to Valence
-- Version: 001_add_federation
-- Description: Adds federation tables and extends beliefs table for federation

-- =============================================================================
-- NOTE: This migration assumes the base schema from schema.sql has been applied.
-- Run this AFTER the initial schema.sql to add federation support.
-- =============================================================================

BEGIN;

-- =============================================================================
-- 1. ADD FEDERATION COLUMNS TO EXISTING BELIEFS TABLE
-- =============================================================================

-- Add federation columns to beliefs (backward compatible)
ALTER TABLE beliefs
    ADD COLUMN IF NOT EXISTS origin_node_id UUID,
    ADD COLUMN IF NOT EXISTS is_local BOOLEAN NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS federation_id UUID,
    ADD COLUMN IF NOT EXISTS visibility TEXT NOT NULL DEFAULT 'private',
    ADD COLUMN IF NOT EXISTS share_level TEXT NOT NULL DEFAULT 'belief_only';

-- Add constraints (only if they don't exist)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'beliefs_valid_visibility'
    ) THEN
        ALTER TABLE beliefs ADD CONSTRAINT beliefs_valid_visibility
            CHECK (visibility IN ('private', 'trusted', 'federated', 'public'));
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'beliefs_valid_share_level'
    ) THEN
        ALTER TABLE beliefs ADD CONSTRAINT beliefs_valid_share_level
            CHECK (share_level IN ('belief_only', 'with_provenance', 'full'));
    END IF;
END $$;

-- Add indexes for federation queries
CREATE INDEX IF NOT EXISTS idx_beliefs_local ON beliefs(is_local);
CREATE INDEX IF NOT EXISTS idx_beliefs_visibility ON beliefs(visibility);
CREATE INDEX IF NOT EXISTS idx_beliefs_federation ON beliefs(federation_id)
    WHERE federation_id IS NOT NULL;

-- =============================================================================
-- 2. CREATE FEDERATION TABLES (if not exists - for idempotency)
-- =============================================================================

-- Federation Nodes
CREATE TABLE IF NOT EXISTS federation_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    did TEXT NOT NULL UNIQUE,
    federation_endpoint TEXT,
    mcp_endpoint TEXT,
    public_key_multibase TEXT NOT NULL,
    name TEXT,
    domains TEXT[] DEFAULT '{}',
    capabilities TEXT[] NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'discovered',
    trust_phase TEXT NOT NULL DEFAULT 'observer',
    phase_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    protocol_version TEXT NOT NULL DEFAULT '1.0',
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ,
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

-- Add foreign key from beliefs to federation_nodes (now that table exists)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'beliefs_origin_node_fk'
    ) THEN
        ALTER TABLE beliefs ADD CONSTRAINT beliefs_origin_node_fk
            FOREIGN KEY (origin_node_id) REFERENCES federation_nodes(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_beliefs_origin_node ON beliefs(origin_node_id)
    WHERE origin_node_id IS NOT NULL;

-- Belief Provenance
CREATE TABLE IF NOT EXISTS belief_provenance (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    federation_id UUID NOT NULL,
    origin_node_id UUID NOT NULL REFERENCES federation_nodes(id),
    origin_belief_id UUID NOT NULL,
    origin_signature TEXT NOT NULL,
    signed_at TIMESTAMPTZ NOT NULL,
    signature_verified BOOLEAN DEFAULT FALSE,
    hop_count INTEGER NOT NULL DEFAULT 1,
    federation_path TEXT[] NOT NULL DEFAULT '{}',
    share_level TEXT NOT NULL DEFAULT 'belief_only',
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

-- Node Trust
CREATE TABLE IF NOT EXISTS node_trust (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    trust JSONB NOT NULL DEFAULT '{"overall": 0.1}',
    beliefs_received INTEGER DEFAULT 0,
    beliefs_corroborated INTEGER DEFAULT 0,
    beliefs_disputed INTEGER DEFAULT 0,
    sync_requests_served INTEGER DEFAULT 0,
    aggregation_participations INTEGER DEFAULT 0,
    endorsements_received INTEGER DEFAULT 0,
    endorsements_given INTEGER DEFAULT 0,
    relationship_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_interaction_at TIMESTAMPTZ,
    manual_trust_adjustment NUMERIC(3,2) DEFAULT 0,
    adjustment_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(node_id)
);

CREATE INDEX IF NOT EXISTS idx_node_trust_node ON node_trust(node_id);
CREATE INDEX IF NOT EXISTS idx_node_trust_overall ON node_trust(
    ((trust->>'overall')::numeric) DESC
);

-- User Node Trust
CREATE TABLE IF NOT EXISTS user_node_trust (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    trust_preference TEXT NOT NULL DEFAULT 'automatic',
    manual_trust_score NUMERIC(3,2),
    reason TEXT,
    domain_overrides JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(node_id),
    CONSTRAINT user_node_trust_valid_preference CHECK (
        trust_preference IN ('blocked', 'reduced', 'automatic', 'elevated', 'anchor')
    )
);

CREATE INDEX IF NOT EXISTS idx_user_node_trust_node ON user_node_trust(node_id);
CREATE INDEX IF NOT EXISTS idx_user_node_trust_pref ON user_node_trust(trust_preference);

-- Belief Trust Annotations
CREATE TABLE IF NOT EXISTS belief_trust_annotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    type TEXT NOT NULL,
    source_node_id UUID REFERENCES federation_nodes(id) ON DELETE SET NULL,
    corroboration_attestation JSONB,
    confidence_delta NUMERIC(3,2) DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    CONSTRAINT belief_trust_annotations_valid_type CHECK (
        type IN ('corroboration', 'dispute', 'endorsement', 'flag')
    )
);

CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_belief ON belief_trust_annotations(belief_id);
CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_type ON belief_trust_annotations(type);
CREATE INDEX IF NOT EXISTS idx_belief_trust_annotations_source ON belief_trust_annotations(source_node_id);

-- Aggregated Beliefs
CREATE TABLE IF NOT EXISTS aggregated_beliefs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash TEXT NOT NULL,
    query_domain TEXT[] NOT NULL DEFAULT '{}',
    query_semantic TEXT,
    collective_confidence NUMERIC(3,2) NOT NULL,
    agreement_score NUMERIC(3,2),
    contributor_count INTEGER NOT NULL,
    node_count INTEGER NOT NULL,
    total_belief_count INTEGER,
    stance_summary TEXT,
    key_factors TEXT[],
    privacy_epsilon NUMERIC(5,4) NOT NULL,
    privacy_delta NUMERIC(10,9) NOT NULL,
    privacy_mechanism TEXT NOT NULL DEFAULT 'laplace',
    aggregator_node_id UUID REFERENCES federation_nodes(id),
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_query ON aggregated_beliefs(query_hash);
CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_domain ON aggregated_beliefs USING GIN (query_domain);
CREATE INDEX IF NOT EXISTS idx_aggregated_beliefs_computed ON aggregated_beliefs(computed_at DESC);

-- Aggregation Sources
CREATE TABLE IF NOT EXISTS aggregation_sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    aggregated_belief_id UUID NOT NULL REFERENCES aggregated_beliefs(id) ON DELETE CASCADE,
    source_hash TEXT NOT NULL,
    contribution_weight NUMERIC(3,2) NOT NULL,
    local_confidence NUMERIC(3,2),
    local_belief_count INTEGER,
    contributed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_aggregation_sources_aggregate ON aggregation_sources(aggregated_belief_id);

-- Tension Resolutions
CREATE TABLE IF NOT EXISTS tension_resolutions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tension_id UUID REFERENCES tensions(id) ON DELETE SET NULL,
    is_cross_node BOOLEAN NOT NULL DEFAULT FALSE,
    node_a_id UUID REFERENCES federation_nodes(id),
    node_b_id UUID REFERENCES federation_nodes(id),
    proposed_resolution TEXT NOT NULL,
    resolution_rationale TEXT,
    proposed_by_node_id UUID REFERENCES federation_nodes(id),
    proposed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consensus_method TEXT NOT NULL DEFAULT 'trust_weighted',
    consensus_threshold NUMERIC(3,2) DEFAULT 0.6,
    current_support NUMERIC(3,2) DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'proposed',
    resolved_at TIMESTAMPTZ,
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

-- Consensus Votes
CREATE TABLE IF NOT EXISTS consensus_votes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    resolution_id UUID NOT NULL REFERENCES tension_resolutions(id) ON DELETE CASCADE,
    voter_node_id UUID NOT NULL REFERENCES federation_nodes(id),
    vote TEXT NOT NULL,
    vote_weight NUMERIC(3,2) NOT NULL,
    rationale TEXT,
    signature TEXT NOT NULL,
    voted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(resolution_id, voter_node_id),
    CONSTRAINT consensus_votes_valid_vote CHECK (
        vote IN ('support', 'oppose', 'abstain')
    )
);

CREATE INDEX IF NOT EXISTS idx_consensus_votes_resolution ON consensus_votes(resolution_id);
CREATE INDEX IF NOT EXISTS idx_consensus_votes_voter ON consensus_votes(voter_node_id);

-- Sync State
CREATE TABLE IF NOT EXISTS sync_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    last_received_cursor TEXT,
    last_sent_cursor TEXT,
    vector_clock JSONB NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'idle',
    beliefs_sent INTEGER DEFAULT 0,
    beliefs_received INTEGER DEFAULT 0,
    last_sync_duration_ms INTEGER,
    last_error TEXT,
    error_count INTEGER DEFAULT 0,
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

-- Sync Events
CREATE TABLE IF NOT EXISTS sync_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    direction TEXT NOT NULL DEFAULT 'inbound',
    belief_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sync_events_node ON sync_events(node_id);
CREATE INDEX IF NOT EXISTS idx_sync_events_type ON sync_events(event_type);
CREATE INDEX IF NOT EXISTS idx_sync_events_time ON sync_events(occurred_at DESC);

-- Sync Outbound Queue
CREATE TABLE IF NOT EXISTS sync_outbound_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_node_id UUID REFERENCES federation_nodes(id) ON DELETE CASCADE,
    operation TEXT NOT NULL,
    belief_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
    payload JSONB,
    priority INTEGER NOT NULL DEFAULT 5,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_attempt_at TIMESTAMPTZ,
    last_error TEXT,
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

-- =============================================================================
-- 3. CREATE FEDERATION VIEWS
-- =============================================================================

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
    bp.origin_node_id as provenance_origin_node_id,
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

-- =============================================================================
-- 4. UPDATE EXISTING BELIEFS TO BE LOCAL
-- =============================================================================

-- Mark all existing beliefs as local (if not already set)
UPDATE beliefs SET is_local = TRUE WHERE is_local IS NULL;
UPDATE beliefs SET visibility = 'private' WHERE visibility IS NULL;
UPDATE beliefs SET share_level = 'belief_only' WHERE share_level IS NULL;

COMMIT;

-- =============================================================================
-- ROLLBACK SCRIPT (run separately if needed)
-- =============================================================================

-- To rollback this migration, run the following:
--
-- BEGIN;
--
-- DROP VIEW IF EXISTS sync_status_overview;
-- DROP VIEW IF EXISTS federated_beliefs;
-- DROP VIEW IF EXISTS federation_nodes_with_trust;
--
-- DROP TABLE IF EXISTS sync_outbound_queue;
-- DROP TABLE IF EXISTS sync_events;
-- DROP TABLE IF EXISTS sync_state;
-- DROP TABLE IF EXISTS consensus_votes;
-- DROP TABLE IF EXISTS tension_resolutions;
-- DROP TABLE IF EXISTS aggregation_sources;
-- DROP TABLE IF EXISTS aggregated_beliefs;
-- DROP TABLE IF EXISTS belief_trust_annotations;
-- DROP TABLE IF EXISTS user_node_trust;
-- DROP TABLE IF EXISTS node_trust;
-- DROP TABLE IF EXISTS belief_provenance;
--
-- ALTER TABLE beliefs DROP CONSTRAINT IF EXISTS beliefs_origin_node_fk;
-- DROP TABLE IF EXISTS federation_nodes;
--
-- ALTER TABLE beliefs
--     DROP COLUMN IF EXISTS origin_node_id,
--     DROP COLUMN IF EXISTS is_local,
--     DROP COLUMN IF EXISTS federation_id,
--     DROP COLUMN IF EXISTS visibility,
--     DROP COLUMN IF EXISTS share_level;
--
-- COMMIT;
