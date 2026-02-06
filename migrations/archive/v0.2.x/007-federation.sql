-- ============================================================================
-- Migration 007: Federation Tables
-- ============================================================================
-- Implements peer-to-peer federation for distributed Valence networks.
-- Allows beliefs to be shared and synchronized across trusted peer nodes.
--
-- Tables:
--   - federation_nodes: Known peer nodes in the federation network
--   - node_trust: Trust information for each peer node
--   - sync_state: Tracks sync progress with each peer
--
-- Adds to beliefs:
--   - is_local: Whether belief originated locally
--   - origin_node_id: Source node for federated beliefs
-- ============================================================================

-- ============================================================================
-- PHASE 1: ENUM TYPES
-- ============================================================================

-- Federation node status (matches models.NodeStatus)
DO $$ BEGIN
    CREATE TYPE federation_node_status AS ENUM (
        'discovered',   -- Found but not yet connected
        'connecting',   -- Connection in progress
        'active',       -- Connected and syncing
        'suspended',    -- Temporarily suspended
        'unreachable'   -- Cannot connect
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Trust phase (matches models.TrustPhase)
DO $$ BEGIN
    CREATE TYPE trust_phase AS ENUM (
        'observer',     -- Days 1-7: Read-only
        'contributor',  -- Days 7-30: Limited contribution
        'participant',  -- Day 30+: Full participation
        'anchor'        -- Earned: Can vouch for others
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Sync state status (matches models.SyncStatus)
DO $$ BEGIN
    CREATE TYPE sync_status AS ENUM (
        'idle',       -- No sync in progress
        'syncing',    -- Sync currently active
        'error',      -- Last sync failed
        'paused'      -- Manually paused
    );
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- ============================================================================
-- PHASE 2: FEDERATION_NODES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS federation_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Identity
    did TEXT UNIQUE NOT NULL,                    -- Decentralized Identifier (did:vkb:...)
    
    -- Connection endpoints
    federation_endpoint TEXT,                    -- Federation API endpoint URL
    mcp_endpoint TEXT,                           -- Model Context Protocol endpoint
    
    -- Cryptographic identity
    public_key_multibase TEXT NOT NULL DEFAULT '', -- Node's public key (multibase encoded)
    
    -- Profile
    name TEXT,                                   -- Human-readable name
    domains TEXT[] DEFAULT '{}',                 -- Knowledge domains this node specializes in
    
    -- Capabilities
    capabilities TEXT[] DEFAULT '{}',            -- Supported protocol features
    
    -- Status and trust phase
    status federation_node_status NOT NULL DEFAULT 'discovered',
    trust_phase trust_phase NOT NULL DEFAULT 'observer',
    phase_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Protocol
    protocol_version TEXT NOT NULL DEFAULT '1.0',
    
    -- Timestamps
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ,
    
    -- Metadata (extensible)
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Indexes for federation_nodes
CREATE INDEX IF NOT EXISTS idx_federation_nodes_did 
    ON federation_nodes(did);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_status 
    ON federation_nodes(status) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_federation_nodes_trust_phase 
    ON federation_nodes(trust_phase);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_last_seen 
    ON federation_nodes(last_seen_at DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_federation_nodes_domains 
    ON federation_nodes USING GIN(domains);

-- ============================================================================
-- PHASE 3: NODE_TRUST TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS node_trust (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Reference to peer node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    
    -- Trust dimensions stored as JSONB
    -- Contains: overall, belief_accuracy, extraction_quality, curation_accuracy,
    --           uptime_reliability, contribution_consistency, endorsement_strength,
    --           domain_expertise (nested object)
    trust JSONB NOT NULL DEFAULT '{"overall": 0.1}',
    
    -- Trust factors / statistics
    beliefs_received INTEGER NOT NULL DEFAULT 0,
    beliefs_sent INTEGER NOT NULL DEFAULT 0,
    beliefs_corroborated INTEGER NOT NULL DEFAULT 0,
    beliefs_disputed INTEGER NOT NULL DEFAULT 0,
    sync_requests_served INTEGER NOT NULL DEFAULT 0,
    aggregation_participations INTEGER NOT NULL DEFAULT 0,
    
    -- Social trust
    endorsements_received INTEGER NOT NULL DEFAULT 0,
    endorsements_given INTEGER NOT NULL DEFAULT 0,
    
    -- Relationship timeline
    relationship_started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_interaction_at TIMESTAMPTZ,
    last_sync_at TIMESTAMPTZ,
    
    -- Extended node info (from DID document)
    capabilities JSONB,                          -- Detailed capability info
    protocol_version TEXT,                       -- Peer's protocol version
    public_key_multibase TEXT,                   -- Cached public key
    profile JSONB,                               -- Cached profile info
    services JSONB,                              -- Cached service endpoints
    
    -- Manual adjustments
    manual_trust_adjustment REAL NOT NULL DEFAULT 0.0,
    adjustment_reason TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Each node has exactly one trust record
    CONSTRAINT node_trust_node_unique UNIQUE (node_id)
);

-- Indexes for node_trust
CREATE INDEX IF NOT EXISTS idx_node_trust_node 
    ON node_trust(node_id);
CREATE INDEX IF NOT EXISTS idx_node_trust_overall 
    ON node_trust(((trust->>'overall')::numeric) DESC);
CREATE INDEX IF NOT EXISTS idx_node_trust_last_interaction 
    ON node_trust(last_interaction_at DESC NULLS LAST);

-- Trigger to update modified_at
CREATE OR REPLACE FUNCTION update_node_trust_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_node_trust_modified_at ON node_trust;
CREATE TRIGGER trg_node_trust_modified_at
    BEFORE UPDATE ON node_trust
    FOR EACH ROW
    EXECUTE FUNCTION update_node_trust_modified_at();

-- ============================================================================
-- PHASE 4: SYNC_STATE TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS sync_state (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Reference to peer node
    node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
    
    -- Cursor tracking for incremental sync
    last_received_cursor TEXT,              -- Last cursor received from peer
    last_sent_cursor TEXT,                  -- Last cursor sent to peer
    
    -- Vector clock for conflict detection
    vector_clock JSONB NOT NULL DEFAULT '{}',
    
    -- Status
    status sync_status NOT NULL DEFAULT 'idle',
    
    -- Statistics
    beliefs_sent INTEGER NOT NULL DEFAULT 0,
    beliefs_received INTEGER NOT NULL DEFAULT 0,
    last_sync_duration_ms INTEGER,
    
    -- Error tracking
    last_error TEXT,
    error_count INTEGER NOT NULL DEFAULT 0,
    
    -- Scheduling
    last_sync_at TIMESTAMPTZ,
    next_sync_scheduled TIMESTAMPTZ,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Each node has exactly one sync state record
    CONSTRAINT sync_state_node_unique UNIQUE (node_id)
);

-- Indexes for sync_state
CREATE INDEX IF NOT EXISTS idx_sync_state_node 
    ON sync_state(node_id);
CREATE INDEX IF NOT EXISTS idx_sync_state_status 
    ON sync_state(status);
CREATE INDEX IF NOT EXISTS idx_sync_state_next_sync 
    ON sync_state(next_sync_scheduled ASC NULLS LAST) 
    WHERE status != 'error';
CREATE INDEX IF NOT EXISTS idx_sync_state_errors 
    ON sync_state(error_count DESC) 
    WHERE error_count > 0;

-- Trigger to update modified_at
CREATE OR REPLACE FUNCTION update_sync_state_modified_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.modified_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sync_state_modified_at ON sync_state;
CREATE TRIGGER trg_sync_state_modified_at
    BEFORE UPDATE ON sync_state
    FOR EACH ROW
    EXECUTE FUNCTION update_sync_state_modified_at();

-- ============================================================================
-- PHASE 5: ADD FEDERATION COLUMNS TO BELIEFS
-- ============================================================================

-- Flag for locally-originated beliefs
ALTER TABLE beliefs 
    ADD COLUMN IF NOT EXISTS is_local BOOLEAN NOT NULL DEFAULT TRUE;

-- Reference to origin node for federated beliefs
ALTER TABLE beliefs 
    ADD COLUMN IF NOT EXISTS origin_node_id UUID REFERENCES federation_nodes(id) ON DELETE SET NULL;

-- Index for finding federated beliefs
CREATE INDEX IF NOT EXISTS idx_beliefs_is_local 
    ON beliefs(is_local) WHERE is_local = FALSE;
CREATE INDEX IF NOT EXISTS idx_beliefs_origin_node 
    ON beliefs(origin_node_id) WHERE origin_node_id IS NOT NULL;

-- ============================================================================
-- PHASE 6: HELPER FUNCTIONS
-- ============================================================================

-- Get active federation nodes for sync
CREATE OR REPLACE FUNCTION get_active_federation_nodes()
RETURNS TABLE (
    id UUID,
    did TEXT,
    name TEXT,
    federation_endpoint TEXT,
    trust_phase trust_phase,
    last_seen_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fn.id,
        fn.did,
        fn.name,
        fn.federation_endpoint,
        fn.trust_phase,
        fn.last_seen_at
    FROM federation_nodes fn
    WHERE fn.status = 'active'
    ORDER BY fn.last_seen_at DESC NULLS LAST;
END;
$$ LANGUAGE plpgsql STABLE;

-- Get nodes due for sync
CREATE OR REPLACE FUNCTION get_nodes_due_for_sync()
RETURNS TABLE (
    node_id UUID,
    did TEXT,
    federation_endpoint TEXT,
    last_sync_at TIMESTAMPTZ,
    next_sync_scheduled TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        fn.id,
        fn.did,
        fn.federation_endpoint,
        ss.last_sync_at,
        ss.next_sync_scheduled
    FROM federation_nodes fn
    JOIN sync_state ss ON ss.node_id = fn.id
    WHERE fn.status = 'active'
      AND ss.status != 'syncing'
      AND (ss.next_sync_scheduled IS NULL OR ss.next_sync_scheduled <= NOW())
    ORDER BY ss.next_sync_scheduled ASC NULLS FIRST;
END;
$$ LANGUAGE plpgsql STABLE;

-- Record sync start
CREATE OR REPLACE FUNCTION sync_start(p_node_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE sync_state
    SET status = 'syncing',
        modified_at = NOW()
    WHERE node_id = p_node_id;
    
    -- Create sync_state if doesn't exist
    INSERT INTO sync_state (node_id, status)
    VALUES (p_node_id, 'syncing')
    ON CONFLICT (node_id) DO UPDATE
    SET status = 'syncing', modified_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Record sync completion
CREATE OR REPLACE FUNCTION sync_complete(
    p_node_id UUID,
    p_sent INTEGER,
    p_received INTEGER,
    p_duration_ms INTEGER,
    p_sent_cursor TEXT,
    p_received_cursor TEXT,
    p_next_sync TIMESTAMPTZ DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    UPDATE sync_state
    SET status = 'idle',
        beliefs_sent = beliefs_sent + p_sent,
        beliefs_received = beliefs_received + p_received,
        last_sync_duration_ms = p_duration_ms,
        last_sent_cursor = COALESCE(p_sent_cursor, last_sent_cursor),
        last_received_cursor = COALESCE(p_received_cursor, last_received_cursor),
        last_sync_at = NOW(),
        next_sync_scheduled = p_next_sync,
        last_error = NULL,
        error_count = 0,
        modified_at = NOW()
    WHERE node_id = p_node_id;
    
    -- Update node last_seen_at
    UPDATE federation_nodes
    SET last_seen_at = NOW(),
        last_sync_at = NOW()
    WHERE id = p_node_id;
    
    -- Update node_trust last_sync_at
    UPDATE node_trust
    SET last_sync_at = NOW(),
        last_interaction_at = NOW()
    WHERE node_id = p_node_id;
END;
$$ LANGUAGE plpgsql;

-- Record sync error
CREATE OR REPLACE FUNCTION sync_error(
    p_node_id UUID,
    p_error TEXT,
    p_retry_at TIMESTAMPTZ DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    UPDATE sync_state
    SET status = 'error',
        last_error = p_error,
        error_count = error_count + 1,
        next_sync_scheduled = COALESCE(p_retry_at, NOW() + INTERVAL '5 minutes' * POWER(2, LEAST(error_count, 6))),
        modified_at = NOW()
    WHERE node_id = p_node_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PHASE 7: COMMENTS
-- ============================================================================

COMMENT ON TABLE federation_nodes IS 'Known peer nodes in the federation network';
COMMENT ON COLUMN federation_nodes.did IS 'Decentralized Identifier (DID) for the node';
COMMENT ON COLUMN federation_nodes.federation_endpoint IS 'URL for the federation protocol endpoint';
COMMENT ON COLUMN federation_nodes.mcp_endpoint IS 'URL for the Model Context Protocol endpoint';
COMMENT ON COLUMN federation_nodes.public_key_multibase IS 'Public key in multibase encoding for signature verification';
COMMENT ON COLUMN federation_nodes.domains IS 'Knowledge domains this node specializes in';
COMMENT ON COLUMN federation_nodes.trust_phase IS 'Current trust establishment phase';
COMMENT ON COLUMN federation_nodes.capabilities IS 'Protocol features supported by this node';

COMMENT ON TABLE node_trust IS 'Trust dimensions and statistics for federation peers';
COMMENT ON COLUMN node_trust.trust IS 'JSONB containing trust dimensions (overall, belief_accuracy, etc.)';
COMMENT ON COLUMN node_trust.beliefs_received IS 'Total beliefs received from this node';
COMMENT ON COLUMN node_trust.beliefs_corroborated IS 'Beliefs from this node that were corroborated';
COMMENT ON COLUMN node_trust.beliefs_disputed IS 'Beliefs from this node that were disputed';
COMMENT ON COLUMN node_trust.relationship_started_at IS 'When we first connected to this node';

COMMENT ON TABLE sync_state IS 'Tracks synchronization state with each federation peer';
COMMENT ON COLUMN sync_state.vector_clock IS 'Lamport vector clock for causal ordering';
COMMENT ON COLUMN sync_state.last_received_cursor IS 'Cursor for incremental pull from peer';
COMMENT ON COLUMN sync_state.last_sent_cursor IS 'Cursor for incremental push to peer';

COMMENT ON COLUMN beliefs.is_local IS 'TRUE if belief originated locally, FALSE if received via federation';
COMMENT ON COLUMN beliefs.origin_node_id IS 'Source federation node for beliefs received via sync';
