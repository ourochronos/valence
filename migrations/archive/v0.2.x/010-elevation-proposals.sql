-- Elevation proposals schema for Valence
-- Adds elevation_proposals and elevation_history tables
-- Closes #94

-- Create elevation_proposals table
CREATE TABLE IF NOT EXISTS elevation_proposals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    
    -- Participants
    owner_did TEXT NOT NULL,      -- DID of content owner
    proposer_did TEXT NOT NULL,   -- DID of proposer
    
    -- Elevation details
    from_level TEXT NOT NULL,     -- Current ShareLevel value
    to_level TEXT NOT NULL,       -- Proposed ShareLevel value
    
    -- Status
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, approved, rejected, withdrawn, expired
    
    -- Timestamps
    proposed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    
    -- Decision info
    decision_reason TEXT,
    decided_by TEXT,
    
    -- Transform for elevation
    transform JSONB NOT NULL DEFAULT '{"transform_type": "none"}',
    
    -- Result
    elevated_belief_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,
    
    -- Proposal metadata
    justification TEXT,
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('pending', 'approved', 'rejected', 'withdrawn', 'expired')),
    CONSTRAINT valid_levels CHECK (
        from_level IN ('private', 'direct', 'bounded', 'cascading', 'public') AND
        to_level IN ('private', 'direct', 'bounded', 'cascading', 'public')
    )
);

-- Create elevation_history table for tracking completed elevations
CREATE TABLE IF NOT EXISTS elevation_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id UUID NOT NULL,  -- May reference deleted belief, so no FK
    proposal_id UUID NOT NULL REFERENCES elevation_proposals(id) ON DELETE CASCADE,
    
    -- Elevation details
    from_level TEXT NOT NULL,
    to_level TEXT NOT NULL,
    
    -- Participants
    proposer_did TEXT NOT NULL,
    approver_did TEXT NOT NULL,
    
    -- Timestamps
    proposed_at TIMESTAMPTZ NOT NULL,
    approved_at TIMESTAMPTZ NOT NULL,
    
    -- Transform applied
    transform JSONB NOT NULL,
    
    -- Links
    original_belief_id UUID NOT NULL,
    elevated_belief_id UUID NOT NULL,
    
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_elevation_proposals_belief ON elevation_proposals(belief_id);
CREATE INDEX IF NOT EXISTS idx_elevation_proposals_owner ON elevation_proposals(owner_did);
CREATE INDEX IF NOT EXISTS idx_elevation_proposals_proposer ON elevation_proposals(proposer_did);
CREATE INDEX IF NOT EXISTS idx_elevation_proposals_status ON elevation_proposals(status);
CREATE INDEX IF NOT EXISTS idx_elevation_proposals_pending ON elevation_proposals(status, expires_at) 
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_elevation_history_belief ON elevation_history(belief_id);
CREATE INDEX IF NOT EXISTS idx_elevation_history_original ON elevation_history(original_belief_id);
CREATE INDEX IF NOT EXISTS idx_elevation_history_elevated ON elevation_history(elevated_belief_id);

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_elevation_proposals_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS elevation_proposals_updated ON elevation_proposals;
CREATE TRIGGER elevation_proposals_updated
    BEFORE UPDATE ON elevation_proposals
    FOR EACH ROW
    EXECUTE FUNCTION update_elevation_proposals_timestamp();

-- Comments for documentation
COMMENT ON TABLE elevation_proposals IS 'Tracks proposals to elevate content from private to more public levels';
COMMENT ON COLUMN elevation_proposals.from_level IS 'Current ShareLevel: private, direct, bounded, cascading, public';
COMMENT ON COLUMN elevation_proposals.to_level IS 'Proposed ShareLevel - must be more public than from_level';
COMMENT ON COLUMN elevation_proposals.transform IS 'Transformation to apply: {transform_type, transformed_content, redactions, transform_metadata}';
COMMENT ON TABLE elevation_history IS 'Audit trail of completed elevations - who proposed, when approved, what transform applied';
