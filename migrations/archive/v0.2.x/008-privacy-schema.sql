-- Privacy schema for Valence
-- Adds share_policy to beliefs and creates consent_chains table
-- Closes #45, #46, #47

-- Add share_policy to beliefs
ALTER TABLE beliefs 
ADD COLUMN IF NOT EXISTS share_policy JSONB DEFAULT '{"level": "private", "enforcement": "policy"}';

-- Index for querying beliefs by share level (closes #49)
CREATE INDEX IF NOT EXISTS idx_beliefs_share_level 
ON beliefs ((share_policy->>'level'));

-- GIN index for flexible share_policy queries
CREATE INDEX IF NOT EXISTS idx_beliefs_share_policy 
ON beliefs USING GIN (share_policy);

-- Create consent_chains table for tracking sharing provenance
CREATE TABLE IF NOT EXISTS consent_chains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    
    -- Origin of the share
    origin_sharer TEXT NOT NULL,  -- DID of original sharer
    origin_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    origin_policy JSONB NOT NULL,
    origin_signature BYTEA NOT NULL,
    
    -- Chain of hops (each hop is {sharer, timestamp, policy_at_hop, signature})
    hops JSONB[] DEFAULT ARRAY[]::JSONB[],
    
    -- Integrity - hash of entire chain for tamper detection
    chain_hash BYTEA NOT NULL,
    
    -- Status
    revoked BOOLEAN DEFAULT FALSE,
    revoked_at TIMESTAMPTZ,
    revoked_by TEXT,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_consent_chains_belief ON consent_chains(belief_id);
CREATE INDEX IF NOT EXISTS idx_consent_chains_sharer ON consent_chains(origin_sharer);
CREATE INDEX IF NOT EXISTS idx_consent_chains_revoked ON consent_chains(revoked) WHERE revoked = false;

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_consent_chains_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS consent_chains_updated ON consent_chains;
CREATE TRIGGER consent_chains_updated
    BEFORE UPDATE ON consent_chains
    FOR EACH ROW
    EXECUTE FUNCTION update_consent_chains_timestamp();

-- Add comment for documentation
COMMENT ON TABLE consent_chains IS 'Tracks the provenance chain of shared beliefs, enabling consent verification and revocation';
COMMENT ON COLUMN consent_chains.hops IS 'Array of hop objects: {sharer: DID, timestamp: ISO8601, policy_at_hop: SharePolicy, signature: base64}';
COMMENT ON COLUMN consent_chains.chain_hash IS 'SHA-256 hash of concatenated hop signatures for tamper detection';
