-- Trust Graph Schema for Valence Privacy
-- Stores trust relationships between DIDs with multi-dimensional trust scores
-- Closes #58

-- Trust edges table
-- Represents directed trust from source_did to target_did
CREATE TABLE IF NOT EXISTS trust_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Trust relationship endpoints (DIDs)
    source_did TEXT NOT NULL,  -- The truster
    target_did TEXT NOT NULL,  -- The trusted
    
    -- Multi-dimensional trust scores (0.0 to 1.0)
    -- Based on Valence trust model dimensions
    competence REAL NOT NULL DEFAULT 0.5,       -- Ability to perform tasks correctly
    integrity REAL NOT NULL DEFAULT 0.5,        -- Honesty and consistency
    confidentiality REAL NOT NULL DEFAULT 0.5,  -- Ability to keep secrets
    judgment REAL NOT NULL DEFAULT 0.5,         -- Quality of decisions/recommendations
    
    -- Optional domain scoping (NULL = general trust)
    domain TEXT,
    
    -- Delegation settings for transitive trust
    can_delegate BOOLEAN NOT NULL DEFAULT FALSE,  -- Whether trust can be transitively delegated
    delegation_depth INTEGER NOT NULL DEFAULT 0,  -- Max delegation chain length (0 = no limit)
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,  -- Optional expiration
    
    -- Constraints
    CONSTRAINT trust_scores_valid CHECK (
        competence >= 0.0 AND competence <= 1.0 AND
        integrity >= 0.0 AND integrity <= 1.0 AND
        confidentiality >= 0.0 AND confidentiality <= 1.0 AND
        judgment >= 0.0 AND judgment <= 1.0
    ),
    
    -- Cannot trust yourself
    CONSTRAINT no_self_trust CHECK (source_did != target_did)
);

-- Composite unique constraint: one edge per (source, target, domain) tuple
-- NULL domain is treated as distinct (general trust)
CREATE UNIQUE INDEX IF NOT EXISTS idx_trust_edges_unique 
ON trust_edges (source_did, target_did, COALESCE(domain, ''));

-- Index for finding all edges FROM a given DID (who do I trust?)
CREATE INDEX IF NOT EXISTS idx_trust_edges_source 
ON trust_edges (source_did);

-- Index for finding all edges TO a given DID (who trusts me?)
CREATE INDEX IF NOT EXISTS idx_trust_edges_target 
ON trust_edges (target_did);

-- Index for domain-scoped queries
CREATE INDEX IF NOT EXISTS idx_trust_edges_domain 
ON trust_edges (domain) WHERE domain IS NOT NULL;

-- Composite index for common graph traversal patterns
CREATE INDEX IF NOT EXISTS idx_trust_edges_source_domain 
ON trust_edges (source_did, domain);

-- Index for expiration cleanup
CREATE INDEX IF NOT EXISTS idx_trust_edges_expires 
ON trust_edges (expires_at) WHERE expires_at IS NOT NULL;

-- Trigger for updated_at
CREATE OR REPLACE FUNCTION update_trust_edges_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trust_edges_updated ON trust_edges;
CREATE TRIGGER trust_edges_updated
    BEFORE UPDATE ON trust_edges
    FOR EACH ROW
    EXECUTE FUNCTION update_trust_edges_timestamp();

-- Documentation
COMMENT ON TABLE trust_edges IS 'Directed trust graph edges between DIDs with multi-dimensional trust scores';
COMMENT ON COLUMN trust_edges.source_did IS 'DID of the entity expressing trust';
COMMENT ON COLUMN trust_edges.target_did IS 'DID of the entity being trusted';
COMMENT ON COLUMN trust_edges.competence IS 'Trust in ability to perform tasks (0-1)';
COMMENT ON COLUMN trust_edges.integrity IS 'Trust in honesty and consistency (0-1)';
COMMENT ON COLUMN trust_edges.confidentiality IS 'Trust in ability to keep information private (0-1)';
COMMENT ON COLUMN trust_edges.judgment IS 'Trust in quality of decisions and recommendations (0-1)';
COMMENT ON COLUMN trust_edges.domain IS 'Optional domain scope (NULL for general trust)';
COMMENT ON COLUMN trust_edges.expires_at IS 'Optional expiration timestamp for time-limited trust';
