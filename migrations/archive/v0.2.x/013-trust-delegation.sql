-- Trust Delegation Policy for Valence Privacy
-- Adds delegation control to trust edges
-- Closes #68

-- Add delegation policy columns to trust_edges
ALTER TABLE trust_edges
ADD COLUMN IF NOT EXISTS can_delegate BOOLEAN NOT NULL DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS delegation_depth INTEGER NOT NULL DEFAULT 0;

-- Constraint for delegation_depth (must be non-negative)
ALTER TABLE trust_edges
ADD CONSTRAINT delegation_depth_valid CHECK (delegation_depth >= 0);

-- Index for finding delegatable edges (useful for transitive trust queries)
CREATE INDEX IF NOT EXISTS idx_trust_edges_delegatable
ON trust_edges (source_did, can_delegate) WHERE can_delegate = TRUE;

-- Documentation
COMMENT ON COLUMN trust_edges.can_delegate IS 'Whether this trust can be transitively delegated (default FALSE = non-transitive)';
COMMENT ON COLUMN trust_edges.delegation_depth IS 'Maximum delegation chain length (0 = no limit when can_delegate=TRUE)';
