-- Migration 003: Add unique constraints for federation IDs
-- Fixes: #235 (TOCTOU race), #237 (missing unique constraint on belief_provenance.federation_id)
BEGIN;

-- Add unique constraint on beliefs.federation_id (where not null)
-- This enables ON CONFLICT handling for atomic upserts
CREATE UNIQUE INDEX IF NOT EXISTS idx_beliefs_federation_unique
ON beliefs(federation_id) WHERE federation_id IS NOT NULL;

-- Add unique constraint on belief_provenance.federation_id
-- Ensures each federated belief has exactly one provenance record
CREATE UNIQUE INDEX IF NOT EXISTS idx_belief_provenance_federation_unique
ON belief_provenance(federation_id);

COMMIT;
