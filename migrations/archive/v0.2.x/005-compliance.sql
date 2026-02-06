-- Migration 005: Compliance infrastructure
-- Implements GDPR Article 17 compliance and PII tracking
-- Reference: spec/compliance/COMPLIANCE.md §3

-- ============================================================================
-- Tombstones: Deletion records for federation propagation
-- ============================================================================

CREATE TABLE IF NOT EXISTS tombstones (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- What was deleted
    target_type VARCHAR(50) NOT NULL,  -- 'belief', 'aggregate', 'membership', 'user'
    target_id UUID NOT NULL,
    
    -- Who and when
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(64) NOT NULL,  -- Hashed user ID for privacy
    
    -- Legal basis
    reason VARCHAR(50) NOT NULL,  -- DeletionReason enum value
    legal_basis TEXT,
    
    -- Cryptographic erasure tracking
    encryption_key_revoked BOOLEAN DEFAULT FALSE,
    key_revocation_timestamp TIMESTAMPTZ,
    
    -- Federation propagation tracking
    propagation_started TIMESTAMPTZ,
    acknowledged_by JSONB DEFAULT '{}',  -- {peer_did: timestamp}
    
    -- Verification
    signature BYTEA,
    
    -- Audit index
    CONSTRAINT tombstone_reason_valid CHECK (
        reason IN (
            'user_request', 'consent_withdrawal', 'legal_order',
            'policy_violation', 'data_accuracy', 'security_incident'
        )
    )
);

-- Index for federation propagation queries
CREATE INDEX IF NOT EXISTS idx_tombstones_propagation 
    ON tombstones (propagation_started) 
    WHERE propagation_started IS NOT NULL;

-- Index for audit queries
CREATE INDEX IF NOT EXISTS idx_tombstones_created_by 
    ON tombstones (created_by, created_at);

-- Index for target lookup
CREATE INDEX IF NOT EXISTS idx_tombstones_target 
    ON tombstones (target_type, target_id);

-- ============================================================================
-- Belief federation opt-out tracking
-- ============================================================================

-- Add opt_out_federation flag to beliefs
ALTER TABLE beliefs 
    ADD COLUMN IF NOT EXISTS opt_out_federation BOOLEAN DEFAULT FALSE;

-- Add PII scan result tracking
ALTER TABLE beliefs 
    ADD COLUMN IF NOT EXISTS pii_classification SMALLINT DEFAULT 0;

-- Index for federation queries (exclude opted-out beliefs)
CREATE INDEX IF NOT EXISTS idx_beliefs_federation 
    ON beliefs (status, opt_out_federation) 
    WHERE status = 'active' AND opt_out_federation = FALSE;

-- ============================================================================
-- Audit log for compliance
-- ============================================================================

CREATE TABLE IF NOT EXISTS compliance_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    sequence BIGSERIAL NOT NULL,
    
    -- Event details
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    actor_hash VARCHAR(64) NOT NULL,  -- Privacy-preserving actor ID
    
    -- Encrypted event details (for sensitive events)
    details JSONB,
    details_encrypted BYTEA,
    
    -- Tamper-evident chain
    previous_hash BYTEA,
    entry_hash BYTEA GENERATED ALWAYS AS (
        sha256(
            COALESCE(previous_hash, ''::bytea) || 
            sequence::text::bytea || 
            timestamp::text::bytea ||
            event_type::bytea
        )
    ) STORED,
    
    -- Signature for verification
    signature BYTEA
);

-- Index for audit queries
CREATE INDEX IF NOT EXISTS idx_audit_log_type 
    ON compliance_audit_log (event_type, timestamp);

CREATE INDEX IF NOT EXISTS idx_audit_log_actor 
    ON compliance_audit_log (actor_hash, timestamp);

-- ============================================================================
-- Consent records
-- ============================================================================

CREATE TABLE IF NOT EXISTS consent_records (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Who consented
    subject_id VARCHAR(256) NOT NULL,  -- DID or user identifier
    subject_hash VARCHAR(64) NOT NULL, -- Hashed for privacy queries
    
    -- What was consented to
    consent_type VARCHAR(50) NOT NULL,
    scope VARCHAR(256) NOT NULL,  -- Federation ID, belief ID, etc.
    
    -- When and how
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address_hash VARCHAR(64),
    user_agent_hash VARCHAR(64),
    
    -- Consent text version (for compliance)
    consent_text_version VARCHAR(20) NOT NULL,
    explicit_acknowledgments TEXT[],
    
    -- Revocation
    revoked_at TIMESTAMPTZ,
    revocation_reason TEXT,
    
    CONSTRAINT consent_type_valid CHECK (
        consent_type IN (
            'federation_membership', 'sharing_consent', 
            'processing_consent', 'cross_federation'
        )
    )
);

-- Index for subject queries
CREATE INDEX IF NOT EXISTS idx_consent_subject 
    ON consent_records (subject_hash, consent_type);

-- Index for active consents
CREATE INDEX IF NOT EXISTS idx_consent_active 
    ON consent_records (subject_hash, consent_type) 
    WHERE revoked_at IS NULL;

-- ============================================================================
-- Functions for compliance operations
-- ============================================================================

-- Function to record audit events with hash chain
CREATE OR REPLACE FUNCTION record_audit_event(
    p_event_type VARCHAR(50),
    p_actor_hash VARCHAR(64),
    p_details JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    v_previous_hash BYTEA;
    v_new_id UUID;
BEGIN
    -- Get previous entry hash for chain
    SELECT entry_hash INTO v_previous_hash
    FROM compliance_audit_log
    ORDER BY sequence DESC
    LIMIT 1;
    
    -- Insert new entry
    INSERT INTO compliance_audit_log (
        event_type, actor_hash, details, previous_hash
    ) VALUES (
        p_event_type, p_actor_hash, p_details, v_previous_hash
    ) RETURNING id INTO v_new_id;
    
    RETURN v_new_id;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Comments for documentation
-- ============================================================================

COMMENT ON TABLE tombstones IS 
    'Deletion records for GDPR compliance and federation propagation. '
    'Retained for 7 years per COMPLIANCE.md §7.';

COMMENT ON TABLE compliance_audit_log IS 
    'Tamper-evident audit log for compliance verification. '
    'Hash chain ensures integrity. Retained per COMPLIANCE.md §7.';

COMMENT ON TABLE consent_records IS 
    'Consent records for GDPR compliance. '
    'Tracks consent grants and revocations with full history.';

COMMENT ON COLUMN beliefs.opt_out_federation IS 
    'If TRUE, belief will not be shared via federation (Issue #26).';

COMMENT ON COLUMN beliefs.pii_classification IS 
    'PII classification level (0-4) from automated scanning (Issue #35).';
