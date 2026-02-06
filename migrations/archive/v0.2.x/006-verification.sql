-- Migration: Verification Protocol with Staking
-- Implements adversarial verification system per spec/components/verification-protocol/

-- ============================================================================
-- Enum Types
-- ============================================================================

CREATE TYPE verification_result AS ENUM (
    'confirmed',
    'contradicted', 
    'uncertain',
    'partial'
);

CREATE TYPE verification_status AS ENUM (
    'pending',
    'accepted',
    'disputed',
    'overturned',
    'rejected',
    'expired'
);

CREATE TYPE stake_type AS ENUM (
    'standard',
    'bounty',
    'challenge'
);

CREATE TYPE evidence_type AS ENUM (
    'belief',
    'external',
    'observation',
    'derivation',
    'testimony'
);

CREATE TYPE evidence_contribution AS ENUM (
    'supports',
    'contradicts',
    'context',
    'qualifies'
);

CREATE TYPE dispute_type AS ENUM (
    'evidence_invalid',
    'evidence_fabricated',
    'evidence_insufficient',
    'reasoning_flawed',
    'conflict_of_interest',
    'new_evidence'
);

CREATE TYPE dispute_outcome AS ENUM (
    'upheld',
    'overturned',
    'modified',
    'dismissed'
);

CREATE TYPE dispute_status AS ENUM (
    'pending',
    'resolved',
    'expired'
);

CREATE TYPE resolution_method AS ENUM (
    'automatic',
    'jury',
    'expert',
    'appeal'
);

-- ============================================================================
-- Reputation Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS reputations (
    identity_id TEXT PRIMARY KEY,  -- DID
    overall NUMERIC(5,4) NOT NULL DEFAULT 0.5 CHECK (overall >= 0.1 AND overall <= 1.0),
    by_domain JSONB NOT NULL DEFAULT '{}',
    verification_count INTEGER NOT NULL DEFAULT 0,
    discrepancy_finds INTEGER NOT NULL DEFAULT 0,
    stake_at_risk NUMERIC(8,6) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_reputations_overall ON reputations(overall DESC);
CREATE INDEX idx_reputations_modified ON reputations(modified_at DESC);

-- ============================================================================
-- Verifications Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS verifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verifier_id TEXT NOT NULL,  -- DID
    belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
    holder_id TEXT NOT NULL,  -- DID (cached)
    result verification_result NOT NULL,
    evidence JSONB NOT NULL DEFAULT '[]',
    stake JSONB NOT NULL,  -- {amount, type, locked_until, escrow_id}
    reasoning TEXT,
    result_details JSONB,
    status verification_status NOT NULL DEFAULT 'pending',
    dispute_id UUID,
    signature BYTEA,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    accepted_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT verifier_not_holder CHECK (verifier_id != holder_id),
    CONSTRAINT valid_stake CHECK (
        stake->>'amount' IS NOT NULL AND
        (stake->>'amount')::numeric >= 0
    )
);

CREATE INDEX idx_verifications_belief ON verifications(belief_id);
CREATE INDEX idx_verifications_verifier ON verifications(verifier_id);
CREATE INDEX idx_verifications_holder ON verifications(holder_id);
CREATE INDEX idx_verifications_status ON verifications(status);
CREATE INDEX idx_verifications_result ON verifications(result);
CREATE INDEX idx_verifications_created ON verifications(created_at DESC);

-- Unique constraint: one verification per verifier per belief
CREATE UNIQUE INDEX idx_verifications_unique_verifier_belief 
    ON verifications(verifier_id, belief_id) 
    WHERE status NOT IN ('rejected', 'expired');

-- ============================================================================
-- Disputes Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS disputes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    verification_id UUID NOT NULL REFERENCES verifications(id) ON DELETE CASCADE,
    disputer_id TEXT NOT NULL,  -- DID
    counter_evidence JSONB NOT NULL DEFAULT '[]',
    stake JSONB NOT NULL,
    dispute_type dispute_type NOT NULL,
    reasoning TEXT NOT NULL,
    proposed_result verification_result,
    status dispute_status NOT NULL DEFAULT 'pending',
    outcome dispute_outcome,
    resolution_reasoning TEXT,
    resolution_method resolution_method,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    
    CONSTRAINT valid_dispute_stake CHECK (
        stake->>'amount' IS NOT NULL AND
        (stake->>'amount')::numeric >= 0
    )
);

CREATE INDEX idx_disputes_verification ON disputes(verification_id);
CREATE INDEX idx_disputes_disputer ON disputes(disputer_id);
CREATE INDEX idx_disputes_status ON disputes(status);
CREATE INDEX idx_disputes_created ON disputes(created_at DESC);

-- Add foreign key from verifications to disputes
ALTER TABLE verifications 
    ADD CONSTRAINT fk_verification_dispute 
    FOREIGN KEY (dispute_id) REFERENCES disputes(id);

-- ============================================================================
-- Stake Positions Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS stake_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_id TEXT NOT NULL REFERENCES reputations(identity_id),
    amount NUMERIC(8,6) NOT NULL CHECK (amount >= 0),
    type stake_type NOT NULL,
    verification_id UUID REFERENCES verifications(id),
    dispute_id UUID REFERENCES disputes(id),
    locked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    unlocks_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'locked' CHECK (status IN ('locked', 'pending_return', 'forfeited', 'returned'))
);

CREATE INDEX idx_stake_positions_identity ON stake_positions(identity_id);
CREATE INDEX idx_stake_positions_verification ON stake_positions(verification_id);
CREATE INDEX idx_stake_positions_dispute ON stake_positions(dispute_id);
CREATE INDEX idx_stake_positions_status ON stake_positions(status);
CREATE INDEX idx_stake_positions_unlocks ON stake_positions(unlocks_at) WHERE status = 'locked';

-- ============================================================================
-- Reputation Events Table (Audit Trail)
-- ============================================================================

CREATE TABLE IF NOT EXISTS reputation_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    identity_id TEXT NOT NULL REFERENCES reputations(identity_id),
    delta NUMERIC(8,6) NOT NULL,
    old_value NUMERIC(5,4) NOT NULL,
    new_value NUMERIC(5,4) NOT NULL,
    reason TEXT NOT NULL,
    dimension TEXT NOT NULL DEFAULT 'overall',
    verification_id UUID REFERENCES verifications(id),
    dispute_id UUID REFERENCES disputes(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_reputation_events_identity ON reputation_events(identity_id);
CREATE INDEX idx_reputation_events_created ON reputation_events(created_at DESC);
CREATE INDEX idx_reputation_events_verification ON reputation_events(verification_id);
CREATE INDEX idx_reputation_events_dispute ON reputation_events(dispute_id);

-- ============================================================================
-- Discrepancy Bounties Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS discrepancy_bounties (
    belief_id UUID PRIMARY KEY REFERENCES beliefs(id) ON DELETE CASCADE,
    holder_id TEXT NOT NULL,
    base_amount NUMERIC(8,6) NOT NULL,
    confidence_premium NUMERIC(5,4) NOT NULL,
    age_factor NUMERIC(4,2) NOT NULL DEFAULT 1.0,
    total_bounty NUMERIC(8,6) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    claimed BOOLEAN NOT NULL DEFAULT FALSE,
    claimed_by TEXT,
    claimed_at TIMESTAMPTZ
);

CREATE INDEX idx_bounties_holder ON discrepancy_bounties(holder_id);
CREATE INDEX idx_bounties_total ON discrepancy_bounties(total_bounty DESC) WHERE NOT claimed;
CREATE INDEX idx_bounties_expires ON discrepancy_bounties(expires_at) WHERE NOT claimed AND expires_at IS NOT NULL;

-- ============================================================================
-- Functions
-- ============================================================================

-- Function to get or create reputation
CREATE OR REPLACE FUNCTION get_or_create_reputation(p_identity_id TEXT)
RETURNS reputations AS $$
DECLARE
    v_rep reputations;
BEGIN
    SELECT * INTO v_rep FROM reputations WHERE identity_id = p_identity_id;
    
    IF NOT FOUND THEN
        INSERT INTO reputations (identity_id)
        VALUES (p_identity_id)
        RETURNING * INTO v_rep;
    END IF;
    
    RETURN v_rep;
END;
$$ LANGUAGE plpgsql;

-- Function to lock stake
CREATE OR REPLACE FUNCTION lock_stake(
    p_identity_id TEXT,
    p_amount NUMERIC,
    p_type stake_type,
    p_verification_id UUID DEFAULT NULL,
    p_dispute_id UUID DEFAULT NULL,
    p_lockup_days INTEGER DEFAULT 7
)
RETURNS UUID AS $$
DECLARE
    v_position_id UUID;
    v_available NUMERIC;
BEGIN
    -- Get or create reputation
    PERFORM get_or_create_reputation(p_identity_id);
    
    -- Check available stake
    SELECT (overall * 0.2) - stake_at_risk INTO v_available
    FROM reputations WHERE identity_id = p_identity_id;
    
    IF v_available < p_amount THEN
        RAISE EXCEPTION 'Insufficient available reputation for stake. Available: %, Requested: %', v_available, p_amount;
    END IF;
    
    -- Create stake position
    INSERT INTO stake_positions (
        identity_id, amount, type, verification_id, dispute_id, 
        unlocks_at
    )
    VALUES (
        p_identity_id, p_amount, p_type, p_verification_id, p_dispute_id,
        NOW() + (p_lockup_days || ' days')::INTERVAL
    )
    RETURNING id INTO v_position_id;
    
    -- Update reputation stake_at_risk
    UPDATE reputations
    SET stake_at_risk = stake_at_risk + p_amount,
        modified_at = NOW()
    WHERE identity_id = p_identity_id;
    
    RETURN v_position_id;
END;
$$ LANGUAGE plpgsql;

-- Function to release stake
CREATE OR REPLACE FUNCTION release_stake(
    p_identity_id TEXT,
    p_verification_id UUID DEFAULT NULL,
    p_dispute_id UUID DEFAULT NULL,
    p_forfeit BOOLEAN DEFAULT FALSE
)
RETURNS NUMERIC AS $$
DECLARE
    v_amount NUMERIC := 0;
    v_position stake_positions;
BEGIN
    FOR v_position IN 
        SELECT * FROM stake_positions 
        WHERE identity_id = p_identity_id
          AND status = 'locked'
          AND (p_verification_id IS NULL OR verification_id = p_verification_id)
          AND (p_dispute_id IS NULL OR dispute_id = p_dispute_id)
    LOOP
        -- Update position status
        UPDATE stake_positions
        SET status = CASE WHEN p_forfeit THEN 'forfeited' ELSE 'returned' END
        WHERE id = v_position.id;
        
        v_amount := v_amount + v_position.amount;
    END LOOP;
    
    -- Update reputation
    IF v_amount > 0 THEN
        UPDATE reputations
        SET stake_at_risk = stake_at_risk - v_amount,
            modified_at = NOW()
        WHERE identity_id = p_identity_id;
    END IF;
    
    RETURN v_amount;
END;
$$ LANGUAGE plpgsql;

-- Function to apply reputation update
CREATE OR REPLACE FUNCTION apply_reputation_update(
    p_identity_id TEXT,
    p_delta NUMERIC,
    p_reason TEXT,
    p_dimension TEXT DEFAULT 'overall',
    p_verification_id UUID DEFAULT NULL,
    p_dispute_id UUID DEFAULT NULL
)
RETURNS reputation_events AS $$
DECLARE
    v_old NUMERIC;
    v_new NUMERIC;
    v_event reputation_events;
BEGIN
    -- Get current value
    SELECT overall INTO v_old FROM reputations WHERE identity_id = p_identity_id;
    IF NOT FOUND THEN
        PERFORM get_or_create_reputation(p_identity_id);
        v_old := 0.5;
    END IF;
    
    -- Calculate new value with bounds
    v_new := GREATEST(0.1, LEAST(1.0, v_old + p_delta));
    
    -- Update reputation
    UPDATE reputations
    SET overall = v_new,
        modified_at = NOW()
    WHERE identity_id = p_identity_id;
    
    -- Log event
    INSERT INTO reputation_events (
        identity_id, delta, old_value, new_value, reason, dimension,
        verification_id, dispute_id
    )
    VALUES (
        p_identity_id, p_delta, v_old, v_new, p_reason, p_dimension,
        p_verification_id, p_dispute_id
    )
    RETURNING * INTO v_event;
    
    RETURN v_event;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate bounty for a belief
CREATE OR REPLACE FUNCTION calculate_belief_bounty(p_belief_id UUID)
RETURNS NUMERIC AS $$
DECLARE
    v_confidence NUMERIC;
    v_age_days INTEGER;
    v_holder_stake NUMERIC := 0.01;  -- Default stake
    v_bounty NUMERIC;
BEGIN
    SELECT 
        COALESCE((confidence->>'overall')::NUMERIC, 0.7),
        EXTRACT(DAY FROM NOW() - created_at)::INTEGER
    INTO v_confidence, v_age_days
    FROM beliefs WHERE id = p_belief_id;
    
    IF NOT FOUND THEN
        RETURN 0;
    END IF;
    
    -- bounty = holder_stake × 0.5 × confidence² × age_factor
    v_bounty := v_holder_stake * 0.5 * (v_confidence * v_confidence) * LEAST(2.0, 1.0 + v_age_days / 30.0);
    
    RETURN v_bounty;
END;
$$ LANGUAGE plpgsql;

-- Function to get verification summary for a belief
CREATE OR REPLACE FUNCTION get_verification_summary(p_belief_id UUID)
RETURNS JSONB AS $$
DECLARE
    v_summary JSONB;
BEGIN
    SELECT jsonb_build_object(
        'total', COUNT(*),
        'by_result', jsonb_object_agg(COALESCE(result::text, 'none'), result_count),
        'by_status', jsonb_object_agg(COALESCE(status::text, 'none'), status_count),
        'total_stake', COALESCE(SUM((stake->>'amount')::NUMERIC), 0),
        'average_stake', COALESCE(AVG((stake->>'amount')::NUMERIC), 0)
    )
    INTO v_summary
    FROM (
        SELECT 
            result,
            status,
            stake,
            COUNT(*) OVER (PARTITION BY result) as result_count,
            COUNT(*) OVER (PARTITION BY status) as status_count
        FROM verifications
        WHERE belief_id = p_belief_id
    ) sub;
    
    RETURN COALESCE(v_summary, '{}'::JSONB);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Triggers
-- ============================================================================

-- Trigger to auto-create bounty when high-confidence belief is created
CREATE OR REPLACE FUNCTION create_bounty_on_belief()
RETURNS TRIGGER AS $$
DECLARE
    v_confidence NUMERIC;
    v_bounty NUMERIC;
BEGIN
    v_confidence := COALESCE((NEW.confidence->>'overall')::NUMERIC, 0.5);
    
    -- Only create bounty for beliefs with confidence > 0.5
    IF v_confidence > 0.5 THEN
        v_bounty := calculate_belief_bounty(NEW.id);
        
        INSERT INTO discrepancy_bounties (
            belief_id, holder_id, base_amount, confidence_premium, 
            total_bounty
        )
        VALUES (
            NEW.id,
            COALESCE(NEW.holder_id, 'local'),
            0.01 * 0.5,  -- base_stake × bounty_multiplier
            v_confidence * v_confidence,
            v_bounty
        )
        ON CONFLICT (belief_id) DO UPDATE
        SET confidence_premium = EXCLUDED.confidence_premium,
            total_bounty = EXCLUDED.total_bounty;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Only create trigger if beliefs table exists and has holder_id column
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'holder_id'
    ) THEN
        DROP TRIGGER IF EXISTS trg_create_bounty ON beliefs;
        CREATE TRIGGER trg_create_bounty
            AFTER INSERT OR UPDATE OF confidence ON beliefs
            FOR EACH ROW EXECUTE FUNCTION create_bounty_on_belief();
    END IF;
END $$;

-- Trigger to update verification count on acceptance
CREATE OR REPLACE FUNCTION update_verification_count()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status = 'accepted' AND OLD.status = 'pending' THEN
        UPDATE reputations
        SET verification_count = verification_count + 1,
            modified_at = NOW()
        WHERE identity_id = NEW.verifier_id;
        
        -- Increment discrepancy_finds if contradicted
        IF NEW.result = 'contradicted' THEN
            UPDATE reputations
            SET discrepancy_finds = discrepancy_finds + 1
            WHERE identity_id = NEW.verifier_id;
        END IF;
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_verification_count
    AFTER UPDATE OF status ON verifications
    FOR EACH ROW EXECUTE FUNCTION update_verification_count();

-- ============================================================================
-- Views
-- ============================================================================

-- View for verification stats per belief
CREATE OR REPLACE VIEW belief_verification_stats AS
SELECT 
    b.id as belief_id,
    b.content,
    COUNT(v.id) as verification_count,
    COUNT(CASE WHEN v.result = 'confirmed' THEN 1 END) as confirmed_count,
    COUNT(CASE WHEN v.result = 'contradicted' THEN 1 END) as contradicted_count,
    COUNT(CASE WHEN v.result = 'uncertain' THEN 1 END) as uncertain_count,
    COUNT(CASE WHEN v.result = 'partial' THEN 1 END) as partial_count,
    SUM((v.stake->>'amount')::NUMERIC) as total_stake,
    AVG((v.stake->>'amount')::NUMERIC) as avg_stake,
    COALESCE(db.total_bounty, 0) as bounty_available
FROM beliefs b
LEFT JOIN verifications v ON b.id = v.belief_id AND v.status = 'accepted'
LEFT JOIN discrepancy_bounties db ON b.id = db.belief_id AND NOT db.claimed
GROUP BY b.id, b.content, db.total_bounty;

-- View for verifier leaderboard
CREATE OR REPLACE VIEW verifier_leaderboard AS
SELECT 
    r.identity_id,
    r.overall as reputation,
    r.verification_count,
    r.discrepancy_finds,
    CASE WHEN r.verification_count > 0 
         THEN r.discrepancy_finds::NUMERIC / r.verification_count 
         ELSE 0 
    END as discrepancy_rate,
    COUNT(CASE WHEN v.status = 'overturned' THEN 1 END) as overturned_count,
    CASE WHEN r.verification_count > 0
         THEN (r.verification_count - COUNT(CASE WHEN v.status = 'overturned' THEN 1 END))::NUMERIC / r.verification_count
         ELSE 1.0
    END as accuracy_rate
FROM reputations r
LEFT JOIN verifications v ON r.identity_id = v.verifier_id
GROUP BY r.identity_id, r.overall, r.verification_count, r.discrepancy_finds
ORDER BY r.overall DESC, r.verification_count DESC;

-- View for pending disputes
CREATE OR REPLACE VIEW pending_disputes AS
SELECT 
    d.*,
    v.belief_id,
    v.verifier_id as original_verifier,
    v.result as original_result,
    v.stake as original_stake,
    b.content as belief_content
FROM disputes d
JOIN verifications v ON d.verification_id = v.id
JOIN beliefs b ON v.belief_id = b.id
WHERE d.status = 'pending'
ORDER BY d.created_at ASC;

-- ============================================================================
-- Add holder_id to beliefs if not exists (for bounty tracking)
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'beliefs' AND column_name = 'holder_id'
    ) THEN
        ALTER TABLE beliefs ADD COLUMN holder_id TEXT DEFAULT 'local';
    END IF;
END $$;

-- ============================================================================
-- Comments
-- ============================================================================

COMMENT ON TABLE verifications IS 'Adversarial verifications of beliefs with reputation staking';
COMMENT ON TABLE disputes IS 'Challenges to verification results';
COMMENT ON TABLE reputations IS 'Agent reputation scores for verification system';
COMMENT ON TABLE stake_positions IS 'Locked reputation stakes';
COMMENT ON TABLE reputation_events IS 'Audit trail of all reputation changes';
COMMENT ON TABLE discrepancy_bounties IS 'Bounties for finding contradictions in high-confidence beliefs';

COMMENT ON FUNCTION lock_stake IS 'Lock reputation as stake for verification or dispute';
COMMENT ON FUNCTION release_stake IS 'Release or forfeit staked reputation';
COMMENT ON FUNCTION apply_reputation_update IS 'Apply reputation change with bounds and logging';
COMMENT ON FUNCTION calculate_belief_bounty IS 'Calculate discrepancy bounty for a belief';
