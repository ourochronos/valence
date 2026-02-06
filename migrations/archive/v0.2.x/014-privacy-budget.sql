-- ============================================================================
-- Migration 014: Privacy Budget Persistence
-- ============================================================================
-- Persists differential privacy budget state across restarts.
-- Closes #144
--
-- Tables:
--   - privacy_budgets: Federation privacy budget state
-- ============================================================================

-- ============================================================================
-- PHASE 1: PRIVACY_BUDGETS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS privacy_budgets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Federation identifier (string for flexibility with different ID formats)
    federation_id TEXT UNIQUE NOT NULL,
    
    -- Budget configuration
    daily_epsilon_budget REAL NOT NULL DEFAULT 10.0,
    daily_delta_budget REAL NOT NULL DEFAULT 0.0001,
    budget_period_hours INTEGER NOT NULL DEFAULT 24,
    
    -- Current spend
    spent_epsilon REAL NOT NULL DEFAULT 0.0,
    spent_delta REAL NOT NULL DEFAULT 0.0,
    queries_today INTEGER NOT NULL DEFAULT 0,
    
    -- Period tracking
    period_start TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Full state (JSONB for topic_budgets and requester_budgets)
    -- Stores serialized state for complex nested structures
    topic_budgets JSONB NOT NULL DEFAULT '{}',
    requester_budgets JSONB NOT NULL DEFAULT '{}',
    
    -- Schema version for future migrations
    schema_version INTEGER NOT NULL DEFAULT 1,
    
    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by federation_id
CREATE INDEX IF NOT EXISTS idx_privacy_budgets_federation_id 
    ON privacy_budgets(federation_id);

-- Index for finding budgets that need period reset
CREATE INDEX IF NOT EXISTS idx_privacy_budgets_period_start 
    ON privacy_budgets(period_start);

-- ============================================================================
-- PHASE 2: TRIGGERS
-- ============================================================================

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_privacy_budgets_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS privacy_budgets_updated ON privacy_budgets;
CREATE TRIGGER privacy_budgets_updated
    BEFORE UPDATE ON privacy_budgets
    FOR EACH ROW
    EXECUTE FUNCTION update_privacy_budgets_timestamp();

-- ============================================================================
-- PHASE 3: HELPER FUNCTIONS
-- ============================================================================

-- Get budget for a federation, creating if not exists
CREATE OR REPLACE FUNCTION get_or_create_privacy_budget(
    p_federation_id TEXT,
    p_daily_epsilon REAL DEFAULT 10.0,
    p_daily_delta REAL DEFAULT 0.0001
)
RETURNS privacy_budgets AS $$
DECLARE
    v_budget privacy_budgets;
BEGIN
    -- Try to get existing
    SELECT * INTO v_budget
    FROM privacy_budgets
    WHERE federation_id = p_federation_id;
    
    -- Create if not exists
    IF NOT FOUND THEN
        INSERT INTO privacy_budgets (
            federation_id,
            daily_epsilon_budget,
            daily_delta_budget
        ) VALUES (
            p_federation_id,
            p_daily_epsilon,
            p_daily_delta
        )
        RETURNING * INTO v_budget;
    END IF;
    
    RETURN v_budget;
END;
$$ LANGUAGE plpgsql;

-- Reset budget period if needed
CREATE OR REPLACE FUNCTION maybe_reset_budget_period(
    p_federation_id TEXT,
    p_period_hours INTEGER DEFAULT 24
)
RETURNS BOOLEAN AS $$
DECLARE
    v_budget privacy_budgets;
    v_hours_elapsed REAL;
BEGIN
    SELECT * INTO v_budget
    FROM privacy_budgets
    WHERE federation_id = p_federation_id;
    
    IF NOT FOUND THEN
        RETURN FALSE;
    END IF;
    
    -- Calculate hours since period start
    v_hours_elapsed := EXTRACT(EPOCH FROM (NOW() - v_budget.period_start)) / 3600;
    
    IF v_hours_elapsed > p_period_hours THEN
        UPDATE privacy_budgets
        SET spent_epsilon = 0.0,
            spent_delta = 0.0,
            queries_today = 0,
            topic_budgets = '{}',
            period_start = NOW()
        WHERE federation_id = p_federation_id;
        
        RETURN TRUE;
    END IF;
    
    RETURN FALSE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PHASE 4: COMMENTS
-- ============================================================================

COMMENT ON TABLE privacy_budgets IS 'Persisted differential privacy budget state for federations';
COMMENT ON COLUMN privacy_budgets.federation_id IS 'Unique federation identifier (UUID as string)';
COMMENT ON COLUMN privacy_budgets.daily_epsilon_budget IS 'Maximum epsilon consumption per period';
COMMENT ON COLUMN privacy_budgets.daily_delta_budget IS 'Maximum delta consumption per period';
COMMENT ON COLUMN privacy_budgets.spent_epsilon IS 'Epsilon consumed in current period';
COMMENT ON COLUMN privacy_budgets.spent_delta IS 'Delta consumed in current period';
COMMENT ON COLUMN privacy_budgets.topic_budgets IS 'Per-topic query tracking (JSONB: topic_hash -> TopicBudget)';
COMMENT ON COLUMN privacy_budgets.requester_budgets IS 'Per-requester rate limiting (JSONB: requester_id -> RequesterBudget)';
COMMENT ON COLUMN privacy_budgets.period_start IS 'Start of current budget period';
