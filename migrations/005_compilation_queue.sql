-- Migration 005: Add compilation_queue table
-- Issue #491: Compilation guard — queue compilations when inference unavailable

-- Create compilation_queue table
CREATE TABLE IF NOT EXISTS compilation_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_ids TEXT[] NOT NULL,
    title_hint TEXT,
    queued_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    attempts INTEGER DEFAULT 0 NOT NULL,
    last_attempt TIMESTAMPTZ,
    status TEXT DEFAULT 'pending' NOT NULL,
    CONSTRAINT compilation_queue_status_check CHECK (status IN ('pending', 'processing', 'failed'))
);

-- Add index for efficient queue processing
CREATE INDEX idx_compilation_queue_status ON compilation_queue(status, queued_at);

-- Grant permissions
ALTER TABLE compilation_queue OWNER TO valence;
