-- Migration 016: Exchange compaction for completed sessions (#359)
-- Adds compacted_summary JSONB to vkb_sessions for storing exchange summaries.
-- Compaction keeps first+last N exchanges verbatim, replaces middle with summary.

ALTER TABLE vkb_sessions ADD COLUMN IF NOT EXISTS compacted_summary JSONB;
ALTER TABLE vkb_sessions ADD COLUMN IF NOT EXISTS compacted_at TIMESTAMPTZ;

-- Index for finding compaction candidates
CREATE INDEX IF NOT EXISTS idx_vkb_sessions_compaction
    ON vkb_sessions(status, compacted_at)
    WHERE status IN ('completed', 'abandoned');
