-- Migration 004: Session ingestion tables
-- Issue #469: Add sessions and session_messages tables
-- Date: 2026-02-24

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    channel TEXT,
    participants TEXT[] DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    parent_session_id TEXT REFERENCES sessions(session_id),
    subagent_label TEXT,
    subagent_model TEXT,
    subagent_task TEXT,
    current_chunk_index INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE session_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    chunk_index INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    speaker TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    flushed_at TIMESTAMPTZ
);

CREATE INDEX idx_session_messages_unflushed ON session_messages (session_id) WHERE flushed_at IS NULL;
CREATE INDEX idx_sessions_stale ON sessions (last_activity_at) WHERE status = 'active';
CREATE INDEX idx_sessions_parent ON sessions (parent_session_id) WHERE parent_session_id IS NOT NULL;
