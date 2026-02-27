-- Migration 002: Source sections for tree-indexed embeddings
-- Each section maps to a tree node from build_tree_index() and holds an
-- embedding of the original source content slice [start_char, end_char).

CREATE TABLE IF NOT EXISTS source_sections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id UUID NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    tree_path TEXT NOT NULL,          -- "0", "0.2", "0.2.1" (dot-separated indices)
    title TEXT,
    summary TEXT,
    start_char INT NOT NULL,
    end_char INT NOT NULL,
    depth INT NOT NULL DEFAULT 0,
    embedding vector(1536),
    content_hash TEXT,                -- md5 of content slice, detects staleness
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_id, tree_path)
);

CREATE INDEX IF NOT EXISTS idx_source_sections_source_id
    ON source_sections(source_id);

-- ivfflat needs rows to exist before building; start with exact search,
-- switch to ivfflat when row count > 1000.
CREATE INDEX IF NOT EXISTS idx_source_sections_embedding
    ON source_sections USING hnsw (embedding vector_cosine_ops);
