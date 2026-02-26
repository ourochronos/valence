-- Add epistemic_type to articles for different lifecycle rules
-- Types: episodic (decays), semantic (persists), procedural (pinned)

DO $$ BEGIN
    CREATE TYPE epistemic_type AS ENUM ('episodic', 'semantic', 'procedural');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

ALTER TABLE articles
    ADD COLUMN IF NOT EXISTS epistemic_type epistemic_type NOT NULL DEFAULT 'semantic';

-- Index for filtering by type in decay/eviction queries
CREATE INDEX IF NOT EXISTS idx_articles_epistemic_type ON articles (epistemic_type) WHERE status = 'active';
