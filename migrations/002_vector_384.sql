-- Migration 002: Standardize embedding columns on VECTOR(384)
--
-- The schema originally specified VECTOR(1536) for OpenAI text-embedding-3-small,
-- but the default model is BAAI/bge-small-en-v1.5 which produces 384-dim vectors.
-- All existing embeddings in the database are 384-dim. This migration aligns
-- the schema with reality.

-- Drop views that depend on embedding columns (CASCADE would work too, but explicit is safer)
DROP VIEW IF EXISTS beliefs_current CASCADE;
DROP VIEW IF EXISTS beliefs_with_entities CASCADE;
DROP VIEW IF EXISTS federated_beliefs CASCADE;
DROP VIEW IF EXISTS vkb_patterns_overview CASCADE;
DROP VIEW IF EXISTS vkb_sessions_overview CASCADE;

-- Resize embedding columns
ALTER TABLE beliefs ALTER COLUMN embedding TYPE VECTOR(384);
ALTER TABLE vkb_exchanges ALTER COLUMN embedding TYPE VECTOR(384);
ALTER TABLE vkb_patterns ALTER COLUMN embedding TYPE VECTOR(384);

-- Recreate dropped views
CREATE OR REPLACE VIEW beliefs_current AS
SELECT * FROM beliefs WHERE status = 'active' AND superseded_by_id IS NULL;

CREATE OR REPLACE VIEW beliefs_with_entities AS
SELECT b.*,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'subject') as subjects,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'object') as objects,
    array_agg(DISTINCT e.name) FILTER (WHERE be.role = 'context') as contexts
FROM beliefs b
LEFT JOIN belief_entities be ON b.id = be.belief_id
LEFT JOIN entities e ON be.entity_id = e.id
GROUP BY b.id;

CREATE OR REPLACE VIEW vkb_sessions_overview AS
SELECT s.*, COUNT(DISTINCT e.id) as exchange_count, COUNT(DISTINCT si.id) as insight_count
FROM vkb_sessions s
LEFT JOIN vkb_exchanges e ON s.id = e.session_id
LEFT JOIN vkb_session_insights si ON s.id = si.session_id
GROUP BY s.id;

CREATE OR REPLACE VIEW vkb_patterns_overview AS
SELECT p.*, array_length(p.evidence, 1) as evidence_count FROM vkb_patterns p;

-- Update default embedding type from OpenAI to local
UPDATE embedding_types SET is_default = FALSE WHERE id = 'openai_text3_small';

INSERT INTO embedding_types (id, provider, model, dimensions, is_default, status)
VALUES ('local_bge_small', 'local', 'BAAI/bge-small-en-v1.5', 384, TRUE, 'active')
ON CONFLICT (id) DO UPDATE SET is_default = TRUE, status = 'active';
