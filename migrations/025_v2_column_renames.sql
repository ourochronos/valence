-- Migration 025: Schema column renames — remove v1 naming remnants (DR-12)
-- WU-17: Rename belief_* columns to article_* equivalents
--
-- Tables affected:
--   contentions:          belief_a_id → article_id, belief_b_id → related_article_id
--   usage_traces:         belief_id → article_id
--   article_entities:     belief_id → article_id
--   vkb_session_insights: belief_id → article_id
--   belief_corroborations: belief_id → article_id

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────────
-- 1. contentions: belief_a_id → article_id, belief_b_id → related_article_id
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE contentions RENAME COLUMN belief_a_id TO article_id;
ALTER TABLE contentions RENAME COLUMN belief_b_id TO related_article_id;

-- Rename FK constraints
ALTER TABLE contentions RENAME CONSTRAINT contentions_belief_a_id_fkey TO contentions_article_id_fkey;
ALTER TABLE contentions RENAME CONSTRAINT contentions_belief_b_id_fkey TO contentions_related_article_id_fkey;

-- Rename indexes
ALTER INDEX idx_contentions_belief_a RENAME TO idx_contentions_article;
ALTER INDEX idx_contentions_belief_b RENAME TO idx_contentions_related_article;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. usage_traces: belief_id → article_id
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE usage_traces RENAME COLUMN belief_id TO article_id;
ALTER TABLE usage_traces RENAME CONSTRAINT usage_traces_belief_id_fkey TO usage_traces_article_id_fkey;
ALTER INDEX idx_usage_traces_belief RENAME TO idx_usage_traces_article;

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. article_entities: belief_id → article_id
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE article_entities RENAME COLUMN belief_id TO article_id;
ALTER TABLE article_entities RENAME CONSTRAINT article_entities_belief_id_fkey TO article_entities_article_id_fkey;
-- The PK constraint name (article_entities_pkey) does not embed the column name; no rename needed.

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. vkb_session_insights: belief_id → article_id
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE vkb_session_insights RENAME COLUMN belief_id TO article_id;
ALTER TABLE vkb_session_insights RENAME CONSTRAINT vkb_session_insights_belief_id_fkey TO vkb_session_insights_article_id_fkey;
ALTER TABLE vkb_session_insights RENAME CONSTRAINT vkb_session_insights_session_id_belief_id_key TO vkb_session_insights_session_id_article_id_key;
ALTER INDEX idx_vkb_session_insights_belief RENAME TO idx_vkb_session_insights_article;

-- ─────────────────────────────────────────────────────────────────────────────
-- 5. belief_corroborations: belief_id → article_id
-- ─────────────────────────────────────────────────────────────────────────────

ALTER TABLE belief_corroborations RENAME COLUMN belief_id TO article_id;
ALTER TABLE belief_corroborations RENAME CONSTRAINT belief_corroborations_belief_id_fkey TO belief_corroborations_article_id_fkey;
ALTER INDEX idx_belief_corroborations_belief RENAME TO idx_belief_corroborations_article;

COMMIT;
