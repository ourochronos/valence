"""Migration 025: Schema column renames — remove v1 naming remnants (DR-12)

WU-17: Renames belief_* columns to article_* equivalents throughout the schema.

Tables affected:
  contentions:           belief_a_id → article_id, belief_b_id → related_article_id
  usage_traces:          belief_id → article_id
  article_entities:      belief_id → article_id
  vkb_session_insights:  belief_id → article_id
  belief_corroborations: belief_id → article_id
"""

version = "025"
description = "v2_column_renames"

_STATEMENTS = [
    # contentions
    ("rename contentions.belief_a_id → article_id",
     "ALTER TABLE contentions RENAME COLUMN belief_a_id TO article_id"),
    ("rename contentions.belief_b_id → related_article_id",
     "ALTER TABLE contentions RENAME COLUMN belief_b_id TO related_article_id"),
    ("rename contentions FK belief_a",
     "ALTER TABLE contentions RENAME CONSTRAINT contentions_belief_a_id_fkey TO contentions_article_id_fkey"),
    ("rename contentions FK belief_b",
     "ALTER TABLE contentions RENAME CONSTRAINT contentions_belief_b_id_fkey TO contentions_related_article_id_fkey"),
    ("rename idx_contentions_belief_a",
     "ALTER INDEX IF EXISTS idx_contentions_belief_a RENAME TO idx_contentions_article"),
    ("rename idx_contentions_belief_b",
     "ALTER INDEX IF EXISTS idx_contentions_belief_b RENAME TO idx_contentions_related_article"),

    # usage_traces
    ("rename usage_traces.belief_id → article_id",
     "ALTER TABLE usage_traces RENAME COLUMN belief_id TO article_id"),
    ("rename usage_traces FK",
     "ALTER TABLE usage_traces RENAME CONSTRAINT usage_traces_belief_id_fkey TO usage_traces_article_id_fkey"),
    ("rename idx_usage_traces_belief",
     "ALTER INDEX IF EXISTS idx_usage_traces_belief RENAME TO idx_usage_traces_article"),

    # article_entities
    ("rename article_entities.belief_id → article_id",
     "ALTER TABLE article_entities RENAME COLUMN belief_id TO article_id"),
    ("rename article_entities FK",
     "ALTER TABLE article_entities RENAME CONSTRAINT article_entities_belief_id_fkey TO article_entities_article_id_fkey"),

    # vkb_session_insights
    ("rename vkb_session_insights.belief_id → article_id",
     "ALTER TABLE vkb_session_insights RENAME COLUMN belief_id TO article_id"),
    ("rename vkb_session_insights FK",
     "ALTER TABLE vkb_session_insights RENAME CONSTRAINT vkb_session_insights_belief_id_fkey TO vkb_session_insights_article_id_fkey"),
    ("rename vkb_session_insights unique constraint",
     "ALTER TABLE vkb_session_insights RENAME CONSTRAINT vkb_session_insights_session_id_belief_id_key TO vkb_session_insights_session_id_article_id_key"),
    ("rename idx_vkb_session_insights_belief",
     "ALTER INDEX IF EXISTS idx_vkb_session_insights_belief RENAME TO idx_vkb_session_insights_article"),

    # belief_corroborations
    ("rename belief_corroborations.belief_id → article_id",
     "ALTER TABLE belief_corroborations RENAME COLUMN belief_id TO article_id"),
    ("rename belief_corroborations FK",
     "ALTER TABLE belief_corroborations RENAME CONSTRAINT belief_corroborations_belief_id_fkey TO belief_corroborations_article_id_fkey"),
    ("rename idx_belief_corroborations_belief",
     "ALTER INDEX IF EXISTS idx_belief_corroborations_belief RENAME TO idx_belief_corroborations_article"),
]


def up(conn) -> None:
    with conn.cursor() as cur:
        for _desc, sql in _STATEMENTS:
            try:
                cur.execute(sql)
            except Exception as exc:
                # Column/index already renamed (e.g., migration partially applied)
                if "does not exist" in str(exc) or "already exists" in str(exc):
                    pass
                else:
                    raise


def down(conn) -> None:
    """Reverse: rename article_* columns back to belief_* names."""
    reversals = [
        "ALTER TABLE contentions RENAME COLUMN article_id TO belief_a_id",
        "ALTER TABLE contentions RENAME COLUMN related_article_id TO belief_b_id",
        "ALTER TABLE usage_traces RENAME COLUMN article_id TO belief_id",
        "ALTER TABLE article_entities RENAME COLUMN article_id TO belief_id",
        "ALTER TABLE vkb_session_insights RENAME COLUMN article_id TO belief_id",
        "ALTER TABLE belief_corroborations RENAME COLUMN article_id TO belief_id",
    ]
    with conn.cursor() as cur:
        for sql in reversals:
            try:
                cur.execute(sql)
            except Exception:
                pass
