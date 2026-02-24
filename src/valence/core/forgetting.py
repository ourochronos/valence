"""Explicit and organic forgetting for the Valence v2 knowledge system.

Sources are append-only — use supersession for corrections.

Two forgetting operations remain:

1. ``remove_article`` — Delete an article. Sources are untouched.
2. ``evict_lowest``   — Organic forgetting when over capacity.
"""

from __future__ import annotations

import json
import logging

from valence.core.db import get_cursor

from .response import ValenceResponse, err, ok

logger = logging.getLogger(__name__)


async def remove_source(source_id: str) -> ValenceResponse:
    """[DEPRECATED] Sources are append-only. Use supersession."""
    return err("remove_source is deprecated. Sources are append-only. Use supersession (ingest --supersedes) for corrections.")


async def remove_article(article_id: str) -> ValenceResponse:
    """Delete an article. Sources are NOT affected.

    Cascades to article_sources, article_mutations, mutation_queue via FK.
    """
    if not article_id or not article_id.strip():
        return err("article_id is required")

    with get_cursor() as cur:
        cur.execute(
            "SELECT id, title, usage_score FROM articles WHERE id = %s::uuid",
            (article_id,),
        )
        row = cur.fetchone()
        if not row:
            return err(f"Article not found: {article_id}")

        title = row.get("title")
        cur.execute("DELETE FROM articles WHERE id = %s::uuid", (article_id,))

    logger.info("remove_article: deleted article %s (title=%r)", article_id, title)
    return ok(data={"article_id": article_id})


async def evict_lowest(count: int = 10) -> ValenceResponse:
    """Organic forgetting: evict lowest-score non-pinned articles when over capacity."""
    count = max(1, count)
    evicted: list[str] = []

    with get_cursor() as cur:
        cur.execute(
            "SELECT value FROM system_config WHERE key = 'bounded_memory'",
        )
        config_row = cur.fetchone()
        if config_row:
            try:
                bounded = json.loads(config_row["value"]) if isinstance(config_row["value"], str) else config_row["value"]
                max_articles = int(bounded.get("max_articles", 10000))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                max_articles = 10000
        else:
            max_articles = 10000

        cur.execute(
            "SELECT COUNT(*) AS cnt FROM articles WHERE status = 'active'",
        )
        current_count = int(cur.fetchone()["cnt"])

        if current_count <= max_articles:
            return ok(data=[])

        overflow = current_count - max_articles
        to_evict = min(count, overflow)

        cur.execute(
            """
            SELECT id::text
            FROM articles
            WHERE pinned = FALSE
              AND status = 'active'
            ORDER BY usage_score ASC
            LIMIT %s
            """,
            (to_evict,),
        )
        candidates = [row["id"] for row in cur.fetchall()]

    for article_id in candidates:
        result = await remove_article(article_id)
        if result.success:
            evicted.append(article_id)
        else:
            logger.warning("evict_lowest: failed to evict %s: %s", article_id, result.error)

    logger.info("evict_lowest: evicted %d article(s)", len(evicted))
    return ok(data=evicted)
