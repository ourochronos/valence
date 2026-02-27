# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Explicit and organic forgetting for the Valence v2 knowledge system.

Sources are append-only — use supersession for corrections.

Three forgetting operations:

1. ``remove_article`` — Delete an article permanently. Sources are untouched.
2. ``archive_lowest``  — Organic forgetting: archive lowest-score articles when over capacity.
3. ``evict_lowest``    — Alias for archive_lowest (backwards compat).

Archived articles remain searchable but with a capped score (rank floor),
so they only surface when directly relevant.
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


async def archive_lowest(count: int = 10) -> ValenceResponse:
    """Organic forgetting: archive lowest-score non-pinned articles when over capacity.

    Archived articles are NOT deleted — they remain searchable with a capped
    score (rank floor) so they only surface when directly relevant.
    """
    count = max(1, count)
    archived: list[str] = []

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
        to_archive = min(count, overflow)

        cur.execute(
            """
            UPDATE articles
            SET status = 'archived'
            WHERE id IN (
                SELECT id
                FROM articles
                WHERE pinned = FALSE
                  AND status = 'active'
                ORDER BY usage_score ASC
                LIMIT %s
            )
            RETURNING id::text
            """,
            (to_archive,),
        )
        archived = [row["id"] for row in cur.fetchall()]

    logger.info("archive_lowest: archived %d article(s)", len(archived))
    return ok(data=archived)


async def evict_lowest(count: int = 10) -> ValenceResponse:
    """Backwards-compatible alias for archive_lowest."""
    return await archive_lowest(count)
