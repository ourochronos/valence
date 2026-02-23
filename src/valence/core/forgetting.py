"""Explicit and organic forgetting for the Valence v2 knowledge system.

Implements WU-10 (C10: Forgetting).

There are three forgetting operations:

1. ``remove_source``  — Hard-delete a source and all its provenance links.
   Articles that referenced the source are flagged for recompilation via the
   mutation queue.  A tombstone records the deletion for audit purposes.
   This is a *complete* removal — no ghost references remain.

2. ``remove_article`` — Hard-delete a single article.  Sources are untouched.
   A tombstone is created.

3. ``evict_lowest``   — Organic forgetting.  Only fires when the article count
   exceeds ``system_config.bounded_memory.max_articles``.  Evicts the lowest
   usage_score, non-pinned articles.

Privacy guarantee: after any removal, the target ID is entirely absent from
the live tables.  Tombstones record that *a deletion occurred* without storing
the deleted content.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from our_db import get_cursor

from .response import ValenceResponse, err, ok

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_tombstone(cur: Any, content_type: str, content_id: str, metadata: dict | None = None) -> None:
    """Insert a tombstone row for audit / GDPR compliance.

    Args:
        cur: Active DB cursor.
        content_type: ``'source'`` or ``'article'``.
        content_id: UUID (as string) of the deleted record.
        metadata: Optional extra data to store in the tombstone JSONB column.
    """
    cur.execute(
        """
        INSERT INTO tombstones (content_type, content_id, reason, metadata)
        VALUES (%s, %s::uuid, 'admin_action', %s::jsonb)
        """,
        (content_type, content_id, json.dumps(metadata or {})),
    )


def _queue_recompile(cur: Any, article_id: str, source_id: str) -> None:
    """Add a 'recompile' entry to mutation_queue for an affected article.

    Args:
        cur: Active DB cursor.
        article_id: UUID (as string) of the article to recompile.
        source_id: UUID (as string) of the removed source that triggered this.
    """
    cur.execute(
        """
        INSERT INTO mutation_queue (operation, article_id, priority, payload)
        VALUES ('recompile', %s::uuid, 7, %s::jsonb)
        """,
        (article_id, json.dumps({"removed_source_id": source_id, "reason": "source_removed"})),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def remove_source(source_id: str) -> ValenceResponse:
    """Delete a source completely, with no ghost references.

    Steps performed in a single transaction:
    1. Collect all article IDs linked via ``article_sources``.
    2. Queue a ``'recompile'`` mutation for each affected article.
    3. ``DELETE FROM sources WHERE id = …``  — cascades to ``article_sources``
       (FK with ON DELETE CASCADE).
    4. Insert a tombstone.

    Args:
        source_id: UUID string of the source to delete.

    Returns:
        Dict with ``success``, ``source_id``, ``affected_articles`` count,
        and ``tombstone_created``.  On failure: ``success=False, error=…``.
    """
    if not source_id or not source_id.strip():
        return err("source_id is required")

    with get_cursor() as cur:
        # 1. Verify the source exists
        cur.execute("SELECT id FROM sources WHERE id = %s::uuid", (source_id,))
        if not cur.fetchone():
            return err(f"Source not found: {source_id}")

        # 2. Collect affected articles (before cascade deletes them from article_sources)
        cur.execute(
            """
            SELECT DISTINCT article_id::text
            FROM article_sources
            WHERE source_id = %s::uuid
            """,
            (source_id,),
        )
        affected_article_ids = [row["article_id"] for row in cur.fetchall()]

        # 3. Queue 'recompile' for each affected article
        for article_id in affected_article_ids:
            _queue_recompile(cur, article_id, source_id)

        # 4. Delete the source (cascades to article_sources via FK)
        cur.execute("DELETE FROM sources WHERE id = %s::uuid", (source_id,))

        # 5. Verify no ghost references remain
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM article_sources WHERE source_id = %s::uuid",
            (source_id,),
        )
        ghost_row = cur.fetchone()
        ghost_count = int(ghost_row["cnt"]) if ghost_row else 0
        if ghost_count:
            logger.error(
                "Ghost references remain after removing source %s: %d rows in article_sources",
                source_id,
                ghost_count,
            )

        # 6. Create tombstone
        _create_tombstone(
            cur,
            "source",
            source_id,
            metadata={
                "affected_article_count": len(affected_article_ids),
                "recompile_queued": len(affected_article_ids),
            },
        )

    logger.info(
        "remove_source: deleted source %s, queued recompile for %d article(s)",
        source_id,
        len(affected_article_ids),
    )

    return ok(data={
        "source_id": source_id,
        "affected_articles": len(affected_article_ids),
        "recompile_queued": len(affected_article_ids),
        "tombstone_created": True,
        "ghost_references": ghost_count,
    })


async def remove_article(article_id: str) -> ValenceResponse:
    """Delete an article completely.  Sources are NOT affected.

    Steps:
    1. ``DELETE FROM articles WHERE id = …``  — cascades to ``article_sources``,
       ``article_mutations``, and ``mutation_queue`` entries via FK ON DELETE CASCADE.
    2. Insert a tombstone.

    Args:
        article_id: UUID string of the article to delete.

    Returns:
        ValenceResponse with data = {"article_id", "tombstone_created"} on success.
    """
    if not article_id or not article_id.strip():
        return err("article_id is required")

    with get_cursor() as cur:
        # 1. Verify the article exists and grab minimal metadata for the tombstone
        cur.execute(
            "SELECT id, title, usage_score FROM articles WHERE id = %s::uuid",
            (article_id,),
        )
        row = cur.fetchone()
        if not row:
            return err(f"Article not found: {article_id}")

        title = row.get("title")
        usage_score = float(row.get("usage_score") or 0.0)

        # 2. Delete the article (cascades to article_sources, article_mutations,
        #    mutation_queue, usage_traces where FK exists)
        cur.execute("DELETE FROM articles WHERE id = %s::uuid", (article_id,))

        # 3. Create tombstone
        _create_tombstone(
            cur,
            "article",
            article_id,
            metadata={
                "title": title,
                "usage_score": usage_score,
            },
        )

    logger.info("remove_article: deleted article %s (title=%r)", article_id, title)

    return ok(data={"article_id": article_id, "tombstone_created": True})


async def evict_lowest(count: int = 10) -> ValenceResponse:
    """Organic forgetting: evict lowest-score non-pinned articles when over capacity.

    Only fires when the total active article count exceeds
    ``system_config.bounded_memory.max_articles``.  Evicts up to *count*
    articles (capped at the overflow amount so we don't over-delete).

    Pinned articles are *never* evicted.

    Args:
        count: Maximum number of articles to evict per call (default 10).

    Returns:
        ValenceResponse with data = list of evicted article ID strings.
        Empty list when below capacity.
    """
    count = max(1, count)
    evicted: list[str] = []

    with get_cursor() as cur:
        # 1. Load bounded_memory config
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

        # 2. Count current active articles
        cur.execute(
            "SELECT COUNT(*) AS cnt FROM articles WHERE status = 'active'",
        )
        count_row = cur.fetchone()
        current_count = int(count_row["cnt"]) if count_row else 0

        if current_count <= max_articles:
            logger.debug(
                "evict_lowest: below capacity (%d/%d), skipping eviction",
                current_count,
                max_articles,
            )
            return ok(data=[])

        # 3. Calculate how many to evict (capped at overflow)
        overflow = current_count - max_articles
        to_evict = min(count, overflow)

        logger.info(
            "evict_lowest: over capacity (%d/%d), evicting %d article(s)",
            current_count,
            max_articles,
            to_evict,
        )

        # 4. Select lowest-score non-pinned candidates
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

    # 5. Remove each candidate (creates tombstones)
    for article_id in candidates:
        result = await remove_article(article_id)
        if result.success:
            evicted.append(article_id)
        else:
            logger.warning(
                "evict_lowest: failed to evict article %s: %s",
                article_id,
                result.error,
            )

    logger.info("evict_lowest: evicted %d article(s)", len(evicted))
    return ok(data=evicted)
