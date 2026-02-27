# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Usage tracking and self-organization for the Valence v2 knowledge system.

Implements WU-09 (C4: Self-Organization, C8: Freshness Awareness).

Usage scoring model
-------------------
Each article accumulates a usage_score that reflects:

  score = Σ exp(-λ·days_since_retrieval)   (retrieval recency sum)
        + log(1 + source_count) × 0.5      (connection richness bonus)

Where λ = 0.01 (configurable via VALENCE_USAGE_DECAY_RATE env var).

This gives:
- Heavy weight to recent retrievals (exponential decay)
- Natural accumulation for frequently-accessed articles
- A small bonus for well-sourced articles

Pinned articles are excluded from decay candidate lists entirely.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import UTC, datetime
from typing import Any

from valence.core.db import get_cursor

from .response import ValenceResponse, ok

logger = logging.getLogger(__name__)

# Exponential decay rate: λ per day. Default 0.01 ≈ half-life ≈ 69 days.
_DECAY_RATE: float = float(os.environ.get("VALENCE_USAGE_DECAY_RATE", "0.01"))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _decayed_weight(retrieved_at: datetime, now: datetime | None = None) -> float:
    """Compute exp(-λ · days_since_retrieval) for a single retrieval timestamp."""
    if now is None:
        now = datetime.now(UTC)
    # Make both tz-aware for safe subtraction
    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    days = max(0.0, (now - retrieved_at).total_seconds() / 86_400)
    return math.exp(-_DECAY_RATE * days)


def _compute_score(retrieval_timestamps: list[datetime], source_count: int) -> float:
    """Compute usage score from retrieval timestamps and connection count.

    Args:
        retrieval_timestamps: List of datetimes when the article was retrieved.
        source_count: Number of linked sources (article_sources rows).

    Returns:
        Float usage score ≥ 0.
    """
    now = datetime.now(UTC)
    recency_sum = sum(_decayed_weight(ts, now) for ts in retrieval_timestamps)
    connection_bonus = math.log1p(source_count) * 0.5
    return recency_sum + connection_bonus


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a DB row to a plain serialisable dict."""
    d = dict(row)
    for key in ("id", "source_id"):
        if d.get(key) is not None:
            d[key] = str(d[key])
    for key, val in d.items():
        if hasattr(val, "isoformat"):
            d[key] = val.isoformat()
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def record_usage(article_id: str, query: str, tool: str) -> ValenceResponse:
    """Record article access in usage_traces and update usage_score.

    Inserts one row into usage_traces and immediately recomputes the
    usage_score for the affected article.

    Args:
        article_id: UUID string of the article that was accessed.
        query: The query string that led to the retrieval.
        tool: The tool name that performed the retrieval.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO usage_traces (article_id, query_text, tool_name)
            VALUES (%s, %s, %s)
            """,
            (article_id, query, tool),
        )

        # Fetch all retrieval timestamps for this article
        cur.execute(
            """
            SELECT retrieved_at
            FROM usage_traces
            WHERE article_id = %s
              AND retrieved_at IS NOT NULL
            """,
            (article_id,),
        )
        timestamps = [row["retrieved_at"] for row in cur.fetchall()]

        # Fetch source count
        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM article_sources
            WHERE article_id = %s
            """,
            (article_id,),
        )
        source_count_row = cur.fetchone()
        source_count = int(source_count_row["cnt"]) if source_count_row else 0

        score = _compute_score(timestamps, source_count)

        cur.execute(
            """
            UPDATE articles
            SET usage_score = %s
            WHERE id = %s
            """,
            (score, article_id),
        )

    logger.debug(
        "Recorded usage for article %s (query=%r tool=%r) → score=%.4f",
        article_id,
        query,
        tool,
        score,
    )
    return ok()


async def compute_usage_scores() -> ValenceResponse:
    """Batch recompute usage_score for all articles.

    Idempotent: running multiple times produces the same result.
    Designed to run in < 5 s for 1 K articles via a single SQL UPDATE.

    Score formula (computed in SQL for efficiency):

        recency_sum     = SUM(exp(-λ · EXTRACT(epoch FROM (NOW() - retrieved_at)) / 86400))
        connection_bonus = ln(1 + COUNT(DISTINCT source_id)) * 0.5
        score           = COALESCE(recency_sum, 0) + connection_bonus

    Returns:
        Number of articles whose usage_score was updated.
    """
    decay_rate = _DECAY_RATE

    with get_cursor() as cur:
        cur.execute(
            """
            WITH retrieval_scores AS (
                SELECT
                    article_id,
                    SUM(
                        EXP(
                            -%s * EXTRACT(epoch FROM (NOW() - retrieved_at)) / 86400.0
                        )
                    ) AS recency_sum
                FROM usage_traces
                WHERE article_id IS NOT NULL
                  AND retrieved_at IS NOT NULL
                GROUP BY article_id
            ),
            source_counts AS (
                SELECT
                    article_id,
                    COUNT(DISTINCT source_id) AS cnt
                FROM article_sources
                GROUP BY article_id
            ),
            computed AS (
                SELECT
                    a.id AS article_id,
                    COALESCE(rs.recency_sum, 0.0)
                        + LN(1.0 + COALESCE(sc.cnt, 0)) * 0.5 AS new_score
                FROM articles a
                LEFT JOIN retrieval_scores rs ON rs.article_id = a.id
                LEFT JOIN source_counts sc ON sc.article_id = a.id
            )
            UPDATE articles
            SET usage_score = computed.new_score
            FROM computed
            WHERE articles.id = computed.article_id
              AND articles.usage_score IS DISTINCT FROM computed.new_score
            """,
            (decay_rate,),
        )
        updated_count = cur.rowcount

    logger.info("compute_usage_scores: updated %d article(s)", updated_count)
    return ok(data=updated_count)


async def get_decay_candidates(limit: int = 100) -> ValenceResponse:
    """Return articles with the lowest usage_score, excluding pinned articles.

    Used by the organic forgetting subsystem (C10) to identify articles
    that are candidates for eviction or review.

    Args:
        limit: Maximum number of candidates to return (default 100).

    Returns:
        List of article dicts ordered by usage_score ascending (lowest first).
    """
    limit = max(1, min(limit, 1000))

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, title, content, usage_score, pinned, status,
                   domain_path, created_at, modified_at
            FROM articles
            WHERE pinned = FALSE
              AND status = 'active'
            ORDER BY usage_score ASC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    return ok(data=[_row_to_dict(row) for row in rows])


async def backfill_confidence_scores() -> ValenceResponse:
    """Batch-recompute confidence and corroboration_count for all articles.

    For each article:
    - confidence_source = avg(source.reliability) for linked sources
    - corroboration_count = number of linked sources
    - confidence = {"overall": min(0.95, avg_reliability + source_bonus)}

    Returns number of articles updated.
    """

    with get_cursor() as cur:
        cur.execute(
            """
            WITH source_stats AS (
                SELECT
                    a_s.article_id,
                    AVG(s.reliability) AS avg_reliability,
                    COUNT(DISTINCT a_s.source_id) AS source_count
                FROM article_sources a_s
                JOIN sources s ON s.id = a_s.source_id
                GROUP BY a_s.article_id
            )
            UPDATE articles a
            SET
                confidence_source = COALESCE(ss.avg_reliability, 0.5),
                corroboration_count = COALESCE(ss.source_count, 0),
                confidence = jsonb_build_object(
                    'overall',
                    LEAST(0.95,
                        COALESCE(ss.avg_reliability, 0.5)
                        + CASE
                            WHEN COALESCE(ss.source_count, 0) > 1
                            THEN LEAST(0.15, LN(1 + ss.source_count - 1) * 0.1)
                            ELSE 0.0
                          END
                    )
                )
            FROM source_stats ss
            WHERE a.id = ss.article_id
              AND a.status = 'active'
            """
        )
        updated = cur.rowcount

    logger.info("backfill_confidence_scores: updated %d article(s)", updated)
    return ok(data=updated)
