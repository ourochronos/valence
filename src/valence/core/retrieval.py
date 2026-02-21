"""Unified retrieval and ranking for the Valence v2 knowledge system.

Implements WU-05 (C9 — Knowledge Retrieval, C8 — Freshness Awareness).

retrieve() is the single entry-point for agents and tools:
  - Searches articles (compiled knowledge) via full-text + vector similarity.
  - Searches ungrouped sources (not linked to any article) via full-text.
  - Ranks by: relevance * 0.5 + confidence * 0.35 + freshness * 0.15.
  - Queues recompile in mutation_queue for any ungrouped sources surfaced.
  - Records usage_traces for every article result returned.

All DB operations are synchronous (psycopg2) run inside asyncio.to_thread()
to match the existing async surface established in WU-03/WU-04.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from our_db import get_cursor

from .ranking import compute_confidence_score, multi_signal_rank

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (sync — called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _compute_freshness_days(article: dict[str, Any]) -> float:
    """Return days since last article update (compiled_at or modified_at or created_at).

    Returns a float number of days (0 = just updated).
    """
    for col in ("compiled_at", "modified_at", "created_at"):
        val = article.get(col)
        if val is None:
            continue
        if isinstance(val, str):
            try:
                val = datetime.fromisoformat(val)
            except (ValueError, TypeError):
                continue
        if isinstance(val, datetime):
            if val.tzinfo is None:
                val = val.replace(tzinfo=UTC)
            return max(0.0, (datetime.now(UTC) - val).total_seconds() / 86400.0)
    return 90.0  # Unknown age → treat as stale


def _freshness_score(days: float, decay_rate: float = 0.01) -> float:
    """Convert age-in-days to a [0, 1] freshness score.

    Uses the same exponential decay as recency_score in ranking.py.
    Default: half-life ≈ 69 days.
    """
    import math

    return max(0.0, min(1.0, math.exp(-decay_rate * days)))


def _build_provenance_summary(
    article_id: str,
    cur: Any,
) -> dict[str, Any]:
    """Query article_sources for a single article and build provenance summary."""
    cur.execute(
        """
        SELECT
            COUNT(DISTINCT asrc.source_id)          AS source_count,
            array_agg(DISTINCT asrc.relationship)   AS relationship_types
        FROM article_sources asrc
        WHERE asrc.article_id = %s
        """,
        (article_id,),
    )
    row = cur.fetchone()
    if not row:
        return {"source_count": 0, "relationship_types": []}
    return {
        "source_count": int(row.get("source_count") or 0),
        "relationship_types": [r for r in (row.get("relationship_types") or []) if r],
    }


def _has_active_contentions(article_id: str, cur: Any) -> bool:
    """Return True if there are any non-resolved contentions for this article."""
    cur.execute(
        """
        SELECT 1
        FROM contentions
        WHERE (article_id_a = %s OR article_id_b = %s)
          AND status != 'resolved'
        LIMIT 1
        """,
        (article_id, article_id),
    )
    return cur.fetchone() is not None


def _record_usage_trace(
    article_id: str,
    query: str,
    tool_name: str,
    final_score: float,
    cur: Any,
    session_id: str | None = None,
) -> None:
    """Insert a row into usage_traces for each article result."""
    try:
        cur.execute(
            """
            INSERT INTO usage_traces (belief_id, query_text, tool_name, final_score, session_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (article_id, query, tool_name, final_score, session_id),
        )
    except Exception as exc:
        # Non-fatal — log and continue
        logger.warning("Failed to record usage trace for article %s: %s", article_id, exc)


def _queue_recompile(source_id: str, query: str, cur: Any) -> None:
    """Queue a recompile mutation for an ungrouped source.

    Uses the smallest available article_id that exists in the articles table
    as a placeholder — this is required by the FK constraint on mutation_queue.
    If no articles exist yet, we skip quietly (nothing to recompile against).
    """
    try:
        # Recompile entries in mutation_queue reference an article_id (FK).
        # For ungrouped sources, we have no article yet. We enqueue via a sentinel
        # by looking up any existing active article as the owner. The compilation
        # job will pick up the source_id from the payload.
        cur.execute(
            """
            SELECT id FROM articles WHERE status = 'active' ORDER BY created_at LIMIT 1
            """,
        )
        row = cur.fetchone()
        if row is None:
            logger.debug("No articles yet; skipping recompile queue for source %s", source_id)
            return

        article_id = str(row["id"])
        cur.execute(
            """
            INSERT INTO mutation_queue (operation, article_id, priority, payload)
            VALUES ('recompile', %s, 5, %s::jsonb)
            """,
            (article_id, f'{{"source_id": "{source_id}", "query": "{query}"}}'),
        )
        logger.debug("Queued recompile for ungrouped source %s", source_id)
    except Exception as exc:
        logger.warning("Failed to queue recompile for source %s: %s", source_id, exc)


def _search_articles_sync(query: str, limit: int) -> list[dict[str, Any]]:
    """Full-text search over articles, with optional vector re-ranking."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT a.*,
                   ts_rank(a.content_tsv, websearch_to_tsquery('english', %s)) AS text_rank
            FROM articles a
            WHERE a.content_tsv @@ websearch_to_tsquery('english', %s)
              AND a.status = 'active'
              AND a.superseded_by_id IS NULL
            ORDER BY text_rank DESC
            LIMIT %s
            """,
            (query, query, limit * 2),
        )
        rows = cur.fetchall()

        if not rows:
            return []

        # Normalise text ranks to [0, 1]
        max_rank = max(float(r.get("text_rank", 0) or 0) for r in rows) or 1.0
        results = []
        for row in rows:
            d = dict(row)
            # Serialise non-JSON types
            for key in list(d.keys()):
                import uuid

                if isinstance(d[key], datetime):
                    d[key] = d[key].isoformat()
                elif isinstance(d[key], uuid.UUID):
                    d[key] = str(d[key])
            d.pop("content_tsv", None)
            d.pop("embedding", None)

            text_rel = float(row.get("text_rank", 0) or 0) / max_rank
            d["similarity"] = text_rel
            d["text_relevance"] = text_rel
            d["type"] = "article"
            results.append(d)

        # Attach provenance summary, freshness, contention flags
        for d in results:
            article_id = d.get("id") or d.get("article_id")
            d["provenance_summary"] = _build_provenance_summary(article_id, cur)
            freshness_days = _compute_freshness_days(d)
            d["freshness"] = freshness_days
            d["freshness_score"] = _freshness_score(freshness_days)
            d["active_contentions"] = _has_active_contentions(article_id, cur)

        return results


def _search_ungrouped_sources_sync(query: str, limit: int) -> list[dict[str, Any]]:
    """Search sources that are NOT linked to any article."""
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT s.id, s.type, s.title, s.url, s.content, s.reliability,
                   s.fingerprint, s.created_at,
                   ts_rank(s.content_tsv, websearch_to_tsquery('english', %s)) AS text_rank
            FROM sources s
            WHERE s.content_tsv @@ websearch_to_tsquery('english', %s)
              AND NOT EXISTS (
                  SELECT 1
                  FROM article_sources asrc
                  WHERE asrc.source_id = s.id
              )
            ORDER BY text_rank DESC
            LIMIT %s
            """,
            (query, query, limit),
        )
        rows = cur.fetchall()

        if not rows:
            return []

        max_rank = max(float(r.get("text_rank", 0) or 0) for r in rows) or 1.0
        results = []
        for row in rows:
            d = dict(row)
            import uuid

            for key in list(d.keys()):
                if isinstance(d[key], datetime):
                    d[key] = d[key].isoformat()
                elif isinstance(d[key], uuid.UUID):
                    d[key] = str(d[key])

            text_rel = float(row.get("text_rank", 0) or 0) / max_rank
            d["similarity"] = text_rel
            d["type"] = "source"
            # Sources use reliability as confidence
            d["confidence"] = {"overall": float(d.get("reliability", 0.5))}
            # Freshness from created_at
            freshness_days = _compute_freshness_days(d)
            d["freshness"] = freshness_days
            d["freshness_score"] = _freshness_score(freshness_days)
            d["active_contentions"] = False
            d["provenance_summary"] = {"source_count": 1, "relationship_types": []}
            results.append(d)

        return results


def _retrieve_sync(
    query: str,
    limit: int,
    include_sources: bool,
    session_id: str | None,
) -> list[dict[str, Any]]:
    """Synchronous implementation of unified retrieval."""

    # --- Article search ---
    article_results = _search_articles_sync(query, limit)

    # --- Ungrouped source search (always, unless caller opts out) ---
    source_results: list[dict[str, Any]] = []
    if include_sources:
        source_results = _search_ungrouped_sources_sync(query, limit)

    # --- Merge + rank ---
    all_results = article_results + source_results

    if not all_results:
        return []

    # Inject freshness_score as the recency signal into multi_signal_rank.
    # multi_signal_rank uses created_at for recency; we override by injecting
    # a synthetic "created_at" that represents our freshness_score directly.
    # Instead, we use multi_signal_rank's recency path but substitute the
    # pre-computed freshness_score into the result so the downstream consumer
    # can see it.  For ranking purposes we set a synthetic created_at to NOW
    # minus freshness_days (already computed) so the built-in recency calc
    # reproduces the freshness_score.
    now = datetime.now(UTC)
    for r in all_results:
        freshness_days = r.get("freshness", 90.0)
        # Write a synthetic created_at back into the dict so multi_signal_rank
        # calculates the same exponential-decay recency value we already computed.
        r["_original_created_at"] = r.get("created_at")
        r["created_at"] = (now - timedelta(days=freshness_days)).isoformat()

    ranked = multi_signal_rank(
        all_results,
        semantic_weight=0.50,
        confidence_weight=0.35,
        recency_weight=0.15,
    )

    # Restore original created_at
    for r in ranked:
        orig = r.pop("_original_created_at", None)
        if orig is not None:
            r["created_at"] = orig

    ranked = ranked[:limit]

    # --- Side effects ---
    with get_cursor() as cur:
        for item in ranked:
            if item.get("type") == "article":
                article_id = item.get("id")
                if article_id:
                    _record_usage_trace(
                        article_id=article_id,
                        query=query,
                        tool_name="knowledge_search",
                        final_score=float(item.get("final_score", 0)),
                        cur=cur,
                        session_id=session_id,
                    )
            elif item.get("type") == "source":
                source_id = item.get("id")
                if source_id:
                    _queue_recompile(source_id=source_id, query=query, cur=cur)

    return ranked


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def retrieve(
    query: str,
    limit: int = 10,
    include_sources: bool = False,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Unified retrieval: search articles and ungrouped sources.

    Ranks results by: relevance * 0.5 + confidence * 0.35 + freshness * 0.15.

    Ungrouped sources that match are surfaced with ``type="source"`` and trigger
    a ``recompile`` entry in the mutation_queue (deferred compilation).

    Usage traces are recorded in ``usage_traces`` for every article returned.

    Args:
        query:           Natural-language search query.
        limit:           Maximum results to return (default 10).
        include_sources: If True, include ungrouped sources in results and
                         queue their compilation.
        session_id:      Optional session ID to attach to usage traces.

    Returns:
        List of result dicts, each containing:
          - id, content, title (articles) / type (sources)
          - type: "article" or "source"
          - final_score: combined ranking score
          - confidence: confidence dict with 'overall' key
          - freshness: days since last update (float)
          - freshness_score: [0, 1] score
          - provenance_summary: {source_count, relationship_types}
          - active_contentions: bool
    """
    if not query or not query.strip():
        return []

    limit = max(1, min(limit, 200))

    return await asyncio.to_thread(
        _retrieve_sync,
        query,
        limit,
        include_sources,
        session_id,
    )
