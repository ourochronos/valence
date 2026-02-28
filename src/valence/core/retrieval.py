# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Unified retrieval and ranking for the Valence v2 knowledge system.

Implements WU-05 (C9 — Knowledge Retrieval, C8 — Freshness Awareness).

retrieve() is the single entry-point for agents and tools:
  - Searches articles (compiled knowledge) via hybrid search (vector + text).
  - Combines signals using Reciprocal Rank Fusion (RRF).
  - Searches ungrouped sources (not linked to any article) via hybrid search.
  - Ranks by: relevance * 0.5 + confidence * 0.35 + freshness * 0.15.
  - Queues recompile in mutation_queue for any ungrouped sources surfaced.
  - Records usage_traces for every article result returned.

All DB operations are synchronous (psycopg2) run inside asyncio.to_thread()
to match the existing async surface established in WU-03/WU-04.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import UTC, datetime, timedelta
from typing import Any

from valence.core.db import get_cursor, serialize_row
from valence.core.embeddings import generate_embedding
from valence.core.sources import resolve_supersession_head_sync

from .ranking import multi_signal_rank
from .response import ValenceResponse, ok

logger = logging.getLogger(__name__)

# RRF constant (standard value from literature)
RRF_K = 60
# Default rank for items not found by one search method
DEFAULT_RANK = 1000
# Maximum final_score for archived articles (organic forgetting rank floor)
ARCHIVED_RANK_FLOOR = 0.1
# Grace period: new articles get a novelty boost that decays linearly over this duration
GRACE_PERIOD_HOURS = 48
# Maximum novelty boost multiplier (at creation, decays linearly to 1.0)
NOVELTY_BOOST_MAX = 1.5

# ---------------------------------------------------------------------------
# Temporal mode weight presets
# ---------------------------------------------------------------------------

TEMPORAL_WEIGHT_PRESETS: dict[str, dict[str, float]] = {
    "default": {"semantic": 0.50, "confidence": 0.35, "recency": 0.15},
    "prefer_recent": {"semantic": 0.35, "confidence": 0.15, "recency": 0.50},
    "prefer_stable": {"semantic": 0.40, "confidence": 0.45, "recency": 0.15},
}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _try_generate_embedding(query: str) -> str | None:
    """Generate a query embedding, returning pgvector literal or None on failure."""
    try:
        vec = generate_embedding(query)
        return "[" + ",".join(str(v) for v in vec) + "]"
    except Exception as exc:
        logger.warning("Embedding generation failed, falling back to text-only: %s", exc)
        return None


def _extract_rrf_scores(d: dict[str, Any], rrf_min: float, rrf_range: float) -> None:
    """Normalize RRF score and attach search signal fields to result dict (in-place)."""
    raw_rrf = float(d.get("rrf_score", 0) or 0)
    normalized = (raw_rrf - rrf_min) / rrf_range if rrf_range > 0 else 0.5

    d["similarity"] = normalized  # consumed by multi_signal_rank
    d["rrf_score"] = raw_rrf
    d["vec_score"] = float(d.get("vec_score", 0) or 0)
    d["text_score"] = float(d.get("text_score", 0) or 0)
    d["vec_rank"] = int(d.get("vec_rank", 0) or 0)
    d["text_rank"] = int(d.get("text_rank", 0) or 0)


def _rrf_range(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Return (max_rrf, min_rrf, range) for a set of rows."""
    scores = [float(r.get("rrf_score", 0) or 0) for r in rows]
    mx = max(scores) if scores else 1.0
    mn = min(scores) if scores else 0.0
    return mx, mn, (mx - mn) if mx > mn else 1.0


def _compute_freshness_days(article: dict[str, Any]) -> float:
    """Return days since last article update (compiled_at or modified_at or created_at)."""
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

    Uses exponential decay. Default: half-life ≈ 69 days.
    """
    return max(0.0, min(1.0, math.exp(-decay_rate * days)))


# ---------------------------------------------------------------------------
# Provenance / metadata helpers
# ---------------------------------------------------------------------------


def _build_provenance_summary(article_id: str, cur: Any) -> dict[str, Any]:
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
        WHERE (article_id = %s OR related_article_id = %s)
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
    """Insert a row into usage_traces and incrementally update usage_score."""
    try:
        cur.execute(
            """
            INSERT INTO usage_traces (article_id, query_text, tool_name, final_score, session_id)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (article_id, query, tool_name, final_score, session_id),
        )
        # Incremental score bump: add 1.0 for this retrieval (fresh, no decay).
        # Full batch recompute via compute_usage_scores() corrects drift.
        cur.execute(
            """
            UPDATE articles
            SET usage_score = COALESCE(usage_score, 0) + 1.0
            WHERE id = %s
            """,
            (article_id,),
        )
    except Exception as exc:
        logger.warning("Failed to record usage trace for article %s: %s", article_id, exc)


def _queue_recompile(source_id: str, query: str, cur: Any) -> None:
    """Queue a recompile mutation for an ungrouped source."""
    try:
        cur.execute(
            "SELECT id FROM articles WHERE status IN ('active', 'archived') ORDER BY created_at LIMIT 1",
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


# ---------------------------------------------------------------------------
# Search: articles
# ---------------------------------------------------------------------------

_ARTICLE_HYBRID_SQL = """
WITH vec AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector) AS vec_rank,
           1 - (embedding <=> %(emb)s::vector) AS vec_score
    FROM articles
    WHERE status IN ('active', 'archived')
      AND superseded_by_id IS NULL
      AND embedding IS NOT NULL
    ORDER BY embedding <=> %(emb)s::vector
    LIMIT %(search_limit)s
),
txt AS (
    SELECT id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', %(q)s)) DESC
           ) AS text_rank,
           ts_rank(content_tsv, websearch_to_tsquery('english', %(q)s)) AS text_score
    FROM articles
    WHERE status IN ('active', 'archived')
      AND superseded_by_id IS NULL
      AND content_tsv @@ websearch_to_tsquery('english', %(q)s)
    LIMIT %(search_limit)s
)
SELECT a.*,
       COALESCE(v.vec_rank, %(default_rank)s) AS vec_rank,
       COALESCE(v.vec_score, 0) AS vec_score,
       COALESCE(t.text_rank, %(default_rank)s) AS text_rank,
       COALESCE(t.text_score, 0) AS text_score,
       (1.0 / (%(k)s + COALESCE(v.vec_rank, %(default_rank)s)))
       + (1.0 / (%(k)s + COALESCE(t.text_rank, %(default_rank)s))) AS rrf_score
FROM articles a
LEFT JOIN vec v ON v.id = a.id
LEFT JOIN txt t ON t.id = a.id
WHERE v.id IS NOT NULL OR t.id IS NOT NULL
ORDER BY rrf_score DESC
LIMIT %(lim)s
"""

_ARTICLE_TEXT_ONLY_SQL = """
SELECT a.*,
       ts_rank(a.content_tsv, websearch_to_tsquery('english', %(q)s)) AS text_score,
       0.0 AS vec_score,
       ROW_NUMBER() OVER (
           ORDER BY ts_rank(a.content_tsv, websearch_to_tsquery('english', %(q)s)) DESC
       ) AS text_rank,
       %(default_rank)s AS vec_rank,
       (1.0 / (%(k)s + ROW_NUMBER() OVER (
           ORDER BY ts_rank(a.content_tsv, websearch_to_tsquery('english', %(q)s)) DESC
       ))) AS rrf_score
FROM articles a
WHERE a.content_tsv @@ websearch_to_tsquery('english', %(q)s)
  AND a.status IN ('active', 'archived')
  AND a.superseded_by_id IS NULL
ORDER BY rrf_score DESC
LIMIT %(lim)s
"""


def _search_articles_sync(query: str, limit: int) -> list[dict[str, Any]]:
    """Hybrid search: vector KNN + full-text combined via RRF."""
    embedding_str = _try_generate_embedding(query)

    params = {
        "q": query,
        "k": RRF_K,
        "default_rank": DEFAULT_RANK,
        "lim": limit,
    }

    with get_cursor() as cur:
        if embedding_str is None:
            cur.execute(_ARTICLE_TEXT_ONLY_SQL, params)
        else:
            params["emb"] = embedding_str
            params["search_limit"] = limit * 2
            cur.execute(_ARTICLE_HYBRID_SQL, params)

        rows = cur.fetchall()
        if not rows:
            return []

        _, rrf_min, rrf_range = _rrf_range(rows)

        results = []
        for row in rows:
            d = serialize_row(row)
            d.pop("content_tsv", None)
            d.pop("embedding", None)
            _extract_rrf_scores(d, rrf_min, rrf_range)
            d["type"] = "article"
            results.append(d)

        # Attach provenance, freshness, contention flags
        for d in results:
            article_id = d.get("id") or d.get("article_id")
            if article_id:
                article_id = str(article_id)
                d["provenance_summary"] = _build_provenance_summary(article_id, cur)
                freshness_days = _compute_freshness_days(d)
                d["freshness"] = freshness_days
                d["freshness_score"] = _freshness_score(freshness_days)
                d["active_contentions"] = _has_active_contentions(article_id, cur)
            else:
                d["provenance_summary"] = {}
                d["freshness"] = None
                d["freshness_score"] = 0.0
                d["active_contentions"] = False

        return results


# ---------------------------------------------------------------------------
# Search: ungrouped sources
# ---------------------------------------------------------------------------

_SOURCE_HYBRID_SQL = """
WITH vec AS (
    SELECT id,
           ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s::vector) AS vec_rank,
           1 - (embedding <=> %(emb)s::vector) AS vec_score
    FROM sources
    WHERE embedding IS NOT NULL
      AND NOT EXISTS (
          SELECT 1 FROM article_sources asrc WHERE asrc.source_id = sources.id
      )
    ORDER BY embedding <=> %(emb)s::vector
    LIMIT %(search_limit)s
),
txt AS (
    SELECT id,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank(content_tsv, websearch_to_tsquery('english', %(q)s)) DESC
           ) AS text_rank,
           ts_rank(content_tsv, websearch_to_tsquery('english', %(q)s)) AS text_score
    FROM sources
    WHERE content_tsv @@ websearch_to_tsquery('english', %(q)s)
      AND NOT EXISTS (
          SELECT 1 FROM article_sources asrc WHERE asrc.source_id = sources.id
      )
    LIMIT %(search_limit)s
)
SELECT s.id, s.type, s.title, s.url, s.content, s.reliability,
       s.fingerprint, s.created_at, s.supersedes_id,
       EXISTS (SELECT 1 FROM sources s2 WHERE s2.supersedes_id = s.id) AS is_superseded,
       COALESCE(v.vec_rank, %(default_rank)s) AS vec_rank,
       COALESCE(v.vec_score, 0) AS vec_score,
       COALESCE(t.text_rank, %(default_rank)s) AS text_rank,
       COALESCE(t.text_score, 0) AS text_score,
       (1.0 / (%(k)s + COALESCE(v.vec_rank, %(default_rank)s)))
       + (1.0 / (%(k)s + COALESCE(t.text_rank, %(default_rank)s))) AS rrf_score
FROM sources s
LEFT JOIN vec v ON v.id = s.id
LEFT JOIN txt t ON t.id = s.id
WHERE v.id IS NOT NULL OR t.id IS NOT NULL
ORDER BY rrf_score DESC
LIMIT %(lim)s
"""

_SOURCE_TEXT_ONLY_SQL = """
SELECT s.id, s.type, s.title, s.url, s.content, s.reliability,
       s.fingerprint, s.created_at, s.supersedes_id,
       EXISTS (SELECT 1 FROM sources s2 WHERE s2.supersedes_id = s.id) AS is_superseded,
       ts_rank(s.content_tsv, websearch_to_tsquery('english', %(q)s)) AS text_score,
       0.0 AS vec_score,
       ROW_NUMBER() OVER (
           ORDER BY ts_rank(s.content_tsv, websearch_to_tsquery('english', %(q)s)) DESC
       ) AS text_rank,
       %(default_rank)s AS vec_rank,
       (1.0 / (%(k)s + ROW_NUMBER() OVER (
           ORDER BY ts_rank(s.content_tsv, websearch_to_tsquery('english', %(q)s)) DESC
       ))) AS rrf_score
FROM sources s
WHERE s.content_tsv @@ websearch_to_tsquery('english', %(q)s)
  AND NOT EXISTS (
      SELECT 1 FROM article_sources asrc WHERE asrc.source_id = s.id
  )
ORDER BY rrf_score DESC
LIMIT %(lim)s
"""


def _search_ungrouped_sources_sync(query: str, limit: int) -> list[dict[str, Any]]:
    """Hybrid search for sources not linked to any article."""
    embedding_str = _try_generate_embedding(query)

    params = {
        "q": query,
        "k": RRF_K,
        "default_rank": DEFAULT_RANK,
        "lim": limit,
    }

    with get_cursor() as cur:
        if embedding_str is None:
            cur.execute(_SOURCE_TEXT_ONLY_SQL, params)
        else:
            params["emb"] = embedding_str
            params["search_limit"] = limit * 2
            cur.execute(_SOURCE_HYBRID_SQL, params)

        rows = cur.fetchall()
        if not rows:
            return []

        _, rrf_min, rrf_range = _rrf_range(rows)

        results = []
        for row in rows:
            d = serialize_row(row)
            _extract_rrf_scores(d, rrf_min, rrf_range)
            d["type"] = "source"
            d["confidence"] = {"overall": float(d.get("reliability", 0.5))}
            freshness_days = _compute_freshness_days(d)
            d["freshness"] = freshness_days
            d["freshness_score"] = _freshness_score(freshness_days)
            d["active_contentions"] = False
            d["provenance_summary"] = {"source_count": 1, "relationship_types": []}

            # Supersession chain flattening: is_superseded comes from the SQL EXISTS
            # subquery so no extra per-row query is needed.
            source_id_str = str(d.get("id", ""))
            d["is_superseded"] = bool(d.pop("is_superseded", False))
            # supersession_head_id resolved lazily below after the loop
            d["supersession_head_id"] = None

            results.append(d)

        # Resolve supersession head IDs for superseded sources in a single cursor context.
        superseded = [d for d in results if d.get("is_superseded")]
        if superseded:
            with get_cursor() as chain_cur:
                for d in superseded:
                    sid = str(d.get("id", ""))
                    d["supersession_head_id"] = resolve_supersession_head_sync(sid, chain_cur) if sid else sid
        # For non-superseded sources the head is themselves.
        for d in results:
            if not d.get("is_superseded"):
                d["supersession_head_id"] = str(d.get("id", ""))

        # De-rank superseded sources: halve their similarity score so the head
        # of each chain floats to the top.
        for d in results:
            if d.get("is_superseded"):
                d["similarity"] = d.get("similarity", 0.0) * 0.5
                d["confidence"] = {"overall": d["confidence"]["overall"] * 0.5}

        return results


# ---------------------------------------------------------------------------
# Search: source sections (tree-indexed embeddings)
# ---------------------------------------------------------------------------

_SECTION_HYBRID_SQL = """
WITH vec AS (
    SELECT ss.id,
           ss.source_id,
           ROW_NUMBER() OVER (ORDER BY ss.embedding <=> %(emb)s::vector) AS vec_rank,
           1 - (ss.embedding <=> %(emb)s::vector) AS vec_score
    FROM source_sections ss
    WHERE ss.embedding IS NOT NULL
    ORDER BY ss.embedding <=> %(emb)s::vector
    LIMIT %(search_limit)s
)
SELECT s.id AS source_id, s.type, s.title AS source_title, s.url,
       s.reliability, s.created_at,
       ss.id AS section_id, ss.tree_path, ss.title AS section_title,
       ss.summary, ss.start_char, ss.end_char, ss.depth,
       COALESCE(v.vec_score, 0) AS vec_score,
       COALESCE(v.vec_rank, %(default_rank)s) AS vec_rank,
       %(default_rank)s AS text_rank,
       0.0 AS text_score,
       (1.0 / (%(k)s + COALESCE(v.vec_rank, %(default_rank)s))) AS rrf_score
FROM source_sections ss
JOIN sources s ON s.id = ss.source_id
JOIN vec v ON v.id = ss.id
ORDER BY rrf_score DESC
LIMIT %(lim)s
"""


def _search_source_sections_sync(query: str, limit: int) -> list[dict[str, Any]]:
    """Vector search over source_sections for tree-indexed content."""
    embedding_str = _try_generate_embedding(query)
    if embedding_str is None:
        return []  # Sections are vector-only (no tsv column)

    params = {
        "q": query,
        "emb": embedding_str,
        "k": RRF_K,
        "default_rank": DEFAULT_RANK,
        "search_limit": limit * 2,
        "lim": limit,
    }

    with get_cursor() as cur:
        cur.execute(_SECTION_HYBRID_SQL, params)
        rows = cur.fetchall()
        if not rows:
            return []

        _, rrf_min, rrf_range = _rrf_range(rows)

        results = []
        for row in rows:
            d = dict(row)
            # Extract the actual content slice from the source
            cur.execute("SELECT content FROM sources WHERE id = %s", (d["source_id"],))
            src_row = cur.fetchone()
            if src_row:
                full_content = src_row["content"] or ""
                d["content"] = full_content[d["start_char"] : d["end_char"]]
            else:
                d["content"] = ""

            d["id"] = str(d["section_id"])
            d["source_id"] = str(d["source_id"])
            d["type"] = "source_section"
            d["title"] = d.get("section_title") or d.get("source_title") or "Untitled"
            d["confidence"] = {"overall": float(d.get("reliability", 0.5))}

            _extract_rrf_scores(d, rrf_min, rrf_range)
            freshness_days = _compute_freshness_days(d)
            d["freshness"] = freshness_days
            d["freshness_score"] = _freshness_score(freshness_days)
            d["active_contentions"] = False
            d["provenance_summary"] = {"source_count": 1, "relationship_types": []}

            # Clean up internal keys
            for key in ("section_id", "source_title", "section_title", "reliability"):
                d.pop(key, None)

            results.append(d)

        return results


# ---------------------------------------------------------------------------
# Unified retrieval
# ---------------------------------------------------------------------------


def _retrieve_sync(
    query: str,
    limit: int,
    include_sources: bool,
    session_id: str | None,
    temporal_mode: str = "default",
) -> list[dict[str, Any]]:
    """Synchronous implementation of unified retrieval."""
    article_results = _search_articles_sync(query, limit)

    source_results: list[dict[str, Any]] = []
    if include_sources:
        # Prefer tree-indexed section search; fall back to whole-source search
        source_results = _search_source_sections_sync(query, limit)
        if not source_results:
            source_results = _search_ungrouped_sources_sync(query, limit)

    all_results = article_results + source_results
    if not all_results:
        return []

    # multi_signal_rank uses created_at for recency — substitute freshness_days
    # so the built-in exponential decay reproduces our freshness_score.
    now = datetime.now(UTC)
    for r in all_results:
        freshness_days = r.get("freshness", 90.0)
        r["_original_created_at"] = r.get("created_at")
        r["created_at"] = (now - timedelta(days=freshness_days)).isoformat()

    weights = TEMPORAL_WEIGHT_PRESETS.get(temporal_mode, TEMPORAL_WEIGHT_PRESETS["default"])
    ranked = multi_signal_rank(
        all_results,
        semantic_weight=weights["semantic"],
        confidence_weight=weights["confidence"],
        recency_weight=weights["recency"],
    )

    # Restore original created_at
    for r in ranked:
        orig = r.pop("_original_created_at", None)
        if orig is not None:
            r["created_at"] = orig

    ranked = ranked[:limit]

    # Apply rank floor to archived articles
    for r in ranked:
        if r.get("status") == "archived" and r.get("final_score", 0) > ARCHIVED_RANK_FLOOR:
            r["final_score"] = ARCHIVED_RANK_FLOOR

    # Grace period novelty boost: new articles get a decaying score multiplier
    grace_cutoff = now - timedelta(hours=GRACE_PERIOD_HOURS)
    for r in ranked:
        created_at = r.get("created_at")
        if not created_at:
            continue
        try:
            if isinstance(created_at, str):
                # Parse ISO format, handle with/without timezone
                ca = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                if ca.tzinfo is None:
                    ca = ca.replace(tzinfo=UTC)
            else:
                ca = created_at
                if ca.tzinfo is None:
                    ca = ca.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue
        if ca > grace_cutoff:
            # Linear decay: full boost at creation, 1.0x at grace_cutoff
            elapsed = (now - ca).total_seconds()
            grace_seconds = GRACE_PERIOD_HOURS * 3600
            decay = max(0.0, 1.0 - elapsed / grace_seconds)
            boost = 1.0 + (NOVELTY_BOOST_MAX - 1.0) * decay
            r["final_score"] = r.get("final_score", 0) * boost

    # Re-sort after score adjustments (rank floor + boost)
    ranked.sort(key=lambda r: r.get("final_score", 0), reverse=True)

    # Side effects: usage traces + recompile queue
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
    temporal_mode: str = "default",
) -> ValenceResponse:
    """Unified retrieval: search articles and ungrouped sources.

    Ranks results by: relevance * 0.5 + confidence * 0.35 + freshness * 0.15.
    Use temporal_mode to adjust ranking weights:
      - "default":       semantic=0.50, confidence=0.35, recency=0.15
      - "prefer_recent": semantic=0.35, confidence=0.15, recency=0.50
      - "prefer_stable": semantic=0.40, confidence=0.45, recency=0.15

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
        ValenceResponse with data = list of result dicts, each containing:
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
        return ok(data=[])

    limit = max(1, min(limit, 200))

    results = await asyncio.to_thread(
        _retrieve_sync,
        query,
        limit,
        include_sources,
        session_id,
        temporal_mode,
    )
    return ok(data=results)
