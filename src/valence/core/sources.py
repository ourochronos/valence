# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Source ingestion and retrieval for the Valence v2 knowledge system (C1).

Sources are immutable raw knowledge inputs — documents, conversations, web pages,
code snippets, observations, tool outputs, and user inputs.

Embedding computation is intentionally deferred to first retrieval need (WU-12).
Duplicates are detected via SHA-256 fingerprint and rejected idempotently.

WU-14: All public functions now return ValenceResponse (C12, DR-10).
"""

from __future__ import annotations

import hashlib
import json
import logging

from valence.core.db import get_cursor, serialize_row

from .response import ValenceResponse, err, ok

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SOURCE_TYPES = frozenset(["document", "conversation", "web", "code", "observation", "tool_output", "user_input"])

# Reliability defaults per source type (C1 spec)
RELIABILITY_DEFAULTS: dict[str, float] = {
    "document": 0.8,
    "code": 0.8,
    "web": 0.6,
    "conversation": 0.5,
    "observation": 0.4,
    "tool_output": 0.7,
    "user_input": 0.75,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_fingerprint(content: str) -> str:
    """Return SHA-256 hex digest of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Public API — all return ValenceResponse (WU-14)
# ---------------------------------------------------------------------------


async def ingest_source(
    content: str,
    source_type: str,
    title: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
    supersedes: str | None = None,
) -> ValenceResponse:
    """Ingest a source into the knowledge substrate.

    Stores content, assigns a reliability score by source type, generates a
    SHA-256 fingerprint for deduplication, and returns the stored record.
    Embedding computation is deferred to first retrieval need.

    Args:
        content: Raw text content of the source.
        source_type: One of document, conversation, web, code, observation,
            tool_output, user_input.
        title: Optional human-readable title.
        url: Optional canonical URL for the source.
        metadata: Optional JSON-serialisable metadata dict.
        supersedes: Optional UUID of a source this one supersedes (for updates).

    Returns:
        ValenceResponse with data = source dict on success.
        On validation failure: success=False, error describes the problem.
        On duplicate fingerprint: success=False, error contains existing_id.
    """
    if not content or not content.strip():
        return err("content must be non-empty")

    if source_type not in VALID_SOURCE_TYPES:
        return err(f"Invalid source_type '{source_type}'. Must be one of: {', '.join(sorted(VALID_SOURCE_TYPES))}")

    fingerprint = _compute_fingerprint(content)
    reliability = RELIABILITY_DEFAULTS.get(source_type, 0.5)
    metadata_json = json.dumps(metadata or {})

    with get_cursor() as cur:
        # Reject duplicates by fingerprint
        cur.execute(
            "SELECT id FROM sources WHERE fingerprint = %s LIMIT 1",
            (fingerprint,),
        )
        existing = cur.fetchone()
        if existing:
            existing_id = str(existing["id"])
            return err(f"Duplicate source: fingerprint {fingerprint!r} already exists (existing_id={existing_id})")

        # Validate supersedes target exists if provided
        if supersedes:
            cur.execute(
                "SELECT id FROM sources WHERE id = %s::uuid",
                (supersedes,),
            )
            if not cur.fetchone():
                return err(f"Superseded source not found: {supersedes}")

        cur.execute(
            """
            INSERT INTO sources (type, title, url, content, fingerprint, reliability,
                                 content_hash, metadata, supersedes_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::uuid)
            RETURNING id, type, title, url, content, fingerprint, reliability,
                      content_hash, metadata, created_at, supersedes_id
            """,
            (
                source_type,
                title,
                url,
                content,
                fingerprint,
                reliability,
                fingerprint,  # content_hash = fingerprint for backward compat
                metadata_json,
                supersedes,
            ),
        )
        row = cur.fetchone()

    result = serialize_row(row)
    if supersedes:
        logger.info(
            "Ingested source id=%s type=%s fingerprint=%s, supersedes=%s",
            result["id"],
            source_type,
            fingerprint[:8],
            supersedes,
        )
    else:
        logger.info("Ingested source id=%s type=%s fingerprint=%s", result["id"], source_type, fingerprint[:8])

    # Run post-ingest pipeline (embed + tree + auto-compile)
    source_id = result["id"]
    try:
        from valence.core.ingest_pipeline import run_source_pipeline

        pipeline_result = await run_source_pipeline(source_id, batch_mode=False)
        if pipeline_result.success:
            result["pipeline"] = pipeline_result.data
        else:
            logger.warning("Ingest pipeline failed for %s: %s", source_id, pipeline_result.error)
            result["pipeline_error"] = pipeline_result.error
    except Exception as exc:
        logger.warning("Ingest pipeline raised for %s: %s", source_id, exc)
        result["pipeline_error"] = str(exc)

    return ok(data=result)


async def get_source(source_id: str) -> ValenceResponse:
    """Retrieve a source by ID.

    Args:
        source_id: UUID string of the source.

    Returns:
        ValenceResponse with data = source dict on success, or error when not found.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, type, title, url, content, fingerprint, reliability,
                   content_hash, metadata, created_at
            FROM sources
            WHERE id = %s
            """,
            (source_id,),
        )
        row = cur.fetchone()

    if row is None:
        return err(f"Source not found: {source_id}")

    return ok(data=serialize_row(row))


async def search_sources(query: str, limit: int = 20) -> ValenceResponse:
    """Full-text search over source content via the content_tsv column.

    Args:
        query: Search terms (passed to websearch_to_tsquery).
        limit: Maximum results to return (default 20).

    Returns:
        ValenceResponse with data = list of source dicts ordered by relevance.
        Empty list when query is blank.
    """
    if not query or not query.strip():
        return ok(data=[])

    limit = max(1, min(limit, 200))

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, type, title, url, content, fingerprint, reliability,
                   content_hash, metadata, created_at,
                   ts_rank(content_tsv, websearch_to_tsquery('english', %s)) AS rank
            FROM sources
            WHERE content_tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, limit),
        )
        rows = cur.fetchall()

    results = []
    for row in rows:
        d = serialize_row(dict(row))
        d["rank"] = float(row.get("rank", 0.0))
        results.append(d)
    return ok(data=results)


async def list_sources(
    source_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ValenceResponse:
    """List sources, optionally filtered by type.

    Args:
        source_type: Optional filter (e.g. 'document', 'web').
        limit: Page size (default 50, max 200).
        offset: Pagination offset (default 0).

    Returns:
        ValenceResponse with data = list of source dicts ordered by created_at
        descending.
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    with get_cursor() as cur:
        if source_type:
            cur.execute(
                """
                SELECT id, type, title, url, content, fingerprint, reliability,
                       content_hash, metadata, created_at
                FROM sources
                WHERE type = %s
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (source_type, limit, offset),
            )
        else:
            cur.execute(
                """
                SELECT id, type, title, url, content, fingerprint, reliability,
                       content_hash, metadata, created_at
                FROM sources
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
                """,
                (limit, offset),
            )
        rows = cur.fetchall()

    return ok(data=[serialize_row(row) for row in rows])


async def find_similar_ungrouped(
    source_id: str,
    threshold: float = 0.8,
    limit: int = 20,
) -> list[str]:
    """Find ungrouped sources whose embedding is cosine-similar to *source_id*.

    "Ungrouped" means the source has no rows in article_sources (it is not yet
    part of any compiled article).  Sources that are already grouped are excluded
    so we do not double-compile them.

    Args:
        source_id: UUID of the reference source.
        threshold: Minimum cosine similarity (1 − cosine distance) to include.
        limit: Maximum number of similar source IDs to return.

    Returns:
        Ordered list of similar source UUIDs (excluding *source_id* itself),
        closest first.  Empty list if the source has no embedding or no similar
        ungrouped neighbours are found.
    """
    with get_cursor() as cur:
        cur.execute("SELECT embedding::text FROM sources WHERE id = %s", (source_id,))
        row = cur.fetchone()

    if not row:
        return []

    raw_emb = row[0] if isinstance(row, tuple) else row.get("embedding") or ""
    raw_emb = str(raw_emb).strip()
    if not raw_emb or raw_emb == "None":
        return []

    # vec_str is already in pgvector "[…]" format straight from the DB cast
    vec_str = raw_emb
    try:
        with get_cursor() as cur:
            cur.execute(
                """
                SELECT s.id::text
                FROM sources s
                WHERE s.id != %s::uuid
                  AND s.embedding IS NOT NULL
                  AND s.pipeline_status IN ('indexed', 'complete')
                  AND NOT EXISTS (
                      SELECT 1 FROM article_sources asrc WHERE asrc.source_id = s.id
                  )
                  AND 1 - (s.embedding <=> %s::vector) > %s
                ORDER BY s.embedding <=> %s::vector
                LIMIT %s
                """,
                (source_id, vec_str, threshold, vec_str, limit),
            )
            rows = cur.fetchall()
    except Exception as exc:
        logger.warning("find_similar_ungrouped query failed for %s: %s", source_id, exc)
        return []

    return [r[0] if isinstance(r, tuple) else str(r.get("id", "")) for r in rows if r]
