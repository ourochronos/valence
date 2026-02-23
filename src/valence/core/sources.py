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
from typing import Any

from valence.core.db import get_cursor

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


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a DB row (RealDictRow or dict) to a plain dict with serialisable values."""
    d = dict(row)
    # Ensure UUID fields are strings
    for key in ("id", "session_id"):
        if d.get(key) is not None:
            d[key] = str(d[key])
    # Ensure datetime is string-serialisable
    if d.get("created_at") is not None:
        d["created_at"] = d["created_at"].isoformat() if hasattr(d["created_at"], "isoformat") else str(d["created_at"])
    # Ensure metadata is a plain dict
    if d.get("metadata") is not None and not isinstance(d["metadata"], dict):
        try:
            d["metadata"] = json.loads(d["metadata"])
        except Exception:
            d["metadata"] = {}
    # Strip the generated TSV column — callers don't need it
    d.pop("content_tsv", None)
    # Strip embedding bytes — not useful in Python dicts
    d.pop("embedding", None)
    return d


# ---------------------------------------------------------------------------
# Public API — all return ValenceResponse (WU-14)
# ---------------------------------------------------------------------------


async def ingest_source(
    content: str,
    source_type: str,
    title: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
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

        cur.execute(
            """
            INSERT INTO sources (type, title, url, content, fingerprint, reliability,
                                 content_hash, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            RETURNING id, type, title, url, content, fingerprint, reliability,
                      content_hash, metadata, created_at
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
            ),
        )
        row = cur.fetchone()

    result = _row_to_dict(row)
    logger.info("Ingested source id=%s type=%s fingerprint=%s", result["id"], source_type, fingerprint[:8])
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

    return ok(data=_row_to_dict(row))


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
        d = _row_to_dict(dict(row))
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

    return ok(data=[_row_to_dict(row) for row in rows])
