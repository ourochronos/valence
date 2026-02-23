"""MCP tool wrappers for source ingestion and retrieval (C1).

All functions are synchronous (conforming to the MCP dispatch protocol).
Business-logic constants and helpers are imported from ``valence.core.sources``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...core.sources import (
    RELIABILITY_DEFAULTS,
    VALID_SOURCE_TYPES,
    _compute_fingerprint,
    _row_to_dict,
)
from ._common import get_cursor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool implementations (sync)
# ---------------------------------------------------------------------------


def source_ingest(
    content: str,
    source_type: str,
    title: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
) -> dict[str, Any]:
    """Ingest a new source into the knowledge substrate.

    Assigns a reliability score by type, generates a SHA-256 fingerprint
    for deduplication. Embedding is deferred to first retrieval.

    Returns a result dict with ``success`` key; on duplicate returns
    ``{"success": False, "duplicate": True, "existing_id": ...}``.
    """
    if not content or not content.strip():
        return {"success": False, "error": "content must be non-empty"}

    if source_type not in VALID_SOURCE_TYPES:
        return {
            "success": False,
            "error": f"Invalid source_type '{source_type}'. Must be one of: {', '.join(sorted(VALID_SOURCE_TYPES))}",
        }

    fingerprint = _compute_fingerprint(content)
    reliability = RELIABILITY_DEFAULTS.get(source_type, 0.5)
    metadata_json = json.dumps(metadata or {})

    with get_cursor() as cur:
        cur.execute(
            "SELECT id FROM sources WHERE fingerprint = %s LIMIT 1",
            (fingerprint,),
        )
        existing = cur.fetchone()
        if existing:
            return {
                "success": False,
                "duplicate": True,
                "existing_id": str(existing["id"]),
                "error": "Duplicate source: fingerprint already exists",
            }

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
                fingerprint,
                metadata_json,
            ),
        )
        row = cur.fetchone()

    source = _row_to_dict(row)
    return {"success": True, "source": source}


def source_get(source_id: str) -> dict[str, Any]:
    """Retrieve a source by UUID.

    Returns ``{"success": True, "source": {...}}`` or
    ``{"success": False, "error": "Source not found: <id>"}``.
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
        return {"success": False, "error": f"Source not found: {source_id}"}

    return {"success": True, "source": _row_to_dict(row)}


def source_search(query: str, limit: int = 20) -> dict[str, Any]:
    """Full-text search over source content.

    Uses the ``content_tsv`` GIN index and ``websearch_to_tsquery`` for
    phrase/keyword matching.  Results are ordered by relevance descending.
    """
    if not query or not query.strip():
        return {"success": True, "sources": [], "total_count": 0}

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

    sources = []
    for row in rows:
        d = _row_to_dict(dict(row))
        d["rank"] = float(row.get("rank", 0.0))
        sources.append(d)

    return {"success": True, "sources": sources, "total_count": len(sources)}


def source_list(
    source_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """List sources, optionally filtered by type.

    Results are ordered by ``created_at`` descending.
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

    sources = [_row_to_dict(row) for row in rows]
    return {"success": True, "sources": sources, "total_count": len(sources)}
