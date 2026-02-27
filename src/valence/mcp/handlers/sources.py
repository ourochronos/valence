# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Source tool handlers."""

from __future__ import annotations

import json
import logging
from typing import Any

from valence.core.db import get_cursor, serialize_row
from valence.core.sources import (
    RELIABILITY_DEFAULTS,
    VALID_SOURCE_TYPES,
    _compute_fingerprint,
)

logger = logging.getLogger(__name__)


def source_ingest(
    content: str,
    source_type: str,
    title: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Ingest a new source into the knowledge substrate."""
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

        if supersedes:
            cur.execute("SELECT id FROM sources WHERE id = %s::uuid", (supersedes,))
            if not cur.fetchone():
                return {"success": False, "error": f"Superseded source not found: {supersedes}"}

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
                fingerprint,
                metadata_json,
                supersedes,
            ),
        )
        row = cur.fetchone()

    source = serialize_row(row)

    # Run post-ingest pipeline synchronously via run_async
    source_id = source["id"]
    try:
        import asyncio
        from valence.core.ingest_pipeline import run_source_pipeline

        pipeline_result = asyncio.run(run_source_pipeline(source_id, batch_mode=False))
        if pipeline_result.success:
            source["pipeline"] = pipeline_result.data
        else:
            logger.warning("Ingest pipeline failed for %s: %s", source_id, pipeline_result.error)
            source["pipeline_error"] = pipeline_result.error
    except Exception as exc:
        logger.warning("Ingest pipeline raised for %s: %s", source_id, exc)
        source["pipeline_error"] = str(exc)

    return {"success": True, "source": source}


def source_get(source_id: str) -> dict[str, Any]:
    """Retrieve a source by UUID."""
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

    if not row:
        return {"success": False, "error": f"Source not found: {source_id}"}

    source = serialize_row(row)
    return {"success": True, "source": source}


def source_search(query: str, limit: int = 20) -> dict[str, Any]:
    """Full-text search over source content."""
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 200))

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, type, title, url, content, fingerprint, reliability,
                   content_hash, metadata, created_at,
                   ts_rank_cd(content_tsv, websearch_to_tsquery('english', %s)) AS rank
            FROM sources
            WHERE content_tsv @@ websearch_to_tsquery('english', %s)
            ORDER BY rank DESC
            LIMIT %s
            """,
            (query, query, limit),
        )
        rows = cur.fetchall()

    sources = [serialize_row(row) for row in rows]
    return {"success": True, "sources": sources, "total_count": len(sources)}


def source_list(limit: int = 50) -> dict[str, Any]:
    """List recent sources."""
    limit = max(1, min(int(limit), 200))

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, type, title, url, fingerprint, reliability, created_at, metadata
            FROM sources
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()

    sources = []
    for row in rows:
        d = dict(row)
        d["id"] = str(d["id"])
        if d.get("created_at"):
            d["created_at"] = d["created_at"].isoformat()
        sources.append(d)

    return {"success": True, "sources": sources, "total_count": len(sources)}
