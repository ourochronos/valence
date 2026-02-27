# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Source tool handlers."""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from valence.core.db import get_cursor, serialize_row

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run an async coroutine from sync context.

    Uses the existing event loop if one is running, otherwise creates a new one.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Shouldn't happen in MCP sync handler context — fall through to asyncio.run
            raise RuntimeError("Event loop already running")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def source_ingest(
    content: str,
    source_type: str,
    title: str | None = None,
    url: str | None = None,
    metadata: dict | None = None,
    supersedes: str | None = None,
) -> dict[str, Any]:
    """Ingest a new source into the knowledge substrate.

    Delegates to core/sources.py:ingest_source() which handles deduplication,
    storage, and the full post-ingest pipeline (tree-index → embed → compose →
    auto-compile).  This is the single ingest entry point; the pipeline is
    wired in core, not here.
    """
    from valence.core.sources import ingest_source

    result = _run_async(
        ingest_source(
            content=content,
            source_type=source_type,
            title=title,
            url=url,
            metadata=metadata,
            supersedes=supersedes,
        )
    )

    if not result.success:
        response: dict[str, Any] = {"success": False, "error": result.error}
        if result.error and "Duplicate source" in result.error:
            response["duplicate"] = True
            # Extract existing_id from error message: "... (existing_id=<uuid>)"
            m = re.search(r"existing_id=([0-9a-f-]+)", result.error or "")
            if m:
                response["existing_id"] = m.group(1)
        return response

    return {"success": True, "source": result.data}


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
