# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Tree-section embedding pipeline for sources.

Every source gets a tree index (via build_tree_index), then each tree node —
leaf through root — gets an embedding of its actual content slice.  This gives
multi-granularity vector search: leaf nodes match specific topics, branch
nodes match broader themes.

Pipeline:
    1. Ensure source has a tree_index (build if missing)
    2. Flatten tree to nodes with tree_path ("0", "0.2", "0.2.1")
    3. For each node, embed source.content[start_char:end_char]
    4. Store in source_sections table

Usage:
    count = await embed_source_sections(source_id)
    total = await embed_all_sources(batch_size=10)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from valence.core.db import get_cursor
from valence.core.embeddings import generate_embedding, vector_to_pgvector
from valence.core.response import ValenceResponse, err, ok

logger = logging.getLogger(__name__)


def _flatten_tree(
    nodes: list[dict[str, Any]],
    prefix: str = "",
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Flatten a tree_index into a list of nodes with tree_path and depth.

    Each node gets:
        tree_path: "0", "1", "0.0", "0.1", "0.0.2" etc.
        depth: 0 for root-level, 1 for children, etc.
        title, summary, start_char, end_char from the tree node.
    """
    flat: list[dict[str, Any]] = []
    for i, node in enumerate(nodes):
        path = f"{prefix}{i}" if not prefix else f"{prefix}.{i}"
        flat.append(
            {
                "tree_path": path,
                "depth": depth,
                "title": node.get("title", ""),
                "summary": node.get("summary", ""),
                "start_char": node.get("start_char", 0),
                "end_char": node.get("end_char", 0),
            }
        )
        children = node.get("children", [])
        if children:
            flat.extend(_flatten_tree(children, prefix=path, depth=depth + 1))
    return flat


def _embed_and_upsert(
    source_id: str,
    content: str,
    flat_nodes: list[dict[str, Any]],
) -> int:
    """Synchronous: embed each section's content slice and upsert to DB.

    Returns number of sections embedded.
    """
    count = 0
    with get_cursor() as cur:
        for node in flat_nodes:
            start = node["start_char"]
            end = node["end_char"]
            slice_text = content[start:end].strip()

            if not slice_text:
                continue

            content_hash = hashlib.md5(slice_text.encode()).hexdigest()

            # Check if section already exists and is current
            cur.execute(
                """
                SELECT id, content_hash FROM source_sections
                WHERE source_id = %s AND tree_path = %s
                """,
                (source_id, node["tree_path"]),
            )
            existing = cur.fetchone()
            if existing and existing["content_hash"] == content_hash:
                count += 1
                continue

            # Generate embedding
            try:
                vec = generate_embedding(slice_text)
                vec_str = vector_to_pgvector(vec)
            except Exception as exc:
                logger.warning(
                    "Embedding failed for source %s path %s: %s",
                    source_id,
                    node["tree_path"],
                    exc,
                )
                continue

            # Upsert section
            cur.execute(
                """
                INSERT INTO source_sections
                    (source_id, tree_path, title, summary,
                     start_char, end_char, depth,
                     embedding, content_hash)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                ON CONFLICT (source_id, tree_path) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    start_char = EXCLUDED.start_char,
                    end_char = EXCLUDED.end_char,
                    depth = EXCLUDED.depth,
                    embedding = EXCLUDED.embedding,
                    content_hash = EXCLUDED.content_hash
                """,
                (
                    source_id,
                    node["tree_path"],
                    node["title"],
                    node["summary"],
                    start,
                    end,
                    node["depth"],
                    vec_str,
                    content_hash,
                ),
            )
            count += 1
    return count


async def embed_source_sections(source_id: str) -> ValenceResponse:
    """Build tree index and embed all sections for a single source.

    Returns ValenceResponse with count of sections embedded.
    """
    try:
        # Fetch source
        with get_cursor() as cur:
            cur.execute(
                "SELECT id, content, title, metadata FROM sources WHERE id = %s",
                (source_id,),
            )
            row = cur.fetchone()
            if not row:
                return err(f"Source not found: {source_id}")

        source = dict(row)
        content = source["content"] or ""
        metadata = source["metadata"] or {}

        if not content.strip():
            return ok(data={"source_id": source_id, "sections_embedded": 0})

        # Step 1: Get tree index if it exists (don't build — that's a separate operation)
        tree_index = metadata.get("tree_index")
        if isinstance(tree_index, dict):
            # Handle {"nodes": [...]} wrapper format
            tree_index = tree_index.get("nodes", [])
        if not tree_index or not isinstance(tree_index, list):
            # No tree — fall back to single section covering entire content
            tree_index = [
                {
                    "title": source.get("title") or "Full content",
                    "summary": "",
                    "start_char": 0,
                    "end_char": len(content),
                    "children": [],
                }
            ]

        # Step 2: Flatten tree
        flat_nodes = _flatten_tree(tree_index)
        if not flat_nodes:
            return ok(data={"source_id": source_id, "sections_embedded": 0})

        # Step 3: Embed and upsert (sync DB/HTTP in thread pool)
        embedded = await asyncio.to_thread(_embed_and_upsert, source_id, content, flat_nodes)

        logger.info(
            "Embedded %d/%d sections for source %s",
            embedded,
            len(flat_nodes),
            source_id,
        )
        return ok(data={"source_id": source_id, "sections_embedded": embedded})

    except Exception as exc:
        logger.error("embed_source_sections failed for %s: %s", source_id, exc)
        return err(f"Failed to embed source sections: {exc}")


async def embed_all_sources(batch_size: int = 10) -> ValenceResponse:
    """Embed tree sections for all sources that need it.

    A source needs embedding if it has no rows in source_sections.

    Args:
        batch_size: Max sources to process per run.

    Returns:
        ValenceResponse with counts.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT s.id
            FROM sources s
            WHERE s.content IS NOT NULL
              AND s.content != ''
              AND NOT EXISTS (
                  SELECT 1 FROM source_sections ss WHERE ss.source_id = s.id
              )
            ORDER BY s.created_at DESC
            LIMIT %s
            """,
            (batch_size,),
        )
        source_ids = [str(r["id"]) for r in cur.fetchall()]

    if not source_ids:
        return ok(data={"processed": 0, "total_sections": 0, "remaining": 0})

    total_sections = 0
    errors = 0
    for sid in source_ids:
        result = await embed_source_sections(sid)
        if result.success:
            total_sections += result.data.get("sections_embedded", 0)
        else:
            errors += 1
            logger.warning("Failed to embed source %s: %s", sid, result.error)

    # Count remaining
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) as n FROM sources s
            WHERE s.content IS NOT NULL
              AND s.content != ''
              AND NOT EXISTS (
                  SELECT 1 FROM source_sections ss WHERE ss.source_id = s.id
              )
            """
        )
        remaining = cur.fetchone()["n"]

    return ok(
        data={
            "processed": len(source_ids),
            "total_sections": total_sections,
            "errors": errors,
            "remaining": remaining,
        }
    )
