# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Unified source ingest pipeline (issue #565).

Pipeline stages (dependency-ordered):
    1. Tree-index   — trivial single-node tree for small sources, LLM tree for large
    2. Embed sections — embed each tree node's content slice
    3. Compose source embedding — mean of section vectors (no truncation)
    4. Check compilation candidates — auto-compile clusters of 3+ ungrouped sources

Usage::
    result = await run_source_pipeline(source_id)
    # result.data = {"source_id": ..., "sections": N, "compiled": bool}

Invocation modes:
    - Single ingest: called inline before returning to the caller
    - Batch: enqueued as 'source_pipeline' mutation_queue task

Pipeline status on sources.pipeline_status:
    pending  → inserted, pipeline not started
    indexed  → tree built, sections embedded, source embedding composed
    complete → compilation check done
    failed   → pipeline failed at some stage
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from valence.core.db import get_cursor
from valence.core.embeddings import generate_embedding, vector_to_pgvector
from valence.core.response import ValenceResponse, err, ok
from valence.core.section_embeddings import _flatten_tree, _embed_and_upsert

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sources smaller than this (chars) get a trivial single-node tree
SMALL_SOURCE_CHARS = 7000  # ~2000 tokens

# Minimum cluster size to trigger auto-compilation
AUTO_COMPILE_MIN_CLUSTER = 3

# Cosine similarity threshold for compilation candidates
COMPILE_SIMILARITY_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Return element-wise mean of a list of equal-length float vectors."""
    if not vectors:
        return []
    n = len(vectors)
    dims = len(vectors[0])
    result = [0.0] * dims
    for vec in vectors:
        for i, v in enumerate(vec):
            result[i] += v
    return [x / n for x in result]


def _set_pipeline_status(source_id: str, status: str) -> None:
    """Update pipeline_status on the source row."""
    with get_cursor() as cur:
        cur.execute(
            "UPDATE sources SET pipeline_status = %s WHERE id = %s",
            (status, source_id),
        )


def _get_section_vectors(source_id: str) -> list[list[float]]:
    """Fetch all section embeddings for a source from source_sections."""
    with get_cursor() as cur:
        cur.execute(
            "SELECT embedding::text FROM source_sections WHERE source_id = %s",
            (source_id,),
        )
        rows = cur.fetchall()

    vectors = []
    for row in rows:
        raw = row[0] if isinstance(row, tuple) else row.get("embedding") or row.get(0)
        if raw is None:
            continue
        # pgvector returns strings like "[0.1,0.2,...]"
        raw = str(raw).strip()
        if raw.startswith("["):
            try:
                vec = [float(x) for x in raw[1:-1].split(",")]
                vectors.append(vec)
            except (ValueError, IndexError):
                pass
    return vectors


def _update_source_embedding(source_id: str, embedding: list[float]) -> None:
    """Write the composed source-level embedding to sources.embedding."""
    vec_str = vector_to_pgvector(embedding)
    with get_cursor() as cur:
        cur.execute(
            "UPDATE sources SET embedding = %s::vector WHERE id = %s",
            (vec_str, source_id),
        )


def _enqueue_pipeline_task(source_id: str) -> None:
    """Add a source_pipeline task to mutation_queue for batch processing."""
    payload = json.dumps({"source_id": source_id})
    with get_cursor() as cur:
        # mutation_queue.article_id is NOT NULL in the schema — use a sentinel
        # We'll store source_id in payload and use a dummy uuid for article_id.
        # Actually check schema first — some versions allow NULL for source ops.
        try:
            cur.execute(
                """
                INSERT INTO mutation_queue (operation, article_id, priority, payload)
                VALUES ('source_pipeline', NULL, 2, %s::jsonb)
                """,
                (payload,),
            )
        except Exception:
            # Fall back: article_id may be NOT NULL; pass source's own id cast
            cur.execute(
                """
                INSERT INTO mutation_queue (operation, article_id, priority, payload)
                SELECT 'source_pipeline', %s::uuid, 2, %s::jsonb
                """,
                (source_id, payload),
            )


# ---------------------------------------------------------------------------
# Stage 2: Tree index (build or trivial)
# ---------------------------------------------------------------------------


async def _stage_tree_index(source_id: str, content: str, metadata: dict) -> list[dict]:
    """Return tree nodes for the source, building if needed."""
    existing_tree = metadata.get("tree_index")
    if isinstance(existing_tree, dict):
        existing_tree = existing_tree.get("nodes", [])

    if existing_tree and isinstance(existing_tree, list):
        return existing_tree

    if len(content) < SMALL_SOURCE_CHARS:
        # Trivial single-node tree for small sources — no LLM call
        return [
            {
                "title": metadata.get("title") or "Full content",
                "summary": "",
                "start_char": 0,
                "end_char": len(content),
                "children": [],
            }
        ]

    # Large source — call build_tree_index (may do LLM call)
    try:
        from valence.core.tree_index import build_tree_index

        result = await build_tree_index(source_id=source_id, force=False)
        if result.success:
            # Re-read from DB (build_tree_index stores in metadata)
            with get_cursor() as cur:
                cur.execute("SELECT metadata FROM sources WHERE id = %s", (source_id,))
                row = cur.fetchone()
            if row:
                meta = row["metadata"] if hasattr(row, "__getitem__") else {}
                if isinstance(meta, str):
                    meta = json.loads(meta)
                tree = meta.get("tree_index", [])
                if isinstance(tree, dict):
                    tree = tree.get("nodes", [])
                if tree:
                    return tree
        # Fall back to trivial tree on LLM failure
        logger.warning("Tree build failed for %s, falling back to single-node tree", source_id)
    except Exception as exc:
        logger.warning("build_tree_index raised for %s: %s — using single-node tree", source_id, exc)

    return [
        {
            "title": "Full content",
            "summary": "",
            "start_char": 0,
            "end_char": len(content),
            "children": [],
        }
    ]


# ---------------------------------------------------------------------------
# Stage 5: Auto-compilation check
# ---------------------------------------------------------------------------


async def _stage_auto_compile(source_id: str) -> bool:
    """Find ungrouped sources similar to this one and auto-compile clusters.

    Returns True if at least one compilation was triggered.
    """
    # Get this source's embedding
    with get_cursor() as cur:
        cur.execute("SELECT embedding::text FROM sources WHERE id = %s", (source_id,))
        row = cur.fetchone()
    if not row:
        return False

    raw_emb = row[0] if isinstance(row, tuple) else (row.get("embedding") or "")
    raw_emb = str(raw_emb).strip()
    if not raw_emb or raw_emb == "None":
        return False

    # Find ungrouped sources with cosine similarity > threshold
    vec_str = raw_emb  # already in pgvector format
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
                LIMIT 20
                """,
                (source_id, vec_str, COMPILE_SIMILARITY_THRESHOLD, vec_str),
            )
            similar_rows = cur.fetchall()
    except Exception as exc:
        logger.warning("Auto-compile similarity query failed: %s", exc)
        return False

    similar_ids = [
        r[0] if isinstance(r, tuple) else str(r.get("id", ""))
        for r in similar_rows
        if r
    ]

    # Check if this source itself is ungrouped
    with get_cursor() as cur:
        cur.execute(
            "SELECT 1 FROM article_sources WHERE source_id = %s::uuid LIMIT 1",
            (source_id,),
        )
        already_grouped = cur.fetchone() is not None

    if not already_grouped:
        candidate_ids = [source_id] + similar_ids
    else:
        candidate_ids = similar_ids

    if len(candidate_ids) < AUTO_COMPILE_MIN_CLUSTER:
        logger.debug(
            "Auto-compile: only %d candidates for %s (need %d), skipping",
            len(candidate_ids),
            source_id,
            AUTO_COMPILE_MIN_CLUSTER,
        )
        return False

    # Compile the cluster
    try:
        from valence.core.compilation import compile_article

        result = await compile_article(candidate_ids[:10])  # cap at 10
        if result.success:
            logger.info(
                "Auto-compiled %d sources (including %s) → article %s",
                len(candidate_ids),
                source_id,
                (result.data or {}).get("id", "?"),
            )
            return True
        else:
            logger.warning("Auto-compile failed: %s", result.error)
    except Exception as exc:
        logger.warning("Auto-compile raised: %s", exc)

    return False


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


async def run_source_pipeline(
    source_id: str,
    batch_mode: bool = False,
) -> ValenceResponse:
    """Run the full ingest pipeline for a source.

    Stages:
        1. Fetch source
        2. Tree-index (trivial for small, LLM for large)
        3. Embed sections (embed each tree node content slice)
        4. Compose source embedding (mean of section vectors)
        5. Auto-compilation check

    Args:
        source_id: UUID of the source to process.
        batch_mode: If True, enqueue as mutation_queue task instead of running inline.

    Returns:
        ValenceResponse with data = {source_id, sections_embedded, compiled}.
    """
    if batch_mode:
        try:
            await asyncio.to_thread(_enqueue_pipeline_task, source_id)
            return ok(data={"source_id": source_id, "queued": True})
        except Exception as exc:
            logger.error("Failed to enqueue pipeline task for %s: %s", source_id, exc)
            return err(f"Failed to enqueue pipeline task: {exc}")

    # --- Fetch source ---
    try:
        with get_cursor() as cur:
            cur.execute(
                "SELECT id, content, title, metadata FROM sources WHERE id = %s",
                (source_id,),
            )
            row = cur.fetchone()
        if not row:
            return err(f"Source not found: {source_id}")
    except Exception as exc:
        return err(f"DB error fetching source: {exc}")

    content = row["content"] or ""
    metadata = row["metadata"] or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except Exception:
            metadata = {}

    if not content.strip():
        _set_pipeline_status(source_id, "complete")
        return ok(data={"source_id": source_id, "sections_embedded": 0, "compiled": False})

    # --- Stage 2: Tree index ---
    try:
        tree_nodes = await _stage_tree_index(source_id, content, metadata)
    except Exception as exc:
        logger.error("Tree index stage failed for %s: %s", source_id, exc)
        _set_pipeline_status(source_id, "failed")
        return err(f"Tree index stage failed: {exc}")

    # --- Stage 3: Embed sections ---
    flat_nodes = _flatten_tree(tree_nodes)
    try:
        sections_embedded = await asyncio.to_thread(
            _embed_and_upsert, source_id, content, flat_nodes
        )
    except Exception as exc:
        logger.error("Section embedding stage failed for %s: %s", source_id, exc)
        _set_pipeline_status(source_id, "failed")
        return err(f"Section embedding stage failed: {exc}")

    # --- Stage 4: Compose source embedding ---
    try:
        section_vectors = await asyncio.to_thread(_get_section_vectors, source_id)
        if section_vectors:
            composed = _mean_vector(section_vectors)
            await asyncio.to_thread(_update_source_embedding, source_id, composed)
            logger.info(
                "Composed source embedding for %s from %d section vectors",
                source_id,
                len(section_vectors),
            )
        else:
            logger.warning("No section vectors found for %s after embedding; skipping compose", source_id)
    except Exception as exc:
        logger.error("Source embedding composition failed for %s: %s", source_id, exc)
        # Don't fail the pipeline — source is still indexed
        _set_pipeline_status(source_id, "failed")
        return err(f"Source embedding composition failed: {exc}")

    _set_pipeline_status(source_id, "indexed")

    # --- Stage 5: Auto-compile ---
    compiled = False
    try:
        compiled = await _stage_auto_compile(source_id)
    except Exception as exc:
        logger.warning("Auto-compile stage failed for %s: %s (non-fatal)", source_id, exc)

    _set_pipeline_status(source_id, "complete")

    return ok(
        data={
            "source_id": source_id,
            "sections_embedded": sections_embedded,
            "compiled": compiled,
        }
    )
