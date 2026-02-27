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

from valence.core.db import get_cursor
from valence.core.embeddings import compose_embedding, get_section_vectors, store_source_embedding
from valence.core.response import ValenceResponse, err, ok
from valence.core.section_embeddings import embed_source_sections

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


def _set_pipeline_status(source_id: str, status: str) -> None:
    """Update pipeline_status on the source row."""
    with get_cursor() as cur:
        cur.execute(
            "UPDATE sources SET pipeline_status = %s WHERE id = %s",
            (status, source_id),
        )


def _enqueue_pipeline_task(source_id: str) -> None:
    """Add a source_pipeline task to mutation_queue for batch processing.

    article_id is NOT NULL in the schema; we store source_id there for
    source-pipeline tasks (the real source_id is also in payload).
    """
    payload = json.dumps({"source_id": source_id})
    with get_cursor() as cur:
        cur.execute(
            """
            INSERT INTO mutation_queue (operation, article_id, priority, payload)
            SELECT 'source_pipeline', %s::uuid, 2, %s::jsonb
            """,
            (source_id, payload),
        )


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
        3. Embed sections via section_embeddings.embed_source_sections
        4. Compose source embedding via embeddings.compose_embedding +
           embeddings.store_source_embedding
        5. Auto-compilation check via sources.find_similar_ungrouped +
           compilation.compile_article

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

    # --- Stage 1: Fetch source ---
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

    # --- Stage 2: Tree index (build for large sources without existing tree) ---
    if len(content) >= SMALL_SOURCE_CHARS and not metadata.get("tree_index"):
        try:
            from valence.core.tree_index import build_tree_index

            result = await build_tree_index(source_id=source_id, force=False)
            if not result.success:
                logger.warning(
                    "Tree build failed for %s: %s — embed stage will fall back to single-node",
                    source_id,
                    result.error,
                )
        except Exception as exc:
            logger.warning("build_tree_index raised for %s: %s — continuing", source_id, exc)

    # --- Stage 3: Embed sections ---
    try:
        embed_result = await embed_source_sections(source_id)
        if not embed_result.success:
            logger.error("Section embedding stage failed for %s: %s", source_id, embed_result.error)
            _set_pipeline_status(source_id, "failed")
            return err(f"Section embedding stage failed: {embed_result.error}")
        sections_embedded = embed_result.data.get("sections_embedded", 0)
    except Exception as exc:
        logger.error("Section embedding stage raised for %s: %s", source_id, exc)
        _set_pipeline_status(source_id, "failed")
        return err(f"Section embedding stage failed: {exc}")

    # --- Stage 4: Compose source embedding ---
    try:
        section_vectors = await asyncio.to_thread(get_section_vectors, source_id)
        if section_vectors:
            composed = compose_embedding(section_vectors)
            await asyncio.to_thread(store_source_embedding, source_id, composed)
            logger.info(
                "Composed source embedding for %s from %d section vectors",
                source_id,
                len(section_vectors),
            )
        else:
            logger.warning("No section vectors found for %s after embedding; skipping compose", source_id)
    except Exception as exc:
        logger.error("Source embedding composition failed for %s: %s", source_id, exc)
        _set_pipeline_status(source_id, "failed")
        return err(f"Source embedding composition failed: {exc}")

    _set_pipeline_status(source_id, "indexed")

    # --- Stage 5: Auto-compile ---
    compiled = False
    try:
        from valence.core.compilation import compile_article
        from valence.core.sources import find_similar_ungrouped

        similar_ids = await find_similar_ungrouped(source_id, COMPILE_SIMILARITY_THRESHOLD)

        with get_cursor() as cur:
            cur.execute(
                "SELECT 1 FROM article_sources WHERE source_id = %s::uuid LIMIT 1",
                (source_id,),
            )
            already_grouped = cur.fetchone() is not None

        candidate_ids = ([source_id] if not already_grouped else []) + similar_ids

        if len(candidate_ids) >= AUTO_COMPILE_MIN_CLUSTER:
            compile_result = await compile_article(candidate_ids[:10])
            if compile_result.success:
                logger.info(
                    "Auto-compiled %d sources (including %s) → article %s",
                    len(candidate_ids),
                    source_id,
                    (compile_result.data or {}).get("id", "?"),
                )
                compiled = True
            else:
                logger.warning("Auto-compile failed: %s", compile_result.error)
        else:
            logger.debug(
                "Auto-compile: only %d candidates for %s (need %d), skipping",
                len(candidate_ids),
                source_id,
                AUTO_COMPILE_MIN_CLUSTER,
            )
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
