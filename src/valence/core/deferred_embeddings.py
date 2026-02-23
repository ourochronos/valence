"""Deferred embedding pipeline for sources and articles.

WU-12 (C1, C9): Compute follows use — embeddings are NOT computed on ingest.
They are computed on first retrieval need, then cached in the DB column.

Key insight:
- Sources are ingested cheaply (no embedding). Embedding is deferred.
- Articles DO get embeddings on create (compiled content, worth the compute).
  That is handled in WU-04 / articles.py.
- This module provides the "lazy + batch" layer for sources.

Typical usage:
    # On retrieval for a source that may lack an embedding:
    was_computed = await ensure_embedding('sources', source_id)

    # Offline batch fill (cron / maintenance):
    count = await compute_missing_embeddings('sources', batch_size=100)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Tables that support deferred embeddings and the text column to embed
_EMBEDDABLE_TABLES: dict[str, str] = {
    "sources": "content",
    "articles": "content",
}

# Module-level import for patchability in tests.
# our_db is required; our_embeddings is optional (graceful degradation).
from our_db import get_cursor  # noqa: E402

# ---------------------------------------------------------------------------
# Internal sync helpers (run in thread pool via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _check_needs_embedding_sync(table: str, row_id: str) -> bool:
    """Synchronous: return True if embedding IS NULL for the given row.

    Raises ValueError for unknown tables.
    Raises LookupError if the row does not exist.
    """
    if table not in _EMBEDDABLE_TABLES:
        raise ValueError(f"Table '{table}' is not supported for deferred embeddings")

    with get_cursor() as cur:
        cur.execute(
            f"SELECT embedding IS NULL AS needs_embed FROM {table} WHERE id = %s",  # noqa: S608
            (row_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"Row not found: {table}.id = {row_id}")
        return bool(row["needs_embed"])


def _compute_embedding_sync(table: str, row_id: str) -> bool:
    """Synchronous: compute and store embedding for a row.

    Returns True if embedding was computed, False if it already existed
    (idempotent — safe to call multiple times).

    Silently skips if the embedding service is unavailable.
    """
    if table not in _EMBEDDABLE_TABLES:
        raise ValueError(f"Table '{table}' is not supported for deferred embeddings")

    text_col = _EMBEDDABLE_TABLES[table]

    # Fetch text content and check whether embedding is already present
    with get_cursor() as cur:
        cur.execute(
            f"SELECT {text_col}, embedding IS NULL AS needs_embed FROM {table} WHERE id = %s",  # noqa: S608
            (row_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise LookupError(f"Row not found: {table}.id = {row_id}")

        if not row["needs_embed"]:
            return False  # Already has an embedding

        text = row[text_col]
        if not text:
            logger.debug("Row %s.%s has no text content, skipping embedding", table, row_id)
            return False

    # Generate embedding (may be slow — runs in thread pool)
    try:
        from our_embeddings.service import generate_embedding, vector_to_pgvector

        vector = generate_embedding(text)
        embedding_str = vector_to_pgvector(vector)
    except Exception as exc:
        logger.warning(
            "Embedding unavailable for %s.%s, will retry later: %s",
            table,
            row_id,
            exc,
        )
        return False

    # Persist the embedding
    with get_cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET embedding = %s::vector WHERE id = %s AND embedding IS NULL",  # noqa: S608
            (embedding_str, row_id),
        )

    logger.debug("Computed and stored embedding for %s.%s", table, row_id)
    return True


def _batch_fill_sync(table: str, batch_size: int) -> int:
    """Synchronous: compute embeddings for up to batch_size rows missing them.

    Returns the number of embeddings computed.
    """
    if table not in _EMBEDDABLE_TABLES:
        raise ValueError(f"Table '{table}' is not supported for deferred embeddings")

    text_col = _EMBEDDABLE_TABLES[table]

    # Collect candidate IDs
    with get_cursor() as cur:
        cur.execute(
            f"""
            SELECT id, {text_col}
            FROM {table}
            WHERE embedding IS NULL
              AND {text_col} IS NOT NULL
              AND {text_col} != ''
            ORDER BY id
            LIMIT %s
            """,  # noqa: S608
            (batch_size,),
        )
        rows: list[dict[str, Any]] = cur.fetchall()

    if not rows:
        return 0

    try:
        from our_embeddings.service import generate_embedding, vector_to_pgvector
    except Exception as exc:
        logger.warning("Embedding service unavailable for batch fill: %s", exc)
        return 0

    computed = 0
    for row in rows:
        row_id = str(row["id"])
        text = row[text_col]
        try:
            vector = generate_embedding(text)
            embedding_str = vector_to_pgvector(vector)
            with get_cursor() as cur:
                cur.execute(
                    f"UPDATE {table} SET embedding = %s::vector WHERE id = %s AND embedding IS NULL",  # noqa: S608
                    (embedding_str, row_id),
                )
            computed += 1
            logger.debug("Batch-computed embedding for %s.%s", table, row_id)
        except Exception as exc:
            logger.warning("Failed to compute embedding for %s.%s: %s", table, row_id, exc)

    logger.info("compute_missing_embeddings: computed %d/%d for table=%s", computed, len(rows), table)
    return computed


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def needs_embedding(table: str, row_id: str) -> bool:
    """Check if a row is missing its embedding.

    Args:
        table:  DB table name (currently: 'sources' or 'articles').
        row_id: UUID of the row to check.

    Returns:
        True  — embedding IS NULL (needs to be computed).
        False — embedding already present.

    Raises:
        ValueError:   Unknown/unsupported table.
        LookupError:  Row not found.
    """
    return await asyncio.to_thread(_check_needs_embedding_sync, table, row_id)


async def ensure_embedding(table: str, row_id: str) -> bool:
    """Check if row has an embedding; compute and store if not.

    This is the primary entry point for lazy (compute-follows-use) embedding.
    Safe to call repeatedly — idempotent.

    Args:
        table:  DB table name ('sources' or 'articles').
        row_id: UUID of the row to embed.

    Returns:
        True  — embedding was just computed (was missing).
        False — embedding already existed; nothing was done.

    Raises:
        ValueError:  Unknown/unsupported table.
        LookupError: Row not found.
    """
    return await asyncio.to_thread(_compute_embedding_sync, table, row_id)


async def compute_missing_embeddings(table: str = "sources", batch_size: int = 100) -> int:
    """Batch-fill missing embeddings for the given table.

    Designed for offline/maintenance use: fills up to `batch_size` rows
    that currently have `embedding IS NULL`.

    Args:
        table:      DB table to process (default: 'sources').
        batch_size: Maximum number of rows to process in one call.

    Returns:
        Number of embeddings successfully computed (0–batch_size).

    Raises:
        ValueError: Unknown/unsupported table.
    """
    if table not in _EMBEDDABLE_TABLES:
        raise ValueError(f"Table '{table}' is not supported for deferred embeddings")
    return await asyncio.to_thread(_batch_fill_sync, table, batch_size)
