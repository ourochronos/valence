"""Provenance tracking for Valence v2 knowledge articles.

Provides functions to link sources to articles, retrieve provenance chains,
and trace which sources likely contributed a specific claim via text similarity.

Implements WU-04 (C5 — provenance tracking).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from our_db import get_cursor

from .embedding_interop import text_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_RELATIONSHIPS = {"originates", "confirms", "supersedes", "contradicts", "contends"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert a DB row dict to a JSON-safe plain dict."""
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, UUID):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def link_source(
    article_id: str,
    source_id: str,
    relationship: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Add an article_sources provenance row.

    Args:
        article_id: UUID of the article.
        source_id: UUID of the source.
        relationship: One of ``originates``, ``confirms``, ``supersedes``,
            ``contradicts``, ``contends``.
        notes: Optional free-text notes about the relationship.

    Returns:
        Dict with ``success`` and the created ``link`` row, or ``error``.
    """
    if relationship not in VALID_RELATIONSHIPS:
        return {
            "success": False,
            "error": f"relationship must be one of {sorted(VALID_RELATIONSHIPS)}",
        }

    with get_cursor() as cur:
        # Verify article exists
        cur.execute("SELECT id FROM articles WHERE id = %s", (article_id,))
        if not cur.fetchone():
            return {"success": False, "error": f"Article not found: {article_id}"}

        # Verify source exists
        cur.execute("SELECT id FROM sources WHERE id = %s", (source_id,))
        if not cur.fetchone():
            return {"success": False, "error": f"Source not found: {source_id}"}

        cur.execute(
            """
            INSERT INTO article_sources (article_id, source_id, relationship, notes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (article_id, source_id, relationship) DO UPDATE
                SET notes = EXCLUDED.notes
            RETURNING *
            """,
            (article_id, source_id, relationship, notes),
        )
        row = cur.fetchone()
        return {"success": True, "link": _serialize_row(dict(row))}


def get_provenance(article_id: str) -> list[dict[str, Any]]:
    """Return all provenance entries for an article, with source details.

    Args:
        article_id: UUID of the article.

    Returns:
        List of provenance dicts (empty list if article has no sources).
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                asrc.id           AS link_id,
                asrc.article_id,
                asrc.source_id,
                asrc.relationship,
                asrc.added_at,
                asrc.notes,
                s.type            AS source_type,
                s.title           AS source_title,
                s.url             AS source_url,
                s.reliability,
                s.created_at      AS source_created_at
            FROM article_sources asrc
            JOIN sources s ON s.id = asrc.source_id
            WHERE asrc.article_id = %s
            ORDER BY asrc.added_at
            """,
            (article_id,),
        )
        rows = cur.fetchall()
        return [_serialize_row(dict(r)) for r in rows]


def trace_claim(article_id: str, claim_text: str) -> list[dict[str, Any]]:
    """On-demand claim tracing: find sources that likely contributed a claim.

    Uses text-based TF-IDF cosine similarity (from embedding_interop) to
    compare the claim against each source's content. Returns sources sorted
    by similarity descending.

    Args:
        article_id: UUID of the article containing the claim.
        claim_text: The specific claim text to trace.

    Returns:
        List of source dicts with an added ``claim_similarity`` score (0–1),
        sorted highest-first. Only sources with similarity > 0 are returned.
    """
    if not claim_text or not claim_text.strip():
        return []

    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                s.id,
                s.type,
                s.title,
                s.url,
                s.content,
                s.reliability,
                s.created_at,
                asrc.relationship
            FROM article_sources asrc
            JOIN sources s ON s.id = asrc.source_id
            WHERE asrc.article_id = %s
              AND s.content IS NOT NULL
            """,
            (article_id,),
        )
        rows = cur.fetchall()

    if not rows:
        return []

    results = []
    for row in rows:
        row_dict = _serialize_row(dict(row))
        source_content = row_dict.get("content") or ""
        similarity = text_similarity(claim_text, source_content)
        if similarity > 0:
            row_dict["claim_similarity"] = round(similarity, 4)
            results.append(row_dict)

    results.sort(key=lambda x: x["claim_similarity"], reverse=True)
    return results


def get_mutation_history(article_id: str) -> list[dict[str, Any]]:
    """Return the mutation history for an article.

    Args:
        article_id: UUID of the article.

    Returns:
        List of mutation dicts ordered by creation time.
    """
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT id, mutation_type, article_id, related_article_id,
                   trigger_source_id, summary, created_at
            FROM article_mutations
            WHERE article_id = %s
            ORDER BY created_at
            """,
            (article_id,),
        )
        rows = cur.fetchall()
        return [_serialize_row(dict(r)) for r in rows]
