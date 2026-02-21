"""Article CRUD and versioning for the Valence v2 knowledge system.

Articles are the core knowledge unit — compiled from one or more sources,
versioned, and linked to provenance via article_sources and article_mutations.

Implements WU-04 (C2 partial, C5).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from typing import Any
from uuid import UUID

from our_db import get_cursor

from .embedding_interop import text_similarity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

VALID_RELATIONSHIP_TYPES = {"originates", "confirms", "supersedes", "contradicts", "contends"}
VALID_AUTHOR_TYPES = {"system", "operator", "agent"}


def _count_tokens(content: str) -> int:
    """Approximate token count: word count * 1.3."""
    return max(1, int(len(content.split()) * 1.3))


def _content_hash(content: str) -> str:
    """SHA-256 hash of normalised content for deduplication."""
    return hashlib.sha256(content.strip().lower().encode()).hexdigest()


def _compute_embedding(content: str) -> str | None:
    """Generate embedding string for pgvector. Returns None if unavailable."""
    if os.environ.get("VALENCE_ASYNC_EMBEDDINGS"):
        return None
    try:
        from our_embeddings.service import generate_embedding, vector_to_pgvector

        return vector_to_pgvector(generate_embedding(content))
    except Exception:
        return None


def _row_to_article(row: dict[str, Any]) -> dict[str, Any]:
    """Serialise a DB article row to a plain dict."""
    article: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, datetime):
            article[k] = v.isoformat()
        elif k == "confidence" and isinstance(v, str):
            try:
                article[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                article[k] = v
        else:
            article[k] = v
    # Always stringify UUIDs
    for uuid_col in ("id", "source_id", "supersedes_id", "superseded_by_id", "holder_id"):
        if uuid_col in article and article[uuid_col] is not None:
            article[uuid_col] = str(article[uuid_col])
    return article


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_article(
    content: str,
    title: str | None = None,
    source_ids: list[str] | None = None,
    author_type: str = "system",
    domain_path: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new article.

    Links provided sources with the 'originates' relationship, records a
    'created' mutation, and computes an embedding + token count.

    Args:
        content: Article body text.
        title: Optional human-readable title.
        source_ids: UUIDs of sources that originated this article.
        author_type: 'system', 'operator', or 'agent'.
        domain_path: Hierarchical domain tags (e.g. ['python', 'stdlib']).

    Returns:
        Dict with ``success``, ``article``, and optionally ``error``.
    """
    if not content or not content.strip():
        return {"success": False, "error": "content is required"}
    if author_type not in VALID_AUTHOR_TYPES:
        return {"success": False, "error": f"author_type must be one of {sorted(VALID_AUTHOR_TYPES)}"}

    embedding_str = _compute_embedding(content)
    token_count = _count_tokens(content)
    content_hash = _content_hash(content)
    confidence = {"overall": 0.7}

    with get_cursor() as cur:
        # Create the article
        cur.execute(
            """
            INSERT INTO articles
                (content, title, author_type, domain_path, size_tokens, confidence,
                 content_hash, embedding, compiled_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, NOW())
            RETURNING *
            """,
            (
                content,
                title,
                author_type,
                domain_path or [],
                token_count,
                json.dumps(confidence),
                content_hash,
                embedding_str,
            ),
        )
        row = cur.fetchone()
        article = _row_to_article(dict(row))
        article_id = article["id"]

        # Link sources with 'originates' relationship
        if source_ids:
            for source_id in source_ids:
                try:
                    cur.execute(
                        """
                        INSERT INTO article_sources (article_id, source_id, relationship)
                        VALUES (%s, %s, 'originates')
                        ON CONFLICT (article_id, source_id, relationship) DO NOTHING
                        """,
                        (article_id, source_id),
                    )
                except Exception as exc:
                    logger.warning("Failed to link source %s to article %s: %s", source_id, article_id, exc)

        # Record 'created' mutation
        cur.execute(
            """
            INSERT INTO article_mutations (mutation_type, article_id, summary)
            VALUES ('created', %s, %s)
            """,
            (article_id, f"Article created with {len(source_ids or [])} source(s)"),
        )

        return {"success": True, "article": article}


def get_article(
    article_id: str,
    include_provenance: bool = False,
) -> dict[str, Any]:
    """Retrieve an article by ID.

    Args:
        article_id: UUID of the article.
        include_provenance: If True, attach the list of linked sources.

    Returns:
        Dict with ``success`` and ``article`` (optionally ``article.provenance``).
    """
    with get_cursor() as cur:
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Article not found: {article_id}"}

        article = _row_to_article(dict(row))

        if include_provenance:
            from uuid import UUID as _UUID
            cur.execute(
                """
                SELECT
                    asrc.id,
                    asrc.source_id,
                    asrc.relationship,
                    asrc.added_at,
                    asrc.notes,
                    s.type       AS source_type,
                    s.title      AS source_title,
                    s.url        AS source_url,
                    s.reliability
                FROM article_sources asrc
                JOIN sources s ON s.id = asrc.source_id
                WHERE asrc.article_id = %s
                ORDER BY asrc.added_at
                """,
                (article_id,),
            )
            provenance = []
            for prow in cur.fetchall():
                entry = {
                    k: (v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, _UUID) else v)
                    for k, v in dict(prow).items()
                }
                provenance.append(entry)
            article["provenance"] = provenance

        return {"success": True, "article": article}


def update_article(
    article_id: str,
    content: str,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Update article content, increment version, and record mutation.

    Args:
        article_id: UUID of the article to update.
        content: New article body text.
        source_id: Optional UUID of the source that triggered the update.

    Returns:
        Dict with ``success`` and updated ``article``.
    """
    if not content or not content.strip():
        return {"success": False, "error": "content is required"}

    embedding_str = _compute_embedding(content)
    token_count = _count_tokens(content)
    content_hash = _content_hash(content)

    with get_cursor() as cur:
        cur.execute(
            """
            UPDATE articles
            SET content      = %s,
                size_tokens  = %s,
                content_hash = %s,
                embedding    = %s::vector,
                version      = version + 1,
                modified_at  = NOW(),
                compiled_at  = NOW()
            WHERE id = %s
            RETURNING *
            """,
            (content, token_count, content_hash, embedding_str, article_id),
        )
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Article not found: {article_id}"}

        article = _row_to_article(dict(row))

        # Optionally link the triggering source
        if source_id:
            try:
                cur.execute(
                    """
                    INSERT INTO article_sources (article_id, source_id, relationship)
                    VALUES (%s, %s, 'confirms')
                    ON CONFLICT (article_id, source_id, relationship) DO NOTHING
                    """,
                    (article_id, source_id),
                )
            except Exception as exc:
                logger.warning("Failed to link source %s on update: %s", source_id, exc)

        # Record 'updated' mutation
        cur.execute(
            """
            INSERT INTO article_mutations (mutation_type, article_id, trigger_source_id, summary)
            VALUES ('updated', %s, %s, 'Content updated')
            """,
            (article_id, source_id),
        )

        return {"success": True, "article": article}


def search_articles(
    query: str,
    limit: int = 10,
    domain_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Search articles via full-text search, optionally augmented with vector similarity.

    Full-text search is always performed. If embeddings are available, results
    are re-ranked by combined semantic + text relevance.

    Args:
        query: Search query string.
        limit: Maximum number of results.
        domain_filter: Optional list of domain path segments to restrict search.

    Returns:
        List of article dicts, ordered by relevance.
    """
    if not query or not query.strip():
        return []

    with get_cursor() as cur:
        # Full-text search
        sql = """
            SELECT a.*,
                   ts_rank(a.content_tsv, websearch_to_tsquery('english', %s)) AS relevance
            FROM articles a
            WHERE a.content_tsv @@ websearch_to_tsquery('english', %s)
              AND a.status = 'active'
              AND a.superseded_by_id IS NULL
        """
        params: list[Any] = [query, query]

        if domain_filter:
            sql += " AND a.domain_path && %s"
            params.append(domain_filter)

        sql += " ORDER BY relevance DESC LIMIT %s"
        params.append(limit * 2)  # Over-fetch for re-ranking

        cur.execute(sql, params)
        rows = cur.fetchall()

        # Build results list
        results = []
        max_relevance = max((float(r.get("relevance", 0)) for r in rows), default=1.0) or 1.0
        for row in rows:
            article = _row_to_article(dict(row))
            article["text_relevance"] = float(row.get("relevance", 0)) / max_relevance
            article["similarity"] = article["text_relevance"]
            results.append(article)

        # Optionally re-rank with vector similarity
        embedding_str = _compute_embedding(query)
        if embedding_str and results:
            ids = [r["id"] for r in results]
            placeholders = ",".join(["%s"] * len(ids))
            cur.execute(
                f"""
                SELECT id, 1 - (embedding <=> %s::vector) AS cosine
                FROM articles
                WHERE id IN ({placeholders})
                  AND embedding IS NOT NULL
                """,
                [embedding_str, *ids],
            )
            cosine_map = {str(r["id"]): float(r["cosine"]) for r in cur.fetchall()}
            for article in results:
                cosine = cosine_map.get(article["id"], 0.0)
                text_rel = article["text_relevance"]
                article["similarity"] = 0.5 * cosine + 0.5 * text_rel

            results.sort(key=lambda x: x.get("similarity", 0), reverse=True)

        # Trim to requested limit
        return results[:limit]
