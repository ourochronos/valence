# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

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

from valence.core.db import get_cursor, serialize_row

from .response import ValenceResponse, err, ok

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
        from valence.core.embeddings import generate_embedding, vector_to_pgvector

        return vector_to_pgvector(generate_embedding(content))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def create_article(
    content: str,
    title: str | None = None,
    source_ids: list[str] | None = None,
    author_type: str = "system",
    domain_path: list[str] | None = None,
    epistemic_type: str = "semantic",
) -> ValenceResponse:
    """Create a new article.

    Links provided sources with the 'originates' relationship, records a
    'created' mutation, and computes an embedding + token count.

    Args:
        content: Article body text.
        title: Optional human-readable title.
        source_ids: UUIDs of sources that originated this article.
        author_type: 'system', 'operator', or 'agent'.
        domain_path: Hierarchical domain tags (e.g. ['python', 'stdlib']).
        epistemic_type: 'episodic', 'semantic', or 'procedural'.

    Returns:
        Dict with ``success``, ``article``, and optionally ``error``.
    """
    if not content or not content.strip():
        return err("content is required")
    if author_type not in VALID_AUTHOR_TYPES:
        return err(f"author_type must be one of {sorted(VALID_AUTHOR_TYPES)}")
    valid_epistemic_types = ("episodic", "semantic", "procedural")
    if epistemic_type not in valid_epistemic_types:
        return err(f"epistemic_type must be one of {sorted(valid_epistemic_types)}")

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
                 content_hash, epistemic_type, embedding, compiled_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NOW())
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
                epistemic_type,
                embedding_str,
            ),
        )
        row = cur.fetchone()
        article = serialize_row(dict(row))
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

        return ok(data=article)


async def get_article(
    article_id: str,
    include_provenance: bool = False,
) -> ValenceResponse:
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
            return err(f"Article not found: {article_id}")

        article = serialize_row(dict(row))

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
                entry = {k: (v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, _UUID) else v) for k, v in dict(prow).items()}
                provenance.append(entry)
            article["provenance"] = provenance

        return ok(data=article)


async def update_article(
    article_id: str,
    content: str,
    source_id: str | None = None,
    epistemic_type: str | None = None,
) -> ValenceResponse:
    """Update article content, increment version, and record mutation.

    Args:
        article_id: UUID of the article to update.
        content: New article body text.
        source_id: Optional UUID of the source that triggered the update.
        epistemic_type: Optional new epistemic type ('episodic', 'semantic', 'procedural').

    Returns:
        Dict with ``success`` and updated ``article``.
    """
    if not content or not content.strip():
        return err("content is required")
    valid_epistemic_types = ("episodic", "semantic", "procedural")
    if epistemic_type is not None and epistemic_type not in valid_epistemic_types:
        return err(f"epistemic_type must be one of {sorted(valid_epistemic_types)}")

    embedding_str = _compute_embedding(content)
    token_count = _count_tokens(content)
    content_hash = _content_hash(content)

    with get_cursor() as cur:
        set_clauses = [
            "content      = %s",
            "size_tokens  = %s",
            "content_hash = %s",
            "embedding    = %s::vector",
            "version      = version + 1",
            "modified_at  = NOW()",
            "compiled_at  = NOW()",
        ]
        params: list[str | int | None] = [content, token_count, content_hash, embedding_str]

        if epistemic_type is not None:
            set_clauses.append("epistemic_type = %s")
            params.append(epistemic_type)

        params.append(article_id)
        cur.execute(
            f"""
            UPDATE articles
            SET {", ".join(set_clauses)}
            WHERE id = %s
            RETURNING *
            """,
            params,
        )
        row = cur.fetchone()
        if not row:
            return err(f"Article not found: {article_id}")

        article = serialize_row(dict(row))

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

        return ok(data=article)


# ---------------------------------------------------------------------------
# Right-Sizing: Split and Merge (WU-07, C3, C6)
# ---------------------------------------------------------------------------


def _split_content_at_midpoint(content: str) -> tuple[str, str]:
    """Split content at the nearest paragraph boundary to the midpoint.

    Tries to split at a blank-line boundary near the midpoint for cleaner
    splits. Falls back to a hard midpoint split if no paragraph boundary
    is found within 20% of the midpoint.

    Returns:
        (first_half, second_half) tuple.
    """
    mid = len(content) // 2
    tolerance = max(50, len(content) // 5)

    # Find paragraph boundary near midpoint
    best_pos = None
    for separator in ("\n\n", "\n## ", "\n# ", "\n"):
        search_start = max(0, mid - tolerance)
        search_end = min(len(content), mid + tolerance)
        region = content[search_start:search_end]
        idx = region.rfind(separator)
        if idx == -1:
            idx = region.find(separator)
        if idx != -1:
            best_pos = search_start + idx + len(separator)
            break

    if best_pos is None or best_pos <= 0 or best_pos >= len(content):
        best_pos = mid

    return content[:best_pos].rstrip(), content[best_pos:].lstrip()


def _build_split_prompt(content: str, target_tokens: int) -> str:
    """Build the LLM prompt for topic-aware article splitting.

    Args:
        content: The article content to split.
        target_tokens: Target token count per section (informational).

    Returns:
        Prompt string for the LLM.
    """
    from valence.core.inference import TASK_OUTPUT_SCHEMAS, TASK_SPLIT

    schema = TASK_OUTPUT_SCHEMAS[TASK_SPLIT]

    return f"""You are splitting an oversized knowledge article into two coherent parts.

ARTICLE CONTENT:
{content}

Find the best natural topic boundary to split this article. Prefer a split point that:
- Separates distinct topics or themes (NOT arbitrary character counts)
- Occurs near the midpoint (roughly ~{target_tokens} tokens per part)
- Falls at a paragraph or section break

Respond with the character index where the split should occur, and descriptive titles for each resulting part.
IMPORTANT: Use titles that reflect the actual content of each part — NEVER use generic titles like "part 1", "part 2", or sequential numbering.

Respond ONLY with valid JSON matching this exact schema (no markdown fences):
{schema}"""


async def split_article(article_id: str) -> ValenceResponse:
    """Split an oversized article by recompiling from its original sources.

    Instead of mechanically dividing article content (which creates copies-of-
    copies), this fetches the article's linked sources and recompiles them using
    the v3 multi-article compilation prompt. The LLM decides natural topic
    boundaries, producing multiple right-sized articles.

    The original article is ARCHIVED (status='archived'). New articles inherit
    the source set with LLM-identified relationships. Mutations of type 'split'
    are recorded.

    Falls back to mechanical midpoint split if inference is unavailable and
    source content cannot be fetched.

    Args:
        article_id: UUID of the article to split.

    Returns:
        Dict with ``success`` and ``data`` containing {"articles": [...]}.
    """
    from valence.core.compilation import compile_article

    with get_cursor() as cur:
        # Fetch original article
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()
        if not row:
            return err(f"Article not found: {article_id}")
        original = serialize_row(dict(row))

        content = original.get("content") or ""
        if len(content.split()) < 4:
            return err(f"Article {article_id} content too short to split (< 4 words)")

        # Fetch linked source IDs
        cur.execute(
            "SELECT source_id::text FROM article_sources WHERE article_id = %s",
            (article_id,),
        )
        source_ids = list({r["source_id"] for r in cur.fetchall()})

    if not source_ids:
        return err(f"Article {article_id} has no linked sources; cannot split from ground truth")

    # Recompile from sources — v3 prompt naturally produces multiple articles
    compile_result = await compile_article(
        source_ids,
        title_hint=original.get("title"),
    )

    if not compile_result.success:
        return err(f"Recompilation during split failed: {compile_result.error}")

    data = compile_result.data or {}
    all_articles = data.get("_all_articles", [data])

    if len(all_articles) < 2:
        # LLM decided content is one coherent topic — nothing to split.
        # Archive the new single article (we don't need a duplicate) and keep original.
        with get_cursor() as cur:
            for art in all_articles:
                cur.execute(
                    "UPDATE articles SET status = 'archived', modified_at = NOW() WHERE id = %s",
                    (art["id"],),
                )
        return err(f"Article {article_id} content is topically coherent; LLM did not split")

    # Archive the original
    with get_cursor() as cur:
        cur.execute(
            "UPDATE articles SET status = 'archived', modified_at = NOW() WHERE id = %s",
            (article_id,),
        )

        # Record split mutations
        for art in all_articles:
            cur.execute(
                """
                INSERT INTO article_mutations
                    (mutation_type, article_id, related_article_id, summary)
                VALUES ('split', %s, %s, 'Created from source-grounded split of original article')
                """,
                (art["id"], article_id),
            )
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, summary)
            VALUES ('split', %s, %s)
            """,
            (article_id, f"Article archived: split into {len(all_articles)} source-grounded articles"),
        )

    logger.info(
        "Split article %s → %d new articles from %d sources",
        article_id,
        len(all_articles),
        len(source_ids),
    )
    return ok(data={"articles": all_articles}, degraded=compile_result.degraded)


async def merge_articles(
    article_id_a: str,
    article_id_b: str,
) -> ValenceResponse:
    """Merge two related articles by recompiling from their combined sources.

    Instead of concatenating article text (which creates copies-of-copies),
    this collects source IDs from both articles and recompiles from ground
    truth. The LLM produces a coherent synthesis. Both original articles are
    archived. Mutations of type 'merged' are recorded.

    Args:
        article_id_a: UUID of the first article to merge.
        article_id_b: UUID of the second article to merge.

    Returns:
        The newly created merged article(s) as a plain dict.
    """
    from valence.core.compilation import compile_article

    with get_cursor() as cur:
        # Verify both articles exist
        for aid in (article_id_a, article_id_b):
            cur.execute("SELECT id FROM articles WHERE id = %s", (aid,))
            if not cur.fetchone():
                return err(f"Article not found: {aid}")

        # Collect title hints
        cur.execute("SELECT title FROM articles WHERE id = %s", (article_id_a,))
        title_a = cur.fetchone()["title"] or ""
        cur.execute("SELECT title FROM articles WHERE id = %s", (article_id_b,))
        title_b = cur.fetchone()["title"] or ""

        # Collect unique source IDs from both articles
        cur.execute(
            """
            SELECT DISTINCT source_id::text
            FROM article_sources
            WHERE article_id IN (%s, %s)
            """,
            (article_id_a, article_id_b),
        )
        source_ids = [r["source_id"] for r in cur.fetchall()]

    if not source_ids:
        return err("Neither article has linked sources; cannot merge from ground truth")

    # Build a title hint from the two originals
    title_hint: str | None
    if title_a and title_b and title_a != title_b:
        title_hint = f"{title_a} + {title_b}"
    else:
        title_hint = title_a or title_b or None

    # Compile from combined sources
    compile_result = await compile_article(source_ids, title_hint=title_hint)

    if not compile_result.success:
        return err(f"Recompilation during merge failed: {compile_result.error}")

    data = compile_result.data or {}
    all_articles = data.get("_all_articles", [data])

    # Archive both originals and record mutations
    with get_cursor() as cur:
        for aid in (article_id_a, article_id_b):
            cur.execute(
                "UPDATE articles SET status = 'archived', modified_at = NOW() WHERE id = %s",
                (aid,),
            )

        for art in all_articles:
            cur.execute(
                """
                INSERT INTO article_mutations
                    (mutation_type, article_id, related_article_id, summary)
                VALUES ('merged', %s, %s, 'Created by source-grounded merge')
                """,
                (art["id"], article_id_a),
            )

        for aid in (article_id_a, article_id_b):
            cur.execute(
                """
                INSERT INTO article_mutations
                    (mutation_type, article_id, summary)
                VALUES ('merged', %s, %s)
                """,
                (aid, f"Article archived: merged via source-grounded recompilation into {len(all_articles)} article(s)"),
            )

    logger.info(
        "Merged articles %s + %s → %d new article(s) from %d sources",
        article_id_a,
        article_id_b,
        len(all_articles),
        len(source_ids),
    )

    if len(all_articles) == 1:
        return ok(data=all_articles[0])
    return ok(data={"articles": all_articles, "primary": all_articles[0]})


async def search_articles(
    query: str,
    limit: int = 10,
    domain_filter: list[str] | None = None,
) -> ValenceResponse:
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
        return ok(data=[])

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
            article = serialize_row(dict(row))
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
        return ok(data=results[:limit])
