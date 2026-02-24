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

from valence.core.db import get_cursor

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


async def create_article(
    content: str,
    title: str | None = None,
    source_ids: list[str] | None = None,
    author_type: str = "system",
    domain_path: list[str] | None = None,
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

    Returns:
        Dict with ``success``, ``article``, and optionally ``error``.
    """
    if not content or not content.strip():
        return err("content is required")
    if author_type not in VALID_AUTHOR_TYPES:
        return err(f"author_type must be one of {sorted(VALID_AUTHOR_TYPES)}")

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
                entry = {k: (v.isoformat() if isinstance(v, datetime) else str(v) if isinstance(v, _UUID) else v) for k, v in dict(prow).items()}
                provenance.append(entry)
            article["provenance"] = provenance

        return ok(data=article)


async def update_article(
    article_id: str,
    content: str,
    source_id: str | None = None,
) -> ValenceResponse:
    """Update article content, increment version, and record mutation.

    Args:
        article_id: UUID of the article to update.
        content: New article body text.
        source_id: Optional UUID of the source that triggered the update.

    Returns:
        Dict with ``success`` and updated ``article``.
    """
    if not content or not content.strip():
        return err("content is required")

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
            return err(f"Article not found: {article_id}")

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
    """Split an oversized article into two articles using topic-aware splitting.

    Uses LLM inference (TASK_SPLIT) to identify the best split point based on
    topic boundaries, not mechanical character counts. The original article is
    ARCHIVED (status='archived'), and two new articles are created with
    descriptive titles.

    Both new articles inherit the full source set from the original. Mutations
    of type 'split' are recorded on all three articles with cross-references.

    Falls back to mechanical midpoint split if LLM is unavailable.

    Args:
        article_id: UUID of the article to split.

    Returns:
        Dict with ``success`` and ``data`` containing {"part_a": ..., "part_b": ...}.

    Raises:
        ValueError: If the article is not found or content is too short to split.
    """
    from valence.core.inference import TASK_SPLIT, InferenceSchemaError, provider, validate_output

    with get_cursor() as cur:
        # Fetch original article
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id,))
        row = cur.fetchone()
        if not row:
            return err(f"Article not found: {article_id}")
        original = _row_to_article(dict(row))

        content = original.get("content") or ""
        if len(content.split()) < 4:
            return err(f"Article {article_id} content too short to split (< 4 words)")

        # Fetch all sources for the original article
        cur.execute(
            """
            SELECT source_id, relationship, notes
            FROM article_sources
            WHERE article_id = %s
            """,
            (article_id,),
        )
        sources = [dict(r) for r in cur.fetchall()]

        # --- LLM-based topic-aware split ---
        target_tokens = _count_tokens(content) // 2
        prompt = _build_split_prompt(content, target_tokens)
        is_degraded = False

        try:
            result = await provider.infer(TASK_SPLIT, prompt)
            if result.degraded:
                raise InferenceSchemaError(result.error or "LLM unavailable")
            parsed = validate_output(TASK_SPLIT, result.content)
            split_index = parsed["split_index"]
            part_a_title = parsed["part_a_title"]
            part_b_title = parsed["part_b_title"]
            reasoning = parsed.get("reasoning", "")
            logger.info("Topic-aware split for article %s: %s", article_id, reasoning)
        except (InferenceSchemaError, ValueError, KeyError) as exc:
            logger.warning("LLM split unavailable (%s), using fallback midpoint split", exc)
            is_degraded = True
            # Fallback: mechanical split at paragraph boundary near midpoint
            first_half, second_half = _split_content_at_midpoint(content)
            split_index = len(first_half)
            orig_title = original.get("title") or "Article"
            part_a_title = f"{orig_title} (part 1)"
            part_b_title = f"{orig_title} (part 2)"

        # Ensure valid split index
        if split_index <= 0 or split_index >= len(content):
            logger.warning("Invalid split_index=%d for content len=%d, using midpoint", split_index, len(content))
            split_index = len(content) // 2

        # Split content at LLM-identified index
        first_half = content[:split_index].rstrip()
        second_half = content[split_index:].lstrip()

        if not first_half or not second_half:
            # Safety fallback if split produced empty content
            first_half, second_half = _split_content_at_midpoint(content)

        # --- Archive the original article ---
        cur.execute(
            "UPDATE articles SET status = 'archived', modified_at = NOW() WHERE id = %s",
            (article_id,),
        )

        # --- Create first new article (part A) ---
        first_token_count = _count_tokens(first_half)
        first_hash = _content_hash(first_half)
        first_embedding = _compute_embedding(first_half)

        cur.execute(
            """
            INSERT INTO articles
                (content, title, author_type, domain_path, size_tokens, confidence,
                 content_hash, embedding, compiled_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector, NOW())
            RETURNING *
            """,
            (
                first_half,
                part_a_title,
                original.get("author_type", "system"),
                original.get("domain_path") or [],
                first_token_count,
                '{"overall": 0.7}',
                first_hash,
                first_embedding,
            ),
        )
        part_a_row = cur.fetchone()
        part_a = _row_to_article(dict(part_a_row))
        part_a_id = part_a["id"]

        # --- Create second new article (part B) ---
        second_token_count = _count_tokens(second_half)
        second_hash = _content_hash(second_half)
        second_embedding = _compute_embedding(second_half)

        cur.execute(
            """
            INSERT INTO articles
                (content, title, author_type, domain_path, size_tokens, confidence,
                 content_hash, embedding, compiled_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector, NOW())
            RETURNING *
            """,
            (
                second_half,
                part_b_title,
                original.get("author_type", "system"),
                original.get("domain_path") or [],
                second_token_count,
                '{"overall": 0.7}',
                second_hash,
                second_embedding,
            ),
        )
        part_b_row = cur.fetchone()
        part_b = _row_to_article(dict(part_b_row))
        part_b_id = part_b["id"]

        # --- Inherit full source set for both new articles ---
        for src in sources:
            src_id = str(src["source_id"])
            rel = src["relationship"]
            notes = src.get("notes")

            # Link to part A
            cur.execute(
                """
                INSERT INTO article_sources (article_id, source_id, relationship, notes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, source_id, relationship) DO NOTHING
                """,
                (part_a_id, src_id, rel, notes),
            )
            # Link to part B
            cur.execute(
                """
                INSERT INTO article_sources (article_id, source_id, relationship, notes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, source_id, relationship) DO NOTHING
                """,
                (part_b_id, src_id, rel, notes),
            )

        # --- Record 'split' mutations on all three articles ---
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, related_article_id, summary)
            VALUES ('split', %s, %s, 'Article archived: split into two topic-aware parts')
            """,
            (article_id, part_a_id),
        )
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, related_article_id, summary)
            VALUES ('split', %s, %s, 'Article created from first topic section of original')
            """,
            (part_a_id, article_id),
        )
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, related_article_id, summary)
            VALUES ('split', %s, %s, 'Article created from second topic section of original')
            """,
            (part_b_id, article_id),
        )

    logger.info("Split article %s → part_a=%s (%s), part_b=%s (%s)", article_id, part_a_id, part_a_title, part_b_id, part_b_title)
    return ok(data={"part_a": part_a, "part_b": part_b}, degraded=is_degraded)


async def merge_articles(
    article_id_a: str,
    article_id_b: str,
) -> ValenceResponse:
    """Merge two related articles into a single new article.

    Creates a new article whose content is the concatenation of both source
    articles. The source sets are combined (deduplicated by source_id +
    relationship). Both original articles are archived (status='archived').
    Mutations of type 'merged' are recorded on all three articles.

    Args:
        article_id_a: UUID of the first article to merge.
        article_id_b: UUID of the second article to merge.

    Returns:
        The newly created merged article as a plain dict.

    Raises:
        ValueError: If either article is not found.
    """
    with get_cursor() as cur:
        # Fetch both articles
        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id_a,))
        row_a = cur.fetchone()
        if not row_a:
            return err(f"Article not found: {article_id_a}")
        article_a = _row_to_article(dict(row_a))

        cur.execute("SELECT * FROM articles WHERE id = %s", (article_id_b,))
        row_b = cur.fetchone()
        if not row_b:
            return err(f"Article not found: {article_id_b}")
        article_b = _row_to_article(dict(row_b))

        # Fetch sources for both articles
        cur.execute(
            """
            SELECT source_id, relationship, notes
            FROM article_sources
            WHERE article_id = %s
            """,
            (article_id_a,),
        )
        sources_a = [dict(r) for r in cur.fetchall()]

        cur.execute(
            """
            SELECT source_id, relationship, notes
            FROM article_sources
            WHERE article_id = %s
            """,
            (article_id_b,),
        )
        sources_b = [dict(r) for r in cur.fetchall()]

        # Deduplicate sources by (source_id, relationship)
        seen: set[tuple] = set()
        combined_sources: list[dict] = []
        for src in sources_a + sources_b:
            key = (str(src["source_id"]), src["relationship"])
            if key not in seen:
                seen.add(key)
                combined_sources.append(src)

        # Build merged content
        content_a = (article_a.get("content") or "").strip()
        content_b = (article_b.get("content") or "").strip()
        title_a = article_a.get("title") or ""
        title_b = article_b.get("title") or ""

        if title_a and title_b and title_a != title_b:
            merged_title = f"{title_a} + {title_b}"
        elif title_a:
            merged_title = title_a
        elif title_b:
            merged_title = title_b
        else:
            merged_title = None

        separator = "\n\n---\n\n"
        merged_content = f"{content_a}{separator}{content_b}"

        token_count = _count_tokens(merged_content)
        content_hash = _content_hash(merged_content)
        embedding_str = _compute_embedding(merged_content)
        author_type = article_a.get("author_type", "system")

        # --- Create merged article ---
        cur.execute(
            """
            INSERT INTO articles
                (content, title, author_type, domain_path, size_tokens, confidence,
                 content_hash, embedding, compiled_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s::vector, NOW())
            RETURNING *
            """,
            (
                merged_content,
                merged_title,
                author_type,
                article_a.get("domain_path") or [],
                token_count,
                '{"overall": 0.7}',
                content_hash,
                embedding_str,
            ),
        )
        merged_row = cur.fetchone()
        merged_article = _row_to_article(dict(merged_row))
        merged_id = merged_article["id"]

        # --- Archive both originals ---
        for aid in (article_id_a, article_id_b):
            cur.execute(
                "UPDATE articles SET status = 'archived', modified_at = NOW() WHERE id = %s",
                (aid,),
            )

        # --- Link combined sources to merged article ---
        for src in combined_sources:
            src_id = str(src["source_id"])
            rel = src["relationship"]
            notes = src.get("notes")
            cur.execute(
                """
                INSERT INTO article_sources (article_id, source_id, relationship, notes)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (article_id, source_id, relationship) DO NOTHING
                """,
                (merged_id, src_id, rel, notes),
            )

        # --- Record 'merged' mutations on all three articles ---
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, related_article_id, summary)
            VALUES ('merged', %s, %s, 'Article archived: merged into new article')
            """,
            (article_id_a, merged_id),
        )
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, related_article_id, summary)
            VALUES ('merged', %s, %s, 'Article archived: merged into new article')
            """,
            (article_id_b, merged_id),
        )
        cur.execute(
            """
            INSERT INTO article_mutations
                (mutation_type, article_id, related_article_id, summary)
            VALUES ('merged', %s, %s, 'Article created by merging two source articles')
            """,
            (merged_id, article_id_a),
        )

    logger.info("Merged articles %s + %s → new article %s", article_id_a, article_id_b, merged_id)
    return ok(data=merged_article)


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
        return ok(data=results[:limit])
