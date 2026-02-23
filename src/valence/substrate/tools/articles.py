"""MCP tool wrappers for article operations (WU-04).

These functions are the thin MCP-facing layer that delegates to the core
articles / provenance modules. Each function accepts and returns plain
JSON-serialisable dicts compatible with the MCP tool protocol.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Article tools
# ---------------------------------------------------------------------------


def article_create(
    content: str,
    title: str | None = None,
    source_ids: list[str] | None = None,
    author_type: str = "system",
    domain_path: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new article.

    MCP tool: ``article_create``

    Args:
        content: Article body text (required).
        title: Optional human-readable title.
        source_ids: UUIDs of source documents this article originates from.
        author_type: 'system', 'operator', or 'agent'.
        domain_path: Hierarchical domain tags (e.g. ['python', 'stdlib']).

    Returns:
        MCP result dict with ``success`` and ``article`` or ``error``.
    """
    from ...core.articles import create_article

    result = _run_async(
        create_article(
            content=content,
            title=title,
            source_ids=source_ids,
            author_type=author_type,
            domain_path=domain_path,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "article": result.data}


def article_get(
    article_id: str,
    include_provenance: bool = False,
) -> dict[str, Any]:
    """Get an article by ID.

    MCP tool: ``article_get``

    Args:
        article_id: UUID of the article.
        include_provenance: Include the list of linked sources in the response.

    Returns:
        MCP result dict with ``success`` and ``article`` or ``error``.
    """
    from ...core.articles import get_article

    result = _run_async(get_article(article_id=article_id, include_provenance=include_provenance))
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "article": result.data}


def article_update(
    article_id: str,
    content: str,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Update an article's content.

    MCP tool: ``article_update``

    Increments the article version, records an 'updated' mutation, and
    optionally links the triggering source with a 'confirms' relationship.

    Args:
        article_id: UUID of the article to update.
        content: New article body text.
        source_id: Optional UUID of the source that triggered the update.

    Returns:
        MCP result dict with ``success`` and updated ``article`` or ``error``.
    """
    from ...core.articles import update_article

    result = _run_async(
        update_article(
            article_id=article_id,
            content=content,
            source_id=source_id,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "article": result.data}


def article_search(
    query: str,
    limit: int = 10,
    domain_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Search articles via full-text (and optional semantic) search.

    MCP tool: ``article_search``

    Args:
        query: Search query string.
        limit: Maximum number of results (default 10, max 50).
        domain_filter: Optional domain path segments to restrict results.

    Returns:
        MCP result dict with ``success``, ``articles``, and ``total_count``.
    """
    from ...core.articles import search_articles

    limit = min(max(1, limit), 50)
    result = _run_async(search_articles(query=query, limit=limit, domain_filter=domain_filter))
    if not result.success:
        return {"success": False, "error": result.error}
    articles = result.data or []
    return {
        "success": True,
        "articles": articles,
        "total_count": len(articles),
    }


# ---------------------------------------------------------------------------
# Provenance tools
# ---------------------------------------------------------------------------


def provenance_link(
    article_id: str,
    source_id: str,
    relationship: str,
    notes: str | None = None,
) -> dict[str, Any]:
    """Link a source to an article with a provenance relationship.

    MCP tool: ``provenance_link``

    Args:
        article_id: UUID of the article.
        source_id: UUID of the source.
        relationship: One of originates, confirms, supersedes, contradicts, contends.
        notes: Optional notes about the relationship.

    Returns:
        MCP result dict with ``success`` and ``link`` or ``error``.
    """
    from ...core.provenance import link_source

    result = _run_async(
        link_source(
            article_id=article_id,
            source_id=source_id,
            relationship=relationship,
            notes=notes,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "link": result.data}


def provenance_get(article_id: str) -> dict[str, Any]:
    """Get the full provenance list for an article.

    MCP tool: ``provenance_get``

    Args:
        article_id: UUID of the article.

    Returns:
        MCP result dict with ``success``, ``provenance`` list, and ``count``.
    """
    from ...core.provenance import get_provenance

    result = _run_async(get_provenance(article_id=article_id))
    if not result.success:
        return {"success": False, "error": result.error}
    provenance = result.data or []
    return {
        "success": True,
        "provenance": provenance,
        "count": len(provenance),
    }


def provenance_trace(article_id: str, claim_text: str) -> dict[str, Any]:
    """Trace which sources likely contributed a specific claim.

    MCP tool: ``provenance_trace``

    Uses text-based TF-IDF similarity to find sources whose content
    overlaps with the given claim text.

    Args:
        article_id: UUID of the article.
        claim_text: The specific claim to trace.

    Returns:
        MCP result dict with ``success``, ``sources`` list (sorted by
        ``claim_similarity`` descending), and ``count``.
    """
    from ...core.provenance import trace_claim

    result = _run_async(trace_claim(article_id=article_id, claim_text=claim_text))
    if not result.success:
        return {"success": False, "error": result.error}
    sources = result.data or []
    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
    }


# ---------------------------------------------------------------------------
# Article compilation / right-sizing tools (WU-06, WU-07)
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context (event-loop aware)."""
    import asyncio
    import concurrent.futures

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def article_compile(
    source_ids: list[str],
    title_hint: str | None = None,
) -> dict[str, Any]:
    """Compile one or more sources into a new knowledge article.

    MCP tool: ``article_compile``

    Uses LLM summarization to produce a coherent article from the given
    sources. The new article is linked to all sources with appropriate
    provenance relationships.

    Args:
        source_ids: UUIDs of source documents to compile.
        title_hint: Optional hint for the article title.

    Returns:
        MCP result dict with ``success`` and ``article`` or ``error``.
    """
    if not source_ids:
        return {"success": False, "error": "source_ids must be a non-empty list"}

    try:
        from ...core.compilation import compile_article

        result = _run_async(compile_article(source_ids=source_ids, title_hint=title_hint))
        if not result.success:
            return {"success": False, "error": result.error}
        out: dict[str, Any] = {"success": True, "article": result.data}
        if result.degraded:
            out["degraded"] = True
        return out
    except Exception as exc:
        logger.exception("article_compile failed")
        return {"success": False, "error": str(exc)}


def article_split(article_id: str) -> dict[str, Any]:
    """Split an oversized article into two smaller articles.

    MCP tool: ``article_split``

    The original article retains its ID and part of the content. A new
    article is created for the remainder. Both articles inherit all
    provenance sources. Mutation records are written for both.

    Args:
        article_id: UUID of the article to split.

    Returns:
        MCP result dict with ``success``, ``original`` (updated article),
        and ``new`` (newly created article), or ``error``.
    """
    if not article_id:
        return {"success": False, "error": "article_id is required"}

    try:
        from ...core.articles import split_article

        result = _run_async(split_article(article_id=article_id))
        if not result.success:
            return {"success": False, "error": result.error}
        data = result.data or {}
        return {
            "success": True,
            "original": data.get("original"),
            "new": data.get("new"),
        }
    except Exception as exc:
        logger.exception("article_split failed for %s", article_id)
        return {"success": False, "error": str(exc)}


def article_merge(article_id_a: str, article_id_b: str) -> dict[str, Any]:
    """Merge two related articles into one.

    MCP tool: ``article_merge``

    A new article is created with the combined content. Both originals are
    archived. The new article inherits the union of provenance sources from
    both originals. Mutation records are written.

    Args:
        article_id_a: UUID of the first article.
        article_id_b: UUID of the second article.

    Returns:
        MCP result dict with ``success`` and ``article`` (merged result),
        or ``error``.
    """
    if not article_id_a or not article_id_b:
        return {"success": False, "error": "article_id_a and article_id_b are both required"}

    try:
        from ...core.articles import merge_articles

        result = _run_async(merge_articles(article_id_a=article_id_a, article_id_b=article_id_b))
        if not result.success:
            return {"success": False, "error": result.error}
        return {"success": True, "article": result.data}
    except Exception as exc:
        logger.exception("article_merge failed for %s + %s", article_id_a, article_id_b)
        return {"success": False, "error": str(exc)}
