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

    return create_article(
        content=content,
        title=title,
        source_ids=source_ids,
        author_type=author_type,
        domain_path=domain_path,
    )


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

    return get_article(article_id=article_id, include_provenance=include_provenance)


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

    return update_article(
        article_id=article_id,
        content=content,
        source_id=source_id,
    )


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
    results = search_articles(query=query, limit=limit, domain_filter=domain_filter)
    return {
        "success": True,
        "articles": results,
        "total_count": len(results),
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

    return link_source(
        article_id=article_id,
        source_id=source_id,
        relationship=relationship,
        notes=notes,
    )


def provenance_get(article_id: str) -> dict[str, Any]:
    """Get the full provenance list for an article.

    MCP tool: ``provenance_get``

    Args:
        article_id: UUID of the article.

    Returns:
        MCP result dict with ``success``, ``provenance`` list, and ``count``.
    """
    from ...core.provenance import get_provenance

    provenance = get_provenance(article_id=article_id)
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

    sources = trace_claim(article_id=article_id, claim_text=claim_text)
    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
    }
