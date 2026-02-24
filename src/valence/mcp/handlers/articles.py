# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Article and retrieval tool handlers."""

from __future__ import annotations

import logging
from typing import Any

from valence.core.response import ValenceResponse

from ._utils import run_async

logger = logging.getLogger(__name__)


def knowledge_search(
    query: str,
    limit: int = 10,
    include_sources: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Unified knowledge retrieval."""
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 200))

    try:
        from valence.core.retrieval import retrieve

        results = run_async(retrieve(query, limit=limit, include_sources=include_sources, session_id=session_id))
    except Exception as exc:
        logger.exception("knowledge_search failed for query %r: %s", query, exc)
        return {"success": False, "error": str(exc)}

    if isinstance(results, ValenceResponse):
        if not results.success:
            return {"success": False, "error": results.error}
        result_list = results.data or []
    else:
        result_list = results

    return {
        "success": True,
        "results": result_list,
        "total_count": len(result_list),
        "query": query,
    }


def article_create(
    content: str,
    title: str | None = None,
    source_ids: list[str] | None = None,
    author_type: str = "system",
    domain_path: list[str] | None = None,
) -> dict[str, Any]:
    """Create a new article."""
    from valence.core.articles import create_article

    result = run_async(
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
    """Get an article by ID."""
    from valence.core.articles import get_article

    result = run_async(get_article(article_id=article_id, include_provenance=include_provenance))
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "article": result.data}


def article_update(
    article_id: str,
    content: str,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Update an article's content."""
    from valence.core.articles import update_article

    result = run_async(
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
    limit: int = 20,
    domain_filter: list[str] | None = None,
) -> dict[str, Any]:
    """Search articles by content."""
    from valence.core.articles import search_articles

    result = run_async(
        search_articles(
            query=query,
            limit=limit,
            domain_filter=domain_filter,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    articles = result.data or []
    return {
        "success": True,
        "articles": articles,
        "count": len(articles),
    }


def article_compile(
    source_ids: list[str],
    title_hint: str | None = None,
) -> dict[str, Any]:
    """Compile one or more sources into a new knowledge article."""
    if not source_ids:
        return {"success": False, "error": "source_ids must be a non-empty list"}

    try:
        from valence.core.compilation import compile_article

        result = run_async(compile_article(source_ids=source_ids, title_hint=title_hint))
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
    """Split an oversized article into two smaller articles."""
    if not article_id:
        return {"success": False, "error": "article_id is required"}

    try:
        from valence.core.articles import split_article

        result = run_async(split_article(article_id=article_id))
        if not result.success:
            return {"success": False, "error": result.error}
        return {"success": True, **result.data}
    except Exception as exc:
        logger.exception("article_split failed")
        return {"success": False, "error": str(exc)}


def article_merge(article_id_a: str, article_id_b: str) -> dict[str, Any]:
    """Merge two related articles into one."""
    if not article_id_a or not article_id_b:
        return {"success": False, "error": "Both article_id_a and article_id_b are required"}

    try:
        from valence.core.articles import merge_articles

        result = run_async(merge_articles(article_id_a=article_id_a, article_id_b=article_id_b))
        if not result.success:
            return {"success": False, "error": result.error}
        return {"success": True, "merged_article": result.data}
    except Exception as exc:
        logger.exception("article_merge failed")
        return {"success": False, "error": str(exc)}
