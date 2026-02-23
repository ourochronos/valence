"""MCP tool wrapper for unified knowledge retrieval (C9).

Exposes the ``knowledge_search`` tool which calls ``valence.core.retrieval.retrieve``
synchronously (the MCP dispatch layer is sync; asyncio is handled internally).

Tool contract:
    knowledge_search(query, limit=10, include_sources=False) -> dict
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool implementation (sync — MCP dispatch protocol)
# ---------------------------------------------------------------------------


def knowledge_search(
    query: str,
    limit: int = 10,
    include_sources: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Unified knowledge retrieval.

    Searches compiled articles and, optionally, ungrouped raw sources.
    Results are ranked by: relevance × 0.5 + confidence × 0.35 + freshness × 0.15.

    Ungrouped sources that match will be queued for compilation (``mutation_queue``).
    Article retrievals are recorded in ``usage_traces`` for self-organisation.

    Args:
        query:           Natural-language search query.
        limit:           Maximum results to return (default 10, max 200).
        include_sources: Include ungrouped raw sources in results (default False).
        session_id:      Optional session ID for usage trace attribution.

    Returns:
        ``{"success": True, "results": [...], "total_count": int}`` on success.
        ``{"success": False, "error": "..."}`` on validation failure.

    Each result dict contains:
        - ``id``: UUID of the article or source
        - ``type``: ``"article"`` or ``"source"``
        - ``content``: Text content
        - ``title``: Optional title
        - ``final_score``: Combined ranking score [0, 1]
        - ``confidence``: Confidence dict with ``"overall"`` key
        - ``freshness``: Days since last update (float)
        - ``freshness_score``: Freshness as [0, 1] decay score
        - ``provenance_summary``: ``{source_count, relationship_types}``
        - ``active_contentions``: bool — are there unresolved contentions?
    """
    if not query or not query.strip():
        return {"success": False, "error": "query must be non-empty"}

    limit = max(1, min(int(limit), 200))

    try:
        from ...core.retrieval import retrieve

        # Run the async retrieve() in the current thread's event loop (or a new one).
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already inside an event loop (e.g., during async tests).
                # Use asyncio.run_coroutine_threadsafe or run in a new thread.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        retrieve(query, limit=limit, include_sources=include_sources, session_id=session_id),
                    )
                    results = future.result(timeout=30)
            else:
                results = loop.run_until_complete(retrieve(query, limit=limit, include_sources=include_sources, session_id=session_id))
        except RuntimeError:
            # No current event loop
            results = asyncio.run(retrieve(query, limit=limit, include_sources=include_sources, session_id=session_id))

    except Exception as exc:
        logger.exception("knowledge_search failed for query %r: %s", query, exc)
        return {"success": False, "error": str(exc)}

    # retrieve() now returns ValenceResponse — unwrap to MCP dict format
    from ...core.response import ValenceResponse

    if isinstance(results, ValenceResponse):
        if not results.success:
            return {"success": False, "error": results.error}
        result_list = results.data or []
    else:
        result_list = results  # backward compat

    return {
        "success": True,
        "results": result_list,
        "total_count": len(result_list),
        "query": query,
    }
