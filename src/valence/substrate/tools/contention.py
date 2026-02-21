"""MCP tool wrappers for contention detection and surfacing (C7 — WU-08).

All public functions are synchronous (conforming to the MCP dispatch protocol)
and delegate to the async core functions via ``asyncio.run`` / event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from . import _common
from ._common import get_cursor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event-loop helper (same pattern used across substrate tools)
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine from sync context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an already-running loop (e.g. tests with pytest-asyncio).
            # Create a new event loop in a thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def contention_detect(article_id: str, source_id: str) -> dict[str, Any]:
    """Detect whether a source contradicts or contends with an article.

    Invokes the LLM (if configured) to assess materiality and type of
    disagreement. A contention record is created only when materiality is
    above the configured threshold (default 0.3).

    Args:
        article_id: UUID of the existing article.
        source_id: UUID of the incoming source to check against the article.

    Returns:
        ``{"success": True, "contention": {...}}`` if a contention was
        created, or ``{"success": True, "contention": null}`` when materiality
        is below threshold (no contention persisted).  On error:
        ``{"success": False, "error": "..."}``.
    """
    from ...core.contention import detect_contention

    if not article_id or not source_id:
        return {"success": False, "error": "article_id and source_id are required"}

    try:
        contention = _run(detect_contention(article_id, source_id))
        return {"success": True, "contention": contention}
    except Exception as exc:
        logger.exception("contention_detect failed: %s", exc)
        return {"success": False, "error": str(exc)}


def contention_list(
    article_id: str | None = None,
    status: str = "detected",
) -> dict[str, Any]:
    """List contentions, optionally filtered by article and status.

    Args:
        article_id: Optional UUID — return only contentions involving this article.
        status: Filter by status: ``detected``, ``resolved``, ``dismissed``,
            or ``null``/empty string to return all.

    Returns:
        ``{"success": True, "contentions": [...], "total_count": N}``.
    """
    from ...core.contention import list_contentions

    # Allow callers to pass empty string to mean "no filter"
    effective_status = status if status else None

    try:
        contentions = _run(list_contentions(article_id=article_id, status=effective_status))
        return {
            "success": True,
            "contentions": contentions,
            "total_count": len(contentions),
        }
    except Exception as exc:
        logger.exception("contention_list failed: %s", exc)
        return {"success": False, "error": str(exc)}


def contention_resolve(
    contention_id: str,
    resolution: str,
    rationale: str,
) -> dict[str, Any]:
    """Resolve a contention.

    Resolution types:

    - ``supersede_a``: Article's position wins; source is noted but article
      content is unchanged.
    - ``supersede_b``: Source's position replaces the article content.
    - ``accept_both``: Both perspectives are valid; the article is annotated
      with a contention note.
    - ``dismiss``: Contention is not material; dismissed without any change.

    Args:
        contention_id: UUID of the contention to resolve.
        resolution: One of ``supersede_a``, ``supersede_b``, ``accept_both``,
            ``dismiss``.
        rationale: Free-text rationale recorded on the contention row.

    Returns:
        ``{"success": True, "contention": {...}}`` with the updated contention,
        plus ``"article": {...}`` when the article was modified (supersede_b).
    """
    from ...core.contention import resolve_contention

    if not contention_id:
        return {"success": False, "error": "contention_id is required"}
    if not resolution:
        return {"success": False, "error": "resolution is required"}

    try:
        result = _run(resolve_contention(contention_id, resolution, rationale or ""))
        return result
    except Exception as exc:
        logger.exception("contention_resolve failed: %s", exc)
        return {"success": False, "error": str(exc)}
