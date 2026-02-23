"""MCP admin tool wrappers for forgetting, stats, and maintenance (WU-10).

Exposes three admin tools:

- ``admin_forget``       — Remove a source or article (explicit forgetting, C10).
- ``admin_stats``        — Health stats: article/source counts, queue depth, etc.
- ``admin_maintenance``  — Trigger maintenance: queue processing, score recompute.

All functions are synchronous (conforming to the MCP dispatch protocol).
Async core functions are driven via ``asyncio.run`` / event-loop bridging,
matching the pattern used in ``substrate/tools/retrieval.py``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper: run async coroutine from sync context
# ---------------------------------------------------------------------------


def _run_async(coro: Any) -> Any:
    """Drive an async coroutine from a synchronous call site.

    Handles both cases: called from within a running event loop (test fixtures,
    async contexts) and called from a plain synchronous context.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# admin_forget
# ---------------------------------------------------------------------------


def admin_forget(
    target_type: str,
    target_id: str,
) -> dict[str, Any]:
    """Remove a source or article from the knowledge system.

    This is a **complete, irreversible deletion** — no ghost references remain.

    For ``source`` targets:
    - The source is deleted from ``sources`` (cascades to ``article_sources``).
    - Affected articles are queued for recompilation in ``mutation_queue``.
    - A tombstone is created in ``tombstones``.

    For ``article`` targets:
    - The article and its provenance links are deleted.
    - Sources are **not** affected.
    - A tombstone is created.

    MCP tool: ``admin_forget``

    Args:
        target_type: ``'source'`` or ``'article'``.
        target_id:   UUID string of the record to delete.

    Returns:
        MCP result dict with ``success`` and deletion summary, or ``error``.

    Example::

        admin_forget("source", "3fa85f64-5717-4562-b3fc-2c963f66afa6")
        # → {"success": True, "source_id": "...", "affected_articles": 3,
        #     "recompile_queued": 3, "tombstone_created": True}

        admin_forget("article", "7c9e6679-7425-40de-944b-e07fc1f90ae7")
        # → {"success": True, "article_id": "...", "tombstone_created": True}
    """
    if target_type not in ("source", "article"):
        return {
            "success": False,
            "error": f"target_type must be 'source' or 'article', got '{target_type}'",
        }
    if not target_id or not target_id.strip():
        return {"success": False, "error": "target_id is required"}

    try:
        if target_type == "source":
            from ...core.forgetting import remove_source

            return _run_async(remove_source(target_id))
        else:
            from ...core.forgetting import remove_article

            return _run_async(remove_article(target_id))
    except Exception as exc:
        logger.exception("admin_forget failed: target_type=%s id=%s", target_type, target_id)
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# admin_stats
# ---------------------------------------------------------------------------


def admin_stats() -> dict[str, Any]:
    """Return health and usage statistics for the knowledge system.

    Queries:
    - Article count (total, active, pinned)
    - Source count
    - Pending mutation_queue depth (by operation type)
    - Tombstone count (last 30 days)
    - Bounded memory capacity (from system_config)

    MCP tool: ``admin_stats``

    Returns:
        Dict with ``success`` and ``stats`` sub-dict.

    Example::

        admin_stats()
        # → {"success": True, "stats": {
        #       "articles": {"total": 8450, "active": 8200, "pinned": 12},
        #       "sources": {"total": 31000},
        #       "mutation_queue": {"pending": 5, "by_operation": {...}},
        #       "tombstones": {"last_30_days": 14},
        #       "capacity": {"max_articles": 10000, "utilization_pct": 82.0}
        #    }}
    """
    try:
        from valence.lib.our_db import get_cursor

        with get_cursor() as cur:
            # Articles
            cur.execute(
                """
                SELECT
                    COUNT(*)                             AS total,
                    COUNT(*) FILTER (WHERE status = 'active') AS active,
                    COUNT(*) FILTER (WHERE pinned = TRUE)     AS pinned
                FROM articles
                """
            )
            art_row = cur.fetchone() or {}
            articles = {
                "total": int(art_row.get("total") or 0),
                "active": int(art_row.get("active") or 0),
                "pinned": int(art_row.get("pinned") or 0),
            }

            # Sources
            cur.execute("SELECT COUNT(*) AS total FROM sources")
            src_row = cur.fetchone() or {}
            sources = {"total": int(src_row.get("total") or 0)}

            # Mutation queue
            cur.execute(
                """
                SELECT operation, COUNT(*) AS cnt
                FROM mutation_queue
                WHERE status = 'pending'
                GROUP BY operation
                """
            )
            queue_rows = cur.fetchall()
            by_op: dict[str, int] = {}
            total_pending = 0
            for qrow in queue_rows:
                op = qrow["operation"]
                cnt = int(qrow["cnt"])
                by_op[op] = cnt
                total_pending += cnt
            mutation_queue = {"pending": total_pending, "by_operation": by_op}

            # Tombstones (last 30 days)
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM tombstones
                WHERE deleted_at >= NOW() - INTERVAL '30 days'
                """
            )
            ts_row = cur.fetchone() or {}
            tombstones = {"last_30_days": int(ts_row.get("cnt") or 0)}

            # Capacity from system_config
            cur.execute("SELECT value FROM system_config WHERE key = 'bounded_memory'")
            cfg_row = cur.fetchone()
            if cfg_row:
                import json

                cfg = json.loads(cfg_row["value"]) if isinstance(cfg_row["value"], str) else cfg_row["value"]
                max_articles = int(cfg.get("max_articles", 10000))
            else:
                max_articles = 10000

            active = articles["active"]
            utilization = round(active / max_articles * 100, 1) if max_articles else 0.0
            capacity = {
                "max_articles": max_articles,
                "current_active": active,
                "utilization_pct": utilization,
                "over_capacity": active > max_articles,
            }

        return {
            "success": True,
            "stats": {
                "articles": articles,
                "sources": sources,
                "mutation_queue": mutation_queue,
                "tombstones": tombstones,
                "capacity": capacity,
            },
        }

    except Exception as exc:
        logger.exception("admin_stats failed")
        return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# admin_maintenance
# ---------------------------------------------------------------------------


def admin_maintenance(
    recompute_scores: bool = False,
    process_queue: bool = False,
    evict_if_over_capacity: bool = False,
    evict_count: int = 10,
) -> dict[str, Any]:
    """Trigger maintenance operations for the knowledge system.

    Operations (any combination may be requested):

    - ``recompute_scores``        — Batch-recompute usage_score for all articles.
    - ``process_queue``           — Process pending entries in mutation_queue
                                    (recompile, split, merge_candidate, decay_check).
    - ``evict_if_over_capacity``  — Run organic forgetting (evict_lowest) if the
                                    article count exceeds the configured maximum.

    MCP tool: ``admin_maintenance``

    Args:
        recompute_scores:        Recompute usage scores batch-wise (default False).
        process_queue:           Process the mutation queue (default False).
        evict_if_over_capacity:  Run organic eviction if over capacity (default False).
        evict_count:             Max articles to evict per run (default 10).

    Returns:
        Dict with ``success`` and per-operation result summaries.

    Example::

        admin_maintenance(recompute_scores=True, evict_if_over_capacity=True)
        # → {"success": True, "results": {
        #       "recompute_scores": {"updated": 8200},
        #       "evict_if_over_capacity": {"evicted": []}
        #    }}
    """
    if not any([recompute_scores, process_queue, evict_if_over_capacity]):
        return {
            "success": False,
            "error": ("At least one operation must be requested: recompute_scores, process_queue, evict_if_over_capacity"),
        }

    results: dict[str, Any] = {}
    errors: list[str] = []

    # ---- recompute_scores ----
    if recompute_scores:
        try:
            from ...core.usage import compute_usage_scores

            updated = _run_async(compute_usage_scores())
            results["recompute_scores"] = {"updated": updated}
        except Exception as exc:
            logger.exception("admin_maintenance: recompute_scores failed")
            errors.append(f"recompute_scores: {exc}")
            results["recompute_scores"] = {"error": str(exc)}

    # ---- process_queue ----
    if process_queue:
        try:
            from ...core.compilation import process_mutation_queue

            processed = process_mutation_queue()
            results["process_queue"] = {"processed": processed}
        except Exception as exc:
            logger.exception("admin_maintenance: process_queue failed")
            errors.append(f"process_queue: {exc}")
            results["process_queue"] = {"error": str(exc)}

    # ---- evict_if_over_capacity ----
    if evict_if_over_capacity:
        try:
            from ...core.forgetting import evict_lowest

            evicted = _run_async(evict_lowest(count=evict_count))
            results["evict_if_over_capacity"] = {"evicted": evicted, "count": len(evicted)}
        except Exception as exc:
            logger.exception("admin_maintenance: evict_if_over_capacity failed")
            errors.append(f"evict_if_over_capacity: {exc}")
            results["evict_if_over_capacity"] = {"error": str(exc)}

    return {
        "success": len(errors) == 0,
        "results": results,
        **({"errors": errors} if errors else {}),
    }
