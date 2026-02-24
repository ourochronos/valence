# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Admin tool handlers."""

from __future__ import annotations

import logging
from typing import Any

from valence.core.health import DatabaseStats

from ._utils import run_async

logger = logging.getLogger(__name__)


def admin_forget(target_type: str, target_id: str) -> dict[str, Any]:
    """Permanently remove a source or article."""
    if target_type not in ["source", "article"]:
        return {"success": False, "error": "target_type must be 'source' or 'article'"}

    try:
        if target_type == "source":
            from valence.core.forgetting import remove_source

            result = run_async(remove_source(target_id))
        else:
            from valence.core.forgetting import remove_article

            result = run_async(remove_article(target_id))

        if not result.success:
            return {"success": False, "error": result.error}
        return {"success": True, **result.data}
    except Exception as exc:
        logger.exception("admin_forget failed")
        return {"success": False, "error": str(exc)}


def admin_stats() -> dict[str, Any]:
    """Return health and capacity statistics."""
    try:
        stats = DatabaseStats.collect()
        return {"success": True, "stats": stats.to_dict()}
    except Exception as exc:
        logger.exception("admin_stats failed")
        return {"success": False, "error": str(exc)}


def admin_maintenance(
    recompute_scores: bool = False,
    process_queue: bool = False,
    evict_if_over_capacity: bool = False,
    evict_count: int = 10,
) -> dict[str, Any]:
    """Trigger maintenance operations."""
    results = {}

    try:
        if recompute_scores:
            from valence.core.usage import compute_usage_scores

            result = run_async(compute_usage_scores())
            results["recompute_scores"] = result.success
            if not result.success:
                results["recompute_scores_error"] = result.error

        if process_queue:
            from valence.core.compilation import process_mutation_queue

            result = run_async(process_mutation_queue())
            results["process_queue"] = result.success
            if not result.success:
                results["process_queue_error"] = result.error

        if evict_if_over_capacity:
            from valence.core.forgetting import evict_lowest

            result = run_async(evict_lowest(count=evict_count))
            results["evict_if_over_capacity"] = result.success
            if not result.success:
                results["evict_error"] = result.error

        return {"success": True, "maintenance_results": results}
    except Exception as exc:
        logger.exception("admin_maintenance failed")
        return {"success": False, "error": str(exc)}
