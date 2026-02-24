# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Contention tool handlers."""

from __future__ import annotations

from typing import Any

from ._utils import run_async


def contention_detect(
    article_id: str,
    source_id: str,
) -> dict[str, Any]:
    """Detect a contention between an article and a source."""
    from valence.core.contention import detect_contention

    result = run_async(
        detect_contention(
            article_id=article_id,
            source_id=source_id,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "contention": result.data}


def contention_list(
    article_id: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """List active contentions."""
    from valence.core.contention import list_contentions

    kwargs: dict[str, Any] = {}
    if article_id is not None:
        kwargs["article_id"] = article_id
    if status is not None:
        kwargs["status"] = status
    result = run_async(list_contentions(**kwargs))
    if not result.success:
        return {"success": False, "error": result.error}
    contentions = result.data or []
    return {
        "success": True,
        "contentions": contentions,
        "total_count": len(contentions),
    }


def contention_resolve(
    contention_id: str,
    resolution: str,
    rationale: str,
) -> dict[str, Any]:
    """Resolve a contention."""
    from valence.core.contention import resolve_contention

    result = run_async(
        resolve_contention(
            contention_id=contention_id,
            resolution=resolution,
            rationale=rationale,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "contention": result.data}
