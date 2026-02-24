# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Provenance tool handlers."""

from __future__ import annotations

from typing import Any

from ._utils import run_async


def provenance_link(
    article_id: str,
    source_id: str,
    relationship: str = "confirms",
) -> dict[str, Any]:
    """Link a source to an article with a provenance relationship."""
    from valence.core.provenance import link_source

    result = run_async(
        link_source(
            article_id=article_id,
            source_id=source_id,
            relationship=relationship,
        )
    )
    if not result.success:
        return {"success": False, "error": result.error}
    return {"success": True, "link": result.data}


def provenance_get(article_id: str) -> dict[str, Any]:
    """Get all provenance sources for an article."""
    from valence.core.provenance import get_provenance

    result = run_async(get_provenance(article_id=article_id))
    if not result.success:
        return {"success": False, "error": result.error}
    provenance = result.data or []
    return {
        "success": True,
        "provenance": provenance,
        "count": len(provenance),
    }


def provenance_trace(article_id: str, claim_text: str) -> dict[str, Any]:
    """Trace which sources likely contributed a specific claim."""
    from valence.core.provenance import trace_claim

    result = run_async(trace_claim(article_id=article_id, claim_text=claim_text))
    if not result.success:
        return {"success": False, "error": result.error}
    sources = result.data or []
    return {
        "success": True,
        "sources": sources,
        "count": len(sources),
    }
