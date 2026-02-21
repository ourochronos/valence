"""Substrate tool dispatch and handler mapping.

Provides:
    SUBSTRATE_HANDLERS -- tool name to handler function mapping
    handle_substrate_tool -- dispatch function
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .admin import admin_forget, admin_maintenance, admin_stats
from .articles import (
    article_compile,
    article_create,
    article_get,
    article_merge,
    article_search,
    article_split,
    article_update,
    provenance_get,
    provenance_link,
    provenance_trace,
)
from .contention import contention_detect, contention_list, contention_resolve
from .entities import entity_get, entity_search
from .retrieval import knowledge_search
from .sources import source_get, source_ingest, source_list, source_search

# Tool name to handler mapping
SUBSTRATE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    # Source tools (WU-03)
    "source_ingest": source_ingest,
    "source_get": source_get,
    "source_search": source_search,
    "source_list": source_list,
    # Article tools (WU-04, WU-06, WU-07)
    "article_create": article_create,
    "article_get": article_get,
    "article_update": article_update,
    "article_search": article_search,
    "article_compile": article_compile,
    "article_split": article_split,
    "article_merge": article_merge,
    # Provenance tools (WU-04)
    "provenance_link": provenance_link,
    "provenance_get": provenance_get,
    "provenance_trace": provenance_trace,
    # Entity tools
    "entity_get": entity_get,
    "entity_search": entity_search,
    # Contention tools (WU-08)
    "contention_detect": contention_detect,
    "contention_list": contention_list,
    "contention_resolve": contention_resolve,
    # Retrieval (WU-05)
    "knowledge_search": knowledge_search,
    # Admin tools (WU-10 / WU-11)
    "admin_forget": admin_forget,
    "admin_stats": admin_stats,
    "admin_maintenance": admin_maintenance,
}


def handle_substrate_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle a substrate tool call.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        Tool result dictionary
    """
    handler = SUBSTRATE_HANDLERS.get(name)
    if handler is None:
        return {"success": False, "error": f"Unknown substrate tool: {name}"}

    return handler(**arguments)
