"""Substrate tool dispatch and handler mapping.

Provides:
    SUBSTRATE_HANDLERS -- tool name to handler function mapping
    handle_substrate_tool -- dispatch function
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .articles import (
    article_create,
    article_get,
    article_search,
    article_update,
    provenance_get,
    provenance_link,
    provenance_trace,
)
from .entities import entity_get, entity_search
from .tensions import tension_list, tension_resolve

# Tool name to handler mapping
SUBSTRATE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    # Article tools (WU-04)
    "article_create": article_create,
    "article_get": article_get,
    "article_update": article_update,
    "article_search": article_search,
    # Provenance tools (WU-04)
    "provenance_link": provenance_link,
    "provenance_get": provenance_get,
    "provenance_trace": provenance_trace,
    # Entity tools
    "entity_get": entity_get,
    "entity_search": entity_search,
    # Contention tools (legacy name kept for compat)
    "tension_list": tension_list,
    "tension_resolve": tension_resolve,
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
