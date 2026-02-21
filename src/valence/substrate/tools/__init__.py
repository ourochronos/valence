"""Substrate tool definitions and implementations.

Tool implementations for the Valence v2 knowledge system (WU-04).

This package re-exports all public tool functions so that existing imports
continue to work after the beliefs→articles migration.
"""

from __future__ import annotations

# Re-export shared utilities (also ensures patch target backward compat)
from ._common import _validate_enum, get_cursor  # noqa: F401

# Re-export article tool implementations (WU-04, replaces beliefs.py)
from .articles import (  # noqa: F401
    article_create,
    article_get,
    article_search,
    article_update,
    provenance_get,
    provenance_link,
    provenance_trace,
)

# Re-export tool definitions
from .definitions import SUBSTRATE_TOOLS  # noqa: F401
from .entities import entity_get, entity_search  # noqa: F401

# Re-export handler dispatch
from .handlers import SUBSTRATE_HANDLERS, handle_substrate_tool  # noqa: F401
from .tensions import tension_list, tension_resolve  # noqa: F401

__all__ = [
    # Definitions
    "SUBSTRATE_TOOLS",
    # Handlers
    "SUBSTRATE_HANDLERS",
    "handle_substrate_tool",
    # Article tools
    "article_create",
    "article_get",
    "article_update",
    "article_search",
    # Provenance tools
    "provenance_link",
    "provenance_get",
    "provenance_trace",
    # Entities
    "entity_get",
    "entity_search",
    # Tensions
    "tension_list",
    "tension_resolve",
    # Shared (for patching)
    "_validate_enum",
    "get_cursor",
]
