"""Substrate tool definitions and implementations.

Tool implementations for the Valence v2 knowledge system (WU-11).

This package re-exports all public tool functions so that existing imports
continue to work after the v2 migration.
"""

from __future__ import annotations

# Re-export shared utilities (also ensures patch target backward compat)
from ._common import _validate_enum, get_cursor  # noqa: F401

# Re-export article tool implementations (WU-04, WU-06, WU-07)
from .articles import (  # noqa: F401
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

# Re-export tool definitions
from .definitions import SUBSTRATE_TOOLS  # noqa: F401
from .entities import entity_get, entity_search  # noqa: F401

# Re-export handler dispatch
from .handlers import SUBSTRATE_HANDLERS, handle_substrate_tool  # noqa: F401

# Re-export contention tools (WU-08)
from .contention import (  # noqa: F401
    contention_detect,
    contention_list,
    contention_resolve,
)

# Backward-compat aliases (tension_ → contention_)
tension_list = contention_list  # noqa: F401
tension_resolve = contention_resolve  # noqa: F401

# Re-export source tools (WU-03)
from .sources import (  # noqa: F401
    source_get,
    source_ingest,
    source_list,
    source_search,
)

# Re-export admin tools (WU-10)
from .admin import admin_forget, admin_maintenance, admin_stats  # noqa: F401

# Re-export retrieval (WU-05)
from .retrieval import knowledge_search  # noqa: F401

__all__ = [
    # Definitions
    "SUBSTRATE_TOOLS",
    # Handlers
    "SUBSTRATE_HANDLERS",
    "handle_substrate_tool",
    # Article tools
    "article_compile",
    "article_create",
    "article_get",
    "article_merge",
    "article_search",
    "article_split",
    "article_update",
    # Provenance tools
    "provenance_link",
    "provenance_get",
    "provenance_trace",
    # Entities
    "entity_get",
    "entity_search",
    # Contention tools
    "contention_detect",
    "contention_list",
    "contention_resolve",
    # Backward-compat tension aliases
    "tension_list",
    "tension_resolve",
    # Sources
    "source_get",
    "source_ingest",
    "source_list",
    "source_search",
    # Admin
    "admin_forget",
    "admin_maintenance",
    "admin_stats",
    # Retrieval
    "knowledge_search",
    # Shared (for patching)
    "_validate_enum",
    "get_cursor",
]
