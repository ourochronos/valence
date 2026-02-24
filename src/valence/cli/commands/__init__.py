"""CLI command modules for Valence.

Each module exposes a ``register(subparsers)`` function that wires up
its argparse sub-commands and sets ``parser.set_defaults(func=handler)``.
"""

from . import (
    articles,
    compile,
    config_cmd,
    conflicts,
    embeddings,
    ingest,
    maintenance,
    memory,
    migration,
    provenance,
    sources,
    stats,
    status,
    unified_search,
)
from .articles import cmd_articles_create, cmd_articles_get, cmd_articles_search
from .compile import cmd_compile
from .conflicts import cmd_conflicts
from .embeddings import cmd_embeddings
from .ingest import cmd_ingest
from .maintenance import cmd_maintenance
from .memory import cmd_memory_import, cmd_memory_import_dir, cmd_memory_list, cmd_memory_search
from .migration import cmd_migrate
from .provenance import cmd_provenance_get, cmd_provenance_link, cmd_provenance_trace
from .sources import cmd_sources_get, cmd_sources_ingest, cmd_sources_list, cmd_sources_search
from .stats import cmd_stats
from .status import cmd_status
from .unified_search import cmd_search

# All command modules with register() functions, in registration order.
# New v2 modules are registered first for discoverability; legacy modules follow.
COMMAND_MODULES = [
    unified_search,  # valence search
    ingest,  # valence ingest
    compile,  # valence compile
    status,  # valence status
    memory,  # valence memory
    sources,
    articles,
    provenance,
    config_cmd,
    conflicts,
    stats,
    embeddings,
    migration,
    maintenance,
]

__all__ = [
    "COMMAND_MODULES",
    # New v2 commands
    "cmd_search",
    "cmd_ingest",
    "cmd_compile",
    "cmd_status",
    # v2 commands
    "cmd_articles_create",
    "cmd_articles_get",
    "cmd_articles_search",
    "cmd_provenance_get",
    "cmd_provenance_link",
    "cmd_provenance_trace",
    "cmd_sources_get",
    "cmd_sources_ingest",
    "cmd_sources_list",
    "cmd_sources_search",
    # Memory commands
    "cmd_memory_import",
    "cmd_memory_import_dir",
    "cmd_memory_list",
    "cmd_memory_search",
    # Legacy commands
    "cmd_conflicts",
    "cmd_embeddings",
    "cmd_migrate",
    "cmd_stats",
    "cmd_maintenance",
]
