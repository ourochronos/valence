"""CLI command modules for Valence.

Each module exposes a ``register(subparsers)`` function that wires up
its argparse sub-commands and sets ``parser.set_defaults(func=handler)``.
"""

from . import (
    articles,
    beliefs,
    config_cmd,
    conflicts,
    embeddings,
    io,
    maintenance,
    migration,
    provenance,
    qos,
    sources,
    stats,
)
from .articles import cmd_articles_create, cmd_articles_get, cmd_articles_search
from .beliefs import cmd_add, cmd_init, cmd_list, cmd_query
from .conflicts import cmd_conflicts
from .embeddings import cmd_embeddings
from .io import cmd_export, cmd_import
from .maintenance import cmd_maintenance
from .migration import cmd_migrate
from .provenance import cmd_provenance_get, cmd_provenance_link, cmd_provenance_trace
from .qos import cmd_qos
from .sources import cmd_sources_ingest, cmd_sources_list, cmd_sources_search
from .stats import cmd_stats

# All command modules with register() functions, in registration order.
# New v2 modules (sources, articles, provenance) are registered first for
# discoverability; legacy modules follow for backward compatibility.
COMMAND_MODULES = [
    sources,
    articles,
    provenance,
    config_cmd,
    beliefs,
    conflicts,
    stats,
    io,
    embeddings,
    migration,
    qos,
    maintenance,
]

__all__ = [
    "COMMAND_MODULES",
    # v2 commands
    "cmd_articles_create",
    "cmd_articles_get",
    "cmd_articles_search",
    "cmd_provenance_get",
    "cmd_provenance_link",
    "cmd_provenance_trace",
    "cmd_sources_ingest",
    "cmd_sources_list",
    "cmd_sources_search",
    # Legacy commands
    "cmd_add",
    "cmd_conflicts",
    "cmd_embeddings",
    "cmd_export",
    "cmd_import",
    "cmd_init",
    "cmd_list",
    "cmd_migrate",
    "cmd_qos",
    "cmd_query",
    "cmd_stats",
    "cmd_maintenance",
]
