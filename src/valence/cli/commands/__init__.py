"""CLI command modules for Valence."""

from .beliefs import cmd_add, cmd_init, cmd_list, cmd_query
from .conflicts import cmd_conflicts
from .discovery import cmd_discover
from .embeddings import cmd_embeddings
from .federation import cmd_query_federated
from .io import cmd_export, cmd_import
from .migration import cmd_migrate, cmd_migrate_visibility
from .peers import cmd_peer, cmd_peer_add, cmd_peer_list, cmd_peer_remove
from .resources import cmd_resources
from .schema import cmd_schema
from .stats import cmd_stats
from .trust import cmd_trust

__all__ = [
    "cmd_add",
    "cmd_conflicts",
    "cmd_discover",
    "cmd_embeddings",
    "cmd_export",
    "cmd_import",
    "cmd_init",
    "cmd_list",
    "cmd_migrate",
    "cmd_migrate_visibility",
    "cmd_peer",
    "cmd_peer_add",
    "cmd_peer_list",
    "cmd_peer_remove",
    "cmd_query",
    "cmd_query_federated",
    "cmd_resources",
    "cmd_schema",
    "cmd_stats",
    "cmd_trust",
]
