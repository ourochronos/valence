"""CLI command modules for Valence.

Each module exposes a ``register(subparsers)`` function that wires up
its argparse sub-commands and sets ``parser.set_defaults(func=handler)``.
"""

from . import (
    attestations,
    beliefs,
    conflicts,
    discovery,
    embeddings,
    identity,
    io,
    maintenance,
    migration,
    peers,
    qos,
    resources,
    schema,
    stats,
    trust,
)
from .attestations import cmd_attestations
from .beliefs import cmd_add, cmd_init, cmd_list, cmd_query
from .conflicts import cmd_conflicts
from .discovery import cmd_discover
from .embeddings import cmd_embeddings
from .federation import cmd_query_federated
from .identity import cmd_identity, register_identity_commands
from .io import cmd_export, cmd_import
from .maintenance import cmd_maintenance
from .migration import cmd_migrate, cmd_migrate_visibility
from .peers import cmd_peer, cmd_peer_add, cmd_peer_list, cmd_peer_remove
from .qos import cmd_qos
from .resources import cmd_resources
from .schema import cmd_schema
from .stats import cmd_stats
from .trust import cmd_trust

# All command modules with register() functions, in registration order.
COMMAND_MODULES = [
    beliefs,
    conflicts,
    stats,
    discovery,
    peers,
    io,
    trust,
    embeddings,
    attestations,
    resources,
    migration,
    schema,
    qos,
    identity,
    maintenance,
]

__all__ = [
    "COMMAND_MODULES",
    "cmd_add",
    "cmd_attestations",
    "cmd_conflicts",
    "cmd_discover",
    "cmd_embeddings",
    "cmd_export",
    "cmd_identity",
    "cmd_import",
    "cmd_init",
    "cmd_list",
    "cmd_migrate",
    "cmd_migrate_visibility",
    "cmd_peer",
    "cmd_peer_add",
    "cmd_peer_list",
    "cmd_peer_remove",
    "cmd_qos",
    "cmd_query",
    "cmd_query_federated",
    "cmd_resources",
    "cmd_schema",
    "cmd_stats",
    "cmd_maintenance",
    "cmd_trust",
    "register_identity_commands",
]
