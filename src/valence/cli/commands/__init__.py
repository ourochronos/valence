"""CLI command modules.

This package organizes CLI commands into logical groups:
- beliefs: Core belief operations (init, add, query, list, stats)
- trust: Trust network management (peer commands, trust check)
- federation: Federated knowledge sharing (export, import)
- admin: Administrative operations (discover, conflicts, migrate-visibility)
"""

from .beliefs import (
    # Helper functions
    get_db_connection,
    get_embedding,
    format_confidence,
    format_age,
    compute_confidence_score,
    compute_recency_score,
    multi_signal_rank,
    # Commands
    cmd_init,
    cmd_add,
    cmd_query,
    cmd_list,
    cmd_stats,
)

from .trust import (
    cmd_peer,
    cmd_peer_add,
    cmd_peer_list,
    cmd_peer_remove,
    cmd_trust,
    cmd_trust_check,
)

from .federation import (
    cmd_query_federated,
    cmd_export,
    cmd_import,
)

from .admin import (
    cmd_discover,
    cmd_conflicts,
    cmd_migrate_visibility,
)

__all__ = [
    # Helper functions
    "get_db_connection",
    "get_embedding",
    "format_confidence",
    "format_age",
    "compute_confidence_score",
    "compute_recency_score",
    "multi_signal_rank",
    # Belief commands
    "cmd_init",
    "cmd_add",
    "cmd_query",
    "cmd_list",
    "cmd_stats",
    # Trust commands
    "cmd_peer",
    "cmd_peer_add",
    "cmd_peer_list",
    "cmd_peer_remove",
    "cmd_trust",
    "cmd_trust_check",
    # Federation commands
    "cmd_query_federated",
    "cmd_export",
    "cmd_import",
    # Admin commands
    "cmd_discover",
    "cmd_conflicts",
    "cmd_migrate_visibility",
]
