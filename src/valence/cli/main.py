#!/usr/bin/env python3
"""
Valence CLI - Personal knowledge substrate for AI agents.

Commands:
  valence init              Initialize database (creates schema)
  valence add <content>     Add a new belief
  valence query <text>      Search beliefs with derivation chains
  valence list              List recent beliefs
  valence conflicts         Detect contradicting beliefs
  valence stats             Show database statistics
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Re-export for backward compatibility (tests import from valence.cli.main)
from .commands import (
    COMMAND_MODULES,
    cmd_add,
    cmd_attestations,
    cmd_conflicts,
    cmd_discover,
    cmd_embeddings,
    cmd_export,
    cmd_identity,
    cmd_import,
    cmd_init,
    cmd_list,
    cmd_maintenance,
    cmd_migrate,
    cmd_migrate_visibility,
    cmd_peer,
    cmd_peer_add,
    cmd_peer_list,
    cmd_peer_remove,
    cmd_qos,
    cmd_query,
    cmd_query_federated,
    cmd_resources,
    cmd_schema,
    cmd_stats,
    cmd_trust,
    register_identity_commands,
)
from .utils import (
    compute_confidence_score,
    compute_recency_score,
    format_age,
    format_confidence,
    get_db_connection,
    get_embedding,
    multi_signal_rank,
)

__all__ = [
    "app",
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
    "cmd_maintenance",
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
    "cmd_trust",
    "compute_confidence_score",
    "compute_recency_score",
    "format_age",
    "format_confidence",
    "get_db_connection",
    "get_embedding",
    "main",
    "multi_signal_rank",
    "register_identity_commands",
]

# Try to load .env from common locations
for env_path in [Path.cwd() / ".env", Path.home() / ".valence" / ".env"]:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())
        break


def app() -> argparse.ArgumentParser:
    """Build the argument parser.

    Each command module in ``valence.cli.commands`` provides a
    ``register(subparsers)`` function that adds its parser(s) and sets
    ``parser.set_defaults(func=handler)`` so dispatch is automatic.
    """
    parser = argparse.ArgumentParser(
        prog="valence",
        description="Personal knowledge substrate for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  valence init                            Initialize database
  valence add "Fact here" -d tech         Add belief with domain
  valence query "search terms"            Search beliefs
  valence query "terms" --scope federated Include peer beliefs
  valence list -n 20                      List recent beliefs
  valence conflicts                       Detect contradictions
  valence stats                           Show statistics

Network:
  valence discover                        Discover network routers
  valence discover --seed <url>           Use custom seed

Federation (Week 2):
  valence peer add <did> --trust 0.8      Add trusted peer
  valence peer list                       Show trusted peers
  valence export --to <did> -o file.json  Export beliefs for peer
  valence import file.json --from <did>   Import from peer
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Let each command module register its own parsers
    for module in COMMAND_MODULES:
        module.register(subparsers)

    return parser


def main() -> int:
    """Main entry point."""
    parser = app()
    args = parser.parse_args()

    # Dispatch via the func default set by each command's register()
    handler = getattr(args, "func", None)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
