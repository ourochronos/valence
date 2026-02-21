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
    cmd_articles_create,
    cmd_articles_get,
    cmd_articles_search,
    cmd_conflicts,
    cmd_embeddings,
    cmd_export,
    cmd_import,
    cmd_init,
    cmd_list,
    cmd_maintenance,
    cmd_migrate,
    cmd_provenance_get,
    cmd_provenance_link,
    cmd_provenance_trace,
    cmd_qos,
    cmd_query,
    cmd_sources_ingest,
    cmd_sources_list,
    cmd_sources_search,
    cmd_stats,
)
from .config import CLIConfig, set_cli_config
from .utils import (
    compute_confidence_score,
    compute_recency_score,
    format_age,
    format_confidence,
    multi_signal_rank,
)

__all__ = [
    "app",
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
    "cmd_maintenance",
    "cmd_migrate",
    "cmd_qos",
    "cmd_query",
    "cmd_stats",
    "compute_confidence_score",
    "compute_recency_score",
    "format_age",
    "format_confidence",
    "main",
    "multi_signal_rank",
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
v2 Knowledge System:
  valence sources list                    List knowledge sources
  valence sources ingest "content" -t web Ingest a new source
  valence sources search "python"         Search sources
  valence articles search "query"         Search compiled articles
  valence articles get <id> --provenance  Get article with provenance
  valence articles create "content"       Create a new article
  valence provenance trace <id> "claim"   Trace claim to sources
  valence provenance get <id>             List all sources for article

Legacy:
  valence query "search terms"            Search beliefs (legacy)
  valence add "Fact here" -d tech         Add belief (legacy)
  valence list -n 20                      List recent beliefs (legacy)
  valence stats                           Show statistics
  valence conflicts                       Detect contradictions

Global:
  valence --json stats                    Output as JSON
  valence --server http://host:8420 query Use remote server
        """,
    )

    # Global flags for REST client mode
    parser.add_argument("--server", metavar="URL", help="Server URL (env: VALENCE_SERVER_URL, default: http://127.0.0.1:8420)")
    parser.add_argument("--token", metavar="TOKEN", help="Auth token (env: VALENCE_TOKEN)")
    parser.add_argument("--output", choices=["json", "text", "table"], help="Output format (env: VALENCE_OUTPUT, default: text)")
    parser.add_argument("--json", action="store_const", const="json", dest="output", help="Shorthand for --output json")
    parser.add_argument("--timeout", type=float, metavar="SECS", help="Request timeout in seconds (default: 30)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Let each command module register its own parsers
    for module in COMMAND_MODULES:
        module.register(subparsers)

    return parser


def main() -> int:
    """Main entry point."""
    parser = app()
    args = parser.parse_args()

    # Initialize CLI config from global flags (overrides env and file)
    config = CLIConfig.load(
        server_url=getattr(args, "server", None),
        token=getattr(args, "token", None),
        output=getattr(args, "output", None),
        timeout=getattr(args, "timeout", None),
    )
    set_cli_config(config)

    # Dispatch via the func default set by each command's register()
    handler = getattr(args, "func", None)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
