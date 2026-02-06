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

# Re-export for backward compatibility
from .commands import (
    cmd_add,
    cmd_conflicts,
    cmd_discover,
    cmd_embeddings,
    cmd_export,
    cmd_import,
    cmd_init,
    cmd_list,
    cmd_migrate_visibility,
    cmd_peer,
    cmd_peer_add,
    cmd_peer_list,
    cmd_peer_remove,
    cmd_query,
    cmd_query_federated,
    cmd_stats,
    cmd_trust,
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
    "cmd_conflicts",
    "cmd_discover",
    "cmd_embeddings",
    "cmd_export",
    "cmd_import",
    "cmd_init",
    "cmd_list",
    "cmd_migrate_visibility",
    "cmd_peer",
    "cmd_peer_add",
    "cmd_peer_list",
    "cmd_peer_remove",
    "cmd_query",
    "cmd_query_federated",
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
    """Build the argument parser."""
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

    # init
    init_parser = subparsers.add_parser("init", help="Initialize database schema")
    init_parser.add_argument("--force", "-f", action="store_true", help="Recreate schema even if exists")

    # add
    add_parser = subparsers.add_parser("add", help="Add a new belief")
    add_parser.add_argument("content", help="Belief content")
    add_parser.add_argument("--confidence", "-c", help="Confidence (JSON or float 0-1)")
    add_parser.add_argument("--domain", "-d", action="append", help="Domain tag (repeatable)")
    add_parser.add_argument(
        "--derivation-type",
        "-t",
        choices=[
            "observation",
            "inference",
            "aggregation",
            "hearsay",
            "assumption",
            "correction",
            "synthesis",
        ],
        default="observation",
        help="How this belief was derived",
    )
    add_parser.add_argument("--derived-from", help="UUID of source belief this was derived from")
    add_parser.add_argument("--method", "-m", help="Method description for derivation")

    # query
    query_parser = subparsers.add_parser("query", help="Search beliefs with multi-signal ranking")
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    query_parser.add_argument("--threshold", "-t", type=float, default=0.3, help="Min semantic similarity")
    query_parser.add_argument("--domain", "-d", help="Filter by domain")
    query_parser.add_argument("--chain", action="store_true", help="Show full supersession chains")
    query_parser.add_argument(
        "--scope",
        "-s",
        choices=["local", "federated"],
        default="local",
        help="Search scope: local (default) or federated (include peer beliefs)",
    )
    # Multi-signal ranking options (Valence Query Protocol)
    query_parser.add_argument(
        "--recency-weight",
        "-r",
        type=float,
        default=0.15,
        help="Recency weight 0.0-1.0 (default 0.15). Higher = prefer newer beliefs",
    )
    query_parser.add_argument(
        "--min-confidence",
        "-c",
        type=float,
        default=None,
        help="Filter beliefs below this confidence threshold (0.0-1.0)",
    )
    query_parser.add_argument(
        "--explain",
        "-e",
        action="store_true",
        help="Show detailed score breakdown per result",
    )

    # list
    list_parser = subparsers.add_parser("list", help="List recent beliefs")
    list_parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    list_parser.add_argument("--domain", "-d", help="Filter by domain")

    # conflicts
    conflicts_parser = subparsers.add_parser("conflicts", help="Detect contradicting beliefs")
    conflicts_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.85,
        help="Similarity threshold for conflict detection",
    )
    conflicts_parser.add_argument(
        "--auto-record",
        "-r",
        action="store_true",
        help="Automatically record detected conflicts as tensions",
    )

    # stats
    subparsers.add_parser("stats", help="Show database statistics")

    # ========================================================================
    # DISCOVER command (Network bootstrap)
    # ========================================================================

    discover_parser = subparsers.add_parser("discover", help="Discover network routers via seeds")
    discover_parser.add_argument(
        "--seed",
        "-s",
        action="append",
        dest="seeds",
        help="Custom seed URL (repeatable)",
    )
    discover_parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of routers to request (default: 5)",
    )
    discover_parser.add_argument("--region", "-r", help="Preferred region")
    discover_parser.add_argument(
        "--feature",
        "-f",
        action="append",
        dest="features",
        help="Required feature (repeatable)",
    )
    discover_parser.add_argument("--refresh", action="store_true", help="Force refresh (bypass cache)")
    discover_parser.add_argument("--no-verify", action="store_true", help="Skip router signature verification")
    discover_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    discover_parser.add_argument("--stats", action="store_true", help="Show discovery statistics")

    # ========================================================================
    # PEER commands (Week 2 Federation)
    # ========================================================================

    peer_parser = subparsers.add_parser("peer", help="Manage trusted peers")
    peer_subparsers = peer_parser.add_subparsers(dest="peer_command", required=True)

    # peer add
    peer_add_parser = peer_subparsers.add_parser("add", help="Add or update a trusted peer")
    peer_add_parser.add_argument("did", help="Peer DID (e.g., did:vkb:web:alice.example.com)")
    peer_add_parser.add_argument(
        "--trust",
        type=float,
        required=True,
        help="Trust level 0.0-1.0 (e.g., 0.8 for 80%% trust)",
    )
    peer_add_parser.add_argument("--name", help="Human-readable name for this peer")
    peer_add_parser.add_argument("--notes", help="Notes about this peer")

    # peer list
    peer_subparsers.add_parser("list", help="List trusted peers")

    # peer remove
    peer_remove_parser = peer_subparsers.add_parser("remove", help="Remove a peer")
    peer_remove_parser.add_argument("did", help="Peer DID to remove")

    # ========================================================================
    # EXPORT command
    # ========================================================================

    export_parser = subparsers.add_parser("export", help="Export beliefs for sharing")
    export_parser.add_argument("--to", dest="to", help="Recipient DID (for filtering)")
    export_parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    export_parser.add_argument("--domain", "-d", action="append", help="Filter by domain")
    export_parser.add_argument("--min-confidence", type=float, default=0.0, help="Minimum confidence threshold")
    export_parser.add_argument("--limit", "-n", type=int, default=1000, help="Max beliefs")
    export_parser.add_argument(
        "--include-federated",
        action="store_true",
        help="Include beliefs received from other peers",
    )

    # ========================================================================
    # IMPORT command
    # ========================================================================

    import_parser = subparsers.add_parser("import", help="Import beliefs from a peer")
    import_parser.add_argument("file", help="Import file (JSON) or - for stdin")
    import_parser.add_argument("--from", dest="source", help="Source peer DID (overrides package)")
    import_parser.add_argument("--trust", type=float, help="Override trust level (otherwise uses registry)")

    # ========================================================================
    # TRUST commands
    # ========================================================================

    trust_parser = subparsers.add_parser("trust", help="Trust network management")
    trust_subparsers = trust_parser.add_subparsers(dest="trust_command", required=True)

    # trust check
    trust_check_parser = trust_subparsers.add_parser("check", help="Check for trust concentration issues")
    trust_check_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    trust_check_parser.add_argument(
        "--single-threshold",
        type=float,
        default=None,
        help="Custom threshold for single node dominance (default: 30%%)",
    )
    trust_check_parser.add_argument(
        "--top3-threshold",
        type=float,
        default=None,
        help="Custom threshold for top 3 nodes dominance (default: 50%%)",
    )
    trust_check_parser.add_argument(
        "--min-sources",
        type=int,
        default=None,
        help="Minimum trusted sources (default: 3)",
    )

    # trust watch
    trust_watch_parser = trust_subparsers.add_parser("watch", help="Watch an entity (see content without reputation boost)")
    trust_watch_parser.add_argument("entity", help="DID of entity to watch")
    trust_watch_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # trust unwatch
    trust_unwatch_parser = trust_subparsers.add_parser("unwatch", help="Remove a watch relationship")
    trust_unwatch_parser.add_argument("entity", help="DID of entity to unwatch")
    trust_unwatch_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # trust distrust
    trust_distrust_parser = trust_subparsers.add_parser("distrust", help="Mark an entity as distrusted (negative reputation)")
    trust_distrust_parser.add_argument("entity", help="DID of entity to distrust")
    trust_distrust_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # trust ignore
    trust_ignore_parser = trust_subparsers.add_parser("ignore", help="Ignore an entity (block content)")
    trust_ignore_parser.add_argument("entity", help="DID of entity to ignore")
    trust_ignore_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # ========================================================================
    # EMBEDDINGS commands
    # ========================================================================

    embeddings_parser = subparsers.add_parser("embeddings", help="Embedding management")
    embeddings_subparsers = embeddings_parser.add_subparsers(dest="embeddings_command", required=True)

    # embeddings backfill
    backfill_parser = embeddings_subparsers.add_parser("backfill", help="Backfill missing or outdated embeddings")
    backfill_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Number of records to process per batch (default: 100)",
    )
    backfill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backfilled without making changes",
    )
    backfill_parser.add_argument(
        "--content-type",
        "-t",
        choices=["belief", "exchange", "pattern"],
        default=None,
        help="Content type to backfill (default: all)",
    )
    backfill_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-embed all records even if embedding exists (for provider migration)",
    )

    # ========================================================================
    # MIGRATE-VISIBILITY command
    # ========================================================================

    subparsers.add_parser(
        "migrate-visibility",
        help="Migrate existing beliefs from visibility to SharePolicy",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = app()
    args = parser.parse_args()

    commands = {
        "init": cmd_init,
        "add": cmd_add,
        "query": cmd_query,
        "list": cmd_list,
        "conflicts": cmd_conflicts,
        "stats": cmd_stats,
        "discover": cmd_discover,
        "peer": cmd_peer,
        "export": cmd_export,
        "import": cmd_import,
        "trust": cmd_trust,
        "embeddings": cmd_embeddings,
        "migrate-visibility": cmd_migrate_visibility,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
