"""Stats command."""

from __future__ import annotations

import argparse
import logging

from ..utils import get_db_connection

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the stats command on the CLI parser."""
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)


def cmd_stats(args: argparse.Namespace) -> int:
    """Show database statistics."""
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) as total FROM beliefs")
        total = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) as active FROM beliefs WHERE status = 'active' AND superseded_by_id IS NULL")
        active = cur.fetchone()["active"]

        cur.execute("SELECT COUNT(*) as with_emb FROM beliefs WHERE embedding IS NOT NULL")
        with_embedding = cur.fetchone()["with_emb"]

        cur.execute("SELECT COUNT(*) as tensions FROM tensions WHERE status = 'detected'")
        tensions = cur.fetchone()["tensions"]

        try:
            cur.execute("SELECT COUNT(DISTINCT d) as count FROM beliefs, LATERAL unnest(domain_path) as d")
            domains = cur.fetchone()["count"]
        except (Exception,) as e:
            logger.debug(f"Could not count domains (column may not exist): {e}")
            domains = 0

        # Count federated beliefs
        try:
            cur.execute("SELECT COUNT(*) as federated FROM beliefs WHERE is_local = FALSE")
            federated = cur.fetchone()["federated"]
        except (Exception,) as e:
            logger.debug(f"Could not count federated beliefs (column may not exist): {e}")
            federated = 0

        print("📊 Valence Statistics")
        print("─" * 30)
        print(f"  Total beliefs:      {total}")
        print(f"  Active beliefs:     {active}")
        print(f"  Local beliefs:      {active - federated}")
        print(f"  Federated beliefs:  {federated}")
        print(f"  With embeddings:    {with_embedding}")
        print(f"  Unique domains:     {domains}")
        print(f"  Unresolved tensions:{tensions}")

        return 0

    except Exception as e:
        print(f"❌ Stats failed: {e}")
        return 1
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
