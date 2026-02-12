"""Maintenance commands: retention, archive, compact, views, vacuum, all (#363)."""

from __future__ import annotations

import argparse
import json

from ..utils import get_db_connection


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register maintenance command on the CLI parser."""
    maint_parser = subparsers.add_parser("maintenance", help="Run database maintenance operations")
    maint_parser.add_argument("--retention", action="store_true", help="Apply retention policies (delete old data)")
    maint_parser.add_argument("--archive", action="store_true", help="Archive stale superseded beliefs")
    maint_parser.add_argument("--tombstones", action="store_true", help="Clean up expired tombstones")
    maint_parser.add_argument("--compact", action="store_true", help="Compact exchanges in completed sessions")
    maint_parser.add_argument("--views", action="store_true", help="Refresh materialized views")
    maint_parser.add_argument("--vacuum", action="store_true", help="Run VACUUM ANALYZE on key tables")
    maint_parser.add_argument("--all", action="store_true", dest="run_all", help="Run full maintenance cycle")
    maint_parser.add_argument("--dry-run", action="store_true", help="Report what would happen without making changes")
    maint_parser.add_argument("--json", action="store_true", dest="output_json", help="Output as JSON")
    maint_parser.set_defaults(func=cmd_maintenance)


def cmd_maintenance(args: argparse.Namespace) -> int:
    """Run maintenance operations."""
    from ...core.maintenance import (
        MaintenanceResult,
        apply_retention,
        archive_beliefs,
        cleanup_tombstones,
        compact_exchanges,
        refresh_views,
        run_full_maintenance,
        vacuum_analyze,
    )

    # Determine what to run
    run_all = args.run_all
    specific = args.retention or args.archive or args.tombstones or args.compact or args.views or args.vacuum

    if not run_all and not specific:
        print("No operation specified. Use --all for full cycle, or specify individual operations.")
        print("  --retention   Apply retention policies")
        print("  --archive     Archive stale beliefs")
        print("  --tombstones  Clean expired tombstones")
        print("  --compact     Compact exchanges in completed sessions")
        print("  --views       Refresh materialized views")
        print("  --vacuum      VACUUM ANALYZE key tables")
        print("  --all         Run all of the above in order")
        print("  --dry-run     Preview without changes")
        return 1

    conn = None
    cur = None
    results: list[MaintenanceResult] = []

    try:
        conn = get_db_connection()

        if run_all:
            # Full maintenance needs autocommit for VACUUM
            if not args.dry_run:
                conn.autocommit = True
            cur = conn.cursor()
            results = run_full_maintenance(cur, dry_run=args.dry_run)
        else:
            # Individual operations
            cur = conn.cursor()

            if args.retention:
                results.extend(apply_retention(cur, dry_run=args.dry_run))

            if args.archive:
                results.append(archive_beliefs(cur, dry_run=args.dry_run))

            if args.tombstones:
                results.append(cleanup_tombstones(cur, dry_run=args.dry_run))

            if args.compact:
                results.append(compact_exchanges(cur, dry_run=args.dry_run))

            if args.views and not args.dry_run:
                results.append(refresh_views(cur))
            elif args.views and args.dry_run:
                results.append(MaintenanceResult(
                    operation="refresh_views",
                    details={"note": "skipped in dry-run mode"},
                    dry_run=True,
                ))

            if args.vacuum and not args.dry_run:
                conn.autocommit = True
                results.append(vacuum_analyze(cur))
            elif args.vacuum and args.dry_run:
                results.append(MaintenanceResult(
                    operation="vacuum_analyze",
                    details={"note": "skipped in dry-run mode"},
                    dry_run=True,
                ))

            if not conn.autocommit:
                conn.commit()

        # Output results
        if args.output_json:
            output = [{"operation": r.operation, "dry_run": r.dry_run, **r.details} for r in results]
            print(json.dumps(output, indent=2, default=str))
        else:
            dry_label = " (dry run)" if args.dry_run else ""
            print(f"Maintenance Report{dry_label}")
            print("=" * 50)
            for r in results:
                print(f"  {r}")
            print("=" * 50)
            print(f"  {len(results)} operation(s) completed")

        return 0

    except Exception as e:
        print(f"Maintenance failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
