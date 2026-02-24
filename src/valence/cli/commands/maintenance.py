"""Maintenance commands: retention, archive, compact, views, vacuum, all (#363)."""

from __future__ import annotations

import argparse

from ..config import get_cli_config
from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register maintenance command on the CLI parser."""
    maint_parser = subparsers.add_parser("maintenance", help="Run database maintenance operations")

    # Create subcommands for maintenance
    maint_subparsers = maint_parser.add_subparsers(dest="maintenance_command", help="Maintenance subcommands")

    # Run maintenance operations (default behavior)
    run_parser = maint_subparsers.add_parser("run", help="Run maintenance operations")
    run_parser.add_argument("--retention", action="store_true", help="Apply retention policies (delete old data)")
    run_parser.add_argument("--archive", action="store_true", help="Archive stale superseded beliefs")
    run_parser.add_argument("--tombstones", action="store_true", help="Clean up expired tombstones")
    run_parser.add_argument("--compact", action="store_true", help="Compact exchanges in completed sessions")
    run_parser.add_argument("--views", action="store_true", help="Refresh materialized views")
    run_parser.add_argument("--vacuum", action="store_true", help="Run VACUUM ANALYZE on key tables")
    run_parser.add_argument("--all", action="store_true", dest="run_all", help="Run full maintenance cycle")
    run_parser.add_argument("--dry-run", action="store_true", help="Report what would happen without making changes")
    run_parser.set_defaults(func=cmd_maintenance_run)

    # Schedule maintenance
    schedule_parser = maint_subparsers.add_parser("schedule", help="Configure or view maintenance schedule")
    schedule_parser.add_argument("--interval", type=int, metavar="HOURS", help="Set maintenance interval in hours")
    schedule_parser.add_argument("--disable", action="store_true", help="Disable scheduled maintenance")
    schedule_parser.set_defaults(func=cmd_maintenance_schedule)

    # For backward compatibility, if no subcommand, treat as 'run'
    maint_parser.set_defaults(func=cmd_maintenance_compat)


def cmd_maintenance_run(args: argparse.Namespace) -> int:
    """Run maintenance operations via REST API."""
    config = get_cli_config()

    # Build request body from flags
    body: dict = {}
    if args.run_all:
        body["all"] = True
    if args.retention:
        body["retention"] = True
    if args.archive:
        body["archive"] = True
    if args.tombstones:
        body["tombstones"] = True
    if args.compact:
        body["compact"] = True
    if args.views:
        body["views"] = True
    if args.vacuum:
        body["vacuum"] = True
    if args.dry_run:
        body["dry_run"] = True

    if not args.run_all and not any(getattr(args, op) for op in ("retention", "archive", "tombstones", "compact", "views", "vacuum")):
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

    params: dict = {"output": config.output}

    client = get_client()
    try:
        result = client.post("/admin/maintenance", body=body, params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_maintenance_schedule(args: argparse.Namespace) -> int:
    """Configure or view maintenance schedule."""
    from ...core.db import get_cursor
    from ...core.maintenance import (
        disable_maintenance_schedule,
        get_maintenance_schedule,
        set_maintenance_schedule,
    )

    try:
        with get_cursor() as cur:
            if args.disable:
                # Disable scheduled maintenance
                disable_maintenance_schedule(cur)
                print("Scheduled maintenance disabled")
                return 0

            if args.interval is not None:
                # Set maintenance interval
                if args.interval <= 0:
                    print("Error: interval must be positive")
                    return 1

                schedule = set_maintenance_schedule(cur, args.interval)
                print("Maintenance schedule updated:")
                print(f"  Interval: {schedule['interval_hours']} hours")
                print(f"  Enabled: {schedule['enabled']}")
                print(f"  Last run: {schedule.get('last_run', 'Never')}")
                return 0

            # Show current schedule
            schedule = get_maintenance_schedule(cur)
            if schedule is None:
                print("Scheduled maintenance is disabled or not configured")
                print("\nTo enable:")
                print("  valence maintenance schedule --interval HOURS")
                return 0

            print("Current maintenance schedule:")
            print("  Enabled: True")
            print(f"  Interval: {schedule.get('interval_hours', 24)} hours")
            last_run = schedule.get("last_run")
            print(f"  Last run: {last_run if last_run else 'Never'}")
            return 0

    except Exception as e:
        output_error(f"Error managing schedule: {e}")
        return 1


def cmd_maintenance_compat(args: argparse.Namespace) -> int:
    """Backward compatibility handler when no subcommand specified."""
    # Check if any run flags are present
    has_run_flags = any(
        getattr(args, attr, False) for attr in ("retention", "archive", "tombstones", "compact", "views", "vacuum", "run_all", "dry_run")
    )

    if has_run_flags:
        # Treat as 'run' command
        return cmd_maintenance_run(args)
    else:
        # Show help
        print("Usage: valence maintenance <subcommand>")
        print("")
        print("Subcommands:")
        print("  run       Run maintenance operations")
        print("  schedule  Configure or view maintenance schedule")
        print("")
        print("Run 'valence maintenance <subcommand> --help' for more info")
        return 1


# Keep old name for backward compatibility
def cmd_maintenance(args: argparse.Namespace) -> int:
    """Legacy entry point - redirects to appropriate handler."""
    return cmd_maintenance_compat(args)
