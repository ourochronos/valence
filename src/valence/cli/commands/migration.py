# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Migration commands for Valence CLI.

Provides:
  valence migrate up [--dry-run]
  valence migrate down [--dry-run]
  valence migrate status
  valence migrate create NAME  (local-only, scaffolds migration file)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import get_cli_config
from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register migrate commands on the CLI parser."""
    migrate_parser = subparsers.add_parser("migrate", help="Database migration management")
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_command", required=True)

    # migrate up
    migrate_up = migrate_subparsers.add_parser("up", help="Apply pending migrations")
    migrate_up.add_argument("--dry-run", action="store_true", help="Show what would be applied")

    # migrate down
    migrate_down = migrate_subparsers.add_parser("down", help="Rollback the last migration")
    migrate_down.add_argument("--dry-run", action="store_true", help="Show what would be rolled back")

    # migrate status
    migrate_subparsers.add_parser("status", help="Show migration status")

    # migrate create (local-only — scaffolds a file, no server call)
    migrate_create = migrate_subparsers.add_parser("create", help="Scaffold a new migration")
    migrate_create.add_argument("name", help="Migration name (e.g. add_users_table)")

    migrate_parser.set_defaults(func=cmd_migrate)


def cmd_migrate(args: argparse.Namespace) -> int:
    """Dispatch migrate subcommands."""
    subcmd = getattr(args, "migrate_command", None)
    if subcmd is None:
        print("Usage: valence migrate {up|down|status|create}", file=sys.stderr)
        return 1

    handlers = {
        "up": cmd_migrate_up,
        "down": cmd_migrate_down,
        "status": cmd_migrate_status,
        "create": cmd_migrate_create,
    }
    handler = handlers.get(subcmd)
    if handler is None:
        print(f"Unknown migrate command: {subcmd}", file=sys.stderr)
        return 1
    return handler(args)


def cmd_migrate_up(args: argparse.Namespace) -> int:
    """Apply pending migrations via REST API."""
    dry_run = getattr(args, "dry_run", False)
    client = get_client()
    try:
        result = client.post("/admin/migrate/up", body={"dry_run": dry_run})
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_migrate_down(args: argparse.Namespace) -> int:
    """Rollback the last migration via REST API."""
    dry_run = getattr(args, "dry_run", False)
    client = get_client()
    try:
        result = client.post("/admin/migrate/down", body={"dry_run": dry_run})
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_migrate_status(args: argparse.Namespace) -> int:
    """Show migration status via REST API."""
    config = get_cli_config()
    params: dict = {"output": config.output}

    client = get_client()
    try:
        result = client.get("/admin/migrate/status", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_migrate_create(args: argparse.Namespace) -> int:
    """Create a new migration scaffold (local-only, no server call)."""
    from ...core.migrations import MigrationRunner

    name = args.name
    migrations_dir = _get_migrations_dir()

    try:
        path = MigrationRunner.create_migration(migrations_dir, name)
        print(f"Created migration: {path.name}")
        print(f"   Edit: {path}")
        return 0
    except Exception as e:
        output_error(f"Failed to create migration: {e}")
        return 1


def _get_migrations_dir() -> Path:
    """Resolve the migrations directory (repo root / migrations)."""
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent.parent.parent
    return repo_root / "migrations"
