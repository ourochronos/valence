"""Migration commands for Valence CLI.

Provides:
  valence migrate up [--to VERSION] [--dry-run]
  valence migrate down [--to VERSION] [--dry-run]
  valence migrate status
  valence migrate create NAME
  valence migrate bootstrap [--dry-run]
  valence migrate-visibility  (legacy)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ..utils import get_db_connection

logger = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register migrate and migrate-visibility commands on the CLI parser."""
    # migrate
    migrate_parser = subparsers.add_parser("migrate", help="Database migration management")
    migrate_subparsers = migrate_parser.add_subparsers(dest="migrate_command", required=True)

    # migrate up
    migrate_up = migrate_subparsers.add_parser("up", help="Apply pending migrations")
    migrate_up.add_argument("--to", help="Apply up to this version (inclusive)")
    migrate_up.add_argument("--dry-run", action="store_true", help="Show what would be applied")

    # migrate down
    migrate_down = migrate_subparsers.add_parser("down", help="Rollback migrations")
    migrate_down.add_argument("--to", help="Rollback to this version (exclusive — it stays applied)")
    migrate_down.add_argument("--dry-run", action="store_true", help="Show what would be rolled back")

    # migrate status
    migrate_subparsers.add_parser("status", help="Show migration status")

    # migrate create
    migrate_create = migrate_subparsers.add_parser("create", help="Scaffold a new migration")
    migrate_create.add_argument("name", help="Migration name (e.g. add_users_table)")

    # migrate bootstrap
    migrate_bootstrap = migrate_subparsers.add_parser("bootstrap", help="Bootstrap fresh database")
    migrate_bootstrap.add_argument("--dry-run", action="store_true", help="Show what would be applied")

    migrate_parser.set_defaults(func=cmd_migrate)

    # migrate-visibility (legacy)
    mv_parser = subparsers.add_parser(
        "migrate-visibility",
        help="Migrate existing beliefs from visibility to SharePolicy",
    )
    mv_parser.set_defaults(func=cmd_migrate_visibility)


def _get_migrations_dir() -> Path:
    """Resolve the migrations directory (repo root / migrations)."""
    # Walk up from this file to find the repo root
    # This file: src/valence/cli/commands/migration.py
    # Repo root: ../../../../
    here = Path(__file__).resolve()
    repo_root = here.parent.parent.parent.parent.parent
    migrations_dir = repo_root / "migrations"
    return migrations_dir


def _make_runner():
    """Create a MigrationRunner with CLI-appropriate connection factory."""
    from ...core.migrations import MigrationRunner

    return MigrationRunner(
        migrations_dir=_get_migrations_dir(),
        connection_factory=get_db_connection,
    )


def cmd_migrate(args: argparse.Namespace) -> int:
    """Dispatch migrate subcommands."""
    subcmd = getattr(args, "migrate_command", None)
    if subcmd is None:
        print("Usage: valence migrate {up|down|status|create|bootstrap}", file=sys.stderr)
        return 1

    handlers = {
        "up": cmd_migrate_up,
        "down": cmd_migrate_down,
        "status": cmd_migrate_status,
        "create": cmd_migrate_create,
        "bootstrap": cmd_migrate_bootstrap,
    }
    handler = handlers.get(subcmd)
    if handler is None:
        print(f"Unknown migrate command: {subcmd}", file=sys.stderr)
        return 1
    return handler(args)


def cmd_migrate_up(args: argparse.Namespace) -> int:
    """Apply pending migrations."""
    runner = _make_runner()
    target = getattr(args, "to", None)
    dry_run = getattr(args, "dry_run", False)

    try:
        applied = runner.up(target=target, dry_run=dry_run)
        if not applied:
            print("✅ No pending migrations.")
        else:
            prefix = "[DRY RUN] " if dry_run else ""
            print(f"\n{prefix}Applied {len(applied)} migration(s):")
            for v in applied:
                print(f"  ✓ {v}")
        return 0
    except Exception as e:
        print(f"❌ Migration failed: {e}", file=sys.stderr)
        return 1


def cmd_migrate_down(args: argparse.Namespace) -> int:
    """Rollback migrations."""
    runner = _make_runner()
    target = getattr(args, "to", None)
    dry_run = getattr(args, "dry_run", False)

    try:
        rolled_back = runner.down(target=target, dry_run=dry_run)
        if not rolled_back:
            print("✅ Nothing to rollback.")
        else:
            prefix = "[DRY RUN] " if dry_run else ""
            print(f"\n{prefix}Rolled back {len(rolled_back)} migration(s):")
            for v in rolled_back:
                print(f"  ↩ {v}")
        return 0
    except Exception as e:
        print(f"❌ Rollback failed: {e}", file=sys.stderr)
        return 1


def cmd_migrate_status(args: argparse.Namespace) -> int:
    """Show migration status."""
    runner = _make_runner()

    try:
        statuses = runner.status()
        if not statuses:
            print("No migrations found.")
            return 0

        print(f"\n{'Version':<10} {'Description':<30} {'State':<20} {'Applied At'}")
        print("─" * 85)
        for s in statuses:
            applied_str = s.applied_at.strftime("%Y-%m-%d %H:%M:%S") if s.applied_at else ""
            state_icon = {"applied": "✓", "pending": "•", "checksum_mismatch": "⚠"}
            icon = state_icon.get(s.state, "?")
            print(f"  {icon} {s.version:<8} {s.description:<30} {s.state:<20} {applied_str}")
        print()
        return 0
    except Exception as e:
        print(f"❌ Failed to get status: {e}", file=sys.stderr)
        return 1


def cmd_migrate_create(args: argparse.Namespace) -> int:
    """Create a new migration scaffold."""
    from ...core.migrations import MigrationRunner

    name = args.name
    migrations_dir = _get_migrations_dir()

    try:
        path = MigrationRunner.create_migration(migrations_dir, name)
        print(f"✅ Created migration: {path.name}")
        print(f"   Edit: {path}")
        return 0
    except Exception as e:
        print(f"❌ Failed to create migration: {e}", file=sys.stderr)
        return 1


def cmd_migrate_bootstrap(args: argparse.Namespace) -> int:
    """Bootstrap a fresh database with all migrations."""
    runner = _make_runner()
    dry_run = getattr(args, "dry_run", False)

    try:
        applied = runner.bootstrap(dry_run=dry_run)
        if not applied:
            print("✅ No migrations to apply.")
        else:
            prefix = "[DRY RUN] " if dry_run else ""
            print(f"\n{prefix}Bootstrapped with {len(applied)} migration(s):")
            for v in applied:
                print(f"  ✓ {v}")
        return 0
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Bootstrap failed: {e}", file=sys.stderr)
        return 1


# --------------------------------------------------------------------------
# Legacy command (kept for backward compat)
# --------------------------------------------------------------------------


def cmd_migrate_visibility(args: argparse.Namespace) -> int:
    """Migrate existing beliefs from old visibility to SharePolicy."""
    from oro_privacy.migration import migrate_all_beliefs_sync

    print("🔄 Migrating visibility to SharePolicy...")

    try:
        conn = get_db_connection()

        cur = conn.cursor()
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.columns
                WHERE table_name = 'beliefs' AND column_name = 'share_policy'
            )
        """
        )
        has_column = cur.fetchone()["exists"]

        if not has_column:
            print("⚠️  share_policy column not found. Adding it...")
            cur.execute("ALTER TABLE beliefs ADD COLUMN IF NOT EXISTS share_policy JSONB")
            conn.commit()
            print("✅ share_policy column added")

        cur.close()

        result = migrate_all_beliefs_sync(conn)

        print("\n📊 Migration Results:")
        print(f"   Total beliefs:     {result['total']}")
        print(f"   Needed migration:  {result['needed_migration']}")
        print(f"   Migrated:          {result['migrated']}")

        if result["needed_migration"] == 0:
            print("\n✅ All beliefs already have share_policy set")
        else:
            print(f"\n✅ Successfully migrated {result['needed_migration']} beliefs")

        conn.close()
        return 0

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
