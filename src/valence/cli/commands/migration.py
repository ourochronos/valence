"""Migration commands."""

from __future__ import annotations

import argparse

from ..utils import get_db_connection


def cmd_migrate_visibility(args: argparse.Namespace) -> int:
    """Migrate existing beliefs from old visibility to SharePolicy."""
    from ...privacy.migration import migrate_all_beliefs_sync

    print("🔄 Migrating visibility to SharePolicy...")

    try:
        conn = get_db_connection()

        # Check if share_policy column exists
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
            cur.execute(
                """
                ALTER TABLE beliefs
                ADD COLUMN IF NOT EXISTS share_policy JSONB
            """
            )
            conn.commit()
            print("✅ share_policy column added")

        cur.close()

        # Run migration
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
