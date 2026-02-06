"""Embeddings management commands: backfill."""

from __future__ import annotations

import argparse
import logging
import sys

from ..utils import get_db_connection

logger = logging.getLogger(__name__)

VALID_CONTENT_TYPES = ("belief", "exchange", "pattern")


def _count_missing_embeddings(cur, content_type: str) -> int:
    """Count records missing embeddings for a given content type."""
    if content_type == "belief":
        cur.execute("SELECT COUNT(*) as count FROM beliefs WHERE embedding IS NULL AND status = 'active'")
    elif content_type == "exchange":
        cur.execute("SELECT COUNT(*) as count FROM exchanges WHERE embedding IS NULL")
    elif content_type == "pattern":
        cur.execute("SELECT COUNT(*) as count FROM patterns WHERE embedding IS NULL")
    else:
        return 0
    return cur.fetchone()["count"]


def _count_all_embeddings(cur, content_type: str) -> int:
    """Count all records (for --force mode)."""
    if content_type == "belief":
        cur.execute("SELECT COUNT(*) as count FROM beliefs WHERE status = 'active'")
    elif content_type == "exchange":
        cur.execute("SELECT COUNT(*) as count FROM exchanges")
    elif content_type == "pattern":
        cur.execute("SELECT COUNT(*) as count FROM patterns")
    else:
        return 0
    return cur.fetchone()["count"]


def cmd_embeddings(args: argparse.Namespace) -> int:
    """Route embeddings subcommands."""
    if args.embeddings_command == "backfill":
        return cmd_embeddings_backfill(args)

    print("❌ Unknown embeddings command. Use 'valence embeddings backfill'.", file=sys.stderr)
    return 1


def cmd_embeddings_backfill(args: argparse.Namespace) -> int:
    """Backfill missing or outdated embeddings.

    Wraps the backfill_embeddings() service function with CLI niceties:
    progress output, dry-run mode, content type filtering, and force re-embed.
    """
    from ...embeddings.service import backfill_embeddings, embed_content

    batch_size: int = args.batch_size
    dry_run: bool = args.dry_run
    force: bool = args.force
    content_type_filter: str | None = args.content_type

    # Determine which content types to process
    if content_type_filter:
        if content_type_filter not in VALID_CONTENT_TYPES:
            print(
                f"❌ Invalid content type '{content_type_filter}'. Must be one of: {', '.join(VALID_CONTENT_TYPES)}",
                file=sys.stderr,
            )
            return 1
        content_types = [content_type_filter]
    else:
        content_types = list(VALID_CONTENT_TYPES)

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Gather counts
        total_to_process = 0
        type_counts: dict[str, int] = {}
        for ct in content_types:
            if force:
                count = _count_all_embeddings(cur, ct)
            else:
                count = _count_missing_embeddings(cur, ct)
            type_counts[ct] = count
            total_to_process += count

        # Show summary
        print("🧠 Embeddings Backfill")
        print("─" * 40)
        for ct in content_types:
            label = "total" if force else "missing"
            print(f"  {ct:>10}: {type_counts[ct]} {label}")
        print(f"  {'total':>10}: {total_to_process}")
        if force:
            print("  ⚡ Force mode: re-embedding all records")
        print(f"  📦 Batch size: {batch_size}")
        print()

        if total_to_process == 0:
            print("✅ Nothing to backfill — all embeddings are up to date.")
            return 0

        if dry_run:
            print("🔍 Dry run — no changes made.")
            return 0

        # Process each content type
        grand_total = 0
        for ct in content_types:
            if type_counts[ct] == 0:
                continue

            if force:
                # Force mode: fetch all records and re-embed them
                count = _backfill_force(cur, conn, ct, batch_size, embed_content)
            else:
                # Normal mode: use the service function
                count = backfill_embeddings(ct, batch_size=batch_size)

            grand_total += count
            print(f"  ✅ {ct}: {count} embedded")

        print()
        print(f"🎉 Backfill complete: {grand_total} embeddings generated.")
        return 0

    except Exception as e:
        print(f"❌ Backfill failed: {e}", file=sys.stderr)
        logger.exception("Backfill error")
        return 1
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def _backfill_force(cur, conn, content_type: str, batch_size: int, embed_fn) -> int:
    """Force re-embed all records of a given content type."""
    from ...core.exceptions import DatabaseException, EmbeddingException

    if content_type == "belief":
        cur.execute(
            "SELECT id, content FROM beliefs WHERE status = 'active' LIMIT %s",
            (batch_size,),
        )
    elif content_type == "exchange":
        cur.execute(
            "SELECT id, content FROM exchanges LIMIT %s",
            (batch_size,),
        )
    elif content_type == "pattern":
        cur.execute(
            "SELECT id, description as content FROM patterns LIMIT %s",
            (batch_size,),
        )
    else:
        return 0

    rows = cur.fetchall()
    count = 0

    for row in rows:
        try:
            embed_fn(content_type, str(row["id"]), row["content"])
            count += 1
        except (EmbeddingException, DatabaseException) as e:
            logger.error(f"Failed to embed {content_type} {row['id']}: {e}")

    return count
