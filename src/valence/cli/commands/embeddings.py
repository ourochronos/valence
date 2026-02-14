"""Embeddings management commands: backfill, migrate, status."""

from __future__ import annotations

import argparse
import logging
import sys

from ..utils import get_db_connection

logger = logging.getLogger(__name__)

VALID_CONTENT_TYPES = ("belief", "exchange", "pattern")

# Known embedding models and their dimensions
KNOWN_MODELS = {
    "BAAI/bge-small-en-v1.5": {"provider": "local", "dims": 384, "type_id": "local_bge_small"},
    "text-embedding-3-small": {"provider": "openai", "dims": 1536, "type_id": "openai_text3_small"},
    "text-embedding-3-large": {"provider": "openai", "dims": 3072, "type_id": "openai_text3_large"},
    "text-embedding-ada-002": {"provider": "openai", "dims": 1536, "type_id": "openai_ada_002"},
}


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register embeddings commands on the CLI parser."""
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

    # embeddings migrate
    migrate_parser = embeddings_subparsers.add_parser(
        "migrate",
        help="Migrate to a different embedding model (changes VECTOR dimensions, clears old embeddings)",
    )
    migrate_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help=f"Embedding model name. Known models: {', '.join(KNOWN_MODELS.keys())}. Or use custom with --dims.",
    )
    migrate_parser.add_argument(
        "--dims",
        "-d",
        type=int,
        default=None,
        help="Vector dimensions (auto-detected for known models)",
    )
    migrate_parser.add_argument(
        "--provider",
        "-p",
        default=None,
        help="Provider name (auto-detected for known models, e.g. 'local', 'openai')",
    )
    migrate_parser.add_argument(
        "--type-id",
        default=None,
        help="Embedding type ID for the DB (auto-generated if not specified)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without making changes",
    )

    # embeddings status
    embeddings_subparsers.add_parser("status", help="Show current embedding configuration and coverage")

    embeddings_parser.set_defaults(func=cmd_embeddings)


def _count_missing_embeddings(cur, content_type: str) -> int:
    """Count records missing embeddings for a given content type."""
    if content_type == "belief":
        cur.execute("SELECT COUNT(*) as count FROM beliefs WHERE embedding IS NULL AND status = 'active'")
    elif content_type == "exchange":
        cur.execute("SELECT COUNT(*) as count FROM vkb_exchanges WHERE embedding IS NULL")
    elif content_type == "pattern":
        cur.execute("SELECT COUNT(*) as count FROM vkb_patterns WHERE embedding IS NULL")
    else:
        return 0
    return cur.fetchone()["count"]


def _count_all_embeddings(cur, content_type: str) -> int:
    """Count all records (for --force mode)."""
    if content_type == "belief":
        cur.execute("SELECT COUNT(*) as count FROM beliefs WHERE status = 'active'")
    elif content_type == "exchange":
        cur.execute("SELECT COUNT(*) as count FROM vkb_exchanges")
    elif content_type == "pattern":
        cur.execute("SELECT COUNT(*) as count FROM vkb_patterns")
    else:
        return 0
    return cur.fetchone()["count"]


def cmd_embeddings(args: argparse.Namespace) -> int:
    """Route embeddings subcommands."""
    if args.embeddings_command == "backfill":
        return cmd_embeddings_backfill(args)
    elif args.embeddings_command == "migrate":
        return cmd_embeddings_migrate(args)
    elif args.embeddings_command == "status":
        return cmd_embeddings_status(args)

    print("❌ Unknown embeddings command. Use 'valence embeddings {backfill,migrate,status}'.", file=sys.stderr)
    return 1


def cmd_embeddings_backfill(args: argparse.Namespace) -> int:
    """Backfill missing or outdated embeddings.

    Wraps the backfill_embeddings() service function with CLI niceties:
    progress output, dry-run mode, content type filtering, and force re-embed.
    """
    from our_embeddings.service import backfill_embeddings, embed_content

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
            "SELECT id, content FROM vkb_exchanges LIMIT %s",
            (batch_size,),
        )
    elif content_type == "pattern":
        cur.execute(
            "SELECT id, description as content FROM vkb_patterns LIMIT %s",
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


def cmd_embeddings_migrate(args: argparse.Namespace) -> int:
    """Migrate embedding dimensions to a different model.

    Changes VECTOR column dimensions, clears old embeddings, and updates
    the embedding_types table. Run `valence embeddings backfill --force`
    after migration to re-embed all content.
    """
    model: str = args.model
    dims: int | None = args.dims
    provider: str | None = args.provider
    type_id: str | None = args.type_id
    dry_run: bool = args.dry_run

    # Resolve model info
    if model in KNOWN_MODELS:
        info = KNOWN_MODELS[model]
        dims = dims or info["dims"]
        provider = provider or info["provider"]
        type_id = type_id or info["type_id"]
    else:
        if dims is None:
            print(f"❌ Unknown model '{model}'. Specify --dims for custom models.", file=sys.stderr)
            return 1
        provider = provider or "custom"
        type_id = type_id or model.replace("/", "_").replace("-", "_").lower()

    print("🔄 Embedding Migration")
    print("─" * 40)
    print(f"  Model:      {model}")
    print(f"  Provider:   {provider}")
    print(f"  Dimensions: {dims}")
    print(f"  Type ID:    {type_id}")
    print()

    if dry_run:
        print("🔍 Dry run — showing what would happen:")
        print("  1. Register new embedding type in embedding_types table")
        print("  2. Drop existing embedding indexes")
        print(f"  3. ALTER VECTOR columns to VECTOR({dims})")
        print("  4. NULL out all existing embeddings (incompatible dimensions)")
        print("  5. Clear old embedding coverage records")
        print("  6. Recreate embedding indexes")
        print()
        print("  After migration, run: valence embeddings backfill --force")
        return 0

    conn = None
    try:
        conn = get_db_connection()
        conn.autocommit = False
        cur = conn.cursor()

        # Check current state
        cur.execute("SELECT id, dimensions, is_default FROM embedding_types WHERE is_default = TRUE")
        current = cur.fetchone()
        if current:
            if current["dimensions"] == dims and current["id"] == type_id:
                print(f"✅ Already configured for {model} ({dims} dims). No migration needed.")
                return 0
            print(f"  Current: {current['id']} ({current['dimensions']} dims)")

        # Execute migration via the helper function
        cur.execute(
            "SELECT * FROM migrate_embedding_dimensions(%s, %s, %s, %s)",
            (dims, type_id, provider, model),
        )
        steps = cur.fetchall()
        for step_row in steps:
            print(f"  ✅ {step_row['step']}: {step_row['detail']}")

        conn.commit()
        print()
        print(f"🎉 Migration to {model} ({dims} dims) complete.")
        print("   Run: valence embeddings backfill --force")
        return 0

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"❌ Migration failed: {e}", file=sys.stderr)
        logger.exception("Embedding migration error")
        return 1
    finally:
        if conn:
            conn.close()


def cmd_embeddings_status(args: argparse.Namespace) -> int:
    """Show current embedding configuration and coverage."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get embedding types
        cur.execute("SELECT id, provider, model, dimensions, is_default, status FROM embedding_types ORDER BY is_default DESC, id")
        types = cur.fetchall()

        print("🧠 Embedding Status")
        print("─" * 50)

        if not types:
            print("  No embedding types configured.")
        else:
            print("  Configured Models:")
            for t in types:
                default_marker = " ★" if t["is_default"] else ""
                print(f"    {t['id']}: {t['provider']}/{t['model']} ({t['dimensions']} dims) [{t['status']}]{default_marker}")

        # Get column dimensions from information_schema
        cur.execute("""
            SELECT c.table_name, c.column_name, c.udt_name,
                   CASE WHEN c.udt_name = 'vector' THEN
                       (SELECT atttypmod FROM pg_attribute a JOIN pg_class cl ON a.attrelid = cl.oid
                        WHERE cl.relname = c.table_name AND a.attname = c.column_name AND a.atttypmod > 0)
                   END as dims
            FROM information_schema.columns c
            WHERE c.udt_name = 'vector'
              AND c.table_schema = 'public'
            ORDER BY c.table_name
        """)
        vector_cols = cur.fetchall()

        if vector_cols:
            print()
            print("  VECTOR Columns:")
            for vc in vector_cols:
                dims_str = f"VECTOR({vc['dims']})" if vc["dims"] else "VECTOR(?)"
                print(f"    {vc['table_name']}.{vc['column_name']}: {dims_str}")

        # Get coverage stats
        cur.execute("""
            SELECT ec.embedding_type_id, ec.content_type, COUNT(*) as count
            FROM embedding_coverage ec
            GROUP BY ec.embedding_type_id, ec.content_type
            ORDER BY ec.embedding_type_id, ec.content_type
        """)
        coverage = cur.fetchall()

        # Get totals
        cur.execute("SELECT COUNT(*) as count FROM beliefs WHERE status = 'active'")
        total_beliefs = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) as count FROM beliefs WHERE embedding IS NOT NULL AND status = 'active'")
        embedded_beliefs = cur.fetchone()["count"]

        print()
        print("  Coverage:")
        print(f"    Beliefs: {embedded_beliefs}/{total_beliefs} embedded")

        if coverage:
            print()
            print("  Coverage by Type:")
            for c in coverage:
                print(f"    {c['embedding_type_id']}/{c['content_type']}: {c['count']}")

        print()
        return 0

    except Exception as e:
        print(f"❌ Status check failed: {e}", file=sys.stderr)
        logger.exception("Embedding status error")
        return 1
    finally:
        if conn:
            conn.close()
