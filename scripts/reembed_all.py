#!/usr/bin/env python3
"""
Re-embed all beliefs with local embedding model (bge-small-en-v1.5).

This script populates the embedding_384 columns added by migration 009.
Supports batch processing, progress tracking, resume capability, and dry-run mode.

Usage:
    # Default: process all tables
    python scripts/reembed_all.py

    # Dry run - show what would be done
    python scripts/reembed_all.py --dry-run

    # Process specific table
    python scripts/reembed_all.py --table beliefs

    # Custom batch size
    python scripts/reembed_all.py --batch-size 50

    # Resume from previous progress
    python scripts/reembed_all.py  # Automatically resumes from .reembed_progress.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from oro_embeddings.providers.local import generate_embeddings_batch

# Default database URL (can be overridden via --database-url or env)
DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://valence:valence@localhost:5432/valence"
)

# Tables and their content columns
TABLE_CONFIG = {
    "beliefs": "content",
    "vkb_exchanges": "content",
    "vkb_patterns": "pattern",
}


def parse_database_url(url: str) -> dict[str, Any]:
    """Parse a PostgreSQL URL into connection parameters."""
    from urllib.parse import urlparse, parse_qs

    parsed = urlparse(url)
    params = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") or "valence",
        "user": parsed.username or "valence",
        "password": parsed.password or "valence",
    }
    return params


def get_connection(database_url: str):
    """Create a database connection."""
    params = parse_database_url(database_url)
    return psycopg2.connect(**params)


def count_needs_embedding(cursor, table: str) -> int:
    """Count rows that need embedding."""
    content_col = TABLE_CONFIG[table]
    cursor.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE embedding_384 IS NULL AND {content_col} IS NOT NULL
    """)
    return cursor.fetchone()[0]


def count_total(cursor, table: str) -> int:
    """Count total rows in table."""
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    return cursor.fetchone()[0]


def count_embedded(cursor, table: str) -> int:
    """Count rows with embeddings."""
    cursor.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE embedding_384 IS NOT NULL
    """)
    return cursor.fetchone()[0]


def fetch_batch(cursor, table: str, batch_size: int, offset: int) -> list[dict]:
    """Fetch a batch of rows needing embedding."""
    content_col = TABLE_CONFIG[table]
    cursor.execute(f"""
        SELECT id, {content_col} as content FROM {table}
        WHERE embedding_384 IS NULL AND {content_col} IS NOT NULL
        ORDER BY id
        LIMIT %s OFFSET %s
    """, (batch_size, offset))
    return cursor.fetchall()


def update_embeddings(conn, cursor, table: str, updates: list[tuple]) -> None:
    """Update embeddings in batch using execute_values for efficiency."""
    if not updates:
        return
    
    # Use execute_values for efficient batch updates
    execute_values(
        cursor,
        f"""
        UPDATE {table} AS t SET
            embedding_384 = v.embedding::vector(384)
        FROM (VALUES %s) AS v(id, embedding)
        WHERE t.id = v.id::uuid
        """,
        updates,
        template="(%s, %s)"
    )
    conn.commit()


def load_progress(progress_file: str) -> dict:
    """Load progress from file."""
    if progress_file and Path(progress_file).exists():
        with open(progress_file) as f:
            return json.load(f)
    return {}


def save_progress(progress_file: str, progress: dict) -> None:
    """Save progress to file."""
    if progress_file:
        with open(progress_file, "w") as f:
            json.dump(progress, f)


def format_rate(count: int, elapsed: float) -> str:
    """Format processing rate as items/sec."""
    if elapsed > 0:
        rate = count / elapsed
        return f"{rate:.1f}/sec"
    return "N/A"


def reembed_table(
    conn,
    table: str,
    batch_size: int = 100,
    dry_run: bool = False,
    progress_file: str | None = None,
    verbose: bool = True,
) -> dict:
    """Re-embed all rows in a table.
    
    Returns:
        dict with stats: {"processed": int, "elapsed": float, "rate": float}
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Load progress if resuming
    progress = load_progress(progress_file) if progress_file else {}
    offset = progress.get(table, 0)
    
    needs_embedding = count_needs_embedding(cursor, table)
    total = count_total(cursor, table)
    
    if verbose:
        print(f"\n{table}:")
        print(f"  Total rows: {total}")
        print(f"  Needs embedding: {needs_embedding}")
        if offset > 0:
            print(f"  Resuming from offset: {offset}")
    
    if dry_run:
        if verbose:
            print(f"  [DRY RUN] Would process {needs_embedding} rows")
        return {"processed": 0, "elapsed": 0, "rate": 0}
    
    if needs_embedding == 0:
        if verbose:
            print("  ✓ Already complete")
        return {"processed": 0, "elapsed": 0, "rate": 0}
    
    start_time = time.time()
    processed = 0
    
    try:
        # Import tqdm for progress bar (optional)
        from tqdm import tqdm
        pbar = tqdm(total=needs_embedding, initial=offset, desc=f"  {table}", unit="rows")
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        if verbose:
            print(f"  Processing...")
    
    while True:
        rows = fetch_batch(cursor, table, batch_size, offset)
        if not rows:
            break
        
        # Extract content
        contents = [row["content"] for row in rows]
        
        # Generate embeddings in batch (no tqdm from provider - we have our own)
        embeddings = generate_embeddings_batch(contents, batch_size=32, show_progress=False)
        
        # Prepare updates: (id, embedding_as_string)
        updates = [
            (str(row["id"]), "[" + ",".join(map(str, emb)) + "]")
            for row, emb in zip(rows, embeddings)
        ]
        
        # Update database
        update_embeddings(conn, cursor, table, updates)
        
        batch_count = len(rows)
        offset += batch_count
        processed += batch_count
        
        if use_tqdm:
            pbar.update(batch_count)
        
        # Save progress periodically
        if progress_file:
            progress[table] = offset
            save_progress(progress_file, progress)
    
    if use_tqdm:
        pbar.close()
    
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0
    
    if verbose:
        print(f"  ✓ Complete: {processed} rows in {elapsed:.1f}s ({rate:.1f}/sec)")
    
    cursor.close()
    
    return {"processed": processed, "elapsed": elapsed, "rate": rate}


def verify_embeddings(conn, tables: list[str], verbose: bool = True) -> dict:
    """Verify embedding coverage after processing.
    
    Returns:
        dict mapping table -> {"total": int, "embedded": int, "pct": float}
    """
    cursor = conn.cursor()
    results = {}
    
    if verbose:
        print("\nVerification:")
    
    for table in tables:
        total = count_total(cursor, table)
        embedded = count_embedded(cursor, table)
        pct = (100 * embedded / total) if total > 0 else 0
        
        results[table] = {
            "total": total,
            "embedded": embedded,
            "pct": pct,
        }
        
        if verbose:
            print(f"  {table}: {embedded}/{total} embedded ({pct:.1f}%)")
    
    cursor.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Re-embed all data with local model (bge-small-en-v1.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--database-url",
        default=DEFAULT_DATABASE_URL,
        help="PostgreSQL connection URL (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of rows to process per batch (default: %(default)s)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--progress-file",
        default=".reembed_progress.json",
        help="File to track progress for resume (default: %(default)s)"
    )
    parser.add_argument(
        "--table",
        choices=list(TABLE_CONFIG.keys()),
        help="Only process specific table"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Don't save progress (disable resume)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )
    args = parser.parse_args()
    
    # Determine which tables to process
    tables = [args.table] if args.table else list(TABLE_CONFIG.keys())
    
    # Connect to database
    if not args.quiet:
        print(f"Connecting to database...")
    
    try:
        conn = get_connection(args.database_url)
    except Exception as e:
        print(f"Error: Failed to connect to database: {e}", file=sys.stderr)
        sys.exit(1)
    
    progress_file = None if args.no_progress else args.progress_file
    
    try:
        start = time.time()
        total_processed = 0
        
        for table in tables:
            result = reembed_table(
                conn,
                table,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                progress_file=progress_file,
                verbose=not args.quiet,
            )
            total_processed += result["processed"]
        
        elapsed = time.time() - start
        
        if not args.quiet:
            print(f"\n{'[DRY RUN] ' if args.dry_run else ''}✓ Complete in {elapsed:.1f}s")
            if total_processed > 0:
                print(f"  Total processed: {total_processed} rows ({total_processed/elapsed:.1f}/sec)")
        
        # Verify (unless dry run)
        if not args.dry_run:
            verify_embeddings(conn, tables, verbose=not args.quiet)
        
        # Clean up progress file on success
        if progress_file and not args.dry_run and Path(progress_file).exists():
            Path(progress_file).unlink()
            if not args.quiet:
                print(f"\n  Cleaned up progress file: {progress_file}")
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
