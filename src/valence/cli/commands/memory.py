"""Memory CLI commands — import, list, and search agent memories.

Commands:
    valence memory import <path>        Import a markdown file as a memory
    valence memory import-dir <dir>     Import all .md files from a directory
    valence memory list                 List recent memories
    valence memory search <query>       Search memories from CLI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from valence.core.db import get_cursor

from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the memory sub-command group."""
    memory_parser = subparsers.add_parser(
        "memory",
        help="Import and manage agent memories",
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command", required=True)

    # --- import ---
    import_p = memory_sub.add_parser(
        "import",
        help="Import a markdown file as a memory source",
    )
    import_p.add_argument("path", type=Path, help="Path to markdown file")
    import_p.add_argument(
        "--context",
        help="Memory context (e.g., 'migration', 'session:main')",
    )
    import_p.add_argument(
        "--importance",
        type=float,
        default=0.5,
        help="Importance score (0.0-1.0, default 0.5)",
    )
    import_p.add_argument(
        "--tags",
        nargs="+",
        help="Optional tags",
    )
    import_p.set_defaults(func=cmd_memory_import)

    # --- import-dir ---
    import_dir_p = memory_sub.add_parser(
        "import-dir",
        help="Import all .md files from a directory as memories",
    )
    import_dir_p.add_argument("directory", type=Path, help="Directory containing .md files")
    import_dir_p.add_argument(
        "--context",
        help="Memory context for all files",
    )
    import_dir_p.add_argument(
        "--importance",
        type=float,
        default=0.5,
        help="Default importance (0.0-1.0, default 0.5)",
    )
    import_dir_p.add_argument(
        "--tags",
        nargs="+",
        help="Default tags for all files",
    )
    import_dir_p.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively import from subdirectories",
    )
    import_dir_p.set_defaults(func=cmd_memory_import_dir)

    # --- list ---
    list_p = memory_sub.add_parser("list", help="List recent memories")
    list_p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=20,
        help="Maximum results (default 20)",
    )
    list_p.add_argument(
        "--tags",
        nargs="+",
        help="Filter by tags",
    )
    list_p.set_defaults(func=cmd_memory_list)

    # --- search ---
    search_p = memory_sub.add_parser("search", help="Search memories")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Maximum results (default 10)",
    )
    search_p.add_argument(
        "--tags",
        nargs="+",
        help="Filter by tags",
    )
    search_p.add_argument(
        "--min-confidence",
        type=float,
        help="Minimum confidence threshold (0.0-1.0)",
    )
    search_p.set_defaults(func=cmd_memory_search)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_memory_import(args: argparse.Namespace) -> int:
    """Import a markdown file as a memory source."""
    from valence.mcp.handlers.memory import memory_store

    path: Path = args.path
    if not path.exists():
        output_error(f"File not found: {path}")
        return 1

    if not path.is_file():
        output_error(f"Not a file: {path}")
        return 1

    try:
        content = path.read_text()
    except Exception as e:
        output_error(f"Failed to read {path}: {e}")
        return 1

    result = memory_store(
        content=content,
        context=args.context,
        importance=args.importance,
        tags=args.tags,
    )

    output_result(result)
    return 0 if result.get("success") else 1


def cmd_memory_import_dir(args: argparse.Namespace) -> int:
    """Import all .md files from a directory as memories."""
    from valence.mcp.handlers.memory import memory_store

    directory: Path = args.directory
    if not directory.exists():
        output_error(f"Directory not found: {directory}")
        return 1

    if not directory.is_dir():
        output_error(f"Not a directory: {directory}")
        return 1

    # Find all .md files
    if args.recursive:
        md_files = list(directory.rglob("*.md"))
    else:
        md_files = list(directory.glob("*.md"))

    if not md_files:
        output_error(f"No .md files found in {directory}")
        return 1

    imported = []
    failed = []

    for path in md_files:
        try:
            content = path.read_text()

            result = memory_store(
                content=content,
                context=args.context or f"file:{path.relative_to(directory)}",
                importance=args.importance,
                tags=args.tags,
            )

            if result.get("success"):
                imported.append(
                    {
                        "file": str(path.relative_to(directory)),
                        "memory_id": result.get("memory_id"),
                        "title": result.get("title"),
                    }
                )
            else:
                failed.append(
                    {
                        "file": str(path.relative_to(directory)),
                        "error": result.get("error"),
                    }
                )
        except Exception as e:
            failed.append(
                {
                    "file": str(path.relative_to(directory)),
                    "error": str(e),
                }
            )

    output_result(
        {
            "success": True,
            "imported": imported,
            "failed": failed,
            "total_files": len(md_files),
            "imported_count": len(imported),
            "failed_count": len(failed),
        }
    )

    return 0 if not failed else 1


def cmd_memory_list(args: argparse.Namespace) -> int:
    """List recent memories."""
    limit = max(1, min(args.limit, 200))
    tags = args.tags

    with get_cursor() as cur:
        if tags:
            # Filter by tags using JSONB containment
            cur.execute(
                """
                SELECT s.id, s.title, s.content, s.metadata, s.created_at
                FROM sources s
                WHERE s.type = 'observation'
                  AND s.metadata->>'memory' = 'true'
                  AND s.metadata->'tags' ?| %s
                ORDER BY s.created_at DESC
                LIMIT %s
                """,
                (tags, limit),
            )
        else:
            cur.execute(
                """
                SELECT s.id, s.title, s.content, s.metadata, s.created_at
                FROM sources s
                WHERE s.type = 'observation'
                  AND s.metadata->>'memory' = 'true'
                ORDER BY s.created_at DESC
                LIMIT %s
                """,
                (limit,),
            )

        rows = cur.fetchall()

    memories = []
    for row in rows:
        metadata = row.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        # Truncate content for display
        content = row.get("content", "")
        if len(content) > 200:
            content = content[:197] + "..."

        memories.append(
            {
                "memory_id": str(row["id"]),
                "title": row.get("title"),
                "content_preview": content,
                "importance": metadata.get("importance", 0.5),
                "context": metadata.get("context"),
                "tags": metadata.get("tags", []),
                "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
            }
        )

    output_result(
        {
            "success": True,
            "memories": memories,
            "count": len(memories),
        }
    )

    return 0


def cmd_memory_search(args: argparse.Namespace) -> int:
    """Search memories from CLI."""
    from valence.mcp.handlers.memory import memory_recall

    result = memory_recall(
        query=args.query,
        limit=args.limit,
        min_confidence=args.min_confidence,
        tags=args.tags,
    )

    output_result(result)
    return 0 if result.get("success") else 1
