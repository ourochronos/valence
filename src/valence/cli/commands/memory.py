# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Memory CLI commands — import, list, and search agent memories.

Commands:
    valence memory import <path>        Import a markdown file as a memory
    valence memory import-dir <dir>     Import all .md files from a directory
    valence memory list                 List recent memories
    valence memory search <query>       Search memories from CLI
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result

# Constants
SNIPPET_TRUNCATE_LENGTH = 200  # Max content length before truncation


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

    # --- status ---
    status_p = memory_sub.add_parser("status", help="Memory system statistics")
    status_p.set_defaults(func=cmd_memory_status_cli)

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

    # --- store ---
    store_p = memory_sub.add_parser("store", help="Store a memory directly from content string")
    store_p.add_argument("content", nargs="?", default=None, help="Memory content (omit to read from stdin)")
    store_p.add_argument("--stdin", action="store_true", help="Read content from stdin")
    store_p.add_argument(
        "--context",
        help="Where this memory came from (e.g., 'session:main', 'conversation:user')",
    )
    store_p.add_argument(
        "--importance",
        type=float,
        default=0.5,
        help="Importance score (0.0-1.0, default 0.5)",
    )
    store_p.add_argument(
        "--tags",
        action="append",
        dest="tags",
        metavar="TAG",
        help="Tag to apply (repeatable: --tags foo --tags bar)",
    )
    store_p.add_argument(
        "--supersedes-id",
        dest="supersedes_id",
        help="UUID of a previous memory this replaces",
    )
    store_p.set_defaults(func=cmd_memory_store)

    # --- forget ---
    forget_p = memory_sub.add_parser("forget", help="Mark a memory as forgotten (soft delete)")
    forget_p.add_argument("memory_id", help="UUID of the memory to forget")
    forget_p.add_argument(
        "--reason",
        help="Optional reason for forgetting this memory",
    )
    forget_p.set_defaults(func=cmd_memory_forget)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_memory_import(args: argparse.Namespace) -> int:
    """Import a markdown file as a memory source via REST API."""
    client = get_client()

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

    body: dict[str, Any] = {"content": content}
    if args.context:
        body["context"] = args.context
    if args.importance is not None:
        body["importance"] = args.importance
    if args.tags:
        body["tags"] = args.tags

    try:
        result = client.post("/memory", body=body)
    except (ValenceConnectionError, ValenceAPIError) as e:
        output_error(str(e))
        return 1

    output_result(result)
    return 0 if result.get("success", True) else 1


def cmd_memory_import_dir(args: argparse.Namespace) -> int:
    """Import all .md files from a directory as memories via REST API."""
    client = get_client()

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
            file_content = path.read_text()
            body: dict[str, Any] = {
                "content": file_content,
                "context": args.context or f"file:{path.relative_to(directory)}",
            }
            if args.importance is not None:
                body["importance"] = args.importance
            if args.tags:
                body["tags"] = args.tags

            result = client.post("/memory", body=body)

            if result.get("success", True):
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
        except (ValenceConnectionError, ValenceAPIError) as e:
            failed.append(
                {
                    "file": str(path.relative_to(directory)),
                    "error": str(e),
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
            "success": len(failed) == 0,
            "imported": imported,
            "failed": failed,
            "imported_count": len(imported),
            "failed_count": len(failed),
        }
    )

    return 0 if not failed else 1


def cmd_memory_list(args: argparse.Namespace) -> int:
    """List recent memories via REST API."""
    client = get_client()
    params: dict[str, str] = {"limit": str(args.limit)}
    if args.tags:
        params["tags"] = ",".join(args.tags)

    try:
        result = client.get("/memory", params=params)
        output_result(result)
        return 0 if result.get("success", True) else 1
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_memory_search(args: argparse.Namespace) -> int:
    """Search memories via REST API."""
    client = get_client()
    params: dict[str, str] = {"query": args.query, "limit": str(args.limit)}
    if args.min_confidence is not None:
        params["min_confidence"] = str(args.min_confidence)
    if args.tags:
        params["tags"] = ",".join(args.tags)

    try:
        result = client.get("/memory/search", params=params)
        output_result(result)
        return 0 if result.get("success", True) else 1
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_memory_store(args: argparse.Namespace) -> int:
    """Store a memory via REST API."""
    client = get_client()
    content = args.content
    if content is None:
        content = sys.stdin.read()
    body: dict[str, Any] = {"content": content}
    if getattr(args, "context", None):
        body["context"] = args.context
    if getattr(args, "importance", None) is not None:
        body["importance"] = args.importance
    if getattr(args, "tags", None):
        body["tags"] = args.tags
    if getattr(args, "supersedes_id", None):
        body["supersedes_id"] = args.supersedes_id

    try:
        result = client.post("/memory", body=body)
        output_result(result)
        return 0 if result.get("success", True) else 1
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_memory_forget(args: argparse.Namespace) -> int:
    """Mark a memory as forgotten via REST API."""
    client = get_client()
    body: dict[str, Any] = {}
    if getattr(args, "reason", None):
        body["reason"] = args.reason

    try:
        result = client.post(f"/memory/{args.memory_id}/forget", body=body)
        output_result(result)
        return 0 if result.get("success", True) else 1
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_memory_status_cli(args: argparse.Namespace) -> int:
    """Show memory system statistics via REST API."""
    client = get_client()
    try:
        result = client.get("/memory/status")
        output_result(result)
        return 0 if result.get("success", True) else 1
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
