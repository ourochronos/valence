# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Sources commands — list, get, ingest, search sources (C1).

Commands:
    valence sources list                        List recent sources
    valence sources get <source_id>             Get source by ID
    valence sources ingest <content>            Ingest a new source
    valence sources search <query>              Full-text search over sources
"""

from __future__ import annotations

import argparse

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the sources sub-command group."""
    sources_parser = subparsers.add_parser("sources", help="Manage knowledge sources")
    sources_sub = sources_parser.add_subparsers(dest="sources_command", required=True)

    # --- list ---
    list_p = sources_sub.add_parser("list", help="List recent sources")
    list_p.add_argument("--type", dest="source_type", help="Filter by source type")
    list_p.add_argument("--limit", "-n", type=int, default=20, help="Max results (default 20)")
    list_p.add_argument("--offset", type=int, default=0, help="Pagination offset")
    list_p.set_defaults(func=cmd_sources_list)

    # --- get ---
    get_p = sources_sub.add_parser("get", help="Get a source by ID")
    get_p.add_argument("source_id", help="UUID of the source")
    get_p.set_defaults(func=cmd_sources_get)

    # --- ingest ---
    ingest_p = sources_sub.add_parser("ingest", help="Ingest a new source")
    ingest_p.add_argument("content", help="Source text content")
    ingest_p.add_argument(
        "--type",
        "-t",
        dest="source_type",
        default="document",
        choices=["document", "conversation", "web", "code", "observation", "tool_output", "user_input"],
        help="Source type (default: document)",
    )
    ingest_p.add_argument("--title", help="Optional human-readable title")
    ingest_p.add_argument("--url", help="Optional canonical URL")
    ingest_p.set_defaults(func=cmd_sources_ingest)

    # --- search ---
    search_p = sources_sub.add_parser("search", help="Full-text search over sources")
    search_p.add_argument("query", help="Search terms")
    search_p.add_argument("--limit", "-n", type=int, default=20, help="Max results (default 20)")
    search_p.set_defaults(func=cmd_sources_search)

    # --- delete ---
    delete_p = sources_sub.add_parser("delete", help="Permanently remove a source")
    delete_p.add_argument("source_id", help="UUID of the source to delete")
    delete_p.set_defaults(func=cmd_sources_delete)

    # --- index ---
    index_p = sources_sub.add_parser("index", help="Build tree index for a source")
    index_p.add_argument("source_id", help="UUID of the source to index")
    index_p.add_argument("--force", action="store_true", help="Rebuild existing index")
    index_p.set_defaults(func=cmd_sources_index)

    # --- tree ---
    tree_p = sources_sub.add_parser("tree", help="Get tree index for a source")
    tree_p.add_argument("source_id", help="UUID of the source")
    tree_p.set_defaults(func=cmd_sources_tree)

    # --- region ---
    region_p = sources_sub.add_parser("region", help="Get text region from a source by char offsets")
    region_p.add_argument("source_id", help="UUID of the source")
    region_p.add_argument("--start", type=int, required=True, help="Start character offset")
    region_p.add_argument("--end", type=int, required=True, help="End character offset")
    region_p.set_defaults(func=cmd_sources_region)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_sources_list(args: argparse.Namespace) -> int:
    """List recent sources."""
    client = get_client()
    params: dict = {
        "limit": args.limit,
        "offset": args.offset,
    }
    if getattr(args, "source_type", None):
        params["source_type"] = args.source_type

    try:
        result = client.get("/sources", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sources_get(args: argparse.Namespace) -> int:
    """Get a source by ID."""
    client = get_client()
    try:
        result = client.get(f"/sources/{args.source_id}")
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sources_ingest(args: argparse.Namespace) -> int:
    """Ingest a new source."""
    client = get_client()
    body: dict = {
        "content": args.content,
        "source_type": args.source_type,
    }
    if getattr(args, "title", None):
        body["title"] = args.title
    if getattr(args, "url", None):
        body["url"] = args.url

    try:
        result = client.post("/sources", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sources_search(args: argparse.Namespace) -> int:
    """Full-text search over sources."""
    client = get_client()
    body: dict = {
        "query": args.query,
        "limit": args.limit,
    }

    try:
        result = client.post("/sources/search", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sources_delete(args: argparse.Namespace) -> int:
    """Permanently remove a source."""
    client = get_client()

    try:
        result = client.delete(f"/sources/{args.source_id}")
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_sources_index(args: argparse.Namespace) -> int:
    """Build tree index for a source."""
    client = get_client()
    try:
        body = {"force": args.force}
        result = client.post(f"/sources/{args.source_id}/index", json=body)
        output_result(result)
        data = result.get("data", {})
        if data:
            print(f"\n  Nodes: {data.get('node_count', '?')}")
            print(f"  Method: {data.get('method', '?')}")
            print(f"  Source tokens: ~{data.get('token_estimate', '?')}")
            issues = data.get("issues", [])
            if issues:
                print(f"  Issues: {len(issues)}")
                for issue in issues:
                    print(f"    ⚠️  {issue}")
        return 0
    except ValenceConnectionError as exc:
        output_error(f"Connection error: {exc}")
        return 1
    except ValenceAPIError as exc:
        output_error(str(exc))
        return 1


def cmd_sources_tree(args: argparse.Namespace) -> int:
    """Get tree index for a source."""

    client = get_client()
    try:
        result = client.get(f"/sources/{args.source_id}/tree")
        data = result.get("data", {})
        if data:
            # Pretty-print the tree structure
            def print_tree(nodes, indent=0):
                for node in nodes:
                    prefix = "  " * indent
                    title = node.get("title", "<untitled>")
                    start = node.get("start_char", "?")
                    end = node.get("end_char", "?")
                    summary = node.get("summary", "")
                    children = node.get("children", [])
                    marker = "├──" if children else "└──"
                    print(f"{prefix}{marker} [{start}:{end}] {title}")
                    if summary:
                        print(f"{prefix}    {summary}")
                    if children:
                        print_tree(children, indent + 1)

            print_tree(data.get("nodes", []))
        else:
            output_result(result)
        return 0
    except ValenceConnectionError as exc:
        output_error(f"Connection error: {exc}")
        return 1
    except ValenceAPIError as exc:
        output_error(str(exc))
        return 1


def cmd_sources_region(args: argparse.Namespace) -> int:
    """Get text region from a source by char offsets."""
    client = get_client()
    try:
        result = client.get(
            f"/sources/{args.source_id}/region",
            params={"start": args.start, "end": args.end},
        )
        data = result.get("data", {})
        if data:
            print(data.get("text", ""))
            print(f"\n--- [{data.get('start_char')}:{data.get('end_char')}] ~{data.get('token_estimate', '?')} tokens ---")
        else:
            output_result(result)
        return 0
    except ValenceConnectionError as exc:
        output_error(f"Connection error: {exc}")
        return 1
    except ValenceAPIError as exc:
        output_error(str(exc))
        return 1
