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
