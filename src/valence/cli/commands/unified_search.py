# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Unified search command — search across articles and sources.

Commands:
    valence search <query>              Search articles and sources
    valence search <query> --articles-only  Search only articles
    valence search <query> --sources-only   Search only sources
"""

from __future__ import annotations

import argparse

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the unified search command."""
    search_p = subparsers.add_parser(
        "search",
        help="Search knowledge (articles and sources)",
    )
    search_p.add_argument("query", help="Search query")
    search_p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=5,
        help="Max results per type (default 5)",
    )
    search_p.add_argument(
        "--articles-only",
        action="store_true",
        help="Search only articles",
    )
    search_p.add_argument(
        "--sources-only",
        action="store_true",
        help="Search only sources",
    )
    search_p.set_defaults(func=cmd_search)


def cmd_search(args: argparse.Namespace) -> int:
    """Unified search handler."""
    client = get_client()

    if args.articles_only and args.sources_only:
        output_error("Cannot specify both --articles-only and --sources-only")
        return 1

    try:
        results = {}

        # Search articles unless sources-only
        if not args.sources_only:
            article_body = {"query": args.query, "limit": args.limit}
            articles_result = client.post("/articles/search", body=article_body)
            results["articles"] = articles_result.get("results", [])

        # Search sources unless articles-only
        if not args.articles_only:
            source_body = {"query": args.query, "limit": args.limit}
            sources_result = client.post("/sources/search", body=source_body)
            results["sources"] = sources_result.get("results", [])

        # Output combined results
        output_result(results)
        return 0

    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
