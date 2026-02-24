# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Articles commands — search, get, create, and manage knowledge articles (C2, C3).

Commands:
    valence articles search <query>         Search articles (knowledge_search)
    valence articles get <article_id>       Get article by ID
    valence articles create <content>       Create a new article
    valence articles list                   List recent articles
"""

from __future__ import annotations

import argparse

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the articles sub-command group."""
    articles_parser = subparsers.add_parser("articles", help="Manage knowledge articles")
    articles_sub = articles_parser.add_subparsers(dest="articles_command", required=True)

    # --- search ---
    search_p = articles_sub.add_parser("search", help="Search articles by query")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--limit", "-n", type=int, default=10, help="Max results (default 10)")
    search_p.add_argument("--domain", "-d", action="append", dest="domain_filter", help="Domain path filter (repeatable)")
    search_p.set_defaults(func=cmd_articles_search)

    # --- get ---
    get_p = articles_sub.add_parser("get", help="Get an article by ID")
    get_p.add_argument("article_id", help="UUID of the article")
    get_p.add_argument(
        "--provenance",
        "-p",
        action="store_true",
        help="Include provenance sources",
    )
    get_p.set_defaults(func=cmd_articles_get)

    # --- create ---
    create_p = articles_sub.add_parser("create", help="Create a new article")
    create_p.add_argument("content", help="Article content")
    create_p.add_argument("--title", "-t", help="Optional article title")
    create_p.add_argument("--domain", "-d", action="append", dest="domain_path", help="Domain path tag (repeatable)")
    create_p.add_argument(
        "--author-type",
        choices=["system", "operator", "agent"],
        default="agent",
        help="Author type (default: agent)",
    )
    create_p.set_defaults(func=cmd_articles_create)

    # --- list ---
    list_p = articles_sub.add_parser("list", help="List recent articles (search with empty-ish query)")
    list_p.add_argument("--limit", "-n", type=int, default=10, help="Max results (default 10)")
    list_p.set_defaults(func=cmd_articles_list)

    # --- split ---
    split_p = articles_sub.add_parser("split", help="Split an article into two topic-aware parts")
    split_p.add_argument("article_id", help="UUID of the article to split")
    split_p.set_defaults(func=cmd_articles_split)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_articles_search(args: argparse.Namespace) -> int:
    """Search articles via REST API."""
    client = get_client()
    body: dict = {
        "query": args.query,
        "limit": args.limit,
    }
    if getattr(args, "domain_filter", None):
        body["domain_filter"] = args.domain_filter

    try:
        result = client.post("/articles/search", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_articles_get(args: argparse.Namespace) -> int:
    """Get an article by ID."""
    client = get_client()
    params: dict = {}
    if getattr(args, "provenance", False):
        params["include_provenance"] = "true"

    try:
        result = client.get(f"/articles/{args.article_id}", params=params or None)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_articles_create(args: argparse.Namespace) -> int:
    """Create a new article."""
    client = get_client()
    body: dict = {
        "content": args.content,
        "author_type": args.author_type,
    }
    if getattr(args, "title", None):
        body["title"] = args.title
    if getattr(args, "domain_path", None):
        body["domain_path"] = args.domain_path

    try:
        result = client.post("/articles", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_articles_list(args: argparse.Namespace) -> int:
    """List recent articles (via search endpoint with a broad query)."""
    client = get_client()
    body: dict = {
        "query": "*",
        "limit": args.limit,
    }

    try:
        result = client.post("/articles/search", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_articles_split(args: argparse.Namespace) -> int:
    """Split an article into two topic-aware parts."""
    client = get_client()

    try:
        result = client.post(f"/articles/{args.article_id}/split", body={})
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
