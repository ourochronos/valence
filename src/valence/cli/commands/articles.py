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
    search_p.add_argument(
        "--epistemic-type",
        choices=["episodic", "semantic", "procedural"],
        help="Filter results by epistemic type",
    )
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
    create_p.add_argument(
        "--epistemic-type",
        choices=["episodic", "semantic", "procedural"],
        default="semantic",
        help="Knowledge type: episodic (decays), semantic (persists), procedural (pinned)",
    )
    create_p.set_defaults(func=cmd_articles_create)

    # --- update ---
    update_p = articles_sub.add_parser("update", help="Update an article's content")
    update_p.add_argument("article_id", help="UUID of the article to update")
    update_p.add_argument("--content", "-c", required=True, help="New content for the article")
    update_p.add_argument("--source-id", help="Optional source ID that prompted this update")
    update_p.add_argument(
        "--epistemic-type",
        choices=["episodic", "semantic", "procedural"],
        help="Optional new epistemic type classification",
    )
    update_p.set_defaults(func=cmd_articles_update)

    # --- merge ---
    merge_p = articles_sub.add_parser("merge", help="Merge two related articles into one")
    merge_p.add_argument("article_id_1", help="UUID of the first article")
    merge_p.add_argument("article_id_2", help="UUID of the second article")
    merge_p.set_defaults(func=cmd_articles_merge)

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
    if getattr(args, "epistemic_type", None):
        body["epistemic_type"] = args.epistemic_type

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
    if getattr(args, "epistemic_type", None):
        body["epistemic_type"] = args.epistemic_type

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


def cmd_articles_update(args: argparse.Namespace) -> int:
    """Update an article's content."""
    client = get_client()
    body: dict = {
        "content": args.content,
    }
    if getattr(args, "source_id", None):
        body["source_id"] = args.source_id
    if getattr(args, "epistemic_type", None):
        body["epistemic_type"] = args.epistemic_type

    try:
        result = client.put(f"/articles/{args.article_id}", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_articles_merge(args: argparse.Namespace) -> int:
    """Merge two related articles into one."""
    client = get_client()
    body: dict = {
        "article_id_a": args.article_id_1,
        "article_id_b": args.article_id_2,
    }

    try:
        result = client.post("/articles/merge", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
