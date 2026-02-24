# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Provenance commands — trace claims and view source provenance (C5).

Commands:
    valence provenance get <article_id>                  Get provenance for an article
    valence provenance trace <article_id> <claim_text>   Trace a claim to sources
    valence provenance link <article_id> <source_id>     Link a source to an article
"""

from __future__ import annotations

import argparse

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the provenance sub-command group."""
    prov_parser = subparsers.add_parser("provenance", help="Manage source provenance for articles")
    prov_sub = prov_parser.add_subparsers(dest="provenance_command", required=True)

    # --- get ---
    get_p = prov_sub.add_parser("get", help="Get the full provenance list for an article")
    get_p.add_argument("article_id", help="UUID of the article")
    get_p.set_defaults(func=cmd_provenance_get)

    # --- trace ---
    trace_p = prov_sub.add_parser("trace", help="Trace a specific claim back to contributing sources")
    trace_p.add_argument("article_id", help="UUID of the article")
    trace_p.add_argument("claim_text", help="The claim text to trace")
    trace_p.set_defaults(func=cmd_provenance_trace)

    # --- link ---
    link_p = prov_sub.add_parser("link", help="Link a source to an article with a relationship")
    link_p.add_argument("article_id", help="UUID of the article")
    link_p.add_argument("source_id", help="UUID of the source")
    link_p.add_argument(
        "--relationship",
        "-r",
        choices=["originates", "confirms", "supersedes", "contradicts", "contends"],
        default="confirms",
        help="Relationship type (default: confirms)",
    )
    link_p.add_argument("--notes", help="Optional notes about the relationship")
    link_p.set_defaults(func=cmd_provenance_link)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_provenance_get(args: argparse.Namespace) -> int:
    """Get provenance list for an article."""
    client = get_client()
    try:
        result = client.get(f"/articles/{args.article_id}/provenance")
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_provenance_trace(args: argparse.Namespace) -> int:
    """Trace a claim to its contributing sources."""
    client = get_client()
    body: dict = {
        "claim_text": args.claim_text,
    }

    try:
        result = client.post(f"/articles/{args.article_id}/provenance/trace", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_provenance_link(args: argparse.Namespace) -> int:
    """Link a source to an article."""
    client = get_client()
    body: dict = {
        "source_id": args.source_id,
        "relationship": args.relationship,
    }
    if getattr(args, "notes", None):
        body["notes"] = args.notes

    try:
        result = client.post(f"/articles/{args.article_id}/provenance/link", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
