# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Top-level ingest command — shorthand for sources ingest.

Commands:
    valence ingest <file_or_url>           Ingest a source
    valence ingest <content>               Ingest content from argument
    valence ingest -                       Ingest from stdin
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the top-level ingest command."""
    ingest_p = subparsers.add_parser(
        "ingest",
        help="Ingest a source (file, URL, or content)",
    )
    ingest_p.add_argument(
        "content",
        help='Content, file path, URL, or "-" for stdin',
    )
    ingest_p.add_argument(
        "--title",
        "-t",
        help="Optional human-readable title",
    )
    ingest_p.add_argument(
        "--type",
        dest="source_type",
        default="document",
        choices=["document", "conversation", "web", "code", "observation", "tool_output", "user_input"],
        help="Source type (default: document)",
    )
    ingest_p.add_argument(
        "--metadata",
        help='Optional JSON metadata, e.g. \'{"author":"name"}\'',
    )
    ingest_p.add_argument(
        "--supersedes",
        help="Optional UUID of a source this one supersedes/replaces",
    )
    ingest_p.set_defaults(func=cmd_ingest)


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest content from file, URL, stdin, or direct content."""
    import json

    content = args.content

    # Handle stdin
    if content == "-":
        content = sys.stdin.read()
    # Handle file paths
    elif Path(content).exists():
        content = Path(content).read_text()
    # Handle URLs (basic check)
    elif content.startswith(("http://", "https://")):
        # Could enhance to fetch URL content, for now just store URL as content
        pass
    # Otherwise treat as direct content
    else:
        pass

    client = get_client()
    body: dict = {
        "content": content,
        "source_type": args.source_type,
    }

    if args.title:
        body["title"] = args.title

    # Handle metadata JSON
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
            body["metadata"] = metadata
        except json.JSONDecodeError as e:
            output_error(f"Invalid JSON in --metadata: {e}")
            return 1

    # Add URL field if it's a URL
    if args.content.startswith(("http://", "https://")):
        body["url"] = args.content

    # Add supersession link if provided
    if args.supersedes:
        body["supersedes"] = args.supersedes

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
