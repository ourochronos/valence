"""Compilation command — compile sources into articles.

Commands:
    valence compile [source_ids...]        Compile specific sources into article
    valence compile --auto                 Auto-group and compile all unlinked sources
"""

from __future__ import annotations

import argparse

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the compile command."""
    compile_p = subparsers.add_parser(
        "compile",
        help="Compile sources into articles",
    )
    compile_p.add_argument(
        "source_ids",
        nargs="*",
        help="Source IDs to compile (optional if --auto)",
    )
    compile_p.add_argument(
        "--title",
        help="Title hint for compilation",
    )
    compile_p.add_argument(
        "--auto",
        action="store_true",
        help="Auto-group and compile all unlinked sources",
    )
    compile_p.set_defaults(func=cmd_compile)


def cmd_compile(args: argparse.Namespace) -> int:
    """Compile sources into an article."""
    client = get_client()

    if args.auto:
        # Auto-compilation mode: find unlinked sources and compile them
        try:
            # Use a dedicated endpoint if it exists, or implement logic here
            result = client.post("/articles/compile/auto", body={})
            output_result(result)
            return 0
        except ValenceAPIError as e:
            # If endpoint doesn't exist yet, provide helpful error
            if e.status_code == 404:
                output_error(
                    "Auto-compilation endpoint not yet implemented. "
                    "Specify source IDs manually or wait for server update."
                )
            else:
                output_error(e.message)
            return 1
        except ValenceConnectionError as e:
            output_error(str(e))
            return 1

    # Manual compilation mode
    if not args.source_ids:
        output_error("Must provide source_ids or use --auto")
        return 1

    body: dict = {
        "source_ids": args.source_ids,
    }

    if args.title:
        body["title_hint"] = args.title

    try:
        result = client.post("/articles/compile", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
