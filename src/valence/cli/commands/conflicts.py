# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Conflict detection and resolution commands."""

from __future__ import annotations

import argparse

from ..config import get_cli_config
from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the conflicts command group on the CLI parser."""
    conflicts_parser = subparsers.add_parser("conflicts", help="Manage conflicts and contentions")
    conflicts_sub = conflicts_parser.add_subparsers(dest="conflicts_command", required=True)

    # --- list (default) ---
    list_p = conflicts_sub.add_parser("list", help="Detect contradicting beliefs")
    list_p.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.85,
        help="Similarity threshold for conflict detection",
    )
    list_p.add_argument(
        "--auto-record",
        "-r",
        action="store_true",
        help="Automatically record detected conflicts as tensions",
    )
    list_p.set_defaults(func=cmd_conflicts_list)

    # --- resolve ---
    resolve_p = conflicts_sub.add_parser("resolve", help="Resolve a contention")
    resolve_p.add_argument("contention_id", help="UUID of the contention to resolve")
    resolve_p.add_argument(
        "--resolution",
        required=True,
        help="Resolution action (e.g., 'accept_source', 'keep_article', 'merge')",
    )
    resolve_p.add_argument(
        "--rationale",
        required=True,
        help="Explanation for the resolution",
    )
    resolve_p.set_defaults(func=cmd_conflicts_resolve)


def cmd_conflicts_list(args: argparse.Namespace) -> int:
    """Detect beliefs that may contradict each other via REST API."""
    config = get_cli_config()
    params: dict = {"output": config.output}
    if args.threshold:
        params["threshold"] = str(args.threshold)
    if args.auto_record:
        params["auto_record"] = "true"

    client = get_client()
    try:
        result = client.get("/beliefs/conflicts", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_conflicts_resolve(args: argparse.Namespace) -> int:
    """Resolve a contention."""
    client = get_client()
    body: dict = {
        "resolution": args.resolution,
        "rationale": args.rationale,
    }

    try:
        result = client.post(f"/contentions/{args.contention_id}/resolve", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


# Legacy single-command entry point (for backward compatibility)
def cmd_conflicts(args: argparse.Namespace) -> int:
    """Legacy entry point - redirect to list."""
    return cmd_conflicts_list(args)
