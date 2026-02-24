# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Conflict detection command."""

from __future__ import annotations

import argparse

from ..config import get_cli_config
from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the conflicts command on the CLI parser."""
    conflicts_parser = subparsers.add_parser("conflicts", help="Detect contradicting beliefs")
    conflicts_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.85,
        help="Similarity threshold for conflict detection",
    )
    conflicts_parser.add_argument(
        "--auto-record",
        "-r",
        action="store_true",
        help="Automatically record detected conflicts as tensions",
    )
    conflicts_parser.set_defaults(func=cmd_conflicts)


def cmd_conflicts(args: argparse.Namespace) -> int:
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
