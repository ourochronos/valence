# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Stats command."""

from __future__ import annotations

import argparse

from ..config import get_cli_config
from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the stats command on the CLI parser."""
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)


def cmd_stats(args: argparse.Namespace) -> int:
    """Show database statistics via REST API."""
    config = get_cli_config()
    params: dict = {"output": config.output}

    client = get_client()
    try:
        result = client.get("/stats", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
