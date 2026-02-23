"""System status command — show article count, source count, embeddings, etc.

Commands:
    valence status              Show system status
"""

from __future__ import annotations

import argparse

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the status command."""
    status_p = subparsers.add_parser(
        "status",
        help="Show system status (article count, source count, embeddings, DB info)",
    )
    status_p.set_defaults(func=cmd_status)


def cmd_status(args: argparse.Namespace) -> int:
    """Show comprehensive system status."""
    client = get_client()

    try:
        # Try to get status from dedicated endpoint
        result = client.get("/status")
        output_result(result)
        return 0
    except ValenceAPIError as e:
        if e.status_code == 404:
            # Fallback: aggregate from stats endpoint
            return _status_from_stats(client)
        output_error(e.message)
        return 1
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1


def _status_from_stats(client) -> int:
    """Build status response from /stats endpoint."""
    try:
        stats = client.get("/stats")

        # Extract counts from stats
        status = {
            "articles": {
                "total": stats.get("articles", {}).get("total", 0),
                "active": stats.get("articles", {}).get("active", 0),
            },
            "sources": {
                "total": stats.get("sources", {}).get("total", 0),
            },
            "embeddings": {
                "coverage": f"{stats.get('embeddings', {}).get('coverage_pct', 0):.1f}%",
                "total": stats.get("embeddings", {}).get("total", 0),
            },
            "database": {
                "connected": True,
            },
        }

        output_result(status)
        return 0
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
