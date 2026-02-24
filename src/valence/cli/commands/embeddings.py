# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Embeddings management commands: backfill, migrate, status."""

from __future__ import annotations

import argparse
import sys

from ..config import get_cli_config
from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register embeddings commands on the CLI parser."""
    embeddings_parser = subparsers.add_parser("embeddings", help="Embedding management")
    embeddings_subparsers = embeddings_parser.add_subparsers(dest="embeddings_command", required=True)

    # embeddings backfill
    backfill_parser = embeddings_subparsers.add_parser("backfill", help="Backfill missing embeddings")
    backfill_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=50,
        help="Number of records to process per batch (default: 50)",
    )
    backfill_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backfilled without making changes",
    )

    # embeddings migrate
    migrate_parser = embeddings_subparsers.add_parser(
        "migrate",
        help="Migrate to a different embedding model",
    )
    migrate_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Embedding model name",
    )
    migrate_parser.add_argument(
        "--dims",
        "-d",
        type=int,
        default=None,
        help="Vector dimensions (auto-detected for known models)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without making changes",
    )

    # embeddings status
    embeddings_subparsers.add_parser("status", help="Show embedding configuration and coverage")

    embeddings_parser.set_defaults(func=cmd_embeddings)


def cmd_embeddings(args: argparse.Namespace) -> int:
    """Route embeddings subcommands."""
    if args.embeddings_command == "backfill":
        return cmd_embeddings_backfill(args)
    elif args.embeddings_command == "migrate":
        return cmd_embeddings_migrate(args)
    elif args.embeddings_command == "status":
        return cmd_embeddings_status(args)

    print("Unknown embeddings command. Use 'valence embeddings {backfill,migrate,status}'.", file=sys.stderr)
    return 1


def cmd_embeddings_backfill(args: argparse.Namespace) -> int:
    """Backfill missing embeddings via REST API."""
    body: dict = {
        "batch_size": args.batch_size,
        "dry_run": args.dry_run,
    }

    client = get_client()
    try:
        result = client.post("/admin/embeddings/backfill", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_embeddings_migrate(args: argparse.Namespace) -> int:
    """Migrate embedding model/dimensions via REST API."""
    body: dict = {"dry_run": args.dry_run}
    if args.model:
        body["model"] = args.model
    if args.dims:
        body["dims"] = args.dims

    client = get_client()
    try:
        result = client.post("/admin/embeddings/migrate", body=body)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1


def cmd_embeddings_status(args: argparse.Namespace) -> int:
    """Show embedding status via REST API."""
    config = get_cli_config()
    params: dict = {"output": config.output}

    client = get_client()
    try:
        result = client.get("/admin/embeddings/status", params=params)
        output_result(result)
        return 0
    except ValenceConnectionError as e:
        output_error(str(e))
        return 1
    except ValenceAPIError as e:
        output_error(e.message)
        return 1
