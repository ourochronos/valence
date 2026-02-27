# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""CLI commands for token management (valence auth)."""

from __future__ import annotations

import argparse

from valence.server.cli import cmd_create, cmd_list, cmd_revoke, cmd_verify


def _inject_token_file(args: argparse.Namespace) -> None:
    """Ensure args.token_file is set to the default if not provided."""
    from pathlib import Path

    if not hasattr(args, "token_file") or args.token_file is None:
        args.token_file = Path.home() / ".valence" / "tokens.json"


def _wrap(fn):
    """Wrap a server.cli command to inject token_file default."""

    def wrapper(args: argparse.Namespace) -> int:
        _inject_token_file(args)
        return fn(args)

    return wrapper


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``valence auth`` command group."""
    auth_parser = subparsers.add_parser("auth", help="Manage authentication tokens")
    auth_sub = auth_parser.add_subparsers(dest="auth_command", required=True)

    auth_parser.add_argument(
        "--token-file",
        type=lambda s: __import__("pathlib").Path(s),
        default=None,
        help="Path to token storage file (default: ~/.valence/tokens.json)",
    )

    # create-token
    create_p = auth_sub.add_parser("create-token", help="Create a new authentication token")
    create_p.add_argument(
        "--client-id",
        "-c",
        required=True,
        help="Client identifier (e.g., 'claude-code-laptop')",
    )
    create_p.add_argument(
        "--description",
        "-d",
        default="",
        help="Human-readable description",
    )
    create_p.add_argument(
        "--scopes",
        "-s",
        default="mcp:access",
        help="Comma-separated list of scopes (default: mcp:access)",
    )
    create_p.add_argument(
        "--expires-days",
        "-e",
        type=int,
        default=None,
        help="Token expires after N days (default: never)",
    )
    create_p.set_defaults(func=_wrap(cmd_create))

    # list-tokens
    list_p = auth_sub.add_parser("list-tokens", help="List all tokens")
    list_p.set_defaults(func=_wrap(cmd_list))

    # revoke-token
    revoke_p = auth_sub.add_parser("revoke-token", help="Revoke a token")
    revoke_p.add_argument(
        "--client-id",
        "-c",
        help="Revoke all tokens for this client ID",
    )
    revoke_p.add_argument(
        "--hash",
        help="Token hash to revoke",
    )
    revoke_p.add_argument(
        "--token",
        "-t",
        help="Raw token to revoke",
    )
    revoke_p.set_defaults(func=_wrap(cmd_revoke))

    # verify
    verify_p = auth_sub.add_parser("verify", help="Verify a token is valid")
    verify_p.add_argument("token", help="Token to verify")
    verify_p.set_defaults(func=_wrap(cmd_verify))
