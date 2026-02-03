"""Command-line interface for Valence server management."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from .auth import TokenStore, hash_token


def cmd_create(args: argparse.Namespace) -> int:
    """Create a new token."""
    store = TokenStore(args.token_file)

    # Parse expiration
    expires_at = None
    if args.expires_days:
        expires_at = time.time() + (args.expires_days * 24 * 60 * 60)

    # Parse scopes
    scopes = args.scopes.split(",") if args.scopes else ["mcp:access"]

    # Create token
    raw_token = store.create(
        client_id=args.client_id,
        description=args.description,
        scopes=scopes,
        expires_at=expires_at,
    )

    print(f"Token created for client '{args.client_id}'")
    print()
    print("=" * 60)
    print("IMPORTANT: Save this token now. It will not be shown again.")
    print("=" * 60)
    print()
    print(f"Token: {raw_token}")
    print()
    print("To use with Claude Code:")
    print(
        f'  claude mcp add --transport http valence https://your-domain/mcp --header "Authorization: Bearer {raw_token}"'
    )
    print()

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List all tokens."""
    store = TokenStore(args.token_file)
    tokens = store.list_tokens()

    if not tokens:
        print("No tokens found.")
        return 0

    print(f"{'Client ID':<20} {'Description':<30} {'Created':<20} {'Expires':<20}")
    print("-" * 90)

    for token in tokens:
        created = datetime.fromtimestamp(token.created_at).strftime("%Y-%m-%d %H:%M")
        if token.expires_at:
            expires = datetime.fromtimestamp(token.expires_at).strftime("%Y-%m-%d %H:%M")
            if token.is_expired():
                expires += " (EXPIRED)"
        else:
            expires = "Never"

        desc = token.description[:27] + "..." if len(token.description) > 30 else token.description
        print(f"{token.client_id:<20} {desc:<30} {created:<20} {expires:<20}")

    print()
    print(f"Total: {len(tokens)} token(s)")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    """Revoke a token."""
    store = TokenStore(args.token_file)

    # If client ID provided, find and revoke all tokens for that client
    if args.client_id:
        tokens = store.get_by_client_id(args.client_id)
        if not tokens:
            print(f"No tokens found for client '{args.client_id}'")
            return 1

        for token in tokens:
            store.revoke(token.token_hash)
            print(f"Revoked token for client '{token.client_id}'")
        return 0

    # If hash provided, revoke directly
    if args.hash:
        if store.revoke(args.hash):
            print(f"Token revoked.")
            return 0
        else:
            print(f"Token not found: {args.hash}")
            return 1

    # If raw token provided, hash it first
    if args.token:
        token_hash = hash_token(args.token)
        if store.revoke(token_hash):
            print(f"Token revoked.")
            return 0
        else:
            print(f"Token not found or invalid.")
            return 1

    print("Must provide --client-id, --hash, or --token")
    return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a token is valid."""
    store = TokenStore(args.token_file)
    token = store.verify(args.token)

    if token:
        print(f"Token is VALID")
        print(f"  Client ID: {token.client_id}")
        print(f"  Scopes: {', '.join(token.scopes)}")
        print(f"  Created: {datetime.fromtimestamp(token.created_at)}")
        if token.expires_at:
            print(f"  Expires: {datetime.fromtimestamp(token.expires_at)}")
        else:
            print(f"  Expires: Never")
        return 0
    else:
        print("Token is INVALID")
        return 1


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Valence token management CLI",
        prog="valence-token",
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=Path("/opt/valence/config/tokens.json"),
        help="Path to token storage file",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new token")
    create_parser.add_argument(
        "--client-id",
        "-c",
        required=True,
        help="Client identifier (e.g., 'claude-code-laptop')",
    )
    create_parser.add_argument(
        "--description",
        "-d",
        default="",
        help="Human-readable description",
    )
    create_parser.add_argument(
        "--scopes",
        "-s",
        default="mcp:access",
        help="Comma-separated list of scopes (default: mcp:access)",
    )
    create_parser.add_argument(
        "--expires-days",
        "-e",
        type=int,
        default=None,
        help="Token expires after N days (default: never)",
    )

    # List command
    subparsers.add_parser("list", help="List all tokens")

    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke a token")
    revoke_parser.add_argument(
        "--client-id",
        "-c",
        help="Revoke all tokens for this client ID",
    )
    revoke_parser.add_argument(
        "--hash",
        help="Token hash to revoke",
    )
    revoke_parser.add_argument(
        "--token",
        "-t",
        help="Raw token to revoke",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify a token")
    verify_parser.add_argument("token", help="Token to verify")

    args = parser.parse_args()

    if args.command == "create":
        return cmd_create(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "revoke":
        return cmd_revoke(args)
    elif args.command == "verify":
        return cmd_verify(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
