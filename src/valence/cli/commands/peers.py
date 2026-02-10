"""Peer management commands."""

from __future__ import annotations

import argparse

from ..utils import format_age


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the peer command on the CLI parser."""
    peer_parser = subparsers.add_parser("peer", help="Manage trusted peers")
    peer_subparsers = peer_parser.add_subparsers(dest="peer_command", required=True)

    # peer add
    peer_add_parser = peer_subparsers.add_parser("add", help="Add or update a trusted peer")
    peer_add_parser.add_argument("did", help="Peer DID (e.g., did:vkb:web:alice.example.com)")
    peer_add_parser.add_argument(
        "--trust",
        type=float,
        required=True,
        help="Trust level 0.0-1.0 (e.g., 0.8 for 80%% trust)",
    )
    peer_add_parser.add_argument("--name", help="Human-readable name for this peer")
    peer_add_parser.add_argument("--notes", help="Notes about this peer")

    # peer list
    peer_subparsers.add_parser("list", help="List trusted peers")

    # peer remove
    peer_remove_parser = peer_subparsers.add_parser("remove", help="Remove a peer")
    peer_remove_parser.add_argument("did", help="Peer DID to remove")

    peer_parser.set_defaults(func=cmd_peer)


def cmd_peer_add(args: argparse.Namespace) -> int:
    """Add a trusted peer to the local registry."""
    from oro_federation.peer_sync import get_trust_registry

    try:
        registry = get_trust_registry()
        peer = registry.add_peer(
            did=args.did,
            trust_level=args.trust,
            name=args.name,
            notes=args.notes,
        )

        print("✅ Peer added/updated")
        print(f"   DID:   {peer.did}")
        print(f"   Trust: {peer.trust_level:.0%}")
        if peer.name:
            print(f"   Name:  {peer.name}")
        return 0

    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to add peer: {e}")
        return 1


def cmd_peer_list(args: argparse.Namespace) -> int:
    """List trusted peers."""
    from oro_federation.peer_sync import get_trust_registry

    try:
        registry = get_trust_registry()
        peers = registry.list_peers()

        if not peers:
            print("📭 No trusted peers")
            print("\n💡 Add a peer with: valence peer add <did> --trust 0.8")
            return 0

        print(f"👥 {len(peers)} trusted peer(s)\n")

        for p in peers:
            trust_bar = "█" * int(p.trust_level * 10) + "░" * (10 - int(p.trust_level * 10))
            name_str = f" ({p.name})" if p.name else ""
            last_sync = format_age(p.last_sync_at) if p.last_sync_at else "never"

            print(f"  {p.did}{name_str}")
            print(f"    Trust: [{trust_bar}] {p.trust_level:.0%}")
            print(f"    Stats: ↓{p.beliefs_received} received, ↑{p.beliefs_sent} sent")
            print(f"    Synced: {last_sync}")
            print()

        return 0

    except Exception as e:
        print(f"❌ Failed to list peers: {e}")
        return 1


def cmd_peer_remove(args: argparse.Namespace) -> int:
    """Remove a peer from the trust registry."""
    from oro_federation.peer_sync import get_trust_registry

    try:
        registry = get_trust_registry()

        if registry.remove_peer(args.did):
            print(f"✅ Removed peer: {args.did}")
            return 0
        else:
            print(f"⚠️  Peer not found: {args.did}")
            return 1

    except Exception as e:
        print(f"❌ Failed to remove peer: {e}")
        return 1


def cmd_peer(args: argparse.Namespace) -> int:
    """Dispatch peer subcommands."""
    if args.peer_command == "add":
        return cmd_peer_add(args)
    elif args.peer_command == "list":
        return cmd_peer_list(args)
    elif args.peer_command == "remove":
        return cmd_peer_remove(args)
    else:
        print(f"Unknown peer command: {args.peer_command}")
        return 1
