"""Usage attestation CLI commands.

Commands:
  valence attestations list [--resource <id>] [--user <did>] [--success-only] [--json]
  valence attestations add <resource-id> [--success/--failure] [--feedback "text"] [--user <did>]
  valence attestations stats [<resource-id>] [--json]
  valence attestations trust <resource-id> [--json]

Part of Issue #271: Social — Usage attestations.
"""

from __future__ import annotations

import argparse
import json
from uuid import UUID


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register attestations commands with the CLI parser."""
    att_parser = subparsers.add_parser(
        "attestations",
        help="Manage usage attestations for shared resources",
    )
    att_subparsers = att_parser.add_subparsers(dest="attestations_command", required=True)

    # attestations list
    list_parser = att_subparsers.add_parser("list", help="List usage attestations")
    list_parser.add_argument("--resource", "-r", help="Filter by resource UUID")
    list_parser.add_argument("--user", "-u", help="Filter by user DID")
    list_parser.add_argument("--success-only", action="store_true", help="Show only successful attestations")
    list_parser.add_argument("--failure-only", action="store_true", help="Show only failed attestations")
    list_parser.add_argument("--limit", "-n", type=int, default=50, help="Max results (default: 50)")
    list_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # attestations add
    add_parser = att_subparsers.add_parser("add", help="Add a usage attestation")
    add_parser.add_argument("resource_id", help="Resource UUID")
    add_parser.add_argument("--user", "-u", default="did:vkb:local", help="User DID (default: did:vkb:local)")
    add_parser.add_argument("--failure", action="store_true", help="Mark as failed usage (default: success)")
    add_parser.add_argument("--feedback", "-f", help="Optional feedback text")
    add_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # attestations stats
    stats_parser = att_subparsers.add_parser("stats", help="Show attestation statistics")
    stats_parser.add_argument("resource_id", nargs="?", help="Resource UUID (omit for all resources)")
    stats_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # attestations trust
    trust_parser = att_subparsers.add_parser("trust", help="Compute trust signal from attestations")
    trust_parser.add_argument("resource_id", help="Resource UUID")
    trust_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")


def cmd_attestations(args: argparse.Namespace) -> int:
    """Dispatch attestations subcommands."""
    subcmd = getattr(args, "attestations_command", None)
    if subcmd == "list":
        return cmd_attestations_list(args)
    elif subcmd == "add":
        return cmd_attestations_add(args)
    elif subcmd == "stats":
        return cmd_attestations_stats(args)
    elif subcmd == "trust":
        return cmd_attestations_trust(args)
    else:
        print("Usage: valence attestations {list|add|stats|trust}")
        return 1


def _build_service():
    """Build the attestation service with its dependencies."""
    from ...core.attestation_service import AttestationService
    from ...core.resource_sharing import ResourceSharingService

    sharing = ResourceSharingService()
    return AttestationService(sharing)


def cmd_attestations_list(args: argparse.Namespace) -> int:
    """List usage attestations."""
    from ...core.attestation_service import AttestationFilter

    service = _build_service()

    # Build filter
    resource_id = None
    if hasattr(args, "resource") and args.resource:
        try:
            resource_id = UUID(args.resource)
        except ValueError:
            print(f"❌ Invalid UUID: {args.resource}")
            return 1

    success_filter = None
    if getattr(args, "success_only", False):
        success_filter = True
    elif getattr(args, "failure_only", False):
        success_filter = False

    filt = AttestationFilter(
        resource_id=resource_id,
        user_did=getattr(args, "user", None),
        success=success_filter,
        limit=getattr(args, "limit", 50),
    )

    attestations = service.get_attestations(filt)

    if getattr(args, "json", False):
        print(json.dumps([a.to_dict() for a in attestations], indent=2))
        return 0

    if not attestations:
        print("📭 No attestations found")
        return 0

    print(f"📋 Attestations ({len(attestations)})")
    print("─" * 70)
    for a in attestations:
        icon = "✅" if a.success else "❌"
        feedback_str = f'  "{a.feedback}"' if a.feedback else ""
        print(f"  {icon} {a.id}")
        print(f"     resource={a.resource_id}  user={a.user_did}")
        print(f"     {a.created_at.isoformat()}{feedback_str}")
    print()
    return 0


def cmd_attestations_add(args: argparse.Namespace) -> int:
    """Add a usage attestation."""
    try:
        resource_id = UUID(args.resource_id)
    except ValueError:
        print(f"❌ Invalid UUID: {args.resource_id}")
        return 1

    service = _build_service()
    success = not getattr(args, "failure", False)
    user_did = getattr(args, "user", "did:vkb:local")
    feedback = getattr(args, "feedback", None)

    try:
        attestation = service.add_attestation(
            resource_id=resource_id,
            user_did=user_did,
            success=success,
            feedback=feedback,
        )
    except Exception as e:
        print(f"❌ {e}")
        return 1

    if getattr(args, "json", False):
        print(json.dumps(attestation.to_dict(), indent=2))
        return 0

    icon = "✅" if attestation.success else "❌"
    print(f"{icon} Attestation recorded: {attestation.id}")
    print(f"   Resource: {attestation.resource_id}")
    print(f"   User: {attestation.user_did}")
    print(f"   Success: {attestation.success}")
    if attestation.feedback:
        print(f"   Feedback: {attestation.feedback}")
    return 0


def cmd_attestations_stats(args: argparse.Namespace) -> int:
    """Show attestation statistics."""
    service = _build_service()

    resource_id_str = getattr(args, "resource_id", None)

    if resource_id_str:
        # Single resource stats
        try:
            resource_id = UUID(resource_id_str)
        except ValueError:
            print(f"❌ Invalid UUID: {resource_id_str}")
            return 1

        stats = service.get_stats(resource_id)

        if getattr(args, "json", False):
            print(json.dumps(stats.to_dict(), indent=2))
            return 0

        if stats.total == 0:
            print(f"📭 No attestations for resource {resource_id}")
            return 0

        print(f"📊 Attestation Stats: {resource_id}")
        print("─" * 50)
        print(f"  Total:        {stats.total}")
        print(f"  Successes:    {stats.successes}")
        print(f"  Failures:     {stats.failures}")
        rate_str = f"{stats.success_rate:.0%}" if stats.success_rate is not None else "N/A"
        print(f"  Success rate: {rate_str}")
        print(f"  Unique users: {stats.unique_users}")
        if stats.latest_at:
            print(f"  Latest:       {stats.latest_at.isoformat()}")
        print()
    else:
        # All resources
        all_stats = service.get_all_stats()

        if getattr(args, "json", False):
            print(json.dumps([s.to_dict() for s in all_stats], indent=2))
            return 0

        if not all_stats:
            print("📭 No attestations recorded yet")
            return 0

        print(f"📊 Attestation Stats ({len(all_stats)} resources)")
        print("─" * 70)
        for s in all_stats:
            rate_str = f"{s.success_rate:.0%}" if s.success_rate is not None else "N/A"
            print(f"  {s.resource_id}  total={s.total}  rate={rate_str}  users={s.unique_users}")
        print()

    return 0


def cmd_attestations_trust(args: argparse.Namespace) -> int:
    """Compute trust signal from attestation patterns."""
    try:
        resource_id = UUID(args.resource_id)
    except ValueError:
        print(f"❌ Invalid UUID: {args.resource_id}")
        return 1

    service = _build_service()
    signal = service.compute_trust_signal(resource_id)

    if getattr(args, "json", False):
        print(json.dumps(signal.to_dict(), indent=2))
        return 0

    if signal.overall == 0.0:
        print(f"📭 No attestations for resource {resource_id} — no trust signal available")
        return 0

    print(f"🔐 Trust Signal: {resource_id}")
    print("─" * 50)
    print(f"  Success rate:    {signal.success_rate:.1%}")
    print(f"  Diversity score: {signal.diversity_score:.3f}")
    print(f"  Volume score:    {signal.volume_score:.3f}")
    print("  ─────────────────────")
    print(f"  Overall:         {signal.overall:.3f}")
    print()
    return 0
