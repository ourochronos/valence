"""Network discovery command."""

from __future__ import annotations

import argparse
import sys


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the discover command on the CLI parser."""
    discover_parser = subparsers.add_parser("discover", help="Discover network routers via seeds")
    discover_parser.add_argument(
        "--seed",
        "-s",
        action="append",
        dest="seeds",
        help="Custom seed URL (repeatable)",
    )
    discover_parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=5,
        help="Number of routers to request (default: 5)",
    )
    discover_parser.add_argument("--region", "-r", help="Preferred region")
    discover_parser.add_argument(
        "--feature",
        "-f",
        action="append",
        dest="features",
        help="Required feature (repeatable)",
    )
    discover_parser.add_argument("--refresh", action="store_true", help="Force refresh (bypass cache)")
    discover_parser.add_argument("--no-verify", action="store_true", help="Skip router signature verification")
    discover_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    discover_parser.add_argument("--stats", action="store_true", help="Show discovery statistics")
    discover_parser.set_defaults(func=cmd_discover)


def cmd_discover(args: argparse.Namespace) -> int:
    """Discover network routers via seed nodes."""
    import asyncio

    async def run_discovery():
        from oro_network.discovery import (
            DiscoveryClient,
            DiscoveryError,
            NoSeedsAvailableError,
        )

        # Create client with custom settings
        client = DiscoveryClient(
            verify_signatures=not args.no_verify,
        )

        # Add custom seeds
        if args.seeds:
            for seed in args.seeds:
                client.add_seed(seed)

        # Build preferences
        preferences = {}
        if args.region:
            preferences["region"] = args.region
        if args.features:
            preferences["features"] = args.features

        try:
            routers = await client.discover_routers(
                count=args.count,
                preferences=preferences if preferences else None,
                force_refresh=args.refresh,
            )
        except NoSeedsAvailableError as e:
            print(f"❌ No seeds available: {e}", file=sys.stderr)
            return 1
        except DiscoveryError as e:
            print(f"❌ Discovery failed: {e}", file=sys.stderr)
            return 1

        # JSON output
        if args.json:
            import json

            output = {
                "routers": [r.to_dict() for r in routers],
                "count": len(routers),
            }
            if args.stats:
                output["stats"] = client.get_stats()
            print(json.dumps(output, indent=2))
            return 0

        # Human-readable output
        if not routers:
            print("📭 No routers discovered")
            return 0

        print(f"📡 Discovered {len(routers)} router(s)\n")

        for i, router in enumerate(routers, 1):
            router_id = router.router_id
            if len(router_id) > 30:
                router_id = router_id[:27] + "..."

            endpoints = ", ".join(router.endpoints[:2])
            if len(router.endpoints) > 2:
                endpoints += f" (+{len(router.endpoints) - 2} more)"

            regions = ", ".join(router.regions) if router.regions else "unspecified"
            features = ", ".join(router.features) if router.features else "none"

            load = router.capacity.get("current_load_pct", "?")
            uptime = router.health.get("uptime_pct", "?")

            print(f"{i}. {router_id}")
            print(f"   Endpoints: {endpoints}")
            print(f"   Regions:   {regions}")
            print(f"   Features:  {features}")
            print(f"   Load: {load}% | Uptime: {uptime}%")
            print()

        # Show stats if requested
        if args.stats:
            stats = client.get_stats()
            print("─" * 40)
            print("📊 Discovery Statistics:")
            print(f"   Queries:            {stats['queries']}")
            print(f"   Cache hits:         {stats['cache_hits']}")
            print(f"   Seed failures:      {stats['seed_failures']}")
            print(f"   Signature failures: {stats['signature_failures']}")

        return 0

    return asyncio.run(run_discovery())
