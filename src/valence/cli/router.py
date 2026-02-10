#!/usr/bin/env python3
"""
Valence Router CLI - Start and manage router nodes.

Commands:
  valence-router start           Start a router node
  valence-router status          Query router status

Examples:
  # Start a router on default port
  valence-router start

  # Start with custom port and seed
  valence-router start --port 8472 --seed https://seed.valence.network:8470

  # Check router status
  valence-router status --host localhost --port 8471
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


async def cmd_start(args: argparse.Namespace) -> int:
    """Start a router node."""
    from oro_network import RouterNode

    router = RouterNode(
        host=args.host,
        port=args.port,
        max_connections=args.max_connections,
        seed_url=args.seed,
        heartbeat_interval=args.heartbeat_interval,
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Received shutdown signal")
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await router.start()

        if not args.json:
            print(f"Router node started on {args.host}:{args.port}")
            if args.seed:
                print(f"Registered with seed: {args.seed}")
            print("Press Ctrl+C to stop")

        # Wait for shutdown signal
        await shutdown_event.wait()

    except Exception as e:
        logger.exception(f"Router error: {e}")
        return 1

    finally:
        await router.stop()
        if not args.json:
            print("Router stopped")

    return 0


async def cmd_status(args: argparse.Namespace) -> int:
    """Query router status."""
    url = f"http://{args.host}:{args.port}/status"

    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    if args.json:
                        print(json.dumps(data, indent=2))
                    else:
                        _print_status(data)

                    return 0
                else:
                    text = await response.text()
                    print(f"❌ Error: HTTP {response.status}: {text}", file=sys.stderr)
                    return 1

    except aiohttp.ClientConnectorError:
        print(f"❌ Cannot connect to router at {args.host}:{args.port}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


def _print_status(data: dict[str, Any]) -> None:
    """Print router status in human-readable format."""
    print("═" * 60)
    print("               VALENCE ROUTER STATUS")
    print("═" * 60)
    print()

    running = data.get("running", False)
    status_icon = "🟢" if running else "🔴"
    print(f"{status_icon} Status: {'Running' if running else 'Stopped'}")
    print(f"📍 Address: {data.get('host', '?')}:{data.get('port', '?')}")

    seed = data.get("seed_url")
    if seed:
        print(f"🌱 Seed: {seed}")

    print()
    print("📊 Connections:")
    conns = data.get("connections", {})
    print(f"   Current: {conns.get('current', 0)}/{conns.get('max', 0)}")
    print(f"   Total: {conns.get('total', 0)}")

    print()
    print("📬 Queues:")
    queues = data.get("queues", {})
    print(f"   Nodes with queued messages: {queues.get('nodes', 0)}")
    print(f"   Total queued messages: {queues.get('total_messages', 0)}")

    print()
    print("📈 Metrics:")
    metrics = data.get("metrics", {})
    print(f"   Messages relayed: {metrics.get('messages_relayed', 0)}")
    print(f"   Messages queued: {metrics.get('messages_queued', 0)}")
    print(f"   Messages delivered: {metrics.get('messages_delivered', 0)}")

    print()


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="valence-router",
        description="Valence Router Node - Relay encrypted messages between nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start a router on default port
  valence-router start

  # Start with custom port and seed
  valence-router start --port 8472 --seed https://seed.valence.network:8470

  # Check router status
  valence-router status --host localhost --port 8471

Environment Variables:
  VALENCE_ROUTER_HOST          Bind address (default: 0.0.0.0)
  VALENCE_ROUTER_PORT          Listen port (default: 8471)
  VALENCE_ROUTER_SEED          Seed node URL for registration
        """,
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start a router node",
        description="Start a Valence router node that relays encrypted messages.",
    )
    start_parser.add_argument(
        "--host",
        default="0.0.0.0",  # nosec B104
        help="Bind address (default: 0.0.0.0)",
    )
    start_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8471,
        help="Listen port (default: 8471)",
    )
    start_parser.add_argument(
        "--seed",
        "-s",
        help="Seed node URL for registration (e.g., https://seed.example.com:8470)",
    )
    start_parser.add_argument(
        "--max-connections",
        "-m",
        type=int,
        default=100,
        help="Maximum concurrent connections (default: 100)",
    )
    start_parser.add_argument(
        "--heartbeat-interval",
        type=int,
        default=300,
        help="Seconds between seed heartbeats (default: 300)",
    )

    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Query router status",
        description="Get status information from a running router node.",
    )
    status_parser.add_argument(
        "--host",
        default="localhost",
        help="Router host (default: localhost)",
    )
    status_parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8471,
        help="Router port (default: 8471)",
    )

    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s: %(message)s",
        )

    if args.command == "start":
        return await cmd_start(args)
    elif args.command == "status":
        return await cmd_status(args)
    else:
        parser = create_parser()
        parser.print_help()
        return 0


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    return asyncio.run(async_main(args))


# For CLI entry point
app = main


if __name__ == "__main__":
    sys.exit(main())
