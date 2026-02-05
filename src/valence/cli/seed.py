#!/usr/bin/env python3
"""
Valence Seed Node CLI - Bootstrap router discovery.

Commands:
  valence-seed start      Start the seed node server
  valence-seed status     Check seed node status (local or remote)

Environment Variables:
  VALENCE_SEED_HOST       Host to bind to (default: 0.0.0.0)
  VALENCE_SEED_PORT       Port to listen on (default: 8470)
  VALENCE_SEED_ID         Seed node identifier (auto-generated if not set)
  VALENCE_SEED_PEERS      Comma-separated list of peer seed URLs

Example:
  # Start seed node on default port
  valence-seed start
  
  # Start on custom port with peer seeds
  valence-seed start --port 8471 --peer https://seed1.valence.network
  
  # Check status of a remote seed
  valence-seed status --url https://seed.valence.network
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


def get_config_from_env() -> dict:
    """Get seed node configuration from core config."""
    from ..core.config import get_config
    core_config = get_config()
    config = {}
    
    if core_config.seed_host:
        config["host"] = core_config.seed_host
    
    if core_config.seed_port:
        config["port"] = core_config.seed_port
    
    if core_config.seed_id:
        config["seed_id"] = core_config.seed_id
    
    if core_config.seed_peers:
        config["known_seeds"] = [p.strip() for p in core_config.seed_peers.split(",") if p.strip()]
    
    return config


# =============================================================================
# COMMANDS
# =============================================================================


async def cmd_start(args: argparse.Namespace) -> int:
    """Start the seed node server."""
    from valence.network.seed import SeedNode, SeedConfig
    
    # Build config from env + CLI args
    config_dict = get_config_from_env()
    
    # CLI args override env
    if args.host:
        config_dict["host"] = args.host
    if args.port:
        config_dict["port"] = args.port
    if args.seed_id:
        config_dict["seed_id"] = args.seed_id
    if args.peer:
        config_dict["known_seeds"] = config_dict.get("known_seeds", []) + args.peer
    
    config = SeedConfig(**config_dict)
    node = SeedNode(config=config)
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()
    
    def handle_signal():
        logger.info("Received shutdown signal")
        stop_event.set()
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_signal)
    
    # Start the server
    await node.start()
    
    print(f"Seed node started: {node.seed_id}")
    print(f"Listening on {config.host}:{config.port}")
    if config.known_seeds:
        print(f"Known peers: {', '.join(config.known_seeds)}")
    print("\nEndpoints:")
    print(f"  POST /discover  - Get router list")
    print(f"  POST /register  - Register a router")
    print(f"  POST /heartbeat - Router health check")
    print(f"  GET  /status    - Seed node status")
    print(f"  GET  /health    - Health check")
    print("\nPress Ctrl+C to stop")
    
    # Wait for shutdown signal
    await stop_event.wait()
    
    print("\nShutting down...")
    await node.stop()
    
    return 0


async def cmd_status(args: argparse.Namespace) -> int:
    """Check seed node status."""
    from ..core.config import get_config
    url = args.url
    
    # Default to local if no URL provided
    if not url:
        port = args.port or get_config().seed_port
        url = f"http://localhost:{port}"
    
    # Normalize URL
    if not url.startswith("http"):
        url = f"http://{url}"
    url = url.rstrip("/")
    
    status_url = f"{url}/status"
    
    if not args.json:
        print(f"Checking seed node at {url}...")
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(status_url) as response:
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}", file=sys.stderr)
                    return 1
                
                data = await response.json()
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    
    # Human-readable output
    print(f"\n✅ Seed Node: {data.get('seed_id', 'unknown')}")
    print(f"   Status: {data.get('status', 'unknown')}")
    
    routers = data.get("routers", {})
    print(f"\n📡 Routers:")
    print(f"   Total registered: {routers.get('total', 0)}")
    print(f"   Healthy: {routers.get('healthy', 0)}")
    
    known_seeds = data.get("known_seeds", 0)
    print(f"\n🌐 Known peer seeds: {known_seeds}")
    
    return 0


async def cmd_discover(args: argparse.Namespace) -> int:
    """Discover routers from a seed node."""
    from ..core.config import get_config
    url = args.url
    
    # Default to local if no URL provided
    if not url:
        port = args.port or get_config().seed_port
        url = f"http://localhost:{port}"
    
    # Normalize URL
    if not url.startswith("http"):
        url = f"http://{url}"
    url = url.rstrip("/")
    
    discover_url = f"{url}/discover"
    
    # Build request body
    body = {"requested_count": args.count}
    
    preferences = {}
    if args.region:
        preferences["region"] = args.region
    if args.feature:
        preferences["features"] = args.feature
    if preferences:
        body["preferences"] = preferences
    
    if not args.json:
        print(f"Discovering routers from {url}...")
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                discover_url,
                json=body,
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    print(f"❌ Error: HTTP {response.status}", file=sys.stderr)
                    return 1
                
                data = await response.json()
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1
    
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    
    # Human-readable output
    routers = data.get("routers", [])
    
    print(f"\n📡 Discovered {len(routers)} routers from {data.get('seed_id', 'unknown')}\n")
    
    if not routers:
        print("No routers available matching your criteria.")
        return 0
    
    for i, r in enumerate(routers, 1):
        router_id = r.get("router_id", "unknown")
        if len(router_id) > 30:
            router_id = router_id[:27] + "..."
        
        endpoints = ", ".join(r.get("endpoints", [])[:2])
        regions = ", ".join(r.get("regions", []))
        load = r.get("capacity", {}).get("current_load_pct", "?")
        uptime = r.get("health", {}).get("uptime_pct", "?")
        
        print(f"{i}. {router_id}")
        print(f"   Endpoints: {endpoints}")
        print(f"   Regions: {regions or 'unspecified'}")
        print(f"   Load: {load}% | Uptime: {uptime}%")
        print()
    
    # Show other seeds
    other_seeds = data.get("other_seeds", [])
    if other_seeds:
        print(f"Other seeds available: {', '.join(other_seeds)}")
    
    return 0


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="valence-seed",
        description="Valence Seed Node - Router discovery bootstrap service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start seed node
  valence-seed start
  
  # Start on custom port with peer seeds
  valence-seed start --port 8471 --peer https://seed1.valence.network
  
  # Check local seed status
  valence-seed status
  
  # Check remote seed status
  valence-seed status --url https://seed.valence.network
  
  # Discover routers
  valence-seed discover --count 10 --region us-west

Environment Variables:
  VALENCE_SEED_HOST       Host to bind to (default: 0.0.0.0)
  VALENCE_SEED_PORT       Port to listen on (default: 8470)
  VALENCE_SEED_ID         Seed node identifier
  VALENCE_SEED_PEERS      Comma-separated peer seed URLs
        """,
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # start command
    start_parser = subparsers.add_parser(
        "start",
        help="Start the seed node server",
        description="Start the seed node server to handle router discovery.",
    )
    start_parser.add_argument(
        "--host", "-H",
        default=None,
        help="Host to bind to (default: 0.0.0.0)",
    )
    start_parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port to listen on (default: 8470)",
    )
    start_parser.add_argument(
        "--seed-id",
        default=None,
        help="Seed node identifier (auto-generated if not set)",
    )
    start_parser.add_argument(
        "--peer",
        action="append",
        default=[],
        help="Peer seed URL (can be repeated)",
    )
    
    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Check seed node status",
        description="Check the status of a local or remote seed node.",
    )
    status_parser.add_argument(
        "--url", "-u",
        default=None,
        help="Seed node URL (default: local)",
    )
    status_parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port for local seed (default: 8470)",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover routers from a seed node",
        description="Query a seed node for available routers.",
    )
    discover_parser.add_argument(
        "--url", "-u",
        default=None,
        help="Seed node URL (default: local)",
    )
    discover_parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="Port for local seed (default: 8470)",
    )
    discover_parser.add_argument(
        "--count", "-n",
        type=int,
        default=5,
        help="Number of routers to request (default: 5)",
    )
    discover_parser.add_argument(
        "--region", "-r",
        default=None,
        help="Preferred region",
    )
    discover_parser.add_argument(
        "--feature", "-f",
        action="append",
        default=[],
        help="Required feature (can be repeated)",
    )
    discover_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    return parser


async def async_main(args: argparse.Namespace) -> int:
    """Async main entry point."""
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    if args.command == "start":
        return await cmd_start(args)
    elif args.command == "status":
        return await cmd_status(args)
    elif args.command == "discover":
        return await cmd_discover(args)
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
