#!/usr/bin/env python3
"""
Valence Federation CLI - Node management and federation operations.

Commands:
  valence-federation discover <endpoint>     Discover and register a remote node
  valence-federation list                    List known federation nodes
  valence-federation status                  Show federation status
  valence-federation trust <node_id> <level> Set trust level for a node
  valence-federation sync <node_id>          Trigger sync with a node

Environment Variables:
  VALENCE_FEDERATION_PRIVATE_KEY   Ed25519 private key (hex) for request signing
  VALENCE_FEDERATION_PUBLIC_KEY    Ed25519 public key (multibase) for DID
  VALENCE_FEDERATION_ENDPOINT      Local federation endpoint URL

Example:
  # Discover a remote node
  valence-federation discover https://valence.example.com
  
  # List active federation nodes
  valence-federation list --status active
  
  # Set trust level for a node
  valence-federation trust abc123-def456 elevated --reason "Trusted research partner"
  
  # Trigger sync with all active nodes
  valence-federation sync
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import logging
import sys
import time
from datetime import datetime
from typing import Any
from uuid import UUID

import aiohttp

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================


def get_private_key() -> bytes | None:
    """Get the Ed25519 private key from config."""
    from ..core.config import get_config
    config = get_config()
    key_hex = config.federation_private_key
    if key_hex:
        return bytes.fromhex(key_hex)
    return None


def get_public_key_multibase() -> str | None:
    """Get the Ed25519 public key in multibase format."""
    from ..core.config import get_config
    return get_config().federation_public_key


def get_local_did() -> str | None:
    """Get the local node's DID."""
    from ..core.config import get_config
    return get_config().federation_did


# =============================================================================
# REQUEST SIGNING (VFP Protocol)
# =============================================================================


def sign_request(
    method: str,
    path: str,
    body: bytes,
    private_key: bytes,
) -> dict[str, str]:
    """Sign a federation request per VFP protocol.
    
    Returns headers to include in the request:
    - X-VFP-DID: The sender's DID
    - X-VFP-Signature: Base64-encoded Ed25519 signature
    - X-VFP-Timestamp: Unix timestamp
    - X-VFP-Nonce: Random nonce
    
    The signature covers: method + path + timestamp + nonce + body_hash
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        print("Error: cryptography library required for request signing", file=sys.stderr)
        print("Install with: pip install cryptography", file=sys.stderr)
        sys.exit(1)
    
    import secrets
    
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(16)
    body_hash = hashlib.sha256(body).hexdigest()
    
    # Construct message to sign
    message = f"{method} {path} {timestamp} {nonce} {body_hash}"
    message_bytes = message.encode("utf-8")
    
    # Sign with Ed25519
    private_key_obj = Ed25519PrivateKey.from_private_bytes(private_key)
    signature = private_key_obj.sign(message_bytes)
    signature_b64 = base64.b64encode(signature).decode("ascii")
    
    did = get_local_did()
    
    return {
        "X-VFP-DID": did or "",
        "X-VFP-Signature": signature_b64,
        "X-VFP-Timestamp": timestamp,
        "X-VFP-Nonce": nonce,
    }


# =============================================================================
# HTTP CLIENT
# =============================================================================


async def federation_request(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    sign: bool = False,
) -> dict[str, Any]:
    """Make a federation HTTP request.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        body: Optional JSON body
        sign: Whether to sign the request with VFP headers
        
    Returns:
        Response JSON or error dict
    """
    headers = {"Content-Type": "application/json"}
    body_bytes = json.dumps(body).encode() if body else b""
    
    if sign:
        private_key = get_private_key()
        if not private_key:
            return {
                "error": "VALENCE_FEDERATION_PRIVATE_KEY not set",
                "hint": "Set the environment variable with your Ed25519 private key (hex)",
            }
        
        # Extract path from URL for signing
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path = parsed.path
        
        vfp_headers = sign_request(method, path, body_bytes, private_key)
        headers.update(vfp_headers)
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(
                method,
                url,
                headers=headers,
                data=body_bytes if body else None,
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    text = await response.text()
                    return {
                        "error": f"HTTP {response.status}",
                        "message": text[:500],
                    }
    except aiohttp.ClientError as e:
        return {"error": f"Network error: {e}"}
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# COMMANDS
# =============================================================================


async def cmd_discover(args: argparse.Namespace) -> int:
    """Discover and register a remote node."""
    endpoint = args.endpoint
    
    # Normalize endpoint
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    endpoint = endpoint.rstrip("/")
    
    # Fetch node metadata
    metadata_url = f"{endpoint}/.well-known/vfp-node-metadata"
    if not args.json:
        print(f"Discovering node at {metadata_url}...")
    
    result = await federation_request("GET", metadata_url)
    
    if "error" in result:
        print(f"❌ Discovery failed: {result['error']}", file=sys.stderr)
        if "message" in result:
            print(f"   {result['message']}", file=sys.stderr)
        return 1
    
    # JSON output mode - just print and return
    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    
    # Display discovered node info (human-readable)
    did = result.get("id", "unknown")
    print(f"\n✅ Discovered node: {did}")
    
    # Profile info
    profile = result.get("vfp:profile", {})
    if profile.get("name"):
        print(f"   Name: {profile['name']}")
    if profile.get("domains"):
        print(f"   Domains: {', '.join(profile['domains'])}")
    
    # Capabilities
    capabilities = result.get("vfp:capabilities", [])
    if capabilities:
        print(f"   Capabilities: {', '.join(capabilities)}")
    
    # Services
    services = result.get("service", [])
    for svc in services:
        svc_type = svc.get("type", "unknown")
        svc_endpoint = svc.get("serviceEndpoint", "")
        print(f"   Service ({svc_type}): {svc_endpoint}")
    
    # Verification method
    vms = result.get("verificationMethod", [])
    if vms:
        print(f"   Public Key: {vms[0].get('publicKeyMultibase', 'unknown')[:20]}...")
    
    # Register if requested
    if args.register:
        print(f"\nRegistering node...")
        # Use the federation tools to register
        try:
            from ..federation.discovery import register_node
            from ..federation.identity import DIDDocument
            
            did_doc = DIDDocument.from_dict(result)
            node = register_node(did_doc)
            
            if node:
                print(f"✅ Registered as node ID: {node.id}")
                print(f"   Status: {node.status.value}")
                print(f"   Trust Phase: {node.trust_phase.value}")
            else:
                print("⚠️  Registration failed (node may already exist)", file=sys.stderr)
        except Exception as e:
            print(f"❌ Registration error: {e}", file=sys.stderr)
            return 1
    
    return 0


async def cmd_list(args: argparse.Namespace) -> int:
    """List known federation nodes."""
    try:
        from ..federation.tools import federation_node_list
        
        result = federation_node_list(
            status=args.status,
            trust_phase=args.trust_phase,
            include_trust=True,
            limit=args.limit,
        )
        
        if not result.get("success"):
            print(f"❌ Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            return 1
        
        nodes = result.get("nodes", [])
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0
        
        if not nodes:
            print("No federation nodes found.")
            return 0
        
        print(f"Federation Nodes ({len(nodes)}):\n")
        print(f"{'ID':<36}  {'DID':<40}  {'Status':<12}  {'Trust':<8}  {'Phase':<12}")
        print("-" * 120)
        
        for entry in nodes:
            node = entry.get("node", entry)
            trust = entry.get("trust", {})
            
            node_id = node.get("id", "?")[:36]
            did = node.get("did", "?")
            if len(did) > 40:
                did = did[:37] + "..."
            status = node.get("status", "?")
            trust_score = trust.get("trust", {}).get("overall", 0) if trust else 0
            phase = node.get("trust_phase", "?")
            
            print(f"{node_id}  {did:<40}  {status:<12}  {trust_score:>6.1%}  {phase:<12}")
        
        return 0
        
    except ImportError:
        print("❌ Federation module not available. Is the database configured?", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


async def cmd_status(args: argparse.Namespace) -> int:
    """Show federation status."""
    try:
        from ..federation.tools import federation_sync_status
        from ..core.db import get_cursor
        
        # Get overall federation stats
        with get_cursor() as cur:
            # Node counts by status
            cur.execute("""
                SELECT status, COUNT(*) as count
                FROM federation_nodes
                GROUP BY status
            """)
            nodes_by_status = {row["status"]: row["count"] for row in cur.fetchall()}
            
            # Total sync stats
            cur.execute("""
                SELECT
                    COUNT(*) as total_peers,
                    SUM(beliefs_sent) as beliefs_sent,
                    SUM(beliefs_received) as beliefs_received,
                    MAX(last_sync_at) as last_sync
                FROM sync_state
            """)
            sync_stats = cur.fetchone()
            
            # Belief counts
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE is_local = TRUE) as local_beliefs,
                    COUNT(*) FILTER (WHERE is_local = FALSE) as federated_beliefs
                FROM beliefs
                WHERE status = 'active'
            """)
            belief_stats = cur.fetchone()
        
        if args.json:
            result = {
                "nodes": {
                    "total": sum(nodes_by_status.values()),
                    "by_status": nodes_by_status,
                },
                "sync": {
                    "peers": sync_stats["total_peers"] or 0,
                    "beliefs_sent": sync_stats["beliefs_sent"] or 0,
                    "beliefs_received": sync_stats["beliefs_received"] or 0,
                    "last_sync": sync_stats["last_sync"].isoformat() if sync_stats["last_sync"] else None,
                },
                "beliefs": {
                    "local": belief_stats["local_beliefs"] or 0,
                    "federated": belief_stats["federated_beliefs"] or 0,
                },
            }
            print(json.dumps(result, indent=2, default=str))
            return 0
        
        # Human-readable output
        print("═══════════════════════════════════════════════════════════════")
        print("                     FEDERATION STATUS")
        print("═══════════════════════════════════════════════════════════════\n")
        
        total_nodes = sum(nodes_by_status.values())
        print(f"📡 Nodes: {total_nodes} total")
        for status, count in sorted(nodes_by_status.items()):
            icon = {"active": "🟢", "discovered": "🔵", "connecting": "🟡", "suspended": "🟠", "unreachable": "🔴"}.get(status, "⚪")
            print(f"   {icon} {status}: {count}")
        
        print(f"\n🔄 Sync:")
        print(f"   Peers: {sync_stats['total_peers'] or 0}")
        print(f"   Beliefs sent: {sync_stats['beliefs_sent'] or 0}")
        print(f"   Beliefs received: {sync_stats['beliefs_received'] or 0}")
        if sync_stats["last_sync"]:
            print(f"   Last sync: {sync_stats['last_sync']}")
        
        print(f"\n📚 Beliefs:")
        print(f"   Local: {belief_stats['local_beliefs'] or 0}")
        print(f"   Federated: {belief_stats['federated_beliefs'] or 0}")
        
        # Show local node identity
        did = get_local_did()
        pub_key = get_public_key_multibase()
        if did or pub_key:
            print(f"\n🔑 Local Identity:")
            if did:
                print(f"   DID: {did}")
            if pub_key:
                print(f"   Public Key: {pub_key[:30]}...")
        
        print()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


async def cmd_trust(args: argparse.Namespace) -> int:
    """Set trust level for a node."""
    node_id = args.node_id
    level = args.level
    
    valid_levels = ["blocked", "reduced", "automatic", "elevated", "anchor"]
    if level not in valid_levels:
        print(f"❌ Invalid trust level: {level}", file=sys.stderr)
        print(f"   Valid levels: {', '.join(valid_levels)}", file=sys.stderr)
        return 1
    
    try:
        from ..federation.tools import federation_trust_set_preference, federation_trust_get
        
        # Get current trust first
        current = federation_trust_get(node_id, include_details=True)
        
        if not current.get("success"):
            print(f"❌ Node not found: {node_id}", file=sys.stderr)
            return 1
        
        # Set new preference
        result = federation_trust_set_preference(
            node_id=node_id,
            preference=level,
            manual_score=args.score,
            reason=args.reason,
        )
        
        if not result.get("success"):
            print(f"❌ Error: {result.get('error', 'Unknown error')}", file=sys.stderr)
            return 1
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0
        
        print(f"✅ Trust preference updated for node {node_id}")
        print(f"   Previous: {current.get('effective_trust', 0):.1%}")
        print(f"   New level: {level}")
        print(f"   Effective trust: {result.get('effective_trust', 0):.1%}")
        if args.reason:
            print(f"   Reason: {args.reason}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


async def cmd_sync(args: argparse.Namespace) -> int:
    """Trigger sync with a node or all active nodes."""
    try:
        from ..federation.tools import federation_sync_trigger, federation_sync_status
        
        node_id = args.node_id
        
        if node_id:
            print(f"Triggering sync with node {node_id}...")
        else:
            print("Triggering sync with all active nodes...")
        
        result = federation_sync_trigger(node_id=node_id)
        
        if not result.get("success"):
            print(f"❌ Sync failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
            return 1
        
        if args.json:
            print(json.dumps(result, indent=2, default=str))
            return 0
        
        print("✅ Sync triggered")
        
        if result.get("queued_nodes"):
            print(f"   Queued nodes: {result['queued_nodes']}")
        if result.get("beliefs_queued"):
            print(f"   Beliefs queued: {result['beliefs_queued']}")
        
        # Show current sync status
        if args.wait:
            print("\nWaiting for sync to complete...")
            # Simple polling (in production, use websockets or events)
            for _ in range(30):  # 30 second timeout
                await asyncio.sleep(1)
                status = federation_sync_status(node_id=node_id)
                syncing = any(
                    s.get("status") == "syncing"
                    for s in status.get("sync_states", [])
                )
                if not syncing:
                    print("✅ Sync completed")
                    break
            else:
                print("⚠️  Sync still in progress (timeout)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="valence-federation",
        description="Valence Federation CLI - Node management and federation operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover a remote node
  valence-federation discover https://valence.example.com
  
  # List active nodes
  valence-federation list --status active
  
  # Show federation status
  valence-federation status
  
  # Set trust level
  valence-federation trust abc123-def456 elevated --reason "Trusted partner"
  
  # Trigger sync
  valence-federation sync

Environment Variables:
  VALENCE_FEDERATION_PRIVATE_KEY   Ed25519 private key (hex) for signing
  VALENCE_FEDERATION_PUBLIC_KEY    Ed25519 public key (multibase)
  VALENCE_FEDERATION_DID           Local node's DID
        """,
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # discover command
    discover_parser = subparsers.add_parser(
        "discover",
        help="Discover and register a remote node",
        description="Discover a federation node from its URL or DID and optionally register it.",
    )
    discover_parser.add_argument(
        "endpoint",
        help="Node URL (e.g., https://valence.example.com) or DID",
    )
    discover_parser.add_argument(
        "--register", "-r",
        action="store_true",
        default=True,
        help="Register the node after discovery (default: True)",
    )
    discover_parser.add_argument(
        "--no-register",
        action="store_false",
        dest="register",
        help="Don't register the node, just discover",
    )
    discover_parser.add_argument(
        "--json",
        action="store_true",
        help="Output DID document as JSON",
    )
    
    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List known federation nodes",
        description="List all known federation nodes with their status and trust levels.",
    )
    list_parser.add_argument(
        "--status", "-s",
        choices=["discovered", "connecting", "active", "suspended", "unreachable"],
        help="Filter by node status",
    )
    list_parser.add_argument(
        "--trust-phase", "-t",
        choices=["observer", "contributor", "participant", "anchor"],
        help="Filter by trust phase",
    )
    list_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=50,
        help="Maximum nodes to list (default: 50)",
    )
    list_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show federation status",
        description="Show overall federation status including node counts, sync stats, and local identity.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # trust command
    trust_parser = subparsers.add_parser(
        "trust",
        help="Set trust level for a node",
        description="Set the trust preference for a federation node.",
    )
    trust_parser.add_argument(
        "node_id",
        help="Node UUID",
    )
    trust_parser.add_argument(
        "level",
        choices=["blocked", "reduced", "automatic", "elevated", "anchor"],
        help="Trust level to set",
    )
    trust_parser.add_argument(
        "--score",
        type=float,
        help="Manual trust score override (0.0 to 1.0)",
    )
    trust_parser.add_argument(
        "--reason",
        help="Reason for the trust preference",
    )
    trust_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    
    # sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Trigger sync with a node or all active nodes",
        description="Trigger synchronization with federation nodes.",
    )
    sync_parser.add_argument(
        "node_id",
        nargs="?",
        help="Specific node UUID to sync with (syncs all if omitted)",
    )
    sync_parser.add_argument(
        "--wait", "-w",
        action="store_true",
        help="Wait for sync to complete",
    )
    sync_parser.add_argument(
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
    
    if args.command == "discover":
        return await cmd_discover(args)
    elif args.command == "list":
        return await cmd_list(args)
    elif args.command == "status":
        return await cmd_status(args)
    elif args.command == "trust":
        return await cmd_trust(args)
    elif args.command == "sync":
        return await cmd_sync(args)
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
