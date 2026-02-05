#!/usr/bin/env python3
"""Example 03: Federation - Connect to peers and share knowledge.

This example demonstrates Valence's federation capabilities:
1. Creating a federation node with DID identity
2. Discovering and connecting to peers
3. Sharing beliefs across the network
4. Querying peers for knowledge

Requirements:
    - `pip install valence` or run from source
    - No database required (in-memory for demo)

Usage:
    python examples/03_federation.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# Add src to path when running from source
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Use dynamic import to avoid database dependencies
import importlib.util


def load_module(name: str, path: Path):
    """Load a module directly without going through __init__.py."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def print_header(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_step(step: int, text: str) -> None:
    """Print a step description."""
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


async def main() -> None:
    """Run the federation example."""
    print_header("Valence Example 03: Federation")

    print("""
Valence enables sovereign knowledge bases to form trust networks.
Each node maintains its own beliefs while sharing with trusted peers.

Key concepts:
- DID identity: Cryptographic identity (did:vkb:...)
- Peer discovery: Finding and connecting to other nodes
- Belief sharing: Signed beliefs with provenance
- Trust-weighted queries: Results weighted by peer reputation
""")

    # Load federation modules directly to avoid database dependencies
    identity = load_module(
        "valence.federation.identity",
        src_path / "valence/federation/identity.py"
    )
    peers = load_module(
        "valence.federation.peers",
        src_path / "valence/federation/peers.py"
    )
    server = load_module(
        "valence.federation.server",
        src_path / "valence/federation/server.py"
    )

    FederationNode = server.FederationNode

    # =========================================================================
    # Step 1: Create federation nodes
    # =========================================================================
    print_step(1, "Creating federation nodes with DID identities")

    # Create two nodes that will communicate
    node_local = FederationNode(name="LocalNode", port=8401)
    node_peer = FederationNode(name="PeerNode", port=8402)

    print("Created two federation nodes:\n")
    print(f"  Local Node:")
    print(f"    Name: {node_local.name}")
    print(f"    DID: {node_local.identity.did}")
    print(f"    Endpoint: {node_local.endpoint}")
    print(f"    Public Key: {node_local.identity.public_key_multibase[:30]}...")

    print(f"\n  Peer Node:")
    print(f"    Name: {node_peer.name}")
    print(f"    DID: {node_peer.identity.did}")
    print(f"    Endpoint: {node_peer.endpoint}")

    print("\n  DIDs are derived from ed25519 keypairs, providing:")
    print("    - Cryptographic identity verification")
    print("    - Message signing and authentication")
    print("    - No central authority required")

    # =========================================================================
    # Step 2: Start federation servers
    # =========================================================================
    print_step(2, "Starting federation servers")

    # Start both nodes
    task_local = node_local.start_background()
    task_peer = node_peer.start_background()

    # Wait for servers to start
    await asyncio.sleep(1)
    print("✓ Both federation servers running")

    # =========================================================================
    # Step 3: Discover and connect to peer
    # =========================================================================
    print_step(3, "Discovering and connecting to peer")

    # Local node introduces itself to peer
    result = await node_local.introduce_to(node_peer.endpoint)

    if result.get("success"):
        print(f"✓ Introduction successful!")
        print(f"  Peer says: \"{result.get('message')}\"")
    else:
        print(f"✗ Introduction failed: {result.get('error')}")
        return

    # Peer introduces back (bidirectional connection)
    result = await node_peer.introduce_to(node_local.endpoint)

    if result.get("success"):
        print(f"✓ Bidirectional connection established")
    else:
        print(f"✗ Reverse introduction failed: {result.get('error')}")

    # Show peer registries
    local_peers = node_local.peer_store.list_peers()
    peer_peers = node_peer.peer_store.list_peers()

    print(f"\n  Peer registries:")
    print(f"    LocalNode knows: {[p.name for p in local_peers]}")
    print(f"    PeerNode knows: {[p.name for p in peer_peers]}")

    # =========================================================================
    # Step 4: Share beliefs with peer
    # =========================================================================
    print_step(4, "Sharing beliefs with peer")

    peer_did = node_peer.identity.did

    beliefs_to_share = [
        {
            "content": "Distributed systems should fail gracefully, not catastrophically.",
            "confidence": 0.92,
            "domains": ["tech", "architecture"],
        },
        {
            "content": "Code review catches bugs but also spreads knowledge.",
            "confidence": 0.88,
            "domains": ["tech", "practices"],
        },
        {
            "content": "The best API is one that's hard to misuse.",
            "confidence": 0.85,
            "domains": ["tech", "design"],
        },
    ]

    print("Sharing beliefs from LocalNode → PeerNode:\n")

    for belief in beliefs_to_share:
        result = await node_local.share_belief(
            peer_did=peer_did,
            content=belief["content"],
            confidence=belief["confidence"],
            domains=belief["domains"],
        )

        if result.get("success"):
            print(f"  ✓ [{belief['confidence']:.0%}] {belief['content'][:45]}...")
        else:
            print(f"  ✗ Failed: {result.get('error')}")

    print(f"\n  LocalNode beliefs: {len(node_local.beliefs)}")
    print(f"  PeerNode beliefs: {len(node_peer.beliefs)} (received)")

    # =========================================================================
    # Step 5: Query peer for knowledge
    # =========================================================================
    print_step(5, "Querying peer for knowledge")

    # PeerNode shares some beliefs back
    local_did = node_local.identity.did

    await node_peer.share_belief(
        peer_did=local_did,
        content="Testing in production requires excellent observability.",
        confidence=0.78,
        domains=["tech", "devops"],
    )

    await node_peer.share_belief(
        peer_did=local_did,
        content="Documentation should be written for your future self.",
        confidence=0.94,
        domains=["tech", "practices"],
    )

    print("PeerNode shared beliefs back to LocalNode")

    # Query for tech beliefs
    print("\nLocalNode queries PeerNode for 'architecture' beliefs:")

    result = await node_local.query_peer(
        peer_did=peer_did,
        query="",
        domains=["architecture"],
        min_confidence=0.5,
    )

    if result.get("success"):
        print(f"  Found {result.get('total', 0)} results:")
        for belief in result.get("results", []):
            print(f"    [{belief['confidence']:.0%}] {belief['content'][:50]}...")
    else:
        print(f"  Query failed: {result.get('error')}")

    # Query by content
    print("\nPeerNode queries LocalNode for 'code review':")

    result = await node_peer.query_peer(
        peer_did=local_did,
        query="code review",
        min_confidence=0.0,
    )

    if result.get("success"):
        print(f"  Found {result.get('total', 0)} results:")
        for belief in result.get("results", []):
            print(f"    [{belief['confidence']:.0%}] {belief['content'][:50]}...")

    # =========================================================================
    # Step 6: Check trust after interactions
    # =========================================================================
    print_step(6, "Trust scores after interactions")

    local_view = node_local.peer_store.get_peer(peer_did)
    peer_view = node_peer.peer_store.get_peer(local_did)

    if local_view:
        print(f"  LocalNode's view of PeerNode:")
        print(f"    Trust score: {local_view.trust_score:.2%}")
        print(f"    Beliefs sent: {local_view.beliefs_sent}")
        print(f"    Beliefs received: {local_view.beliefs_received}")
        print(f"    Queries: {local_view.queries_sent} sent, {local_view.queries_received} received")

    if peer_view:
        print(f"\n  PeerNode's view of LocalNode:")
        print(f"    Trust score: {peer_view.trust_score:.2%}")
        print(f"    Beliefs sent: {peer_view.beliefs_sent}")
        print(f"    Beliefs received: {peer_view.beliefs_received}")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\nShutting down servers...")
    await node_local.stop()
    await node_peer.stop()

    task_local.cancel()
    task_peer.cancel()

    try:
        await task_local
    except asyncio.CancelledError:
        pass

    try:
        await task_peer
    except asyncio.CancelledError:
        pass

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Summary")

    print("""
What we demonstrated:

✓ DID Identity: Cryptographic identities for each node
✓ Peer Discovery: Nodes introduced themselves and connected
✓ Belief Sharing: Signed beliefs shared between peers
✓ Cross-Node Queries: Querying peers for relevant knowledge
✓ Trust Tracking: Interactions updated trust scores

Federation principles:
- Sovereignty: Each node owns its data completely
- Opt-in sharing: You choose what to share and with whom
- Cryptographic verification: All messages are signed
- Trust-weighted results: Peer reputation affects relevance

Production considerations:
- Use persistent storage instead of in-memory
- Configure privacy parameters for aggregation
- Set up DNS/DID verification for public nodes
- Implement backup and recovery procedures

For more details, see:
- spec/FEDERATION.md - Federation protocol specification
- spec/TRUST.md - Trust system specification
- docs/VISION.md - Project vision and principles
""")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExample interrupted.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nThis example runs without a database connection.")
        print("If you see import errors, check that valence is installed correctly:")
        print("  pip install -e .")
        sys.exit(1)
