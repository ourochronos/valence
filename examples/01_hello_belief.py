#!/usr/bin/env python3
"""Example 01: Hello Belief - Create and query beliefs.

This example demonstrates the core Valence workflow:
1. Creating a belief with confidence and domains
2. Querying beliefs by content
3. Updating beliefs with supersession

Requirements:
    - PostgreSQL with pgvector running (see README for Docker setup)
    - Environment variable VKB_DB_PASSWORD set
    - `pip install valence` or run from source

Usage:
    python examples/01_hello_belief.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add src to path when running from source
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from valence.core import (
    Belief,
    DimensionalConfidence,
    get_connection,
)
from valence.substrate.tools import (
    belief_create,
    belief_query,
    belief_get,
    belief_supersede,
)


def main() -> None:
    """Run the hello belief example."""
    print("=" * 60)
    print("  Valence Example 01: Hello Belief")
    print("=" * 60)
    print()

    # =========================================================================
    # Step 1: Create a simple belief
    # =========================================================================
    print("[Step 1] Creating a belief...")
    print("-" * 40)

    result = belief_create(
        content="Python is excellent for rapid prototyping.",
        confidence={"overall": 0.85},
        domain_path=["tech", "programming", "python"],
        source_type="observation",
    )

    if result["success"]:
        belief = result["belief"]
        print(f"✓ Created belief: {belief['id'][:8]}...")
        print(f"  Content: {belief['content']}")
        print(f"  Confidence: {belief['confidence']['overall']:.0%}")
        print(f"  Domains: {belief['domain_path']}")
        belief_id = belief["id"]
    else:
        print(f"✗ Failed: {result.get('error')}")
        return

    print()

    # =========================================================================
    # Step 2: Create a belief with dimensional confidence
    # =========================================================================
    print("[Step 2] Creating a belief with 6D confidence...")
    print("-" * 40)

    # Dimensional confidence tracks multiple aspects of certainty
    result = belief_create(
        content="PostgreSQL handles concurrent writes better than SQLite for multi-user applications.",
        confidence={
            "source_reliability": 0.9,      # Trusted sources (benchmarks, docs)
            "method_quality": 0.85,         # Derived from testing
            "internal_consistency": 0.95,   # Aligns with other knowledge
            "temporal_freshness": 1.0,      # Current as of now
            "corroboration": 0.7,           # Multiple sources agree
            "domain_applicability": 0.8,    # Applies broadly
        },
        domain_path=["tech", "databases"],
        source_type="inference",
        entities=[
            {"name": "PostgreSQL", "type": "tool", "role": "subject"},
            {"name": "SQLite", "type": "tool", "role": "object"},
        ],
    )

    if result["success"]:
        belief = result["belief"]
        print(f"✓ Created belief with 6D confidence:")
        print(f"  Content: {belief['content'][:60]}...")
        print(f"  Overall confidence: {belief['confidence']['overall']:.0%}")
        print(f"  Source reliability: {belief['confidence']['source_reliability']:.0%}")
        print(f"  Corroboration: {belief['confidence']['corroboration']:.0%}")
    else:
        print(f"✗ Failed: {result.get('error')}")

    print()

    # =========================================================================
    # Step 3: Query beliefs
    # =========================================================================
    print("[Step 3] Querying beliefs about Python...")
    print("-" * 40)

    result = belief_query(
        query="Python programming",
        domain_filter=["tech"],
        limit=5,
    )

    if result["success"]:
        print(f"✓ Found {result['total_count']} beliefs:")
        for i, b in enumerate(result["beliefs"], 1):
            print(f"  {i}. [{b['confidence']['overall']:.0%}] {b['content'][:50]}...")
    else:
        print(f"✗ Query failed: {result.get('error')}")

    print()

    # =========================================================================
    # Step 4: Get a specific belief with details
    # =========================================================================
    print("[Step 4] Getting belief details...")
    print("-" * 40)

    result = belief_get(
        belief_id=belief_id,
        include_history=True,
        include_tensions=True,
    )

    if result["success"]:
        b = result["belief"]
        print(f"✓ Belief details:")
        print(f"  ID: {b['id']}")
        print(f"  Content: {b['content']}")
        print(f"  Status: {b['status']}")
        print(f"  Created: {b['created_at']}")
        if result.get("history"):
            print(f"  History: {len(result['history'])} versions")
        if result.get("tensions"):
            print(f"  Tensions: {len(result['tensions'])} active")
    else:
        print(f"✗ Failed: {result.get('error')}")

    print()

    # =========================================================================
    # Step 5: Supersede a belief (update with history)
    # =========================================================================
    print("[Step 5] Superseding a belief...")
    print("-" * 40)

    result = belief_supersede(
        old_belief_id=belief_id,
        new_content="Python is excellent for rapid prototyping and has mature async support.",
        reason="Updated to include async capabilities",
        confidence={"overall": 0.9},
    )

    if result["success"]:
        new_belief = result["new_belief"]
        print(f"✓ Superseded belief:")
        print(f"  Old ID: {result['old_belief_id'][:8]}...")
        print(f"  New ID: {new_belief['id'][:8]}...")
        print(f"  Reason: {result['reason']}")
        print(f"  New content: {new_belief['content']}")
    else:
        print(f"✗ Failed: {result.get('error')}")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print("""
What we demonstrated:

✓ Creating beliefs with simple or dimensional confidence
✓ Tagging beliefs with domains for organization
✓ Linking beliefs to entities (people, tools, concepts)
✓ Querying beliefs with text search and domain filters
✓ Superseding beliefs while maintaining history

Next steps:
- See 02_trust_graph.py for trust operations
- See 03_federation.py for peer-to-peer sharing
""")


if __name__ == "__main__":
    # Check for database connection
    if not os.environ.get("VKB_DB_PASSWORD"):
        print("Warning: VKB_DB_PASSWORD not set. Using default 'valence'.")
        os.environ["VKB_DB_PASSWORD"] = "valence"

    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker run -d --name valence-db -p 5432:5432 \\")
        print("    -e POSTGRES_PASSWORD=valence \\")
        print("    -e POSTGRES_USER=valence \\")
        print("    -e POSTGRES_DB=valence \\")
        print("    ankane/pgvector")
        print("\nThen initialize the database:")
        print("  valence init")
        sys.exit(1)
