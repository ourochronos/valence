#!/usr/bin/env python3
"""Example 01: Hello Article — Create and search knowledge articles.

This example demonstrates the core Valence v2 workflow:
1. Ingesting a source and creating an article
2. Searching knowledge via knowledge_search
3. Updating articles with versioning

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

from valence.substrate.tools import (
    article_create,
    article_get,
    article_update,
    knowledge_search,
)


def main() -> None:
    """Run the hello article example."""
    print("=" * 60)
    print("  Valence Example 01: Hello Article")
    print("=" * 60)
    print()

    # =========================================================================
    # Step 1: Create a simple article
    # =========================================================================
    print("[Step 1] Creating an article...")
    print("-" * 40)

    result = article_create(
        content="Python is excellent for rapid prototyping.",
        domain_path=["tech", "programming", "python"],
    )

    if result["success"]:
        article = result["article"]
        print(f"  Created article: {article['id'][:8]}...")
        print(f"  Content: {article['content']}")
        article_id = article["id"]
    else:
        print(f"  Failed: {result.get('error')}")
        return

    print()

    # =========================================================================
    # Step 2: Create an article with a title
    # =========================================================================
    print("[Step 2] Creating a titled article...")
    print("-" * 40)

    result = article_create(
        content="PostgreSQL handles concurrent writes better than SQLite for multi-user applications.",
        title="PostgreSQL vs SQLite Concurrency",
        domain_path=["tech", "databases"],
    )

    if result["success"]:
        article = result["article"]
        print(f"  Created article: {article['id'][:8]}...")
        print(f"  Title: {article.get('title', '(none)')}")
        print(f"  Content: {article['content'][:60]}...")
    else:
        print(f"  Failed: {result.get('error')}")

    print()

    # =========================================================================
    # Step 3: Search knowledge
    # =========================================================================
    print("[Step 3] Searching knowledge about Python...")
    print("-" * 40)

    result = knowledge_search(
        query="Python programming",
        limit=5,
    )

    if result["success"]:
        print(f"  Found {result['total_count']} results:")
        for i, r in enumerate(result["results"], 1):
            content_preview = r.get("content", "")[:50]
            print(f"  {i}. [{r['type']}] {content_preview}...")
    else:
        print(f"  Search failed: {result.get('error')}")

    print()

    # =========================================================================
    # Step 4: Get a specific article with details
    # =========================================================================
    print("[Step 4] Getting article details...")
    print("-" * 40)

    result = article_get(
        article_id=article_id,
        include_provenance=True,
    )

    if result["success"]:
        a = result["article"]
        print(f"  Article details:")
        print(f"  ID: {a['id']}")
        print(f"  Content: {a['content']}")
        print(f"  Version: {a.get('version', 1)}")
    else:
        print(f"  Failed: {result.get('error')}")

    print()

    # =========================================================================
    # Step 5: Update an article (versioned)
    # =========================================================================
    print("[Step 5] Updating an article...")
    print("-" * 40)

    result = article_update(
        article_id=article_id,
        content="Python is excellent for rapid prototyping and has mature async support.",
    )

    if result["success"]:
        updated = result["article"]
        print(f"  Updated article:")
        print(f"  ID: {updated['id']}")
        print(f"  Version: {updated.get('version', '?')}")
        print(f"  Content: {updated['content']}")
    else:
        print(f"  Failed: {result.get('error')}")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print("""
What we demonstrated:

  Creating articles with domain classification
  Searching knowledge with knowledge_search
  Getting article details with provenance
  Updating articles with version tracking

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
