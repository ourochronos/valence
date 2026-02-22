#!/usr/bin/env python3
"""Valence v2 Demo — walk through the core value loop.

Prerequisites:
    - PostgreSQL running with valence database initialized
    - pip install -e .
    - VKB_DB_HOST, VKB_DB_NAME, VKB_DB_USER env vars set (or defaults)

Usage:
    python examples/demo.py
"""

from __future__ import annotations

import json
import sys

# Import v2 tool handlers directly
from valence.substrate.tools.sources import source_ingest
from valence.substrate.tools.articles import article_create, article_get, article_update
from valence.substrate.tools.retrieval import knowledge_search
from valence.substrate.tools.entities import entity_search
from valence.substrate.tools.contention import contention_list
from valence.vkb.tools.sessions import session_start, session_end
from valence.vkb.tools.insights import insight_extract


def pp(label: str, data: dict) -> None:
    """Pretty-print a result."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if isinstance(data, dict):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)


def main() -> int:
    print("Valence v2 Demo")
    print("================")
    print("Walking through the core value loop.\n")

    # 1. Start a session
    print("[1/7] Starting a session...")
    session = session_start(platform="api", project_context="demo")
    pp("Session started", session)
    session_id = session.get("session_id")

    if not session.get("success"):
        print("ERROR: Could not start session. Is the database running?")
        return 1

    # 2. Ingest sources and create articles
    print("\n[2/7] Ingesting sources and creating articles...")

    s1 = source_ingest(
        content="Python's GIL limits true parallelism for CPU-bound tasks",
        source_type="observation",
    )
    pp("Source 1 ingested", s1)

    s2 = source_ingest(
        content="asyncio provides effective concurrency for I/O-bound Python applications",
        source_type="observation",
    )
    pp("Source 2 ingested", s2)

    a1 = article_create(
        content="PostgreSQL with pgvector enables efficient semantic search at scale",
        domain_path=["tech", "databases"],
    )
    pp("Article 1 created", a1)

    # 3. Query semantically
    print("\n[3/7] Searching the knowledge base...")

    results = knowledge_search(query="Python concurrency patterns", include_sources=True)
    pp("Search: 'Python concurrency patterns'", results)

    results2 = knowledge_search(query="database vector search")
    pp("Search: 'database vector search'", results2)

    # 4. Update an article (versioned)
    print("\n[4/7] Updating an article (versioned)...")

    if a1.get("success") and a1.get("article", {}).get("id"):
        updated = article_update(
            article_id=a1["article"]["id"],
            content="PostgreSQL with pgvector enables efficient semantic search at scale; also supports HNSW indexing for fast ANN",
        )
        pp("Article updated", updated)

    # 5. Check entities
    print("\n[5/7] Searching entities...")

    entities = entity_search(query="Python")
    pp("Entity search: 'Python'", entities)

    # 6. Check contentions
    print("\n[6/7] Listing contentions...")

    contentions = contention_list()
    pp("Contentions", contentions)

    # 7. Extract insight and close session
    print("\n[7/7] Extracting insight and closing session...")

    if session_id:
        insight = insight_extract(
            session_id=session_id,
            content="Valence v2 demo successfully demonstrates the core value loop: ingest, search, update, contention",
            domain_path=["meta", "valence"],
        )
        pp("Insight extracted", insight)

        closed = session_end(
            session_id=session_id,
            summary="Demonstrated Valence v2 core value loop with sources, articles, search, and contentions",
            themes=["valence", "demo", "knowledge-substrate"],
        )
        pp("Session closed", closed)

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60)
    print("\nWhat happened:")
    print("  1. Started a tracked session")
    print("  2. Ingested sources and created articles")
    print("  3. Searched the knowledge base (articles + sources)")
    print("  4. Updated an article (full version history preserved)")
    print("  5. Searched entities (auto-created from articles)")
    print("  6. Listed contentions (contradictions)")
    print("  7. Extracted an insight and closed the session")
    print("\nThis is the core value loop. Over time, articles accumulate,")
    print("patterns emerge, and the substrate learns what matters to you.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
