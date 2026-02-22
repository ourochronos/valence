#!/usr/bin/env python3
"""Valence v2 smoke test — full ingest → query → compile → retrieve loop.

Updated for WU-14: all core functions now return ValenceResponse.
Updated for WU-18: --live flag enables real Gemini inference backend.

Usage:
    # Default: mock LLM (for CI / offline testing)
    .venv/bin/python scripts/smoke_test_v2.py

    # Live: real Gemini 2.0 Flash inference
    .venv/bin/python scripts/smoke_test_v2.py --live
"""
import argparse
import asyncio
import json
import os
import sys

# Point at v2 database (can be overridden by environment variables)
os.environ.setdefault("VKB_DB_HOST", "localhost")
os.environ.setdefault("VKB_DB_PORT", "5434")
os.environ.setdefault("VKB_DB_NAME", "valence_v2")
os.environ.setdefault("VKB_DB_USER", "valence")
os.environ.setdefault("VKB_DB_PASSWORD", "valence")

from valence.core import sources, articles, provenance, retrieval, compilation, contention, usage, forgetting
from valence.core.inference import provider as inference_provider

# ---------------------------------------------------------------------------
# Mock backend (default, no LLM required)
# ---------------------------------------------------------------------------

# Simple LLM backend for testing — returns responses matching task schemas (DR-11 / WU-16)
def simple_llm(prompt: str) -> str:
    """Minimal mock that returns valid task-schema JSON without a real LLM."""
    import re
    # Extract source ids from prompt (format: "Source id=<uuid>: Title")
    source_ids = re.findall(r'Source id=([^\s:]+):', prompt)

    # TASK_COMPILE: "compiling a knowledge article"
    if "compiling a knowledge article" in prompt:
        source_relationships = [
            {"source_id": sid, "relationship": "originates"}
            for sid in source_ids
        ] if source_ids else []
        return json.dumps({
            "title": "Python Programming Language",
            "content": "Python is a high-level programming language created by Guido van Rossum.",
            "source_relationships": source_relationships,
        })

    # TASK_UPDATE: "updating a knowledge article with new source"
    if "updating a knowledge article" in prompt:
        return json.dumps({
            "content": "Python is a programming language with diverse opinions on its suitability.",
            "relationship": "contends",
            "changes_summary": "Added alternative perspective from new source.",
        })

    # TASK_CONTENTION: "is_contention" schema
    if "is_contention" in prompt or "contradict" in prompt.lower() or "contends with" in prompt.lower():
        return json.dumps({
            "is_contention": True,
            "materiality": 0.7,
            "explanation": "Source disputes Python's suitability for production systems.",
        })

    # Supersede merge (free-form, expects {"content": ...})
    if "superseding content" in prompt.lower() or "rewrite the following article" in prompt.lower():
        return json.dumps({"content": "Merged article content."})

    return json.dumps({"content": "Generic result.", "title": "Result"})


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valence v2 smoke test — full ingest → compile → retrieve loop",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help=(
            "Use the real Gemini 2.0 Flash backend instead of the mock LLM. "
            "Requires the 'gemini' CLI to be installed and authenticated."
        ),
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use with --live (default: gemini-2.5-flash)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Backend setup
# ---------------------------------------------------------------------------

def _configure_backend(args: argparse.Namespace) -> None:
    """Configure the inference provider based on CLI flags."""
    if args.live:
        from valence.core.backends.gemini_cli import create_gemini_backend
        backend = create_gemini_backend(model=args.model)
        inference_provider.configure(backend)
        print(f"[smoke_test] Using LIVE Gemini backend: {args.model}")
    else:
        inference_provider.configure(simple_llm)
        print("[smoke_test] Using mock LLM backend (pass --live for real inference)")

async def smoke_test():
    print("=" * 60)
    print("VALENCE V2 SMOKE TEST")
    print("=" * 60)
    
    # Step 1: Ingest sources
    print("\n--- Step 1: Ingest Sources ---")
    r = await sources.ingest_source(
        content="Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes code readability and simplicity.",
        source_type="document",
        title="Python Overview"
    )
    if not r.success:
        print(f"  ✗ Source 1 failed: {r.error}")
        return False
    s1 = r.data
    print(f"  ✓ Source 1 ingested: {s1['id']}")

    r = await sources.ingest_source(
        content="Python was first released in 1991 and has become one of the most popular programming languages in the world, used extensively in AI and data science.",
        source_type="web",
        title="Python Popularity"
    )
    if not r.success:
        print(f"  ✗ Source 2 failed: {r.error}")
        return False
    s2 = r.data
    print(f"  ✓ Source 2 ingested: {s2['id']}")

    r = await sources.ingest_source(
        content="JavaScript, not Python, is the most important programming language. Python is slow and unsuitable for production systems.",
        source_type="conversation",
        title="Controversial Take"
    )
    if not r.success:
        print(f"  ✗ Source 3 failed: {r.error}")
        return False
    s3 = r.data
    print(f"  ✓ Source 3 ingested: {s3['id']}")

    # Step 2: List sources
    print("\n--- Step 2: List Sources ---")
    r = await sources.list_sources()
    if not r.success:
        print(f"  ✗ List failed: {r.error}")
        return False
    print(f"  ✓ {len(r.data)} sources found")

    # Step 3: Search sources
    print("\n--- Step 3: Search Sources ---")
    r = await sources.search_sources("Python programming")
    if not r.success:
        print(f"  ✗ Search failed: {r.error}")
        return False
    print(f"  ✓ Search returned {len(r.data)} results")

    # Step 4: Compile article from sources
    print("\n--- Step 4: Compile Article ---")
    r = await compilation.compile_article(
        source_ids=[s1['id'], s2['id']],
        title_hint="Python Programming Language"
    )
    if not r.success:
        print(f"  ✗ Compilation failed: {r.error}")
        return False
    article = r.data
    print(f"  ✓ Article compiled: {article['id']}")
    print(f"    Title: {article.get('title', 'N/A')}")

    # Step 5: Get article with provenance
    print("\n--- Step 5: Get Article + Provenance ---")
    r = await articles.get_article(article['id'], include_provenance=True)
    if not r.success:
        print(f"  ✗ Article get failed: {r.error}")
        return False
    print(f"  ✓ Article retrieved: {r.data.get('title', 'N/A')}")

    r2 = await provenance.get_provenance(article['id'])
    if not r2.success:
        print(f"  ✗ Provenance get failed: {r2.error}")
        return False
    prov = r2.data or []
    print(f"  ✓ Provenance: {len(prov)} sources linked")
    for p in prov:
        print(f"    - {p.get('relationship', '?')}: {p.get('source_title', p.get('source_id', '?'))}")

    # Step 6: Retrieve (unified search)
    print("\n--- Step 6: Unified Retrieval ---")
    r = await retrieval.retrieve("Python programming language")
    if not r.success:
        print(f"  ✗ Retrieval failed: {r.error}")
        return False
    results = r.data or []
    print(f"  ✓ Retrieval returned {len(results)} results")
    for item in results[:3]:
        print(f"    - [{item.get('type', '?')}] {item.get('title', 'N/A')} (score: {item.get('final_score', '?')})")

    # Step 7: Update article with new source
    print("\n--- Step 7: Incremental Update ---")
    r = await compilation.update_article_from_source(article['id'], s3['id'])
    if not r.success:
        print(f"  ✗ Update failed: {r.error}")
        return False
    updated = r.data.get("article", {})
    print(f"  ✓ Article updated, version: {updated.get('version', '?')}")

    # Step 8: Check contention
    print("\n--- Step 8: Contention Check ---")
    r = await contention.detect_contention(article['id'], s3['id'])
    if not r.success:
        print(f"  ✗ Contention check failed: {r.error}")
        return False
    if r.data:
        print(f"  ✓ Contention detected: materiality={r.data.get('materiality', '?')}")
    else:
        print(f"  ✓ No contention detected (below threshold)")

    # Step 9: Usage scores
    print("\n--- Step 9: Usage Scores ---")
    r = await usage.compute_usage_scores()
    if not r.success:
        print(f"  ✗ Usage scores failed: {r.error}")
        return False
    print(f"  ✓ Usage scores computed for {r.data} articles")

    r = await usage.get_decay_candidates(limit=5)
    if not r.success:
        print(f"  ✗ Decay candidates failed: {r.error}")
        return False
    print(f"  ✓ Decay candidates: {len(r.data)}")

    # Step 10: Forgetting
    print("\n--- Step 10: Source Removal ---")
    r = await forgetting.remove_source(s3['id'])
    if not r.success:
        print(f"  ✗ Forgetting failed: {r.error}")
        return False
    print(f"  ✓ Source removed, tombstone created")

    r = await sources.list_sources()
    if r.success:
        print(f"  ✓ {len(r.data)} sources remain")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE ✓")
    print("=" * 60)
    return True

if __name__ == "__main__":
    _args = _parse_args()
    _configure_backend(_args)
    success = asyncio.run(smoke_test())
    sys.exit(0 if success else 1)
