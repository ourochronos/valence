#!/usr/bin/env python3
"""Valence v2 smoke test — full ingest → query → compile → retrieve loop.

Updated for WU-14: all core functions now return ValenceResponse.
"""
import asyncio
import json
import os
import sys

# Point at v2 database
os.environ["VKB_DB_HOST"] = "localhost"
os.environ["VKB_DB_PORT"] = "5434"
os.environ["VKB_DB_NAME"] = "valence_v2"
os.environ["VKB_DB_USER"] = "valence"
os.environ["VKB_DB_PASSWORD"] = "valence"

from valence.core import sources, articles, provenance, retrieval, compilation, contention, usage, forgetting
from valence.core.inference import provider as inference_provider

# Simple LLM backend for testing — just concatenates and formats
def simple_llm(prompt: str) -> str:
    """Minimal compilation that extracts key info without a real LLM."""
    import re
    # Extract source content from prompt for inclusion in output
    source_snippets = re.findall(r'Content:\s*(.+?)(?:\n|$)', prompt)
    combined = ' '.join(source_snippets) if source_snippets else 'Compiled from provided sources.'
    
    if "compile" in prompt.lower() or "summarize" in prompt.lower() or "sources to compile" in prompt.lower():
        return json.dumps({
            "title": "Python Programming Language",
            "content": f"Python is a high-level programming language. {combined[:500]}",
            "relationships": [{"source_index": 0, "relationship": "originates"}]
        })
    if "update" in prompt.lower() or "incorporate" in prompt.lower():
        return json.dumps({
            "title": "Python Programming Language (Updated)",
            "content": f"Python is a programming language with diverse opinions. {combined[:500]}",
            "relationship": "contends"
        })
    if "contention" in prompt.lower() or "contradict" in prompt.lower() or "contends" in prompt.lower():
        return json.dumps({
            "contends": True,
            "materiality": 0.7,
            "explanation": "Source disputes Python's suitability for production systems."
        })
    return json.dumps({"content": combined[:500], "title": "Result"})

inference_provider.configure(simple_llm)

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
    success = asyncio.run(smoke_test())
    sys.exit(0 if success else 1)
