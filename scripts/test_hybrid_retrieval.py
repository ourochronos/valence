#!/usr/bin/env python3
"""Live verification script for hybrid retrieval with RRF.

Tests real queries against the database to verify the hybrid retrieval
implementation works correctly with actual data.

DO NOT include this in the test suite — it requires a live database.
"""

import os
import sys

# Set up environment — reads OPENAI_API_KEY from .env or environment
from pathlib import Path

# Load .env file if present
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())

os.environ.setdefault('ORO_DB_PORT', '5433')
os.environ.setdefault('ORO_DB_HOST', '127.0.0.1')
os.environ.setdefault('ORO_DB_USER', 'valence')
os.environ.setdefault('ORO_DB_PASSWORD', 'valence')
os.environ.setdefault('ORO_DB_NAME', 'valence')
os.environ.setdefault('VALENCE_EMBEDDING_PROVIDER', 'openai')
os.environ.setdefault('VALENCE_EMBEDDING_MODEL', 'text-embedding-3-small')
os.environ.setdefault('VALENCE_EMBEDDING_DIMS', '1536')

if not os.environ.get('OPENAI_API_KEY'):
    print("ERROR: OPENAI_API_KEY not set. Add it to .env or export it.")
    sys.exit(1)

from valence.core.retrieval import _search_articles_sync


def print_result_breakdown(query: str, results: list):
    """Print detailed score breakdown for results."""
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"Results: {len(results)}")
    print(f"{'='*80}\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('title', 'Untitled')} (ID: {result.get('id', 'N/A')[:8]}...)")
        print(f"   Vec Rank: {result.get('vec_rank', 'N/A'):>4} | Vec Score: {result.get('vec_score', 0):.4f}")
        print(f"   Text Rank: {result.get('text_rank', 'N/A'):>3} | Text Score: {result.get('text_score', 0):.4f}")
        print(f"   RRF Score: {result.get('rrf_score', 0):.6f}")
        print(f"   Similarity (normalized): {result.get('similarity', 0):.4f}")
        print(f"   Freshness: {result.get('freshness_score', 0):.4f} ({result.get('freshness', 0):.1f} days)")
        
        # Show confidence if available
        confidence = result.get('confidence', {})
        if isinstance(confidence, dict):
            overall = confidence.get('overall', 0)
            print(f"   Confidence: {overall:.4f}")
        
        print()


def test_discord_configuration():
    """Test query that previously returned 0 results.
    
    Should now find discord-related articles via semantic search.
    """
    print("\n" + "="*80)
    print("TEST 1: Discord Configuration (previously 0 results)")
    print("="*80)
    
    results = _search_articles_sync("discord configuration", limit=5)
    print_result_breakdown("discord configuration", results)
    
    if len(results) == 0:
        print("❌ FAILED: Still returning 0 results")
        return False
    else:
        print(f"✅ PASSED: Found {len(results)} results")
        # Check if any result mentions discord
        has_discord = any("discord" in str(r.get('content', '')).lower() or 
                         "discord" in str(r.get('title', '')).lower() 
                         for r in results)
        if has_discord:
            print("✅ PASSED: Results contain discord content")
        else:
            print("⚠️  WARNING: Results don't mention discord explicitly")
        return True


def test_delegation_compliance():
    """Test query with exact keyword match.
    
    Should still find exact matches via text search component.
    """
    print("\n" + "="*80)
    print("TEST 2: Delegation Compliance (exact keyword match)")
    print("="*80)
    
    results = _search_articles_sync("delegation compliance", limit=5)
    print_result_breakdown("delegation compliance", results)
    
    if len(results) == 0:
        print("❌ FAILED: No results found")
        return False
    else:
        print(f"✅ PASSED: Found {len(results)} results")
        # Check text_score is significant for top result
        if results[0].get('text_score', 0) > 0:
            print(f"✅ PASSED: Text search active (text_score={results[0].get('text_score', 0):.4f})")
        return True


def test_model_economics():
    """Test semantic query that may not have exact keyword match.
    
    Should find provider economics via vector similarity.
    """
    print("\n" + "="*80)
    print("TEST 3: Model Economics (semantic match)")
    print("="*80)
    
    results = _search_articles_sync("model economics", limit=5)
    print_result_breakdown("model economics", results)
    
    if len(results) == 0:
        print("❌ FAILED: No results found")
        return False
    else:
        print(f"✅ PASSED: Found {len(results)} results")
        # Check vec_score is significant for top result
        if results[0].get('vec_score', 0) > 0:
            print(f"✅ PASSED: Vector search active (vec_score={results[0].get('vec_score', 0):.4f})")
        return True


def main():
    """Run all live verification tests."""
    print("\n" + "="*80)
    print("HYBRID RETRIEVAL LIVE VERIFICATION")
    print("="*80)
    
    tests = [
        test_discord_configuration,
        test_delegation_compliance,
        test_model_economics,
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append((test.__name__, passed))
        except Exception as e:
            print(f"❌ ERROR in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test.__name__, False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, passed_test in results:
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
