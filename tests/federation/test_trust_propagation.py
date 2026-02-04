"""Tests for transitive trust propagation.

Tests cover:
- Basic propagation through trust chains
- Decay factor behavior
- Max hops limit
- Cache behavior
- Edge cases (cycles, disconnected nodes)
- Query result weighting
"""

import time
from uuid import uuid4, UUID

import pytest

from valence.federation.trust_propagation import (
    TrustPropagation,
    TrustCache,
    TransitiveTrustResult,
    TrustEdge,
    DEFAULT_DECAY_FACTOR,
    DEFAULT_MAX_HOPS,
    weight_query_results_by_trust,
    compute_transitive_trust,
    get_trust_propagation,
    invalidate_trust_cache,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def node_ids():
    """Generate node IDs for testing."""
    return {
        "alice": uuid4(),
        "bob": uuid4(),
        "carol": uuid4(),
        "dave": uuid4(),
        "eve": uuid4(),
        "frank": uuid4(),
    }


@pytest.fixture
def simple_trust_graph(node_ids):
    """Create a simple trust graph for testing.
    
    Alice -> Bob (0.8)
    Bob -> Carol (0.9)
    Carol -> Dave (0.7)
    Alice -> Eve (0.5)
    Eve -> Dave (0.6)
    """
    graph = {
        node_ids["alice"]: {
            node_ids["bob"]: 0.8,
            node_ids["eve"]: 0.5,
        },
        node_ids["bob"]: {
            node_ids["carol"]: 0.9,
        },
        node_ids["carol"]: {
            node_ids["dave"]: 0.7,
        },
        node_ids["eve"]: {
            node_ids["dave"]: 0.6,
        },
    }
    return graph


def make_trust_getter(graph: dict[UUID, dict[UUID, float]]):
    """Create a trust getter function from a graph dict."""
    def getter(node_id: UUID, domain: str | None = None) -> list[tuple[UUID, float]]:
        edges = graph.get(node_id, {})
        return list(edges.items())
    return getter


# =============================================================================
# TRUST CACHE TESTS
# =============================================================================


class TestTrustCache:
    """Tests for TrustCache."""
    
    def test_cache_set_and_get(self, node_ids):
        """Test basic cache operations."""
        cache = TrustCache(ttl_seconds=60)
        
        result = TransitiveTrustResult(
            from_node_id=node_ids["alice"],
            to_node_id=node_ids["bob"],
            direct_trust=0.8,
            transitive_trust=0.8,
            path_count=1,
            shortest_path_length=1,
            computation_time_ms=1.0,
        )
        
        cache.set(result)
        
        cached = cache.get(node_ids["alice"], node_ids["bob"])
        assert cached is not None
        assert cached.transitive_trust == 0.8
        assert cached.cached is True
    
    def test_cache_miss(self, node_ids):
        """Test cache miss returns None."""
        cache = TrustCache()
        
        result = cache.get(node_ids["alice"], node_ids["bob"])
        assert result is None
    
    def test_cache_expiry(self, node_ids):
        """Test cache entries expire after TTL."""
        cache = TrustCache(ttl_seconds=1)  # 1 second TTL
        
        result = TransitiveTrustResult(
            from_node_id=node_ids["alice"],
            to_node_id=node_ids["bob"],
            direct_trust=0.8,
            transitive_trust=0.8,
            path_count=1,
            shortest_path_length=1,
            computation_time_ms=1.0,
        )
        
        cache.set(result)
        
        # Should be cached
        assert cache.get(node_ids["alice"], node_ids["bob"]) is not None
        
        # Wait for expiry
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get(node_ids["alice"], node_ids["bob"]) is None
    
    def test_cache_invalidate_node(self, node_ids):
        """Test invalidating cache entries for a specific node."""
        cache = TrustCache()
        
        # Add multiple entries
        for target in ["bob", "carol", "dave"]:
            result = TransitiveTrustResult(
                from_node_id=node_ids["alice"],
                to_node_id=node_ids[target],
                direct_trust=0.5,
                transitive_trust=0.5,
                path_count=1,
                shortest_path_length=1,
                computation_time_ms=1.0,
            )
            cache.set(result)
        
        assert cache.size == 3
        
        # Invalidate alice
        removed = cache.invalidate(node_ids["alice"])
        assert removed == 3
        assert cache.size == 0
    
    def test_cache_clear(self, node_ids):
        """Test clearing entire cache."""
        cache = TrustCache()
        
        for target in ["bob", "carol"]:
            result = TransitiveTrustResult(
                from_node_id=node_ids["alice"],
                to_node_id=node_ids[target],
                direct_trust=0.5,
                transitive_trust=0.5,
                path_count=1,
                shortest_path_length=1,
                computation_time_ms=1.0,
            )
            cache.set(result)
        
        assert cache.size == 2
        cache.clear()
        assert cache.size == 0


# =============================================================================
# TRUST PROPAGATION TESTS
# =============================================================================


class TestTrustPropagation:
    """Tests for TrustPropagation engine."""
    
    def test_direct_trust(self, node_ids, simple_trust_graph):
        """Test that direct trust is correctly identified."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["bob"],
        )
        
        assert result.direct_trust == 0.8
        assert result.transitive_trust >= 0.8
        assert result.shortest_path_length == 1
    
    def test_one_hop_transitive(self, node_ids, simple_trust_graph):
        """Test transitive trust through one intermediate node."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph),
            decay_factor=0.8,
        )
        
        # Alice -> Bob -> Carol
        # Decay is applied at each hop:
        # - Edge Alice->Bob: 1.0 * 0.8 * 0.8 = 0.64
        # - Edge Bob->Carol: 0.64 * 0.9 * 0.8 = 0.4608
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        
        assert result.direct_trust is None
        assert result.transitive_trust == pytest.approx(0.4608, rel=0.01)
        assert result.shortest_path_length == 2
    
    def test_two_hop_transitive(self, node_ids, simple_trust_graph):
        """Test transitive trust through two intermediate nodes."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph),
            decay_factor=0.8,
        )
        
        # Alice -> Bob -> Carol -> Dave
        # Path 1: 0.8 * 0.8 (decay) * 0.9 * 0.8 * 0.7 * 0.8 = 0.258048
        # Path 2: Alice -> Eve -> Dave = 0.5 * 0.8 * 0.6 * 0.8 = 0.192
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["dave"],
        )
        
        assert result.direct_trust is None
        # Should take the higher path (through Bob/Carol)
        assert result.transitive_trust == pytest.approx(0.258, rel=0.1)
        assert result.path_count >= 2  # Multiple paths to Dave
    
    def test_no_path(self, node_ids, simple_trust_graph):
        """Test when no trust path exists."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        # Frank is not in the graph at all
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["frank"],
        )
        
        assert result.transitive_trust == 0.0
        assert result.path_count == 0
        assert result.shortest_path_length is None
    
    def test_self_trust(self, node_ids, simple_trust_graph):
        """Test trust to self is always 1.0."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["alice"],
        )
        
        assert result.transitive_trust == 1.0
        assert result.shortest_path_length == 0
    
    def test_max_hops_limit(self, node_ids):
        """Test that max_hops limits propagation depth."""
        # Create a long chain: A -> B -> C -> D -> E -> F
        chain = {}
        prev = node_ids["alice"]
        for name in ["bob", "carol", "dave", "eve", "frank"]:
            chain[prev] = {node_ids[name]: 0.9}
            prev = node_ids[name]
        
        engine = TrustPropagation(
            trust_getter=make_trust_getter(chain),
            max_hops=2,
            decay_factor=1.0,  # No decay for clearer testing
        )
        
        # Alice can reach Carol (2 hops)
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        assert result.transitive_trust > 0
        
        # Alice cannot reach Dave (3 hops, exceeds max)
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["dave"],
        )
        assert result.transitive_trust == 0.0
    
    def test_min_trust_threshold(self, node_ids):
        """Test that paths below min_trust are not followed."""
        graph = {
            node_ids["alice"]: {
                node_ids["bob"]: 0.05,  # Very low trust
            },
            node_ids["bob"]: {
                node_ids["carol"]: 0.9,
            },
        }
        
        engine = TrustPropagation(
            trust_getter=make_trust_getter(graph),
            min_trust_threshold=0.1,
        )
        
        # Should not propagate through Bob due to low trust
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        
        assert result.transitive_trust == 0.0
    
    def test_decay_factor(self, node_ids):
        """Test different decay factors."""
        graph = {
            node_ids["alice"]: {node_ids["bob"]: 1.0},
            node_ids["bob"]: {node_ids["carol"]: 1.0},
        }
        
        # High decay (0.9)
        engine_high = TrustPropagation(
            trust_getter=make_trust_getter(graph),
            decay_factor=0.9,
        )
        result_high = engine_high.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        
        # Low decay (0.5)
        engine_low = TrustPropagation(
            trust_getter=make_trust_getter(graph),
            decay_factor=0.5,
        )
        result_low = engine_low.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        
        # High decay should result in more trust
        assert result_high.transitive_trust > result_low.transitive_trust
        assert result_high.transitive_trust == pytest.approx(0.81, rel=0.01)  # 1.0 * 1.0 * 0.9^2
        assert result_low.transitive_trust == pytest.approx(0.25, rel=0.01)   # 1.0 * 1.0 * 0.5^2
    
    def test_caching(self, node_ids, simple_trust_graph):
        """Test that results are cached."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        # First computation
        result1 = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        assert not result1.cached
        
        # Second computation should be cached
        result2 = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        assert result2.cached
        assert result2.transitive_trust == result1.transitive_trust
        
        # Check stats
        stats = engine.get_stats()
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1
    
    def test_cache_invalidation(self, node_ids, simple_trust_graph):
        """Test cache invalidation."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        # Populate cache
        engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["carol"],
        )
        
        assert engine.cache.size == 1
        
        # Invalidate
        engine.invalidate_node(node_ids["alice"])
        
        assert engine.cache.size == 0
    
    def test_compute_trust_for_all_peers(self, node_ids, simple_trust_graph):
        """Test batch computation for all reachable peers."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        results = engine.compute_trust_for_all_peers(node_ids["alice"])
        
        # Should have results for Bob, Carol, Dave, Eve
        assert len(results) >= 4
        assert node_ids["bob"] in results
        assert node_ids["carol"] in results
        assert node_ids["dave"] in results
        assert node_ids["eve"] in results
    
    def test_cycle_handling(self, node_ids):
        """Test that cycles in trust graph don't cause infinite loops."""
        # Create a cycle: Alice -> Bob -> Carol -> Alice
        graph = {
            node_ids["alice"]: {node_ids["bob"]: 0.8},
            node_ids["bob"]: {node_ids["carol"]: 0.8},
            node_ids["carol"]: {node_ids["alice"]: 0.8, node_ids["dave"]: 0.8},
        }
        
        engine = TrustPropagation(
            trust_getter=make_trust_getter(graph)
        )
        
        # Should complete without hanging
        result = engine.compute_transitive_trust(
            node_ids["alice"],
            node_ids["dave"],
        )
        
        assert result.transitive_trust > 0
        assert result.computation_time_ms < 1000  # Should be fast


# =============================================================================
# TRANSITIVE TRUST RESULT TESTS
# =============================================================================


class TestTransitiveTrustResult:
    """Tests for TransitiveTrustResult."""
    
    def test_effective_trust_with_direct(self, node_ids):
        """Test effective_trust returns max of direct and transitive."""
        result = TransitiveTrustResult(
            from_node_id=node_ids["alice"],
            to_node_id=node_ids["bob"],
            direct_trust=0.8,
            transitive_trust=0.6,
            path_count=2,
            shortest_path_length=1,
            computation_time_ms=1.0,
        )
        
        assert result.effective_trust == 0.8
    
    def test_effective_trust_transitive_higher(self, node_ids):
        """Test effective_trust when transitive is higher."""
        result = TransitiveTrustResult(
            from_node_id=node_ids["alice"],
            to_node_id=node_ids["bob"],
            direct_trust=0.3,
            transitive_trust=0.7,
            path_count=3,
            shortest_path_length=2,
            computation_time_ms=1.0,
        )
        
        assert result.effective_trust == 0.7
    
    def test_effective_trust_no_direct(self, node_ids):
        """Test effective_trust when no direct trust."""
        result = TransitiveTrustResult(
            from_node_id=node_ids["alice"],
            to_node_id=node_ids["carol"],
            direct_trust=None,
            transitive_trust=0.5,
            path_count=1,
            shortest_path_length=2,
            computation_time_ms=1.0,
        )
        
        assert result.effective_trust == 0.5
    
    def test_to_dict(self, node_ids):
        """Test serialization to dict."""
        result = TransitiveTrustResult(
            from_node_id=node_ids["alice"],
            to_node_id=node_ids["bob"],
            direct_trust=0.8,
            transitive_trust=0.8,
            path_count=1,
            shortest_path_length=1,
            computation_time_ms=1.5,
        )
        
        d = result.to_dict()
        
        assert d["direct_trust"] == 0.8
        assert d["transitive_trust"] == 0.8
        assert d["effective_trust"] == 0.8
        assert d["path_count"] == 1
        assert "computed_at" in d


# =============================================================================
# QUERY WEIGHTING TESTS
# =============================================================================


class TestQueryWeighting:
    """Tests for weighting query results by trust."""
    
    def test_weight_query_results(self, node_ids, simple_trust_graph):
        """Test adding trust weights to query results."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        # Simulate query results from different nodes
        results = [
            {"content": "belief 1", "origin_node_id": str(node_ids["bob"])},
            {"content": "belief 2", "origin_node_id": str(node_ids["carol"])},
            {"content": "belief 3", "origin_node_id": str(node_ids["frank"])},  # Unknown
        ]
        
        weighted = weight_query_results_by_trust(
            results,
            node_ids["alice"],
            propagation=engine,
        )
        
        # Should be sorted by trust
        assert weighted[0]["origin_node_id"] == str(node_ids["bob"])  # Direct trust
        assert weighted[0]["trust_weight"] == pytest.approx(0.8, rel=0.01)
        
        # Unknown node should have 0 weight
        frank_result = next(r for r in weighted if r["origin_node_id"] == str(node_ids["frank"]))
        assert frank_result["trust_weight"] == 0.0
    
    def test_weight_empty_results(self, node_ids, simple_trust_graph):
        """Test weighting empty results."""
        engine = TrustPropagation(
            trust_getter=make_trust_getter(simple_trust_graph)
        )
        
        weighted = weight_query_results_by_trust(
            [],
            node_ids["alice"],
            propagation=engine,
        )
        
        assert weighted == []


# =============================================================================
# MODULE FUNCTION TESTS
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_trust_propagation(self):
        """Test getting default propagation instance."""
        # Reset global state
        import valence.federation.trust_propagation as tp
        tp._default_propagation = None
        
        engine1 = get_trust_propagation()
        engine2 = get_trust_propagation()
        
        assert engine1 is engine2  # Same instance
    
    def test_invalidate_trust_cache(self, node_ids):
        """Test cache invalidation via module function."""
        # This uses the global engine which may not be set up with our test getter
        # Just verify it doesn't crash
        count = invalidate_trust_cache()
        assert count >= 0


# =============================================================================
# INTEGRATION-LIKE TESTS
# =============================================================================


class TestIntegration:
    """Integration-style tests for trust propagation."""
    
    def test_realistic_trust_network(self):
        """Test with a more realistic trust network."""
        # Create 10 nodes with various trust relationships
        nodes = {f"node_{i}": uuid4() for i in range(10)}
        
        # Build a semi-connected network
        graph = {
            nodes["node_0"]: {  # Hub node
                nodes["node_1"]: 0.9,
                nodes["node_2"]: 0.8,
                nodes["node_3"]: 0.7,
            },
            nodes["node_1"]: {
                nodes["node_4"]: 0.85,
                nodes["node_5"]: 0.6,
            },
            nodes["node_2"]: {
                nodes["node_4"]: 0.9,  # Another path to node_4
                nodes["node_6"]: 0.75,
            },
            nodes["node_3"]: {
                nodes["node_7"]: 0.8,
            },
            nodes["node_4"]: {
                nodes["node_8"]: 0.7,
            },
            nodes["node_5"]: {
                nodes["node_8"]: 0.65,  # Another path to node_8
            },
            nodes["node_6"]: {
                nodes["node_9"]: 0.9,
            },
            nodes["node_7"]: {
                nodes["node_9"]: 0.85,  # Another path to node_9
            },
        }
        
        engine = TrustPropagation(
            trust_getter=make_trust_getter(graph),
            decay_factor=0.8,
            max_hops=4,
        )
        
        # Test trust to various nodes
        results = engine.compute_trust_for_all_peers(nodes["node_0"])
        
        # Node 1 should have highest trust (direct, high)
        assert results[nodes["node_1"]].transitive_trust >= results[nodes["node_4"]].transitive_trust
        
        # Node 8 should be reachable through multiple paths
        assert results[nodes["node_8"]].path_count >= 2
        
        # Node 9 should have lower trust (further away)
        assert results[nodes["node_9"]].transitive_trust < results[nodes["node_1"]].transitive_trust
        
        # Verify caching is working
        stats = engine.get_stats()
        assert stats["computations"] > 0
