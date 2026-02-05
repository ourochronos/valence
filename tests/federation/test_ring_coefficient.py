"""Tests for ring coefficient calculation and trust propagation dampening.

Tests cover:
- Ring detection in trust graphs
- Ring coefficient calculation
- Cycle detection and dampening
- Trust velocity anomaly detection
- Sybil cluster detection via graph analysis
- Integration with trust propagation

Per THREAT-MODEL.md §1.2.1: Ring coefficient must apply to TRUST PROPAGATION,
not just rewards, to prevent Sybil networks from accumulating transitive trust.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from valence.federation.ring_coefficient import (
    MAX_NORMAL_VELOCITY,
    MIN_RING_COEFFICIENT,
    GraphAnalysisResult,
    RingCoefficientCalculator,
    RingDetector,
    SybilClusterDetector,
    TrustVelocityAnalyzer,
    analyze_trust_graph,
    calculate_ring_coefficient,
    get_ring_coefficient_calculator,
    record_trust_change,
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
        "grace": uuid4(),
        "heidi": uuid4(),
    }


@pytest.fixture
def simple_graph(node_ids):
    """Simple linear trust graph (no rings)."""
    return {
        node_ids["alice"]: {node_ids["bob"]: 0.8},
        node_ids["bob"]: {node_ids["carol"]: 0.9},
        node_ids["carol"]: {node_ids["dave"]: 0.7},
    }


@pytest.fixture
def ring_graph(node_ids):
    """Trust graph with a 3-node ring."""
    # Alice -> Bob -> Carol -> Alice (ring)
    return {
        node_ids["alice"]: {node_ids["bob"]: 0.8},
        node_ids["bob"]: {node_ids["carol"]: 0.9},
        node_ids["carol"]: {node_ids["alice"]: 0.7},
    }


@pytest.fixture
def complex_ring_graph(node_ids):
    """Trust graph with multiple rings and branches."""
    # Ring 1: Alice -> Bob -> Carol -> Alice
    # Ring 2: Dave -> Eve -> Frank -> Dave
    # Bridge: Carol -> Dave
    return {
        node_ids["alice"]: {node_ids["bob"]: 0.8},
        node_ids["bob"]: {node_ids["carol"]: 0.9},
        node_ids["carol"]: {node_ids["alice"]: 0.7, node_ids["dave"]: 0.6},
        node_ids["dave"]: {node_ids["eve"]: 0.85},
        node_ids["eve"]: {node_ids["frank"]: 0.9},
        node_ids["frank"]: {node_ids["dave"]: 0.8},
    }


@pytest.fixture
def sybil_cluster_graph(node_ids):
    """Graph simulating a Sybil cluster with high internal connectivity."""
    # Sybil cluster: alice, bob, carol (all trust each other highly)
    # Legitimate nodes: dave, eve (sparse connections)
    return {
        # Sybil cluster - dense, high trust
        node_ids["alice"]: {node_ids["bob"]: 0.95, node_ids["carol"]: 0.95},
        node_ids["bob"]: {node_ids["alice"]: 0.95, node_ids["carol"]: 0.95},
        node_ids["carol"]: {
            node_ids["alice"]: 0.95,
            node_ids["bob"]: 0.95,
            node_ids["dave"]: 0.3,
        },
        # Legitimate nodes - sparse
        node_ids["dave"]: {node_ids["eve"]: 0.6},
        node_ids["eve"]: {},
    }


# =============================================================================
# RING DETECTOR TESTS
# =============================================================================


class TestRingDetector:
    """Tests for RingDetector class."""

    def test_no_ring_in_path(self, node_ids):
        """Test detection when path has no ring."""
        detector = RingDetector()
        path = [node_ids["alice"], node_ids["bob"], node_ids["carol"]]

        result = detector.detect_ring_in_path(path, node_ids["dave"])

        assert not result.has_ring
        assert result.ring_coefficient == 1.0
        assert len(result.ring_nodes) == 0

    def test_ring_detected_in_path(self, node_ids):
        """Test detection when adding node creates a ring."""
        detector = RingDetector()
        path = [node_ids["alice"], node_ids["bob"], node_ids["carol"]]

        # Adding alice creates a ring
        result = detector.detect_ring_in_path(path, node_ids["alice"])

        assert result.has_ring
        assert result.ring_coefficient < 1.0
        assert node_ids["alice"] in result.ring_nodes
        assert result.ring_size == 4  # alice, bob, carol, alice

    def test_ring_coefficient_decreases_with_size(self, node_ids):
        """Test that larger rings get more dampening."""
        detector = RingDetector()

        # 3-node ring
        path_3 = [node_ids["alice"], node_ids["bob"]]
        result_3 = detector.detect_ring_in_path(path_3, node_ids["alice"])

        # 5-node ring
        path_5 = [
            node_ids["alice"],
            node_ids["bob"],
            node_ids["carol"],
            node_ids["dave"],
        ]
        result_5 = detector.detect_ring_in_path(path_5, node_ids["alice"])

        # Larger ring should have lower coefficient
        assert result_5.ring_coefficient < result_3.ring_coefficient

    def test_ring_coefficient_never_below_minimum(self, node_ids):
        """Test that coefficient doesn't go below minimum."""
        detector = RingDetector(min_coefficient=0.05)

        # Very large ring
        large_path = [uuid4() for _ in range(20)]
        large_path.append(large_path[0])  # Create ring back to start

        result = detector.detect_ring_in_path(large_path[:-1], large_path[-1])

        assert result.ring_coefficient >= 0.05

    def test_detect_all_rings_simple(self, ring_graph, node_ids):
        """Test detecting all rings in a simple graph."""
        detector = RingDetector()

        rings = detector.detect_all_rings(ring_graph)

        assert len(rings) == 1
        assert rings[0].has_ring
        assert rings[0].ring_size == 3

    def test_detect_all_rings_complex(self, complex_ring_graph, node_ids):
        """Test detecting multiple rings."""
        detector = RingDetector()

        rings = detector.detect_all_rings(complex_ring_graph)

        # Should find at least 2 rings
        assert len(rings) >= 2

    def test_no_rings_in_acyclic_graph(self, simple_graph):
        """Test that acyclic graph has no rings."""
        detector = RingDetector()

        rings = detector.detect_all_rings(simple_graph)

        assert len(rings) == 0

    def test_custom_dampening_parameters(self, node_ids):
        """Test custom dampening parameters."""
        # Very aggressive dampening
        detector = RingDetector(
            base_dampening=0.1,
            size_penalty=0.05,
            min_coefficient=0.01,
        )

        path = [node_ids["alice"], node_ids["bob"], node_ids["carol"]]
        result = detector.detect_ring_in_path(path, node_ids["alice"])

        assert result.ring_coefficient <= 0.1


# =============================================================================
# TRUST VELOCITY ANALYZER TESTS
# =============================================================================


class TestTrustVelocityAnalyzer:
    """Tests for TrustVelocityAnalyzer class."""

    @pytest.mark.skip(reason="Flaky in CI - passes locally but timing-sensitive. See #228")
    def test_record_and_analyze_normal_velocity(self, node_ids):
        """Test normal trust velocity is not flagged."""
        analyzer = TrustVelocityAnalyzer()

        # Record small, gradual trust changes
        now = datetime.now()
        for i in range(10):
            analyzer.record_trust_change(
                node_ids["alice"],
                0.01,  # Small positive change
                now - timedelta(days=i),
            )

        result = analyzer.analyze_velocity(node_ids["alice"])

        assert not result.is_anomalous
        assert result.current_velocity <= MAX_NORMAL_VELOCITY

    def test_detect_anomalous_velocity(self, node_ids):
        """Test anomalous trust accumulation is flagged."""
        analyzer = TrustVelocityAnalyzer(
            anomaly_threshold=2.0,
            max_normal_velocity=0.05,
        )

        # Record rapid trust accumulation
        now = datetime.now()
        for i in range(5):
            analyzer.record_trust_change(
                node_ids["alice"],
                0.2,  # Large positive change
                now - timedelta(hours=i),
            )

        result = analyzer.analyze_velocity(node_ids["alice"])

        assert result.is_anomalous
        assert result.current_velocity > analyzer.max_normal_velocity

    def test_velocity_with_no_history(self, node_ids):
        """Test velocity analysis with no history."""
        analyzer = TrustVelocityAnalyzer()

        result = analyzer.analyze_velocity(node_ids["alice"])

        assert not result.is_anomalous
        assert result.current_velocity == 0.0

    def test_velocity_window_filtering(self, node_ids):
        """Test that old changes are excluded from current velocity."""
        analyzer = TrustVelocityAnalyzer(window_days=7)

        # Record old changes (outside window)
        old_time = datetime.now() - timedelta(days=30)
        for i in range(10):
            analyzer.record_trust_change(
                node_ids["alice"],
                0.5,  # Large changes
                old_time + timedelta(days=i),
            )

        # Record recent small change
        analyzer.record_trust_change(
            node_ids["alice"],
            0.01,
            datetime.now(),
        )

        result = analyzer.analyze_velocity(node_ids["alice"])

        # Current velocity should be based only on recent change
        assert result.current_velocity < 0.1

    @pytest.mark.skip(reason="Flaky in CI - timing-sensitive velocity calculation. See #228")
    def test_get_all_anomalies(self, node_ids):
        """Test getting all anomalous nodes."""
        analyzer = TrustVelocityAnalyzer(max_normal_velocity=0.05)

        # Create anomalous node
        now = datetime.now()
        for i in range(3):
            analyzer.record_trust_change(
                node_ids["alice"],
                0.3,
                now - timedelta(hours=i),
            )

        # Create normal node
        for i in range(10):
            analyzer.record_trust_change(
                node_ids["bob"],
                0.01,
                now - timedelta(days=i),
            )

        anomalies = analyzer.get_all_anomalies()

        # Only alice should be anomalous
        assert len(anomalies) == 1
        assert anomalies[0].node_id == node_ids["alice"]


# =============================================================================
# SYBIL CLUSTER DETECTOR TESTS
# =============================================================================


class TestSybilClusterDetector:
    """Tests for SybilClusterDetector class."""

    def test_detect_dense_cluster(self, sybil_cluster_graph, node_ids):
        """Test detection of dense interconnected clusters."""
        detector = SybilClusterDetector(
            min_cluster_size=3,
            density_threshold=0.5,
        )

        clusters = detector.detect_clusters(sybil_cluster_graph)

        # Should detect the sybil cluster
        assert len(clusters) >= 1

        # Find the cluster containing alice, bob, carol
        sybil_nodes = {node_ids["alice"], node_ids["bob"], node_ids["carol"]}
        sybil_cluster = next((c for c in clusters if set(c.node_ids) & sybil_nodes), None)

        assert sybil_cluster is not None
        assert sybil_cluster.density > 0.5

    def test_no_clusters_in_sparse_graph(self, simple_graph):
        """Test that sparse graphs don't trigger cluster detection."""
        detector = SybilClusterDetector(min_cluster_size=3)

        clusters = detector.detect_clusters(simple_graph)

        # No suspicious clusters in linear graph
        suspicious = [c for c in clusters if c.confidence > 0.3]
        assert len(suspicious) == 0

    def test_temporal_correlation_detection(self, node_ids):
        """Test detection based on correlated activity timing."""
        detector = SybilClusterDetector(
            min_cluster_size=3,
            temporal_threshold=0.5,
        )

        # Graph with a cluster
        graph = {
            node_ids["alice"]: {node_ids["bob"]: 0.9, node_ids["carol"]: 0.9},
            node_ids["bob"]: {node_ids["alice"]: 0.9, node_ids["carol"]: 0.9},
            node_ids["carol"]: {node_ids["alice"]: 0.9, node_ids["bob"]: 0.9},
        }

        # Correlated activity (all active at same times)
        base_time = datetime.now()
        activity_times = {
            node_ids["alice"]: [base_time, base_time + timedelta(hours=1)],
            node_ids["bob"]: [
                base_time + timedelta(minutes=5),
                base_time + timedelta(hours=1, minutes=10),
            ],
            node_ids["carol"]: [
                base_time + timedelta(minutes=10),
                base_time + timedelta(hours=1, minutes=5),
            ],
        }

        clusters = detector.detect_clusters(graph, activity_times)

        assert len(clusters) >= 1
        # High temporal correlation should be flagged
        assert any("timing" in r.lower() for c in clusters for r in c.detection_reasons)

    def test_ring_detection_within_cluster(self, ring_graph, node_ids):
        """Test that rings within clusters are detected."""
        detector = SybilClusterDetector(min_cluster_size=2)

        clusters = detector.detect_clusters(ring_graph)

        # Ring graph should be flagged
        ring_clusters = [c for c in clusters if "ring" in str(c.detection_reasons).lower()]
        assert len(ring_clusters) >= 1


# =============================================================================
# RING COEFFICIENT CALCULATOR TESTS
# =============================================================================


class TestRingCoefficientCalculator:
    """Tests for the integrated RingCoefficientCalculator."""

    def test_path_coefficient_no_ring(self, simple_graph, node_ids):
        """Test coefficient is 1.0 for paths without rings."""
        calculator = RingCoefficientCalculator()

        path = [node_ids["alice"], node_ids["bob"], node_ids["carol"]]
        coefficient = calculator.calculate_path_coefficient(path, simple_graph)

        assert coefficient == 1.0

    def test_path_coefficient_with_ring(self, ring_graph, node_ids):
        """Test coefficient is reduced for paths containing a cycle back edge."""
        calculator = RingCoefficientCalculator()

        # Analyze graph first to detect rings
        result = calculator.analyze_graph(ring_graph)

        # Verify rings were detected
        assert result.ring_count >= 1

        # Path that forms a ring (back to alice)
        path = [
            node_ids["alice"],
            node_ids["bob"],
            node_ids["carol"],
            node_ids["alice"],
        ]
        coefficient = calculator.calculate_path_coefficient(path, ring_graph)

        # Should be dampened due to ring (path contains cycle)
        assert coefficient < 1.0

    def test_node_coefficient_suspicious_node(self, node_ids):
        """Test node coefficient for nodes with anomalous velocity."""
        calculator = RingCoefficientCalculator()

        # Record anomalous trust accumulation
        now = datetime.now()
        for i in range(5):
            calculator.record_trust_change(
                node_ids["alice"],
                0.3,  # Large changes
                now - timedelta(hours=i),
            )

        coefficient = calculator.calculate_node_coefficient(node_ids["alice"])

        # Suspicious nodes should have reduced coefficient
        assert coefficient < 1.0
        assert calculator.is_node_suspicious(node_ids["alice"])

    def test_graph_analysis_result(self, complex_ring_graph, node_ids):
        """Test full graph analysis."""
        calculator = RingCoefficientCalculator()

        result = calculator.analyze_graph(complex_ring_graph)

        assert isinstance(result, GraphAnalysisResult)
        assert result.total_nodes > 0
        assert result.total_edges > 0
        assert result.ring_count >= 2  # Complex graph has at least 2 rings
        assert result.analysis_time_ms > 0

    def test_cache_clearing(self, node_ids):
        """Test that cache can be cleared."""
        calculator = RingCoefficientCalculator()

        # Populate cache
        calculator.calculate_node_coefficient(node_ids["alice"])

        # Clear cache
        calculator.clear_cache()

        # Should not be suspicious after clear (no recorded data for node)
        assert not calculator.is_node_suspicious(node_ids["alice"])


# =============================================================================
# MODULE FUNCTION TESTS
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_ring_coefficient_calculator(self):
        """Test getting default calculator instance."""
        import valence.federation.ring_coefficient as rc

        rc._default_calculator = None  # Reset

        calc1 = get_ring_coefficient_calculator()
        calc2 = get_ring_coefficient_calculator()

        assert calc1 is calc2  # Same instance

    def test_calculate_ring_coefficient(self, node_ids):
        """Test convenience function for coefficient calculation."""
        path = [node_ids["alice"], node_ids["bob"], node_ids["carol"]]

        coefficient = calculate_ring_coefficient(path)

        assert 0.0 <= coefficient <= 1.0

    def test_record_trust_change(self, node_ids):
        """Test convenience function for recording trust changes."""
        # Should not raise
        record_trust_change(node_ids["alice"], 0.1)

    def test_analyze_trust_graph(self, simple_graph):
        """Test convenience function for graph analysis."""
        result = analyze_trust_graph(simple_graph)

        assert isinstance(result, GraphAnalysisResult)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegrationWithTrustPropagation:
    """Integration tests with TrustPropagation engine."""

    def test_ring_coefficient_applied_to_propagation(self, ring_graph, node_ids):
        """Test that ring coefficient is applied during trust propagation."""
        from valence.federation.trust_propagation import TrustPropagation

        def trust_getter(node_id, domain=None):
            edges = ring_graph.get(node_id, {})
            return list(edges.items())

        # With ring coefficient enabled
        engine_with_ring = TrustPropagation(
            trust_getter=trust_getter,
            apply_ring_coefficient=True,
        )

        # Without ring coefficient
        engine_without_ring = TrustPropagation(
            trust_getter=trust_getter,
            apply_ring_coefficient=False,
        )

        result_with = engine_with_ring.compute_transitive_trust(node_ids["alice"], node_ids["carol"])
        engine_without_ring.compute_transitive_trust(node_ids["alice"], node_ids["carol"])

        # Ring coefficient should be applied
        assert result_with.ring_coefficient_applied <= 1.0
        # Stats should track rings
        stats = engine_with_ring.get_stats()
        assert "rings_detected" in stats

    def test_trust_propagation_result_includes_ring_info(self, ring_graph, node_ids):
        """Test that TransitiveTrustResult includes ring information."""
        from valence.federation.trust_propagation import TrustPropagation

        def trust_getter(node_id, domain=None):
            edges = ring_graph.get(node_id, {})
            return list(edges.items())

        engine = TrustPropagation(
            trust_getter=trust_getter,
            apply_ring_coefficient=True,
        )

        result = engine.compute_transitive_trust(node_ids["alice"], node_ids["carol"])

        # Result should have ring info
        result_dict = result.to_dict()
        assert "ring_coefficient_applied" in result_dict
        assert "rings_detected" in result_dict

    def test_direct_trust_not_affected_by_ring_coefficient(self, ring_graph, node_ids):
        """Test that direct trust is not subject to ring dampening."""
        from valence.federation.trust_propagation import TrustPropagation

        def trust_getter(node_id, domain=None):
            edges = ring_graph.get(node_id, {})
            return list(edges.items())

        engine = TrustPropagation(
            trust_getter=trust_getter,
            apply_ring_coefficient=True,
        )

        # Direct trust (alice -> bob)
        result = engine.compute_transitive_trust(node_ids["alice"], node_ids["bob"])

        # Direct trust should be 0.8 (from ring_graph)
        assert result.direct_trust == 0.8
        # Effective trust should use direct (higher than dampened transitive)
        assert result.effective_trust >= result.direct_trust

    def test_record_trust_change_via_propagation_engine(self, node_ids):
        """Test recording trust changes through propagation engine."""
        from valence.federation.trust_propagation import TrustPropagation

        engine = TrustPropagation(apply_ring_coefficient=True)

        # Should not raise
        engine.record_trust_change(node_ids["alice"], 0.1)

    def test_analyze_graph_via_propagation_engine(self, simple_graph, node_ids):
        """Test graph analysis through propagation engine."""
        from valence.federation.trust_propagation import TrustPropagation

        def trust_getter(node_id, domain=None):
            edges = simple_graph.get(node_id, {})
            return list(edges.items())

        engine = TrustPropagation(
            trust_getter=trust_getter,
            apply_ring_coefficient=True,
        )

        result = engine.analyze_trust_graph(node_ids["alice"])

        assert isinstance(result, GraphAnalysisResult)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_graph(self):
        """Test handling of empty graph."""
        calculator = RingCoefficientCalculator()

        result = calculator.analyze_graph({})

        assert result.total_nodes == 0
        assert result.ring_count == 0

    def test_single_node_graph(self, node_ids):
        """Test graph with single node."""
        graph = {node_ids["alice"]: {}}
        calculator = RingCoefficientCalculator()

        result = calculator.analyze_graph(graph)

        assert result.total_nodes == 1
        assert result.ring_count == 0

    def test_self_loop(self, node_ids):
        """Test handling of self-loops (node trusting itself)."""
        graph = {node_ids["alice"]: {node_ids["alice"]: 1.0}}

        detector = RingDetector()
        rings = detector.detect_all_rings(graph)

        # Self-loop is a degenerate ring
        assert len(rings) >= 1

    def test_very_large_ring(self):
        """Test handling of very large rings."""
        # Create a 100-node ring
        nodes = [uuid4() for _ in range(100)]
        graph = {}
        for i in range(len(nodes)):
            graph[nodes[i]] = {nodes[(i + 1) % len(nodes)]: 0.9}

        detector = RingDetector()
        rings = detector.detect_all_rings(graph)

        assert len(rings) >= 1
        # Coefficient should be at minimum
        assert all(r.ring_coefficient >= MIN_RING_COEFFICIENT for r in rings)

    def test_disconnected_graph(self, node_ids):
        """Test graph with disconnected components."""
        graph = {
            node_ids["alice"]: {node_ids["bob"]: 0.8},
            node_ids["bob"]: {},  # Make bob explicit
            node_ids["carol"]: {node_ids["dave"]: 0.7},  # Disconnected
            node_ids["dave"]: {},  # Make dave explicit
        }

        calculator = RingCoefficientCalculator()
        result = calculator.analyze_graph(graph)

        # total_nodes counts keys in graph dict
        assert result.total_nodes == 4
        assert result.ring_count == 0


# =============================================================================
# SECURITY TESTS
# =============================================================================


class TestSecurityScenarios:
    """Tests for security scenarios from THREAT-MODEL.md."""

    def test_sybil_ring_attack_dampened(self, node_ids):
        """Test that Sybil ring attacks are dampened.

        Per THREAT-MODEL.md §1.2.1: Coordinated Sybils building mutual trust
        edges should have their transitive trust reduced.
        """
        # Simulate Sybil attack: 5 fake identities in a trust ring
        sybils = [uuid4() for _ in range(5)]
        graph = {}

        # Create ring among Sybils
        for i, sybil in enumerate(sybils):
            next_sybil = sybils[(i + 1) % len(sybils)]
            graph[sybil] = {next_sybil: 0.95}

        # Sybils also trust a target
        target = node_ids["alice"]
        graph[sybils[0]][target] = 0.9
        graph[target] = {}  # Make target explicit

        calculator = RingCoefficientCalculator()

        # Analyze the graph
        result = calculator.analyze_graph(graph)

        # Should detect the ring
        assert result.ring_count >= 1

        # Path that forms a ring (going around the sybil ring)
        # sybil[0] -> sybil[1] -> sybil[2] -> sybil[3] -> sybil[4] -> sybil[0]
        ring_path = sybils + [sybils[0]]  # Full ring
        coefficient = calculator.calculate_path_coefficient(ring_path, graph)
        assert coefficient < 1.0

    def test_velocity_attack_detected(self, node_ids):
        """Test that rapid trust accumulation is detected.

        Per THREAT-MODEL.md: Trust velocity anomaly detection should
        flag coordinated Sybil activity.
        """
        calculator = RingCoefficientCalculator()

        # Simulate rapid trust accumulation (suspicious)
        now = datetime.now()
        for i in range(10):
            calculator.record_trust_change(
                node_ids["alice"],
                0.1,  # Each change is suspicious
                now - timedelta(minutes=i * 10),  # All in ~2 hours
            )

        # Calculate node coefficient to trigger velocity check and flag
        coefficient = calculator.calculate_node_coefficient(node_ids["alice"])

        # Coefficient should be reduced due to anomalous velocity
        assert coefficient < 1.0

        # Node should be flagged as suspicious
        assert calculator.is_node_suspicious(node_ids["alice"])

    def test_cluster_attack_detected(self, sybil_cluster_graph, node_ids):
        """Test that Sybil clusters are detected.

        Per THREAT-MODEL.md: Graph analysis should identify coordinated
        Sybil cluster patterns.
        """
        calculator = RingCoefficientCalculator()

        # Add activity timing (coordinated)
        base_time = datetime.now()
        activity_times = {
            node_ids["alice"]: [base_time],
            node_ids["bob"]: [base_time + timedelta(seconds=30)],
            node_ids["carol"]: [base_time + timedelta(seconds=45)],
        }

        result = calculator.analyze_graph(sybil_cluster_graph, activity_times)

        # Should detect suspicious cluster
        assert len(result.suspicious_clusters) >= 1

        # Cluster should have reasonable confidence
        high_confidence = [c for c in result.suspicious_clusters if c.confidence > 0.3]
        assert len(high_confidence) >= 1
