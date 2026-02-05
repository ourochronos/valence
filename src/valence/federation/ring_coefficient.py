"""Ring Coefficient for Trust Propagation.

Implements cycle detection and dampening for trust graphs to prevent
Sybil networks from accumulating transitive trust through coordinated rings.

Per THREAT-MODEL.md §1.2.1: Ring detection penalty must apply to TRUST
PROPAGATION, not just rewards. This module provides:

1. Ring/cycle detection in trust graphs
2. Ring coefficient calculation (dampening factor)
3. Trust velocity anomaly detection
4. Coordinated Sybil cluster detection via graph analysis

Key concepts:
- Ring coefficient: A dampening factor (0.0-1.0) applied when trust
  would flow through cycles in the graph
- Trust velocity: Rate of trust accumulation over time
- Sybil cluster: Group of nodes with suspiciously coordinated trust patterns
"""

from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Ring coefficient parameters
DEFAULT_RING_DAMPENING = 0.3  # Base dampening when ring detected
RING_SIZE_PENALTY = 0.1  # Additional penalty per ring member
MIN_RING_COEFFICIENT = 0.05  # Minimum coefficient (never fully zero)
MAX_RING_SIZE_PENALTY = 0.8  # Cap on ring size penalty

# Trust velocity parameters
VELOCITY_WINDOW_DAYS = 7  # Window for velocity calculation
VELOCITY_ANOMALY_THRESHOLD = 3.0  # Std deviations above mean
MAX_NORMAL_VELOCITY = 0.1  # Max trust gain per day considered normal

# Sybil cluster detection parameters
MIN_CLUSTER_SIZE = 3  # Minimum nodes for a cluster
CLUSTER_DENSITY_THRESHOLD = 0.6  # Edge density threshold
TEMPORAL_CORRELATION_THRESHOLD = 0.7  # Timing correlation threshold


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class RingDetectionResult:
    """Result of ring/cycle detection in trust graph."""

    has_ring: bool
    ring_nodes: list[UUID] = field(default_factory=list)
    ring_coefficient: float = 1.0  # 1.0 = no dampening
    ring_size: int = 0
    detection_path: list[UUID] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_ring": self.has_ring,
            "ring_nodes": [str(n) for n in self.ring_nodes],
            "ring_coefficient": self.ring_coefficient,
            "ring_size": self.ring_size,
            "detection_path": [str(n) for n in self.detection_path],
        }


@dataclass
class TrustVelocityResult:
    """Result of trust velocity analysis."""

    node_id: UUID
    current_velocity: float  # Trust change per day
    historical_mean: float
    historical_std: float
    is_anomalous: bool
    anomaly_score: float  # How many std deviations above mean
    trust_changes: list[tuple[datetime, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": str(self.node_id),
            "current_velocity": self.current_velocity,
            "historical_mean": self.historical_mean,
            "historical_std": self.historical_std,
            "is_anomalous": self.is_anomalous,
            "anomaly_score": self.anomaly_score,
        }


@dataclass
class SybilCluster:
    """A detected cluster of potentially coordinated Sybil nodes."""

    cluster_id: str
    node_ids: list[UUID]
    density: float  # Edge density within cluster
    temporal_correlation: float  # How synchronized their activity is
    confidence: float  # Confidence this is a Sybil cluster
    detection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "node_ids": [str(n) for n in self.node_ids],
            "density": self.density,
            "temporal_correlation": self.temporal_correlation,
            "confidence": self.confidence,
            "detection_reasons": self.detection_reasons,
        }


@dataclass
class GraphAnalysisResult:
    """Result of full graph analysis for Sybil detection."""

    total_nodes: int
    total_edges: int
    ring_count: int
    suspicious_clusters: list[SybilCluster]
    velocity_anomalies: list[TrustVelocityResult]
    analysis_time_ms: float
    analyzed_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "ring_count": self.ring_count,
            "suspicious_clusters": [c.to_dict() for c in self.suspicious_clusters],
            "velocity_anomalies": [v.to_dict() for v in self.velocity_anomalies],
            "analysis_time_ms": self.analysis_time_ms,
            "analyzed_at": self.analyzed_at.isoformat(),
        }


# =============================================================================
# RING DETECTOR
# =============================================================================


class RingDetector:
    """Detects rings/cycles in trust graphs and calculates dampening coefficients.

    A ring occurs when trust flows in a cycle: A trusts B trusts C trusts A.
    Such rings can be exploited by Sybil networks to artificially inflate
    transitive trust. The ring coefficient dampens trust propagation through
    detected cycles.

    Algorithm:
    1. Track visited nodes during propagation
    2. When a cycle is detected (back edge to visited node), calculate coefficient
    3. Coefficient decreases with ring size (larger rings = more suspicious)
    4. Minimum coefficient prevents complete trust elimination

    Example:
        >>> detector = RingDetector()
        >>> result = detector.detect_ring(graph, path=[a, b, c], target=a)
        >>> if result.has_ring:
        ...     trust *= result.ring_coefficient
    """

    def __init__(
        self,
        base_dampening: float = DEFAULT_RING_DAMPENING,
        size_penalty: float = RING_SIZE_PENALTY,
        min_coefficient: float = MIN_RING_COEFFICIENT,
    ):
        """Initialize the ring detector.

        Args:
            base_dampening: Base coefficient when ring detected (0.0-1.0)
            size_penalty: Additional penalty per ring member
            min_coefficient: Floor for the coefficient
        """
        self.base_dampening = base_dampening
        self.size_penalty = size_penalty
        self.min_coefficient = min_coefficient

    def detect_ring_in_path(
        self,
        path: list[UUID],
        target: UUID,
    ) -> RingDetectionResult:
        """Detect if adding target to path creates a ring.

        Args:
            path: Current path of node IDs being traversed
            target: Next node to visit

        Returns:
            RingDetectionResult with ring info and coefficient
        """
        if target in path:
            # Found a ring!
            ring_start_idx = path.index(target)
            ring_nodes = path[ring_start_idx:] + [target]
            ring_size = len(ring_nodes)

            coefficient = self._calculate_coefficient(ring_size)

            return RingDetectionResult(
                has_ring=True,
                ring_nodes=ring_nodes,
                ring_coefficient=coefficient,
                ring_size=ring_size,
                detection_path=list(path),
            )

        return RingDetectionResult(
            has_ring=False,
            ring_coefficient=1.0,
        )

    def detect_all_rings(
        self,
        graph: dict[UUID, dict[UUID, float]],
    ) -> list[RingDetectionResult]:
        """Detect all rings in a trust graph using DFS.

        Args:
            graph: Adjacency dict mapping node_id -> {neighbor_id -> trust}

        Returns:
            List of all detected rings
        """
        rings: list[RingDetectionResult] = []
        visited: set[UUID] = set()
        rec_stack: set[UUID] = set()

        def dfs(node: UUID, path: list[UUID]) -> None:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, {}):
                if neighbor not in visited:
                    dfs(neighbor, path + [node])
                elif neighbor in rec_stack:
                    # Found a cycle
                    ring_start = path.index(neighbor) if neighbor in path else len(path)
                    ring_nodes = path[ring_start:] + [node, neighbor]

                    # Avoid duplicate rings (same nodes, different start)
                    ring_set = frozenset(ring_nodes[:-1])  # Exclude duplicate of start
                    if not any(frozenset(r.ring_nodes[:-1]) == ring_set for r in rings):
                        coefficient = self._calculate_coefficient(len(ring_nodes) - 1)
                        rings.append(
                            RingDetectionResult(
                                has_ring=True,
                                ring_nodes=ring_nodes,
                                ring_coefficient=coefficient,
                                ring_size=len(ring_nodes) - 1,
                                detection_path=path + [node],
                            )
                        )

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return rings

    def _calculate_coefficient(self, ring_size: int) -> float:
        """Calculate the ring coefficient based on ring size.

        Larger rings get more severe dampening as they indicate
        more coordinated Sybil activity.

        Formula:
            coefficient = base_dampening - (ring_size - 2) * size_penalty
            coefficient = max(coefficient, min_coefficient)

        Args:
            ring_size: Number of nodes in the ring

        Returns:
            Ring coefficient between min_coefficient and base_dampening
        """
        # Base penalty for any ring
        coefficient = self.base_dampening

        # Additional penalty for larger rings (starts at size 3)
        if ring_size > 2:
            size_penalty = min(
                (ring_size - 2) * self.size_penalty,
                MAX_RING_SIZE_PENALTY,
            )
            coefficient -= size_penalty

        return max(self.min_coefficient, coefficient)

    def get_ring_coefficient_for_path(
        self,
        graph: dict[UUID, dict[UUID, float]],
        path: list[UUID],
    ) -> float:
        """Calculate cumulative ring coefficient for a path.

        Checks for any rings along the path and returns the product
        of all ring coefficients encountered.

        Args:
            graph: Trust graph
            path: Path to check

        Returns:
            Cumulative ring coefficient (product of all ring coefficients)
        """
        if len(path) < 2:
            return 1.0

        coefficient = 1.0
        visited: set[UUID] = set()
        visited_list: list[UUID] = []

        for node in path:
            if node in visited:
                # Ring detected
                ring_start = visited_list.index(node) if node in visited_list else 0
                ring_size = len(visited) - ring_start + 1
                coefficient *= self._calculate_coefficient(ring_size)
            visited.add(node)
            visited_list.append(node)

        return coefficient


# =============================================================================
# TRUST VELOCITY ANALYZER
# =============================================================================


class TrustVelocityAnalyzer:
    """Analyzes trust velocity to detect anomalous accumulation patterns.

    Trust velocity is the rate of trust change over time. Sybil networks
    often show abnormally high trust velocity as coordinated nodes
    rapidly endorse each other.

    Detection methods:
    1. Calculate velocity over sliding window
    2. Compare to historical baseline
    3. Flag anomalies exceeding threshold std deviations
    """

    def __init__(
        self,
        window_days: int = VELOCITY_WINDOW_DAYS,
        anomaly_threshold: float = VELOCITY_ANOMALY_THRESHOLD,
        max_normal_velocity: float = MAX_NORMAL_VELOCITY,
    ):
        """Initialize the velocity analyzer.

        Args:
            window_days: Days to consider for velocity calculation
            anomaly_threshold: Std deviations for anomaly detection
            max_normal_velocity: Maximum velocity considered normal
        """
        self.window_days = window_days
        self.anomaly_threshold = anomaly_threshold
        self.max_normal_velocity = max_normal_velocity

        # Track trust changes: node_id -> [(timestamp, trust_delta)]
        self._trust_history: dict[UUID, list[tuple[datetime, float]]] = defaultdict(list)

    def record_trust_change(
        self,
        node_id: UUID,
        trust_delta: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a trust change for velocity tracking.

        Args:
            node_id: Node that received trust change
            trust_delta: Amount of trust change (positive or negative)
            timestamp: When the change occurred (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        self._trust_history[node_id].append((timestamp, trust_delta))

        # Prune old entries
        cutoff = datetime.now() - timedelta(days=self.window_days * 2)
        self._trust_history[node_id] = [(ts, delta) for ts, delta in self._trust_history[node_id] if ts > cutoff]

    def analyze_velocity(
        self,
        node_id: UUID,
        trust_changes: list[tuple[datetime, float]] | None = None,
        reference_time: datetime | None = None,
    ) -> TrustVelocityResult:
        """Analyze trust velocity for a node.

        Args:
            node_id: Node to analyze
            trust_changes: Optional explicit history (uses tracked if None)
            reference_time: Reference time for cutoff calculations (defaults to now).
                           Useful for deterministic testing.

        Returns:
            TrustVelocityResult with velocity analysis
        """
        if trust_changes is None:
            trust_changes = self._trust_history.get(node_id, [])

        if not trust_changes:
            return TrustVelocityResult(
                node_id=node_id,
                current_velocity=0.0,
                historical_mean=0.0,
                historical_std=0.0,
                is_anomalous=False,
                anomaly_score=0.0,
            )

        # Use reference_time for deterministic testing, otherwise now
        now = reference_time or datetime.now()

        # Calculate current velocity (last window_days)
        cutoff = now - timedelta(days=self.window_days)
        recent_changes = [(ts, delta) for ts, delta in trust_changes if ts > cutoff]

        if recent_changes:
            total_change = sum(delta for _, delta in recent_changes)
            days_span = max(1, (now - min(ts for ts, _ in recent_changes)).days)
            current_velocity = total_change / days_span
        else:
            current_velocity = 0.0

        # Calculate historical baseline (all data)
        daily_velocities = self._calculate_daily_velocities(trust_changes)

        if len(daily_velocities) >= 2:
            historical_mean = sum(daily_velocities) / len(daily_velocities)
            variance = sum((v - historical_mean) ** 2 for v in daily_velocities) / len(daily_velocities)
            historical_std = math.sqrt(variance)
        else:
            historical_mean = current_velocity
            historical_std = 0.0

        # Calculate anomaly score
        if historical_std > 0:
            anomaly_score = abs(current_velocity - historical_mean) / historical_std
        else:
            anomaly_score = 0.0 if current_velocity <= self.max_normal_velocity else float("inf")

        # Determine if anomalous
        is_anomalous = anomaly_score > self.anomaly_threshold or current_velocity > self.max_normal_velocity

        return TrustVelocityResult(
            node_id=node_id,
            current_velocity=current_velocity,
            historical_mean=historical_mean,
            historical_std=historical_std,
            is_anomalous=is_anomalous,
            anomaly_score=anomaly_score,
            trust_changes=trust_changes,
        )

    def _calculate_daily_velocities(
        self,
        trust_changes: list[tuple[datetime, float]],
    ) -> list[float]:
        """Group trust changes by day and return daily velocities."""
        if not trust_changes:
            return []

        # Group by day
        daily: dict[str, float] = defaultdict(float)
        for ts, delta in trust_changes:
            day_key = ts.strftime("%Y-%m-%d")
            daily[day_key] += delta

        return list(daily.values())

    def get_all_anomalies(
        self,
        reference_time: datetime | None = None,
    ) -> list[TrustVelocityResult]:
        """Get all nodes with anomalous trust velocity.

        Args:
            reference_time: Reference time for velocity calculations (defaults to now).
                           Useful for deterministic testing.
        """
        anomalies = []
        for node_id in self._trust_history:
            result = self.analyze_velocity(node_id, reference_time=reference_time)
            if result.is_anomalous:
                anomalies.append(result)
        return anomalies


# =============================================================================
# SYBIL CLUSTER DETECTOR
# =============================================================================


class SybilClusterDetector:
    """Detects coordinated Sybil clusters using graph analysis.

    Sybil clusters are groups of fake identities controlled by the same
    attacker. They often exhibit:

    1. High internal connectivity (dense subgraph)
    2. Temporal correlation (act together)
    3. Similar trust patterns (trust same nodes)
    4. Ring structures (mutual trust loops)

    This detector uses graph analysis techniques to identify suspicious
    clusters for human review.
    """

    def __init__(
        self,
        min_cluster_size: int = MIN_CLUSTER_SIZE,
        density_threshold: float = CLUSTER_DENSITY_THRESHOLD,
        temporal_threshold: float = TEMPORAL_CORRELATION_THRESHOLD,
    ):
        """Initialize the cluster detector.

        Args:
            min_cluster_size: Minimum nodes to consider a cluster
            density_threshold: Edge density threshold for suspicion
            temporal_threshold: Activity correlation threshold
        """
        self.min_cluster_size = min_cluster_size
        self.density_threshold = density_threshold
        self.temporal_threshold = temporal_threshold

    def detect_clusters(
        self,
        graph: dict[UUID, dict[UUID, float]],
        activity_times: dict[UUID, list[datetime]] | None = None,
    ) -> list[SybilCluster]:
        """Detect potential Sybil clusters in the trust graph.

        Args:
            graph: Trust graph adjacency dict
            activity_times: Optional dict of node activity timestamps

        Returns:
            List of detected suspicious clusters
        """
        clusters: list[SybilCluster] = []

        # Find strongly connected components
        sccs = self._find_strongly_connected_components(graph)

        for i, scc in enumerate(sccs):
            if len(scc) < self.min_cluster_size:
                continue

            # Calculate cluster metrics
            density = self._calculate_density(graph, scc)
            temporal_corr = self._calculate_temporal_correlation(scc, activity_times)

            reasons = []
            confidence = 0.0

            # Check density
            if density > self.density_threshold:
                reasons.append(f"High internal density ({density:.2f})")
                confidence += 0.3

            # Check temporal correlation
            if temporal_corr > self.temporal_threshold:
                reasons.append(f"Correlated activity timing ({temporal_corr:.2f})")
                confidence += 0.3

            # Check for complete rings
            ring_count = self._count_rings_in_subgraph(graph, scc)
            if ring_count > 0:
                reasons.append(f"Contains {ring_count} ring(s)")
                confidence += 0.2 * min(ring_count, 2)

            # Check for uniform trust patterns
            trust_uniformity = self._check_trust_uniformity(graph, scc)
            if trust_uniformity > 0.8:
                reasons.append(f"Uniform trust patterns ({trust_uniformity:.2f})")
                confidence += 0.2

            if reasons:
                clusters.append(
                    SybilCluster(
                        cluster_id=f"cluster_{i}",
                        node_ids=list(scc),
                        density=density,
                        temporal_correlation=temporal_corr,
                        confidence=min(1.0, confidence),
                        detection_reasons=reasons,
                    )
                )

        return clusters

    def _find_strongly_connected_components(
        self,
        graph: dict[UUID, dict[UUID, float]],
    ) -> list[set[UUID]]:
        """Find strongly connected components using Tarjan's algorithm."""
        index_counter = [0]
        stack: list[UUID] = []
        lowlinks: dict[UUID, int] = {}
        index: dict[UUID, int] = {}
        on_stack: dict[UUID, bool] = {}
        sccs: list[set[UUID]] = []

        def strongconnect(node: UUID) -> None:
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True

            for neighbor in graph.get(node, {}):
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack.get(neighbor, False):
                    lowlinks[node] = min(lowlinks[node], index[neighbor])

            if lowlinks[node] == index[node]:
                scc: set[UUID] = set()
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.add(w)
                    if w == node:
                        break
                sccs.append(scc)

        for node in graph:
            if node not in index:
                strongconnect(node)

        return sccs

    def _calculate_density(
        self,
        graph: dict[UUID, dict[UUID, float]],
        nodes: set[UUID],
    ) -> float:
        """Calculate edge density within a node set."""
        if len(nodes) < 2:
            return 0.0

        edge_count = 0
        for node in nodes:
            for neighbor in graph.get(node, {}):
                if neighbor in nodes:
                    edge_count += 1

        max_edges = len(nodes) * (len(nodes) - 1)  # Directed graph
        return edge_count / max_edges if max_edges > 0 else 0.0

    def _calculate_temporal_correlation(
        self,
        nodes: set[UUID],
        activity_times: dict[UUID, list[datetime]] | None,
    ) -> float:
        """Calculate temporal correlation of node activities."""
        if not activity_times or len(nodes) < 2:
            return 0.0

        # Get activity timestamps for cluster nodes
        node_times = []
        for node in nodes:
            times = activity_times.get(node, [])
            if times:
                node_times.append(times)

        if len(node_times) < 2:
            return 0.0

        # Count overlapping activity windows (within 1 hour)
        window = timedelta(hours=1)
        overlap_count = 0
        total_comparisons = 0

        for i, times_i in enumerate(node_times):
            for times_j in node_times[i + 1 :]:
                for t_i in times_i:
                    for t_j in times_j:
                        total_comparisons += 1
                        if abs(t_i - t_j) <= window:
                            overlap_count += 1

        return overlap_count / total_comparisons if total_comparisons > 0 else 0.0

    def _count_rings_in_subgraph(
        self,
        graph: dict[UUID, dict[UUID, float]],
        nodes: set[UUID],
    ) -> int:
        """Count rings within a subgraph."""
        # Build subgraph
        subgraph = {n: {m: t for m, t in graph.get(n, {}).items() if m in nodes} for n in nodes}

        detector = RingDetector()
        rings = detector.detect_all_rings(subgraph)
        return len(rings)

    def _check_trust_uniformity(
        self,
        graph: dict[UUID, dict[UUID, float]],
        nodes: set[UUID],
    ) -> float:
        """Check how uniform trust values are within cluster."""
        trust_values = []
        for node in nodes:
            for neighbor, trust in graph.get(node, {}).items():
                if neighbor in nodes:
                    trust_values.append(trust)

        if len(trust_values) < 2:
            return 0.0

        mean = sum(trust_values) / len(trust_values)
        variance = sum((v - mean) ** 2 for v in trust_values) / len(trust_values)
        std = math.sqrt(variance)

        # Low std relative to mean = high uniformity
        if mean > 0:
            coefficient_of_variation = std / mean
            return max(0.0, 1.0 - coefficient_of_variation)
        return 0.0


# =============================================================================
# INTEGRATED RING COEFFICIENT CALCULATOR
# =============================================================================


class RingCoefficientCalculator:
    """Integrated ring coefficient calculator for trust propagation.

    Combines ring detection, velocity analysis, and cluster detection
    to produce a comprehensive dampening coefficient for trust propagation.

    This is the main entry point for applying ring coefficients to
    transitive trust calculations.
    """

    def __init__(
        self,
        ring_detector: RingDetector | None = None,
        velocity_analyzer: TrustVelocityAnalyzer | None = None,
        cluster_detector: SybilClusterDetector | None = None,
    ):
        """Initialize the calculator.

        Args:
            ring_detector: RingDetector instance (creates default if None)
            velocity_analyzer: TrustVelocityAnalyzer (creates default if None)
            cluster_detector: SybilClusterDetector (creates default if None)
        """
        self.ring_detector = ring_detector or RingDetector()
        self.velocity_analyzer = velocity_analyzer or TrustVelocityAnalyzer()
        self.cluster_detector = cluster_detector or SybilClusterDetector()

        # Cache for node coefficients
        self._node_coefficients: dict[UUID, float] = {}
        self._suspicious_nodes: set[UUID] = set()

    def calculate_path_coefficient(
        self,
        path: list[UUID],
        graph: dict[UUID, dict[UUID, float]] | None = None,
    ) -> float:
        """Calculate ring coefficient for a trust path.

        Args:
            path: List of node IDs in the trust path
            graph: Optional trust graph for additional analysis

        Returns:
            Coefficient to multiply trust by (0.0-1.0)
        """
        coefficient = 1.0

        # Check for rings in path
        if len(path) >= 2:
            visited = set()
            for i, node in enumerate(path):
                if node in visited:
                    # Ring detected
                    result = self.ring_detector.detect_ring_in_path(path[:i], node)
                    coefficient *= result.ring_coefficient
                visited.add(node)

        # Apply node-specific penalties
        for node in path:
            if node in self._suspicious_nodes:
                coefficient *= 0.5  # Suspicious node penalty
            elif node in self._node_coefficients:
                coefficient *= self._node_coefficients[node]

        return coefficient

    def calculate_node_coefficient(
        self,
        node_id: UUID,
        graph: dict[UUID, dict[UUID, float]] | None = None,
    ) -> float:
        """Calculate ring coefficient for a specific node.

        Based on the node's involvement in rings, velocity anomalies,
        and suspicious clusters.

        Args:
            node_id: Node to calculate coefficient for
            graph: Trust graph for analysis

        Returns:
            Coefficient for the node (0.0-1.0)
        """
        if node_id in self._node_coefficients:
            return self._node_coefficients[node_id]

        coefficient = 1.0

        # Check velocity
        velocity_result = self.velocity_analyzer.analyze_velocity(node_id)
        if velocity_result.is_anomalous:
            # Reduce coefficient based on anomaly severity
            anomaly_penalty = min(0.5, velocity_result.anomaly_score * 0.1)
            coefficient *= 1.0 - anomaly_penalty
            self._suspicious_nodes.add(node_id)

        # Check rings involving this node (if graph provided)
        if graph:
            rings = self.ring_detector.detect_all_rings(graph)
            node_rings = [r for r in rings if node_id in r.ring_nodes]
            if node_rings:
                # Apply worst ring coefficient
                worst_ring_coeff = min(r.ring_coefficient for r in node_rings)
                coefficient *= worst_ring_coeff

        self._node_coefficients[node_id] = coefficient
        return coefficient

    def analyze_graph(
        self,
        graph: dict[UUID, dict[UUID, float]],
        activity_times: dict[UUID, list[datetime]] | None = None,
    ) -> GraphAnalysisResult:
        """Perform full graph analysis for Sybil detection.

        Args:
            graph: Trust graph to analyze
            activity_times: Optional activity timestamps per node

        Returns:
            GraphAnalysisResult with all findings
        """
        import time

        start = time.time()

        # Count nodes and edges
        total_nodes = len(graph)
        total_edges = sum(len(edges) for edges in graph.values())

        # Detect rings
        rings = self.ring_detector.detect_all_rings(graph)

        # Detect clusters
        clusters = self.cluster_detector.detect_clusters(graph, activity_times)

        # Mark suspicious nodes
        for cluster in clusters:
            if cluster.confidence > 0.5:
                self._suspicious_nodes.update(cluster.node_ids)

        # Get velocity anomalies
        velocity_anomalies = self.velocity_analyzer.get_all_anomalies()

        elapsed_ms = (time.time() - start) * 1000

        return GraphAnalysisResult(
            total_nodes=total_nodes,
            total_edges=total_edges,
            ring_count=len(rings),
            suspicious_clusters=clusters,
            velocity_anomalies=velocity_anomalies,
            analysis_time_ms=elapsed_ms,
        )

    def record_trust_change(
        self,
        node_id: UUID,
        trust_delta: float,
        timestamp: datetime | None = None,
    ) -> None:
        """Record a trust change for velocity tracking.

        Args:
            node_id: Node that received trust change
            trust_delta: Amount of change
            timestamp: When the change occurred
        """
        self.velocity_analyzer.record_trust_change(node_id, trust_delta, timestamp)

        # Invalidate cached coefficient
        if node_id in self._node_coefficients:
            del self._node_coefficients[node_id]

    def clear_cache(self) -> None:
        """Clear cached coefficients."""
        self._node_coefficients.clear()

    def is_node_suspicious(self, node_id: UUID) -> bool:
        """Check if a node has been flagged as suspicious."""
        return node_id in self._suspicious_nodes


# =============================================================================
# MODULE-LEVEL FUNCTIONS
# =============================================================================


# Default calculator instance
_default_calculator: RingCoefficientCalculator | None = None
_default_calculator_lock = threading.Lock()


def get_ring_coefficient_calculator() -> RingCoefficientCalculator:
    """Get the default RingCoefficientCalculator instance.

    Thread-safe initialization using double-checked locking pattern.
    """
    global _default_calculator
    if _default_calculator is None:
        with _default_calculator_lock:
            # Double-check after acquiring lock
            if _default_calculator is None:
                _default_calculator = RingCoefficientCalculator()
    return _default_calculator


def calculate_ring_coefficient(
    path: list[UUID],
    graph: dict[UUID, dict[UUID, float]] | None = None,
) -> float:
    """Calculate ring coefficient for a path (convenience function)."""
    return get_ring_coefficient_calculator().calculate_path_coefficient(path, graph)


def record_trust_change(
    node_id: UUID,
    trust_delta: float,
    timestamp: datetime | None = None,
) -> None:
    """Record a trust change for velocity tracking (convenience function)."""
    get_ring_coefficient_calculator().record_trust_change(node_id, trust_delta, timestamp)


def analyze_trust_graph(
    graph: dict[UUID, dict[UUID, float]],
    activity_times: dict[UUID, list[datetime]] | None = None,
) -> GraphAnalysisResult:
    """Analyze a trust graph for Sybil patterns (convenience function)."""
    return get_ring_coefficient_calculator().analyze_graph(graph, activity_times)
