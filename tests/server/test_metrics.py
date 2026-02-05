"""Tests for Prometheus metrics endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from valence.server.metrics import (
    LATENCY_BUCKETS,
    HistogramData,
    MetricsCollector,
    metrics_endpoint,
)


class TestHistogramData:
    """Tests for HistogramData class."""

    def test_observe_increments_count(self):
        """Observe increments count and sum."""
        h = HistogramData()
        h.observe(0.1)
        assert h.count == 1
        assert h.sum == 0.1

    def test_observe_fills_buckets(self):
        """Observations fill appropriate buckets."""
        h = HistogramData()
        h.observe(0.003)  # Should be in 0.005 bucket
        h.observe(0.05)  # Should be in 0.05 bucket
        h.observe(0.5)  # Should be in 0.5 bucket

        # Check cumulative bucket counts
        assert h.buckets[0.005] == 1  # Only 0.003
        assert h.buckets[0.05] == 2  # 0.003 and 0.05
        assert h.buckets[0.5] == 3  # All three

    def test_observe_large_value(self):
        """Large values only increment count, not buckets."""
        h = HistogramData()
        h.observe(100.0)  # Larger than all buckets

        assert h.count == 1
        assert h.sum == 100.0
        # No bucket should be incremented
        for bucket in LATENCY_BUCKETS:
            assert h.buckets.get(bucket, 0) == 0


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_record_request(self):
        """Recording a request updates counts and histograms."""
        collector = MetricsCollector()
        collector.record_request("GET", "/api/v1/health", 200, 0.05)

        assert collector._request_counts[("GET", "/api/v1/health", 200)] == 1
        assert collector._latency_histograms[("GET", "/api/v1/health")].count == 1

    def test_record_multiple_requests(self):
        """Multiple requests accumulate correctly."""
        collector = MetricsCollector()
        collector.record_request("GET", "/api/v1/health", 200, 0.05)
        collector.record_request("GET", "/api/v1/health", 200, 0.1)
        collector.record_request("GET", "/api/v1/health", 500, 0.2)

        assert collector._request_counts[("GET", "/api/v1/health", 200)] == 2
        assert collector._request_counts[("GET", "/api/v1/health", 500)] == 1

    def test_normalize_path_uuid(self):
        """UUID-like path segments are normalized."""
        collector = MetricsCollector()
        path = "/api/v1/beliefs/123e4567-e89b-12d3-a456-426614174000"
        normalized = collector._normalize_path(path)
        assert normalized == "/api/v1/beliefs/{id}"

    def test_normalize_path_numeric(self):
        """Numeric path segments are normalized."""
        collector = MetricsCollector()
        path = "/api/v1/users/12345/data"
        normalized = collector._normalize_path(path)
        assert normalized == "/api/v1/users/{id}/data"

    def test_connection_tracking(self):
        """Connection increment and decrement work correctly."""
        collector = MetricsCollector()
        assert collector.get_active_connections() == 0

        collector.increment_connections()
        assert collector.get_active_connections() == 1

        collector.increment_connections()
        assert collector.get_active_connections() == 2

        collector.decrement_connections()
        assert collector.get_active_connections() == 1

    def test_connection_decrement_floor(self):
        """Connection count doesn't go negative."""
        collector = MetricsCollector()
        collector.decrement_connections()
        collector.decrement_connections()
        assert collector.get_active_connections() == 0

    def test_format_prometheus_request_counts(self):
        """Prometheus output includes request counts."""
        collector = MetricsCollector()
        collector.record_request("GET", "/health", 200, 0.01)
        collector.record_request("POST", "/api/v1/mcp", 200, 0.1)

        output = collector.format_prometheus()

        assert "valence_http_requests_total" in output
        assert 'method="GET"' in output
        assert 'method="POST"' in output
        assert 'status="200"' in output

    def test_format_prometheus_histograms(self):
        """Prometheus output includes histogram buckets."""
        collector = MetricsCollector()
        collector.record_request("GET", "/health", 200, 0.01)

        output = collector.format_prometheus()

        assert "valence_http_request_duration_seconds_bucket" in output
        assert "valence_http_request_duration_seconds_sum" in output
        assert "valence_http_request_duration_seconds_count" in output
        assert 'le="0.01"' in output
        assert 'le="+Inf"' in output

    def test_format_prometheus_active_connections(self):
        """Prometheus output includes active connections gauge."""
        collector = MetricsCollector()
        collector.increment_connections()
        collector.increment_connections()

        output = collector.format_prometheus()

        assert "valence_active_connections 2" in output

    @patch("valence.server.metrics.MetricsCollector._collect_database_metrics")
    @patch("valence.server.metrics.MetricsCollector._collect_federation_metrics")
    def test_format_prometheus_includes_db_metrics(self, mock_fed, mock_db):
        """Prometheus output includes database metrics."""
        mock_db.return_value = [
            "",
            "# HELP valence_beliefs_total Total number of beliefs",
            "# TYPE valence_beliefs_total gauge",
            "valence_beliefs_total 42",
        ]
        mock_fed.return_value = []

        collector = MetricsCollector()
        output = collector.format_prometheus()

        assert "valence_beliefs_total 42" in output

    @patch("valence.server.metrics.MetricsCollector._collect_database_metrics")
    @patch("valence.server.metrics.MetricsCollector._collect_federation_metrics")
    def test_format_prometheus_includes_federation_metrics(self, mock_fed, mock_db):
        """Prometheus output includes federation metrics."""
        mock_db.return_value = []
        mock_fed.return_value = [
            "",
            "# HELP valence_federation_peers_total Number of peers",
            "# TYPE valence_federation_peers_total gauge",
            "valence_federation_peers_total 5",
        ]

        collector = MetricsCollector()
        output = collector.format_prometheus()

        assert "valence_federation_peers_total 5" in output


class TestDatabaseMetrics:
    """Tests for database metrics collection."""

    @patch("valence.core.db.DatabaseStats")
    def test_collect_database_metrics_success(self, mock_stats_class):
        """Database metrics are collected when available."""
        mock_stats = MagicMock()
        mock_stats.to_dict.return_value = {
            "beliefs": 100,
            "entities": 50,
            "sessions": 10,
            "exchanges": 200,
            "patterns": 5,
            "tensions": 2,
        }
        mock_stats_class.collect.return_value = mock_stats

        collector = MetricsCollector()
        lines = collector._collect_database_metrics()

        assert any("valence_beliefs_total 100" in line for line in lines)
        assert any("valence_entities_total 50" in line for line in lines)

    def test_collect_database_metrics_error(self):
        """Database errors are handled gracefully."""
        collector = MetricsCollector()

        # Mock the import to raise an exception
        with patch("valence.core.db.DatabaseStats.collect", side_effect=Exception("DB error")):
            lines = collector._collect_database_metrics()

        # Should return something (either metrics or error comment)
        assert len(lines) > 0


class TestFederationMetrics:
    """Tests for federation metrics collection."""

    @patch("valence.federation.peer_sync.get_trust_registry")
    def test_collect_federation_metrics_success(self, mock_get_registry):
        """Federation metrics are collected when available."""
        mock_peer1 = MagicMock()
        mock_peer1.trust_level = 0.9
        mock_peer1.beliefs_received = 10
        mock_peer1.beliefs_sent = 5

        mock_peer2 = MagicMock()
        mock_peer2.trust_level = 0.6
        mock_peer2.beliefs_received = 20
        mock_peer2.beliefs_sent = 15

        mock_registry = MagicMock()
        mock_registry.list_peers.return_value = [mock_peer1, mock_peer2]
        mock_get_registry.return_value = mock_registry

        collector = MetricsCollector()
        lines = collector._collect_federation_metrics()

        assert any("valence_federation_peers_total 2" in line for line in lines)
        assert any('level="high"' in line and "1" in line for line in lines)
        assert any('level="medium"' in line and "1" in line for line in lines)
        assert any("valence_federation_beliefs_received_total 30" in line for line in lines)
        assert any("valence_federation_beliefs_sent_total 20" in line for line in lines)

    @patch("valence.federation.peer_sync.get_trust_registry")
    def test_collect_federation_metrics_no_peers(self, mock_get_registry):
        """Empty registry returns zero metrics."""
        mock_registry = MagicMock()
        mock_registry.list_peers.return_value = []
        mock_get_registry.return_value = mock_registry

        collector = MetricsCollector()
        lines = collector._collect_federation_metrics()

        assert any("valence_federation_peers_total 0" in line for line in lines)


@pytest.mark.asyncio
async def test_metrics_endpoint():
    """Test the metrics HTTP endpoint."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    app = Starlette(routes=[Route("/metrics", metrics_endpoint)])
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "valence_active_connections" in response.text


@pytest.mark.asyncio
async def test_metrics_endpoint_content_type():
    """Metrics endpoint returns correct Prometheus content type."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    app = Starlette(routes=[Route("/metrics", metrics_endpoint)])
    client = TestClient(app)

    response = client.get("/metrics")

    # Prometheus expects this specific content type
    assert "text/plain" in response.headers["content-type"]
    assert "version=0.0.4" in response.headers["content-type"]
