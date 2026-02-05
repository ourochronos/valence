"""Prometheus metrics for Valence server.

Provides /metrics endpoint with Prometheus text format.
Implements simple text format without prometheus_client dependency.

Metrics exported:
- valence_http_request_duration_seconds: Request latency histogram
- valence_http_requests_total: Request count by endpoint/status
- valence_active_connections: Currently active connections
- valence_beliefs_total: Total belief count in database
- valence_federation_peers_total: Number of federation peers
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

logger = logging.getLogger(__name__)

# Histogram bucket boundaries (in seconds)
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class HistogramData:
    """Histogram metric data."""

    buckets: dict[float, int] = field(default_factory=lambda: defaultdict(int))
    sum: float = 0.0
    count: int = 0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.sum += value
        self.count += 1
        for bucket in LATENCY_BUCKETS:
            if value <= bucket:
                self.buckets[bucket] += 1


class MetricsCollector:
    """Thread-safe metrics collector.

    Collects request metrics and provides Prometheus text format output.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Request metrics: {(method, path, status): count}
        self._request_counts: dict[tuple[str, str, int], int] = defaultdict(int)

        # Latency histogram: {(method, path): HistogramData}
        self._latency_histograms: dict[tuple[str, str], HistogramData] = defaultdict(HistogramData)

        # Active connections gauge
        self._active_connections: int = 0

    def record_request(
        self, method: str, path: str, status_code: int, duration_seconds: float
    ) -> None:
        """Record a completed request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path (normalized)
            status_code: HTTP response status code
            duration_seconds: Request duration in seconds
        """
        # Normalize path to avoid cardinality explosion
        normalized_path = self._normalize_path(path)

        with self._lock:
            self._request_counts[(method, normalized_path, status_code)] += 1
            self._latency_histograms[(method, normalized_path)].observe(duration_seconds)

    def _normalize_path(self, path: str) -> str:
        """Normalize path to prevent label cardinality explosion.

        Replaces UUID-like segments and numeric IDs with placeholders.
        """
        parts = path.split("/")
        normalized = []
        for part in parts:
            # Replace UUIDs
            if len(part) == 36 and part.count("-") == 4:
                normalized.append("{id}")
            # Replace numeric IDs
            elif part.isdigit():
                normalized.append("{id}")
            else:
                normalized.append(part)
        return "/".join(normalized)

    def increment_connections(self) -> None:
        """Increment active connection count."""
        with self._lock:
            self._active_connections += 1

    def decrement_connections(self) -> None:
        """Decrement active connection count."""
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    def get_active_connections(self) -> int:
        """Get current active connection count."""
        with self._lock:
            return self._active_connections

    def format_prometheus(self) -> str:
        """Format all metrics in Prometheus text format.

        Returns:
            Prometheus text format string
        """
        lines: list[str] = []

        with self._lock:
            # Request count metric
            lines.append("# HELP valence_http_requests_total Total HTTP requests")
            lines.append("# TYPE valence_http_requests_total counter")
            for (method, path, status), count in sorted(self._request_counts.items()):
                metric = "valence_http_requests_total"
                labels = f'method="{method}",path="{path}",status="{status}"'
                lines.append(f"{metric}{{{labels}}} {count}")

            # Latency histogram
            lines.append("")
            lines.append("# HELP valence_http_request_duration_seconds HTTP request latency")
            lines.append("# TYPE valence_http_request_duration_seconds histogram")
            for (method, path), histogram in sorted(self._latency_histograms.items()):
                base_labels = f'method="{method}",path="{path}"'
                # Cumulative bucket counts
                cumulative = 0
                for bucket in LATENCY_BUCKETS:
                    cumulative += histogram.buckets.get(bucket, 0)
                    bucket_metric = "valence_http_request_duration_seconds_bucket"
                    lines.append(f'{bucket_metric}{{{base_labels},le="{bucket}"}} {cumulative}')
                # +Inf bucket
                bucket_metric = "valence_http_request_duration_seconds_bucket"
                lines.append(f'{bucket_metric}{{{base_labels},le="+Inf"}} {histogram.count}')
                # Sum and count
                sum_metric = "valence_http_request_duration_seconds_sum"
                lines.append(f"{sum_metric}{{{base_labels}}} {histogram.sum:.6f}")
                count_metric = "valence_http_request_duration_seconds_count"
                lines.append(f"{count_metric}{{{base_labels}}} {histogram.count}")

            # Active connections gauge
            lines.append("")
            lines.append("# HELP valence_active_connections Currently active HTTP connections")
            lines.append("# TYPE valence_active_connections gauge")
            lines.append(f"valence_active_connections {self._active_connections}")

        # Database metrics (collected on demand)
        db_metrics = self._collect_database_metrics()
        lines.extend(db_metrics)

        # Federation metrics
        fed_metrics = self._collect_federation_metrics()
        lines.extend(fed_metrics)

        lines.append("")
        return "\n".join(lines)

    def _collect_database_metrics(self) -> list[str]:
        """Collect database-related metrics."""
        lines: list[str] = []

        try:
            from ..core.db import DatabaseStats

            stats = DatabaseStats.collect()
            stats_dict = stats.to_dict()

            lines.append("")
            lines.append("# HELP valence_beliefs_total Total beliefs in database")
            lines.append("# TYPE valence_beliefs_total gauge")
            lines.append(f"valence_beliefs_total {stats_dict.get('beliefs', 0)}")

            lines.append("")
            lines.append("# HELP valence_entities_total Total entities in database")
            lines.append("# TYPE valence_entities_total gauge")
            lines.append(f"valence_entities_total {stats_dict.get('entities', 0)}")

            lines.append("")
            lines.append("# HELP valence_sessions_total Total VKB sessions")
            lines.append("# TYPE valence_sessions_total gauge")
            lines.append(f"valence_sessions_total {stats_dict.get('sessions', 0)}")

            lines.append("")
            lines.append("# HELP valence_exchanges_total Total VKB exchanges")
            lines.append("# TYPE valence_exchanges_total gauge")
            lines.append(f"valence_exchanges_total {stats_dict.get('exchanges', 0)}")

            lines.append("")
            lines.append("# HELP valence_patterns_total Total VKB patterns")
            lines.append("# TYPE valence_patterns_total gauge")
            lines.append(f"valence_patterns_total {stats_dict.get('patterns', 0)}")

            lines.append("")
            lines.append("# HELP valence_tensions_total Total belief tensions")
            lines.append("# TYPE valence_tensions_total gauge")
            lines.append(f"valence_tensions_total {stats_dict.get('tensions', 0)}")

        except Exception as e:
            logger.debug(f"Could not collect database metrics: {e}")
            # Return placeholder when DB unavailable
            lines.append("")
            lines.append("# HELP valence_beliefs_total Total beliefs in database")
            lines.append("# TYPE valence_beliefs_total gauge")
            lines.append("# valence_beliefs_total unavailable (database error)")

        return lines

    def _collect_federation_metrics(self) -> list[str]:
        """Collect federation-related metrics."""
        lines: list[str] = []

        try:
            from ..federation.peer_sync import get_trust_registry

            registry = get_trust_registry()
            peers = registry.list_peers()

            lines.append("")
            lines.append("# HELP valence_federation_peers_total Trusted federation peers")
            lines.append("# TYPE valence_federation_peers_total gauge")
            lines.append(f"valence_federation_peers_total {len(peers)}")

            # Breakdown by trust level ranges
            high_trust = sum(1 for p in peers if p.trust_level >= 0.8)
            medium_trust = sum(1 for p in peers if 0.5 <= p.trust_level < 0.8)
            low_trust = sum(1 for p in peers if p.trust_level < 0.5)

            lines.append("")
            lines.append("# HELP valence_federation_peers_by_trust Peers by trust level")
            lines.append("# TYPE valence_federation_peers_by_trust gauge")
            lines.append(f'valence_federation_peers_by_trust{{level="high"}} {high_trust}')
            lines.append(f'valence_federation_peers_by_trust{{level="medium"}} {medium_trust}')
            lines.append(f'valence_federation_peers_by_trust{{level="low"}} {low_trust}')

            # Total beliefs exchanged with peers
            total_received = sum(p.beliefs_received for p in peers)
            total_sent = sum(p.beliefs_sent for p in peers)

            lines.append("")
            lines.append("# HELP valence_federation_beliefs_received_total Beliefs from peers")
            lines.append("# TYPE valence_federation_beliefs_received_total counter")
            lines.append(f"valence_federation_beliefs_received_total {total_received}")

            lines.append("")
            lines.append("# HELP valence_federation_beliefs_sent_total Beliefs sent to peers")
            lines.append("# TYPE valence_federation_beliefs_sent_total counter")
            lines.append(f"valence_federation_beliefs_sent_total {total_sent}")

        except Exception as e:
            logger.debug(f"Could not collect federation metrics: {e}")
            lines.append("")
            lines.append("# HELP valence_federation_peers_total Trusted federation peers")
            lines.append("# TYPE valence_federation_peers_total gauge")
            lines.append("valence_federation_peers_total 0")

        return lines


# Global metrics collector instance
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class MetricsMiddleware(BaseHTTPMiddleware):
    """Starlette middleware for collecting request metrics.

    Tracks request count, latency, and active connections.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and record metrics."""
        collector = get_metrics_collector()

        # Skip metrics endpoint to avoid recursion
        if request.url.path == "/metrics" or request.url.path == "/api/v1/metrics":
            return await call_next(request)

        collector.increment_connections()
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            duration = time.perf_counter() - start_time

            collector.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_seconds=duration,
            )

            return response
        except Exception:
            duration = time.perf_counter() - start_time
            collector.record_request(
                method=request.method,
                path=request.url.path,
                status_code=500,
                duration_seconds=duration,
            )
            raise
        finally:
            collector.decrement_connections()


async def metrics_endpoint(request: Request) -> PlainTextResponse:
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text exposition format.
    """
    collector = get_metrics_collector()
    metrics_text = collector.format_prometheus()

    return PlainTextResponse(
        content=metrics_text,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
