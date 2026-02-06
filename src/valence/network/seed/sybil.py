"""
Sybil resistance mechanisms for seed node.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from valence.network.seed.reputation import RateLimiter, ReputationManager

if TYPE_CHECKING:
    from valence.network.seed.config import SeedConfig

logger = logging.getLogger(__name__)


class CorrelationDetector:
    """
    Detects correlated router behavior patterns (Issue #117).

    Sybil attackers often control multiple routers that exhibit:
    - Synchronized heartbeats (within seconds of each other)
    - Similar endpoint patterns (same IP ranges, sequential ports)
    - Correlated up/down patterns
    - Matching capacity metrics

    This detector identifies suspicious correlations and flags routers.
    """

    def __init__(self, config: SeedConfig, reputation_manager: ReputationManager):
        self.config = config
        self.reputation = reputation_manager

        # Track heartbeat timestamps per router for correlation analysis
        self._heartbeat_times: dict[str, list[float]] = defaultdict(list)
        self._max_heartbeat_history = 20  # Keep last N heartbeats per router

        # Track endpoint patterns
        self._endpoint_patterns: dict[str, list[str]] = {}

        # Correlation groups (routers suspected of being controlled together)
        self._correlation_groups: list[set] = []

    def record_heartbeat(self, router_id: str, timestamp: float) -> None:
        """Record heartbeat timestamp for correlation analysis."""
        if not self.config.correlation_detection_enabled:
            return

        times = self._heartbeat_times[router_id]
        times.append(timestamp)

        # Keep only recent heartbeats
        if len(times) > self._max_heartbeat_history:
            self._heartbeat_times[router_id] = times[-self._max_heartbeat_history :]

    def record_endpoint(self, router_id: str, endpoints: list[str]) -> None:
        """Record router endpoints for pattern analysis."""
        if not self.config.correlation_detection_enabled:
            return

        self._endpoint_patterns[router_id] = endpoints

    def check_heartbeat_correlation(self, router_id: str, timestamp: float) -> list[str]:
        """
        Check if this heartbeat correlates suspiciously with other routers.

        Returns list of router IDs that heartbeat within the suspicious window.
        """
        if not self.config.correlation_detection_enabled:
            return []

        window = self.config.correlation_heartbeat_window_seconds
        correlated = []

        for other_id, times in self._heartbeat_times.items():
            if other_id == router_id:
                continue

            # Check if any recent heartbeat from other router is within window
            for t in times[-5:]:  # Check last 5 heartbeats
                if abs(timestamp - t) < window:
                    correlated.append(other_id)
                    break

        return correlated

    def check_endpoint_similarity(self, router_id: str, endpoints: list[str]) -> list[tuple]:
        """
        Check for suspicious endpoint similarity with existing routers.

        Returns list of (router_id, similarity_score) tuples for suspicious matches.
        """
        if not self.config.correlation_detection_enabled:
            return []

        suspicious = []

        for other_id, other_endpoints in self._endpoint_patterns.items():
            if other_id == router_id:
                continue

            similarity = self._compute_endpoint_similarity(endpoints, other_endpoints)
            if similarity >= self.config.correlation_endpoint_similarity_threshold:
                suspicious.append((other_id, similarity))

        return suspicious

    def _compute_endpoint_similarity(self, endpoints1: list[str], endpoints2: list[str]) -> float:
        """
        Compute similarity score between two endpoint lists.

        Checks for:
        - Same IP addresses
        - Same /24 subnet
        - Sequential port numbers
        """
        if not endpoints1 or not endpoints2:
            return 0.0

        score = 0.0
        comparisons = 0

        for ep1 in endpoints1:
            for ep2 in endpoints2:
                comparisons += 1

                # Parse endpoints
                try:
                    host1, port1_str = ep1.rsplit(":", 1) if ":" in ep1 else (ep1, "8471")
                    host2, port2_str = ep2.rsplit(":", 1) if ":" in ep2 else (ep2, "8471")
                    port1_int, port2_int = int(port1_str), int(port2_str)
                except (ValueError, AttributeError):
                    continue

                # Same IP = very suspicious
                if host1 == host2:
                    score += 1.0
                else:
                    # Same /24 subnet
                    parts1 = host1.split(".")
                    parts2 = host2.split(".")
                    if len(parts1) == 4 and len(parts2) == 4:
                        if parts1[:3] == parts2[:3]:
                            score += 0.7
                        elif parts1[:2] == parts2[:2]:
                            score += 0.3

                # Sequential ports on similar IPs
                if abs(port1_int - port2_int) <= 10:
                    score += 0.2

        return score / comparisons if comparisons > 0 else 0.0

    def analyze_and_flag(self, router_id: str, timestamp: float, endpoints: list[str]) -> list[str]:
        """
        Analyze router behavior and flag if suspicious.

        Returns list of flags applied.
        """
        if not self.config.correlation_detection_enabled:
            return []

        flags = []

        # Check heartbeat correlation
        correlated_heartbeats = self.check_heartbeat_correlation(router_id, timestamp)
        if len(correlated_heartbeats) >= self.config.correlation_min_suspicious_events:
            flags.append("correlated_heartbeats")
            # Apply penalty to this router
            self.reputation.apply_penalty(
                router_id,
                self.config.correlation_penalty_score,
                "correlated_heartbeats",
            )
            logger.warning(f"Correlated heartbeats detected: {router_id[:20]}... correlates with {len(correlated_heartbeats)} other routers")

        # Check endpoint similarity
        similar_endpoints = self.check_endpoint_similarity(router_id, endpoints)
        if similar_endpoints:
            flags.append("similar_endpoints")
            for other_id, similarity in similar_endpoints:
                logger.warning(f"Similar endpoints detected: {router_id[:20]}... and {other_id[:20]}... (similarity={similarity:.2f})")

            # Apply penalty if highly suspicious
            if any(s >= 0.9 for _, s in similar_endpoints):
                self.reputation.apply_penalty(
                    router_id,
                    self.config.correlation_penalty_score,
                    "highly_similar_endpoints",
                )

        # Record for future analysis
        self.record_heartbeat(router_id, timestamp)
        self.record_endpoint(router_id, endpoints)

        return flags

    def remove_router(self, router_id: str) -> None:
        """Remove router from correlation tracking."""
        self._heartbeat_times.pop(router_id, None)
        self._endpoint_patterns.pop(router_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Get correlation detector statistics."""
        return {
            "tracked_routers": len(self._heartbeat_times),
            "endpoint_patterns": len(self._endpoint_patterns),
            "correlation_groups": len(self._correlation_groups),
        }


class SybilResistance:
    """
    Unified Sybil resistance manager (Issue #117).

    Coordinates:
    - Rate limiting
    - Reputation management
    - Correlation detection
    - Adaptive PoW difficulty
    """

    def __init__(self, config: SeedConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.reputation = ReputationManager(config)
        self.correlation = CorrelationDetector(config, self.reputation)

    def check_registration(self, router_id: str, source_ip: str, endpoints: list[str]) -> tuple[bool, str | None]:
        """
        Check if a registration should be allowed.

        Args:
            router_id: The router attempting to register
            source_ip: Source IP of the registration request
            endpoints: Advertised endpoints

        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        # Check rate limits
        allowed, reason = self.rate_limiter.check_rate_limit(source_ip)
        if not allowed:
            return False, reason

        # Check endpoint similarity with existing routers (pre-registration check)
        similar = self.correlation.check_endpoint_similarity(router_id, endpoints)
        if any(s >= 0.95 for _, s in similar):
            return False, "endpoint_collision:nearly_identical_endpoints_exist"

        return True, None

    def on_registration_success(self, router_id: str, source_ip: str, endpoints: list[str]) -> None:
        """Called when a registration succeeds."""
        self.rate_limiter.record_registration(router_id, source_ip, success=True)
        self.reputation.register_router(router_id)
        self.correlation.record_endpoint(router_id, endpoints)

    def on_registration_failure(self, router_id: str, source_ip: str) -> None:
        """Called when a registration fails."""
        self.rate_limiter.record_registration(router_id, source_ip, success=False)

    def on_heartbeat(self, router_id: str, timestamp: float, endpoints: list[str]) -> list[str]:
        """
        Process a heartbeat for Sybil resistance analysis.

        Returns list of suspicious behavior flags.
        """
        self.reputation.record_heartbeat(router_id)
        flags = self.correlation.analyze_and_flag(router_id, timestamp, endpoints)
        return flags

    def on_missed_heartbeat(self, router_id: str) -> None:
        """Called when a router misses a heartbeat."""
        self.reputation.record_missed_heartbeat(router_id)

    def on_router_removed(self, router_id: str) -> None:
        """Called when a router is removed from the registry."""
        self.reputation.remove_router(router_id)
        self.correlation.remove_router(router_id)

    def get_adaptive_pow_difficulty(self, base_difficulty: int, source_ip: str) -> int:
        """
        Get PoW difficulty adjusted for current network conditions.

        Increases difficulty when registration rate is high (potential attack).
        """
        if not self.config.adaptive_pow_enabled:
            return base_difficulty

        rate = self.rate_limiter.get_registration_rate()

        if rate > self.config.adaptive_pow_threshold_per_hour:
            # Scale difficulty based on how much we exceed threshold
            excess_factor = rate / self.config.adaptive_pow_threshold_per_hour
            additional_bits = int(excess_factor * self.config.adaptive_pow_difficulty_step)
            adjusted = min(
                base_difficulty + additional_bits,
                self.config.adaptive_pow_max_difficulty,
            )

            if adjusted > base_difficulty:
                logger.info(f"Adaptive PoW: difficulty increased {base_difficulty} → {adjusted} (rate={rate}/hour)")

            return adjusted

        return base_difficulty

    def is_trusted_for_discovery(self, router_id: str) -> bool:
        """Check if router should be included in discovery based on reputation."""
        return self.reputation.is_trusted_for_discovery(router_id)

    def get_trust_factor(self, router_id: str) -> float:
        """Get trust factor for router scoring."""
        return self.reputation.get_trust_factor(router_id)

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive Sybil resistance statistics."""
        return {
            "rate_limiter": self.rate_limiter.get_stats(),
            "reputation": self.reputation.get_stats(),
            "correlation": self.correlation.get_stats(),
            "adaptive_pow": {
                "enabled": self.config.adaptive_pow_enabled,
                "threshold_per_hour": self.config.adaptive_pow_threshold_per_hour,
                "current_rate": self.rate_limiter.get_registration_rate(),
            },
        }
