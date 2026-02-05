"""
Sybil resistance for seed nodes (Issue #117).

Implements rate limiting, reputation management, and correlation detection
to protect against Sybil attacks.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import SeedConfig

logger = logging.getLogger(__name__)


@dataclass
class RegistrationEvent:
    """Record of a registration attempt for rate limiting."""
    router_id: str
    source_ip: str
    subnet: str  # /24 subnet
    timestamp: float
    success: bool


@dataclass 
class ReputationRecord:
    """
    Reputation state for a router.
    
    New routers start with a low trust score that increases over time
    with consistent good behavior (heartbeats, uptime).
    """
    router_id: str
    score: float  # 0.0 to 1.0
    registered_at: float
    last_heartbeat: float
    heartbeat_count: int
    missed_heartbeats: int
    flags: List[str] = field(default_factory=list)  # Suspicious behavior flags
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "router_id": self.router_id,
            "score": round(self.score, 3),
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
            "heartbeat_count": self.heartbeat_count,
            "missed_heartbeats": self.missed_heartbeats,
            "flags": self.flags.copy(),
        }


class RateLimiter:
    """
    Rate limiter for registration requests (Issue #117).
    
    Implements:
    - Per-IP rate limiting
    - Per-subnet (/24) rate limiting  
    - Cooldown period between registrations from same IP
    - Sliding window tracking
    """
    
    def __init__(self, config: "SeedConfig"):
        self.config = config
        self._registration_events: List[RegistrationEvent] = []
        self._last_registration_by_ip: Dict[str, float] = {}
    
    def _get_subnet(self, ip: str) -> str:
        """Extract /24 subnet from IP address."""
        try:
            parts = ip.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        except Exception:
            pass
        return ip  # Return original if not a valid IPv4
    
    def _cleanup_old_events(self, now: float) -> None:
        """Remove events outside the rate limit window."""
        cutoff = now - self.config.rate_limit_window_seconds
        self._registration_events = [
            e for e in self._registration_events if e.timestamp > cutoff
        ]
    
    def check_rate_limit(self, source_ip: str) -> tuple[bool, Optional[str]]:
        """
        Check if a registration from this IP is allowed.
        
        Args:
            source_ip: The IP address attempting to register
            
        Returns:
            Tuple of (allowed: bool, reason: Optional[str])
        """
        if not self.config.rate_limit_enabled:
            return True, None
        
        now = time.time()
        self._cleanup_old_events(now)
        subnet = self._get_subnet(source_ip)
        
        # Check cooldown since last registration from this IP
        last_reg = self._last_registration_by_ip.get(source_ip, 0)
        if now - last_reg < self.config.rate_limit_cooldown_seconds:
            remaining = int(self.config.rate_limit_cooldown_seconds - (now - last_reg))
            return False, f"cooldown_active:retry_after={remaining}s"
        
        # Count recent registrations from this IP
        ip_count = sum(1 for e in self._registration_events if e.source_ip == source_ip)
        if ip_count >= self.config.rate_limit_max_per_ip:
            return False, f"ip_limit_exceeded:max={self.config.rate_limit_max_per_ip}/hour"
        
        # Count recent registrations from this subnet
        subnet_count = sum(1 for e in self._registration_events if e.subnet == subnet)
        if subnet_count >= self.config.rate_limit_max_per_subnet:
            return False, f"subnet_limit_exceeded:max={self.config.rate_limit_max_per_subnet}/hour"
        
        return True, None
    
    def record_registration(self, router_id: str, source_ip: str, success: bool) -> None:
        """Record a registration attempt."""
        now = time.time()
        subnet = self._get_subnet(source_ip)
        
        self._registration_events.append(RegistrationEvent(
            router_id=router_id,
            source_ip=source_ip,
            subnet=subnet,
            timestamp=now,
            success=success,
        ))
        
        if success:
            self._last_registration_by_ip[source_ip] = now
    
    def get_registration_rate(self) -> float:
        """Get registrations per hour (for adaptive PoW)."""
        now = time.time()
        self._cleanup_old_events(now)
        
        # Count successful registrations in last hour
        hour_ago = now - 3600
        recent = [e for e in self._registration_events if e.timestamp > hour_ago and e.success]
        return len(recent)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        now = time.time()
        self._cleanup_old_events(now)
        
        return {
            "total_events": len(self._registration_events),
            "unique_ips": len(set(e.source_ip for e in self._registration_events)),
            "unique_subnets": len(set(e.subnet for e in self._registration_events)),
            "registrations_per_hour": self.get_registration_rate(),
            "window_seconds": self.config.rate_limit_window_seconds,
        }


class ReputationManager:
    """
    Reputation system for routers (Issue #117).
    
    New routers start with a low trust score that increases over time.
    Good behavior (consistent heartbeats) improves score.
    Bad behavior (missed heartbeats, suspicious patterns) decreases score.
    """
    
    def __init__(self, config: "SeedConfig"):
        self.config = config
        self._reputation: Dict[str, ReputationRecord] = {}
    
    def register_router(self, router_id: str) -> ReputationRecord:
        """Create initial reputation for a new router."""
        now = time.time()
        record = ReputationRecord(
            router_id=router_id,
            score=self.config.reputation_initial_score,
            registered_at=now,
            last_heartbeat=now,
            heartbeat_count=0,
            missed_heartbeats=0,
            flags=[],
        )
        self._reputation[router_id] = record
        logger.debug(
            f"New router reputation: {router_id[:20]}... "
            f"initial_score={record.score}"
        )
        return record
    
    def get_reputation(self, router_id: str) -> Optional[ReputationRecord]:
        """Get reputation record for a router."""
        return self._reputation.get(router_id)
    
    def record_heartbeat(self, router_id: str) -> Optional[ReputationRecord]:
        """
        Record a successful heartbeat and boost reputation.
        
        Also applies time-based reputation decay (increasing trust over time).
        """
        record = self._reputation.get(router_id)
        if not record:
            return None
        
        now = time.time()
        record.last_heartbeat = now
        record.heartbeat_count += 1
        
        # Boost score for successful heartbeat
        record.score = min(
            self.config.reputation_max_score,
            record.score + self.config.reputation_boost_per_heartbeat
        )
        
        # Apply time-based trust increase (decay toward full trust)
        hours_since_registration = (now - record.registered_at) / 3600
        if hours_since_registration > 0:
            time_factor = min(1.0, hours_since_registration / self.config.reputation_decay_period_hours)
            # Blend toward max score based on time
            target_score = self.config.reputation_max_score
            record.score = record.score + (target_score - record.score) * time_factor * 0.01
        
        return record
    
    def record_missed_heartbeat(self, router_id: str) -> Optional[ReputationRecord]:
        """Record a missed heartbeat and penalize reputation."""
        record = self._reputation.get(router_id)
        if not record:
            return None
        
        record.missed_heartbeats += 1
        record.score = max(
            0.0,
            record.score - self.config.reputation_penalty_missed_heartbeat
        )
        
        logger.debug(
            f"Missed heartbeat: {router_id[:20]}... "
            f"score={record.score:.3f}, missed={record.missed_heartbeats}"
        )
        return record
    
    def apply_penalty(self, router_id: str, penalty: float, reason: str) -> Optional[ReputationRecord]:
        """Apply a reputation penalty with a reason flag."""
        record = self._reputation.get(router_id)
        if not record:
            return None
        
        record.score = max(0.0, record.score - penalty)
        if reason not in record.flags:
            record.flags.append(reason)
        
        logger.warning(
            f"Reputation penalty: {router_id[:20]}... "
            f"penalty={penalty}, reason={reason}, new_score={record.score:.3f}"
        )
        return record
    
    def is_trusted_for_discovery(self, router_id: str) -> bool:
        """Check if router has enough reputation to be included in discovery."""
        if not self.config.reputation_enabled:
            return True
        
        record = self._reputation.get(router_id)
        if not record:
            return True  # Unknown routers default to included
        
        return record.score >= self.config.reputation_min_score_for_discovery
    
    def get_trust_factor(self, router_id: str) -> float:
        """
        Get a trust factor (0.0-1.0) for scoring routers in discovery.
        
        Higher reputation = higher factor = preferred in selection.
        """
        if not self.config.reputation_enabled:
            return 1.0
        
        record = self._reputation.get(router_id)
        if not record:
            return self.config.reputation_initial_score
        
        return record.score
    
    def remove_router(self, router_id: str) -> None:
        """Remove reputation record for a router."""
        self._reputation.pop(router_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reputation system statistics."""
        if not self._reputation:
            return {
                "total_routers": 0,
                "avg_score": 0.0,
                "below_threshold": 0,
                "flagged": 0,
            }
        
        scores = [r.score for r in self._reputation.values()]
        return {
            "total_routers": len(self._reputation),
            "avg_score": round(sum(scores) / len(scores), 3),
            "min_score": round(min(scores), 3),
            "max_score": round(max(scores), 3),
            "below_threshold": sum(
                1 for r in self._reputation.values()
                if r.score < self.config.reputation_min_score_for_discovery
            ),
            "flagged": sum(1 for r in self._reputation.values() if r.flags),
        }


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
    
    def __init__(self, config: "SeedConfig", reputation_manager: ReputationManager):
        self.config = config
        self.reputation = reputation_manager
        
        # Track heartbeat timestamps per router for correlation analysis
        self._heartbeat_times: Dict[str, List[float]] = defaultdict(list)
        self._max_heartbeat_history = 20  # Keep last N heartbeats per router
        
        # Track endpoint patterns
        self._endpoint_patterns: Dict[str, List[str]] = {}
        
        # Correlation groups (routers suspected of being controlled together)
        self._correlation_groups: List[set] = []
    
    def record_heartbeat(self, router_id: str, timestamp: float) -> None:
        """Record heartbeat timestamp for correlation analysis."""
        if not self.config.correlation_detection_enabled:
            return
        
        times = self._heartbeat_times[router_id]
        times.append(timestamp)
        
        # Keep only recent heartbeats
        if len(times) > self._max_heartbeat_history:
            self._heartbeat_times[router_id] = times[-self._max_heartbeat_history:]
    
    def record_endpoint(self, router_id: str, endpoints: List[str]) -> None:
        """Record router endpoints for pattern analysis."""
        if not self.config.correlation_detection_enabled:
            return
        
        self._endpoint_patterns[router_id] = endpoints
    
    def check_heartbeat_correlation(self, router_id: str, timestamp: float) -> List[str]:
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
    
    def check_endpoint_similarity(self, router_id: str, endpoints: List[str]) -> List[tuple]:
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
    
    def _compute_endpoint_similarity(self, endpoints1: List[str], endpoints2: List[str]) -> float:
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
    
    def analyze_and_flag(self, router_id: str, timestamp: float, endpoints: List[str]) -> List[str]:
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
                "correlated_heartbeats"
            )
            logger.warning(
                f"Correlated heartbeats detected: {router_id[:20]}... "
                f"correlates with {len(correlated_heartbeats)} other routers"
            )
        
        # Check endpoint similarity
        similar_endpoints = self.check_endpoint_similarity(router_id, endpoints)
        if similar_endpoints:
            flags.append("similar_endpoints")
            for other_id, similarity in similar_endpoints:
                logger.warning(
                    f"Similar endpoints detected: {router_id[:20]}... and "
                    f"{other_id[:20]}... (similarity={similarity:.2f})"
                )
            
            # Apply penalty if highly suspicious
            if any(s >= 0.9 for _, s in similar_endpoints):
                self.reputation.apply_penalty(
                    router_id,
                    self.config.correlation_penalty_score,
                    "highly_similar_endpoints"
                )
        
        # Record for future analysis
        self.record_heartbeat(router_id, timestamp)
        self.record_endpoint(router_id, endpoints)
        
        return flags
    
    def remove_router(self, router_id: str) -> None:
        """Remove router from correlation tracking."""
        self._heartbeat_times.pop(router_id, None)
        self._endpoint_patterns.pop(router_id, None)
    
    def get_stats(self) -> Dict[str, Any]:
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
    
    def __init__(self, config: "SeedConfig"):
        self.config = config
        self.rate_limiter = RateLimiter(config)
        self.reputation = ReputationManager(config)
        self.correlation = CorrelationDetector(config, self.reputation)
    
    def check_registration(self, router_id: str, source_ip: str, endpoints: List[str]) -> tuple[bool, Optional[str]]:
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
    
    def on_registration_success(self, router_id: str, source_ip: str, endpoints: List[str]) -> None:
        """Called when a registration succeeds."""
        self.rate_limiter.record_registration(router_id, source_ip, success=True)
        self.reputation.register_router(router_id)
        self.correlation.record_endpoint(router_id, endpoints)
    
    def on_registration_failure(self, router_id: str, source_ip: str) -> None:
        """Called when a registration fails."""
        self.rate_limiter.record_registration(router_id, source_ip, success=False)
    
    def on_heartbeat(self, router_id: str, timestamp: float, endpoints: List[str]) -> List[str]:
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
                self.config.adaptive_pow_max_difficulty
            )
            
            if adjusted > base_difficulty:
                logger.info(
                    f"Adaptive PoW: difficulty increased {base_difficulty} → {adjusted} "
                    f"(rate={rate}/hour)"
                )
            
            return adjusted
        
        return base_difficulty
    
    def is_trusted_for_discovery(self, router_id: str) -> bool:
        """Check if router should be included in discovery based on reputation."""
        return self.reputation.is_trusted_for_discovery(router_id)
    
    def get_trust_factor(self, router_id: str) -> float:
        """Get trust factor for router scoring."""
        return self.reputation.get_trust_factor(router_id)
    
    def get_stats(self) -> Dict[str, Any]:
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
