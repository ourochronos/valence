"""
Reputation and rate limiting for Sybil resistance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from valence.network.seed.config import SeedConfig

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
    flags: list[str] = field(default_factory=list)  # Suspicious behavior flags

    def to_dict(self) -> dict[str, Any]:
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

    def __init__(self, config: SeedConfig):
        self.config = config
        self._registration_events: list[RegistrationEvent] = []
        self._last_registration_by_ip: dict[str, float] = {}

    def _get_subnet(self, ip: str) -> str:
        """Extract /24 subnet from IP address."""
        try:
            parts = ip.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        except (ValueError, IndexError):
            pass
        return ip  # Return original if not a valid IPv4

    def _cleanup_old_events(self, now: float) -> None:
        """Remove events outside the rate limit window."""
        cutoff = now - self.config.rate_limit_window_seconds
        self._registration_events = [e for e in self._registration_events if e.timestamp > cutoff]

    def check_rate_limit(self, source_ip: str) -> tuple[bool, str | None]:
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
            return (
                False,
                f"ip_limit_exceeded:max={self.config.rate_limit_max_per_ip}/hour",
            )

        # Count recent registrations from this subnet
        subnet_count = sum(1 for e in self._registration_events if e.subnet == subnet)
        if subnet_count >= self.config.rate_limit_max_per_subnet:
            return (
                False,
                f"subnet_limit_exceeded:max={self.config.rate_limit_max_per_subnet}/hour",
            )

        return True, None

    def record_registration(self, router_id: str, source_ip: str, success: bool) -> None:
        """Record a registration attempt."""
        now = time.time()
        subnet = self._get_subnet(source_ip)

        self._registration_events.append(
            RegistrationEvent(
                router_id=router_id,
                source_ip=source_ip,
                subnet=subnet,
                timestamp=now,
                success=success,
            )
        )

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

    def get_stats(self) -> dict[str, Any]:
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

    def __init__(self, config: SeedConfig):
        self.config = config
        self._reputation: dict[str, ReputationRecord] = {}

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
        logger.debug(f"New router reputation: {router_id[:20]}... initial_score={record.score}")
        return record

    def get_reputation(self, router_id: str) -> ReputationRecord | None:
        """Get reputation record for a router."""
        return self._reputation.get(router_id)

    def record_heartbeat(self, router_id: str) -> ReputationRecord | None:
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
            record.score + self.config.reputation_boost_per_heartbeat,
        )

        # Apply time-based trust increase (decay toward full trust)
        hours_since_registration = (now - record.registered_at) / 3600
        if hours_since_registration > 0:
            time_factor = min(
                1.0,
                hours_since_registration / self.config.reputation_decay_period_hours,
            )
            # Blend toward max score based on time
            target_score = self.config.reputation_max_score
            record.score = record.score + (target_score - record.score) * time_factor * 0.01

        return record

    def record_missed_heartbeat(self, router_id: str) -> ReputationRecord | None:
        """Record a missed heartbeat and penalize reputation."""
        record = self._reputation.get(router_id)
        if not record:
            return None

        record.missed_heartbeats += 1
        record.score = max(0.0, record.score - self.config.reputation_penalty_missed_heartbeat)

        logger.debug(f"Missed heartbeat: {router_id[:20]}... score={record.score:.3f}, missed={record.missed_heartbeats}")
        return record

    def apply_penalty(self, router_id: str, penalty: float, reason: str) -> ReputationRecord | None:
        """Apply a reputation penalty with a reason flag."""
        record = self._reputation.get(router_id)
        if not record:
            return None

        record.score = max(0.0, record.score - penalty)
        if reason not in record.flags:
            record.flags.append(reason)

        logger.warning(f"Reputation penalty: {router_id[:20]}... penalty={penalty}, reason={reason}, new_score={record.score:.3f}")
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

    def get_stats(self) -> dict[str, Any]:
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
            "below_threshold": sum(1 for r in self._reputation.values() if r.score < self.config.reputation_min_score_for_discovery),
            "flagged": sum(1 for r in self._reputation.values() if r.flags),
        }
