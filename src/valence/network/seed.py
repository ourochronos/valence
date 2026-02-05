"""
Valence Seed Node - The "phone book" for router discovery.

Seed nodes maintain a registry of active routers and help new nodes
bootstrap into the network. They implement:

1. Router registration: Routers announce themselves to seed nodes
2. Discovery: Nodes request router lists based on preferences
3. Health tracking: Monitor router availability via heartbeats
4. IP diversity: Ensure returned routers span different networks

Protocol:
- POST /discover - Get a list of routers matching preferences
- POST /register - Register a router with the seed node
- POST /heartbeat - Periodic health check from routers
- GET /status - Seed node status (public)

Security:
- Routers must sign registration with their Ed25519 key
- Proof-of-work required for anti-Sybil protection
- Heartbeats include router signature for verification
- No sensitive data stored - just public routing info
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import aiohttp
from aiohttp import web
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

if TYPE_CHECKING:
    from valence.network.messages import SeedRevocation, SeedRevocationList

logger = logging.getLogger(__name__)


# =============================================================================
# HEALTH MONITORING
# =============================================================================


class HealthStatus(Enum):
    """Health status for monitored routers."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    REMOVED = "removed"


@dataclass
class HealthState:
    """Health state for a monitored router."""
    status: HealthStatus
    missed_heartbeats: int
    last_heartbeat: float
    last_probe: float
    probe_latency_ms: float
    warnings: List[str] = field(default_factory=list)
    recovery_time: Optional[float] = None  # Timestamp when router recovered from unhealthy
    previous_status: Optional[HealthStatus] = None  # Previous status before current
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize health state."""
        return {
            "status": self.status.value,
            "missed_heartbeats": self.missed_heartbeats,
            "last_heartbeat": self.last_heartbeat,
            "last_probe": self.last_probe,
            "probe_latency_ms": self.probe_latency_ms,
            "warnings": self.warnings.copy(),
            "recovery_time": self.recovery_time,
        }


class HealthMonitor:
    """
    Monitors router health for seed node.
    
    Implements:
    - Heartbeat tracking with miss counting
    - Status state machine: HEALTHY → WARNING → DEGRADED → UNHEALTHY → REMOVED
    - Active probing of router subsets
    - Integration with discovery to filter unhealthy routers
    """
    
    def __init__(self, seed: "SeedNode"):
        self.seed = seed
        self.health_states: Dict[str, HealthState] = {}
        self.heartbeat_interval = 300  # 5 minutes
        self.probe_interval = 900  # 15 minutes
        self.check_interval = 60  # 1 minute
        self.probe_sample_size = 10
        self.high_latency_threshold_ms = 1000.0
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self) -> None:
        """Start health monitoring loops."""
        if self._running:
            return
        
        self._running = True
        self._tasks = [
            asyncio.create_task(self._heartbeat_checker()),
            asyncio.create_task(self._active_prober()),
        ]
        logger.info("Health monitor started")
    
    async def stop(self) -> None:
        """Stop health monitoring loops."""
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks = []
        logger.info("Health monitor stopped")
    
    async def _heartbeat_checker(self) -> None:
        """Check for missed heartbeats every minute."""
        while self._running:
            await asyncio.sleep(self.check_interval)
            
            now = time.time()
            removed_routers: List[str] = []
            
            for router_id, state in list(self.health_states.items()):
                time_since_heartbeat = now - state.last_heartbeat
                missed = int(time_since_heartbeat / self.heartbeat_interval)
                
                if missed != state.missed_heartbeats:
                    old_status = state.status
                    old_missed = state.missed_heartbeats
                    state.missed_heartbeats = missed
                    state.status = self._compute_status(missed)
                    
                    if state.status != old_status:
                        logger.info(
                            f"Router {router_id[:20]}... health: "
                            f"{old_status.value} → {state.status.value} "
                            f"(missed={missed})"
                        )
                    
                    # Notify Sybil resistance of missed heartbeats (Issue #117)
                    new_misses = missed - old_missed
                    if new_misses > 0 and hasattr(self.seed, '_sybil_resistance') and self.seed._sybil_resistance:
                        for _ in range(new_misses):
                            self.seed.sybil_resistance.on_missed_heartbeat(router_id)
                    
                    if state.status == HealthStatus.REMOVED:
                        removed_routers.append(router_id)
            
            # Remove expired routers
            for router_id in removed_routers:
                self.seed.router_registry.pop(router_id, None)
                self.health_states.pop(router_id, None)
                # Notify Sybil resistance of router removal (Issue #117)
                if hasattr(self.seed, '_sybil_resistance') and self.seed._sybil_resistance:
                    self.seed.sybil_resistance.on_router_removed(router_id)
                logger.info(f"Removed router {router_id[:20]}... due to missed heartbeats")
    
    def _compute_status(self, missed: int) -> HealthStatus:
        """
        Compute health status from missed heartbeats.
        
        State machine:
        - 0 missed: HEALTHY
        - 1 missed (5 min): WARNING
        - 2 missed (10 min): DEGRADED
        - 3-5 missed (15-25 min): UNHEALTHY
        - 6+ missed (30+ min): REMOVED
        """
        if missed == 0:
            return HealthStatus.HEALTHY
        elif missed == 1:
            return HealthStatus.WARNING
        elif missed == 2:
            return HealthStatus.DEGRADED
        elif missed >= 6:
            return HealthStatus.REMOVED
        else:  # 3, 4, 5
            return HealthStatus.UNHEALTHY
    
    async def _active_prober(self) -> None:
        """Actively probe a subset of routers every probe_interval."""
        while self._running:
            await asyncio.sleep(self.probe_interval)
            
            # Probe subset of routers
            routers = list(self.seed.router_registry.values())
            if not routers:
                continue
            
            sample_size = min(self.probe_sample_size, len(routers))
            sample = random.sample(routers, sample_size)
            
            logger.debug(f"Probing {len(sample)} routers")
            
            for router in sample:
                await self._probe_router(router)
    
    async def _probe_router(self, router: "RouterRecord") -> None:
        """
        Probe router health endpoint.
        
        Records probe latency and warnings for high latency or failures.
        """
        if not router.endpoints:
            return
        
        endpoint = router.endpoints[0]
        
        try:
            # Parse endpoint
            if ":" in endpoint:
                host, port = endpoint.rsplit(":", 1)
            else:
                host = endpoint
                port = "8471"
            
            url = f"http://{host}:{port}/health"
            start = time.time()
            
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    latency_ms = (time.time() - start) * 1000
                    
                    if router.router_id in self.health_states:
                        state = self.health_states[router.router_id]
                        state.last_probe = time.time()
                        state.probe_latency_ms = latency_ms
                        
                        # Check for high latency
                        if latency_ms > self.high_latency_threshold_ms:
                            if "high_latency" not in state.warnings:
                                state.warnings.append("high_latency")
                                logger.debug(
                                    f"Router {router.router_id[:20]}... "
                                    f"high latency: {latency_ms:.1f}ms"
                                )
                        else:
                            # Remove high_latency warning if latency is now OK
                            if "high_latency" in state.warnings:
                                state.warnings.remove("high_latency")
                    
                    logger.debug(
                        f"Probe successful: {router.router_id[:20]}... "
                        f"latency={latency_ms:.1f}ms"
                    )
                    
        except asyncio.TimeoutError:
            if router.router_id in self.health_states:
                state = self.health_states[router.router_id]
                if "probe_timeout" not in state.warnings:
                    state.warnings.append("probe_timeout")
            logger.debug(f"Probe timeout: {router.router_id[:20]}...")
            
        except Exception as e:
            if router.router_id in self.health_states:
                state = self.health_states[router.router_id]
                if "probe_failed" not in state.warnings:
                    state.warnings.append("probe_failed")
            logger.debug(f"Probe failed: {router.router_id[:20]}... - {e}")
    
    def record_heartbeat(self, router_id: str, metrics: Optional[Dict[str, Any]] = None) -> HealthState:
        """
        Record a heartbeat from router.
        
        Creates health state if new, resets missed count and status to healthy.
        Tracks recovery time for thundering herd prevention.
        
        Args:
            router_id: The router's ID
            metrics: Optional metrics from heartbeat (unused, for future extension)
            
        Returns:
            The current HealthState for this router
        """
        now = time.time()
        
        if router_id not in self.health_states:
            self.health_states[router_id] = HealthState(
                status=HealthStatus.HEALTHY,
                missed_heartbeats=0,
                last_heartbeat=now,
                last_probe=0,
                probe_latency_ms=0,
                warnings=[],
                recovery_time=None,  # New router, no recovery
                previous_status=None,
            )
            logger.debug(f"New health state for router {router_id[:20]}...")
        else:
            state = self.health_states[router_id]
            old_status = state.status
            state.previous_status = old_status
            state.last_heartbeat = now
            state.missed_heartbeats = 0
            state.status = HealthStatus.HEALTHY
            # Clear transient warnings on heartbeat
            state.warnings = [w for w in state.warnings if w not in ["probe_timeout", "probe_failed"]]
            
            # Track recovery time for thundering herd prevention
            # If router was DEGRADED or UNHEALTHY and is now HEALTHY
            if old_status in (HealthStatus.DEGRADED, HealthStatus.UNHEALTHY):
                state.recovery_time = now
                logger.info(
                    f"Router {router_id[:20]}... recovered: "
                    f"{old_status.value} → healthy (recovery ramp started)"
                )
            elif old_status == HealthStatus.WARNING:
                # Minor recovery, no ramp needed
                logger.info(
                    f"Router {router_id[:20]}... recovered: "
                    f"{old_status.value} → healthy"
                )
            
            # Clear recovery time if router has been healthy long enough
            # (handled by _get_recovery_factor checking elapsed time)
        
        return self.health_states[router_id]
    
    def get_health_state(self, router_id: str) -> Optional[HealthState]:
        """Get health state for a router."""
        return self.health_states.get(router_id)
    
    def is_healthy_for_discovery(self, router_id: str) -> bool:
        """
        Check if router should be included in discovery results.
        
        Routers with HEALTHY or WARNING status are included.
        DEGRADED, UNHEALTHY, and REMOVED are excluded.
        """
        state = self.health_states.get(router_id)
        if state is None:
            # No health state = new router, assume healthy
            return True
        
        return state.status in (HealthStatus.HEALTHY, HealthStatus.WARNING)
    
    def get_stats(self) -> Dict[str, int]:
        """Get health monitoring statistics."""
        stats = {status.value: 0 for status in HealthStatus}
        for state in self.health_states.values():
            stats[state.status.value] += 1
        stats["total"] = len(self.health_states)
        return stats


# =============================================================================
# SYBIL RESISTANCE (Issue #117)
# =============================================================================


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


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class SeedConfig:
    """Configuration for seed node."""
    
    host: str = "0.0.0.0"
    port: int = 8470
    
    # Router health thresholds
    min_uptime_pct: float = 90.0  # Minimum uptime to be considered healthy
    max_stale_seconds: float = 600.0  # Max time since last heartbeat
    
    # Selection weights for load-aware balancing
    weight_health: float = 0.25
    weight_load: float = 0.35  # Primary load factor (connection ratio)
    weight_queue: float = 0.15  # Queue depth penalty
    weight_throughput: float = 0.10  # Message rate consideration
    weight_region: float = 0.10
    weight_random: float = 0.05
    
    # Thundering herd prevention
    recovery_ramp_duration: float = 300.0  # 5 min ramp-up period for recovered routers
    recovery_initial_weight: float = 0.2  # Start at 20% weight when coming back online
    
    # Legacy weight (for backward compatibility)
    weight_capacity: float = 0.3  # Deprecated, use weight_load
    
    # Seed node identity
    seed_id: Optional[str] = None
    
    # Known other seeds for redundancy
    known_seeds: List[str] = field(default_factory=list)
    
    # Proof-of-work difficulty (leading zero bits required)
    pow_difficulty_base: int = 16  # First router from IP
    pow_difficulty_second: int = 20  # Second router from same IP
    pow_difficulty_third_plus: int = 24  # Third+ router from same IP
    
    # Enable/disable signature and PoW verification
    verify_signatures: bool = True
    verify_pow: bool = True
    
    # Endpoint probing settings
    probe_endpoints: bool = True
    probe_timeout_seconds: float = 5.0
    
    # Seed peering / gossip settings
    peer_seeds: List[str] = field(default_factory=list)  # URLs of peer seeds
    gossip_enabled: bool = True
    gossip_interval_seconds: float = 300.0  # 5 minutes
    gossip_batch_size: int = 20  # Max routers per gossip exchange
    gossip_timeout_seconds: float = 10.0  # Timeout for gossip requests
    gossip_max_router_age_seconds: float = 1800.0  # Don't propagate routers older than 30 min
    
    # Misbehavior report settings (Issue #119)
    misbehavior_reports_enabled: bool = True
    misbehavior_min_reports_to_flag: int = 3  # Minimum unique reporters to flag a router
    misbehavior_report_window_seconds: float = 3600.0  # Time window for counting reports
    misbehavior_verify_reporter_signature: bool = True  # Verify reporter signatures
    misbehavior_max_reports_per_router: int = 100  # Limit stored reports per router
    misbehavior_flag_severity_threshold: float = 0.5  # Min avg severity to flag
    
    # ==========================================================================
    # Seed Revocation Settings (Issue #121)
    # ==========================================================================
    
    # Enable/disable seed revocation handling
    seed_revocation_enabled: bool = True
    
    # Verify revocation signatures (set False only for testing)
    seed_revocation_verify_signatures: bool = True
    
    # Path to out-of-band revocation list file (optional)
    # If set, the seed will load revocations from this file on startup
    # and periodically check for updates
    seed_revocation_list_path: Optional[str] = None
    
    # Interval to check for revocation list file updates (seconds)
    seed_revocation_list_check_interval: float = 3600.0  # 1 hour
    
    # Whether to propagate revocations via gossip
    seed_revocation_gossip_enabled: bool = True
    
    # Maximum age of revocation to accept (prevents replay of old revocations)
    seed_revocation_max_age_seconds: float = 86400.0 * 30  # 30 days
    
    # Trusted authority public keys for revocation list verification (hex-encoded)
    # If empty, only self-signed revocations are accepted
    seed_revocation_trusted_authorities: List[str] = field(default_factory=list)
    
    # ==========================================================================
    # Sybil Resistance Settings (Issue #117)
    # ==========================================================================
    
    # Rate limiting per IP/subnet
    rate_limit_enabled: bool = True
    rate_limit_window_seconds: float = 3600.0  # 1 hour window
    rate_limit_max_per_ip: int = 5  # Max registrations per IP per window
    rate_limit_max_per_subnet: int = 10  # Max registrations per /24 subnet per window
    rate_limit_cooldown_seconds: float = 300.0  # 5 min cooldown between registrations from same IP
    
    # Reputation system for new routers
    reputation_enabled: bool = True
    reputation_initial_score: float = 0.5  # New routers start at 50% trust (0.0 - 1.0)
    reputation_decay_period_hours: float = 24.0  # Hours to reach full trust
    reputation_min_score_for_discovery: float = 0.3  # Minimum score to be included in discovery
    reputation_boost_per_heartbeat: float = 0.01  # Score boost per successful heartbeat
    reputation_penalty_missed_heartbeat: float = 0.05  # Score penalty per missed heartbeat
    reputation_max_score: float = 1.0  # Maximum reputation score
    
    # Correlated behavior detection
    correlation_detection_enabled: bool = True
    correlation_heartbeat_window_seconds: float = 30.0  # Heartbeats within this window are suspicious
    correlation_min_suspicious_events: int = 5  # Min correlated events to flag
    correlation_endpoint_similarity_threshold: float = 0.8  # 80% endpoint similarity is suspicious
    correlation_penalty_score: float = 0.2  # Reputation penalty for correlated behavior
    
    # Adaptive PoW difficulty based on network-wide registration rate
    adaptive_pow_enabled: bool = True
    adaptive_pow_threshold_per_hour: int = 100  # If >100 registrations/hour, increase difficulty
    adaptive_pow_max_difficulty: int = 28  # Maximum difficulty bits
    adaptive_pow_difficulty_step: int = 2  # Increase difficulty by this many bits when threshold hit
    
    def __post_init__(self):
        if self.seed_id is None:
            self.seed_id = f"seed-{secrets.token_hex(8)}"


# =============================================================================
# REGIONAL ROUTING
# =============================================================================


# ISO 3166-1 alpha-2 country code to continent mapping
# Continents: AF (Africa), AN (Antarctica), AS (Asia), EU (Europe),
#             NA (North America), OC (Oceania), SA (South America)
COUNTRY_TO_CONTINENT: Dict[str, str] = {
    # North America
    "US": "NA", "CA": "NA", "MX": "NA", "GT": "NA", "BZ": "NA", "HN": "NA",
    "SV": "NA", "NI": "NA", "CR": "NA", "PA": "NA", "CU": "NA", "JM": "NA",
    "HT": "NA", "DO": "NA", "PR": "NA", "BS": "NA", "BB": "NA", "TT": "NA",
    
    # South America
    "BR": "SA", "AR": "SA", "CL": "SA", "CO": "SA", "PE": "SA", "VE": "SA",
    "EC": "SA", "BO": "SA", "PY": "SA", "UY": "SA", "GY": "SA", "SR": "SA",
    
    # Europe
    "GB": "EU", "DE": "EU", "FR": "EU", "IT": "EU", "ES": "EU", "PT": "EU",
    "NL": "EU", "BE": "EU", "CH": "EU", "AT": "EU", "SE": "EU", "NO": "EU",
    "DK": "EU", "FI": "EU", "IE": "EU", "PL": "EU", "CZ": "EU", "SK": "EU",
    "HU": "EU", "RO": "EU", "BG": "EU", "GR": "EU", "HR": "EU", "SI": "EU",
    "RS": "EU", "UA": "EU", "BY": "EU", "LT": "EU", "LV": "EU", "EE": "EU",
    "LU": "EU", "MT": "EU", "CY": "EU", "IS": "EU", "AL": "EU", "MK": "EU",
    "BA": "EU", "ME": "EU", "MD": "EU", "XK": "EU",
    
    # Asia
    "CN": "AS", "JP": "AS", "KR": "AS", "IN": "AS", "ID": "AS", "TH": "AS",
    "VN": "AS", "MY": "AS", "SG": "AS", "PH": "AS", "TW": "AS", "HK": "AS",
    "BD": "AS", "PK": "AS", "LK": "AS", "NP": "AS", "MM": "AS", "KH": "AS",
    "LA": "AS", "MN": "AS", "KZ": "AS", "UZ": "AS", "TM": "AS", "TJ": "AS",
    "KG": "AS", "AZ": "AS", "AM": "AS", "GE": "AS",
    
    # Middle East (part of Asia)
    "TR": "AS", "IR": "AS", "IQ": "AS", "SA": "AS", "AE": "AS", "IL": "AS",
    "JO": "AS", "LB": "AS", "SY": "AS", "YE": "AS", "OM": "AS", "KW": "AS",
    "QA": "AS", "BH": "AS", "PS": "AS", "AF": "AS",
    
    # Africa
    "ZA": "AF", "EG": "AF", "NG": "AF", "KE": "AF", "ET": "AF", "GH": "AF",
    "TZ": "AF", "UG": "AF", "MA": "AF", "DZ": "AF", "TN": "AF", "LY": "AF",
    "SD": "AF", "AO": "AF", "MZ": "AF", "ZW": "AF", "ZM": "AF", "BW": "AF",
    "NA": "AF", "CI": "AF", "CM": "AF", "SN": "AF", "ML": "AF", "NE": "AF",
    "BF": "AF", "MG": "AF", "MW": "AF", "RW": "AF", "SO": "AF", "CD": "AF",
    "CG": "AF", "GA": "AF", "MU": "AF", "SC": "AF", "CV": "AF",
    
    # Oceania
    "AU": "OC", "NZ": "OC", "FJ": "OC", "PG": "OC", "NC": "OC", "VU": "OC",
    "WS": "OC", "TO": "OC", "PF": "OC", "GU": "OC", "FM": "OC", "SB": "OC",
    
    # Russia spans Europe/Asia - categorize as Europe for routing
    "RU": "EU",
}


def get_continent(country_code: Optional[str]) -> Optional[str]:
    """
    Get continent code from ISO 3166-1 alpha-2 country code.
    
    Args:
        country_code: Two-letter country code (e.g., "US", "DE")
        
    Returns:
        Two-letter continent code or None if not found
    """
    if not country_code:
        return None
    return COUNTRY_TO_CONTINENT.get(country_code.upper())


def compute_region_score(
    router_region: Optional[str],
    preferred_region: Optional[str],
) -> float:
    """
    Compute region match score for router selection.
    
    Scoring tiers:
    - Same region (country): 1.0 (full match)
    - Same continent: 0.5 (partial match)
    - Different continent or unknown: 0.0 (no match)
    
    Args:
        router_region: Router's region (country code)
        preferred_region: Requested region preference (country code)
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not preferred_region or not router_region:
        return 0.0
    
    # Normalize to uppercase
    router_region = router_region.upper()
    preferred_region = preferred_region.upper()
    
    # Same country = full match
    if router_region == preferred_region:
        return 1.0
    
    # Same continent = partial match
    router_continent = get_continent(router_region)
    preferred_continent = get_continent(preferred_region)
    
    if router_continent and preferred_continent and router_continent == preferred_continent:
        return 0.5
    
    # Different continent or unknown
    return 0.0


# =============================================================================
# SEED REVOCATION MANAGER (Issue #121)
# =============================================================================


@dataclass
class SeedRevocationRecord:
    """Record of a seed revocation for storage."""
    seed_id: str
    revocation_id: str
    reason: str
    reason_detail: str
    timestamp: float
    effective_at: float
    issuer_id: str
    signature: str
    received_at: float = field(default_factory=time.time)
    source: str = "direct"  # "direct", "gossip", or "file"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "seed_id": self.seed_id,
            "revocation_id": self.revocation_id,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "timestamp": self.timestamp,
            "effective_at": self.effective_at,
            "issuer_id": self.issuer_id,
            "signature": self.signature,
            "received_at": self.received_at,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedRevocationRecord":
        """Deserialize from dict."""
        return cls(
            seed_id=data["seed_id"],
            revocation_id=data["revocation_id"],
            reason=data.get("reason", ""),
            reason_detail=data.get("reason_detail", ""),
            timestamp=data.get("timestamp", 0),
            effective_at=data.get("effective_at", 0),
            issuer_id=data.get("issuer_id", ""),
            signature=data.get("signature", ""),
            received_at=data.get("received_at", time.time()),
            source=data.get("source", "direct"),
        )
    
    @property
    def is_effective(self) -> bool:
        """Check if the revocation is currently effective."""
        return time.time() >= self.effective_at


class SeedRevocationManager:
    """
    Manages seed revocations for a seed node (Issue #121).
    
    Implements:
    - Revocation storage and lookup
    - Signature verification for revocations
    - Out-of-band revocation list loading from file
    - Integration with gossip for revocation propagation
    
    Revocations can be:
    - Self-signed (seed revokes itself)
    - Authority-signed (trusted authority revokes seed)
    
    Nodes query the seed for revocations and honor them by
    excluding revoked seeds from discovery.
    """
    
    def __init__(self, config: "SeedConfig"):
        self.config = config
        self._revocations: Dict[str, SeedRevocationRecord] = {}  # seed_id -> record
        self._revocation_list_mtime: float = 0  # Last modification time of file
        self._check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def is_seed_revoked(self, seed_id: str) -> bool:
        """
        Check if a seed is revoked.
        
        Args:
            seed_id: The seed ID to check
            
        Returns:
            True if the seed is revoked and the revocation is effective
        """
        if not self.config.seed_revocation_enabled:
            return False
        
        record = self._revocations.get(seed_id)
        if record is None:
            return False
        
        return record.is_effective
    
    def get_revocation(self, seed_id: str) -> Optional[SeedRevocationRecord]:
        """Get revocation record for a seed."""
        return self._revocations.get(seed_id)
    
    def get_all_revocations(self) -> List[SeedRevocationRecord]:
        """Get all revocation records."""
        return list(self._revocations.values())
    
    def get_revoked_seed_ids(self) -> set:
        """Get set of all currently revoked seed IDs."""
        now = time.time()
        return {
            seed_id for seed_id, record in self._revocations.items()
            if record.effective_at <= now
        }
    
    def add_revocation(
        self,
        revocation_data: Dict[str, Any],
        source: str = "direct",
        verify: bool = True,
    ) -> tuple[bool, Optional[str]]:
        """
        Add a seed revocation.
        
        Args:
            revocation_data: Revocation dict (from SeedRevocation.to_dict())
            source: Source of revocation ("direct", "gossip", "file")
            verify: Whether to verify the signature
            
        Returns:
            Tuple of (success: bool, error_reason: Optional[str])
        """
        if not self.config.seed_revocation_enabled:
            return False, "revocation_disabled"
        
        seed_id = revocation_data.get("seed_id")
        if not seed_id:
            return False, "missing_seed_id"
        
        revocation_id = revocation_data.get("revocation_id")
        if not revocation_id:
            return False, "missing_revocation_id"
        
        timestamp = revocation_data.get("timestamp", 0)
        issuer_id = revocation_data.get("issuer_id", "")
        signature = revocation_data.get("signature", "")
        
        # Check revocation age
        now = time.time()
        age = now - timestamp
        if age > self.config.seed_revocation_max_age_seconds:
            return False, f"revocation_too_old:age={int(age)}s"
        
        # Verify signature if enabled
        if verify and self.config.seed_revocation_verify_signatures:
            is_valid, error = self._verify_revocation_signature(revocation_data)
            if not is_valid:
                return False, f"invalid_signature:{error}"
        
        # Check if we already have this revocation
        existing = self._revocations.get(seed_id)
        if existing and existing.revocation_id == revocation_id:
            # Already have this exact revocation
            return True, None
        
        # If we have an older revocation for this seed, keep the newer one
        if existing and existing.timestamp > timestamp:
            # Existing is newer, ignore this one
            logger.debug(
                f"Ignoring older revocation for {seed_id[:20]}... "
                f"(existing={existing.timestamp}, new={timestamp})"
            )
            return True, None
        
        # Create and store the revocation record
        record = SeedRevocationRecord(
            seed_id=seed_id,
            revocation_id=revocation_id,
            reason=revocation_data.get("reason", ""),
            reason_detail=revocation_data.get("reason_detail", ""),
            timestamp=timestamp,
            effective_at=revocation_data.get("effective_at", timestamp),
            issuer_id=issuer_id,
            signature=signature,
            received_at=now,
            source=source,
        )
        
        self._revocations[seed_id] = record
        
        logger.info(
            f"Added seed revocation: {seed_id[:20]}... "
            f"reason={record.reason}, issuer={issuer_id[:20]}..., source={source}"
        )
        
        return True, None
    
    def _verify_revocation_signature(
        self,
        revocation_data: Dict[str, Any],
    ) -> tuple[bool, Optional[str]]:
        """
        Verify the signature on a revocation.
        
        A revocation is valid if:
        1. It's self-signed (issuer_id == seed_id) and signature is valid
        2. It's signed by a trusted authority in config.seed_revocation_trusted_authorities
        
        Args:
            revocation_data: Revocation dict
            
        Returns:
            Tuple of (is_valid: bool, error: Optional[str])
        """
        seed_id = revocation_data.get("seed_id", "")
        issuer_id = revocation_data.get("issuer_id", "")
        signature_hex = revocation_data.get("signature", "")
        
        if not signature_hex:
            return False, "missing_signature"
        
        if not issuer_id:
            return False, "missing_issuer_id"
        
        # Determine if this is self-signed or authority-signed
        is_self_signed = issuer_id == seed_id
        is_authority = issuer_id in self.config.seed_revocation_trusted_authorities
        
        if not is_self_signed and not is_authority:
            return False, "untrusted_issuer"
        
        # Build the signed data (same as SeedRevocation.get_signable_data())
        signed_data = {
            "type": revocation_data.get("type", "seed_revocation"),
            "revocation_id": revocation_data.get("revocation_id"),
            "seed_id": seed_id,
            "reason": revocation_data.get("reason", ""),
            "reason_detail": revocation_data.get("reason_detail", ""),
            "timestamp": revocation_data.get("timestamp"),
            "effective_at": revocation_data.get("effective_at"),
            "issuer_id": issuer_id,
        }
        message = json.dumps(signed_data, sort_keys=True, separators=(',', ':')).encode()
        
        try:
            # Parse public key from issuer_id (hex-encoded Ed25519 public key)
            public_key_bytes = bytes.fromhex(issuer_id)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            
            # Parse and verify signature
            signature = bytes.fromhex(signature_hex)
            public_key.verify(signature, message)
            
            return True, None
            
        except InvalidSignature:
            return False, "signature_verification_failed"
        except (ValueError, TypeError) as e:
            return False, f"invalid_key_or_signature_format:{e}"
        except Exception as e:
            return False, f"verification_error:{e}"
    
    def load_revocation_list_from_file(self, file_path: str) -> tuple[int, List[str]]:
        """
        Load revocations from an out-of-band file.
        
        The file should contain a JSON SeedRevocationList.
        
        Args:
            file_path: Path to the revocation list file
            
        Returns:
            Tuple of (loaded_count: int, errors: List[str])
        """
        errors = []
        loaded_count = 0
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            errors.append(f"file_not_found:{file_path}")
            return 0, errors
        except json.JSONDecodeError as e:
            errors.append(f"invalid_json:{e}")
            return 0, errors
        except Exception as e:
            errors.append(f"read_error:{e}")
            return 0, errors
        
        # Import here to avoid circular imports
        from valence.network.messages import SeedRevocationList
        
        try:
            revocation_list = SeedRevocationList.from_dict(data)
        except Exception as e:
            errors.append(f"parse_error:{e}")
            return 0, errors
        
        # Verify list signature if we have trusted authorities
        if self.config.seed_revocation_verify_signatures:
            if revocation_list.authority_id:
                is_valid, error = self._verify_list_signature(revocation_list)
                if not is_valid:
                    errors.append(f"list_signature_invalid:{error}")
                    return 0, errors
        
        # Process each revocation
        for revocation in revocation_list.revocations:
            success, error = self.add_revocation(
                revocation.to_dict(),
                source="file",
                verify=self.config.seed_revocation_verify_signatures,
            )
            if success:
                loaded_count += 1
            else:
                errors.append(f"revocation_error:{revocation.seed_id}:{error}")
        
        logger.info(
            f"Loaded {loaded_count} revocations from {file_path} "
            f"(version={revocation_list.version}, errors={len(errors)})"
        )
        
        return loaded_count, errors
    
    def _verify_list_signature(
        self,
        revocation_list: "SeedRevocationList",
    ) -> tuple[bool, Optional[str]]:
        """Verify the signature on a revocation list."""
        authority_id = revocation_list.authority_id
        signature_hex = revocation_list.signature
        
        if not authority_id:
            return False, "missing_authority_id"
        
        if authority_id not in self.config.seed_revocation_trusted_authorities:
            return False, "untrusted_authority"
        
        if not signature_hex:
            return False, "missing_signature"
        
        message = revocation_list.get_signable_bytes()
        
        try:
            public_key_bytes = bytes.fromhex(authority_id)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            signature = bytes.fromhex(signature_hex)
            public_key.verify(signature, message)
            return True, None
        except InvalidSignature:
            return False, "signature_verification_failed"
        except Exception as e:
            return False, f"verification_error:{e}"
    
    async def start(self) -> None:
        """Start the revocation manager (file watching loop)."""
        if self._running:
            return
        
        self._running = True
        
        # Load initial revocations from file if configured
        if self.config.seed_revocation_list_path:
            try:
                loaded, errors = self.load_revocation_list_from_file(
                    self.config.seed_revocation_list_path
                )
                if errors:
                    logger.warning(f"Revocation list load errors: {errors}")
            except Exception as e:
                logger.error(f"Failed to load revocation list: {e}")
        
        # Start file check loop if path is configured
        if self.config.seed_revocation_list_path:
            self._check_task = asyncio.create_task(self._file_check_loop())
        
        logger.info("Seed revocation manager started")
    
    async def stop(self) -> None:
        """Stop the revocation manager."""
        self._running = False
        
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        
        logger.info("Seed revocation manager stopped")
    
    async def _file_check_loop(self) -> None:
        """Periodically check for revocation list file updates."""
        while self._running:
            await asyncio.sleep(self.config.seed_revocation_list_check_interval)
            
            if not self.config.seed_revocation_list_path:
                continue
            
            try:
                mtime = os.path.getmtime(self.config.seed_revocation_list_path)
                if mtime > self._revocation_list_mtime:
                    logger.info("Revocation list file updated, reloading...")
                    self.load_revocation_list_from_file(
                        self.config.seed_revocation_list_path
                    )
                    self._revocation_list_mtime = mtime
            except FileNotFoundError:
                pass  # File doesn't exist yet
            except Exception as e:
                logger.error(f"Error checking revocation list file: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get revocation manager statistics."""
        now = time.time()
        effective_count = sum(
            1 for r in self._revocations.values()
            if r.effective_at <= now
        )
        
        return {
            "enabled": self.config.seed_revocation_enabled,
            "total_revocations": len(self._revocations),
            "effective_revocations": effective_count,
            "revocation_list_path": self.config.seed_revocation_list_path,
            "revoked_seeds": list(self.get_revoked_seed_ids()),
        }
    
    def get_revocations_for_gossip(self) -> List[Dict[str, Any]]:
        """
        Get revocations to share in gossip exchange.
        
        Returns recent revocations that should be propagated.
        """
        if not self.config.seed_revocation_gossip_enabled:
            return []
        
        now = time.time()
        max_age = self.config.seed_revocation_max_age_seconds
        
        revocations = []
        for record in self._revocations.values():
            # Only share revocations within age limit
            if now - record.timestamp > max_age:
                continue
            
            revocations.append(record.to_dict())
        
        return revocations
    
    def process_gossip_revocations(self, revocations: List[Dict[str, Any]]) -> int:
        """
        Process revocations received via gossip.
        
        Args:
            revocations: List of revocation dicts from gossip
            
        Returns:
            Number of revocations added
        """
        added = 0
        
        for rev_data in revocations:
            success, error = self.add_revocation(
                rev_data,
                source="gossip",
                verify=self.config.seed_revocation_verify_signatures,
            )
            if success:
                added += 1
            elif error:
                logger.debug(f"Gossip revocation rejected: {error}")
        
        return added


# =============================================================================
# SEED PEERING / GOSSIP
# =============================================================================


class SeedPeerManager:
    """
    Manages peering and gossip between seed nodes.
    
    Implements:
    - Periodic gossip exchanges with peer seeds
    - Router registry synchronization (subset-based, not full)
    - Deduplication of routers by ID
    - Filtering of stale/unhealthy routers before propagation
    
    Gossip protocol:
    - Each seed maintains a list of peer seed URLs
    - Periodically (configurable interval), seeds exchange router subsets
    - Only healthy, fresh routers are shared
    - Routers are deduplicated by router_id (newer wins)
    """
    
    def __init__(self, seed: "SeedNode"):
        self.seed = seed
        self._running = False
        self._gossip_task: Optional[asyncio.Task] = None
        self._peer_states: Dict[str, Dict[str, Any]] = {}  # peer_url -> state
        
    @property
    def peer_seeds(self) -> List[str]:
        """Get list of peer seed URLs."""
        return self.seed.config.peer_seeds
    
    @property
    def gossip_interval(self) -> float:
        """Get gossip interval in seconds."""
        return self.seed.config.gossip_interval_seconds
    
    @property
    def batch_size(self) -> int:
        """Get max routers per gossip exchange."""
        return self.seed.config.gossip_batch_size
    
    async def start(self) -> None:
        """Start the gossip loop."""
        if self._running:
            return
        
        if not self.seed.config.gossip_enabled:
            logger.info("Gossip disabled by configuration")
            return
        
        if not self.peer_seeds:
            logger.info("No peer seeds configured, gossip not started")
            return
        
        self._running = True
        self._gossip_task = asyncio.create_task(self._gossip_loop())
        logger.info(
            f"Seed peer manager started with {len(self.peer_seeds)} peers, "
            f"gossip interval={self.gossip_interval}s"
        )
    
    async def stop(self) -> None:
        """Stop the gossip loop."""
        self._running = False
        if self._gossip_task:
            self._gossip_task.cancel()
            try:
                await self._gossip_task
            except asyncio.CancelledError:
                pass
            self._gossip_task = None
        logger.info("Seed peer manager stopped")
    
    async def _gossip_loop(self) -> None:
        """Periodically gossip with peer seeds."""
        # Initial delay to let seed start up
        await asyncio.sleep(5)
        
        while self._running:
            try:
                await self._gossip_round()
            except Exception as e:
                logger.error(f"Gossip round failed: {e}")
            
            await asyncio.sleep(self.gossip_interval)
    
    async def _gossip_round(self) -> None:
        """Execute one round of gossip with all peers."""
        if not self.peer_seeds:
            return
        
        logger.debug(f"Starting gossip round with {len(self.peer_seeds)} peers")
        
        # Gather routers to share
        routers_to_share = self._select_routers_for_gossip()
        
        if not routers_to_share:
            logger.debug("No routers to share in gossip")
        
        # Exchange with each peer concurrently
        tasks = [
            self._exchange_with_peer(peer_url, routers_to_share)
            for peer_url in self.peer_seeds
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if r is True)
        logger.info(
            f"Gossip round complete: {successful}/{len(self.peer_seeds)} peers exchanged, "
            f"shared {len(routers_to_share)} routers"
        )
    
    def _select_routers_for_gossip(self) -> List[Dict[str, Any]]:
        """
        Select a subset of routers to share in gossip.
        
        Filters:
        - Only healthy routers (based on health monitor)
        - Only fresh routers (seen within max_router_age)
        - Limited to batch_size
        - Randomized selection for fairness
        """
        now = time.time()
        max_age = self.seed.config.gossip_max_router_age_seconds
        
        candidates = []
        for router_id, router in self.seed.router_registry.items():
            # Check freshness
            last_seen = router.health.get("last_seen", 0)
            if now - last_seen > max_age:
                continue
            
            # Check health via health monitor
            if not self.seed.health_monitor.is_healthy_for_discovery(router_id):
                continue
            
            # Check legacy health status
            if not self.seed._is_healthy(router, now):
                continue
            
            candidates.append(router)
        
        # Randomize and limit
        if len(candidates) > self.batch_size:
            candidates = random.sample(candidates, self.batch_size)
        
        return [r.to_dict() for r in candidates]
    
    async def _exchange_with_peer(
        self,
        peer_url: str,
        routers_to_share: List[Dict[str, Any]],
    ) -> bool:
        """
        Exchange router info with a single peer seed.
        
        Returns True if exchange was successful.
        """
        try:
            # Normalize URL
            if not peer_url.startswith("http"):
                peer_url = f"http://{peer_url}"
            
            exchange_url = f"{peer_url.rstrip('/')}/gossip/exchange"
            
            # Include revocations in gossip exchange (Issue #121)
            revocations_to_share = []
            if hasattr(self.seed, 'revocation_manager'):
                revocations_to_share = self.seed.revocation_manager.get_revocations_for_gossip()
            
            payload = {
                "seed_id": self.seed.seed_id,
                "timestamp": time.time(),
                "routers": routers_to_share,
                "revocations": revocations_to_share,  # Issue #121
            }
            
            timeout = aiohttp.ClientTimeout(
                total=self.seed.config.gossip_timeout_seconds
            )
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(exchange_url, json=payload) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"Gossip exchange with {peer_url} failed: "
                            f"status {resp.status}"
                        )
                        self._update_peer_state(peer_url, success=False)
                        return False
                    
                    data = await resp.json()
                    
                    # Process received routers
                    received_routers = data.get("routers", [])
                    merged_count = self._merge_routers(received_routers, peer_url)
                    
                    # Process received revocations (Issue #121)
                    revocations_merged = 0
                    received_revocations = data.get("revocations", [])
                    if received_revocations and hasattr(self.seed, 'revocation_manager'):
                        revocations_merged = self.seed.revocation_manager.process_gossip_revocations(
                            received_revocations
                        )
                    
                    self._update_peer_state(peer_url, success=True)
                    
                    logger.debug(
                        f"Gossip with {peer_url}: sent {len(routers_to_share)} routers, "
                        f"received {len(received_routers)}, merged {merged_count}; "
                        f"sent {len(revocations_to_share)} revocations, "
                        f"received {len(received_revocations)}, merged {revocations_merged}"
                    )
                    
                    return True
                    
        except asyncio.TimeoutError:
            logger.warning(f"Gossip exchange with {peer_url} timed out")
            self._update_peer_state(peer_url, success=False, error="timeout")
            return False
        except Exception as e:
            logger.warning(f"Gossip exchange with {peer_url} failed: {e}")
            self._update_peer_state(peer_url, success=False, error=str(e))
            return False
    
    def _merge_routers(
        self,
        received_routers: List[Dict[str, Any]],
        source_peer: str,
    ) -> int:
        """
        Merge received routers into local registry.
        
        Deduplication rules:
        - If router_id doesn't exist, add it
        - If router_id exists, keep the one with more recent last_seen
        
        Returns number of routers actually merged (new or updated).
        """
        now = time.time()
        max_age = self.seed.config.gossip_max_router_age_seconds
        merged = 0
        
        for router_data in received_routers:
            try:
                router_id = router_data.get("router_id")
                if not router_id:
                    continue
                
                # Check freshness of received router
                last_seen = router_data.get("health", {}).get("last_seen", 0)
                if now - last_seen > max_age:
                    logger.debug(
                        f"Skipping stale router {router_id[:20]}... from gossip "
                        f"(age={now - last_seen:.0f}s)"
                    )
                    continue
                
                # Check if we should merge
                existing = self.seed.router_registry.get(router_id)
                
                if existing is None:
                    # New router - add it
                    router = RouterRecord.from_dict(router_data)
                    router.source_ip = f"gossip:{source_peer}"
                    self.seed.router_registry[router_id] = router
                    
                    # Initialize health state
                    self.seed.health_monitor.record_heartbeat(router_id)
                    
                    merged += 1
                    logger.debug(
                        f"Added router {router_id[:20]}... from gossip "
                        f"(peer={source_peer})"
                    )
                else:
                    # Existing router - compare freshness
                    existing_last_seen = existing.health.get("last_seen", 0)
                    
                    if last_seen > existing_last_seen:
                        # Received router is newer - update
                        router = RouterRecord.from_dict(router_data)
                        router.source_ip = existing.source_ip  # Preserve original source
                        router.registered_at = existing.registered_at  # Preserve registration time
                        self.seed.router_registry[router_id] = router
                        
                        merged += 1
                        logger.debug(
                            f"Updated router {router_id[:20]}... from gossip "
                            f"(peer={source_peer}, delta={last_seen - existing_last_seen:.0f}s)"
                        )
                        
            except Exception as e:
                logger.warning(f"Failed to merge router from gossip: {e}")
                continue
        
        return merged
    
    def _update_peer_state(
        self,
        peer_url: str,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """Update state tracking for a peer."""
        now = time.time()
        
        if peer_url not in self._peer_states:
            self._peer_states[peer_url] = {
                "first_seen": now,
                "successful_exchanges": 0,
                "failed_exchanges": 0,
                "last_success": None,
                "last_failure": None,
                "last_error": None,
            }
        
        state = self._peer_states[peer_url]
        
        if success:
            state["successful_exchanges"] += 1
            state["last_success"] = now
        else:
            state["failed_exchanges"] += 1
            state["last_failure"] = now
            state["last_error"] = error
    
    def get_peer_stats(self) -> Dict[str, Any]:
        """Get statistics about peer connections."""
        return {
            "peer_count": len(self.peer_seeds),
            "gossip_enabled": self.seed.config.gossip_enabled,
            "gossip_interval_seconds": self.gossip_interval,
            "peer_states": {
                url: {
                    "successful": state["successful_exchanges"],
                    "failed": state["failed_exchanges"],
                    "last_success": state["last_success"],
                    "last_error": state["last_error"],
                }
                for url, state in self._peer_states.items()
            },
        }
    
    async def handle_gossip_exchange(self, request: web.Request) -> web.Response:
        """
        Handle incoming gossip exchange from peer seed.
        
        POST /gossip/exchange
        {
            "seed_id": "peer-seed-001",
            "timestamp": 1706789012.345,
            "routers": [...]
        }
        
        Response:
        {
            "seed_id": "this-seed-id",
            "timestamp": 1706789012.456,
            "routers": [...]  // Our routers to share back
        }
        """
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"status": "error", "reason": "invalid_json", "detail": str(e)},
                status=400
            )
        
        peer_seed_id = data.get("seed_id", "unknown")
        received_routers = data.get("routers", [])
        received_revocations = data.get("revocations", [])
        
        # Merge received routers
        merged_count = self._merge_routers(received_routers, peer_seed_id)
        
        # Process received revocations (Issue #121)
        revocations_merged = 0
        if received_revocations and hasattr(self.seed, 'revocation_manager'):
            revocations_merged = self.seed.revocation_manager.process_gossip_revocations(
                received_revocations
            )
        
        # Select our routers to send back
        routers_to_share = self._select_routers_for_gossip()
        
        # Include revocations in response (Issue #121)
        revocations_to_share = []
        if hasattr(self.seed, 'revocation_manager'):
            revocations_to_share = self.seed.revocation_manager.get_revocations_for_gossip()
        
        logger.debug(
            f"Gossip exchange from {peer_seed_id}: "
            f"received {len(received_routers)} routers, merged {merged_count}; "
            f"received {len(received_revocations)} revocations, merged {revocations_merged}"
        )
        
        return web.json_response({
            "seed_id": self.seed.seed_id,
            "timestamp": time.time(),
            "routers": routers_to_share,
            "revocations": revocations_to_share,
        })


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class RouterRecord:
    """Record of a registered router."""
    
    router_id: str  # Ed25519 public key (hex)
    endpoints: List[str]  # ["ip:port", ...]
    capacity: Dict[str, Any]  # {max_connections, current_load_pct, bandwidth_mbps}
    health: Dict[str, Any]  # {last_seen, uptime_pct, avg_latency_ms, status}
    regions: List[str]  # Geographic regions served
    features: List[str]  # Supported features/protocols
    registered_at: float  # Unix timestamp
    router_signature: str  # Signature of registration data
    proof_of_work: Optional[Dict[str, Any]] = None  # PoW proof
    source_ip: Optional[str] = None  # IP address that registered this router
    region: Optional[str] = None  # ISO 3166-1 alpha-2 country code (e.g., "US", "DE")
    coordinates: Optional[List[float]] = None  # [latitude, longitude]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "router_id": self.router_id,
            "endpoints": self.endpoints,
            "capacity": self.capacity,
            "health": self.health,
            "regions": self.regions,
            "features": self.features,
            "registered_at": self.registered_at,
            "router_signature": self.router_signature,
        }
        if self.region:
            result["region"] = self.region
        if self.coordinates:
            result["coordinates"] = self.coordinates
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterRecord":
        """Create from dictionary."""
        coords = data.get("coordinates")
        if coords and isinstance(coords, (list, tuple)) and len(coords) == 2:
            coordinates = [float(coords[0]), float(coords[1])]
        else:
            coordinates = None
            
        return cls(
            router_id=data["router_id"],
            endpoints=data.get("endpoints", []),
            capacity=data.get("capacity", {}),
            health=data.get("health", {}),
            regions=data.get("regions", []),
            features=data.get("features", []),
            registered_at=data.get("registered_at", time.time()),
            router_signature=data.get("router_signature", ""),
            proof_of_work=data.get("proof_of_work"),
            source_ip=data.get("source_ip"),
            region=data.get("region"),
            coordinates=coordinates,
        )


# =============================================================================
# SEED NODE
# =============================================================================


@dataclass
class SeedNode:
    """
    Seed node for router discovery.
    
    Maintains a registry of routers and responds to discovery requests
    from nodes looking to connect to the network.
    """
    
    config: SeedConfig = field(default_factory=SeedConfig)
    router_registry: Dict[str, RouterRecord] = field(default_factory=dict)
    
    # Track routers per IP for PoW difficulty scaling
    _ip_router_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Health monitoring
    _health_monitor: Optional[HealthMonitor] = field(default=None, repr=False)
    
    # Seed peering / gossip
    _peer_manager: Optional[SeedPeerManager] = field(default=None, repr=False)
    
    # Seed revocation management (Issue #121)
    _revocation_manager: Optional[SeedRevocationManager] = field(default=None, repr=False)
    
    # Sybil resistance (Issue #117)
    _sybil_resistance: Optional[SybilResistance] = field(default=None, repr=False)
    
    # Misbehavior reports (Issue #119): {router_id: {reporter_id: report_data}}
    _misbehavior_reports: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict, repr=False)
    
    # Flagged routers from misbehavior reports: {router_id: flag_timestamp}
    _misbehavior_flagged_routers: Dict[str, float] = field(default_factory=dict, repr=False)
    
    # Runtime state
    _app: Optional[web.Application] = field(default=None, repr=False)
    _runner: Optional[web.AppRunner] = field(default=None, repr=False)
    _site: Optional[web.TCPSite] = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)
    
    @property
    def seed_id(self) -> str:
        """Get the seed node's ID."""
        # seed_id is set in SeedConfig.__post_init__ if None
        assert self.config.seed_id is not None
        return self.config.seed_id
    
    @property
    def known_seeds(self) -> List[str]:
        """Get list of known peer seeds."""
        return self.config.known_seeds
    
    @property
    def health_monitor(self) -> HealthMonitor:
        """Get the health monitor, creating if needed."""
        if self._health_monitor is None:
            self._health_monitor = HealthMonitor(self)
        return self._health_monitor
    
    @property
    def peer_manager(self) -> SeedPeerManager:
        """Get the peer manager, creating if needed."""
        if self._peer_manager is None:
            self._peer_manager = SeedPeerManager(self)
        return self._peer_manager
    
    @property
    def sybil_resistance(self) -> SybilResistance:
        """Get the Sybil resistance manager, creating if needed (Issue #117)."""
        if not hasattr(self, '_sybil_resistance') or self._sybil_resistance is None:
            self._sybil_resistance = SybilResistance(self.config)
        return self._sybil_resistance
    
    @property
    def revocation_manager(self) -> SeedRevocationManager:
        """Get the seed revocation manager, creating if needed (Issue #121)."""
        if self._revocation_manager is None:
            self._revocation_manager = SeedRevocationManager(self.config)
        return self._revocation_manager
    
    # -------------------------------------------------------------------------
    # ROUTER SELECTION
    # -------------------------------------------------------------------------
    
    def _get_subnet(self, endpoint: str) -> Optional[str]:
        """Extract /16 subnet from an endpoint for diversity checking."""
        try:
            # Handle "ip:port" format
            host = endpoint.split(":")[0] if ":" in endpoint else endpoint
            
            # Skip non-IPv4 addresses
            parts = host.split(".")
            if len(parts) != 4:
                return None
            
            # Return /16 subnet (first two octets)
            return f"{parts[0]}.{parts[1]}"
        except Exception:
            return None
    
    def _is_healthy(self, router: RouterRecord, now: float) -> bool:
        """Check if a router is healthy based on uptime and freshness."""
        uptime = router.health.get("uptime_pct", 0)
        last_seen = router.health.get("last_seen", 0)
        
        if uptime < self.config.min_uptime_pct:
            return False
        
        if now - last_seen > self.config.max_stale_seconds:
            return False
        
        return True
    
    def _score_router(self, router: RouterRecord, preferences: Dict[str, Any]) -> float:
        """
        Score a router for selection using load-aware balancing.
        
        Higher score = better candidate.
        
        Load metrics considered:
        - Connection load: current_connections / max_connections
        - Queue depth: penalize routers with deep queues
        - Message throughput: consider current message rate
        - Recovery state: ramp up slowly to avoid thundering herd
        
        Args:
            router: The router to score
            preferences: Selection preferences (region, features, etc.)
            
        Returns:
            Score between 0.0 and ~1.0 (can exceed with bonuses)
        """
        score = 0.0
        now = time.time()
        
        # Health component (0-1, weighted)
        uptime = router.health.get("uptime_pct", 0) / 100.0
        score += uptime * self.config.weight_health
        
        # Load component (0-1, weighted) - prefer lower connection load
        # Use connection ratio if available, fall back to load_pct
        current_conn = router.capacity.get("active_connections", 0)
        max_conn = router.capacity.get("max_connections", 100)
        if max_conn > 0 and current_conn > 0:
            connection_load = current_conn / max_conn
        else:
            connection_load = router.capacity.get("current_load_pct", 0) / 100.0
        
        # Invert: low load = high score
        load_score = 1.0 - min(connection_load, 1.0)
        score += load_score * self.config.weight_load
        
        # Queue depth penalty (0-1, weighted)
        # Deep queues indicate backpressure - penalize proportionally
        queue_depth = router.capacity.get("queue_depth", 0)
        # Normalize: 0 queue = 1.0, 100+ queue = 0.0
        queue_penalty = max(0.0, 1.0 - (queue_depth / 100.0))
        score += queue_penalty * self.config.weight_queue
        
        # Throughput consideration (0-1, weighted)
        # Moderate throughput is good (router is active), very high may indicate stress
        msg_per_sec = router.capacity.get("messages_per_sec", 0)
        # Normalize: 0-50 msg/s is good, 50-100 starts to penalize
        if msg_per_sec <= 50:
            throughput_score = 1.0
        elif msg_per_sec <= 100:
            throughput_score = 1.0 - ((msg_per_sec - 50) / 100.0)
        else:
            throughput_score = 0.5  # Still usable but not preferred
        score += throughput_score * self.config.weight_throughput
        
        # Recovery ramp-up (thundering herd prevention)
        # If router recently recovered, scale down its attractiveness
        recovery_factor = self._get_recovery_factor(router, now)
        if recovery_factor < 1.0:
            # Scale down the non-random portion of score
            non_random_score = score
            score = non_random_score * recovery_factor
        
        # Region match bonus with tiered geographic scoring
        # Supports both 'preferred_region' (new) and 'region' (legacy) keys
        preferred_region = preferences.get("preferred_region") or preferences.get("region")
        if preferred_region:
            # Use new region field (country code) for geographic routing
            # Tiered: same country = 1.0, same continent = 0.5, different = 0.0
            region_score = compute_region_score(router.region, preferred_region)
            score += region_score * self.config.weight_region
            
            # Legacy: also check regions list for backward compatibility
            if region_score == 0 and preferred_region in router.regions:
                score += self.config.weight_region * 0.25  # Smaller bonus for legacy match
        
        # Feature match bonus (partial)
        required_features = set(preferences.get("features", []))
        if required_features:
            router_features = set(router.features)
            match_ratio = len(required_features & router_features) / len(required_features)
            score += match_ratio * 0.05  # Small bonus for feature match
        
        # Deterministic random factor (based on router_id for consistency)
        # This provides some load distribution even among equally-scored routers
        random_component = (hash(router.router_id) % 100) / 100.0
        score += random_component * self.config.weight_random
        
        return score
    
    def _get_recovery_factor(self, router: RouterRecord, now: float) -> float:
        """
        Calculate recovery factor for thundering herd prevention.
        
        When a router comes back online after being unhealthy, we don't want
        all traffic to immediately flood it. This method returns a factor
        between recovery_initial_weight and 1.0 that ramps up over time.
        
        Args:
            router: The router to check
            now: Current timestamp
            
        Returns:
            Factor between 0.0 and 1.0 to multiply the router's score
        """
        router_id = router.router_id
        health_state = self.health_monitor.get_health_state(router_id)
        
        if health_state is None:
            return 1.0  # New router, no recovery needed
        
        # Check if router recently recovered from unhealthy state
        # We track this via the health state's last status change
        last_heartbeat = health_state.last_heartbeat
        
        # If router was marked as recovered recently (within ramp duration)
        # we check by comparing last_heartbeat time to now
        # A router that just sent its first heartbeat after being down
        # will have a very recent last_heartbeat
        
        # Check the router's recovery time if tracked
        recovery_time = getattr(health_state, 'recovery_time', None)
        if recovery_time is None:
            return 1.0  # No recovery tracking, full weight
        
        elapsed = now - recovery_time
        if elapsed >= self.config.recovery_ramp_duration:
            return 1.0  # Fully ramped up
        
        # Linear ramp from initial_weight to 1.0 over ramp_duration
        progress = elapsed / self.config.recovery_ramp_duration
        factor = self.config.recovery_initial_weight + (1.0 - self.config.recovery_initial_weight) * progress
        
        return factor
    
    def select_routers(
        self,
        count: int,
        preferences: Optional[Dict[str, Any]] = None,
        include_unhealthy: bool = False,
    ) -> List[RouterRecord]:
        """
        Select routers based on health, capacity, and preferences.
        
        Args:
            count: Number of routers to return
            preferences: Optional dict with region, features, etc.
            include_unhealthy: If True, skip health monitor filtering
            
        Returns:
            List of RouterRecord objects (up to count)
        """
        preferences = preferences or {}
        now = time.time()
        
        # Get healthy candidates (both legacy health check and health monitor)
        candidates = []
        for r in self.router_registry.values():
            # Legacy health check (based on uptime and staleness)
            if not self._is_healthy(r, now):
                continue
            
            # Health monitor check (based on heartbeat tracking)
            if not include_unhealthy and not self.health_monitor.is_healthy_for_discovery(r.router_id):
                continue
            
            # Sybil resistance: Filter by reputation (Issue #117)
            if not self.sybil_resistance.is_trusted_for_discovery(r.router_id):
                logger.debug(
                    f"Router {r.router_id[:20]}... excluded from discovery "
                    f"(low reputation)"
                )
                continue
            
            candidates.append(r)
        
        if not candidates:
            logger.warning("No healthy routers available")
            return []
        
        # Score and sort, incorporating trust factor
        scored = []
        for r in candidates:
            base_score = self._score_router(r, preferences)
            trust_factor = self.sybil_resistance.get_trust_factor(r.router_id)
            # Apply trust factor as a multiplier (0.5 to 1.0 range typically)
            # This gives established routers an advantage over new ones
            final_score = base_score * (0.5 + 0.5 * trust_factor)
            scored.append((final_score, r))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Select with IP diversity
        selected: List[RouterRecord] = []
        seen_subnets: set[Optional[str]] = set()
        
        for _, router in scored:
            # Get subnet from first endpoint
            subnet = None
            if router.endpoints:
                subnet = self._get_subnet(router.endpoints[0])
            
            # Skip if we already have a router from this subnet
            # (None subnets are always allowed - could be IPv6 or hostname)
            if subnet is not None and subnet in seen_subnets:
                continue
            
            selected.append(router)
            if subnet is not None:
                seen_subnets.add(subnet)
            
            if len(selected) >= count:
                break
        
        logger.debug(
            f"Selected {len(selected)} routers from {len(candidates)} candidates "
            f"(registry has {len(self.router_registry)} total)"
        )
        
        return selected
    
    # -------------------------------------------------------------------------
    # VERIFICATION METHODS
    # -------------------------------------------------------------------------
    
    def _get_pow_difficulty(self, source_ip: str) -> int:
        """
        Get required PoW difficulty based on number of routers from this IP
        and current network conditions (adaptive PoW).
        
        Anti-Sybil measures (Issue #117):
        - More routers from same IP = harder PoW
        - Higher network-wide registration rate = harder PoW (adaptive)
        """
        count = self._ip_router_count.get(source_ip, 0)
        
        # Base difficulty based on IP router count
        if count == 0:
            base_difficulty = self.config.pow_difficulty_base
        elif count == 1:
            base_difficulty = self.config.pow_difficulty_second
        else:
            base_difficulty = self.config.pow_difficulty_third_plus
        
        # Apply adaptive PoW based on network registration rate
        return self.sybil_resistance.get_adaptive_pow_difficulty(base_difficulty, source_ip)
    
    def _verify_signature(
        self,
        router_id: str,
        data: Dict[str, Any],
        signature: str,
    ) -> bool:
        """
        Verify Ed25519 signature of registration data.
        
        Args:
            router_id: Hex-encoded Ed25519 public key
            data: The registration data that was signed
            signature: Hex-encoded signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        if not self.config.verify_signatures:
            return True
        
        try:
            # Parse public key from router_id (hex-encoded)
            public_key_bytes = bytes.fromhex(router_id)
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            
            # Reconstruct the signed message (exclude signature from data)
            signed_data = {k: v for k, v in data.items() if k != "signature"}
            message = json.dumps(signed_data, sort_keys=True, separators=(',', ':')).encode()
            
            # Verify signature
            signature_bytes = bytes.fromhex(signature)
            public_key.verify(signature_bytes, message)
            
            return True
            
        except (ValueError, InvalidSignature) as e:
            logger.warning(f"Signature verification failed for {router_id[:20]}...: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error verifying signature: {e}")
            return False
    
    def _verify_pow(
        self,
        router_id: str,
        proof_of_work: Dict[str, Any],
        required_difficulty: int,
    ) -> bool:
        """
        Verify proof-of-work for anti-Sybil protection.
        
        PoW format:
        {
            "challenge": "<seed-provided or router-generated challenge>",
            "nonce": <integer nonce>,
            "difficulty": <difficulty level achieved>
        }
        
        Verification: sha256(challenge || nonce || router_id) must have
        `required_difficulty` leading zero bits.
        
        Args:
            router_id: The router's public key (hex)
            proof_of_work: The PoW proof dict
            required_difficulty: Number of leading zero bits required
            
        Returns:
            True if PoW is valid, False otherwise
        """
        if not self.config.verify_pow:
            return True
        
        if not proof_of_work:
            logger.warning(f"Missing proof_of_work for {router_id[:20]}...")
            return False
        
        try:
            challenge = proof_of_work.get("challenge", "")
            nonce = proof_of_work.get("nonce", 0)
            
            # Construct the hash input
            hash_input = f"{challenge}{nonce}{router_id}".encode()
            hash_result = hashlib.sha256(hash_input).digest()
            
            # Count leading zero bits
            leading_zeros = 0
            for byte in hash_result:
                if byte == 0:
                    leading_zeros += 8
                else:
                    # Count leading zeros in this byte
                    for i in range(7, -1, -1):
                        if byte & (1 << i):
                            break
                        leading_zeros += 1
                    break
            
            if leading_zeros >= required_difficulty:
                return True
            else:
                logger.warning(
                    f"PoW insufficient for {router_id[:20]}...: "
                    f"got {leading_zeros} bits, need {required_difficulty}"
                )
                return False
                
        except Exception as e:
            logger.error(f"PoW verification error: {e}")
            return False
    
    async def _probe_endpoint(self, endpoint: str) -> bool:
        """
        Probe router endpoint to verify reachability.
        
        Connects to the router's health endpoint to verify it's accessible.
        
        Args:
            endpoint: "host:port" string
            
        Returns:
            True if endpoint is reachable, False otherwise
        """
        if not self.config.probe_endpoints:
            return True
        
        try:
            # Parse endpoint
            port: int
            if ":" in endpoint:
                host, port_str = endpoint.rsplit(":", 1)
                port = int(port_str)
            else:
                host = endpoint
                port = 8471  # Default router port
            
            # Try HTTP health check
            url = f"http://{host}:{port}/health"
            timeout = aiohttp.ClientTimeout(total=self.config.probe_timeout_seconds)
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        logger.debug(f"Endpoint probe successful: {endpoint}")
                        return True
                    else:
                        logger.warning(
                            f"Endpoint probe failed for {endpoint}: "
                            f"status {response.status}"
                        )
                        return False
                        
        except asyncio.TimeoutError:
            logger.warning(f"Endpoint probe timeout for {endpoint}")
            return False
        except Exception as e:
            logger.warning(f"Endpoint probe failed for {endpoint}: {e}")
            return False
    
    def _determine_health_status(self, router: RouterRecord) -> str:
        """
        Determine router health status based on metrics.
        
        Returns: "healthy", "warning", or "degraded"
        """
        load_pct = router.capacity.get("current_load_pct", 0)
        uptime_pct = router.health.get("uptime_pct", 100)
        
        if load_pct > 90 or uptime_pct < 95:
            return "degraded"
        elif load_pct > 70 or uptime_pct < 99:
            return "warning"
        else:
            return "healthy"
    
    # -------------------------------------------------------------------------
    # HTTP HANDLERS
    # -------------------------------------------------------------------------
    
    async def handle_discover(self, request: web.Request) -> web.Response:
        """
        Handle discovery requests from nodes.
        
        POST /discover
        {
            "requested_count": 5,
            "preferences": {
                "region": "us-west",
                "features": ["ipv6", "quic"]
            }
        }
        
        Response:
        {
            "seed_id": "seed-abc123",
            "timestamp": 1706789012.345,
            "routers": [...],
            "other_seeds": ["https://seed2.valence.network"]
        }
        """
        try:
            data = await request.json()
        except Exception:
            data = {}
        
        requested_count = data.get("requested_count", 5)
        preferences = data.get("preferences", {})
        
        # Limit to reasonable range
        requested_count = max(1, min(requested_count, 20))
        
        routers = self.select_routers(requested_count, preferences)
        
        response = {
            "seed_id": self.seed_id,
            "timestamp": time.time(),
            "routers": [r.to_dict() for r in routers],
            "other_seeds": self.known_seeds,
        }
        
        logger.info(
            f"Discovery request: requested={requested_count}, "
            f"returned={len(routers)}, preferences={preferences}"
        )
        
        return web.json_response(response)
    
    async def handle_register(self, request: web.Request) -> web.Response:
        """
        Handle router registration with validation.
        
        POST /register
        {
            "router_id": "<hex-encoded Ed25519 public key>",
            "endpoints": ["192.168.1.100:8471"],
            "capacity": {"max_connections": 1000, "bandwidth_mbps": 100},
            "regions": ["us-west", "us-central"],
            "features": ["ipv6", "quic"],
            "proof_of_work": {"challenge": "...", "nonce": 12345, "difficulty": 16},
            "timestamp": 1706789012.345,
            "signature": "<hex-encoded Ed25519 signature>"
        }
        
        Registration flow:
        1. Validate required fields
        2. Verify Ed25519 signature
        3. Check proof-of-work (anti-Sybil)
        4. Probe endpoint reachability
        5. Register router
        """
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"status": "rejected", "reason": "invalid_json", "detail": str(e)},
                status=400
            )
        
        # Get source IP for PoW difficulty calculation
        source_ip = request.remote or "unknown"
        
        # Validate required fields
        router_id = data.get("router_id")
        if not router_id:
            return web.json_response(
                {"status": "rejected", "reason": "missing_router_id"},
                status=400
            )
        
        endpoints = data.get("endpoints", [])
        if not endpoints:
            return web.json_response(
                {"status": "rejected", "reason": "missing_endpoints"},
                status=400
            )
        
        signature = data.get("signature", "")
        proof_of_work = data.get("proof_of_work")
        
        # Check if updating existing router (skip some verifications for updates)
        is_update = router_id in self.router_registry
        
        # Sybil resistance: Rate limiting check (Issue #117)
        # Only apply rate limiting to new registrations
        if not is_update:
            allowed, reason = self.sybil_resistance.check_registration(router_id, source_ip, endpoints)
            if not allowed:
                logger.warning(
                    f"Registration blocked by Sybil resistance: {router_id[:20]}... "
                    f"reason={reason}, source_ip={source_ip}"
                )
                self.sybil_resistance.on_registration_failure(router_id, source_ip)
                return web.json_response(
                    {
                        "status": "rejected",
                        "reason": "rate_limited",
                        "detail": reason,
                    },
                    status=429  # Too Many Requests
                )
        
        # Verify Ed25519 signature
        if not self._verify_signature(router_id, data, signature):
            if not is_update:
                self.sybil_resistance.on_registration_failure(router_id, source_ip)
            return web.json_response(
                {"status": "rejected", "reason": "invalid_signature"},
                status=400
            )
        
        # Check proof of work (only for new registrations)
        if not is_update:
            required_difficulty = self._get_pow_difficulty(source_ip)
            if not self._verify_pow(router_id, proof_of_work, required_difficulty):
                self.sybil_resistance.on_registration_failure(router_id, source_ip)
                return web.json_response(
                    {
                        "status": "rejected",
                        "reason": "insufficient_pow",
                        "required_difficulty": required_difficulty,
                    },
                    status=400
                )
        
        # Probe endpoint reachability (only for new registrations or changed endpoints)
        if not is_update or endpoints != self.router_registry[router_id].endpoints:
            if not await self._probe_endpoint(endpoints[0]):
                return web.json_response(
                    {"status": "rejected", "reason": "unreachable"},
                    status=400
                )
        
        now = time.time()
        
        # Parse coordinates if provided
        coords_data = data.get("coordinates")
        if coords_data and isinstance(coords_data, (list, tuple)) and len(coords_data) == 2:
            coordinates = [float(coords_data[0]), float(coords_data[1])]
        else:
            coordinates = None
        
        # Create or update record
        record = RouterRecord(
            router_id=router_id,
            endpoints=endpoints,
            capacity=data.get("capacity", {}),
            health={
                "last_seen": now,
                "uptime_pct": 100.0,  # Assume healthy on registration
                "avg_latency_ms": data.get("avg_latency_ms", 0),
                "status": "healthy",
            },
            regions=data.get("regions", []),
            features=data.get("features", []),
            registered_at=now if not is_update else self.router_registry[router_id].registered_at,
            router_signature=signature,
            proof_of_work=proof_of_work,
            source_ip=source_ip,
            region=data.get("region"),  # ISO 3166-1 alpha-2 country code
            coordinates=coordinates,
        )
        
        self.router_registry[router_id] = record
        
        # Track IP router count for new registrations
        if not is_update:
            self._ip_router_count[source_ip] += 1
            # Notify Sybil resistance manager of successful registration (Issue #117)
            self.sybil_resistance.on_registration_success(router_id, source_ip, endpoints)
        
        action = "updated" if is_update else "registered"
        region_info = f", region={record.region}" if record.region else ""
        
        # Include reputation info in response for new registrations
        reputation_info = {}
        if not is_update:
            rep = self.sybil_resistance.reputation.get_reputation(router_id)
            if rep:
                reputation_info = {
                    "initial_reputation": rep.score,
                    "reputation_decay_hours": self.config.reputation_decay_period_hours,
                }
        
        logger.info(
            f"Router {action}: {router_id[:20]}... "
            f"endpoints={endpoints}, regions={record.regions}{region_info}, source_ip={source_ip}"
        )
        
        response = {
            "status": "accepted",
            "action": action,
            "router_id": router_id,
            "seed_id": self.seed_id,
        }
        response.update(reputation_info)
        
        return web.json_response(response)
    
    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """
        Handle router heartbeats.
        
        POST /heartbeat
        {
            "router_id": "<hex-encoded Ed25519 public key>",
            "current_connections": 350,
            "load_pct": 35.5,
            "messages_relayed": 12500,
            "uptime_pct": 99.8,
            "avg_latency_ms": 12.5,
            "timestamp": 1706789012.345,
            "signature": "<hex-encoded signature>"
        }
        
        Response includes health status: healthy/warning/degraded
        """
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"status": "error", "reason": "invalid_json", "detail": str(e)},
                status=400
            )
        
        router_id = data.get("router_id")
        if not router_id:
            return web.json_response(
                {"status": "error", "reason": "missing_router_id"},
                status=400
            )
        
        # Check if router is registered
        if router_id not in self.router_registry:
            return web.json_response(
                {"status": "error", "reason": "not_registered", "hint": "Call /register first"},
                status=404
            )
        
        # Verify signature if provided
        signature = data.get("signature", "")
        if signature and self.config.verify_signatures:
            if not self._verify_signature(router_id, data, signature):
                return web.json_response(
                    {"status": "error", "reason": "invalid_signature"},
                    status=400
                )
        
        # Update health and capacity
        record = self.router_registry[router_id]
        now = time.time()
        
        record.health["last_seen"] = now
        
        # Record heartbeat with health monitor
        self.health_monitor.record_heartbeat(router_id, data)
        
        if "uptime_pct" in data:
            record.health["uptime_pct"] = float(data["uptime_pct"])
        if "avg_latency_ms" in data:
            record.health["avg_latency_ms"] = float(data["avg_latency_ms"])
        
        # Update capacity metrics
        if "load_pct" in data:
            record.capacity["current_load_pct"] = float(data["load_pct"])
        elif "current_load_pct" in data:
            record.capacity["current_load_pct"] = float(data["current_load_pct"])
            
        if "current_connections" in data:
            record.capacity["active_connections"] = int(data["current_connections"])
        elif "active_connections" in data:
            record.capacity["active_connections"] = int(data["active_connections"])
        
        # Store max_connections for load ratio calculation
        if "max_connections" in data:
            record.capacity["max_connections"] = int(data["max_connections"])
            
        if "messages_relayed" in data:
            record.capacity["messages_relayed"] = int(data["messages_relayed"])
        
        # New load metrics for distributed load balancing
        if "messages_per_sec" in data:
            record.capacity["messages_per_sec"] = float(data["messages_per_sec"])
        
        if "queue_depth" in data:
            record.capacity["queue_depth"] = int(data["queue_depth"])
        
        # Update endpoints if provided
        if "endpoints" in data:
            record.endpoints = data["endpoints"]
        
        # Determine health status
        health_status = self._determine_health_status(record)
        record.health["status"] = health_status
        
        # Sybil resistance: Analyze heartbeat for correlated behavior (Issue #117)
        sybil_flags = self.sybil_resistance.on_heartbeat(router_id, now, record.endpoints)
        
        # Get current reputation
        reputation = self.sybil_resistance.get_trust_factor(router_id)
        
        logger.debug(
            f"Heartbeat from {router_id[:20]}...: "
            f"load={record.capacity.get('current_load_pct')}%, "
            f"connections={record.capacity.get('active_connections')}, "
            f"status={health_status}, reputation={reputation:.3f}"
        )
        
        response = {
            "status": "ok",
            "health_status": health_status,
            "router_id": router_id,
            "seed_id": self.seed_id,
            "next_heartbeat_in": 300,  # 5 minutes
            "reputation": round(reputation, 3),
        }
        
        # Include warnings if suspicious behavior detected
        if sybil_flags:
            response["warnings"] = sybil_flags
        
        return web.json_response(response)
    
    async def handle_status(self, request: web.Request) -> web.Response:
        """
        Handle status requests (public endpoint).
        
        GET /status
        """
        now = time.time()
        healthy_count = sum(
            1 for r in self.router_registry.values()
            if self._is_healthy(r, now)
        )
        
        # Get health monitor stats
        health_stats = self.health_monitor.get_stats()
        
        # Get peer manager stats
        peer_stats = self.peer_manager.get_peer_stats() if self._peer_manager else {}
        
        # Get Sybil resistance stats (Issue #117)
        sybil_stats = self.sybil_resistance.get_stats()
        
        # Get misbehavior stats (Issue #119)
        misbehavior_stats = self.get_misbehavior_stats()
        
        # Get revocation stats (Issue #121)
        revocation_stats = self.revocation_manager.get_stats() if self._revocation_manager else {}
        
        return web.json_response({
            "seed_id": self.seed_id,
            "status": "running" if self._running else "stopped",
            "timestamp": now,
            "routers": {
                "total": len(self.router_registry),
                "healthy": healthy_count,
            },
            "health_monitor": health_stats,
            "sybil_resistance": sybil_stats,
            "misbehavior": misbehavior_stats,
            "revocations": revocation_stats,
            "known_seeds": len(self.known_seeds),
            "peering": peer_stats,
        })
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint for load balancers."""
        return web.json_response({"status": "ok"})
    
    # -------------------------------------------------------------------------
    # MISBEHAVIOR REPORTS (Issue #119)
    # -------------------------------------------------------------------------
    
    async def handle_misbehavior_report(self, request: web.Request) -> web.Response:
        """
        Handle misbehavior report from nodes (Issue #119).
        
        POST /report_misbehavior
        {
            "report_id": "uuid",
            "reporter_id": "node public key hex",
            "router_id": "router public key hex",
            "misbehavior_type": "message_drop|message_delay|ack_failure|...",
            "evidence": [...],
            "metrics": {...},
            "severity": 0.0-1.0,
            "timestamp": 1234567890.123,
            "signature": "hex signature"
        }
        
        Seeds aggregate reports from multiple nodes. When a router receives
        reports from enough unique reporters, it gets flagged.
        """
        if not self.config.misbehavior_reports_enabled:
            return web.json_response(
                {"status": "rejected", "reason": "misbehavior_reports_disabled"},
                status=400
            )
        
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"status": "error", "reason": "invalid_json", "detail": str(e)},
                status=400
            )
        
        router_id = data.get("router_id")
        reporter_id = data.get("reporter_id")
        
        if not router_id:
            return web.json_response(
                {"status": "error", "reason": "missing_router_id"},
                status=400
            )
        
        if not reporter_id:
            return web.json_response(
                {"status": "error", "reason": "missing_reporter_id"},
                status=400
            )
        
        # Verify reporter signature if enabled
        if self.config.misbehavior_verify_reporter_signature:
            signature = data.get("signature", "")
            if signature:
                # Create data for signature verification (exclude signature field)
                verify_data = {k: v for k, v in data.items() if k != "signature"}
                if not self._verify_signature(reporter_id, verify_data, signature):
                    return web.json_response(
                        {"status": "error", "reason": "invalid_signature"},
                        status=400
                    )
        
        # Store the report
        now = time.time()
        
        # Initialize storage for this router if needed
        if router_id not in self._misbehavior_reports:
            self._misbehavior_reports[router_id] = {}
        
        # Store report from this reporter
        self._misbehavior_reports[router_id][reporter_id] = {
            "report_id": data.get("report_id"),
            "misbehavior_type": data.get("misbehavior_type"),
            "severity": data.get("severity", 0.0),
            "timestamp": data.get("timestamp", now),
            "received_at": now,
        }
        
        # Prune old reports outside the window
        window_cutoff = now - self.config.misbehavior_report_window_seconds
        self._misbehavior_reports[router_id] = {
            r_id: r_data 
            for r_id, r_data in self._misbehavior_reports[router_id].items()
            if r_data.get("received_at", 0) >= window_cutoff
        }
        
        # Limit reports per router
        if len(self._misbehavior_reports[router_id]) > self.config.misbehavior_max_reports_per_router:
            # Keep most recent reports
            sorted_reports = sorted(
                self._misbehavior_reports[router_id].items(),
                key=lambda x: x[1].get("received_at", 0),
                reverse=True
            )
            self._misbehavior_reports[router_id] = dict(
                sorted_reports[:self.config.misbehavior_max_reports_per_router]
            )
        
        # Check if router should be flagged
        reports = self._misbehavior_reports[router_id]
        unique_reporters = len(reports)
        
        router_flagged = False
        if unique_reporters >= self.config.misbehavior_min_reports_to_flag:
            # Calculate average severity
            severities = [r.get("severity", 0) for r in reports.values()]
            avg_severity = sum(severities) / len(severities) if severities else 0
            
            if avg_severity >= self.config.misbehavior_flag_severity_threshold:
                # Flag the router
                if router_id not in self._misbehavior_flagged_routers:
                    self._misbehavior_flagged_routers[router_id] = now
                    router_flagged = True
                    logger.warning(
                        f"FLAGGED router {router_id[:20]}... for misbehavior: "
                        f"{unique_reporters} reporters, avg_severity={avg_severity:.2f}"
                    )
        
        logger.info(
            f"Received misbehavior report for router {router_id[:20]}... "
            f"from {reporter_id[:20]}...: type={data.get('misbehavior_type')}, "
            f"severity={data.get('severity')}"
        )
        
        return web.json_response({
            "status": "accepted",
            "report_id": data.get("report_id"),
            "router_id": router_id,
            "reports_for_router": unique_reporters,
            "router_flagged": router_flagged,
        })
    
    def is_router_misbehavior_flagged(self, router_id: str) -> bool:
        """
        Check if a router has been flagged for misbehavior.
        
        Args:
            router_id: Router to check
            
        Returns:
            True if router is flagged
        """
        return router_id in self._misbehavior_flagged_routers
    
    def get_misbehavior_stats(self) -> Dict[str, Any]:
        """
        Get misbehavior report statistics.
        
        Returns:
            Dict with misbehavior tracking stats
        """
        now = time.time()
        return {
            "enabled": self.config.misbehavior_reports_enabled,
            "routers_with_reports": len(self._misbehavior_reports),
            "flagged_routers": len(self._misbehavior_flagged_routers),
            "flagged_router_ids": list(self._misbehavior_flagged_routers.keys()),
            "config": {
                "min_reports_to_flag": self.config.misbehavior_min_reports_to_flag,
                "report_window_seconds": self.config.misbehavior_report_window_seconds,
                "flag_severity_threshold": self.config.misbehavior_flag_severity_threshold,
            },
        }
    
    def clear_router_misbehavior_flag(self, router_id: str) -> bool:
        """
        Clear the misbehavior flag for a router.
        
        Args:
            router_id: Router to unflag
            
        Returns:
            True if flag was cleared, False if router wasn't flagged
        """
        if router_id in self._misbehavior_flagged_routers:
            del self._misbehavior_flagged_routers[router_id]
            logger.info(f"Cleared misbehavior flag for router {router_id[:20]}...")
            return True
        return False
    
    # -------------------------------------------------------------------------
    # SEED REVOCATION (Issue #121)
    # -------------------------------------------------------------------------
    
    async def handle_revoke_seed(self, request: web.Request) -> web.Response:
        """
        Handle seed revocation submission (Issue #121).
        
        POST /revoke_seed
        {
            "revocation_id": "uuid",
            "seed_id": "seed public key hex",
            "reason": "key_compromise|malicious_behavior|retired|admin_action|security_audit",
            "reason_detail": "Optional detailed explanation",
            "timestamp": 1234567890.123,
            "effective_at": 1234567890.123,
            "issuer_id": "issuer public key hex (seed itself or authority)",
            "signature": "Ed25519 signature hex"
        }
        
        The revocation must be signed by either:
        - The seed being revoked (self-revocation)
        - A trusted authority (from config.seed_revocation_trusted_authorities)
        """
        if not self.config.seed_revocation_enabled:
            return web.json_response(
                {"status": "rejected", "reason": "revocation_disabled"},
                status=400
            )
        
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"status": "error", "reason": "invalid_json", "detail": str(e)},
                status=400
            )
        
        seed_id = data.get("seed_id")
        if not seed_id:
            return web.json_response(
                {"status": "error", "reason": "missing_seed_id"},
                status=400
            )
        
        # Add the revocation
        success, error = self.revocation_manager.add_revocation(data, source="direct")
        
        if not success:
            return web.json_response(
                {"status": "rejected", "reason": error},
                status=400
            )
        
        # Get the record for response
        record = self.revocation_manager.get_revocation(seed_id)
        
        logger.info(
            f"Accepted seed revocation: {seed_id[:20]}... "
            f"reason={data.get('reason')}"
        )
        
        return web.json_response({
            "status": "accepted",
            "seed_id": seed_id,
            "revocation_id": record.revocation_id if record else data.get("revocation_id"),
            "effective_at": record.effective_at if record else data.get("effective_at"),
        })
    
    async def handle_get_revocations(self, request: web.Request) -> web.Response:
        """
        Get current seed revocations (Issue #121).
        
        GET /revocations
        GET /revocations?seed_id=<seed_id>
        
        Returns list of all revocations, or single revocation if seed_id specified.
        """
        if not self.config.seed_revocation_enabled:
            return web.json_response(
                {"status": "error", "reason": "revocation_disabled"},
                status=400
            )
        
        seed_id = request.query.get("seed_id")
        
        if seed_id:
            # Return single revocation
            record = self.revocation_manager.get_revocation(seed_id)
            if record is None:
                return web.json_response({
                    "revoked": False,
                    "seed_id": seed_id,
                })
            
            return web.json_response({
                "revoked": True,
                "seed_id": seed_id,
                "revocation": record.to_dict(),
            })
        
        # Return all revocations
        revocations = self.revocation_manager.get_all_revocations()
        revoked_ids = self.revocation_manager.get_revoked_seed_ids()
        
        return web.json_response({
            "total": len(revocations),
            "effective": len(revoked_ids),
            "revoked_seed_ids": list(revoked_ids),
            "revocations": [r.to_dict() for r in revocations],
        })
    
    def is_seed_revoked(self, seed_id: str) -> bool:
        """
        Check if a seed is revoked (Issue #121).
        
        Args:
            seed_id: The seed ID to check
            
        Returns:
            True if the seed is revoked
        """
        return self.revocation_manager.is_seed_revoked(seed_id)
    
    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------
    
    def _create_app(self) -> web.Application:
        """Create the aiohttp application."""
        app = web.Application()
        
        # Discovery endpoints
        app.router.add_post("/discover", self.handle_discover)
        app.router.add_post("/register", self.handle_register)
        app.router.add_post("/heartbeat", self.handle_heartbeat)
        
        # Misbehavior report endpoint (Issue #119)
        app.router.add_post("/report_misbehavior", self.handle_misbehavior_report)
        
        # Seed revocation endpoints (Issue #121)
        app.router.add_post("/revoke_seed", self.handle_revoke_seed)
        app.router.add_get("/revocations", self.handle_get_revocations)
        
        # Gossip endpoints (seed-to-seed)
        app.router.add_post("/gossip/exchange", self.peer_manager.handle_gossip_exchange)
        
        # Status endpoints
        app.router.add_get("/status", self.handle_status)
        app.router.add_get("/health", self.handle_health)
        
        return app
    
    async def start(self) -> None:
        """Start the seed node server."""
        if self._running:
            logger.warning("Seed node already running")
            return
        
        self._app = self._create_app()
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        
        self._site = web.TCPSite(
            self._runner,
            self.config.host,
            self.config.port,
        )
        await self._site.start()
        
        # Start health monitoring
        await self.health_monitor.start()
        
        # Start seed peering / gossip
        await self.peer_manager.start()
        
        # Start revocation manager (Issue #121)
        await self.revocation_manager.start()
        
        self._running = True
        logger.info(
            f"Seed node {self.seed_id} listening on "
            f"{self.config.host}:{self.config.port}"
        )
    
    async def stop(self) -> None:
        """Stop the seed node server."""
        if not self._running:
            return
        
        # Stop revocation manager (Issue #121)
        if self._revocation_manager:
            await self._revocation_manager.stop()
        
        # Stop seed peering / gossip
        if self._peer_manager:
            await self._peer_manager.stop()
        
        # Stop health monitoring
        if self._health_monitor:
            await self._health_monitor.stop()
        
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        
        self._running = False
        self._app = None
        self._runner = None
        self._site = None
        
        logger.info(f"Seed node {self.seed_id} stopped")
    
    async def run_forever(self) -> None:
        """Start and run until interrupted."""
        await self.start()
        
        try:
            # Keep running until cancelled
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_seed_node(
    host: str = "0.0.0.0",
    port: int = 8470,
    known_seeds: Optional[List[str]] = None,
    peer_seeds: Optional[List[str]] = None,
    **kwargs,
) -> SeedNode:
    """Create a seed node with the given configuration."""
    config = SeedConfig(
        host=host,
        port=port,
        known_seeds=known_seeds or [],
        peer_seeds=peer_seeds or [],
        **kwargs,
    )
    return SeedNode(config=config)


async def run_seed_node(
    host: str = "0.0.0.0",
    port: int = 8470,
    **kwargs,
) -> None:
    """Create and run a seed node (convenience function)."""
    node = create_seed_node(host=host, port=port, **kwargs)
    await node.run_forever()
