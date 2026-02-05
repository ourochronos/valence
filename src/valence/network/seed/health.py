"""
Seed node health monitoring.

Tracks router health via heartbeats and active probing.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from .core import SeedNode
    from .registry import RouterRecord

logger = logging.getLogger(__name__)

# Use cryptographically secure RNG for router sampling
_secure_random = secrets.SystemRandom()


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
            sample = _secure_random.sample(routers, sample_size)
            
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
