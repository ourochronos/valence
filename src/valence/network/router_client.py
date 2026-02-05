"""
Router Client - Handles router interaction for NodeClient.

This module manages:
- Router selection (weighted by health)
- Router rotation for eclipse attack resistance
- Back-pressure handling
- Failover logic

Extracted from NodeClient as part of god class decomposition (Issue #128).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .discovery import DiscoveryClient, RouterInfo
    from .node import RouterConnection, FailoverState, PendingAck
    from .connection_manager import ConnectionManager

logger = logging.getLogger(__name__)


@dataclass
class RouterClientConfig:
    """Configuration for RouterClient."""
    
    # Failover config
    initial_cooldown: float = 60.0  # 1 minute
    max_cooldown: float = 3600.0  # 1 hour
    reconnect_delay: float = 1.0
    failover_connect_timeout: float = 3.0
    
    # Router rotation (Eclipse mitigation - Issue #118)
    rotation_enabled: bool = True
    rotation_interval: float = 3600.0  # 1 hour
    rotation_max_age: float = 7200.0  # 2 hours
    
    # Selection weights
    own_observation_weight: float = 0.7
    peer_observation_weight: float = 0.3


class RouterClient:
    """
    Handles router selection and failover for a node.
    
    Responsible for:
    - Weighted router selection based on health
    - Router rotation for eclipse resistance
    - Back-pressure handling
    - Failover to alternative routers
    """
    
    def __init__(
        self,
        connection_manager: "ConnectionManager",
        discovery: "DiscoveryClient",
        config: Optional[RouterClientConfig] = None,
        get_aggregated_health: Optional[Callable[[str], float]] = None,
        is_router_flagged: Optional[Callable[[str], bool]] = None,
        flagged_router_penalty: float = 0.1,
    ):
        """
        Initialize the RouterClient.
        
        Args:
            connection_manager: ConnectionManager for connections
            discovery: DiscoveryClient for finding routers
            config: Router client configuration
            get_aggregated_health: Function to get health score for router
            is_router_flagged: Function to check if router is flagged
            flagged_router_penalty: Penalty multiplier for flagged routers
        """
        self.connection_manager = connection_manager
        self.discovery = discovery
        self.config = config or RouterClientConfig()
        
        # Callbacks
        self.get_aggregated_health = get_aggregated_health
        self.is_router_flagged = is_router_flagged
        self.flagged_router_penalty = flagged_router_penalty
        
        # Rotation state
        self._last_rotation: float = 0.0
        
        # Direct mode (graceful degradation)
        self.direct_mode: bool = False
        
        # Statistics
        self._stats: Dict[str, int] = {
            "failovers": 0,
            "routers_rotated": 0,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router client statistics."""
        return {
            **self._stats,
            "direct_mode": self.direct_mode,
            "last_rotation": self._last_rotation,
        }
    
    def select_router(self, exclude_back_pressured: bool = True) -> Optional["RouterInfo"]:
        """
        Select the best router based on health metrics.
        
        Uses weighted random selection combining:
        - Own observations (direct connection metrics)
        - Peer observations (gossip from other nodes)
        - Back-pressure status
        
        Args:
            exclude_back_pressured: If True, exclude routers under back-pressure
            
        Returns:
            Selected RouterInfo or None if no routers available
        """
        connections = self.connection_manager.connections
        if not connections:
            return None
        
        # Filter to healthy connections
        candidates = [
            conn for conn in connections.values()
            if not conn.websocket.closed
        ]
        
        if not candidates:
            return None
        
        # Filter out routers under back-pressure if requested
        if exclude_back_pressured:
            non_bp_candidates = [
                conn for conn in candidates
                if not conn.is_under_back_pressure
            ]
            
            if non_bp_candidates:
                candidates = non_bp_candidates
            elif candidates:
                logger.warning(
                    f"All {len(candidates)} routers under back-pressure, "
                    "selecting least loaded"
                )
                candidates.sort(key=lambda c: c.back_pressure_until)
        
        # Calculate weights using health scores
        weights = []
        for conn in candidates:
            # Get aggregated health score if available
            if self.get_aggregated_health:
                aggregated = self.get_aggregated_health(conn.router.router_id)
            else:
                aggregated = conn.health_score
            
            # Blend aggregated and direct scores
            direct_score = conn.health_score
            combined_score = (aggregated * 0.6) + (direct_score * 0.4)
            
            # Penalty for back-pressured routers
            if conn.is_under_back_pressure:
                combined_score *= 0.1
            
            # Penalty for flagged routers
            if self.is_router_flagged and self.is_router_flagged(conn.router.router_id):
                combined_score *= self.flagged_router_penalty
            
            weights.append(max(0.01, combined_score))
        
        # Weighted random selection
        selected = random.choices(candidates, weights=weights, k=1)[0]
        return selected.router
    
    async def handle_router_failure(
        self,
        router_id: str,
        on_retry_messages: Optional[Callable] = None,
        failure_type: str = "connection",
        error_code: Optional[str] = None,
    ) -> bool:
        """
        Handle router failure with intelligent failover.
        
        Args:
            router_id: The failed router's ID
            on_retry_messages: Callback to retry pending messages
            failure_type: Type of failure
            error_code: Optional error code
            
        Returns:
            True if failover was successful
        """
        from .node import FailoverState
        
        conn = self.connection_manager.connections.get(router_id)
        if not conn:
            return False
        
        fail_time = time.time()
        self._stats["failovers"] += 1
        logger.warning(f"Router {router_id[:16]}... failed, initiating failover")
        
        # Update failover state with exponential backoff
        failover_states = self.connection_manager.failover_states
        if router_id not in failover_states:
            failover_states[router_id] = FailoverState(
                router_id=router_id,
                failed_at=fail_time,
                fail_count=1,
                cooldown_until=fail_time + self.config.initial_cooldown,
                queued_messages=[],
            )
        else:
            state = failover_states[router_id]
            state.fail_count += 1
            state.failed_at = fail_time
            cooldown = min(
                self.config.initial_cooldown * (2 ** (state.fail_count - 1)),
                self.config.max_cooldown
            )
            state.cooldown_until = fail_time + cooldown
        
        # Close the failed connection
        await self.connection_manager.close_connection(router_id, conn)
        
        # Discover alternative routers
        exclude_ids = [router_id] + [
            r_id for r_id, state in failover_states.items()
            if state.is_in_cooldown() and r_id != router_id
        ]
        exclude_ids.extend(self.connection_manager.connections.keys())
        
        try:
            alternatives = await self.discovery.discover_routers(
                count=3,
                preferences={"exclude": exclude_ids},
                force_refresh=True,
            )
        except Exception as e:
            logger.warning(f"Failed to discover alternative routers: {e}")
            alternatives = []
        
        # Sort by health
        if alternatives:
            alternatives.sort(
                key=lambda r: (
                    -r.health.get("uptime_pct", 0),
                    r.health.get("avg_latency_ms", 999),
                ),
            )
        
        # Try to connect to alternatives
        connected = False
        for router in alternatives:
            if self.connection_manager.config.enforce_ip_diversity:
                if not self.connection_manager.check_ip_diversity(router):
                    continue
            
            try:
                await asyncio.wait_for(
                    self.connection_manager.connect_to_router(router),
                    timeout=self.config.failover_connect_timeout,
                )
                connected = True
                logger.info(f"Failover successful: connected to {router.router_id[:16]}...")
                break
            except (asyncio.TimeoutError, Exception) as e:
                logger.warning(f"Failover connection failed: {e}")
                continue
        
        # Retry pending messages if connected
        if connected and on_retry_messages:
            try:
                await on_retry_messages()
            except Exception as e:
                logger.warning(f"Failed to retry messages: {e}")
        
        # Handle graceful degradation
        if not connected:
            self._enable_direct_mode()
        elif self.direct_mode:
            self._disable_direct_mode()
        
        # Ensure minimum connections
        if self.connection_manager.connection_count < self.connection_manager.config.min_connections:
            await asyncio.sleep(self.config.reconnect_delay)
            await self.connection_manager.ensure_connections()
        
        return connected
    
    async def rotate_router(self, router_id: str, reason: str = "periodic") -> bool:
        """
        Rotate a specific router connection.
        
        Disconnects from the router and connects to a new, diverse router.
        
        Args:
            router_id: Router to rotate out
            reason: Reason for rotation
            
        Returns:
            True if rotation was successful
        """
        conn = self.connection_manager.connections.get(router_id)
        if not conn:
            return False
        
        self._last_rotation = time.time()
        self._stats["routers_rotated"] += 1
        
        # Disconnect from old router
        await self.connection_manager.close_connection(router_id, conn)
        
        # Find a new diverse router
        exclude_ids = set(self.connection_manager.connections.keys())
        exclude_ids.add(router_id)
        
        for r_id, state in self.connection_manager.failover_states.items():
            if state.is_in_cooldown():
                exclude_ids.add(r_id)
        
        try:
            routers = await self.discovery.discover_routers(
                count=5,
                preferences={"exclude": list(exclude_ids)},
            )
            
            for router in routers:
                if not self.connection_manager.check_ip_diversity(router):
                    continue
                if not self.connection_manager.check_asn_diversity(router):
                    continue
                
                try:
                    await self.connection_manager.connect_to_router(router)
                    logger.info(
                        f"Rotation complete: {router_id[:16]}... -> {router.router_id[:16]}... "
                        f"(reason: {reason})"
                    )
                    return True
                except Exception as e:
                    logger.debug(f"Failed to connect to rotation candidate: {e}")
                    continue
            
            logger.warning(f"Could not find diverse replacement for {router_id[:16]}...")
            return False
            
        except Exception as e:
            logger.warning(f"Rotation failed: {e}")
            return False
    
    def handle_back_pressure(
        self,
        conn: "RouterConnection",
        active: bool,
        load_pct: float = 0.0,
        retry_after_ms: int = 1000,
        reason: str = "",
    ) -> None:
        """
        Handle back-pressure signal from router.
        
        Args:
            conn: The router connection
            active: Whether back-pressure is active
            load_pct: Current load percentage
            retry_after_ms: Suggested retry delay
            reason: Reason for back-pressure
        """
        router_id = conn.router.router_id
        
        if active:
            conn.back_pressure_active = True
            conn.back_pressure_until = time.time() + (retry_after_ms / 1000)
            conn.back_pressure_retry_ms = retry_after_ms
            
            logger.warning(
                f"Router {router_id[:16]}... signaled BACK-PRESSURE: "
                f"load={load_pct:.1f}%, retry_after={retry_after_ms}ms"
            )
        else:
            conn.back_pressure_active = False
            conn.back_pressure_until = 0.0
            logger.info(f"Router {router_id[:16]}... RELEASED back-pressure")
    
    def _enable_direct_mode(self) -> None:
        """Enable direct mode as graceful degradation."""
        if not self.direct_mode:
            self.direct_mode = True
            logger.warning(
                "No routers available - enabling direct mode "
                "(will attempt P2P for known peers)"
            )
    
    def _disable_direct_mode(self) -> None:
        """Disable direct mode when routers become available."""
        if self.direct_mode:
            self.direct_mode = False
            logger.info("Router connection restored - disabling direct mode")
    
    async def check_rotation_needed(self) -> Optional[str]:
        """
        Check if any router needs rotation due to age.
        
        Returns:
            Router ID that needs rotation, or None
        """
        if not self.config.rotation_enabled:
            return None
        
        now = time.time()
        connections = self.connection_manager.connections
        timestamps = self.connection_manager._connection_timestamps
        
        # Check for connections exceeding max age
        for router_id, conn in connections.items():
            conn_time = timestamps.get(router_id, conn.connected_at)
            age = now - conn_time
            
            if age >= self.config.rotation_max_age:
                return router_id
        
        # Check periodic rotation
        if len(connections) > self.connection_manager.config.min_connections:
            time_since_rotation = now - self._last_rotation
            
            if time_since_rotation >= self.config.rotation_interval:
                # Return oldest connection
                return min(
                    connections.keys(),
                    key=lambda rid: timestamps.get(rid, connections[rid].connected_at)
                )
        
        return None
