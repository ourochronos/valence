"""
Connection Manager - Handles connection lifecycle for NodeClient.

This module manages:
- Multi-router connection establishment and teardown
- IP/ASN diversity enforcement for eclipse attack resistance
- Connection state tracking (subnets, ASNs)
- Failover state management

Extracted from NodeClient as part of god class decomposition (Issue #128).
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import aiohttp
from aiohttp import WSMsgType

if TYPE_CHECKING:
    from .discovery import DiscoveryClient, RouterInfo
    from .node import RouterConnection, FailoverState

logger = logging.getLogger(__name__)


@dataclass
class ConnectionManagerConfig:
    """Configuration for ConnectionManager."""
    
    min_connections: int = 3
    target_connections: int = 5
    max_connections: int = 8
    
    # IP diversity
    enforce_ip_diversity: bool = True
    ip_diversity_prefix: int = 16  # /16 subnet diversity
    
    # ASN diversity (Eclipse mitigation - Issue #118)
    min_diverse_subnets: int = 2
    min_diverse_asns: int = 2
    asn_diversity_enabled: bool = True


class ConnectionManager:
    """
    Manages router connections for a node.
    
    Handles:
    - Connection establishment and teardown
    - IP and ASN diversity enforcement
    - Subnet tracking
    - Connection state
    """
    
    def __init__(
        self,
        node_id: str,
        discovery: "DiscoveryClient",
        config: Optional[ConnectionManagerConfig] = None,
        on_connection_established: Optional[Callable[..., Any]] = None,
        on_connection_lost: Optional[Callable[..., Any]] = None,
    ):
        """
        Initialize the ConnectionManager.
        
        Args:
            node_id: This node's ID (Ed25519 public key hex)
            discovery: DiscoveryClient for finding routers
            config: Connection configuration
            on_connection_established: Callback when connection is established
            on_connection_lost: Callback when connection is lost
        """
        self.node_id = node_id
        self.discovery = discovery
        self.config = config or ConnectionManagerConfig()
        
        # Callbacks
        self.on_connection_established = on_connection_established
        self.on_connection_lost = on_connection_lost
        
        # Connection state
        self.connections: Dict[str, "RouterConnection"] = {}
        self._connected_subnets: Set[str] = set()
        self._connected_asns: Set[str] = set()
        self._connection_timestamps: Dict[str, float] = {}
        
        # Failover state
        self.failover_states: Dict[str, "FailoverState"] = {}
        
        # Statistics
        self._stats: Dict[str, int] = {
            "connections_established": 0,
            "connections_failed": 0,
            "diversity_rejections": 0,
        }
    
    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.connections)
    
    @property
    def is_healthy(self) -> bool:
        """Check if we have minimum required connections."""
        return self.connection_count >= self.config.min_connections
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            **self._stats,
            "active_connections": self.connection_count,
            "connected_subnets": len(self._connected_subnets),
            "connected_asns": len(self._connected_asns),
            "routers_in_cooldown": sum(
                1 for state in self.failover_states.values()
                if state.is_in_cooldown()
            ),
        }
    
    async def ensure_connections(self) -> None:
        """Ensure we have enough router connections."""
        attempts = 0
        max_attempts = 3
        
        while self.connection_count < self.config.target_connections and attempts < max_attempts:
            needed = self.config.target_connections - self.connection_count
            
            # Get excluded router IDs (already connected + in cooldown)
            excluded_ids = set(self.connections.keys())
            
            # Also exclude routers in cooldown
            for router_id, state in self.failover_states.items():
                if state.is_in_cooldown():
                    excluded_ids.add(router_id)
                    logger.debug(
                        f"Excluding router {router_id[:16]}... from discovery "
                        f"(in cooldown for {state.remaining_cooldown():.1f}s)"
                    )
            
            # Discover routers
            try:
                routers = await self.discovery.discover_routers(
                    count=needed * 2,  # Request extra for filtering
                    preferences={"region": "any"},
                )
            except Exception as e:
                logger.warning(f"Router discovery failed: {e}")
                attempts += 1
                continue
            
            # Filter and connect
            for router in routers:
                if self.connection_count >= self.config.target_connections:
                    break
                
                if router.router_id in excluded_ids:
                    continue
                
                # Check IP diversity
                if self.config.enforce_ip_diversity:
                    if not self.check_ip_diversity(router):
                        logger.debug(
                            f"Skipping router {router.router_id[:16]}... "
                            f"(IP diversity check failed)"
                        )
                        continue
                
                # Check ASN diversity
                if not self.check_asn_diversity(router):
                    continue
                
                try:
                    await self.connect_to_router(router)
                except Exception as e:
                    logger.warning(
                        f"Failed to connect to router {router.router_id[:16]}...: {e}"
                    )
                    self._stats["connections_failed"] += 1
                    continue
            
            attempts += 1
        
        if self.connection_count < self.config.min_connections:
            logger.warning(
                f"Only {self.connection_count} connections established "
                f"(minimum: {self.config.min_connections})"
            )
    
    def check_ip_diversity(self, router: "RouterInfo") -> bool:
        """
        Check if connecting to this router maintains IP diversity.
        
        We require routers to be in different /16 subnets to prevent
        a single network operator from controlling all our connections.
        """
        if not router.endpoints:
            return False
        
        try:
            endpoint = router.endpoints[0]
            
            # Handle IPv6 bracket notation: [ipv6]:port
            if endpoint.startswith("["):
                bracket_end = endpoint.find("]")
                if bracket_end == -1:
                    return True
                host = endpoint[1:bracket_end]
            else:
                host = endpoint.split(":")[0]
            
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                return True  # Hostname, allow it
            
            if ip.version == 4:
                network = ipaddress.ip_network(
                    f"{ip}/{self.config.ip_diversity_prefix}", strict=False
                )
            else:
                network = ipaddress.ip_network(f"{ip}/48", strict=False)
            
            subnet_key = str(network)
            
            if subnet_key in self._connected_subnets:
                self._stats["diversity_rejections"] += 1
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"IP diversity check error: {e}")
            return True
    
    def check_asn_diversity(self, router: "RouterInfo") -> bool:
        """
        Check if connecting to this router maintains ASN diversity.
        
        ASN diversity helps prevent eclipse attacks where an attacker
        controls multiple IPs in the same network.
        """
        if not self.config.asn_diversity_enabled:
            return True
        
        asn = router.asn or router.capacity.get("asn") or router.health.get("asn")
        if not asn:
            return True
        
        asn_str = str(asn)
        
        if asn_str in self._connected_asns:
            if len(self._connected_asns) >= self.config.min_diverse_asns:
                self._stats["diversity_rejections"] += 1
                logger.debug(
                    f"Rejecting router {router.router_id[:16]}... - "
                    f"already connected to ASN {asn_str}"
                )
                return False
        
        return True
    
    def check_diversity_requirements(self) -> bool:
        """Check if current connections meet diversity requirements."""
        if len(self._connected_subnets) < min(
            self.config.min_diverse_subnets, self.connection_count
        ):
            return False
        
        if self.config.asn_diversity_enabled and self._connected_asns:
            if len(self._connected_asns) < min(
                self.config.min_diverse_asns, self.connection_count
            ):
                return False
        
        return True
    
    def _add_subnet(self, router: "RouterInfo") -> None:
        """Track the subnet and ASN of a connected router."""
        if not router.endpoints:
            return
        
        try:
            endpoint = router.endpoints[0]
            
            if endpoint.startswith("["):
                bracket_end = endpoint.find("]")
                if bracket_end == -1:
                    return
                host = endpoint[1:bracket_end]
            else:
                host = endpoint.split(":")[0]
            
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                return
            
            if ip.version == 4:
                network = ipaddress.ip_network(
                    f"{ip}/{self.config.ip_diversity_prefix}", strict=False
                )
            else:
                network = ipaddress.ip_network(f"{ip}/48", strict=False)
            
            self._connected_subnets.add(str(network))
            
        except Exception:
            pass
        
        # Track ASN
        asn = router.asn or router.capacity.get("asn") or router.health.get("asn")
        if asn:
            self._connected_asns.add(str(asn))
    
    def _remove_subnet(self, router: "RouterInfo") -> None:
        """Remove subnet and ASN tracking when disconnecting."""
        if not router.endpoints:
            return
        
        try:
            endpoint = router.endpoints[0]
            
            if endpoint.startswith("["):
                bracket_end = endpoint.find("]")
                if bracket_end == -1:
                    return
                host = endpoint[1:bracket_end]
            else:
                host = endpoint.split(":")[0]
            
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                return
            
            if ip.version == 4:
                network = ipaddress.ip_network(
                    f"{ip}/{self.config.ip_diversity_prefix}", strict=False
                )
            else:
                network = ipaddress.ip_network(f"{ip}/48", strict=False)
            
            self._connected_subnets.discard(str(network))
            
        except Exception:
            pass
        
        asn = router.asn or router.capacity.get("asn") or router.health.get("asn")
        if asn:
            self._connected_asns.discard(str(asn))
    
    async def connect_to_router(self, router: "RouterInfo") -> "RouterConnection":
        """
        Establish WebSocket connection to a router.
        
        Args:
            router: Router to connect to
            
        Returns:
            RouterConnection object
            
        Raises:
            ConnectionError: If connection fails
        """
        from .node import RouterConnection
        
        if not router.endpoints:
            raise ConnectionError("Router has no endpoints")
        
        for endpoint in router.endpoints:
            try:
                ws_url = f"wss://{endpoint}/ws"
                
                session = aiohttp.ClientSession()
                try:
                    ws = await session.ws_connect(
                        ws_url,
                        heartbeat=30,
                        timeout=aiohttp.ClientTimeout(total=10),
                    )
                except Exception:
                    await session.close()
                    continue
                
                # Identify ourselves
                await ws.send_json({
                    "type": "identify",
                    "node_id": self.node_id,
                })
                
                # Wait for identification response
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    if msg.type == WSMsgType.TEXT:
                        import json
                        response = json.loads(msg.data)
                        if response.get("type") != "identified":
                            raise ConnectionError(
                                f"Unexpected response: {response.get('type')}"
                            )
                    else:
                        raise ConnectionError(f"Unexpected message type: {msg.type}")
                except asyncio.TimeoutError:
                    await ws.close()
                    await session.close()
                    raise ConnectionError("Identification timeout")
                
                # Create connection record
                now = time.time()
                conn = RouterConnection(
                    router=router,
                    websocket=ws,
                    session=session,
                    connected_at=now,
                    last_seen=now,
                )
                
                self.connections[router.router_id] = conn
                self._add_subnet(router)
                self._connection_timestamps[router.router_id] = now
                self._stats["connections_established"] += 1
                
                logger.info(
                    f"Connected to router {router.router_id[:16]}... "
                    f"at {endpoint}"
                )
                
                # Notify callback
                if self.on_connection_established:
                    try:
                        self.on_connection_established(router.router_id, conn)
                    except Exception as e:
                        logger.warning(f"Connection callback error: {e}")
                
                return conn
                
            except ConnectionError:
                raise
            except Exception as e:
                logger.debug(f"Failed to connect to {endpoint}: {e}")
                continue
        
        raise ConnectionError("Failed to connect to any endpoint for router")
    
    async def close_connection(
        self,
        router_id: str,
        conn: Optional["RouterConnection"] = None,
    ) -> None:
        """Close a router connection and clean up."""
        if conn is None:
            conn = self.connections.get(router_id)
        
        if conn is None:
            return
        
        try:
            if not conn.websocket.closed:
                await conn.websocket.close()
        except Exception:
            pass
        
        try:
            await conn.session.close()
        except Exception:
            pass
        
        self._remove_subnet(conn.router)
        self.connections.pop(router_id, None)
        self._connection_timestamps.pop(router_id, None)
        
        # Notify callback
        if self.on_connection_lost:
            try:
                self.on_connection_lost(router_id)
            except Exception as e:
                logger.warning(f"Connection lost callback error: {e}")
    
    async def close_all(self) -> None:
        """Close all connections."""
        for router_id, conn in list(self.connections.items()):
            await self.close_connection(router_id, conn)
        
        self.connections.clear()
        self._connected_subnets.clear()
        self._connected_asns.clear()
        self._connection_timestamps.clear()
    
    def get_connection(self, router_id: str) -> Optional["RouterConnection"]:
        """Get a connection by router ID."""
        return self.connections.get(router_id)
    
    def get_healthy_connections(self) -> List["RouterConnection"]:
        """Get list of healthy (non-closed) connections."""
        return [
            conn for conn in self.connections.values()
            if not conn.websocket.closed
        ]
    
    def clear_router_cooldown(self, router_id: str) -> bool:
        """Manually clear cooldown for a router."""
        if router_id in self.failover_states:
            state = self.failover_states[router_id]
            state.cooldown_until = 0
            state.fail_count = 0
            logger.info(f"Cleared cooldown for router {router_id[:16]}...")
            return True
        return False
    
    def get_failover_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current failover states for all routers."""
        return {
            router_id: {
                "failed_at": state.failed_at,
                "fail_count": state.fail_count,
                "cooldown_until": state.cooldown_until,
                "in_cooldown": state.is_in_cooldown(),
                "remaining_cooldown": state.remaining_cooldown(),
            }
            for router_id, state in self.failover_states.items()
        }
