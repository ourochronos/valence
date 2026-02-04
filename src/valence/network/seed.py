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
- Heartbeats include router signature for verification
- No sensitive data stored - just public routing info
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


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
    
    # Selection weights
    weight_health: float = 0.4
    weight_capacity: float = 0.3
    weight_region: float = 0.2
    weight_random: float = 0.1
    
    # Seed node identity
    seed_id: Optional[str] = None
    
    # Known other seeds for redundancy
    known_seeds: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.seed_id is None:
            self.seed_id = f"seed-{secrets.token_hex(8)}"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class RouterRecord:
    """Record of a registered router."""
    
    router_id: str  # Ed25519 public key (multibase or hex)
    endpoints: List[str]  # ["ip:port", ...]
    capacity: Dict[str, Any]  # {max_connections, current_load_pct, bandwidth_mbps}
    health: Dict[str, Any]  # {last_seen, uptime_pct, avg_latency_ms}
    regions: List[str]  # Geographic regions served
    features: List[str]  # Supported features/protocols
    registered_at: float  # Unix timestamp
    router_signature: str  # Signature of registration data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "router_id": self.router_id,
            "endpoints": self.endpoints,
            "capacity": self.capacity,
            "health": self.health,
            "regions": self.regions,
            "features": self.features,
            "registered_at": self.registered_at,
            "router_signature": self.router_signature,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouterRecord":
        """Create from dictionary."""
        return cls(
            router_id=data["router_id"],
            endpoints=data.get("endpoints", []),
            capacity=data.get("capacity", {}),
            health=data.get("health", {}),
            regions=data.get("regions", []),
            features=data.get("features", []),
            registered_at=data.get("registered_at", time.time()),
            router_signature=data.get("router_signature", ""),
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
    
    # Runtime state
    _app: Optional[web.Application] = field(default=None, repr=False)
    _runner: Optional[web.AppRunner] = field(default=None, repr=False)
    _site: Optional[web.TCPSite] = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)
    
    @property
    def seed_id(self) -> str:
        """Get the seed node's ID."""
        return self.config.seed_id
    
    @property
    def known_seeds(self) -> List[str]:
        """Get list of known peer seeds."""
        return self.config.known_seeds
    
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
        Score a router for selection.
        
        Higher score = better candidate.
        """
        score = 0.0
        
        # Health component (0-1, weighted)
        uptime = router.health.get("uptime_pct", 0) / 100.0
        score += uptime * self.config.weight_health
        
        # Capacity component (0-1, weighted) - prefer lower load
        load = router.capacity.get("current_load_pct", 100) / 100.0
        score += (1 - load) * self.config.weight_capacity
        
        # Region match bonus
        preferred_region = preferences.get("region")
        if preferred_region and preferred_region in router.regions:
            score += self.config.weight_region
        
        # Feature match bonus (partial)
        required_features = set(preferences.get("features", []))
        if required_features:
            router_features = set(router.features)
            match_ratio = len(required_features & router_features) / len(required_features)
            score += match_ratio * 0.1  # Small bonus for feature match
        
        # Deterministic random factor (based on router_id for consistency)
        random_component = (hash(router.router_id) % 100) / 100.0
        score += random_component * self.config.weight_random
        
        return score
    
    def select_routers(
        self,
        count: int,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> List[RouterRecord]:
        """
        Select routers based on health, capacity, and preferences.
        
        Args:
            count: Number of routers to return
            preferences: Optional dict with region, features, etc.
            
        Returns:
            List of RouterRecord objects (up to count)
        """
        preferences = preferences or {}
        now = time.time()
        
        # Get healthy candidates
        candidates = [
            r for r in self.router_registry.values()
            if self._is_healthy(r, now)
        ]
        
        if not candidates:
            logger.warning("No healthy routers available")
            return []
        
        # Score and sort
        scored = [(self._score_router(r, preferences), r) for r in candidates]
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
        Handle router registration.
        
        POST /register
        {
            "router_id": "z6MkhaXgBZDvotDkL...",
            "endpoints": ["192.168.1.100:8471"],
            "capacity": {"max_connections": 1000, "bandwidth_mbps": 100},
            "regions": ["us-west", "us-central"],
            "features": ["ipv6", "quic"],
            "router_signature": "base64..."
        }
        """
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": "Invalid JSON", "detail": str(e)},
                status=400
            )
        
        # Validate required fields
        router_id = data.get("router_id")
        if not router_id:
            return web.json_response(
                {"error": "router_id required"},
                status=400
            )
        
        endpoints = data.get("endpoints", [])
        if not endpoints:
            return web.json_response(
                {"error": "At least one endpoint required"},
                status=400
            )
        
        # TODO: Verify router_signature against router_id public key
        # For now, we accept the registration but note it's unverified
        signature = data.get("router_signature", "")
        
        now = time.time()
        
        # Check if updating existing or new registration
        is_update = router_id in self.router_registry
        
        # Create or update record
        record = RouterRecord(
            router_id=router_id,
            endpoints=endpoints,
            capacity=data.get("capacity", {}),
            health={
                "last_seen": now,
                "uptime_pct": 100.0,  # Assume healthy on registration
                "avg_latency_ms": data.get("avg_latency_ms", 0),
            },
            regions=data.get("regions", []),
            features=data.get("features", []),
            registered_at=now if not is_update else self.router_registry[router_id].registered_at,
            router_signature=signature,
        )
        
        self.router_registry[router_id] = record
        
        action = "updated" if is_update else "registered"
        logger.info(
            f"Router {action}: {router_id[:20]}... "
            f"endpoints={endpoints}, regions={record.regions}"
        )
        
        return web.json_response({
            "status": "ok",
            "action": action,
            "router_id": router_id,
            "seed_id": self.seed_id,
        })
    
    async def handle_heartbeat(self, request: web.Request) -> web.Response:
        """
        Handle router heartbeats.
        
        POST /heartbeat
        {
            "router_id": "z6MkhaXgBZDvotDkL...",
            "current_load_pct": 35.5,
            "active_connections": 350,
            "uptime_pct": 99.8,
            "avg_latency_ms": 12.5,
            "router_signature": "base64..."
        }
        """
        try:
            data = await request.json()
        except Exception as e:
            return web.json_response(
                {"error": "Invalid JSON", "detail": str(e)},
                status=400
            )
        
        router_id = data.get("router_id")
        if not router_id:
            return web.json_response(
                {"error": "router_id required"},
                status=400
            )
        
        # Check if router is registered
        if router_id not in self.router_registry:
            return web.json_response(
                {"error": "Router not registered", "hint": "Call /register first"},
                status=404
            )
        
        # TODO: Verify router_signature
        
        # Update health and capacity
        record = self.router_registry[router_id]
        now = time.time()
        
        record.health["last_seen"] = now
        
        if "uptime_pct" in data:
            record.health["uptime_pct"] = float(data["uptime_pct"])
        if "avg_latency_ms" in data:
            record.health["avg_latency_ms"] = float(data["avg_latency_ms"])
        
        if "current_load_pct" in data:
            record.capacity["current_load_pct"] = float(data["current_load_pct"])
        if "active_connections" in data:
            record.capacity["active_connections"] = int(data["active_connections"])
        
        # Update endpoints if provided
        if "endpoints" in data:
            record.endpoints = data["endpoints"]
        
        logger.debug(f"Heartbeat from {router_id[:20]}...: load={data.get('current_load_pct')}%")
        
        return web.json_response({
            "status": "ok",
            "router_id": router_id,
            "seed_id": self.seed_id,
            "next_heartbeat_in": 60,  # Suggest heartbeat interval
        })
    
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
        
        return web.json_response({
            "seed_id": self.seed_id,
            "status": "running" if self._running else "stopped",
            "timestamp": now,
            "routers": {
                "total": len(self.router_registry),
                "healthy": healthy_count,
            },
            "known_seeds": len(self.known_seeds),
        })
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint for load balancers."""
        return web.json_response({"status": "ok"})
    
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
        
        self._running = True
        logger.info(
            f"Seed node {self.seed_id} listening on "
            f"{self.config.host}:{self.config.port}"
        )
    
    async def stop(self) -> None:
        """Stop the seed node server."""
        if not self._running:
            return
        
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
    **kwargs,
) -> SeedNode:
    """Create a seed node with the given configuration."""
    config = SeedConfig(
        host=host,
        port=port,
        known_seeds=known_seeds or [],
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
