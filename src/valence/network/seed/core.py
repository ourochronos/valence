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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp import web
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

from .config import SeedConfig, compute_region_score
from .health import HealthMonitor, HealthStatus, HealthState
from .sybil import SybilResistance
from .revocation import SeedRevocationManager
from .gossip import SeedPeerManager
from .registry import RouterRecord

logger = logging.getLogger(__name__)


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
