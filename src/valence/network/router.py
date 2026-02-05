"""
Valence Router Node - Relays encrypted messages without reading them.

The router node is a key component of the Valence mesh network. It:
- Accepts WebSocket connections from other nodes
- Relays encrypted messages to their destinations
- Queues messages for offline nodes
- Maintains no persistent state about message contents (privacy-preserving)

Messages are end-to-end encrypted; routers only see routing metadata.

Registration Protocol:
- Router generates Ed25519 keypair for identity
- Registration includes PoW (proof-of-work) for anti-Sybil
- Heartbeats sent every 5 minutes with load metrics
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import aiohttp
from aiohttp import WSMsgType, web
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger(__name__)


class RegistrationError(Exception):
    """Raised when router registration fails."""
    pass


@dataclass
class Connection:
    """Represents a connected node."""

    node_id: str
    websocket: web.WebSocketResponse
    connected_at: float
    last_seen: float


# =============================================================================
# CIRCUIT DATA STRUCTURES (Issue #115)
# =============================================================================


@dataclass
class CircuitHopState:
    """
    State for a single circuit hop at this router.
    
    Tracks the shared key and routing info for forwarding messages.
    The router only knows the previous and next hop, not the full path.
    """
    circuit_id: str
    shared_key: bytes  # 32-byte AES key derived from DH exchange
    prev_hop: Optional[str]  # Node ID of previous hop (or originator)
    next_hop: Optional[str]  # Node ID of next hop (or None if exit)
    created_at: float
    message_count: int = 0
    
    def is_exit(self) -> bool:
        """Check if this router is the exit node for this circuit."""
        return self.next_hop is None


@dataclass
class CircuitState:
    """
    All circuit state at this router.
    """
    # circuit_id -> CircuitHopState
    circuits: Dict[str, CircuitHopState] = field(default_factory=dict)
    
    # Circuit limits
    max_circuits: int = 1000
    circuit_timeout: float = 600.0  # 10 minutes
    
    def add_circuit(self, hop_state: CircuitHopState) -> bool:
        """Add a circuit hop. Returns False if at capacity."""
        if len(self.circuits) >= self.max_circuits:
            return False
        self.circuits[hop_state.circuit_id] = hop_state
        return True
    
    def get_circuit(self, circuit_id: str) -> Optional[CircuitHopState]:
        """Get circuit state by ID."""
        return self.circuits.get(circuit_id)
    
    def remove_circuit(self, circuit_id: str) -> None:
        """Remove a circuit."""
        self.circuits.pop(circuit_id, None)
    
    def cleanup_expired(self) -> int:
        """Remove expired circuits. Returns count removed."""
        now = time.time()
        expired = [
            cid for cid, state in self.circuits.items()
            if now - state.created_at > self.circuit_timeout
        ]
        for cid in expired:
            del self.circuits[cid]
        return len(expired)


@dataclass
class NodeConnectionHistory:
    """
    Tracks connection history for a node (Issue #111).
    
    Used to recognize reconnecting nodes and provide state recovery hints.
    """
    
    node_id: str
    first_seen: float
    last_connected: float
    last_disconnected: float
    connection_count: int
    total_messages_delivered: int = 0
    
    def time_since_disconnect(self) -> float:
        """Get seconds since last disconnect."""
        return time.time() - self.last_disconnected


@dataclass
class QueuedMessage:
    """A message queued for an offline node."""

    message_id: str
    payload: str  # Encrypted payload (base64 or similar)
    queued_at: float
    ttl: int


@dataclass
class RouterNode:
    """Router node that relays encrypted messages between Valence nodes.

    The router is privacy-preserving:
    - Message payloads are end-to-end encrypted; router cannot read them
    - Only routing metadata (next_hop, ttl) is visible
    - No per-user tracking or logging

    Attributes:
        host: Bind address for the WebSocket server
        port: Port to listen on
        max_connections: Maximum concurrent connections
        seed_url: URL of seed node to register with (optional)
        heartbeat_interval: Seconds between seed heartbeats
        regions: Geographic regions this router serves
        features: Supported features/protocols
        region: Primary region (ISO 3166-1 alpha-2 country code, e.g., "US", "DE")
        coordinates: Optional (latitude, longitude) for precise location
    """

    host: str = "0.0.0.0"
    port: int = 8471
    max_connections: int = 100
    seed_url: str | None = None
    heartbeat_interval: int = 300  # 5 minutes
    
    # Router identity and capabilities
    regions: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    
    # Regional location (for geographic routing preference)
    region: str | None = None  # ISO 3166-1 alpha-2 country code (e.g., "US", "DE", "JP")
    coordinates: tuple[float, float] | None = None  # (latitude, longitude)
    
    # Ed25519 identity keypair (generated on init if not provided)
    _private_key: Ed25519PrivateKey | None = field(default=None, repr=False)
    _router_id: str | None = field(default=None, repr=False)

    # Runtime state
    connections: dict[str, Connection] = field(default_factory=dict)
    offline_queues: dict[str, list[QueuedMessage]] = field(default_factory=dict)
    
    # Connection history tracking (Issue #111)
    connection_history: dict[str, NodeConnectionHistory] = field(default_factory=dict)
    max_history_entries: int = 10000  # Limit memory usage
    history_max_age: float = 86400.0  # 24 hours

    # Metrics (aggregate only for privacy)
    messages_relayed: int = 0
    messages_queued: int = 0
    messages_delivered: int = 0
    connections_total: int = 0
    reconnections_total: int = 0  # Track reconnections
    
    # Registration state
    _registered: bool = field(default=False, repr=False)
    _seed_id: str | None = field(default=None, repr=False)

    # Internal state
    _app: web.Application | None = field(default=None, repr=False)
    _runner: web.AppRunner | None = field(default=None, repr=False)
    _heartbeat_task: asyncio.Task | None = field(default=None, repr=False)
    _running: bool = field(default=False, repr=False)

    # Queue limits
    MAX_QUEUE_SIZE: int = 1000
    MAX_QUEUE_AGE: int = 3600  # 1 hour
    
    # PoW settings
    POW_DIFFICULTY: int = 16  # Default difficulty (leading zero bits)
    
    # Back-pressure settings
    back_pressure_threshold: float = 80.0  # Activate at 80% load
    back_pressure_release_threshold: float = 60.0  # Release at 60% load
    back_pressure_retry_ms: int = 1000  # Suggest 1 second retry delay
    _back_pressure_active: bool = field(default=False, repr=False)
    _back_pressure_nodes: set = field(default_factory=set, repr=False)  # Nodes notified
    
    # Circuit state (Issue #115)
    _circuit_state: CircuitState = field(default_factory=CircuitState, repr=False)
    _circuit_cleanup_task: asyncio.Task | None = field(default=None, repr=False)
    circuit_cleanup_interval: float = 60.0  # Check for expired circuits every minute
    
    # Circuit metrics
    circuits_created: int = 0
    circuits_relayed: int = 0
    circuits_destroyed: int = 0
    
    def __post_init__(self):
        """Initialize identity keypair if not provided."""
        if self._private_key is None:
            self._private_key = Ed25519PrivateKey.generate()
        
        if self._router_id is None:
            # Router ID is the hex-encoded public key
            public_key = self._private_key.public_key()
            self._router_id = public_key.public_bytes_raw().hex()
    
    @property
    def router_id(self) -> str:
        """Get the router's public identity (hex-encoded Ed25519 public key)."""
        assert self._router_id is not None, "Router ID not initialized"
        return self._router_id
    
    @property
    def endpoints(self) -> list[str]:
        """Get the router's advertised endpoints."""
        return [f"{self.host}:{self.port}"]
    
    def get_capacity(self) -> dict[str, Any]:
        """Get current router capacity metrics."""
        return {
            "max_connections": self.max_connections,
            "current_connections": len(self.connections),
            "bandwidth_mbps": 100,  # Could be configurable
        }
    
    def get_load_pct(self) -> float:
        """Calculate current load percentage (0-100).
        
        Load is based on:
        - Connection utilization (connections / max_connections)
        - Queue pressure (total queued messages / MAX_QUEUE_SIZE)
        
        Returns:
            Load percentage from 0.0 to 100.0
        """
        # Connection-based load
        conn_load = (len(self.connections) / self.max_connections * 100) if self.max_connections > 0 else 0
        
        # Queue-based load (total messages across all queues)
        total_queued = sum(len(q) for q in self.offline_queues.values())
        queue_load = (total_queued / self.MAX_QUEUE_SIZE * 100) if self.MAX_QUEUE_SIZE > 0 else 0
        
        # Combined load: weighted average (connections are more critical)
        return (conn_load * 0.7) + (queue_load * 0.3)
    
    async def _check_back_pressure(self) -> None:
        """Check load and activate/release back-pressure as needed."""
        load_pct = self.get_load_pct()
        
        if not self._back_pressure_active and load_pct >= self.back_pressure_threshold:
            # Activate back-pressure
            await self._activate_back_pressure(load_pct)
        elif self._back_pressure_active and load_pct <= self.back_pressure_release_threshold:
            # Release back-pressure
            await self._release_back_pressure(load_pct)
    
    async def _activate_back_pressure(self, load_pct: float) -> None:
        """Activate back-pressure and notify all connected nodes.
        
        Args:
            load_pct: Current load percentage for the notification
        """
        self._back_pressure_active = True
        logger.warning(
            f"Back-pressure ACTIVATED at {load_pct:.1f}% load "
            f"(threshold: {self.back_pressure_threshold}%)"
        )
        
        # Notify all connected nodes
        message = {
            "type": "back_pressure",
            "active": True,
            "load_pct": round(load_pct, 1),
            "retry_after_ms": self.back_pressure_retry_ms,
            "reason": "Router under heavy load",
        }
        
        await self._broadcast_to_nodes(message)
        self._back_pressure_nodes = set(self.connections.keys())
    
    async def _release_back_pressure(self, load_pct: float) -> None:
        """Release back-pressure and notify previously notified nodes.
        
        Args:
            load_pct: Current load percentage for the notification
        """
        self._back_pressure_active = False
        logger.info(
            f"Back-pressure RELEASED at {load_pct:.1f}% load "
            f"(threshold: {self.back_pressure_release_threshold}%)"
        )
        
        # Notify nodes that were previously notified of back-pressure
        message = {
            "type": "back_pressure",
            "active": False,
            "load_pct": round(load_pct, 1),
            "retry_after_ms": 0,
            "reason": "Load returned to normal",
        }
        
        # Send to nodes that were notified of back-pressure
        for node_id in list(self._back_pressure_nodes):
            if node_id in self.connections:
                try:
                    conn = self.connections[node_id]
                    await conn.websocket.send_json(message)
                except Exception as e:
                    logger.debug(f"Failed to send back-pressure release to {node_id}: {e}")
        
        self._back_pressure_nodes.clear()
    
    async def _broadcast_to_nodes(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all connected nodes.
        
        Args:
            message: Message to broadcast
        """
        for node_id, conn in list(self.connections.items()):
            try:
                await conn.websocket.send_json(message)
            except Exception as e:
                logger.debug(f"Failed to broadcast to {node_id}: {e}")
    
    @property
    def is_back_pressure_active(self) -> bool:
        """Check if back-pressure is currently active."""
        return self._back_pressure_active
    
    def _sign(self, data: dict[str, Any]) -> str:
        """Sign data with router's Ed25519 private key.
        
        Args:
            data: Dictionary to sign (will be JSON-serialized)
            
        Returns:
            Hex-encoded signature
        """
        assert self._private_key is not None, "Private key not set"
        message = json.dumps(data, sort_keys=True, separators=(',', ':')).encode()
        signature = self._private_key.sign(message)
        return signature.hex()
    
    def _generate_pow(self, difficulty: int | None = None) -> dict[str, Any]:
        """Generate proof-of-work for registration.
        
        PoW: Find nonce where sha256(challenge || nonce || router_id)
        has `difficulty` leading zero bits.
        
        Args:
            difficulty: Number of leading zero bits required (default: POW_DIFFICULTY)
            
        Returns:
            Dict with challenge, nonce, and difficulty
        """
        if difficulty is None:
            difficulty = self.POW_DIFFICULTY
        
        # Generate random challenge
        challenge = secrets.token_hex(16)
        
        nonce = 0
        while True:
            # Construct hash input
            hash_input = f"{challenge}{nonce}{self.router_id}".encode()
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
            
            if leading_zeros >= difficulty:
                logger.debug(f"PoW found: nonce={nonce}, zeros={leading_zeros}")
                return {
                    "challenge": challenge,
                    "nonce": nonce,
                    "difficulty": difficulty,
                }
            
            nonce += 1
            
            # Safety limit to avoid infinite loop in tests with high difficulty
            if nonce > 10_000_000:
                raise RuntimeError(f"PoW generation exceeded limit at difficulty {difficulty}")

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse | web.Response:
        """Handle incoming WebSocket connections from nodes."""
        if len(self.connections) >= self.max_connections:
            logger.warning("Max connections reached, rejecting new connection")
            return web.Response(status=503, text="Server at capacity")

        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)

        node_id: str | None = None
        self.connections_total += 1

        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning("Received invalid JSON")
                        continue

                    msg_type = data.get("type")

                    if msg_type == "identify":
                        node_id = await self._handle_identify(data, ws)

                    elif msg_type == "relay":
                        await self._handle_relay(data)

                    elif msg_type == "ping":
                        await ws.send_json({"type": "pong", "timestamp": time.time()})
                        if node_id and node_id in self.connections:
                            self.connections[node_id].last_seen = time.time()

                    else:
                        logger.debug(f"Unknown message type: {msg_type}")

                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"WebSocket error: {ws.exception()}")
                    break

                elif msg.type == WSMsgType.CLOSE:
                    break

        except Exception as e:
            logger.exception(f"Error handling WebSocket: {e}")

        finally:
            if node_id and node_id in self.connections:
                del self.connections[node_id]
                # Update connection history with disconnect time (Issue #111)
                if node_id in self.connection_history:
                    self.connection_history[node_id].last_disconnected = time.time()
                logger.info(f"Node disconnected: {node_id[:16]}...")

        return ws

    async def _handle_identify(
        self, data: dict[str, Any], ws: web.WebSocketResponse
    ) -> str | None:
        """Handle node identification message.
        
        Enhanced with reconnection tracking (Issue #111):
        - Recognizes returning nodes
        - Provides queued message count in response
        - Tracks connection history for recovery hints
        """
        node_id = data.get("node_id")
        if not node_id:
            await ws.send_json({"type": "error", "message": "Missing node_id"})
            return None

        # Check if already connected
        if node_id in self.connections:
            old_conn = self.connections[node_id]
            if not old_conn.websocket.closed:
                # Existing connection still active - reject new one
                await ws.send_json(
                    {"type": "error", "message": "Node already connected"}
                )
                return None

        now = time.time()
        is_reconnection = False
        queued_count = len(self.offline_queues.get(node_id, []))
        
        # Update connection history (Issue #111)
        if node_id in self.connection_history:
            history = self.connection_history[node_id]
            is_reconnection = True
            history.last_connected = now
            history.connection_count += 1
            self.reconnections_total += 1
            
            logger.info(
                f"Node reconnected: {node_id[:16]}... "
                f"(away for {history.time_since_disconnect():.1f}s, "
                f"queued: {queued_count})"
            )
        else:
            # New node - create history entry
            self.connection_history[node_id] = NodeConnectionHistory(
                node_id=node_id,
                first_seen=now,
                last_connected=now,
                last_disconnected=0.0,
                connection_count=1,
            )
            logger.info(f"Node connected (new): {node_id[:16]}...")
        
        # Prune old history entries to prevent memory bloat
        self._prune_connection_history()

        # Register connection
        self.connections[node_id] = Connection(
            node_id=node_id,
            websocket=ws,
            connected_at=now,
            last_seen=now,
        )

        # Build response with recovery hints (Issue #111)
        response = {
            "type": "identified",
            "node_id": node_id,
            "timestamp": now,
            "is_reconnection": is_reconnection,
            "queued_messages": queued_count,
        }
        
        # Add time since disconnect for reconnections
        if is_reconnection and node_id in self.connection_history:
            history = self.connection_history[node_id]
            if history.last_disconnected > 0:
                response["time_since_disconnect"] = now - history.last_disconnected
        
        await ws.send_json(response)

        # Deliver any queued messages
        delivered = await self._deliver_queued(node_id, ws)
        if delivered > 0:
            logger.info(f"Delivered {delivered} queued messages to {node_id[:16]}...")
            # Update history
            if node_id in self.connection_history:
                self.connection_history[node_id].total_messages_delivered += delivered

        return node_id
    
    def _prune_connection_history(self) -> None:
        """Remove old connection history entries to prevent memory bloat."""
        if len(self.connection_history) <= self.max_history_entries:
            return
        
        now = time.time()
        
        # Remove entries older than max age
        to_remove = [
            node_id for node_id, history in self.connection_history.items()
            if (now - history.last_connected) > self.history_max_age
            and node_id not in self.connections  # Don't remove active connections
            and node_id not in self.offline_queues  # Don't remove nodes with queued messages
        ]
        
        for node_id in to_remove:
            del self.connection_history[node_id]
        
        if to_remove:
            logger.debug(f"Pruned {len(to_remove)} old connection history entries")

    async def _handle_relay(self, data: dict[str, Any]) -> None:
        """Relay a message to its destination.

        The payload is encrypted and opaque to us. We only look at routing metadata.
        """
        message_id = data.get("message_id")
        next_hop = data.get("next_hop")
        payload = data.get("payload")  # Encrypted - we can't read it
        ttl = data.get("ttl", 10)

        if not all([message_id, next_hop, payload]):
            logger.warning("Relay message missing required fields")
            return
        
        # Type assertions for mypy (we've verified these are not None above)
        assert message_id is not None
        assert next_hop is not None
        assert payload is not None

        if ttl <= 0:
            logger.debug(f"Dropping expired message {message_id}")
            return

        self.messages_relayed += 1

        if next_hop in self.connections:
            # Deliver directly to connected node
            conn = self.connections[next_hop]
            try:
                await conn.websocket.send_json(
                    {
                        "type": "deliver",
                        "message_id": message_id,
                        "payload": payload,
                        "ttl": ttl - 1,
                    }
                )
                self.messages_delivered += 1
                logger.debug(f"Delivered message {message_id} to {next_hop}")
            except Exception as e:
                logger.warning(f"Failed to deliver to {next_hop}: {e}")
                # Queue for retry
                self._queue_message(
                    next_hop,
                    QueuedMessage(
                        message_id=message_id,
                        payload=payload,
                        queued_at=time.time(),
                        ttl=ttl - 1,
                    ),
                )
        else:
            # Node offline - queue message
            self._queue_message(
                next_hop,
                QueuedMessage(
                    message_id=message_id,
                    payload=payload,
                    queued_at=time.time(),
                    ttl=ttl - 1,
                ),
            )
            logger.debug(f"Queued message {message_id} for offline node {next_hop}")
        
        # Check if we need to activate/release back-pressure
        await self._check_back_pressure()

    def _queue_message(self, node_id: str, msg: QueuedMessage) -> bool:
        """Queue a message for an offline node.

        Returns True if queued, False if queue is full.
        """
        if node_id not in self.offline_queues:
            self.offline_queues[node_id] = []

        queue = self.offline_queues[node_id]

        # Enforce queue size limit
        if len(queue) >= self.MAX_QUEUE_SIZE:
            logger.warning(f"Queue full for {node_id}, dropping oldest message")
            queue.pop(0)

        queue.append(msg)
        self.messages_queued += 1
        return True

    async def _deliver_queued(
        self, node_id: str, ws: web.WebSocketResponse
    ) -> int:
        """Deliver queued messages to a newly connected node.

        Returns the number of messages delivered.
        """
        if node_id not in self.offline_queues:
            return 0

        queue = self.offline_queues[node_id]
        delivered = 0
        now = time.time()

        for msg in queue:
            # Check if message has expired
            if now - msg.queued_at >= self.MAX_QUEUE_AGE:
                continue

            try:
                await ws.send_json(
                    {
                        "type": "deliver",
                        "message_id": msg.message_id,
                        "payload": msg.payload,
                        "ttl": msg.ttl,
                    }
                )
                delivered += 1
                self.messages_delivered += 1
            except Exception as e:
                logger.warning(f"Failed to deliver queued message: {e}")
                break

        # Clear the queue
        del self.offline_queues[node_id]

        return delivered

    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy",
                "connections": len(self.connections),
                "queued_nodes": len(self.offline_queues),
                "back_pressure": {
                    "active": self._back_pressure_active,
                    "load_pct": round(self.get_load_pct(), 1),
                },
                "metrics": {
                    "messages_relayed": self.messages_relayed,
                    "messages_queued": self.messages_queued,
                    "messages_delivered": self.messages_delivered,
                    "connections_total": self.connections_total,
                },
            }
        )

    async def handle_status(self, request: web.Request) -> web.Response:
        """Detailed status endpoint."""
        return web.json_response(
            {
                "host": self.host,
                "port": self.port,
                "running": self._running,
                "seed_url": self.seed_url,
                "connections": {
                    "current": len(self.connections),
                    "max": self.max_connections,
                    "total": self.connections_total,
                },
                "queues": {
                    "nodes": len(self.offline_queues),
                    "total_messages": sum(
                        len(q) for q in self.offline_queues.values()
                    ),
                },
                "back_pressure": {
                    "active": self._back_pressure_active,
                    "load_pct": round(self.get_load_pct(), 1),
                    "threshold": self.back_pressure_threshold,
                    "release_threshold": self.back_pressure_release_threshold,
                    "nodes_notified": len(self._back_pressure_nodes),
                },
                "circuits": self.get_circuit_stats(),
                "metrics": {
                    "messages_relayed": self.messages_relayed,
                    "messages_queued": self.messages_queued,
                    "messages_delivered": self.messages_delivered,
                },
            }
        )

    async def register_with_seed(self, seed_url: str | None = None) -> bool:
        """Register this router with the seed node.
        
        Registration includes:
        - Ed25519 signed payload
        - Proof-of-work for anti-Sybil protection
        - Router capabilities (regions, features, capacity)

        Args:
            seed_url: Override seed URL (uses self.seed_url if not provided)

        Returns:
            True if registration successful.
            
        Raises:
            RegistrationError: If registration is rejected by seed.
        """
        url = seed_url or self.seed_url
        if not url:
            return False

        registration_url = f"{url.rstrip('/')}/register"
        
        # Generate proof-of-work
        logger.info(f"Generating proof-of-work (difficulty={self.POW_DIFFICULTY})...")
        proof_of_work = self._generate_pow()
        
        # Build registration payload
        registration = {
            "router_id": self.router_id,
            "endpoints": self.endpoints,
            "capacity": self.get_capacity(),
            "regions": self.regions,
            "features": self.features,
            "region": self.region,  # ISO 3166-1 alpha-2 country code
            "coordinates": list(self.coordinates) if self.coordinates else None,
            "proof_of_work": proof_of_work,
            "timestamp": time.time(),
        }
        
        # Sign the registration
        registration["signature"] = self._sign(registration)

        try:
            timeout = aiohttp.ClientTimeout(total=30)  # Longer timeout for PoW verification
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(registration_url, json=registration) as response:
                    result = await response.json()
                    
                    if result.get("status") == "accepted":
                        self._registered = True
                        self._seed_id = result.get("seed_id")
                        logger.info(
                            f"Registered with seed node {self._seed_id}: {url}"
                        )
                        return True
                    else:
                        reason = result.get("reason", "unknown")
                        logger.warning(
                            f"Seed registration rejected: {reason}"
                        )
                        raise RegistrationError(reason)
                        
        except RegistrationError:
            raise
        except Exception as e:
            logger.warning(f"Failed to register with seed: {e}")
            return False
    
    async def _register_with_seed(self) -> bool:
        """Legacy method for backward compatibility."""
        try:
            return await self.register_with_seed()
        except RegistrationError:
            return False

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to the seed node."""
        while self._running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                if not self._running:
                    break

                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")

    async def _send_heartbeat(self) -> bool:
        """Send a heartbeat to the seed node.
        
        Heartbeat includes enhanced load metrics for distributed load balancing:
        - current_connections: Active WebSocket connections
        - max_connections: Connection capacity
        - load_pct: Estimated load percentage (connections/max)
        - messages_relayed: Total messages relayed since start
        - messages_per_sec: Message throughput rate
        - queue_depth: Total queued messages for offline nodes
        - uptime_pct: Estimated uptime (placeholder)

        Returns True if heartbeat acknowledged.
        """
        if not self.seed_url:
            return False

        heartbeat_url = f"{self.seed_url.rstrip('/')}/heartbeat"
        
        # Calculate load percentage based on connections
        connection_load = (len(self.connections) / self.max_connections) * 100 if self.max_connections > 0 else 0
        
        # Calculate queue depth (total messages across all offline queues)
        queue_depth = sum(len(q) for q in self.offline_queues.values())
        
        # Calculate messages per second (using last heartbeat interval)
        # Store previous values for rate calculation
        now = time.time()
        if not hasattr(self, '_last_heartbeat_time'):
            self._last_heartbeat_time = now
            self._last_messages_relayed = self.messages_relayed
        
        elapsed = now - self._last_heartbeat_time
        if elapsed > 0:
            messages_per_sec = (self.messages_relayed - self._last_messages_relayed) / elapsed
        else:
            messages_per_sec = 0.0
        
        # Update tracking for next heartbeat
        self._last_heartbeat_time = now
        self._last_messages_relayed = self.messages_relayed
        
        # Build heartbeat payload with enhanced load metrics
        heartbeat = {
            "router_id": self.router_id,
            "current_connections": len(self.connections),
            "max_connections": self.max_connections,
            "load_pct": round(connection_load, 1),
            "messages_relayed": self.messages_relayed,
            "messages_per_sec": round(messages_per_sec, 2),
            "queue_depth": queue_depth,
            "uptime_pct": 99.9,  # Placeholder - could track actual uptime
            "timestamp": time.time(),
        }
        
        # Sign the heartbeat
        heartbeat["signature"] = self._sign(heartbeat)

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(heartbeat_url, json=heartbeat) as response:
                    if response.status == 200:
                        result = await response.json()
                        health_status = result.get("health_status", "unknown")
                        next_interval = result.get("next_heartbeat_in", self.heartbeat_interval)
                        
                        logger.debug(
                            f"Heartbeat acknowledged: status={health_status}, "
                            f"next_in={next_interval}s"
                        )
                        return True
                    else:
                        text = await response.text()
                        logger.warning(f"Heartbeat failed ({response.status}): {text}")
                        return False
        except Exception as e:
            logger.warning(f"Heartbeat error: {e}")
            return False

    async def start(self) -> None:
        """Start the router node."""
        if self._running:
            logger.warning("Router already running")
            return

        self._app = web.Application()
        self._app.router.add_get("/ws", self.handle_websocket)
        self._app.router.add_get("/health", self.handle_health)
        self._app.router.add_get("/status", self.handle_status)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()

        self._running = True
        logger.info(f"Router node listening on {self.host}:{self.port}")

        # Register with seed if configured
        if self.seed_url:
            await self._register_with_seed()
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        """Stop the router node."""
        if not self._running:
            return

        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Close all WebSocket connections
        for node_id, conn in list(self.connections.items()):
            try:
                await conn.websocket.close()
            except Exception:
                pass
        self.connections.clear()

        # Cleanup runner
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

        self._app = None
        logger.info("Router node stopped")
    
    # -------------------------------------------------------------------------
    # CIRCUIT HANDLING (Issue #115 - Enhanced Privacy)
    # -------------------------------------------------------------------------
    
    async def _handle_circuit_create(
        self,
        data: dict[str, Any],
        ws: web.WebSocketResponse,
        sender_id: str | None,
    ) -> None:
        """
        Handle circuit creation request.
        
        Process:
        1. Generate ephemeral X25519 keypair
        2. Derive shared key via ECDH with sender's ephemeral key
        3. Store circuit state
        4. If next_hop specified, extend circuit
        5. Send circuit_created response
        """
        circuit_id = data.get("circuit_id")
        sender_ephemeral_hex = data.get("ephemeral_public")
        next_hop = data.get("next_hop")
        extend_payload = data.get("extend_payload")
        
        if not circuit_id or not sender_ephemeral_hex:
            await ws.send_json({
                "type": "error",
                "message": "Missing circuit_id or ephemeral_public",
            })
            return
        
        # Check if we already have this circuit
        if self._circuit_state.get_circuit(circuit_id):
            await ws.send_json({
                "type": "error",
                "message": "Circuit already exists",
                "circuit_id": circuit_id,
            })
            return
        
        try:
            # Generate our ephemeral keypair
            our_private = X25519PrivateKey.generate()
            our_public = our_private.public_key()
            
            # Derive shared key
            sender_ephemeral = X25519PublicKey.from_public_bytes(
                bytes.fromhex(sender_ephemeral_hex)
            )
            shared_secret = our_private.exchange(sender_ephemeral)
            
            # Derive circuit key using HKDF
            shared_key = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=None,
                info=f"valence-circuit-v1:{circuit_id}".encode()
            ).derive(shared_secret)
            
            # Store circuit state
            hop_state = CircuitHopState(
                circuit_id=circuit_id,
                shared_key=shared_key,
                prev_hop=sender_id,
                next_hop=next_hop,
                created_at=time.time(),
            )
            
            if not self._circuit_state.add_circuit(hop_state):
                await ws.send_json({
                    "type": "error",
                    "message": "Circuit capacity reached",
                    "circuit_id": circuit_id,
                })
                return
            
            self.circuits_created += 1
            
            # If there's a next hop, we need to extend the circuit
            extend_response = None
            if next_hop and extend_payload:
                # In a full implementation, we would:
                # 1. Connect to next_hop router
                # 2. Decrypt extend_payload with our shared key
                # 3. Send circuit_create to next hop
                # 4. Include their response in our response
                # For now, we just note the next hop
                logger.debug(f"Circuit {circuit_id[:8]}... will extend to {next_hop[:16]}...")
            
            # Send response
            await ws.send_json({
                "type": "circuit_created",
                "circuit_id": circuit_id,
                "ephemeral_public": our_public.public_bytes_raw().hex(),
                "extend_response": extend_response,
            })
            
            logger.info(
                f"Circuit {circuit_id[:8]}... created, "
                f"prev={sender_id[:16] if sender_id else 'unknown'}..., "
                f"next={next_hop[:16] if next_hop else 'exit'}..."
            )
            
        except Exception as e:
            logger.warning(f"Circuit create failed: {e}")
            await ws.send_json({
                "type": "error",
                "message": f"Circuit creation failed: {e}",
                "circuit_id": circuit_id,
            })
    
    async def _handle_circuit_relay(
        self,
        data: dict[str, Any],
        sender_id: str | None,
    ) -> None:
        """
        Handle circuit relay message.
        
        Process:
        1. Look up circuit state
        2. Decrypt one layer of onion
        3. Forward to next hop (or deliver if we're exit)
        """
        circuit_id = data.get("circuit_id")
        payload_hex = data.get("payload")
        direction = data.get("direction", "forward")
        
        if not circuit_id or not payload_hex:
            logger.debug("Circuit relay missing circuit_id or payload")
            return
        
        hop_state = self._circuit_state.get_circuit(circuit_id)
        if not hop_state:
            logger.debug(f"Unknown circuit: {circuit_id[:8]}...")
            return
        
        hop_state.message_count += 1
        self.circuits_relayed += 1
        
        try:
            payload = bytes.fromhex(payload_hex)
            
            if direction == "forward":
                # Peel one layer
                inner_payload, routing_next_hop = self._peel_onion_layer(
                    payload, hop_state.shared_key
                )
                
                if hop_state.is_exit() or not routing_next_hop:
                    # We're the exit node - deliver to final recipient
                    await self._deliver_circuit_payload(inner_payload, hop_state)
                else:
                    # Forward to next hop
                    await self._forward_circuit_relay(
                        circuit_id, inner_payload, routing_next_hop, "forward"
                    )
            
            else:  # backward
                # Add one layer for backward direction
                outer_payload = self._add_backward_layer(payload, hop_state.shared_key)
                
                # Forward back to previous hop
                if hop_state.prev_hop:
                    await self._forward_circuit_relay(
                        circuit_id, outer_payload, hop_state.prev_hop, "backward"
                    )
                    
        except Exception as e:
            logger.warning(f"Circuit relay failed: {e}")
    
    def _peel_onion_layer(
        self,
        payload: bytes,
        key: bytes,
    ) -> Tuple[bytes, Optional[str]]:
        """
        Peel one layer of onion encryption.
        
        Returns (inner_payload, next_hop_or_none)
        """
        # Extract nonce (first 12 bytes)
        nonce = payload[:12]
        ciphertext = payload[12:]
        
        # Decrypt
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        
        # Parse routing header
        # Format: 1 byte has_next + 64 bytes next_hop (padded) + content
        has_next = plaintext[0] == 1
        next_hop_bytes = plaintext[1:65]
        content = plaintext[65:]
        
        if has_next:
            next_hop = next_hop_bytes.rstrip(b'\x00').decode()
        else:
            next_hop = None
        
        return content, next_hop
    
    def _add_backward_layer(
        self,
        payload: bytes,
        key: bytes,
    ) -> bytes:
        """
        Add one layer of encryption for backward direction.
        """
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, payload, None)
        return nonce + ciphertext
    
    async def _deliver_circuit_payload(
        self,
        payload: bytes,
        hop_state: CircuitHopState,
    ) -> None:
        """
        Deliver the final payload from a circuit.
        
        Parse the inner payload to find the recipient and forward.
        """
        try:
            # Parse the inner payload
            inner_data = json.loads(payload.decode())
            recipient_id = inner_data.get("recipient_id")
            message_payload_hex = inner_data.get("payload")
            message_id = inner_data.get("message_id")
            
            if not recipient_id or not message_payload_hex:
                logger.warning("Circuit payload missing recipient_id or payload")
                return
            
            # Check if recipient is connected
            if recipient_id in self.connections:
                conn = self.connections[recipient_id]
                await conn.websocket.send_json({
                    "type": "deliver",
                    "message_id": message_id,
                    "payload": json.loads(bytes.fromhex(message_payload_hex).decode()),
                    "ttl": 1,
                })
                self.messages_delivered += 1
                logger.debug(
                    f"Delivered circuit message {message_id[:8] if message_id else 'unknown'}... "
                    f"to {recipient_id[:16]}..."
                )
            else:
                # Queue for offline delivery
                self._queue_message(
                    recipient_id,
                    QueuedMessage(
                        message_id=message_id or str(time.time()),
                        payload=message_payload_hex,
                        queued_at=time.time(),
                        ttl=1,
                    ),
                )
                logger.debug(f"Queued circuit message for offline node {recipient_id[:16]}...")
                
        except Exception as e:
            logger.warning(f"Failed to deliver circuit payload: {e}")
    
    async def _forward_circuit_relay(
        self,
        circuit_id: str,
        payload: bytes,
        next_hop: str,
        direction: str,
    ) -> None:
        """
        Forward a circuit relay message to the next hop.
        """
        # Check if next hop is connected
        if next_hop not in self.connections:
            logger.debug(f"Next hop {next_hop[:16]}... not connected")
            return
        
        conn = self.connections[next_hop]
        
        try:
            await conn.websocket.send_json({
                "type": "circuit_relay",
                "circuit_id": circuit_id,
                "payload": payload.hex(),
                "direction": direction,
            })
        except Exception as e:
            logger.warning(f"Failed to forward circuit relay: {e}")
    
    async def _handle_circuit_destroy(
        self,
        data: dict[str, Any],
    ) -> None:
        """
        Handle circuit destroy message.
        
        Clean up circuit state and optionally propagate to next hop.
        """
        circuit_id = data.get("circuit_id")
        reason = data.get("reason", "")
        
        if not circuit_id:
            return
        
        hop_state = self._circuit_state.get_circuit(circuit_id)
        if not hop_state:
            return
        
        # Propagate destroy to next hop if applicable
        if hop_state.next_hop and hop_state.next_hop in self.connections:
            conn = self.connections[hop_state.next_hop]
            try:
                await conn.websocket.send_json({
                    "type": "circuit_destroy",
                    "circuit_id": circuit_id,
                    "reason": reason,
                })
            except Exception:
                pass
        
        # Remove circuit state
        self._circuit_state.remove_circuit(circuit_id)
        self.circuits_destroyed += 1
        
        logger.info(f"Circuit {circuit_id[:8]}... destroyed: {reason}")
    
    async def _circuit_cleanup_loop(self) -> None:
        """
        Periodically clean up expired circuits.
        """
        while self._running:
            try:
                await asyncio.sleep(self.circuit_cleanup_interval)
                if not self._running:
                    break
                
                removed = self._circuit_state.cleanup_expired()
                if removed > 0:
                    self.circuits_destroyed += removed
                    logger.info(f"Cleaned up {removed} expired circuits")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Circuit cleanup error: {e}")
    
    def get_circuit_stats(self) -> dict[str, Any]:
        """Get circuit-related statistics."""
        return {
            "circuits_active": len(self._circuit_state.circuits),
            "circuits_created": self.circuits_created,
            "circuits_relayed": self.circuits_relayed,
            "circuits_destroyed": self.circuits_destroyed,
        }

    async def run_forever(self) -> None:
        """Start the router and run until interrupted."""
        await self.start()
        
        # Start circuit cleanup task
        self._circuit_cleanup_task = asyncio.create_task(self._circuit_cleanup_loop())
        
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # Cancel circuit cleanup task
            if self._circuit_cleanup_task:
                self._circuit_cleanup_task.cancel()
                try:
                    await self._circuit_cleanup_task
                except asyncio.CancelledError:
                    pass
            await self.stop()
