"""
Valence Node Client - User nodes connect to routers for message relay.

User nodes maintain connections to multiple routers for redundancy and
load balancing. This module provides:

- Multi-router connection management
- Weighted router selection based on health metrics
- Keepalive and automatic failure detection
- Message queueing during failover
- IP diversity enforcement (different /16 subnets)
- Connection state persistence and recovery (Issue #111)

Architecture:
- Each node connects to target_connections (default 5) routers
- Routers are selected based on ACK success rate and load
- Connections are monitored via periodic pings
- Failed connections are replaced automatically via discovery
- State is persisted locally for recovery after disconnection

Refactored in Issue #128 to decompose the god class into focused modules:
- ConnectionManager: connection lifecycle
- MessageHandler: message routing
- RouterClient: router interaction
- HealthMonitor: health checks

NodeClient now serves as a facade that coordinates these components.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp
from aiohttp import WSMsgType
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey

from .discovery import DiscoveryClient, RouterInfo
from .crypto import encrypt_message, decrypt_message
from .messages import (
    AckMessage, 
    DeliverPayload, 
    HealthGossip, 
    RouterHealthObservation,
    MisbehaviorType,
    RouterBehaviorMetrics,
    MisbehaviorEvidence,
    MisbehaviorReport,
    NetworkBaseline,
    pad_message,
    get_padded_size,
    CoverMessage,
    Circuit,
    CircuitHop,
)
from .config import (
    TrafficAnalysisMitigationConfig,
    PrivacyLevel,
    BatchingConfig,
    TimingJitterConfig,
    ConstantRateConfig,
    MixNetworkConfig,
)
from .connection_manager import ConnectionManager, ConnectionManagerConfig
from .message_handler import MessageHandler, MessageHandlerConfig
from .router_client import RouterClient, RouterClientConfig
from .health_monitor import HealthMonitor, HealthMonitorConfig

logger = logging.getLogger(__name__)


# =============================================================================
# STATE PERSISTENCE (Issue #111)
# =============================================================================

STATE_VERSION = 1


@dataclass
class ConnectionState:
    """
    Serializable connection state for persistence and recovery.
    
    This captures the essential state needed to recover after a
    disconnect/reconnect cycle.
    """
    
    version: int = STATE_VERSION
    node_id: str = ""
    saved_at: float = 0.0
    sequence_number: int = 0
    pending_acks: List[Dict[str, Any]] = field(default_factory=list)
    message_queue: List[Dict[str, Any]] = field(default_factory=list)
    seen_messages: List[str] = field(default_factory=list)
    failover_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "node_id": self.node_id,
            "saved_at": self.saved_at,
            "sequence_number": self.sequence_number,
            "pending_acks": self.pending_acks,
            "message_queue": self.message_queue,
            "seen_messages": self.seen_messages,
            "failover_states": self.failover_states,
            "stats": self.stats,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConnectionState":
        return cls(
            version=data.get("version", 1),
            node_id=data.get("node_id", ""),
            saved_at=data.get("saved_at", 0.0),
            sequence_number=data.get("sequence_number", 0),
            pending_acks=data.get("pending_acks", []),
            message_queue=data.get("message_queue", []),
            seen_messages=data.get("seen_messages", []),
            failover_states=data.get("failover_states", {}),
            stats=data.get("stats", {}),
        )
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConnectionState":
        return cls.from_dict(json.loads(json_str))


class StateConflictError(Exception):
    """Raised when there's a conflict between saved and current state."""
    pass


class StaleStateError(Exception):
    """Raised when saved state is too old to be useful."""
    pass


# =============================================================================
# ACK TRACKING
# =============================================================================


@dataclass
class PendingAck:
    """Tracks a message awaiting acknowledgment."""
    
    message_id: str
    recipient_id: str
    content: bytes
    recipient_public_key: X25519PublicKey
    sent_at: float
    router_id: str
    timeout_ms: int = 30000
    retries: int = 0
    max_retries: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "recipient_id": self.recipient_id,
            "content": self.content.hex() if isinstance(self.content, bytes) else self.content,
            "recipient_public_key_hex": self.recipient_public_key.public_bytes_raw().hex(),
            "sent_at": self.sent_at,
            "router_id": self.router_id,
            "timeout_ms": self.timeout_ms,
            "retries": self.retries,
            "max_retries": self.max_retries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingAck":
        content = data["content"]
        if isinstance(content, str):
            content = bytes.fromhex(content)
        
        pub_key_hex = data["recipient_public_key_hex"]
        pub_key = X25519PublicKey.from_public_bytes(bytes.fromhex(pub_key_hex))
        
        return cls(
            message_id=data["message_id"],
            recipient_id=data["recipient_id"],
            content=content,
            recipient_public_key=pub_key,
            sent_at=data["sent_at"],
            router_id=data["router_id"],
            timeout_ms=data.get("timeout_ms", 30000),
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 2),
        )


# =============================================================================
# EXCEPTIONS
# =============================================================================


class NodeError(Exception):
    """Base exception for node errors."""
    pass


class ConnectionError(NodeError):
    """Raised when connection to router fails."""
    pass


class NoRoutersAvailableError(NodeError):
    """Raised when no routers are available."""
    pass


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class RouterConnection:
    """Represents an active connection to a router."""
    
    router: RouterInfo
    websocket: aiohttp.ClientWebSocketResponse
    session: aiohttp.ClientSession
    connected_at: float
    last_seen: float
    messages_sent: int = 0
    messages_received: int = 0
    ack_pending: int = 0
    ack_success: int = 0
    ack_failure: int = 0
    ping_latency_ms: float = 0.0
    back_pressure_active: bool = False
    back_pressure_until: float = 0.0
    back_pressure_retry_ms: int = 1000
    
    @property
    def is_under_back_pressure(self) -> bool:
        if not self.back_pressure_active:
            return False
        return time.time() < self.back_pressure_until
    
    @property
    def ack_success_rate(self) -> float:
        total = self.ack_success + self.ack_failure
        if total == 0:
            return 1.0
        return self.ack_success / total
    
    @property
    def health_score(self) -> float:
        ack_score = self.ack_success_rate
        latency_score = max(0, 1.0 - (self.ping_latency_ms / 500))
        load_pct = self.router.capacity.get("current_load_pct", 0)
        load_score = 1.0 - (load_pct / 100)
        return (ack_score * 0.5) + (latency_score * 0.3) + (load_score * 0.2)


@dataclass
class PendingMessage:
    """Message queued for delivery during failover."""
    
    message_id: str
    recipient_id: str
    content: bytes
    recipient_public_key: X25519PublicKey
    queued_at: float
    retries: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "recipient_id": self.recipient_id,
            "content": self.content.hex() if isinstance(self.content, bytes) else self.content,
            "recipient_public_key_hex": self.recipient_public_key.public_bytes_raw().hex(),
            "queued_at": self.queued_at,
            "retries": self.retries,
            "max_retries": self.max_retries,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingMessage":
        content = data["content"]
        if isinstance(content, str):
            content = bytes.fromhex(content)
        
        pub_key_hex = data["recipient_public_key_hex"]
        pub_key = X25519PublicKey.from_public_bytes(bytes.fromhex(pub_key_hex))
        
        return cls(
            message_id=data["message_id"],
            recipient_id=data["recipient_id"],
            content=content,
            recipient_public_key=pub_key,
            queued_at=data["queued_at"],
            retries=data.get("retries", 0),
            max_retries=data.get("max_retries", 3),
        )


@dataclass
class FailoverState:
    """Tracks failover state for a router."""
    
    router_id: str
    failed_at: float
    fail_count: int
    cooldown_until: float
    queued_messages: List[PendingMessage] = field(default_factory=list)
    
    def is_in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until
    
    def remaining_cooldown(self) -> float:
        return max(0, self.cooldown_until - time.time())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "router_id": self.router_id,
            "failed_at": self.failed_at,
            "fail_count": self.fail_count,
            "cooldown_until": self.cooldown_until,
            "queued_messages": [msg.to_dict() for msg in self.queued_messages],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailoverState":
        queued = [
            PendingMessage.from_dict(msg_data)
            for msg_data in data.get("queued_messages", [])
        ]
        return cls(
            router_id=data["router_id"],
            failed_at=data["failed_at"],
            fail_count=data["fail_count"],
            cooldown_until=data["cooldown_until"],
            queued_messages=queued,
        )


@dataclass
class CoverTrafficConfig:
    """Configuration for cover traffic generation."""
    
    enabled: bool = False
    rate_per_minute: float = 2.0
    idle_threshold_seconds: float = 30.0
    pad_messages: bool = True
    target_peers: List[str] = field(default_factory=list)
    randomize_timing: bool = True
    min_interval_seconds: float = 15.0
    max_interval_seconds: float = 60.0
    
    def get_next_interval(self) -> float:
        if not self.enabled:
            return float('inf')
        
        base_interval = 60.0 / self.rate_per_minute if self.rate_per_minute > 0 else 60.0
        
        if self.randomize_timing:
            interval = random.expovariate(1.0 / base_interval)
            interval = max(self.min_interval_seconds, min(self.max_interval_seconds, interval))
        else:
            interval = base_interval
        
        return interval


# =============================================================================
# NODE CLIENT - FACADE
# =============================================================================


@dataclass
class NodeClient:
    """
    User node that connects to routers for message relay.
    
    The node maintains connections to multiple routers for redundancy.
    Messages are encrypted end-to-end; routers only see routing metadata.
    
    This class serves as a facade coordinating:
    - ConnectionManager: connection lifecycle
    - MessageHandler: message routing
    - RouterClient: router interaction
    - HealthMonitor: health checks
    
    Example:
        node = NodeClient(
            node_id="abc123...",
            private_key=my_ed25519_private_key,
            encryption_private_key=my_x25519_private_key,
        )
        await node.start()
        await node.send_message(recipient_id, recipient_pub_key, b"Hello!")
        await node.stop()
    """
    
    # Identity
    node_id: str
    private_key: Ed25519PrivateKey
    encryption_private_key: X25519PrivateKey
    
    # Connection config
    min_connections: int = 3
    target_connections: int = 5
    max_connections: int = 8
    
    # Seed configuration (Issue #108)
    seed_urls: List[str] = field(default_factory=list)
    
    # Timing config
    keepalive_interval: float = 2.0
    ping_timeout: float = 3.0
    maintenance_interval: float = 60.0
    reconnect_delay: float = 1.0
    failover_connect_timeout: float = 3.0
    
    # Failover config
    initial_cooldown: float = 60.0
    max_cooldown: float = 3600.0
    missed_pings_threshold: int = 2
    
    # IP diversity
    enforce_ip_diversity: bool = True
    ip_diversity_prefix: int = 16
    
    # Eclipse attack mitigations (Issue #118)
    min_diverse_subnets: int = 2
    min_diverse_asns: int = 2
    asn_diversity_enabled: bool = True
    rotation_interval: float = 3600.0
    rotation_max_age: float = 7200.0
    rotation_enabled: bool = True
    anomaly_window: float = 60.0
    anomaly_threshold: int = 3
    anomaly_detection_enabled: bool = True
    oob_verification_enabled: bool = False
    oob_verification_url: Optional[str] = None
    oob_verification_interval: float = 3600.0
    
    # State
    connections: Dict[str, RouterConnection] = field(default_factory=dict)
    discovery: DiscoveryClient = field(default_factory=DiscoveryClient)
    message_queue: List[PendingMessage] = field(default_factory=list)
    
    # ACK tracking
    pending_acks: Dict[str, PendingAck] = field(default_factory=dict)
    seen_messages: Set[str] = field(default_factory=set)
    
    # Failover state tracking
    failover_states: Dict[str, FailoverState] = field(default_factory=dict)
    direct_mode: bool = False
    _missed_pings: Dict[str, int] = field(default_factory=dict, repr=False)
    
    # ACK configuration
    default_ack_timeout_ms: int = 30000
    max_seen_messages: int = 10000
    
    # State persistence (Issue #111)
    state_file: Optional[str] = None
    state_save_interval: float = 30.0
    max_state_age: float = 86400.0
    recover_state_on_start: bool = True
    
    # Cover traffic (Issue #116)
    cover_traffic: CoverTrafficConfig = field(default_factory=CoverTrafficConfig)
    
    # Traffic analysis mitigation (Issue #120)
    traffic_analysis_mitigation: TrafficAnalysisMitigationConfig = field(
        default_factory=TrafficAnalysisMitigationConfig
    )
    
    # Callbacks
    on_message: Optional[Callable[[str, bytes], None]] = None
    on_ack_timeout: Optional[Callable[[str, str], None]] = None
    on_state_recovered: Optional[Callable[[ConnectionState], None]] = None
    
    # Misbehavior detection (Issue #119)
    misbehavior_detection_enabled: bool = True
    min_messages_for_detection: int = 20
    delivery_rate_threshold: float = 0.75
    latency_threshold_stddevs: float = 2.0
    ack_failure_threshold: float = 0.30
    mild_severity_threshold: float = 0.3
    severe_severity_threshold: float = 0.7
    auto_avoid_flagged_routers: bool = True
    flagged_router_penalty: float = 0.1
    report_to_seeds: bool = True
    report_cooldown_seconds: float = 300.0
    max_evidence_per_report: int = 10
    baseline_min_samples: int = 3
    on_router_flagged: Optional[Callable[[str, Any], None]] = None
    
    # Gossip configuration
    gossip_interval: float = 30.0
    gossip_ttl: int = 2
    own_observation_weight: float = 0.7
    peer_observation_weight: float = 0.3
    max_observations_per_gossip: int = 10
    max_peer_observations: int = 100
    observation_max_age: float = 300.0
    
    # Circuit configuration (Issue #115)
    circuit_min_hops: int = 2
    circuit_max_hops: int = 3
    circuit_lifetime: float = 600.0
    circuit_max_messages: int = 100
    circuit_build_timeout: float = 10.0
    use_circuits: bool = False
    
    # Queue limits
    MAX_QUEUE_SIZE: int = 1000
    MAX_QUEUE_AGE: float = 3600.0
    
    # Internal state
    _running: bool = field(default=False, repr=False)
    _tasks: List[asyncio.Task] = field(default_factory=list, repr=False)
    _connected_subnets: Set[str] = field(default_factory=set, repr=False)
    _connected_asns: Set[str] = field(default_factory=set, repr=False)
    _state_sequence: int = field(default=0, repr=False)
    _last_state_save: float = field(default=0.0, repr=False)
    
    # Component instances (initialized in __post_init__)
    _connection_manager: Optional[ConnectionManager] = field(default=None, repr=False)
    _message_handler: Optional[MessageHandler] = field(default=None, repr=False)
    _router_client: Optional[RouterClient] = field(default=None, repr=False)
    _health_monitor: Optional[HealthMonitor] = field(default=None, repr=False)
    
    # Statistics
    _stats: Dict[str, int] = field(default_factory=lambda: {
        "messages_sent": 0,
        "messages_received": 0,
        "messages_queued": 0,
        "messages_dropped": 0,
        "messages_deduplicated": 0,
        "connections_established": 0,
        "connections_failed": 0,
        "failovers": 0,
        "ack_successes": 0,
        "ack_failures": 0,
        "acks_sent": 0,
        "gossip_sent": 0,
        "gossip_received": 0,
        "cover_messages_sent": 0,
        "cover_messages_received": 0,
        "bytes_padded": 0,
        "routers_rotated": 0,
        "diversity_rejections": 0,
        "anomalies_detected": 0,
        "oob_verifications": 0,
        "oob_verification_failures": 0,
        "batched_messages": 0,
        "batch_flushes": 0,
        "jitter_delays_applied": 0,
        "total_jitter_ms": 0,
        "constant_rate_padding_sent": 0,
        "messages_with_jitter": 0,
    })
    
    def __post_init__(self):
        """Initialize component managers after dataclass initialization."""
        self._init_components()
    
    def _init_components(self):
        """Initialize the component managers."""
        # ConnectionManager
        conn_config = ConnectionManagerConfig(
            min_connections=self.min_connections,
            target_connections=self.target_connections,
            max_connections=self.max_connections,
            enforce_ip_diversity=self.enforce_ip_diversity,
            ip_diversity_prefix=self.ip_diversity_prefix,
            min_diverse_subnets=self.min_diverse_subnets,
            min_diverse_asns=self.min_diverse_asns,
            asn_diversity_enabled=self.asn_diversity_enabled,
        )
        self._connection_manager = ConnectionManager(
            node_id=self.node_id,
            discovery=self.discovery,
            config=conn_config,
            on_connection_established=self._on_connection_established,
            on_connection_lost=self._on_connection_lost,
        )
        
        # HealthMonitor
        health_config = HealthMonitorConfig(
            gossip_interval=self.gossip_interval,
            gossip_ttl=self.gossip_ttl,
            own_observation_weight=self.own_observation_weight,
            peer_observation_weight=self.peer_observation_weight,
            max_observations_per_gossip=self.max_observations_per_gossip,
            max_peer_observations=self.max_peer_observations,
            observation_max_age=self.observation_max_age,
            keepalive_interval=self.keepalive_interval,
            ping_timeout=self.ping_timeout,
            missed_pings_threshold=self.missed_pings_threshold,
            misbehavior_detection_enabled=self.misbehavior_detection_enabled,
            min_messages_for_detection=self.min_messages_for_detection,
            delivery_rate_threshold=self.delivery_rate_threshold,
            latency_threshold_stddevs=self.latency_threshold_stddevs,
            ack_failure_threshold=self.ack_failure_threshold,
            mild_severity_threshold=self.mild_severity_threshold,
            severe_severity_threshold=self.severe_severity_threshold,
            auto_avoid_flagged_routers=self.auto_avoid_flagged_routers,
            flagged_router_penalty=self.flagged_router_penalty,
            report_cooldown_seconds=self.report_cooldown_seconds,
            max_evidence_per_report=self.max_evidence_per_report,
            baseline_min_samples=self.baseline_min_samples,
            anomaly_detection_enabled=self.anomaly_detection_enabled,
            anomaly_window=self.anomaly_window,
            anomaly_threshold=self.anomaly_threshold,
        )
        self._health_monitor = HealthMonitor(
            node_id=self.node_id,
            discovery=self.discovery,
            config=health_config,
            on_router_flagged=self.on_router_flagged,
        )
        
        # RouterClient
        router_config = RouterClientConfig(
            initial_cooldown=self.initial_cooldown,
            max_cooldown=self.max_cooldown,
            reconnect_delay=self.reconnect_delay,
            failover_connect_timeout=self.failover_connect_timeout,
            rotation_enabled=self.rotation_enabled,
            rotation_interval=self.rotation_interval,
            rotation_max_age=self.rotation_max_age,
            own_observation_weight=self.own_observation_weight,
            peer_observation_weight=self.peer_observation_weight,
        )
        self._router_client = RouterClient(
            connection_manager=self._connection_manager,
            discovery=self.discovery,
            config=router_config,
            get_aggregated_health=self._health_monitor.get_aggregated_health,
            is_router_flagged=self._health_monitor.is_router_flagged,
            flagged_router_penalty=self.flagged_router_penalty,
        )
        
        # MessageHandler
        msg_config = MessageHandlerConfig(
            default_ack_timeout_ms=self.default_ack_timeout_ms,
            max_seen_messages=self.max_seen_messages,
            max_queue_size=self.MAX_QUEUE_SIZE,
            max_queue_age=self.MAX_QUEUE_AGE,
        )
        self._message_handler = MessageHandler(
            node_id=self.node_id,
            private_key=self.private_key,
            encryption_private_key=self.encryption_private_key,
            config=msg_config,
            traffic_mitigation_config=self.traffic_analysis_mitigation,
            on_message=self.on_message,
            on_ack_timeout=self.on_ack_timeout,
        )
        
        # Share state references for backward compatibility
        self._connection_manager.failover_states = self.failover_states
    
    def _on_connection_established(self, router_id: str, conn: RouterConnection) -> None:
        """Callback when a connection is established."""
        self.connections[router_id] = conn
        self._stats["connections_established"] += 1
        
        # Start receive loop
        self._tasks.append(
            asyncio.create_task(self._receive_loop(router_id))
        )
        
        # Disable direct mode if enabled
        if self._router_client and self._router_client.direct_mode:
            self._router_client._disable_direct_mode()
    
    def _on_connection_lost(self, router_id: str) -> None:
        """Callback when a connection is lost."""
        self.connections.pop(router_id, None)
        self._missed_pings.pop(router_id, None)
    
    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------
    
    async def start(self) -> None:
        """Start the node - discover routers and connect."""
        if self._running:
            logger.warning("Node already running")
            return
        
        self._running = True
        logger.info(f"Starting node {self.node_id[:16]}...")
        
        # Re-initialize components if needed
        if self._connection_manager is None:
            self._init_components()
        
        # State recovery (Issue #111)
        if self.recover_state_on_start and self.state_file:
            try:
                recovered = await self._recover_state()
                if recovered:
                    logger.info(
                        f"Recovered state: {len(self.pending_acks)} pending ACKs, "
                        f"{len(self.message_queue)} queued messages"
                    )
            except (StaleStateError, StateConflictError) as e:
                logger.warning(f"State recovery skipped: {e}")
            except Exception as e:
                logger.warning(f"State recovery failed: {e}")
        
        # Configure discovery with seed URLs (Issue #108)
        if self.seed_urls:
            for seed_url in self.seed_urls:
                self.discovery.add_seed(seed_url)
            logger.info(f"Configured {len(self.seed_urls)} seed URLs for discovery")
        
        # Establish connections
        await self._connection_manager.ensure_connections()
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._connection_maintenance()))
        self._tasks.append(asyncio.create_task(self._keepalive_loop()))
        self._tasks.append(asyncio.create_task(self._queue_processor()))
        self._tasks.append(asyncio.create_task(self._gossip_loop()))
        
        if self.state_file:
            self._tasks.append(asyncio.create_task(self._state_persistence_loop()))
        
        if self.traffic_analysis_mitigation.batching.enabled:
            self._tasks.append(asyncio.create_task(self._batch_flush_loop()))
        
        if self.traffic_analysis_mitigation.constant_rate.enabled:
            self._tasks.append(asyncio.create_task(self._constant_rate_loop()))
        
        logger.info(f"Node started with {len(self.connections)} router connections")
    
    async def stop(self) -> None:
        """Stop the node and close all connections."""
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping node...")
        
        # Save state before stopping
        if self.state_file:
            try:
                await self._save_state()
                logger.info("State saved for recovery")
            except Exception as e:
                logger.warning(f"Failed to save state: {e}")
        
        # Cancel background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        
        # Close all connections
        await self._connection_manager.close_all()
        self.connections.clear()
        
        logger.info("Node stopped")
    
    async def send_message(
        self,
        recipient_id: str,
        recipient_public_key: X25519PublicKey,
        content: bytes,
        require_ack: bool = True,
        ack_timeout_ms: Optional[int] = None,
        bypass_mitigations: bool = False,
    ) -> str:
        """
        Send an encrypted message to a recipient via router.
        
        Args:
            recipient_id: Recipient's node ID
            recipient_public_key: Recipient's X25519 public key
            content: Raw message bytes
            require_ack: Whether to require acknowledgment
            ack_timeout_ms: ACK timeout in milliseconds
            bypass_mitigations: Skip batching/jitter
            
        Returns:
            Message ID
        """
        return await self._message_handler.send_message(
            recipient_id=recipient_id,
            recipient_public_key=recipient_public_key,
            content=content,
            router_selector=self._select_router,
            send_via_router=self._send_via_router,
            require_ack=require_ack,
            ack_timeout_ms=ack_timeout_ms,
            bypass_mitigations=bypass_mitigations,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        stats = dict(self._stats)
        
        if self._connection_manager:
            stats.update(self._connection_manager.get_stats())
        if self._health_monitor:
            stats.update(self._health_monitor.get_stats())
        if self._router_client:
            stats.update(self._router_client.get_stats())
        if self._message_handler:
            stats.update(self._message_handler.get_stats())
        
        return {
            **stats,
            "active_connections": len(self.connections),
            "queued_messages": len(self.message_queue),
            "pending_acks": len(self.pending_acks),
            "seen_messages_cached": len(self.seen_messages),
            "direct_mode": self._router_client.direct_mode if self._router_client else False,
        }
    
    def get_connections(self) -> List[Dict[str, Any]]:
        """Get information about active connections."""
        return [
            {
                "router_id": router_id[:16] + "...",
                "endpoint": conn.router.endpoints[0] if conn.router.endpoints else "unknown",
                "connected_at": conn.connected_at,
                "last_seen": conn.last_seen,
                "health_score": round(conn.health_score, 3),
                "ack_success_rate": round(conn.ack_success_rate, 3),
                "ping_latency_ms": round(conn.ping_latency_ms, 1),
                "messages_sent": conn.messages_sent,
                "messages_received": conn.messages_received,
                "back_pressure_active": conn.back_pressure_active,
                "under_back_pressure": conn.is_under_back_pressure,
            }
            for router_id, conn in self.connections.items()
        ]
    
    # -------------------------------------------------------------------------
    # ROUTER SELECTION (delegates to RouterClient)
    # -------------------------------------------------------------------------
    
    def _select_router(self, exclude_back_pressured: bool = True) -> Optional[RouterInfo]:
        """Select the best router based on health metrics."""
        return self._router_client.select_router(exclude_back_pressured)
    
    # -------------------------------------------------------------------------
    # MESSAGE SENDING
    # -------------------------------------------------------------------------
    
    async def _send_via_router(
        self,
        router: RouterInfo,
        message_id: str,
        recipient_id: str,
        recipient_public_key: X25519PublicKey,
        content: bytes,
        require_ack: bool = True,
        is_cover_traffic: bool = False,
    ) -> None:
        """Send an encrypted message via a specific router."""
        conn = self.connections.get(router.router_id)
        if not conn or conn.websocket.closed:
            raise ConnectionError(f"Not connected to router {router.router_id[:16]}...")
        
        # Encrypt for recipient
        payload_with_ack = {
            "content": content.decode() if isinstance(content, bytes) else content,
            "message_id": message_id,
            "require_ack": require_ack,
            "sender_id": self.node_id,
        }
        encrypted = encrypt_message(
            json.dumps(payload_with_ack).encode(),
            recipient_public_key,
            self.private_key
        )
        
        # Send via router
        await conn.websocket.send_json({
            "type": "relay",
            "message_id": message_id,
            "next_hop": recipient_id,
            "payload": encrypted,
            "ttl": 10,
        })
        
        conn.messages_sent += 1
        conn.ack_pending += 1
        
        if is_cover_traffic:
            self._stats["cover_messages_sent"] += 1
        else:
            self._stats["messages_sent"] += 1
        
        logger.debug(f"Sent message {message_id} via router {router.router_id[:16]}...")
    
    # -------------------------------------------------------------------------
    # MESSAGE RECEIVING
    # -------------------------------------------------------------------------
    
    async def _receive_loop(self, router_id: str) -> None:
        """Receive messages from a router connection."""
        conn = self.connections.get(router_id)
        if not conn:
            return
        
        try:
            async for msg in conn.websocket:
                if not self._running:
                    break
                
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    
                    msg_type = data.get("type")
                    
                    if msg_type == "deliver":
                        await self._handle_deliver(data, conn)
                    elif msg_type == "pong":
                        await self._handle_pong(data, conn)
                    elif msg_type == "ack":
                        await self._handle_ack(data, conn)
                    elif msg_type == "back_pressure":
                        self._handle_back_pressure(data, conn)
                    elif msg_type == "gossip":
                        await self._handle_gossip_message(data, conn)
                    elif msg_type == "error":
                        logger.warning(f"Router error: {data.get('message', 'unknown')}")
                    
                    conn.last_seen = time.time()
                
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSED):
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Receive loop error for router {router_id[:16]}...: {e}")
        
        finally:
            if router_id in self.connections:
                await self._handle_router_failure(router_id)
    
    async def _handle_deliver(self, data: Dict[str, Any], conn: RouterConnection) -> None:
        """Handle an incoming message delivery."""
        relay_message_id = data.get("message_id")
        payload = data.get("payload")
        
        if not payload:
            return
        
        conn.messages_received += 1
        self._stats["messages_received"] += 1
        
        try:
            sender_public_hex = payload.get("sender_public")
            if not sender_public_hex:
                return
            
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            sender_public = Ed25519PublicKey.from_public_bytes(bytes.fromhex(sender_public_hex))
            
            plaintext = decrypt_message(payload, self.encryption_private_key, sender_public)
            
            try:
                inner_payload = json.loads(plaintext.decode())
            except (ValueError, UnicodeDecodeError):
                inner_payload = {"content": plaintext.decode()}
            
            inner_message_id = inner_payload.get("message_id")
            require_ack = inner_payload.get("require_ack", False)
            sender_id = inner_payload.get("sender_id", sender_public_hex)
            content = inner_payload.get("content", plaintext)
            
            # Check for ACK message
            if isinstance(content, str):
                try:
                    content_data = json.loads(content)
                    if content_data.get("type") == "ack":
                        ack = AckMessage.from_dict(content_data)
                        self._message_handler.handle_e2e_ack(ack)
                        return
                except (ValueError, TypeError):
                    pass
            
            # Idempotent delivery
            if inner_message_id and self._message_handler.is_duplicate_message(inner_message_id):
                self._stats["messages_deduplicated"] += 1
                return
            
            # Deliver to callback
            if self.on_message:
                if isinstance(content, str):
                    content_bytes = content.encode()
                else:
                    content_bytes = content if isinstance(content, bytes) else str(content).encode()
                await self.on_message(sender_id, content_bytes)
            
            if require_ack and inner_message_id:
                self._stats["acks_sent"] += 1
            
        except Exception as e:
            logger.warning(f"Failed to process message {relay_message_id}: {e}")
    
    async def _handle_pong(self, data: Dict[str, Any], conn: RouterConnection) -> None:
        """Handle pong response."""
        sent_at = data.get("sent_at")
        if sent_at:
            conn.ping_latency_ms = (time.time() - sent_at) * 1000
            self._health_monitor.update_observation(conn.router.router_id, conn)
    
    async def _handle_ack(self, data: Dict[str, Any], conn: RouterConnection) -> None:
        """Handle acknowledgment for a sent message."""
        message_id = data.get("message_id")
        success = data.get("success", True)
        
        conn.ack_pending = max(0, conn.ack_pending - 1)
        
        if success:
            conn.ack_success += 1
        else:
            conn.ack_failure += 1
        
        self._health_monitor.update_observation(conn.router.router_id, conn)
        self._health_monitor.record_ack_outcome(conn.router.router_id, success)
        self._message_handler.handle_ack(message_id, success)
    
    def _handle_back_pressure(self, data: Dict[str, Any], conn: RouterConnection) -> None:
        """Handle back-pressure signal from router."""
        self._router_client.handle_back_pressure(
            conn=conn,
            active=data.get("active", True),
            load_pct=data.get("load_pct", 0),
            retry_after_ms=data.get("retry_after_ms", 1000),
            reason=data.get("reason", ""),
        )
    
    async def _handle_gossip_message(self, data: Dict[str, Any], conn: RouterConnection) -> None:
        """Handle incoming health gossip."""
        payload = data.get("payload", {})
        try:
            gossip = HealthGossip.from_dict(payload)
            self._health_monitor.handle_gossip(gossip)
            self._stats["gossip_received"] += 1
        except Exception as e:
            logger.warning(f"Failed to process gossip: {e}")
    
    async def _handle_router_failure(self, router_id: str) -> None:
        """Handle router failure with intelligent failover."""
        self._health_monitor.record_failure_event(router_id, "connection")
        
        async def retry_messages():
            pending_for_router = [
                pending for msg_id, pending in self.pending_acks.items()
                if pending.router_id == router_id
            ]
            for pending in pending_for_router:
                try:
                    await self._message_handler._retry_message(
                        pending.message_id,
                        self._send_via_router,
                        self._select_router,
                    )
                except Exception as e:
                    logger.warning(f"Failed to retry message: {e}")
        
        await self._router_client.handle_router_failure(
            router_id=router_id,
            on_retry_messages=retry_messages,
        )
        
        self._stats["failovers"] += 1
    
    # -------------------------------------------------------------------------
    # BACKGROUND TASKS
    # -------------------------------------------------------------------------
    
    async def _keepalive_loop(self) -> None:
        """Send periodic pings to detect connection failures."""
        while self._running:
            try:
                await asyncio.sleep(self.keepalive_interval)
                if not self._running:
                    break
                
                for router_id, conn in list(self.connections.items()):
                    if conn.websocket.closed:
                        await self._handle_router_failure(router_id)
                        continue
                    
                    try:
                        sent_at = time.time()
                        await asyncio.wait_for(
                            conn.websocket.send_json({"type": "ping", "sent_at": sent_at}),
                            timeout=self.ping_timeout,
                        )
                        self._health_monitor.track_ping_response(router_id, True)
                    except asyncio.TimeoutError:
                        if self._health_monitor.track_ping_response(router_id, False):
                            await self._handle_router_failure(router_id)
                    except Exception as e:
                        logger.warning(f"Ping error for router {router_id[:16]}...: {e}")
                        await self._handle_router_failure(router_id)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Keepalive loop error: {e}")
    
    async def _connection_maintenance(self) -> None:
        """Periodically check and maintain connections."""
        while self._running:
            try:
                await asyncio.sleep(self.maintenance_interval)
                if not self._running:
                    break
                
                # Remove stale connections
                now = time.time()
                for router_id, conn in list(self.connections.items()):
                    if now - conn.last_seen > self.keepalive_interval * 2:
                        await self._handle_router_failure(router_id)
                
                # Ensure we have enough connections
                if len(self.connections) < self.min_connections:
                    await self._connection_manager.ensure_connections()
                
                # Check router rotation
                if self._router_client:
                    router_to_rotate = await self._router_client.check_rotation_needed()
                    if router_to_rotate:
                        await self._router_client.rotate_router(router_to_rotate, "periodic")
                        self._stats["routers_rotated"] += 1
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Maintenance loop error: {e}")
    
    async def _queue_processor(self) -> None:
        """Process queued messages when routers become available."""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                if not self._running:
                    break
                
                if self.message_queue and self.connections:
                    count = await self._message_handler.process_queue(
                        self._select_router,
                        self._send_via_router,
                    )
                    if count > 0:
                        logger.debug(f"Processed {count} queued messages")
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Queue processor error: {e}")
    
    async def _gossip_loop(self) -> None:
        """Periodically broadcast health gossip."""
        while self._running:
            try:
                await asyncio.sleep(self.gossip_interval)
                if not self._running:
                    break
                
                # Update observations
                for router_id, conn in self.connections.items():
                    self._health_monitor.update_observation(router_id, conn)
                
                # Create and broadcast gossip
                if self._health_monitor._own_observations:
                    gossip = self._health_monitor.create_gossip_message()
                    await self._broadcast_gossip(gossip)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Gossip loop error: {e}")
    
    async def _broadcast_gossip(self, gossip: HealthGossip) -> None:
        """Broadcast gossip message to all connected routers."""
        gossip_data = gossip.to_dict()
        
        for router_id, conn in list(self.connections.items()):
            if conn.websocket.closed:
                continue
            
            try:
                await conn.websocket.send_json({"type": "gossip", "payload": gossip_data})
                self._stats["gossip_sent"] += 1
            except Exception as e:
                logger.debug(f"Failed to send gossip via {router_id[:16]}...: {e}")
    
    async def _batch_flush_loop(self) -> None:
        """Background task to flush message batches."""
        while self._running:
            try:
                batch_interval = self.traffic_analysis_mitigation.batching.get_effective_interval()
                
                try:
                    await asyncio.wait_for(
                        self._message_handler._pending_batch_event.wait(),
                        timeout=batch_interval,
                    )
                    self._message_handler._pending_batch_event.clear()
                except asyncio.TimeoutError:
                    pass
                
                if not self._running:
                    break
                
                await self._message_handler.flush_batch()
                self._stats["batch_flushes"] += 1
            
            except asyncio.CancelledError:
                await self._message_handler.flush_batch()
                break
            except Exception as e:
                logger.warning(f"Batch flush loop error: {e}")
    
    async def _constant_rate_loop(self) -> None:
        """Background task to send messages at constant rate."""
        cr_config = self.traffic_analysis_mitigation.constant_rate
        
        while self._running:
            try:
                send_interval = cr_config.get_send_interval()
                await asyncio.sleep(send_interval)
                
                if not self._running:
                    break
                
                # Try to get a real message from batch
                message_to_send = None
                async with self._message_handler._batch_lock:
                    if self._message_handler._message_batch:
                        message_to_send = self._message_handler._message_batch.pop(0)
                
                if message_to_send:
                    try:
                        await self._message_handler._send_message_direct(
                            message_id=message_to_send["message_id"],
                            recipient_id=message_to_send["recipient_id"],
                            recipient_public_key=message_to_send["recipient_public_key"],
                            content=message_to_send["content"],
                            require_ack=message_to_send["require_ack"],
                            timeout_ms=message_to_send["timeout_ms"],
                            router_selector=self._select_router,
                            send_via_router=self._send_via_router,
                        )
                    except Exception as e:
                        logger.warning(f"Constant-rate send failed: {e}")
                else:
                    await self._send_padding_message()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Constant-rate loop error: {e}")
    
    async def _send_padding_message(self) -> None:
        """Send a padding message for constant-rate traffic."""
        if not self.connections:
            return
        
        cr_config = self.traffic_analysis_mitigation.constant_rate
        padding_content = os.urandom(cr_config.pad_to_size)
        fake_recipient_id = os.urandom(32).hex()
        fake_recipient_key = X25519PrivateKey.generate().public_key()
        
        router = self._select_router()
        if not router:
            return
        
        try:
            await self._send_via_router(
                router=router,
                message_id=str(uuid.uuid4()),
                recipient_id=fake_recipient_id,
                recipient_public_key=fake_recipient_key,
                content=padding_content,
                require_ack=False,
                is_cover_traffic=True,
            )
            self._stats["constant_rate_padding_sent"] += 1
        except Exception as e:
            logger.debug(f"Failed to send padding message: {e}")
    
    # -------------------------------------------------------------------------
    # STATE PERSISTENCE (Issue #111)
    # -------------------------------------------------------------------------
    
    def _get_connection_state(self) -> ConnectionState:
        """Create a ConnectionState snapshot."""
        self._state_sequence += 1
        
        return ConnectionState(
            version=STATE_VERSION,
            node_id=self.node_id,
            saved_at=time.time(),
            sequence_number=self._state_sequence,
            pending_acks=[ack.to_dict() for ack in self.pending_acks.values()],
            message_queue=[msg.to_dict() for msg in self.message_queue],
            seen_messages=list(self.seen_messages)[-self.max_seen_messages:],
            failover_states={
                router_id: state.to_dict()
                for router_id, state in self.failover_states.items()
            },
            stats=dict(self._stats),
        )
    
    async def _save_state(self) -> None:
        """Save current connection state to disk."""
        if not self.state_file:
            return
        
        state = self._get_connection_state()
        state_json = state.to_json()
        
        state_path = Path(self.state_file)
        temp_path = state_path.with_suffix('.tmp')
        
        try:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path.write_text(state_json)
            temp_path.rename(state_path)
            self._last_state_save = time.time()
        except Exception as e:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise IOError(f"Failed to save state: {e}") from e
    
    async def _load_state(self) -> Optional[ConnectionState]:
        """Load saved connection state from disk."""
        if not self.state_file:
            return None
        
        state_path = Path(self.state_file)
        if not state_path.exists():
            return None
        
        try:
            state_json = state_path.read_text()
            state = ConnectionState.from_json(state_json)
        except Exception as e:
            logger.warning(f"Failed to parse state file: {e}")
            return None
        
        if state.version != STATE_VERSION:
            return None
        
        if state.node_id != self.node_id:
            raise StateConflictError(f"State file belongs to different node")
        
        state_age = time.time() - state.saved_at
        if state_age > self.max_state_age:
            raise StaleStateError(f"State is {state_age/3600:.1f} hours old")
        
        return state
    
    async def _recover_state(self) -> bool:
        """Recover pending state from saved state file."""
        state = await self._load_state()
        if not state:
            return False
        
        # Restore pending ACKs
        for ack_data in state.pending_acks:
            try:
                ack = PendingAck.from_dict(ack_data)
                age = time.time() - ack.sent_at
                if age < (ack.timeout_ms / 1000) * 2:
                    self.pending_acks[ack.message_id] = ack
            except Exception as e:
                logger.warning(f"Failed to recover pending ACK: {e}")
        
        # Restore message queue
        for msg_data in state.message_queue:
            try:
                msg = PendingMessage.from_dict(msg_data)
                age = time.time() - msg.queued_at
                if age < self.MAX_QUEUE_AGE:
                    self.message_queue.append(msg)
            except Exception as e:
                logger.warning(f"Failed to recover queued message: {e}")
        
        # Restore seen messages
        self.seen_messages.update(state.seen_messages)
        
        # Restore failover states
        for router_id, fs_data in state.failover_states.items():
            try:
                fs = FailoverState.from_dict(fs_data)
                if fs.is_in_cooldown():
                    self.failover_states[router_id] = fs
            except Exception as e:
                logger.warning(f"Failed to recover failover state: {e}")
        
        # Restore statistics
        for key, value in state.stats.items():
            if key in self._stats:
                self._stats[key] = value
        
        self._state_sequence = state.sequence_number
        
        if self.on_state_recovered:
            try:
                self.on_state_recovered(state)
            except Exception as e:
                logger.warning(f"on_state_recovered callback error: {e}")
        
        try:
            Path(self.state_file).unlink()
        except Exception:
            pass
        
        return True
    
    async def _state_persistence_loop(self) -> None:
        """Background task to periodically save state."""
        while self._running:
            try:
                await asyncio.sleep(self.state_save_interval)
                if not self._running:
                    break
                
                if self.pending_acks or self.message_queue:
                    await self._save_state()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"State persistence error: {e}")
    
    def delete_state_file(self) -> bool:
        """Delete the state file if it exists."""
        if not self.state_file:
            return False
        
        state_path = Path(self.state_file)
        if state_path.exists():
            state_path.unlink()
            return True
        return False
    
    # -------------------------------------------------------------------------
    # DELEGATE METHODS (expose component functionality)
    # -------------------------------------------------------------------------
    
    def clear_router_cooldown(self, router_id: str) -> bool:
        """Manually clear cooldown for a router."""
        return self._connection_manager.clear_router_cooldown(router_id)
    
    def get_failover_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current failover states for all routers."""
        return self._connection_manager.get_failover_states()
    
    def get_health_observations(self) -> Dict[str, Any]:
        """Get current health observation state."""
        return self._health_monitor.get_health_observations()
    
    def is_router_flagged(self, router_id: str) -> bool:
        """Check if a router has been flagged for misbehavior."""
        return self._health_monitor.is_router_flagged(router_id)
    
    def get_flagged_routers(self) -> Dict[str, Dict[str, Any]]:
        """Get all flagged routers with their reports."""
        return self._health_monitor.get_flagged_routers()
    
    def clear_router_flag(self, router_id: str) -> bool:
        """Clear the misbehavior flag for a router."""
        return self._health_monitor.clear_router_flag(router_id)
    
    def get_misbehavior_detection_stats(self) -> Dict[str, Any]:
        """Get statistics for misbehavior detection."""
        return self._health_monitor.get_misbehavior_detection_stats()
    
    def get_anomaly_alerts(self) -> List[Dict[str, Any]]:
        """Get recent anomaly alerts."""
        return self._health_monitor.get_anomaly_alerts()
    
    def clear_anomaly_alerts(self) -> int:
        """Clear anomaly alerts."""
        return self._health_monitor.clear_anomaly_alerts()
    
    def set_privacy_level(self, level: PrivacyLevel) -> None:
        """Set the traffic analysis mitigation privacy level."""
        old_level = self.traffic_analysis_mitigation.privacy_level
        self.traffic_analysis_mitigation = TrafficAnalysisMitigationConfig.from_privacy_level(level)
        
        if self._message_handler:
            self._message_handler.traffic_mitigation_config = self.traffic_analysis_mitigation
        
        logger.info(f"Privacy level changed: {old_level.value} -> {level.value}")
    
    def get_traffic_analysis_mitigation_stats(self) -> Dict[str, Any]:
        """Get traffic analysis mitigation statistics."""
        tam = self.traffic_analysis_mitigation
        return {
            "privacy_level": tam.privacy_level.value,
            "batching": {
                "enabled": tam.batching.enabled,
                "batch_interval_ms": tam.batching.batch_interval_ms,
            },
            "jitter": {
                "enabled": tam.jitter.enabled,
            },
            "constant_rate": {
                "enabled": tam.constant_rate.enabled,
            },
            "stats": {
                k: v for k, v in self._stats.items()
                if k.startswith(("batched", "batch_", "jitter", "constant_rate"))
            },
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_node_client(
    private_key: Ed25519PrivateKey,
    encryption_private_key: X25519PrivateKey,
    discovery_client: Optional[DiscoveryClient] = None,
    seed_urls: Optional[List[str]] = None,
    **kwargs,
) -> NodeClient:
    """
    Create a node client with the given keys.
    
    Args:
        private_key: Ed25519 private key for signing
        encryption_private_key: X25519 private key for decryption
        discovery_client: Optional pre-configured discovery client
        seed_urls: Optional list of seed URLs for router discovery
        **kwargs: Additional NodeClient parameters
        
    Returns:
        Configured NodeClient
    """
    node_id = private_key.public_key().public_bytes_raw().hex()
    
    if seed_urls:
        kwargs["seed_urls"] = seed_urls
    
    client = NodeClient(
        node_id=node_id,
        private_key=private_key,
        encryption_private_key=encryption_private_key,
        **kwargs,
    )
    
    if discovery_client:
        client.discovery = discovery_client
    
    return client
