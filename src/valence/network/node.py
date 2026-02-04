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
"""

from __future__ import annotations

import asyncio
import ipaddress
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
    # Malicious router detection (Issue #119)
    MisbehaviorType,
    RouterBehaviorMetrics,
    MisbehaviorEvidence,
    MisbehaviorReport,
    NetworkBaseline,
    # Message padding for traffic analysis resistance
    pad_message,
    get_padded_size,
)
from .config import (
    TrafficAnalysisMitigationConfig,
    PrivacyLevel,
    BatchingConfig,
    TimingJitterConfig,
    ConstantRateConfig,
    MixNetworkConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# STATE PERSISTENCE (Issue #111)
# =============================================================================

# Version for state file format - increment on breaking changes
STATE_VERSION = 1


@dataclass
class ConnectionState:
    """
    Serializable connection state for persistence and recovery.
    
    This captures the essential state needed to recover after a
    disconnect/reconnect cycle:
    - Pending ACKs (messages awaiting acknowledgment)
    - Message queue (messages pending delivery)
    - Seen messages (for idempotent delivery)
    - Failover states (router cooldown tracking)
    
    Attributes:
        version: State format version for compatibility checking
        node_id: The node's identity
        saved_at: Timestamp when state was saved
        sequence_number: Monotonic counter to detect stale state
        pending_acks: List of serialized PendingAck records
        message_queue: List of serialized PendingMessage records
        seen_messages: Set of message IDs already processed
        failover_states: Dict of router_id -> serialized FailoverState
        stats: Accumulated statistics
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
        """Serialize to dictionary for JSON storage."""
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
        """Deserialize from dictionary."""
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
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "ConnectionState":
        """Deserialize from JSON string."""
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
    timeout_ms: int = 30000  # 30 seconds default
    retries: int = 0
    max_retries: int = 2  # Retry once via same router, then via different router
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for state persistence (without the public key object)."""
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
        """Deserialize from state persistence."""
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
        
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
    session: aiohttp.ClientSession  # Keep session for cleanup
    connected_at: float
    last_seen: float
    messages_sent: int = 0
    messages_received: int = 0
    ack_pending: int = 0
    ack_success: int = 0
    ack_failure: int = 0
    ping_latency_ms: float = 0.0
    
    # Back-pressure state
    back_pressure_active: bool = False
    back_pressure_until: float = 0.0  # Timestamp when back-pressure expires
    back_pressure_retry_ms: int = 1000  # Suggested retry delay
    
    @property
    def is_under_back_pressure(self) -> bool:
        """Check if this router is currently under back-pressure."""
        if not self.back_pressure_active:
            return False
        # Check if back-pressure has expired (retry_after_ms elapsed)
        return time.time() < self.back_pressure_until
    
    @property
    def ack_success_rate(self) -> float:
        """Calculate ACK success rate (0.0 to 1.0)."""
        total = self.ack_success + self.ack_failure
        if total == 0:
            return 1.0  # Assume good until proven otherwise
        return self.ack_success / total
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0.0 to 1.0)."""
        # Combine ACK success rate, latency, and load
        ack_score = self.ack_success_rate
        
        # Latency penalty (>500ms is bad)
        latency_score = max(0, 1.0 - (self.ping_latency_ms / 500))
        
        # Load from router capacity
        load_pct = self.router.capacity.get("current_load_pct", 0)
        load_score = 1.0 - (load_pct / 100)
        
        # Weighted combination
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
        """Serialize for state persistence."""
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
        """Deserialize from state persistence."""
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
        
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
    """
    Tracks failover state for a router.
    
    Used to implement exponential backoff for flapping routers
    and track queued messages during failover.
    """
    
    router_id: str
    failed_at: float
    fail_count: int
    cooldown_until: float
    queued_messages: List[PendingMessage] = field(default_factory=list)
    
    def is_in_cooldown(self) -> bool:
        """Check if router is still in cooldown period."""
        return time.time() < self.cooldown_until
    
    def remaining_cooldown(self) -> float:
        """Get remaining cooldown time in seconds."""
        return max(0, self.cooldown_until - time.time())
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for state persistence."""
        return {
            "router_id": self.router_id,
            "failed_at": self.failed_at,
            "fail_count": self.fail_count,
            "cooldown_until": self.cooldown_until,
            "queued_messages": [msg.to_dict() for msg in self.queued_messages],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailoverState":
        """Deserialize from state persistence."""
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


# =============================================================================
# COVER TRAFFIC CONFIGURATION (Issue #116)
# =============================================================================


@dataclass
class CoverTrafficConfig:
    """
    Configuration for cover traffic generation.
    
    Cover traffic helps resist traffic analysis by sending dummy messages
    when the node is idle. These messages are indistinguishable from real
    traffic to routers (encrypted, same sizes, same routing format).
    
    Attributes:
        enabled: Whether cover traffic is active
        rate_per_minute: Average cover messages per minute when idle
        idle_threshold_seconds: Seconds without sending before considered idle
        pad_messages: Whether to pad real messages to bucket sizes
        target_peers: List of peer node IDs to send cover traffic to.
                     If empty, cover traffic is sent to random routers.
        randomize_timing: Add jitter to timing (harder to fingerprint)
        min_interval_seconds: Minimum seconds between cover messages
        max_interval_seconds: Maximum seconds between cover messages
    """
    enabled: bool = False
    rate_per_minute: float = 2.0  # Default: ~2 msgs/min when idle
    idle_threshold_seconds: float = 30.0  # Consider idle after 30s no activity
    pad_messages: bool = True  # Pad real messages to bucket sizes
    target_peers: List[str] = field(default_factory=list)
    randomize_timing: bool = True
    min_interval_seconds: float = 15.0
    max_interval_seconds: float = 60.0
    
    def get_next_interval(self) -> float:
        """
        Get the next interval for sending cover traffic.
        
        Uses exponential distribution centered on rate_per_minute,
        with optional jitter for randomization.
        """
        if not self.enabled:
            return float('inf')
        
        # Base interval from rate
        base_interval = 60.0 / self.rate_per_minute if self.rate_per_minute > 0 else 60.0
        
        if self.randomize_timing:
            # Add randomization using exponential distribution
            interval = random.expovariate(1.0 / base_interval)
            # Clamp to min/max
            interval = max(self.min_interval_seconds, min(self.max_interval_seconds, interval))
        else:
            interval = base_interval
        
        return interval


# =============================================================================
# NODE CLIENT
# =============================================================================


@dataclass
class NodeClient:
    """
    User node that connects to routers for message relay.
    
    The node maintains connections to multiple routers for redundancy.
    Messages are encrypted end-to-end; routers only see routing metadata.
    
    Example:
        node = NodeClient(
            node_id="abc123...",
            private_key=my_ed25519_private_key,
            encryption_private_key=my_x25519_private_key,
        )
        await node.start()
        await node.send_message(recipient_id, recipient_pub_key, b"Hello!")
        await node.stop()
    
    With custom seeds (Issue #108 - Seed Redundancy):
        node = NodeClient(
            node_id="abc123...",
            private_key=my_ed25519_private_key,
            encryption_private_key=my_x25519_private_key,
            seed_urls=[
                "https://primary-seed.example.com:8470",
                "https://backup-seed.example.com:8470",
            ],
        )
    
    Attributes:
        node_id: Our Ed25519 public key (hex)
        private_key: Our Ed25519 private key for signing
        encryption_private_key: Our X25519 private key for decryption
        min_connections: Minimum router connections to maintain
        target_connections: Ideal number of router connections
        max_connections: Maximum router connections allowed
        seed_urls: List of seed URLs for router discovery (primary first)
    """
    
    # Identity
    node_id: str  # Ed25519 public key (hex)
    private_key: Ed25519PrivateKey
    encryption_private_key: X25519PrivateKey
    
    # Connection config
    min_connections: int = 3
    target_connections: int = 5
    max_connections: int = 8
    
    # Seed configuration (Issue #108)
    seed_urls: List[str] = field(default_factory=list)
    
    # Timing config
    keepalive_interval: float = 2.0  # seconds between pings (fast detection)
    ping_timeout: float = 3.0  # seconds to wait for pong
    maintenance_interval: float = 60.0  # seconds between maintenance runs
    reconnect_delay: float = 1.0  # seconds before reconnecting after failure
    failover_connect_timeout: float = 3.0  # seconds to wait when connecting during failover
    
    # Failover config
    initial_cooldown: float = 60.0  # 1 minute initial cooldown for failed router
    max_cooldown: float = 3600.0  # 1 hour maximum cooldown
    missed_pings_threshold: int = 2  # consecutive missed pings before failure
    
    # IP diversity: require different /16 subnets
    enforce_ip_diversity: bool = True
    ip_diversity_prefix: int = 16  # /16 subnet diversity
    
    # ==========================================================================
    # ECLIPSE ATTACK MITIGATIONS (Issue #118)
    # ==========================================================================
    
    # Router diversity enforcement - require routers from different ASNs/subnets
    # This prevents an attacker from surrounding the node with controlled routers
    min_diverse_subnets: int = 2  # Minimum different /16 subnets required
    min_diverse_asns: int = 2  # Minimum different ASNs (if available)
    asn_diversity_enabled: bool = True  # Enable ASN-based diversity
    
    # Periodic router rotation - prevent long-term eclipse
    rotation_interval: float = 3600.0  # 1 hour - rotate one router periodically
    rotation_max_age: float = 7200.0  # 2 hours - max time for any single connection
    rotation_enabled: bool = True  # Enable periodic rotation
    
    # Anomaly detection - detect coordinated router failures
    anomaly_window: float = 60.0  # Window to detect correlated failures
    anomaly_threshold: int = 3  # Number of similar failures to trigger alert
    anomaly_detection_enabled: bool = True  # Enable anomaly detection
    
    # Out-of-band seed verification
    oob_verification_enabled: bool = False  # Disabled by default
    oob_verification_url: Optional[str] = None  # URL for OOB verification
    oob_verification_interval: float = 3600.0  # 1 hour between checks
    
    # State
    connections: Dict[str, RouterConnection] = field(default_factory=dict)
    discovery: DiscoveryClient = field(default_factory=DiscoveryClient)
    message_queue: List[PendingMessage] = field(default_factory=list)
    
    # ACK tracking
    pending_acks: Dict[str, PendingAck] = field(default_factory=dict)
    seen_messages: Set[str] = field(default_factory=set)  # For idempotent delivery
    
    # Failover state tracking
    failover_states: Dict[str, FailoverState] = field(default_factory=dict)
    direct_mode: bool = False  # Graceful degradation when no routers available
    _missed_pings: Dict[str, int] = field(default_factory=dict, repr=False)  # Track consecutive missed pings
    
    # ACK configuration
    default_ack_timeout_ms: int = 30000  # 30 seconds
    max_seen_messages: int = 10000  # Limit seen message cache size
    
    # State persistence configuration (Issue #111)
    state_file: Optional[str] = None  # Path to state file, None to disable
    state_save_interval: float = 30.0  # Save state every N seconds
    max_state_age: float = 86400.0  # Maximum age of state to recover (24 hours)
    recover_state_on_start: bool = True  # Whether to recover state on start
    
    # Cover traffic configuration (Issue #116)
    cover_traffic: CoverTrafficConfig = field(default_factory=CoverTrafficConfig)
    
    # Traffic analysis mitigation configuration (Issue #120)
    traffic_analysis_mitigation: TrafficAnalysisMitigationConfig = field(
        default_factory=TrafficAnalysisMitigationConfig
    )
    
    # Callbacks for message handling
    on_message: Optional[Callable[[str, bytes], None]] = None
    on_ack_timeout: Optional[Callable[[str, str], None]] = None  # (message_id, recipient_id)
    on_state_recovered: Optional[Callable[[ConnectionState], None]] = None  # Called after state recovery
    
    # Internal state
    _running: bool = field(default=False, repr=False)
    _tasks: List[asyncio.Task] = field(default_factory=list, repr=False)
    _connected_subnets: Set[str] = field(default_factory=set, repr=False)
    _connected_asns: Set[str] = field(default_factory=set, repr=False)  # ASN diversity tracking (Issue #118)
    _state_sequence: int = field(default=0, repr=False)  # Monotonic counter for conflict detection
    _last_state_save: float = field(default=0.0, repr=False)
    
    # Eclipse mitigation state (Issue #118)
    _connection_timestamps: Dict[str, float] = field(default_factory=dict, repr=False)  # router_id -> connect time
    _last_rotation: float = field(default=0.0, repr=False)  # Last router rotation time
    _failure_events: List[Dict[str, Any]] = field(default_factory=list, repr=False)  # Recent failure events
    _anomaly_alerts: List[Dict[str, Any]] = field(default_factory=list, repr=False)  # Detected anomalies
    _last_oob_verification: float = field(default=0.0, repr=False)  # Last OOB verification time
    _oob_verified_routers: Set[str] = field(default_factory=set, repr=False)  # Routers verified via OOB
    
    # Cover traffic internal state (Issue #116)
    _last_real_message_time: float = field(default=0.0, repr=False)
    _cover_traffic_task: Optional[asyncio.Task] = field(default=None, repr=False)
    
    # Traffic analysis mitigation internal state (Issue #120)
    _message_batch: List[Dict[str, Any]] = field(default_factory=list, repr=False)
    _batch_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _batch_flush_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _constant_rate_task: Optional[asyncio.Task] = field(default=None, repr=False)
    _last_batch_flush: float = field(default=0.0, repr=False)
    _last_constant_rate_send: float = field(default=0.0, repr=False)
    _pending_batch_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    
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
        # Cover traffic stats (Issue #116)
        "cover_messages_sent": 0,
        "cover_messages_received": 0,
        "bytes_padded": 0,
        # Eclipse mitigation stats (Issue #118)
        "routers_rotated": 0,
        "diversity_rejections": 0,
        "anomalies_detected": 0,
        "oob_verifications": 0,
        "oob_verification_failures": 0,
        # Traffic analysis mitigation stats (Issue #120)
        "batched_messages": 0,
        "batch_flushes": 0,
        "jitter_delays_applied": 0,
        "total_jitter_ms": 0,
        "constant_rate_padding_sent": 0,
        "messages_with_jitter": 0,
    })
    
    # -------------------------------------------------------------------------
    # HEALTH GOSSIP CONFIGURATION
    # -------------------------------------------------------------------------
    
    # Gossip timing
    gossip_interval: float = 30.0  # Seconds between gossip broadcasts
    gossip_ttl: int = 2  # Max hops for gossip propagation
    
    # Observation aggregation weights
    own_observation_weight: float = 0.7  # Weight for own observations
    peer_observation_weight: float = 0.3  # Weight for peer observations
    
    # Observation limits
    max_observations_per_gossip: int = 10  # Limit observations per gossip message
    max_peer_observations: int = 100  # Max peer observations to cache
    observation_max_age: float = 300.0  # Max age of observations (5 minutes)
    
    # Peer health observations: {peer_node_id: {router_id: RouterHealthObservation}}
    _peer_observations: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)
    
    # Our own router health observations: {router_id: RouterHealthObservation}
    _own_observations: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # Connected peers (nodes we can gossip with)
    _connected_peers: Set[str] = field(default_factory=set, repr=False)
    
    # -------------------------------------------------------------------------
    # CIRCUIT CONFIGURATION (Issue #115 - Enhanced Privacy)
    # -------------------------------------------------------------------------
    
    # Circuit building
    circuit_min_hops: int = 2  # Minimum hops for privacy
    circuit_max_hops: int = 3  # Maximum hops (latency vs privacy tradeoff)
    circuit_lifetime: float = 600.0  # 10 minutes default lifetime
    circuit_max_messages: int = 100  # Rotate circuit after this many messages
    circuit_build_timeout: float = 10.0  # Timeout for circuit establishment
    
    # Circuit state
    _circuits: Dict[str, Circuit] = field(default_factory=dict, repr=False)
    _circuit_keys: Dict[str, List[bytes]] = field(default_factory=dict, repr=False)  # circuit_id -> hop keys
    _active_circuit_id: Optional[str] = field(default=None, repr=False)  # Currently selected circuit
    _circuit_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    
    # Ephemeral keys for circuit establishment (circuit_id -> list of private keys per hop)
    _circuit_ephemeral_keys: Dict[str, List[Any]] = field(default_factory=dict, repr=False)
    
    # Circuit usage enabled flag
    use_circuits: bool = False  # Disabled by default for backward compatibility
    
    # Queue limits
    MAX_QUEUE_SIZE: int = 1000
    MAX_QUEUE_AGE: float = 3600.0  # 1 hour
    
    # -------------------------------------------------------------------------
    # MALICIOUS ROUTER DETECTION (Issue #119)
    # -------------------------------------------------------------------------
    
    # Detection configuration
    misbehavior_detection_enabled: bool = True
    
    # Thresholds for detecting misbehavior
    min_messages_for_detection: int = 20  # Minimum samples before flagging
    delivery_rate_threshold: float = 0.75  # Flag if delivery rate below this
    latency_threshold_stddevs: float = 2.0  # Flag if latency > N stddevs above baseline
    ack_failure_threshold: float = 0.30  # Flag if ACK failure rate exceeds this
    
    # Severity thresholds
    mild_severity_threshold: float = 0.3  # Below this = mild misbehavior
    severe_severity_threshold: float = 0.7  # Above this = severe misbehavior
    
    # Auto-avoidance configuration
    auto_avoid_flagged_routers: bool = True
    flagged_router_penalty: float = 0.1  # Multiply weight by this for flagged routers
    
    # Reporting configuration
    report_to_seeds: bool = True
    report_cooldown_seconds: float = 300.0  # Don't report same router within 5 min
    max_evidence_per_report: int = 10  # Limit evidence items per report
    
    # Network baseline configuration
    baseline_update_interval: float = 60.0  # Seconds between baseline updates
    baseline_min_samples: int = 3  # Min routers for baseline calculation
    
    # Per-router behavior metrics: {router_id: RouterBehaviorMetrics}
    _router_behavior_metrics: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # Network baseline for comparison
    _network_baseline: Any = field(default=None, repr=False)
    
    # Flagged routers: {router_id: MisbehaviorReport}
    _flagged_routers: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    # Report tracking: {router_id: last_report_timestamp}
    _last_misbehavior_reports: Dict[str, float] = field(default_factory=dict, repr=False)
    
    # Callback for misbehavior detection
    on_router_flagged: Optional[Callable[[str, Any], None]] = None  # (router_id, report)
    
    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------
    
    async def start(self) -> None:
        """
        Start the node - discover routers and connect.
        
        This initiates router discovery, establishes connections,
        and starts background maintenance tasks.
        
        If seed_urls were provided during initialization, they will be
        configured on the discovery client (primary seed first, with
        automatic fallback to secondaries).
        
        If state recovery is enabled and a state file exists, the node
        will attempt to recover pending state from the previous session.
        """
        if self._running:
            logger.warning("Node already running")
            return
        
        self._running = True
        logger.info(f"Starting node {self.node_id[:16]}...")
        
        # Attempt state recovery (Issue #111)
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
        
        # Configure discovery client with seed URLs (Issue #108)
        if self.seed_urls:
            for seed_url in self.seed_urls:
                self.discovery.add_seed(seed_url)
            logger.info(f"Configured {len(self.seed_urls)} seed URLs for discovery")
        
        # Initial connection establishment
        await self._ensure_connections()
        
        # Start background tasks
        self._tasks.append(asyncio.create_task(self._connection_maintenance()))
        self._tasks.append(asyncio.create_task(self._keepalive_loop()))
        self._tasks.append(asyncio.create_task(self._queue_processor()))
        self._tasks.append(asyncio.create_task(self._gossip_loop()))
        
        # Start state persistence task if enabled (Issue #111)
        if self.state_file:
            self._tasks.append(asyncio.create_task(self._state_persistence_loop()))
        
        # Start traffic analysis mitigation tasks (Issue #120)
        if self.traffic_analysis_mitigation.batching.enabled:
            self._batch_flush_task = asyncio.create_task(self._batch_flush_loop())
            self._tasks.append(self._batch_flush_task)
            logger.info(
                f"Message batching enabled: interval={self.traffic_analysis_mitigation.batching.batch_interval_ms}ms, "
                f"max_size={self.traffic_analysis_mitigation.batching.max_batch_size}"
            )
        
        if self.traffic_analysis_mitigation.constant_rate.enabled:
            self._constant_rate_task = asyncio.create_task(self._constant_rate_loop())
            self._tasks.append(self._constant_rate_task)
            logger.info(
                f"Constant-rate sending enabled: {self.traffic_analysis_mitigation.constant_rate.messages_per_minute} msg/min"
            )
        
        logger.info(
            f"Node started with {len(self.connections)} router connections"
        )
    
    async def stop(self) -> None:
        """Stop the node and close all connections.
        
        If state persistence is enabled, saves the current state
        before shutting down for recovery on next start.
        """
        if not self._running:
            return
        
        self._running = False
        logger.info("Stopping node...")
        
        # Save state before stopping (Issue #111)
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
        for router_id, conn in list(self.connections.items()):
            await self._close_connection(router_id, conn)
        self.connections.clear()
        self._connected_subnets.clear()
        
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
        
        The message is encrypted end-to-end using the recipient's X25519
        public key. The router only sees the encrypted payload and
        routing metadata.
        
        Traffic Analysis Mitigations (Issue #120):
        - If batching is enabled, message is queued and sent with batch
        - If jitter is enabled, a random delay is added before sending
        - If constant-rate is enabled, message waits for next send slot
        - Messages may be padded to fixed sizes for traffic analysis resistance
        
        Args:
            recipient_id: Recipient's node ID (Ed25519 public key hex)
            recipient_public_key: Recipient's X25519 public key for encryption
            content: Raw message bytes to encrypt and send
            require_ack: Whether to require end-to-end acknowledgment
            ack_timeout_ms: ACK timeout in milliseconds (default: 30000)
            bypass_mitigations: Skip batching/jitter (for internal use)
            
        Returns:
            Message ID (UUID string)
            
        Raises:
            NoRoutersAvailableError: If no routers are connected and queue is full
        """
        message_id = str(uuid.uuid4())
        timeout_ms = ack_timeout_ms or self.default_ack_timeout_ms
        
        # Update real message time for cover traffic tracking
        self._last_real_message_time = time.time()
        
        # Apply traffic analysis mitigations if enabled and not bypassed
        tam = self.traffic_analysis_mitigation
        
        # Pad message content if enabled (for traffic analysis resistance)
        if tam.batching.enabled or tam.constant_rate.enabled:
            if tam.constant_rate.enabled and tam.constant_rate.pad_to_size > 0:
                original_size = len(content)
                content = pad_message(content, tam.constant_rate.pad_to_size)
                if len(content) > original_size:
                    self._stats["bytes_padded"] += len(content) - original_size
        
        # Route through batching if enabled
        if tam.batching.enabled and not bypass_mitigations:
            return await self._add_to_batch(
                message_id=message_id,
                recipient_id=recipient_id,
                recipient_public_key=recipient_public_key,
                content=content,
                require_ack=require_ack,
                timeout_ms=timeout_ms,
            )
        
        # Route through constant-rate if enabled (without batching)
        if tam.constant_rate.enabled and not bypass_mitigations:
            return await self._schedule_constant_rate_send(
                message_id=message_id,
                recipient_id=recipient_id,
                recipient_public_key=recipient_public_key,
                content=content,
                require_ack=require_ack,
                timeout_ms=timeout_ms,
            )
        
        # Apply timing jitter if enabled
        if tam.jitter.enabled and not bypass_mitigations:
            jitter_delay = tam.jitter.get_jitter_delay()
            if jitter_delay > 0:
                self._stats["jitter_delays_applied"] += 1
                self._stats["total_jitter_ms"] += int(jitter_delay * 1000)
                self._stats["messages_with_jitter"] += 1
                logger.debug(f"Applying {jitter_delay*1000:.1f}ms jitter to message {message_id}")
                await asyncio.sleep(jitter_delay)
        
        # Direct send path (no batching/constant-rate)
        return await self._send_message_direct(
            message_id=message_id,
            recipient_id=recipient_id,
            recipient_public_key=recipient_public_key,
            content=content,
            require_ack=require_ack,
            timeout_ms=timeout_ms,
        )
    
    async def _send_message_direct(
        self,
        message_id: str,
        recipient_id: str,
        recipient_public_key: X25519PublicKey,
        content: bytes,
        require_ack: bool,
        timeout_ms: int,
    ) -> str:
        """
        Send a message directly without traffic analysis mitigations.
        
        This is the core send path used by both direct sends and
        batched/scheduled sends after mitigation processing.
        """
        # Select best router
        router = self._select_router()
        if not router:
            # Queue message for later delivery
            if len(self.message_queue) >= self.MAX_QUEUE_SIZE:
                self._stats["messages_dropped"] += 1
                raise NoRoutersAvailableError(
                    "No routers available and message queue full"
                )
            
            self.message_queue.append(PendingMessage(
                message_id=message_id,
                recipient_id=recipient_id,
                content=content,
                recipient_public_key=recipient_public_key,
                queued_at=time.time(),
            ))
            self._stats["messages_queued"] += 1
            logger.debug(f"Message {message_id} queued (no routers available)")
            return message_id
        
        # Send via selected router
        await self._send_via_router(
            router,
            message_id,
            recipient_id,
            recipient_public_key,
            content,
            require_ack=require_ack,
        )
        
        # Track pending ACK if required
        if require_ack:
            self.pending_acks[message_id] = PendingAck(
                message_id=message_id,
                recipient_id=recipient_id,
                content=content,
                recipient_public_key=recipient_public_key,
                sent_at=time.time(),
                router_id=router.router_id,
                timeout_ms=timeout_ms,
            )
            asyncio.create_task(self._wait_for_ack(message_id))
        
        return message_id
    
    def get_stats(self) -> Dict[str, Any]:
        """Get node statistics."""
        return {
            **self._stats,
            "active_connections": len(self.connections),
            "queued_messages": len(self.message_queue),
            "connected_subnets": len(self._connected_subnets),
            "connected_asns": len(self._connected_asns),  # Eclipse mitigation (Issue #118)
            "pending_acks": len(self.pending_acks),
            "seen_messages_cached": len(self.seen_messages),
            "routers_in_cooldown": sum(
                1 for state in self.failover_states.values()
                if state.is_in_cooldown()
            ),
            "routers_under_back_pressure": sum(
                1 for conn in self.connections.values()
                if conn.is_under_back_pressure
            ),
            "direct_mode": self.direct_mode,
            "own_health_observations": len(self._own_observations),
            "peer_observation_sources": len(self._peer_observations),
            # Eclipse mitigation stats (Issue #118)
            "diversity_met": self._check_diversity_requirements(),
            "recent_anomalies": len(self._anomaly_alerts),
            "oob_verified_routers": len(self._oob_verified_routers),
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
    # STATE PERSISTENCE (Issue #111)
    # -------------------------------------------------------------------------
    
    def _get_connection_state(self) -> ConnectionState:
        """
        Create a ConnectionState snapshot of current recoverable state.
        
        Returns:
            ConnectionState object with serialized pending state
        """
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
        """
        Save current connection state to disk.
        
        State is saved atomically by writing to a temp file first,
        then renaming to prevent corruption on crash.
        
        Raises:
            IOError: If state file cannot be written
        """
        if not self.state_file:
            return
        
        state = self._get_connection_state()
        state_json = state.to_json()
        
        # Write atomically via temp file
        state_path = Path(self.state_file)
        temp_path = state_path.with_suffix('.tmp')
        
        try:
            # Ensure parent directory exists
            state_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to temp file
            temp_path.write_text(state_json)
            
            # Atomic rename
            temp_path.rename(state_path)
            
            self._last_state_save = time.time()
            logger.debug(
                f"Saved state: seq={state.sequence_number}, "
                f"pending_acks={len(state.pending_acks)}, "
                f"queue={len(state.message_queue)}"
            )
        except Exception as e:
            # Clean up temp file on error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise IOError(f"Failed to save state: {e}") from e
    
    async def _load_state(self) -> Optional[ConnectionState]:
        """
        Load saved connection state from disk.
        
        Returns:
            ConnectionState if file exists and is valid, None otherwise
            
        Raises:
            StaleStateError: If saved state is too old
            StateConflictError: If saved state doesn't match this node
        """
        if not self.state_file:
            return None
        
        state_path = Path(self.state_file)
        if not state_path.exists():
            logger.debug("No state file found")
            return None
        
        try:
            state_json = state_path.read_text()
            state = ConnectionState.from_json(state_json)
        except Exception as e:
            logger.warning(f"Failed to parse state file: {e}")
            return None
        
        # Version check
        if state.version != STATE_VERSION:
            logger.warning(
                f"State version mismatch: file={state.version}, "
                f"current={STATE_VERSION}"
            )
            return None
        
        # Node ID check - prevent applying state from different node
        if state.node_id != self.node_id:
            raise StateConflictError(
                f"State file belongs to different node: {state.node_id[:16]}..."
            )
        
        # Age check
        state_age = time.time() - state.saved_at
        if state_age > self.max_state_age:
            raise StaleStateError(
                f"State is {state_age/3600:.1f} hours old "
                f"(max: {self.max_state_age/3600:.1f} hours)"
            )
        
        logger.info(
            f"Loaded state: seq={state.sequence_number}, "
            f"age={state_age:.0f}s, pending_acks={len(state.pending_acks)}"
        )
        return state
    
    async def _recover_state(self) -> bool:
        """
        Recover pending state from saved state file.
        
        This restores:
        - Pending ACKs (messages awaiting acknowledgment)
        - Message queue (messages pending delivery)
        - Seen messages (for idempotent delivery)
        - Failover states (router cooldown tracking)
        - Statistics
        
        Returns:
            True if state was recovered, False if no state to recover
            
        Raises:
            StaleStateError: If saved state is too old
            StateConflictError: If saved state doesn't match this node
        """
        state = await self._load_state()
        if not state:
            return False
        
        # Restore pending ACKs
        recovered_acks = 0
        for ack_data in state.pending_acks:
            try:
                ack = PendingAck.from_dict(ack_data)
                # Only recover if not expired
                age = time.time() - ack.sent_at
                if age < (ack.timeout_ms / 1000) * 2:  # 2x timeout as max age
                    self.pending_acks[ack.message_id] = ack
                    recovered_acks += 1
                    # Schedule ACK wait
                    asyncio.create_task(self._wait_for_ack(ack.message_id))
            except Exception as e:
                logger.warning(f"Failed to recover pending ACK: {e}")
        
        # Restore message queue
        recovered_msgs = 0
        for msg_data in state.message_queue:
            try:
                msg = PendingMessage.from_dict(msg_data)
                # Only recover if not expired
                age = time.time() - msg.queued_at
                if age < self.MAX_QUEUE_AGE:
                    self.message_queue.append(msg)
                    recovered_msgs += 1
            except Exception as e:
                logger.warning(f"Failed to recover queued message: {e}")
        
        # Restore seen messages
        self.seen_messages.update(state.seen_messages)
        
        # Restore failover states (only if still in cooldown)
        for router_id, fs_data in state.failover_states.items():
            try:
                fs = FailoverState.from_dict(fs_data)
                # Only restore if still in cooldown
                if fs.is_in_cooldown():
                    self.failover_states[router_id] = fs
            except Exception as e:
                logger.warning(f"Failed to recover failover state: {e}")
        
        # Restore statistics
        for key, value in state.stats.items():
            if key in self._stats:
                self._stats[key] = value
        
        # Update sequence number to prevent conflicts
        self._state_sequence = state.sequence_number
        
        logger.info(
            f"Recovered state: {recovered_acks} pending ACKs, "
            f"{recovered_msgs} queued messages, "
            f"{len(self.seen_messages)} seen messages"
        )
        
        # Notify callback if registered
        if self.on_state_recovered:
            try:
                self.on_state_recovered(state)
            except Exception as e:
                logger.warning(f"on_state_recovered callback error: {e}")
        
        # Delete state file after successful recovery
        try:
            Path(self.state_file).unlink()
            logger.debug("Deleted state file after recovery")
        except Exception as e:
            logger.warning(f"Failed to delete state file: {e}")
        
        return True
    
    async def _state_persistence_loop(self) -> None:
        """Background task to periodically save state."""
        while self._running:
            try:
                await asyncio.sleep(self.state_save_interval)
                if not self._running:
                    break
                
                # Only save if there's something to save
                if self.pending_acks or self.message_queue:
                    await self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"State persistence error: {e}")
    
    def delete_state_file(self) -> bool:
        """
        Delete the state file if it exists.
        
        Useful for testing or when you want to start fresh.
        
        Returns:
            True if file was deleted, False if it didn't exist
        """
        if not self.state_file:
            return False
        
        state_path = Path(self.state_file)
        if state_path.exists():
            state_path.unlink()
            logger.info("Deleted state file")
            return True
        return False
    
    # -------------------------------------------------------------------------
    # CONNECTION MANAGEMENT
    # -------------------------------------------------------------------------
    
    async def _ensure_connections(self) -> None:
        """Ensure we have enough router connections."""
        attempts = 0
        max_attempts = 3
        
        while len(self.connections) < self.target_connections and attempts < max_attempts:
            needed = self.target_connections - len(self.connections)
            
            # Get excluded router IDs (already connected + in cooldown)
            excluded_ids = set(self.connections.keys())
            
            # Also exclude routers in cooldown (flapping protection)
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
                if len(self.connections) >= self.target_connections:
                    break
                
                if router.router_id in excluded_ids:
                    continue
                
                # Check IP diversity
                if self.enforce_ip_diversity:
                    if not self._check_ip_diversity(router):
                        logger.debug(
                            f"Skipping router {router.router_id[:16]}... "
                            f"(IP diversity check failed)"
                        )
                        continue
                
                try:
                    await self._connect_to_router(router)
                    # Disable direct mode if we have a connection
                    if self.direct_mode and self.connections:
                        self._disable_direct_mode()
                except Exception as e:
                    logger.warning(
                        f"Failed to connect to router {router.router_id[:16]}...: {e}"
                    )
                    self._stats["connections_failed"] += 1
                    continue
            
            attempts += 1
        
        if len(self.connections) < self.min_connections:
            logger.warning(
                f"Only {len(self.connections)} connections established "
                f"(minimum: {self.min_connections})"
            )
            # Enable direct mode if below minimum
            if not self.connections:
                self._enable_direct_mode()
    
    def _check_ip_diversity(self, router: RouterInfo) -> bool:
        """
        Check if connecting to this router maintains IP diversity.
        
        We require routers to be in different /16 subnets to prevent
        a single network operator from controlling all our connections.
        
        Args:
            router: Router to check
            
        Returns:
            True if router passes diversity check
        """
        if not router.endpoints:
            return False
        
        try:
            # Parse IP from first endpoint
            # Formats: "ip:port", "[ipv6]:port", "hostname:port"
            endpoint = router.endpoints[0]
            
            # Handle IPv6 bracket notation: [ipv6]:port
            if endpoint.startswith("["):
                bracket_end = endpoint.find("]")
                if bracket_end == -1:
                    return True  # Malformed, allow
                host = endpoint[1:bracket_end]
            else:
                # IPv4 or hostname: ip:port
                host = endpoint.split(":")[0]
            
            # Try to parse as IP address
            try:
                ip = ipaddress.ip_address(host)
            except ValueError:
                # Not an IP, could be hostname - allow it
                return True
            
            # Get /16 subnet for IPv4, /48 for IPv6
            if ip.version == 4:
                network = ipaddress.ip_network(
                    f"{ip}/{self.ip_diversity_prefix}", strict=False
                )
            else:
                # IPv6: use /48 prefix
                network = ipaddress.ip_network(f"{ip}/48", strict=False)
            
            subnet_key = str(network)
            
            if subnet_key in self._connected_subnets:
                self._stats["diversity_rejections"] += 1
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"IP diversity check error: {e}")
            return True  # Allow on error
    
    def _check_asn_diversity(self, router: RouterInfo) -> bool:
        """
        Check if connecting to this router maintains ASN diversity.
        
        ASN (Autonomous System Number) diversity helps prevent eclipse attacks
        where an attacker controls multiple IPs in the same network.
        
        Args:
            router: Router to check
            
        Returns:
            True if router passes ASN diversity check
        """
        if not self.asn_diversity_enabled:
            return True
        
        # Get ASN from router info (if provided by seed)
        asn = router.asn or router.capacity.get("asn") or router.health.get("asn")
        if not asn:
            # No ASN info available - allow but log
            logger.debug(f"No ASN info for router {router.router_id[:16]}...")
            return True
        
        asn_str = str(asn)
        
        # Check if we already have a router from this ASN
        if asn_str in self._connected_asns:
            # Only reject if we have enough diversity already
            if len(self._connected_asns) >= self.min_diverse_asns:
                self._stats["diversity_rejections"] += 1
                logger.debug(
                    f"Rejecting router {router.router_id[:16]}... - "
                    f"already connected to ASN {asn_str}"
                )
                return False
        
        return True
    
    def _check_diversity_requirements(self) -> bool:
        """
        Check if current connections meet diversity requirements.
        
        Used to detect potential eclipse attack scenarios.
        
        Returns:
            True if diversity requirements are met
        """
        # Check subnet diversity
        if len(self._connected_subnets) < min(self.min_diverse_subnets, len(self.connections)):
            return False
        
        # Check ASN diversity (if enabled and data available)
        if self.asn_diversity_enabled and self._connected_asns:
            if len(self._connected_asns) < min(self.min_diverse_asns, len(self.connections)):
                return False
        
        return True
    
    def _add_subnet(self, router: RouterInfo) -> None:
        """Track the subnet and ASN of a connected router."""
        if not router.endpoints:
            return
        
        try:
            endpoint = router.endpoints[0]
            
            # Handle IPv6 bracket notation: [ipv6]:port
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
                    f"{ip}/{self.ip_diversity_prefix}", strict=False
                )
            else:
                network = ipaddress.ip_network(f"{ip}/48", strict=False)
            
            self._connected_subnets.add(str(network))
            
        except Exception:
            pass
        
        # Track ASN (Issue #118 - Eclipse mitigation)
        asn = router.asn or router.capacity.get("asn") or router.health.get("asn")
        if asn:
            self._connected_asns.add(str(asn))
    
    def _remove_subnet(self, router: RouterInfo) -> None:
        """Remove subnet and ASN tracking when disconnecting from router."""
        if not router.endpoints:
            return
        
        try:
            endpoint = router.endpoints[0]
            
            # Handle IPv6 bracket notation: [ipv6]:port
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
                    f"{ip}/{self.ip_diversity_prefix}", strict=False
                )
            else:
                network = ipaddress.ip_network(f"{ip}/48", strict=False)
            
            self._connected_subnets.discard(str(network))
            
        except Exception:
            pass
        
        # Remove ASN tracking (Issue #118 - Eclipse mitigation)
        asn = router.asn or router.capacity.get("asn") or router.health.get("asn")
        if asn:
            self._connected_asns.discard(str(asn))
    
    async def _connect_to_router(self, router: RouterInfo) -> None:
        """
        Establish WebSocket connection to a router.
        
        Args:
            router: Router to connect to
            
        Raises:
            ConnectionError: If connection fails
        """
        if not router.endpoints:
            raise ConnectionError("Router has no endpoints")
        
        endpoint = router.endpoints[0]
        
        # Try each endpoint
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
                        response = await asyncio.get_event_loop().run_in_executor(
                            None, lambda: __import__('json').loads(msg.data)
                        )
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
                self._stats["connections_established"] += 1
                
                # Track connection time for eclipse mitigation (Issue #118)
                self._connection_timestamps[router.router_id] = now
                
                # Start receive loop
                self._tasks.append(
                    asyncio.create_task(self._receive_loop(router.router_id))
                )
                
                logger.info(
                    f"Connected to router {router.router_id[:16]}... "
                    f"at {endpoint}"
                )
                return
                
            except ConnectionError:
                raise
            except Exception as e:
                logger.debug(f"Failed to connect to {endpoint}: {e}")
                continue
        
        raise ConnectionError(f"Failed to connect to any endpoint for router")
    
    async def _close_connection(
        self,
        router_id: str,
        conn: RouterConnection,
    ) -> None:
        """Close a router connection and clean up."""
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
    
    # -------------------------------------------------------------------------
    # MESSAGE HANDLING
    # -------------------------------------------------------------------------
    
    async def _send_via_router(
        self,
        router: RouterInfo,
        message_id: str,
        recipient_id: str,
        recipient_public_key: X25519PublicKey,
        content: bytes,
        require_ack: bool = True,
    ) -> None:
        """Send an encrypted message via a specific router."""
        conn = self.connections.get(router.router_id)
        if not conn or conn.websocket.closed:
            raise ConnectionError(f"Not connected to router {router.router_id[:16]}...")
        
        # Encrypt for recipient - include message_id and require_ack in payload
        # The content should include ACK metadata for the recipient
        payload_with_ack = {
            "content": content.decode() if isinstance(content, bytes) else content,
            "message_id": message_id,
            "require_ack": require_ack,
            "sender_id": self.node_id,
        }
        encrypted = encrypt_message(
            __import__('json').dumps(payload_with_ack).encode(),
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
        self._stats["messages_sent"] += 1
        
        logger.debug(f"Sent message {message_id} via router {router.router_id[:16]}...")
    
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
                        data = __import__('json').loads(msg.data)
                    except __import__('json').JSONDecodeError:
                        logger.warning("Received invalid JSON from router")
                        continue
                    
                    msg_type = data.get("type")
                    
                    if msg_type == "deliver":
                        await self._handle_deliver(data, conn)
                    
                    elif msg_type == "pong":
                        await self._handle_pong(data, conn)
                    
                    elif msg_type == "ack":
                        await self._handle_ack(data, conn)
                    
                    elif msg_type == "back_pressure":
                        await self._handle_back_pressure(data, conn)
                    
                    elif msg_type == "error":
                        logger.warning(
                            f"Router error: {data.get('message', 'unknown')}"
                        )
                    
                    elif msg_type == "gossip":
                        await self._handle_gossip_message(data, conn)
                    
                    else:
                        logger.debug(f"Unknown message type from router: {msg_type}")
                    
                    conn.last_seen = time.time()
                
                elif msg.type == WSMsgType.ERROR:
                    logger.warning(f"WebSocket error: {conn.websocket.exception()}")
                    break
                
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED):
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Receive loop error for router {router_id[:16]}...: {e}")
        
        finally:
            # Handle disconnection
            if router_id in self.connections:
                await self._handle_router_failure(router_id)
    
    async def _handle_deliver(
        self,
        data: Dict[str, Any],
        conn: RouterConnection,
    ) -> None:
        """Handle an incoming message delivery."""
        relay_message_id = data.get("message_id")
        payload = data.get("payload")
        
        if not payload:
            logger.warning(f"Received delivery without payload: {relay_message_id}")
            return
        
        conn.messages_received += 1
        self._stats["messages_received"] += 1
        
        # Decrypt and deliver to callback
        try:
            # Extract sender public key from payload
            sender_public_hex = payload.get("sender_public")
            if not sender_public_hex:
                logger.warning("Received message without sender public key")
                return
            
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            sender_public = Ed25519PublicKey.from_public_bytes(
                bytes.fromhex(sender_public_hex)
            )
            
            # Decrypt
            plaintext = decrypt_message(
                payload,
                self.encryption_private_key,
                sender_public,
            )
            
            # Parse the decrypted payload
            try:
                inner_payload = __import__('json').loads(plaintext.decode())
            except (ValueError, UnicodeDecodeError):
                # Fallback for non-JSON payloads (legacy)
                inner_payload = {"content": plaintext.decode()}
            
            # Extract message metadata
            inner_message_id = inner_payload.get("message_id")
            require_ack = inner_payload.get("require_ack", False)
            sender_id = inner_payload.get("sender_id", sender_public_hex)
            content = inner_payload.get("content", plaintext)
            
            # Check if this is an ACK message
            if isinstance(content, str):
                try:
                    content_data = __import__('json').loads(content)
                    if content_data.get("type") == "ack":
                        ack = AckMessage.from_dict(content_data)
                        await self._handle_e2e_ack(ack)
                        return
                except (ValueError, TypeError):
                    pass
            
            # Idempotent delivery - skip if we've already seen this message
            if inner_message_id and self._is_duplicate_message(inner_message_id):
                logger.debug(f"Duplicate message {inner_message_id}, skipping")
                self._stats["messages_deduplicated"] += 1
                # Still send ACK for duplicates (sender may not have received our first ACK)
                if require_ack:
                    # Get sender's encryption key from directory (simplified: use signing key)
                    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
                    # Note: In production, we'd look up the sender's X25519 key
                    # For now, we log a warning
                    logger.debug(f"Would send ACK for duplicate {inner_message_id}")
                return
            
            # Deliver to callback
            if self.on_message:
                if isinstance(content, str):
                    content_bytes = content.encode()
                else:
                    content_bytes = content if isinstance(content, bytes) else str(content).encode()
                await self.on_message(sender_id, content_bytes)
            
            # Send E2E ACK if requested
            if require_ack and inner_message_id:
                # Note: We need the sender's X25519 public key for encrypted ACK
                # For now, log that we would send an ACK
                # In production, this would look up the key from a directory
                self._stats["acks_sent"] = self._stats.get("acks_sent", 0) + 1
                logger.debug(f"ACK requested for {inner_message_id} from {sender_id[:16]}...")
            
        except Exception as e:
            logger.warning(f"Failed to process message {relay_message_id}: {e}")
    
    async def _handle_pong(
        self,
        data: Dict[str, Any],
        conn: RouterConnection,
    ) -> None:
        """Handle pong response to our ping."""
        sent_at = data.get("sent_at")
        if sent_at:
            conn.ping_latency_ms = (time.time() - sent_at) * 1000
            
            # Update our own health observation for this router
            self._update_own_observation(conn.router.router_id, conn)
    
    async def _handle_ack(
        self,
        data: Dict[str, Any],
        conn: RouterConnection,
    ) -> None:
        """Handle acknowledgment for a sent message."""
        message_id = data.get("message_id")
        success = data.get("success", True)
        
        conn.ack_pending = max(0, conn.ack_pending - 1)
        
        if success:
            conn.ack_success += 1
        else:
            conn.ack_failure += 1
            logger.debug(f"Message {message_id} delivery failed")
        
        # Update health observation after ACK
        self._update_own_observation(conn.router.router_id, conn)
        
        # Record ACK outcome for misbehavior detection (Issue #119)
        self._record_ack_outcome(conn.router.router_id, success)
    
    async def _handle_gossip_message(
        self,
        data: Dict[str, Any],
        conn: RouterConnection,
    ) -> None:
        """Handle incoming health gossip from router."""
        payload = data.get("payload", {})
        
        try:
            gossip = HealthGossip.from_dict(payload)
            self._handle_gossip(gossip)
            
            # Optionally propagate if TTL > 1
            if gossip.ttl > 1:
                # Decrement TTL and forward to other routers
                gossip.ttl -= 1
                await self._propagate_gossip(gossip, exclude_router=conn.router.router_id)
                
        except Exception as e:
            logger.warning(f"Failed to process gossip: {e}")
    
    async def _propagate_gossip(
        self,
        gossip: HealthGossip,
        exclude_router: Optional[str] = None,
    ) -> None:
        """
        Propagate gossip to other connected routers (excluding source).
        
        This enables multi-hop gossip propagation for better network coverage.
        """
        gossip_data = gossip.to_dict()
        
        for router_id, conn in list(self.connections.items()):
            # Skip the router we received from
            if router_id == exclude_router:
                continue
            
            if conn.websocket.closed:
                continue
            
            try:
                await conn.websocket.send_json({
                    "type": "gossip",
                    "payload": gossip_data,
                })
                logger.debug(f"Propagated gossip to router {router_id[:16]}...")
            except Exception as e:
                logger.debug(f"Failed to propagate gossip to {router_id[:16]}...: {e}")
    
    async def _handle_back_pressure(
        self,
        data: Dict[str, Any],
        conn: RouterConnection,
    ) -> None:
        """Handle back-pressure signal from router.
        
        When a router signals back-pressure, we should:
        1. Mark the router as under back-pressure
        2. Avoid sending to this router until the suggested retry delay
        3. Try alternative routers for pending messages
        
        Args:
            data: Back-pressure message with active, load_pct, retry_after_ms
            conn: The router connection that sent the signal
        """
        active = data.get("active", True)
        load_pct = data.get("load_pct", 0)
        retry_after_ms = data.get("retry_after_ms", 1000)
        reason = data.get("reason", "")
        
        router_id = conn.router.router_id
        
        if active:
            # Router is under load - mark it and set retry time
            conn.back_pressure_active = True
            conn.back_pressure_until = time.time() + (retry_after_ms / 1000)
            conn.back_pressure_retry_ms = retry_after_ms
            
            logger.warning(
                f"Router {router_id[:16]}... signaled BACK-PRESSURE: "
                f"load={load_pct:.1f}%, retry_after={retry_after_ms}ms, reason={reason}"
            )
            
            # Update stats
            self._stats["back_pressure_received"] = self._stats.get("back_pressure_received", 0) + 1
        else:
            # Back-pressure released
            conn.back_pressure_active = False
            conn.back_pressure_until = 0.0
            
            logger.info(
                f"Router {router_id[:16]}... RELEASED back-pressure: load={load_pct:.1f}%"
            )
            
            # Update stats
            self._stats["back_pressure_released"] = self._stats.get("back_pressure_released", 0) + 1
    
    # -------------------------------------------------------------------------
    # E2E ACK HANDLING
    # -------------------------------------------------------------------------
    
    async def _wait_for_ack(self, message_id: str) -> None:
        """Wait for an E2E ACK with timeout, then retry if needed."""
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        await asyncio.sleep(pending.timeout_ms / 1000)
        
        # Check if ACK was received while we waited
        if message_id not in self.pending_acks:
            return  # ACK received, all good
        
        # Timeout - attempt retry
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        if pending.retries < 1:
            # First retry - try same router
            pending.retries += 1
            logger.debug(f"ACK timeout for {message_id}, retrying via same router")
            await self._retry_message(message_id)
        else:
            # Second timeout - try different router
            logger.debug(f"ACK timeout for {message_id}, trying different router")
            await self._retry_via_different_router(message_id)
    
    async def _retry_message(self, message_id: str) -> None:
        """Retry sending a message via the same router."""
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        conn = self.connections.get(pending.router_id)
        if not conn or conn.websocket.closed:
            # Router unavailable, try different one
            await self._retry_via_different_router(message_id)
            return
        
        try:
            await self._send_via_router(
                conn.router,
                message_id,
                pending.recipient_id,
                pending.recipient_public_key,
                pending.content,
                require_ack=True,
            )
            pending.sent_at = time.time()
            # Schedule another wait
            asyncio.create_task(self._wait_for_ack(message_id))
        except Exception as e:
            logger.warning(f"Retry failed for {message_id}: {e}")
            await self._retry_via_different_router(message_id)
    
    async def _retry_via_different_router(self, message_id: str) -> None:
        """Retry sending a message via a different router."""
        pending = self.pending_acks.get(message_id)
        if not pending:
            return
        
        # Find a different router
        original_router_id = pending.router_id
        new_router = None
        
        for router_id, conn in self.connections.items():
            if router_id != original_router_id and not conn.websocket.closed:
                new_router = conn.router
                break
        
        if not new_router:
            # No alternative router, mark as failed
            logger.warning(
                f"No alternative router for {message_id}, giving up"
            )
            self._handle_ack_failure(message_id, pending)
            return
        
        try:
            pending.retries += 1
            pending.router_id = new_router.router_id
            
            await self._send_via_router(
                new_router,
                message_id,
                pending.recipient_id,
                pending.recipient_public_key,
                pending.content,
                require_ack=True,
            )
            pending.sent_at = time.time()
            # Schedule another wait
            asyncio.create_task(self._wait_for_ack(message_id))
        except Exception as e:
            logger.warning(f"Retry via different router failed for {message_id}: {e}")
            self._handle_ack_failure(message_id, pending)
    
    def _handle_ack_failure(self, message_id: str, pending: PendingAck) -> None:
        """Handle final ACK failure after all retries exhausted."""
        self.pending_acks.pop(message_id, None)
        
        # Update router stats for the last router used
        router_conn = self.connections.get(pending.router_id)
        if router_conn:
            router_conn.ack_failure += 1
        
        self._stats["ack_failures"] = self._stats.get("ack_failures", 0) + 1
        
        # Record delivery failure for misbehavior detection (Issue #119)
        self._record_delivery_outcome(
            pending.router_id, 
            message_id, 
            delivered=False,
        )
        
        # Notify callback if registered
        if self.on_ack_timeout:
            try:
                self.on_ack_timeout(message_id, pending.recipient_id)
            except Exception as e:
                logger.warning(f"on_ack_timeout callback error: {e}")
        
        logger.warning(
            f"Message {message_id} to {pending.recipient_id[:16]}... "
            f"failed after {pending.retries} retries"
        )
    
    async def _handle_e2e_ack(self, ack: AckMessage) -> None:
        """Handle an E2E acknowledgment from the recipient."""
        message_id = ack.original_message_id
        
        if message_id not in self.pending_acks:
            logger.debug(f"Received ACK for unknown message {message_id}")
            return
        
        pending = self.pending_acks.pop(message_id)
        
        # Verify signature (recipient signed the message_id)
        # For now, we trust the ACK came through encrypted channel
        # TODO: Verify signature if recipient's signing key is known
        
        # Update router stats
        router_conn = self.connections.get(pending.router_id)
        if router_conn:
            router_conn.ack_success += 1
        
        self._stats["ack_successes"] = self._stats.get("ack_successes", 0) + 1
        
        latency_ms = (ack.received_at - pending.sent_at) * 1000
        
        # Record successful delivery for misbehavior detection (Issue #119)
        self._record_delivery_outcome(
            pending.router_id,
            message_id,
            delivered=True,
            latency_ms=latency_ms,
        )
        
        logger.debug(
            f"E2E ACK received for {message_id} from {ack.recipient_id[:16]}... "
            f"(latency: {latency_ms:.1f}ms)"
        )
    
    async def _send_ack_to_sender(
        self,
        message_id: str,
        sender_id: str,
        sender_public_key: X25519PublicKey,
        reply_router_id: Optional[str] = None,
    ) -> None:
        """Send an E2E ACK back to the message sender."""
        # Create ACK message
        ack = AckMessage(
            original_message_id=message_id,
            received_at=time.time(),
            recipient_id=self.node_id,
            signature=self._sign_ack(message_id),
        )
        
        # Select router - prefer the one that delivered the message
        router = None
        if reply_router_id and reply_router_id in self.connections:
            conn = self.connections[reply_router_id]
            if not conn.websocket.closed:
                router = conn.router
        
        if not router:
            router = self._select_router()
        
        if not router:
            logger.warning(f"No router available to send ACK for {message_id}")
            return
        
        # Send ACK as a message
        ack_content = __import__('json').dumps(ack.to_dict()).encode()
        await self._send_via_router(
            router,
            str(uuid.uuid4()),  # ACK has its own message_id
            sender_id,
            sender_public_key,
            ack_content,
            require_ack=False,  # ACKs don't require ACKs (no infinite loop)
        )
        
        logger.debug(f"Sent E2E ACK for {message_id} to {sender_id[:16]}...")
    
    def _sign_ack(self, message_id: str) -> str:
        """Sign a message_id to prove we received it."""
        signature = self.private_key.sign(message_id.encode())
        return signature.hex()
    
    def _is_duplicate_message(self, message_id: str) -> bool:
        """Check if we've already seen this message (idempotent delivery)."""
        if message_id in self.seen_messages:
            return True
        
        # Add to seen messages
        self.seen_messages.add(message_id)
        
        # Prune if too large (FIFO-ish, just clear half)
        if len(self.seen_messages) > self.max_seen_messages:
            # Convert to list, remove oldest half
            seen_list = list(self.seen_messages)
            self.seen_messages = set(seen_list[len(seen_list)//2:])
        
        return False
    
    # -------------------------------------------------------------------------
    # ROUTER SELECTION
    # -------------------------------------------------------------------------
    
    def _select_router(self, exclude_back_pressured: bool = True) -> Optional[RouterInfo]:
        """
        Select the best router based on aggregated health metrics.
        
        Uses weighted random selection combining:
        - Own observations (direct connection metrics)
        - Peer observations (gossip from other nodes)
        - Back-pressure status (avoid overloaded routers)
        
        Own observations are weighted higher (default 0.7 vs 0.3 for peers)
        to ensure we trust our own experience more than hearsay.
        
        Args:
            exclude_back_pressured: If True, exclude routers under back-pressure.
                                   Set to False to include all routers as fallback.
        
        Returns:
            Selected RouterInfo or None if no routers available
        """
        if not self.connections:
            return None
        
        # Filter to healthy connections
        candidates = [
            conn for conn in self.connections.values()
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
                # All routers are under back-pressure - log warning and use anyway
                logger.warning(
                    f"All {len(candidates)} routers under back-pressure, "
                    "selecting least loaded"
                )
                # Sort by back_pressure_until (prefer ones that will release soonest)
                candidates.sort(key=lambda c: c.back_pressure_until)
        
        # Calculate weights using aggregated health (own + peer observations)
        weights = []
        for conn in candidates:
            # Get aggregated score combining own and peer observations
            aggregated = self._get_aggregated_health(conn.router.router_id)
            
            # Combine with direct connection health for robustness
            direct_score = conn.health_score
            
            # Blend: 60% aggregated (includes peer gossip), 40% direct
            combined_score = (aggregated * 0.6) + (direct_score * 0.4)
            
            # Apply penalty for back-pressured routers (if included)
            if conn.is_under_back_pressure:
                combined_score *= 0.1  # 90% penalty
            
            # Apply penalty for flagged routers (Issue #119 - Malicious Router Detection)
            if self.auto_avoid_flagged_routers and self.is_router_flagged(conn.router.router_id):
                combined_score *= self.flagged_router_penalty  # Heavy penalty for misbehaving routers
            
            weights.append(max(0.01, combined_score))
        
        # Weighted random selection
        selected = random.choices(candidates, weights=weights, k=1)[0]
        return selected.router
    
    # -------------------------------------------------------------------------
    # BACKGROUND TASKS
    # -------------------------------------------------------------------------
    
    async def _keepalive_loop(self) -> None:
        """
        Send periodic pings to detect connection failures.
        
        Uses fast detection with configurable keepalive interval (default 2s)
        and consecutive missed ping tracking for reliability.
        """
        while self._running:
            try:
                await asyncio.sleep(self.keepalive_interval)
                if not self._running:
                    break
                
                for router_id, conn in list(self.connections.items()):
                    if conn.websocket.closed:
                        logger.debug(f"WebSocket closed for router {router_id[:16]}...")
                        await self._handle_router_failure(router_id)
                        continue
                    
                    try:
                        sent_at = time.time()
                        await asyncio.wait_for(
                            conn.websocket.send_json({
                                "type": "ping",
                                "sent_at": sent_at,
                            }),
                            timeout=self.ping_timeout,
                        )
                        # Reset missed pings on successful ping
                        self._missed_pings[router_id] = 0
                        
                    except asyncio.TimeoutError:
                        # Track consecutive missed pings
                        missed = self._missed_pings.get(router_id, 0) + 1
                        self._missed_pings[router_id] = missed
                        
                        logger.warning(
                            f"Ping timeout for router {router_id[:16]}... "
                            f"({missed}/{self.missed_pings_threshold} missed)"
                        )
                        
                        # Trigger failure after threshold consecutive misses
                        if missed >= self.missed_pings_threshold:
                            logger.warning(
                                f"Router {router_id[:16]}... unresponsive "
                                f"({missed} consecutive missed pings)"
                            )
                            await self._handle_router_failure(router_id)
                            
                    except Exception as e:
                        # Other errors (connection reset, etc.) - immediate failure
                        logger.warning(
                            f"Ping error for router {router_id[:16]}...: {e}"
                        )
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
                    # Check for stale connection (no activity in 2x keepalive)
                    if now - conn.last_seen > self.keepalive_interval * 2:
                        logger.warning(
                            f"Stale connection to router {router_id[:16]}..."
                        )
                        await self._handle_router_failure(router_id)
                
                # Ensure we have enough connections
                if len(self.connections) < self.min_connections:
                    logger.info(
                        f"Connection count ({len(self.connections)}) below minimum "
                        f"({self.min_connections}), reconnecting..."
                    )
                    await self._ensure_connections()
                
                # Optionally add more connections if below target
                elif len(self.connections) < self.target_connections:
                    await self._ensure_connections()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Maintenance loop error: {e}")
    
    async def _queue_processor(self) -> None:
        """Process queued messages when routers become available."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                if not self._running:
                    break
                
                if not self.message_queue:
                    continue
                
                if not self.connections:
                    continue
                
                # Process oldest messages first
                now = time.time()
                processed = []
                
                for i, msg in enumerate(self.message_queue):
                    # Check message age
                    if now - msg.queued_at > self.MAX_QUEUE_AGE:
                        processed.append(i)
                        self._stats["messages_dropped"] += 1
                        logger.debug(f"Dropped expired message {msg.message_id}")
                        continue
                    
                    # Try to send
                    router = self._select_router()
                    if not router:
                        break
                    
                    try:
                        await self._send_via_router(
                            router,
                            msg.message_id,
                            msg.recipient_id,
                            msg.recipient_public_key,
                            msg.content,
                        )
                        processed.append(i)
                        logger.debug(f"Sent queued message {msg.message_id}")
                    except Exception as e:
                        msg.retries += 1
                        if msg.retries >= msg.max_retries:
                            processed.append(i)
                            self._stats["messages_dropped"] += 1
                            logger.warning(
                                f"Dropped message {msg.message_id} after "
                                f"{msg.retries} retries: {e}"
                            )
                
                # Remove processed messages (in reverse order to maintain indices)
                for i in reversed(processed):
                    self.message_queue.pop(i)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Queue processor error: {e}")
    
    async def _handle_router_failure(
        self,
        router_id: str,
        failure_type: str = "connection",
        error_code: Optional[str] = None,
    ) -> None:
        """
        Handle router failure with intelligent failover.
        
        This implements robust failover logic:
        1. Fast detection (already detected via keepalive)
        2. Update failover state with exponential backoff
        3. Preserve pending messages for retry
        4. Query seed for alternative routers
        5. Connect to alternatives with timeout
        6. Flush queued messages on new connection
        7. Enable direct mode as graceful degradation
        8. Record failure for eclipse anomaly detection (Issue #118)
        """
        conn = self.connections.pop(router_id, None)
        if not conn:
            return
        
        fail_time = time.time()
        self._stats["failovers"] += 1
        logger.warning(f"Router {router_id[:16]}... failed, initiating failover")
        
        # Record failure for eclipse anomaly detection (Issue #118)
        self._record_failure_event(router_id, failure_type, error_code)
        
        # 1. Update failover state with exponential backoff
        if router_id not in self.failover_states:
            self.failover_states[router_id] = FailoverState(
                router_id=router_id,
                failed_at=fail_time,
                fail_count=1,
                cooldown_until=fail_time + self.initial_cooldown,
                queued_messages=[],
            )
            logger.debug(f"Router {router_id[:16]}... entered cooldown for {self.initial_cooldown}s")
        else:
            state = self.failover_states[router_id]
            state.fail_count += 1
            state.failed_at = fail_time
            # Exponential backoff for flapping routers
            cooldown = min(
                self.initial_cooldown * (2 ** (state.fail_count - 1)),
                self.max_cooldown
            )
            state.cooldown_until = fail_time + cooldown
            logger.debug(
                f"Router {router_id[:16]}... fail_count={state.fail_count}, "
                f"cooldown={cooldown}s (flapping protection)"
            )
        
        # 2. Get messages that were pending on this router
        pending_for_router = [
            pending for msg_id, pending in self.pending_acks.items()
            if pending.router_id == router_id
        ]
        
        if pending_for_router:
            logger.info(
                f"Preserving {len(pending_for_router)} pending messages "
                f"from failed router {router_id[:16]}..."
            )
        
        # Close the failed connection
        await self._close_connection(router_id, conn)
        
        # Clear missed pings tracking for this router
        self._missed_pings.pop(router_id, None)
        
        # 3. Query seed for alternative routers (exclude failed and in-cooldown routers)
        exclude_ids = [router_id] + [
            r_id for r_id, state in self.failover_states.items()
            if state.is_in_cooldown() and r_id != router_id
        ]
        
        # Also exclude currently connected routers
        exclude_ids.extend(self.connections.keys())
        
        try:
            alternatives = await self.discovery.discover_routers(
                count=3,
                preferences={"exclude": exclude_ids},
                force_refresh=True,  # Get fresh data during failover
            )
        except Exception as e:
            logger.warning(f"Failed to discover alternative routers: {e}")
            alternatives = []
        
        # 4. Sort alternatives by health metrics
        if alternatives:
            alternatives.sort(
                key=lambda r: (
                    -r.health.get("uptime_pct", 0),
                    r.health.get("avg_latency_ms", 999),
                ),
            )
        
        # 5. Try to connect to alternative routers with timeout
        connected = False
        for router in alternatives:
            # Check IP diversity if enforced
            if self.enforce_ip_diversity and not self._check_ip_diversity(router):
                logger.debug(
                    f"Skipping router {router.router_id[:16]}... "
                    f"(IP diversity check failed)"
                )
                continue
            
            try:
                await asyncio.wait_for(
                    self._connect_to_router(router),
                    timeout=self.failover_connect_timeout,
                )
                connected = True
                logger.info(
                    f"Failover successful: connected to {router.router_id[:16]}... "
                    f"at {router.endpoints[0] if router.endpoints else 'unknown'}"
                )
                break
            except asyncio.TimeoutError:
                logger.warning(
                    f"Failover connection to {router.router_id[:16]}... "
                    f"timed out after {self.failover_connect_timeout}s"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"Failover connection to {router.router_id[:16]}... "
                    f"failed: {e}"
                )
                continue
        
        # 6. If connected, retry pending messages
        if connected and pending_for_router:
            logger.info(f"Retrying {len(pending_for_router)} messages on new router")
            for pending in pending_for_router:
                try:
                    await self._retry_message(pending.message_id)
                except Exception as e:
                    logger.warning(
                        f"Failed to retry message {pending.message_id}: {e}"
                    )
        
        # 7. If still below minimum connections, try to get more
        if not connected or len(self.connections) < self.min_connections:
            if not connected:
                # Enable direct mode as graceful degradation
                self._enable_direct_mode()
            
            # Schedule background reconnection attempt
            if self._running:
                await asyncio.sleep(self.reconnect_delay)
                if self._running:
                    await self._ensure_connections()
    
    def _enable_direct_mode(self) -> None:
        """
        Enable direct mode as graceful degradation.
        
        When no routers are available, the node can attempt direct P2P
        connections to known peers (if their endpoints are cached).
        """
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
    
    def clear_router_cooldown(self, router_id: str) -> bool:
        """
        Manually clear cooldown for a router.
        
        Useful for testing or when you know a router has been fixed.
        
        Returns:
            True if cooldown was cleared, False if router wasn't in cooldown
        """
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


    # -------------------------------------------------------------------------
    # HEALTH GOSSIP
    # -------------------------------------------------------------------------
    
    def _update_own_observation(self, router_id: str, conn: RouterConnection) -> None:
        """
        Update our own health observation for a router based on connection metrics.
        
        Called after receiving pong, after ACK success/failure, etc.
        """
        now = time.time()
        
        obs = RouterHealthObservation(
            router_id=router_id,
            latency_ms=conn.ping_latency_ms,
            success_rate=conn.ack_success_rate,
            failure_count=conn.ack_failure,
            success_count=conn.ack_success,
            last_seen=now,
            load_pct=conn.router.capacity.get("current_load_pct", 0),
        )
        
        self._own_observations[router_id] = obs
    
    def _get_aggregated_health(self, router_id: str) -> float:
        """
        Get aggregated health score for a router combining own and peer observations.
        
        Own observations are weighted higher (default 0.7 vs 0.3 for peers).
        
        Returns:
            Health score from 0.0 to 1.0
        """
        own_obs = self._own_observations.get(router_id)
        
        # Collect peer observations for this router
        peer_obs_list = []
        now = time.time()
        
        for peer_id, peer_data in self._peer_observations.items():
            obs = peer_data.get(router_id)
            if obs:
                # Check observation age
                obs_age = now - obs.last_seen if hasattr(obs, 'last_seen') else float('inf')
                if obs_age <= self.observation_max_age:
                    peer_obs_list.append(obs)
        
        # Calculate own health score
        if own_obs:
            own_score = self._calculate_observation_score(own_obs)
        else:
            own_score = None
        
        # Calculate average peer health score
        if peer_obs_list:
            peer_scores = [self._calculate_observation_score(obs) for obs in peer_obs_list]
            peer_score = sum(peer_scores) / len(peer_scores)
        else:
            peer_score = None
        
        # Combine with weights
        if own_score is not None and peer_score is not None:
            return (own_score * self.own_observation_weight + 
                    peer_score * self.peer_observation_weight)
        elif own_score is not None:
            return own_score
        elif peer_score is not None:
            # If we only have peer data, use it but trust it less
            return peer_score * 0.8  # 20% penalty for no direct observation
        else:
            return 0.5  # No data, neutral score
    
    def _calculate_observation_score(self, obs: RouterHealthObservation) -> float:
        """
        Calculate health score from a single observation.
        
        Combines:
        - Success rate (50% weight)
        - Latency score (30% weight)  
        - Load score (20% weight)
        """
        # Success rate component
        success_score = obs.success_rate * 0.5
        
        # Latency component (lower is better, cap at 500ms)
        latency_score = max(0, 1.0 - (obs.latency_ms / 500)) * 0.3
        
        # Load component (lower is better)
        load_score = (1.0 - (obs.load_pct / 100)) * 0.2
        
        return success_score + latency_score + load_score
    
    def _sample_observations_for_gossip(self) -> List[RouterHealthObservation]:
        """
        Sample observations to include in gossip message.
        
        Prioritizes recent observations and limits to max_observations_per_gossip
        to keep gossip lightweight.
        """
        now = time.time()
        
        # Filter to recent observations
        recent_obs = []
        for router_id, obs in self._own_observations.items():
            obs_age = now - obs.last_seen if obs.last_seen > 0 else float('inf')
            if obs_age <= self.observation_max_age:
                recent_obs.append((obs_age, obs))
        
        # Sort by recency (most recent first)
        recent_obs.sort(key=lambda x: x[0])
        
        # Take up to max_observations_per_gossip
        return [obs for _, obs in recent_obs[:self.max_observations_per_gossip]]
    
    def _create_gossip_message(self) -> HealthGossip:
        """Create a health gossip message with our observations."""
        observations = self._sample_observations_for_gossip()
        
        return HealthGossip(
            source_node_id=self.node_id,
            timestamp=time.time(),
            observations=observations,
            ttl=self.gossip_ttl,
        )
    
    async def _gossip_loop(self) -> None:
        """
        Periodically broadcast health gossip to connected peers.
        
        Gossip is sent via all connected routers to reach peer nodes.
        """
        while self._running:
            try:
                await asyncio.sleep(self.gossip_interval)
                if not self._running:
                    break
                
                # Update observations from current connections
                for router_id, conn in self.connections.items():
                    self._update_own_observation(router_id, conn)
                
                # Create and broadcast gossip
                if self._own_observations:
                    gossip = self._create_gossip_message()
                    await self._broadcast_gossip(gossip)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Gossip loop error: {e}")
    
    async def _broadcast_gossip(self, gossip: HealthGossip) -> None:
        """
        Broadcast gossip message to all connected routers.
        
        The router will forward to other connected nodes.
        """
        gossip_data = gossip.to_dict()
        
        for router_id, conn in list(self.connections.items()):
            if conn.websocket.closed:
                continue
            
            try:
                await conn.websocket.send_json({
                    "type": "gossip",
                    "payload": gossip_data,
                })
                self._stats["gossip_sent"] += 1
                logger.debug(
                    f"Sent gossip with {len(gossip.observations)} observations "
                    f"via router {router_id[:16]}..."
                )
            except Exception as e:
                logger.debug(f"Failed to send gossip via {router_id[:16]}...: {e}")
    
    def _handle_gossip(self, gossip: HealthGossip) -> None:
        """
        Handle incoming health gossip from a peer.
        
        Updates peer observation cache, pruning old entries if needed.
        """
        source = gossip.source_node_id
        
        # Don't process our own gossip
        if source == self.node_id:
            return
        
        # Check TTL for propagation
        if gossip.ttl <= 0:
            return
        
        self._stats["gossip_received"] += 1
        
        # Initialize peer observation dict if needed
        if source not in self._peer_observations:
            self._peer_observations[source] = {}
        
        # Update observations from this peer
        for obs in gossip.observations:
            self._peer_observations[source][obs.router_id] = obs
        
        # Prune if too many peer observations
        self._prune_peer_observations()
        
        logger.debug(
            f"Received gossip from {source[:16]}... with "
            f"{len(gossip.observations)} observations"
        )
    
    def _prune_peer_observations(self) -> None:
        """
        Prune old or excess peer observations to limit memory usage.
        """
        now = time.time()
        total_obs = 0
        
        # First pass: remove old observations
        for peer_id in list(self._peer_observations.keys()):
            peer_data = self._peer_observations[peer_id]
            for router_id in list(peer_data.keys()):
                obs = peer_data[router_id]
                obs_age = now - obs.last_seen if obs.last_seen > 0 else float('inf')
                if obs_age > self.observation_max_age:
                    del peer_data[router_id]
            
            # Remove peer if no observations left
            if not peer_data:
                del self._peer_observations[peer_id]
            else:
                total_obs += len(peer_data)
        
        # Second pass: if still over limit, remove oldest
        if total_obs > self.max_peer_observations:
            # Collect all observations with age
            all_obs = []
            for peer_id, peer_data in self._peer_observations.items():
                for router_id, obs in peer_data.items():
                    all_obs.append((peer_id, router_id, obs.last_seen))
            
            # Sort by age (oldest first)
            all_obs.sort(key=lambda x: x[2])
            
            # Remove oldest until under limit
            to_remove = len(all_obs) - self.max_peer_observations
            for i in range(to_remove):
                peer_id, router_id, _ = all_obs[i]
                if peer_id in self._peer_observations:
                    self._peer_observations[peer_id].pop(router_id, None)
    
    def get_health_observations(self) -> Dict[str, Any]:
        """
        Get current health observation state for debugging/monitoring.
        
        Returns:
            Dict with own observations, peer observation count, and aggregated scores
        """
        aggregated_scores = {}
        for router_id in self._own_observations.keys():
            aggregated_scores[router_id] = round(self._get_aggregated_health(router_id), 3)
        
        return {
            "own_observations": {
                router_id: obs.to_dict()
                for router_id, obs in self._own_observations.items()
            },
            "peer_observation_count": sum(
                len(peer_data) for peer_data in self._peer_observations.values()
            ),
            "peers_with_observations": len(self._peer_observations),
            "aggregated_health_scores": aggregated_scores,
        }
    
    # -------------------------------------------------------------------------
    # CIRCUIT BUILDING (Issue #115 - Enhanced Privacy)
    # -------------------------------------------------------------------------
    
    async def build_circuit(
        self,
        hop_count: Optional[int] = None,
        exclude_routers: Optional[List[str]] = None,
    ) -> Optional[Circuit]:
        """
        Build a new circuit through multiple routers for enhanced privacy.
        
        Args:
            hop_count: Number of hops (default: circuit_min_hops to circuit_max_hops)
            exclude_routers: Router IDs to exclude from circuit
            
        Returns:
            Circuit object if successful, None if circuit building failed
        """
        async with self._circuit_lock:
            if hop_count is None:
                hop_count = random.randint(self.circuit_min_hops, self.circuit_max_hops)
            hop_count = max(self.circuit_min_hops, min(hop_count, self.circuit_max_hops))
            
            exclude = set(exclude_routers or [])
            
            # Discover routers for circuit
            try:
                routers = await self.discovery.discover_routers(
                    count=hop_count * 2,
                    preferences={"exclude": list(exclude)},
                )
            except Exception as e:
                logger.warning(f"Failed to discover routers for circuit: {e}")
                return None
            
            if len(routers) < hop_count:
                logger.warning(f"Not enough routers for circuit")
                return None
            
            # Select diverse routers
            selected = self._select_diverse_routers(routers, hop_count)
            if len(selected) < hop_count:
                return None
            
            # Create circuit
            circuit = Circuit(
                hops=[CircuitHop(router_id=r.router_id) for r in selected],
                expires_at=time.time() + self.circuit_lifetime,
                max_messages=self.circuit_max_messages,
            )
            
            # For now, simulate circuit creation with placeholder keys
            # In a full implementation, this would do the DH key exchange
            hop_keys = [os.urandom(32) for _ in selected]
            
            self._circuits[circuit.circuit_id] = circuit
            self._circuit_keys[circuit.circuit_id] = hop_keys
            self._stats["circuits_built"] = self._stats.get("circuits_built", 0) + 1
            
            logger.info(f"Circuit {circuit.circuit_id[:8]}... built with {hop_count} hops")
            return circuit
    
    def _select_diverse_routers(
        self,
        routers: List[RouterInfo],
        count: int,
    ) -> List[RouterInfo]:
        """Select routers ensuring network diversity."""
        selected = []
        used_subnets = set()
        
        for router in routers:
            if len(selected) >= count:
                break
            
            subnet = self._get_router_subnet(router)
            if subnet and subnet in used_subnets:
                continue
            
            selected.append(router)
            if subnet:
                used_subnets.add(subnet)
        
        # Fill remaining if needed
        if len(selected) < count:
            for router in routers:
                if len(selected) >= count:
                    break
                if router not in selected:
                    selected.append(router)
        
        return selected
    
    def _get_router_subnet(self, router: RouterInfo) -> Optional[str]:
        """Extract /16 subnet from router endpoint."""
        if not router.endpoints:
            return None
        
        try:
            endpoint = router.endpoints[0]
            if endpoint.startswith("["):
                bracket_end = endpoint.find("]")
                if bracket_end == -1:
                    return None
                host = endpoint[1:bracket_end]
            else:
                host = endpoint.split(":")[0]
            
            ip = ipaddress.ip_address(host)
            if ip.version == 4:
                network = ipaddress.ip_network(f"{ip}/16", strict=False)
            else:
                network = ipaddress.ip_network(f"{ip}/48", strict=False)
            return str(network)
        except Exception:
            return None
    
    async def _destroy_circuit(
        self,
        circuit_id: str,
        reason: str = "",
    ) -> None:
        """Tear down a circuit and clean up resources."""
        circuit = self._circuits.pop(circuit_id, None)
        self._circuit_keys.pop(circuit_id, None)
        self._circuit_ephemeral_keys.pop(circuit_id, None)
        
        if self._active_circuit_id == circuit_id:
            self._active_circuit_id = None
        
        self._stats["circuits_destroyed"] = self._stats.get("circuits_destroyed", 0) + 1
        if circuit:
            logger.info(f"Circuit {circuit_id[:8]}... destroyed: {reason}")
    
    def get_circuit_stats(self) -> Dict[str, Any]:
        """Get circuit-related statistics."""
        circuits_info = []
        for circuit_id, circuit in self._circuits.items():
            circuits_info.append({
                "circuit_id": circuit_id[:8] + "...",
                "hop_count": circuit.hop_count,
                "message_count": circuit.message_count,
                "is_expired": circuit.is_expired,
                "needs_rotation": circuit.needs_rotation,
            })
        
        return {
            "circuits_active": len(self._circuits),
            "circuits_built": self._stats.get("circuits_built", 0),
            "circuits_destroyed": self._stats.get("circuits_destroyed", 0),
            "messages_via_circuit": self._stats.get("messages_via_circuit", 0),
            "active_circuit_id": self._active_circuit_id[:8] + "..." if self._active_circuit_id else None,
            "circuits": circuits_info,
        }
    
    # -------------------------------------------------------------------------
    # COVER TRAFFIC (Issue #116)
    # -------------------------------------------------------------------------
    
    def _is_idle(self) -> bool:
        """
        Check if the node is considered idle for cover traffic purposes.
        
        A node is idle if no real messages have been sent within the
        idle threshold configured in cover_traffic.idle_threshold_seconds.
        
        Returns:
            True if node is idle and should generate cover traffic
        """
        if not self.cover_traffic.enabled:
            return False
        
        if self._last_real_message_time == 0:
            # No messages sent yet - consider idle
            return True
        
        elapsed = time.time() - self._last_real_message_time
        return elapsed >= self.cover_traffic.idle_threshold_seconds
    
    def _get_cover_target(self) -> Optional[str]:
        """
        Select a target for cover traffic.
        
        If target_peers is configured, randomly selects from that list.
        Otherwise, returns None (cover traffic will go to a random router
        which will discard it as unroutable).
        
        Returns:
            Target node ID or None
        """
        if self.cover_traffic.target_peers:
            return random.choice(self.cover_traffic.target_peers)
        
        # No target peers configured - generate fake recipient ID
        # This message will be sent to a router but can't be delivered
        # (no such node exists), which is fine for cover traffic
        return os.urandom(32).hex()
    
    async def _generate_cover_message(self) -> None:
        """
        Generate and send a single cover traffic message.
        
        Cover messages are:
        - Encrypted (routers can't read them)
        - Padded to standard bucket sizes
        - Routed to target peers (or discarded by routers if no target)
        - Indistinguishable from real traffic
        """
        if not self.connections:
            logger.debug("No routers connected, skipping cover traffic")
            return
        
        # Select router and target
        router = self._select_router()
        if not router:
            return
        
        target_id = self._get_cover_target()
        
        # Create cover message content
        cover_msg = CoverMessage()
        content = cover_msg.to_bytes()
        
        # Generate a fake recipient key for encryption
        # (the message won't be decrypted anyway)
        fake_key = X25519PrivateKey.generate().public_key()
        
        try:
            await self._send_via_router(
                router=router,
                message_id=cover_msg.message_id,
                recipient_id=target_id,
                recipient_public_key=fake_key,
                content=content,
                require_ack=False,  # Cover traffic doesn't need ACKs
                is_cover_traffic=True,
            )
        except Exception as e:
            logger.debug(f"Failed to send cover traffic: {e}")
    
    async def _cover_traffic_loop(self) -> None:
        """
        Background task to generate cover traffic when idle.
        
        Sends dummy messages at configurable rate when the node hasn't
        sent real messages within the idle threshold. This obscures
        real communication patterns from traffic analysis.
        """
        logger.debug("Cover traffic loop started")
        
        while self._running:
            try:
                # Get next interval based on config
                interval = self.cover_traffic.get_next_interval()
                await asyncio.sleep(interval)
                
                if not self._running:
                    break
                
                # Only generate cover traffic when idle
                if self._is_idle():
                    await self._generate_cover_message()
                else:
                    logger.debug(
                        f"Node active (last message {time.time() - self._last_real_message_time:.1f}s ago), "
                        "skipping cover traffic"
                    )
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Cover traffic loop error: {e}")
        
        logger.debug("Cover traffic loop stopped")
    
    def get_cover_traffic_stats(self) -> Dict[str, Any]:
        """
        Get cover traffic statistics.
        
        Returns:
            Dict with cover traffic metrics
        """
        return {
            "enabled": self.cover_traffic.enabled,
            "rate_per_minute": self.cover_traffic.rate_per_minute,
            "pad_messages": self.cover_traffic.pad_messages,
            "is_idle": self._is_idle(),
            "cover_messages_sent": self._stats.get("cover_messages_sent", 0),
            "cover_messages_received": self._stats.get("cover_messages_received", 0),
            "bytes_padded": self._stats.get("bytes_padded", 0),
            "last_real_message_ago": (
                time.time() - self._last_real_message_time
                if self._last_real_message_time > 0 else None
            ),
        }
    
    def enable_cover_traffic(
        self,
        rate_per_minute: float = 2.0,
        pad_messages: bool = True,
        target_peers: Optional[List[str]] = None,
    ) -> None:
        """
        Enable cover traffic at runtime.
        
        Args:
            rate_per_minute: Messages per minute when idle
            pad_messages: Whether to pad real messages to bucket sizes
            target_peers: Optional list of peer node IDs for cover traffic
        """
        self.cover_traffic.enabled = True
        self.cover_traffic.rate_per_minute = rate_per_minute
        self.cover_traffic.pad_messages = pad_messages
        if target_peers:
            self.cover_traffic.target_peers = target_peers
        
        # Start the cover traffic task if not running
        if self._running and (
            self._cover_traffic_task is None or 
            self._cover_traffic_task.done()
        ):
            self._cover_traffic_task = asyncio.create_task(self._cover_traffic_loop())
            self._tasks.append(self._cover_traffic_task)
        
        logger.info(
            f"Cover traffic enabled: {rate_per_minute:.1f} msg/min when idle, "
            f"padding={'on' if pad_messages else 'off'}"
        )
    
    def disable_cover_traffic(self) -> None:
        """Disable cover traffic at runtime."""
        self.cover_traffic.enabled = False
        
        # Cancel the task if running
        if self._cover_traffic_task and not self._cover_traffic_task.done():
            self._cover_traffic_task.cancel()
        
        logger.info("Cover traffic disabled")

    # =========================================================================
    # TRAFFIC ANALYSIS MITIGATIONS (Issue #120)
    # =========================================================================
    
    async def _add_to_batch(
        self,
        message_id: str,
        recipient_id: str,
        recipient_public_key: X25519PublicKey,
        content: bytes,
        require_ack: bool,
        timeout_ms: int,
    ) -> str:
        """
        Add a message to the current batch for delayed sending.
        
        Messages are collected in a batch and sent together at regular
        intervals to obscure individual message timing.
        
        Args:
            message_id: Unique message identifier
            recipient_id: Recipient's node ID
            recipient_public_key: Recipient's encryption key
            content: Message content (possibly padded)
            require_ack: Whether ACK is required
            timeout_ms: ACK timeout
            
        Returns:
            Message ID
        """
        async with self._batch_lock:
            batch_entry = {
                "message_id": message_id,
                "recipient_id": recipient_id,
                "recipient_public_key": recipient_public_key,
                "content": content,
                "require_ack": require_ack,
                "timeout_ms": timeout_ms,
                "queued_at": time.time(),
            }
            self._message_batch.append(batch_entry)
            self._stats["batched_messages"] += 1
            
            logger.debug(
                f"Message {message_id} added to batch "
                f"(batch size: {len(self._message_batch)})"
            )
            
            # Check if batch is full - trigger immediate flush
            if len(self._message_batch) >= self.traffic_analysis_mitigation.batching.max_batch_size:
                logger.debug(f"Batch full, triggering flush")
                self._pending_batch_event.set()
        
        return message_id
    
    async def _batch_flush_loop(self) -> None:
        """
        Background task to flush message batches at regular intervals.
        
        Sends accumulated messages when:
        - Batch interval has elapsed
        - Batch is full (max_batch_size reached)
        - At least min_batch_size messages are queued (or timeout)
        """
        logger.debug("Batch flush loop started")
        
        while self._running:
            try:
                batch_interval = self.traffic_analysis_mitigation.batching.get_effective_interval()
                
                # Wait for interval or until batch is full
                try:
                    await asyncio.wait_for(
                        self._pending_batch_event.wait(),
                        timeout=batch_interval,
                    )
                    # Event was set (batch full) - reset it
                    self._pending_batch_event.clear()
                except asyncio.TimeoutError:
                    # Interval elapsed - normal flush
                    pass
                
                if not self._running:
                    break
                
                await self._flush_batch()
                
            except asyncio.CancelledError:
                # Final flush before shutdown
                await self._flush_batch()
                break
            except Exception as e:
                logger.warning(f"Batch flush loop error: {e}")
        
        logger.debug("Batch flush loop stopped")
    
    async def _flush_batch(self) -> None:
        """
        Flush all messages in the current batch.
        
        If randomize_order is enabled, messages are shuffled before sending
        to further obscure timing patterns.
        """
        async with self._batch_lock:
            if not self._message_batch:
                return
            
            batch = self._message_batch.copy()
            self._message_batch.clear()
        
        # Randomize order if configured
        if self.traffic_analysis_mitigation.batching.randomize_order:
            random.shuffle(batch)
        
        self._stats["batch_flushes"] += 1
        self._last_batch_flush = time.time()
        
        logger.debug(f"Flushing batch of {len(batch)} messages")
        
        # Apply jitter between messages in batch
        jitter_config = self.traffic_analysis_mitigation.jitter
        
        for entry in batch:
            try:
                # Apply inter-message jitter if enabled
                if jitter_config.enabled:
                    jitter_delay = jitter_config.get_jitter_delay()
                    if jitter_delay > 0:
                        self._stats["jitter_delays_applied"] += 1
                        self._stats["total_jitter_ms"] += int(jitter_delay * 1000)
                        await asyncio.sleep(jitter_delay)
                
                # Send the message
                await self._send_message_direct(
                    message_id=entry["message_id"],
                    recipient_id=entry["recipient_id"],
                    recipient_public_key=entry["recipient_public_key"],
                    content=entry["content"],
                    require_ack=entry["require_ack"],
                    timeout_ms=entry["timeout_ms"],
                )
                
            except Exception as e:
                logger.warning(
                    f"Failed to send batched message {entry['message_id']}: {e}"
                )
                # Re-queue failed messages
                self.message_queue.append(PendingMessage(
                    message_id=entry["message_id"],
                    recipient_id=entry["recipient_id"],
                    content=entry["content"],
                    recipient_public_key=entry["recipient_public_key"],
                    queued_at=entry["queued_at"],
                ))
    
    async def _schedule_constant_rate_send(
        self,
        message_id: str,
        recipient_id: str,
        recipient_public_key: X25519PublicKey,
        content: bytes,
        require_ack: bool,
        timeout_ms: int,
    ) -> str:
        """
        Schedule a message for constant-rate sending.
        
        In constant-rate mode, messages are sent at fixed intervals.
        If there's no real message to send, a padding message is sent instead.
        Real messages are queued and sent in the next available slot.
        
        Args:
            message_id: Unique message identifier
            recipient_id: Recipient's node ID
            recipient_public_key: Recipient's encryption key
            content: Message content
            require_ack: Whether ACK is required
            timeout_ms: ACK timeout
            
        Returns:
            Message ID
        """
        cr_config = self.traffic_analysis_mitigation.constant_rate
        
        # If bursting is allowed and we have capacity, send immediately
        if cr_config.allow_burst:
            async with self._batch_lock:
                pending_count = len(self._message_batch)
                if pending_count < cr_config.max_burst_size:
                    # Can burst - apply jitter and send
                    if self.traffic_analysis_mitigation.jitter.enabled:
                        jitter_delay = self.traffic_analysis_mitigation.jitter.get_jitter_delay()
                        if jitter_delay > 0:
                            self._stats["jitter_delays_applied"] += 1
                            self._stats["total_jitter_ms"] += int(jitter_delay * 1000)
                            await asyncio.sleep(jitter_delay)
                    
                    return await self._send_message_direct(
                        message_id=message_id,
                        recipient_id=recipient_id,
                        recipient_public_key=recipient_public_key,
                        content=content,
                        require_ack=require_ack,
                        timeout_ms=timeout_ms,
                    )
        
        # Queue for next constant-rate slot
        async with self._batch_lock:
            batch_entry = {
                "message_id": message_id,
                "recipient_id": recipient_id,
                "recipient_public_key": recipient_public_key,
                "content": content,
                "require_ack": require_ack,
                "timeout_ms": timeout_ms,
                "queued_at": time.time(),
            }
            self._message_batch.append(batch_entry)
        
        logger.debug(f"Message {message_id} queued for constant-rate send")
        return message_id
    
    async def _constant_rate_loop(self) -> None:
        """
        Background task to send messages at a constant rate.
        
        Sends one message per interval. If no real messages are queued,
        sends a padding message to maintain constant traffic rate.
        """
        logger.debug("Constant-rate sending loop started")
        
        cr_config = self.traffic_analysis_mitigation.constant_rate
        
        while self._running:
            try:
                send_interval = cr_config.get_send_interval()
                
                # Wait for next send slot
                await asyncio.sleep(send_interval)
                
                if not self._running:
                    break
                
                self._last_constant_rate_send = time.time()
                
                # Try to get a real message from the batch
                message_to_send = None
                async with self._batch_lock:
                    if self._message_batch:
                        message_to_send = self._message_batch.pop(0)
                
                if message_to_send:
                    # Send real message
                    try:
                        await self._send_message_direct(
                            message_id=message_to_send["message_id"],
                            recipient_id=message_to_send["recipient_id"],
                            recipient_public_key=message_to_send["recipient_public_key"],
                            content=message_to_send["content"],
                            require_ack=message_to_send["require_ack"],
                            timeout_ms=message_to_send["timeout_ms"],
                        )
                    except Exception as e:
                        logger.warning(
                            f"Constant-rate send failed for {message_to_send['message_id']}: {e}"
                        )
                else:
                    # Send padding message to maintain constant rate
                    await self._send_padding_message()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Constant-rate loop error: {e}")
        
        logger.debug("Constant-rate sending loop stopped")
    
    async def _send_padding_message(self) -> None:
        """
        Send a padding message to maintain constant traffic rate.
        
        Padding messages are encrypted and indistinguishable from real
        traffic to network observers. They're sent to random targets
        or discarded by routers (if no valid target).
        """
        if not self.connections:
            return
        
        cr_config = self.traffic_analysis_mitigation.constant_rate
        
        # Create padding content
        padding_content = os.urandom(cr_config.pad_to_size)
        
        # Generate a fake recipient (message will be discarded)
        fake_recipient_id = os.urandom(32).hex()
        fake_recipient_key = X25519PrivateKey.generate().public_key()
        
        # Select a router
        router = self._select_router()
        if not router:
            return
        
        try:
            # Send padding message (no ACK needed, will be discarded)
            padding_msg_id = str(uuid.uuid4())
            await self._send_via_router(
                router=router,
                message_id=padding_msg_id,
                recipient_id=fake_recipient_id,
                recipient_public_key=fake_recipient_key,
                content=padding_content,
                require_ack=False,
                is_cover_traffic=True,
            )
            self._stats["constant_rate_padding_sent"] += 1
            logger.debug(f"Sent constant-rate padding message {padding_msg_id[:8]}...")
            
        except Exception as e:
            logger.debug(f"Failed to send padding message: {e}")
    
    def set_privacy_level(self, level: PrivacyLevel) -> None:
        """
        Set the traffic analysis mitigation privacy level.
        
        Updates all mitigation settings to match the preset level.
        Changes take effect immediately for new messages.
        
        Args:
            level: The desired privacy level
        """
        old_level = self.traffic_analysis_mitigation.privacy_level
        self.traffic_analysis_mitigation = TrafficAnalysisMitigationConfig.from_privacy_level(level)
        
        logger.info(f"Privacy level changed: {old_level.value} -> {level.value}")
        
        # Start/stop tasks based on new settings
        if self._running:
            asyncio.create_task(self._update_mitigation_tasks())
    
    async def _update_mitigation_tasks(self) -> None:
        """Update mitigation background tasks based on current config."""
        tam = self.traffic_analysis_mitigation
        
        # Handle batch flush task
        if tam.batching.enabled:
            if self._batch_flush_task is None or self._batch_flush_task.done():
                self._batch_flush_task = asyncio.create_task(self._batch_flush_loop())
                self._tasks.append(self._batch_flush_task)
                logger.debug("Started batch flush task")
        else:
            if self._batch_flush_task and not self._batch_flush_task.done():
                self._batch_flush_task.cancel()
                # Flush any pending messages
                await self._flush_batch()
                logger.debug("Stopped batch flush task")
        
        # Handle constant-rate task
        if tam.constant_rate.enabled:
            if self._constant_rate_task is None or self._constant_rate_task.done():
                self._constant_rate_task = asyncio.create_task(self._constant_rate_loop())
                self._tasks.append(self._constant_rate_task)
                logger.debug("Started constant-rate task")
        else:
            if self._constant_rate_task and not self._constant_rate_task.done():
                self._constant_rate_task.cancel()
                logger.debug("Stopped constant-rate task")
    
    def get_traffic_analysis_mitigation_stats(self) -> Dict[str, Any]:
        """
        Get traffic analysis mitigation statistics and status.
        
        Returns:
            Dict with mitigation config and statistics
        """
        tam = self.traffic_analysis_mitigation
        
        return {
            "privacy_level": tam.privacy_level.value,
            "batching": {
                "enabled": tam.batching.enabled,
                "batch_interval_ms": tam.batching.batch_interval_ms,
                "min_batch_size": tam.batching.min_batch_size,
                "max_batch_size": tam.batching.max_batch_size,
                "randomize_order": tam.batching.randomize_order,
                "current_batch_size": len(self._message_batch),
                "last_flush": self._last_batch_flush,
            },
            "jitter": {
                "enabled": tam.jitter.enabled,
                "min_delay_ms": tam.jitter.min_delay_ms,
                "max_delay_ms": tam.jitter.max_delay_ms,
                "distribution": tam.jitter.distribution,
            },
            "constant_rate": {
                "enabled": tam.constant_rate.enabled,
                "messages_per_minute": tam.constant_rate.messages_per_minute,
                "pad_to_size": tam.constant_rate.pad_to_size,
                "allow_burst": tam.constant_rate.allow_burst,
                "last_send": self._last_constant_rate_send,
            },
            "mix_network": {
                "enabled": tam.mix_network.enabled,
                "provider_url": tam.mix_network.provider_url,
            },
            "stats": {
                "batched_messages": self._stats.get("batched_messages", 0),
                "batch_flushes": self._stats.get("batch_flushes", 0),
                "jitter_delays_applied": self._stats.get("jitter_delays_applied", 0),
                "total_jitter_ms": self._stats.get("total_jitter_ms", 0),
                "avg_jitter_ms": (
                    self._stats.get("total_jitter_ms", 0) / 
                    max(1, self._stats.get("jitter_delays_applied", 1))
                ),
                "constant_rate_padding_sent": self._stats.get("constant_rate_padding_sent", 0),
                "messages_with_jitter": self._stats.get("messages_with_jitter", 0),
            },
            "latency_estimate": tam.estimate_latency_overhead(),
        }

    # =========================================================================
    # ECLIPSE ATTACK MITIGATIONS (Issue #118)
    # =========================================================================
    
    async def _router_rotation_loop(self) -> None:
        """
        Periodically rotate router connections to prevent long-term eclipse.
        
        This loop:
        1. Checks for connections exceeding max age and rotates them
        2. Periodically rotates one random connection for diversity
        3. Ensures diversity requirements are maintained after rotation
        """
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                if not self._running:
                    break
                
                now = time.time()
                rotated = False
                
                # Check for connections exceeding max age
                for router_id, conn in list(self.connections.items()):
                    conn_time = self._connection_timestamps.get(router_id, conn.connected_at)
                    age = now - conn_time
                    
                    if age >= self.rotation_max_age:
                        logger.info(
                            f"Rotating router {router_id[:16]}... due to max age "
                            f"({age/3600:.1f}h > {self.rotation_max_age/3600:.1f}h)"
                        )
                        await self._rotate_router(router_id, "max_age")
                        rotated = True
                        break  # One rotation per cycle
                
                # Periodic rotation (if not already rotated for max age)
                if not rotated and len(self.connections) > self.min_connections:
                    time_since_rotation = now - self._last_rotation
                    
                    if time_since_rotation >= self.rotation_interval:
                        # Select oldest connection for rotation
                        oldest_router_id = min(
                            self.connections.keys(),
                            key=lambda rid: self._connection_timestamps.get(
                                rid, self.connections[rid].connected_at
                            )
                        )
                        
                        logger.info(
                            f"Periodic rotation of router {oldest_router_id[:16]}... "
                            f"(connected for {(now - self._connection_timestamps.get(oldest_router_id, now))/3600:.1f}h)"
                        )
                        await self._rotate_router(oldest_router_id, "periodic")
                
                # Check diversity requirements
                if not self._check_diversity_requirements():
                    logger.warning(
                        "Diversity requirements not met after rotation - "
                        "potential eclipse risk"
                    )
                    # Try to add more diverse connections
                    await self._ensure_connections()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Router rotation loop error: {e}")
    
    async def _rotate_router(self, router_id: str, reason: str) -> bool:
        """
        Rotate a specific router connection.
        
        Disconnects from the router and attempts to connect to a new,
        diverse router to replace it.
        
        Args:
            router_id: Router to rotate out
            reason: Reason for rotation (for logging)
            
        Returns:
            True if rotation was successful
        """
        conn = self.connections.get(router_id)
        if not conn:
            return False
        
        # Record the rotation
        self._last_rotation = time.time()
        self._stats["routers_rotated"] += 1
        
        # Disconnect from old router
        await self._close_connection(router_id, conn)
        self.connections.pop(router_id, None)
        self._connection_timestamps.pop(router_id, None)
        self._remove_subnet(conn.router)
        
        # Try to connect to a new diverse router
        try:
            # Exclude current connections and recently rotated router
            exclude_ids = set(self.connections.keys())
            exclude_ids.add(router_id)
            
            # Also exclude routers in cooldown
            for r_id, state in self.failover_states.items():
                if state.is_in_cooldown():
                    exclude_ids.add(r_id)
            
            routers = await self.discovery.discover_routers(
                count=5,
                preferences={"exclude": list(exclude_ids)},
            )
            
            # Find a diverse router
            for router in routers:
                if self.enforce_ip_diversity and not self._check_ip_diversity(router):
                    continue
                if not self._check_asn_diversity(router):
                    continue
                
                try:
                    await self._connect_to_router(router)
                    self._connection_timestamps[router.router_id] = time.time()
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
            logger.warning(f"Rotation failed for {router_id[:16]}...: {e}")
            return False
    
    def _record_failure_event(
        self,
        router_id: str,
        failure_type: str,
        error_code: Optional[str] = None,
    ) -> None:
        """
        Record a router failure event for anomaly detection.
        
        Tracks failure events with timestamps to detect correlated failures
        that might indicate an eclipse attack.
        
        Args:
            router_id: The router that failed
            failure_type: Type of failure (e.g., "connection", "timeout", "error")
            error_code: Optional specific error code
        """
        if not self.anomaly_detection_enabled:
            return
        
        now = time.time()
        
        event = {
            "router_id": router_id,
            "failure_type": failure_type,
            "error_code": error_code,
            "timestamp": now,
        }
        
        self._failure_events.append(event)
        
        # Prune old events
        cutoff = now - self.anomaly_window
        self._failure_events = [
            e for e in self._failure_events
            if e["timestamp"] >= cutoff
        ]
        
        # Check for anomalies
        self._detect_anomalies()
    
    def _detect_anomalies(self) -> Optional[Dict[str, Any]]:
        """
        Detect anomalous patterns in router failures.
        
        Looks for:
        - Multiple routers failing at the same time
        - Same error occurring across multiple routers
        - All routers from same subnet/ASN failing
        
        Returns:
            Anomaly details if detected, None otherwise
        """
        now = time.time()
        cutoff = now - self.anomaly_window
        
        # Recent failures only
        recent_failures = [
            e for e in self._failure_events
            if e["timestamp"] >= cutoff
        ]
        
        if len(recent_failures) < self.anomaly_threshold:
            return None
        
        # Check for same failure type across multiple routers
        failure_type_counts: Dict[str, List[str]] = {}
        for event in recent_failures:
            ft = event["failure_type"]
            if ft not in failure_type_counts:
                failure_type_counts[ft] = []
            failure_type_counts[ft].append(event["router_id"])
        
        for failure_type, router_ids in failure_type_counts.items():
            unique_routers = set(router_ids)
            if len(unique_routers) >= self.anomaly_threshold:
                anomaly = {
                    "type": "correlated_failures",
                    "failure_type": failure_type,
                    "affected_routers": list(unique_routers),
                    "count": len(unique_routers),
                    "window_seconds": self.anomaly_window,
                    "detected_at": now,
                }
                
                self._anomaly_alerts.append(anomaly)
                self._stats["anomalies_detected"] += 1
                
                # Keep only recent anomalies
                one_hour_ago = now - 3600
                self._anomaly_alerts = [
                    a for a in self._anomaly_alerts
                    if a["detected_at"] >= one_hour_ago
                ]
                
                logger.warning(
                    f"ECLIPSE ANOMALY DETECTED: {len(unique_routers)} routers "
                    f"experienced '{failure_type}' within {self.anomaly_window}s"
                )
                
                return anomaly
        
        # Check for same error code
        error_code_counts: Dict[str, List[str]] = {}
        for event in recent_failures:
            ec = event.get("error_code")
            if ec:
                if ec not in error_code_counts:
                    error_code_counts[ec] = []
                error_code_counts[ec].append(event["router_id"])
        
        for error_code, router_ids in error_code_counts.items():
            unique_routers = set(router_ids)
            if len(unique_routers) >= self.anomaly_threshold:
                anomaly = {
                    "type": "same_error",
                    "error_code": error_code,
                    "affected_routers": list(unique_routers),
                    "count": len(unique_routers),
                    "window_seconds": self.anomaly_window,
                    "detected_at": now,
                }
                
                self._anomaly_alerts.append(anomaly)
                self._stats["anomalies_detected"] += 1
                
                logger.warning(
                    f"ECLIPSE ANOMALY DETECTED: {len(unique_routers)} routers "
                    f"returned same error '{error_code}' within {self.anomaly_window}s"
                )
                
                return anomaly
        
        return None
    
    async def _oob_verification_loop(self) -> None:
        """
        Periodically verify router information via out-of-band channel.
        
        OOB verification provides an independent check that our connected
        routers are legitimate and not part of an eclipse attack.
        """
        while self._running:
            try:
                await asyncio.sleep(self.oob_verification_interval)
                if not self._running:
                    break
                
                if not self.oob_verification_url:
                    continue
                
                await self._perform_oob_verification()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"OOB verification loop error: {e}")
    
    async def _perform_oob_verification(self) -> bool:
        """
        Perform out-of-band verification of connected routers.
        
        Contacts an independent verification service to confirm that
        our connected routers are registered and legitimate.
        
        Returns:
            True if verification passed, False otherwise
        """
        if not self.oob_verification_url:
            return True
        
        now = time.time()
        self._last_oob_verification = now
        self._stats["oob_verifications"] += 1
        
        # Get list of connected router IDs
        connected_router_ids = list(self.connections.keys())
        
        if not connected_router_ids:
            return True  # No routers to verify
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.oob_verification_url,
                    json={
                        "node_id": self.node_id,
                        "router_ids": connected_router_ids,
                        "timestamp": now,
                    },
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"OOB verification failed: HTTP {resp.status}"
                        )
                        self._stats["oob_verification_failures"] += 1
                        return False
                    
                    data = await resp.json()
                    
                    # Process verification results
                    verified_ids = set(data.get("verified", []))
                    unknown_ids = set(data.get("unknown", []))
                    suspicious_ids = set(data.get("suspicious", []))
                    
                    self._oob_verified_routers = verified_ids
                    
                    if suspicious_ids:
                        logger.warning(
                            f"OOB verification flagged {len(suspicious_ids)} "
                            f"suspicious routers: {[s[:16] + '...' for s in suspicious_ids]}"
                        )
                        
                        # Consider disconnecting from suspicious routers
                        for router_id in suspicious_ids:
                            if router_id in self.connections:
                                logger.warning(
                                    f"Disconnecting from suspicious router {router_id[:16]}..."
                                )
                                await self._rotate_router(router_id, "oob_suspicious")
                    
                    if unknown_ids:
                        logger.info(
                            f"OOB verification: {len(unknown_ids)} routers unknown "
                            f"(may be new registrations)"
                        )
                    
                    logger.info(
                        f"OOB verification complete: {len(verified_ids)} verified, "
                        f"{len(unknown_ids)} unknown, {len(suspicious_ids)} suspicious"
                    )
                    
                    return len(suspicious_ids) == 0
                    
        except aiohttp.ClientError as e:
            logger.warning(f"OOB verification network error: {e}")
            self._stats["oob_verification_failures"] += 1
            return False
        except Exception as e:
            logger.warning(f"OOB verification error: {e}")
            self._stats["oob_verification_failures"] += 1
            return False
    
    def get_eclipse_mitigation_status(self) -> Dict[str, Any]:
        """
        Get comprehensive eclipse mitigation status.
        
        Returns:
            Dict with diversity metrics, rotation status, anomaly alerts, etc.
        """
        now = time.time()
        
        # Connection age info
        connection_ages = {}
        for router_id, conn in self.connections.items():
            conn_time = self._connection_timestamps.get(router_id, conn.connected_at)
            connection_ages[router_id[:16] + "..."] = {
                "age_hours": (now - conn_time) / 3600,
                "subnet": self._get_router_subnet(conn.router),
            }
        
        return {
            # Diversity status
            "diversity": {
                "subnets_connected": len(self._connected_subnets),
                "asns_connected": len(self._connected_asns),
                "min_subnets_required": self.min_diverse_subnets,
                "min_asns_required": self.min_diverse_asns,
                "requirements_met": self._check_diversity_requirements(),
            },
            
            # Rotation status
            "rotation": {
                "enabled": self.rotation_enabled,
                "last_rotation": self._last_rotation,
                "rotation_interval_hours": self.rotation_interval / 3600,
                "max_connection_age_hours": self.rotation_max_age / 3600,
                "routers_rotated_total": self._stats.get("routers_rotated", 0),
            },
            
            # Anomaly detection
            "anomaly_detection": {
                "enabled": self.anomaly_detection_enabled,
                "recent_failures": len(self._failure_events),
                "anomaly_window_seconds": self.anomaly_window,
                "anomaly_threshold": self.anomaly_threshold,
                "recent_anomalies": len(self._anomaly_alerts),
                "total_anomalies_detected": self._stats.get("anomalies_detected", 0),
            },
            
            # OOB verification
            "oob_verification": {
                "enabled": self.oob_verification_enabled,
                "url": self.oob_verification_url,
                "last_verification": self._last_oob_verification,
                "verified_routers": len(self._oob_verified_routers),
                "total_verifications": self._stats.get("oob_verifications", 0),
                "verification_failures": self._stats.get("oob_verification_failures", 0),
            },
            
            # Connection details
            "connections": connection_ages,
            
            # Recent anomaly alerts
            "recent_anomaly_alerts": self._anomaly_alerts[-5:],  # Last 5 alerts
        }
    
    def get_anomaly_alerts(self) -> List[Dict[str, Any]]:
        """Get recent anomaly alerts for monitoring."""
        return list(self._anomaly_alerts)
    
    def clear_anomaly_alerts(self) -> int:
        """Clear anomaly alerts (for testing or after investigation)."""
        count = len(self._anomaly_alerts)
        self._anomaly_alerts.clear()
        return count

    # =========================================================================
    # MALICIOUS ROUTER DETECTION (Issue #119)
    # =========================================================================
    
    def _get_router_metrics(self, router_id: str) -> RouterBehaviorMetrics:
        """
        Get or create behavior metrics for a router.
        
        Args:
            router_id: The router's ID
            
        Returns:
            RouterBehaviorMetrics for the router
        """
        if router_id not in self._router_behavior_metrics:
            self._router_behavior_metrics[router_id] = RouterBehaviorMetrics(
                router_id=router_id,
                first_seen=time.time(),
            )
        return self._router_behavior_metrics[router_id]
    
    def _record_message_sent(self, router_id: str, message_id: str) -> None:
        """
        Record that a message was sent through a router.
        
        Called when sending a message to track delivery outcomes.
        
        Args:
            router_id: Router that received the message
            message_id: The message ID for correlation
        """
        if not self.misbehavior_detection_enabled:
            return
        
        metrics = self._get_router_metrics(router_id)
        metrics.messages_sent += 1
        metrics.last_updated = time.time()
    
    def _record_delivery_outcome(
        self,
        router_id: str,
        message_id: str,
        delivered: bool,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Record the outcome of a message delivery attempt.
        
        Called when we receive an ACK (or timeout) to track router reliability.
        
        Args:
            router_id: Router that handled the message
            message_id: The message ID
            delivered: Whether the message was delivered successfully
            latency_ms: End-to-end latency if delivered
        """
        if not self.misbehavior_detection_enabled:
            return
        
        metrics = self._get_router_metrics(router_id)
        metrics.record_delivery(delivered)
        
        if delivered and latency_ms is not None:
            metrics.record_latency(latency_ms)
        
        # Check for misbehavior after recording
        self._check_router_behavior(router_id)
    
    def _record_ack_outcome(self, router_id: str, success: bool) -> None:
        """
        Record an ACK success or failure for a router.
        
        Args:
            router_id: Router that handled the message
            success: Whether ACK was received successfully
        """
        if not self.misbehavior_detection_enabled:
            return
        
        metrics = self._get_router_metrics(router_id)
        metrics.record_ack(success)
        
        # Update stats
        if success:
            self._stats["misbehavior_ack_success"] = self._stats.get("misbehavior_ack_success", 0) + 1
        else:
            self._stats["misbehavior_ack_failure"] = self._stats.get("misbehavior_ack_failure", 0) + 1
        
        # Check for misbehavior after recording
        self._check_router_behavior(router_id)
    
    def _update_network_baseline(self) -> None:
        """
        Update the network baseline from all router metrics.
        
        The baseline represents "normal" network behavior and is used
        to detect anomalies.
        """
        if not self._router_behavior_metrics:
            return
        
        # Collect stats from all routers with enough samples
        delivery_rates = []
        latencies = []
        ack_rates = []
        
        for router_id, metrics in self._router_behavior_metrics.items():
            # Skip flagged routers - they skew the baseline
            if metrics.flagged:
                continue
            
            if metrics.messages_sent >= self.min_messages_for_detection:
                delivery_rates.append(metrics.delivery_rate)
                ack_rates.append(metrics.ack_success_rate)
                
                if metrics.latency_samples > 0:
                    latencies.append(metrics.avg_latency_ms)
        
        if len(delivery_rates) < self.baseline_min_samples:
            # Not enough data for baseline
            if self._network_baseline is None:
                # Use default baseline
                self._network_baseline = NetworkBaseline()
            return
        
        # Calculate statistics
        avg_delivery = sum(delivery_rates) / len(delivery_rates)
        avg_ack = sum(ack_rates) / len(ack_rates)
        
        # Standard deviation for delivery rate
        if len(delivery_rates) > 1:
            variance = sum((r - avg_delivery) ** 2 for r in delivery_rates) / len(delivery_rates)
            delivery_stddev = variance ** 0.5
        else:
            delivery_stddev = 0.05  # Default
        
        # Latency statistics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            if len(latencies) > 1:
                lat_variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
                latency_stddev = lat_variance ** 0.5
            else:
                latency_stddev = 50.0  # Default
        else:
            avg_latency = 100.0
            latency_stddev = 50.0
        
        self._network_baseline = NetworkBaseline(
            avg_delivery_rate=avg_delivery,
            avg_latency_ms=avg_latency,
            avg_ack_success_rate=avg_ack,
            sample_count=len(delivery_rates),
            last_updated=time.time(),
            delivery_rate_stddev=max(delivery_stddev, 0.01),  # Avoid zero stddev
            latency_stddev_ms=max(latency_stddev, 10.0),
        )
        
        logger.debug(
            f"Updated network baseline: delivery_rate={avg_delivery:.3f}±{delivery_stddev:.3f}, "
            f"latency={avg_latency:.1f}±{latency_stddev:.1f}ms, samples={len(delivery_rates)}"
        )
    
    def _check_router_behavior(self, router_id: str) -> Optional[MisbehaviorReport]:
        """
        Check if a router's behavior is anomalous and should be flagged.
        
        Compares the router's metrics against the network baseline
        and flags if anomalous.
        
        Args:
            router_id: Router to check
            
        Returns:
            MisbehaviorReport if flagged, None otherwise
        """
        if not self.misbehavior_detection_enabled:
            return None
        
        metrics = self._get_router_metrics(router_id)
        
        # Need minimum samples before detection
        if metrics.messages_sent < self.min_messages_for_detection:
            return None
        
        # Already flagged - don't re-check
        if metrics.flagged:
            return None
        
        # Update baseline periodically
        baseline = self._network_baseline
        if baseline is None:
            self._update_network_baseline()
            baseline = self._network_baseline
        
        if baseline is None:
            # Still no baseline - use defaults
            baseline = NetworkBaseline()
        
        # Check for anomalies
        evidence_list: List[MisbehaviorEvidence] = []
        anomaly_score = 0.0
        misbehavior_type = ""
        
        # Check delivery rate
        delivery_rate = metrics.delivery_rate
        if delivery_rate < self.delivery_rate_threshold:
            if baseline.is_delivery_rate_anomalous(delivery_rate, self.latency_threshold_stddevs):
                evidence_list.append(MisbehaviorEvidence(
                    misbehavior_type=MisbehaviorType.MESSAGE_DROP,
                    delivery_rate_baseline=baseline.avg_delivery_rate,
                    delivery_rate_observed=delivery_rate,
                    description=f"Delivery rate {delivery_rate:.1%} below threshold {self.delivery_rate_threshold:.1%}"
                ))
                anomaly_score += 0.4
                misbehavior_type = MisbehaviorType.MESSAGE_DROP
        
        # Check ACK failure rate
        ack_failure_rate = 1.0 - metrics.ack_success_rate
        if ack_failure_rate > self.ack_failure_threshold:
            evidence_list.append(MisbehaviorEvidence(
                misbehavior_type=MisbehaviorType.ACK_FAILURE,
                description=f"ACK failure rate {ack_failure_rate:.1%} exceeds threshold {self.ack_failure_threshold:.1%}"
            ))
            anomaly_score += 0.3
            if not misbehavior_type:
                misbehavior_type = MisbehaviorType.ACK_FAILURE
        
        # Check latency
        if metrics.latency_samples > 0 and metrics.avg_latency_ms > 0:
            if baseline.is_latency_anomalous(metrics.avg_latency_ms, self.latency_threshold_stddevs):
                evidence_list.append(MisbehaviorEvidence(
                    misbehavior_type=MisbehaviorType.MESSAGE_DELAY,
                    expected_latency_ms=baseline.avg_latency_ms,
                    actual_latency_ms=metrics.avg_latency_ms,
                    description=f"Latency {metrics.avg_latency_ms:.1f}ms exceeds baseline {baseline.avg_latency_ms:.1f}ms"
                ))
                anomaly_score += 0.3
                if not misbehavior_type:
                    misbehavior_type = MisbehaviorType.MESSAGE_DELAY
        
        # If no anomalies found, return None
        if not evidence_list:
            return None
        
        # Calculate severity (0.0 to 1.0)
        severity = min(1.0, anomaly_score)
        
        # Compute overall anomaly score for the metrics
        metrics.anomaly_score = severity
        
        # Only flag if severity exceeds threshold
        if severity < self.mild_severity_threshold:
            return None
        
        # Flag the router
        metrics.flagged = True
        metrics.flag_reason = misbehavior_type
        
        # Create misbehavior report
        report = MisbehaviorReport(
            reporter_id=self.node_id,
            router_id=router_id,
            misbehavior_type=misbehavior_type,
            evidence=evidence_list[:self.max_evidence_per_report],
            metrics=metrics,
            severity=severity,
        )
        
        # Store in flagged routers
        self._flagged_routers[router_id] = report
        self._stats["routers_flagged"] = self._stats.get("routers_flagged", 0) + 1
        
        logger.warning(
            f"FLAGGED ROUTER {router_id[:16]}... for {misbehavior_type}: "
            f"severity={severity:.2f}, delivery_rate={delivery_rate:.1%}, "
            f"ack_success={metrics.ack_success_rate:.1%}"
        )
        
        # Notify callback if registered
        if self.on_router_flagged:
            try:
                self.on_router_flagged(router_id, report)
            except Exception as e:
                logger.warning(f"on_router_flagged callback error: {e}")
        
        # Report to seeds if enabled
        if self.report_to_seeds:
            try:
                # Only create task if there's a running event loop
                loop = asyncio.get_running_loop()
                loop.create_task(self._report_misbehavior_to_seeds(report))
            except RuntimeError:
                # No running event loop - skip async reporting
                # This happens in synchronous test contexts
                logger.debug(
                    f"Skipping async misbehavior report (no event loop) "
                    f"for router {router_id[:16]}..."
                )
        
        return report
    
    async def _report_misbehavior_to_seeds(self, report: MisbehaviorReport) -> bool:
        """
        Report router misbehavior to seed nodes.
        
        Seeds aggregate reports from multiple nodes to identify
        systematically misbehaving routers.
        
        Args:
            report: The misbehavior report to submit
            
        Returns:
            True if report was submitted successfully
        """
        router_id = report.router_id
        
        # Check cooldown
        last_report = self._last_misbehavior_reports.get(router_id, 0)
        if time.time() - last_report < self.report_cooldown_seconds:
            logger.debug(
                f"Skipping misbehavior report for {router_id[:16]}... "
                f"(cooldown not expired)"
            )
            return False
        
        # Sign the report
        report_data = report.to_dict()
        report_data.pop("signature", None)  # Remove signature before signing
        signature_data = json.dumps(report_data, sort_keys=True).encode()
        report.signature = self.private_key.sign(signature_data).hex()
        
        # Submit to seeds via discovery client
        try:
            submitted = await self.discovery.report_misbehavior(report)
            
            if submitted:
                self._last_misbehavior_reports[router_id] = time.time()
                self._stats["misbehavior_reports_sent"] = self._stats.get("misbehavior_reports_sent", 0) + 1
                logger.info(
                    f"Reported misbehavior for router {router_id[:16]}... to seeds"
                )
            
            return submitted
            
        except Exception as e:
            logger.warning(f"Failed to report misbehavior to seeds: {e}")
            return False
    
    def is_router_flagged(self, router_id: str) -> bool:
        """
        Check if a router has been flagged for misbehavior.
        
        Args:
            router_id: Router to check
            
        Returns:
            True if router is flagged
        """
        metrics = self._router_behavior_metrics.get(router_id)
        if metrics and metrics.flagged:
            return True
        return router_id in self._flagged_routers
    
    def get_flagged_routers(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all flagged routers with their reports.
        
        Returns:
            Dict of router_id -> report details
        """
        result = {}
        for router_id, report in self._flagged_routers.items():
            result[router_id] = {
                "misbehavior_type": report.misbehavior_type,
                "severity": report.severity,
                "flagged_at": report.timestamp,
                "evidence_count": len(report.evidence),
            }
        return result
    
    def clear_router_flag(self, router_id: str) -> bool:
        """
        Clear the misbehavior flag for a router.
        
        Useful for testing or after a router has been verified as fixed.
        
        Args:
            router_id: Router to unflag
            
        Returns:
            True if flag was cleared, False if router wasn't flagged
        """
        cleared = False
        
        if router_id in self._flagged_routers:
            del self._flagged_routers[router_id]
            cleared = True
        
        metrics = self._router_behavior_metrics.get(router_id)
        if metrics and metrics.flagged:
            metrics.flagged = False
            metrics.flag_reason = ""
            metrics.anomaly_score = 0.0
            cleared = True
        
        if cleared:
            logger.info(f"Cleared misbehavior flag for router {router_id[:16]}...")
        
        return cleared
    
    def get_router_behavior_metrics(self, router_id: str) -> Optional[Dict[str, Any]]:
        """
        Get behavior metrics for a specific router.
        
        Args:
            router_id: Router to get metrics for
            
        Returns:
            Dict of metrics or None if no data
        """
        metrics = self._router_behavior_metrics.get(router_id)
        if metrics:
            return metrics.to_dict()
        return None
    
    def get_all_router_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get behavior metrics for all tracked routers.
        
        Returns:
            Dict of router_id -> metrics
        """
        return {
            router_id: metrics.to_dict()
            for router_id, metrics in self._router_behavior_metrics.items()
        }
    
    def get_network_baseline(self) -> Optional[Dict[str, Any]]:
        """
        Get the current network baseline.
        
        Returns:
            Dict of baseline metrics or None if not computed
        """
        if self._network_baseline:
            return self._network_baseline.to_dict()
        return None
    
    def get_misbehavior_detection_stats(self) -> Dict[str, Any]:
        """
        Get statistics for misbehavior detection.
        
        Returns:
            Dict with detection statistics
        """
        return {
            "enabled": self.misbehavior_detection_enabled,
            "routers_tracked": len(self._router_behavior_metrics),
            "routers_flagged": len(self._flagged_routers),
            "reports_sent": self._stats.get("misbehavior_reports_sent", 0),
            "ack_success_total": self._stats.get("misbehavior_ack_success", 0),
            "ack_failure_total": self._stats.get("misbehavior_ack_failure", 0),
            "baseline": self.get_network_baseline(),
            "thresholds": {
                "delivery_rate": self.delivery_rate_threshold,
                "ack_failure": self.ack_failure_threshold,
                "latency_stddevs": self.latency_threshold_stddevs,
                "min_messages": self.min_messages_for_detection,
            },
            "flagged_routers": list(self._flagged_routers.keys()),
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
        seed_urls: Optional list of seed URLs for router discovery.
                   Primary seed should be first, with fallbacks following.
                   These are added to the discovery client as custom seeds.
        **kwargs: Additional NodeClient parameters
        
    Returns:
        Configured NodeClient
        
    Example:
        # With seed redundancy (Issue #108)
        node = create_node_client(
            private_key=my_private_key,
            encryption_private_key=my_enc_key,
            seed_urls=[
                "https://primary.seed.com:8470",
                "https://backup1.seed.com:8470",
                "https://backup2.seed.com:8470",
            ],
        )
    """
    # Derive node ID from public key
    node_id = private_key.public_key().public_bytes_raw().hex()
    
    # Include seed_urls in kwargs if provided
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
