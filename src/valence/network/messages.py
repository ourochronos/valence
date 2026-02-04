"""
Message formats for Valence Relay Protocol.

RelayMessage: What routers see (encrypted payload, routing info)
DeliverPayload: What recipients see after decryption (actual content)
AckRequest: Configuration for acknowledgment behavior
AckMessage: End-to-end acknowledgment that proves recipient received message
BackPressureMessage: Router signals load status to connected nodes

Cover Traffic (Issue #116):
MessagePadding: Utilities to pad messages to fixed sizes
CoverMessage: Dummy message indistinguishable from real traffic

Circuit Messages (Issue #115):
CircuitCreateMessage: Request to establish circuit hop
CircuitCreatedMessage: Confirmation of circuit hop establishment
CircuitRelayMessage: Message relayed through circuit
CircuitDestroyMessage: Teardown circuit
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import json
import os
import time
import uuid


# =============================================================================
# MESSAGE PADDING (Issue #116 - Cover Traffic)
# =============================================================================

# Fixed message size buckets for traffic analysis resistance
# Messages are padded to the next bucket size to hide actual content length
MESSAGE_SIZE_BUCKETS = [
    1024,       # 1 KB - small messages (typical chat)
    4096,       # 4 KB - medium messages
    16384,      # 16 KB - larger messages
    65536,      # 64 KB - max standard message
]

# Padding byte used for PKCS7-style padding
PADDING_MAGIC = b'\x80'  # Start of padding marker
PADDING_FILL = b'\x00'   # Fill byte


def get_padded_size(content_length: int) -> int:
    """
    Get the bucket size for a given content length.
    
    Returns the smallest bucket that fits the content, or the content
    length if it exceeds all buckets (no padding for very large messages).
    
    Args:
        content_length: Size of the unpadded content in bytes
        
    Returns:
        Target padded size in bytes
    """
    for bucket in MESSAGE_SIZE_BUCKETS:
        if content_length < bucket - 1:  # -1 for padding marker
            return bucket
    # Content exceeds all buckets, return as-is
    return content_length


def pad_message(content: bytes, target_size: Optional[int] = None) -> bytes:
    """
    Pad a message to a fixed bucket size for traffic analysis resistance.
    
    Uses a simple padding scheme:
    - Appends PADDING_MAGIC (0x80) as marker
    - Fills remaining space with PADDING_FILL (0x00)
    
    This ensures padded messages are indistinguishable from each other
    at the same bucket size, hiding the actual content length.
    
    Args:
        content: Raw message bytes to pad
        target_size: Optional specific target size. If None, uses next bucket.
        
    Returns:
        Padded message bytes of exactly target_size
        
    Example:
        >>> padded = pad_message(b"Hello")
        >>> len(padded)
        1024
        >>> unpad_message(padded)
        b'Hello'
    """
    if target_size is None:
        target_size = get_padded_size(len(content))
    
    # Calculate padding needed
    padding_needed = target_size - len(content) - 1  # -1 for marker
    
    if padding_needed < 0:
        # Content already too large for target, return as-is with marker
        return content + PADDING_MAGIC
    
    # Build padded message: content + marker + fill
    return content + PADDING_MAGIC + (PADDING_FILL * padding_needed)


def unpad_message(padded: bytes) -> bytes:
    """
    Remove padding from a padded message.
    
    Finds the PADDING_MAGIC marker and returns content before it.
    If no marker found, returns the original bytes (unpadded message).
    
    Args:
        padded: Padded message bytes
        
    Returns:
        Original unpadded content bytes
        
    Raises:
        ValueError: If padding is malformed (marker in unexpected position)
    """
    # Find the padding marker (search from end for efficiency)
    try:
        marker_pos = padded.rindex(PADDING_MAGIC)
    except ValueError:
        # No marker found - message wasn't padded
        return padded
    
    # Verify everything after marker is padding fill
    padding_section = padded[marker_pos + 1:]
    if padding_section and not all(b == 0 for b in padding_section):
        # Invalid padding - marker wasn't actually padding
        # This could happen if content contains 0x80 naturally
        # Search for the last valid marker
        for i in range(len(padded) - 1, -1, -1):
            if padded[i:i+1] == PADDING_MAGIC:
                if all(b == 0 for b in padded[i+1:]):
                    return padded[:i]
        # No valid padding found, return as-is
        return padded
    
    return padded[:marker_pos]


def calculate_padding_overhead(content_length: int) -> Tuple[int, float]:
    """
    Calculate padding overhead for a given content length.
    
    Args:
        content_length: Size of unpadded content
        
    Returns:
        Tuple of (padded_size, overhead_percentage)
    """
    padded_size = get_padded_size(content_length)
    if padded_size == content_length:
        return (content_length, 0.0)
    overhead_pct = ((padded_size - content_length) / content_length) * 100
    return (padded_size, overhead_pct)


# =============================================================================
# BACK-PRESSURE MESSAGES
# =============================================================================


@dataclass
class BackPressureMessage:
    """
    Back-pressure signal from router to connected nodes.
    
    When a router is under heavy load, it sends this message to
    connected nodes to request they slow down or try alternative routers.
    
    Attributes:
        type: Always "back_pressure"
        active: True when back-pressure is active, False when released
        load_pct: Current load percentage (0-100)
        retry_after_ms: Suggested delay before retrying (milliseconds)
        reason: Human-readable reason for back-pressure
    """
    type: str = field(default="back_pressure", init=False)
    active: bool = True
    load_pct: float = 0.0
    retry_after_ms: int = 1000
    reason: str = ""
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "active": self.active,
            "load_pct": self.load_pct,
            "retry_after_ms": self.retry_after_ms,
            "reason": self.reason,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "BackPressureMessage":
        """Deserialize from dict."""
        return cls(
            active=data.get("active", True),
            load_pct=data.get("load_pct", 0.0),
            retry_after_ms=data.get("retry_after_ms", 1000),
            reason=data.get("reason", ""),
        )


# =============================================================================
# ACKNOWLEDGMENT MESSAGES
# =============================================================================


@dataclass
class AckRequest:
    """
    Configuration for message acknowledgment behavior.
    
    Attached to outgoing messages to specify whether ACK is required
    and timeout settings.
    """
    message_id: str
    require_ack: bool = True
    ack_timeout_ms: int = 30000  # 30 seconds default


@dataclass
class AckMessage:
    """
    End-to-end acknowledgment message.
    
    Sent by the recipient back to the sender to prove message delivery.
    The signature proves the recipient actually received and processed
    the message (not just that it was relayed).
    """
    type: str = field(default="ack", init=False)
    original_message_id: str = ""
    received_at: float = 0.0
    recipient_id: str = ""
    signature: str = ""  # Hex-encoded signature proving receipt
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "original_message_id": self.original_message_id,
            "received_at": self.received_at,
            "recipient_id": self.recipient_id,
            "signature": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "AckMessage":
        """Deserialize from dict."""
        return cls(
            original_message_id=data.get("original_message_id", ""),
            received_at=data.get("received_at", 0.0),
            recipient_id=data.get("recipient_id", ""),
            signature=data.get("signature", ""),
        )


@dataclass
class RelayMessage:
    """
    Message format for router relay - router sees only this.
    
    The payload is an encrypted blob that routers cannot decrypt.
    Routers only need next_hop to forward the message.
    """
    message_id: str
    next_hop: str  # Recipient node ID or "local"
    payload: str   # Encrypted blob (hex), router cannot decrypt
    ttl: int
    timestamp: float
    
    @classmethod
    def create(
        cls,
        next_hop: str,
        payload: str,
        ttl: int = 10,
        message_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> "RelayMessage":
        """Create a new relay message with auto-generated ID and timestamp."""
        return cls(
            message_id=message_id or str(uuid.uuid4()),
            next_hop=next_hop,
            payload=payload,
            ttl=ttl,
            timestamp=timestamp or time.time()
        )
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": "relay",
            "message_id": self.message_id,
            "next_hop": self.next_hop,
            "payload": self.payload,
            "ttl": self.ttl,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RelayMessage":
        """Deserialize from dict."""
        return cls(
            message_id=data["message_id"],
            next_hop=data["next_hop"],
            payload=data["payload"],
            ttl=data["ttl"],
            timestamp=data["timestamp"]
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "RelayMessage":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# HEALTH GOSSIP MESSAGES
# =============================================================================


@dataclass
class RouterHealthObservation:
    """
    A single router health observation from a node's perspective.
    
    Nodes track health metrics for routers they connect to and share
    these observations with peers via gossip.
    """
    router_id: str
    latency_ms: float = 0.0
    success_rate: float = 1.0  # 0.0 to 1.0
    failure_count: int = 0
    success_count: int = 0
    last_seen: float = 0.0
    load_pct: float = 0.0
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "router_id": self.router_id,
            "latency_ms": self.latency_ms,
            "success_rate": self.success_rate,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_seen": self.last_seen,
            "load_pct": self.load_pct,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RouterHealthObservation":
        """Deserialize from dict."""
        return cls(
            router_id=data.get("router_id", ""),
            latency_ms=data.get("latency_ms", 0.0),
            success_rate=data.get("success_rate", 1.0),
            failure_count=data.get("failure_count", 0),
            success_count=data.get("success_count", 0),
            last_seen=data.get("last_seen", 0.0),
            load_pct=data.get("load_pct", 0.0),
        )


@dataclass
class HealthGossip:
    """
    Health gossip message for sharing router observations between nodes.
    
    Nodes periodically share their router health observations with peers
    to improve collective routing decisions. Observations are sampled
    to keep gossip lightweight.
    
    Attributes:
        type: Always "health_gossip"
        source_node_id: The node sharing these observations
        timestamp: When the gossip was generated
        observations: List of router health observations (sampled)
        ttl: Hop limit for gossip propagation (default 2)
    """
    type: str = field(default="health_gossip", init=False)
    source_node_id: str = ""
    timestamp: float = 0.0
    observations: list = field(default_factory=list)  # List[RouterHealthObservation]
    ttl: int = 2  # Limit propagation depth
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "source_node_id": self.source_node_id,
            "timestamp": self.timestamp,
            "observations": [
                obs.to_dict() if hasattr(obs, 'to_dict') else obs
                for obs in self.observations
            ],
            "ttl": self.ttl,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "HealthGossip":
        """Deserialize from dict."""
        observations = [
            RouterHealthObservation.from_dict(obs) if isinstance(obs, dict) else obs
            for obs in data.get("observations", [])
        ]
        return cls(
            source_node_id=data.get("source_node_id", ""),
            timestamp=data.get("timestamp", 0.0),
            observations=observations,
            ttl=data.get("ttl", 2),
        )


@dataclass
class DeliverPayload:
    """
    Inner payload - only recipient can decrypt and see this.
    
    Contains the actual message content, sender identity,
    and optional reply path for responses.
    """
    sender_id: str
    message_type: str  # "belief", "query", "response", "ack"
    content: dict
    reply_path: Optional[str] = None  # Encrypted return path
    timestamp: float = field(default_factory=time.time)
    message_id: Optional[str] = None  # For ACK correlation
    require_ack: bool = False  # Whether sender wants ACK
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "sender_id": self.sender_id,
            "message_type": self.message_type,
            "content": self.content,
            "reply_path": self.reply_path,
            "timestamp": self.timestamp,
            "message_id": self.message_id,
            "require_ack": self.require_ack,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DeliverPayload":
        """Deserialize from dict."""
        return cls(
            sender_id=data["sender_id"],
            message_type=data["message_type"],
            content=data["content"],
            reply_path=data.get("reply_path"),
            timestamp=data.get("timestamp", time.time()),
            message_id=data.get("message_id"),
            require_ack=data.get("require_ack", False),
        )
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for encryption."""
        return json.dumps(self.to_dict()).encode()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "DeliverPayload":
        """Deserialize from bytes after decryption."""
        return cls.from_dict(json.loads(data.decode()))


# =============================================================================
# COVER TRAFFIC MESSAGES (Issue #116)
# =============================================================================


@dataclass
class CoverMessage:
    """
    Cover traffic message - indistinguishable from real traffic.
    
    Cover messages are sent when a node is idle to obscure real
    communication patterns. They look identical to real messages
    from a router's perspective:
    - Same encryption (routers can't read content)
    - Same padded sizes (all messages fit bucket sizes)
    - Same routing metadata format
    
    The recipient (another node in the network) recognizes cover
    traffic by the message_type and silently discards it without
    triggering callbacks.
    
    Attributes:
        type: Always "cover" - used by recipient to identify
        message_id: Unique ID (same format as real messages)
        timestamp: When the cover message was generated
        nonce: Random bytes to ensure uniqueness and proper padding
    """
    type: str = field(default="cover", init=False)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    nonce: str = field(default_factory=lambda: os.urandom(32).hex())
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "nonce": self.nonce,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CoverMessage":
        """Deserialize from dict."""
        msg = cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", time.time()),
            nonce=data.get("nonce", os.urandom(32).hex()),
        )
        return msg
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for encryption."""
        return json.dumps(self.to_dict()).encode()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "CoverMessage":
        """Deserialize from bytes."""
        return cls.from_dict(json.loads(data.decode()))
    
    @staticmethod
    def is_cover_message(data: dict) -> bool:
        """Check if a decrypted payload is cover traffic."""
        return data.get("type") == "cover"


def generate_cover_content(target_bucket: Optional[int] = None) -> bytes:
    """
    Generate random cover content that fills a message bucket.
    
    The content is designed to:
    - Fill a specific bucket size when padded
    - Look like encrypted content (random bytes)
    - Be efficiently generated
    
    Args:
        target_bucket: Desired bucket size. If None, randomly selects one.
        
    Returns:
        Random bytes sized to fit the target bucket after JSON serialization
    """
    if target_bucket is None:
        # Random bucket selection weighted toward smaller (more common)
        import random
        weights = [0.5, 0.3, 0.15, 0.05]  # Favor smaller buckets
        target_bucket = random.choices(MESSAGE_SIZE_BUCKETS, weights=weights, k=1)[0]
    
    # Account for JSON overhead of CoverMessage (~150 bytes typical)
    # and padding overhead (1 byte marker)
    json_overhead = 200  # Conservative estimate
    content_size = max(16, target_bucket - json_overhead - 1)
    
    return os.urandom(content_size)


# =============================================================================
# CIRCUIT MESSAGES (Issue #115 - Privacy-Enhanced Routing)
# =============================================================================


@dataclass
class CircuitHop:
    """
    Represents a single hop in a circuit.
    
    Each hop contains the router ID and the shared key established
    during circuit creation (via Diffie-Hellman key exchange).
    """
    router_id: str
    shared_key: bytes = field(default=b"", repr=False)  # 32-byte AES key
    
    def to_dict(self) -> dict:
        """Serialize to dict (excluding secret key)."""
        return {
            "router_id": self.router_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitHop":
        """Deserialize from dict."""
        return cls(
            router_id=data["router_id"],
        )


@dataclass
class Circuit:
    """
    Represents an established circuit through multiple routers.
    
    A circuit provides enhanced privacy by routing messages through
    2-3 routers, with layered (onion) encryption. Each router only
    knows the previous and next hop, never the full path.
    
    Attributes:
        circuit_id: Unique identifier for this circuit
        hops: List of CircuitHop objects (in order from node to destination)
        created_at: Timestamp when circuit was established
        expires_at: Timestamp when circuit should be torn down
        message_count: Number of messages sent through this circuit
        max_messages: Maximum messages before rotation (default 100)
    """
    circuit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    hops: List[CircuitHop] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0  # 0 means no expiry set
    message_count: int = 0
    max_messages: int = 100
    
    def __post_init__(self):
        if self.expires_at == 0.0:
            # Default 10 minute lifetime
            self.expires_at = self.created_at + 600
    
    @property
    def is_expired(self) -> bool:
        """Check if circuit has expired."""
        return time.time() > self.expires_at
    
    @property
    def needs_rotation(self) -> bool:
        """Check if circuit needs rotation (expired or too many messages)."""
        return self.is_expired or self.message_count >= self.max_messages
    
    @property
    def hop_count(self) -> int:
        """Number of hops in the circuit."""
        return len(self.hops)
    
    def to_dict(self) -> dict:
        """Serialize to dict (excluding secret keys)."""
        return {
            "circuit_id": self.circuit_id,
            "hops": [hop.to_dict() for hop in self.hops],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "message_count": self.message_count,
            "max_messages": self.max_messages,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Circuit":
        """Deserialize from dict."""
        return cls(
            circuit_id=data["circuit_id"],
            hops=[CircuitHop.from_dict(h) for h in data.get("hops", [])],
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at", 0.0),
            message_count=data.get("message_count", 0),
            max_messages=data.get("max_messages", 100),
        )


@dataclass
class CircuitCreateMessage:
    """
    Request to create a circuit hop at a router.
    
    Sent to each router in the circuit path during establishment.
    The router responds with CircuitCreatedMessage containing its
    ephemeral public key for the Diffie-Hellman key exchange.
    
    Attributes:
        type: Always "circuit_create"
        circuit_id: Unique circuit identifier
        ephemeral_public: Sender's ephemeral X25519 public key (hex)
        next_hop: Router ID of the next hop (None for exit node)
        extend_payload: Encrypted payload for next hop (onion layer)
    """
    type: str = field(default="circuit_create", init=False)
    circuit_id: str = ""
    ephemeral_public: str = ""  # Hex-encoded X25519 public key
    next_hop: Optional[str] = None  # None means this is the exit node
    extend_payload: Optional[str] = None  # Encrypted for next router
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "circuit_id": self.circuit_id,
            "ephemeral_public": self.ephemeral_public,
            "next_hop": self.next_hop,
            "extend_payload": self.extend_payload,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitCreateMessage":
        """Deserialize from dict."""
        return cls(
            circuit_id=data.get("circuit_id", ""),
            ephemeral_public=data.get("ephemeral_public", ""),
            next_hop=data.get("next_hop"),
            extend_payload=data.get("extend_payload"),
        )


@dataclass
class CircuitCreatedMessage:
    """
    Response confirming circuit hop establishment.
    
    Sent by a router after successfully processing CircuitCreateMessage.
    Contains the router's ephemeral public key for completing the
    Diffie-Hellman key exchange.
    
    Attributes:
        type: Always "circuit_created"
        circuit_id: The circuit identifier
        ephemeral_public: Router's ephemeral X25519 public key (hex)
        extend_response: Encrypted response from next hop (if extended)
    """
    type: str = field(default="circuit_created", init=False)
    circuit_id: str = ""
    ephemeral_public: str = ""  # Hex-encoded X25519 public key
    extend_response: Optional[str] = None  # Response from next hop
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "circuit_id": self.circuit_id,
            "ephemeral_public": self.ephemeral_public,
            "extend_response": self.extend_response,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitCreatedMessage":
        """Deserialize from dict."""
        return cls(
            circuit_id=data.get("circuit_id", ""),
            ephemeral_public=data.get("ephemeral_public", ""),
            extend_response=data.get("extend_response"),
        )


@dataclass
class CircuitRelayMessage:
    """
    Message relayed through an established circuit.
    
    The payload is onion-encrypted: each router peels one layer
    and forwards to the next hop. Only the final recipient can
    read the innermost payload.
    
    Attributes:
        type: Always "circuit_relay"
        circuit_id: The circuit this message is traveling through
        payload: Onion-encrypted payload (hex)
        direction: "forward" (toward recipient) or "backward" (toward sender)
    """
    type: str = field(default="circuit_relay", init=False)
    circuit_id: str = ""
    payload: str = ""  # Hex-encoded onion payload
    direction: str = "forward"  # "forward" or "backward"
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "circuit_id": self.circuit_id,
            "payload": self.payload,
            "direction": self.direction,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitRelayMessage":
        """Deserialize from dict."""
        return cls(
            circuit_id=data.get("circuit_id", ""),
            payload=data.get("payload", ""),
            direction=data.get("direction", "forward"),
        )


@dataclass
class CircuitDestroyMessage:
    """
    Request to tear down a circuit.
    
    Sent when a circuit expires, has been used for max messages,
    or is explicitly closed. Each router should clean up its
    circuit state upon receiving this.
    
    Attributes:
        type: Always "circuit_destroy"
        circuit_id: The circuit to destroy
        reason: Optional reason for teardown
    """
    type: str = field(default="circuit_destroy", init=False)
    circuit_id: str = ""
    reason: str = ""
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "circuit_id": self.circuit_id,
            "reason": self.reason,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitDestroyMessage":
        """Deserialize from dict."""
        return cls(
            circuit_id=data.get("circuit_id", ""),
            reason=data.get("reason", ""),
        )


@dataclass
class CircuitExtendMessage:
    """
    Internal message to extend circuit to next hop.
    
    This is the decrypted content of extend_payload in CircuitCreateMessage.
    It contains the information needed to create the next hop.
    
    Attributes:
        next_router_id: Router ID to extend to
        ephemeral_public: Client's ephemeral key for next hop
        next_extend_payload: Encrypted payload for hop after next (if any)
    """
    next_router_id: str = ""
    ephemeral_public: str = ""
    next_extend_payload: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "next_router_id": self.next_router_id,
            "ephemeral_public": self.ephemeral_public,
            "next_extend_payload": self.next_extend_payload,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CircuitExtendMessage":
        """Deserialize from dict."""
        return cls(
            next_router_id=data.get("next_router_id", ""),
            ephemeral_public=data.get("ephemeral_public", ""),
            next_extend_payload=data.get("next_extend_payload"),
        )
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for encryption."""
        return json.dumps(self.to_dict()).encode()
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "CircuitExtendMessage":
        """Deserialize from bytes."""
        return cls.from_dict(json.loads(data.decode()))


# =============================================================================
# MALICIOUS ROUTER DETECTION MESSAGES (Issue #119)
# =============================================================================


class MisbehaviorType:
    """Types of router misbehavior that can be reported."""
    MESSAGE_DROP = "message_drop"  # Router drops messages
    MESSAGE_DELAY = "message_delay"  # Router delays messages excessively
    MESSAGE_MODIFY = "message_modify"  # Router modifies message content
    ACK_FAILURE = "ack_failure"  # Router fails to deliver ACKs
    SELECTIVE_DROP = "selective_drop"  # Router drops messages for specific recipients
    PERFORMANCE_DEGRADATION = "performance_degradation"  # General performance issues


@dataclass
class RouterBehaviorMetrics:
    """
    Metrics tracking a router's behavior over time.
    
    These metrics are used to detect malicious or misbehaving routers
    by comparing against network baseline.
    
    Attributes:
        router_id: The router being tracked
        messages_sent: Total messages sent through this router
        messages_delivered: Messages confirmed delivered (ACK received)
        messages_dropped: Messages that never got ACK (suspected dropped)
        avg_latency_ms: Average message latency in milliseconds
        latency_samples: Number of latency samples collected
        latency_sum_ms: Sum of all latency samples (for avg calculation)
        ack_success_count: Number of successful ACKs received
        ack_failure_count: Number of ACK failures/timeouts
        first_seen: Timestamp when we first observed this router
        last_updated: Timestamp of last metric update
        anomaly_score: Computed anomaly score (0.0 = normal, 1.0 = highly anomalous)
        flagged: Whether this router has been flagged for misbehavior
        flag_reason: Reason for flagging (if flagged)
    """
    router_id: str
    messages_sent: int = 0
    messages_delivered: int = 0
    messages_dropped: int = 0
    avg_latency_ms: float = 0.0
    latency_samples: int = 0
    latency_sum_ms: float = 0.0
    ack_success_count: int = 0
    ack_failure_count: int = 0
    first_seen: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    anomaly_score: float = 0.0
    flagged: bool = False
    flag_reason: str = ""
    
    @property
    def delivery_rate(self) -> float:
        """Calculate message delivery rate (0.0 to 1.0)."""
        total = self.messages_sent
        if total == 0:
            return 1.0  # Assume good until proven otherwise
        return self.messages_delivered / total
    
    @property
    def ack_success_rate(self) -> float:
        """Calculate ACK success rate (0.0 to 1.0)."""
        total = self.ack_success_count + self.ack_failure_count
        if total == 0:
            return 1.0  # Assume good until proven otherwise
        return self.ack_success_count / total
    
    def record_latency(self, latency_ms: float) -> None:
        """Record a latency sample and update running average."""
        self.latency_samples += 1
        self.latency_sum_ms += latency_ms
        self.avg_latency_ms = self.latency_sum_ms / self.latency_samples
        self.last_updated = time.time()
    
    def record_delivery(self, success: bool) -> None:
        """Record a message delivery outcome."""
        self.messages_sent += 1
        if success:
            self.messages_delivered += 1
        else:
            self.messages_dropped += 1
        self.last_updated = time.time()
    
    def record_ack(self, success: bool) -> None:
        """Record an ACK outcome."""
        if success:
            self.ack_success_count += 1
        else:
            self.ack_failure_count += 1
        self.last_updated = time.time()
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission/storage."""
        return {
            "router_id": self.router_id,
            "messages_sent": self.messages_sent,
            "messages_delivered": self.messages_delivered,
            "messages_dropped": self.messages_dropped,
            "avg_latency_ms": self.avg_latency_ms,
            "latency_samples": self.latency_samples,
            "latency_sum_ms": self.latency_sum_ms,
            "ack_success_count": self.ack_success_count,
            "ack_failure_count": self.ack_failure_count,
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
            "anomaly_score": self.anomaly_score,
            "flagged": self.flagged,
            "flag_reason": self.flag_reason,
            "delivery_rate": self.delivery_rate,
            "ack_success_rate": self.ack_success_rate,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RouterBehaviorMetrics":
        """Deserialize from dict."""
        metrics = cls(
            router_id=data.get("router_id", ""),
            messages_sent=data.get("messages_sent", 0),
            messages_delivered=data.get("messages_delivered", 0),
            messages_dropped=data.get("messages_dropped", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            latency_samples=data.get("latency_samples", 0),
            latency_sum_ms=data.get("latency_sum_ms", 0.0),
            ack_success_count=data.get("ack_success_count", 0),
            ack_failure_count=data.get("ack_failure_count", 0),
            first_seen=data.get("first_seen", time.time()),
            last_updated=data.get("last_updated", time.time()),
            anomaly_score=data.get("anomaly_score", 0.0),
            flagged=data.get("flagged", False),
            flag_reason=data.get("flag_reason", ""),
        )
        return metrics


@dataclass
class MisbehaviorEvidence:
    """
    Evidence of router misbehavior for a specific incident.
    
    Collected when a misbehavior is detected to support the report.
    """
    timestamp: float = field(default_factory=time.time)
    misbehavior_type: str = ""  # One of MisbehaviorType values
    message_id: Optional[str] = None  # Related message ID if applicable
    expected_latency_ms: float = 0.0  # Expected latency based on baseline
    actual_latency_ms: float = 0.0  # Actual observed latency
    delivery_rate_baseline: float = 0.0  # Network baseline delivery rate
    delivery_rate_observed: float = 0.0  # Observed delivery rate for this router
    description: str = ""  # Human-readable description
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "timestamp": self.timestamp,
            "misbehavior_type": self.misbehavior_type,
            "message_id": self.message_id,
            "expected_latency_ms": self.expected_latency_ms,
            "actual_latency_ms": self.actual_latency_ms,
            "delivery_rate_baseline": self.delivery_rate_baseline,
            "delivery_rate_observed": self.delivery_rate_observed,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MisbehaviorEvidence":
        """Deserialize from dict."""
        return cls(
            timestamp=data.get("timestamp", time.time()),
            misbehavior_type=data.get("misbehavior_type", ""),
            message_id=data.get("message_id"),
            expected_latency_ms=data.get("expected_latency_ms", 0.0),
            actual_latency_ms=data.get("actual_latency_ms", 0.0),
            delivery_rate_baseline=data.get("delivery_rate_baseline", 0.0),
            delivery_rate_observed=data.get("delivery_rate_observed", 0.0),
            description=data.get("description", ""),
        )


@dataclass
class MisbehaviorReport:
    """
    Report of router misbehavior sent to seed nodes.
    
    Nodes generate these reports when they detect routers exhibiting
    anomalous behavior (dropping messages, excessive delays, etc.).
    Seeds aggregate reports from multiple nodes to identify
    systematically misbehaving routers.
    
    Attributes:
        type: Always "misbehavior_report"
        report_id: Unique identifier for this report
        reporter_id: Node ID of the reporter
        router_id: ID of the misbehaving router
        misbehavior_type: Type of misbehavior detected
        evidence: List of evidence supporting the report
        metrics: Behavioral metrics for the router
        severity: Severity level (0.0 to 1.0, higher = more severe)
        timestamp: When the report was generated
        signature: Reporter's signature (hex) for verification
    """
    type: str = field(default="misbehavior_report", init=False)
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reporter_id: str = ""
    router_id: str = ""
    misbehavior_type: str = ""
    evidence: List[MisbehaviorEvidence] = field(default_factory=list)
    metrics: Optional[RouterBehaviorMetrics] = None
    severity: float = 0.0
    timestamp: float = field(default_factory=time.time)
    signature: str = ""
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "report_id": self.report_id,
            "reporter_id": self.reporter_id,
            "router_id": self.router_id,
            "misbehavior_type": self.misbehavior_type,
            "evidence": [e.to_dict() for e in self.evidence],
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "signature": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MisbehaviorReport":
        """Deserialize from dict."""
        evidence = [
            MisbehaviorEvidence.from_dict(e) 
            for e in data.get("evidence", [])
        ]
        metrics_data = data.get("metrics")
        metrics = RouterBehaviorMetrics.from_dict(metrics_data) if metrics_data else None
        
        report = cls(
            reporter_id=data.get("reporter_id", ""),
            router_id=data.get("router_id", ""),
            misbehavior_type=data.get("misbehavior_type", ""),
            evidence=evidence,
            metrics=metrics,
            severity=data.get("severity", 0.0),
            timestamp=data.get("timestamp", time.time()),
            signature=data.get("signature", ""),
        )
        report.report_id = data.get("report_id", report.report_id)
        return report
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "MisbehaviorReport":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class NetworkBaseline:
    """
    Network-wide baseline metrics for comparison.
    
    Used to determine if a router's behavior is anomalous by comparing
    against the network average.
    
    Attributes:
        avg_delivery_rate: Average delivery rate across all routers
        avg_latency_ms: Average latency across all routers
        avg_ack_success_rate: Average ACK success rate across all routers
        sample_count: Number of routers used to compute baseline
        last_updated: When the baseline was last computed
        delivery_rate_stddev: Standard deviation of delivery rates
        latency_stddev_ms: Standard deviation of latencies
    """
    avg_delivery_rate: float = 0.95
    avg_latency_ms: float = 100.0
    avg_ack_success_rate: float = 0.95
    sample_count: int = 0
    last_updated: float = field(default_factory=time.time)
    delivery_rate_stddev: float = 0.05
    latency_stddev_ms: float = 50.0
    
    def is_delivery_rate_anomalous(self, rate: float, threshold_stddevs: float = 2.0) -> bool:
        """Check if a delivery rate is anomalously low."""
        threshold = self.avg_delivery_rate - (threshold_stddevs * self.delivery_rate_stddev)
        return rate < threshold
    
    def is_latency_anomalous(self, latency_ms: float, threshold_stddevs: float = 2.0) -> bool:
        """Check if latency is anomalously high."""
        threshold = self.avg_latency_ms + (threshold_stddevs * self.latency_stddev_ms)
        return latency_ms > threshold
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "avg_delivery_rate": self.avg_delivery_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_ack_success_rate": self.avg_ack_success_rate,
            "sample_count": self.sample_count,
            "last_updated": self.last_updated,
            "delivery_rate_stddev": self.delivery_rate_stddev,
            "latency_stddev_ms": self.latency_stddev_ms,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NetworkBaseline":
        """Deserialize from dict."""
        return cls(
            avg_delivery_rate=data.get("avg_delivery_rate", 0.95),
            avg_latency_ms=data.get("avg_latency_ms", 100.0),
            avg_ack_success_rate=data.get("avg_ack_success_rate", 0.95),
            sample_count=data.get("sample_count", 0),
            last_updated=data.get("last_updated", time.time()),
            delivery_rate_stddev=data.get("delivery_rate_stddev", 0.05),
            latency_stddev_ms=data.get("latency_stddev_ms", 50.0),
        )


# =============================================================================
# SEED REVOCATION MESSAGES (Issue #121)
# =============================================================================


class RevocationReason:
    """Standard reasons for seed revocation."""
    KEY_COMPROMISE = "key_compromise"  # Private key was compromised
    MALICIOUS_BEHAVIOR = "malicious_behavior"  # Seed exhibited malicious behavior
    RETIRED = "retired"  # Seed is being retired from service
    ADMIN_ACTION = "admin_action"  # Administrative revocation
    SECURITY_AUDIT = "security_audit"  # Revoked due to security audit findings


@dataclass
class SeedRevocation:
    """
    Seed revocation message for revoking compromised or malicious seeds.
    
    When a seed is revoked, this message is broadcast via gossip to all
    other seeds. Nodes honor revocations by stopping trust of the revoked
    seed for discovery purposes.
    
    Security:
    - Must be signed by the seed being revoked (proving ownership) OR
    - Signed by a trusted authority (for cases where seed is compromised)
    - Includes timestamp to prevent replay attacks
    - Includes reason for audit trail
    
    Attributes:
        type: Always "seed_revocation"
        revocation_id: Unique identifier for this revocation
        seed_id: The seed being revoked
        reason: Reason for revocation (one of RevocationReason values)
        reason_detail: Optional detailed explanation
        timestamp: When the revocation was issued (UTC epoch)
        effective_at: When the revocation takes effect (UTC epoch, defaults to timestamp)
        issuer_id: ID of who issued the revocation (seed itself or authority)
        signature: Ed25519 signature proving authorization (hex-encoded)
    """
    type: str = field(default="seed_revocation", init=False)
    revocation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    seed_id: str = ""  # The seed being revoked
    reason: str = ""  # One of RevocationReason values
    reason_detail: str = ""  # Optional detailed explanation
    timestamp: float = field(default_factory=time.time)
    effective_at: float = 0.0  # When revocation takes effect (0 = immediate)
    issuer_id: str = ""  # Who issued this (the seed itself or trusted authority)
    signature: str = ""  # Ed25519 signature (hex)
    
    def __post_init__(self):
        if self.effective_at == 0.0:
            self.effective_at = self.timestamp
    
    @property
    def is_effective(self) -> bool:
        """Check if the revocation is currently effective."""
        return time.time() >= self.effective_at
    
    def get_signable_data(self) -> dict:
        """
        Get the data that should be signed.
        
        Excludes the signature field itself and includes all security-relevant
        fields in a canonical order.
        """
        return {
            "type": self.type,
            "revocation_id": self.revocation_id,
            "seed_id": self.seed_id,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "timestamp": self.timestamp,
            "effective_at": self.effective_at,
            "issuer_id": self.issuer_id,
        }
    
    def get_signable_bytes(self) -> bytes:
        """Get canonical bytes for signing."""
        return json.dumps(self.get_signable_data(), sort_keys=True, separators=(',', ':')).encode()
    
    def to_dict(self) -> dict:
        """Serialize to dict for transmission."""
        return {
            "type": self.type,
            "revocation_id": self.revocation_id,
            "seed_id": self.seed_id,
            "reason": self.reason,
            "reason_detail": self.reason_detail,
            "timestamp": self.timestamp,
            "effective_at": self.effective_at,
            "issuer_id": self.issuer_id,
            "signature": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SeedRevocation":
        """Deserialize from dict."""
        revocation = cls(
            seed_id=data.get("seed_id", ""),
            reason=data.get("reason", ""),
            reason_detail=data.get("reason_detail", ""),
            timestamp=data.get("timestamp", time.time()),
            effective_at=data.get("effective_at", 0.0),
            issuer_id=data.get("issuer_id", ""),
            signature=data.get("signature", ""),
        )
        revocation.revocation_id = data.get("revocation_id", revocation.revocation_id)
        return revocation
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "SeedRevocation":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class SeedRevocationList:
    """
    A signed list of seed revocations for out-of-band distribution.
    
    This allows nodes to receive revocations via file (e.g., fetched from
    a trusted URL) without requiring network connectivity to seeds.
    
    The list is signed by a trusted authority to prevent tampering.
    
    Attributes:
        version: List version (monotonically increasing)
        generated_at: When the list was generated
        revocations: List of SeedRevocation objects
        authority_id: ID of the signing authority
        signature: Ed25519 signature of the list (hex)
    """
    version: int = 1
    generated_at: float = field(default_factory=time.time)
    revocations: List[SeedRevocation] = field(default_factory=list)
    authority_id: str = ""
    signature: str = ""
    
    def get_signable_data(self) -> dict:
        """Get the data that should be signed."""
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "revocations": [r.to_dict() for r in self.revocations],
            "authority_id": self.authority_id,
        }
    
    def get_signable_bytes(self) -> bytes:
        """Get canonical bytes for signing."""
        return json.dumps(self.get_signable_data(), sort_keys=True, separators=(',', ':')).encode()
    
    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "revocations": [r.to_dict() for r in self.revocations],
            "authority_id": self.authority_id,
            "signature": self.signature,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SeedRevocationList":
        """Deserialize from dict."""
        revocations = [
            SeedRevocation.from_dict(r) for r in data.get("revocations", [])
        ]
        return cls(
            version=data.get("version", 1),
            generated_at=data.get("generated_at", time.time()),
            revocations=revocations,
            authority_id=data.get("authority_id", ""),
            signature=data.get("signature", ""),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "SeedRevocationList":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def get_revoked_seed_ids(self) -> set:
        """
        Get set of all revoked seed IDs that are currently effective.
        
        Returns:
            Set of seed IDs that are revoked
        """
        now = time.time()
        return {
            r.seed_id for r in self.revocations
            if r.effective_at <= now
        }
    
    def is_seed_revoked(self, seed_id: str) -> bool:
        """
        Check if a specific seed is revoked.
        
        Args:
            seed_id: The seed ID to check
            
        Returns:
            True if the seed is revoked and the revocation is effective
        """
        return seed_id in self.get_revoked_seed_ids()
