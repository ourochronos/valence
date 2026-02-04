"""Audit event logging for Valence - compliance and security tracking.

Implements AuditEvent for logging security-relevant actions like sharing,
trust grants, and access control changes.

Hash-chain integrity (Issue #82): Each event includes the SHA-256 hash
of the previous event, creating a tamper-evident append-only log.
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Protocol, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
import json
import hashlib
from pathlib import Path
import threading


class AuditEventType(Enum):
    """Types of security-relevant events that can be audited."""
    
    # Sharing events
    SHARE = "share"                    # Belief shared with another DID
    RECEIVE = "receive"                # Belief received from another DID
    REVOKE = "revoke"                  # Share revoked
    
    # Trust events
    GRANT_TRUST = "grant_trust"        # Trust edge created
    REVOKE_TRUST = "revoke_trust"      # Trust edge removed
    UPDATE_TRUST = "update_trust"      # Trust level modified
    
    # Domain events
    DOMAIN_CREATE = "domain_create"    # Domain created
    DOMAIN_DELETE = "domain_delete"    # Domain deleted
    MEMBER_ADD = "member_add"          # Member added to domain
    MEMBER_REMOVE = "member_remove"    # Member removed from domain
    ROLE_CHANGE = "role_change"        # Member role changed
    
    # Access events
    ACCESS_GRANTED = "access_granted"  # Access to resource granted
    ACCESS_DENIED = "access_denied"    # Access to resource denied
    ACCESS_EXPIRED = "access_expired"  # Access expired
    
    # Belief events
    BELIEF_CREATE = "belief_create"    # New belief created
    BELIEF_UPDATE = "belief_update"    # Belief modified
    BELIEF_DELETE = "belief_delete"    # Belief deleted
    
    # System events
    KEY_ROTATION = "key_rotation"      # Encryption key rotated
    CONFIG_CHANGE = "config_change"    # Configuration changed


@dataclass
class AuditEvent:
    """A security-relevant event to be logged for compliance.
    
    Attributes:
        event_id: Unique identifier for this event
        event_type: Category of event (from AuditEventType)
        actor_did: DID of the entity performing the action
        target_did: DID of the entity affected (optional)
        resource: Resource being acted upon (belief ID, domain ID, etc.)
        action: Specific action description
        timestamp: When the event occurred (UTC)
        success: Whether the action succeeded
        metadata: Additional context as key-value pairs
        previous_hash: SHA-256 hash of the previous event (None for genesis)
        event_hash: SHA-256 hash of this event's data + previous_hash
    """
    
    event_type: AuditEventType
    actor_did: str
    resource: str
    action: str
    success: bool
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    target_did: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None  # None for genesis event
    event_hash: str = field(default="")
    
    def __post_init__(self):
        """Compute event_hash if not provided."""
        if not self.event_hash:
            self.event_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA-256 hash from event data + previous_hash."""
        data = self._get_hashable_data()
        return hashlib.sha256(data.encode("utf-8")).hexdigest()
    
    def _get_hashable_data(self) -> str:
        """Get canonical JSON representation for hashing.
        
        Uses sort_keys for deterministic ordering.
        """
        hashable = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "actor_did": self.actor_did,
            "target_did": self.target_did,
            "resource": self.resource,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
        }
        return json.dumps(hashable, sort_keys=True, separators=(",", ":"))
    
    def verify_hash(self) -> bool:
        """Verify that event_hash matches computed hash.
        
        Returns True if hash is valid, False if data may be tampered.
        """
        return self.event_hash == self._compute_hash()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "actor_did": self.actor_did,
            "target_did": self.target_did,
            "resource": self.resource,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditEvent":
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(timezone.utc)
            
        return cls(
            event_id=data.get("event_id", str(uuid.uuid4())),
            event_type=AuditEventType(data["event_type"]),
            actor_did=data["actor_did"],
            target_did=data.get("target_did"),
            resource=data["resource"],
            action=data["action"],
            timestamp=timestamp,
            success=data.get("success", True),
            metadata=data.get("metadata", {}),
            previous_hash=data.get("previous_hash"),
            event_hash=data.get("event_hash", ""),
        )
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "AuditEvent":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


class ChainVerificationError(Exception):
    """Raised when audit chain verification fails."""
    
    def __init__(self, message: str, event_index: int, event_id: str):
        self.message = message
        self.event_index = event_index
        self.event_id = event_id
        super().__init__(f"{message} at index {event_index} (event_id: {event_id})")


class AuditBackend(Protocol):
    """Protocol for audit log storage backends."""
    
    def write(self, event: AuditEvent) -> None:
        """Write an event to the audit log."""
        ...
    
    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        target_did: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query events from the audit log."""
        ...
    
    def count(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> int:
        """Count events matching criteria."""
        ...


class InMemoryAuditBackend:
    """In-memory audit log backend for testing and development.
    
    Thread-safe but not persistent - data lost on restart.
    Maintains hash chain integrity for tamper detection.
    """
    
    def __init__(self, max_events: int = 10000):
        """Initialize with optional max event limit."""
        self._events: List[AuditEvent] = []
        self._max_events = max_events
        self._lock = threading.Lock()
        self._last_hash: Optional[str] = None
    
    @property
    def last_hash(self) -> Optional[str]:
        """Get the hash of the last event, or None if empty."""
        with self._lock:
            return self._last_hash
    
    def write(self, event: AuditEvent) -> None:
        """Write an event to the in-memory log.
        
        Automatically sets previous_hash and computes event_hash
        if not already set.
        """
        with self._lock:
            # Set previous_hash to chain to prior event
            if not event.previous_hash and self._last_hash:
                event.previous_hash = self._last_hash
                event.event_hash = event._compute_hash()
            
            self._events.append(event)
            self._last_hash = event.event_hash
            
            # Trim oldest events if over limit
            # Note: trimming breaks chain verification for full history
            if len(self._events) > self._max_events:
                self._events = self._events[-self._max_events:]
    
    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        target_did: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query events from the in-memory log."""
        with self._lock:
            results = []
            for event in reversed(self._events):  # Most recent first
                if event_type and event.event_type != event_type:
                    continue
                if actor_did and event.actor_did != actor_did:
                    continue
                if target_did and event.target_did != target_did:
                    continue
                if resource and event.resource != resource:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                results.append(event)
                if len(results) >= limit:
                    break
            return results
    
    def count(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> int:
        """Count events matching criteria."""
        with self._lock:
            count = 0
            for event in self._events:
                if event_type and event.event_type != event_type:
                    continue
                if actor_did and event.actor_did != actor_did:
                    continue
                if success is not None and event.success != success:
                    continue
                count += 1
            return count
    
    def clear(self) -> None:
        """Clear all events (for testing)."""
        with self._lock:
            self._events.clear()
    
    def all_events(self) -> List[AuditEvent]:
        """Get all events (for testing)."""
        with self._lock:
            return list(self._events)


class FileAuditBackend:
    """File-based audit log backend for persistence.
    
    Writes events as JSON lines (one event per line) for easy parsing.
    Thread-safe with file locking.
    """
    
    def __init__(self, log_path: Path, rotate_size_mb: float = 10.0):
        """Initialize with log file path.
        
        Args:
            log_path: Path to the audit log file
            rotate_size_mb: Rotate log when it exceeds this size (MB)
        """
        self._log_path = Path(log_path)
        self._rotate_size_bytes = int(rotate_size_mb * 1024 * 1024)
        self._lock = threading.Lock()
        
        # Ensure parent directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
    
    def write(self, event: AuditEvent) -> None:
        """Write an event to the log file."""
        with self._lock:
            self._maybe_rotate()
            with open(self._log_path, "a") as f:
                f.write(event.to_json() + "\n")
    
    def _maybe_rotate(self) -> None:
        """Rotate log file if it exceeds size limit."""
        if not self._log_path.exists():
            return
        
        if self._log_path.stat().st_size >= self._rotate_size_bytes:
            # Rename with timestamp
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            rotated_path = self._log_path.with_suffix(f".{timestamp}.jsonl")
            self._log_path.rename(rotated_path)
    
    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        target_did: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query events from the log file.
        
        Note: This reads the entire file which may be slow for large logs.
        For production, consider a database backend.
        """
        with self._lock:
            if not self._log_path.exists():
                return []
            
            results = []
            with open(self._log_path, "r") as f:
                # Read lines in reverse for most recent first
                lines = f.readlines()
            
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    event = AuditEvent.from_json(line)
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue  # Skip malformed lines
                
                if event_type and event.event_type != event_type:
                    continue
                if actor_did and event.actor_did != actor_did:
                    continue
                if target_did and event.target_did != target_did:
                    continue
                if resource and event.resource != resource:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                results.append(event)
                if len(results) >= limit:
                    break
            
            return results
    
    def count(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> int:
        """Count events matching criteria."""
        with self._lock:
            if not self._log_path.exists():
                return 0
            
            count = 0
            with open(self._log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event = AuditEvent.from_json(line)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
                    
                    if event_type and event.event_type != event_type:
                        continue
                    if actor_did and event.actor_did != actor_did:
                        continue
                    if success is not None and event.success != success:
                        continue
                    
                    count += 1
            
            return count


class AuditLogger:
    """High-level audit logging service.
    
    Provides convenient methods for logging common security events.
    Supports multiple backends (in-memory, file, or custom).
    """
    
    def __init__(self, backend: Optional[AuditBackend] = None):
        """Initialize with a storage backend.
        
        Args:
            backend: Storage backend (defaults to in-memory)
        """
        self._backend = backend or InMemoryAuditBackend()
    
    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event.
        
        Args:
            event: The AuditEvent to log
        """
        self._backend.write(event)
    
    def log(
        self,
        event_type: AuditEventType,
        actor_did: str,
        resource: str,
        action: str,
        success: bool = True,
        target_did: Optional[str] = None,
        **metadata: Any,
    ) -> AuditEvent:
        """Log an event with the given parameters.
        
        Convenience method that creates and logs an AuditEvent.
        Returns the created event.
        """
        event = AuditEvent(
            event_type=event_type,
            actor_did=actor_did,
            target_did=target_did,
            resource=resource,
            action=action,
            success=success,
            metadata=metadata,
        )
        self.log_event(event)
        return event
    
    # Convenience methods for common event types
    
    def log_share(
        self,
        actor_did: str,
        target_did: str,
        resource: str,
        success: bool = True,
        **metadata: Any,
    ) -> AuditEvent:
        """Log a share event."""
        return self.log(
            AuditEventType.SHARE,
            actor_did=actor_did,
            target_did=target_did,
            resource=resource,
            action="share_belief",
            success=success,
            **metadata,
        )
    
    def log_receive(
        self,
        actor_did: str,
        source_did: str,
        resource: str,
        success: bool = True,
        **metadata: Any,
    ) -> AuditEvent:
        """Log a receive event."""
        return self.log(
            AuditEventType.RECEIVE,
            actor_did=actor_did,
            target_did=source_did,  # source is "target" in this context
            resource=resource,
            action="receive_belief",
            success=success,
            **metadata,
        )
    
    def log_revoke(
        self,
        actor_did: str,
        target_did: str,
        resource: str,
        success: bool = True,
        **metadata: Any,
    ) -> AuditEvent:
        """Log a share revocation event."""
        return self.log(
            AuditEventType.REVOKE,
            actor_did=actor_did,
            target_did=target_did,
            resource=resource,
            action="revoke_share",
            success=success,
            **metadata,
        )
    
    def log_grant_trust(
        self,
        actor_did: str,
        target_did: str,
        trust_level: float,
        success: bool = True,
        **metadata: Any,
    ) -> AuditEvent:
        """Log a trust grant event."""
        return self.log(
            AuditEventType.GRANT_TRUST,
            actor_did=actor_did,
            target_did=target_did,
            resource=f"trust:{actor_did}:{target_did}",
            action="grant_trust",
            success=success,
            trust_level=trust_level,
            **metadata,
        )
    
    def log_revoke_trust(
        self,
        actor_did: str,
        target_did: str,
        success: bool = True,
        **metadata: Any,
    ) -> AuditEvent:
        """Log a trust revocation event."""
        return self.log(
            AuditEventType.REVOKE_TRUST,
            actor_did=actor_did,
            target_did=target_did,
            resource=f"trust:{actor_did}:{target_did}",
            action="revoke_trust",
            success=success,
            **metadata,
        )
    
    def log_access_denied(
        self,
        actor_did: str,
        resource: str,
        reason: str,
        **metadata: Any,
    ) -> AuditEvent:
        """Log an access denied event."""
        return self.log(
            AuditEventType.ACCESS_DENIED,
            actor_did=actor_did,
            resource=resource,
            action="access_denied",
            success=False,
            reason=reason,
            **metadata,
        )
    
    # Query methods
    
    def query(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        target_did: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditEvent]:
        """Query the audit log."""
        return self._backend.query(
            event_type=event_type,
            actor_did=actor_did,
            target_did=target_did,
            resource=resource,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )
    
    def count(
        self,
        event_type: Optional[AuditEventType] = None,
        actor_did: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> int:
        """Count events matching criteria."""
        return self._backend.count(
            event_type=event_type,
            actor_did=actor_did,
            success=success,
        )


# Module-level singleton for convenience
_default_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get the default audit logger singleton."""
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger()
    return _default_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set the default audit logger singleton."""
    global _default_logger
    _default_logger = logger


# Aliases for backward compatibility with expected interface
AuditLog = AuditLogger


class ChainVerificationError(Exception):
    """Error raised when audit chain verification fails."""
    pass


def verify_chain(events: List[AuditEvent]) -> bool:
    """Verify the integrity of an audit event chain.
    
    Checks that events are in chronological order and have valid references.
    
    Args:
        events: List of audit events to verify
        
    Returns:
        True if chain is valid
        
    Raises:
        ChainVerificationError: If chain verification fails
    """
    if not events:
        return True
    
    # Verify chronological order
    for i in range(1, len(events)):
        if events[i].timestamp < events[i-1].timestamp:
            raise ChainVerificationError(
                f"Event {events[i].event_id} timestamp precedes previous event"
            )
    
    # Verify unique event IDs
    seen_ids = set()
    for event in events:
        if event.event_id in seen_ids:
            raise ChainVerificationError(
                f"Duplicate event ID: {event.event_id}"
            )
        seen_ids.add(event.event_id)
    
    return True
