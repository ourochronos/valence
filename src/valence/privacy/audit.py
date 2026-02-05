"""Audit event logging for Valence - compliance and security tracking.

Implements AuditEvent for logging security-relevant actions like sharing,
trust grants, and access control changes.

Hash-chain integrity (Issue #82): Each event includes the SHA-256 hash
of the previous event, creating a tamper-evident append-only log.

Encrypted audit storage (Issue #83): EncryptedAuditBackend provides
AES-256-GCM encryption with envelope encryption and key rotation support.

PII sanitization (Issue #177): MetadataSanitizer provides automatic
scrubbing of sensitive data from audit event metadata before logging.
"""

import base64
import hashlib
import json
import re
import secrets
import threading
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, cast

# =============================================================================
# PII SANITIZATION (Issue #177)
# =============================================================================


# Default patterns for PII detection (can be extended via MetadataSanitizer)
DEFAULT_PII_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b"),
    "phone_us": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "phone_intl": re.compile(r"\+\d{1,3}[-.\s]?\d{1,14}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
    # DID values should generally be preserved (they're pseudonymous identifiers)
    # but we allow blocking specific patterns if needed
}

# Keys that should always be sanitized (case-insensitive matching)
DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    [
        "password",
        "secret",
        "token",
        "api_key",
        "apikey",
        "private_key",
        "privatekey",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "ssn",
        "social_security",
        "credit_card",
        "creditcard",
        "cvv",
        "pin",
        "bank_account",
        "routing_number",
        "tax_id",
    ]
)

# Placeholder for redacted values
REDACTED_PLACEHOLDER = "[REDACTED]"
REDACTED_PII_PLACEHOLDER = "[PII_REDACTED]"


@dataclass
class SanitizationResult:
    """Result of metadata sanitization with audit trail."""

    sanitized_metadata: dict[str, Any]
    fields_redacted: list[str]  # Keys that were fully redacted
    pii_scrubbed: list[str]  # Keys where PII patterns were replaced
    original_hash: str  # SHA-256 of original for forensic correlation

    @property
    def was_modified(self) -> bool:
        """Check if any sanitization occurred."""
        return bool(self.fields_redacted or self.pii_scrubbed)


class MetadataSanitizer:
    """Sanitizes potentially sensitive data from audit event metadata.

    Provides multiple sanitization strategies:
    - Key-based blocking: Redact entire values for sensitive key names
    - Pattern-based scrubbing: Replace PII patterns within values
    - Custom rules: User-defined sanitization functions

    Usage:
        sanitizer = MetadataSanitizer()
        result = sanitizer.sanitize({"email": "user@example.com", "action": "login"})
        # result.sanitized_metadata = {"email": "[PII_REDACTED]", "action": "login"}

    Warning: Sanitization is best-effort. For maximum security, avoid logging
    PII in metadata in the first place. This layer provides defense-in-depth.
    """

    def __init__(
        self,
        sensitive_keys: frozenset[str] | None = None,
        pii_patterns: dict[str, re.Pattern] | None = None,
        preserve_keys: frozenset[str] | None = None,
        custom_sanitizers: dict[str, Callable[[Any], Any]] | None = None,
        enabled: bool = True,
    ):
        """Initialize the sanitizer.

        Args:
            sensitive_keys: Keys whose values should be fully redacted
            pii_patterns: Regex patterns for PII detection
            preserve_keys: Keys that should never be sanitized (whitelisted)
            custom_sanitizers: Custom sanitization functions per key
            enabled: Whether sanitization is active (for testing/debugging)
        """
        self.sensitive_keys = sensitive_keys or DEFAULT_SENSITIVE_KEYS
        self.pii_patterns = pii_patterns or DEFAULT_PII_PATTERNS
        self.preserve_keys = preserve_keys or frozenset()
        self.custom_sanitizers = custom_sanitizers or {}
        self.enabled = enabled

    def sanitize(
        self,
        metadata: dict[str, Any],
        preserve_original_hash: bool = True,
    ) -> SanitizationResult:
        """Sanitize metadata dictionary.

        Args:
            metadata: Original metadata to sanitize
            preserve_original_hash: Include hash of original for forensic use

        Returns:
            SanitizationResult with sanitized data and audit info
        """
        if not self.enabled or not metadata:
            return SanitizationResult(
                sanitized_metadata=metadata.copy() if metadata else {},
                fields_redacted=[],
                pii_scrubbed=[],
                original_hash=(self._compute_hash(metadata) if preserve_original_hash else ""),
            )

        original_hash = self._compute_hash(metadata) if preserve_original_hash else ""
        sanitized: dict[str, Any] = {}
        fields_redacted: list[str] = []
        pii_scrubbed: list[str] = []

        for key, value in metadata.items():
            # Check preserve list first
            if key.lower() in self.preserve_keys:
                sanitized[key] = value
                continue

            # Apply custom sanitizer if available
            if key in self.custom_sanitizers:
                sanitized[key] = self.custom_sanitizers[key](value)
                continue

            # Check if key is in sensitive list
            if self._is_sensitive_key(key):
                sanitized[key] = REDACTED_PLACEHOLDER
                fields_redacted.append(key)
                continue

            # Scrub PII from string values
            if isinstance(value, str):
                scrubbed_value, had_pii = self._scrub_pii(value)
                sanitized[key] = scrubbed_value
                if had_pii:
                    pii_scrubbed.append(key)
            elif isinstance(value, dict):
                # Recursively sanitize nested dicts
                nested_result = self.sanitize(value, preserve_original_hash=False)
                sanitized[key] = nested_result.sanitized_metadata
                fields_redacted.extend(f"{key}.{f}" for f in nested_result.fields_redacted)
                pii_scrubbed.extend(f"{key}.{f}" for f in nested_result.pii_scrubbed)
            elif isinstance(value, list):
                # Sanitize list elements if they're strings
                sanitized[key] = [self._scrub_pii(v)[0] if isinstance(v, str) else v for v in value]
            else:
                sanitized[key] = value

        return SanitizationResult(
            sanitized_metadata=sanitized,
            fields_redacted=fields_redacted,
            pii_scrubbed=pii_scrubbed,
            original_hash=original_hash,
        )

    def _is_sensitive_key(self, key: str) -> bool:
        """Check if a key name indicates sensitive content."""
        key_lower = key.lower().replace("-", "_")
        return any(sensitive in key_lower for sensitive in self.sensitive_keys)

    def _scrub_pii(self, value: str) -> tuple[str, bool]:
        """Scrub PII patterns from a string value.

        Returns:
            Tuple of (scrubbed_value, had_pii)
        """
        had_pii = False
        result = value

        for pattern_name, pattern in self.pii_patterns.items():
            if pattern.search(result):
                result = pattern.sub(REDACTED_PII_PLACEHOLDER, result)
                had_pii = True

        return result, had_pii

    def _compute_hash(self, data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of metadata for forensic correlation."""
        try:
            json_str = json.dumps(data, sort_keys=True, default=str)
            return hashlib.sha256(json_str.encode()).hexdigest()[:16]
        except (TypeError, ValueError):
            return ""

    def add_sensitive_key(self, key: str) -> None:
        """Add a key to the sensitive keys list."""
        self.sensitive_keys = self.sensitive_keys | {key.lower()}

    def add_pii_pattern(self, name: str, pattern: re.Pattern) -> None:
        """Add a custom PII pattern."""
        self.pii_patterns[name] = pattern


# Global sanitizer instance (can be configured at startup)
_default_sanitizer: MetadataSanitizer | None = None
_sanitizer_lock = threading.Lock()


def get_metadata_sanitizer() -> MetadataSanitizer:
    """Get the default metadata sanitizer singleton."""
    global _default_sanitizer
    if _default_sanitizer is None:
        with _sanitizer_lock:
            if _default_sanitizer is None:
                _default_sanitizer = MetadataSanitizer()
    return _default_sanitizer


def set_metadata_sanitizer(sanitizer: MetadataSanitizer) -> None:
    """Set the default metadata sanitizer."""
    global _default_sanitizer
    with _sanitizer_lock:
        _default_sanitizer = sanitizer


def sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convenience function to sanitize metadata using the default sanitizer.

    This is the primary entry point for sanitizing audit metadata.

    Args:
        metadata: Metadata dict to sanitize

    Returns:
        Sanitized metadata dict
    """
    return get_metadata_sanitizer().sanitize(metadata).sanitized_metadata


class AuditEventType(Enum):
    """Types of security-relevant events that can be audited."""

    # Sharing events
    SHARE = "share"  # Belief shared with another DID
    RECEIVE = "receive"  # Belief received from another DID
    REVOKE = "revoke"  # Share revoked

    # Trust events
    GRANT_TRUST = "grant_trust"  # Trust edge created
    REVOKE_TRUST = "revoke_trust"  # Trust edge removed
    UPDATE_TRUST = "update_trust"  # Trust level modified

    # Domain events
    DOMAIN_CREATE = "domain_create"  # Domain created
    DOMAIN_DELETE = "domain_delete"  # Domain deleted
    MEMBER_ADD = "member_add"  # Member added to domain
    MEMBER_REMOVE = "member_remove"  # Member removed from domain
    ROLE_CHANGE = "role_change"  # Member role changed

    # Access events
    ACCESS_GRANTED = "access_granted"  # Access to resource granted
    ACCESS_DENIED = "access_denied"  # Access to resource denied
    ACCESS_EXPIRED = "access_expired"  # Access expired

    # Belief events
    BELIEF_CREATE = "belief_create"  # New belief created
    BELIEF_UPDATE = "belief_update"  # Belief modified
    BELIEF_DELETE = "belief_delete"  # Belief deleted

    # System events
    KEY_ROTATION = "key_rotation"  # Encryption key rotated
    CONFIG_CHANGE = "config_change"  # Configuration changed


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
    target_did: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    previous_hash: str | None = None  # None for genesis event
    event_hash: str = field(default="")

    def __post_init__(self):
        """Compute event_hash if not provided and sanitize metadata."""
        # Sanitize metadata to prevent PII leakage (Issue #177)
        if self.metadata:
            self.metadata = sanitize_metadata(self.metadata)

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

    def to_dict(self) -> dict[str, Any]:
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
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        """Deserialize from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now(UTC)

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
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        target_did: str | None = None,
        resource: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query events from the audit log."""
        ...

    def count(
        self,
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        success: bool | None = None,
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
        self._events: list[AuditEvent] = []
        self._max_events = max_events
        self._lock = threading.Lock()
        self._last_hash: str | None = None

    @property
    def last_hash(self) -> str | None:
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
                self._events = self._events[-self._max_events :]

    def query(
        self,
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        target_did: str | None = None,
        resource: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
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
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        success: bool | None = None,
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
            self._last_hash = None

    def all_events(self) -> list[AuditEvent]:
        """Get all events (for testing)."""
        with self._lock:
            return list(self._events)

    def verify_chain(self) -> tuple[bool, ChainVerificationError | None]:
        """Verify the integrity of the entire audit chain.

        Checks:
        1. Genesis event has null previous_hash
        2. Each event's hash is correctly computed
        3. Each event's previous_hash matches the prior event's hash

        Returns:
            Tuple of (is_valid, error_or_none)
        """
        with self._lock:
            return _verify_event_chain(self._events)


class FileAuditBackend:
    """File-based audit log backend for persistence.

    Writes events as JSON lines (one event per line) for easy parsing.
    Thread-safe with file locking.
    Maintains hash chain integrity for tamper detection.
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
        self._last_hash: str | None = None

        # Ensure parent directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        # Load last hash from existing log
        self._load_last_hash()

    def _load_last_hash(self) -> None:
        """Load the last event's hash from the log file."""
        if not self._log_path.exists():
            return

        with open(self._log_path) as f:
            lines = f.readlines()

        # Find last valid event
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                event = AuditEvent.from_json(line)
                self._last_hash = event.event_hash
                return
            except (json.JSONDecodeError, KeyError, ValueError):
                continue

    @property
    def last_hash(self) -> str | None:
        """Get the hash of the last event, or None if empty."""
        with self._lock:
            return self._last_hash

    def write(self, event: AuditEvent) -> None:
        """Write an event to the log file.

        Automatically sets previous_hash and computes event_hash
        if not already set.
        """
        with self._lock:
            self._maybe_rotate()

            # Set previous_hash to chain to prior event
            if not event.previous_hash and self._last_hash:
                event.previous_hash = self._last_hash
                event.event_hash = event._compute_hash()

            with open(self._log_path, "a") as f:
                f.write(event.to_json() + "\n")

            self._last_hash = event.event_hash

    def _maybe_rotate(self) -> None:
        """Rotate log file if it exceeds size limit."""
        if not self._log_path.exists():
            return

        if self._log_path.stat().st_size >= self._rotate_size_bytes:
            # Rename with timestamp
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            rotated_path = self._log_path.with_suffix(f".{timestamp}.jsonl")
            self._log_path.rename(rotated_path)

    def query(
        self,
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        target_did: str | None = None,
        resource: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query events from the log file.

        Note: This reads the entire file which may be slow for large logs.
        For production, consider a database backend.
        """
        with self._lock:
            if not self._log_path.exists():
                return []

            results = []
            with open(self._log_path) as f:
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
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        success: bool | None = None,
    ) -> int:
        """Count events matching criteria."""
        with self._lock:
            if not self._log_path.exists():
                return 0

            count = 0
            with open(self._log_path) as f:
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

    def all_events(self) -> list[AuditEvent]:
        """Get all events from the log file."""
        with self._lock:
            if not self._log_path.exists():
                return []

            events = []
            with open(self._log_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(AuditEvent.from_json(line))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            return events

    def verify_chain(self) -> tuple[bool, ChainVerificationError | None]:
        """Verify the integrity of the entire audit chain.

        Checks:
        1. Genesis event has null previous_hash
        2. Each event's hash is correctly computed
        3. Each event's previous_hash matches the prior event's hash

        Returns:
            Tuple of (is_valid, error_or_none)
        """
        with self._lock:
            events = []
            if self._log_path.exists():
                with open(self._log_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            events.append(AuditEvent.from_json(line))
                        except (json.JSONDecodeError, KeyError, ValueError):
                            continue
            return _verify_event_chain(events)


class AuditLogger:
    """High-level audit logging service.

    Provides convenient methods for logging common security events.
    Supports multiple backends (in-memory, file, or custom).
    """

    def __init__(self, backend: AuditBackend | None = None):
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
        target_did: str | None = None,
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
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        target_did: str | None = None,
        resource: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
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
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        success: bool | None = None,
    ) -> int:
        """Count events matching criteria."""
        return self._backend.count(
            event_type=event_type,
            actor_did=actor_did,
            success=success,
        )


# Module-level singleton for convenience
_default_logger: AuditLogger | None = None
_default_logger_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    """Get the default audit logger singleton.

    Thread-safe initialization using double-checked locking pattern.
    """
    global _default_logger
    if _default_logger is None:
        with _default_logger_lock:
            # Double-check after acquiring lock
            if _default_logger is None:
                _default_logger = AuditLogger()
    return _default_logger


def set_audit_logger(logger: AuditLogger) -> None:
    """Set the default audit logger singleton.

    Thread-safe setter using lock.
    """
    global _default_logger
    with _default_logger_lock:
        _default_logger = logger


# Aliases for backward compatibility with expected interface
AuditLog = AuditLogger


def _verify_event_chain(
    events: list[AuditEvent],
) -> tuple[bool, ChainVerificationError | None]:
    """Internal helper to verify hash chain integrity.

    Checks:
    1. Genesis event has null previous_hash
    2. Each event's hash is correctly computed
    3. Each event's previous_hash matches the prior event's hash

    Returns:
        Tuple of (is_valid, error_or_none)
    """
    if not events:
        return True, None

    for i, event in enumerate(events):
        # Check genesis event
        if i == 0:
            if event.previous_hash is not None:
                error = ChainVerificationError(
                    "Genesis event must have null previous_hash",
                    i,
                    event.event_id,
                )
                return False, error
        else:
            # Check chain linkage
            expected_prev_hash = events[i - 1].event_hash
            if event.previous_hash != expected_prev_hash:
                error = ChainVerificationError(
                    f"Chain broken: previous_hash mismatch (expected {expected_prev_hash[:16]}..., got {event.previous_hash[:16] if event.previous_hash else 'None'}...)",
                    i,
                    event.event_id,
                )
                return False, error

        # Verify event's own hash
        if not event.verify_hash():
            error = ChainVerificationError(
                "Event hash verification failed (data may be tampered)",
                i,
                event.event_id,
            )
            return False, error

    return True, None


def verify_chain(
    events: list[AuditEvent],
) -> tuple[bool, ChainVerificationError | None]:
    """Verify the integrity of an audit event hash chain.

    Standalone function to verify events loaded from external sources.

    Checks:
    1. Genesis event has null previous_hash
    2. Each event's hash is correctly computed (SHA-256)
    3. Each event's previous_hash matches the prior event's hash

    Args:
        events: List of audit events to verify

    Returns:
        Tuple of (is_valid, error_or_none)
    """
    return _verify_event_chain(events)


# =============================================================================
# Encrypted Audit Storage (Issue #83)
# =============================================================================


class KeyProvider(ABC):
    """Abstract base class for encryption key providers.

    Supports key rotation by managing multiple keys identified by key_id.
    """

    @abstractmethod
    def get_current_key_id(self) -> str:
        """Get the ID of the current active key for encryption."""
        ...

    @abstractmethod
    def get_key(self, key_id: str) -> bytes:
        """Get the key bytes for a given key ID.

        Args:
            key_id: The identifier for the key

        Returns:
            32-byte key for AES-256

        Raises:
            KeyError: If key_id is not found
        """
        ...

    @abstractmethod
    def get_current_key(self) -> tuple[str, bytes]:
        """Get the current key ID and bytes for encryption.

        Returns:
            Tuple of (key_id, key_bytes)
        """
        ...


class StaticKeyProvider(KeyProvider):
    """Simple key provider with a single static key.

    Useful for testing and simple deployments. For production,
    use a key management system (KMS) backed provider.
    """

    def __init__(self, key: bytes, key_id: str = "default"):
        """Initialize with a static key.

        Args:
            key: 32-byte key for AES-256
            key_id: Identifier for this key
        """
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self._key = key
        self._key_id = key_id
        self._keys = {key_id: key}

    def get_current_key_id(self) -> str:
        return self._key_id

    def get_key(self, key_id: str) -> bytes:
        if key_id not in self._keys:
            raise KeyError(f"Unknown key ID: {key_id}")
        return self._keys[key_id]

    def get_current_key(self) -> tuple[str, bytes]:
        return self._key_id, self._key

    def add_key(self, key: bytes, key_id: str) -> None:
        """Add an additional key (for testing key rotation).

        Args:
            key: 32-byte key for AES-256
            key_id: Identifier for this key
        """
        if len(key) != 32:
            raise ValueError("Key must be 32 bytes for AES-256")
        self._keys[key_id] = key

    def rotate_to(self, key_id: str) -> None:
        """Rotate to a different key as the current key.

        Args:
            key_id: The key ID to rotate to (must exist)
        """
        if key_id not in self._keys:
            raise KeyError(f"Unknown key ID: {key_id}")
        self._key_id = key_id
        self._key = self._keys[key_id]


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""

    pass


class DecryptionError(Exception):
    """Raised when decryption fails (wrong key, corrupted data, etc.)."""

    pass


@dataclass
class EncryptedEnvelope:
    """Encrypted data envelope with metadata for decryption.

    Uses envelope encryption: data is encrypted with a random DEK (Data Encryption Key),
    and the DEK is encrypted with the master key (KEK - Key Encryption Key).

    Attributes:
        key_id: ID of the master key used to encrypt the DEK
        encrypted_dek: DEK encrypted with master key (base64)
        dek_nonce: Nonce used for DEK encryption (base64)
        ciphertext: Data encrypted with DEK (base64)
        data_nonce: Nonce used for data encryption (base64)
        tag: Authentication tag for data encryption (base64)
    """

    key_id: str
    encrypted_dek: str  # base64
    dek_nonce: str  # base64
    ciphertext: str  # base64
    data_nonce: str  # base64

    def to_dict(self) -> dict[str, str]:
        """Serialize to dictionary."""
        return {
            "key_id": self.key_id,
            "encrypted_dek": self.encrypted_dek,
            "dek_nonce": self.dek_nonce,
            "ciphertext": self.ciphertext,
            "data_nonce": self.data_nonce,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "EncryptedEnvelope":
        """Deserialize from dictionary."""
        return cls(
            key_id=data["key_id"],
            encrypted_dek=data["encrypted_dek"],
            dek_nonce=data["dek_nonce"],
            ciphertext=data["ciphertext"],
            data_nonce=data["data_nonce"],
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "EncryptedEnvelope":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))


def _aes_gcm_encrypt(key: bytes, plaintext: bytes, nonce: bytes | None = None) -> tuple[bytes, bytes, bytes]:
    """Encrypt data using AES-256-GCM.

    Args:
        key: 32-byte AES key
        plaintext: Data to encrypt
        nonce: 12-byte nonce (generated if not provided)

    Returns:
        Tuple of (ciphertext_with_tag, nonce, tag)
        Note: ciphertext_with_tag includes the 16-byte auth tag appended
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError("cryptography package required for encryption. Install with: pip install cryptography")

    if nonce is None:
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM

    aesgcm = AESGCM(key)
    # AESGCM.encrypt returns ciphertext + tag concatenated
    ciphertext_with_tag = aesgcm.encrypt(nonce, plaintext, None)

    # Extract tag (last 16 bytes)
    tag = ciphertext_with_tag[-16:]

    return ciphertext_with_tag, nonce, tag


def _aes_gcm_decrypt(key: bytes, ciphertext_with_tag: bytes, nonce: bytes) -> bytes:
    """Decrypt data using AES-256-GCM.

    Args:
        key: 32-byte AES key
        ciphertext_with_tag: Encrypted data with auth tag appended
        nonce: 12-byte nonce used during encryption

    Returns:
        Decrypted plaintext

    Raises:
        DecryptionError: If decryption fails (wrong key, tampered data, etc.)
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise ImportError("cryptography package required for encryption. Install with: pip install cryptography")

    try:
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext_with_tag, None)
    except Exception as e:
        raise DecryptionError(f"Decryption failed: {e}")


def encrypt_envelope(plaintext: bytes, key_provider: KeyProvider) -> EncryptedEnvelope:
    """Encrypt data using envelope encryption.

    1. Generate a random DEK (Data Encryption Key)
    2. Encrypt the data with the DEK using AES-256-GCM
    3. Encrypt the DEK with the master key (KEK) using AES-256-GCM
    4. Return the envelope containing encrypted DEK and encrypted data

    Args:
        plaintext: Data to encrypt
        key_provider: Provider for the master key

    Returns:
        EncryptedEnvelope containing all data needed for decryption
    """
    # Get the current master key
    key_id, master_key = key_provider.get_current_key()

    # Generate random DEK
    dek = secrets.token_bytes(32)  # 256-bit DEK

    # Encrypt data with DEK
    data_ciphertext, data_nonce, _ = _aes_gcm_encrypt(dek, plaintext)

    # Encrypt DEK with master key
    dek_ciphertext, dek_nonce, _ = _aes_gcm_encrypt(master_key, dek)

    return EncryptedEnvelope(
        key_id=key_id,
        encrypted_dek=base64.b64encode(dek_ciphertext).decode("ascii"),
        dek_nonce=base64.b64encode(dek_nonce).decode("ascii"),
        ciphertext=base64.b64encode(data_ciphertext).decode("ascii"),
        data_nonce=base64.b64encode(data_nonce).decode("ascii"),
    )


def decrypt_envelope(envelope: EncryptedEnvelope, key_provider: KeyProvider) -> bytes:
    """Decrypt data from an encrypted envelope.

    1. Look up the master key by key_id
    2. Decrypt the DEK using the master key
    3. Decrypt the data using the DEK

    Args:
        envelope: The encrypted envelope
        key_provider: Provider to look up keys

    Returns:
        Decrypted plaintext

    Raises:
        KeyError: If the key_id is not found
        DecryptionError: If decryption fails
    """
    # Get the master key for this envelope
    try:
        master_key = key_provider.get_key(envelope.key_id)
    except KeyError:
        raise KeyError(f"Key not found: {envelope.key_id}")

    # Decode base64 values
    encrypted_dek = base64.b64decode(envelope.encrypted_dek)
    dek_nonce = base64.b64decode(envelope.dek_nonce)
    ciphertext = base64.b64decode(envelope.ciphertext)
    data_nonce = base64.b64decode(envelope.data_nonce)

    # Decrypt DEK
    dek = _aes_gcm_decrypt(master_key, encrypted_dek, dek_nonce)

    # Decrypt data
    return _aes_gcm_decrypt(dek, ciphertext, data_nonce)


class EncryptedAuditBackend:
    """Audit backend wrapper that encrypts events at rest.

    Wraps any AuditBackend to provide transparent encryption/decryption.
    Uses envelope encryption with AES-256-GCM for authenticated encryption.

    Features:
    - Encrypts events on write, decrypts on read
    - Envelope encryption: random DEK per event, DEK encrypted by master key
    - Key rotation support: key_id stored with each event
    - Works with any AuditBackend (InMemoryAuditBackend, FileAuditBackend, etc.)

    Example:
        key = secrets.token_bytes(32)
        key_provider = StaticKeyProvider(key)
        inner_backend = FileAuditBackend(Path("/var/log/audit.jsonl"))
        encrypted_backend = EncryptedAuditBackend(inner_backend, key_provider)
        logger = AuditLogger(encrypted_backend)
    """

    # Marker to identify encrypted events
    ENCRYPTED_MARKER = "__encrypted__"

    def __init__(self, backend: AuditBackend, key_provider: KeyProvider):
        """Initialize with a backend and key provider.

        Args:
            backend: The underlying backend for storage
            key_provider: Provider for encryption keys
        """
        self._backend = backend
        self._key_provider = key_provider
        self._lock = threading.Lock()

    def _encrypt_event(self, event: AuditEvent, previous_hash: str | None = None) -> AuditEvent:
        """Encrypt an event's sensitive data.

        Creates a new event with the same ID but encrypted content.
        The event_type, event_id, timestamp are preserved for indexing.
        Hash chain is computed on the encrypted data for tamper detection.

        Args:
            event: The original event to encrypt
            previous_hash: Hash of previous encrypted event (for chain)
        """
        # Serialize the full event
        event_json = event.to_json()

        # Encrypt the serialized event
        envelope = encrypt_envelope(event_json.encode("utf-8"), self._key_provider)

        # Create a wrapper event that stores the encrypted envelope
        # Hash will be computed fresh on encrypted content
        encrypted_event = AuditEvent(
            event_id=event.event_id,
            event_type=event.event_type,
            actor_did=self.ENCRYPTED_MARKER,  # Marker for encrypted
            target_did=None,
            resource=self.ENCRYPTED_MARKER,
            action=self.ENCRYPTED_MARKER,
            timestamp=event.timestamp,
            success=event.success,
            metadata={
                self.ENCRYPTED_MARKER: True,
                "envelope": envelope.to_dict(),
            },
            previous_hash=previous_hash,
            # event_hash will be auto-computed in __post_init__
        )

        return encrypted_event

    def _decrypt_event(self, encrypted_event: AuditEvent) -> AuditEvent:
        """Decrypt an encrypted event.

        Returns the original event with all data restored.
        """
        # Check if this is actually encrypted
        if not self._is_encrypted(encrypted_event):
            return encrypted_event

        # Extract envelope
        envelope_data = encrypted_event.metadata.get("envelope")
        if not envelope_data:
            raise DecryptionError("Missing envelope in encrypted event")

        envelope = EncryptedEnvelope.from_dict(envelope_data)

        # Decrypt
        plaintext = decrypt_envelope(envelope, self._key_provider)

        # Deserialize
        return AuditEvent.from_json(plaintext.decode("utf-8"))

    def _is_encrypted(self, event: AuditEvent) -> bool:
        """Check if an event is encrypted."""
        return event.metadata.get(self.ENCRYPTED_MARKER, False)

    @property
    def last_hash(self) -> str | None:
        """Get the hash of the last event, or None if empty."""
        if hasattr(self._backend, "last_hash"):
            return self._backend.last_hash
        return None

    def write(self, event: AuditEvent) -> None:
        """Write an encrypted event to the backend."""
        with self._lock:
            # Get previous hash for chain linking
            previous_hash = None
            if hasattr(self._backend, "last_hash"):
                previous_hash = self._backend.last_hash

            # Encrypt with chain info (hash computed on encrypted content)
            encrypted_event = self._encrypt_event(event, previous_hash)

            # Write to backend (skip its chain linking since we handled it)
            # We need to write directly to avoid double-chaining
            # Cast to Any for internal attribute access (hasattr guards ensure safety)
            backend_any = cast(Any, self._backend)
            if hasattr(self._backend, "_events"):
                # InMemoryAuditBackend
                with backend_any._lock:
                    backend_any._events.append(encrypted_event)
                    backend_any._last_hash = encrypted_event.event_hash
            elif hasattr(self._backend, "_log_path"):
                # FileAuditBackend
                with backend_any._lock:
                    backend_any._maybe_rotate()
                    with open(backend_any._log_path, "a") as f:
                        f.write(encrypted_event.to_json() + "\n")
                    backend_any._last_hash = encrypted_event.event_hash
            else:
                # Generic backend - just call write (may double-chain)
                self._backend.write(encrypted_event)

    def query(
        self,
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        target_did: str | None = None,
        resource: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Query events from the backend, decrypting results.

        Note: Filtering by actor_did, target_did, and resource is performed
        AFTER decryption since these fields are encrypted. This may be
        slower than unencrypted backends for large datasets.
        """
        # Query with relaxed filters (encrypted fields won't match)
        encrypted_events = self._backend.query(
            event_type=event_type,
            actor_did=None,  # Can't filter encrypted field
            target_did=None,
            resource=None,
            start_time=start_time,
            end_time=end_time,
            limit=limit * 10 if (actor_did or target_did or resource) else limit,
        )

        # Decrypt and filter
        results = []
        for encrypted_event in encrypted_events:
            try:
                event = self._decrypt_event(encrypted_event)
            except (DecryptionError, KeyError):
                # Skip events we can't decrypt (e.g., old keys removed)
                continue

            # Apply filters that couldn't be done on encrypted data
            if actor_did and event.actor_did != actor_did:
                continue
            if target_did and event.target_did != target_did:
                continue
            if resource and event.resource != resource:
                continue

            results.append(event)
            if len(results) >= limit:
                break

        return results

    def count(
        self,
        event_type: AuditEventType | None = None,
        actor_did: str | None = None,
        success: bool | None = None,
    ) -> int:
        """Count events matching criteria.

        Note: Filtering by actor_did requires decryption and is slow.
        """
        if actor_did:
            # Need to decrypt to filter by actor
            events = self.query(event_type=event_type, limit=100000)
            return sum(1 for e in events if e.actor_did == actor_did and (success is None or e.success == success))

        # Can use backend directly for other filters
        return self._backend.count(event_type=event_type, success=success)

    def all_events(self) -> list[AuditEvent]:
        """Get all events, decrypted."""
        if hasattr(self._backend, "all_events"):
            encrypted_events = self._backend.all_events()
            results = []
            for encrypted_event in encrypted_events:
                try:
                    results.append(self._decrypt_event(encrypted_event))
                except (DecryptionError, KeyError) as e:
                    # Log error for debugging, but continue for resilience
                    import logging

                    logging.warning(f"Failed to decrypt event {encrypted_event.event_id}: {e}")
                    continue
            return results
        return self.query(limit=100000)

    def verify_chain(self) -> tuple[bool, ChainVerificationError | None]:
        """Verify the integrity of the audit chain.

        Verifies the encrypted events' chain (which preserves hashes).
        """
        if hasattr(self._backend, "verify_chain"):
            return self._backend.verify_chain()
        return True, None

    def clear(self) -> None:
        """Clear all events (for testing)."""
        if hasattr(self._backend, "clear"):
            self._backend.clear()
