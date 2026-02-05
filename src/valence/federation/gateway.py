"""Gateway nodes for external federation sharing.

Gateway nodes bridge federations for controlled external sharing, providing:
- Inbound: receive external shares, validate, route to members
- Outbound: send shares to external federations via their gateways
- Rate limiting and access control
- Audit logging of all gateway traffic

Issue #88: Implement gateway nodes for external sharing
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
from uuid import UUID, uuid4

from ..core.db import get_cursor
from ..core.exceptions import ValenceException, ValidationException
from .models import (
    FederationNode,
    FederatedBelief,
    BeliefProvenance,
    NodeTrust,
    Visibility,
    ShareLevel,
    NodeStatus,
)
from .protocol import ErrorCode


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Rate limiting defaults
DEFAULT_RATE_LIMIT_WINDOW = 60  # seconds
DEFAULT_RATE_LIMIT_MAX_REQUESTS = 100  # per window
DEFAULT_RATE_LIMIT_MAX_BELIEFS = 1000  # beliefs per window

# Access control
MIN_TRUST_FOR_GATEWAY = 0.3  # Minimum trust to use gateway
DEFAULT_MAX_BELIEF_SIZE = 65536  # 64KB max belief content

# Audit log retention
DEFAULT_AUDIT_RETENTION_DAYS = 90


# =============================================================================
# ENUMS
# =============================================================================


class GatewayCapability(str, Enum):
    """Capabilities a gateway can support."""
    
    INBOUND_SHARE = "inbound_share"  # Accept external shares
    OUTBOUND_SHARE = "outbound_share"  # Send to external federations
    BELIEF_RELAY = "belief_relay"  # Relay beliefs between federations
    TRUST_BRIDGE = "trust_bridge"  # Bridge trust attestations
    PRIVACY_AGGREGATE = "privacy_aggregate"  # Privacy-preserving aggregation
    QUERY_FORWARD = "query_forward"  # Forward queries to external federations


class GatewayStatus(str, Enum):
    """Status of a gateway node."""
    
    INITIALIZING = "initializing"
    ACTIVE = "active"
    DEGRADED = "degraded"  # Partial functionality
    RATE_LIMITED = "rate_limited"
    SUSPENDED = "suspended"
    OFFLINE = "offline"


class AuditEventType(str, Enum):
    """Types of gateway audit events."""
    
    INBOUND_SHARE = "inbound_share"
    OUTBOUND_SHARE = "outbound_share"
    ACCESS_DENIED = "access_denied"
    RATE_LIMITED = "rate_limited"
    VALIDATION_FAILED = "validation_failed"
    ROUTE_SUCCESS = "route_success"
    ROUTE_FAILED = "route_failed"
    TRUST_CHECK = "trust_check"
    CONFIG_CHANGE = "config_change"


class ShareDirection(str, Enum):
    """Direction of a share operation."""
    
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class ValidationResult(str, Enum):
    """Result of share validation."""
    
    VALID = "valid"
    INVALID_FORMAT = "invalid_format"
    INVALID_SIGNATURE = "invalid_signature"
    UNTRUSTED_SOURCE = "untrusted_source"
    RATE_LIMITED = "rate_limited"
    SIZE_EXCEEDED = "size_exceeded"
    BLOCKED_FEDERATION = "blocked_federation"


# =============================================================================
# EXCEPTIONS
# =============================================================================


class GatewayException(ValenceException):
    """Base exception for gateway-related errors."""
    pass


class RateLimitException(GatewayException):
    """Raised when rate limit is exceeded."""
    
    def __init__(
        self,
        message: str,
        federation_id: UUID | None = None,
        retry_after: float | None = None,
    ):
        details: dict[str, str | float] = {}
        if federation_id:
            details["federation_id"] = str(federation_id)
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, details)
        self.federation_id = federation_id
        self.retry_after = retry_after


class AccessDeniedException(GatewayException):
    """Raised when access is denied to gateway."""
    
    def __init__(
        self,
        message: str,
        federation_id: UUID | None = None,
        reason: str | None = None,
    ):
        details = {}
        if federation_id:
            details["federation_id"] = str(federation_id)
        if reason:
            details["reason"] = reason
        super().__init__(message, details)
        self.federation_id = federation_id
        self.reason = reason


class ValidationFailedException(GatewayException):
    """Raised when share validation fails."""
    
    def __init__(
        self,
        message: str,
        result: ValidationResult | None = None,
        details: dict | None = None,
    ):
        super().__init__(message, details or {})
        self.result = result


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class GatewayConfig:
    """Configuration for a gateway node."""
    
    # Rate limiting
    rate_limit_window: int = DEFAULT_RATE_LIMIT_WINDOW
    rate_limit_max_requests: int = DEFAULT_RATE_LIMIT_MAX_REQUESTS
    rate_limit_max_beliefs: int = DEFAULT_RATE_LIMIT_MAX_BELIEFS
    
    # Access control
    min_trust_for_access: float = MIN_TRUST_FOR_GATEWAY
    max_belief_size: int = DEFAULT_MAX_BELIEF_SIZE
    
    # Blocked federations
    blocked_federations: set[UUID] = field(default_factory=set)
    
    # Allowed capabilities
    enabled_capabilities: set[GatewayCapability] = field(
        default_factory=lambda: {
            GatewayCapability.INBOUND_SHARE,
            GatewayCapability.OUTBOUND_SHARE,
        }
    )
    
    # Audit settings
    audit_retention_days: int = DEFAULT_AUDIT_RETENTION_DAYS
    audit_enabled: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rate_limit_window": self.rate_limit_window,
            "rate_limit_max_requests": self.rate_limit_max_requests,
            "rate_limit_max_beliefs": self.rate_limit_max_beliefs,
            "min_trust_for_access": self.min_trust_for_access,
            "max_belief_size": self.max_belief_size,
            "blocked_federations": [str(f) for f in self.blocked_federations],
            "enabled_capabilities": [c.value for c in self.enabled_capabilities],
            "audit_retention_days": self.audit_retention_days,
            "audit_enabled": self.audit_enabled,
        }


@dataclass
class RateLimitState:
    """Rate limit state for a federation."""
    
    federation_id: UUID
    window_start: float = field(default_factory=time.time)
    request_count: int = 0
    belief_count: int = 0
    
    def reset_if_expired(self, window_seconds: int) -> None:
        """Reset counters if window has expired."""
        now = time.time()
        if now - self.window_start >= window_seconds:
            self.window_start = now
            self.request_count = 0
            self.belief_count = 0
    
    def is_request_allowed(self, max_requests: int) -> bool:
        """Check if a request is allowed."""
        return self.request_count < max_requests
    
    def is_belief_allowed(self, max_beliefs: int, count: int = 1) -> bool:
        """Check if belief(s) can be accepted."""
        return self.belief_count + count <= max_beliefs
    
    def record_request(self) -> None:
        """Record a request."""
        self.request_count += 1
    
    def record_beliefs(self, count: int = 1) -> None:
        """Record belief(s)."""
        self.belief_count += count


@dataclass
class AuditEntry:
    """An entry in the gateway audit log."""
    
    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_type: AuditEventType = AuditEventType.INBOUND_SHARE
    
    # Source/destination
    source_federation_id: UUID | None = None
    source_gateway_endpoint: str | None = None
    target_federation_id: UUID | None = None
    target_gateway_endpoint: str | None = None
    
    # Share details
    direction: ShareDirection | None = None
    belief_ids: list[UUID] = field(default_factory=list)
    belief_count: int = 0
    
    # Result
    success: bool = True
    error_code: ErrorCode | None = None
    error_message: str | None = None
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "source_federation_id": str(self.source_federation_id) if self.source_federation_id else None,
            "source_gateway_endpoint": self.source_gateway_endpoint,
            "target_federation_id": str(self.target_federation_id) if self.target_federation_id else None,
            "target_gateway_endpoint": self.target_gateway_endpoint,
            "direction": self.direction.value if self.direction else None,
            "belief_ids": [str(b) for b in self.belief_ids],
            "belief_count": self.belief_count,
            "success": self.success,
            "error_code": self.error_code.value if self.error_code else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class InboundShare:
    """An inbound share from an external federation."""
    
    id: UUID = field(default_factory=uuid4)
    source_federation_id: UUID | None = None
    source_federation_did: str = ""
    source_gateway_endpoint: str = ""
    
    # Beliefs being shared
    beliefs: list[FederatedBelief] = field(default_factory=list)
    
    # Signature for verification
    signature: str = ""
    
    # Metadata
    received_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass 
class OutboundShare:
    """An outbound share to an external federation."""
    
    id: UUID = field(default_factory=uuid4)
    target_federation_id: UUID | None = None
    target_federation_did: str = ""
    target_gateway_endpoint: str = ""
    
    # Beliefs to share
    beliefs: list[FederatedBelief] = field(default_factory=list)
    
    # Share settings
    share_level: ShareLevel = ShareLevel.BELIEF_ONLY
    
    # Status
    sent_at: datetime | None = None
    acknowledged_at: datetime | None = None
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ShareResult:
    """Result of a share operation."""
    
    success: bool = True
    share_id: UUID | None = None
    
    # For inbound: routed destinations
    routed_to: list[UUID] = field(default_factory=list)
    
    # For outbound: acknowledgment
    acknowledged: bool = False
    
    # Errors
    error_code: ErrorCode | None = None
    error_message: str | None = None
    
    # Stats
    belief_count: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "share_id": str(self.share_id) if self.share_id else None,
            "routed_to": [str(r) for r in self.routed_to],
            "acknowledged": self.acknowledged,
            "error_code": self.error_code.value if self.error_code else None,
            "error_message": self.error_message,
            "belief_count": self.belief_count,
        }


# =============================================================================
# GATEWAY NODE
# =============================================================================


class GatewayNode:
    """A gateway node for bridging federations.
    
    Gateway nodes enable controlled sharing between federations:
    - Inbound: receive and validate external shares, route to members
    - Outbound: send shares to external federations
    - Rate limiting to prevent abuse
    - Access control based on trust
    - Audit logging for all operations
    """
    
    def __init__(
        self,
        federation_id: UUID,
        endpoint: str,
        capabilities: list[GatewayCapability] | None = None,
        config: GatewayConfig | None = None,
    ):
        """Initialize a gateway node.
        
        Args:
            federation_id: ID of the federation this gateway serves
            endpoint: External endpoint URL for this gateway
            capabilities: List of supported capabilities
            config: Gateway configuration
        """
        self.id = uuid4()
        self.federation_id = federation_id
        self.endpoint = endpoint
        self.capabilities = set(capabilities or [
            GatewayCapability.INBOUND_SHARE,
            GatewayCapability.OUTBOUND_SHARE,
        ])
        self.config = config or GatewayConfig()
        self.status = GatewayStatus.INITIALIZING
        
        # Rate limit state per federation
        self._rate_limits: dict[UUID, RateLimitState] = {}
        
        # Audit log (in-memory, can be persisted)
        self._audit_log: list[AuditEntry] = []
        
        # Trust cache (federation_id -> trust_score)
        self._trust_cache: dict[UUID, tuple[float, datetime]] = {}
        self._trust_cache_ttl = timedelta(minutes=5)
        
        # Callbacks for routing
        self._inbound_handler: Callable[[InboundShare], Awaitable[ShareResult]] | None = None
        self._outbound_handler: Callable[[OutboundShare], Awaitable[ShareResult]] | None = None
        
        # Known external gateways
        self._external_gateways: dict[UUID, str] = {}  # federation_id -> endpoint
        
        self.created_at = datetime.now(UTC)
        
        logger.info(
            f"Gateway node initialized for federation {federation_id} "
            f"at {endpoint} with capabilities: {[c.value for c in self.capabilities]}"
        )
    
    # =========================================================================
    # PROPERTIES
    # =========================================================================
    
    @property
    def is_active(self) -> bool:
        """Check if gateway is active."""
        return self.status == GatewayStatus.ACTIVE
    
    def has_capability(self, capability: GatewayCapability) -> bool:
        """Check if gateway has a capability."""
        return (
            capability in self.capabilities and
            capability in self.config.enabled_capabilities
        )
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    async def start(self) -> None:
        """Start the gateway node."""
        logger.info(f"Starting gateway {self.id} for federation {self.federation_id}")
        self.status = GatewayStatus.ACTIVE
        self._audit(
            AuditEventType.CONFIG_CHANGE,
            metadata={"action": "start"},
        )
    
    async def stop(self) -> None:
        """Stop the gateway node."""
        logger.info(f"Stopping gateway {self.id}")
        self.status = GatewayStatus.OFFLINE
        self._audit(
            AuditEventType.CONFIG_CHANGE,
            metadata={"action": "stop"},
        )
    
    def set_inbound_handler(
        self,
        handler: Callable[[InboundShare], Awaitable[ShareResult]],
    ) -> None:
        """Set the handler for inbound shares."""
        self._inbound_handler = handler
    
    def set_outbound_handler(
        self,
        handler: Callable[[OutboundShare], Awaitable[ShareResult]],
    ) -> None:
        """Set the handler for outbound shares."""
        self._outbound_handler = handler
    
    def register_external_gateway(
        self,
        federation_id: UUID,
        endpoint: str,
    ) -> None:
        """Register an external gateway for a federation."""
        self._external_gateways[federation_id] = endpoint
        logger.info(f"Registered external gateway for {federation_id}: {endpoint}")
    
    def unregister_external_gateway(self, federation_id: UUID) -> None:
        """Unregister an external gateway."""
        if federation_id in self._external_gateways:
            del self._external_gateways[federation_id]
            logger.info(f"Unregistered external gateway for {federation_id}")
    
    # =========================================================================
    # RATE LIMITING
    # =========================================================================
    
    def _get_rate_limit_state(self, federation_id: UUID) -> RateLimitState:
        """Get or create rate limit state for a federation."""
        if federation_id not in self._rate_limits:
            self._rate_limits[federation_id] = RateLimitState(federation_id=federation_id)
        state = self._rate_limits[federation_id]
        state.reset_if_expired(self.config.rate_limit_window)
        return state
    
    def _check_rate_limit(
        self,
        federation_id: UUID,
        belief_count: int = 0,
    ) -> tuple[bool, float | None]:
        """Check if request is within rate limits.
        
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        state = self._get_rate_limit_state(federation_id)
        
        # Check request count
        if not state.is_request_allowed(self.config.rate_limit_max_requests):
            retry_after = self.config.rate_limit_window - (time.time() - state.window_start)
            return False, retry_after
        
        # Check belief count
        if belief_count > 0 and not state.is_belief_allowed(
            self.config.rate_limit_max_beliefs,
            belief_count,
        ):
            retry_after = self.config.rate_limit_window - (time.time() - state.window_start)
            return False, retry_after
        
        return True, None
    
    def _record_rate_limit(
        self,
        federation_id: UUID,
        belief_count: int = 0,
    ) -> None:
        """Record a request for rate limiting."""
        state = self._get_rate_limit_state(federation_id)
        state.record_request()
        if belief_count > 0:
            state.record_beliefs(belief_count)
    
    # =========================================================================
    # ACCESS CONTROL
    # =========================================================================
    
    async def _get_federation_trust(self, federation_id: UUID) -> float:
        """Get trust score for a federation, using cache if available."""
        now = datetime.now(UTC)
        
        # Check cache
        if federation_id in self._trust_cache:
            score, cached_at = self._trust_cache[federation_id]
            if now - cached_at < self._trust_cache_ttl:
                return score
        
        # Query trust (in production, this would query the trust system)
        # For now, return a default trust score
        try:
            with get_cursor() as cur:
                cur.execute(
                    """
                    SELECT trust_score
                    FROM federation_node_trust
                    WHERE source_node_id = %s AND target_node_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                    """,
                    (str(self.federation_id), str(federation_id)),
                )
                row = cur.fetchone()
                score = row["trust_score"] if row else 0.5
        except Exception as e:
            logger.warning(f"Failed to query trust for {federation_id}: {e}")
            score = 0.5  # Default to neutral trust
        
        # Cache the result
        self._trust_cache[federation_id] = (score, now)
        return score
    
    async def _check_access(
        self,
        federation_id: UUID,
        capability: GatewayCapability,
    ) -> tuple[bool, str | None]:
        """Check if a federation has access for a capability.
        
        Returns:
            Tuple of (allowed, reason_if_denied)
        """
        # Check if federation is blocked
        if federation_id in self.config.blocked_federations:
            return False, "federation_blocked"
        
        # Check if capability is enabled
        if not self.has_capability(capability):
            return False, "capability_disabled"
        
        # Check trust threshold
        trust = await self._get_federation_trust(federation_id)
        if trust < self.config.min_trust_for_access:
            return False, f"trust_below_threshold:{trust:.2f}"
        
        return True, None
    
    # =========================================================================
    # VALIDATION
    # =========================================================================
    
    def _validate_share(
        self,
        beliefs: list[FederatedBelief],
        source_federation_id: UUID,
    ) -> ValidationResult:
        """Validate an incoming share."""
        # Check for blocked federation
        if source_federation_id in self.config.blocked_federations:
            return ValidationResult.BLOCKED_FEDERATION
        
        # Check belief sizes
        for belief in beliefs:
            content_size = len(belief.content.encode("utf-8"))
            if content_size > self.config.max_belief_size:
                return ValidationResult.SIZE_EXCEEDED
        
        # Additional validation can be added here:
        # - Signature verification
        # - Schema validation
        # - Content filtering
        
        return ValidationResult.VALID
    
    # =========================================================================
    # AUDIT LOGGING
    # =========================================================================
    
    def _audit(
        self,
        event_type: AuditEventType,
        source_federation_id: UUID | None = None,
        target_federation_id: UUID | None = None,
        direction: ShareDirection | None = None,
        belief_ids: list[UUID] | None = None,
        success: bool = True,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AuditEntry:
        """Create an audit log entry."""
        if not self.config.audit_enabled:
            # Return entry but don't store
            return AuditEntry(
                event_type=event_type,
                source_federation_id=source_federation_id,
                target_federation_id=target_federation_id,
                direction=direction,
                belief_ids=belief_ids or [],
                belief_count=len(belief_ids) if belief_ids else 0,
                success=success,
                error_code=error_code,
                error_message=error_message,
                metadata=metadata or {},
            )
        
        entry = AuditEntry(
            event_type=event_type,
            source_federation_id=source_federation_id,
            source_gateway_endpoint=self._external_gateways.get(source_federation_id) if source_federation_id else None,
            target_federation_id=target_federation_id,
            target_gateway_endpoint=self._external_gateways.get(target_federation_id) if target_federation_id else None,
            direction=direction,
            belief_ids=belief_ids or [],
            belief_count=len(belief_ids) if belief_ids else 0,
            success=success,
            error_code=error_code,
            error_message=error_message,
            metadata=metadata or {},
        )
        
        self._audit_log.append(entry)
        
        # Prune old entries
        cutoff = datetime.now(UTC) - timedelta(days=self.config.audit_retention_days)
        self._audit_log = [e for e in self._audit_log if e.timestamp >= cutoff]
        
        logger.debug(f"Audit: {event_type.value} - success={success}")
        
        return entry
    
    def get_audit_log(
        self,
        event_type: AuditEventType | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[AuditEntry]:
        """Get audit log entries."""
        entries = self._audit_log
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        
        # Return most recent first
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    # =========================================================================
    # INBOUND OPERATIONS
    # =========================================================================
    
    async def receive_share(self, share: InboundShare) -> ShareResult:
        """Receive and process an inbound share from an external federation.
        
        Args:
            share: The inbound share to process
            
        Returns:
            ShareResult with routing information
            
        Raises:
            RateLimitException: If rate limit exceeded
            AccessDeniedException: If access denied
            ValidationFailedException: If validation fails
        """
        if not self.is_active:
            return ShareResult(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message="Gateway not active",
            )
        
        if not self.has_capability(GatewayCapability.INBOUND_SHARE):
            return ShareResult(
                success=False,
                error_code=ErrorCode.VISIBILITY_DENIED,
                error_message="Inbound shares not supported",
            )
        
        source_id = share.source_federation_id
        if not source_id:
            return ShareResult(
                success=False,
                error_code=ErrorCode.INVALID_REQUEST,
                error_message="Source federation ID required",
            )
        
        # Check rate limit
        belief_count = len(share.beliefs)
        allowed, retry_after = self._check_rate_limit(source_id, belief_count)
        if not allowed:
            self._audit(
                AuditEventType.RATE_LIMITED,
                source_federation_id=source_id,
                direction=ShareDirection.INBOUND,
                success=False,
                error_code=ErrorCode.RATE_LIMITED,
            )
            raise RateLimitException(
                "Rate limit exceeded",
                federation_id=source_id,
                retry_after=retry_after,
            )
        
        # Check access
        access_allowed, deny_reason = await self._check_access(
            source_id,
            GatewayCapability.INBOUND_SHARE,
        )
        if not access_allowed:
            self._audit(
                AuditEventType.ACCESS_DENIED,
                source_federation_id=source_id,
                direction=ShareDirection.INBOUND,
                success=False,
                error_code=ErrorCode.TRUST_INSUFFICIENT,
                metadata={"reason": deny_reason},
            )
            raise AccessDeniedException(
                "Access denied",
                federation_id=source_id,
                reason=deny_reason,
            )
        
        # Validate share
        validation_result = self._validate_share(share.beliefs, source_id)
        if validation_result != ValidationResult.VALID:
            self._audit(
                AuditEventType.VALIDATION_FAILED,
                source_federation_id=source_id,
                direction=ShareDirection.INBOUND,
                belief_ids=[b.id for b in share.beliefs if b.id],
                success=False,
                error_code=ErrorCode.INVALID_REQUEST,
                metadata={"validation_result": validation_result.value},
            )
            raise ValidationFailedException(
                f"Validation failed: {validation_result.value}",
                result=validation_result,
            )
        
        # Record rate limit
        self._record_rate_limit(source_id, belief_count)
        
        # Route to handler
        if self._inbound_handler:
            try:
                result = await self._inbound_handler(share)
                self._audit(
                    AuditEventType.INBOUND_SHARE,
                    source_federation_id=source_id,
                    direction=ShareDirection.INBOUND,
                    belief_ids=[b.id for b in share.beliefs if b.id],
                    success=result.success,
                    error_code=result.error_code,
                    error_message=result.error_message,
                    metadata={"routed_to": [str(r) for r in result.routed_to]},
                )
                return result
            except Exception as e:
                logger.error(f"Inbound handler failed: {e}")
                self._audit(
                    AuditEventType.ROUTE_FAILED,
                    source_federation_id=source_id,
                    direction=ShareDirection.INBOUND,
                    belief_ids=[b.id for b in share.beliefs if b.id],
                    success=False,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message=str(e),
                )
                return ShareResult(
                    success=False,
                    share_id=share.id,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message=str(e),
                    belief_count=belief_count,
                )
        
        # Default: accept and log
        self._audit(
            AuditEventType.INBOUND_SHARE,
            source_federation_id=source_id,
            direction=ShareDirection.INBOUND,
            belief_ids=[b.id for b in share.beliefs if b.id],
            success=True,
        )
        
        return ShareResult(
            success=True,
            share_id=share.id,
            belief_count=belief_count,
        )
    
    # =========================================================================
    # OUTBOUND OPERATIONS
    # =========================================================================
    
    async def send_share(self, share: OutboundShare) -> ShareResult:
        """Send an outbound share to an external federation.
        
        Args:
            share: The outbound share to send
            
        Returns:
            ShareResult with acknowledgment status
            
        Raises:
            AccessDeniedException: If access denied
            GatewayException: If sending fails
        """
        if not self.is_active:
            return ShareResult(
                success=False,
                error_code=ErrorCode.INTERNAL_ERROR,
                error_message="Gateway not active",
            )
        
        if not self.has_capability(GatewayCapability.OUTBOUND_SHARE):
            return ShareResult(
                success=False,
                error_code=ErrorCode.VISIBILITY_DENIED,
                error_message="Outbound shares not supported",
            )
        
        target_id = share.target_federation_id
        if not target_id:
            return ShareResult(
                success=False,
                error_code=ErrorCode.INVALID_REQUEST,
                error_message="Target federation ID required",
            )
        
        # Check if we know the target gateway
        if target_id not in self._external_gateways and not share.target_gateway_endpoint:
            return ShareResult(
                success=False,
                error_code=ErrorCode.NODE_NOT_FOUND,
                error_message="Unknown target federation gateway",
            )
        
        # Check access (verify we're allowed to share with this federation)
        access_allowed, deny_reason = await self._check_access(
            target_id,
            GatewayCapability.OUTBOUND_SHARE,
        )
        if not access_allowed:
            self._audit(
                AuditEventType.ACCESS_DENIED,
                target_federation_id=target_id,
                direction=ShareDirection.OUTBOUND,
                success=False,
                error_code=ErrorCode.TRUST_INSUFFICIENT,
                metadata={"reason": deny_reason},
            )
            raise AccessDeniedException(
                "Access denied for outbound share",
                federation_id=target_id,
                reason=deny_reason,
            )
        
        belief_count = len(share.beliefs)
        
        # Send via handler
        if self._outbound_handler:
            try:
                share.sent_at = datetime.now(UTC)
                result = await self._outbound_handler(share)
                
                self._audit(
                    AuditEventType.OUTBOUND_SHARE,
                    target_federation_id=target_id,
                    direction=ShareDirection.OUTBOUND,
                    belief_ids=[b.id for b in share.beliefs if b.id],
                    success=result.success,
                    error_code=result.error_code,
                    error_message=result.error_message,
                    metadata={"acknowledged": result.acknowledged},
                )
                return result
            except Exception as e:
                logger.error(f"Outbound handler failed: {e}")
                self._audit(
                    AuditEventType.ROUTE_FAILED,
                    target_federation_id=target_id,
                    direction=ShareDirection.OUTBOUND,
                    belief_ids=[b.id for b in share.beliefs if b.id],
                    success=False,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message=str(e),
                )
                return ShareResult(
                    success=False,
                    share_id=share.id,
                    error_code=ErrorCode.INTERNAL_ERROR,
                    error_message=str(e),
                    belief_count=belief_count,
                )
        
        # No handler - log as success (dry run)
        self._audit(
            AuditEventType.OUTBOUND_SHARE,
            target_federation_id=target_id,
            direction=ShareDirection.OUTBOUND,
            belief_ids=[b.id for b in share.beliefs if b.id],
            success=True,
            metadata={"dry_run": True},
        )
        
        return ShareResult(
            success=True,
            share_id=share.id,
            belief_count=belief_count,
        )
    
    # =========================================================================
    # ADMIN OPERATIONS
    # =========================================================================
    
    def block_federation(self, federation_id: UUID) -> None:
        """Block a federation from using this gateway."""
        self.config.blocked_federations.add(federation_id)
        self._audit(
            AuditEventType.CONFIG_CHANGE,
            metadata={
                "action": "block_federation",
                "federation_id": str(federation_id),
            },
        )
        logger.info(f"Blocked federation {federation_id}")
    
    def unblock_federation(self, federation_id: UUID) -> None:
        """Unblock a federation."""
        self.config.blocked_federations.discard(federation_id)
        self._audit(
            AuditEventType.CONFIG_CHANGE,
            metadata={
                "action": "unblock_federation",
                "federation_id": str(federation_id),
            },
        )
        logger.info(f"Unblocked federation {federation_id}")
    
    def update_config(self, **kwargs: Any) -> None:
        """Update gateway configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self._audit(
            AuditEventType.CONFIG_CHANGE,
            metadata={"updates": kwargs},
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get gateway statistics."""
        now = datetime.now(UTC)
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_entries = [e for e in self._audit_log if e.timestamp >= last_hour]
        daily_entries = [e for e in self._audit_log if e.timestamp >= last_day]
        
        return {
            "gateway_id": str(self.id),
            "federation_id": str(self.federation_id),
            "endpoint": self.endpoint,
            "status": self.status.value,
            "capabilities": [c.value for c in self.capabilities],
            "uptime_seconds": (now - self.created_at).total_seconds(),
            "known_gateways": len(self._external_gateways),
            "stats": {
                "last_hour": {
                    "total_events": len(recent_entries),
                    "inbound_shares": len([e for e in recent_entries if e.event_type == AuditEventType.INBOUND_SHARE]),
                    "outbound_shares": len([e for e in recent_entries if e.event_type == AuditEventType.OUTBOUND_SHARE]),
                    "access_denied": len([e for e in recent_entries if e.event_type == AuditEventType.ACCESS_DENIED]),
                    "rate_limited": len([e for e in recent_entries if e.event_type == AuditEventType.RATE_LIMITED]),
                },
                "last_day": {
                    "total_events": len(daily_entries),
                    "success_rate": (
                        len([e for e in daily_entries if e.success]) / len(daily_entries)
                        if daily_entries else 1.0
                    ),
                },
            },
            "rate_limits": {
                str(k): {
                    "request_count": v.request_count,
                    "belief_count": v.belief_count,
                }
                for k, v in self._rate_limits.items()
            },
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Convert gateway to dictionary."""
        return {
            "id": str(self.id),
            "federation_id": str(self.federation_id),
            "endpoint": self.endpoint,
            "capabilities": [c.value for c in self.capabilities],
            "status": self.status.value,
            "config": self.config.to_dict(),
            "created_at": self.created_at.isoformat(),
            "external_gateways": {
                str(k): v for k, v in self._external_gateways.items()
            },
        }


# =============================================================================
# GATEWAY REGISTRY
# =============================================================================


class GatewayRegistry:
    """Registry for managing gateway nodes.
    
    Provides a centralized way to manage multiple gateway nodes
    and route traffic appropriately.
    """
    
    def __init__(self) -> None:
        self._gateways: dict[UUID, GatewayNode] = {}  # gateway_id -> gateway
        self._by_federation: dict[UUID, UUID] = {}  # federation_id -> gateway_id
    
    def register(self, gateway: GatewayNode) -> None:
        """Register a gateway node."""
        self._gateways[gateway.id] = gateway
        self._by_federation[gateway.federation_id] = gateway.id
        logger.info(f"Registered gateway {gateway.id} for federation {gateway.federation_id}")
    
    def unregister(self, gateway_id: UUID) -> GatewayNode | None:
        """Unregister a gateway node."""
        gateway = self._gateways.pop(gateway_id, None)
        if gateway:
            self._by_federation.pop(gateway.federation_id, None)
            logger.info(f"Unregistered gateway {gateway_id}")
        return gateway
    
    def get(self, gateway_id: UUID) -> GatewayNode | None:
        """Get a gateway by ID."""
        return self._gateways.get(gateway_id)
    
    def get_by_federation(self, federation_id: UUID) -> GatewayNode | None:
        """Get a gateway by federation ID."""
        gateway_id = self._by_federation.get(federation_id)
        if gateway_id:
            return self._gateways.get(gateway_id)
        return None
    
    def list_gateways(
        self,
        status: GatewayStatus | None = None,
        capability: GatewayCapability | None = None,
    ) -> list[GatewayNode]:
        """List gateways with optional filters."""
        gateways = list(self._gateways.values())
        
        if status:
            gateways = [g for g in gateways if g.status == status]
        
        if capability:
            gateways = [g for g in gateways if g.has_capability(capability)]
        
        return gateways
    
    async def start_all(self) -> None:
        """Start all registered gateways."""
        for gateway in self._gateways.values():
            await gateway.start()
    
    async def stop_all(self) -> None:
        """Stop all registered gateways."""
        for gateway in self._gateways.values():
            await gateway.stop()


# Global registry instance
_gateway_registry = GatewayRegistry()


def get_gateway_registry() -> GatewayRegistry:
    """Get the global gateway registry."""
    return _gateway_registry


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_gateway(
    federation_id: UUID,
    endpoint: str,
    capabilities: list[GatewayCapability] | None = None,
    config: GatewayConfig | None = None,
    register: bool = True,
) -> GatewayNode:
    """Create a new gateway node.
    
    Args:
        federation_id: ID of the federation this gateway serves
        endpoint: External endpoint URL
        capabilities: Optional list of capabilities
        config: Optional gateway configuration
        register: Whether to register in global registry
        
    Returns:
        The created gateway node
    """
    gateway = GatewayNode(
        federation_id=federation_id,
        endpoint=endpoint,
        capabilities=capabilities,
        config=config,
    )
    
    if register:
        get_gateway_registry().register(gateway)
    
    return gateway
