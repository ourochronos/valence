"""Capability-based access control for Valence.

Implements OCAP (object-capability) style authorization:
- Capabilities are unforgeable tokens granting specific permissions
- Signed by issuer to prevent tampering
- Short TTL (time-to-live) for security
- Stored for audit trail

A capability grants a holder permission to perform specific actions on a resource.
Unlike role-based access, capabilities are bearer tokens - whoever holds the
capability can use it (within constraints).
"""

from __future__ import annotations

import functools
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Awaitable, TypeVar, ParamSpec, Union
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.exceptions import InvalidSignature
import hashlib
import jwt

# Type variables for decorator
P = ParamSpec('P')
T = TypeVar('T')

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default TTL: 1 hour
DEFAULT_TTL_SECONDS = 3600

# Maximum TTL: 24 hours (to limit exposure of compromised capabilities)
MAX_TTL_SECONDS = 86400  # 24 * 60 * 60


class CapabilityAction(str, Enum):
    """Standard actions for capabilities."""
    
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    DELEGATE = "delegate"  # Can issue sub-capabilities
    ADMIN = "admin"
    
    # Belief-specific actions
    QUERY = "query"
    EMBED = "embed"
    FEDERATE = "federate"


# =============================================================================
# ERRORS
# =============================================================================

class CapabilityError(Exception):
    """Base error for capability operations."""
    pass


class CapabilityExpiredError(CapabilityError):
    """Capability has expired."""
    pass


class CapabilityInvalidSignatureError(CapabilityError):
    """Capability signature is invalid."""
    pass


class CapabilityRevokedError(CapabilityError):
    """Capability has been revoked."""
    pass


class CapabilityNotFoundError(CapabilityError):
    """Capability not found in store."""
    pass


class CapabilityTTLExceededError(CapabilityError):
    """Requested TTL exceeds maximum allowed."""
    pass


class CapabilityInsufficientPermissionError(CapabilityError):
    """Issuer lacks permission to grant this capability."""
    pass


class CapabilityValidationError(CapabilityError):
    """Capability validation failed with detailed errors."""
    
    def __init__(self, message: str, errors: List[str]) -> None:
        super().__init__(message)
        self.errors = errors


# Alias for compatibility (JWT-based validation uses this)
CapabilityInvalidError = CapabilityInvalidSignatureError


# =============================================================================
# CAPABILITY MODEL
# =============================================================================

@dataclass
class Capability:
    """An unforgeable authorization token.
    
    Capabilities are:
    - Bearer tokens: whoever holds it can use it
    - Signed: cryptographically verified issuer
    - Time-limited: short TTL reduces exposure
    - Specific: grants particular actions on particular resources
    - Auditable: stored for tracking who granted what
    
    Attributes:
        id: Unique capability identifier
        issuer_did: DID of the entity that issued this capability
        holder_did: DID of the entity authorized to use this capability
        resource: Resource this capability grants access to (URI or identifier)
        actions: List of permitted actions on the resource
        issued_at: When this capability was issued
        expires_at: When this capability expires (required)
        signature: Ed25519 signature of the capability payload
        parent_id: ID of parent capability (for delegation chains)
        metadata: Additional context (e.g., constraints, domain)
        revoked_at: When this capability was revoked (None if active)
        revocation_reason: Reason for revocation (required when revoked)
    """
    
    id: str
    issuer_did: str
    holder_did: str
    resource: str
    actions: List[str]
    issued_at: datetime
    expires_at: datetime
    signature: Optional[str] = None  # Hex-encoded Ed25519 signature
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    revoked_at: Optional[datetime] = None
    revocation_reason: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate capability fields."""
        if not self.actions:
            raise ValueError("Capability must grant at least one action")
        
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be after issued_at")
        
        if self.holder_did == self.issuer_did:
            # Self-issued capabilities are allowed (for root capabilities)
            pass
    
    @property
    def is_expired(self) -> bool:
        """Check if this capability has expired."""
        now = datetime.now(timezone.utc)
        # Handle timezone-naive expires_at
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now > expires
    
    @property
    def is_revoked(self) -> bool:
        """Check if this capability has been revoked."""
        return self.revoked_at is not None
    
    @property
    def is_valid(self) -> bool:
        """Check if this capability is currently valid (not expired or revoked)."""
        return not self.is_expired and not self.is_revoked
    
    @property
    def ttl_seconds(self) -> float:
        """Return remaining TTL in seconds (negative if expired)."""
        now = datetime.now(timezone.utc)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        delta = expires - now
        return delta.total_seconds()
    
    def has_action(self, action: str) -> bool:
        """Check if this capability grants a specific action."""
        # ADMIN grants all actions
        if CapabilityAction.ADMIN.value in self.actions:
            return True
        return action in self.actions
    
    def payload_bytes(self) -> bytes:
        """Get canonical bytes for signing."""
        payload = {
            "id": self.id,
            "issuer_did": self.issuer_did,
            "holder_did": self.holder_did,
            "resource": self.resource,
            "actions": sorted(self.actions),  # Canonical ordering
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "parent_id": self.parent_id,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "issuer_did": self.issuer_did,
            "holder_did": self.holder_did,
            "resource": self.resource,
            "actions": self.actions,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "signature": self.signature,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revocation_reason": self.revocation_reason,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Capability":
        """Deserialize from dictionary."""
        # Parse datetimes - handle both naive and aware
        issued_at = datetime.fromisoformat(data["issued_at"])
        expires_at = datetime.fromisoformat(data["expires_at"])
        
        # Ensure timezone-aware (assume UTC if naive)
        if issued_at.tzinfo is None:
            issued_at = issued_at.replace(tzinfo=timezone.utc)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        
        revoked_at = None
        if data.get("revoked_at"):
            revoked_at = datetime.fromisoformat(data["revoked_at"])
            if revoked_at.tzinfo is None:
                revoked_at = revoked_at.replace(tzinfo=timezone.utc)
        
        return cls(
            id=data["id"],
            issuer_did=data["issuer_did"],
            holder_did=data["holder_did"],
            resource=data["resource"],
            actions=data["actions"],
            issued_at=issued_at,
            expires_at=expires_at,
            signature=data.get("signature"),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {}),
            revoked_at=revoked_at,
            revocation_reason=data.get("revocation_reason"),
        )
    
    def to_jwt(self, secret: str, algorithm: str = "HS256") -> str:
        """Serialize capability to a signed JWT.
        
        The JWT contains the full capability data in claims, allowing
        stateless verification by any party that knows the secret.
        
        Note: This is separate from the Ed25519 signature used for
        cryptographic capability verification. JWT is useful for
        transporting capabilities over HTTP APIs.
        
        Args:
            secret: Signing secret (should be at least 32 bytes)
            algorithm: JWT signing algorithm (default HS256)
        
        Returns:
            Signed JWT string
        """
        payload = {
            # Standard JWT claims
            "jti": self.id,  # JWT ID = capability ID
            "iss": self.issuer_did,  # Issuer
            "sub": self.holder_did,  # Subject (holder)
            "iat": int(self.issued_at.timestamp()),  # Issued at
            "exp": int(self.expires_at.timestamp()),  # Expiration
            # Custom claims for capability data
            "resource": self.resource,
            "actions": self.actions,
            "metadata": self.metadata,
            "parent_id": self.parent_id,
        }
        return jwt.encode(payload, secret, algorithm=algorithm)
    
    @classmethod
    def from_jwt(
        cls,
        token: str,
        secret: str,
        algorithm: str = "HS256",
        verify_exp: bool = True,
    ) -> "Capability":
        """Deserialize and verify a capability from JWT.
        
        Args:
            token: JWT string
            secret: Signing secret for verification
            algorithm: Expected signing algorithm
            verify_exp: Whether to verify expiration (default True)
        
        Returns:
            Capability instance
        
        Raises:
            CapabilityExpiredError: If the capability has expired
            CapabilityInvalidError: If the JWT is invalid
        """
        try:
            options = {"verify_exp": verify_exp}
            payload = jwt.decode(
                token,
                secret,
                algorithms=[algorithm],
                options=options,
            )
        except jwt.ExpiredSignatureError as e:
            raise CapabilityExpiredError("Capability has expired") from e
        except jwt.InvalidTokenError as e:
            raise CapabilityInvalidSignatureError(f"Invalid capability token: {e}") from e
        
        # Convert timestamps back to datetime
        issued_at = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        
        return cls(
            id=payload["jti"],
            issuer_did=payload["iss"],
            holder_did=payload["sub"],
            resource=payload["resource"],
            actions=payload["actions"],
            issued_at=issued_at,
            expires_at=expires_at,
            metadata=payload.get("metadata", {}),
            parent_id=payload.get("parent_id"),
            # JWT-deserialized capabilities don't carry Ed25519 signatures
            signature=None,
        )
    
    @classmethod
    def create(
        cls,
        issuer_did: str,
        holder_did: str,
        resource: str,
        actions: List[str],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        metadata: Optional[Dict[str, Any]] = None,
        capability_id: Optional[str] = None,
    ) -> "Capability":
        """Create a new capability (without cryptographic signature).
        
        This is a convenience method for creating capabilities for JWT-based
        transport. For Ed25519-signed capabilities, use CapabilityService.issue().
        
        Args:
            issuer_did: DID of the issuing entity
            holder_did: DID of the entity receiving the capability
            resource: Resource identifier this grants access to
            actions: List of permitted actions on the resource
            ttl_seconds: Time-to-live in seconds (default 1 hour)
            metadata: Optional additional context (replaces 'constraints')
            capability_id: Optional explicit ID (auto-generated if not provided)
        
        Returns:
            A new Capability instance (unsigned)
        """
        now = datetime.now(timezone.utc)
        return cls(
            id=capability_id or str(uuid.uuid4()),
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=list(actions),  # Ensure it's a list copy
            issued_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            metadata=metadata or {},
        )


# =============================================================================
# CAPABILITY VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Result of capability validation with detailed error information.
    
    Provides comprehensive validation feedback:
    - is_valid: Overall validation result
    - errors: List of specific validation failures
    - checked_at: Timestamp of validation
    - capability_id: ID of validated capability
    - resource: Resource that was validated against
    - action: Action that was validated against
    
    Example:
        result = validate_capability(cap, "valence://beliefs/science", "read")
        if not result.is_valid:
            for error in result.errors:
                print(f"Validation failed: {error}")
    """
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    capability_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    
    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid
    
    def raise_if_invalid(self) -> None:
        """Raise CapabilityValidationError if validation failed.
        
        Raises:
            CapabilityValidationError: If is_valid is False
        """
        if not self.is_valid:
            raise CapabilityValidationError(
                f"Capability validation failed: {'; '.join(self.errors)}",
                errors=self.errors,
            )


def validate_capability(
    capability: Capability,
    resource: str,
    action: str,
    *,
    check_revocation: bool = True,
    store: Optional["CapabilityStore"] = None,
) -> ValidationResult:
    """Validate a capability for accessing a protected resource.
    
    Performs comprehensive validation:
    1. Expiration check
    2. Revocation check (local flag)
    3. Resource match
    4. Action permission
    
    Note: This is synchronous validation. For async revocation checking
    against a store, use validate_capability_async().
    
    Args:
        capability: The capability to validate
        resource: Resource identifier to validate against
        action: Action to validate against
        check_revocation: Whether to check local revocation flag (default True)
        store: Optional store for additional revocation check (not used in sync version)
    
    Returns:
        ValidationResult with is_valid and detailed errors
    
    Example:
        cap = Capability.create(...)
        result = validate_capability(cap, "valence://data/users", "read")
        if result.is_valid:
            # Grant access
            pass
        else:
            # Log errors
            for err in result.errors:
                logger.warning(err)
    """
    errors: List[str] = []
    
    # Check expiration
    if capability.is_expired:
        expires = capability.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        errors.append(f"Capability expired at {expires.isoformat()}")
    
    # Check revocation (local flag only in sync version)
    if check_revocation and capability.is_revoked:
        revoked = capability.revoked_at
        if revoked and revoked.tzinfo is None:
            revoked = revoked.replace(tzinfo=timezone.utc)
        errors.append(f"Capability was revoked at {revoked.isoformat() if revoked else 'unknown time'}")
    
    # Check resource match
    if capability.resource != resource:
        errors.append(
            f"Resource mismatch: capability grants access to '{capability.resource}', "
            f"but '{resource}' was requested"
        )
    
    # Check action permission
    if not capability.has_action(action):
        errors.append(
            f"Action not permitted: capability grants {capability.actions}, "
            f"but '{action}' was requested"
        )
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        capability_id=capability.id,
        resource=resource,
        action=action,
    )


async def validate_capability_async(
    capability: Capability,
    resource: str,
    action: str,
    *,
    store: Optional["CapabilityStore"] = None,
) -> ValidationResult:
    """Validate a capability asynchronously, including store revocation check.
    
    This async version additionally checks the store for revocation status,
    which is useful when capabilities might be revoked server-side.
    
    Args:
        capability: The capability to validate
        resource: Resource identifier to validate against
        action: Action to validate against
        store: Optional store to check for revocation
    
    Returns:
        ValidationResult with is_valid and detailed errors
    """
    # Start with synchronous validation (without local revocation check if store provided)
    result = validate_capability(
        capability,
        resource,
        action,
        check_revocation=(store is None),  # Skip local check if we'll check store
    )
    
    errors = list(result.errors)
    
    # Check store revocation if store provided
    if store is not None:
        # Check local revocation flag first
        if capability.is_revoked:
            revoked = capability.revoked_at
            if revoked and revoked.tzinfo is None:
                revoked = revoked.replace(tzinfo=timezone.utc)
            errors.append(f"Capability was revoked at {revoked.isoformat() if revoked else 'unknown time'}")
        # Also check store
        elif await store.is_revoked(capability.id):
            errors.append(f"Capability {capability.id} was revoked (store check)")
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        checked_at=datetime.now(timezone.utc),
        capability_id=capability.id,
        resource=resource,
        action=action,
    )


# =============================================================================
# CAPABILITY DECORATOR
# =============================================================================

def requires_capability(
    resource: Union[str, Callable[..., str]],
    action: str,
    *,
    capability_param: str = "capability",
    store_param: Optional[str] = None,
    raise_on_invalid: bool = True,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator for protecting functions with capability-based access control.
    
    Validates that the provided capability grants access to the specified
    resource and action before executing the decorated function.
    
    Args:
        resource: Resource identifier (string) or callable that extracts
                  resource from function arguments
        action: Required action (e.g., "read", "write")
        capability_param: Name of the parameter containing the capability
                          (default: "capability")
        store_param: Name of the parameter containing the store for async
                     revocation checking (optional)
        raise_on_invalid: Whether to raise CapabilityValidationError on
                          validation failure (default True). If False,
                          returns None for sync functions or a ValidationResult
                          for functions that return Optional types.
    
    Returns:
        Decorated function that validates capability before execution
    
    Example:
        @requires_capability("valence://beliefs/science", "read")
        async def read_beliefs(capability: Capability) -> List[Belief]:
            return await fetch_beliefs()
        
        @requires_capability(
            lambda self, domain: f"valence://beliefs/{domain}",
            "write"
        )
        async def write_belief(
            self,
            domain: str,
            content: str,
            capability: Capability
        ) -> Belief:
            return await store_belief(domain, content)
    
    Raises:
        CapabilityValidationError: If validation fails and raise_on_invalid=True
        ValueError: If capability_param not found in function arguments
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        import asyncio
        import inspect
        
        # Check if function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract capability from arguments
            cap = _extract_param(func, args, kwargs, capability_param)
            if cap is None:
                raise ValueError(f"Required parameter '{capability_param}' not provided")
            
            # Resolve resource (might be callable)
            resolved_resource = _resolve_resource(resource, func, args, kwargs)
            
            # Extract optional store
            store = None
            if store_param:
                store = _extract_param(func, args, kwargs, store_param)
            
            # Validate
            result = validate_capability(cap, resolved_resource, action, store=store)
            
            if not result.is_valid:
                if raise_on_invalid:
                    result.raise_if_invalid()
                return None  # type: ignore
            
            return func(*args, **kwargs)
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Extract capability from arguments
            cap = _extract_param(func, args, kwargs, capability_param)
            if cap is None:
                raise ValueError(f"Required parameter '{capability_param}' not provided")
            
            # Resolve resource (might be callable)
            resolved_resource = _resolve_resource(resource, func, args, kwargs)
            
            # Extract optional store
            store = None
            if store_param:
                store = _extract_param(func, args, kwargs, store_param)
            
            # Validate (use async version if store provided)
            if store is not None:
                result = await validate_capability_async(cap, resolved_resource, action, store=store)
            else:
                result = validate_capability(cap, resolved_resource, action)
            
            if not result.is_valid:
                if raise_on_invalid:
                    result.raise_if_invalid()
                return None  # type: ignore
            
            return await func(*args, **kwargs)
        
        return async_wrapper if is_async else sync_wrapper  # type: ignore
    
    return decorator


def _extract_param(
    func: Callable,
    args: tuple,
    kwargs: dict,
    param_name: str,
) -> Any:
    """Extract a parameter value from function arguments."""
    import inspect
    
    # Check kwargs first
    if param_name in kwargs:
        return kwargs[param_name]
    
    # Get function signature to map positional args
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())
    
    try:
        idx = params.index(param_name)
        if idx < len(args):
            return args[idx]
    except ValueError:
        pass
    
    return None


def _resolve_resource(
    resource: Union[str, Callable],
    func: Callable,
    args: tuple,
    kwargs: dict,
) -> str:
    """Resolve resource identifier from string or callable."""
    import inspect
    
    if callable(resource) and not isinstance(resource, str):
        # Build kwargs dict from positional args of the decorated function
        func_sig = inspect.signature(func)
        func_params = list(func_sig.parameters.keys())
        
        # Create combined kwargs from function
        combined = dict(kwargs)
        for i, val in enumerate(args):
            if i < len(func_params):
                combined[func_params[i]] = val
        
        # Get the resource callable's signature to know what args it expects
        resource_sig = inspect.signature(resource)
        resource_params = set(resource_sig.parameters.keys())
        
        # Filter combined to only include params the resource callable expects
        filtered = {k: v for k, v in combined.items() if k in resource_params}
        
        # Call resource resolver with only expected arguments
        return resource(**filtered)
    
    return resource


# =============================================================================
# CAPABILITY STORE (Abstract Interface)
# =============================================================================

class CapabilityStore:
    """Abstract interface for capability persistence.
    
    Implementations can use different backends:
    - In-memory (for testing)
    - PostgreSQL (for production)
    - File-based (for single-node deployment)
    """
    
    async def save(self, capability: Capability) -> None:
        """Persist a capability for audit trail."""
        raise NotImplementedError
    
    async def get(self, capability_id: str) -> Optional[Capability]:
        """Retrieve a capability by ID."""
        raise NotImplementedError
    
    async def list_by_holder(self, holder_did: str, include_expired: bool = False) -> List[Capability]:
        """List capabilities held by a DID."""
        raise NotImplementedError
    
    async def list_by_issuer(self, issuer_did: str, include_expired: bool = False) -> List[Capability]:
        """List capabilities issued by a DID."""
        raise NotImplementedError
    
    async def list_by_resource(self, resource: str, include_expired: bool = False) -> List[Capability]:
        """List capabilities for a resource."""
        raise NotImplementedError
    
    async def revoke(self, capability_id: str, reason: str) -> bool:
        """Mark a capability as revoked with a reason.
        
        Revocation is immutable - once revoked, a capability cannot be un-revoked.
        
        Args:
            capability_id: ID of capability to revoke
            reason: Reason for revocation (required)
        
        Returns:
            True if found and revoked, False if not found or already revoked
        """
        raise NotImplementedError
    
    async def revoke_by_issuer(self, issuer_did: str, reason: str) -> int:
        """Revoke all capabilities issued by a DID.
        
        Args:
            issuer_did: DID of the issuer
            reason: Reason for revocation
        
        Returns:
            Number of capabilities revoked
        """
        raise NotImplementedError
    
    async def revoke_by_holder(self, holder_did: str, reason: str) -> int:
        """Revoke all capabilities held by a DID.
        
        Args:
            holder_did: DID of the holder
            reason: Reason for revocation
        
        Returns:
            Number of capabilities revoked
        """
        raise NotImplementedError
    
    async def is_revoked(self, capability_id: str) -> bool:
        """Check if a capability is revoked."""
        raise NotImplementedError
    
    async def get_revocation_info(self, capability_id: str) -> Optional[tuple[datetime, str]]:
        """Get revocation timestamp and reason for a capability.
        
        Returns:
            Tuple of (revoked_at, reason) if revoked, None if not revoked or not found
        """
        raise NotImplementedError
    
    async def cleanup_expired(self, older_than_days: int = 30) -> int:
        """Remove expired capabilities older than specified days. Returns count removed."""
        raise NotImplementedError


class InMemoryCapabilityStore(CapabilityStore):
    """In-memory capability store for testing."""
    
    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}
    
    async def save(self, capability: Capability) -> None:
        self._capabilities[capability.id] = capability
    
    async def get(self, capability_id: str) -> Optional[Capability]:
        return self._capabilities.get(capability_id)
    
    async def list_by_holder(self, holder_did: str, include_expired: bool = False) -> List[Capability]:
        result = []
        for cap in self._capabilities.values():
            if cap.holder_did == holder_did:
                if include_expired or cap.is_valid:
                    result.append(cap)
        return result
    
    async def list_by_issuer(self, issuer_did: str, include_expired: bool = False) -> List[Capability]:
        result = []
        for cap in self._capabilities.values():
            if cap.issuer_did == issuer_did:
                if include_expired or cap.is_valid:
                    result.append(cap)
        return result
    
    async def list_by_resource(self, resource: str, include_expired: bool = False) -> List[Capability]:
        result = []
        for cap in self._capabilities.values():
            if cap.resource == resource:
                if include_expired or cap.is_valid:
                    result.append(cap)
        return result
    
    async def revoke(self, capability_id: str, reason: str) -> bool:
        """Mark a capability as revoked with a reason.
        
        Revocation is immutable - already revoked capabilities return False.
        """
        if capability_id in self._capabilities:
            cap = self._capabilities[capability_id]
            # Immutable: cannot re-revoke
            if cap.is_revoked:
                return False
            cap.revoked_at = datetime.now(timezone.utc)
            cap.revocation_reason = reason
            return True
        return False
    
    async def revoke_by_issuer(self, issuer_did: str, reason: str) -> int:
        """Revoke all capabilities issued by a DID."""
        count = 0
        for cap in self._capabilities.values():
            if cap.issuer_did == issuer_did and not cap.is_revoked:
                cap.revoked_at = datetime.now(timezone.utc)
                cap.revocation_reason = reason
                count += 1
        return count
    
    async def revoke_by_holder(self, holder_did: str, reason: str) -> int:
        """Revoke all capabilities held by a DID."""
        count = 0
        for cap in self._capabilities.values():
            if cap.holder_did == holder_did and not cap.is_revoked:
                cap.revoked_at = datetime.now(timezone.utc)
                cap.revocation_reason = reason
                count += 1
        return count
    
    async def is_revoked(self, capability_id: str) -> bool:
        cap = self._capabilities.get(capability_id)
        if cap:
            return cap.is_revoked
        return False
    
    async def get_revocation_info(self, capability_id: str) -> Optional[tuple[datetime, str]]:
        """Get revocation timestamp and reason for a capability."""
        cap = self._capabilities.get(capability_id)
        if cap and cap.is_revoked:
            return (cap.revoked_at, cap.revocation_reason)
        return None
    
    async def cleanup_expired(self, older_than_days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        to_remove = []
        for cap_id, cap in self._capabilities.items():
            if cap.is_expired:
                expires = cap.expires_at
                if expires.tzinfo is None:
                    expires = expires.replace(tzinfo=timezone.utc)
                if expires < cutoff:
                    to_remove.append(cap_id)
        
        for cap_id in to_remove:
            del self._capabilities[cap_id]
        
        return len(to_remove)


# =============================================================================
# KEY RESOLVER (for looking up issuer public keys)
# =============================================================================

# Type alias for key resolver function
KeyResolver = Callable[[str], Awaitable[Optional[Ed25519PublicKey]]]


async def null_key_resolver(did: str) -> Optional[Ed25519PublicKey]:
    """Default key resolver that always returns None."""
    return None


# =============================================================================
# CAPABILITY SERVICE
# =============================================================================

class CapabilityService:
    """Service for issuing and verifying capabilities.
    
    Manages the lifecycle of capabilities:
    - Issue: Create signed capabilities with TTL
    - Verify: Check signature and expiration
    - Revoke: Invalidate capabilities before expiration
    - Delegate: Issue sub-capabilities from existing capabilities
    
    Configuration:
    - default_ttl_seconds: Default TTL for new capabilities (1 hour)
    - max_ttl_seconds: Maximum allowed TTL (24 hours)
    """
    
    def __init__(
        self,
        store: Optional[CapabilityStore] = None,
        key_resolver: Optional[KeyResolver] = None,
        default_ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_ttl_seconds: int = MAX_TTL_SECONDS,
    ) -> None:
        """Initialize the capability service.
        
        Args:
            store: Capability store for persistence (uses in-memory if None)
            key_resolver: Async function to resolve DIDs to public keys
            default_ttl_seconds: Default TTL for new capabilities
            max_ttl_seconds: Maximum allowed TTL
        """
        self.store = store or InMemoryCapabilityStore()
        self.key_resolver = key_resolver or null_key_resolver
        self.default_ttl_seconds = default_ttl_seconds
        self.max_ttl_seconds = max_ttl_seconds
    
    async def issue(
        self,
        issuer_did: str,
        holder_did: str,
        resource: str,
        actions: List[str],
        issuer_private_key: Ed25519PrivateKey,
        ttl_seconds: Optional[int] = None,
        parent_capability: Optional[Capability] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Capability:
        """Issue a new signed capability.
        
        Args:
            issuer_did: DID of the issuer
            holder_did: DID of the holder receiving the capability
            resource: Resource identifier (URI or path)
            actions: List of permitted actions
            issuer_private_key: Issuer's Ed25519 private key for signing
            ttl_seconds: Time-to-live in seconds (uses default if None)
            parent_capability: Parent capability for delegation chains
            metadata: Additional context data
            
        Returns:
            Signed Capability object
            
        Raises:
            CapabilityTTLExceededError: If requested TTL exceeds maximum
            CapabilityInsufficientPermissionError: If parent doesn't allow delegation
        """
        # Determine TTL
        effective_ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        
        # Enforce maximum TTL
        if effective_ttl > self.max_ttl_seconds:
            raise CapabilityTTLExceededError(
                f"Requested TTL ({effective_ttl}s) exceeds maximum ({self.max_ttl_seconds}s)"
            )
        
        # If delegating from parent, validate delegation rights
        parent_id = None
        if parent_capability:
            if not parent_capability.has_action(CapabilityAction.DELEGATE.value):
                raise CapabilityInsufficientPermissionError(
                    "Parent capability does not grant delegation rights"
                )
            
            if parent_capability.is_expired:
                raise CapabilityExpiredError("Parent capability has expired")
            
            if parent_capability.is_revoked:
                raise CapabilityRevokedError("Parent capability has been revoked")
            
            # Check that requested actions are subset of parent's actions
            parent_actions = set(parent_capability.actions)
            has_admin = CapabilityAction.ADMIN.value in parent_actions
            
            if not has_admin:
                requested_actions = set(actions)
                if not requested_actions.issubset(parent_actions):
                    extra = requested_actions - parent_actions
                    raise CapabilityInsufficientPermissionError(
                        f"Cannot delegate actions not in parent: {extra}"
                    )
            
            # Delegated capability TTL cannot exceed parent's remaining TTL
            parent_ttl = parent_capability.ttl_seconds
            if effective_ttl > parent_ttl:
                effective_ttl = int(parent_ttl)
                logger.info(f"Capped delegated capability TTL to parent's remaining TTL: {effective_ttl}s")
            
            parent_id = parent_capability.id
        
        # Generate capability
        now = datetime.now(timezone.utc)
        capability = Capability(
            id=str(uuid.uuid4()),
            issuer_did=issuer_did,
            holder_did=holder_did,
            resource=resource,
            actions=actions,
            issued_at=now,
            expires_at=now + timedelta(seconds=effective_ttl),
            parent_id=parent_id,
            metadata=metadata or {},
        )
        
        # Sign the capability
        payload = capability.payload_bytes()
        signature = issuer_private_key.sign(payload)
        capability.signature = signature.hex()
        
        # Store for audit
        await self.store.save(capability)
        
        logger.info(
            f"Issued capability {capability.id}: "
            f"{issuer_did} -> {holder_did} for {resource} "
            f"(actions={actions}, ttl={effective_ttl}s)"
        )
        
        return capability
    
    async def verify(
        self,
        capability: Capability,
        issuer_public_key: Optional[Ed25519PublicKey] = None,
    ) -> bool:
        """Verify a capability's signature and validity.
        
        Args:
            capability: The capability to verify
            issuer_public_key: Issuer's public key (resolved via key_resolver if None)
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            CapabilityExpiredError: If capability has expired
            CapabilityRevokedError: If capability has been revoked
            CapabilityInvalidSignatureError: If signature verification fails
        """
        # Check expiration
        if capability.is_expired:
            raise CapabilityExpiredError(f"Capability {capability.id} has expired")
        
        # Check revocation (both local record and store)
        if capability.is_revoked:
            raise CapabilityRevokedError(f"Capability {capability.id} has been revoked")
        
        if await self.store.is_revoked(capability.id):
            raise CapabilityRevokedError(f"Capability {capability.id} has been revoked")
        
        # Get issuer's public key
        if issuer_public_key is None:
            issuer_public_key = await self.key_resolver(capability.issuer_did)
            if issuer_public_key is None:
                raise CapabilityInvalidSignatureError(
                    f"Cannot resolve public key for {capability.issuer_did}"
                )
        
        # Verify signature
        if not capability.signature:
            raise CapabilityInvalidSignatureError("Capability has no signature")
        
        try:
            payload = capability.payload_bytes()
            signature = bytes.fromhex(capability.signature)
            issuer_public_key.verify(signature, payload)
        except InvalidSignature:
            raise CapabilityInvalidSignatureError(
                f"Invalid signature on capability {capability.id}"
            )
        
        return True
    
    async def revoke(self, capability_id: str, reason: str) -> bool:
        """Revoke a capability by ID.
        
        Revocation is immutable - once revoked, a capability cannot be un-revoked.
        
        Args:
            capability_id: ID of capability to revoke
            reason: Reason for revocation (required for audit trail)
            
        Returns:
            True if capability was found and revoked, False if not found or already revoked
        
        Raises:
            ValueError: If reason is empty
        """
        if not reason or not reason.strip():
            raise ValueError("Revocation reason is required")
        
        result = await self.store.revoke(capability_id, reason)
        if result:
            logger.info(f"Revoked capability {capability_id}: {reason}")
        return result
    
    async def revoke_by_issuer(self, issuer_did: str, reason: str) -> int:
        """Revoke all capabilities issued by a DID.
        
        Use this for bulk revocation when an issuer is compromised or
        needs to invalidate all issued capabilities.
        
        Args:
            issuer_did: DID of the issuer
            reason: Reason for revocation (required for audit trail)
            
        Returns:
            Number of capabilities revoked
        
        Raises:
            ValueError: If reason is empty
        """
        if not reason or not reason.strip():
            raise ValueError("Revocation reason is required")
        
        count = await self.store.revoke_by_issuer(issuer_did, reason)
        if count > 0:
            logger.info(f"Revoked {count} capabilities issued by {issuer_did}: {reason}")
        return count
    
    async def revoke_by_holder(self, holder_did: str, reason: str) -> int:
        """Revoke all capabilities held by a DID.
        
        Use this for bulk revocation when a holder is compromised or
        their access needs to be terminated.
        
        Args:
            holder_did: DID of the holder
            reason: Reason for revocation (required for audit trail)
            
        Returns:
            Number of capabilities revoked
        
        Raises:
            ValueError: If reason is empty
        """
        if not reason or not reason.strip():
            raise ValueError("Revocation reason is required")
        
        count = await self.store.revoke_by_holder(holder_did, reason)
        if count > 0:
            logger.info(f"Revoked {count} capabilities held by {holder_did}: {reason}")
        return count
    
    async def get_revocation_info(self, capability_id: str) -> Optional[tuple[datetime, str]]:
        """Get revocation information for a capability.
        
        Args:
            capability_id: ID of capability to check
            
        Returns:
            Tuple of (revoked_at, reason) if revoked, None otherwise
        """
        return await self.store.get_revocation_info(capability_id)
    
    async def get(self, capability_id: str) -> Optional[Capability]:
        """Get a capability by ID.
        
        Args:
            capability_id: ID of capability to retrieve
            
        Returns:
            Capability if found, None otherwise
        """
        return await self.store.get(capability_id)
    
    async def list_holder_capabilities(
        self,
        holder_did: str,
        include_expired: bool = False,
    ) -> List[Capability]:
        """List capabilities held by a DID.
        
        Args:
            holder_did: DID of the holder
            include_expired: Include expired capabilities
            
        Returns:
            List of capabilities
        """
        return await self.store.list_by_holder(holder_did, include_expired)
    
    async def list_issuer_capabilities(
        self,
        issuer_did: str,
        include_expired: bool = False,
    ) -> List[Capability]:
        """List capabilities issued by a DID.
        
        Args:
            issuer_did: DID of the issuer
            include_expired: Include expired capabilities
            
        Returns:
            List of capabilities
        """
        return await self.store.list_by_issuer(issuer_did, include_expired)
    
    async def check_access(
        self,
        holder_did: str,
        resource: str,
        action: str,
        issuer_public_key: Optional[Ed25519PublicKey] = None,
    ) -> Optional[Capability]:
        """Check if a holder has access to perform an action on a resource.
        
        Args:
            holder_did: DID of the entity requesting access
            resource: Resource identifier
            action: Action to perform
            issuer_public_key: Optional public key for verification
            
        Returns:
            Valid capability granting access, or None if no access
        """
        capabilities = await self.store.list_by_holder(holder_did, include_expired=False)
        
        for cap in capabilities:
            # Check resource match
            if cap.resource != resource:
                continue
            
            # Check action match
            if not cap.has_action(action):
                continue
            
            # Verify signature
            try:
                await self.verify(cap, issuer_public_key)
                return cap
            except CapabilityError:
                continue
        
        return None
    
    async def validate_capability(
        self,
        capability: Capability,
        resource: str,
        action: str,
        *,
        verify_signature: bool = True,
        issuer_public_key: Optional[Ed25519PublicKey] = None,
    ) -> ValidationResult:
        """Validate a capability for accessing a protected resource.
        
        Performs comprehensive validation:
        1. Expiration check
        2. Revocation check (both local and store)
        3. Resource match
        4. Action permission
        5. Optional signature verification
        
        This method integrates with the CapabilityService's store and key resolver.
        
        Args:
            capability: The capability to validate
            resource: Resource identifier to validate against
            action: Action to validate against
            verify_signature: Whether to verify the capability signature (default True)
            issuer_public_key: Optional public key for signature verification
        
        Returns:
            ValidationResult with is_valid and detailed errors
        
        Example:
            result = await service.validate_capability(cap, "valence://data/users", "read")
            if result.is_valid:
                # Grant access
                pass
            else:
                logger.warning(f"Access denied: {result.errors}")
        """
        # Start with async validation (includes store revocation check)
        result = await validate_capability_async(
            capability,
            resource,
            action,
            store=self.store,
        )
        
        errors = list(result.errors)
        
        # Optionally verify signature
        if verify_signature and result.is_valid:
            try:
                await self.verify(capability, issuer_public_key)
            except CapabilityExpiredError as e:
                # Already checked, but signature verify also checks
                if not any("expired" in err.lower() for err in errors):
                    errors.append(str(e))
            except CapabilityRevokedError as e:
                if not any("revoked" in err.lower() for err in errors):
                    errors.append(str(e))
            except CapabilityInvalidSignatureError as e:
                errors.append(f"Signature verification failed: {e}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            checked_at=datetime.now(timezone.utc),
            capability_id=capability.id,
            resource=resource,
            action=action,
        )
    
    async def delegate(
        self,
        parent_capability: Capability,
        new_holder_did: str,
        delegator_private_key: Ed25519PrivateKey,
        actions: Optional[List[str]] = None,
        ttl_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Capability:
        """Delegate a capability to a new holder.
        
        Creates a new capability derived from the parent capability.
        The new capability:
        - Cannot exceed parent's permissions
        - Cannot outlive parent's expiration
        - Records parent_id for audit trail
        
        Args:
            parent_capability: Capability being delegated
            new_holder_did: DID receiving the delegated capability
            delegator_private_key: Delegator's private key for signing
            actions: Actions to delegate (subset of parent, or all if None)
            ttl_seconds: TTL for delegated capability
            metadata: Additional context
            
        Returns:
            New delegated Capability
        """
        # Use parent's actions if none specified
        effective_actions = actions if actions is not None else parent_capability.actions.copy()
        
        # The delegator becomes the issuer of the new capability
        # (the holder of the parent capability)
        return await self.issue(
            issuer_did=parent_capability.holder_did,
            holder_did=new_holder_did,
            resource=parent_capability.resource,
            actions=effective_actions,
            issuer_private_key=delegator_private_key,
            ttl_seconds=ttl_seconds,
            parent_capability=parent_capability,
            metadata=metadata,
        )


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_default_service: Optional[CapabilityService] = None


def get_capability_service() -> CapabilityService:
    """Get the default capability service singleton."""
    global _default_service
    if _default_service is None:
        _default_service = CapabilityService()
    return _default_service


def set_capability_service(service: CapabilityService) -> None:
    """Set the default capability service singleton."""
    global _default_service
    _default_service = service


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def issue_capability(
    issuer_did: str,
    holder_did: str,
    resource: str,
    actions: List[str],
    issuer_private_key: Ed25519PrivateKey,
    ttl_seconds: Optional[int] = None,
) -> Capability:
    """Issue a capability using the default service."""
    return await get_capability_service().issue(
        issuer_did=issuer_did,
        holder_did=holder_did,
        resource=resource,
        actions=actions,
        issuer_private_key=issuer_private_key,
        ttl_seconds=ttl_seconds,
    )


async def verify_capability(
    capability: Capability,
    issuer_public_key: Optional[Ed25519PublicKey] = None,
) -> bool:
    """Verify a capability using the default service."""
    return await get_capability_service().verify(capability, issuer_public_key)


async def revoke_capability(capability_id: str, reason: str) -> bool:
    """Revoke a capability using the default service.
    
    Args:
        capability_id: ID of capability to revoke
        reason: Reason for revocation (required)
    
    Returns:
        True if capability was found and revoked
    """
    return await get_capability_service().revoke(capability_id, reason)
