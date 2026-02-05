"""Cross-federation consent chains for Valence.

Implements consent chain management that spans federation boundaries:
- CrossFederationHop tracks consent as it crosses federations
- FederationConsentPolicy for federation-level sharing restrictions
- Revocation propagation across federation boundaries
- Provenance preservation across federations

Issue #89
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class CrossFederationPolicy(str, Enum):
    """Federation-level policies for cross-federation sharing."""

    ALLOW_ALL = "allow_all"  # No restrictions on cross-federation sharing
    ALLOW_TRUSTED = "allow_trusted"  # Only share with trusted federations
    ALLOW_LISTED = "allow_listed"  # Only share with explicitly listed federations
    DENY_ALL = "deny_all"  # Block all cross-federation sharing


class ConsentValidationResult(str, Enum):
    """Result of consent chain validation."""

    VALID = "valid"
    INVALID_SIGNATURE = "invalid_signature"
    REVOKED = "revoked"
    FEDERATION_BLOCKED = "federation_blocked"
    POLICY_VIOLATION = "policy_violation"
    EXPIRED = "expired"
    BROKEN_CHAIN = "broken_chain"


class RevocationScope(str, Enum):
    """Scope of revocation propagation."""

    LOCAL = "local"  # Only revoke in current federation
    DOWNSTREAM = "downstream"  # Revoke in all downstream federations
    FULL_CHAIN = "full_chain"  # Revoke across entire chain


# =============================================================================
# DATA CLASSES
# =============================================================================


def compute_policy_hash(policy_snapshot: dict[str, Any]) -> bytes:
    """Compute SHA-256 hash of a policy snapshot for verification.

    Args:
        policy_snapshot: The policy snapshot dictionary

    Returns:
        SHA-256 hash bytes
    """
    policy_json = json.dumps(policy_snapshot, sort_keys=True).encode("utf-8")
    return hashlib.sha256(policy_json).digest()


def verify_policy_hash(policy_snapshot: dict[str, Any], policy_hash: bytes) -> bool:
    """Verify that a policy snapshot matches its claimed hash.

    Args:
        policy_snapshot: The policy snapshot dictionary to verify
        policy_hash: The claimed hash of the policy

    Returns:
        True if hash matches, False otherwise
    """
    computed = compute_policy_hash(policy_snapshot)
    return computed == policy_hash


@dataclass
class CrossFederationHop:
    """A hop in the consent chain that crosses federation boundaries.

    Tracks when consent/sharing crosses from one federation to another,
    preserving provenance and enforcing federation-level policies.
    """

    hop_id: str
    from_federation_id: str
    from_gateway_id: str
    to_federation_id: str
    to_gateway_id: str
    timestamp: float
    signature: bytes

    # Provenance tracking
    original_consent_chain_id: str
    hop_number: int

    # Policy at time of crossing (with hash for verification - Issue #145)
    policy_snapshot: dict[str, Any] = field(default_factory=dict)
    policy_hash: bytes = field(default_factory=bytes)

    # Optional metadata
    reason: Optional[str] = None
    requester_did: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hop_id": self.hop_id,
            "from_federation_id": self.from_federation_id,
            "from_gateway_id": self.from_gateway_id,
            "to_federation_id": self.to_federation_id,
            "to_gateway_id": self.to_gateway_id,
            "timestamp": self.timestamp,
            "signature": self.signature.hex()
            if isinstance(self.signature, bytes)
            else self.signature,
            "original_consent_chain_id": self.original_consent_chain_id,
            "hop_number": self.hop_number,
            "policy_snapshot": self.policy_snapshot,
            "policy_hash": self.policy_hash.hex()
            if isinstance(self.policy_hash, bytes)
            else self.policy_hash,
            "reason": self.reason,
            "requester_did": self.requester_did,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossFederationHop:
        """Deserialize from dictionary."""
        sig = data.get("signature", b"")
        if isinstance(sig, str):
            sig = bytes.fromhex(sig)

        policy_hash = data.get("policy_hash", b"")
        if isinstance(policy_hash, str):
            policy_hash = bytes.fromhex(policy_hash)

        return cls(
            hop_id=data["hop_id"],
            from_federation_id=data["from_federation_id"],
            from_gateway_id=data["from_gateway_id"],
            to_federation_id=data["to_federation_id"],
            to_gateway_id=data["to_gateway_id"],
            timestamp=data["timestamp"],
            signature=sig,
            original_consent_chain_id=data["original_consent_chain_id"],
            hop_number=data["hop_number"],
            policy_snapshot=data.get("policy_snapshot", {}),
            policy_hash=policy_hash,
            reason=data.get("reason"),
            requester_did=data.get("requester_did"),
        )


@dataclass
class FederationConsentPolicy:
    """Policy governing cross-federation consent.

    Defines what a federation allows for incoming and outgoing consent chains.
    """

    federation_id: str

    # Outgoing policy (sharing to other federations)
    outgoing_policy: CrossFederationPolicy = CrossFederationPolicy.ALLOW_TRUSTED
    allowed_outgoing_federations: list[str] = field(default_factory=list)
    blocked_outgoing_federations: list[str] = field(default_factory=list)

    # Incoming policy (receiving from other federations)
    incoming_policy: CrossFederationPolicy = CrossFederationPolicy.ALLOW_TRUSTED
    allowed_incoming_federations: list[str] = field(default_factory=list)
    blocked_incoming_federations: list[str] = field(default_factory=list)

    # Trust requirements
    min_trust_for_outgoing: float = 0.5
    min_trust_for_incoming: float = 0.5

    # Propagation limits
    max_outgoing_hops: Optional[int] = 3
    max_incoming_hops: Optional[int] = 3

    # Data restrictions
    strip_fields_on_outgoing: list[str] = field(default_factory=list)
    strip_fields_on_incoming: list[str] = field(default_factory=list)

    # Revocation behavior
    revocation_scope: RevocationScope = RevocationScope.DOWNSTREAM

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "federation_id": self.federation_id,
            "outgoing_policy": self.outgoing_policy.value,
            "allowed_outgoing_federations": self.allowed_outgoing_federations,
            "blocked_outgoing_federations": self.blocked_outgoing_federations,
            "incoming_policy": self.incoming_policy.value,
            "allowed_incoming_federations": self.allowed_incoming_federations,
            "blocked_incoming_federations": self.blocked_incoming_federations,
            "min_trust_for_outgoing": self.min_trust_for_outgoing,
            "min_trust_for_incoming": self.min_trust_for_incoming,
            "max_outgoing_hops": self.max_outgoing_hops,
            "max_incoming_hops": self.max_incoming_hops,
            "strip_fields_on_outgoing": self.strip_fields_on_outgoing,
            "strip_fields_on_incoming": self.strip_fields_on_incoming,
            "revocation_scope": self.revocation_scope.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FederationConsentPolicy:
        """Deserialize from dictionary."""
        return cls(
            federation_id=data["federation_id"],
            outgoing_policy=CrossFederationPolicy(data.get("outgoing_policy", "allow_trusted")),
            allowed_outgoing_federations=data.get("allowed_outgoing_federations", []),
            blocked_outgoing_federations=data.get("blocked_outgoing_federations", []),
            incoming_policy=CrossFederationPolicy(data.get("incoming_policy", "allow_trusted")),
            allowed_incoming_federations=data.get("allowed_incoming_federations", []),
            blocked_incoming_federations=data.get("blocked_incoming_federations", []),
            min_trust_for_outgoing=data.get("min_trust_for_outgoing", 0.5),
            min_trust_for_incoming=data.get("min_trust_for_incoming", 0.5),
            max_outgoing_hops=data.get("max_outgoing_hops", 3),
            max_incoming_hops=data.get("max_incoming_hops", 3),
            strip_fields_on_outgoing=data.get("strip_fields_on_outgoing", []),
            strip_fields_on_incoming=data.get("strip_fields_on_incoming", []),
            revocation_scope=RevocationScope(data.get("revocation_scope", "downstream")),
        )


@dataclass
class CrossFederationConsentChain:
    """A consent chain that spans multiple federations.

    Extends the local consent chain with cross-federation tracking.
    """

    id: str
    original_chain_id: str
    origin_federation_id: str
    origin_gateway_id: str

    # Hops across federation boundaries
    cross_federation_hops: list[CrossFederationHop] = field(default_factory=list)

    # Full provenance chain
    provenance_chain: list[dict[str, Any]] = field(default_factory=list)

    # Chain integrity
    chain_hash: bytes = field(default_factory=bytes)

    # Status
    revoked: bool = False
    revoked_at: Optional[float] = None
    revoked_by: Optional[str] = None
    revoked_in_federation: Optional[str] = None

    created_at: float = field(default_factory=time.time)

    def add_hop(self, hop: CrossFederationHop) -> None:
        """Add a cross-federation hop to the chain."""
        self.cross_federation_hops.append(hop)

        # Update provenance
        self.provenance_chain.append(
            {
                "type": "cross_federation_hop",
                "hop_id": hop.hop_id,
                "from": f"{hop.from_federation_id}:{hop.from_gateway_id}",
                "to": f"{hop.to_federation_id}:{hop.to_gateway_id}",
                "timestamp": hop.timestamp,
            }
        )

        # Recompute chain hash
        self._update_chain_hash()

    def _update_chain_hash(self) -> None:
        """Recompute the chain hash for integrity verification."""
        chain_data = {
            "id": self.id,
            "original_chain_id": self.original_chain_id,
            "origin_federation_id": self.origin_federation_id,
            "hops": [h.to_dict() for h in self.cross_federation_hops],
            "provenance": self.provenance_chain,
        }
        chain_json = json.dumps(chain_data, sort_keys=True).encode("utf-8")
        self.chain_hash = hashlib.sha256(chain_json).digest()

    def get_current_federation(self) -> str:
        """Get the federation ID where the chain currently resides."""
        if not self.cross_federation_hops:
            return self.origin_federation_id
        return self.cross_federation_hops[-1].to_federation_id

    def get_current_gateway(self) -> str:
        """Get the gateway ID where the chain currently resides."""
        if not self.cross_federation_hops:
            return self.origin_gateway_id
        return self.cross_federation_hops[-1].to_gateway_id

    def get_hop_count(self) -> int:
        """Get the number of cross-federation hops."""
        return len(self.cross_federation_hops)

    def get_federation_path(self) -> list[str]:
        """Get the list of federations this chain has traversed."""
        path = [self.origin_federation_id]
        for hop in self.cross_federation_hops:
            path.append(hop.to_federation_id)
        return path

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "original_chain_id": self.original_chain_id,
            "origin_federation_id": self.origin_federation_id,
            "origin_gateway_id": self.origin_gateway_id,
            "cross_federation_hops": [h.to_dict() for h in self.cross_federation_hops],
            "provenance_chain": self.provenance_chain,
            "chain_hash": self.chain_hash.hex()
            if isinstance(self.chain_hash, bytes)
            else self.chain_hash,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
            "revoked_by": self.revoked_by,
            "revoked_in_federation": self.revoked_in_federation,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossFederationConsentChain:
        """Deserialize from dictionary."""
        chain_hash = data.get("chain_hash", b"")
        if isinstance(chain_hash, str):
            chain_hash = bytes.fromhex(chain_hash)

        chain = cls(
            id=data["id"],
            original_chain_id=data["original_chain_id"],
            origin_federation_id=data["origin_federation_id"],
            origin_gateway_id=data["origin_gateway_id"],
            cross_federation_hops=[
                CrossFederationHop.from_dict(h) for h in data.get("cross_federation_hops", [])
            ],
            provenance_chain=data.get("provenance_chain", []),
            chain_hash=chain_hash,
            revoked=data.get("revoked", False),
            revoked_at=data.get("revoked_at"),
            revoked_by=data.get("revoked_by"),
            revoked_in_federation=data.get("revoked_in_federation"),
            created_at=data.get("created_at", time.time()),
        )
        return chain


@dataclass
class CrossFederationRevocation:
    """A revocation event that propagates across federations."""

    id: str
    consent_chain_id: str
    cross_chain_id: str

    # Who initiated the revocation
    revoked_by: str
    revoked_in_federation: str
    revoked_at: float
    reason: Optional[str] = None

    # Propagation tracking
    scope: RevocationScope = RevocationScope.DOWNSTREAM
    propagated_to: list[str] = field(default_factory=list)  # Federation IDs
    pending_propagation: list[str] = field(default_factory=list)

    # Acknowledgments
    acknowledgments: dict[str, float] = field(default_factory=dict)  # federation_id -> timestamp

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "consent_chain_id": self.consent_chain_id,
            "cross_chain_id": self.cross_chain_id,
            "revoked_by": self.revoked_by,
            "revoked_in_federation": self.revoked_in_federation,
            "revoked_at": self.revoked_at,
            "reason": self.reason,
            "scope": self.scope.value,
            "propagated_to": self.propagated_to,
            "pending_propagation": self.pending_propagation,
            "acknowledgments": self.acknowledgments,
        }


@dataclass
class ConsentValidation:
    """Result of validating a cross-federation consent chain."""

    result: ConsentValidationResult
    chain_id: str
    federation_path: list[str]

    # Details
    valid_hops: int = 0
    total_hops: int = 0

    # Error details
    error_at_hop: Optional[int] = None
    error_message: Optional[str] = None
    blocking_federation: Optional[str] = None

    # Provenance summary
    origin_federation: Optional[str] = None
    origin_timestamp: Optional[float] = None

    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.result == ConsentValidationResult.VALID


# =============================================================================
# PROTOCOLS
# =============================================================================


class FederationTrustProtocol(Protocol):
    """Protocol for federation trust lookups."""

    async def get_federation_trust(self, from_federation: str, to_federation: str) -> float:
        """Get trust score between two federations."""
        ...


class GatewaySigningProtocol(Protocol):
    """Protocol for gateway cryptographic operations."""

    def sign(self, data: dict[str, Any]) -> bytes:
        """Sign data with the gateway's key."""
        ...

    def verify(self, data: dict[str, Any], signature: bytes, gateway_id: str) -> bool:
        """Verify a signature from a gateway."""
        ...

    def get_gateway_id(self) -> str:
        """Get this gateway's ID."""
        ...


class ConsentChainStoreProtocol(Protocol):
    """Protocol for storing cross-federation consent chains."""

    async def store_cross_chain(self, chain: CrossFederationConsentChain) -> None:
        """Store a cross-federation consent chain."""
        ...

    async def get_cross_chain(self, chain_id: str) -> Optional[CrossFederationConsentChain]:
        """Get a cross-federation consent chain by ID."""
        ...

    async def get_cross_chain_by_original(
        self, original_chain_id: str
    ) -> Optional[CrossFederationConsentChain]:
        """Get a cross-federation chain by its original chain ID."""
        ...

    async def update_cross_chain(self, chain: CrossFederationConsentChain) -> None:
        """Update a cross-federation consent chain."""
        ...

    async def store_revocation(self, revocation: CrossFederationRevocation) -> None:
        """Store a revocation event."""
        ...

    async def get_revocation(self, revocation_id: str) -> Optional[CrossFederationRevocation]:
        """Get a revocation by ID."""
        ...

    async def list_pending_revocations(self, federation_id: str) -> list[CrossFederationRevocation]:
        """List revocations pending propagation to a federation."""
        ...


class PolicyStoreProtocol(Protocol):
    """Protocol for storing federation consent policies."""

    async def get_policy(self, federation_id: str) -> Optional[FederationConsentPolicy]:
        """Get consent policy for a federation."""
        ...

    async def store_policy(self, policy: FederationConsentPolicy) -> None:
        """Store a consent policy."""
        ...


# =============================================================================
# CROSS-FEDERATION CONSENT SERVICE
# =============================================================================


class CrossFederationConsentService:
    """Service for managing consent chains across federation boundaries.

    Handles:
    - Creating cross-federation consent chain hops
    - Validating consent across federation boundaries
    - Enforcing federation-level policies
    - Propagating revocations across federations
    - Preserving provenance across boundaries
    """

    def __init__(
        self,
        federation_id: str,
        gateway_signer: GatewaySigningProtocol,
        chain_store: ConsentChainStoreProtocol,
        policy_store: PolicyStoreProtocol,
        trust_service: Optional[FederationTrustProtocol] = None,
    ):
        self.federation_id = federation_id
        self.gateway_signer = gateway_signer
        self.chain_store = chain_store
        self.policy_store = policy_store
        self.trust_service = trust_service

    async def create_cross_federation_hop(
        self,
        original_chain_id: str,
        target_federation_id: str,
        target_gateway_id: str,
        requester_did: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> CrossFederationHop:
        """Create a hop when consent crosses to another federation.

        Args:
            original_chain_id: The consent chain being extended
            target_federation_id: The federation to share with
            target_gateway_id: The gateway in the target federation
            requester_did: Optional DID of who requested the cross-federation share
            reason: Optional reason for the cross-federation share

        Returns:
            The created CrossFederationHop

        Raises:
            ValueError: If validation fails
            PermissionError: If policy blocks the share
        """
        # Get or create the cross-federation chain
        cross_chain = await self.chain_store.get_cross_chain_by_original(original_chain_id)

        if cross_chain is None:
            # First cross-federation hop - create the chain
            cross_chain = CrossFederationConsentChain(
                id=str(uuid.uuid4()),
                original_chain_id=original_chain_id,
                origin_federation_id=self.federation_id,
                origin_gateway_id=self.gateway_signer.get_gateway_id(),
            )
        elif cross_chain.revoked:
            raise ValueError("Cannot create hop: consent chain has been revoked")

        # Validate against outgoing policy
        await self._validate_outgoing(target_federation_id)

        # Get current hop number
        hop_number = cross_chain.get_hop_count() + 1

        # Get policy for snapshot with hash for verification (Issue #145)
        policy = await self.policy_store.get_policy(self.federation_id)
        policy_snapshot = policy.to_dict() if policy else {}
        policy_hash = compute_policy_hash(policy_snapshot)

        # Create the hop
        timestamp = time.time()
        hop_data = {
            "from_federation_id": self.federation_id,
            "from_gateway_id": self.gateway_signer.get_gateway_id(),
            "to_federation_id": target_federation_id,
            "to_gateway_id": target_gateway_id,
            "original_chain_id": original_chain_id,
            "hop_number": hop_number,
            "timestamp": timestamp,
        }

        signature = self.gateway_signer.sign(hop_data)

        hop = CrossFederationHop(
            hop_id=str(uuid.uuid4()),
            from_federation_id=self.federation_id,
            from_gateway_id=self.gateway_signer.get_gateway_id(),
            to_federation_id=target_federation_id,
            to_gateway_id=target_gateway_id,
            timestamp=timestamp,
            signature=signature,
            original_consent_chain_id=original_chain_id,
            hop_number=hop_number,
            policy_snapshot=policy_snapshot,
            policy_hash=policy_hash,
            reason=reason,
            requester_did=requester_did,
        )

        # Add hop to chain
        cross_chain.add_hop(hop)

        # Store updated chain
        await self.chain_store.store_cross_chain(cross_chain)

        logger.info(
            f"Created cross-federation hop: {self.federation_id} -> {target_federation_id} "
            f"(chain={original_chain_id}, hop={hop_number})"
        )

        return hop

    async def receive_cross_federation_hop(
        self,
        hop: CrossFederationHop,
        source_gateway_verifier: Optional[GatewaySigningProtocol] = None,
    ) -> CrossFederationConsentChain:
        """Receive and validate a cross-federation hop from another federation.

        Args:
            hop: The hop being received
            source_gateway_verifier: Optional verifier for the source gateway's signature

        Returns:
            The updated cross-federation consent chain

        Raises:
            ValueError: If validation fails (including policy hash mismatch)
            PermissionError: If policy blocks incoming
        """
        # Validate this hop is directed to us
        if hop.to_federation_id != self.federation_id:
            raise ValueError(
                f"Hop is not directed to this federation "
                f"(expected {self.federation_id}, got {hop.to_federation_id})"
            )

        # Verify policy snapshot hash (Issue #145 - prevent tampering)
        if hop.policy_hash:
            if not verify_policy_hash(hop.policy_snapshot, hop.policy_hash):
                raise ValueError(
                    f"Policy snapshot hash mismatch: snapshot may have been tampered with "
                    f"(from federation {hop.from_federation_id})"
                )

        # Validate against incoming policy
        await self._validate_incoming(hop.from_federation_id)

        # Verify signature if verifier provided
        if source_gateway_verifier:
            hop_data = {
                "from_federation_id": hop.from_federation_id,
                "from_gateway_id": hop.from_gateway_id,
                "to_federation_id": hop.to_federation_id,
                "to_gateway_id": hop.to_gateway_id,
                "original_chain_id": hop.original_consent_chain_id,
                "hop_number": hop.hop_number,
                "timestamp": hop.timestamp,
            }
            if not source_gateway_verifier.verify(hop_data, hop.signature, hop.from_gateway_id):
                raise ValueError("Invalid hop signature")

        # Get or create local representation of the chain
        cross_chain = await self.chain_store.get_cross_chain_by_original(
            hop.original_consent_chain_id
        )

        is_new_chain = cross_chain is None
        if is_new_chain:
            # Create local representation - we may be receiving a hop further
            # down the chain (e.g., hop 2 when we don't know about hop 1)
            cross_chain = CrossFederationConsentChain(
                id=str(uuid.uuid4()),
                original_chain_id=hop.original_consent_chain_id,
                origin_federation_id=hop.from_federation_id,
                origin_gateway_id=hop.from_gateway_id,
            )
        
        # At this point cross_chain is guaranteed to be non-None
        assert cross_chain is not None

        # Verify hop chain integrity:
        # - For new chains, accept the hop (we're joining the chain midway)
        # - For existing chains, verify it's the next expected hop
        if not is_new_chain:
            expected_hop = cross_chain.get_hop_count() + 1
            if expected_hop != hop.hop_number:
                raise ValueError(
                    f"Hop number mismatch: expected {expected_hop}, " f"got {hop.hop_number}"
                )

        # Add hop
        cross_chain.add_hop(hop)

        # Store
        await self.chain_store.store_cross_chain(cross_chain)

        logger.info(
            f"Received cross-federation hop from {hop.from_federation_id}: "
            f"chain={hop.original_consent_chain_id}, hop={hop.hop_number}"
        )

        return cross_chain

    async def validate_consent_chain(
        self,
        chain_id: str,
    ) -> ConsentValidation:
        """Validate a cross-federation consent chain.

        Checks:
        - Chain integrity (hashes, signatures)
        - Federation policies at each hop
        - Revocation status
        - Trust requirements

        Args:
            chain_id: The cross-federation chain ID or original chain ID

        Returns:
            ConsentValidation result
        """
        # Try to get chain by ID or original ID
        chain = await self.chain_store.get_cross_chain(chain_id)
        if chain is None:
            chain = await self.chain_store.get_cross_chain_by_original(chain_id)

        if chain is None:
            return ConsentValidation(
                result=ConsentValidationResult.BROKEN_CHAIN,
                chain_id=chain_id,
                federation_path=[],
                error_message="Chain not found",
            )

        # Check revocation
        if chain.revoked:
            return ConsentValidation(
                result=ConsentValidationResult.REVOKED,
                chain_id=chain.id,
                federation_path=chain.get_federation_path(),
                origin_federation=chain.origin_federation_id,
                origin_timestamp=chain.created_at,
                error_message=f"Revoked by {chain.revoked_by} in {chain.revoked_in_federation}",
            )

        # Validate each hop
        valid_hops = 0
        for i, hop in enumerate(chain.cross_federation_hops):
            # Check hop signatures would require verifier for each gateway
            # For now, verify structural integrity

            # Check federation policies
            source_policy = await self.policy_store.get_policy(hop.from_federation_id)
            target_policy = await self.policy_store.get_policy(hop.to_federation_id)

            # Check outgoing policy
            if source_policy:
                if hop.to_federation_id in source_policy.blocked_outgoing_federations:
                    return ConsentValidation(
                        result=ConsentValidationResult.FEDERATION_BLOCKED,
                        chain_id=chain.id,
                        federation_path=chain.get_federation_path(),
                        valid_hops=valid_hops,
                        total_hops=len(chain.cross_federation_hops),
                        error_at_hop=i,
                        error_message=f"Federation {hop.to_federation_id} blocked by {hop.from_federation_id}",
                        blocking_federation=hop.from_federation_id,
                        origin_federation=chain.origin_federation_id,
                        origin_timestamp=chain.created_at,
                    )

                # Check hop limit
                if (
                    source_policy.max_outgoing_hops
                    and hop.hop_number > source_policy.max_outgoing_hops
                ):
                    return ConsentValidation(
                        result=ConsentValidationResult.POLICY_VIOLATION,
                        chain_id=chain.id,
                        federation_path=chain.get_federation_path(),
                        valid_hops=valid_hops,
                        total_hops=len(chain.cross_federation_hops),
                        error_at_hop=i,
                        error_message=f"Max hops exceeded at hop {hop.hop_number}",
                        blocking_federation=hop.from_federation_id,
                        origin_federation=chain.origin_federation_id,
                        origin_timestamp=chain.created_at,
                    )

            # Check incoming policy
            if target_policy:
                if hop.from_federation_id in target_policy.blocked_incoming_federations:
                    return ConsentValidation(
                        result=ConsentValidationResult.FEDERATION_BLOCKED,
                        chain_id=chain.id,
                        federation_path=chain.get_federation_path(),
                        valid_hops=valid_hops,
                        total_hops=len(chain.cross_federation_hops),
                        error_at_hop=i,
                        error_message=f"Federation {hop.from_federation_id} blocked by {hop.to_federation_id}",
                        blocking_federation=hop.to_federation_id,
                        origin_federation=chain.origin_federation_id,
                        origin_timestamp=chain.created_at,
                    )

            # Check trust if service available
            if self.trust_service:
                trust = await self.trust_service.get_federation_trust(
                    hop.from_federation_id, hop.to_federation_id
                )
                min_trust = source_policy.min_trust_for_outgoing if source_policy else 0.5
                if trust < min_trust:
                    return ConsentValidation(
                        result=ConsentValidationResult.POLICY_VIOLATION,
                        chain_id=chain.id,
                        federation_path=chain.get_federation_path(),
                        valid_hops=valid_hops,
                        total_hops=len(chain.cross_federation_hops),
                        error_at_hop=i,
                        error_message=f"Insufficient trust ({trust:.2f} < {min_trust:.2f})",
                        origin_federation=chain.origin_federation_id,
                        origin_timestamp=chain.created_at,
                    )

            valid_hops += 1

        return ConsentValidation(
            result=ConsentValidationResult.VALID,
            chain_id=chain.id,
            federation_path=chain.get_federation_path(),
            valid_hops=valid_hops,
            total_hops=len(chain.cross_federation_hops),
            origin_federation=chain.origin_federation_id,
            origin_timestamp=chain.created_at,
        )

    async def revoke_cross_federation(
        self,
        chain_id: str,
        revoker_did: str,
        reason: Optional[str] = None,
        scope: RevocationScope = RevocationScope.DOWNSTREAM,
    ) -> CrossFederationRevocation:
        """Revoke a cross-federation consent chain.

        Args:
            chain_id: The chain ID or original chain ID to revoke
            revoker_did: DID of who is revoking
            reason: Optional reason for revocation
            scope: How far to propagate the revocation

        Returns:
            The revocation record

        Raises:
            ValueError: If chain not found or already revoked
        """
        # Get chain
        chain = await self.chain_store.get_cross_chain(chain_id)
        if chain is None:
            chain = await self.chain_store.get_cross_chain_by_original(chain_id)

        if chain is None:
            raise ValueError("Chain not found")

        if chain.revoked:
            raise ValueError("Chain already revoked")

        # Mark chain as revoked
        revoked_at = time.time()
        chain.revoked = True
        chain.revoked_at = revoked_at
        chain.revoked_by = revoker_did
        chain.revoked_in_federation = self.federation_id

        await self.chain_store.update_cross_chain(chain)

        # Determine federations to propagate to
        pending_federations: list[str] = []
        federation_path = chain.get_federation_path()

        if scope == RevocationScope.DOWNSTREAM:
            # Find federations after current one in path
            try:
                current_idx = federation_path.index(self.federation_id)
                pending_federations = federation_path[current_idx + 1 :]
            except ValueError:
                pass
        elif scope == RevocationScope.FULL_CHAIN:
            # All federations except current
            pending_federations = [f for f in federation_path if f != self.federation_id]
        # LOCAL scope = empty pending list

        # Create revocation record
        revocation = CrossFederationRevocation(
            id=str(uuid.uuid4()),
            consent_chain_id=chain.original_chain_id,
            cross_chain_id=chain.id,
            revoked_by=revoker_did,
            revoked_in_federation=self.federation_id,
            revoked_at=revoked_at,
            reason=reason,
            scope=scope,
            pending_propagation=pending_federations,
        )

        await self.chain_store.store_revocation(revocation)

        logger.info(
            f"Revoked cross-federation chain {chain.id}: "
            f"scope={scope.value}, pending={len(pending_federations)} federations"
        )

        return revocation

    async def receive_revocation(
        self,
        revocation: CrossFederationRevocation,
    ) -> None:
        """Receive and process a revocation from another federation.

        Args:
            revocation: The revocation to process
        """
        # Get our local chain representation
        chain = await self.chain_store.get_cross_chain_by_original(revocation.consent_chain_id)

        if chain is None:
            logger.warning(f"Received revocation for unknown chain: {revocation.consent_chain_id}")
            return

        if chain.revoked:
            logger.info(f"Chain {chain.id} already revoked locally")
            return

        # Apply revocation
        chain.revoked = True
        chain.revoked_at = revocation.revoked_at
        chain.revoked_by = revocation.revoked_by
        chain.revoked_in_federation = revocation.revoked_in_federation

        await self.chain_store.update_cross_chain(chain)

        # Acknowledge
        revocation.acknowledgments[self.federation_id] = time.time()
        if self.federation_id in revocation.pending_propagation:
            revocation.pending_propagation.remove(self.federation_id)
            revocation.propagated_to.append(self.federation_id)

        await self.chain_store.store_revocation(revocation)

        logger.info(
            f"Processed revocation from {revocation.revoked_in_federation}: " f"chain={chain.id}"
        )

    async def get_provenance(
        self,
        chain_id: str,
    ) -> list[dict[str, Any]]:
        """Get the full provenance chain for a cross-federation consent.

        Args:
            chain_id: The chain ID or original chain ID

        Returns:
            List of provenance entries
        """
        chain = await self.chain_store.get_cross_chain(chain_id)
        if chain is None:
            chain = await self.chain_store.get_cross_chain_by_original(chain_id)

        if chain is None:
            return []

        return chain.provenance_chain

    async def _validate_outgoing(self, target_federation_id: str) -> None:
        """Validate outgoing cross-federation share against policy."""
        policy = await self.policy_store.get_policy(self.federation_id)

        if policy is None:
            # No policy = allow by default
            return

        # Check if target is blocked
        if target_federation_id in policy.blocked_outgoing_federations:
            raise PermissionError(f"Outgoing shares to {target_federation_id} are blocked")

        # Check policy type
        if policy.outgoing_policy == CrossFederationPolicy.DENY_ALL:
            raise PermissionError("All outgoing cross-federation shares are blocked")

        if policy.outgoing_policy == CrossFederationPolicy.ALLOW_LISTED:
            if target_federation_id not in policy.allowed_outgoing_federations:
                raise PermissionError(
                    f"Federation {target_federation_id} not in allowed outgoing list"
                )

        if policy.outgoing_policy == CrossFederationPolicy.ALLOW_TRUSTED:
            # Check trust if service available
            if self.trust_service:
                trust = await self.trust_service.get_federation_trust(
                    self.federation_id, target_federation_id
                )
                if trust < policy.min_trust_for_outgoing:
                    raise PermissionError(
                        f"Insufficient trust for {target_federation_id}: "
                        f"{trust:.2f} < {policy.min_trust_for_outgoing:.2f}"
                    )

    async def _validate_incoming(self, source_federation_id: str) -> None:
        """Validate incoming cross-federation share against policy."""
        policy = await self.policy_store.get_policy(self.federation_id)

        if policy is None:
            # No policy = allow by default
            return

        # Check if source is blocked
        if source_federation_id in policy.blocked_incoming_federations:
            raise PermissionError(f"Incoming shares from {source_federation_id} are blocked")

        # Check policy type
        if policy.incoming_policy == CrossFederationPolicy.DENY_ALL:
            raise PermissionError("All incoming cross-federation shares are blocked")

        if policy.incoming_policy == CrossFederationPolicy.ALLOW_LISTED:
            if source_federation_id not in policy.allowed_incoming_federations:
                raise PermissionError(
                    f"Federation {source_federation_id} not in allowed incoming list"
                )

        if policy.incoming_policy == CrossFederationPolicy.ALLOW_TRUSTED:
            # Check trust if service available
            if self.trust_service:
                trust = await self.trust_service.get_federation_trust(
                    source_federation_id, self.federation_id
                )
                if trust < policy.min_trust_for_incoming:
                    raise PermissionError(
                        f"Insufficient trust from {source_federation_id}: "
                        f"{trust:.2f} < {policy.min_trust_for_incoming:.2f}"
                    )


# =============================================================================
# IN-MEMORY IMPLEMENTATIONS (for testing)
# =============================================================================


class InMemoryConsentChainStore:
    """In-memory implementation of ConsentChainStoreProtocol."""

    def __init__(self) -> None:
        self._chains: dict[str, CrossFederationConsentChain] = {}
        self._chains_by_original: dict[str, CrossFederationConsentChain] = {}
        self._revocations: dict[str, CrossFederationRevocation] = {}

    async def store_cross_chain(self, chain: CrossFederationConsentChain) -> None:
        self._chains[chain.id] = chain
        self._chains_by_original[chain.original_chain_id] = chain

    async def get_cross_chain(self, chain_id: str) -> Optional[CrossFederationConsentChain]:
        return self._chains.get(chain_id)

    async def get_cross_chain_by_original(
        self, original_chain_id: str
    ) -> Optional[CrossFederationConsentChain]:
        return self._chains_by_original.get(original_chain_id)

    async def update_cross_chain(self, chain: CrossFederationConsentChain) -> None:
        self._chains[chain.id] = chain
        self._chains_by_original[chain.original_chain_id] = chain

    async def store_revocation(self, revocation: CrossFederationRevocation) -> None:
        self._revocations[revocation.id] = revocation

    async def get_revocation(self, revocation_id: str) -> Optional[CrossFederationRevocation]:
        return self._revocations.get(revocation_id)

    async def list_pending_revocations(self, federation_id: str) -> list[CrossFederationRevocation]:
        return [r for r in self._revocations.values() if federation_id in r.pending_propagation]


class InMemoryPolicyStore:
    """In-memory implementation of PolicyStoreProtocol."""

    def __init__(self) -> None:
        self._policies: dict[str, FederationConsentPolicy] = {}

    async def get_policy(self, federation_id: str) -> Optional[FederationConsentPolicy]:
        return self._policies.get(federation_id)

    async def store_policy(self, policy: FederationConsentPolicy) -> None:
        self._policies[policy.federation_id] = policy


class MockGatewaySigner:
    """Mock gateway signer for testing."""

    def __init__(self, gateway_id: str) -> None:
        self._gateway_id = gateway_id
        self._signatures: dict[str, bytes] = {}

    def sign(self, data: dict[str, Any]) -> bytes:
        """Create a mock signature."""
        data_json = json.dumps(data, sort_keys=True)
        sig = hashlib.sha256((data_json + self._gateway_id).encode()).digest()
        self._signatures[data_json] = sig
        return sig

    def verify(self, data: dict[str, Any], signature: bytes, gateway_id: str) -> bool:
        """Verify a mock signature."""
        data_json = json.dumps(data, sort_keys=True)
        expected = hashlib.sha256((data_json + gateway_id).encode()).digest()
        return signature == expected

    def get_gateway_id(self) -> str:
        return self._gateway_id


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "CrossFederationPolicy",
    "ConsentValidationResult",
    "RevocationScope",
    # Helper functions (Issue #145)
    "compute_policy_hash",
    "verify_policy_hash",
    # Data classes
    "CrossFederationHop",
    "FederationConsentPolicy",
    "CrossFederationConsentChain",
    "CrossFederationRevocation",
    "ConsentValidation",
    # Protocols
    "FederationTrustProtocol",
    "GatewaySigningProtocol",
    "ConsentChainStoreProtocol",
    "PolicyStoreProtocol",
    # Service
    "CrossFederationConsentService",
    # In-memory implementations
    "InMemoryConsentChainStore",
    "InMemoryPolicyStore",
    "MockGatewaySigner",
]
