"""Federation trust relationships.

This module contains types and functions for managing trust between federations:
- FederationTrustEdge: Trust relationship between federations
- FederationMembershipRegistry: Registry tracking DID-to-federation membership
- Federation trust service methods and convenience functions
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from .edges import CLOCK_SKEW_TOLERANCE, TrustEdge

if TYPE_CHECKING:
    from .service import TrustService

logger = logging.getLogger(__name__)

# Federation ID prefix for storage in trust graph
FEDERATION_PREFIX = "federation:"


@dataclass
class FederationTrustEdge:
    """Trust relationship between federations.

    Similar to TrustEdge but represents trust between federations rather than
    individual DIDs. When two federations have a trust relationship, members
    of one federation inherit a base level of trust toward members of the other.

    Federation trust is stored in the same trust graph but with a `federation:`
    prefix on the IDs to distinguish them from individual DID trust edges.

    Attributes:
        source_federation: The trusting federation's ID
        target_federation: The trusted federation's ID
        competence: Trust in the federation's collective competence (0.0-1.0)
        integrity: Trust in the federation's integrity (0.0-1.0)
        confidentiality: Trust in the federation's discretion (0.0-1.0)
        judgment: Trust in the federation's ability to vouch for members (0.0-1.0)
        inheritance_factor: How much federation trust is inherited by members (0.0-1.0)
                           Default 0.5 means members get 50% of federation trust as base
        domain: Optional domain scope for the trust relationship
        id: Database ID (set after persistence)
        created_at: When this edge was created
        updated_at: When this edge was last updated
        expires_at: Optional expiration time

    Example:
        >>> # Federation A trusts Federation B
        >>> fed_trust = FederationTrustEdge(
        ...     source_federation="acme-corp",
        ...     target_federation="globex-inc",
        ...     competence=0.8,
        ...     integrity=0.9,
        ...     confidentiality=0.7,
        ...     judgment=0.6,
        ...     inheritance_factor=0.5,
        ... )
        >>> # Members of acme-corp will have base trust of 0.4, 0.45, 0.35 toward
        >>> # members of globex-inc (50% of federation trust)
    """

    source_federation: str
    target_federation: str
    competence: float = 0.5
    integrity: float = 0.5
    confidentiality: float = 0.5
    judgment: float = 0.3  # Moderate default - federation's judgment about members
    inheritance_factor: float = 0.5  # How much members inherit (0.5 = 50%)
    domain: str | None = None
    id: UUID | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: datetime | None = None

    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        # Validate all trust scores
        for dim_name, dim_value in [
            ("competence", self.competence),
            ("integrity", self.integrity),
            ("confidentiality", self.confidentiality),
            ("judgment", self.judgment),
            ("inheritance_factor", self.inheritance_factor),
        ]:
            if not 0.0 <= dim_value <= 1.0:
                raise ValueError(f"{dim_name} must be between 0.0 and 1.0, got {dim_value}")

        # Cannot trust yourself (federation self-trust is implicit)
        if self.source_federation == self.target_federation:
            raise ValueError("Cannot create federation trust edge to self")

    def __hash__(self) -> int:
        """Hash for use in sets and as dict keys."""
        return hash((self.source_federation, self.target_federation, self.domain))

    def __eq__(self, other: object) -> bool:
        """Equality based on federation IDs and domain."""
        if not isinstance(other, FederationTrustEdge):
            return False
        return self.source_federation == other.source_federation and self.target_federation == other.target_federation and self.domain == other.domain

    @property
    def overall_trust(self) -> float:
        """Calculate overall trust as geometric mean of trust dimensions."""
        values = [self.competence, self.integrity, self.confidentiality, self.judgment]
        if any(v == 0 for v in values):
            return 0.0
        product = 1.0
        for v in values:
            product *= v
        return product ** (1.0 / len(values))

    @property
    def source_did(self) -> str:
        """Return source as DID-style identifier for storage compatibility."""
        return f"{FEDERATION_PREFIX}{self.source_federation}"

    @property
    def target_did(self) -> str:
        """Return target as DID-style identifier for storage compatibility."""
        return f"{FEDERATION_PREFIX}{self.target_federation}"

    def is_expired(self) -> bool:
        """Check if this trust edge has expired.

        Includes clock skew tolerance to handle clock drift between federation nodes.
        A trust edge is only considered expired if the current time exceeds
        expires_at + CLOCK_SKEW_TOLERANCE.
        """
        if self.expires_at is None:
            return False
        now = datetime.now(UTC)
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        # Add clock skew tolerance to handle drift between federation nodes
        return now > expires + CLOCK_SKEW_TOLERANCE

    def to_trust_edge(self) -> TrustEdge:
        """Convert to a TrustEdge for storage.

        Uses federation: prefix on DIDs so federation edges can be stored
        alongside regular DID trust edges in the same graph.
        """
        return TrustEdge(
            source_did=self.source_did,
            target_did=self.target_did,
            competence=self.competence,
            integrity=self.integrity,
            confidentiality=self.confidentiality,
            judgment=self.judgment,
            domain=self.domain,
            can_delegate=False,  # Federation trust doesn't delegate transitively
            delegation_depth=0,
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.expires_at,
        )

    @classmethod
    def from_trust_edge(cls, edge: TrustEdge, inheritance_factor: float = 0.5) -> FederationTrustEdge:
        """Create a FederationTrustEdge from a TrustEdge.

        Strips the federation: prefix from source/target DIDs.

        Args:
            edge: The TrustEdge to convert
            inheritance_factor: The inheritance factor to apply (not stored in TrustEdge)

        Returns:
            FederationTrustEdge instance

        Raises:
            ValueError: If the edge doesn't have federation: prefix on both DIDs
        """
        if not edge.source_did.startswith(FEDERATION_PREFIX):
            raise ValueError(f"Source DID must start with {FEDERATION_PREFIX}")
        if not edge.target_did.startswith(FEDERATION_PREFIX):
            raise ValueError(f"Target DID must start with {FEDERATION_PREFIX}")

        return cls(
            source_federation=edge.source_did[len(FEDERATION_PREFIX) :],
            target_federation=edge.target_did[len(FEDERATION_PREFIX) :],
            competence=edge.competence,
            integrity=edge.integrity,
            confidentiality=edge.confidentiality,
            judgment=edge.judgment,
            inheritance_factor=inheritance_factor,
            domain=edge.domain,
            id=edge.id,
            created_at=edge.created_at,
            updated_at=edge.updated_at,
            expires_at=edge.expires_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_federation": self.source_federation,
            "target_federation": self.target_federation,
            "competence": self.competence,
            "integrity": self.integrity,
            "confidentiality": self.confidentiality,
            "judgment": self.judgment,
            "inheritance_factor": self.inheritance_factor,
            "domain": self.domain,
            "id": str(self.id) if self.id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "overall_trust": self.overall_trust,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FederationTrustEdge:
        """Deserialize from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(UTC)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now(UTC)

        expires_at = data.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)

        edge_id = data.get("id")
        if isinstance(edge_id, str):
            edge_id = UUID(edge_id)

        return cls(
            source_federation=data["source_federation"],
            target_federation=data["target_federation"],
            competence=float(data.get("competence", 0.5)),
            integrity=float(data.get("integrity", 0.5)),
            confidentiality=float(data.get("confidentiality", 0.5)),
            judgment=float(data.get("judgment", 0.3)),
            inheritance_factor=float(data.get("inheritance_factor", 0.5)),
            domain=data.get("domain"),
            id=edge_id,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
        )


class FederationMembershipRegistry:
    """Registry tracking which DIDs belong to which federations.

    This is a simple in-memory registry for mapping DIDs to federation IDs.
    In a production system, this would be backed by a database or external
    federation membership service.

    The registry is used when computing effective trust between two DIDs
    to check if federation-level trust should apply.
    """

    def __init__(self) -> None:
        """Initialize empty registry."""
        # DID -> federation_id mapping
        self._memberships: dict[str, str] = {}
        # federation_id -> set of DIDs
        self._federation_members: dict[str, set[str]] = {}

    def register_member(self, did: str, federation_id: str) -> None:
        """Register a DID as a member of a federation.

        Args:
            did: The DID to register
            federation_id: The federation the DID belongs to
        """
        # Remove from old federation if any
        old_fed = self._memberships.get(did)
        if old_fed and old_fed != federation_id:
            self._federation_members.get(old_fed, set()).discard(did)

        # Add to new federation
        self._memberships[did] = federation_id
        if federation_id not in self._federation_members:
            self._federation_members[federation_id] = set()
        self._federation_members[federation_id].add(did)

    def unregister_member(self, did: str) -> bool:
        """Remove a DID from its federation.

        Args:
            did: The DID to unregister

        Returns:
            True if the DID was unregistered, False if not found
        """
        fed_id = self._memberships.pop(did, None)
        if fed_id:
            self._federation_members.get(fed_id, set()).discard(did)
            return True
        return False

    def get_federation(self, did: str) -> str | None:
        """Get the federation ID for a DID.

        Args:
            did: The DID to look up

        Returns:
            Federation ID or None if not a member of any federation
        """
        return self._memberships.get(did)

    def get_members(self, federation_id: str) -> set[str]:
        """Get all DIDs in a federation.

        Args:
            federation_id: The federation to query

        Returns:
            Set of DIDs (may be empty)
        """
        return self._federation_members.get(federation_id, set()).copy()

    def clear(self) -> None:
        """Clear all memberships."""
        self._memberships.clear()
        self._federation_members.clear()


# Global federation membership registry
_federation_registry: FederationMembershipRegistry | None = None


def get_federation_registry() -> FederationMembershipRegistry:
    """Get the global federation membership registry."""
    global _federation_registry
    if _federation_registry is None:
        _federation_registry = FederationMembershipRegistry()
    return _federation_registry


# =============================================================================
# FEDERATION TRUST SERVICE METHODS (added to TrustService)
# =============================================================================


def _extend_trust_service() -> None:
    """Extend TrustService with federation trust methods.

    This is called at module load to add federation methods to TrustService.
    Using this pattern to keep the federation code together while extending
    the existing class.
    """
    from .service import TrustService

    def set_federation_trust(
        self: TrustService,
        source_federation: str,
        target_federation: str,
        competence: float,
        integrity: float,
        confidentiality: float,
        judgment: float = 0.3,
        inheritance_factor: float = 0.5,
        domain: str | None = None,
        expires_at: datetime | None = None,
    ) -> FederationTrustEdge:
        """Set trust from one federation to another.

        Creates or updates a federation trust relationship. Members of the source
        federation will inherit a base level of trust toward members of the target
        federation, scaled by the inheritance_factor.

        Args:
            source_federation: The trusting federation's ID
            target_federation: The trusted federation's ID
            competence: Trust in the federation's competence (0-1)
            integrity: Trust in the federation's integrity (0-1)
            confidentiality: Trust in the federation's discretion (0-1)
            judgment: Trust in the federation's member vetting (0-1, default 0.3)
            inheritance_factor: How much trust members inherit (0-1, default 0.5)
            domain: Optional domain scope
            expires_at: Optional expiration time

        Returns:
            The created/updated FederationTrustEdge

        Example:
            >>> service.set_federation_trust(
            ...     source_federation="acme-corp",
            ...     target_federation="globex-inc",
            ...     competence=0.8,
            ...     integrity=0.9,
            ...     confidentiality=0.7,
            ... )
        """
        fed_edge = FederationTrustEdge(
            source_federation=source_federation,
            target_federation=target_federation,
            competence=competence,
            integrity=integrity,
            confidentiality=confidentiality,
            judgment=judgment,
            inheritance_factor=inheritance_factor,
            domain=domain,
            expires_at=expires_at,
        )

        # Convert to TrustEdge for storage
        trust_edge = fed_edge.to_trust_edge()

        # Store using existing grant_trust mechanism
        stored = self.grant_trust(
            source_did=trust_edge.source_did,
            target_did=trust_edge.target_did,
            competence=trust_edge.competence,
            integrity=trust_edge.integrity,
            confidentiality=trust_edge.confidentiality,
            judgment=trust_edge.judgment,
            domain=trust_edge.domain,
            can_delegate=False,
            expires_at=trust_edge.expires_at,
        )

        # Return as FederationTrustEdge
        fed_edge.id = stored.id
        fed_edge.created_at = stored.created_at
        fed_edge.updated_at = stored.updated_at

        logger.debug(f"Set federation trust: {source_federation} -> {target_federation}, competence={competence}, inheritance={inheritance_factor}")

        return fed_edge

    def get_federation_trust(
        self: TrustService,
        source_federation: str,
        target_federation: str,
        domain: str | None = None,
    ) -> FederationTrustEdge | None:
        """Get trust from one federation to another.

        Args:
            source_federation: The trusting federation's ID
            target_federation: The trusted federation's ID
            domain: Optional domain scope

        Returns:
            FederationTrustEdge if found and not expired, None otherwise
        """
        source_did = f"{FEDERATION_PREFIX}{source_federation}"
        target_did = f"{FEDERATION_PREFIX}{target_federation}"

        edge = self.get_trust(source_did, target_did, domain)
        if edge is None:
            return None

        # Convert back to FederationTrustEdge
        # Note: inheritance_factor is not stored, defaults to 0.5
        return FederationTrustEdge.from_trust_edge(edge)

    def revoke_federation_trust(
        self: TrustService,
        source_federation: str,
        target_federation: str,
        domain: str | None = None,
    ) -> bool:
        """Revoke trust from one federation to another.

        Args:
            source_federation: The trusting federation's ID
            target_federation: The trusted federation's ID
            domain: Optional domain scope

        Returns:
            True if revoked, False if not found
        """
        source_did = f"{FEDERATION_PREFIX}{source_federation}"
        target_did = f"{FEDERATION_PREFIX}{target_federation}"

        result = self.revoke_trust(source_did, target_did, domain)
        if result:
            logger.debug(f"Revoked federation trust: {source_federation} -> {target_federation}")
        return result

    def list_federation_trusts_from(
        self: TrustService,
        source_federation: str,
        domain: str | None = None,
    ) -> list[FederationTrustEdge]:
        """List all federations trusted by a federation.

        Args:
            source_federation: The trusting federation's ID
            domain: Optional domain scope

        Returns:
            List of FederationTrustEdge objects
        """
        source_did = f"{FEDERATION_PREFIX}{source_federation}"
        edges = self.list_trusted(source_did, domain)

        result = []
        for edge in edges:
            if edge.target_did.startswith(FEDERATION_PREFIX):
                try:
                    result.append(FederationTrustEdge.from_trust_edge(edge))
                except ValueError:
                    pass  # Not a federation edge
        return result

    def list_federation_trusts_to(
        self: TrustService,
        target_federation: str,
        domain: str | None = None,
    ) -> list[FederationTrustEdge]:
        """List all federations that trust a federation.

        Args:
            target_federation: The trusted federation's ID
            domain: Optional domain scope

        Returns:
            List of FederationTrustEdge objects
        """
        target_did = f"{FEDERATION_PREFIX}{target_federation}"
        edges = self.list_trusters(target_did, domain)

        result = []
        for edge in edges:
            if edge.source_did.startswith(FEDERATION_PREFIX):
                try:
                    result.append(FederationTrustEdge.from_trust_edge(edge))
                except ValueError:
                    pass  # Not a federation edge
        return result

    def get_effective_trust_with_federation(
        self: TrustService,
        source_did: str,
        target_did: str,
        domain: str | None = None,
        registry: FederationMembershipRegistry | None = None,
        inheritance_factor: float = 0.5,
    ) -> TrustEdge | None:
        """Get effective trust considering federation relationships.

        Computes trust from source to target, taking into account any
        federation-level trust that might apply. If both DIDs are members
        of federations with a trust relationship, the federation trust
        provides a base level that can be inherited.

        The effective trust formula:
        - If direct trust exists: use direct trust
        - If no direct trust but federation trust exists:
          effective = federation_trust * inheritance_factor
        - If both exist: use max of direct and inherited

        Args:
            source_did: The trusting DID
            target_did: The trusted DID
            domain: Optional domain scope
            registry: Federation membership registry (uses global if None)
            inheritance_factor: How much federation trust is inherited (0-1)

        Returns:
            TrustEdge representing effective trust, or None if no trust exists
        """
        if registry is None:
            registry = get_federation_registry()

        # Get direct trust first
        direct_trust = self.get_trust(source_did, target_did, domain)

        # Look up federation memberships
        source_fed = registry.get_federation(source_did)
        target_fed = registry.get_federation(target_did)

        # If both DIDs are in federations, check for federation trust
        federation_trust: FederationTrustEdge | None = None
        if source_fed and target_fed and source_fed != target_fed:
            federation_trust = self.get_federation_trust(source_fed, target_fed, domain)  # type: ignore[attr-defined]

        # If no federation trust, return direct trust (may be None)
        if federation_trust is None:
            return direct_trust

        # Compute inherited trust from federation
        inherited = TrustEdge(
            source_did=source_did,
            target_did=target_did,
            competence=federation_trust.competence * inheritance_factor,
            integrity=federation_trust.integrity * inheritance_factor,
            confidentiality=federation_trust.confidentiality * inheritance_factor,
            judgment=federation_trust.judgment * inheritance_factor,
            domain=domain,
        )

        # If no direct trust, return inherited
        if direct_trust is None:
            return inherited

        # Both exist: combine by taking max of each dimension
        # This ensures direct trust can override federation trust in either direction
        return TrustEdge(
            source_did=source_did,
            target_did=target_did,
            competence=max(direct_trust.competence, inherited.competence),
            integrity=max(direct_trust.integrity, inherited.integrity),
            confidentiality=max(direct_trust.confidentiality, inherited.confidentiality),
            judgment=max(direct_trust.judgment, inherited.judgment),
            domain=domain,
            can_delegate=direct_trust.can_delegate,
            delegation_depth=direct_trust.delegation_depth,
            id=direct_trust.id,
            created_at=direct_trust.created_at,
            updated_at=direct_trust.updated_at,
            expires_at=direct_trust.expires_at,
        )

    # Attach methods to TrustService class (dynamically extended)
    TrustService.set_federation_trust = set_federation_trust  # type: ignore[attr-defined]
    TrustService.get_federation_trust = get_federation_trust  # type: ignore[attr-defined]
    TrustService.revoke_federation_trust = revoke_federation_trust  # type: ignore[attr-defined]
    TrustService.list_federation_trusts_from = list_federation_trusts_from  # type: ignore[attr-defined]
    TrustService.list_federation_trusts_to = list_federation_trusts_to  # type: ignore[attr-defined]
    TrustService.get_effective_trust_with_federation = get_effective_trust_with_federation  # type: ignore[attr-defined]


# Extend TrustService at module load
_extend_trust_service()


# =============================================================================
# FEDERATION TRUST CONVENIENCE FUNCTIONS
# =============================================================================


def set_federation_trust(
    source_federation: str,
    target_federation: str,
    competence: float,
    integrity: float,
    confidentiality: float,
    judgment: float = 0.3,
    inheritance_factor: float = 0.5,
    domain: str | None = None,
    expires_at: datetime | None = None,
) -> FederationTrustEdge:
    """Set federation trust (convenience function using default service)."""
    from .service import get_trust_service

    return get_trust_service().set_federation_trust(  # type: ignore[attr-defined]
        source_federation=source_federation,
        target_federation=target_federation,
        competence=competence,
        integrity=integrity,
        confidentiality=confidentiality,
        judgment=judgment,
        inheritance_factor=inheritance_factor,
        domain=domain,
        expires_at=expires_at,
    )


def get_federation_trust(
    source_federation: str,
    target_federation: str,
    domain: str | None = None,
) -> FederationTrustEdge | None:
    """Get federation trust (convenience function using default service)."""
    from .service import get_trust_service

    return get_trust_service().get_federation_trust(  # type: ignore[attr-defined]
        source_federation=source_federation,
        target_federation=target_federation,
        domain=domain,
    )


def revoke_federation_trust(
    source_federation: str,
    target_federation: str,
    domain: str | None = None,
) -> bool:
    """Revoke federation trust (convenience function using default service)."""
    from .service import get_trust_service

    return get_trust_service().revoke_federation_trust(  # type: ignore[attr-defined]
        source_federation=source_federation,
        target_federation=target_federation,
        domain=domain,
    )


def get_effective_trust_with_federation(
    source_did: str,
    target_did: str,
    domain: str | None = None,
    registry: FederationMembershipRegistry | None = None,
    inheritance_factor: float = 0.5,
) -> TrustEdge | None:
    """Get effective trust considering federation (convenience function)."""
    from .service import get_trust_service

    return get_trust_service().get_effective_trust_with_federation(  # type: ignore[attr-defined]
        source_did=source_did,
        target_did=target_did,
        domain=domain,
        registry=registry,
        inheritance_factor=inheritance_factor,
    )


def register_federation_member(did: str, federation_id: str) -> None:
    """Register a DID as a member of a federation."""
    get_federation_registry().register_member(did, federation_id)


def unregister_federation_member(did: str) -> bool:
    """Unregister a DID from its federation."""
    return get_federation_registry().unregister_member(did)


def get_did_federation(did: str) -> str | None:
    """Get the federation ID for a DID."""
    return get_federation_registry().get_federation(did)
