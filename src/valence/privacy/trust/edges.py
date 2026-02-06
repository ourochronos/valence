"""Trust edge types and decay models for the trust graph.

This module contains the core data structures for representing trust relationships:
- DecayModel: Enum for how trust decays over time
- TrustEdge: An edge in the trust graph with multi-dimensional trust scores
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Clock skew tolerance for temporal comparisons across federation nodes.
# This accounts for clock drift between nodes that may cause false expiration
# or premature rejection of otherwise valid trust edges.
# Default: 5 minutes (300 seconds). Configurable via environment or settings.
CLOCK_SKEW_TOLERANCE = timedelta(minutes=5)


class RelationshipType(StrEnum):
    """Types of relationships in the trust graph.

    Separates 'seeing content' from 'giving reputation weight':

    | Relationship | See content? | Reputation boost? | Affects worldview? |
    |---|---|---|---|
    | TRUST | Yes | Yes | Yes |
    | WATCH | Yes | No  | No  |
    | DISTRUST | Optional | Negative | Inverse |
    | IGNORE | No | No  | No  |
    """

    TRUST = "trust"  # Full trust: content visible, positive reputation, affects worldview
    WATCH = "watch"  # Attention only: content visible, no reputation or worldview effect
    DISTRUST = "distrust"  # Negative trust: optional content, negative reputation, inverse worldview
    IGNORE = "ignore"  # Block: no content, no reputation, no worldview effect

    @classmethod
    def from_string(cls, value: str) -> RelationshipType:
        """Convert string to RelationshipType, case-insensitive."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.TRUST  # Default to trust for backward compat

    @property
    def shows_content(self) -> bool:
        """Whether this relationship type makes content visible."""
        return self in (RelationshipType.TRUST, RelationshipType.WATCH)

    @property
    def affects_reputation(self) -> bool:
        """Whether this relationship type contributes to reputation scores."""
        return self in (RelationshipType.TRUST, RelationshipType.DISTRUST)

    @property
    def reputation_sign(self) -> int:
        """Sign of reputation contribution: +1, 0, or -1."""
        if self == RelationshipType.TRUST:
            return 1
        elif self == RelationshipType.DISTRUST:
            return -1
        return 0

    @property
    def affects_worldview(self) -> bool:
        """Whether this relationship type affects the entity's worldview."""
        return self in (RelationshipType.TRUST, RelationshipType.DISTRUST)


class DecayModel(StrEnum):
    """Models for how trust decays over time."""

    NONE = "none"  # No decay - trust stays constant
    LINEAR = "linear"  # trust - (decay_rate * days)
    EXPONENTIAL = "exponential"  # trust * (retention_rate ^ days)

    @classmethod
    def from_string(cls, value: str) -> DecayModel:
        """Convert string to DecayModel, case-insensitive."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.EXPONENTIAL  # Default


@dataclass
class TrustEdge:
    """An edge in the trust graph with multi-dimensional trust scores and decay.

    Represents direct trust from one DID to another across four dimensions,
    with optional time-based decay that reduces trust over time unless refreshed.

    Attributes:
        source_did: The trusting entity's DID
        target_did: The trusted entity's DID
        competence: Trust in their ability (0.0 to 1.0, default 0.5)
        integrity: Trust in their honesty (0.0 to 1.0, default 0.5)
        confidentiality: Trust in their discretion (0.0 to 1.0, default 0.5)
        judgment: Trust in their ability to evaluate others (0.0 to 1.0, default 0.1)
                  Very low default because trusting someone's judgment about
                  third parties requires more evidence than direct trust.
        domain: Optional domain-specific trust context
        decay_rate: Rate of trust decay per day (0.0 = no decay, default)
                    For EXPONENTIAL: daily loss rate (0.1 = 10% loss/day)
                    For LINEAR: absolute daily loss (0.1 = -0.1/day)
        decay_model: How decay is applied (NONE, LINEAR, EXPONENTIAL)
        last_refreshed: When trust was last confirmed/refreshed
        id: Database ID (set after persistence)
        created_at: When this edge was created
        updated_at: When this edge was last updated
        expires_at: Optional expiration time
    """

    source_did: str
    target_did: str
    competence: float = 0.5
    integrity: float = 0.5
    confidentiality: float = 0.5
    judgment: float = 0.1  # Very low default - trust in judgment must be earned
    domain: str | None = None
    relationship_type: RelationshipType = RelationshipType.TRUST  # Default for backward compat
    can_delegate: bool = False  # Default: non-transitive trust
    delegation_depth: int = 0  # 0 = no limit when can_delegate=True
    decay_rate: float = 0.0  # 0.0 = no decay
    decay_model: DecayModel = DecayModel.EXPONENTIAL
    last_refreshed: datetime = field(default_factory=lambda: datetime.now(UTC))
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
        ]:
            if not 0.0 <= dim_value <= 1.0:
                raise ValueError(f"{dim_name} must be between 0.0 and 1.0, got {dim_value}")

        # Validate decay_rate
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be between 0.0 and 1.0, got {self.decay_rate}")

        # Validate delegation_depth
        if self.delegation_depth < 0:
            raise ValueError(f"delegation_depth must be >= 0, got {self.delegation_depth}")

        # Convert string decay_model to enum if needed
        if isinstance(self.decay_model, str):
            self.decay_model = DecayModel.from_string(self.decay_model)

        # Convert string relationship_type to enum if needed
        if isinstance(self.relationship_type, str):
            self.relationship_type = RelationshipType.from_string(self.relationship_type)

        # Cannot trust yourself (that's not a trust edge)
        if self.source_did == self.target_did:
            raise ValueError("Cannot create trust edge to self")

    def __hash__(self) -> int:
        """Hash for use in sets and as dict keys."""
        return hash((self.source_did, self.target_did, self.domain))

    def __eq__(self, other: object) -> bool:
        """Equality based on DIDs and domain."""
        if not isinstance(other, TrustEdge):
            return False
        return self.source_did == other.source_did and self.target_did == other.target_did and self.domain == other.domain

    @property
    def overall_trust(self) -> float:
        """Calculate overall trust as geometric mean of all dimensions.

        Geometric mean ensures that zero in any dimension results in zero overall.
        This captures the intuition that trust requires all dimensions.
        """
        values = [self.competence, self.integrity, self.confidentiality, self.judgment]

        # Handle zeros - any zero dimension means zero overall trust
        if any(v == 0 for v in values):
            return 0.0

        # Geometric mean: nth root of product
        product = 1.0
        for v in values:
            product *= v

        return product ** (1.0 / len(values))

    def is_expired(self) -> bool:
        """Check if this trust edge has expired.

        Includes clock skew tolerance to handle clock drift between federation nodes.
        A trust edge is only considered expired if the current time exceeds
        expires_at + CLOCK_SKEW_TOLERANCE.
        """
        if self.expires_at is None:
            return False
        now = datetime.now(UTC)
        # Handle timezone-naive datetimes
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=UTC)
        # Add clock skew tolerance to handle drift between federation nodes
        return now > expires + CLOCK_SKEW_TOLERANCE

    @property
    def days_since_refresh(self) -> float:
        """Calculate days elapsed since last refresh."""
        now = datetime.now(UTC)
        # Handle timezone-naive datetimes
        if self.last_refreshed.tzinfo is None:
            last_refreshed = self.last_refreshed.replace(tzinfo=UTC)
        else:
            last_refreshed = self.last_refreshed
        delta = now - last_refreshed
        return delta.total_seconds() / 86400.0  # seconds per day

    def _apply_decay_to_value(self, value: float, days: float) -> float:
        """Apply decay to a single trust dimension value.

        Args:
            value: The base trust value
            days: Days since last refresh

        Returns:
            The decayed value
        """
        if self.decay_rate == 0.0 or self.decay_model == DecayModel.NONE:
            return value

        if self.decay_model == DecayModel.LINEAR:
            return max(0.0, value - (self.decay_rate * days))

        elif self.decay_model == DecayModel.EXPONENTIAL:
            retention_rate = 1.0 - self.decay_rate
            return max(0.0, value * (retention_rate**days))

        return value

    def effective_trust(self, as_of: datetime | None = None) -> dict[str, float]:
        """Calculate effective trust for all dimensions after applying decay.

        Args:
            as_of: Calculate decay as of this time (default: now)

        Returns:
            Dict with effective values for each dimension plus 'overall'

        The formula depends on decay_model:
        - NONE: Returns base values unchanged
        - LINEAR: value - (decay_rate * days)
        - EXPONENTIAL: value * ((1 - decay_rate) ^ days)
        """
        # Calculate days since refresh
        if as_of is not None:
            if self.last_refreshed.tzinfo is None:
                last_refreshed = self.last_refreshed.replace(tzinfo=UTC)
            else:
                last_refreshed = self.last_refreshed
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=UTC)
            delta = as_of - last_refreshed
            days = max(0.0, delta.total_seconds() / 86400.0)
        else:
            days = max(0.0, self.days_since_refresh)

        # Apply decay to each dimension
        effective_competence = self._apply_decay_to_value(self.competence, days)
        effective_integrity = self._apply_decay_to_value(self.integrity, days)
        effective_confidentiality = self._apply_decay_to_value(self.confidentiality, days)
        effective_judgment = self._apply_decay_to_value(self.judgment, days)

        # Calculate overall from effective values
        values = [
            effective_competence,
            effective_integrity,
            effective_confidentiality,
            effective_judgment,
        ]
        if any(v == 0 for v in values):
            effective_overall = 0.0
        else:
            product = 1.0
            for v in values:
                product *= v
            effective_overall = product ** (1.0 / len(values))

        return {
            "competence": effective_competence,
            "integrity": effective_integrity,
            "confidentiality": effective_confidentiality,
            "judgment": effective_judgment,
            "overall": effective_overall,
        }

    def refresh_trust(
        self,
        new_values: dict[str, float] | None = None,
        timestamp: datetime | None = None,
    ) -> TrustEdge:
        """Refresh the trust, resetting the decay clock.

        Args:
            new_values: Optionally update trust dimension values
                       Keys can be: competence, integrity, confidentiality, judgment
            timestamp: Time of refresh (default: now)

        Returns:
            Self for method chaining

        Example:
            >>> edge.refresh_trust()  # Reset decay clock only
            >>> edge.refresh_trust({"competence": 0.9})  # Update and reset
        """
        if timestamp is None:
            timestamp = datetime.now(UTC)

        self.last_refreshed = timestamp
        self.updated_at = timestamp

        if new_values:
            for dim_name, dim_value in new_values.items():
                if dim_name in (
                    "competence",
                    "integrity",
                    "confidentiality",
                    "judgment",
                ):
                    if not 0.0 <= dim_value <= 1.0:
                        raise ValueError(f"{dim_name} must be between 0.0 and 1.0, got {dim_value}")
                    setattr(self, dim_name, dim_value)

        logger.debug(f"Refreshed trust edge {self.source_did} -> {self.target_did}: last_refreshed={self.last_refreshed}")

        return self

    def with_decay(
        self,
        decay_rate: float,
        decay_model: DecayModel = DecayModel.EXPONENTIAL,
    ) -> TrustEdge:
        """Create a copy of this edge with decay settings.

        Args:
            decay_rate: The decay rate to set
            decay_model: The decay model to use

        Returns:
            New TrustEdge with decay settings applied
        """
        return TrustEdge(
            source_did=self.source_did,
            target_did=self.target_did,
            competence=self.competence,
            integrity=self.integrity,
            confidentiality=self.confidentiality,
            judgment=self.judgment,
            domain=self.domain,
            relationship_type=self.relationship_type,
            can_delegate=self.can_delegate,
            delegation_depth=self.delegation_depth,
            decay_rate=decay_rate,
            decay_model=decay_model,
            last_refreshed=self.last_refreshed,
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.expires_at,
        )

    def with_delegation(
        self,
        can_delegate: bool = True,
        delegation_depth: int = 0,
    ) -> TrustEdge:
        """Create a copy of this edge with delegation settings.

        Args:
            can_delegate: Whether trust can be transitively delegated
            delegation_depth: Maximum delegation chain length (0 = no limit)

        Returns:
            New TrustEdge with delegation settings applied

        Example:
            >>> # Allow 2-hop delegation
            >>> delegatable_edge = edge.with_delegation(can_delegate=True, delegation_depth=2)
        """
        return TrustEdge(
            source_did=self.source_did,
            target_did=self.target_did,
            competence=self.competence,
            integrity=self.integrity,
            confidentiality=self.confidentiality,
            judgment=self.judgment,
            domain=self.domain,
            relationship_type=self.relationship_type,
            can_delegate=can_delegate,
            delegation_depth=delegation_depth,
            decay_rate=self.decay_rate,
            decay_model=self.decay_model,
            last_refreshed=self.last_refreshed,
            id=self.id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            expires_at=self.expires_at,
        )

    def is_stale(self, min_effective_trust: float = 0.1) -> bool:
        """Check if trust has decayed below a threshold.

        Args:
            min_effective_trust: Minimum acceptable overall trust (default 0.1)

        Returns:
            True if effective overall trust is below threshold
        """
        effective = self.effective_trust()
        return effective["overall"] < min_effective_trust

    def time_until_stale(
        self,
        min_effective_trust: float = 0.1,
    ) -> float | None:
        """Calculate days until overall trust decays below threshold.

        Uses the overall_trust as the basis for calculation.

        Args:
            min_effective_trust: The threshold to check against

        Returns:
            Days until stale, or None if trust won't decay below threshold
        """
        if self.decay_rate == 0.0 or self.decay_model == DecayModel.NONE:
            return None  # Will never become stale from decay

        current_overall = self.overall_trust
        if current_overall <= min_effective_trust:
            return 0.0  # Already stale

        if self.decay_model == DecayModel.LINEAR:
            # overall - (decay_rate * days) = min_effective_trust
            # days = (overall - min_effective_trust) / decay_rate
            days_total = (current_overall - min_effective_trust) / self.decay_rate
            days_elapsed = self.days_since_refresh
            return max(0.0, days_total - days_elapsed)

        elif self.decay_model == DecayModel.EXPONENTIAL:
            # overall * (retention ^ days) = min_effective_trust
            # days = log(min / overall) / log(retention)
            retention = 1.0 - self.decay_rate
            if retention <= 0:
                return 0.0
            if retention >= 1.0:
                return None
            ratio = min_effective_trust / current_overall
            if ratio >= 1.0:
                return 0.0
            days_total = math.log(ratio) / math.log(retention)
            days_elapsed = self.days_since_refresh
            return max(0.0, days_total - days_elapsed)

        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        effective = self.effective_trust()
        return {
            "source_did": self.source_did,
            "target_did": self.target_did,
            "competence": self.competence,
            "integrity": self.integrity,
            "confidentiality": self.confidentiality,
            "judgment": self.judgment,
            "domain": self.domain,
            "relationship_type": self.relationship_type.value,
            "can_delegate": self.can_delegate,
            "delegation_depth": self.delegation_depth,
            "decay_rate": self.decay_rate,
            "decay_model": self.decay_model.value,
            "last_refreshed": (self.last_refreshed.isoformat() if self.last_refreshed else None),
            "id": str(self.id) if self.id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "overall_trust": self.overall_trust,
            "effective_trust": effective,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrustEdge:
        """Deserialize from dictionary."""
        # Parse timestamps
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

        last_refreshed = data.get("last_refreshed")
        if isinstance(last_refreshed, str):
            last_refreshed = datetime.fromisoformat(last_refreshed)
        elif last_refreshed is None:
            last_refreshed = datetime.now(UTC)

        # Parse decay model
        decay_model = data.get("decay_model", "exponential")
        if isinstance(decay_model, str):
            decay_model = DecayModel.from_string(decay_model)

        # Parse relationship type
        relationship_type = data.get("relationship_type", "trust")
        if isinstance(relationship_type, str):
            relationship_type = RelationshipType.from_string(relationship_type)

        # Parse UUID
        edge_id = data.get("id")
        if isinstance(edge_id, str):
            edge_id = UUID(edge_id)

        return cls(
            source_did=data["source_did"],
            target_did=data["target_did"],
            competence=float(data.get("competence", 0.5)),
            integrity=float(data.get("integrity", 0.5)),
            confidentiality=float(data.get("confidentiality", 0.5)),
            judgment=float(data.get("judgment", 0.1)),
            domain=data.get("domain"),
            relationship_type=relationship_type,
            can_delegate=bool(data.get("can_delegate", False)),
            delegation_depth=int(data.get("delegation_depth", 0)),
            decay_rate=float(data.get("decay_rate", 0.0)),
            decay_model=decay_model,
            last_refreshed=last_refreshed,
            id=edge_id,
            created_at=created_at,
            updated_at=updated_at,
            expires_at=expires_at,
        )


# Alias for backward compatibility - TrustEdge is the 4D trust schema
TrustEdge4D = TrustEdge
