"""Trust graph for Valence privacy module.

Implements multi-dimensional trust edges between DIDs with database storage.
Trust has four dimensions:
- competence: Ability to perform tasks correctly
- integrity: Honesty and consistency
- confidentiality: Ability to keep secrets
- judgment: Ability to evaluate others (affects delegated trust)

The judgment dimension is special: it affects how much weight we give to
someone's trust recommendations. Low judgment means their trust in others
is weighted less in transitive trust calculations.

Trust Decay:
Trust can decay over time if not refreshed. This models the natural erosion
of trust when relationships aren't maintained. Decay is configurable via:
- decay_rate: Rate of decay per day (0.0 = no decay)
- decay_model: LINEAR (constant loss) or EXPONENTIAL (percentage loss)
- last_refreshed: When trust was last confirmed/refreshed
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Clock skew tolerance for temporal comparisons across federation nodes.
# This accounts for clock drift between nodes that may cause false expiration
# or premature rejection of otherwise valid trust edges.
# Default: 5 minutes (300 seconds). Configurable via environment or settings.
CLOCK_SKEW_TOLERANCE = timedelta(minutes=5)


class DecayModel(str, Enum):
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


# Singleton store instance
_default_store: Optional[TrustGraphStore] = None


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
        last_refreshed: When trust was last refreshed (defaults to creation time)
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
    domain: Optional[str] = None
    can_delegate: bool = False  # Default: non-transitive trust
    delegation_depth: int = 0  # 0 = no limit when can_delegate=True
    decay_rate: float = 0.0  # 0.0 = no decay
    decay_model: DecayModel = DecayModel.EXPONENTIAL
    last_refreshed: datetime = field(default_factory=lambda: datetime.now(UTC))
    id: Optional[UUID] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None

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
        return (
            self.source_did == other.source_did
            and self.target_did == other.target_did
            and self.domain == other.domain
        )

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
                if dim_name in ("competence", "integrity", "confidentiality", "judgment"):
                    if not 0.0 <= dim_value <= 1.0:
                        raise ValueError(f"{dim_name} must be between 0.0 and 1.0, got {dim_value}")
                    setattr(self, dim_name, dim_value)

        logger.debug(
            f"Refreshed trust edge {self.source_did} -> {self.target_did}: "
            f"last_refreshed={self.last_refreshed}"
        )

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
            "can_delegate": self.can_delegate,
            "delegation_depth": self.delegation_depth,
            "decay_rate": self.decay_rate,
            "decay_model": self.decay_model.value,
            "last_refreshed": self.last_refreshed.isoformat() if self.last_refreshed else None,
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


def compute_delegated_trust(
    direct_edge: TrustEdge,
    delegated_edge: TrustEdge,
    remaining_depth: int | None = None,
    respect_delegation: bool = True,
) -> TrustEdge | None:
    """Compute transitive trust through delegation.

    When A trusts B, and B trusts C, this computes A's delegated trust in C.
    The key insight: A's trust in B's judgment affects how much weight
    B's trust in C gets.

    Delegation policy is enforced when respect_delegation=True:
    - If direct_edge.can_delegate is False, returns None (no transitive trust)
    - If delegation_depth limits are exceeded, returns None

    Formula for each dimension:
        delegated_trust = direct_trust * (judgment_weight * delegated_value)

    Where judgment_weight scales B's opinions based on A's trust in B's judgment.

    Args:
        direct_edge: A's trust in B (the intermediary)
        delegated_edge: B's trust in C (what B thinks of C)
        remaining_depth: Remaining delegation hops allowed (None = use edge's depth)
        respect_delegation: If True, enforce can_delegate policy (default True)

    Returns:
        A new TrustEdge representing A's delegated trust in C, or None if
        delegation is not allowed by policy
    """
    # Check delegation policy on the direct edge (only if respecting delegation)
    if respect_delegation and not direct_edge.can_delegate:
        return None  # This edge does not allow transitive trust

    # Check depth limits
    if remaining_depth is not None and remaining_depth <= 0:
        return None  # Exceeded depth limit

    # A's trust in B's judgment determines how much we weight B's opinions
    judgment_weight = direct_edge.judgment

    # For each dimension, delegated trust is:
    # min(A's direct trust in B, B's trust in C) * judgment_weight
    # The min ensures we don't trust C more than we trust B
    # The judgment_weight scales down based on how much we trust B's judgment

    def delegate_dimension(direct_val: float, delegated_val: float) -> float:
        """Compute delegated trust for a single dimension."""
        # Can't trust C more than we trust B on this dimension
        base = min(direct_val, delegated_val)
        # Scale by how much we trust B's judgment
        return base * judgment_weight

    # Compute new delegation depth for the resulting edge
    # The result can delegate if both edges allow it
    result_can_delegate = direct_edge.can_delegate and delegated_edge.can_delegate

    # New depth is the minimum of both, decremented by 1 if either has a limit
    if direct_edge.delegation_depth == 0 and delegated_edge.delegation_depth == 0:
        result_depth = 0  # Both unlimited
    elif direct_edge.delegation_depth == 0:
        result_depth = max(0, delegated_edge.delegation_depth - 1)
    elif delegated_edge.delegation_depth == 0:
        result_depth = max(0, direct_edge.delegation_depth - 1)
    else:
        result_depth = max(
            0, min(direct_edge.delegation_depth, delegated_edge.delegation_depth) - 1
        )

    return TrustEdge(
        source_did=direct_edge.source_did,
        target_did=delegated_edge.target_did,
        competence=delegate_dimension(direct_edge.competence, delegated_edge.competence),
        integrity=delegate_dimension(direct_edge.integrity, delegated_edge.integrity),
        confidentiality=delegate_dimension(
            direct_edge.confidentiality, delegated_edge.confidentiality
        ),
        # For judgment of C, we also apply A's trust in B's judgment
        judgment=delegate_dimension(direct_edge.judgment, delegated_edge.judgment),
        domain=delegated_edge.domain,  # Use the target edge's domain
        can_delegate=result_can_delegate,
        delegation_depth=result_depth,
    )


def compute_transitive_trust(
    source_did: str,
    target_did: str,
    trust_graph: dict[tuple[str, str], TrustEdge],
    max_hops: int = 3,
    respect_delegation: bool = True,
) -> Optional[TrustEdge]:
    """Compute transitive trust through the graph.

    Uses breadth-first search to find trust paths and combines them.
    Judgment dimension affects how much each hop's trust recommendations
    are weighted.

    Delegation policy is enforced when respect_delegation=True:
    - Only edges with can_delegate=True participate in transitive trust
    - delegation_depth limits how far trust can propagate

    Args:
        source_did: Starting DID (the truster)
        target_did: Ending DID (the trustee)
        trust_graph: Dict mapping (source, target) to TrustEdge
        max_hops: Maximum path length to consider
        respect_delegation: If True, only traverse delegatable edges (default True)

    Returns:
        Combined TrustEdge if paths exist, None otherwise
    """
    # Direct trust check
    if (source_did, target_did) in trust_graph:
        return trust_graph[(source_did, target_did)]

    # BFS for paths
    from collections import deque

    # Queue entries: (current_did, path_of_edges, remaining_depth)
    # remaining_depth tracks the minimum delegation_depth limit along the path
    queue: deque[tuple[str, list[TrustEdge], int | None]] = deque()

    # Find all outgoing edges from source that can delegate (for transitive paths)
    for (src, tgt), edge in trust_graph.items():
        if src == source_did:
            if not respect_delegation or edge.can_delegate:
                # Determine initial remaining depth
                remaining = edge.delegation_depth if edge.delegation_depth > 0 else None
                queue.append((tgt, [edge], remaining))

    found_paths: list[list[TrustEdge]] = []
    visited_at_depth: dict[str, int] = {source_did: 0}

    while queue:
        current_did, path, remaining_depth = queue.popleft()
        current_depth = len(path)

        if current_depth > max_hops:
            continue

        # Found target - check BEFORE depth limit since reaching the target doesn't require another hop
        if current_did == target_did:
            found_paths.append(path)
            continue

        # Check delegation depth limit - only matters for continuing exploration
        if remaining_depth is not None and remaining_depth <= 0:
            continue

        # Explore neighbors
        for (src, tgt), edge in trust_graph.items():
            if src == current_did:
                # Only traverse delegatable edges for intermediate hops
                if respect_delegation and not edge.can_delegate:
                    # But we CAN reach the target if this is the final hop
                    if tgt == target_did:
                        # Final hop doesn't need can_delegate on the last edge
                        next_depth = current_depth + 1
                        if tgt not in visited_at_depth or visited_at_depth[tgt] >= next_depth:
                            visited_at_depth[tgt] = next_depth
                            queue.append((tgt, path + [edge], remaining_depth))
                    continue

                # Avoid cycles and redundant exploration
                next_depth = current_depth + 1
                if tgt not in visited_at_depth or visited_at_depth[tgt] >= next_depth:
                    visited_at_depth[tgt] = next_depth
                    # Update remaining depth
                    if remaining_depth is None:
                        # No limit so far
                        new_remaining = edge.delegation_depth if edge.delegation_depth > 0 else None
                    elif edge.delegation_depth == 0:
                        # This edge has no limit, keep existing limit
                        new_remaining = remaining_depth - 1 if remaining_depth else None
                    else:
                        # Both have limits, take the more restrictive
                        new_remaining = min(remaining_depth, edge.delegation_depth) - 1
                    queue.append((tgt, path + [edge], new_remaining))

    if not found_paths:
        return None

    # Compute delegated trust for each path
    path_trusts: list[TrustEdge] = []
    for path in found_paths:
        # Chain the edges
        result = path[0]
        for next_edge in path[1:]:
            delegated = compute_delegated_trust(
                result, next_edge, respect_delegation=respect_delegation
            )
            if delegated is None:
                # Delegation not allowed along this path
                break
            result = delegated
        else:
            # Only add if we successfully chained all edges
            path_trusts.append(result)

    if not path_trusts:
        return None

    # Combine paths: take max of each dimension (optimistic combination)
    # This represents "trust via the best available path"
    return TrustEdge(
        source_did=source_did,
        target_did=target_did,
        competence=max(t.competence for t in path_trusts),
        integrity=max(t.integrity for t in path_trusts),
        confidentiality=max(t.confidentiality for t in path_trusts),
        judgment=max(t.judgment for t in path_trusts),
    )


class TrustGraphStore:
    """Database storage for trust graph edges.

    Provides CRUD operations for TrustEdge persistence.
    """

    def __init__(self) -> None:
        """Initialize the store."""
        pass

    def add_edge(self, edge: TrustEdge) -> TrustEdge:
        """Add or update a trust edge.

        Uses upsert semantics - updates if edge exists, inserts otherwise.

        Args:
            edge: The trust edge to store

        Returns:
            The stored edge with ID and timestamps populated
        """

        from valence.core.db import get_cursor

        now = datetime.now(UTC)

        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO trust_edges (
                    id, source_did, target_did,
                    competence, integrity, confidentiality, judgment,
                    domain, can_delegate, delegation_depth,
                    created_at, updated_at, expires_at
                )
                VALUES (
                    COALESCE(%s, gen_random_uuid()), %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON CONFLICT (source_did, target_did, COALESCE(domain, ''))
                DO UPDATE SET
                    competence = EXCLUDED.competence,
                    integrity = EXCLUDED.integrity,
                    confidentiality = EXCLUDED.confidentiality,
                    judgment = EXCLUDED.judgment,
                    can_delegate = EXCLUDED.can_delegate,
                    delegation_depth = EXCLUDED.delegation_depth,
                    updated_at = EXCLUDED.updated_at,
                    expires_at = EXCLUDED.expires_at
                RETURNING id, created_at, updated_at
                """,
                (
                    str(edge.id) if edge.id else None,
                    edge.source_did,
                    edge.target_did,
                    edge.competence,
                    edge.integrity,
                    edge.confidentiality,
                    edge.judgment,
                    edge.domain,
                    edge.can_delegate,
                    edge.delegation_depth,
                    edge.created_at or now,
                    now,
                    edge.expires_at,
                ),
            )
            row = cur.fetchone()

            edge.id = row["id"]
            edge.created_at = row["created_at"]
            edge.updated_at = row["updated_at"]

        return edge

    def get_edge(
        self,
        source_did: str,
        target_did: str,
        domain: Optional[str] = None,
    ) -> Optional[TrustEdge]:
        """Get a specific trust edge.

        Args:
            source_did: The trusting DID
            target_did: The trusted DID
            domain: Optional domain filter

        Returns:
            The trust edge if found, None otherwise
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            if domain is not None:
                cur.execute(
                    """
                    SELECT id, source_did, target_did,
                           competence, integrity, confidentiality, judgment,
                           domain, can_delegate, delegation_depth,
                           created_at, updated_at, expires_at
                    FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain = %s
                    """,
                    (source_did, target_did, domain),
                )
            else:
                cur.execute(
                    """
                    SELECT id, source_did, target_did,
                           competence, integrity, confidentiality, judgment,
                           domain, can_delegate, delegation_depth,
                           created_at, updated_at, expires_at
                    FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain IS NULL
                    """,
                    (source_did, target_did),
                )

            row = cur.fetchone()
            if row is None:
                return None

            return TrustEdge(
                id=row["id"],
                source_did=row["source_did"],
                target_did=row["target_did"],
                competence=row["competence"],
                integrity=row["integrity"],
                confidentiality=row["confidentiality"],
                judgment=row["judgment"],
                domain=row["domain"],
                can_delegate=row.get("can_delegate", False),
                delegation_depth=row.get("delegation_depth", 0),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                expires_at=row["expires_at"],
            )

    def get_edges_from(
        self,
        source_did: str,
        domain: Optional[str] = None,
        include_expired: bool = False,
    ) -> list[TrustEdge]:
        """Get all trust edges from a DID.

        Args:
            source_did: The trusting DID
            domain: Optional domain filter
            include_expired: Whether to include expired edges

        Returns:
            List of trust edges
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            query = """
                SELECT id, source_did, target_did,
                       competence, integrity, confidentiality, judgment,
                       domain, can_delegate, delegation_depth,
                       created_at, updated_at, expires_at
                FROM trust_edges
                WHERE source_did = %s
            """
            params: list[Any] = [source_did]

            if domain is not None:
                query += " AND domain = %s"
                params.append(domain)

            if not include_expired:
                query += " AND (expires_at IS NULL OR expires_at > NOW())"

            cur.execute(query, params)
            rows = cur.fetchall()

            return [
                TrustEdge(
                    id=row["id"],
                    source_did=row["source_did"],
                    target_did=row["target_did"],
                    competence=row["competence"],
                    integrity=row["integrity"],
                    confidentiality=row["confidentiality"],
                    judgment=row["judgment"],
                    domain=row["domain"],
                    can_delegate=row.get("can_delegate", False),
                    delegation_depth=row.get("delegation_depth", 0),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    expires_at=row["expires_at"],
                )
                for row in rows
            ]

    def get_edges_to(
        self,
        target_did: str,
        domain: Optional[str] = None,
        include_expired: bool = False,
    ) -> list[TrustEdge]:
        """Get all trust edges to a DID.

        Args:
            target_did: The trusted DID
            domain: Optional domain filter
            include_expired: Whether to include expired edges

        Returns:
            List of trust edges
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            query = """
                SELECT id, source_did, target_did,
                       competence, integrity, confidentiality, judgment,
                       domain, can_delegate, delegation_depth,
                       created_at, updated_at, expires_at
                FROM trust_edges
                WHERE target_did = %s
            """
            params: list[Any] = [target_did]

            if domain is not None:
                query += " AND domain = %s"
                params.append(domain)

            if not include_expired:
                query += " AND (expires_at IS NULL OR expires_at > NOW())"

            cur.execute(query, params)
            rows = cur.fetchall()

            return [
                TrustEdge(
                    id=row["id"],
                    source_did=row["source_did"],
                    target_did=row["target_did"],
                    competence=row["competence"],
                    integrity=row["integrity"],
                    confidentiality=row["confidentiality"],
                    judgment=row["judgment"],
                    domain=row["domain"],
                    can_delegate=row.get("can_delegate", False),
                    delegation_depth=row.get("delegation_depth", 0),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    expires_at=row["expires_at"],
                )
                for row in rows
            ]

    def delete_edge(
        self,
        source_did: str,
        target_did: str,
        domain: Optional[str] = None,
    ) -> bool:
        """Delete a trust edge.

        Args:
            source_did: The trusting DID
            target_did: The trusted DID
            domain: Optional domain filter

        Returns:
            True if an edge was deleted, False otherwise
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            if domain is not None:
                cur.execute(
                    """
                    DELETE FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain = %s
                    RETURNING id
                    """,
                    (source_did, target_did, domain),
                )
            else:
                cur.execute(
                    """
                    DELETE FROM trust_edges
                    WHERE source_did = %s AND target_did = %s AND domain IS NULL
                    RETURNING id
                    """,
                    (source_did, target_did),
                )

            row = cur.fetchone()
            return row is not None

    def delete_edges_from(self, source_did: str) -> int:
        """Delete all trust edges from a DID.

        Args:
            source_did: The trusting DID

        Returns:
            Number of edges deleted
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                DELETE FROM trust_edges
                WHERE source_did = %s
                RETURNING id
                """,
                (source_did,),
            )
            rows = cur.fetchall()
            return len(rows)

    def delete_edges_to(self, target_did: str) -> int:
        """Delete all trust edges to a DID.

        Args:
            target_did: The trusted DID

        Returns:
            Number of edges deleted
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                DELETE FROM trust_edges
                WHERE target_did = %s
                RETURNING id
                """,
                (target_did,),
            )
            rows = cur.fetchall()
            return len(rows)

    def cleanup_expired(self) -> int:
        """Delete all expired trust edges.

        Returns:
            Number of edges deleted
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            cur.execute(
                """
                DELETE FROM trust_edges
                WHERE expires_at IS NOT NULL AND expires_at < NOW()
                RETURNING id
                """,
            )
            rows = cur.fetchall()
            return len(rows)

    def count_edges(
        self,
        source_did: Optional[str] = None,
        target_did: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> int:
        """Count trust edges with optional filters.

        Args:
            source_did: Optional source filter
            target_did: Optional target filter
            domain: Optional domain filter

        Returns:
            Number of matching edges
        """
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            query = "SELECT COUNT(*) as count FROM trust_edges WHERE 1=1"
            params: list[Any] = []

            if source_did is not None:
                query += " AND source_did = %s"
                params.append(source_did)

            if target_did is not None:
                query += " AND target_did = %s"
                params.append(target_did)

            if domain is not None:
                query += " AND domain = %s"
                params.append(domain)

            cur.execute(query, params)
            row = cur.fetchone()
            return row["count"]


def get_trust_graph_store() -> TrustGraphStore:
    """Get the singleton TrustGraphStore instance.

    Returns:
        The shared TrustGraphStore instance
    """
    global _default_store
    if _default_store is None:
        _default_store = TrustGraphStore()
    return _default_store


# =============================================================================
# TRUST SERVICE (Issue #59 - High-Level API)
# =============================================================================


class TrustService:
    """High-level API for managing trust relationships.

    Provides convenient methods for granting, revoking, and querying trust
    relationships. Supports both in-memory storage (for testing) and
    persistent storage via TrustGraphStore.

    Example:
        >>> service = TrustService()
        >>> service.grant_trust(
        ...     source_did="did:key:alice",
        ...     target_did="did:key:bob",
        ...     competence=0.9,
        ...     integrity=0.8,
        ...     confidentiality=0.7,
        ... )
        >>> edge = service.get_trust("did:key:alice", "did:key:bob")
        >>> print(f"Overall trust: {edge.overall_trust:.2f}")

        # List who Alice trusts
        >>> trusted = service.list_trusted("did:key:alice")

        # List who trusts Bob
        >>> trusters = service.list_trusters("did:key:bob")
    """

    def __init__(self, use_memory: bool = False):
        """Initialize the TrustService.

        Args:
            use_memory: If True, use in-memory storage (for testing).
                       If False, use database-backed TrustGraphStore.
        """
        self._use_memory = use_memory
        self._memory_store: dict[tuple[str, str, str], TrustEdge4D] = {}
        self._store: TrustGraphStore | None = None if use_memory else get_trust_graph_store()

    def _make_key(
        self,
        source_did: str,
        target_did: str,
        domain: str | None,
    ) -> tuple[str, str, str]:
        """Create a consistent key for in-memory storage."""
        return (source_did, target_did, domain or "")

    def grant_trust(
        self,
        source_did: str,
        target_did: str,
        competence: float,
        integrity: float,
        confidentiality: float,
        judgment: float = 0.1,
        domain: str | None = None,
        can_delegate: bool = False,
        delegation_depth: int = 0,
        expires_at: datetime | None = None,
    ) -> TrustEdge4D:
        """Grant trust from source to target.

        Creates a new trust edge or updates an existing one.

        Args:
            source_did: DID of the trusting agent
            target_did: DID of the agent being trusted
            competence: Trust in target's ability to perform correctly (0-1)
            integrity: Trust in target's honesty and reliability (0-1)
            confidentiality: Trust in target's ability to keep secrets (0-1)
            judgment: Trust in target's decision-making quality (0-1, default 0.1)
            domain: Optional scope/context for this trust relationship
            can_delegate: Whether trust can be transitively delegated (default False)
            delegation_depth: Maximum delegation chain length (0 = no limit when can_delegate=True)
            expires_at: Optional expiration time

        Returns:
            The created or updated TrustEdge4D

        Raises:
            ValueError: If source_did equals target_did, or scores are invalid
        """
        edge = TrustEdge4D(
            source_did=source_did,
            target_did=target_did,
            competence=competence,
            integrity=integrity,
            confidentiality=confidentiality,
            judgment=judgment,
            domain=domain,
            can_delegate=can_delegate,
            delegation_depth=delegation_depth,
            expires_at=expires_at,
        )

        if self._use_memory:
            key = self._make_key(source_did, target_did, domain)
            existing = self._memory_store.get(key)
            if existing:
                edge.created_at = existing.created_at
                edge.id = existing.id
            self._memory_store[key] = edge
            return edge
        else:
            return self._store.add_edge(edge)

    def revoke_trust(
        self,
        source_did: str,
        target_did: str,
        domain: str | None = None,
    ) -> bool:
        """Revoke trust from source to target.

        Removes the trust edge if it exists.

        Args:
            source_did: DID of the trusting agent
            target_did: DID whose trust is being revoked
            domain: Optional scope to revoke (None revokes general trust)

        Returns:
            True if an edge was revoked, False if not found
        """
        if self._use_memory:
            key = self._make_key(source_did, target_did, domain)
            if key in self._memory_store:
                del self._memory_store[key]
                return True
            return False
        else:
            return self._store.delete_edge(source_did, target_did, domain)

    def get_trust(
        self,
        source_did: str,
        target_did: str,
        domain: str | None = None,
    ) -> TrustEdge4D | None:
        """Get the trust edge from source to target.

        When a domain is specified, first checks for a domain-specific edge,
        then falls back to the global (domain=None) edge if not found.
        Domain-scoped trust overrides global trust for that domain.

        Args:
            source_did: DID of the trusting agent
            target_did: DID of the trusted agent
            domain: Optional scope to query. If specified, will check for
                   domain-specific edge first, then fall back to global.

        Returns:
            TrustEdge4D if found and not expired, None otherwise
        """
        if self._use_memory:
            # Try domain-specific edge first (if domain specified)
            if domain is not None:
                key = self._make_key(source_did, target_did, domain)
                edge = self._memory_store.get(key)
                if edge and not edge.is_expired():
                    return edge
                elif edge and edge.is_expired():
                    del self._memory_store[key]

                # Fall back to global edge
                global_key = self._make_key(source_did, target_did, None)
                global_edge = self._memory_store.get(global_key)
                if global_edge and not global_edge.is_expired():
                    return global_edge
                elif global_edge and global_edge.is_expired():
                    del self._memory_store[global_key]
                return None
            else:
                # Just look for global edge
                key = self._make_key(source_did, target_did, None)
                edge = self._memory_store.get(key)
                if edge and not edge.is_expired():
                    return edge
                elif edge and edge.is_expired():
                    del self._memory_store[key]
                return None
        else:
            # Try domain-specific edge first (if domain specified)
            if domain is not None:
                edge = self._store.get_edge(source_did, target_did, domain)
                if edge and not edge.is_expired():
                    return edge
                elif edge and edge.is_expired():
                    self._store.delete_edge(source_did, target_did, domain)

                # Fall back to global edge
                global_edge = self._store.get_edge(source_did, target_did, None)
                if global_edge and not global_edge.is_expired():
                    return global_edge
                elif global_edge and global_edge.is_expired():
                    self._store.delete_edge(source_did, target_did, None)
                return None
            else:
                # Just look for global edge
                edge = self._store.get_edge(source_did, target_did, None)
                if edge and edge.is_expired():
                    self._store.delete_edge(source_did, target_did, None)
                    return None
                return edge

    def list_trusted(
        self,
        source_did: str,
        domain: str | None = None,
    ) -> list[TrustEdge4D]:
        """List all agents trusted by the source.

        When a domain is specified, returns the effective trust edges for that
        domain context. For each target, returns the domain-specific edge if it
        exists, otherwise returns the global edge. Domain-scoped trust overrides
        global trust for that domain.

        Args:
            source_did: DID of the trusting agent
            domain: Optional scope filter. If specified, returns effective trust
                   for that domain (domain-specific edges override global).
                   If None, returns all edges (both global and domain-scoped).

        Returns:
            List of TrustEdge4D objects (excluding expired)
        """
        if self._use_memory:
            result = []
            expired_keys = []

            if domain is not None:
                # Collect all edges from this source, separating by target
                # For each target: domain-specific edge overrides global
                global_edges: dict[str, TrustEdge4D] = {}  # target -> edge
                domain_edges: dict[str, TrustEdge4D] = {}  # target -> edge

                for key, edge in self._memory_store.items():
                    src, tgt, dom = key
                    if src != source_did:
                        continue

                    if edge.is_expired():
                        expired_keys.append(key)
                        continue

                    if dom == domain:
                        domain_edges[tgt] = edge
                    elif dom == "":  # Global (domain is None stored as "")
                        global_edges[tgt] = edge

                # Domain-specific edges override global for each target
                all_targets = set(global_edges.keys()) | set(domain_edges.keys())
                for target in all_targets:
                    if target in domain_edges:
                        result.append(domain_edges[target])
                    elif target in global_edges:
                        result.append(global_edges[target])
            else:
                # Return all edges (no domain filter)
                for key, edge in self._memory_store.items():
                    src, tgt, dom = key
                    if src != source_did:
                        continue

                    if edge.is_expired():
                        expired_keys.append(key)
                        continue

                    result.append(edge)

            # Clean up expired
            for key in expired_keys:
                del self._memory_store[key]

            return result
        else:
            if domain is not None:
                # Get both domain-specific and global edges
                domain_edges = self._store.get_edges_from(source_did, domain, include_expired=False)
                global_edges = self._store.get_edges_from(source_did, None, include_expired=False)

                # Build result: domain-specific overrides global for each target
                domain_by_target = {e.target_did: e for e in domain_edges}
                global_by_target = {e.target_did: e for e in global_edges}

                result = []
                all_targets = set(domain_by_target.keys()) | set(global_by_target.keys())
                for target in all_targets:
                    if target in domain_by_target:
                        result.append(domain_by_target[target])
                    elif target in global_by_target:
                        result.append(global_by_target[target])

                return result
            else:
                # Return all edges (no domain filter)
                return self._store.get_edges_from(source_did, None, include_expired=False)

    def list_trusters(
        self,
        target_did: str,
        domain: str | None = None,
    ) -> list[TrustEdge4D]:
        """List all agents who trust the target.

        Args:
            target_did: DID of the trusted agent
            domain: Optional scope filter (None returns all)

        Returns:
            List of TrustEdge4D objects (excluding expired)
        """
        if self._use_memory:
            result = []
            expired_keys = []

            for key, edge in self._memory_store.items():
                src, tgt, dom = key
                if tgt != target_did:
                    continue

                # Filter by domain if specified
                if domain is not None and (dom or None) != domain:
                    continue

                if edge.is_expired():
                    expired_keys.append(key)
                    continue

                result.append(edge)

            # Clean up expired
            for key in expired_keys:
                del self._memory_store[key]

            return result
        else:
            return self._store.get_edges_to(target_did, domain, include_expired=False)

    def compute_delegated_trust(
        self,
        source: str,
        target: str,
        domain: str | None = None,
    ) -> TrustEdge4D | None:
        """Compute transitive trust through delegation chains with decay.

        Finds paths from source to target through edges where can_delegate=True,
        applies decay at each hop based on the intermediary's judgment score,
        and respects delegation_depth limits.

        The decay formula at each hop:
            delegated_trust[dim] = min(current[dim], next[dim]) * current.judgment

        This captures the intuition that:
        1. We can't trust the target more than we trust the intermediary
        2. The intermediary's judgment affects how much we weight their recommendation

        Args:
            source: Source DID (the entity seeking trust information)
            target: Target DID (the entity being evaluated)
            domain: Optional domain to scope the trust lookup

        Returns:
            TrustEdge representing delegated trust from source to target,
            or None if no valid delegation path exists.

        Example:
            >>> service = TrustService(use_memory=True)
            >>> # Alice trusts Bob with high judgment, allows delegation depth 2
            >>> edge = TrustEdge(
            ...     source_did="alice", target_did="bob",
            ...     competence=0.8, integrity=0.8, confidentiality=0.8,
            ...     judgment=0.9, can_delegate=True, delegation_depth=2
            ... )
            >>> service._memory_store[("alice", "bob", "")] = edge
            >>> # Bob trusts Carol
            >>> edge2 = TrustEdge(
            ...     source_did="bob", target_did="carol",
            ...     competence=0.9, integrity=0.9, confidentiality=0.9, judgment=0.8
            ... )
            >>> service._memory_store[("bob", "carol", "")] = edge2
            >>> # Compute Alice's delegated trust in Carol
            >>> delegated = service.compute_delegated_trust("alice", "carol")
            >>> # Decay: min(0.8, 0.9) * 0.9 = 0.72 for competence
        """
        from collections import deque

        # Check for direct trust first
        direct = self.get_trust(source, target, domain)
        if direct is not None:
            return direct

        def get_effective_edges(src_did: str) -> list[TrustEdge4D]:
            """Get edges for the given source, filtered by domain.

            For delegated trust computation:
            - If domain is None: only return global (domain-less) edges
            - If domain is specified: use list_trusted's override behavior
              (domain-specific edges take precedence, with global fallback)
            """
            if domain is None:
                # For global trust computation, only use global edges
                all_edges = self.list_trusted(src_did, None)
                return [e for e in all_edges if e.domain is None]
            else:
                # For domain-specific, use the override behavior
                return self.list_trusted(src_did, domain)

        # BFS to find delegation paths
        # Queue entries: (current_did, path_of_edges, remaining_depth)
        # remaining_depth tracks the minimum delegation_depth remaining in the chain
        # None means no limit (unlimited delegation)
        queue: deque[tuple[str, list[TrustEdge4D], int | None]] = deque()

        # Get initial edges from source that allow delegation
        source_edges = get_effective_edges(source)
        for edge in source_edges:
            if edge.can_delegate:
                # delegation_depth=0 means no limit, otherwise it's the max hops allowed
                initial_depth: int | None = (
                    edge.delegation_depth if edge.delegation_depth > 0 else None
                )
                queue.append((edge.target_did, [edge], initial_depth))

        found_paths: list[list[TrustEdge4D]] = []
        # Track visited nodes with (depth, remaining_limit) to avoid inferior paths
        visited: dict[str, tuple[int, int | None]] = {}

        while queue:
            current_did, path, remaining_depth = queue.popleft()
            current_hops = len(path)

            # Found target - collect ALL paths to target (don't prune)
            if current_did == target:
                found_paths.append(path)
                continue

            # Check if we've found a better path to this intermediate node already
            # (only prune intermediate nodes, not the target)
            if current_did in visited:
                prev_hops, prev_remaining = visited[current_did]
                # Skip if we've reached this node in fewer hops
                if prev_hops < current_hops:
                    continue
                # Skip if same hops but better remaining depth
                if prev_hops == current_hops:
                    if prev_remaining is None:  # Unlimited is better
                        continue
                    if remaining_depth is not None and prev_remaining >= remaining_depth:
                        continue

            visited[current_did] = (current_hops, remaining_depth)

            # Can't go further if depth exhausted
            if remaining_depth is not None and remaining_depth <= 0:
                continue

            # Get outgoing edges from current node
            current_edges = get_effective_edges(current_did)
            for edge in current_edges:
                # For intermediate hops, edge must allow delegation
                # For final hop to target, we don't require can_delegate on the last edge
                if edge.target_did == target:
                    # Final hop - doesn't need can_delegate
                    new_remaining = remaining_depth - 1 if remaining_depth is not None else None
                    queue.append((edge.target_did, path + [edge], new_remaining))
                elif edge.can_delegate:
                    # Intermediate hop - must allow delegation
                    # Calculate new remaining depth
                    if remaining_depth is None:
                        # No limit from upstream; use this edge's limit if any
                        new_remaining = edge.delegation_depth if edge.delegation_depth > 0 else None
                    elif edge.delegation_depth == 0:
                        # This edge has no limit; decrement upstream limit
                        new_remaining = remaining_depth - 1
                    else:
                        # Both have limits; take the more restrictive and decrement
                        new_remaining = min(remaining_depth, edge.delegation_depth) - 1

                    # Only continue if we have depth left
                    if new_remaining is None or new_remaining >= 0:
                        queue.append((edge.target_did, path + [edge], new_remaining))

        if not found_paths:
            return None

        # Compute delegated trust for each path with decay
        path_trusts: list[TrustEdge4D] = []
        for path in found_paths:
            # Start with the first edge
            result = path[0]

            # Chain through each subsequent edge, applying decay
            for next_edge in path[1:]:
                result = compute_delegated_trust(result, next_edge)

            path_trusts.append(result)

        # Combine paths: take max of each dimension (optimistic combination)
        # This represents "trust via the best available path"
        return TrustEdge4D(
            source_did=source,
            target_did=target,
            competence=max(t.competence for t in path_trusts),
            integrity=max(t.integrity for t in path_trusts),
            confidentiality=max(t.confidentiality for t in path_trusts),
            judgment=max(t.judgment for t in path_trusts),
            domain=domain,
        )

    def clear(self) -> int:
        """Clear all trust edges (mainly for testing).

        Returns:
            Number of edges cleared
        """
        if self._use_memory:
            count = len(self._memory_store)
            self._memory_store.clear()
            return count
        else:
            # Note: This deletes ALL edges in the database - use with caution
            from valence.core.db import get_cursor

            with get_cursor() as cur:
                cur.execute("DELETE FROM trust_edges RETURNING id")
                count = len(cur.fetchall())
            logger.warning(f"Cleared all {count} trust edges from database")
            return count


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS (Issue #59)
# =============================================================================


# Default service instance (in-memory for easy testing)
_default_service: TrustService | None = None


def get_trust_service(use_memory: bool = True) -> TrustService:
    """Get the default TrustService instance.

    Args:
        use_memory: If True, use in-memory storage. If False, use database.
                   Only affects the first call (instance is cached).

    Returns:
        The TrustService singleton
    """
    global _default_service
    if _default_service is None:
        _default_service = TrustService(use_memory=use_memory)
    return _default_service


def grant_trust(
    source_did: str,
    target_did: str,
    competence: float,
    integrity: float,
    confidentiality: float,
    judgment: float = 0.1,
    domain: str | None = None,
    can_delegate: bool = False,
    delegation_depth: int = 0,
    expires_at: datetime | None = None,
) -> TrustEdge4D:
    """Grant trust (convenience function using default service)."""
    return get_trust_service().grant_trust(
        source_did=source_did,
        target_did=target_did,
        competence=competence,
        integrity=integrity,
        confidentiality=confidentiality,
        judgment=judgment,
        domain=domain,
        can_delegate=can_delegate,
        delegation_depth=delegation_depth,
        expires_at=expires_at,
    )


def revoke_trust(
    source_did: str,
    target_did: str,
    domain: str | None = None,
) -> bool:
    """Revoke trust (convenience function using default service)."""
    return get_trust_service().revoke_trust(
        source_did=source_did,
        target_did=target_did,
        domain=domain,
    )


def get_trust(
    source_did: str,
    target_did: str,
    domain: str | None = None,
) -> TrustEdge4D | None:
    """Get trust (convenience function using default service)."""
    return get_trust_service().get_trust(
        source_did=source_did,
        target_did=target_did,
        domain=domain,
    )


def list_trusted(
    source_did: str,
    domain: str | None = None,
) -> list[TrustEdge4D]:
    """List trusted agents (convenience function using default service)."""
    return get_trust_service().list_trusted(
        source_did=source_did,
        domain=domain,
    )


def list_trusters(
    target_did: str,
    domain: str | None = None,
) -> list[TrustEdge4D]:
    """List trusters (convenience function using default service)."""
    return get_trust_service().list_trusters(
        target_did=target_did,
        domain=domain,
    )


def compute_delegated_trust_from_service(
    source: str,
    target: str,
    domain: str | None = None,
) -> TrustEdge4D | None:
    """Compute delegated trust through the trust graph (convenience function).

    Finds paths from source to target through delegatable edges, applies
    decay at each hop based on the intermediary's judgment, and respects
    delegation_depth limits.

    Args:
        source: Source DID (the entity seeking trust information)
        target: Target DID (the entity being evaluated)
        domain: Optional domain to scope the trust lookup

    Returns:
        TrustEdge representing delegated trust, or None if no path exists.
    """
    return get_trust_service().compute_delegated_trust(
        source=source,
        target=target,
        domain=domain,
    )


# =============================================================================
# FEDERATION TRUST EDGES (Issue #86)
# =============================================================================

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
    domain: Optional[str] = None
    id: Optional[UUID] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None

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
        return (
            self.source_federation == other.source_federation
            and self.target_federation == other.target_federation
            and self.domain == other.domain
        )

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
    def from_trust_edge(
        cls, edge: TrustEdge, inheritance_factor: float = 0.5
    ) -> FederationTrustEdge:
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

        logger.debug(
            f"Set federation trust: {source_federation} -> {target_federation}, "
            f"competence={competence}, inheritance={inheritance_factor}"
        )

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
            federation_trust = self.get_federation_trust(source_fed, target_fed, domain)

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

    # Attach methods to TrustService class
    TrustService.set_federation_trust = set_federation_trust
    TrustService.get_federation_trust = get_federation_trust
    TrustService.revoke_federation_trust = revoke_federation_trust
    TrustService.list_federation_trusts_from = list_federation_trusts_from
    TrustService.list_federation_trusts_to = list_federation_trusts_to
    TrustService.get_effective_trust_with_federation = get_effective_trust_with_federation


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
    return get_trust_service().set_federation_trust(
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
    return get_trust_service().get_federation_trust(
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
    return get_trust_service().revoke_federation_trust(
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
    return get_trust_service().get_effective_trust_with_federation(
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
