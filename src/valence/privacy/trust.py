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
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class DecayModel(str, Enum):
    """Models for how trust decays over time."""
    
    NONE = "none"              # No decay - trust stays constant
    LINEAR = "linear"          # trust - (decay_rate * days)
    EXPONENTIAL = "exponential"  # trust * (retention_rate ^ days)
    
    @classmethod
    def from_string(cls, value: str) -> "DecayModel":
        """Convert string to DecayModel, case-insensitive."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.EXPONENTIAL  # Default

# Singleton store instance
_default_store: Optional["TrustGraphStore"] = None


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
    decay_rate: float = 0.0  # 0.0 = no decay
    decay_model: DecayModel = DecayModel.EXPONENTIAL
    last_refreshed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: Optional[UUID] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
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
        """Check if this trust edge has expired."""
        if self.expires_at is None:
            return False
        now = datetime.now(timezone.utc)
        # Handle timezone-naive datetimes
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=timezone.utc)
        return now > expires
    
    @property
    def days_since_refresh(self) -> float:
        """Calculate days elapsed since last refresh."""
        now = datetime.now(timezone.utc)
        # Handle timezone-naive datetimes
        if self.last_refreshed.tzinfo is None:
            last_refreshed = self.last_refreshed.replace(tzinfo=timezone.utc)
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
            return max(0.0, value * (retention_rate ** days))
        
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
                last_refreshed = self.last_refreshed.replace(tzinfo=timezone.utc)
            else:
                last_refreshed = self.last_refreshed
            if as_of.tzinfo is None:
                as_of = as_of.replace(tzinfo=timezone.utc)
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
        values = [effective_competence, effective_integrity, 
                  effective_confidentiality, effective_judgment]
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
    ) -> "TrustEdge":
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
            timestamp = datetime.now(timezone.utc)
        
        self.last_refreshed = timestamp
        self.updated_at = timestamp
        
        if new_values:
            for dim_name, dim_value in new_values.items():
                if dim_name in ("competence", "integrity", "confidentiality", "judgment"):
                    if not 0.0 <= dim_value <= 1.0:
                        raise ValueError(
                            f"{dim_name} must be between 0.0 and 1.0, got {dim_value}"
                        )
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
    ) -> "TrustEdge":
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
            decay_rate=decay_rate,
            decay_model=decay_model,
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
    def from_dict(cls, data: dict[str, Any]) -> "TrustEdge":
        """Deserialize from dictionary."""
        # Parse timestamps
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
        
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now(timezone.utc)
        
        expires_at = data.get("expires_at")
        if isinstance(expires_at, str):
            expires_at = datetime.fromisoformat(expires_at)
        
        last_refreshed = data.get("last_refreshed")
        if isinstance(last_refreshed, str):
            last_refreshed = datetime.fromisoformat(last_refreshed)
        elif last_refreshed is None:
            last_refreshed = datetime.now(timezone.utc)
        
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
) -> TrustEdge:
    """Compute transitive trust through delegation.
    
    When A trusts B, and B trusts C, this computes A's delegated trust in C.
    The key insight: A's trust in B's judgment affects how much weight
    B's trust in C gets.
    
    Formula for each dimension:
        delegated_trust = direct_trust * (judgment_weight * delegated_value)
        
    Where judgment_weight scales B's opinions based on A's trust in B's judgment.
    
    Args:
        direct_edge: A's trust in B (the intermediary)
        delegated_edge: B's trust in C (what B thinks of C)
        
    Returns:
        A new TrustEdge representing A's delegated trust in C
    """
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
        
        # Check delegation depth limit
        if remaining_depth is not None and remaining_depth <= 0:
            continue
        
        # Found target
        if current_did == target_did:
            found_paths.append(path)
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
            delegated = compute_delegated_trust(result, next_edge)
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
        from uuid import uuid4
        
        now = datetime.now(timezone.utc)
        
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
                           domain, created_at, updated_at, expires_at
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
                           domain, created_at, updated_at, expires_at
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
                       domain, created_at, updated_at, expires_at
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
                       domain, created_at, updated_at, expires_at
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
        
        Args:
            source_did: DID of the trusting agent
            target_did: DID of the trusted agent
            domain: Optional scope to query
            
        Returns:
            TrustEdge4D if found and not expired, None otherwise
        """
        if self._use_memory:
            key = self._make_key(source_did, target_did, domain)
            edge = self._memory_store.get(key)
            if edge and not edge.is_expired():
                return edge
            elif edge and edge.is_expired():
                # Clean up expired edge
                del self._memory_store[key]
            return None
        else:
            edge = self._store.get_edge(source_did, target_did, domain)
            if edge and edge.is_expired():
                # Clean up and return None
                self._store.delete_edge(source_did, target_did, domain)
                return None
            return edge
    
    def list_trusted(
        self,
        source_did: str,
        domain: str | None = None,
    ) -> list[TrustEdge4D]:
        """List all agents trusted by the source.
        
        Args:
            source_did: DID of the trusting agent
            domain: Optional scope filter (None returns all)
            
        Returns:
            List of TrustEdge4D objects (excluding expired)
        """
        if self._use_memory:
            result = []
            expired_keys = []
            
            for key, edge in self._memory_store.items():
                src, tgt, dom = key
                if src != source_did:
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
            return self._store.get_edges_from(source_did, domain, include_expired=False)
    
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
