"""Trust decay for Valence privacy module.

Implements time-based trust decay that reduces trust over time unless refreshed.
Trust represents the confidence in sharing beliefs with another entity.

Key concepts:
- decay_rate: How much trust is retained per day (0.0 = no decay, 0.9 = 10% loss/day)
- last_refreshed: Timestamp when trust was last refreshed
- effective_trust: The actual usable trust after applying decay

Decay Models:
- NONE: No decay (decay_rate = 0.0)
- LINEAR: Trust decreases linearly: trust - (decay_rate * days)
- EXPONENTIAL: Trust decreases exponentially: trust * (retention_rate ^ days)

Example:
    >>> edge = TrustEdge(
    ...     from_node_id=uuid4(),
    ...     to_node_id=uuid4(),
    ...     direct_trust=0.8,
    ...     decay_rate=0.1,  # 10% decay per day
    ...     decay_model=DecayModel.EXPONENTIAL,
    ... )
    >>> edge.effective_trust()  # Returns 0.8 (just created)
    >>> # After 7 days without refresh:
    >>> # 0.8 * (0.9 ^ 7) = 0.8 * 0.478 = 0.382
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
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


@dataclass
class TrustEdge:
    """An edge in the trust graph with time-based decay support.
    
    Represents direct trust from one node to another, with optional
    time-based decay that reduces effective trust over time unless refreshed.
    
    Attributes:
        from_node_id: The trusting node's UUID
        to_node_id: The trusted node's UUID
        direct_trust: Base trust level (0.0 to 1.0)
        domain: Optional domain-specific trust context
        decay_rate: Rate of trust decay per day (0.0 = no decay, default)
                    For EXPONENTIAL: this is the daily loss rate (0.1 = 10% loss/day)
                    For LINEAR: this is the absolute daily loss (0.1 = -0.1/day)
        decay_model: How decay is applied (NONE, LINEAR, EXPONENTIAL)
        last_refreshed: When trust was last refreshed (defaults to creation time)
        created_at: When this trust edge was created
    """
    
    from_node_id: UUID
    to_node_id: UUID
    direct_trust: float  # 0.0 to 1.0
    domain: str | None = None
    decay_rate: float = 0.0  # 0.0 = no decay
    decay_model: DecayModel = DecayModel.EXPONENTIAL
    last_refreshed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self) -> None:
        """Validate fields after initialization."""
        if not 0.0 <= self.direct_trust <= 1.0:
            raise ValueError(f"direct_trust must be between 0.0 and 1.0, got {self.direct_trust}")
        if not 0.0 <= self.decay_rate <= 1.0:
            raise ValueError(f"decay_rate must be between 0.0 and 1.0, got {self.decay_rate}")
        # Convert string decay_model to enum if needed
        if isinstance(self.decay_model, str):
            self.decay_model = DecayModel.from_string(self.decay_model)
    
    def __hash__(self) -> int:
        """Hash for use in sets and as dict keys."""
        return hash((self.from_node_id, self.to_node_id, self.domain))
    
    def __eq__(self, other: object) -> bool:
        """Equality based on nodes and domain."""
        if not isinstance(other, TrustEdge):
            return False
        return (
            self.from_node_id == other.from_node_id
            and self.to_node_id == other.to_node_id
            and self.domain == other.domain
        )
    
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
    
    def effective_trust(self, as_of: datetime | None = None) -> float:
        """Calculate the effective trust after applying decay.
        
        Args:
            as_of: Calculate decay as of this time (default: now)
            
        Returns:
            The effective trust value (0.0 to direct_trust)
            
        The formula depends on decay_model:
        - NONE: Returns direct_trust unchanged
        - LINEAR: direct_trust - (decay_rate * days)
        - EXPONENTIAL: direct_trust * ((1 - decay_rate) ^ days)
        
        When decay_rate is 0.0, returns direct_trust regardless of model.
        """
        # No decay case
        if self.decay_rate == 0.0 or self.decay_model == DecayModel.NONE:
            return self.direct_trust
        
        # Calculate days since refresh
        if as_of is not None:
            # Handle timezone-naive datetimes
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
        
        # Apply decay based on model
        if self.decay_model == DecayModel.LINEAR:
            # Linear: trust decreases by decay_rate per day
            # effective = direct_trust - (decay_rate * days)
            decayed = self.direct_trust - (self.decay_rate * days)
            return max(0.0, decayed)
        
        elif self.decay_model == DecayModel.EXPONENTIAL:
            # Exponential: trust multiplied by retention factor per day
            # retention_rate = 1 - decay_rate
            # effective = direct_trust * (retention_rate ^ days)
            retention_rate = 1.0 - self.decay_rate
            decayed = self.direct_trust * (retention_rate ** days)
            return max(0.0, decayed)
        
        # Fallback (shouldn't reach here)
        return self.direct_trust
    
    def refresh_trust(
        self,
        new_trust: float | None = None,
        timestamp: datetime | None = None,
    ) -> "TrustEdge":
        """Refresh the trust, resetting the decay clock.
        
        Args:
            new_trust: Optionally update the direct_trust value
            timestamp: Time of refresh (default: now)
            
        Returns:
            Self for method chaining
            
        Example:
            >>> edge.refresh_trust()  # Reset decay clock, keep trust
            >>> edge.refresh_trust(new_trust=0.9)  # Update trust and reset clock
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self.last_refreshed = timestamp
        
        if new_trust is not None:
            if not 0.0 <= new_trust <= 1.0:
                raise ValueError(f"new_trust must be between 0.0 and 1.0, got {new_trust}")
            self.direct_trust = new_trust
        
        logger.debug(
            f"Refreshed trust edge {self.from_node_id} -> {self.to_node_id}: "
            f"trust={self.direct_trust}, last_refreshed={self.last_refreshed}"
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
            from_node_id=self.from_node_id,
            to_node_id=self.to_node_id,
            direct_trust=self.direct_trust,
            domain=self.domain,
            decay_rate=decay_rate,
            decay_model=decay_model,
            last_refreshed=self.last_refreshed,
            created_at=self.created_at,
        )
    
    def is_stale(self, min_effective_trust: float = 0.1) -> bool:
        """Check if trust has decayed below a threshold.
        
        Args:
            min_effective_trust: Minimum acceptable trust (default 0.1)
            
        Returns:
            True if effective trust is below threshold
        """
        return self.effective_trust() < min_effective_trust
    
    def time_until_stale(
        self,
        min_effective_trust: float = 0.1,
    ) -> float | None:
        """Calculate days until trust decays below threshold.
        
        Args:
            min_effective_trust: The threshold to check against
            
        Returns:
            Days until stale, or None if trust won't decay below threshold
        """
        if self.decay_rate == 0.0 or self.decay_model == DecayModel.NONE:
            return None  # Will never become stale
        
        if self.direct_trust <= min_effective_trust:
            return 0.0  # Already stale
        
        if self.decay_model == DecayModel.LINEAR:
            # direct_trust - (decay_rate * days) = min_effective_trust
            # days = (direct_trust - min_effective_trust) / decay_rate
            days_total = (self.direct_trust - min_effective_trust) / self.decay_rate
            days_elapsed = self.days_since_refresh
            return max(0.0, days_total - days_elapsed)
        
        elif self.decay_model == DecayModel.EXPONENTIAL:
            # direct_trust * (retention ^ days) = min_effective_trust
            # retention ^ days = min_effective_trust / direct_trust
            # days * log(retention) = log(min_effective_trust / direct_trust)
            # days = log(min / direct) / log(retention)
            import math
            retention = 1.0 - self.decay_rate
            if retention <= 0:
                return 0.0
            if retention >= 1.0:
                return None
            ratio = min_effective_trust / self.direct_trust
            if ratio >= 1.0:
                return 0.0
            days_total = math.log(ratio) / math.log(retention)
            days_elapsed = self.days_since_refresh
            return max(0.0, days_total - days_elapsed)
        
        return None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "from_node_id": str(self.from_node_id),
            "to_node_id": str(self.to_node_id),
            "direct_trust": self.direct_trust,
            "domain": self.domain,
            "decay_rate": self.decay_rate,
            "decay_model": self.decay_model.value,
            "last_refreshed": self.last_refreshed.isoformat(),
            "created_at": self.created_at.isoformat(),
            "effective_trust": self.effective_trust(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrustEdge":
        """Deserialize from dictionary."""
        # Parse timestamps
        last_refreshed = data.get("last_refreshed")
        if isinstance(last_refreshed, str):
            last_refreshed = datetime.fromisoformat(last_refreshed)
        elif last_refreshed is None:
            last_refreshed = datetime.now(timezone.utc)
        
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
        
        # Parse decay model
        decay_model = data.get("decay_model", "exponential")
        if isinstance(decay_model, str):
            decay_model = DecayModel.from_string(decay_model)
        
        return cls(
            from_node_id=UUID(data["from_node_id"]) if isinstance(data["from_node_id"], str) else data["from_node_id"],
            to_node_id=UUID(data["to_node_id"]) if isinstance(data["to_node_id"], str) else data["to_node_id"],
            direct_trust=float(data["direct_trust"]),
            domain=data.get("domain"),
            decay_rate=float(data.get("decay_rate", 0.0)),
            decay_model=decay_model,
            last_refreshed=last_refreshed,
            created_at=created_at,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_trust_edge(
    from_node_id: UUID,
    to_node_id: UUID,
    trust: float,
    domain: str | None = None,
    decay_rate: float = 0.0,
    decay_model: DecayModel | str = DecayModel.EXPONENTIAL,
) -> TrustEdge:
    """Create a new trust edge with optional decay.
    
    Args:
        from_node_id: The trusting node
        to_node_id: The trusted node
        trust: Initial trust level (0.0 to 1.0)
        domain: Optional domain context
        decay_rate: Daily decay rate (0.0 = no decay)
        decay_model: Decay model to use
        
    Returns:
        A new TrustEdge
    """
    if isinstance(decay_model, str):
        decay_model = DecayModel.from_string(decay_model)
    
    return TrustEdge(
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        direct_trust=trust,
        domain=domain,
        decay_rate=decay_rate,
        decay_model=decay_model,
    )


def calculate_decayed_trust(
    base_trust: float,
    days_elapsed: float,
    decay_rate: float,
    decay_model: DecayModel = DecayModel.EXPONENTIAL,
) -> float:
    """Calculate decayed trust without creating a TrustEdge.
    
    Utility function for quick calculations.
    
    Args:
        base_trust: The starting trust value
        days_elapsed: Days since last refresh
        decay_rate: Daily decay rate
        decay_model: Decay model to use
        
    Returns:
        The decayed trust value
    """
    if decay_rate == 0.0 or decay_model == DecayModel.NONE:
        return base_trust
    
    if decay_model == DecayModel.LINEAR:
        return max(0.0, base_trust - (decay_rate * days_elapsed))
    
    elif decay_model == DecayModel.EXPONENTIAL:
        retention = 1.0 - decay_rate
        return max(0.0, base_trust * (retention ** days_elapsed))
    
    return base_trust
