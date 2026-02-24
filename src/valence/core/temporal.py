"""Temporal validity tracking for Valence articles.

Articles can have time-bounded validity and can supersede each other
to form chains of knowledge evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any


@dataclass
class TemporalValidity:
    """Time-bounded validity for a belief.

    A belief can be:
    - Always valid: valid_from=None, valid_until=None
    - Valid from a point: valid_from=datetime, valid_until=None
    - Valid in a range: valid_from=datetime, valid_until=datetime
    - Historically valid: valid_from=datetime, valid_until=datetime (past)
    """

    valid_from: datetime | None = None
    valid_until: datetime | None = None

    @classmethod
    def always_valid(cls) -> TemporalValidity:
        """Create validity that is always true."""
        return cls()

    @classmethod
    def from_now(cls) -> TemporalValidity:
        """Create validity starting from now."""
        return cls(valid_from=datetime.now())

    @classmethod
    def until(cls, end: datetime) -> TemporalValidity:
        """Create validity until a specific time."""
        return cls(valid_until=end)

    @classmethod
    def range(cls, start: datetime, end: datetime) -> TemporalValidity:
        """Create validity for a specific time range."""
        return cls(valid_from=start, valid_until=end)

    @classmethod
    def for_duration(cls, duration: timedelta) -> TemporalValidity:
        """Create validity for a duration from now."""
        now = datetime.now()
        return cls(valid_from=now, valid_until=now + duration)

    def is_valid_at(self, when: datetime | None = None) -> bool:
        """Check if valid at a specific time (default: now)."""
        check_time = when or datetime.now()

        if self.valid_from and check_time < self.valid_from:
            return False

        if self.valid_until and check_time > self.valid_until:
            return False

        return True

    def is_current(self) -> bool:
        """Check if currently valid."""
        return self.is_valid_at(datetime.now())

    def is_expired(self) -> bool:
        """Check if validity has ended."""
        if self.valid_until is None:
            return False
        return datetime.now() > self.valid_until

    def is_future(self) -> bool:
        """Check if validity hasn't started yet."""
        if self.valid_from is None:
            return False
        return datetime.now() < self.valid_from

    def overlaps(self, other: TemporalValidity) -> bool:
        """Check if two temporal validities overlap."""
        # Handle unbounded cases
        if self.valid_from is None and self.valid_until is None:
            return True
        if other.valid_from is None and other.valid_until is None:
            return True

        # Get effective bounds
        self_start = self.valid_from or datetime.min
        self_end = self.valid_until or datetime.max
        other_start = other.valid_from or datetime.min
        other_end = other.valid_until or datetime.max

        # Check for overlap
        return self_start <= other_end and other_start <= self_end

    def contains(self, when: datetime) -> bool:
        """Check if a point in time falls within this validity."""
        return self.is_valid_at(when)

    def duration(self) -> timedelta | None:
        """Get the duration of validity, or None if unbounded."""
        if self.valid_from is None or self.valid_until is None:
            return None
        return self.valid_until - self.valid_from

    def remaining(self) -> timedelta | None:
        """Get remaining valid time, or None if no end."""
        if self.valid_until is None:
            return None
        now = datetime.now()
        if now > self.valid_until:
            return timedelta(0)
        return self.valid_until - now

    def expire_now(self) -> TemporalValidity:
        """Return new validity with valid_until set to now."""
        return TemporalValidity(
            valid_from=self.valid_from,
            valid_until=datetime.now(),
        )

    def extend(self, duration: timedelta) -> TemporalValidity:
        """Extend validity by a duration."""
        if self.valid_until is None:
            return self  # Already unbounded
        return TemporalValidity(
            valid_from=self.valid_from,
            valid_until=self.valid_until + duration,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalValidity:
        """Create from dictionary."""
        valid_from = None
        valid_until = None

        if data.get("valid_from"):
            if isinstance(data["valid_from"], str):
                valid_from = datetime.fromisoformat(data["valid_from"])
            else:
                valid_from = data["valid_from"]

        if data.get("valid_until"):
            if isinstance(data["valid_until"], str):
                valid_until = datetime.fromisoformat(data["valid_until"])
            else:
                valid_until = data["valid_until"]

        return cls(valid_from=valid_from, valid_until=valid_until)

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.valid_from is None and self.valid_until is None:
            return "always valid"
        elif self.valid_from is None:
            # valid_until must be set since we didn't match the first condition
            assert self.valid_until is not None
            return f"valid until {self.valid_until.isoformat()}"
        elif self.valid_until is None:
            return f"valid from {self.valid_from.isoformat()}"
        else:
            return f"valid {self.valid_from.isoformat()} to {self.valid_until.isoformat()}"


@dataclass
class SupersessionChain:
    """Tracks the history of a belief through supersessions.

    When a belief is superseded, it forms a chain:
    original -> v2 -> v3 -> current

    This class helps navigate and query the chain.
    """

    belief_ids: list[str]  # Ordered from oldest to newest
    reasons: list[str | None]  # Why each supersession happened
    timestamps: list[datetime]  # When each supersession happened

    @property
    def original_id(self) -> str:
        """Get the original belief ID."""
        return self.belief_ids[0]

    @property
    def current_id(self) -> str:
        """Get the current (most recent) belief ID."""
        return self.belief_ids[-1]

    @property
    def length(self) -> int:
        """Number of articles in the chain."""
        return len(self.belief_ids)

    @property
    def revision_count(self) -> int:
        """Number of revisions (supersessions)."""
        return len(self.belief_ids) - 1

    def get_at_time(self, when: datetime) -> str | None:
        """Get the belief ID that was current at a specific time."""
        for i, ts in enumerate(self.timestamps):
            if when < ts:
                return self.belief_ids[i] if i > 0 else None
        return self.current_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "belief_ids": self.belief_ids,
            "reasons": self.reasons,
            "timestamps": [ts.isoformat() for ts in self.timestamps],
            "length": self.length,
            "revision_count": self.revision_count,
        }


def calculate_freshness(created_at: datetime, half_life_days: float = 30.0) -> float:
    """Calculate temporal freshness score based on age.

    Uses exponential decay with configurable half-life.
    Returns a value between 0 and 1.
    """
    age = datetime.now() - created_at
    age_days = age.total_seconds() / (24 * 60 * 60)

    # Exponential decay: freshness = 0.5^(age/half_life)
    import math

    freshness = math.pow(0.5, age_days / half_life_days)

    return max(0.0, min(1.0, freshness))


def freshness_label(freshness: float) -> str:
    """Get a human-readable label for freshness."""
    if freshness >= 0.9:
        return "very fresh"
    elif freshness >= 0.7:
        return "fresh"
    elif freshness >= 0.5:
        return "moderately fresh"
    elif freshness >= 0.3:
        return "aging"
    elif freshness >= 0.1:
        return "stale"
    else:
        return "very stale"
