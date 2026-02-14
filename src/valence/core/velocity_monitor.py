"""Velocity anomaly detection for belief creation and sharing (#352).

Tracks per-DID rates of belief creation and sharing within sliding windows.
Flags anomalies when rates exceed configurable thresholds.

Designed to work with or without a database — falls back to in-memory tracking
when no cursor is provided (useful for single-instance deployments).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# In-memory sliding window state (per-instance)
_windows: dict[str, list[float]] = defaultdict(list)


@dataclass
class VelocityConfig:
    """Velocity thresholds for anomaly detection."""

    beliefs_per_hour: int = 100
    shares_per_hour: int = 50
    window_seconds: int = 3600  # 1 hour sliding window


@dataclass
class VelocityResult:
    """Result of a velocity check."""

    allowed: bool
    identity_id: str
    action: str  # "belief_create" or "belief_share"
    current_count: int
    threshold: int
    message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "identity_id": self.identity_id,
            "action": self.action,
            "current_count": self.current_count,
            "threshold": self.threshold,
            "message": self.message,
        }


def check_velocity(
    identity_id: str,
    action: str,
    config: VelocityConfig | None = None,
) -> VelocityResult:
    """Check if an identity's action rate is within velocity limits.

    Uses in-memory sliding window tracking. Each action type gets its own window.

    Args:
        identity_id: DID or entity ID performing the action
        action: "belief_create" or "belief_share"
        config: Velocity thresholds (uses defaults if None)

    Returns:
        VelocityResult indicating if action is allowed
    """
    config = config or VelocityConfig()
    threshold = config.beliefs_per_hour if action == "belief_create" else config.shares_per_hour
    key = f"velocity:{identity_id}:{action}"

    now = time.time()
    window_start = now - config.window_seconds

    # Clean expired entries
    _windows[key] = [t for t in _windows[key] if t > window_start]
    current_count = len(_windows[key])

    if current_count >= threshold:
        logger.warning(f"Velocity limit exceeded: {identity_id} {action} {current_count}/{threshold}")
        return VelocityResult(
            allowed=False,
            identity_id=identity_id,
            action=action,
            current_count=current_count,
            threshold=threshold,
            message=f"Rate limit exceeded: {current_count} {action} in the last hour (limit: {threshold})",
        )

    # Record this action
    _windows[key].append(now)

    return VelocityResult(
        allowed=True,
        identity_id=identity_id,
        action=action,
        current_count=current_count + 1,
        threshold=threshold,
    )


def get_velocity_status(identity_id: str, config: VelocityConfig | None = None) -> dict[str, Any]:
    """Get current velocity status for an identity.

    Args:
        identity_id: DID or entity ID
        config: Velocity thresholds (uses defaults if None)

    Returns:
        Dict with current rates and remaining capacity
    """
    config = config or VelocityConfig()
    now = time.time()
    window_start = now - config.window_seconds

    statuses = {}
    for action, threshold in [("belief_create", config.beliefs_per_hour), ("belief_share", config.shares_per_hour)]:
        key = f"velocity:{identity_id}:{action}"
        _windows[key] = [t for t in _windows[key] if t > window_start]
        count = len(_windows[key])
        statuses[action] = {
            "current": count,
            "threshold": threshold,
            "remaining": max(0, threshold - count),
            "exceeded": count >= threshold,
        }

    return {
        "identity_id": identity_id,
        "window_seconds": config.window_seconds,
        "actions": statuses,
    }


def clear_velocity_state() -> None:
    """Clear all velocity tracking state. Useful for testing."""
    _windows.clear()
