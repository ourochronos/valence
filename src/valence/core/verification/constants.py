"""Constants for the verification protocol.

From spec/components/verification-protocol/REPUTATION.md
"""

from __future__ import annotations


class ReputationConstants:
    """Constants for reputation calculations."""

    # Base rewards
    CONFIRMATION_BASE = 0.001
    CONTRADICTION_BASE = 0.005
    UNCERTAINTY_BASE = 0.0002

    # Bounties
    BOUNTY_MULTIPLIER = 0.5
    FIRST_FINDER_BONUS = 2.0

    # Penalties
    CONTRADICTION_PENALTY_BASE = 0.003
    FRIVOLOUS_DISPUTE_PENALTY = 0.2
    COLLUSION_PENALTY = 0.5

    # Calibration
    CALIBRATION_BONUS_BASE = 0.01
    CALIBRATION_PENALTY_THRESHOLD = 0.4

    # Decay
    MONTHLY_INACTIVITY_DECAY = 0.02
    VERIFICATION_HALF_LIFE_MONTHS = 12

    # Limits
    MAX_DAILY_GAIN = 0.02
    MAX_WEEKLY_GAIN = 0.08
    REPUTATION_FLOOR = 0.1
    MAX_STAKE_RATIO = 0.2

    # Recovery
    MAX_MONTHLY_RECOVERY = 0.03
    PROBATION_DURATION_DAYS = 90

    # Timing
    VALIDATION_WINDOW_HOURS = 1
    ACCEPTANCE_DELAY_HOURS = 24
    DISPUTE_WINDOW_DAYS = 7
    RESOLUTION_TIMEOUT_DAYS = 14
    STAKE_LOCKUP_DAYS = 7
    BOUNTY_STAKE_LOCKUP_DAYS = 14

    # Rate limits
    MAX_VERIFICATIONS_PER_DAY = 50
    MAX_PENDING_VERIFICATIONS = 10
    MAX_VERIFICATIONS_PER_BELIEF = 100
    MAX_VERIFICATION_VELOCITY_PER_HOUR = 5
