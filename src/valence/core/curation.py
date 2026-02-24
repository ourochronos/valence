"""Curation decision framework for auto-capture.

Defines vocabulary and confidence thresholds for different types of
automatically captured knowledge. Used by session-end hooks and
insight extraction.
"""

from __future__ import annotations

from valence.core.config import get_config

# Signal types mapped to base confidence scores.
# Higher confidence = stronger signal that this should be captured.
SIGNAL_CONFIDENCE: dict[str, float] = {
    "explicit_request": 0.90,  # "remember X"
    "decision_with_rationale": 0.80,
    "stated_preference": 0.75,
    "correction": 0.70,
    "project_fact": 0.65,
    "session_summary": 0.50,  # lowered: corroboration will raise it
    "session_theme": 0.50,  # lowered: corroboration will raise it
    "mentioned_in_passing": 0.35,
}

# Minimum confidence for auto-capture
# Note: This is a default constant for tests. Live code should use _get_min_capture_confidence()
# which reads from config (VALENCE_MIN_CAPTURE_CONFIDENCE env var)
MIN_CAPTURE_CONFIDENCE = 0.50


def _get_min_capture_confidence() -> float:
    """Get minimum capture confidence from config."""
    return get_config().min_capture_confidence


# Maximum articles auto-created per session to prevent spam
MAX_AUTO_BELIEFS_PER_SESSION = 10

# Minimum content length for capture
MIN_SUMMARY_LENGTH = 20
MIN_THEME_LENGTH = 10


def should_capture(signal_type: str) -> bool:
    """Check if a signal type meets the minimum capture threshold."""
    confidence = SIGNAL_CONFIDENCE.get(signal_type, 0.0)
    return confidence >= _get_min_capture_confidence()


def get_confidence(signal_type: str) -> float:
    """Get the confidence score for a signal type."""
    return SIGNAL_CONFIDENCE.get(signal_type, 0.5)


def corroboration_confidence(corroboration_count: int) -> float:
    """Map corroboration count to confidence level.

    Implements a step-wise escalation ladder:
    - 0-1 corroborations: 0.50 (tentative)
    - 2 corroborations: 0.65 (emerging)
    - 3+ corroborations: 0.80 (established)
    """
    if corroboration_count <= 1:
        return 0.50
    if corroboration_count == 2:
        return 0.65
    return 0.80
