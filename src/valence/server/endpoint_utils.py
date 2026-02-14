"""Shared utility functions for REST endpoint parameter parsing.

Extracted from substrate_endpoints.py and vkb_endpoints.py to avoid duplication.
"""

from __future__ import annotations


def _parse_bool(value: str | None, default: bool = False) -> bool:
    """Parse a boolean query parameter."""
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes")


def _parse_int(value: str | None, default: int, maximum: int = 1000) -> int:
    """Parse an integer query parameter with a max cap."""
    if value is None:
        return default
    try:
        return min(int(value), maximum)
    except ValueError:
        return default


def _parse_float(value: str | None, default: float | None = None) -> float | None:
    """Parse a float query parameter safely."""
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
