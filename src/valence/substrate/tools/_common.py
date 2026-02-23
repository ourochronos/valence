"""Shared imports and utilities for substrate tool modules.

Centralizes commonly-used imports so that tests can patch a single
location (e.g., ``valence.substrate.tools._common.get_cursor``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Callable
from datetime import datetime
from typing import Any

from valence.lib.our_confidence import (
    DEFAULT_WEIGHTS,
    ConfidenceDimension,
    DimensionalConfidence,
    confidence_label,
)
from valence.lib.our_db import get_cursor
from valence.lib.our_models import Belief, Entity, Tension

from ...core.utils import escape_ilike

logger = logging.getLogger(__name__)


def _validate_enum(value: str, valid_values: list[str], param_name: str) -> dict[str, Any] | None:
    """Return error dict if value is not in valid_values, else None."""
    if value not in valid_values:
        return {"success": False, "error": f"Invalid {param_name} '{value}'. Must be one of: {', '.join(valid_values)}"}
    return None


__all__ = [
    "Belief",
    "Callable",
    "ConfidenceDimension",
    "DEFAULT_WEIGHTS",
    "DimensionalConfidence",
    "Entity",
    "Tension",
    "Any",
    "confidence_label",
    "datetime",
    "escape_ilike",
    "get_cursor",
    "hashlib",
    "json",
    "logger",
    "logging",
    "os",
    "_validate_enum",
]
