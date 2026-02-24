# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Standard response envelope for the Valence v2 knowledge system.

All public core functions return a ``ValenceResponse`` so callers always
receive a consistent ``{success, data, error, degraded}`` structure rather
than ad-hoc dicts or bare exceptions.

Usage::

    from valence.core.response import ok, err, ValenceResponse

    # Success
    return ok(data={"id": "abc"})

    # Failure
    return err("Something went wrong")

    # Degraded (inference fallback was used)
    return ok(data=article, degraded=True)

Implements WU-14 (C12, DR-10).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ValenceResponse:
    """Unified response envelope for all public Valence core functions.

    Attributes:
        success:  True when the operation completed without error.
        data:     Payload returned on success.  None for void operations.
        error:    Human-readable error message on failure.  None on success.
        degraded: True when inference fallback was used (e.g. LLM unavailable,
                  returned concatenation instead of synthesis).  Only meaningful
                  when success=True.
    """

    success: bool
    data: Any = None
    error: str | None = None
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict suitable for REST / MCP responses.

        Only includes keys that carry information:
        - ``success`` is always present.
        - ``data`` is included only when not None.
        - ``error`` is included only when truthy.
        - ``degraded`` is included only when True.
        """
        d: dict[str, Any] = {"success": self.success}
        if self.data is not None:
            d["data"] = self.data
        if self.error:
            d["error"] = self.error
        if self.degraded:
            d["degraded"] = True
        return d


def ok(data: Any = None, degraded: bool = False) -> ValenceResponse:
    """Create a successful ValenceResponse.

    Args:
        data:     Payload to return to the caller.
        degraded: Set to True when a fallback path was used.

    Returns:
        ValenceResponse with success=True.
    """
    return ValenceResponse(success=True, data=data, degraded=degraded)


def err(error: str) -> ValenceResponse:
    """Create a failed ValenceResponse.

    Args:
        error: Human-readable description of the failure.

    Returns:
        ValenceResponse with success=False.
    """
    return ValenceResponse(success=False, error=error)
