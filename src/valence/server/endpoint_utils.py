# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Shared utility functions for REST endpoint parameter parsing and output formatting.

Shared helpers for REST endpoint parameter parsing and output formatting.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response


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


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================


def parse_output_format(request: Request) -> str:
    """Parse ?output= query parameter. Default: json."""
    fmt = request.query_params.get("output", "json")
    if fmt not in ("json", "text", "table"):
        return "json"
    return fmt


def format_response(
    data: dict[str, Any],
    output_format: str,
    text_formatter: Callable[[dict[str, Any]], str] | None = None,
    table_formatter: Callable[[dict[str, Any]], str] | None = None,
    status_code: int = 200,
) -> Response:
    """Return response in the requested output format.

    For JSON mode (default), returns JSONResponse.
    For text/table mode, uses the provided formatter to produce plain text.
    If no formatter is available, falls back to JSON.
    """
    if output_format == "json":
        return JSONResponse(data, status_code=status_code)

    formatter = table_formatter if output_format == "table" and table_formatter else text_formatter
    if formatter:
        formatted_text = formatter(data)
        return PlainTextResponse(formatted_text, status_code=status_code)

    return JSONResponse(data, status_code=status_code)
