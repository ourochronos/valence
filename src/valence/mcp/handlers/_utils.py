# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Shared helpers for MCP tool handlers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_async(coro: Any) -> Any:
    """Run an async coroutine from sync context (event-loop aware).

    Handles:
    - No event loop (create new)
    - Existing but not running loop (use it)
    - Running loop (execute in thread pool to avoid blocking)
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=60)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def validate_enum(value: str, valid_values: list[str], param_name: str) -> dict[str, Any] | None:
    """Return error dict if value is not in valid_values, else None."""
    if value not in valid_values:
        return {"success": False, "error": f"Invalid {param_name} '{value}'. Must be one of: {', '.join(valid_values)}"}
    return None
