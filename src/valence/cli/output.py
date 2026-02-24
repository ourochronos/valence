# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Output formatting for CLI commands.

Handles JSON vs server-formatted text output based on CLI config.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from .config import get_cli_config


def output_result(data: dict[str, Any], output_format: str | None = None) -> None:
    """Print API response in the configured output format.

    If output is "json", pretty-print the full JSON response.
    If the response has a "formatted" key (server-side formatting), print that directly.
    Otherwise fall back to JSON.
    """
    fmt = output_format or get_cli_config().output

    if fmt == "json":
        print(json.dumps(data, indent=2, default=str))
    elif "formatted" in data:
        print(data["formatted"])
    else:
        # Fallback: dump as JSON even in text mode if server didn't format
        print(json.dumps(data, indent=2, default=str))


def output_error(message: str) -> None:
    """Print error message to stderr."""
    print(f"Error: {message}", file=sys.stderr)
