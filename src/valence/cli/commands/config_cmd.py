"""CLI commands for configuring Valence inference backends.

Implements WU-18: ``valence config inference <provider> [options]``.

Supported sub-commands::

    valence config inference show
    valence config inference gemini [--model MODEL]
    valence config inference cerebras --api-key KEY [--model MODEL]
    valence config inference ollama [--host HOST] [--model MODEL]

Configuration is stored in the ``system_config`` table (key = ``inference``).
The server reads this at startup (or on reconfiguration) to configure the
global ``InferenceProvider``.
"""

from __future__ import annotations

import argparse
import json
import sys

from ..http_client import ValenceAPIError, ValenceConnectionError, get_client
from ..output import output_error, output_result

# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``config`` command tree."""
    config_parser = subparsers.add_parser(
        "config",
        help="Configure Valence system settings (inference backend, etc.)",
    )
    config_sub = config_parser.add_subparsers(
        dest="config_command",
        required=True,
        metavar="SUBCOMMAND",
    )

    # valence config inference ...
    inference_parser = config_sub.add_parser(
        "inference",
        help="Configure LLM inference backend",
    )
    inference_sub = inference_parser.add_subparsers(
        dest="inference_provider",
        required=True,
        metavar="PROVIDER",
    )

    # valence config inference show
    show_p = inference_sub.add_parser(
        "show",
        help="Show current inference backend configuration",
    )
    show_p.set_defaults(func=cmd_config_inference_show)

    # valence config inference gemini [--model MODEL]
    gemini_p = inference_sub.add_parser(
        "gemini",
        help="Use Google Gemini Flash via the local 'gemini' CLI (no API key needed)",
    )
    gemini_p.add_argument(
        "--model",
        default="gemini-2.5-flash",
        metavar="MODEL",
        help="Gemini model name (default: gemini-2.5-flash)",
    )
    gemini_p.set_defaults(func=cmd_config_inference_gemini)

    # valence config inference cerebras --api-key KEY [--model MODEL]
    cerebras_p = inference_sub.add_parser(
        "cerebras",
        help="Use Cerebras Cloud (ultra-low-latency classification)",
    )
    cerebras_p.add_argument(
        "--api-key",
        required=True,
        metavar="KEY",
        help="Cerebras API key (https://cloud.cerebras.ai/)",
    )
    cerebras_p.add_argument(
        "--model",
        default="llama-4-scout-17b-16e-instruct",
        metavar="MODEL",
        help="Cerebras model name (default: llama-4-scout-17b-16e-instruct)",
    )
    cerebras_p.set_defaults(func=cmd_config_inference_cerebras)

    # valence config inference ollama [--host HOST] [--model MODEL]
    ollama_p = inference_sub.add_parser(
        "ollama",
        help="Use local Ollama instance for fully offline inference",
    )
    ollama_p.add_argument(
        "--host",
        default="http://localhost:11434",
        metavar="HOST",
        help="Ollama server URL (default: http://localhost:11434)",
    )
    ollama_p.add_argument(
        "--model",
        default="qwen3:30b",
        metavar="MODEL",
        help="Ollama model name (default: qwen3:30b). Must be pulled first: ollama pull <model>",
    )
    ollama_p.set_defaults(func=cmd_config_inference_ollama)

    # Root config fallback
    config_parser.set_defaults(func=lambda args: (config_parser.print_help(), 0)[1])


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_config_inference_show(args: argparse.Namespace) -> int:
    """Display the current inference configuration from system_config."""
    try:
        client = get_client()
        result = client.get("/v2/config/inference")
        output_result(result)
        return 0
    except ValenceConnectionError:
        return _show_config_direct()
    except ValenceAPIError as exc:
        output_error(exc.message)
        return 1


def cmd_config_inference_gemini(args: argparse.Namespace) -> int:
    """Configure Gemini Flash as the inference backend."""
    config_value = {
        "provider": "gemini",
        "model": args.model,
    }
    return _write_inference_config(config_value)


def cmd_config_inference_cerebras(args: argparse.Namespace) -> int:
    """Configure Cerebras as the inference backend."""
    config_value = {
        "provider": "cerebras",
        "api_key": args.api_key,
        "model": args.model,
    }
    return _write_inference_config(config_value)


def cmd_config_inference_ollama(args: argparse.Namespace) -> int:
    """Configure Ollama as the inference backend."""
    config_value = {
        "provider": "ollama",
        "host": args.host,
        "model": args.model,
    }
    return _write_inference_config(config_value)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _display_value(config_value: dict) -> dict:
    """Return a copy with api_key masked for display."""
    disp = dict(config_value)
    if "api_key" in disp:
        disp["api_key"] = "***"
    return disp


def _write_inference_config(config_value: dict) -> int:
    """Write inference config via the REST API, or directly to the DB as fallback."""
    disp = _display_value(config_value)
    try:
        client = get_client()
        result = client.post("/v2/config/inference", body=config_value)
        print(f"✓ Inference backend configured: {json.dumps(disp)}")
        output_result(result)
        return 0
    except ValenceConnectionError:
        # Server not running — write directly to system_config table
        return _write_config_direct("inference", config_value, disp)
    except ValenceAPIError as exc:
        output_error(exc.message)
        return 1


def _show_config_direct() -> int:
    """Read inference config directly from the database (no server required)."""
    try:
        from valence.lib.our_db import get_cursor  # type: ignore[import]

        with get_cursor() as cur:
            cur.execute("SELECT value, updated_at FROM system_config WHERE key = 'inference' LIMIT 1")
            row = cur.fetchone()
        if row is None:
            print("No inference backend configured (running in degraded/mock mode).")
        else:
            val = row["value"]
            if isinstance(val, str):
                val = json.loads(val)
            val = _display_value(val) if isinstance(val, dict) else val
            updated = row.get("updated_at", "?")
            print(f"Inference config (last updated: {updated}):")
            print(json.dumps(val, indent=2))
        return 0
    except Exception as exc:
        print(f"Error reading config from database: {exc}", file=sys.stderr)
        return 1


def _write_config_direct(key: str, value: dict, display_value: dict) -> int:
    """Write a system_config entry directly to the database (no server required)."""
    try:
        from valence.lib.our_db import get_cursor  # type: ignore[import]

        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO system_config (key, value)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = NOW()
                """,
                (key, json.dumps(value)),
            )
        print(f"✓ Inference backend configured (direct DB write): {json.dumps(display_value)}")
        print("  Restart the Valence server to apply the new configuration.")
        return 0
    except Exception as exc:
        print(f"Error writing config to database: {exc}", file=sys.stderr)
        return 1
