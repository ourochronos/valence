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

    # valence config show
    show_all_p = config_sub.add_parser(
        "show",
        help="Show all system configuration",
    )
    show_all_p.set_defaults(func=cmd_config_show_all)

    # valence config set <key> <value>
    set_p = config_sub.add_parser(
        "set",
        help="Set a configuration value",
    )
    set_p.add_argument("key", help="Configuration key")
    set_p.add_argument("value", help="Configuration value (JSON string)")
    set_p.set_defaults(func=cmd_config_set)

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

    # valence config right-sizing ...
    right_sizing_parser = config_sub.add_parser(
        "right-sizing",
        help="Configure article right-sizing (token limits)",
    )
    right_sizing_parser.add_argument(
        "--target",
        type=int,
        metavar="N",
        help="Target token count for compiled articles",
    )
    right_sizing_parser.add_argument(
        "--max",
        type=int,
        metavar="N",
        help="Maximum token count for compiled articles",
    )
    right_sizing_parser.add_argument(
        "--min",
        type=int,
        metavar="N",
        help="Minimum token count for compiled articles",
    )
    right_sizing_parser.set_defaults(func=cmd_config_right_sizing)

    # Root config fallback
    config_parser.set_defaults(func=lambda args: (config_parser.print_help(), 0)[1])


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_config_show_all(args: argparse.Namespace) -> int:
    """Display all system configuration."""
    try:
        from valence.core.db import get_cursor  # type: ignore[import]

        with get_cursor() as cur:
            cur.execute("SELECT key, value, updated_at FROM system_config ORDER BY key")
            rows = cur.fetchall()

        if not rows:
            print("No configuration found.")
            return 0

        config_dict = {}
        for row in rows:
            key = row["key"]
            val = row["value"]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except json.JSONDecodeError:
                    pass
            # Mask sensitive values
            if isinstance(val, dict) and "api_key" in val:
                val = _display_value(val)
            config_dict[key] = val

        output_result(config_dict)
        return 0
    except Exception as exc:
        output_error(f"Error reading configuration: {exc}")
        return 1


def cmd_config_set(args: argparse.Namespace) -> int:
    """Set a configuration value."""
    try:
        # Try to parse value as JSON, fall back to string
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value

        from valence.core.db import get_cursor  # type: ignore[import]

        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO system_config (key, value)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = NOW()
                """,
                (args.key, json.dumps(value)),
            )

        print(f"✓ Configuration set: {args.key} = {value}")
        return 0
    except Exception as exc:
        output_error(f"Error setting configuration: {exc}")
        return 1


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


def cmd_config_right_sizing(args: argparse.Namespace) -> int:
    """Display or update right-sizing configuration."""
    # If no flags provided, show current config
    if args.target is None and args.max is None and args.min is None:
        return _show_right_sizing()

    # Validate values if provided
    if args.min is not None and args.min < 1:
        output_error("Minimum token count must be at least 1")
        return 1
    if args.target is not None and args.target < 1:
        output_error("Target token count must be at least 1")
        return 1
    if args.max is not None and args.max < 1:
        output_error("Maximum token count must be at least 1")
        return 1

    # Get current config to merge with updates
    current = _get_current_right_sizing()

    # Update only the provided values
    if args.target is not None:
        current["target_tokens"] = args.target
    if args.max is not None:
        current["max_tokens"] = args.max
    if args.min is not None:
        current["min_tokens"] = args.min

    # Validate logical constraints
    if current["min_tokens"] > current["target_tokens"]:
        output_error("Minimum token count cannot exceed target token count")
        return 1
    if current["target_tokens"] > current["max_tokens"]:
        output_error("Target token count cannot exceed maximum token count")
        return 1

    # Write to database
    try:
        from valence.core.db import get_cursor  # type: ignore[import]

        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO system_config (key, value)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = NOW()
                """,
                ("right_sizing", json.dumps(current)),
            )

        print("✓ Right-sizing configured:")
        print(f"  Target tokens: {current['target_tokens']}")
        print(f"  Maximum tokens: {current['max_tokens']}")
        print(f"  Minimum tokens: {current['min_tokens']}")
        return 0
    except Exception as exc:
        output_error(f"Error writing right-sizing configuration: {exc}")
        return 1


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
        from valence.core.db import get_cursor  # type: ignore[import]

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
        from valence.core.db import get_cursor  # type: ignore[import]

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


def _get_current_right_sizing() -> dict[str, int]:
    """Read current right-sizing config from database, or return defaults."""
    try:
        from valence.core.db import get_cursor  # type: ignore[import]

        with get_cursor() as cur:
            cur.execute("SELECT value FROM system_config WHERE key = 'right_sizing' LIMIT 1")
            row = cur.fetchone()
            if row:
                val = row["value"]
                if isinstance(val, str):
                    val = json.loads(val)
                if isinstance(val, dict):
                    return val
    except Exception as exc:
        output_error(f"Error reading right-sizing config: {exc}")

    # Return defaults (matching compilation.py DEFAULT_RIGHT_SIZING)
    return {
        "target_tokens": 2000,
        "max_tokens": 4000,
        "min_tokens": 200,
    }


def _show_right_sizing() -> int:
    """Display current right-sizing configuration."""
    try:
        from valence.core.db import get_cursor  # type: ignore[import]

        with get_cursor() as cur:
            cur.execute("SELECT value, updated_at FROM system_config WHERE key = 'right_sizing' LIMIT 1")
            row = cur.fetchone()

        if row is None:
            # Show defaults
            config = _get_current_right_sizing()
            print("Right-sizing configuration (defaults):")
        else:
            val = row["value"]
            if isinstance(val, str):
                val = json.loads(val)
            config = val if isinstance(val, dict) else _get_current_right_sizing()
            updated = row.get("updated_at", "?")
            print(f"Right-sizing configuration (last updated: {updated}):")

        print(f"  Target tokens: {config.get('target_tokens', 2000)}")
        print(f"  Maximum tokens: {config.get('max_tokens', 4000)}")
        print(f"  Minimum tokens: {config.get('min_tokens', 200)}")
        return 0
    except Exception as exc:
        output_error(f"Error reading right-sizing configuration: {exc}")
        return 1
