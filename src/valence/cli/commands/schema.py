"""CLI commands for dimension schema registry.

Usage:
    valence schema list
    valence schema show <name>
    valence schema validate <name> <json>
"""

from __future__ import annotations

import argparse
import json
import sys


def cmd_schema(args: argparse.Namespace) -> int:
    """Router for schema sub-commands."""
    handlers = {
        "list": _cmd_schema_list,
        "show": _cmd_schema_show,
        "validate": _cmd_schema_validate,
    }
    handler = handlers.get(args.schema_command)
    if handler:
        return handler(args)
    print("Unknown schema command. Use: list, show, validate", file=sys.stderr)
    return 1


def _cmd_schema_list(_args: argparse.Namespace) -> int:
    """List all registered schemas."""
    from valence.core.dimension_registry import get_registry

    registry = get_registry()
    schemas = registry.list_schemas()
    if not schemas:
        print("No schemas registered.")
        return 0

    for schema in schemas:
        inherits = f"  (inherits {schema.inherits})" if schema.inherits else ""
        desc = schema.metadata.get("description", "")
        desc_str = f" - {desc}" if desc else ""
        print(f"  {schema.name}{inherits}{desc_str}")
    return 0


def _cmd_schema_show(args: argparse.Namespace) -> int:
    """Show details of a specific schema."""
    from valence.core.dimension_registry import get_registry

    registry = get_registry()
    schema = registry.get(args.name)
    if schema is None:
        print(f"Schema '{args.name}' not found.", file=sys.stderr)
        return 1

    resolved = registry.resolve(args.name)

    print(f"Schema: {schema.name}")
    if schema.inherits:
        print(f"Inherits: {schema.inherits}")
    print(f"Range: [{schema.value_range[0]}, {schema.value_range[1]}]")

    print(f"Dimensions ({len(resolved.dimensions)}):")
    for dim in resolved.dimensions:
        marker = " [required]" if dim in resolved.required else ""
        # Mark inherited dims
        inherited = ""
        if schema.inherits and dim not in schema.dimensions:
            inherited = " (inherited)"
        print(f"  - {dim}{marker}{inherited}")

    if schema.metadata:
        desc = schema.metadata.get("description")
        if desc:
            print(f"Description: {desc}")

    return 0


def _cmd_schema_validate(args: argparse.Namespace) -> int:
    """Validate dimension values against a schema."""
    from valence.core.dimension_registry import get_registry

    try:
        dimensions = json.loads(args.dimensions_json)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(dimensions, dict):
        print("JSON must be an object mapping dimension names to values.", file=sys.stderr)
        return 1

    registry = get_registry()
    result = registry.validate(args.name, dimensions)

    if result.valid:
        print(f"✓ Valid for schema '{args.name}'")
        return 0

    print(f"✗ Invalid for schema '{args.name}':")
    for err in result.errors:
        print(f"  - {err}")
    return 1
