# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Tests for valence auth CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from valence.cli.commands.auth import register


def test_auth_register():
    """Auth command group registers all subcommands."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    # Should parse valid subcommands
    args = parser.parse_args(["auth", "list-tokens"])
    assert args.auth_command == "list-tokens"
    assert hasattr(args, "func")


def test_auth_create_token_args():
    """create-token requires --client-id."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    args = parser.parse_args(["auth", "create-token", "-c", "test-client", "-d", "test desc"])
    assert args.client_id == "test-client"
    assert args.description == "test desc"


def test_auth_list_tokens(tmp_path: Path):
    """list-tokens runs without error on empty token store."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    register(sub)

    args = parser.parse_args(["auth", "--token-file", str(tmp_path / "tokens.json"), "list-tokens"])
    result = args.func(args)
    assert result == 0
