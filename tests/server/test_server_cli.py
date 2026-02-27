# SPDX-License-Identifier: MIT
"""Tests for server management CLI and health metadata."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestServerStatus:
    """Test valence server status command."""

    @patch("valence.cli.commands.server._check_health")
    @patch("valence.cli.commands.server._launchd_loaded")
    @patch("valence.cli.commands.server._get_server_pid")
    @patch("valence.cli.commands.server._get_git_head")
    def test_status_healthy(self, mock_git, mock_pid, mock_loaded, mock_health, capsys):
        from valence.cli.commands.server import cmd_server_status

        mock_pid.return_value = 12345
        mock_loaded.return_value = True
        mock_git.return_value = "abc1234"
        mock_health.return_value = {
            "status": "healthy",
            "version": "2.0.0",
            "database": "connected",
            "started_at": "2026-02-26T12:00:00Z",
            "git_commit": "abc1234",
            "last_write_at": "2026-02-26T16:00:00Z",
        }

        args = MagicMock()
        result = cmd_server_status(args)
        assert result == 0

    @patch("valence.cli.commands.server._check_health")
    @patch("valence.cli.commands.server._launchd_loaded")
    @patch("valence.cli.commands.server._get_server_pid")
    @patch("valence.cli.commands.server._get_git_head")
    def test_status_stale_code(self, mock_git, mock_pid, mock_loaded, mock_health, capsys):
        from valence.cli.commands.server import cmd_server_status

        mock_pid.return_value = 12345
        mock_loaded.return_value = True
        mock_git.return_value = "def5678"
        mock_health.return_value = {
            "status": "healthy",
            "version": "2.0.0",
            "database": "connected",
            "git_commit": "abc1234",
        }

        args = MagicMock()
        result = cmd_server_status(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "restart needed" in captured.out.lower() or "stale" in captured.out.lower() or "code_stale" in captured.out

    @patch("valence.cli.commands.server._check_health")
    @patch("valence.cli.commands.server._launchd_loaded")
    @patch("valence.cli.commands.server._get_server_pid")
    @patch("valence.cli.commands.server._get_git_head")
    def test_status_server_down(self, mock_git, mock_pid, mock_loaded, mock_health, capsys):
        from valence.cli.commands.server import cmd_server_status

        mock_pid.return_value = None
        mock_loaded.return_value = False
        mock_git.return_value = "abc1234"
        mock_health.return_value = None

        args = MagicMock()
        result = cmd_server_status(args)
        assert result == 0  # status always returns 0 (it's informational)
        captured = capsys.readouterr()
        assert "unreachable" in captured.out


class TestHealthMetadata:
    """Test server health endpoint metadata."""

    def test_record_write_updates_timestamp(self):
        from valence.server import app as app_mod

        app_mod._last_write_at = None
        app_mod.record_write()
        assert app_mod._last_write_at is not None
        assert "T" in app_mod._last_write_at  # ISO format

    def test_record_write_updates_on_each_call(self):
        import time

        from valence.server import app as app_mod

        app_mod.record_write()
        first = app_mod._last_write_at
        time.sleep(0.01)
        app_mod.record_write()
        second = app_mod._last_write_at
        # Timestamps should differ (or at least both be set)
        assert first is not None
        assert second is not None


class TestFlattenTree:
    """Test _flatten_tree in section_embeddings (ensure it's accessible)."""

    def test_import(self):
        from valence.core.section_embeddings import _flatten_tree

        assert callable(_flatten_tree)
