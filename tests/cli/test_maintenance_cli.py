"""Tests for unified maintenance CLI (#363).

Tests cover:
1. Argument parsing: --retention, --archive, --tombstones, --views, --vacuum, --all, --dry-run
2. No-op when no flags given (prints help, returns 1)
3. REST client calls to server
4. refresh_views function (core module, not CLI)
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from valence.cli.commands.maintenance import cmd_maintenance
from valence.core.maintenance import refresh_views


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset CLI config singleton for each test."""
    from valence.cli.config import reset_cli_config

    reset_cli_config()
    yield
    reset_cli_config()


class TestRefreshViews:
    """Test the refresh_views function (core module, unchanged)."""

    def test_calls_stored_procedure(self):
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"view_name": "beliefs_current_mat", "refreshed": True, "error_msg": None},
            {"view_name": "domain_statistics", "refreshed": True, "error_msg": None},
        ]
        result = refresh_views(cur)
        assert result.operation == "refresh_views"
        assert result.details["refreshed"] == 2
        assert result.details["failed"] == 0

    def test_reports_failures(self):
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"view_name": "beliefs_current_mat", "refreshed": True, "error_msg": None},
            {"view_name": "reputation_summary", "refreshed": False, "error_msg": "relation does not exist"},
        ]
        result = refresh_views(cur)
        assert result.details["refreshed"] == 1
        assert result.details["failed"] == 1
        assert result.details["errors"][0]["view"] == "reputation_summary"

    def test_concurrent_parameter(self):
        cur = MagicMock()
        cur.fetchall.return_value = []
        refresh_views(cur, concurrent=False)
        args = cur.execute.call_args[0]
        assert args[1] == (False,)


class TestCmdMaintenanceNoOp:
    """Test that no flags = help message."""

    def test_no_flags_returns_1(self, capsys):
        args = argparse.Namespace(
            run_all=False,
            retention=False,
            archive=False,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "--all" in captured.out


class TestCmdMaintenanceREST:
    """Test maintenance commands via REST client."""

    @patch("valence.cli.commands.maintenance.get_client")
    def test_all_calls_server(self, mock_get_client):
        """--all sends POST /admin/maintenance with all=True."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "results": [], "count": 0, "dry_run": False}
        mock_get_client.return_value = mock_client

        args = argparse.Namespace(
            run_all=True,
            retention=False,
            archive=False,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)
        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["all"] is True

    @patch("valence.cli.commands.maintenance.get_client")
    def test_retention_only(self, mock_get_client):
        """--retention sends retention=True."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "results": [], "count": 1, "dry_run": False}
        mock_get_client.return_value = mock_client

        args = argparse.Namespace(
            run_all=False,
            retention=True,
            archive=False,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)
        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["retention"] is True
        assert "all" not in call_body

    @patch("valence.cli.commands.maintenance.get_client")
    def test_dry_run(self, mock_get_client):
        """--dry-run passes dry_run=True."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "results": [], "count": 0, "dry_run": True}
        mock_get_client.return_value = mock_client

        args = argparse.Namespace(
            run_all=True,
            retention=False,
            archive=False,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=True,
        )
        result = cmd_maintenance(args)
        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["dry_run"] is True

    @patch("valence.cli.commands.maintenance.get_client")
    def test_connection_error(self, mock_get_client):
        """Handles connection error."""
        from valence.cli.http_client import ValenceConnectionError

        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceConnectionError("http://127.0.0.1:8420")
        mock_get_client.return_value = mock_client

        args = argparse.Namespace(
            run_all=True,
            retention=False,
            archive=False,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)
        assert result == 1

    @patch("valence.cli.commands.maintenance.get_client")
    def test_api_error(self, mock_get_client):
        """Handles API error."""
        from valence.cli.http_client import ValenceAPIError

        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceAPIError(403, "FORBIDDEN", "Insufficient scope")
        mock_get_client.return_value = mock_client

        args = argparse.Namespace(
            run_all=True,
            retention=False,
            archive=False,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)
        assert result == 1
