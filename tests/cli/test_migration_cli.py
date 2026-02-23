"""Tests for migration CLI commands.

Tests cover:
1. Argument parsing for migrate up/down/status/create
2. REST client calls for migrate up/down/status
3. Local-only migrate create
4. Error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valence.cli.commands.migration import cmd_migrate
from valence.cli.main import app


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset CLI config singleton for each test."""
    from valence.cli.config import reset_cli_config

    reset_cli_config()
    yield
    reset_cli_config()


# ============================================================================
# Argument Parsing
# ============================================================================


class TestMigrateArgParsing:
    """Test CLI argument parsing for migrate commands."""

    def test_migrate_up_defaults(self):
        """Parse migrate up with defaults."""
        parser = app()
        args = parser.parse_args(["migrate", "up"])
        assert args.command == "migrate"
        assert args.migrate_command == "up"
        assert args.dry_run is False

    def test_migrate_up_dry_run(self):
        """Parse migrate up --dry-run."""
        parser = app()
        args = parser.parse_args(["migrate", "up", "--dry-run"])
        assert args.dry_run is True

    def test_migrate_down_defaults(self):
        """Parse migrate down with defaults."""
        parser = app()
        args = parser.parse_args(["migrate", "down"])
        assert args.migrate_command == "down"
        assert args.dry_run is False

    def test_migrate_down_dry_run(self):
        """Parse migrate down --dry-run."""
        parser = app()
        args = parser.parse_args(["migrate", "down", "--dry-run"])
        assert args.dry_run is True

    def test_migrate_status(self):
        """Parse migrate status."""
        parser = app()
        args = parser.parse_args(["migrate", "status"])
        assert args.migrate_command == "status"

    def test_migrate_create(self):
        """Parse migrate create."""
        parser = app()
        args = parser.parse_args(["migrate", "create", "add_users_table"])
        assert args.migrate_command == "create"
        assert args.name == "add_users_table"

    def test_migrate_requires_subcommand(self):
        """migrate without subcommand raises error."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["migrate"])


# ============================================================================
# REST Client Command Tests
# ============================================================================


class TestMigrateUpREST:
    """Test migrate up via REST client."""

    @patch("valence.cli.commands.migration.get_client")
    def test_up_happy_path(self, mock_get_client):
        """Migrate up calls POST /admin/migrate/up."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "applied": ["001_init"], "dry_run": False}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["migrate", "up"])
        result = cmd_migrate(args)

        assert result == 0
        mock_client.post.assert_called_once_with("/admin/migrate/up", body={"dry_run": False})

    @patch("valence.cli.commands.migration.get_client")
    def test_up_dry_run(self, mock_get_client):
        """Migrate up dry run passes dry_run=True."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "applied": ["001_init"], "dry_run": True}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["migrate", "up", "--dry-run"])
        result = cmd_migrate(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["dry_run"] is True

    @patch("valence.cli.commands.migration.get_client")
    def test_up_connection_error(self, mock_get_client):
        """Handles connection error."""
        from valence.cli.http_client import ValenceConnectionError

        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceConnectionError("http://127.0.0.1:8420")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["migrate", "up"])
        result = cmd_migrate(args)

        assert result == 1


class TestMigrateDownREST:
    """Test migrate down via REST client."""

    @patch("valence.cli.commands.migration.get_client")
    def test_down_happy_path(self, mock_get_client):
        """Migrate down calls POST /admin/migrate/down."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "rolled_back": "001_init", "dry_run": False}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["migrate", "down"])
        result = cmd_migrate(args)

        assert result == 0
        mock_client.post.assert_called_once_with("/admin/migrate/down", body={"dry_run": False})


class TestMigrateStatusREST:
    """Test migrate status via REST client."""

    @patch("valence.cli.commands.migration.get_client")
    def test_status_happy_path(self, mock_get_client):
        """Migrate status calls GET /admin/migrate/status."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "migrations": [], "count": 0}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["migrate", "status"])
        result = cmd_migrate(args)

        assert result == 0
        mock_client.get.assert_called_once()
        assert "/admin/migrate/status" in mock_client.get.call_args[0][0]

    @patch("valence.cli.commands.migration.get_client")
    def test_status_api_error(self, mock_get_client):
        """Handles API error."""
        from valence.cli.http_client import ValenceAPIError

        mock_client = MagicMock()
        mock_client.get.side_effect = ValenceAPIError(403, "FORBIDDEN", "Insufficient scope")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["migrate", "status"])
        result = cmd_migrate(args)

        assert result == 1


# ============================================================================
# Command Routing
# ============================================================================


class TestMigrateRouting:
    """Test migrate command routing."""

    def test_dispatches_to_migrate(self):
        """main() dispatches 'migrate' to cmd_migrate via args.func."""
        parser = app()
        args = parser.parse_args(["migrate", "up"])
        assert hasattr(args, "func")
        assert args.func is cmd_migrate
