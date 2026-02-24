"""Tests for maintenance command module (#363)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from valence.cli.commands.maintenance import cmd_maintenance
from valence.cli.http_client import ValenceAPIError, ValenceConnectionError


class TestMaintenance:
    """Test maintenance command."""

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_all(self, mock_get_client, mock_get_config):
        """Run full maintenance cycle."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "retention": {"deleted": 100},
            "archive": {"archived": 50},
            "tombstones": {"cleaned": 10},
            "compact": {"compacted": 25},
            "views": {"refreshed": 3},
            "vacuum": {"analyzed": 5},
        }

        args = MagicMock(
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
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/admin/maintenance"
        body = call_args[1]["body"]
        assert body["all"] is True

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_retention_only(self, mock_get_client, mock_get_config):
        """Run retention operation only."""
        mock_config = MagicMock(output="json")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"retention": {"deleted": 50}}

        args = MagicMock(
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
        body = mock_client.post.call_args[1]["body"]
        assert body.get("retention") is True
        assert body.get("all") is None

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_archive_only(self, mock_get_client, mock_get_config):
        """Run archive operation only."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"archive": {"archived": 30}}

        args = MagicMock(
            run_all=False,
            retention=False,
            archive=True,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body.get("archive") is True

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_tombstones_only(self, mock_get_client, mock_get_config):
        """Run tombstone cleanup only."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"tombstones": {"cleaned": 5}}

        args = MagicMock(
            run_all=False,
            retention=False,
            archive=False,
            tombstones=True,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body.get("tombstones") is True

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_compact_only(self, mock_get_client, mock_get_config):
        """Run compact operation only."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"compact": {"compacted": 15}}

        args = MagicMock(
            run_all=False,
            retention=False,
            archive=False,
            tombstones=False,
            compact=True,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body.get("compact") is True

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_views_only(self, mock_get_client, mock_get_config):
        """Run view refresh only."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"views": {"refreshed": 3}}

        args = MagicMock(
            run_all=False,
            retention=False,
            archive=False,
            tombstones=False,
            compact=False,
            views=True,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body.get("views") is True

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_vacuum_only(self, mock_get_client, mock_get_config):
        """Run vacuum operation only."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"vacuum": {"analyzed": 5}}

        args = MagicMock(
            run_all=False,
            retention=False,
            archive=False,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=True,
            dry_run=False,
        )
        result = cmd_maintenance(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body.get("vacuum") is True

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_multiple_operations(self, mock_get_client, mock_get_config):
        """Run multiple specific operations."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "retention": {"deleted": 50},
            "compact": {"compacted": 20},
            "views": {"refreshed": 3},
        }

        args = MagicMock(
            run_all=False,
            retention=True,
            archive=False,
            tombstones=False,
            compact=True,
            views=True,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body.get("retention") is True
        assert body.get("compact") is True
        assert body.get("views") is True
        assert body.get("archive") is None

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_dry_run(self, mock_get_client, mock_get_config):
        """Run maintenance in dry-run mode."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "dry_run": True,
            "would_delete": 100,
        }

        args = MagicMock(
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
        body = mock_client.post.call_args[1]["body"]
        assert body.get("dry_run") is True

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_dry_run_with_operations(self, mock_get_client, mock_get_config):
        """Run specific operations in dry-run mode."""
        mock_config = MagicMock(output="json")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"dry_run": True}

        args = MagicMock(
            run_all=False,
            retention=True,
            archive=True,
            tombstones=False,
            compact=False,
            views=False,
            vacuum=False,
            dry_run=True,
        )
        result = cmd_maintenance(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body.get("retention") is True
        assert body.get("archive") is True
        assert body.get("dry_run") is True

    def test_maintenance_no_operations_specified(self):
        """Reject maintenance with no operations."""
        args = MagicMock(
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

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_output_formats(self, mock_get_client, mock_get_config):
        """Test different output formats."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"retention": {"deleted": 10}}

        for output_format in ["text", "json", "table"]:
            mock_config = MagicMock(output=output_format)
            mock_get_config.return_value = mock_config

            args = MagicMock(
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
            params = mock_client.post.call_args[1]["params"]
            assert params["output"] == output_format

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_connection_error(self, mock_get_client, mock_get_config):
        """Handle connection errors."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(
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

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_api_error_403(self, mock_get_client, mock_get_config):
        """Handle permission errors."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_get_client.return_value.post.side_effect = ValenceAPIError(403, "forbidden", "Permission denied")

        args = MagicMock(
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

    @patch("valence.cli.commands.maintenance.get_cli_config")
    @patch("valence.cli.commands.maintenance.get_client")
    def test_maintenance_api_error_500(self, mock_get_client, mock_get_config):
        """Handle internal server errors."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_get_client.return_value.post.side_effect = ValenceAPIError(500, "error", "Internal error")

        args = MagicMock(
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

        assert result == 1
