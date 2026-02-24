"""Tests for conflicts command module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from valence.cli.commands.conflicts import cmd_conflicts
from valence.cli.http_client import ValenceAPIError, ValenceConnectionError


class TestConflicts:
    """Test conflicts detection command."""

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_default(self, mock_get_client, mock_get_config):
        """Detect conflicts with default threshold."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "conflicts": [
                {
                    "belief_a": "belief1",
                    "belief_b": "belief2",
                    "similarity": 0.92,
                }
            ]
        }

        args = MagicMock(threshold=0.85, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 0
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "/beliefs/conflicts"
        params = call_args[1]["params"]
        assert params["output"] == "text"
        assert params["threshold"] == "0.85"

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_custom_threshold(self, mock_get_client, mock_get_config):
        """Detect conflicts with custom threshold."""
        mock_config = MagicMock(output="json")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"conflicts": []}

        args = MagicMock(threshold=0.75, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 0
        params = mock_client.get.call_args[1]["params"]
        assert params["threshold"] == "0.75"

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_high_threshold(self, mock_get_client, mock_get_config):
        """Detect conflicts with high threshold."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"conflicts": []}

        args = MagicMock(threshold=0.95, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 0
        params = mock_client.get.call_args[1]["params"]
        assert params["threshold"] == "0.95"

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_with_auto_record(self, mock_get_client, mock_get_config):
        """Detect conflicts and auto-record as tensions."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "conflicts": [{"belief_a": "b1", "belief_b": "b2"}],
            "recorded": 1,
        }

        args = MagicMock(threshold=0.85, auto_record=True)
        result = cmd_conflicts(args)

        assert result == 0
        params = mock_client.get.call_args[1]["params"]
        assert params["auto_record"] == "true"

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_output_format(self, mock_get_client, mock_get_config):
        """Test different output formats."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"conflicts": []}

        for output_format in ["text", "json", "table"]:
            mock_config = MagicMock(output=output_format)
            mock_get_config.return_value = mock_config

            args = MagicMock(threshold=0.85, auto_record=False)
            result = cmd_conflicts(args)

            assert result == 0
            params = mock_client.get.call_args[1]["params"]
            assert params["output"] == output_format

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_no_conflicts_found(self, mock_get_client, mock_get_config):
        """Handle case with no conflicts detected."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"conflicts": []}

        args = MagicMock(threshold=0.85, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 0

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_multiple_results(self, mock_get_client, mock_get_config):
        """Handle multiple conflict results."""
        mock_config = MagicMock(output="json")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "conflicts": [
                {"belief_a": "b1", "belief_b": "b2", "similarity": 0.90},
                {"belief_a": "b3", "belief_b": "b4", "similarity": 0.88},
                {"belief_a": "b5", "belief_b": "b6", "similarity": 0.86},
            ]
        }

        args = MagicMock(threshold=0.85, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 0

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_connection_error(self, mock_get_client, mock_get_config):
        """Handle connection errors."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_get_client.return_value.get.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(threshold=0.85, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 1

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_api_error_500(self, mock_get_client, mock_get_config):
        """Handle internal server errors."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_get_client.return_value.get.side_effect = ValenceAPIError(500, "error", "Internal server error")

        args = MagicMock(threshold=0.85, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 1

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_api_error_400(self, mock_get_client, mock_get_config):
        """Handle bad request errors."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_get_client.return_value.get.side_effect = ValenceAPIError(400, "bad_request", "Invalid threshold")

        args = MagicMock(threshold=-0.5, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 1

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_zero_threshold(self, mock_get_client, mock_get_config):
        """Test with zero threshold (edge case - not added to params due to falsy check)."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"conflicts": []}

        args = MagicMock(threshold=0.0, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 0
        params = mock_client.get.call_args[1]["params"]
        # threshold=0.0 is falsy, so it won't be in params
        assert "threshold" not in params

    @patch("valence.cli.commands.conflicts.get_cli_config")
    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_one_threshold(self, mock_get_client, mock_get_config):
        """Test with threshold of 1.0 (edge case)."""
        mock_config = MagicMock(output="text")
        mock_get_config.return_value = mock_config

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"conflicts": []}

        args = MagicMock(threshold=1.0, auto_record=False)
        result = cmd_conflicts(args)

        assert result == 0
        params = mock_client.get.call_args[1]["params"]
        assert params["threshold"] == "1.0"
