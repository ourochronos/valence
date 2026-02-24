"""Tests for right-sizing configuration CLI commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from valence.cli.commands.config_cmd import (
    _get_current_right_sizing,
    _show_right_sizing,
    cmd_config_right_sizing,
)


class TestRightSizingShow:
    """Test showing right-sizing configuration."""

    @patch("valence.core.db.get_cursor")
    def test_show_defaults(self, mock_get_cursor):
        """Show defaults when no config exists."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        result = _show_right_sizing()

        assert result == 0
        # Called for both _show_right_sizing and _get_current_right_sizing
        assert mock_cursor.execute.call_count >= 1

    @patch("valence.core.db.get_cursor")
    def test_show_existing_config(self, mock_get_cursor):
        """Show existing config from database."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {
            "value": {"target_tokens": 1500, "max_tokens": 3000, "min_tokens": 300},
            "updated_at": "2026-02-23 12:00:00",
        }
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        result = _show_right_sizing()

        assert result == 0

    @patch("valence.core.db.get_cursor")
    def test_show_handles_db_error(self, mock_get_cursor):
        """Handle database errors gracefully."""
        mock_get_cursor.side_effect = Exception("DB connection failed")

        result = _show_right_sizing()

        assert result == 1


class TestRightSizingUpdate:
    """Test updating right-sizing configuration."""

    @patch("valence.core.db.get_cursor")
    def test_update_all_values(self, mock_get_cursor):
        """Update all right-sizing values."""
        mock_cursor = MagicMock()
        # First call for reading current config
        mock_cursor.fetchone.return_value = None
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        args = MagicMock(
            target=1500,
            max=3000,
            min=300,
        )

        result = cmd_config_right_sizing(args)

        assert result == 0
        # Should have called execute twice: once for read, once for write
        assert mock_cursor.execute.call_count == 2

    @patch("valence.core.db.get_cursor")
    def test_update_partial_values(self, mock_get_cursor):
        """Update only some values, keeping others from current config."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"value": {"target_tokens": 2000, "max_tokens": 4000, "min_tokens": 200}}
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        args = MagicMock(
            target=1800,
            max=None,
            min=None,
        )

        result = cmd_config_right_sizing(args)

        assert result == 0

    @patch("valence.cli.commands.config_cmd._show_right_sizing")
    def test_no_args_shows_config(self, mock_show):
        """When no args provided, show current config."""
        mock_show.return_value = 0

        args = MagicMock(
            target=None,
            max=None,
            min=None,
        )

        result = cmd_config_right_sizing(args)

        assert result == 0
        mock_show.assert_called_once()

    def test_reject_zero_values(self):
        """Reject zero or negative values."""
        args = MagicMock(
            target=0,
            max=None,
            min=None,
        )

        result = cmd_config_right_sizing(args)
        assert result == 1

        args.target = None
        args.min = -5
        result = cmd_config_right_sizing(args)
        assert result == 1

    @patch("valence.core.db.get_cursor")
    def test_reject_min_greater_than_target(self, mock_get_cursor):
        """Reject min > target."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        args = MagicMock(
            target=500,
            max=None,
            min=1000,
        )

        result = cmd_config_right_sizing(args)
        assert result == 1

    @patch("valence.core.db.get_cursor")
    def test_reject_target_greater_than_max(self, mock_get_cursor):
        """Reject target > max."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        args = MagicMock(
            target=5000,
            max=3000,
            min=None,
        )

        result = cmd_config_right_sizing(args)
        assert result == 1

    @patch("valence.core.db.get_cursor")
    def test_handles_db_write_error(self, mock_get_cursor):
        """Handle database write errors."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.execute.side_effect = [None, Exception("Write failed")]
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        args = MagicMock(
            target=1500,
            max=None,
            min=None,
        )

        result = cmd_config_right_sizing(args)
        assert result == 1


class TestGetCurrentRightSizing:
    """Test reading current right-sizing config."""

    @patch("valence.core.db.get_cursor")
    def test_returns_db_value(self, mock_get_cursor):
        """Return value from database."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"value": {"target_tokens": 1500, "max_tokens": 3000, "min_tokens": 300}}
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        config = _get_current_right_sizing()

        assert config["target_tokens"] == 1500
        assert config["max_tokens"] == 3000
        assert config["min_tokens"] == 300

    @patch("valence.core.db.get_cursor")
    def test_returns_defaults_when_no_db_value(self, mock_get_cursor):
        """Return defaults when no database value."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        config = _get_current_right_sizing()

        assert config["target_tokens"] == 2000
        assert config["max_tokens"] == 4000
        assert config["min_tokens"] == 200

    @patch("valence.core.db.get_cursor")
    def test_handles_db_error(self, mock_get_cursor):
        """Return defaults when database error occurs."""
        mock_get_cursor.side_effect = Exception("DB connection failed")

        config = _get_current_right_sizing()

        assert config["target_tokens"] == 2000
        assert config["max_tokens"] == 4000
        assert config["min_tokens"] == 200

    @patch("valence.core.db.get_cursor")
    def test_handles_json_string_value(self, mock_get_cursor):
        """Parse JSON string values."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"value": '{"target_tokens": 1500, "max_tokens": 3000, "min_tokens": 300}'}
        mock_get_cursor.return_value.__enter__.return_value = mock_cursor

        config = _get_current_right_sizing()

        assert config["target_tokens"] == 1500
