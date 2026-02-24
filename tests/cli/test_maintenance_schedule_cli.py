"""Tests for maintenance schedule CLI commands."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

from valence.cli.commands.maintenance import (
    cmd_maintenance_schedule,
)


class TestMaintenanceScheduleCLI:
    @patch("valence.core.db.get_cursor")
    @patch("valence.core.maintenance.get_maintenance_schedule")
    def test_schedule_show_disabled(self, mock_get_sched, mock_get_cursor):
        """Test showing schedule when disabled."""
        mock_get_sched.return_value = None
        mock_get_cursor.return_value.__enter__ = MagicMock()
        mock_get_cursor.return_value.__exit__ = MagicMock()

        args = argparse.Namespace(interval=None, disable=False)

        result = cmd_maintenance_schedule(args)

        assert result == 0
        mock_get_sched.assert_called_once()

    @patch("valence.core.db.get_cursor")
    @patch("valence.core.maintenance.get_maintenance_schedule")
    def test_schedule_show_enabled(self, mock_get_sched, mock_get_cursor):
        """Test showing schedule when enabled."""
        mock_get_sched.return_value = {
            "enabled": True,
            "interval_hours": 24,
            "last_run": "2026-01-01T00:00:00+00:00",
        }
        mock_get_cursor.return_value.__enter__ = MagicMock()
        mock_get_cursor.return_value.__exit__ = MagicMock()

        args = argparse.Namespace(interval=None, disable=False)

        result = cmd_maintenance_schedule(args)

        assert result == 0

    @patch("valence.core.db.get_cursor")
    @patch("valence.core.maintenance.set_maintenance_schedule")
    def test_schedule_set_interval(self, mock_set_sched, mock_get_cursor):
        """Test setting maintenance interval."""
        mock_set_sched.return_value = {
            "enabled": True,
            "interval_hours": 12,
            "last_run": None,
        }
        mock_get_cursor.return_value.__enter__ = MagicMock()
        mock_get_cursor.return_value.__exit__ = MagicMock()

        args = argparse.Namespace(interval=12, disable=False)

        result = cmd_maintenance_schedule(args)

        assert result == 0
        mock_set_sched.assert_called_once()

    @patch("valence.core.db.get_cursor")
    @patch("valence.core.maintenance.set_maintenance_schedule")
    def test_schedule_set_invalid_interval(self, mock_set_sched, mock_get_cursor):
        """Test setting invalid interval."""
        mock_get_cursor.return_value.__enter__ = MagicMock()
        mock_get_cursor.return_value.__exit__ = MagicMock()
        # The function validates before calling set_maintenance_schedule

        args = argparse.Namespace(interval=0, disable=False)

        result = cmd_maintenance_schedule(args)

        assert result == 1
        # Should not call set_maintenance_schedule with invalid input
        mock_set_sched.assert_not_called()

    @patch("valence.core.db.get_cursor")
    @patch("valence.core.maintenance.disable_maintenance_schedule")
    def test_schedule_disable(self, mock_disable_sched, mock_get_cursor):
        """Test disabling maintenance schedule."""
        mock_get_cursor.return_value.__enter__ = MagicMock()
        mock_get_cursor.return_value.__exit__ = MagicMock()

        args = argparse.Namespace(interval=None, disable=True)

        result = cmd_maintenance_schedule(args)

        assert result == 0
        mock_disable_sched.assert_called_once()

    @patch("valence.core.db.get_cursor")
    def test_schedule_error_handling(self, mock_get_cursor):
        """Test error handling in schedule command."""
        mock_get_cursor.side_effect = Exception("Database error")

        args = argparse.Namespace(interval=None, disable=False)

        result = cmd_maintenance_schedule(args)

        # Should return error code
        assert result == 1
