"""Tests for unified maintenance CLI (#363).

Tests cover:
1. Argument parsing: --retention, --archive, --tombstones, --views, --vacuum, --all, --dry-run, --json
2. No-op when no flags given (prints help, returns 1)
3. Individual operations dispatch correctly
4. --all runs full maintenance
5. --dry-run skips vacuum and views
6. --json produces valid JSON output
7. refresh_views function
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from valence.cli.commands.maintenance import cmd_maintenance
from valence.core.maintenance import MaintenanceResult, refresh_views


class TestRefreshViews:
    """Test the refresh_views function."""

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
            run_all=False, retention=False, archive=False,
            tombstones=False, compact=False, views=False, vacuum=False,
            dry_run=False, output_json=False,
        )
        result = cmd_maintenance(args)
        assert result == 1
        captured = capsys.readouterr()
        assert "--all" in captured.out


class TestCmdMaintenanceAll:
    """Test --all flag."""

    @patch("valence.cli.commands.maintenance.get_db_connection")
    def test_all_calls_run_full_maintenance(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        mock_cur.fetchone.side_effect = [
            {"archived_count": 5, "freed_embeddings": 3},
            {"count": 1},
        ]
        # fetchall: retention, compaction candidates, refresh_views
        mock_cur.fetchall.side_effect = [
            [{"table_name": "belief_retrievals", "deleted_count": 10}],
            [],  # No compaction candidates
            [{"view_name": "beliefs_current_mat", "refreshed": True, "error_msg": None}],
        ]

        args = argparse.Namespace(
            run_all=True, retention=False, archive=False,
            tombstones=False, compact=False, views=False, vacuum=False,
            dry_run=False, output_json=False,
        )
        result = cmd_maintenance(args)
        assert result == 0

    @patch("valence.cli.commands.maintenance.get_db_connection")
    def test_all_dry_run(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        mock_cur.fetchall.return_value = []  # Used for both retention and compaction
        mock_cur.fetchone.side_effect = [
            {"archived_count": 0, "freed_embeddings": 0},
            {"count": 0},
        ]

        args = argparse.Namespace(
            run_all=True, retention=False, archive=False,
            tombstones=False, compact=False, views=False, vacuum=False,
            dry_run=True, output_json=False,
        )
        result = cmd_maintenance(args)
        assert result == 0


class TestCmdMaintenanceIndividual:
    """Test individual operation flags."""

    @patch("valence.cli.commands.maintenance.get_db_connection")
    def test_retention_only(self, mock_conn_fn):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [
            {"table_name": "belief_retrievals", "deleted_count": 5},
        ]

        args = argparse.Namespace(
            run_all=False, retention=True, archive=False,
            tombstones=False, compact=False, views=False, vacuum=False,
            dry_run=False, output_json=False,
        )
        result = cmd_maintenance(args)
        assert result == 0

    @patch("valence.cli.commands.maintenance.get_db_connection")
    def test_views_dry_run_skipped(self, mock_conn_fn, capsys):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        args = argparse.Namespace(
            run_all=False, retention=False, archive=False,
            tombstones=False, compact=False, views=True, vacuum=False,
            dry_run=True, output_json=False,
        )
        result = cmd_maintenance(args)
        assert result == 0
        captured = capsys.readouterr()
        assert "dry run" in captured.out.lower()


class TestCmdMaintenanceJsonOutput:
    """Test --json output."""

    @patch("valence.cli.commands.maintenance.get_db_connection")
    def test_json_output(self, mock_conn_fn, capsys):
        mock_conn = MagicMock()
        mock_conn_fn.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur
        mock_cur.fetchall.return_value = [
            {"table_name": "belief_retrievals", "deleted_count": 10},
        ]

        args = argparse.Namespace(
            run_all=False, retention=True, archive=False,
            tombstones=False, compact=False, views=False, vacuum=False,
            dry_run=False, output_json=True,
        )
        result = cmd_maintenance(args)
        assert result == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert parsed[0]["operation"] == "retention"
