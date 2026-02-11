"""Tests for session-end hook auto-capture."""

from __future__ import annotations

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# The hook is a standalone script, import it by path
HOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "plugin", "hooks")


@pytest.fixture
def hook_module():
    """Import session-end.py as a module."""
    hook_path = os.path.join(HOOK_DIR, "session-end.py")
    spec = importlib.util.spec_from_file_location("session_end_hook", hook_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mock_conn():
    """Create a mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    dict_cursor = MagicMock()

    # Regular cursor for inserts/updates
    conn.cursor.return_value.__enter__ = MagicMock(return_value=cursor)
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    # Default: no existing belief found (dedup hash check returns None)
    cursor.fetchone.return_value = None

    return conn, cursor, dict_cursor


class TestAutoCapture:
    """Tests for auto_capture_beliefs."""

    def test_summary_creates_belief(self, hook_module, mock_conn):
        """Summary present should create a belief with correct confidence."""
        conn, cursor, _ = mock_conn
        cursor.execute = MagicMock()

        session_data = {
            "summary": "Implemented the valence knowledge substrate with dimensional confidence model.",
            "themes": [],
            "project_context": "valence",
        }

        count = hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)
        assert count == 1

        # Check belief was created with correct confidence
        calls = cursor.execute.call_args_list
        insert_call = [c for c in calls if "INSERT INTO beliefs" in str(c)]
        assert len(insert_call) >= 1
        # Confidence should be 0.50 (SUMMARY_CONFIDENCE, lowered for corroboration)
        args = insert_call[0][0][1]  # params tuple
        assert json.loads(args[2])["overall"] == 0.50

    def test_no_summary_no_beliefs(self, hook_module, mock_conn):
        """No summary should not create beliefs."""
        conn, cursor, _ = mock_conn

        session_data = {"summary": "", "themes": [], "project_context": ""}
        count = hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)
        assert count == 0

    def test_short_summary_skipped(self, hook_module, mock_conn):
        """Summary shorter than MIN_SUMMARY_LENGTH should be skipped."""
        conn, cursor, _ = mock_conn

        session_data = {"summary": "Too short", "themes": [], "project_context": ""}
        count = hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)
        assert count == 0

    def test_themes_create_individual_beliefs(self, hook_module, mock_conn):
        """Each theme should become its own belief."""
        conn, cursor, _ = mock_conn
        cursor.execute = MagicMock()

        session_data = {
            "summary": None,
            "themes": [
                "Dimensional confidence scoring with geometric means",
                "Multi-signal ranking for belief retrieval",
                "Auto-capture in session-end hooks",
            ],
            "project_context": "valence",
        }

        count = hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)
        assert count == 3

    def test_short_themes_skipped(self, hook_module, mock_conn):
        """Themes shorter than MIN_THEME_LENGTH should be skipped."""
        conn, cursor, _ = mock_conn
        cursor.execute = MagicMock()

        session_data = {
            "summary": None,
            "themes": ["short", "This theme is long enough to capture"],
            "project_context": "",
        }

        count = hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)
        assert count == 1

    def test_max_cap_respected(self, hook_module, mock_conn):
        """More than MAX_AUTO_BELIEFS_PER_SESSION themes should be capped."""
        conn, cursor, _ = mock_conn
        cursor.execute = MagicMock()

        # 15 themes + 1 summary = 16 potential beliefs
        session_data = {
            "summary": "A sufficiently long session summary for auto capture testing purposes.",
            "themes": [f"Theme number {i} with enough length to pass" for i in range(15)],
            "project_context": "",
        }

        count = hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)
        assert count == hook_module.MAX_AUTO_BELIEFS_PER_SESSION

    def test_db_error_returns_zero(self, hook_module, mock_conn):
        """DB error during belief creation should return 0, not raise."""
        conn, cursor, _ = mock_conn
        import psycopg2
        cursor.execute = MagicMock(side_effect=psycopg2.Error("test error"))

        session_data = {
            "summary": "This is a long enough summary to trigger capture logic.",
            "themes": [],
            "project_context": "",
        }

        count = hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)
        assert count == 0

    def test_theme_confidence_is_045(self, hook_module, mock_conn):
        """Theme beliefs should have confidence 0.45."""
        conn, cursor, _ = mock_conn
        cursor.execute = MagicMock()

        session_data = {
            "summary": None,
            "themes": ["A theme long enough for testing purposes in auto-capture"],
            "project_context": "",
        }

        hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)

        calls = cursor.execute.call_args_list
        insert_call = [c for c in calls if "INSERT INTO beliefs" in str(c)]
        assert len(insert_call) >= 1
        args = insert_call[0][0][1]
        assert json.loads(args[2])["overall"] == 0.45

    def test_domain_path_from_project_context(self, hook_module, mock_conn):
        """Beliefs should use project_context as domain_path."""
        conn, cursor, _ = mock_conn
        cursor.execute = MagicMock()

        session_data = {
            "summary": "A summary that is long enough to be captured as a belief.",
            "themes": [],
            "project_context": "valence",
        }

        hook_module.auto_capture_beliefs(conn, "test-session-id", session_data)

        calls = cursor.execute.call_args_list
        insert_call = [c for c in calls if "INSERT INTO beliefs" in str(c)]
        args = insert_call[0][0][1]
        assert args[3] == ["valence"]


class TestCloseSession:
    """Tests for close_session."""

    def test_closes_session(self, hook_module, mock_conn):
        """Should update vkb_sessions status to completed."""
        conn, cursor, _ = mock_conn
        cursor.rowcount = 1

        result = hook_module.close_session(conn, "test-id")
        assert result is True

        calls = cursor.execute.call_args_list
        update_call = [c for c in calls if "UPDATE vkb_sessions" in str(c)]
        assert len(update_call) == 1

    def test_no_conn_returns_false(self, hook_module):
        """No connection should return False."""
        assert hook_module.close_session(None, "test-id") is False

    def test_no_session_id_returns_false(self, hook_module, mock_conn):
        """No session_id should return False."""
        conn, _, _ = mock_conn
        assert hook_module.close_session(conn, "") is False


class TestMain:
    """Tests for main entry point."""

    def test_main_outputs_json(self, hook_module, capsys):
        """Main should always output valid JSON."""
        with patch.dict(os.environ, {"VALENCE_SESSION_ID": ""}, clear=False):
            hook_module.main()
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {}

    def test_main_with_session_id(self, hook_module, capsys):
        """Main with session ID should attempt DB operations and still output JSON."""
        with patch.dict(os.environ, {"VALENCE_SESSION_ID": "test-session-id"}, clear=False):
            with patch.object(hook_module, "get_db_connection", return_value=None):
                hook_module.main()
        captured = capsys.readouterr()
        assert json.loads(captured.out) == {}
