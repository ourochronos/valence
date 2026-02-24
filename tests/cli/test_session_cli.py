# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Tests for session CLI commands."""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

from valence.cli.commands.sessions import (
    cmd_sessions_append,
    cmd_sessions_compile,
    cmd_sessions_finalize,
    cmd_sessions_flush,
    cmd_sessions_flush_stale,
    cmd_sessions_list,
    cmd_sessions_search,
    cmd_sessions_show,
    cmd_sessions_start,
)
from valence.cli.http_client import ValenceAPIError, ValenceConnectionError


class Args:
    """Mock argparse.Namespace for testing."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# =============================================================================
# Session Start Tests
# =============================================================================


def test_sessions_start_success():
    """Test successful session start."""
    args = Args(
        session_id="test-session-123",
        platform="openclaw",
        channel="discord",
        parent_session_id=None,
    )

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {
            "session_id": "test-session-123",
            "platform": "openclaw",
            "status": "active",
        }
        mock_get_client.return_value = mock_client

        result = cmd_sessions_start(args)

        assert result == 0
        mock_client.post.assert_called_once_with(
            "/sessions",
            body={
                "session_id": "test-session-123",
                "platform": "openclaw",
                "channel": "discord",
            },
        )


def test_sessions_start_with_parent():
    """Test session start with parent session ID."""
    args = Args(
        session_id="child-session",
        platform="openclaw",
        channel=None,
        parent_session_id="parent-session",
    )

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {"session_id": "child-session"}
        mock_get_client.return_value = mock_client

        result = cmd_sessions_start(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[1]["body"]["parent_session_id"] == "parent-session"


def test_sessions_start_connection_error():
    """Test session start with connection error."""
    args = Args(session_id="test", platform="openclaw", channel=None, parent_session_id=None)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceConnectionError("http://localhost:8420")
        mock_get_client.return_value = mock_client

        result = cmd_sessions_start(args)

        assert result == 1


def test_sessions_start_api_error():
    """Test session start with API error."""
    args = Args(session_id="test", platform="openclaw", channel=None, parent_session_id=None)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceAPIError(400, "INVALID", "Invalid platform")
        mock_get_client.return_value = mock_client

        result = cmd_sessions_start(args)

        assert result == 1


# =============================================================================
# Message Append Tests
# =============================================================================


def test_sessions_append_with_content():
    """Test appending message with --content."""
    args = Args(
        session_id="test-session",
        role="user",
        speaker="chris",
        content="Hello world",
    )

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {"id": 123, "content": "Hello world"}
        mock_get_client.return_value = mock_client

        result = cmd_sessions_append(args)

        assert result == 0
        mock_client.post.assert_called_once_with(
            "/sessions/test-session/messages",
            body={
                "speaker": "chris",
                "role": "user",
                "content": "Hello world",
            },
        )


def test_sessions_append_from_stdin():
    """Test appending message from stdin."""
    args = Args(
        session_id="test-session",
        role="assistant",
        speaker="jane",
        content=None,
    )

    stdin_content = "This is a multi-line\nmessage from stdin"

    with (
        patch("valence.cli.commands.sessions.get_client") as mock_get_client,
        patch("sys.stdin", io.StringIO(stdin_content)),
    ):
        mock_client = MagicMock()
        mock_client.post.return_value = {"id": 124}
        mock_get_client.return_value = mock_client

        result = cmd_sessions_append(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[1]["body"]["content"] == stdin_content


def test_sessions_append_no_content():
    """Test appending message with no content provided."""
    args = Args(
        session_id="test-session",
        role="user",
        speaker="chris",
        content=None,
    )

    with (
        patch("valence.cli.commands.sessions.get_client") as mock_get_client,
        patch("sys.stdin", io.StringIO("")),
    ):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = cmd_sessions_append(args)

        assert result == 1  # Should fail with no content


# =============================================================================
# Flush Tests
# =============================================================================


def test_sessions_flush_with_compile():
    """Test flushing session with compilation."""
    args = Args(session_id="test-session", compile=True)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {
            "session_id": "test-session",
            "chunk_index": 1,
            "message_count": 15,
            "flushed": True,
        }
        mock_get_client.return_value = mock_client

        result = cmd_sessions_flush(args)

        assert result == 0
        mock_client.post.assert_called_once_with(
            "/sessions/test-session/flush",
            params={"compile": "true"},
        )


def test_sessions_flush_no_compile():
    """Test flushing session without compilation."""
    args = Args(session_id="test-session", compile=False)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {"flushed": True}
        mock_get_client.return_value = mock_client

        result = cmd_sessions_flush(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[1]["params"]["compile"] == "false"


# =============================================================================
# Finalize Tests
# =============================================================================


def test_sessions_finalize():
    """Test finalizing session."""
    args = Args(session_id="test-session")

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {
            "session_id": "test-session",
            "status": "completed",
        }
        mock_get_client.return_value = mock_client

        result = cmd_sessions_finalize(args)

        assert result == 0
        mock_client.post.assert_called_once_with("/sessions/test-session/finalize")


# =============================================================================
# Search Tests
# =============================================================================


def test_sessions_search():
    """Test searching conversation sources."""
    args = Args(query="python testing", limit=10)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "results": [
                {"title": "Session about testing", "score": 0.95},
            ]
        }
        mock_get_client.return_value = mock_client

        result = cmd_sessions_search(args)

        assert result == 0
        mock_client.get.assert_called_once_with(
            "/sessions/search",
            params={"q": "python testing", "limit": 10},
        )


# =============================================================================
# List Tests
# =============================================================================


def test_sessions_list_all():
    """Test listing all sessions."""
    args = Args(status=None, since=None, limit=100)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = [
            {"session_id": "s1", "status": "active"},
            {"session_id": "s2", "status": "completed"},
        ]
        mock_get_client.return_value = mock_client

        result = cmd_sessions_list(args)

        assert result == 0
        mock_client.get.assert_called_once_with(
            "/sessions",
            params={"limit": 100},
        )


def test_sessions_list_filtered():
    """Test listing sessions with filters."""
    args = Args(status="active", since="2026-02-24", limit=50)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = []
        mock_get_client.return_value = mock_client

        result = cmd_sessions_list(args)

        assert result == 0
        call_args = mock_client.get.call_args
        assert call_args[1]["params"]["status"] == "active"
        assert call_args[1]["params"]["since"] == "2026-02-24"
        assert call_args[1]["params"]["limit"] == 50


# =============================================================================
# Show Tests
# =============================================================================


def test_sessions_show_without_messages():
    """Test showing session details without messages."""
    args = Args(session_id="test-session", show_messages=False)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.return_value = {
            "session_id": "test-session",
            "platform": "openclaw",
            "status": "active",
        }
        mock_get_client.return_value = mock_client

        result = cmd_sessions_show(args)

        assert result == 0
        mock_client.get.assert_called_once_with("/sessions/test-session")


def test_sessions_show_with_messages():
    """Test showing session details with messages."""
    args = Args(session_id="test-session", show_messages=True)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.get.side_effect = [
            {"session_id": "test-session", "platform": "openclaw"},
            [{"id": 1, "content": "Hello"}, {"id": 2, "content": "World"}],
        ]
        mock_get_client.return_value = mock_client

        result = cmd_sessions_show(args)

        assert result == 0
        # Should have called get twice: session details + messages
        assert mock_client.get.call_count == 2


# =============================================================================
# Compile Tests
# =============================================================================


def test_sessions_compile():
    """Test compiling session into article."""
    args = Args(session_id="test-session")

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {
            "article_id": "abc-123",
            "title": "Session: test-session",
        }
        mock_get_client.return_value = mock_client

        result = cmd_sessions_compile(args)

        assert result == 0
        mock_client.post.assert_called_once_with("/sessions/test-session/compile")


# =============================================================================
# Flush Stale Tests
# =============================================================================


def test_sessions_flush_stale_default():
    """Test flushing stale sessions with default threshold."""
    args = Args(stale_minutes=30)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {
            "flushed": ["session-1", "session-2"],
            "count": 2,
        }
        mock_get_client.return_value = mock_client

        result = cmd_sessions_flush_stale(args)

        assert result == 0
        mock_client.post.assert_called_once_with(
            "/sessions/flush-stale",
            params={"stale_minutes": 30},
        )


def test_sessions_flush_stale_custom_threshold():
    """Test flushing stale sessions with custom threshold."""
    args = Args(stale_minutes=60)

    with patch("valence.cli.commands.sessions.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.post.return_value = {"flushed": [], "count": 0}
        mock_get_client.return_value = mock_client

        result = cmd_sessions_flush_stale(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[1]["params"]["stale_minutes"] == 60
