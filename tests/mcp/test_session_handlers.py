"""Tests for session MCP handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from valence.mcp.handlers.sessions import (
    session_append,
    session_compile,
    session_finalize,
    session_flush,
    session_flush_stale,
    session_get,
    session_list,
    session_search,
    session_start,
)


def create_mock_response(success: bool, data: dict | None = None, error: str | None = None):
    """Create a mock ValenceResponse."""
    mock_response = MagicMock()
    mock_response.success = success
    mock_response.data = data
    mock_response.error = error
    return mock_response


class TestSessionStart:
    """Tests for session_start handler."""

    @patch("valence.mcp.handlers.sessions.upsert_session")
    def test_start_success(self, mock_upsert):
        """Test successful session start."""
        session_data = {
            "session_id": "test-session",
            "platform": "openclaw",
            "status": "active",
            "started_at": "2026-02-24T12:00:00Z",
        }
        mock_upsert.return_value = create_mock_response(success=True, data=session_data)

        result = session_start(session_id="test-session", platform="openclaw")

        assert result["success"] is True
        assert result["session"]["session_id"] == "test-session"
        assert result["session"]["platform"] == "openclaw"
        mock_upsert.assert_called_once()

    @patch("valence.mcp.handlers.sessions.upsert_session")
    def test_start_with_all_params(self, mock_upsert):
        """Test session start with all optional parameters."""
        session_data = {
            "session_id": "child-session",
            "platform": "openclaw",
            "channel": "discord",
            "participants": ["chris", "jane"],
            "parent_session_id": "parent-session",
            "subagent_label": "worker",
        }
        mock_upsert.return_value = create_mock_response(success=True, data=session_data)

        result = session_start(
            session_id="child-session",
            platform="openclaw",
            channel="discord",
            participants=["chris", "jane"],
            parent_session_id="parent-session",
            subagent_label="worker",
            subagent_model="sonnet",
            subagent_task="Test task",
        )

        assert result["success"] is True
        assert result["session"]["channel"] == "discord"
        assert result["session"]["participants"] == ["chris", "jane"]

    @patch("valence.mcp.handlers.sessions.upsert_session")
    def test_start_error(self, mock_upsert):
        """Test session start with error."""
        mock_upsert.return_value = create_mock_response(success=False, error="Invalid session_id")

        result = session_start(session_id="", platform="openclaw")

        assert result["success"] is False
        assert "Invalid session_id" in result["error"]


class TestSessionAppend:
    """Tests for session_append handler."""

    @patch("valence.mcp.handlers.sessions.append_message")
    def test_append_single_message(self, mock_append):
        """Test appending a single message."""
        message_data = {
            "id": 1,
            "session_id": "test-session",
            "speaker": "chris",
            "role": "user",
            "content": "Hello",
        }
        mock_append.return_value = create_mock_response(success=True, data=message_data)

        result = session_append(
            session_id="test-session",
            speaker="chris",
            role="user",
            content="Hello",
        )

        assert result["success"] is True
        assert len(result["messages"]) == 1
        assert result["messages"][0]["speaker"] == "chris"
        mock_append.assert_called_once()

    @patch("valence.mcp.handlers.sessions.append_messages")
    def test_append_batch_messages(self, mock_append_batch):
        """Test appending multiple messages in batch."""
        messages = [
            {"speaker": "chris", "role": "user", "content": "Hello"},
            {"speaker": "jane", "role": "assistant", "content": "Hi there"},
        ]
        mock_append_batch.return_value = create_mock_response(success=True, data=messages)

        result = session_append(session_id="test-session", messages=messages)

        assert result["success"] is True
        assert len(result["messages"]) == 2
        mock_append_batch.assert_called_once()

    def test_append_missing_params(self):
        """Test append with missing required parameters."""
        result = session_append(session_id="test-session")

        assert result["success"] is False
        assert "required" in result["error"].lower()

    @patch("valence.mcp.handlers.sessions.append_message")
    def test_append_with_metadata(self, mock_append):
        """Test appending message with metadata."""
        message_data = {
            "id": 1,
            "session_id": "test-session",
            "speaker": "chris",
            "role": "user",
            "content": "Test",
            "metadata": {"tool": "test_tool"},
        }
        mock_append.return_value = create_mock_response(success=True, data=message_data)

        result = session_append(
            session_id="test-session",
            speaker="chris",
            role="user",
            content="Test",
            message_metadata={"tool": "test_tool"},
        )

        assert result["success"] is True
        assert result["messages"][0]["metadata"]["tool"] == "test_tool"


class TestSessionFlush:
    """Tests for session_flush handler."""

    @patch("valence.mcp.handlers.sessions.flush_session")
    def test_flush_success(self, mock_flush):
        """Test successful session flush."""
        flush_data = {
            "session_id": "test-session",
            "chunk_index": 0,
            "message_count": 5,
            "source_id": str(uuid4()),
            "flushed": True,
        }
        mock_flush.return_value = create_mock_response(success=True, data=flush_data)

        result = session_flush(session_id="test-session", compile=True)

        assert result["success"] is True
        assert result["flush"]["flushed"] is True
        assert result["flush"]["message_count"] == 5
        mock_flush.assert_called_once_with(session_id="test-session", compile=True)

    @patch("valence.mcp.handlers.sessions.flush_session")
    def test_flush_no_messages(self, mock_flush):
        """Test flush with no unflushed messages."""
        flush_data = {
            "session_id": "test-session",
            "message_count": 0,
            "flushed": False,
        }
        mock_flush.return_value = create_mock_response(success=True, data=flush_data)

        result = session_flush(session_id="test-session")

        assert result["success"] is True
        assert result["flush"]["flushed"] is False
        assert result["flush"]["message_count"] == 0

    @patch("valence.mcp.handlers.sessions.flush_session")
    def test_flush_without_compile(self, mock_flush):
        """Test flush without triggering compilation."""
        flush_data = {"session_id": "test-session", "flushed": True}
        mock_flush.return_value = create_mock_response(success=True, data=flush_data)

        result = session_flush(session_id="test-session", compile=False)

        assert result["success"] is True
        mock_flush.assert_called_once_with(session_id="test-session", compile=False)


class TestSessionFinalize:
    """Tests for session_finalize handler."""

    @patch("valence.mcp.handlers.sessions.finalize_session")
    def test_finalize_success(self, mock_finalize):
        """Test successful session finalization."""
        finalize_data = {
            "session_id": "test-session",
            "status": "completed",
            "flush": {"message_count": 3, "flushed": True},
        }
        mock_finalize.return_value = create_mock_response(success=True, data=finalize_data)

        result = session_finalize(session_id="test-session")

        assert result["success"] is True
        assert result["finalization"]["status"] == "completed"
        mock_finalize.assert_called_once()

    @patch("valence.mcp.handlers.sessions.finalize_session")
    def test_finalize_error(self, mock_finalize):
        """Test finalization error."""
        mock_finalize.return_value = create_mock_response(success=False, error="Session not found")

        result = session_finalize(session_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestSessionSearch:
    """Tests for session_search handler."""

    @patch("valence.mcp.handlers.sessions.search_sources")
    def test_search_success(self, mock_search):
        """Test successful session search."""
        sources = [
            {"id": str(uuid4()), "type": "conversation", "title": "Session test-1"},
            {"id": str(uuid4()), "type": "conversation", "title": "Session test-2"},
            {"id": str(uuid4()), "type": "document", "title": "Not a conversation"},
        ]
        mock_search.return_value = create_mock_response(success=True, data=sources)

        result = session_search(query="test query", limit=10)

        assert result["success"] is True
        # Should filter to only conversation types
        assert len(result["results"]) == 2
        assert all(r["type"] == "conversation" for r in result["results"])

    @patch("valence.mcp.handlers.sessions.search_sources")
    def test_search_no_results(self, mock_search):
        """Test search with no results."""
        mock_search.return_value = create_mock_response(success=True, data=[])

        result = session_search(query="nonexistent")

        assert result["success"] is True
        assert len(result["results"]) == 0


class TestSessionList:
    """Tests for session_list handler."""

    @patch("valence.mcp.handlers.sessions.list_sessions")
    def test_list_all(self, mock_list):
        """Test listing all sessions."""
        sessions = [
            {"session_id": "session-1", "platform": "openclaw", "status": "active"},
            {"session_id": "session-2", "platform": "openclaw", "status": "completed"},
        ]
        mock_list.return_value = create_mock_response(success=True, data=sessions)

        result = session_list()

        assert result["success"] is True
        assert len(result["sessions"]) == 2
        mock_list.assert_called_once()

    @patch("valence.mcp.handlers.sessions.list_sessions")
    def test_list_with_filters(self, mock_list):
        """Test listing sessions with filters."""
        sessions = [{"session_id": "active-session", "status": "active"}]
        mock_list.return_value = create_mock_response(success=True, data=sessions)

        result = session_list(status="active", platform="openclaw", limit=5)

        assert result["success"] is True
        assert len(result["sessions"]) == 1
        mock_list.assert_called_once()

    @patch("valence.mcp.handlers.sessions.list_sessions")
    def test_list_with_since(self, mock_list):
        """Test listing sessions with since filter."""
        sessions = [{"session_id": "recent-session"}]
        mock_list.return_value = create_mock_response(success=True, data=sessions)

        result = session_list(since="2026-02-24T12:00:00Z")

        assert result["success"] is True
        mock_list.assert_called_once()

    def test_list_invalid_since(self):
        """Test listing with invalid since timestamp."""
        result = session_list(since="not-a-timestamp")

        assert result["success"] is False
        assert "Invalid ISO timestamp" in result["error"]


class TestSessionGet:
    """Tests for session_get handler."""

    @patch("valence.mcp.handlers.sessions.get_messages")
    @patch("valence.mcp.handlers.sessions.get_session")
    def test_get_with_messages(self, mock_get_session, mock_get_messages):
        """Test getting session with messages."""
        session_data = {"session_id": "test-session", "platform": "openclaw"}
        messages_data = [
            {"id": 1, "speaker": "chris", "role": "user", "content": "Hello"},
            {"id": 2, "speaker": "jane", "role": "assistant", "content": "Hi"},
        ]
        mock_get_session.return_value = create_mock_response(success=True, data=session_data)
        mock_get_messages.return_value = create_mock_response(success=True, data=messages_data)

        result = session_get(session_id="test-session", include_messages=True)

        assert result["success"] is True
        assert result["session"]["session_id"] == "test-session"
        assert len(result["messages"]) == 2
        mock_get_session.assert_called_once()
        mock_get_messages.assert_called_once()

    @patch("valence.mcp.handlers.sessions.get_session")
    def test_get_without_messages(self, mock_get_session):
        """Test getting session without messages."""
        session_data = {"session_id": "test-session", "platform": "openclaw"}
        mock_get_session.return_value = create_mock_response(success=True, data=session_data)

        result = session_get(session_id="test-session", include_messages=False)

        assert result["success"] is True
        assert result["session"]["session_id"] == "test-session"
        assert "messages" not in result

    @patch("valence.mcp.handlers.sessions.get_session")
    def test_get_not_found(self, mock_get_session):
        """Test getting nonexistent session."""
        mock_get_session.return_value = create_mock_response(success=False, error="Session not found")

        result = session_get(session_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()


class TestSessionCompile:
    """Tests for session_compile handler."""

    @patch("valence.mcp.handlers.sessions.compile_article")
    @patch("valence.mcp.handlers.sessions.search_sources")
    def test_compile_success(self, mock_search, mock_compile):
        """Test successful session compilation."""
        sources = [
            {"id": str(uuid4()), "type": "conversation"},
            {"id": str(uuid4()), "type": "conversation"},
        ]
        article_data = {"id": str(uuid4()), "title": "Session Summary"}
        mock_search.return_value = create_mock_response(success=True, data=sources)
        mock_compile.return_value = create_mock_response(success=True, data=article_data)

        result = session_compile(session_id="test-session")

        assert result["success"] is True
        assert result["article"]["title"] == "Session Summary"
        mock_search.assert_called_once()
        mock_compile.assert_called_once()

    @patch("valence.mcp.handlers.sessions.search_sources")
    def test_compile_no_sources(self, mock_search):
        """Test compilation with no sources found."""
        mock_search.return_value = create_mock_response(success=True, data=[])

        result = session_compile(session_id="empty-session")

        assert result["success"] is False
        assert "No sources found" in result["error"]

    @patch("valence.mcp.handlers.sessions.search_sources")
    def test_compile_search_error(self, mock_search):
        """Test compilation when source search fails."""
        mock_search.return_value = create_mock_response(success=False, error="Search failed")

        result = session_compile(session_id="test-session")

        assert result["success"] is False
        assert "Failed to find sources" in result["error"]


class TestSessionFlushStale:
    """Tests for session_flush_stale handler."""

    @patch("valence.mcp.handlers.sessions.flush_stale_sessions")
    def test_flush_stale_success(self, mock_flush_stale):
        """Test flushing stale sessions."""
        flushed_data = [
            {"session_id": "stale-1", "message_count": 3},
            {"session_id": "stale-2", "message_count": 5},
        ]
        mock_flush_stale.return_value = create_mock_response(success=True, data=flushed_data)

        result = session_flush_stale(stale_minutes=30)

        assert result["success"] is True
        assert len(result["flushed_sessions"]) == 2
        mock_flush_stale.assert_called_once_with(stale_minutes=30)

    @patch("valence.mcp.handlers.sessions.flush_stale_sessions")
    def test_flush_stale_custom_threshold(self, mock_flush_stale):
        """Test flushing with custom threshold."""
        mock_flush_stale.return_value = create_mock_response(success=True, data=[])

        result = session_flush_stale(stale_minutes=60)

        assert result["success"] is True
        mock_flush_stale.assert_called_once_with(stale_minutes=60)

    @patch("valence.mcp.handlers.sessions.flush_stale_sessions")
    def test_flush_stale_error(self, mock_flush_stale):
        """Test flush stale with error."""
        mock_flush_stale.return_value = create_mock_response(success=False, error="Database error")

        result = session_flush_stale()

        assert result["success"] is False
        assert "Database error" in result["error"]
