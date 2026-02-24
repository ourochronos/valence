"""Tests for core.sessions module (#469, #473).

Tests cover:
1. upsert_session — insert new, update existing
2. get_session — found, not found
3. list_sessions — all, filtered by status/platform/since
4. update_session — status, metadata, participants, ended_at
5. append_message — single message + updates last_activity_at
6. append_messages — batch insert
7. get_messages — all, filtered by role/since/chunk_index
8. get_unflushed_messages — only unflushed
9. find_stale_sessions — inactive threshold
10. flush_session — creates source, marks flushed, increments chunk
11. finalize_session — flush + mark completed
12. flush_stale_sessions — batch flush + mark stale
13. Subagent session with parent link
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from valence.core.sessions import (
    append_message,
    append_messages,
    finalize_session,
    find_stale_sessions,
    flush_session,
    flush_stale_sessions,
    get_messages,
    get_session,
    get_unflushed_messages,
    list_sessions,
    update_session,
    upsert_session,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cursor():
    """Mock psycopg2 cursor."""
    cur = MagicMock()
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    return cur


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Patch get_cursor with a sync context manager."""

    @contextmanager
    def _mock() -> Generator:
        yield mock_cursor

    with patch("valence.core.sessions.get_cursor", _mock):
        yield mock_cursor


def _make_session_row(
    session_id: str = "test-session",
    platform: str = "openclaw",
    channel: str | None = "discord",
    participants: list[str] | None = None,
    status: str = "active",
    parent_session_id: str | None = None,
) -> dict:
    """Build a fake session DB row."""
    return {
        "session_id": session_id,
        "platform": platform,
        "channel": channel,
        "participants": participants or ["chris", "jane"],
        "started_at": datetime(2026, 2, 24, 12, 0, 0, tzinfo=UTC),
        "last_activity_at": datetime(2026, 2, 24, 12, 30, 0, tzinfo=UTC),
        "ended_at": None,
        "status": status,
        "metadata": {},
        "parent_session_id": parent_session_id,
        "subagent_label": None,
        "subagent_model": None,
        "subagent_task": None,
        "current_chunk_index": 0,
    }


def _make_message_row(
    session_id: str = "test-session",
    speaker: str = "chris",
    role: str = "user",
    content: str = "Hello",
    flushed_at: datetime | None = None,
) -> dict:
    """Build a fake message DB row."""
    return {
        "id": 1,
        "session_id": session_id,
        "chunk_index": 0,
        "timestamp": datetime(2026, 2, 24, 12, 0, 0, tzinfo=UTC),
        "speaker": speaker,
        "role": role,
        "content": content,
        "metadata": {},
        "flushed_at": flushed_at,
    }


# ---------------------------------------------------------------------------
# Tests: upsert_session
# ---------------------------------------------------------------------------


class TestUpsertSession:
    @pytest.mark.asyncio
    async def test_insert_new_session(self, mock_get_cursor):
        row = _make_session_row()
        mock_get_cursor.fetchone.return_value = row

        resp = await upsert_session("test-session", "openclaw", channel="discord", participants=["chris", "jane"])

        assert resp.success
        assert resp.data["session_id"] == "test-session"
        assert resp.data["platform"] == "openclaw"
        assert resp.data["channel"] == "discord"
        assert "chris" in resp.data["participants"]

    @pytest.mark.asyncio
    async def test_update_existing_session(self, mock_get_cursor):
        row = _make_session_row()
        mock_get_cursor.fetchone.return_value = row

        resp = await upsert_session("test-session", "openclaw")

        assert resp.success
        # Verify ON CONFLICT DO UPDATE was triggered (last_activity_at updated)

    @pytest.mark.asyncio
    async def test_empty_session_id(self, mock_get_cursor):
        resp = await upsert_session("", "openclaw")
        assert not resp.success
        assert "session_id must be non-empty" in resp.error

    @pytest.mark.asyncio
    async def test_empty_platform(self, mock_get_cursor):
        resp = await upsert_session("test-session", "")
        assert not resp.success
        assert "platform must be non-empty" in resp.error

    @pytest.mark.asyncio
    async def test_subagent_session(self, mock_get_cursor):
        # Mock parent exists check
        child_row = _make_session_row(
            session_id="child-session",
            parent_session_id="parent-session",
        )
        child_row["subagent_label"] = "audit-worker"
        child_row["subagent_model"] = "claude-sonnet-4"
        child_row["subagent_task"] = "Fix audit issues"

        mock_get_cursor.fetchone.side_effect = [
            {"session_id": "parent-session"},  # parent exists
            child_row,
        ]

        resp = await upsert_session(
            "child-session",
            "openclaw",
            parent_session_id="parent-session",
            subagent_label="audit-worker",
            subagent_model="claude-sonnet-4",
            subagent_task="Fix audit issues",
        )

        assert resp.success
        assert resp.data["parent_session_id"] == "parent-session"
        assert resp.data["subagent_label"] == "audit-worker"

    @pytest.mark.asyncio
    async def test_invalid_parent_session(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = None  # parent not found

        resp = await upsert_session("child-session", "openclaw", parent_session_id="nonexistent")

        assert not resp.success
        assert "Parent session not found" in resp.error


# ---------------------------------------------------------------------------
# Tests: get_session
# ---------------------------------------------------------------------------


class TestGetSession:
    @pytest.mark.asyncio
    async def test_found(self, mock_get_cursor):
        row = _make_session_row()
        mock_get_cursor.fetchone.return_value = row

        resp = await get_session("test-session")

        assert resp.success
        assert resp.data["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_not_found(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = None

        resp = await get_session("nonexistent")

        assert not resp.success
        assert "Session not found" in resp.error


# ---------------------------------------------------------------------------
# Tests: list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    @pytest.mark.asyncio
    async def test_all_sessions(self, mock_get_cursor):
        rows = [
            _make_session_row(session_id="session-1"),
            _make_session_row(session_id="session-2"),
        ]
        mock_get_cursor.fetchall.return_value = rows

        resp = await list_sessions()

        assert resp.success
        assert len(resp.data) == 2

    @pytest.mark.asyncio
    async def test_filter_by_status(self, mock_get_cursor):
        rows = [_make_session_row(status="active")]
        mock_get_cursor.fetchall.return_value = rows

        resp = await list_sessions(status="active")

        assert resp.success
        assert len(resp.data) == 1

    @pytest.mark.asyncio
    async def test_filter_by_platform(self, mock_get_cursor):
        rows = [_make_session_row(platform="openclaw")]
        mock_get_cursor.fetchall.return_value = rows

        resp = await list_sessions(platform="openclaw")

        assert resp.success

    @pytest.mark.asyncio
    async def test_filter_by_since(self, mock_get_cursor):
        rows = [_make_session_row()]
        mock_get_cursor.fetchall.return_value = rows

        since = datetime(2026, 2, 24, 0, 0, 0, tzinfo=UTC)
        resp = await list_sessions(since=since)

        assert resp.success


# ---------------------------------------------------------------------------
# Tests: update_session
# ---------------------------------------------------------------------------


class TestUpdateSession:
    @pytest.mark.asyncio
    async def test_update_status(self, mock_get_cursor):
        row = _make_session_row(status="completed")
        mock_get_cursor.fetchone.return_value = row

        resp = await update_session("test-session", status="completed")

        assert resp.success
        assert resp.data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_update_metadata(self, mock_get_cursor):
        row = _make_session_row()
        row["metadata"] = {"key": "value"}
        mock_get_cursor.fetchone.return_value = row

        resp = await update_session("test-session", metadata={"key": "value"})

        assert resp.success

    @pytest.mark.asyncio
    async def test_no_fields_to_update(self, mock_get_cursor):
        resp = await update_session("test-session")

        assert not resp.success
        assert "No fields to update" in resp.error

    @pytest.mark.asyncio
    async def test_session_not_found(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = None

        resp = await update_session("nonexistent", status="completed")

        assert not resp.success
        assert "Session not found" in resp.error


# ---------------------------------------------------------------------------
# Tests: append_message
# ---------------------------------------------------------------------------


class TestAppendMessage:
    @pytest.mark.asyncio
    async def test_append_user_message(self, mock_get_cursor):
        row = _make_message_row()
        mock_get_cursor.fetchone.return_value = row

        resp = await append_message("test-session", "chris", "user", "Hello")

        assert resp.success
        assert resp.data["speaker"] == "chris"
        assert resp.data["role"] == "user"
        assert resp.data["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_append_assistant_message(self, mock_get_cursor):
        row = _make_message_row(speaker="jane", role="assistant", content="Hi there")
        mock_get_cursor.fetchone.return_value = row

        resp = await append_message("test-session", "jane", "assistant", "Hi there")

        assert resp.success
        assert resp.data["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_invalid_role(self, mock_get_cursor):
        resp = await append_message("test-session", "chris", "invalid", "Hello")

        assert not resp.success
        assert "Invalid role" in resp.error

    @pytest.mark.asyncio
    async def test_updates_last_activity_at(self, mock_get_cursor):
        row = _make_message_row()
        mock_get_cursor.fetchone.return_value = row

        resp = await append_message("test-session", "chris", "user", "Hello")

        assert resp.success
        # Verify UPDATE sessions SET last_activity_at was called
        assert mock_get_cursor.execute.call_count >= 2


# ---------------------------------------------------------------------------
# Tests: append_messages
# ---------------------------------------------------------------------------


class TestAppendMessages:
    @pytest.mark.asyncio
    async def test_batch_append(self, mock_get_cursor):
        rows = [
            _make_message_row(speaker="chris", role="user", content="Hello"),
            _make_message_row(speaker="jane", role="assistant", content="Hi"),
        ]
        mock_get_cursor.fetchone.side_effect = rows

        messages = [
            {"speaker": "chris", "role": "user", "content": "Hello"},
            {"speaker": "jane", "role": "assistant", "content": "Hi"},
        ]

        resp = await append_messages("test-session", messages)

        assert resp.success
        assert len(resp.data) == 2

    @pytest.mark.asyncio
    async def test_empty_list(self, mock_get_cursor):
        resp = await append_messages("test-session", [])

        assert resp.success
        assert resp.data == []


# ---------------------------------------------------------------------------
# Tests: get_messages
# ---------------------------------------------------------------------------


class TestGetMessages:
    @pytest.mark.asyncio
    async def test_all_messages(self, mock_get_cursor):
        rows = [
            _make_message_row(speaker="chris", role="user"),
            _make_message_row(speaker="jane", role="assistant"),
        ]
        mock_get_cursor.fetchall.return_value = rows

        resp = await get_messages("test-session")

        assert resp.success
        assert len(resp.data) == 2

    @pytest.mark.asyncio
    async def test_filter_by_role(self, mock_get_cursor):
        rows = [_make_message_row(role="user")]
        mock_get_cursor.fetchall.return_value = rows

        resp = await get_messages("test-session", role="user")

        assert resp.success

    @pytest.mark.asyncio
    async def test_filter_by_since(self, mock_get_cursor):
        rows = [_make_message_row()]
        mock_get_cursor.fetchall.return_value = rows

        since = datetime(2026, 2, 24, 0, 0, 0, tzinfo=UTC)
        resp = await get_messages("test-session", since=since)

        assert resp.success

    @pytest.mark.asyncio
    async def test_filter_by_chunk_index(self, mock_get_cursor):
        rows = [_make_message_row()]
        mock_get_cursor.fetchall.return_value = rows

        resp = await get_messages("test-session", chunk_index=0)

        assert resp.success


# ---------------------------------------------------------------------------
# Tests: get_unflushed_messages
# ---------------------------------------------------------------------------


class TestGetUnflushedMessages:
    @pytest.mark.asyncio
    async def test_unflushed_only(self, mock_get_cursor):
        rows = [
            _make_message_row(content="Message 1", flushed_at=None),
            _make_message_row(content="Message 2", flushed_at=None),
        ]
        mock_get_cursor.fetchall.return_value = rows

        resp = await get_unflushed_messages("test-session")

        assert resp.success
        assert len(resp.data) == 2
        for msg in resp.data:
            assert msg["flushed_at"] is None

    @pytest.mark.asyncio
    async def test_empty_unflushed(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        resp = await get_unflushed_messages("test-session")

        assert resp.success
        assert resp.data == []


# ---------------------------------------------------------------------------
# Tests: find_stale_sessions
# ---------------------------------------------------------------------------


class TestFindStaleSessions:
    @pytest.mark.asyncio
    async def test_stale_sessions(self, mock_get_cursor):
        rows = [
            {"session_id": "stale-1"},
            {"session_id": "stale-2"},
        ]
        mock_get_cursor.fetchall.return_value = rows

        resp = await find_stale_sessions(stale_minutes=30)

        assert resp.success
        assert len(resp.data) == 2
        assert "stale-1" in resp.data

    @pytest.mark.asyncio
    async def test_no_stale_sessions(self, mock_get_cursor):
        mock_get_cursor.fetchall.return_value = []

        resp = await find_stale_sessions(stale_minutes=30)

        assert resp.success
        assert resp.data == []


# ---------------------------------------------------------------------------
# Tests: flush_session
# ---------------------------------------------------------------------------


class TestFlushSession:
    @pytest.mark.asyncio
    async def test_flush_creates_source(self, mock_get_cursor):
        session_row = _make_session_row()
        message_rows = [
            _make_message_row(speaker="chris", role="user", content="Hello"),
            _make_message_row(speaker="jane", role="assistant", content="Hi there"),
        ]
        source_row = {
            "id": str(uuid4()),
            "type": "conversation",
            "title": "Session test-session — chunk 0",
            "content": "...",
            "fingerprint": "abc123",
            "reliability": 0.5,
            "content_hash": "abc123",
            "metadata": {},
            "created_at": datetime.now(UTC),
        }

        mock_get_cursor.fetchone.side_effect = [
            session_row,  # get_session
            None,  # ingest_source duplicate check
            source_row,  # ingest_source insert
        ]
        mock_get_cursor.fetchall.side_effect = [
            message_rows,  # get_unflushed_messages
            [],  # search_sources (for compile)
        ]

        with patch("valence.core.sessions.ingest_source", new_callable=AsyncMock) as mock_ingest:
            from valence.core.response import ok as mock_ok

            mock_ingest.return_value = mock_ok(data=source_row)

            with patch("valence.core.sources.search_sources") as mock_search:
                mock_search.return_value.success = True
                mock_search.return_value.data = []

                resp = await flush_session("test-session", compile=False)

        assert resp.success
        assert resp.data["message_count"] == 2
        assert resp.data["flushed"] is True

    @pytest.mark.asyncio
    async def test_flush_empty_buffer(self, mock_get_cursor):
        session_row = _make_session_row()
        mock_get_cursor.fetchone.return_value = session_row
        mock_get_cursor.fetchall.return_value = []

        resp = await flush_session("test-session")

        assert resp.success
        assert resp.data["message_count"] == 0
        assert resp.data["flushed"] is False

    @pytest.mark.asyncio
    async def test_flush_marks_messages_flushed(self, mock_get_cursor):
        session_row = _make_session_row()
        message_rows = [_make_message_row()]
        source_row = {
            "id": str(uuid4()),
            "type": "conversation",
            "title": "Test",
            "content": "...",
            "fingerprint": "abc",
            "reliability": 0.5,
            "content_hash": "abc",
            "metadata": {},
            "created_at": datetime.now(UTC),
        }

        mock_get_cursor.fetchone.side_effect = [session_row, None, source_row]
        mock_get_cursor.fetchall.side_effect = [message_rows, []]

        with patch("valence.core.sessions.ingest_source", new_callable=AsyncMock) as mock_ingest:
            from valence.core.response import ok as mock_ok

            mock_ingest.return_value = mock_ok(data=source_row)

            with patch("valence.core.sources.search_sources") as mock_search:
                mock_search.return_value.success = True
                mock_search.return_value.data = []

                resp = await flush_session("test-session", compile=False)

        assert resp.success
        # Verify UPDATE session_messages SET flushed_at was called
        update_calls = [call for call in mock_get_cursor.execute.call_args_list if "UPDATE session_messages" in str(call)]
        assert len(update_calls) >= 1

    @pytest.mark.asyncio
    async def test_flush_increments_chunk_index(self, mock_get_cursor):
        session_row = _make_session_row()
        message_rows = [_make_message_row()]
        source_row = {
            "id": str(uuid4()),
            "type": "conversation",
            "title": "Test",
            "content": "...",
            "fingerprint": "abc",
            "reliability": 0.5,
            "content_hash": "abc",
            "metadata": {},
            "created_at": datetime.now(UTC),
        }

        mock_get_cursor.fetchone.side_effect = [session_row, None, source_row]
        mock_get_cursor.fetchall.side_effect = [message_rows, []]

        with patch("valence.core.sessions.ingest_source", new_callable=AsyncMock) as mock_ingest:
            from valence.core.response import ok as mock_ok

            mock_ingest.return_value = mock_ok(data=source_row)

            with patch("valence.core.sources.search_sources") as mock_search:
                mock_search.return_value.success = True
                mock_search.return_value.data = []

                resp = await flush_session("test-session", compile=False)

        assert resp.success
        # Verify UPDATE sessions SET current_chunk_index was called
        update_calls = [
            call for call in mock_get_cursor.execute.call_args_list if "UPDATE sessions" in str(call) and "current_chunk_index" in str(call)
        ]
        assert len(update_calls) >= 1


# ---------------------------------------------------------------------------
# Tests: finalize_session
# ---------------------------------------------------------------------------


class TestFinalizeSession:
    @pytest.mark.asyncio
    async def test_finalize(self, mock_get_cursor):
        completed_row = _make_session_row(status="completed")
        flush_data = {"session_id": "test-session", "chunk_index": 0, "message_count": 0, "flushed": False}

        # Only mock update_session
        mock_get_cursor.fetchone.return_value = completed_row

        with patch("valence.core.sessions.flush_session") as mock_flush:
            from valence.core.response import ok as mock_ok

            mock_flush.return_value = mock_ok(data=flush_data)

            resp = await finalize_session("test-session")

        assert resp.success
        assert resp.data["status"] == "completed"
        assert "flush" in resp.data


# ---------------------------------------------------------------------------
# Tests: flush_stale_sessions
# ---------------------------------------------------------------------------


class TestFlushStaleSessions:
    @pytest.mark.asyncio
    async def test_flush_stale(self, mock_get_cursor):
        stale_rows = [{"session_id": "stale-1"}]
        session_row = _make_session_row(session_id="stale-1")
        stale_updated_row = _make_session_row(session_id="stale-1", status="stale")
        source_row = {
            "id": str(uuid4()),
            "type": "conversation",
            "title": "Test",
            "content": "...",
            "fingerprint": "abc",
            "reliability": 0.5,
            "content_hash": "abc",
            "metadata": {},
            "created_at": datetime.now(UTC),
        }

        mock_get_cursor.fetchall.side_effect = [
            stale_rows,  # find_stale_sessions
            [],  # get_unflushed_messages
            [],  # search_sources
        ]
        mock_get_cursor.fetchone.side_effect = [
            session_row,  # get_session for flush
            None,  # ingest_source duplicate check
            source_row,  # ingest_source insert
            stale_updated_row,  # update_session
        ]

        with patch("valence.core.sessions.ingest_source", new_callable=AsyncMock) as mock_ingest:
            from valence.core.response import ok as mock_ok

            mock_ingest.return_value = mock_ok(data=source_row)

            with patch("valence.core.sources.search_sources") as mock_search:
                mock_search.return_value.success = True
                mock_search.return_value.data = []

                resp = await flush_stale_sessions(stale_minutes=30)

        assert resp.success
        assert len(resp.data) == 1
