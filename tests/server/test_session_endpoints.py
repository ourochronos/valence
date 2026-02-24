"""Tests for session REST endpoints (#470)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

from valence.core.response import ok


@pytest.fixture(autouse=True)
def reset_stores():
    """Reset all stores between tests."""
    import valence.server.app as app_module
    import valence.server.auth as auth_module
    import valence.server.config as config_module

    config_module._settings = None
    auth_module._token_store = None
    app_module._rate_limits.clear()
    app_module._openapi_spec_cache = None

    yield

    config_module._settings = None
    auth_module._token_store = None
    app_module._rate_limits.clear()
    app_module._openapi_spec_cache = None


@pytest.fixture
def app_env(monkeypatch, tmp_path):
    """Set up application environment."""
    token_file = tmp_path / "tokens.json"
    token_file.write_text('{"tokens": []}')

    monkeypatch.setenv("VALENCE_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("VALENCE_RATE_LIMIT_RPM", "60")
    monkeypatch.setenv("VALENCE_EXTERNAL_URL", "http://localhost:8420")
    monkeypatch.setenv("VALENCE_OAUTH_ENABLED", "false")

    return {"token_file": token_file}


@pytest.fixture
def mock_db():
    """Mock database for tests."""
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value = None

    def mock_context(*args, **kwargs):
        class CM:
            def __enter__(self):
                return mock_cursor

            def __exit__(self, *args):
                pass

        return CM()

    with patch("valence.core.db.get_cursor", mock_context):
        yield mock_cursor


@pytest.fixture
def client(app_env, mock_db) -> TestClient:
    """Create test client."""
    from valence.server.app import create_app

    app = create_app()
    return TestClient(app, raise_server_exceptions=False)


API_V1 = "/api/v1"


# =============================================================================
# SESSION UPSERT
# =============================================================================


class TestUpsertSession:
    """Tests for POST /api/v1/sessions (upsert)."""

    @patch("valence.server.session_endpoints.upsert_session")
    async def test_upsert_new_session(self, mock_upsert, client):
        """Test creating a new session."""
        session_data = {
            "session_id": "test-session-1",
            "platform": "openclaw",
            "channel": "discord",
            "participants": ["chris", "jane"],
            "metadata": {"room": "command"},
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "last_activity_at": datetime.now(UTC).isoformat(),
            "ended_at": None,
            "parent_session_id": None,
            "subagent_label": None,
            "subagent_model": None,
            "subagent_task": None,
            "current_chunk_index": 0,
        }

        mock_upsert.return_value = ok(data=session_data)

        response = client.post(
            f"{API_V1}/sessions",
            json={
                "session_id": "test-session-1",
                "platform": "openclaw",
                "channel": "discord",
                "participants": ["chris", "jane"],
                "metadata": {"room": "command"},
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["session_id"] == "test-session-1"
        assert data["platform"] == "openclaw"
        assert data["channel"] == "discord"
        assert data["participants"] == ["chris", "jane"]

    @patch("valence.server.session_endpoints.upsert_session")
    async def test_upsert_subagent_session(self, mock_upsert, client):
        """Test creating a subagent session."""
        session_data = {
            "session_id": "subagent-1",
            "platform": "openclaw",
            "parent_session_id": "parent-1",
            "subagent_label": "worker:audit",
            "subagent_model": "sonnet-4",
            "subagent_task": "Audit database pool",
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "last_activity_at": datetime.now(UTC).isoformat(),
            "ended_at": None,
            "channel": None,
            "participants": [],
            "metadata": {},
            "current_chunk_index": 0,
        }

        mock_upsert.return_value = ok(data=session_data)

        response = client.post(
            f"{API_V1}/sessions",
            json={
                "session_id": "subagent-1",
                "platform": "openclaw",
                "parent_session_id": "parent-1",
                "subagent_label": "worker:audit",
                "subagent_model": "sonnet-4",
                "subagent_task": "Audit database pool",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["parent_session_id"] == "parent-1"
        assert data["subagent_label"] == "worker:audit"


# =============================================================================
# SESSION LIST
# =============================================================================


class TestListSessions:
    """Tests for GET /api/v1/sessions."""

    @patch("valence.server.session_endpoints.list_sessions")
    async def test_list_all_sessions(self, mock_list, client):
        """Test listing all sessions."""
        sessions_data = [
            {
                "session_id": "session-1",
                "platform": "openclaw",
                "status": "active",
                "started_at": datetime.now(UTC).isoformat(),
                "last_activity_at": datetime.now(UTC).isoformat(),
                "ended_at": None,
                "channel": "discord",
                "participants": ["chris"],
                "metadata": {},
                "parent_session_id": None,
                "subagent_label": None,
                "subagent_model": None,
                "subagent_task": None,
                "current_chunk_index": 0,
            },
            {
                "session_id": "session-2",
                "platform": "claude-code",
                "status": "completed",
                "started_at": datetime.now(UTC).isoformat(),
                "last_activity_at": datetime.now(UTC).isoformat(),
                "ended_at": datetime.now(UTC).isoformat(),
                "channel": None,
                "participants": [],
                "metadata": {},
                "parent_session_id": None,
                "subagent_label": None,
                "subagent_model": None,
                "subagent_task": None,
                "current_chunk_index": 2,
            },
        ]

        mock_list.return_value = ok(data=sessions_data)

        response = client.get(f"{API_V1}/sessions")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["session_id"] == "session-1"
        assert data[1]["session_id"] == "session-2"

    @patch("valence.server.session_endpoints.list_sessions")
    async def test_list_sessions_with_filters(self, mock_list, client):
        """Test listing sessions with filters."""
        sessions_data = [
            {
                "session_id": "session-1",
                "platform": "openclaw",
                "status": "active",
                "started_at": datetime.now(UTC).isoformat(),
                "last_activity_at": datetime.now(UTC).isoformat(),
                "ended_at": None,
                "channel": "discord",
                "participants": ["chris"],
                "metadata": {},
                "parent_session_id": None,
                "subagent_label": None,
                "subagent_model": None,
                "subagent_task": None,
                "current_chunk_index": 0,
            }
        ]

        mock_list.return_value = ok(data=sessions_data)

        response = client.get(f"{API_V1}/sessions?status=active&platform=openclaw&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["status"] == "active"


# =============================================================================
# SESSION GET
# =============================================================================


class TestGetSession:
    """Tests for GET /api/v1/sessions/{session_id}."""

    @patch("valence.server.session_endpoints.get_session")
    async def test_get_session_success(self, mock_get, client):
        """Test getting a session by ID."""
        session_data = {
            "session_id": "test-session",
            "platform": "openclaw",
            "status": "active",
            "started_at": datetime.now(UTC).isoformat(),
            "last_activity_at": datetime.now(UTC).isoformat(),
            "ended_at": None,
            "channel": "discord",
            "participants": ["chris", "jane"],
            "metadata": {"room": "command"},
            "parent_session_id": None,
            "subagent_label": None,
            "subagent_model": None,
            "subagent_task": None,
            "current_chunk_index": 1,
        }

        mock_get.return_value = ok(data=session_data)

        response = client.get(f"{API_V1}/sessions/test-session")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session"
        assert data["current_chunk_index"] == 1


# =============================================================================
# SESSION UPDATE
# =============================================================================


class TestUpdateSession:
    """Tests for PATCH /api/v1/sessions/{session_id}."""

    @patch("valence.server.session_endpoints.update_session")
    async def test_update_session_status(self, mock_update, client):
        """Test updating session status."""
        updated_data = {
            "session_id": "test-session",
            "platform": "openclaw",
            "status": "completed",
            "started_at": datetime.now(UTC).isoformat(),
            "last_activity_at": datetime.now(UTC).isoformat(),
            "ended_at": datetime.now(UTC).isoformat(),
            "channel": "discord",
            "participants": ["chris"],
            "metadata": {},
            "parent_session_id": None,
            "subagent_label": None,
            "subagent_model": None,
            "subagent_task": None,
            "current_chunk_index": 0,
        }

        mock_update.return_value = ok(data=updated_data)

        response = client.patch(f"{API_V1}/sessions/test-session", json={"status": "completed"})

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"


# =============================================================================
# MESSAGES
# =============================================================================


class TestAppendMessages:
    """Tests for POST /api/v1/sessions/{session_id}/messages."""

    @patch("valence.server.session_endpoints.append_message")
    async def test_append_single_message(self, mock_append, client):
        """Test appending a single message."""
        message_data = {
            "id": 1,
            "session_id": "test-session",
            "chunk_index": 0,
            "timestamp": datetime.now(UTC).isoformat(),
            "speaker": "chris",
            "role": "user",
            "content": "Hello, world!",
            "metadata": {},
            "flushed_at": None,
        }

        mock_append.return_value = ok(data=message_data)

        response = client.post(
            f"{API_V1}/sessions/test-session/messages",
            json={"speaker": "chris", "role": "user", "content": "Hello, world!"},
        )

        assert response.status_code == 201
        data = response.json()
        assert data["speaker"] == "chris"
        assert data["content"] == "Hello, world!"

    @patch("valence.server.session_endpoints.append_messages")
    async def test_append_message_batch(self, mock_append, client):
        """Test appending multiple messages."""
        messages_data = [
            {
                "id": 1,
                "session_id": "test-session",
                "chunk_index": 0,
                "timestamp": datetime.now(UTC).isoformat(),
                "speaker": "chris",
                "role": "user",
                "content": "Hello",
                "metadata": {},
                "flushed_at": None,
            },
            {
                "id": 2,
                "session_id": "test-session",
                "chunk_index": 0,
                "timestamp": datetime.now(UTC).isoformat(),
                "speaker": "jane",
                "role": "assistant",
                "content": "Hi there",
                "metadata": {},
                "flushed_at": None,
            },
        ]

        mock_append.return_value = ok(data=messages_data)

        response = client.post(
            f"{API_V1}/sessions/test-session/messages",
            json={
                "messages": [
                    {"speaker": "chris", "role": "user", "content": "Hello"},
                    {"speaker": "jane", "role": "assistant", "content": "Hi there"},
                ]
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert len(data) == 2
        assert data[0]["speaker"] == "chris"
        assert data[1]["speaker"] == "jane"


class TestGetMessages:
    """Tests for GET /api/v1/sessions/{session_id}/messages."""

    @patch("valence.server.session_endpoints.get_messages")
    async def test_get_all_messages(self, mock_get, client):
        """Test getting all messages for a session."""
        messages_data = [
            {
                "id": 1,
                "session_id": "test-session",
                "chunk_index": 0,
                "timestamp": datetime.now(UTC).isoformat(),
                "speaker": "chris",
                "role": "user",
                "content": "Hello",
                "metadata": {},
                "flushed_at": None,
            }
        ]

        mock_get.return_value = ok(data=messages_data)

        response = client.get(f"{API_V1}/sessions/test-session/messages")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["content"] == "Hello"


# =============================================================================
# FLUSH & COMPILE
# =============================================================================


class TestFlush:
    """Tests for POST /api/v1/sessions/{session_id}/flush."""

    @patch("valence.server.session_endpoints.flush_session")
    async def test_flush_session(self, mock_flush, client):
        """Test flushing a session."""
        flush_data = {
            "session_id": "test-session",
            "chunk_index": 0,
            "message_count": 5,
            "source_id": "source-123",
            "flushed": True,
        }

        mock_flush.return_value = ok(data=flush_data)

        response = client.post(f"{API_V1}/sessions/test-session/flush?compile=true")

        assert response.status_code == 200
        data = response.json()
        assert data["flushed"] is True
        assert data["message_count"] == 5
        assert data["source_id"] == "source-123"


class TestFinalize:
    """Tests for POST /api/v1/sessions/{session_id}/finalize."""

    @patch("valence.server.session_endpoints.finalize_session")
    async def test_finalize_session(self, mock_finalize, client):
        """Test finalizing a session."""
        finalize_data = {
            "session_id": "test-session",
            "status": "completed",
            "flush": {
                "session_id": "test-session",
                "chunk_index": 1,
                "message_count": 3,
                "source_id": "source-456",
                "flushed": True,
            },
        }

        mock_finalize.return_value = ok(data=finalize_data)

        response = client.post(f"{API_V1}/sessions/test-session/finalize")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["flush"]["flushed"] is True


class TestFlushStale:
    """Tests for POST /api/v1/sessions/flush-stale."""

    @patch("valence.server.session_endpoints.flush_stale_sessions")
    async def test_flush_stale_sessions(self, mock_flush_stale, client):
        """Test flushing stale sessions."""
        flush_results = [
            {
                "session_id": "stale-1",
                "chunk_index": 0,
                "message_count": 2,
                "source_id": "source-1",
                "flushed": True,
            },
            {
                "session_id": "stale-2",
                "chunk_index": 1,
                "message_count": 4,
                "source_id": "source-2",
                "flushed": True,
            },
        ]

        mock_flush_stale.return_value = ok(data=flush_results)

        response = client.post(f"{API_V1}/sessions/flush-stale?stale_minutes=30")

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["flushed"]) == 2


# =============================================================================
# SEARCH
# =============================================================================


class TestSearch:
    """Tests for GET /api/v1/sessions/search."""

    @patch("valence.core.sources.search_sources")
    async def test_search_sessions(self, mock_search, client):
        """Test searching conversation sources."""
        search_results = [
            {
                "id": "source-1",
                "type": "conversation",
                "title": "Session test-1 — chunk 0",
                "content": "Transcript content",
                "metadata": {"session_key": "test-1"},
            }
        ]

        mock_search.return_value = ok(data=search_results)

        response = client.get(f"{API_V1}/sessions/search?q=transcript&limit=10")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["type"] == "conversation"
