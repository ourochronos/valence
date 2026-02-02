"""Tests for valence.vkb.mcp_server module."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

import pytest


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_cursor():
    """Create a mock cursor for database operations."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""
    from contextlib import contextmanager

    @contextmanager
    def fake_get_cursor(dict_cursor=True):
        yield mock_cursor

    with patch("valence.vkb.mcp_server.get_cursor", fake_get_cursor):
        yield mock_cursor


# ============================================================================
# session_start Tests
# ============================================================================

class TestSessionStart:
    """Tests for session_start function."""

    def test_basic_creation(self, mock_get_cursor):
        """Should create a basic session."""
        from valence.vkb.mcp_server import session_start

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "claude-code",
            "project_context": None,
            "status": "active",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
        }

        result = session_start("claude-code")

        assert result["success"] is True
        assert result["session"]["platform"] == "claude-code"
        assert result["session"]["status"] == "active"

    def test_with_project_context(self, mock_get_cursor):
        """Should accept project context."""
        from valence.vkb.mcp_server import session_start

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "matrix",
            "project_context": "test-project",
            "status": "active",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
        }

        result = session_start("matrix", project_context="test-project")

        assert result["success"] is True
        assert result["session"]["project_context"] == "test-project"

    def test_with_external_room_id(self, mock_get_cursor):
        """Should accept external room ID."""
        from valence.vkb.mcp_server import session_start

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "matrix",
            "project_context": None,
            "status": "active",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": "!room:matrix.org",
            "metadata": {},
        }

        result = session_start("matrix", external_room_id="!room:matrix.org")

        assert result["success"] is True
        assert result["session"]["external_room_id"] == "!room:matrix.org"

    def test_with_metadata(self, mock_get_cursor):
        """Should accept metadata."""
        from valence.vkb.mcp_server import session_start

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "api",
            "project_context": None,
            "status": "active",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {"key": "value"},
        }

        result = session_start("api", metadata={"key": "value"})

        assert result["success"] is True


# ============================================================================
# session_end Tests
# ============================================================================

class TestSessionEnd:
    """Tests for session_end function."""

    def test_basic_end(self, mock_get_cursor):
        """Should end a session."""
        from valence.vkb.mcp_server import session_end

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "claude-code",
            "project_context": None,
            "status": "completed",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": now,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
        }

        result = session_end(str(session_id))

        assert result["success"] is True
        assert result["session"]["status"] == "completed"

    def test_with_summary(self, mock_get_cursor):
        """Should accept summary."""
        from valence.vkb.mcp_server import session_end

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "claude-code",
            "project_context": None,
            "status": "completed",
            "summary": "Worked on tests",
            "themes": [],
            "started_at": now,
            "ended_at": now,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
        }

        result = session_end(str(session_id), summary="Worked on tests")

        assert result["success"] is True
        assert result["session"]["summary"] == "Worked on tests"

    def test_with_themes(self, mock_get_cursor):
        """Should accept themes."""
        from valence.vkb.mcp_server import session_end

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "claude-code",
            "project_context": None,
            "status": "completed",
            "summary": None,
            "themes": ["testing", "python"],
            "started_at": now,
            "ended_at": now,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
        }

        result = session_end(str(session_id), themes=["testing", "python"])

        assert result["success"] is True
        assert result["session"]["themes"] == ["testing", "python"]

    def test_abandoned_status(self, mock_get_cursor):
        """Should accept abandoned status."""
        from valence.vkb.mcp_server import session_end

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "claude-code",
            "project_context": None,
            "status": "abandoned",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": now,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
        }

        result = session_end(str(session_id), status="abandoned")

        assert result["success"] is True
        assert result["session"]["status"] == "abandoned"

    def test_not_found(self, mock_get_cursor):
        """Should return error if session not found."""
        from valence.vkb.mcp_server import session_end

        mock_get_cursor.fetchone.return_value = None

        result = session_end(str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]


# ============================================================================
# session_get Tests
# ============================================================================

class TestSessionGet:
    """Tests for session_get function."""

    def test_basic_get(self, mock_get_cursor):
        """Should get a session by ID."""
        from valence.vkb.mcp_server import session_get

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "claude-code",
            "project_context": "test-project",
            "status": "active",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
            "exchange_count": 5,
            "insight_count": 2,
        }

        result = session_get(str(session_id))

        assert result["success"] is True
        assert result["session"]["project_context"] == "test-project"

    def test_not_found(self, mock_get_cursor):
        """Should return error if session not found."""
        from valence.vkb.mcp_server import session_get

        mock_get_cursor.fetchone.return_value = None

        result = session_get(str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_include_exchanges(self, mock_get_cursor):
        """Should include exchanges when requested."""
        from valence.vkb.mcp_server import session_get

        session_id = uuid4()
        exchange_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "claude-code",
            "project_context": None,
            "status": "active",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": None,
            "metadata": {},
            "exchange_count": 1,
            "insight_count": 0,
        }

        mock_get_cursor.fetchall.return_value = [
            {
                "id": exchange_id,
                "session_id": session_id,
                "sequence": 1,
                "role": "user",
                "content": "Hello",
                "created_at": now,
                "tokens_approx": None,
                "tool_uses": [],
            }
        ]

        result = session_get(str(session_id), include_exchanges=True)

        assert result["success"] is True
        assert "exchanges" in result
        assert len(result["exchanges"]) == 1


# ============================================================================
# session_list Tests
# ============================================================================

class TestSessionList:
    """Tests for session_list function."""

    def test_basic_list(self, mock_get_cursor):
        """Should list sessions."""
        from valence.vkb.mcp_server import session_list

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.return_value = [
            {
                "id": session_id,
                "platform": "claude-code",
                "project_context": None,
                "status": "active",
                "summary": None,
                "themes": [],
                "started_at": now,
                "ended_at": None,
                "claude_session_id": None,
                "external_room_id": None,
                "metadata": {},
                "exchange_count": 0,
                "insight_count": 0,
            }
        ]

        result = session_list()

        assert result["success"] is True
        assert len(result["sessions"]) == 1

    def test_platform_filter(self, mock_get_cursor):
        """Should filter by platform."""
        from valence.vkb.mcp_server import session_list

        mock_get_cursor.fetchall.return_value = []

        result = session_list(platform="matrix")

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "platform = %s" in call_args[0]

    def test_status_filter(self, mock_get_cursor):
        """Should filter by status."""
        from valence.vkb.mcp_server import session_list

        mock_get_cursor.fetchall.return_value = []

        result = session_list(status="completed")

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "status = %s" in call_args[0]


# ============================================================================
# session_find_by_room Tests
# ============================================================================

class TestSessionFindByRoom:
    """Tests for session_find_by_room function."""

    def test_found(self, mock_get_cursor):
        """Should find active session by room ID."""
        from valence.vkb.mcp_server import session_find_by_room

        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": session_id,
            "platform": "matrix",
            "project_context": None,
            "status": "active",
            "summary": None,
            "themes": [],
            "started_at": now,
            "ended_at": None,
            "claude_session_id": None,
            "external_room_id": "!room:matrix.org",
            "metadata": {},
        }

        result = session_find_by_room("!room:matrix.org")

        assert result["success"] is True
        assert result["found"] is True
        assert result["session"]["external_room_id"] == "!room:matrix.org"

    def test_not_found(self, mock_get_cursor):
        """Should return found=False when no session exists."""
        from valence.vkb.mcp_server import session_find_by_room

        mock_get_cursor.fetchone.return_value = None

        result = session_find_by_room("!nonexistent:matrix.org")

        assert result["success"] is True
        assert result["found"] is False
        assert result["session"] is None


# ============================================================================
# exchange_add Tests
# ============================================================================

class TestExchangeAdd:
    """Tests for exchange_add function."""

    def test_basic_add(self, mock_get_cursor):
        """Should add an exchange."""
        from valence.vkb.mcp_server import exchange_add

        session_id = uuid4()
        exchange_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {"next_seq": 1},  # Sequence number
            {
                "id": exchange_id,
                "session_id": session_id,
                "sequence": 1,
                "role": "user",
                "content": "Hello",
                "created_at": now,
                "tokens_approx": None,
                "tool_uses": [],
            },
        ]

        result = exchange_add(str(session_id), "user", "Hello")

        assert result["success"] is True
        assert result["exchange"]["role"] == "user"
        assert result["exchange"]["content"] == "Hello"

    def test_auto_sequence(self, mock_get_cursor):
        """Should auto-increment sequence number."""
        from valence.vkb.mcp_server import exchange_add

        session_id = uuid4()
        exchange_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {"next_seq": 5},  # Already have 4 exchanges
            {
                "id": exchange_id,
                "session_id": session_id,
                "sequence": 5,
                "role": "assistant",
                "content": "Response",
                "created_at": now,
                "tokens_approx": None,
                "tool_uses": [],
            },
        ]

        result = exchange_add(str(session_id), "assistant", "Response")

        assert result["success"] is True
        assert result["exchange"]["sequence"] == 5

    def test_with_tokens(self, mock_get_cursor):
        """Should accept token count."""
        from valence.vkb.mcp_server import exchange_add

        session_id = uuid4()
        exchange_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {"next_seq": 1},
            {
                "id": exchange_id,
                "session_id": session_id,
                "sequence": 1,
                "role": "user",
                "content": "Test",
                "created_at": now,
                "tokens_approx": 150,
                "tool_uses": [],
            },
        ]

        result = exchange_add(
            str(session_id), "user", "Test", tokens_approx=150
        )

        assert result["success"] is True


# ============================================================================
# exchange_list Tests
# ============================================================================

class TestExchangeList:
    """Tests for exchange_list function."""

    def test_basic_list(self, mock_get_cursor):
        """Should list exchanges."""
        from valence.vkb.mcp_server import exchange_list

        session_id = uuid4()
        exchange_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.return_value = [
            {
                "id": exchange_id,
                "session_id": session_id,
                "sequence": 1,
                "role": "user",
                "content": "Hello",
                "created_at": now,
                "tokens_approx": None,
                "tool_uses": [],
            }
        ]

        result = exchange_list(str(session_id))

        assert result["success"] is True
        assert len(result["exchanges"]) == 1

    def test_with_pagination(self, mock_get_cursor):
        """Should support pagination."""
        from valence.vkb.mcp_server import exchange_list

        mock_get_cursor.fetchall.return_value = []

        result = exchange_list(str(uuid4()), limit=10, offset=5)

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "LIMIT" in call_args[0]
        assert "OFFSET" in call_args[0]


# ============================================================================
# pattern_record Tests
# ============================================================================

class TestPatternRecord:
    """Tests for pattern_record function."""

    def test_basic_creation(self, mock_get_cursor):
        """Should create a pattern."""
        from valence.vkb.mcp_server import pattern_record

        pattern_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": pattern_id,
            "type": "preference",
            "description": "Prefers dark mode",
            "evidence": [],
            "occurrence_count": 1,
            "confidence": 0.5,
            "status": "emerging",
            "first_observed": now,
            "last_observed": now,
        }

        result = pattern_record("preference", "Prefers dark mode")

        assert result["success"] is True
        assert result["pattern"]["type"] == "preference"
        assert result["pattern"]["status"] == "emerging"

    def test_with_evidence(self, mock_get_cursor):
        """Should accept evidence session IDs."""
        from valence.vkb.mcp_server import pattern_record

        pattern_id = uuid4()
        session_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": pattern_id,
            "type": "topic_recurrence",
            "description": "Discusses testing often",
            "evidence": [session_id],
            "occurrence_count": 1,
            "confidence": 0.5,
            "status": "emerging",
            "first_observed": now,
            "last_observed": now,
        }

        result = pattern_record(
            "topic_recurrence",
            "Discusses testing often",
            evidence=[str(session_id)]
        )

        assert result["success"] is True

    def test_with_confidence(self, mock_get_cursor):
        """Should accept initial confidence."""
        from valence.vkb.mcp_server import pattern_record

        pattern_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": pattern_id,
            "type": "preference",
            "description": "Test",
            "evidence": [],
            "occurrence_count": 1,
            "confidence": 0.7,
            "status": "emerging",
            "first_observed": now,
            "last_observed": now,
        }

        result = pattern_record("preference", "Test", confidence=0.7)

        assert result["success"] is True


# ============================================================================
# pattern_reinforce Tests
# ============================================================================

class TestPatternReinforce:
    """Tests for pattern_reinforce function."""

    def test_basic_reinforcement(self, mock_get_cursor):
        """Should reinforce a pattern."""
        from valence.vkb.mcp_server import pattern_reinforce

        pattern_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            # Get current pattern
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Test",
                "evidence": [],
                "occurrence_count": 1,
                "confidence": 0.5,
                "status": "emerging",
                "first_observed": now,
                "last_observed": now,
            },
            # Updated pattern
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Test",
                "evidence": [],
                "occurrence_count": 2,
                "confidence": 0.55,
                "status": "emerging",
                "first_observed": now,
                "last_observed": now,
            },
        ]

        result = pattern_reinforce(str(pattern_id))

        assert result["success"] is True
        assert result["pattern"]["occurrence_count"] == 2

    def test_confidence_growth(self, mock_get_cursor):
        """Should increase confidence asymptotically."""
        from valence.vkb.mcp_server import pattern_reinforce

        pattern_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Test",
                "evidence": [],
                "occurrence_count": 1,
                "confidence": 0.9,
                "status": "emerging",
                "first_observed": now,
                "last_observed": now,
            },
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Test",
                "evidence": [],
                "occurrence_count": 2,
                "confidence": 0.91,  # Small increase when already high
                "status": "emerging",
                "first_observed": now,
                "last_observed": now,
            },
        ]

        result = pattern_reinforce(str(pattern_id))

        assert result["success"] is True
        # Confidence should increase but not exceed 0.99
        assert result["pattern"]["confidence"] <= 0.99

    def test_status_transition(self, mock_get_cursor):
        """Should transition to established after enough occurrences."""
        from valence.vkb.mcp_server import pattern_reinforce

        pattern_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Test",
                "evidence": [],
                "occurrence_count": 4,  # At threshold
                "confidence": 0.7,
                "status": "emerging",
                "first_observed": now,
                "last_observed": now,
            },
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Test",
                "evidence": [],
                "occurrence_count": 5,
                "confidence": 0.73,
                "status": "established",  # Transitioned
                "first_observed": now,
                "last_observed": now,
            },
        ]

        result = pattern_reinforce(str(pattern_id))

        assert result["success"] is True
        assert result["pattern"]["status"] == "established"

    def test_not_found(self, mock_get_cursor):
        """Should return error if pattern not found."""
        from valence.vkb.mcp_server import pattern_reinforce

        mock_get_cursor.fetchone.return_value = None

        result = pattern_reinforce(str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]


# ============================================================================
# pattern_list Tests
# ============================================================================

class TestPatternList:
    """Tests for pattern_list function."""

    def test_basic_list(self, mock_get_cursor):
        """Should list patterns."""
        from valence.vkb.mcp_server import pattern_list

        pattern_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.return_value = [
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Test",
                "evidence": [],
                "occurrence_count": 1,
                "confidence": 0.5,
                "status": "emerging",
                "first_observed": now,
                "last_observed": now,
            }
        ]

        result = pattern_list()

        assert result["success"] is True
        assert len(result["patterns"]) == 1

    def test_type_filter(self, mock_get_cursor):
        """Should filter by type."""
        from valence.vkb.mcp_server import pattern_list

        mock_get_cursor.fetchall.return_value = []

        result = pattern_list(type="preference")

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "type = %s" in call_args[0]

    def test_status_filter(self, mock_get_cursor):
        """Should filter by status."""
        from valence.vkb.mcp_server import pattern_list

        mock_get_cursor.fetchall.return_value = []

        result = pattern_list(status="established")

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "status = %s" in call_args[0]

    def test_min_confidence(self, mock_get_cursor):
        """Should filter by minimum confidence."""
        from valence.vkb.mcp_server import pattern_list

        mock_get_cursor.fetchall.return_value = []

        result = pattern_list(min_confidence=0.7)

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "confidence >= %s" in call_args[0]


# ============================================================================
# pattern_search Tests
# ============================================================================

class TestPatternSearch:
    """Tests for pattern_search function."""

    def test_match(self, mock_get_cursor):
        """Should find matching patterns."""
        from valence.vkb.mcp_server import pattern_search

        pattern_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.return_value = [
            {
                "id": pattern_id,
                "type": "preference",
                "description": "Prefers dark mode in IDEs",
                "evidence": [],
                "occurrence_count": 3,
                "confidence": 0.7,
                "status": "emerging",
                "first_observed": now,
                "last_observed": now,
            }
        ]

        result = pattern_search("dark mode")

        assert result["success"] is True
        assert len(result["patterns"]) == 1

    def test_no_match(self, mock_get_cursor):
        """Should handle no matches."""
        from valence.vkb.mcp_server import pattern_search

        mock_get_cursor.fetchall.return_value = []

        result = pattern_search("nonexistent pattern")

        assert result["success"] is True
        assert result["patterns"] == []
        assert result["total_count"] == 0


# ============================================================================
# insight_extract Tests
# ============================================================================

class TestInsightExtract:
    """Tests for insight_extract function."""

    def test_basic_extraction(self, mock_get_cursor):
        """Should extract an insight and create a belief."""
        from valence.vkb.mcp_server import insight_extract

        session_id = uuid4()
        source_id = uuid4()
        belief_id = uuid4()
        insight_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {"id": source_id},  # Get source
            {
                "id": belief_id,
                "content": "User prefers Python",
                "confidence": {"overall": 0.8},
                "domain_path": ["preferences"],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": source_id,
                "extraction_method": "conversation_extraction",
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },  # Create belief
            {"id": insight_id},  # Create insight link
        ]

        result = insight_extract(str(session_id), "User prefers Python")

        assert result["success"] is True
        assert result["belief_id"] == str(belief_id)
        assert result["session_id"] == str(session_id)

    def test_with_domain_path(self, mock_get_cursor):
        """Should accept domain path."""
        from valence.vkb.mcp_server import insight_extract

        session_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {"id": uuid4()},
            {
                "id": belief_id,
                "content": "Test",
                "confidence": {"overall": 0.8},
                "domain_path": ["tech", "preferences"],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": "conversation_extraction",
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
            {"id": uuid4()},
        ]

        result = insight_extract(
            str(session_id),
            "Test",
            domain_path=["tech", "preferences"]
        )

        assert result["success"] is True

    def test_with_entities(self, mock_get_cursor):
        """Should create and link entities."""
        from valence.vkb.mcp_server import insight_extract

        session_id = uuid4()
        belief_id = uuid4()
        entity_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {"id": uuid4()},  # Source
            {
                "id": belief_id,
                "content": "Test",
                "confidence": {"overall": 0.8},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": "conversation_extraction",
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },  # Belief
            {"id": entity_id},  # Entity upsert
            {"id": uuid4()},  # Insight link
        ]

        result = insight_extract(
            str(session_id),
            "Test",
            entities=[{"name": "Python", "type": "tool"}]
        )

        assert result["success"] is True


# ============================================================================
# insight_list Tests
# ============================================================================

class TestInsightList:
    """Tests for insight_list function."""

    def test_basic_list(self, mock_get_cursor):
        """Should list insights from a session."""
        from valence.vkb.mcp_server import insight_list

        session_id = uuid4()
        insight_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.return_value = [
            {
                "id": insight_id,
                "session_id": session_id,
                "belief_id": belief_id,
                "extraction_method": "manual",
                "extracted_at": now,
                "content": "Test belief",
                "confidence": {"overall": 0.8},
                "domain_path": [],
                "belief_created_at": now,
            }
        ]

        result = insight_list(str(session_id))

        assert result["success"] is True
        assert len(result["insights"]) == 1
        assert result["insights"][0]["belief"]["content"] == "Test belief"

    def test_empty_list(self, mock_get_cursor):
        """Should handle sessions with no insights."""
        from valence.vkb.mcp_server import insight_list

        mock_get_cursor.fetchall.return_value = []

        result = insight_list(str(uuid4()))

        assert result["success"] is True
        assert result["insights"] == []
        assert result["total_count"] == 0
