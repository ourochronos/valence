"""Tests for VKB tool handlers.

Tests cover:
- session_start / session_end / session_get / session_list - session management
- session_find_by_room - finding sessions by room ID
- exchange_add / exchange_list - exchange operations
- pattern_record / pattern_reinforce / pattern_list / pattern_search - pattern management
- insight_extract / insight_list - insight extraction
- handle_vkb_tool - routing
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from valence.vkb.tools import (
    VKB_HANDLERS,
    VKB_TOOLS,
    exchange_add,
    exchange_list,
    handle_vkb_tool,
    insight_extract,
    insight_list,
    pattern_list,
    pattern_record,
    pattern_reinforce,
    pattern_search,
    session_end,
    session_find_by_room,
    session_get,
    session_list,
    session_start,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""

    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.vkb.tools.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def sample_session_row():
    """Create a sample session row."""

    def _factory(
        id: UUID | None = None,
        platform: str = "claude-code",
        status: str = "active",
        **kwargs,
    ):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "platform": platform,
            "project_context": kwargs.get("project_context"),
            "status": status,
            "summary": kwargs.get("summary"),
            "themes": kwargs.get("themes", []),
            "started_at": kwargs.get("started_at", now),
            "ended_at": kwargs.get("ended_at"),
            "claude_session_id": kwargs.get("claude_session_id"),
            "external_room_id": kwargs.get("external_room_id"),
            "metadata": kwargs.get("metadata", {}),
            "exchange_count": kwargs.get("exchange_count"),
            "insight_count": kwargs.get("insight_count"),
        }

    return _factory


@pytest.fixture
def sample_exchange_row():
    """Create a sample exchange row."""

    def _factory(
        id: UUID | None = None,
        session_id: UUID | None = None,
        sequence: int = 1,
        role: str = "user",
        content: str = "Test message",
        **kwargs,
    ):
        return {
            "id": id or uuid4(),
            "session_id": session_id or uuid4(),
            "sequence": sequence,
            "role": role,
            "content": content,
            "created_at": kwargs.get("created_at", datetime.now()),
            "tokens_approx": kwargs.get("tokens_approx"),
            "tool_uses": kwargs.get("tool_uses", []),
        }

    return _factory


@pytest.fixture
def sample_pattern_row():
    """Create a sample pattern row."""

    def _factory(
        id: UUID | None = None,
        type: str = "topic_recurrence",
        description: str = "Test pattern",
        **kwargs,
    ):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "type": type,
            "description": description,
            "evidence": kwargs.get("evidence", []),
            "occurrence_count": kwargs.get("occurrence_count", 1),
            "confidence": kwargs.get("confidence", 0.5),
            "status": kwargs.get("status", "emerging"),
            "first_observed": kwargs.get("first_observed", now),
            "last_observed": kwargs.get("last_observed", now),
        }

    return _factory


# =============================================================================
# TOOL DEFINITIONS TESTS
# =============================================================================


class TestToolDefinitions:
    """Tests for tool definitions."""

    def test_vkb_tools_list(self):
        """Test that VKB_TOOLS contains expected tools."""
        tool_names = [t.name for t in VKB_TOOLS]

        assert "session_start" in tool_names
        assert "session_end" in tool_names
        assert "session_get" in tool_names
        assert "session_list" in tool_names
        assert "session_find_by_room" in tool_names
        assert "exchange_add" in tool_names
        assert "exchange_list" in tool_names
        assert "pattern_record" in tool_names
        assert "pattern_reinforce" in tool_names
        assert "pattern_list" in tool_names
        assert "pattern_search" in tool_names
        assert "insight_extract" in tool_names
        assert "insight_list" in tool_names

    def test_handlers_map(self):
        """Test that all tools have handlers."""
        for tool in VKB_TOOLS:
            assert tool.name in VKB_HANDLERS
            assert callable(VKB_HANDLERS[tool.name])

    def test_tool_descriptions_contain_behavioral_hints(self):
        """Test that key tools have behavioral conditioning."""
        session_start_tool = next(t for t in VKB_TOOLS if t.name == "session_start")
        assert "START" in session_start_tool.description

        insight_extract_tool = next(t for t in VKB_TOOLS if t.name == "insight_extract")
        assert "PROACTIVELY" in insight_extract_tool.description


# =============================================================================
# SESSION TESTS
# =============================================================================


class TestSessionStart:
    """Tests for session_start function."""

    def test_session_start_basic(self, mock_get_cursor, sample_session_row):
        """Test basic session start."""
        session_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_session_row(id=session_id)

        result = session_start(platform="claude-code")

        assert result["success"] is True
        assert "session" in result
        mock_get_cursor.execute.assert_called()

    def test_session_start_with_project_context(self, mock_get_cursor, sample_session_row):
        """Test session start with project context."""
        mock_get_cursor.fetchone.return_value = sample_session_row(project_context="valence-repo")

        result = session_start(platform="claude-code", project_context="valence-repo")

        assert result["success"] is True

    def test_session_start_with_external_room(self, mock_get_cursor, sample_session_row):
        """Test session start with external room ID."""
        mock_get_cursor.fetchone.return_value = sample_session_row(external_room_id="!room:slack.com")

        result = session_start(platform="slack", external_room_id="!room:slack.com")

        assert result["success"] is True

    def test_session_start_with_metadata(self, mock_get_cursor, sample_session_row):
        """Test session start with metadata."""
        mock_get_cursor.fetchone.return_value = sample_session_row()

        result = session_start(platform="api", metadata={"client_version": "1.0", "model": "claude-3"})

        assert result["success"] is True


class TestSessionEnd:
    """Tests for session_end function."""

    def test_session_end_basic(self, mock_get_cursor, sample_session_row):
        """Test basic session end."""
        session_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_session_row(id=session_id, status="completed")

        result = session_end(session_id=str(session_id))

        assert result["success"] is True

    def test_session_end_with_summary(self, mock_get_cursor, sample_session_row):
        """Test session end with summary."""
        session_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_session_row(id=session_id, summary="Discussed valence architecture")

        result = session_end(
            session_id=str(session_id),
            summary="Discussed valence architecture",
            themes=["architecture", "design"],
        )

        assert result["success"] is True

    def test_session_end_not_found(self, mock_get_cursor):
        """Test session end when not found."""
        mock_get_cursor.fetchone.return_value = None

        result = session_end(session_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_session_end_abandoned(self, mock_get_cursor, sample_session_row):
        """Test session end with abandoned status."""
        session_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_session_row(id=session_id, status="abandoned")

        result = session_end(session_id=str(session_id), status="abandoned")

        assert result["success"] is True


class TestSessionGet:
    """Tests for session_get function."""

    def test_session_get_basic(self, mock_get_cursor, sample_session_row):
        """Test basic session retrieval."""
        session_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_session_row(id=session_id)

        result = session_get(session_id=str(session_id))

        assert result["success"] is True
        assert "session" in result

    def test_session_get_not_found(self, mock_get_cursor):
        """Test session not found."""
        mock_get_cursor.fetchone.return_value = None

        result = session_get(session_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_session_get_with_exchanges(self, mock_get_cursor, sample_session_row, sample_exchange_row):
        """Test getting session with exchanges."""
        session_id = uuid4()

        mock_get_cursor.fetchone.return_value = sample_session_row(id=session_id)
        mock_get_cursor.fetchall.return_value = [
            sample_exchange_row(session_id=session_id, sequence=1),
            sample_exchange_row(session_id=session_id, sequence=2),
        ]

        result = session_get(session_id=str(session_id), include_exchanges=True, exchange_limit=10)

        assert result["success"] is True
        assert "exchanges" in result


class TestSessionList:
    """Tests for session_list function."""

    def test_session_list_basic(self, mock_get_cursor, sample_session_row):
        """Test basic session listing."""
        mock_get_cursor.fetchall.return_value = [
            sample_session_row(),
            sample_session_row(),
        ]

        result = session_list()

        assert result["success"] is True
        assert len(result["sessions"]) == 2

    def test_session_list_with_platform_filter(self, mock_get_cursor, sample_session_row):
        """Test session list with platform filter."""
        mock_get_cursor.fetchall.return_value = [sample_session_row(platform="slack")]

        result = session_list(platform="slack")

        assert result["success"] is True
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "platform = %s" in sql_call

    def test_session_list_with_status_filter(self, mock_get_cursor, sample_session_row):
        """Test session list with status filter."""
        mock_get_cursor.fetchall.return_value = []

        result = session_list(status="completed")

        assert result["success"] is True


class TestSessionFindByRoom:
    """Tests for session_find_by_room function."""

    def test_session_find_by_room_found(self, mock_get_cursor, sample_session_row):
        """Test finding session by room ID."""
        session_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_session_row(id=session_id, external_room_id="!room:example.com")

        result = session_find_by_room(external_room_id="!room:example.com")

        assert result["success"] is True
        assert result["found"] is True
        assert result["session"] is not None

    def test_session_find_by_room_not_found(self, mock_get_cursor):
        """Test when no session found for room."""
        mock_get_cursor.fetchone.return_value = None

        result = session_find_by_room(external_room_id="!nonexistent:example.com")

        assert result["success"] is True
        assert result["found"] is False
        assert result["session"] is None


# =============================================================================
# EXCHANGE TESTS
# =============================================================================


class TestExchangeAdd:
    """Tests for exchange_add function."""

    def test_exchange_add_basic(self, mock_get_cursor, sample_exchange_row):
        """Test basic exchange add."""
        session_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            {"next_seq": 1},  # sequence query
            sample_exchange_row(session_id=session_id, sequence=1),  # insert result
        ]

        result = exchange_add(session_id=str(session_id), role="user", content="Hello, how are you?")

        assert result["success"] is True
        assert "exchange" in result

    def test_exchange_add_with_tokens(self, mock_get_cursor, sample_exchange_row):
        """Test exchange add with token count."""
        session_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            {"next_seq": 1},
            sample_exchange_row(tokens_approx=150),
        ]

        result = exchange_add(
            session_id=str(session_id),
            role="assistant",
            content="I'm doing well!",
            tokens_approx=150,
        )

        assert result["success"] is True

    def test_exchange_add_with_tool_uses(self, mock_get_cursor, sample_exchange_row):
        """Test exchange add with tool uses."""
        session_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            {"next_seq": 1},
            sample_exchange_row(),
        ]

        result = exchange_add(
            session_id=str(session_id),
            role="assistant",
            content="Let me check that.",
            tool_uses=[{"name": "belief_query", "args": {"query": "test"}}],
        )

        assert result["success"] is True


class TestExchangeList:
    """Tests for exchange_list function."""

    def test_exchange_list_basic(self, mock_get_cursor, sample_exchange_row):
        """Test basic exchange listing."""
        session_id = uuid4()

        mock_get_cursor.fetchall.return_value = [
            sample_exchange_row(sequence=1),
            sample_exchange_row(sequence=2),
        ]

        result = exchange_list(session_id=str(session_id))

        assert result["success"] is True
        assert len(result["exchanges"]) == 2

    def test_exchange_list_with_limit(self, mock_get_cursor, sample_exchange_row):
        """Test exchange list with limit."""
        session_id = uuid4()

        mock_get_cursor.fetchall.return_value = [sample_exchange_row()]

        result = exchange_list(session_id=str(session_id), limit=5, offset=10)

        assert result["success"] is True


# =============================================================================
# PATTERN TESTS
# =============================================================================


class TestPatternRecord:
    """Tests for pattern_record function."""

    def test_pattern_record_basic(self, mock_get_cursor, sample_pattern_row):
        """Test basic pattern recording."""
        pattern_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_pattern_row(id=pattern_id)

        result = pattern_record(type="topic_recurrence", description="User frequently discusses Python")

        assert result["success"] is True
        assert "pattern" in result

    def test_pattern_record_with_evidence(self, mock_get_cursor, sample_pattern_row):
        """Test pattern recording with evidence."""
        session1 = uuid4()
        session2 = uuid4()

        mock_get_cursor.fetchone.return_value = sample_pattern_row(evidence=[session1, session2])

        result = pattern_record(
            type="preference",
            description="User prefers concise responses",
            evidence=[str(session1), str(session2)],
            confidence=0.7,
        )

        assert result["success"] is True

    def test_pattern_record_evidence_passed_as_strings(self, mock_get_cursor, sample_pattern_row):
        """Regression: evidence must be passed as strings, not UUID objects.

        psycopg2 can't adapt list[UUID] to PostgreSQL UUID[]. The SQL must use
        ::uuid[] cast and the params must contain plain strings.
        """
        session1 = uuid4()
        mock_get_cursor.fetchone.return_value = sample_pattern_row(evidence=[session1])

        pattern_record(
            type="preference",
            description="Test pattern",
            evidence=[str(session1)],
        )

        call_args = mock_get_cursor.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]

        # SQL must cast to uuid[]
        assert "::uuid[]" in sql
        # Evidence param (index 2) must be a list of strings, not UUID objects
        evidence_param = params[2]
        assert isinstance(evidence_param, list)
        for item in evidence_param:
            assert isinstance(item, str), f"Expected str, got {type(item)}"

    def test_pattern_record_empty_evidence(self, mock_get_cursor, sample_pattern_row):
        """Regression: empty evidence list should not cause UUID parse errors."""
        mock_get_cursor.fetchone.return_value = sample_pattern_row()

        result = pattern_record(
            type="preference",
            description="Test pattern",
            evidence=[],
        )

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args
        params = call_args[0][1]
        assert params[2] == []


class TestPatternReinforce:
    """Tests for pattern_reinforce function."""

    def test_pattern_reinforce_basic(self, mock_get_cursor, sample_pattern_row):
        """Test basic pattern reinforcement."""
        pattern_id = uuid4()

        # First call returns current pattern, second returns updated
        mock_get_cursor.fetchone.side_effect = [
            sample_pattern_row(id=pattern_id, occurrence_count=3, confidence=0.6),
            sample_pattern_row(id=pattern_id, occurrence_count=4, confidence=0.64),
        ]

        result = pattern_reinforce(pattern_id=str(pattern_id))

        assert result["success"] is True

    def test_pattern_reinforce_with_session(self, mock_get_cursor, sample_pattern_row):
        """Test pattern reinforcement with session evidence."""
        pattern_id = uuid4()
        session_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            sample_pattern_row(id=pattern_id),
            sample_pattern_row(id=pattern_id),
        ]

        result = pattern_reinforce(pattern_id=str(pattern_id), session_id=str(session_id))

        assert result["success"] is True

    def test_pattern_reinforce_evidence_passed_as_strings(self, mock_get_cursor, sample_pattern_row):
        """Regression: evidence must be passed as strings in UPDATE, not UUID objects."""
        pattern_id = uuid4()
        session_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            sample_pattern_row(id=pattern_id, evidence=[uuid4()]),
            sample_pattern_row(id=pattern_id),
        ]

        pattern_reinforce(pattern_id=str(pattern_id), session_id=str(session_id))

        # The second execute call is the UPDATE
        update_call = mock_get_cursor.execute.call_args_list[1]
        sql = update_call[0][0]
        params = update_call[0][1]

        assert "::uuid[]" in sql
        evidence_param = params[0]
        assert isinstance(evidence_param, list)
        for item in evidence_param:
            assert isinstance(item, str), f"Expected str, got {type(item)}"

    def test_pattern_reinforce_not_found(self, mock_get_cursor):
        """Test reinforcing non-existent pattern."""
        mock_get_cursor.fetchone.return_value = None

        result = pattern_reinforce(pattern_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]


class TestPatternList:
    """Tests for pattern_list function."""

    def test_pattern_list_basic(self, mock_get_cursor, sample_pattern_row):
        """Test basic pattern listing."""
        mock_get_cursor.fetchall.return_value = [
            sample_pattern_row(),
            sample_pattern_row(),
        ]

        result = pattern_list()

        assert result["success"] is True
        assert len(result["patterns"]) == 2

    def test_pattern_list_with_type_filter(self, mock_get_cursor, sample_pattern_row):
        """Test pattern list with type filter."""
        mock_get_cursor.fetchall.return_value = [sample_pattern_row(type="preference")]

        result = pattern_list(type="preference")

        assert result["success"] is True

    def test_pattern_list_with_status_filter(self, mock_get_cursor, sample_pattern_row):
        """Test pattern list with status filter."""
        mock_get_cursor.fetchall.return_value = []

        result = pattern_list(status="established")

        assert result["success"] is True

    def test_pattern_list_with_min_confidence(self, mock_get_cursor, sample_pattern_row):
        """Test pattern list with minimum confidence."""
        mock_get_cursor.fetchall.return_value = []

        result = pattern_list(min_confidence=0.7)

        assert result["success"] is True


class TestPatternSearch:
    """Tests for pattern_search function."""

    def test_pattern_search_basic(self, mock_get_cursor, sample_pattern_row):
        """Test basic pattern search."""
        mock_get_cursor.fetchall.return_value = [
            sample_pattern_row(description="User likes Python"),
        ]

        result = pattern_search(query="Python")

        assert result["success"] is True
        assert len(result["patterns"]) == 1


# =============================================================================
# INSIGHT TESTS
# =============================================================================


class TestInsightExtract:
    """Tests for insight_extract function."""

    def test_insight_extract_basic(self, mock_get_cursor):
        """Test basic insight extraction."""
        session_id = uuid4()
        source_id = uuid4()
        belief_id = uuid4()
        insight_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            {"id": source_id},  # source lookup
            {
                "id": belief_id,
                "content": "Test",
                "confidence": {},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": datetime.now(),
                "modified_at": datetime.now(),
                "source_id": source_id,
                "extraction_method": "conversation_extraction",
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
                "opt_out_federation": False,
            },  # belief insert
            {"id": insight_id},  # insight link
        ]

        result = insight_extract(session_id=str(session_id), content="Python is great for data science")

        assert result["success"] is True
        assert "belief_id" in result

    def test_insight_extract_with_domain(self, mock_get_cursor):
        """Test insight extraction with domain path."""
        session_id = uuid4()
        belief_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            {"id": uuid4()},  # source
            {
                "id": belief_id,
                "content": "Test",
                "confidence": {},
                "domain_path": ["tech", "python"],
                "valid_from": None,
                "valid_until": None,
                "created_at": datetime.now(),
                "modified_at": datetime.now(),
                "source_id": uuid4(),
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
                "opt_out_federation": False,
            },
            {"id": uuid4()},  # insight
        ]

        result = insight_extract(
            session_id=str(session_id),
            content="Insight about Python",
            domain_path=["tech", "python"],
        )

        assert result["success"] is True

    def test_insight_extract_with_entities(self, mock_get_cursor):
        """Test insight extraction with entities."""
        session_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            {"id": uuid4()},  # source
            {
                "id": uuid4(),
                "content": "Test",
                "confidence": {},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": datetime.now(),
                "modified_at": datetime.now(),
                "source_id": uuid4(),
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
                "opt_out_federation": False,
            },  # belief
            {"id": uuid4()},  # entity 1
            {"id": uuid4()},  # entity 2
            {"id": uuid4()},  # insight
        ]

        result = insight_extract(
            session_id=str(session_id),
            content="Developers use Python",
            entities=[
                {"name": "developers", "type": "concept"},
                {"name": "Python", "type": "tool"},
            ],
        )

        assert result["success"] is True


class TestInsightList:
    """Tests for insight_list function."""

    def test_insight_list_basic(self, mock_get_cursor):
        """Test basic insight listing."""
        session_id = uuid4()

        mock_get_cursor.fetchall.return_value = [
            {
                "id": uuid4(),
                "session_id": session_id,
                "belief_id": uuid4(),
                "extraction_method": "manual",
                "extracted_at": datetime.now(),
                "content": "Test insight",
                "confidence": {"overall": 0.8},
                "domain_path": ["test"],
                "belief_created_at": datetime.now(),
            }
        ]

        result = insight_list(session_id=str(session_id))

        assert result["success"] is True
        assert len(result["insights"]) == 1


# =============================================================================
# HANDLER ROUTING TESTS
# =============================================================================


class TestHandleVKBTool:
    """Tests for handle_vkb_tool routing function."""

    def test_routes_to_correct_handler(self, mock_get_cursor, sample_session_row):
        """Test that tool calls are routed correctly."""
        mock_get_cursor.fetchone.return_value = sample_session_row()

        result = handle_vkb_tool("session_start", {"platform": "claude-code"})

        assert result["success"] is True

    def test_unknown_tool(self):
        """Test handling unknown tool name."""
        result = handle_vkb_tool("nonexistent_tool", {})

        assert result["success"] is False
        assert "Unknown" in result["error"]

    def test_all_handlers_are_callable(self):
        """Test that all registered handlers work."""
        for name in VKB_HANDLERS:
            assert callable(VKB_HANDLERS[name])
