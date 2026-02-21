"""Tests for substrate tool handlers.

Tests cover:
- belief_query - search beliefs
- belief_create - create beliefs with entities
- belief_supersede - update beliefs with history
- belief_get - get belief with history/tensions
- entity_get / entity_search - entity operations
- tension_list / tension_resolve - tension management
- belief_corroboration - corroboration details
- handle_substrate_tool - routing
"""

from __future__ import annotations

import pytest

pytest.skip(
    "Deprecated: belief system replaced by v2 articles/sources (WU-11)",
    allow_module_level=True,
)

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from valence.substrate.tools import (
    SUBSTRATE_HANDLERS,
    SUBSTRATE_TOOLS,
    belief_create,
    belief_get,
    belief_query,
    belief_supersede,
    entity_get,
    entity_search,
    handle_substrate_tool,
    tension_list,
    tension_resolve,
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

    with patch("valence.substrate.tools._common.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def sample_belief_row():
    """Create a sample belief row."""

    def _factory(id: UUID | None = None, content: str = "Test belief content", **kwargs):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "content": content,
            "confidence": json.dumps(kwargs.get("confidence", {"overall": 0.7})),
            "domain_path": kwargs.get("domain_path", ["test", "domain"]),
            "valid_from": kwargs.get("valid_from"),
            "valid_until": kwargs.get("valid_until"),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
            "source_id": kwargs.get("source_id"),
            "extraction_method": kwargs.get("extraction_method"),
            "supersedes_id": kwargs.get("supersedes_id"),
            "superseded_by_id": kwargs.get("superseded_by_id"),
            "status": kwargs.get("status", "active"),
            "relevance": kwargs.get("relevance", 0.95),
            "opt_out_federation": kwargs.get("opt_out_federation", False),
        }

    return _factory


@pytest.fixture
def sample_entity_row():
    """Create a sample entity row."""

    def _factory(
        id: UUID | None = None,
        name: str = "Test Entity",
        type: str = "concept",
        **kwargs,
    ):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "name": name,
            "type": type,
            "description": kwargs.get("description"),
            "aliases": kwargs.get("aliases", []),
            "canonical_id": kwargs.get("canonical_id"),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
            "role": kwargs.get("role", "subject"),
        }

    return _factory


@pytest.fixture
def sample_tension_row():
    """Create a sample tension row."""

    def _factory(
        id: UUID | None = None,
        belief_a_id: UUID | None = None,
        belief_b_id: UUID | None = None,
        **kwargs,
    ):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "belief_a_id": belief_a_id or uuid4(),
            "belief_b_id": belief_b_id or uuid4(),
            "type": kwargs.get("type", "contradiction"),
            "description": kwargs.get("description"),
            "severity": kwargs.get("severity", "medium"),
            "status": kwargs.get("status", "detected"),
            "resolution": kwargs.get("resolution"),
            "resolved_at": kwargs.get("resolved_at"),
            "detected_at": kwargs.get("detected_at", now),
        }

    return _factory


# =============================================================================
# TOOL DEFINITIONS TESTS
# =============================================================================


class TestToolDefinitions:
    """Tests for tool definitions."""

    def test_substrate_tools_list(self):
        """Test that SUBSTRATE_TOOLS contains expected tools."""
        tool_names = [t.name for t in SUBSTRATE_TOOLS]

        assert "belief_query" in tool_names
        assert "belief_create" in tool_names
        assert "belief_supersede" in tool_names
        assert "belief_get" in tool_names
        assert "entity_get" in tool_names
        assert "entity_search" in tool_names
        assert "tension_list" in tool_names
        assert "tension_resolve" in tool_names

    def test_handlers_map(self):
        """Test that all tools have handlers."""
        for tool in SUBSTRATE_TOOLS:
            assert tool.name in SUBSTRATE_HANDLERS
            assert callable(SUBSTRATE_HANDLERS[tool.name])

    def test_tool_descriptions_contain_behavioral_hints(self):
        """Test that tool descriptions have behavioral conditioning."""
        belief_query_tool = next(t for t in SUBSTRATE_TOOLS if t.name == "belief_query")
        assert "CRITICAL" in belief_query_tool.description
        assert "MUST call" in belief_query_tool.description

        belief_create_tool = next(t for t in SUBSTRATE_TOOLS if t.name == "belief_create")
        assert "PROACTIVELY" in belief_create_tool.description


# =============================================================================
# BELIEF QUERY TESTS
# =============================================================================


class TestBeliefQuery:
    """Tests for belief_query function."""

    def test_belief_query_basic(self, mock_get_cursor, sample_belief_row):
        """Test basic belief query."""
        belief_id = uuid4()
        mock_get_cursor.fetchall.return_value = [sample_belief_row(id=belief_id, content="Python is great")]

        result = belief_query(query="Python")

        assert result["success"] is True
        assert len(result["beliefs"]) == 1
        assert result["beliefs"][0]["content"] == "Python is great"
        assert "relevance_score" in result["beliefs"][0]

    def test_belief_query_with_domain_filter(self, mock_get_cursor, sample_belief_row):
        """Test belief query with domain filter."""
        mock_get_cursor.fetchall.return_value = [sample_belief_row(domain_path=["tech", "python"])]

        result = belief_query(query="Python", domain_filter=["tech", "python"])

        assert result["success"] is True
        # Verify domain filter was passed (check SQL call)
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "domain_path &&" in sql_call

    def test_belief_query_with_entity_filter(self, mock_get_cursor, sample_belief_row):
        """Test belief query with entity filter."""
        entity_id = uuid4()
        mock_get_cursor.fetchall.return_value = []

        result = belief_query(query="test", entity_id=str(entity_id))

        assert result["success"] is True
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "belief_entities" in sql_call

    def test_belief_query_include_superseded(self, mock_get_cursor, sample_belief_row):
        """Test including superseded beliefs."""
        mock_get_cursor.fetchall.return_value = [sample_belief_row(status="superseded")]

        result = belief_query(query="test", include_superseded=True)

        assert result["success"] is True
        # SQL should NOT have status filter
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "status = 'active'" not in sql_call or "include_superseded" in str(calls)

    def test_belief_query_respects_limit(self, mock_get_cursor, sample_belief_row):
        """Test limit parameter."""
        mock_get_cursor.fetchall.return_value = []

        belief_query(query="test", limit=5)

        calls = mock_get_cursor.execute.call_args_list
        params = calls[0][0][1]
        assert 5 in params

    def test_belief_query_filters_revoked_by_default(self, mock_get_cursor, sample_belief_row):
        """Test that revoked beliefs are filtered out by default."""
        mock_get_cursor.fetchall.return_value = []

        result = belief_query(query="test")

        assert result["success"] is True
        assert result["include_revoked"] is False
        # Verify SQL contains revocation filter
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "consent_chains" in sql_call
        assert "revoked = true" in sql_call

    def test_belief_query_include_revoked_explicit(self, mock_get_cursor, sample_belief_row):
        """Test that include_revoked=True bypasses revocation filter."""
        mock_get_cursor.fetchall.return_value = [sample_belief_row(content="Revoked belief content")]

        result = belief_query(query="test", include_revoked=True)

        assert result["success"] is True
        assert result["include_revoked"] is True
        # Verify SQL does NOT contain revocation filter
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "consent_chains" not in sql_call

    def test_belief_query_include_revoked_audit_logging(self, mock_get_cursor, sample_belief_row, caplog):
        """Test that accessing revoked content is audit logged."""
        import logging

        caplog.set_level(logging.INFO)
        mock_get_cursor.fetchall.return_value = []

        belief_query(
            query="test query for audit",
            include_revoked=True,
            user_did="did:key:test123",
        )

        # Check that audit log was written
        assert any(
            "Query includes revoked content" in record.message and "did:key:test123" in record.message and "test query for audit" in record.message
            for record in caplog.records
        )

    def test_belief_query_no_audit_log_when_not_including_revoked(self, mock_get_cursor, sample_belief_row, caplog):
        """Test that no audit log when include_revoked=False."""
        import logging

        caplog.set_level(logging.INFO)
        mock_get_cursor.fetchall.return_value = []

        belief_query(query="test", include_revoked=False)

        # Check that no audit log was written
        assert not any("Query includes revoked content" in record.message for record in caplog.records)

    def test_belief_query_mixed_results_revoked_filtered(self, mock_get_cursor, sample_belief_row):
        """Test that only non-revoked beliefs are returned by default.

        This tests the scenario where some beliefs have revoked consent chains
        and some don't - only the non-revoked ones should be returned.
        """
        # The DB should return only non-revoked due to SQL filter
        non_revoked_belief = sample_belief_row(content="Non-revoked belief")
        mock_get_cursor.fetchall.return_value = [non_revoked_belief]

        result = belief_query(query="belief")

        assert result["success"] is True
        assert len(result["beliefs"]) == 1
        assert result["beliefs"][0]["content"] == "Non-revoked belief"


# =============================================================================
# BELIEF CREATE TESTS
# =============================================================================


class TestBeliefCreate:
    """Tests for belief_create function."""

    @pytest.fixture(autouse=True)
    def no_embeddings(self):
        """Disable embedding-based fuzzy dedup for unit tests."""
        with patch("our_embeddings.service.generate_embedding", side_effect=RuntimeError("test")):
            yield

    def test_belief_create_basic(self, mock_get_cursor, sample_belief_row):
        """Test basic belief creation."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            sample_belief_row(id=belief_id, content="Test content"),  # INSERT RETURNING
        ]

        result = belief_create(content="Test content")

        assert result["success"] is True
        assert "belief" in result
        mock_get_cursor.execute.assert_called()

    def test_belief_create_with_confidence(self, mock_get_cursor, sample_belief_row):
        """Test belief creation with confidence."""
        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            sample_belief_row(),  # INSERT RETURNING
        ]

        result = belief_create(content="Test", confidence={"overall": 0.9, "source_reliability": 0.8})

        assert result["success"] is True

    def test_belief_create_with_domain_path(self, mock_get_cursor, sample_belief_row):
        """Test belief creation with domain path."""
        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            sample_belief_row(),  # INSERT RETURNING
        ]

        result = belief_create(content="Test", domain_path=["tech", "python", "testing"])

        assert result["success"] is True

    def test_belief_create_with_source(self, mock_get_cursor, sample_belief_row):
        """Test belief creation with source."""
        source_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            {"id": source_id},  # source insert
            sample_belief_row(),  # belief insert
        ]

        result = belief_create(content="Test", source_type="conversation", source_ref="session-123")

        assert result["success"] is True

    def test_belief_create_with_opt_out_federation(self, mock_get_cursor, sample_belief_row):
        """Test belief creation with federation opt-out (Issue #26)."""
        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            sample_belief_row(opt_out_federation=True),  # INSERT RETURNING
        ]

        result = belief_create(content="Private belief", opt_out_federation=True)

        assert result["success"] is True

    def test_belief_create_with_entities(self, mock_get_cursor, sample_belief_row):
        """Test belief creation with entity links."""
        belief_id = uuid4()
        entity1_id = uuid4()
        entity2_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            sample_belief_row(id=belief_id),  # belief insert
            {"id": entity1_id},  # entity 1 upsert
            {"id": entity2_id},  # entity 2 upsert
        ]

        result = belief_create(
            content="Python is used by developers",
            entities=[
                {"name": "Python", "type": "tool", "role": "subject"},
                {"name": "developers", "type": "concept", "role": "object"},
            ],
        )

        assert result["success"] is True


# =============================================================================
# BELIEF SUPERSEDE TESTS
# =============================================================================


class TestBeliefSupersede:
    """Tests for belief_supersede function."""

    def test_belief_supersede_success(self, mock_get_cursor, sample_belief_row):
        """Test successful belief supersession."""
        old_id = uuid4()
        new_id = uuid4()

        old_belief = sample_belief_row(id=old_id, content="Old content")
        new_belief = sample_belief_row(id=new_id, content="New content")

        mock_get_cursor.fetchone.side_effect = [
            old_belief,  # Get old belief
            new_belief,  # Create new belief
        ]

        result = belief_supersede(
            old_belief_id=str(old_id),
            new_content="New content",
            reason="Updated information",
        )

        assert result["success"] is True
        assert result["old_belief_id"] == str(old_id)
        assert "new_belief" in result
        assert result["reason"] == "Updated information"

    def test_belief_supersede_not_found(self, mock_get_cursor):
        """Test supersession when old belief not found."""
        mock_get_cursor.fetchone.return_value = None

        result = belief_supersede(old_belief_id=str(uuid4()), new_content="New content", reason="Test")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_belief_supersede_preserves_entities(self, mock_get_cursor, sample_belief_row):
        """Test that supersession copies entity links."""
        old_id = uuid4()
        new_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            sample_belief_row(id=old_id),
            sample_belief_row(id=new_id),
        ]

        result = belief_supersede(old_belief_id=str(old_id), new_content="Updated", reason="Fix")

        assert result["success"] is True
        # Verify entity copy SQL was executed
        calls = [str(c) for c in mock_get_cursor.execute.call_args_list]
        assert any("belief_entities" in c for c in calls)


# =============================================================================
# BELIEF GET TESTS
# =============================================================================


class TestBeliefGet:
    """Tests for belief_get function."""

    def test_belief_get_basic(self, mock_get_cursor, sample_belief_row):
        """Test basic belief retrieval."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_belief_row(id=belief_id)

        result = belief_get(belief_id=str(belief_id))

        assert result["success"] is True
        assert "belief" in result

    def test_belief_get_not_found(self, mock_get_cursor):
        """Test belief not found."""
        mock_get_cursor.fetchone.return_value = None

        result = belief_get(belief_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_belief_get_with_history(self, mock_get_cursor, sample_belief_row):
        """Test getting belief with history chain."""
        belief_id = uuid4()
        older_id = uuid4()

        mock_get_cursor.fetchone.side_effect = [
            sample_belief_row(id=belief_id, supersedes_id=older_id),  # Current belief
            {
                "id": belief_id,
                "supersedes_id": older_id,
                "created_at": datetime.now(),
                "extraction_method": "update",
            },
            {
                "id": older_id,
                "supersedes_id": None,
                "created_at": datetime.now(),
                "extraction_method": None,
            },
        ]
        mock_get_cursor.fetchall.return_value = []

        result = belief_get(belief_id=str(belief_id), include_history=True)

        assert result["success"] is True
        assert "history" in result

    def test_belief_get_with_tensions(self, mock_get_cursor, sample_belief_row, sample_tension_row):
        """Test getting belief with tensions."""
        belief_id = uuid4()

        mock_get_cursor.fetchone.return_value = sample_belief_row(id=belief_id)
        mock_get_cursor.fetchall.side_effect = [
            [],  # entities
            [sample_tension_row(belief_a_id=belief_id)],  # tensions
        ]

        result = belief_get(belief_id=str(belief_id), include_tensions=True)

        assert result["success"] is True
        # Should include tensions if found


# =============================================================================
# ENTITY TESTS
# =============================================================================


class TestEntityGet:
    """Tests for entity_get function."""

    def test_entity_get_basic(self, mock_get_cursor, sample_entity_row):
        """Test basic entity retrieval."""
        entity_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_entity_row(id=entity_id)

        result = entity_get(entity_id=str(entity_id))

        assert result["success"] is True
        assert "entity" in result

    def test_entity_get_not_found(self, mock_get_cursor):
        """Test entity not found."""
        mock_get_cursor.fetchone.return_value = None

        result = entity_get(entity_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_entity_get_with_beliefs(self, mock_get_cursor, sample_entity_row, sample_belief_row):
        """Test getting entity with related beliefs."""
        entity_id = uuid4()

        mock_get_cursor.fetchone.return_value = sample_entity_row(id=entity_id)
        mock_get_cursor.fetchall.return_value = [{**sample_belief_row(), "role": "subject"}]

        result = entity_get(entity_id=str(entity_id), include_beliefs=True, belief_limit=5)

        assert result["success"] is True
        assert "beliefs" in result


class TestEntitySearch:
    """Tests for entity_search function."""

    def test_entity_search_basic(self, mock_get_cursor, sample_entity_row):
        """Test basic entity search."""
        mock_get_cursor.fetchall.return_value = [
            sample_entity_row(name="Python"),
            sample_entity_row(name="PyTorch"),
        ]

        result = entity_search(query="Py")

        assert result["success"] is True
        assert len(result["entities"]) == 2

    def test_entity_search_with_type_filter(self, mock_get_cursor, sample_entity_row):
        """Test entity search with type filter."""
        mock_get_cursor.fetchall.return_value = [sample_entity_row(name="Alice", type="person")]

        result = entity_search(query="Alice", type="person")

        assert result["success"] is True
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "type = %s" in sql_call


# =============================================================================
# TENSION TESTS
# =============================================================================


class TestTensionList:
    """Tests for tension_list function."""

    def test_tension_list_basic(self, mock_get_cursor, sample_tension_row, sample_belief_row):
        """Test basic tension listing."""
        tension = sample_tension_row()
        mock_get_cursor.fetchall.side_effect = [
            [tension],  # tensions
            [  # beliefs for context
                {"id": tension["belief_a_id"], "content": "Belief A"},
                {"id": tension["belief_b_id"], "content": "Belief B"},
            ],
        ]

        result = tension_list()

        assert result["success"] is True
        assert "tensions" in result

    def test_tension_list_with_status_filter(self, mock_get_cursor):
        """Test tension list with status filter."""
        mock_get_cursor.fetchall.return_value = []

        result = tension_list(status="investigating")

        assert result["success"] is True
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "status = %s" in sql_call

    def test_tension_list_with_severity_filter(self, mock_get_cursor):
        """Test tension list with severity filter."""
        mock_get_cursor.fetchall.return_value = []

        result = tension_list(severity="high")

        assert result["success"] is True
        calls = mock_get_cursor.execute.call_args_list
        sql_call = calls[0][0][0]
        assert "CASE severity" in sql_call


class TestTensionResolve:
    """Tests for tension_resolve function."""

    def test_tension_resolve_keep_both(self, mock_get_cursor, sample_tension_row):
        """Test resolving tension by keeping both beliefs."""
        tension_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_tension_row(id=tension_id)

        result = tension_resolve(
            tension_id=str(tension_id),
            resolution="Both are valid in different contexts",
            action="keep_both",
        )

        assert result["success"] is True
        assert result["action"] == "keep_both"

    def test_tension_resolve_supersede_a(self, mock_get_cursor, sample_tension_row, sample_belief_row):
        """Test resolving tension by superseding belief A."""
        tension_id = uuid4()
        belief_a_id = uuid4()
        belief_b_id = uuid4()

        tension = sample_tension_row(id=tension_id, belief_a_id=belief_a_id, belief_b_id=belief_b_id)

        mock_get_cursor.fetchone.side_effect = [
            tension,  # Get tension
            {"content": "Belief B content"},  # Get belief B content
            sample_belief_row(id=belief_a_id),  # For supersession
            sample_belief_row(),  # New belief
        ]

        result = tension_resolve(
            tension_id=str(tension_id),
            resolution="B is more accurate",
            action="supersede_a",
        )

        assert result["success"] is True
        assert result["action"] == "supersede_a"

    def test_tension_resolve_not_found(self, mock_get_cursor):
        """Test resolving non-existent tension."""
        mock_get_cursor.fetchone.return_value = None

        result = tension_resolve(tension_id=str(uuid4()), resolution="Test", action="keep_both")

        assert result["success"] is False
        assert "not found" in result["error"]


