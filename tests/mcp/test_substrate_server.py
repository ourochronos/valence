"""Tests for valence.substrate.mcp_server module.

NOTE (WU-11): Tests referencing the old belief system (belief_query, belief_create,
belief_supersede, belief_get, tension_list, tension_resolve) are skipped because
those tools have been replaced by the v2 article/source system.
See tests/integration/test_v2_integration.py for v2 equivalents.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Skip all tests in this module that reference deleted belief tools
pytestmark = pytest.mark.skip(reason="Deprecated: belief tools replaced by v2 articles/sources (WU-11)")

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

    with patch("valence.substrate.tools._common.get_cursor", fake_get_cursor):
        yield mock_cursor


# ============================================================================
# belief_query Tests
# ============================================================================


class TestBeliefQuery:
    """Tests for belief_query function."""

    def test_basic_search(self, mock_get_cursor):
        """Should search beliefs by query."""
        from valence.substrate.tools import belief_query

        belief_id = uuid4()
        now = datetime.now()
        mock_get_cursor.fetchall.return_value = [
            {
                "id": belief_id,
                "content": "Test belief content",
                "confidence": {"overall": 0.8},
                "domain_path": ["test"],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
                "relevance": 0.95,
            }
        ]

        result = belief_query("test query")

        assert result["success"] is True
        assert len(result["beliefs"]) == 1
        assert result["beliefs"][0]["content"] == "Test belief content"
        assert result["beliefs"][0]["relevance_score"] == 0.95

    def test_with_domain_filter(self, mock_get_cursor):
        """Should filter by domain."""
        from valence.substrate.tools import belief_query

        mock_get_cursor.fetchall.return_value = []

        result = belief_query("test", domain_filter=["tech", "python"])

        assert result["success"] is True
        # Check that SQL includes domain filter
        call_args = mock_get_cursor.execute.call_args[0]
        assert "domain_path" in call_args[0]

    def test_with_entity_filter(self, mock_get_cursor):
        """Should filter by entity."""
        from valence.substrate.tools import belief_query

        mock_get_cursor.fetchall.return_value = []

        result = belief_query("test", entity_id=str(uuid4()))

        assert result["success"] is True
        # Check that SQL includes entity filter
        call_args = mock_get_cursor.execute.call_args[0]
        assert "belief_entities" in call_args[0]

    def test_include_superseded(self, mock_get_cursor):
        """Should include superseded beliefs when requested."""
        from valence.substrate.tools import belief_query

        mock_get_cursor.fetchall.return_value = []

        result = belief_query("test", include_superseded=True)

        assert result["success"] is True
        # Query should NOT filter by status/superseded_by_id
        call_args = mock_get_cursor.execute.call_args[0]
        sql = call_args[0]
        assert "status = 'active'" not in sql

    def test_empty_results(self, mock_get_cursor):
        """Should handle empty results."""
        from valence.substrate.tools import belief_query

        mock_get_cursor.fetchall.return_value = []

        result = belief_query("nonexistent")

        assert result["success"] is True
        assert result["beliefs"] == []
        assert result["total_count"] == 0


# ============================================================================
# belief_create Tests
# ============================================================================


class TestBeliefCreate:
    """Tests for belief_create function."""

    @pytest.fixture(autouse=True)
    def no_embeddings(self):
        """Disable embedding-based fuzzy dedup for unit tests."""
        with patch("valence.lib.our_embeddings.service.generate_embedding", side_effect=RuntimeError("test")):
            yield

    def test_basic_creation(self, mock_get_cursor):
        """Should create a basic belief."""
        from valence.substrate.tools import belief_create

        belief_id = uuid4()
        now = datetime.now()
        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            {
                "id": belief_id,
                "content": "New belief",
                "confidence": {"overall": 0.7},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
        ]

        result = belief_create("New belief")

        assert result["success"] is True
        assert result["belief"]["content"] == "New belief"

    def test_with_confidence(self, mock_get_cursor):
        """Should accept confidence dimensions."""
        from valence.substrate.tools import belief_create

        belief_id = uuid4()
        now = datetime.now()
        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            {
                "id": belief_id,
                "content": "Test",
                "confidence": {"overall": 0.9, "source_reliability": 0.95},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
        ]

        result = belief_create("Test", confidence={"overall": 0.9, "source_reliability": 0.95})

        assert result["success"] is True

    def test_with_source(self, mock_get_cursor):
        """Should create source record."""
        from valence.substrate.tools import belief_create

        source_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            {"id": source_id},  # Source creation
            {
                "id": belief_id,
                "content": "Test",
                "confidence": {"overall": 0.7},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": source_id,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
        ]

        result = belief_create("Test", source_type="document", source_ref="https://example.com")

        assert result["success"] is True

    def test_with_entities(self, mock_get_cursor):
        """Should create and link entities."""
        from valence.substrate.tools import belief_create

        belief_id = uuid4()
        entity_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            None,  # dedup hash check: no match
            # Belief creation
            {
                "id": belief_id,
                "content": "Test about Python",
                "confidence": {"overall": 0.7},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
            # Entity upsert
            {"id": entity_id},
        ]

        result = belief_create(
            "Test about Python",
            entities=[{"name": "Python", "type": "tool", "role": "subject"}],
        )

        assert result["success"] is True


# ============================================================================
# belief_supersede Tests
# ============================================================================


class TestBeliefSupersede:
    """Tests for belief_supersede function."""

    def test_basic_supersession(self, mock_get_cursor):
        """Should supersede an existing belief."""
        from valence.substrate.tools import belief_supersede

        old_id = uuid4()
        new_id = uuid4()
        now = datetime.now()

        # Get old belief, create new belief
        mock_get_cursor.fetchone.side_effect = [
            # Old belief lookup
            {
                "id": old_id,
                "content": "Old content",
                "confidence": {"overall": 0.7},
                "domain_path": ["test"],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
            # New belief creation
            {
                "id": new_id,
                "content": "Updated content",
                "confidence": {"overall": 0.8},
                "domain_path": ["test"],
                "valid_from": now,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": "supersession: Updated information",
                "supersedes_id": old_id,
                "superseded_by_id": None,
                "status": "active",
            },
        ]

        result = belief_supersede(str(old_id), "Updated content", "Updated information")

        assert result["success"] is True
        assert result["old_belief_id"] == str(old_id)
        assert result["reason"] == "Updated information"

    def test_belief_not_found(self, mock_get_cursor):
        """Should return error if belief not found."""
        from valence.substrate.tools import belief_supersede

        mock_get_cursor.fetchone.return_value = None

        result = belief_supersede(str(uuid4()), "New content", "Reason")

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_with_custom_confidence(self, mock_get_cursor):
        """Should accept custom confidence for new belief."""
        from valence.substrate.tools import belief_supersede

        old_id = uuid4()
        new_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            {
                "id": old_id,
                "content": "Old",
                "confidence": {"overall": 0.5},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
            {
                "id": new_id,
                "content": "New",
                "confidence": {"overall": 0.95},
                "domain_path": [],
                "valid_from": now,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": old_id,
                "superseded_by_id": None,
                "status": "active",
            },
        ]

        result = belief_supersede(str(old_id), "New", "Better source", confidence={"overall": 0.95})

        assert result["success"] is True


# ============================================================================
# belief_get Tests
# ============================================================================


class TestBeliefGet:
    """Tests for belief_get function."""

    def test_basic_get(self, mock_get_cursor):
        """Should get a belief by ID."""
        from valence.substrate.tools import belief_get

        belief_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            # Belief lookup
            {
                "id": belief_id,
                "content": "Test belief",
                "confidence": {"overall": 0.8},
                "domain_path": ["test"],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
        ]
        mock_get_cursor.fetchall.return_value = []  # No entities

        result = belief_get(str(belief_id))

        assert result["success"] is True
        assert result["belief"]["content"] == "Test belief"

    def test_not_found(self, mock_get_cursor):
        """Should return error if belief not found."""
        from valence.substrate.tools import belief_get

        mock_get_cursor.fetchone.return_value = None

        result = belief_get(str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_include_history(self, mock_get_cursor):
        """Should include supersession history when requested."""
        from valence.substrate.tools import belief_get

        belief_id = uuid4()
        old_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            # Current belief
            {
                "id": belief_id,
                "content": "Current",
                "confidence": {"overall": 0.8},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": "supersession: updated",
                "supersedes_id": old_id,
                "superseded_by_id": None,
                "status": "active",
            },
            # History: current belief
            {
                "id": belief_id,
                "supersedes_id": old_id,
                "created_at": now,
                "extraction_method": "update",
            },
            # History: old belief
            {
                "id": old_id,
                "supersedes_id": None,
                "created_at": now,
                "extraction_method": "initial",
            },
        ]
        mock_get_cursor.fetchall.return_value = []

        result = belief_get(str(belief_id), include_history=True)

        assert result["success"] is True
        assert "history" in result

    def test_include_tensions(self, mock_get_cursor):
        """Should include tensions when requested."""
        from valence.substrate.tools import belief_get

        belief_id = uuid4()
        tension_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": belief_id,
            "content": "Test",
            "confidence": {"overall": 0.7},
            "domain_path": [],
            "valid_from": None,
            "valid_until": None,
            "created_at": now,
            "modified_at": now,
            "source_id": None,
            "extraction_method": None,
            "supersedes_id": None,
            "superseded_by_id": None,
            "status": "active",
        }

        mock_get_cursor.fetchall.side_effect = [
            [],  # Entities
            [  # Tensions
                {
                    "id": tension_id,
                    "belief_a_id": belief_id,
                    "belief_b_id": uuid4(),
                    "type": "contradiction",
                    "description": None,
                    "severity": "medium",
                    "status": "detected",
                    "resolution": None,
                    "resolved_at": None,
                    "detected_at": now,
                }
            ],
        ]

        result = belief_get(str(belief_id), include_tensions=True)

        assert result["success"] is True
        assert "tensions" in result
        assert len(result["tensions"]) == 1


# ============================================================================
# entity_get Tests
# ============================================================================


class TestEntityGet:
    """Tests for entity_get function."""

    def test_basic_get(self, mock_get_cursor):
        """Should get an entity by ID."""
        from valence.substrate.tools import entity_get

        entity_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": entity_id,
            "name": "Python",
            "type": "tool",
            "description": "A programming language",
            "aliases": ["Python3", "py"],
            "canonical_id": None,
            "created_at": now,
            "modified_at": now,
        }

        result = entity_get(str(entity_id))

        assert result["success"] is True
        assert result["entity"]["name"] == "Python"

    def test_not_found(self, mock_get_cursor):
        """Should return error if entity not found."""
        from valence.substrate.tools import entity_get

        mock_get_cursor.fetchone.return_value = None

        result = entity_get(str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_include_beliefs(self, mock_get_cursor):
        """Should include related beliefs when requested."""
        from valence.substrate.tools import entity_get

        entity_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": entity_id,
            "name": "Python",
            "type": "tool",
            "description": None,
            "aliases": [],
            "canonical_id": None,
            "created_at": now,
            "modified_at": now,
        }

        mock_get_cursor.fetchall.return_value = [
            {
                "id": belief_id,
                "content": "Python is great",
                "confidence": {"overall": 0.9},
                "domain_path": ["tech"],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
                "role": "subject",
            }
        ]

        result = entity_get(str(entity_id), include_beliefs=True)

        assert result["success"] is True
        assert "beliefs" in result
        assert len(result["beliefs"]) == 1


# ============================================================================
# entity_search Tests
# ============================================================================


class TestEntitySearch:
    """Tests for entity_search function."""

    def test_name_match(self, mock_get_cursor):
        """Should search by name."""
        from valence.substrate.tools import entity_search

        entity_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.return_value = [
            {
                "id": entity_id,
                "name": "Python",
                "type": "tool",
                "description": None,
                "aliases": [],
                "canonical_id": None,
                "created_at": now,
                "modified_at": now,
            }
        ]

        result = entity_search("Python")

        assert result["success"] is True
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "Python"

    def test_alias_match(self, mock_get_cursor):
        """Should search by alias."""
        from valence.substrate.tools import entity_search

        mock_get_cursor.fetchall.return_value = []

        result = entity_search("py3")

        assert result["success"] is True
        # Check that aliases are searched
        call_args = mock_get_cursor.execute.call_args[0]
        assert "aliases" in call_args[0]

    def test_type_filter(self, mock_get_cursor):
        """Should filter by entity type."""
        from valence.substrate.tools import entity_search

        mock_get_cursor.fetchall.return_value = []

        result = entity_search("test", type="person")

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "type = %s" in call_args[0]


# ============================================================================
# tension_list Tests
# ============================================================================


class TestTensionList:
    """Tests for tension_list function."""

    def test_basic_list(self, mock_get_cursor):
        """Should list tensions."""
        from valence.substrate.tools import tension_list

        tension_id = uuid4()
        belief_a = uuid4()
        belief_b = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchall.side_effect = [
            # Tensions
            [
                {
                    "id": tension_id,
                    "belief_a_id": belief_a,
                    "belief_b_id": belief_b,
                    "type": "contradiction",
                    "description": None,
                    "severity": "medium",
                    "status": "detected",
                    "resolution": None,
                    "resolved_at": None,
                    "detected_at": now,
                }
            ],
            # Belief content lookup
            [
                {"id": belief_a, "content": "Belief A content"},
                {"id": belief_b, "content": "Belief B content"},
            ],
        ]

        result = tension_list()

        assert result["success"] is True
        assert len(result["tensions"]) == 1

    def test_status_filter(self, mock_get_cursor):
        """Should filter by status."""
        from valence.substrate.tools import tension_list

        mock_get_cursor.fetchall.return_value = []

        result = tension_list(status="detected")

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "status = %s" in call_args[0]

    def test_severity_filter(self, mock_get_cursor):
        """Should filter by minimum severity."""
        from valence.substrate.tools import tension_list

        mock_get_cursor.fetchall.return_value = []

        result = tension_list(severity="high")

        assert result["success"] is True
        call_args = mock_get_cursor.execute.call_args[0]
        assert "severity" in call_args[0]


# ============================================================================
# tension_resolve Tests
# ============================================================================


class TestTensionResolve:
    """Tests for tension_resolve function."""

    def test_supersede_a(self, mock_get_cursor):
        """Should supersede belief A with B."""
        from valence.substrate.tools import tension_resolve

        tension_id = uuid4()
        belief_a = uuid4()
        belief_b = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.side_effect = [
            # Get tension
            {
                "id": tension_id,
                "belief_a_id": belief_a,
                "belief_b_id": belief_b,
                "type": "contradiction",
                "severity": "medium",
                "status": "detected",
                "detected_at": now,
            },
            # Get belief B content
            {"content": "Belief B is correct"},
            # Get belief A for supersession
            {
                "id": belief_a,
                "content": "Old content",
                "confidence": {"overall": 0.5},
                "domain_path": [],
                "valid_from": None,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": None,
                "superseded_by_id": None,
                "status": "active",
            },
            # New belief from supersession
            {
                "id": uuid4(),
                "content": "Belief B is correct",
                "confidence": {"overall": 0.5},
                "domain_path": [],
                "valid_from": now,
                "valid_until": None,
                "created_at": now,
                "modified_at": now,
                "source_id": None,
                "extraction_method": None,
                "supersedes_id": belief_a,
                "superseded_by_id": None,
                "status": "active",
            },
        ]

        result = tension_resolve(str(tension_id), "Belief B is more accurate", "supersede_a")

        assert result["success"] is True
        assert result["action"] == "supersede_a"

    def test_keep_both(self, mock_get_cursor):
        """Should mark tension as accepted when keeping both."""
        from valence.substrate.tools import tension_resolve

        tension_id = uuid4()
        now = datetime.now()

        mock_get_cursor.fetchone.return_value = {
            "id": tension_id,
            "belief_a_id": uuid4(),
            "belief_b_id": uuid4(),
            "type": "partial_overlap",
            "severity": "low",
            "status": "detected",
            "detected_at": now,
        }

        result = tension_resolve(str(tension_id), "Both beliefs are valid in different contexts", "keep_both")

        assert result["success"] is True
        assert result["action"] == "keep_both"

    def test_not_found(self, mock_get_cursor):
        """Should return error if tension not found."""
        from valence.substrate.tools import tension_resolve

        mock_get_cursor.fetchone.return_value = None

        result = tension_resolve(str(uuid4()), "Resolution", "keep_both")

        assert result["success"] is False
        assert "not found" in result["error"]
