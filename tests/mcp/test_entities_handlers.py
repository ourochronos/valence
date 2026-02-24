"""Tests for entity MCP handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.mcp.handlers.entities import entity_get, entity_search


def create_mock_cursor():
    """Create a mock cursor for testing."""
    mock_cursor = MagicMock()
    mock_get_cursor = MagicMock()
    mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_cursor, mock_get_cursor


class TestEntityGet:
    """Tests for entity_get handler."""

    def test_entity_get_success(self):
        """Test successful entity retrieval."""
        entity_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": entity_id,
            "type": "person",
            "name": "John Doe",
            "canonical_id": None,
            "aliases": ["J. Doe"],
            "metadata": {"role": "engineer"},
            "created_at": None,
        }

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_get(str(entity_id))

        assert result["success"] is True
        assert result["entity"]["id"] == str(entity_id)
        assert result["entity"]["name"] == "John Doe"
        assert result["entity"]["type"] == "person"
        mock_cursor.execute.assert_called_once()

    def test_entity_get_not_found(self):
        """Test entity not found."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = None

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_get(str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_entity_get_with_canonical_id(self):
        """Test entity with canonical_id serialization."""
        entity_id = uuid4()
        canonical_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": entity_id,
            "type": "person",
            "name": "Jane Doe",
            "canonical_id": canonical_id,
            "aliases": [],
            "metadata": {},
            "created_at": None,
        }

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_get(str(entity_id))

        assert result["success"] is True
        assert result["entity"]["canonical_id"] == str(canonical_id)

    def test_entity_get_with_beliefs(self):
        """Test entity_get with include_beliefs flag."""
        entity_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": entity_id,
            "type": "concept",
            "name": "Test Concept",
            "canonical_id": None,
            "aliases": [],
            "metadata": {},
            "created_at": None,
        }

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_get(str(entity_id), include_beliefs=True)

        assert result["success"] is True
        assert "beliefs" in result["entity"]
        assert result["entity"]["beliefs"] == []

    def test_entity_get_database_error(self):
        """Test entity_get with database error."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.execute.side_effect = Exception("Database error")

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_get(str(uuid4()))

        assert result["success"] is False
        assert "Database error" in result["error"]

    def test_entity_get_with_created_at(self):
        """Test entity_get with created_at timestamp."""
        from datetime import datetime

        entity_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": entity_id,
            "type": "concept",
            "name": "Test",
            "canonical_id": None,
            "aliases": [],
            "metadata": {},
            "created_at": created_at,
        }

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_get(str(entity_id))

        assert result["success"] is True
        assert result["entity"]["created_at"] == created_at.isoformat()


class TestEntitySearch:
    """Tests for entity_search handler."""

    def test_entity_search_success(self):
        """Test successful entity search."""
        entity_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": entity_id,
                "type": "person",
                "name": "John Doe",
                "canonical_id": None,
                "aliases": ["J. Doe"],
                "metadata": {},
                "created_at": None,
            }
        ]

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_search("John")

        assert result["success"] is True
        assert len(result["entities"]) == 1
        assert result["entities"][0]["name"] == "John Doe"
        assert result["total_count"] == 1

    def test_entity_search_empty_query(self):
        """Test entity_search with empty query."""
        result = entity_search("")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_entity_search_whitespace_query(self):
        """Test entity_search with whitespace-only query."""
        result = entity_search("   ")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_entity_search_with_type_filter(self):
        """Test entity_search with entity_type filter."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": uuid4(),
                "type": "person",
                "name": "John Doe",
                "canonical_id": None,
                "aliases": [],
                "metadata": {},
                "created_at": None,
            }
        ]

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_search("John", entity_type="person")

        assert result["success"] is True
        assert len(result["entities"]) == 1
        # Verify the type filter was used in the query
        call_args = mock_cursor.execute.call_args
        assert "type = %s" in call_args[0][0]

    def test_entity_search_limit_parameter(self):
        """Test entity_search with custom limit."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            entity_search("test", limit=50)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][-1] == 50  # Last parameter should be limit

    def test_entity_search_limit_clamping(self):
        """Test entity_search limit is clamped to max 200."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            entity_search("test", limit=500)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][-1] == 200

    def test_entity_search_limit_minimum(self):
        """Test entity_search limit minimum is 1."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            entity_search("test", limit=0)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][-1] == 1

    def test_entity_search_no_results(self):
        """Test entity_search with no results."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_search("NonExistentEntity")

        assert result["success"] is True
        assert len(result["entities"]) == 0
        assert result["total_count"] == 0

    def test_entity_search_database_error(self):
        """Test entity_search with database error."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.execute.side_effect = Exception("Database error")

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_search("test")

        assert result["success"] is False
        assert "Database error" in result["error"]

    def test_entity_search_multiple_entities(self):
        """Test entity_search returning multiple entities."""
        from datetime import datetime

        created_at = datetime(2024, 1, 15, 10, 30, 0)
        canonical_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": uuid4(),
                "type": "person",
                "name": "John Doe",
                "canonical_id": None,
                "aliases": [],
                "metadata": {},
                "created_at": created_at,
            },
            {
                "id": uuid4(),
                "type": "person",
                "name": "Jane Doe",
                "canonical_id": canonical_id,
                "aliases": [],
                "metadata": {},
                "created_at": created_at,
            },
        ]

        with patch("valence.mcp.handlers.entities.get_cursor", mock_get_cursor):
            result = entity_search("Doe")

        assert result["success"] is True
        assert len(result["entities"]) == 2
        assert result["total_count"] == 2
        # Check datetime and canonical_id conversions
        assert result["entities"][0]["created_at"] == created_at.isoformat()
        assert result["entities"][1]["canonical_id"] == str(canonical_id)
        assert result["entities"][1]["created_at"] == created_at.isoformat()
