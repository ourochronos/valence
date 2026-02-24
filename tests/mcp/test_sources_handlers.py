"""Tests for source MCP handlers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from valence.mcp.handlers.sources import source_get, source_ingest, source_list, source_search


def create_mock_cursor():
    """Create a mock cursor for testing."""
    mock_cursor = MagicMock()
    mock_get_cursor = MagicMock()
    mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_get_cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_cursor, mock_get_cursor


class TestSourceIngest:
    """Tests for source_ingest handler."""

    def test_ingest_success(self):
        """Test successful source ingestion."""
        source_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.side_effect = [
            None,  # No duplicate
            {  # Returned row
                "id": source_id,
                "type": "document",
                "title": "Test Note",
                "url": None,
                "content": "Test content",
                "fingerprint": "abc123",
                "reliability": 0.5,
                "content_hash": "abc123",
                "metadata": {},
                "created_at": None,
                "supersedes_id": None,
            },
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_ingest(
                content="Test content",
                source_type="document",
                title="Test Note",
            )

        assert result["success"] is True
        assert result["source"]["id"] == str(source_id)
        assert result["source"]["title"] == "Test Note"

    def test_ingest_empty_content(self):
        """Test ingesting source with empty content."""
        result = source_ingest(content="", source_type="document")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_ingest_whitespace_content(self):
        """Test ingesting source with whitespace-only content."""
        result = source_ingest(content="   ", source_type="document")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_ingest_invalid_source_type(self):
        """Test ingesting with invalid source_type."""
        result = source_ingest(content="Test", source_type="invalid_type")

        assert result["success"] is False
        assert "Invalid source_type" in result["error"]

    def test_ingest_duplicate_fingerprint(self):
        """Test ingesting duplicate source."""
        existing_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {"id": existing_id}

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_ingest(content="Duplicate content", source_type="document")

        assert result["success"] is False
        assert result["duplicate"] is True
        assert result["existing_id"] == str(existing_id)

    def test_ingest_with_url(self):
        """Test ingesting source with URL."""
        source_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.side_effect = [
            None,
            {
                "id": source_id,
                "type": "web",
                "title": "Test Link",
                "url": "https://example.com",
                "content": "Test",
                "fingerprint": "abc123",
                "reliability": 0.7,
                "content_hash": "abc123",
                "metadata": {},
                "created_at": None,
                "supersedes_id": None,
            },
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_ingest(
                content="Test",
                source_type="web",
                url="https://example.com",
                title="Test Link",
            )

        assert result["success"] is True
        assert result["source"]["url"] == "https://example.com"

    def test_ingest_with_metadata(self):
        """Test ingesting source with metadata."""
        source_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.side_effect = [
            None,
            {
                "id": source_id,
                "type": "document",
                "title": None,
                "url": None,
                "content": "Test",
                "fingerprint": "abc123",
                "reliability": 0.5,
                "content_hash": "abc123",
                "metadata": {"author": "Jane"},
                "created_at": None,
                "supersedes_id": None,
            },
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_ingest(
                content="Test",
                source_type="document",
                metadata={"author": "Jane"},
            )

        assert result["success"] is True
        assert result["source"]["metadata"]["author"] == "Jane"

    def test_ingest_with_supersedes(self):
        """Test ingesting source that supersedes another."""
        old_id = uuid4()
        new_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.side_effect = [
            None,  # No duplicate
            {"id": old_id},  # Superseded source exists
            {  # New source
                "id": new_id,
                "type": "document",
                "title": None,
                "url": None,
                "content": "New version",
                "fingerprint": "def456",
                "reliability": 0.5,
                "content_hash": "def456",
                "metadata": {},
                "created_at": None,
                "supersedes_id": str(old_id),  # Already string in mock
            },
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_ingest(
                content="New version",
                source_type="document",
                supersedes=str(old_id),
            )

        assert result["success"] is True
        assert result["source"]["supersedes_id"] == str(old_id)

    def test_ingest_supersedes_not_found(self):
        """Test ingesting with non-existent supersedes_id."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.side_effect = [
            None,  # No duplicate
            None,  # Superseded source not found
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_ingest(
                content="Test",
                source_type="document",
                supersedes=str(uuid4()),
            )

        assert result["success"] is False
        assert "not found" in result["error"]


class TestSourceGet:
    """Tests for source_get handler."""

    def test_get_success(self):
        """Test successful source retrieval."""
        source_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = {
            "id": source_id,
            "type": "document",
            "title": "Test Note",
            "url": None,
            "content": "Test content",
            "fingerprint": "abc123",
            "reliability": 0.5,
            "content_hash": "abc123",
            "metadata": {},
            "created_at": None,
        }

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_get(str(source_id))

        assert result["success"] is True
        assert result["source"]["id"] == str(source_id)
        assert result["source"]["title"] == "Test Note"

    def test_get_not_found(self):
        """Test getting non-existent source."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchone.return_value = None

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_get(str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]


class TestSourceSearch:
    """Tests for source_search handler."""

    def test_search_success(self):
        """Test successful source search."""
        source_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": source_id,
                "type": "document",
                "title": "Test Note",
                "url": None,
                "content": "Test content",
                "fingerprint": "abc123",
                "reliability": 0.5,
                "content_hash": "abc123",
                "metadata": {},
                "created_at": None,
                "rank": 0.5,
            }
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_search("test query")

        assert result["success"] is True
        assert len(result["sources"]) == 1
        assert result["total_count"] == 1

    def test_search_empty_query(self):
        """Test search with empty query."""
        result = source_search("")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_search_whitespace_query(self):
        """Test search with whitespace-only query."""
        result = source_search("   ")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_search_with_limit(self):
        """Test search with custom limit."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            source_search("test", limit=50)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][-1] == 50

    def test_search_limit_clamping(self):
        """Test search limit is clamped to max 200."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            source_search("test", limit=500)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][-1] == 200

    def test_search_limit_minimum(self):
        """Test search limit minimum is 1."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            source_search("test", limit=0)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][-1] == 1

    def test_search_no_results(self):
        """Test search with no results."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_search("nonexistent")

        assert result["success"] is True
        assert len(result["sources"]) == 0
        assert result["total_count"] == 0


class TestSourceList:
    """Tests for source_list handler."""

    def test_list_success(self):
        """Test successful source listing."""
        source_id = uuid4()
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": source_id,
                "type": "document",
                "title": "Test Note",
                "url": None,
                "fingerprint": "abc123",
                "reliability": 0.5,
                "created_at": None,
                "metadata": {},
            }
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_list()

        assert result["success"] is True
        assert len(result["sources"]) == 1
        assert result["total_count"] == 1

    def test_list_with_limit(self):
        """Test listing with custom limit."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            source_list(limit=100)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][0] == 100

    def test_list_limit_clamping(self):
        """Test list limit is clamped to max 200."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            source_list(limit=500)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][0] == 200

    def test_list_limit_minimum(self):
        """Test list limit minimum is 1."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            source_list(limit=0)

        call_args = mock_cursor.execute.call_args
        assert call_args[0][1][0] == 1

    def test_list_empty(self):
        """Test listing with no sources."""
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = []

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_list()

        assert result["success"] is True
        assert len(result["sources"]) == 0
        assert result["total_count"] == 0

    def test_list_with_created_at(self):
        """Test listing with created_at timestamp."""
        from datetime import datetime

        source_id = uuid4()
        created_at = datetime(2024, 1, 15, 10, 30, 0)
        mock_cursor, mock_get_cursor = create_mock_cursor()
        mock_cursor.fetchall.return_value = [
            {
                "id": source_id,
                "type": "document",
                "title": "Test",
                "url": None,
                "fingerprint": "abc123",
                "reliability": 0.5,
                "created_at": created_at,
                "metadata": {},
            }
        ]

        with patch("valence.mcp.handlers.sources.get_cursor", mock_get_cursor):
            result = source_list()

        assert result["success"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["created_at"] == created_at.isoformat()
