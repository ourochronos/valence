"""Tests for memory MCP tool handlers.

Tests cover:
1. memory_store — creates observation source with memory metadata
2. memory_recall — searches and filters memory sources
3. memory_status — returns memory statistics
4. memory_forget — marks memory as forgotten
5. memory_store with supersedes_id — sets FK correctly
"""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.mcp.handlers.memory import memory_forget, memory_recall, memory_status, memory_store

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
    def _mock(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.mcp.handlers.memory.get_cursor", _mock):
        yield mock_cursor


@pytest.fixture
def mock_source_ingest():
    """Mock source_ingest to return success."""
    with patch("valence.mcp.handlers.memory.source_ingest") as mock:
        mock.return_value = {
            "success": True,
            "source": {
                "id": str(uuid4()),
                "type": "observation",
                "title": "Test Memory",
                "content": "Test content",
                "reliability": 0.4,
                "created_at": datetime(2026, 2, 23, 12, 0, 0).isoformat(),
            },
        }
        yield mock


# ---------------------------------------------------------------------------
# Tests: memory_store
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_empty_content_fails(self):
        result = memory_store(content="")
        assert result["success"] is False
        assert "non-empty" in result["error"]

    def test_whitespace_content_fails(self):
        result = memory_store(content="   \n  ")
        assert result["success"] is False
        assert "non-empty" in result["error"]

    def test_basic_store(self, mock_source_ingest):
        result = memory_store(content="Important fact")
        assert result["success"] is True
        assert "memory_id" in result
        assert result["importance"] == 0.5

        # Check source_ingest was called with correct params
        mock_source_ingest.assert_called_once()
        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["content"] == "Important fact"
        assert call_kwargs["source_type"] == "observation"
        assert call_kwargs["metadata"]["memory"] is True
        assert call_kwargs["metadata"]["importance"] == 0.5

    def test_store_with_context(self, mock_source_ingest):
        result = memory_store(
            content="Test memory",
            context="session:main",
        )
        assert result["success"] is True

        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["metadata"]["context"] == "session:main"

    def test_store_with_tags(self, mock_source_ingest):
        result = memory_store(
            content="Test memory",
            tags=["infrastructure", "decision"],
        )
        assert result["success"] is True
        assert result["tags"] == ["infrastructure", "decision"]

        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["metadata"]["tags"] == ["infrastructure", "decision"]

    def test_store_with_importance(self, mock_source_ingest):
        result = memory_store(
            content="Critical memory",
            importance=0.9,
        )
        assert result["success"] is True
        assert result["importance"] == 0.9

        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["metadata"]["importance"] == 0.9

    def test_importance_clamped_to_range(self, mock_source_ingest):
        # Test upper bound
        result = memory_store(content="Test", importance=1.5)
        assert result["success"] is True
        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["metadata"]["importance"] == 1.0

        # Test lower bound
        mock_source_ingest.reset_mock()
        result = memory_store(content="Test", importance=-0.5)
        assert result["success"] is True
        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["metadata"]["importance"] == 0.0

    def test_store_with_supersedes_id(self, mock_source_ingest):
        old_id = str(uuid4())
        result = memory_store(
            content="Updated memory",
            supersedes_id=old_id,
        )
        assert result["success"] is True

        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["supersedes"] == old_id

    def test_title_from_first_line(self, mock_source_ingest):
        result = memory_store(content="First line\nSecond line")
        assert result["success"] is True

        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["title"] == "First line"

    def test_title_from_context_if_no_short_first_line(self, mock_source_ingest):
        long_line = "x" * 150
        result = memory_store(
            content=long_line,
            context="test:context",
        )
        assert result["success"] is True

        call_kwargs = mock_source_ingest.call_args.kwargs
        assert call_kwargs["title"] == "Memory from test:context"


# ---------------------------------------------------------------------------
# Tests: memory_recall
# ---------------------------------------------------------------------------


class TestMemoryRecall:
    def test_empty_query_fails(self):
        result = memory_recall(query="")
        assert result["success"] is False
        assert "non-empty" in result["error"]

    def test_recall_basic(self, mock_get_cursor):
        """Test basic recall workflow (mocked)."""
        # Mock knowledge_search to return a source
        source_id = str(uuid4())
        with patch("valence.mcp.handlers.articles.knowledge_search") as mock_ks:
            mock_ks.return_value = {
                "success": True,
                "results": [
                    {
                        "type": "source",
                        "id": source_id,
                        "content": "Test memory content",
                        "title": "Test Memory",
                        "reliability": 0.4,
                        "confidence": {"overall": 0.6},
                        "freshness": 5.0,
                        "created_at": "2026-02-18T12:00:00",
                        "final_score": 0.75,
                    }
                ],
            }

            # Mock the cursor to return memory metadata (batch fetch)
            mock_get_cursor.fetchall.return_value = [
                {
                    "id": source_id,
                    "type": "observation",
                    "metadata": json.dumps(
                        {
                            "memory": True,
                            "importance": 0.8,
                            "context": "test",
                            "tags": ["test"],
                        }
                    ),
                }
            ]

            result = memory_recall(query="test query", limit=5)

            assert result["success"] is True
            assert len(result["memories"]) == 1
            memory = result["memories"][0]
            assert memory["memory_id"] == source_id
            assert memory["importance"] == 0.8
            assert memory["tags"] == ["test"]

    def test_recall_filters_non_memory_sources(self, mock_get_cursor):
        """Non-memory sources should be filtered out."""
        source_id = str(uuid4())
        with patch("valence.mcp.handlers.articles.knowledge_search") as mock_ks:
            mock_ks.return_value = {
                "success": True,
                "results": [
                    {
                        "type": "source",
                        "id": source_id,
                        "content": "Not a memory",
                    }
                ],
            }

            # Mock cursor to return non-memory metadata (batch fetch)
            mock_get_cursor.fetchall.return_value = [
                {
                    "id": source_id,
                    "type": "document",
                    "metadata": json.dumps({}),
                }
            ]

            result = memory_recall(query="test", limit=5)

            assert result["success"] is True
            assert len(result["memories"]) == 0

    def test_recall_with_tag_filter(self, mock_get_cursor):
        """Memories without matching tags should be filtered."""
        source_id = str(uuid4())
        with patch("valence.mcp.handlers.articles.knowledge_search") as mock_ks:
            mock_ks.return_value = {
                "success": True,
                "results": [
                    {
                        "type": "source",
                        "id": source_id,
                        "content": "Memory",
                    }
                ],
            }

            # Mock cursor to return memory with different tags (batch fetch)
            mock_get_cursor.fetchall.return_value = [
                {
                    "id": source_id,
                    "type": "observation",
                    "metadata": json.dumps(
                        {
                            "memory": True,
                            "tags": ["other"],
                        }
                    ),
                }
            ]

            result = memory_recall(query="test", tags=["wanted"])

            assert result["success"] is True
            assert len(result["memories"]) == 0

    def test_recall_with_min_confidence(self, mock_get_cursor):
        """Memories below confidence threshold should be filtered."""
        source_id = str(uuid4())
        with patch("valence.mcp.handlers.articles.knowledge_search") as mock_ks:
            mock_ks.return_value = {
                "success": True,
                "results": [
                    {
                        "type": "source",
                        "id": source_id,
                        "content": "Memory",
                        "reliability": 0.3,
                        "confidence": {"overall": 0.3},
                    }
                ],
            }

            # Mock cursor for batch fetch
            mock_get_cursor.fetchall.return_value = [
                {
                    "id": source_id,
                    "type": "observation",
                    "metadata": json.dumps({"memory": True}),
                }
            ]

            result = memory_recall(query="test", min_confidence=0.5)

            assert result["success"] is True
            assert len(result["memories"]) == 0


# ---------------------------------------------------------------------------
# Tests: memory_status
# ---------------------------------------------------------------------------


class TestMemoryStatus:
    def test_status_returns_counts(self, mock_get_cursor):
        """Test that status returns memory counts and stats."""
        # Mock three queries: count, last_memory, compiled_articles, tag_dist
        mock_get_cursor.fetchone.side_effect = [
            {"count": 42},  # memory count
            {"last_memory": datetime(2026, 2, 23, 12, 0, 0)},  # last memory
            {"count": 10},  # compiled articles
        ]
        mock_get_cursor.fetchall.return_value = [
            {"tag": "infrastructure", "count": 15},
            {"tag": "decision", "count": 8},
        ]

        result = memory_status()

        assert result["success"] is True
        assert result["memory_count"] == 42
        assert result["compiled_articles"] == 10
        assert result["last_memory_at"] == "2026-02-23T12:00:00"
        assert result["top_tags"]["infrastructure"] == 15
        assert result["top_tags"]["decision"] == 8


# ---------------------------------------------------------------------------
# Tests: memory_forget
# ---------------------------------------------------------------------------


class TestMemoryForget:
    def test_forget_empty_id_fails(self):
        result = memory_forget(memory_id="")
        assert result["success"] is False
        assert "required" in result["error"]

    def test_forget_nonexistent_memory_fails(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = None

        result = memory_forget(memory_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]

    def test_forget_non_observation_fails(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {
            "id": str(uuid4()),
            "type": "document",
            "metadata": json.dumps({}),
        }

        result = memory_forget(memory_id=str(uuid4()))

        assert result["success"] is False
        assert "not a memory" in result["error"]

    def test_forget_non_memory_source_fails(self, mock_get_cursor):
        mock_get_cursor.fetchone.return_value = {
            "id": str(uuid4()),
            "type": "observation",
            "metadata": json.dumps({}),
        }

        result = memory_forget(memory_id=str(uuid4()))

        assert result["success"] is False
        assert "not marked as a memory" in result["error"]

    def test_forget_sets_metadata(self, mock_get_cursor):
        memory_id = str(uuid4())
        mock_get_cursor.fetchone.return_value = {
            "id": memory_id,
            "type": "observation",
            "metadata": json.dumps(
                {
                    "memory": True,
                    "importance": 0.5,
                }
            ),
        }

        result = memory_forget(
            memory_id=memory_id,
            reason="Outdated information",
        )

        assert result["success"] is True
        assert result["memory_id"] == memory_id
        assert result["forgotten"] is True
        assert result["reason"] == "Outdated information"

        # Check that UPDATE was called
        mock_get_cursor.execute.assert_called()
        update_call = [call for call in mock_get_cursor.execute.call_args_list if "UPDATE sources" in str(call)]
        assert len(update_call) > 0

    def test_forget_without_reason(self, mock_get_cursor):
        memory_id = str(uuid4())
        mock_get_cursor.fetchone.return_value = {
            "id": memory_id,
            "type": "observation",
            "metadata": json.dumps({"memory": True}),
        }

        result = memory_forget(memory_id=memory_id)

        assert result["success"] is True
        assert result["reason"] is None
