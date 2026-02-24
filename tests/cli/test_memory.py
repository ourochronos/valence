"""Tests for memory CLI commands.

Tests cover:
1. valence memory import <path> — imports a markdown file
2. valence memory import-dir <dir> — imports directory of .md files
3. valence memory list — lists memories
4. valence memory search <query> — searches memories
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from valence.cli.commands.memory import (
    cmd_memory_import,
    cmd_memory_import_dir,
    cmd_memory_list,
    cmd_memory_search,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cursor():
    """Mock cursor."""
    cur = MagicMock()
    cur.fetchone.return_value = None
    cur.fetchall.return_value = []
    return cur


@pytest.fixture
def tmp_md_file(tmp_path):
    """Create a temporary markdown file."""
    file = tmp_path / "test-memory.md"
    file.write_text("# Test Memory\n\nThis is a test memory.")
    return file


@pytest.fixture
def tmp_md_dir(tmp_path):
    """Create a directory with multiple markdown files."""
    dir_path = tmp_path / "memories"
    dir_path.mkdir()

    (dir_path / "memory1.md").write_text("First memory")
    (dir_path / "memory2.md").write_text("Second memory")
    (dir_path / "subdir").mkdir()
    (dir_path / "subdir" / "memory3.md").write_text("Third memory")

    return dir_path


# ---------------------------------------------------------------------------
# Tests: memory import
# ---------------------------------------------------------------------------


class TestMemoryImport:
    def test_import_nonexistent_file(self, capsys):
        args = MagicMock()
        args.path = Path("/nonexistent/file.md")
        args.context = None
        args.importance = 0.5
        args.tags = None

        with patch("valence.cli.commands.memory.output_error") as mock_err:
            code = cmd_memory_import(args)
            assert code == 1
            mock_err.assert_called_once()
            assert "not found" in str(mock_err.call_args).lower()

    def test_import_directory_as_file(self, tmp_path):
        args = MagicMock()
        args.path = tmp_path
        args.context = None
        args.importance = 0.5
        args.tags = None

        with patch("valence.cli.commands.memory.output_error") as mock_err:
            code = cmd_memory_import(args)
            assert code == 1
            mock_err.assert_called_once()
            assert "not a file" in str(mock_err.call_args).lower()

    def test_import_success(self, tmp_md_file):
        args = MagicMock()
        args.path = tmp_md_file
        args.context = "test:import"
        args.importance = 0.7
        args.tags = ["test"]

        with patch("valence.mcp.handlers.memory.memory_store") as mock_store:
            mock_store.return_value = {
                "success": True,
                "memory_id": "test-id",
                "title": "Test Memory",
            }

            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_import(args)
                assert code == 0

                # Check memory_store was called
                mock_store.assert_called_once()
                call_kwargs = mock_store.call_args.kwargs
                assert "Test Memory" in call_kwargs["content"]
                assert call_kwargs["context"] == "test:import"
                assert call_kwargs["importance"] == 0.7
                assert call_kwargs["tags"] == ["test"]

                # Check output
                mock_out.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: memory import-dir
# ---------------------------------------------------------------------------


class TestMemoryImportDir:
    def test_import_dir_nonexistent(self):
        args = MagicMock()
        args.directory = Path("/nonexistent")
        args.context = None
        args.importance = 0.5
        args.tags = None
        args.recursive = False

        with patch("valence.cli.commands.memory.output_error") as mock_err:
            code = cmd_memory_import_dir(args)
            assert code == 1
            mock_err.assert_called_once()
            assert "not found" in str(mock_err.call_args).lower()

    def test_import_dir_not_directory(self, tmp_md_file):
        args = MagicMock()
        args.directory = tmp_md_file
        args.context = None
        args.importance = 0.5
        args.tags = None
        args.recursive = False

        with patch("valence.cli.commands.memory.output_error") as mock_err:
            code = cmd_memory_import_dir(args)
            assert code == 1
            mock_err.assert_called_once()
            assert "not a directory" in str(mock_err.call_args).lower()

    def test_import_dir_no_md_files(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        args = MagicMock()
        args.directory = empty_dir
        args.context = None
        args.importance = 0.5
        args.tags = None
        args.recursive = False

        with patch("valence.cli.commands.memory.output_error") as mock_err:
            code = cmd_memory_import_dir(args)
            assert code == 1
            mock_err.assert_called_once()
            assert "no .md files" in str(mock_err.call_args).lower()

    def test_import_dir_success(self, tmp_md_dir):
        args = MagicMock()
        args.directory = tmp_md_dir
        args.context = "test:batch"
        args.importance = 0.6
        args.tags = ["batch"]
        args.recursive = False

        with patch("valence.mcp.handlers.memory.memory_store") as mock_store:
            mock_store.return_value = {
                "success": True,
                "memory_id": "test-id",
                "title": "Memory",
            }

            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_import_dir(args)
                assert code == 0

                # Should import 2 files (not recursive)
                assert mock_store.call_count == 2

                # Check output
                mock_out.assert_called_once()
                result = mock_out.call_args[0][0]
                assert result["imported_count"] == 2
                assert result["failed_count"] == 0

    def test_import_dir_recursive(self, tmp_md_dir):
        args = MagicMock()
        args.directory = tmp_md_dir
        args.context = None
        args.importance = 0.5
        args.tags = None
        args.recursive = True

        with patch("valence.mcp.handlers.memory.memory_store") as mock_store:
            mock_store.return_value = {
                "success": True,
                "memory_id": "test-id",
                "title": "Memory",
            }

            with patch("valence.cli.commands.memory.output_result"):
                code = cmd_memory_import_dir(args)
                assert code == 0

                # Should import 3 files (recursive)
                assert mock_store.call_count == 3


# ---------------------------------------------------------------------------
# Tests: memory list
# ---------------------------------------------------------------------------


class TestMemoryList:
    def test_list_success(self, mock_cursor):
        args = MagicMock()
        args.limit = 20
        args.tags = None

        mock_cursor.fetchall.return_value = [
            {
                "id": "mem-1",
                "title": "Memory 1",
                "content": "Content 1",
                "metadata": json.dumps(
                    {
                        "memory": True,
                        "importance": 0.8,
                        "tags": ["test"],
                    }
                ),
                "created_at": None,
            }
        ]

        with patch("valence.cli.commands.memory.get_cursor") as mock_gc:
            mock_gc.return_value.__enter__ = lambda _: mock_cursor
            mock_gc.return_value.__exit__ = lambda *_: None

            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_list(args)
                assert code == 0

                result = mock_out.call_args[0][0]
                assert result["success"] is True
                assert len(result["memories"]) == 1
                assert result["memories"][0]["importance"] == 0.8

    def test_list_with_tag_filter(self, mock_cursor):
        args = MagicMock()
        args.limit = 20
        args.tags = ["infrastructure"]

        mock_cursor.fetchall.return_value = []

        with patch("valence.cli.commands.memory.get_cursor") as mock_gc:
            mock_gc.return_value.__enter__ = lambda _: mock_cursor
            mock_gc.return_value.__exit__ = lambda *_: None

            with patch("valence.cli.commands.memory.output_result"):
                code = cmd_memory_list(args)
                assert code == 0

                # Verify query includes tag filter
                sql = mock_cursor.execute.call_args[0][0]
                assert "?|" in sql  # JSONB array overlap operator


# ---------------------------------------------------------------------------
# Tests: memory search
# ---------------------------------------------------------------------------


class TestMemorySearch:
    def test_search_success(self):
        args = MagicMock()
        args.query = "test query"
        args.limit = 10
        args.min_confidence = None
        args.tags = None

        with patch("valence.mcp.handlers.memory.memory_recall") as mock_recall:
            mock_recall.return_value = {
                "success": True,
                "memories": [
                    {
                        "memory_id": "mem-1",
                        "content": "Test memory",
                        "score": 0.9,
                    }
                ],
                "count": 1,
            }

            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_search(args)
                assert code == 0

                mock_recall.assert_called_once_with(
                    query="test query",
                    limit=10,
                    min_confidence=None,
                    tags=None,
                )
                mock_out.assert_called_once()

    def test_search_with_filters(self):
        args = MagicMock()
        args.query = "important"
        args.limit = 5
        args.min_confidence = 0.7
        args.tags = ["critical"]

        with patch("valence.mcp.handlers.memory.memory_recall") as mock_recall:
            mock_recall.return_value = {"success": True, "memories": [], "count": 0}

            with patch("valence.cli.commands.memory.output_result"):
                code = cmd_memory_search(args)
                assert code == 0

                call_kwargs = mock_recall.call_args.kwargs
                assert call_kwargs["min_confidence"] == 0.7
                assert call_kwargs["tags"] == ["critical"]
