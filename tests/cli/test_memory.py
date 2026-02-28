"""Tests for memory CLI commands.

Tests cover:
1. valence memory import <path> — imports a markdown file via REST
2. valence memory import-dir <dir> — imports directory of .md files via REST
3. valence memory list — lists memories via REST
4. valence memory search <query> — searches memories via REST
"""

from __future__ import annotations

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
def mock_client():
    """Mock ValenceClient."""
    client = MagicMock()
    client.get.return_value = {"success": True}
    client.post.return_value = {"success": True, "memory_id": "test-id", "title": "Memory"}
    return client


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
    def test_import_nonexistent_file(self):
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

    def test_import_success(self, tmp_md_file, mock_client):
        args = MagicMock()
        args.path = tmp_md_file
        args.context = "test:import"
        args.importance = 0.7
        args.tags = ["test"]

        with patch("valence.cli.commands.memory.get_client", return_value=mock_client):
            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_import(args)
                assert code == 0

                # Check REST call was made
                mock_client.post.assert_called_once()
                call_kwargs = mock_client.post.call_args
                body = call_kwargs.kwargs.get("body") or call_kwargs[1].get("body")
                assert "Test Memory" in body["content"]
                assert body["context"] == "test:import"
                assert body["importance"] == 0.7
                assert body["tags"] == ["test"]

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

    def test_import_dir_success(self, tmp_md_dir, mock_client):
        args = MagicMock()
        args.directory = tmp_md_dir
        args.context = "test:batch"
        args.importance = 0.6
        args.tags = ["batch"]
        args.recursive = False

        with patch("valence.cli.commands.memory.get_client", return_value=mock_client):
            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_import_dir(args)
                assert code == 0

                # Should import 2 files (not recursive)
                assert mock_client.post.call_count == 2

                mock_out.assert_called_once()
                result = mock_out.call_args[0][0]
                assert result["imported_count"] == 2
                assert result["failed_count"] == 0

    def test_import_dir_recursive(self, tmp_md_dir, mock_client):
        args = MagicMock()
        args.directory = tmp_md_dir
        args.context = None
        args.importance = 0.5
        args.tags = None
        args.recursive = True

        with patch("valence.cli.commands.memory.get_client", return_value=mock_client):
            with patch("valence.cli.commands.memory.output_result"):
                code = cmd_memory_import_dir(args)
                assert code == 0

                # Should import 3 files (recursive)
                assert mock_client.post.call_count == 3


# ---------------------------------------------------------------------------
# Tests: memory list
# ---------------------------------------------------------------------------


class TestMemoryList:
    def test_list_success(self, mock_client):
        args = MagicMock()
        args.limit = 20
        args.tags = None

        mock_client.get.return_value = {
            "success": True,
            "memories": [
                {
                    "memory_id": "mem-1",
                    "title": "Memory 1",
                    "content_preview": "Content 1",
                    "importance": 0.8,
                    "tags": ["test"],
                }
            ],
            "count": 1,
        }

        with patch("valence.cli.commands.memory.get_client", return_value=mock_client):
            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_list(args)
                assert code == 0

                mock_client.get.assert_called_once_with("/memory", params={"limit": "20"})
                mock_out.assert_called_once()

    def test_list_with_tag_filter(self, mock_client):
        args = MagicMock()
        args.limit = 20
        args.tags = ["infrastructure"]

        mock_client.get.return_value = {"success": True, "memories": [], "count": 0}

        with patch("valence.cli.commands.memory.get_client", return_value=mock_client):
            with patch("valence.cli.commands.memory.output_result"):
                code = cmd_memory_list(args)
                assert code == 0

                # Verify tags are passed as comma-separated
                call_args = mock_client.get.call_args
                params = call_args.kwargs.get("params") or call_args[1]
                assert params["tags"] == "infrastructure"


# ---------------------------------------------------------------------------
# Tests: memory search
# ---------------------------------------------------------------------------


class TestMemorySearch:
    def test_search_success(self, mock_client):
        args = MagicMock()
        args.query = "test query"
        args.limit = 10
        args.min_confidence = None
        args.tags = None

        mock_client.get.return_value = {
            "success": True,
            "memories": [{"memory_id": "mem-1", "content": "Test memory", "score": 0.9}],
            "count": 1,
        }

        with patch("valence.cli.commands.memory.get_client", return_value=mock_client):
            with patch("valence.cli.commands.memory.output_result") as mock_out:
                code = cmd_memory_search(args)
                assert code == 0

                mock_client.get.assert_called_once_with(
                    "/memory/search",
                    params={"query": "test query", "limit": "10"},
                )
                mock_out.assert_called_once()

    def test_search_with_filters(self, mock_client):
        args = MagicMock()
        args.query = "important"
        args.limit = 5
        args.min_confidence = 0.7
        args.tags = ["critical"]

        mock_client.get.return_value = {"success": True, "memories": [], "count": 0}

        with patch("valence.cli.commands.memory.get_client", return_value=mock_client):
            with patch("valence.cli.commands.memory.output_result"):
                code = cmd_memory_search(args)
                assert code == 0

                call_args = mock_client.get.call_args
                params = call_args.kwargs.get("params") or call_args[1]
                assert params["min_confidence"] == "0.7"
                assert params["tags"] == "critical"
