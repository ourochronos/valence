"""Tests for new CLI commands (search, ingest, compile, status)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from valence.cli.commands.compile import cmd_compile
from valence.cli.commands.ingest import cmd_ingest
from valence.cli.commands.status import cmd_status
from valence.cli.commands.unified_search import cmd_search

# ==============================================================================
# Unified Search Tests
# ==============================================================================


class TestUnifiedSearch:
    """Test the unified search command."""

    @patch("valence.cli.commands.unified_search.get_client")
    def test_search_both(self, mock_get_client):
        """Search both articles and sources."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Mock responses
        mock_client.post.side_effect = [
            {"results": [{"id": "art1", "title": "Article 1"}]},  # articles
            {"results": [{"id": "src1", "title": "Source 1"}]},  # sources
        ]

        args = MagicMock(
            query="test query",
            limit=5,
            articles_only=False,
            sources_only=False,
            epistemic_type=None,
        )

        result = cmd_search(args)

        assert result == 0
        assert mock_client.post.call_count == 2
        mock_client.post.assert_any_call(
            "/articles/search",
            body={"query": "test query", "limit": 5},
        )
        mock_client.post.assert_any_call(
            "/sources/search",
            body={"query": "test query", "limit": 5},
        )

    @patch("valence.cli.commands.unified_search.get_client")
    def test_search_articles_only(self, mock_get_client):
        """Search articles only."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"results": [{"id": "art1"}]}

        args = MagicMock(
            query="test",
            limit=5,
            articles_only=True,
            sources_only=False,
            epistemic_type=None,
        )

        result = cmd_search(args)

        assert result == 0
        assert mock_client.post.call_count == 1
        mock_client.post.assert_called_with(
            "/articles/search",
            body={"query": "test", "limit": 5},
        )

    @patch("valence.cli.commands.unified_search.get_client")
    def test_search_sources_only(self, mock_get_client):
        """Search sources only."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"results": [{"id": "src1"}]}

        args = MagicMock(
            query="test",
            limit=5,
            articles_only=False,
            sources_only=True,
            epistemic_type=None,
        )

        result = cmd_search(args)

        assert result == 0
        assert mock_client.post.call_count == 1
        mock_client.post.assert_called_with(
            "/sources/search",
            body={"query": "test", "limit": 5},
        )

    def test_search_conflicting_flags(self):
        """Reject both --articles-only and --sources-only."""
        args = MagicMock(
            query="test",
            limit=5,
            articles_only=True,
            sources_only=True,
            epistemic_type=None,
        )

        result = cmd_search(args)
        assert result == 1


# ==============================================================================
# Ingest Tests
# ==============================================================================


class TestIngest:
    """Test the ingest command."""

    @patch("valence.cli.commands.ingest.get_client")
    def test_ingest_direct_content(self, mock_get_client):
        """Ingest direct content from argument."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True, "source": {"id": "src1"}}

        args = MagicMock(
            content="This is test content",
            title="Test Title",
            source_type="document",
            metadata=None,
        )

        result = cmd_ingest(args)

        assert result == 0
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/sources"
        body = call_args[1]["body"]
        assert body["content"] == "This is test content"
        assert body["source_type"] == "document"
        assert body["title"] == "Test Title"

    @patch("valence.cli.commands.ingest.get_client")
    @patch("valence.cli.commands.ingest.Path")
    def test_ingest_from_file(self, mock_path, mock_get_client):
        """Ingest content from file."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        # Mock file existence and content
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        mock_file.read_text.return_value = "File content"
        mock_path.return_value = mock_file

        args = MagicMock(
            content="/path/to/file.txt",
            title=None,
            source_type="document",
            metadata=None,
        )

        result = cmd_ingest(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["content"] == "File content"

    @patch("valence.cli.commands.ingest.get_client")
    @patch("valence.cli.commands.ingest.sys")
    def test_ingest_from_stdin(self, mock_sys, mock_get_client):
        """Ingest content from stdin."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        # Mock stdin
        mock_sys.stdin.read.return_value = "Stdin content"

        args = MagicMock(
            content="-",
            title=None,
            source_type="observation",
            metadata=None,
        )

        result = cmd_ingest(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["content"] == "Stdin content"
        assert body["source_type"] == "observation"

    @patch("valence.cli.commands.ingest.get_client")
    def test_ingest_with_metadata(self, mock_get_client):
        """Ingest with JSON metadata."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(
            content="Content",
            title=None,
            source_type="document",
            metadata='{"author":"Alice","tags":["test"]}',
        )

        result = cmd_ingest(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["metadata"] == {"author": "Alice", "tags": ["test"]}

    @patch("valence.cli.commands.ingest.get_client")
    def test_ingest_invalid_metadata(self, mock_get_client):
        """Reject invalid JSON metadata."""
        args = MagicMock(
            content="Content",
            title=None,
            source_type="document",
            metadata='{"invalid": json}',
        )

        result = cmd_ingest(args)
        assert result == 1

    @patch("valence.cli.commands.ingest.get_client")
    def test_ingest_url(self, mock_get_client):
        """Ingest URL adds url field."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(
            content="https://example.com/page",
            title=None,
            source_type="web",
            metadata=None,
        )

        result = cmd_ingest(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["url"] == "https://example.com/page"
        assert body["content"] == "https://example.com/page"


# ==============================================================================
# Compile Tests
# ==============================================================================


class TestCompile:
    """Test the compile command."""

    @patch("valence.cli.commands.compile.get_client")
    def test_compile_specific_sources(self, mock_get_client):
        """Compile specific source IDs."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True, "article": {"id": "art1"}}

        args = MagicMock(
            source_ids=["src1", "src2"],
            title="Test Article",
            auto=False,
            recompile=None,
            recompile_degraded=False,
            drain_queue=False,
        )

        result = cmd_compile(args)

        assert result == 0
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/articles/compile"
        body = call_args[1]["body"]
        assert body["source_ids"] == ["src1", "src2"]
        assert body["title_hint"] == "Test Article"

    @patch("valence.cli.commands.compile.get_client")
    def test_compile_auto(self, mock_get_client):
        """Auto-compile unlinked sources."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"success": True}

        args = MagicMock(
            source_ids=[],
            title=None,
            auto=True,
            recompile=None,
            recompile_degraded=False,
            drain_queue=False,
        )

        result = cmd_compile(args)

        assert result == 0
        mock_client.post.assert_called_with("/articles/compile/auto", body={})

    def test_compile_no_sources_or_auto(self):
        """Reject compile with no sources and no --auto."""
        args = MagicMock(
            source_ids=[],
            title=None,
            auto=False,
            recompile=None,
            recompile_degraded=False,
            drain_queue=False,
        )

        result = cmd_compile(args)
        assert result == 1


# ==============================================================================
# Status Tests
# ==============================================================================


class TestStatus:
    """Test the status command."""

    @patch("valence.cli.commands.status.get_client")
    def test_status_from_endpoint(self, mock_get_client):
        """Get status from dedicated endpoint."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "articles": {"total": 100, "active": 95},
            "sources": {"total": 500},
            "embeddings": {"coverage": "95.0%"},
        }

        args = MagicMock()
        result = cmd_status(args)

        assert result == 0
        mock_client.get.assert_called_with("/status")

    @patch("valence.cli.commands.status.get_client")
    def test_status_fallback_to_stats(self, mock_get_client):
        """Fallback to /stats if /status doesn't exist."""
        from valence.cli.http_client import ValenceAPIError

        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First call to /status returns 404
        # Second call to /stats succeeds
        mock_client.get.side_effect = [
            ValenceAPIError(404, "not_found", "Not found"),
            {
                "articles": {"total": 100, "active": 95},
                "sources": {"total": 500},
                "embeddings": {"coverage_pct": 95.0, "total": 95},
            },
        ]

        args = MagicMock()
        result = cmd_status(args)

        assert result == 0
        assert mock_client.get.call_count == 2
