"""Tests for sources command module."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

from valence.cli.commands.sources import (
    cmd_sources_get,
    cmd_sources_ingest,
    cmd_sources_list,
    cmd_sources_search,
    register,
)
from valence.cli.http_client import ValenceAPIError, ValenceConnectionError


class TestSourcesRegistration:
    """Test sources command registration."""

    def test_register(self):
        """Test that register creates all subcommands."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        register(subparsers)

        # Test that sources subcommand exists and has subcommands
        args_list = parser.parse_args(["sources", "list"])
        assert args_list.sources_command == "list"

        args_get = parser.parse_args(["sources", "get", "src123"])
        assert args_get.sources_command == "get"
        assert args_get.source_id == "src123"

        args_ingest = parser.parse_args(["sources", "ingest", "content"])
        assert args_ingest.sources_command == "ingest"
        assert args_ingest.content == "content"

        args_search = parser.parse_args(["sources", "search", "query"])
        assert args_search.sources_command == "search"
        assert args_search.query == "query"


class TestSourcesList:
    """Test sources list command."""

    @patch("valence.cli.commands.sources.get_client")
    def test_list_default(self, mock_get_client):
        """List sources with default parameters."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "sources": [
                {"id": "src1", "content": "Source 1"},
                {"id": "src2", "content": "Source 2"},
            ]
        }

        args = MagicMock(source_type=None, limit=20, offset=0)
        result = cmd_sources_list(args)

        assert result == 0
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "/sources"
        params = call_args[1]["params"]
        assert params["limit"] == 20
        assert params["offset"] == 0

    @patch("valence.cli.commands.sources.get_client")
    def test_list_with_type_filter(self, mock_get_client):
        """List sources filtered by type."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"sources": []}

        args = MagicMock(source_type="document", limit=20, offset=0)
        result = cmd_sources_list(args)

        assert result == 0
        params = mock_client.get.call_args[1]["params"]
        assert params["source_type"] == "document"

    @patch("valence.cli.commands.sources.get_client")
    def test_list_with_pagination(self, mock_get_client):
        """List sources with custom limit and offset."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {"sources": []}

        args = MagicMock(source_type=None, limit=50, offset=100)
        result = cmd_sources_list(args)

        assert result == 0
        params = mock_client.get.call_args[1]["params"]
        assert params["limit"] == 50
        assert params["offset"] == 100

    @patch("valence.cli.commands.sources.get_client")
    def test_list_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.get.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(source_type=None, limit=20, offset=0)
        result = cmd_sources_list(args)

        assert result == 1

    @patch("valence.cli.commands.sources.get_client")
    def test_list_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.get.side_effect = ValenceAPIError(500, "error", "Internal error")

        args = MagicMock(source_type=None, limit=20, offset=0)
        result = cmd_sources_list(args)

        assert result == 1


class TestSourcesGet:
    """Test sources get command."""

    @patch("valence.cli.commands.sources.get_client")
    def test_get_success(self, mock_get_client):
        """Get source by ID."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "id": "src123",
            "content": "Source content",
            "source_type": "document",
        }

        args = MagicMock(source_id="src123")
        result = cmd_sources_get(args)

        assert result == 0
        mock_client.get.assert_called_with("/sources/src123")

    @patch("valence.cli.commands.sources.get_client")
    def test_get_not_found(self, mock_get_client):
        """Handle source not found."""
        mock_get_client.return_value.get.side_effect = ValenceAPIError(404, "not_found", "Source not found")

        args = MagicMock(source_id="nonexistent")
        result = cmd_sources_get(args)

        assert result == 1

    @patch("valence.cli.commands.sources.get_client")
    def test_get_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.get.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(source_id="src123")
        result = cmd_sources_get(args)

        assert result == 1


class TestSourcesIngest:
    """Test sources ingest command."""

    @patch("valence.cli.commands.sources.get_client")
    def test_ingest_basic(self, mock_get_client):
        """Ingest source with minimal fields."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "id": "new_src",
            "content": "Source content",
        }

        args = MagicMock(
            content="Source content",
            source_type="document",
            title=None,
            url=None,
        )
        result = cmd_sources_ingest(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/sources"
        body = call_args[1]["body"]
        assert body["content"] == "Source content"
        assert body["source_type"] == "document"

    @patch("valence.cli.commands.sources.get_client")
    def test_ingest_with_title(self, mock_get_client):
        """Ingest source with title."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"id": "new_src"}

        args = MagicMock(
            content="Content",
            source_type="document",
            title="Document Title",
            url=None,
        )
        result = cmd_sources_ingest(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["title"] == "Document Title"

    @patch("valence.cli.commands.sources.get_client")
    def test_ingest_with_url(self, mock_get_client):
        """Ingest source with URL."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"id": "new_src"}

        args = MagicMock(
            content="Content",
            source_type="web",
            title=None,
            url="https://example.com",
        )
        result = cmd_sources_ingest(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["url"] == "https://example.com"

    @patch("valence.cli.commands.sources.get_client")
    def test_ingest_all_source_types(self, mock_get_client):
        """Ingest sources with different types."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"id": "new_src"}

        source_types = ["document", "conversation", "web", "code", "observation", "tool_output", "user_input"]
        for src_type in source_types:
            args = MagicMock(
                content="Content",
                source_type=src_type,
                title=None,
                url=None,
            )
            result = cmd_sources_ingest(args)

            assert result == 0
            body = mock_client.post.call_args[1]["body"]
            assert body["source_type"] == src_type

    @patch("valence.cli.commands.sources.get_client")
    def test_ingest_with_all_fields(self, mock_get_client):
        """Ingest source with all optional fields."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"id": "new_src"}

        args = MagicMock(
            content="Full content",
            source_type="web",
            title="Full Title",
            url="https://example.com/page",
        )
        result = cmd_sources_ingest(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["content"] == "Full content"
        assert body["source_type"] == "web"
        assert body["title"] == "Full Title"
        assert body["url"] == "https://example.com/page"

    @patch("valence.cli.commands.sources.get_client")
    def test_ingest_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(content="Content", source_type="document", title=None, url=None)
        result = cmd_sources_ingest(args)

        assert result == 1

    @patch("valence.cli.commands.sources.get_client")
    def test_ingest_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.post.side_effect = ValenceAPIError(400, "invalid", "Invalid content")

        args = MagicMock(content="", source_type="document", title=None, url=None)
        result = cmd_sources_ingest(args)

        assert result == 1


class TestSourcesSearch:
    """Test sources search command."""

    @patch("valence.cli.commands.sources.get_client")
    def test_search_basic(self, mock_get_client):
        """Search sources with basic query."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "results": [
                {"id": "src1", "content": "Match 1"},
                {"id": "src2", "content": "Match 2"},
            ]
        }

        args = MagicMock(query="test query", limit=20)
        result = cmd_sources_search(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/sources/search"
        body = call_args[1]["body"]
        assert body["query"] == "test query"
        assert body["limit"] == 20

    @patch("valence.cli.commands.sources.get_client")
    def test_search_custom_limit(self, mock_get_client):
        """Search sources with custom limit."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"results": []}

        args = MagicMock(query="test", limit=100)
        result = cmd_sources_search(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["limit"] == 100

    @patch("valence.cli.commands.sources.get_client")
    def test_search_empty_results(self, mock_get_client):
        """Handle search with no results."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"results": []}

        args = MagicMock(query="nonexistent", limit=20)
        result = cmd_sources_search(args)

        assert result == 0

    @patch("valence.cli.commands.sources.get_client")
    def test_search_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(query="test", limit=20)
        result = cmd_sources_search(args)

        assert result == 1

    @patch("valence.cli.commands.sources.get_client")
    def test_search_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.post.side_effect = ValenceAPIError(500, "error", "Internal error")

        args = MagicMock(query="test", limit=20)
        result = cmd_sources_search(args)

        assert result == 1
