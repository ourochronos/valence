"""Tests for articles command module."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

from valence.cli.commands.articles import (
    cmd_articles_create,
    cmd_articles_get,
    cmd_articles_list,
    cmd_articles_search,
    register,
)
from valence.cli.http_client import ValenceAPIError, ValenceConnectionError


class TestArticlesRegistration:
    """Test articles command registration."""

    def test_register(self):
        """Test that register creates all subcommands."""
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()

        register(subparsers)

        # Test that articles subcommand exists and has subcommands
        args_search = parser.parse_args(["articles", "search", "test"])
        assert args_search.articles_command == "search"
        assert args_search.query == "test"

        args_get = parser.parse_args(["articles", "get", "art123"])
        assert args_get.articles_command == "get"
        assert args_get.article_id == "art123"

        args_create = parser.parse_args(["articles", "create", "content"])
        assert args_create.articles_command == "create"
        assert args_create.content == "content"

        args_list = parser.parse_args(["articles", "list"])
        assert args_list.articles_command == "list"


class TestArticlesSearch:
    """Test articles search command."""

    @patch("valence.cli.commands.articles.get_client")
    def test_search_basic(self, mock_get_client):
        """Search articles with basic query."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "results": [
                {"id": "art1", "title": "Article 1", "content": "Content 1"},
                {"id": "art2", "title": "Article 2", "content": "Content 2"},
            ]
        }

        args = MagicMock(query="test query", limit=10, domain_filter=None)
        result = cmd_articles_search(args)

        assert result == 0
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/articles/search"
        body = call_args[1]["body"]
        assert body["query"] == "test query"
        assert body["limit"] == 10

    @patch("valence.cli.commands.articles.get_client")
    def test_search_with_domain_filter(self, mock_get_client):
        """Search articles with domain filter."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"results": []}

        args = MagicMock(query="test", limit=5, domain_filter=["tech", "science"])
        result = cmd_articles_search(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["domain_filter"] == ["tech", "science"]

    @patch("valence.cli.commands.articles.get_client")
    def test_search_custom_limit(self, mock_get_client):
        """Search with custom result limit."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"results": []}

        args = MagicMock(query="test", limit=50, domain_filter=None)
        result = cmd_articles_search(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["limit"] == 50

    @patch("valence.cli.commands.articles.get_client")
    def test_search_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(query="test", limit=10, domain_filter=None)
        result = cmd_articles_search(args)

        assert result == 1

    @patch("valence.cli.commands.articles.get_client")
    def test_search_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.post.side_effect = ValenceAPIError(500, "error", "Internal error")

        args = MagicMock(query="test", limit=10, domain_filter=None)
        result = cmd_articles_search(args)

        assert result == 1


class TestArticlesGet:
    """Test articles get command."""

    @patch("valence.cli.commands.articles.get_client")
    def test_get_basic(self, mock_get_client):
        """Get article by ID without provenance."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "id": "art123",
            "title": "Test Article",
            "content": "Article content",
        }

        args = MagicMock(article_id="art123", provenance=False)
        result = cmd_articles_get(args)

        assert result == 0
        mock_client.get.assert_called_with("/articles/art123", params=None)

    @patch("valence.cli.commands.articles.get_client")
    def test_get_with_provenance(self, mock_get_client):
        """Get article with provenance sources."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.get.return_value = {
            "id": "art123",
            "title": "Test",
            "provenance": [{"source_id": "src1"}],
        }

        args = MagicMock(article_id="art123", provenance=True)
        result = cmd_articles_get(args)

        assert result == 0
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "/articles/art123"
        assert call_args[1]["params"]["include_provenance"] == "true"

    @patch("valence.cli.commands.articles.get_client")
    def test_get_not_found(self, mock_get_client):
        """Handle article not found."""
        mock_get_client.return_value.get.side_effect = ValenceAPIError(404, "not_found", "Article not found")

        args = MagicMock(article_id="nonexistent", provenance=False)
        result = cmd_articles_get(args)

        assert result == 1

    @patch("valence.cli.commands.articles.get_client")
    def test_get_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.get.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(article_id="art123", provenance=False)
        result = cmd_articles_get(args)

        assert result == 1


class TestArticlesCreate:
    """Test articles create command."""

    @patch("valence.cli.commands.articles.get_client")
    def test_create_basic(self, mock_get_client):
        """Create article with minimal fields."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "id": "new_art",
            "content": "New article content",
        }

        args = MagicMock(
            content="New article content",
            title=None,
            domain_path=None,
            author_type="agent",
        )
        result = cmd_articles_create(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/articles"
        body = call_args[1]["body"]
        assert body["content"] == "New article content"
        assert body["author_type"] == "agent"

    @patch("valence.cli.commands.articles.get_client")
    def test_create_with_title(self, mock_get_client):
        """Create article with title."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"id": "new_art"}

        args = MagicMock(
            content="Content",
            title="Article Title",
            domain_path=None,
            author_type="agent",
        )
        result = cmd_articles_create(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["title"] == "Article Title"

    @patch("valence.cli.commands.articles.get_client")
    def test_create_with_domain_path(self, mock_get_client):
        """Create article with domain paths."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"id": "new_art"}

        args = MagicMock(
            content="Content",
            title=None,
            domain_path=["tech", "ai", "ml"],
            author_type="agent",
        )
        result = cmd_articles_create(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["domain_path"] == ["tech", "ai", "ml"]

    @patch("valence.cli.commands.articles.get_client")
    def test_create_author_types(self, mock_get_client):
        """Create article with different author types."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"id": "new_art"}

        for author_type in ["system", "operator", "agent"]:
            args = MagicMock(
                content="Content",
                title=None,
                domain_path=None,
                author_type=author_type,
            )
            result = cmd_articles_create(args)

            assert result == 0
            body = mock_client.post.call_args[1]["body"]
            assert body["author_type"] == author_type

    @patch("valence.cli.commands.articles.get_client")
    def test_create_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(content="Content", title=None, domain_path=None, author_type="agent")
        result = cmd_articles_create(args)

        assert result == 1

    @patch("valence.cli.commands.articles.get_client")
    def test_create_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.post.side_effect = ValenceAPIError(400, "invalid", "Invalid content")

        args = MagicMock(content="", title=None, domain_path=None, author_type="agent")
        result = cmd_articles_create(args)

        assert result == 1


class TestArticlesList:
    """Test articles list command."""

    @patch("valence.cli.commands.articles.get_client")
    def test_list_default(self, mock_get_client):
        """List articles with default limit."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {
            "results": [
                {"id": "art1", "title": "Article 1"},
                {"id": "art2", "title": "Article 2"},
            ]
        }

        args = MagicMock(limit=10)
        result = cmd_articles_list(args)

        assert result == 0
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/articles/search"
        body = call_args[1]["body"]
        assert body["query"] == "*"
        assert body["limit"] == 10

    @patch("valence.cli.commands.articles.get_client")
    def test_list_custom_limit(self, mock_get_client):
        """List articles with custom limit."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.post.return_value = {"results": []}

        args = MagicMock(limit=25)
        result = cmd_articles_list(args)

        assert result == 0
        body = mock_client.post.call_args[1]["body"]
        assert body["limit"] == 25

    @patch("valence.cli.commands.articles.get_client")
    def test_list_connection_error(self, mock_get_client):
        """Handle connection errors."""
        mock_get_client.return_value.post.side_effect = ValenceConnectionError("Connection refused")

        args = MagicMock(limit=10)
        result = cmd_articles_list(args)

        assert result == 1

    @patch("valence.cli.commands.articles.get_client")
    def test_list_api_error(self, mock_get_client):
        """Handle API errors."""
        mock_get_client.return_value.post.side_effect = ValenceAPIError(500, "error", "Internal error")

        args = MagicMock(limit=10)
        result = cmd_articles_list(args)

        assert result == 1
