"""Tests for article MCP handlers."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

from valence.core.response import ValenceResponse
from valence.mcp.handlers.articles import (
    article_compile,
    article_create,
    article_get,
    article_merge,
    article_search,
    article_split,
    article_update,
    knowledge_search,
)


class TestKnowledgeSearch:
    """Tests for knowledge_search handler."""

    def test_search_success(self):
        """Test successful knowledge search."""
        results = [
            {
                "article_id": str(uuid4()),
                "content": "Test article",
                "score": 0.9,
            }
        ]

        with patch("valence.mcp.handlers.articles.run_async", return_value=results):
            result = knowledge_search("test query")

        assert result["success"] is True
        assert len(result["results"]) == 1
        assert result["total_count"] == 1
        assert result["query"] == "test query"

    def test_search_empty_query(self):
        """Test search with empty query."""
        result = knowledge_search("")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_search_whitespace_query(self):
        """Test search with whitespace-only query."""
        result = knowledge_search("   ")

        assert result["success"] is False
        assert "must be non-empty" in result["error"]

    def test_search_with_limit(self):
        """Test search with custom limit."""
        with patch("valence.mcp.handlers.articles.run_async", return_value=[]):
            result = knowledge_search("test", limit=50)

        assert result["success"] is True

    def test_search_limit_clamping(self):
        """Test search limit is clamped to max 200."""
        with patch("valence.core.retrieval.retrieve", return_value=[]):
            with patch("valence.mcp.handlers.articles.run_async", return_value=[]):
                result = knowledge_search("test", limit=500)

        assert result["success"] is True

    def test_search_with_sources(self):
        """Test search with include_sources flag."""
        results = [
            {
                "article_id": str(uuid4()),
                "content": "Test",
                "sources": [{"source_id": str(uuid4())}],
            }
        ]

        with patch("valence.mcp.handlers.articles.run_async", return_value=results):
            result = knowledge_search("test", include_sources=True)

        assert result["success"] is True

    def test_search_with_session_id(self):
        """Test search with session_id."""
        with patch("valence.mcp.handlers.articles.run_async", return_value=[]):
            result = knowledge_search("test", session_id="session-123")

        assert result["success"] is True

    def test_search_exception(self):
        """Test search with exception."""
        with patch("valence.mcp.handlers.articles.run_async", side_effect=Exception("Search error")):
            result = knowledge_search("test")

        assert result["success"] is False
        assert "Search error" in result["error"]

    def test_search_valence_response(self):
        """Test search returning ValenceResponse."""
        results = [{"article_id": str(uuid4())}]
        mock_response = ValenceResponse(success=True, data=results)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = knowledge_search("test")

        assert result["success"] is True
        assert len(result["results"]) == 1

    def test_search_valence_response_failure(self):
        """Test search with failed ValenceResponse."""
        mock_response = ValenceResponse(success=False, error="No results")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = knowledge_search("test")

        assert result["success"] is False
        assert "No results" in result["error"]


class TestArticleCreate:
    """Tests for article_create handler."""

    def test_create_success(self):
        """Test successful article creation."""
        article_data = {
            "id": str(uuid4()),
            "content": "Test article",
            "title": "Test Title",
        }
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_create(content="Test article", title="Test Title")

        assert result["success"] is True
        assert result["article"]["id"] == article_data["id"]

    def test_create_with_sources(self):
        """Test creating article with source_ids."""
        article_data = {"id": str(uuid4()), "content": "Test"}
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_create(
                content="Test",
                source_ids=[str(uuid4()), str(uuid4())],
            )

        assert result["success"] is True

    def test_create_with_author_type(self):
        """Test creating article with custom author_type."""
        article_data = {"id": str(uuid4()), "content": "Test"}
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_create(content="Test", author_type="human")

        assert result["success"] is True

    def test_create_with_domain_path(self):
        """Test creating article with domain_path."""
        article_data = {"id": str(uuid4()), "content": "Test"}
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_create(
                content="Test",
                domain_path=["tech", "ai"],
            )

        assert result["success"] is True

    def test_create_failure(self):
        """Test article creation failure."""
        mock_response = ValenceResponse(success=False, error="Creation failed")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_create(content="Test")

        assert result["success"] is False
        assert "Creation failed" in result["error"]


class TestArticleGet:
    """Tests for article_get handler."""

    def test_get_success(self):
        """Test successful article retrieval."""
        article_data = {
            "id": str(uuid4()),
            "content": "Test article",
            "title": "Test",
        }
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_get(article_id=str(uuid4()))

        assert result["success"] is True
        assert result["article"]["content"] == "Test article"

    def test_get_with_provenance(self):
        """Test getting article with provenance."""
        article_data = {
            "id": str(uuid4()),
            "content": "Test",
            "provenance": [{"source_id": str(uuid4())}],
        }
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_get(article_id=str(uuid4()), include_provenance=True)

        assert result["success"] is True

    def test_get_failure(self):
        """Test article retrieval failure."""
        mock_response = ValenceResponse(success=False, error="Article not found")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_get(article_id=str(uuid4()))

        assert result["success"] is False
        assert "not found" in result["error"]


class TestArticleUpdate:
    """Tests for article_update handler."""

    def test_update_success(self):
        """Test successful article update."""
        article_data = {"id": str(uuid4()), "content": "Updated content"}
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_update(
                article_id=str(uuid4()),
                content="Updated content",
            )

        assert result["success"] is True
        assert result["article"]["content"] == "Updated content"

    def test_update_with_source(self):
        """Test updating article with source_id."""
        article_data = {"id": str(uuid4()), "content": "Updated"}
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_update(
                article_id=str(uuid4()),
                content="Updated",
                source_id=str(uuid4()),
            )

        assert result["success"] is True

    def test_update_failure(self):
        """Test article update failure."""
        mock_response = ValenceResponse(success=False, error="Update failed")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_update(article_id=str(uuid4()), content="Test")

        assert result["success"] is False
        assert "Update failed" in result["error"]


class TestArticleSearch:
    """Tests for article_search handler."""

    def test_search_success(self):
        """Test successful article search."""
        articles = [
            {"id": str(uuid4()), "content": "Test article 1"},
            {"id": str(uuid4()), "content": "Test article 2"},
        ]
        mock_response = ValenceResponse(success=True, data=articles)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_search(query="test")

        assert result["success"] is True
        assert len(result["articles"]) == 2
        assert result["count"] == 2

    def test_search_with_limit(self):
        """Test article search with limit."""
        mock_response = ValenceResponse(success=True, data=[])

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_search(query="test", limit=50)

        assert result["success"] is True

    def test_search_with_domain_filter(self):
        """Test article search with domain_filter."""
        mock_response = ValenceResponse(success=True, data=[])

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_search(query="test", domain_filter=["tech"])

        assert result["success"] is True

    def test_search_no_results(self):
        """Test article search with no results."""
        mock_response = ValenceResponse(success=True, data=[])

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_search(query="nonexistent")

        assert result["success"] is True
        assert len(result["articles"]) == 0

    def test_search_none_data(self):
        """Test article search when data is None."""
        mock_response = ValenceResponse(success=True, data=None)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_search(query="test")

        assert result["success"] is True
        assert result["articles"] == []

    def test_search_failure(self):
        """Test article search failure."""
        mock_response = ValenceResponse(success=False, error="Search failed")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_search(query="test")

        assert result["success"] is False
        assert "Search failed" in result["error"]


class TestArticleCompile:
    """Tests for article_compile handler."""

    def test_compile_success(self):
        """Test successful article compilation."""
        article_data = {"id": str(uuid4()), "content": "Compiled article"}
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_compile(source_ids=[str(uuid4())])

        assert result["success"] is True
        assert result["article"]["content"] == "Compiled article"

    def test_compile_empty_source_ids(self):
        """Test compilation with empty source_ids."""
        result = article_compile(source_ids=[])

        assert result["success"] is False
        assert "must be a non-empty list" in result["error"]

    def test_compile_with_title_hint(self):
        """Test compilation with title_hint."""
        article_data = {"id": str(uuid4()), "title": "Custom Title"}
        mock_response = ValenceResponse(success=True, data=article_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_compile(
                source_ids=[str(uuid4())],
                title_hint="Custom Title",
            )

        assert result["success"] is True

    def test_compile_degraded(self):
        """Test compilation with degraded result."""
        article_data = {"id": str(uuid4()), "content": "Partial"}
        mock_response = ValenceResponse(success=True, data=article_data, degraded=True)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_compile(source_ids=[str(uuid4())])

        assert result["success"] is True
        assert result.get("degraded") is True

    def test_compile_failure(self):
        """Test compilation failure."""
        mock_response = ValenceResponse(success=False, error="Compilation failed")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_compile(source_ids=[str(uuid4())])

        assert result["success"] is False
        assert "Compilation failed" in result["error"]

    def test_compile_exception(self):
        """Test compilation with exception."""
        with patch("valence.mcp.handlers.articles.run_async", side_effect=Exception("Compile error")):
            result = article_compile(source_ids=[str(uuid4())])

        assert result["success"] is False
        assert "Compile error" in result["error"]


class TestArticleSplit:
    """Tests for article_split handler."""

    def test_split_success(self):
        """Test successful article split."""
        split_data = {
            "article_a_id": str(uuid4()),
            "article_b_id": str(uuid4()),
        }
        mock_response = ValenceResponse(success=True, data=split_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_split(article_id=str(uuid4()))

        assert result["success"] is True
        assert "article_a_id" in result
        assert "article_b_id" in result

    def test_split_empty_article_id(self):
        """Test split with empty article_id."""
        result = article_split(article_id="")

        assert result["success"] is False
        assert "is required" in result["error"]

    def test_split_failure(self):
        """Test article split failure."""
        mock_response = ValenceResponse(success=False, error="Split failed")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_split(article_id=str(uuid4()))

        assert result["success"] is False
        assert "Split failed" in result["error"]

    def test_split_exception(self):
        """Test split with exception."""
        with patch("valence.mcp.handlers.articles.run_async", side_effect=Exception("Split error")):
            result = article_split(article_id=str(uuid4()))

        assert result["success"] is False
        assert "Split error" in result["error"]


class TestArticleMerge:
    """Tests for article_merge handler."""

    def test_merge_success(self):
        """Test successful article merge."""
        merged_data = {"id": str(uuid4()), "content": "Merged article"}
        mock_response = ValenceResponse(success=True, data=merged_data)

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_merge(
                article_id_a=str(uuid4()),
                article_id_b=str(uuid4()),
            )

        assert result["success"] is True
        assert result["merged_article"]["content"] == "Merged article"

    def test_merge_empty_article_id_a(self):
        """Test merge with empty article_id_a."""
        result = article_merge(article_id_a="", article_id_b=str(uuid4()))

        assert result["success"] is False
        assert "are required" in result["error"]

    def test_merge_empty_article_id_b(self):
        """Test merge with empty article_id_b."""
        result = article_merge(article_id_a=str(uuid4()), article_id_b="")

        assert result["success"] is False
        assert "are required" in result["error"]

    def test_merge_both_empty(self):
        """Test merge with both article_ids empty."""
        result = article_merge(article_id_a="", article_id_b="")

        assert result["success"] is False
        assert "are required" in result["error"]

    def test_merge_failure(self):
        """Test article merge failure."""
        mock_response = ValenceResponse(success=False, error="Merge failed")

        with patch("valence.mcp.handlers.articles.run_async", return_value=mock_response):
            result = article_merge(
                article_id_a=str(uuid4()),
                article_id_b=str(uuid4()),
            )

        assert result["success"] is False
        assert "Merge failed" in result["error"]

    def test_merge_exception(self):
        """Test merge with exception."""
        with patch("valence.mcp.handlers.articles.run_async", side_effect=Exception("Merge error")):
            result = article_merge(
                article_id_a=str(uuid4()),
                article_id_b=str(uuid4()),
            )

        assert result["success"] is False
        assert "Merge error" in result["error"]
