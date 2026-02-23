"""Tests for valence.core.articles module (WU-04).

Updated for WU-14: all public functions are now async and return ValenceResponse.

Uses mock DB cursor via ``get_cursor`` patching. No real PostgreSQL required.
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid4())
SOURCE_ID = str(uuid4())


def _make_article_row(**kwargs) -> dict:
    """Build a minimal article DB row dict."""
    now = datetime.now()
    defaults = {
        "id": ARTICLE_ID,
        "content": "Test article content about Python.",
        "title": "Test Article",
        "author_type": "system",
        "pinned": False,
        "size_tokens": 7,
        "compiled_at": now,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": ["test"],
        "valid_from": None,
        "valid_until": None,
        "created_at": now,
        "modified_at": now,
        "source_id": None,
        "extraction_method": None,
        "extraction_metadata": None,
        "supersedes_id": None,
        "superseded_by_id": None,
        "holder_id": None,
        "version": 1,
        "content_hash": "abc123",
        "status": "active",
    }
    defaults.update(kwargs)
    return defaults


def _make_cursor_mock(fetchone=None, fetchall=None):
    """Create a mock cursor context manager."""
    mock_cur = MagicMock()
    mock_cur.fetchone.return_value = fetchone
    mock_cur.fetchall.return_value = fetchall or []
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    return mock_cur


# ---------------------------------------------------------------------------
# create_article
# ---------------------------------------------------------------------------


class TestCreateArticle:
    """Tests for create_article() — now async, returns ValenceResponse."""

    async def test_missing_content(self):
        from valence.core.articles import create_article

        result = await create_article(content="")
        assert result.success is False
        assert "content" in result.error

    async def test_invalid_author_type(self):
        from valence.core.articles import create_article

        result = await create_article(content="some content", author_type="hacker")
        assert result.success is False
        assert "author_type" in result.error

    async def test_create_minimal(self):
        """create_article with only content succeeds."""
        from valence.core.articles import create_article

        article_row = _make_article_row()
        mock_cur = _make_cursor_mock(fetchone=article_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            result = await create_article(content="Test article content about Python.")

        assert result.success is True
        assert result.data is not None
        assert result.data["content"] == "Test article content about Python."

    async def test_create_with_sources(self):
        """create_article links sources with 'originates' relationship."""
        from valence.core.articles import create_article

        article_row = _make_article_row()
        mock_cur = _make_cursor_mock(fetchone=article_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            result = await create_article(
                content="Test article content.",
                source_ids=[SOURCE_ID],
                author_type="system",
                domain_path=["test"],
            )

        assert result.success is True
        # Confirm article_sources INSERT was called
        calls_str = str(mock_cur.execute.call_args_list)
        assert "article_sources" in calls_str

    async def test_create_records_mutation(self):
        """create_article records a 'created' article_mutations row."""
        from valence.core.articles import create_article

        article_row = _make_article_row()
        mock_cur = _make_cursor_mock(fetchone=article_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            await create_article(content="Content here.")

        calls_str = str(mock_cur.execute.call_args_list)
        assert "article_mutations" in calls_str
        assert "created" in calls_str

    async def test_create_with_title(self):
        """create_article stores title."""
        from valence.core.articles import create_article

        article_row = _make_article_row(title="My Title")
        mock_cur = _make_cursor_mock(fetchone=article_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            result = await create_article(content="Content here.", title="My Title")

        assert result.success is True
        assert result.data["title"] == "My Title"

    def test_create_computes_token_count(self):
        """_count_tokens returns a reasonable integer."""
        from valence.core.articles import _count_tokens

        count = _count_tokens("Hello world this is a test sentence.")
        assert isinstance(count, int)
        assert count > 0


# ---------------------------------------------------------------------------
# get_article
# ---------------------------------------------------------------------------


class TestGetArticle:
    """Tests for get_article() — now async, returns ValenceResponse."""

    async def test_not_found(self):
        from valence.core.articles import get_article

        mock_cur = _make_cursor_mock(fetchone=None)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await get_article(article_id=str(uuid4()))

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_found(self):
        from valence.core.articles import get_article

        article_row = _make_article_row()
        mock_cur = _make_cursor_mock(fetchone=article_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await get_article(article_id=ARTICLE_ID)

        assert result.success is True
        assert result.data["id"] == ARTICLE_ID

    async def test_include_provenance(self):
        """get_article with include_provenance=True attaches provenance list."""
        from valence.core.articles import get_article

        article_row = _make_article_row()
        provenance_row = {
            "id": str(uuid4()),
            "source_id": SOURCE_ID,
            "relationship": "originates",
            "added_at": datetime.now(),
            "notes": None,
            "source_type": "document",
            "source_title": "Ref doc",
            "source_url": None,
            "reliability": 0.8,
        }
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        # First fetchone = article row; subsequent fetchall = provenance
        mock_cur.fetchone.return_value = article_row
        mock_cur.fetchall.return_value = [provenance_row]

        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await get_article(article_id=ARTICLE_ID, include_provenance=True)

        assert result.success is True
        assert "provenance" in result.data
        assert len(result.data["provenance"]) == 1
        assert result.data["provenance"][0]["relationship"] == "originates"


# ---------------------------------------------------------------------------
# update_article
# ---------------------------------------------------------------------------


class TestUpdateArticle:
    """Tests for update_article() — now async, returns ValenceResponse."""

    async def test_missing_content(self):
        from valence.core.articles import update_article

        result = await update_article(article_id=ARTICLE_ID, content="")
        assert result.success is False

    async def test_not_found(self):
        from valence.core.articles import update_article

        mock_cur = _make_cursor_mock(fetchone=None)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            result = await update_article(article_id=str(uuid4()), content="New content.")

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_update_increments_version(self):
        """update_article returns the updated article with incremented version."""
        from valence.core.articles import update_article

        updated_row = _make_article_row(version=2, content="New content.")
        mock_cur = _make_cursor_mock(fetchone=updated_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            result = await update_article(article_id=ARTICLE_ID, content="New content.")

        assert result.success is True
        assert result.data["version"] == 2

    async def test_update_records_mutation(self):
        """update_article inserts an 'updated' article_mutations row."""
        from valence.core.articles import update_article

        updated_row = _make_article_row(version=2, content="New content.")
        mock_cur = _make_cursor_mock(fetchone=updated_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            await update_article(article_id=ARTICLE_ID, content="New content.")

        calls_str = str(mock_cur.execute.call_args_list)
        assert "article_mutations" in calls_str
        assert "updated" in calls_str

    async def test_update_with_source_links_source(self):
        """update_article with source_id links source to article_sources."""
        from valence.core.articles import update_article

        updated_row = _make_article_row(version=2)
        mock_cur = _make_cursor_mock(fetchone=updated_row)

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            await update_article(article_id=ARTICLE_ID, content="Updated.", source_id=SOURCE_ID)

        calls_str = str(mock_cur.execute.call_args_list)
        assert "article_sources" in calls_str


# ---------------------------------------------------------------------------
# search_articles
# ---------------------------------------------------------------------------


class TestSearchArticles:
    """Tests for search_articles() — now async, returns ValenceResponse."""

    async def test_empty_query_returns_empty(self):
        from valence.core.articles import search_articles

        result = await search_articles(query="")
        assert result.success is True
        assert result.data == []

    async def test_returns_list(self):
        from valence.core.articles import search_articles

        article_row = _make_article_row()
        article_row["relevance"] = 1.0
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = [article_row]

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            result = await search_articles(query="python")

        assert result.success is True
        assert isinstance(result.data, list)
        assert len(result.data) >= 1
        assert result.data[0]["content"] == "Test article content about Python."

    async def test_no_results(self):
        from valence.core.articles import search_articles

        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchall.return_value = []

        with patch("valence.core.articles.get_cursor", return_value=mock_cur), patch("valence.core.articles._compute_embedding", return_value=None):
            result = await search_articles(query="totally obscure xyz")

        assert result.success is True
        assert result.data == []


# ---------------------------------------------------------------------------
# Token count helper
# ---------------------------------------------------------------------------


class TestCountTokens:
    """Tests for _count_tokens helper."""

    def test_basic(self):
        from valence.core.articles import _count_tokens

        count = _count_tokens("one two three")
        # 3 words * 1.3 ≈ 3-4
        assert 3 <= count <= 5

    def test_empty(self):
        from valence.core.articles import _count_tokens

        count = _count_tokens("")
        assert count >= 1  # minimum 1

    def test_long(self):
        from valence.core.articles import _count_tokens

        words = " ".join(["word"] * 100)
        count = _count_tokens(words)
        assert count > 100
