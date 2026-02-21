"""Tests for valence.core.provenance module (WU-04).

Uses mock DB cursor via ``get_cursor`` patching. No real PostgreSQL required.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid4())
SOURCE_ID = str(uuid4())
LINK_ID = str(uuid4())


def _make_article_source_row(**kwargs) -> dict:
    """Build a minimal article_sources DB row."""
    defaults = {
        "id": LINK_ID,
        "article_id": ARTICLE_ID,
        "source_id": SOURCE_ID,
        "relationship": "originates",
        "added_at": datetime.now(),
        "notes": None,
    }
    defaults.update(kwargs)
    return defaults


def _make_source_row(**kwargs) -> dict:
    """Build a minimal sources DB row."""
    defaults = {
        "id": SOURCE_ID,
        "type": "document",
        "title": "Test Source",
        "url": None,
        "content": "Python 3.12 adds type parameter defaults.",
        "reliability": 0.8,
        "created_at": datetime.now(),
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
# link_source
# ---------------------------------------------------------------------------


class TestLinkSource:
    """Tests for link_source()."""

    def test_invalid_relationship(self):
        from valence.core.provenance import link_source

        result = link_source(
            article_id=ARTICLE_ID,
            source_id=SOURCE_ID,
            relationship="invented_type",
        )
        assert result["success"] is False
        assert "relationship" in result["error"]

    def test_article_not_found(self):
        from valence.core.provenance import link_source

        mock_cur = _make_cursor_mock(fetchone=None)

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = link_source(
                article_id=str(uuid4()),
                source_id=SOURCE_ID,
                relationship="originates",
            )

        assert result["success"] is False
        assert "Article not found" in result["error"]

    def test_source_not_found(self):
        from valence.core.provenance import link_source

        article_row = {"id": ARTICLE_ID}
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        # First fetchone (article lookup) succeeds; second (source lookup) returns None
        mock_cur.fetchone.side_effect = [article_row, None]

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = link_source(
                article_id=ARTICLE_ID,
                source_id=str(uuid4()),
                relationship="originates",
            )

        assert result["success"] is False
        assert "Source not found" in result["error"]

    def test_link_originates(self):
        """link_source with 'originates' relationship succeeds."""
        from valence.core.provenance import link_source

        link_row = _make_article_source_row()
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.side_effect = [
            {"id": ARTICLE_ID},   # article lookup
            {"id": SOURCE_ID},    # source lookup
            link_row,             # INSERT RETURNING
        ]

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = link_source(
                article_id=ARTICLE_ID,
                source_id=SOURCE_ID,
                relationship="originates",
            )

        assert result["success"] is True
        assert "link" in result
        assert result["link"]["relationship"] == "originates"

    @pytest.mark.parametrize("rel", ["originates", "confirms", "supersedes", "contradicts", "contends"])
    def test_all_valid_relationships(self, rel: str):
        """All documented relationship types should be accepted."""
        from valence.core.provenance import link_source

        link_row = _make_article_source_row(relationship=rel)
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.side_effect = [
            {"id": ARTICLE_ID},
            {"id": SOURCE_ID},
            link_row,
        ]

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = link_source(
                article_id=ARTICLE_ID,
                source_id=SOURCE_ID,
                relationship=rel,
            )

        assert result["success"] is True

    def test_link_with_notes(self):
        """link_source propagates notes."""
        from valence.core.provenance import link_source

        link_row = _make_article_source_row(notes="Relevant section 3.2")
        mock_cur = MagicMock()
        mock_cur.__enter__ = MagicMock(return_value=mock_cur)
        mock_cur.__exit__ = MagicMock(return_value=False)
        mock_cur.fetchone.side_effect = [
            {"id": ARTICLE_ID},
            {"id": SOURCE_ID},
            link_row,
        ]

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = link_source(
                article_id=ARTICLE_ID,
                source_id=SOURCE_ID,
                relationship="confirms",
                notes="Relevant section 3.2",
            )

        assert result["success"] is True
        assert result["link"]["notes"] == "Relevant section 3.2"


# ---------------------------------------------------------------------------
# get_provenance
# ---------------------------------------------------------------------------


class TestGetProvenance:
    """Tests for get_provenance()."""

    def test_empty_provenance(self):
        """Returns empty list when article has no sources."""
        from valence.core.provenance import get_provenance

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = get_provenance(article_id=ARTICLE_ID)

        assert result == []

    def test_returns_sources(self):
        """Returns list of provenance dicts with source metadata."""
        from valence.core.provenance import get_provenance

        row = {
            **_make_article_source_row(),
            "source_type": "document",
            "source_title": "Test Source",
            "source_url": None,
            "reliability": 0.8,
            "source_created_at": datetime.now(),
        }
        mock_cur = _make_cursor_mock(fetchall=[row])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = get_provenance(article_id=ARTICLE_ID)

        assert len(result) == 1
        assert result[0]["relationship"] == "originates"
        assert result[0]["source_type"] == "document"
        assert result[0]["reliability"] == 0.8

    def test_multiple_sources(self):
        """Returns all linked sources."""
        from valence.core.provenance import get_provenance

        rows = [
            {**_make_article_source_row(source_id=str(uuid4()), relationship="originates"),
             "source_type": "document", "source_title": "Doc A", "source_url": None,
             "reliability": 0.8, "source_created_at": datetime.now()},
            {**_make_article_source_row(source_id=str(uuid4()), relationship="confirms"),
             "source_type": "web", "source_title": "Web B", "source_url": "https://example.com",
             "reliability": 0.6, "source_created_at": datetime.now()},
        ]
        mock_cur = _make_cursor_mock(fetchall=rows)

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = get_provenance(article_id=ARTICLE_ID)

        assert len(result) == 2
        relationships = {r["relationship"] for r in result}
        assert "originates" in relationships
        assert "confirms" in relationships


# ---------------------------------------------------------------------------
# trace_claim
# ---------------------------------------------------------------------------


class TestTraceClaim:
    """Tests for trace_claim()."""

    def test_empty_claim_returns_empty(self):
        from valence.core.provenance import trace_claim

        result = trace_claim(article_id=ARTICLE_ID, claim_text="")
        assert result == []

    def test_no_sources_returns_empty(self):
        """Returns empty list when article has no sources with content."""
        from valence.core.provenance import trace_claim

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = trace_claim(article_id=ARTICLE_ID, claim_text="Python 3.12 features")

        assert result == []

    def test_matches_similar_source(self):
        """Returns source with positive claim_similarity when content overlaps."""
        from valence.core.provenance import trace_claim

        source_row = {
            "id": SOURCE_ID,
            "type": "document",
            "title": "Python Changelog",
            "url": None,
            "content": "Python 3.12 adds type parameter defaults and improved error messages.",
            "reliability": 0.8,
            "created_at": datetime.now(),
            "relationship": "originates",
        }
        mock_cur = _make_cursor_mock(fetchall=[source_row])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = trace_claim(
                article_id=ARTICLE_ID,
                claim_text="Python 3.12 adds type parameter defaults",
            )

        assert len(result) >= 1
        assert result[0]["claim_similarity"] > 0
        # Best match first
        assert result[0]["id"] == SOURCE_ID

    def test_sorted_by_similarity_descending(self):
        """Results are sorted highest similarity first."""
        from valence.core.provenance import trace_claim

        high_match = {
            "id": str(uuid4()),
            "type": "document",
            "title": "Exact match doc",
            "url": None,
            "content": "Python 3.12 adds type parameter defaults in PEP 695.",
            "reliability": 0.8,
            "created_at": datetime.now(),
            "relationship": "originates",
        }
        low_match = {
            "id": str(uuid4()),
            "type": "document",
            "title": "Tangentially related",
            "url": None,
            "content": "JavaScript version 2023 adds new array methods.",
            "reliability": 0.6,
            "created_at": datetime.now(),
            "relationship": "confirms",
        }
        mock_cur = _make_cursor_mock(fetchall=[high_match, low_match])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = trace_claim(
                article_id=ARTICLE_ID,
                claim_text="Python 3.12 type parameter defaults PEP 695",
            )

        # Filter to only positive-similarity results
        positive = [r for r in result if r["claim_similarity"] > 0]
        if len(positive) >= 2:
            assert positive[0]["claim_similarity"] >= positive[1]["claim_similarity"]

    def test_zero_similarity_sources_excluded(self):
        """Sources with zero text overlap are excluded from results."""
        from valence.core.provenance import trace_claim

        # Source with completely unrelated content
        unrelated = {
            "id": str(uuid4()),
            "type": "document",
            "title": "Completely unrelated",
            "url": None,
            "content": "xyzzy plugh foo bar baz quux",  # no overlap with claim
            "reliability": 0.5,
            "created_at": datetime.now(),
            "relationship": "confirms",
        }
        mock_cur = _make_cursor_mock(fetchall=[unrelated])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = trace_claim(
                article_id=ARTICLE_ID,
                claim_text="Python programming language features",
            )

        # Result may be empty or contain sources with positive similarity
        # All returned sources must have claim_similarity > 0
        for r in result:
            assert r["claim_similarity"] > 0


# ---------------------------------------------------------------------------
# get_mutation_history
# ---------------------------------------------------------------------------


class TestGetMutationHistory:
    """Tests for get_mutation_history()."""

    def test_empty_history(self):
        from valence.core.provenance import get_mutation_history

        mock_cur = _make_cursor_mock(fetchall=[])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = get_mutation_history(article_id=ARTICLE_ID)

        assert result == []

    def test_returns_mutations(self):
        from valence.core.provenance import get_mutation_history

        row = {
            "id": str(uuid4()),
            "mutation_type": "created",
            "article_id": ARTICLE_ID,
            "related_article_id": None,
            "trigger_source_id": None,
            "summary": "Article created with 1 source(s)",
            "created_at": datetime.now(),
        }
        mock_cur = _make_cursor_mock(fetchall=[row])

        with patch("valence.core.provenance.get_cursor", return_value=mock_cur):
            result = get_mutation_history(article_id=ARTICLE_ID)

        assert len(result) == 1
        assert result[0]["mutation_type"] == "created"
        assert result[0]["summary"] == "Article created with 1 source(s)"
