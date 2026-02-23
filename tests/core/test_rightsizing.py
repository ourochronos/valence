"""Tests for WU-07: Article right-sizing (split and merge).

Tests split_article and merge_articles from valence.core.articles.
Uses mock DB cursor — no real PostgreSQL required.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, call, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Shared IDs & helpers
# ---------------------------------------------------------------------------

ARTICLE_ID = str(uuid4())
ARTICLE_ID_A = str(uuid4())
ARTICLE_ID_B = str(uuid4())
NEW_ARTICLE_ID = str(uuid4())
SOURCE_ID_1 = str(uuid4())
SOURCE_ID_2 = str(uuid4())
SOURCE_ID_3 = str(uuid4())

NOW = datetime.now()


def _article_row(
    article_id: str = ARTICLE_ID,
    content: str = "First paragraph about Python.\n\nSecond paragraph about Ruby.",
    title: str | None = "Test Article",
    version: int = 1,
    status: str = "active",
    **kwargs,
) -> dict:
    """Build a minimal article DB row."""
    return {
        "id": article_id,
        "content": content,
        "title": title,
        "author_type": kwargs.get("author_type", "system"),
        "pinned": False,
        "size_tokens": 20,
        "compiled_at": NOW,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": kwargs.get("domain_path", ["test"]),
        "valid_from": None,
        "valid_until": None,
        "created_at": NOW,
        "modified_at": NOW,
        "source_id": None,
        "extraction_method": None,
        "extraction_metadata": None,
        "supersedes_id": None,
        "superseded_by_id": None,
        "holder_id": None,
        "version": version,
        "content_hash": "abc123",
        "status": status,
    }


def _source_link_row(source_id: str, relationship: str = "originates", notes: str | None = None) -> dict:
    return {"source_id": source_id, "relationship": relationship, "notes": notes}


def _make_cursor(fetchone_seq=None, fetchall_seq=None):
    """Create a mock cursor with context-manager support and sequenced returns."""
    cur = MagicMock()
    if fetchone_seq is not None:
        cur.fetchone.side_effect = list(fetchone_seq)
    else:
        cur.fetchone.return_value = None
    if fetchall_seq is not None:
        cur.fetchall.side_effect = list(fetchall_seq)
    else:
        cur.fetchall.return_value = []
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=False)
    return cur


# ---------------------------------------------------------------------------
# Helpers: _split_content_at_midpoint
# ---------------------------------------------------------------------------


class TestSplitContentAtMidpoint:
    """Unit tests for the midpoint splitting helper."""

    async def test_splits_at_paragraph_boundary(self):
        from valence.core.articles import _split_content_at_midpoint

        content = "First paragraph about Python.\n\nSecond paragraph about Ruby."
        first, second = _split_content_at_midpoint(content)
        assert len(first) > 0
        assert len(second) > 0
        # The two halves together contain all the original content
        combined = first + "\n\n" + second
        # Both pieces should appear in the combined result
        assert "Python" in combined
        assert "Ruby" in combined

    def test_falls_back_to_hard_midpoint(self):
        from valence.core.articles import _split_content_at_midpoint

        # No paragraph separators — should still produce two parts
        content = "a b c d e f g h i j k l m n o p q r s t"
        first, second = _split_content_at_midpoint(content)
        assert len(first) > 0
        assert len(second) > 0

    def test_preserves_all_content(self):
        from valence.core.articles import _split_content_at_midpoint

        content = "Alpha.\n\nBeta.\n\nGamma.\n\nDelta."
        first, second = _split_content_at_midpoint(content)
        # Every word should appear somewhere in the split output
        assert "Alpha" in first or "Alpha" in second
        assert "Delta" in first or "Delta" in second


# ---------------------------------------------------------------------------
# split_article
# ---------------------------------------------------------------------------


class TestSplitArticle:
    """Tests for await split_article()."""

    def _setup_split_cursor(
        self,
        original_content: str = "First portion content here.\n\nSecond portion content here.",
        sources: list[dict] | None = None,
    ):
        """Build a mock cursor configured for a successful split."""
        if sources is None:
            sources = [
                _source_link_row(SOURCE_ID_1, "originates"),
                _source_link_row(SOURCE_ID_2, "confirms"),
            ]
        original = _article_row(
            article_id=ARTICLE_ID,
            content=original_content,
            version=1,
        )
        updated_original = _article_row(
            article_id=ARTICLE_ID,
            content=original_content.split("\n\n")[0] if "\n\n" in original_content else original_content[:20],
            version=2,
        )
        new_article = _article_row(
            article_id=NEW_ARTICLE_ID,
            content="new content",
            title="Test Article (part 2)",
            version=1,
        )
        return _make_cursor(
            fetchone_seq=[original, updated_original, new_article],
            fetchall_seq=[sources],
        )

    async def test_split_returns_two_articles(self):
        from valence.core.articles import split_article

        mock_cur = self._setup_split_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            split_result = await split_article(ARTICLE_ID)
            original = split_result.data["original"]
            new_art = split_result.data["new"]

        assert original is not None
        assert new_art is not None

    async def test_split_original_retains_id(self):
        from valence.core.articles import split_article

        mock_cur = self._setup_split_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            split_result = await split_article(ARTICLE_ID)
            original = split_result.data["original"]

        assert original["id"] == ARTICLE_ID

    async def test_split_new_article_has_different_id(self):
        from valence.core.articles import split_article

        mock_cur = self._setup_split_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            split_result = await split_article(ARTICLE_ID)
            original = split_result.data["original"]
            new_art = split_result.data["new"]

        assert new_art["id"] != original["id"]
        # The new ID comes from the DB insert; mock returns NEW_ARTICLE_ID
        assert new_art["id"] == NEW_ARTICLE_ID

    async def test_split_sources_copied_to_both_articles(self):
        from valence.core.articles import split_article

        mock_cur = self._setup_split_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await split_article(ARTICLE_ID)

        calls_str = str(mock_cur.execute.call_args_list)
        # Both article_sources inserts should reference SOURCE_ID_1 and SOURCE_ID_2
        assert SOURCE_ID_1 in calls_str
        assert SOURCE_ID_2 in calls_str
        # New article ID should appear in article_sources inserts
        assert NEW_ARTICLE_ID in calls_str

    async def test_split_records_mutations_on_both_articles(self):
        from valence.core.articles import split_article

        mock_cur = self._setup_split_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await split_article(ARTICLE_ID)

        calls_str = str(mock_cur.execute.call_args_list)
        assert "article_mutations" in calls_str
        assert "split" in calls_str
        # Both the original and new article IDs appear in mutation inserts
        assert ARTICLE_ID in calls_str
        assert NEW_ARTICLE_ID in calls_str

    async def test_split_raises_on_missing_article(self):
        from valence.core.articles import split_article

        mock_cur = _make_cursor(fetchone_seq=[None])  # article not found
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            result = await split_article(str(uuid4()))
            assert result.success is False
            assert "not found" in result.error.lower()

    async def test_split_raises_on_too_short_content(self):
        from valence.core.articles import split_article

        short_article = _article_row(content="Too short.")
        mock_cur = _make_cursor(fetchone_seq=[short_article])
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            result = await split_article(ARTICLE_ID)
            assert result.success is False
            assert "too short" in result.error.lower()

    async def test_split_sources_include_all_original_sources(self):
        """Both articles should have ALL original sources — not just a subset."""
        from valence.core.articles import split_article

        three_sources = [
            _source_link_row(SOURCE_ID_1, "originates"),
            _source_link_row(SOURCE_ID_2, "confirms"),
            _source_link_row(SOURCE_ID_3, "contends"),
        ]
        mock_cur = self._setup_split_cursor(sources=three_sources)
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await split_article(ARTICLE_ID)

        calls_str = str(mock_cur.execute.call_args_list)
        # All three sources appear in the SQL calls
        assert SOURCE_ID_1 in calls_str
        assert SOURCE_ID_2 in calls_str
        assert SOURCE_ID_3 in calls_str
        # Specifically, the new article should have inserts for all three
        # Count occurrences of NEW_ARTICLE_ID in article_sources inserts
        article_sources_calls = [str(c) for c in mock_cur.execute.call_args_list if "article_sources" in str(c)]
        new_art_source_calls = [c for c in article_sources_calls if NEW_ARTICLE_ID in c]
        assert len(new_art_source_calls) == 3  # one insert per source

    async def test_split_mutation_has_cross_references(self):
        """Mutation for original should reference new article ID and vice versa."""
        from valence.core.articles import split_article

        mock_cur = self._setup_split_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await split_article(ARTICLE_ID)

        mutation_calls = [str(c) for c in mock_cur.execute.call_args_list if "article_mutations" in str(c) and "split" in str(c)]
        # Should be exactly 2 mutation inserts (one for each article)
        assert len(mutation_calls) == 2
        # Both should reference each other
        combined = " ".join(mutation_calls)
        assert ARTICLE_ID in combined
        assert NEW_ARTICLE_ID in combined

    async def test_split_no_sources_still_works(self):
        """split_article succeeds even if the original has no sources."""
        from valence.core.articles import split_article

        original = _article_row(
            content="First part of a fairly long article.\n\nSecond part of a fairly long article.",
        )
        updated_original = _article_row(article_id=ARTICLE_ID, version=2)
        new_article = _article_row(article_id=NEW_ARTICLE_ID, version=1)
        mock_cur = _make_cursor(
            fetchone_seq=[original, updated_original, new_article],
            fetchall_seq=[[]],  # no sources
        )
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            split_result = await split_article(ARTICLE_ID)
            original_out = split_result.data["original"]
            new_art = split_result.data["new"]

        assert original_out["id"] == ARTICLE_ID
        assert new_art["id"] == NEW_ARTICLE_ID


# ---------------------------------------------------------------------------
# merge_articles
# ---------------------------------------------------------------------------


class TestMergeArticles:
    """Tests for await merge_articles()."""

    MERGED_ID = str(uuid4())

    def _setup_merge_cursor(
        self,
        sources_a: list[dict] | None = None,
        sources_b: list[dict] | None = None,
    ):
        """Build a mock cursor configured for a successful merge."""
        if sources_a is None:
            sources_a = [_source_link_row(SOURCE_ID_1, "originates")]
        if sources_b is None:
            sources_b = [_source_link_row(SOURCE_ID_2, "confirms")]

        article_a = _article_row(
            article_id=ARTICLE_ID_A,
            content="Content from article A.",
            title="Article A",
        )
        article_b = _article_row(
            article_id=ARTICLE_ID_B,
            content="Content from article B.",
            title="Article B",
        )
        merged_article = _article_row(
            article_id=self.MERGED_ID,
            content="Content from article A.\n\n---\n\nContent from article B.",
            title="Article A + Article B",
        )
        return _make_cursor(
            fetchone_seq=[article_a, article_b, merged_article],
            fetchall_seq=[sources_a, sources_b],
        )

    async def test_merge_returns_new_article(self):
        from valence.core.articles import merge_articles

        mock_cur = self._setup_merge_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            _mr = await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)
            merge_result = _mr.data
            merged = merge_result

        assert merged is not None
        assert merged["id"] == self.MERGED_ID

    async def test_merge_new_article_has_new_id(self):
        from valence.core.articles import merge_articles

        mock_cur = self._setup_merge_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            _mr = await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)
            merge_result = _mr.data
            merged = merge_result

        assert merged["id"] != ARTICLE_ID_A
        assert merged["id"] != ARTICLE_ID_B

    async def test_merge_originals_are_archived(self):
        from valence.core.articles import merge_articles

        mock_cur = self._setup_merge_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        calls_str = str(mock_cur.execute.call_args_list)
        # Both originals should be archived
        assert "archived" in calls_str
        assert ARTICLE_ID_A in calls_str
        assert ARTICLE_ID_B in calls_str

    async def test_merge_combined_sources(self):
        """Merged article has sources from both originals."""
        from valence.core.articles import merge_articles

        mock_cur = self._setup_merge_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        calls_str = str(mock_cur.execute.call_args_list)
        assert SOURCE_ID_1 in calls_str
        assert SOURCE_ID_2 in calls_str

    async def test_merge_deduplicates_sources(self):
        """If both articles share a source, it appears only once in the merged article."""
        from valence.core.articles import merge_articles

        shared_source = SOURCE_ID_1
        sources_a = [
            _source_link_row(shared_source, "originates"),
            _source_link_row(SOURCE_ID_2, "confirms"),
        ]
        sources_b = [
            _source_link_row(shared_source, "originates"),  # duplicate
            _source_link_row(SOURCE_ID_3, "contends"),
        ]
        mock_cur = self._setup_merge_cursor(sources_a=sources_a, sources_b=sources_b)
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        # Count article_sources INSERT calls for merged article (self.MERGED_ID)
        source_insert_calls = [str(c) for c in mock_cur.execute.call_args_list if "article_sources" in str(c) and self.MERGED_ID in str(c)]
        # 3 unique (source_id, relationship) combos: shared+originates, s2+confirms, s3+contends
        assert len(source_insert_calls) == 3

    async def test_merge_records_mutations_on_all_three(self):
        """Merged mutation recorded for both originals and the new article."""
        from valence.core.articles import merge_articles

        mock_cur = self._setup_merge_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        mutation_calls = [str(c) for c in mock_cur.execute.call_args_list if "article_mutations" in str(c) and "merged" in str(c)]
        # 3 mutations: one for A, one for B, one for the merged article
        assert len(mutation_calls) == 3
        combined = " ".join(mutation_calls)
        assert ARTICLE_ID_A in combined
        assert ARTICLE_ID_B in combined
        assert self.MERGED_ID in combined

    async def test_merge_raises_on_missing_article_a(self):
        from valence.core.articles import merge_articles

        mock_cur = _make_cursor(fetchone_seq=[None])
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            result = await merge_articles(str(uuid4()), ARTICLE_ID_B)
            assert result.success is False
            assert "not found" in result.error.lower()

    async def test_merge_raises_on_missing_article_b(self):
        from valence.core.articles import merge_articles

        article_a = _article_row(article_id=ARTICLE_ID_A)
        mock_cur = _make_cursor(fetchone_seq=[article_a, None])
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            result = await merge_articles(ARTICLE_ID_A, str(uuid4()))
            assert result.success is False
            assert "not found" in result.error.lower()

    async def test_merge_title_combines_both(self):
        """Merged article title should reflect both source titles."""
        from valence.core.articles import merge_articles

        mock_cur = self._setup_merge_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            # The title is built before INSERT; check that it's passed to the DB
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        # Find the INSERT INTO articles call and check title argument
        insert_calls = [c for c in mock_cur.execute.call_args_list if "INSERT INTO articles" in str(c)]
        assert len(insert_calls) == 1
        # The args should contain "Article A + Article B"
        insert_args = str(insert_calls[0])
        assert "Article A + Article B" in insert_args

    async def test_merge_content_contains_both(self):
        """Merged article content is a concatenation of both source article contents."""
        from valence.core.articles import merge_articles

        mock_cur = self._setup_merge_cursor()
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        # The INSERT call's first positional arg after the SQL is the content
        insert_calls = [c for c in mock_cur.execute.call_args_list if "INSERT INTO articles" in str(c)]
        insert_args = str(insert_calls[0])
        assert "Content from article A" in insert_args
        assert "Content from article B" in insert_args

    async def test_merge_union_of_sources(self):
        """Union of sources — all unique (source_id, relationship) pairs preserved."""
        from valence.core.articles import merge_articles

        sources_a = [
            _source_link_row(SOURCE_ID_1, "originates"),
            _source_link_row(SOURCE_ID_2, "confirms"),
        ]
        sources_b = [
            _source_link_row(SOURCE_ID_2, "confirms"),  # duplicate — deduplicated
            _source_link_row(SOURCE_ID_3, "contends"),
        ]
        mock_cur = self._setup_merge_cursor(sources_a=sources_a, sources_b=sources_b)
        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)

        source_inserts = [str(c) for c in mock_cur.execute.call_args_list if "article_sources" in str(c) and self.MERGED_ID in str(c)]
        # 3 unique: (s1, originates), (s2, confirms), (s3, contends)
        assert len(source_inserts) == 3


# ---------------------------------------------------------------------------
# Provenance chain integrity
# ---------------------------------------------------------------------------


class TestProvenanceChainIntegrity:
    """Verify that split and merge preserve provenance per WU-07 spec (C6)."""

    async def test_split_both_articles_have_all_original_sources_via_provenance(self):
        """After split, get_provenance on each article should return the original sources."""
        from valence.core.articles import split_article
        from valence.core.provenance import get_provenance

        # --- Set up split ---
        sources = [
            _source_link_row(SOURCE_ID_1, "originates"),
            _source_link_row(SOURCE_ID_2, "confirms"),
        ]
        original = _article_row(
            content="First part.\n\nSecond part.",
        )
        updated_orig = _article_row(article_id=ARTICLE_ID, version=2)
        new_art_row = _article_row(article_id=NEW_ARTICLE_ID, version=1)
        split_cur = _make_cursor(
            fetchone_seq=[original, updated_orig, new_art_row],
            fetchall_seq=[sources],
        )
        with (
            patch("valence.core.articles.get_cursor", return_value=split_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            split_res = await split_article(ARTICLE_ID)
            orig_result = split_res.data["original"]
            new_result = split_res.data["new"]

        # --- Verify provenance for the original (now updated) article ---
        prov_row_1 = {
            "link_id": str(uuid4()),
            "article_id": ARTICLE_ID,
            "source_id": SOURCE_ID_1,
            "relationship": "originates",
            "added_at": NOW,
            "notes": None,
            "source_type": "document",
            "source_title": "Doc 1",
            "source_url": None,
            "reliability": 0.8,
            "source_created_at": NOW,
        }
        prov_row_2 = {**prov_row_1, "source_id": SOURCE_ID_2, "relationship": "confirms"}
        prov_cur = _make_cursor(fetchall_seq=[[prov_row_1, prov_row_2]])
        with patch("valence.core.provenance.get_cursor", return_value=prov_cur):
            prov_result = await get_provenance(ARTICLE_ID)
            prov = prov_result.data

        assert len(prov) == 2
        source_ids = {p["source_id"] for p in prov}
        assert SOURCE_ID_1 in source_ids
        assert SOURCE_ID_2 in source_ids

    async def test_merge_provenance_shows_all_original_sources(self):
        """After merge, get_provenance on merged article returns all sources from both originals."""
        from valence.core.articles import merge_articles
        from valence.core.provenance import get_provenance

        MERGED_ID = str(uuid4())
        sources_a = [_source_link_row(SOURCE_ID_1, "originates")]
        sources_b = [_source_link_row(SOURCE_ID_2, "confirms")]

        article_a = _article_row(article_id=ARTICLE_ID_A, content="A content.", title="A")
        article_b = _article_row(article_id=ARTICLE_ID_B, content="B content.", title="B")
        merged_row = _article_row(article_id=MERGED_ID, content="A content.\n\n---\n\nB content.")

        merge_cur = _make_cursor(
            fetchone_seq=[article_a, article_b, merged_row],
            fetchall_seq=[sources_a, sources_b],
        )
        with (
            patch("valence.core.articles.get_cursor", return_value=merge_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
        ):
            _mr = await merge_articles(ARTICLE_ID_A, ARTICLE_ID_B)
            merge_result = _mr.data
            merged = merge_result

        # Simulate get_provenance returning both sources for merged article
        prov_row_1 = {
            "link_id": str(uuid4()),
            "article_id": MERGED_ID,
            "source_id": SOURCE_ID_1,
            "relationship": "originates",
            "added_at": NOW,
            "notes": None,
            "source_type": "document",
            "source_title": "Doc 1",
            "source_url": None,
            "reliability": 0.8,
            "source_created_at": NOW,
        }
        prov_row_2 = {**prov_row_1, "source_id": SOURCE_ID_2, "relationship": "confirms"}
        prov_cur = _make_cursor(fetchall_seq=[[prov_row_1, prov_row_2]])
        with patch("valence.core.provenance.get_cursor", return_value=prov_cur):
            prov_result = await get_provenance(MERGED_ID)
            prov = prov_result.data

        assert len(prov) == 2
        source_ids_in_prov = {p["source_id"] for p in prov}
        assert SOURCE_ID_1 in source_ids_in_prov
        assert SOURCE_ID_2 in source_ids_in_prov


# ---------------------------------------------------------------------------
# mutation_queue integration: process_mutation_queue handles 'split'
# ---------------------------------------------------------------------------


class TestMutationQueueSplitIntegration:
    """Verify that process_mutation_queue dispatches 'split' to split_article."""

    async def test_queue_dispatches_split_to_split_article(self):
        """process_mutation_queue with a 'split' item calls split_article."""
        import valence.core.compilation as comp_mod
        from valence.core.compilation import process_mutation_queue

        queue_item = {
            "id": str(uuid4()),
            "operation": "split",
            "article_id": ARTICLE_ID,
            "payload": {"reason": "exceeds_max_tokens", "token_count": 5000},
        }
        # Mock the DB queue claim to return our item
        queue_cur = MagicMock()
        queue_cur.fetchall.return_value = [queue_item]
        queue_cur.__enter__ = MagicMock(return_value=queue_cur)
        queue_cur.__exit__ = MagicMock(return_value=False)

        updated = _article_row(article_id=ARTICLE_ID, version=2)
        new_art = _article_row(article_id=NEW_ARTICLE_ID, version=1)

        with (
            patch.object(comp_mod, "get_cursor", return_value=queue_cur),
            patch(
                "valence.core.articles.split_article",
                return_value=(updated, new_art),
            ) as mock_split,
            patch("valence.core.compilation._set_queue_item_status") as mock_status,
        ):
            count = await process_mutation_queue(batch_size=1)

        # split_article was called with the article ID
        mock_split.assert_called_once_with(ARTICLE_ID)
        # Item was marked completed
        mock_status.assert_called_once()
        status_call_args = mock_status.call_args[0]
        assert status_call_args[1] == "completed"
        assert count.data == 1

    async def test_queue_dispatches_merge_candidate(self):
        """process_mutation_queue with a 'merge_candidate' item calls merge_articles."""
        import valence.core.compilation as comp_mod
        from valence.core.compilation import process_mutation_queue

        CANDIDATE_ID = str(uuid4())
        queue_item = {
            "id": str(uuid4()),
            "operation": "merge_candidate",
            "article_id": ARTICLE_ID_A,
            "payload": {"candidate_article_id": CANDIDATE_ID},
        }
        queue_cur = MagicMock()
        queue_cur.fetchall.return_value = [queue_item]
        queue_cur.__enter__ = MagicMock(return_value=queue_cur)
        queue_cur.__exit__ = MagicMock(return_value=False)

        merged_art = _article_row(article_id=str(uuid4()))

        with (
            patch.object(comp_mod, "get_cursor", return_value=queue_cur),
            patch(
                "valence.core.articles.merge_articles",
                return_value=merged_art,
            ) as mock_merge,
            patch("valence.core.compilation._set_queue_item_status") as mock_status,
        ):
            count = await process_mutation_queue(batch_size=1)

        mock_merge.assert_called_once_with(ARTICLE_ID_A, CANDIDATE_ID)
        status_call_args = mock_status.call_args[0]
        assert status_call_args[1] == "completed"
        assert count.data == 1
