"""Tests for topic-aware article splitting (WU-07 / issue #415).

Tests the new split_article() implementation with LLM-based topic detection
and fallback to mechanical splitting when LLM is unavailable.
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
SOURCE_ID_1 = str(uuid4())
SOURCE_ID_2 = str(uuid4())
PART_A_ID = str(uuid4())
PART_B_ID = str(uuid4())


def _make_article_row(**kwargs) -> dict:
    """Build a minimal article DB row dict."""
    now = datetime.now()
    defaults = {
        "id": ARTICLE_ID,
        "content": "First topic about Python.\n\nSecond topic about JavaScript.",
        "title": "Programming Languages",
        "author_type": "system",
        "pinned": False,
        "size_tokens": 12,
        "compiled_at": now,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": ["programming"],
        "version": 1,
        "content_hash": "abc123",
        "status": "active",
        "created_at": now,
        "modified_at": now,
        "superseded_by_id": None,
    }
    defaults.update(kwargs)
    return defaults


def _make_source_row(source_id=SOURCE_ID_1, **kwargs) -> dict:
    """Build a minimal article_sources relationship row."""
    defaults = {
        "source_id": source_id,
        "relationship": "originates",
        "notes": None,
    }
    defaults.update(kwargs)
    return defaults


def _make_split_response() -> str:
    """Mock LLM response for TASK_SPLIT."""
    return json.dumps(
        {
            "split_index": 30,  # Split after "Python."
            "part_a_title": "Python Overview",
            "part_b_title": "JavaScript Fundamentals",
            "reasoning": "Split at natural topic boundary between Python and JavaScript sections",
        }
    )


def _make_cursor_mock(fetchone_seq=None, fetchall_seq=None):
    """Create a mock cursor with context manager support."""
    mock_cur = MagicMock()
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)

    if fetchone_seq is not None:
        mock_cur.fetchone.side_effect = list(fetchone_seq)
    else:
        mock_cur.fetchone.return_value = None

    if fetchall_seq is not None:
        mock_cur.fetchall.side_effect = list(fetchall_seq)
    else:
        mock_cur.fetchall.return_value = []

    return mock_cur


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSplitArticle:
    """Tests for topic-aware split_article()."""

    async def test_article_not_found(self):
        """split_article returns error when article doesn't exist."""
        from valence.core.articles import split_article

        mock_cur = _make_cursor_mock(fetchone_seq=[None])

        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await split_article(article_id=str(uuid4()))

        assert result.success is False
        assert "not found" in result.error.lower()

    async def test_content_too_short(self):
        """split_article returns error when content < 4 words."""
        from valence.core.articles import split_article

        short_article = _make_article_row(content="Too short")
        mock_cur = _make_cursor_mock(fetchone_seq=[short_article], fetchall_seq=[[]])

        with patch("valence.core.articles.get_cursor", return_value=mock_cur):
            result = await split_article(article_id=ARTICLE_ID)

        assert result.success is False
        assert "too short" in result.error.lower()

    async def test_split_with_llm_success(self):
        """split_article uses LLM to find topic boundary and generate titles."""
        from valence.core.articles import split_article
        from valence.core.inference import InferenceResult

        original = _make_article_row()
        sources = [_make_source_row(SOURCE_ID_1), _make_source_row(SOURCE_ID_2)]

        # Three fetchone calls: original, part_a insert, part_b insert
        part_a_row = _make_article_row(
            id=PART_A_ID,
            content="First topic about Python.",
            title="Python Overview",
        )
        part_b_row = _make_article_row(
            id=PART_B_ID,
            content="Second topic about JavaScript.",
            title="JavaScript Fundamentals",
        )

        mock_cur = _make_cursor_mock(
            fetchone_seq=[original, part_a_row, part_b_row],
            fetchall_seq=[sources],
        )

        # Mock the inference provider to return a successful split
        mock_result = InferenceResult.success(
            content=_make_split_response(),
            task_type="split",
            parsed={
                "split_index": 30,
                "part_a_title": "Python Overview",
                "part_b_title": "JavaScript Fundamentals",
                "reasoning": "Split at natural topic boundary",
            },
        )

        async def mock_infer(*args, **kwargs):
            return mock_result

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
            patch("valence.core.inference.provider") as mock_provider,
        ):
            mock_provider.infer = mock_infer
            result = await split_article(article_id=ARTICLE_ID)

        assert result.success is True
        assert result.degraded is False
        assert "part_a" in result.data
        assert "part_b" in result.data
        assert result.data["part_a"]["title"] == "Python Overview"
        assert result.data["part_b"]["title"] == "JavaScript Fundamentals"

        # Verify original article was archived
        calls_str = str(mock_cur.execute.call_args_list)
        assert "status = 'archived'" in calls_str

    async def test_split_fallback_when_llm_unavailable(self):
        """split_article falls back to mechanical split when LLM unavailable."""
        from valence.core.articles import split_article
        from valence.core.inference import InferenceResult

        original = _make_article_row()
        sources = [_make_source_row()]

        part_a_row = _make_article_row(
            id=PART_A_ID,
            content="First topic about Python.",
            title="Programming Languages (part 1)",
        )
        part_b_row = _make_article_row(
            id=PART_B_ID,
            content="Second topic about JavaScript.",
            title="Programming Languages (part 2)",
        )

        mock_cur = _make_cursor_mock(
            fetchone_seq=[original, part_a_row, part_b_row],
            fetchall_seq=[sources],
        )

        # Mock degraded inference response
        mock_result = InferenceResult.degraded_result(
            task_type="split",
            error="LLM unavailable",
        )

        async def mock_infer(*args, **kwargs):
            return mock_result

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
            patch("valence.core.inference.provider") as mock_provider,
        ):
            mock_provider.infer = mock_infer
            result = await split_article(article_id=ARTICLE_ID)

        assert result.success is True
        assert result.degraded is True
        assert "part_a" in result.data
        assert "part_b" in result.data
        # Fallback titles should include "(part 1)" / "(part 2)"
        assert "(part 1)" in result.data["part_a"]["title"] or "(part 2)" in result.data["part_b"]["title"]

    async def test_split_preserves_sources_on_both_parts(self):
        """split_article links all sources to both new articles."""
        from valence.core.articles import split_article
        from valence.core.inference import InferenceResult

        original = _make_article_row()
        sources = [
            _make_source_row(SOURCE_ID_1, relationship="originates"),
            _make_source_row(SOURCE_ID_2, relationship="confirms"),
        ]

        part_a_row = _make_article_row(id=PART_A_ID, title="Part A")
        part_b_row = _make_article_row(id=PART_B_ID, title="Part B")

        mock_cur = _make_cursor_mock(
            fetchone_seq=[original, part_a_row, part_b_row],
            fetchall_seq=[sources],
        )

        mock_result = InferenceResult.success(
            content=_make_split_response(),
            task_type="split",
            parsed={
                "split_index": 30,
                "part_a_title": "Part A",
                "part_b_title": "Part B",
                "reasoning": "Test split",
            },
        )

        async def mock_infer(*args, **kwargs):
            return mock_result

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
            patch("valence.core.inference.provider") as mock_provider,
        ):
            mock_provider.infer = mock_infer
            await split_article(article_id=ARTICLE_ID)

        # Check that article_sources INSERT was called for both sources × both parts = 4 times
        calls_str = str(mock_cur.execute.call_args_list)
        source_inserts = calls_str.count("INSERT INTO article_sources")
        assert source_inserts == 4  # 2 sources × 2 new articles

    async def test_split_records_mutations(self):
        """split_article records 'split' mutations on all three articles."""
        from valence.core.articles import split_article
        from valence.core.inference import InferenceResult

        original = _make_article_row()
        sources = [_make_source_row()]

        part_a_row = _make_article_row(id=PART_A_ID, title="Part A")
        part_b_row = _make_article_row(id=PART_B_ID, title="Part B")

        mock_cur = _make_cursor_mock(
            fetchone_seq=[original, part_a_row, part_b_row],
            fetchall_seq=[sources],
        )

        mock_result = InferenceResult.success(
            content=_make_split_response(),
            task_type="split",
            parsed={
                "split_index": 30,
                "part_a_title": "Part A",
                "part_b_title": "Part B",
                "reasoning": "Test split",
            },
        )

        async def mock_infer(*args, **kwargs):
            return mock_result

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
            patch("valence.core.inference.provider") as mock_provider,
        ):
            mock_provider.infer = mock_infer
            await split_article(article_id=ARTICLE_ID)

        # Check that article_mutations INSERT was called 3 times (original + part_a + part_b)
        calls_str = str(mock_cur.execute.call_args_list)
        mutation_inserts = calls_str.count("INSERT INTO article_mutations")
        assert mutation_inserts == 3

    async def test_split_handles_invalid_split_index(self):
        """split_article recovers when LLM returns invalid split_index."""
        from valence.core.articles import split_article
        from valence.core.inference import InferenceResult

        long_content = " ".join(["word"] * 50) + ".\n\n" + " ".join(["word"] * 50)
        original = _make_article_row(content=long_content)
        sources = [_make_source_row()]

        part_a_row = _make_article_row(id=PART_A_ID, title="Part A", content=long_content[:50])
        part_b_row = _make_article_row(id=PART_B_ID, title="Part B", content=long_content[50:])

        mock_cur = _make_cursor_mock(
            fetchone_seq=[original, part_a_row, part_b_row],
            fetchall_seq=[sources],
        )

        # Return invalid split_index (negative)
        mock_result = InferenceResult.success(
            content=json.dumps(
                {
                    "split_index": -10,
                    "part_a_title": "Part A",
                    "part_b_title": "Part B",
                    "reasoning": "Invalid index",
                }
            ),
            task_type="split",
            parsed={
                "split_index": -10,
                "part_a_title": "Part A",
                "part_b_title": "Part B",
                "reasoning": "Invalid index",
            },
        )

        async def mock_infer(*args, **kwargs):
            return mock_result

        with (
            patch("valence.core.articles.get_cursor", return_value=mock_cur),
            patch("valence.core.articles._compute_embedding", return_value=None),
            patch("valence.core.inference.provider") as mock_provider,
        ):
            mock_provider.infer = mock_infer
            result = await split_article(article_id=ARTICLE_ID)

        # Should still succeed by using midpoint fallback
        assert result.success is True


class TestBuildSplitPrompt:
    """Tests for _build_split_prompt helper."""

    def test_prompt_includes_content(self):
        """_build_split_prompt includes article content in prompt."""
        from valence.core.articles import _build_split_prompt

        prompt = _build_split_prompt("Test content here.", 100)
        assert "Test content here." in prompt

    def test_prompt_includes_schema(self):
        """_build_split_prompt includes TASK_SPLIT schema."""
        from valence.core.articles import _build_split_prompt

        prompt = _build_split_prompt("Content.", 100)
        assert "split_index" in prompt
        assert "part_a_title" in prompt
        assert "part_b_title" in prompt

    def test_prompt_forbids_generic_titles(self):
        """_build_split_prompt instructs LLM NOT to use generic titles."""
        from valence.core.articles import _build_split_prompt

        prompt = _build_split_prompt("Content.", 100)
        assert "NEVER" in prompt or "never" in prompt
        assert "part 1" in prompt.lower()
