"""Tests for dedup-on-compile feature (issue #73).

Tests that compile_article() deduplicates against existing articles by:
  1. Updating an existing similar article rather than creating a duplicate.
  2. Still creating a new article when content is genuinely novel.
  3. Gracefully degrading if embedding generation fails.

Uses the same mock-cursor pattern as test_compilation.py.
asyncio_mode = auto (pyproject.toml), so no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock, patch
from uuid import uuid4

import valence.core.compilation as compilation_mod
from valence.core.compilation import (
    DEDUP_SIMILARITY_THRESHOLD,
    _find_similar_article,
    compile_article,
)

# ---------------------------------------------------------------------------
# Shared IDs & helpers
# ---------------------------------------------------------------------------

NOW = datetime.now()
SOURCE_ID_1 = str(uuid4())
EXISTING_ARTICLE_ID = str(uuid4())
NEW_ARTICLE_ID = str(uuid4())


def _source_row(sid=None, content="Python is great.") -> dict:
    return {
        "id": sid or SOURCE_ID_1,
        "type": "document",
        "title": "Source Doc",
        "url": None,
        "content": content,
        "reliability": 0.8,
    }


def _article_row(aid=None, content="Compiled content.", version=1, **kw) -> dict:
    from valence.core.compilation import _count_tokens

    defaults = {
        "id": aid or EXISTING_ARTICLE_ID,
        "content": content,
        "title": "Existing Article",
        "author_type": "system",
        "pinned": False,
        "size_tokens": _count_tokens(content),
        "compiled_at": NOW,
        "usage_score": 0,
        "confidence": json.dumps({"overall": 0.7}),
        "domain_path": [],
        "version": version,
        "content_hash": "abc",
        "status": "active",
        "superseded_by_id": None,
        "created_at": NOW,
        "modified_at": NOW,
        "embedding": None,
        "epistemic_type": "semantic",
        "confidence_source": 0.7,
        "corroboration_count": 1,
    }
    defaults.update(kw)
    return defaults


def _make_cursor(fetchone_seq=None, fetchall_seq=None):
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


def _llm_ok(title="Compiled Title", content="Compiled content.", rels=None) -> str:
    if rels is None:
        rels = [{"source_id": SOURCE_ID_1, "relationship": "originates"}]
    return json.dumps({"articles": [{"title": title, "content": content, "source_relationships": rels}]})


_RS = {"max_tokens": 800, "min_tokens": 300, "target_tokens": 550}
_PL = {"max_total_chars": 100_000, "max_source_chars": 50_000}


# ---------------------------------------------------------------------------
# Unit: DEDUP_SIMILARITY_THRESHOLD constant
# ---------------------------------------------------------------------------


class TestDedupConstant:
    def test_threshold_is_0_90(self):
        assert DEDUP_SIMILARITY_THRESHOLD == 0.90


# ---------------------------------------------------------------------------
# Unit: _find_similar_article
# ---------------------------------------------------------------------------


class TestFindSimilarArticle:
    async def test_returns_none_when_no_match(self):
        fake_vector = [0.1] * 1536
        cur = _make_cursor(fetchone_seq=[None])
        with (
            patch.object(compilation_mod, "generate_embedding", return_value=fake_vector),
            patch("valence.core.compilation.get_cursor", return_value=cur),
        ):
            result = await _find_similar_article("some content")
        assert result is None

    async def test_returns_article_dict_when_match_found(self):
        fake_vector = [0.1] * 1536
        existing = _article_row(EXISTING_ARTICLE_ID, "Very similar content")
        cur = _make_cursor(fetchone_seq=[existing])
        with (
            patch.object(compilation_mod, "generate_embedding", return_value=fake_vector),
            patch("valence.core.compilation.get_cursor", return_value=cur),
        ):
            result = await _find_similar_article("Very similar content")
        assert result is not None
        assert result["id"] == EXISTING_ARTICLE_ID

    async def test_graceful_degradation_on_embedding_failure(self):
        """If embedding raises, return None (graceful degradation)."""
        with patch.object(compilation_mod, "generate_embedding", side_effect=ValueError("API key missing")):
            result = await _find_similar_article("some content")
        assert result is None

    async def test_graceful_degradation_on_db_failure(self):
        """If DB query raises, return None (graceful degradation)."""
        fake_vector = [0.1] * 1536
        cur = MagicMock()
        inner_cur = MagicMock()
        inner_cur.execute.side_effect = Exception("DB error")
        cur.__enter__ = MagicMock(return_value=inner_cur)
        cur.__exit__ = MagicMock(return_value=False)
        with (
            patch.object(compilation_mod, "generate_embedding", return_value=fake_vector),
            patch("valence.core.compilation.get_cursor", return_value=cur),
        ):
            result = await _find_similar_article("some content")
        assert result is None

    async def test_uses_custom_threshold(self):
        """Custom threshold is passed to the SQL query."""
        fake_vector = [0.1] * 1536
        cur = _make_cursor(fetchone_seq=[None])
        with (
            patch.object(compilation_mod, "generate_embedding", return_value=fake_vector),
            patch("valence.core.compilation.get_cursor", return_value=cur),
        ):
            await _find_similar_article("content", threshold=0.95)
        call_args_str = str(cur.execute.call_args_list)
        assert "0.95" in call_args_str


# ---------------------------------------------------------------------------
# Integration: compile_article dedup paths
# ---------------------------------------------------------------------------


class TestCompileArticleDedupPath:
    async def test_dedup_fires_updates_existing_article(self):
        """When an existing article is highly similar, compile_article updates it."""
        src = _source_row(SOURCE_ID_1, "Python 3.12 is fast.")
        existing = _article_row(EXISTING_ARTICLE_ID, "Python 3.12 compilation result.")
        updated = _article_row(EXISTING_ARTICLE_ID, "Python 3.12 updated.", version=2)
        llm_resp = _llm_ok(
            content="Python 3.12 updated.",
            rels=[{"source_id": SOURCE_ID_1, "relationship": "confirms"}],
        )
        fake_vector = [0.1] * 1536
        update_cur = _make_cursor(fetchone_seq=[updated])

        async def mock_llm(prompt, **_kw):
            return llm_resp

        with (
            patch("valence.core.compilation.get_cursor") as mock_gc,
            patch.object(compilation_mod, "_get_right_sizing", return_value=_RS),
            patch.object(compilation_mod, "_get_prompt_limits", return_value=_PL),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
            patch.object(compilation_mod, "generate_embedding", return_value=fake_vector),
            patch.object(type(compilation_mod._inference_provider), "available", new_callable=PropertyMock, return_value=True),
        ):
            sources_cur = _make_cursor(fetchall_seq=[[src]])
            dedup_cur = _make_cursor(fetchone_seq=[existing])
            mock_gc.side_effect = [sources_cur, dedup_cur, update_cur]
            result = await compile_article([SOURCE_ID_1])

        assert result.success is True
        assert result.data["id"] == EXISTING_ARTICLE_ID
        assert result.data["version"] == 2
        calls_str = str(update_cur.execute.call_args_list)
        assert "UPDATE articles" in calls_str

    async def test_no_dedup_creates_new_article_when_novel(self):
        """When no similar article exists, compile_article creates a new one normally."""
        src = _source_row(SOURCE_ID_1, "Novel content about quantum computing.")
        new_art = _article_row(NEW_ARTICLE_ID, "Quantum compiled.")
        llm_resp = _llm_ok(
            content="Quantum compiled.",
            rels=[{"source_id": SOURCE_ID_1, "relationship": "originates"}],
        )
        fake_vector = [0.1] * 1536

        async def mock_llm(prompt, **_kw):
            return llm_resp

        with (
            patch("valence.core.compilation.get_cursor") as mock_gc,
            patch.object(compilation_mod, "_get_right_sizing", return_value=_RS),
            patch.object(compilation_mod, "_get_prompt_limits", return_value=_PL),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
            patch.object(compilation_mod, "generate_embedding", return_value=fake_vector),
            patch.object(type(compilation_mod._inference_provider), "available", new_callable=PropertyMock, return_value=True),
        ):
            sources_cur = _make_cursor(fetchall_seq=[[src]])
            dedup_cur = _make_cursor(fetchone_seq=[None])  # No similar article
            create_cur = _make_cursor(fetchone_seq=[new_art])
            mock_gc.side_effect = [sources_cur, dedup_cur, create_cur]
            result = await compile_article([SOURCE_ID_1])

        assert result.success is True
        assert result.data["id"] == NEW_ARTICLE_ID
        calls_str = str(create_cur.execute.call_args_list)
        assert "INSERT INTO articles" in calls_str

    async def test_dedup_skipped_when_embedding_fails(self):
        """When embedding fails, dedup is skipped and a new article is created."""
        src = _source_row(SOURCE_ID_1)
        new_art = _article_row(NEW_ARTICLE_ID)
        llm_resp = _llm_ok(
            rels=[{"source_id": SOURCE_ID_1, "relationship": "originates"}],
        )

        async def mock_llm(prompt, **_kw):
            return llm_resp

        with (
            patch("valence.core.compilation.get_cursor") as mock_gc,
            patch.object(compilation_mod, "_get_right_sizing", return_value=_RS),
            patch.object(compilation_mod, "_get_prompt_limits", return_value=_PL),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
            patch.object(
                compilation_mod,
                "generate_embedding",
                side_effect=ValueError("OPENAI_API_KEY not set"),
            ),
            patch.object(type(compilation_mod._inference_provider), "available", new_callable=PropertyMock, return_value=True),
        ):
            sources_cur = _make_cursor(fetchall_seq=[[src]])
            create_cur = _make_cursor(fetchone_seq=[new_art])
            mock_gc.side_effect = [sources_cur, create_cur]
            result = await compile_article([SOURCE_ID_1])

        assert result.success is True
        assert result.data["id"] == NEW_ARTICLE_ID

    async def test_dedup_logs_when_firing(self, caplog):
        """Dedup logs an info message when it finds an existing similar article."""
        src = _source_row(SOURCE_ID_1)
        existing = _article_row(EXISTING_ARTICLE_ID)
        updated = _article_row(EXISTING_ARTICLE_ID, version=2)
        llm_resp = _llm_ok(
            rels=[{"source_id": SOURCE_ID_1, "relationship": "confirms"}],
        )
        fake_vector = [0.1] * 1536

        async def mock_llm(prompt, **_kw):
            return llm_resp

        with (
            patch("valence.core.compilation.get_cursor") as mock_gc,
            patch.object(compilation_mod, "_get_right_sizing", return_value=_RS),
            patch.object(compilation_mod, "_get_prompt_limits", return_value=_PL),
            patch.object(compilation_mod, "_call_llm", side_effect=mock_llm),
            patch.object(compilation_mod, "generate_embedding", return_value=fake_vector),
            patch.object(type(compilation_mod._inference_provider), "available", new_callable=PropertyMock, return_value=True),
            caplog.at_level(logging.INFO, logger="valence.core.compilation"),
        ):
            sources_cur = _make_cursor(fetchall_seq=[[src]])
            dedup_cur = _make_cursor(fetchone_seq=[existing])
            update_cur = _make_cursor(fetchone_seq=[updated])
            mock_gc.side_effect = [sources_cur, dedup_cur, update_cur]
            await compile_article([SOURCE_ID_1])

        assert any("Dedup" in record.message for record in caplog.records)
        assert any(EXISTING_ARTICLE_ID in record.message for record in caplog.records)
