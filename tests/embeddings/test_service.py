"""Tests for valence.embeddings.service module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


# ============================================================================
# get_openai_client Tests
# ============================================================================

class TestGetOpenaiClient:
    """Tests for get_openai_client function."""

    def test_lazy_initialization(self, env_with_openai_key):
        """Should lazily initialize OpenAI client."""
        from valence.embeddings import service

        # Reset the global client
        service._openai_client = None

        with patch.object(service, "OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client

            client = service.get_openai_client()

            mock_openai.assert_called_once()
            assert client is mock_client

    def test_reuses_existing_client(self, env_with_openai_key):
        """Should reuse existing client."""
        from valence.embeddings import service

        mock_client = MagicMock()
        service._openai_client = mock_client

        result = service.get_openai_client()

        assert result is mock_client

    def test_raises_without_api_key(self, clean_env):
        """Should raise ValueError without API key."""
        from valence.embeddings import service

        service._openai_client = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            service.get_openai_client()


# ============================================================================
# generate_embedding Tests
# ============================================================================

class TestGenerateEmbedding:
    """Tests for generate_embedding function."""

    def test_success(self, env_with_openai_key, mock_openai):
        """Should generate embedding."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        result = service.generate_embedding("test text")

        assert len(result) == 1536
        mock_openai.embeddings.create.assert_called_once()

    def test_truncates_long_text(self, env_with_openai_key, mock_openai):
        """Should truncate text longer than 8000 chars."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        long_text = "x" * 10000
        service.generate_embedding(long_text)

        call_args = mock_openai.embeddings.create.call_args
        assert len(call_args.kwargs["input"]) == 8000

    def test_uses_specified_model(self, env_with_openai_key, mock_openai):
        """Should use specified model."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        service.generate_embedding("test", model="text-embedding-3-large")

        call_args = mock_openai.embeddings.create.call_args
        assert call_args.kwargs["model"] == "text-embedding-3-large"


# ============================================================================
# vector_to_pgvector Tests
# ============================================================================

class TestVectorToPgvector:
    """Tests for vector_to_pgvector function."""

    def test_format_conversion(self):
        """Should convert to pgvector format."""
        from valence.embeddings.service import vector_to_pgvector

        vector = [0.1, 0.2, 0.3]
        result = vector_to_pgvector(vector)

        assert result == "[0.1,0.2,0.3]"

    def test_empty_vector(self):
        """Should handle empty vector."""
        from valence.embeddings.service import vector_to_pgvector

        result = vector_to_pgvector([])
        assert result == "[]"

    def test_float_precision(self):
        """Should preserve float precision."""
        from valence.embeddings.service import vector_to_pgvector

        vector = [0.123456789]
        result = vector_to_pgvector(vector)

        assert "0.123456789" in result


# ============================================================================
# embed_content Tests
# ============================================================================

class TestEmbedContent:
    """Tests for embed_content function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("valence.embeddings.service.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_embed_belief(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should embed belief content."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        # Mock embedding type lookup
        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.embed_content("belief", str(uuid4()), "test content")

            assert result["content_type"] == "belief"
            assert result["dimensions"] == 1536

    def test_embed_exchange(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should embed exchange content."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.embed_content("exchange", str(uuid4()), "test")

            assert result["content_type"] == "exchange"

    def test_embed_pattern(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should embed pattern content."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.embed_content("pattern", str(uuid4()), "test")

            assert result["content_type"] == "pattern"


# ============================================================================
# search_similar Tests
# ============================================================================

class TestSearchSimilar:
    """Tests for search_similar function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("valence.embeddings.service.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_search_all_types(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should search all content types by default."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.search_similar("test query")

            assert isinstance(result, list)
            # Should have searched beliefs, exchanges, and patterns
            assert mock_get_cursor.execute.call_count == 3

    def test_filter_by_content_type(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should filter by content type."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.search_similar("test", content_type="belief")

            # Should only search beliefs
            assert mock_get_cursor.execute.call_count == 1

    def test_returns_sorted_results(self, env_with_openai_key, mock_openai, mock_get_cursor):
        """Should return results sorted by similarity."""
        from valence.embeddings import service

        service._openai_client = mock_openai

        mock_get_cursor.fetchall.side_effect = [
            [  # Beliefs
                {
                    "id": uuid4(),
                    "content": "Test 1",
                    "similarity": 0.7,
                }
            ],
            [  # Exchanges
                {
                    "id": uuid4(),
                    "session_id": uuid4(),
                    "content": "Test 2",
                    "similarity": 0.9,
                }
            ],
            [],  # Patterns
        ]

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            result = service.search_similar("test")

            # Should be sorted by similarity descending
            if len(result) >= 2:
                assert result[0]["similarity"] >= result[1]["similarity"]


# ============================================================================
# backfill_embeddings Tests
# ============================================================================

class TestBackfillEmbeddings:
    """Tests for backfill_embeddings function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        cursor.fetchall.return_value = []
        return cursor

    @pytest.fixture
    def mock_get_cursor(self, mock_cursor):
        """Mock get_cursor context manager."""
        from contextlib import contextmanager

        @contextmanager
        def fake_get_cursor(dict_cursor=True):
            yield mock_cursor

        with patch("valence.embeddings.service.get_cursor", fake_get_cursor):
            yield mock_cursor

    def test_backfill_beliefs(self, env_with_openai_key, mock_get_cursor):
        """Should backfill belief embeddings."""
        from valence.embeddings import service

        belief_id = uuid4()
        mock_get_cursor.fetchall.return_value = [
            {"id": belief_id, "content": "Test belief"}
        ]

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            with patch("valence.embeddings.service.embed_content") as mock_embed:
                result = service.backfill_embeddings("belief", batch_size=10)

                mock_embed.assert_called_once()
                assert result == 1

    def test_handles_errors(self, env_with_openai_key, mock_get_cursor):
        """Should continue on individual embedding errors."""
        from valence.embeddings import service
        from valence.core.exceptions import EmbeddingException

        mock_get_cursor.fetchall.return_value = [
            {"id": uuid4(), "content": "Test 1"},
            {"id": uuid4(), "content": "Test 2"},
        ]

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_type.model = "text-embedding-3-small"
            mock_get_type.return_value = mock_type

            with patch("valence.embeddings.service.embed_content") as mock_embed:
                # First call fails, second succeeds
                mock_embed.side_effect = [
                    EmbeddingException("Failed"),
                    {"content_type": "belief"},
                ]

                result = service.backfill_embeddings("belief")

                # Should have processed 1 successfully
                assert result == 1

    def test_returns_zero_for_unknown_type(self, env_with_openai_key, mock_get_cursor):
        """Should return 0 for unknown content type."""
        from valence.embeddings import service

        with patch("valence.embeddings.service.get_embedding_type") as mock_get_type:
            mock_type = MagicMock()
            mock_type.id = "openai_text3_small"
            mock_get_type.return_value = mock_type

            result = service.backfill_embeddings("unknown_type")

            assert result == 0


# ============================================================================
# Async Function Tests
# ============================================================================

class TestAsyncFunctions:
    """Tests for async wrapper functions."""

    @pytest.mark.asyncio
    async def test_embed_content_async(self, env_with_openai_key):
        """Should wrap embed_content in async."""
        from valence.embeddings import service

        with patch("valence.embeddings.service.embed_content") as mock_embed:
            mock_embed.return_value = {"content_type": "belief"}

            result = await service.embed_content_async(
                "belief", str(uuid4()), "test"
            )

            assert result["content_type"] == "belief"

    @pytest.mark.asyncio
    async def test_search_similar_async(self, env_with_openai_key):
        """Should wrap search_similar in async."""
        from valence.embeddings import service

        with patch("valence.embeddings.service.search_similar") as mock_search:
            mock_search.return_value = []

            result = await service.search_similar_async("test query")

            assert result == []
