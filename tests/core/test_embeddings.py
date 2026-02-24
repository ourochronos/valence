"""Tests for valence.core.embeddings module - embedding generation and provider dispatch.

Tests cover:
- get_embedding_config() configuration loading
- generate_embedding() with various inputs
- Error handling (missing API key, unsupported provider)
- Text truncation for long inputs
- vector_to_pgvector() conversion
- get_embedding_capability() capability reporting
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest


class TestGetEmbeddingConfig:
    """Test embedding configuration loading."""

    @patch("valence.core.embeddings.get_config")
    def test_config_openai(self, mock_get_config):
        """Test config returns OpenAI settings."""
        from valence.core.embeddings import get_embedding_config

        mock_cfg = MagicMock()
        mock_cfg.embedding_provider = "openai"
        mock_cfg.embedding_model = "text-embedding-3-small"
        mock_cfg.embedding_dims = 1536
        mock_get_config.return_value = mock_cfg

        config = get_embedding_config()

        assert config["provider"] == "openai"
        assert config["model"] == "text-embedding-3-small"
        assert config["dims"] == 1536

    @patch("valence.core.embeddings.get_config")
    def test_config_local(self, mock_get_config):
        """Test config returns local settings."""
        from valence.core.embeddings import get_embedding_config

        mock_cfg = MagicMock()
        mock_cfg.embedding_provider = "local"
        mock_cfg.embedding_model = "BAAI/bge-small-en-v1.5"
        mock_cfg.embedding_dims = 384
        mock_get_config.return_value = mock_cfg

        config = get_embedding_config()

        assert config["provider"] == "local"
        assert config["model"] == "BAAI/bge-small-en-v1.5"
        assert config["dims"] == 384


class TestGenerateEmbedding:
    """Test generate_embedding function."""

    @patch("valence.core.embeddings.httpx.post")
    @patch("valence.core.embeddings.get_config")
    @patch("valence.core.embeddings.get_embedding_config")
    def test_generate_embedding_success(self, mock_get_cfg, mock_get_config, mock_post):
        """Test successful embedding generation."""
        from valence.core.embeddings import generate_embedding

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dims": 1536,
        }

        mock_config = MagicMock()
        mock_config.openai_api_key = "test-key-123"
        mock_get_config.return_value = mock_config

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        result = generate_embedding("test text")

        assert result == [0.1, 0.2, 0.3]
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key-123"
        assert call_kwargs["json"]["input"] == "test text"
        assert call_kwargs["json"]["model"] == "text-embedding-3-small"

    @patch("valence.core.embeddings.get_config")
    @patch("valence.core.embeddings.get_embedding_config")
    def test_generate_embedding_no_api_key(self, mock_get_cfg, mock_get_config):
        """Test error when API key not set."""
        from valence.core.embeddings import generate_embedding

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dims": 1536,
        }

        mock_config = MagicMock()
        mock_config.openai_api_key = ""
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError, match="OPENAI_API_KEY not set"):
            generate_embedding("test text")

    @patch("valence.core.embeddings.get_config")
    @patch("valence.core.embeddings.get_embedding_config")
    def test_generate_embedding_unsupported_provider(self, mock_get_cfg, mock_get_config):
        """Test error for unsupported provider."""
        from valence.core.embeddings import generate_embedding

        mock_get_cfg.return_value = {
            "provider": "unsupported",
            "model": "some-model",
            "dims": 512,
        }

        mock_config = MagicMock()
        mock_config.openai_api_key = "test-key"
        mock_get_config.return_value = mock_config

        with pytest.raises(ValueError, match="Unsupported embedding provider"):
            generate_embedding("test text")

    @patch("valence.core.embeddings.httpx.post")
    @patch("valence.core.embeddings.get_config")
    @patch("valence.core.embeddings.get_embedding_config")
    def test_generate_embedding_custom_model(self, mock_get_cfg, mock_get_config, mock_post):
        """Test embedding generation with custom model."""
        from valence.core.embeddings import generate_embedding

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dims": 1536,
        }

        mock_config = MagicMock()
        mock_config.openai_api_key = "test-key"
        mock_get_config.return_value = mock_config

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.5, 0.6]}]}
        mock_post.return_value = mock_response

        result = generate_embedding("test", model="text-embedding-3-large")

        assert result == [0.5, 0.6]
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "text-embedding-3-large"

    @patch("valence.core.embeddings.httpx.post")
    @patch("valence.core.embeddings.get_config")
    @patch("valence.core.embeddings.get_embedding_config")
    def test_generate_embedding_truncates_long_text(self, mock_get_cfg, mock_get_config, mock_post):
        """Test text truncation for very long inputs."""
        from valence.core.embeddings import generate_embedding

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dims": 1536,
        }

        mock_config = MagicMock()
        mock_config.openai_api_key = "test-key"
        mock_get_config.return_value = mock_config

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
        mock_post.return_value = mock_response

        # Text longer than 8000 chars
        long_text = "x" * 9000

        result = generate_embedding(long_text)

        assert result == [0.1, 0.2]
        # Should be truncated to 8000
        call_kwargs = mock_post.call_args[1]
        sent_text = call_kwargs["json"]["input"]
        assert len(sent_text) == 8000

    @patch("valence.core.embeddings.httpx.post")
    @patch("valence.core.embeddings.get_config")
    @patch("valence.core.embeddings.get_embedding_config")
    def test_generate_embedding_http_error(self, mock_get_cfg, mock_get_config, mock_post):
        """Test error handling for HTTP failures."""
        import httpx

        from valence.core.embeddings import generate_embedding

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dims": 1536,
        }

        mock_config = MagicMock()
        mock_config.openai_api_key = "test-key"
        mock_get_config.return_value = mock_config

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("401 Unauthorized", request=Mock(), response=Mock())
        mock_post.return_value = mock_response

        with pytest.raises(httpx.HTTPError):
            generate_embedding("test")

    @patch("valence.core.embeddings.httpx.post")
    @patch("valence.core.embeddings.get_config")
    @patch("valence.core.embeddings.get_embedding_config")
    def test_generate_embedding_timeout(self, mock_get_cfg, mock_get_config, mock_post):
        """Test timeout setting is applied."""
        from valence.core.embeddings import generate_embedding

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dims": 1536,
        }

        mock_config = MagicMock()
        mock_config.openai_api_key = "test-key"
        mock_get_config.return_value = mock_config

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": [{"embedding": [0.1]}]}
        mock_post.return_value = mock_response

        generate_embedding("test")

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["timeout"] == 30.0


class TestVectorToPgvector:
    """Test vector_to_pgvector conversion."""

    def test_vector_to_pgvector_basic(self):
        """Test basic vector conversion."""
        from valence.core.embeddings import vector_to_pgvector

        vector = [0.1, 0.2, 0.3]
        result = vector_to_pgvector(vector)

        assert result == "[0.1,0.2,0.3]"

    def test_vector_to_pgvector_single_element(self):
        """Test conversion of single-element vector."""
        from valence.core.embeddings import vector_to_pgvector

        vector = [0.5]
        result = vector_to_pgvector(vector)

        assert result == "[0.5]"

    def test_vector_to_pgvector_empty(self):
        """Test conversion of empty vector."""
        from valence.core.embeddings import vector_to_pgvector

        vector = []
        result = vector_to_pgvector(vector)

        assert result == "[]"

    def test_vector_to_pgvector_negative(self):
        """Test conversion with negative values."""
        from valence.core.embeddings import vector_to_pgvector

        vector = [-0.5, 0.0, 0.5]
        result = vector_to_pgvector(vector)

        assert result == "[-0.5,0.0,0.5]"

    def test_vector_to_pgvector_large(self):
        """Test conversion of large vector."""
        from valence.core.embeddings import vector_to_pgvector

        vector = [0.1] * 1536
        result = vector_to_pgvector(vector)

        assert result.startswith("[0.1,")
        assert result.endswith(",0.1]")
        assert result.count(",") == 1535  # n-1 commas for n elements

    def test_vector_to_pgvector_precision(self):
        """Test conversion preserves float precision."""
        from valence.core.embeddings import vector_to_pgvector

        vector = [0.123456789, 1.987654321]
        result = vector_to_pgvector(vector)

        assert "0.123456789" in result
        assert "1.987654321" in result


class TestGetEmbeddingCapability:
    """Test get_embedding_capability function."""

    @patch("valence.core.embeddings.get_embedding_config")
    def test_capability_openai_small(self, mock_get_cfg):
        """Test capability report for OpenAI 3-small."""
        from valence.core.embeddings import get_embedding_capability

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dims": 1536,
        }

        cap = get_embedding_capability()

        assert cap["model"] == "text-embedding-3-small"
        assert cap["dimensions"] == 1536
        assert cap["type_id"] == "openai_3_small"
        assert cap["normalization"] == "l2"

    @patch("valence.core.embeddings.get_embedding_config")
    def test_capability_bge_small(self, mock_get_cfg):
        """Test capability report for BGE small."""
        from valence.core.embeddings import get_embedding_capability

        mock_get_cfg.return_value = {
            "provider": "local",
            "model": "BAAI/bge-small-en-v1.5",
            "dims": 384,
        }

        cap = get_embedding_capability()

        assert cap["model"] == "BAAI/bge-small-en-v1.5"
        assert cap["dimensions"] == 384
        assert cap["type_id"] == "bge_small_en_v15"
        assert cap["normalization"] == "l2"

    @patch("valence.core.embeddings.get_embedding_config")
    def test_capability_bge_base(self, mock_get_cfg):
        """Test capability report for BGE base."""
        from valence.core.embeddings import get_embedding_capability

        mock_get_cfg.return_value = {
            "provider": "local",
            "model": "BAAI/bge-base-en-v1.5",
            "dims": 768,
        }

        cap = get_embedding_capability()

        assert cap["dimensions"] == 768
        assert cap["type_id"] == "bge_base_en_v15"

    @patch("valence.core.embeddings.get_embedding_config")
    def test_capability_bge_large(self, mock_get_cfg):
        """Test capability report for BGE large."""
        from valence.core.embeddings import get_embedding_capability

        mock_get_cfg.return_value = {
            "provider": "local",
            "model": "BAAI/bge-large-en-v1.5",
            "dims": 1024,
        }

        cap = get_embedding_capability()

        assert cap["dimensions"] == 1024
        assert cap["type_id"] == "bge_large_en_v15"

    @patch("valence.core.embeddings.get_embedding_config")
    def test_capability_custom_dimensions(self, mock_get_cfg):
        """Test capability report for custom dimensions."""
        from valence.core.embeddings import get_embedding_capability

        mock_get_cfg.return_value = {
            "provider": "local",
            "model": "custom-model",
            "dims": 512,
        }

        cap = get_embedding_capability()

        assert cap["dimensions"] == 512
        assert cap["type_id"] == "custom_512d"
        assert cap["model"] == "custom-model"

    @patch("valence.core.embeddings.get_embedding_config")
    def test_capability_normalization_always_l2(self, mock_get_cfg):
        """Test normalization is always l2."""
        from valence.core.embeddings import get_embedding_capability

        mock_get_cfg.return_value = {
            "provider": "openai",
            "model": "any-model",
            "dims": 256,
        }

        cap = get_embedding_capability()

        assert cap["normalization"] == "l2"
