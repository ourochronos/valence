"""Tests for local embedding provider (sentence-transformers/bge-small-en-v1.5)."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from valence.core.config import clear_config_cache

# ============================================================================
# Helper Function Tests
# ============================================================================


class TestIsLocalPath:
    """Tests for _is_local_path helper function."""

    def test_absolute_path_detected(self, tmp_path):
        """Should detect absolute filesystem paths."""
        from valence.embeddings.providers import local

        # Create a real path
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        assert local._is_local_path(str(model_dir)) is True

    def test_relative_path_detected(self):
        """Should detect relative paths starting with ./ or ../"""
        from valence.embeddings.providers import local

        assert local._is_local_path("./models/bge") is True
        assert local._is_local_path("../models/bge") is True

    def test_home_path_detected(self):
        """Should detect paths starting with ~"""
        from valence.embeddings.providers import local

        assert local._is_local_path("~/models/bge") is True

    def test_huggingface_model_name_not_local(self):
        """Should NOT detect HuggingFace model names as local paths."""
        from valence.embeddings.providers import local

        assert local._is_local_path("BAAI/bge-small-en-v1.5") is False
        assert local._is_local_path("sentence-transformers/all-MiniLM-L6-v2") is False


# ============================================================================
# Model Loading Tests
# ============================================================================


class TestGetModel:
    """Tests for get_model function."""

    def test_lazy_loading(self):
        """Should lazily initialize model on first call."""
        from valence.embeddings.providers import local

        # Reset model
        local.reset_model()

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
            model = local.get_model()

            mock_st.assert_called_once()
            assert model is mock_model

    def test_reuses_cached_model(self):
        """Should reuse cached model on subsequent calls."""
        from valence.embeddings.providers import local

        mock_model = MagicMock()
        local._model = mock_model

        result = local.get_model()

        assert result is mock_model

    def test_respects_device_env(self):
        """Should use VALENCE_EMBEDDING_DEVICE env var."""
        from valence.embeddings.providers import local

        local.reset_model()

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_DEVICE": "cuda"}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()

                    mock_st.assert_called_once()
                    call_kwargs = mock_st.call_args
                    assert call_kwargs[1]["device"] == "cuda"
            finally:
                clear_config_cache()

    def test_respects_model_path_env(self):
        """Should use VALENCE_EMBEDDING_MODEL_PATH env var."""
        from valence.embeddings.providers import local

        local.reset_model()

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        custom_model = "custom/model-path"
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": custom_model}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()

                    mock_st.assert_called_once()
                    call_args = mock_st.call_args
                    assert call_args[0][0] == custom_model
            finally:
                clear_config_cache()


# ============================================================================
# Single Embedding Tests
# ============================================================================


class TestGenerateEmbedding:
    """Tests for generate_embedding function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model that returns proper embeddings."""
        model = MagicMock()
        # Return normalized 384-dim vector
        normalized_vec = np.random.randn(384).astype(np.float32)
        normalized_vec = normalized_vec / np.linalg.norm(normalized_vec)
        model.encode.return_value = normalized_vec
        model.get_sentence_embedding_dimension.return_value = 384
        return model

    def test_returns_list_of_floats(self, mock_model):
        """Should return list of floats."""
        from valence.embeddings.providers import local

        local._model = mock_model

        result = local.generate_embedding("test text")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    def test_returns_384_dimensions(self, mock_model):
        """Should return 384-dimensional vector for BGE model."""
        from valence.embeddings.providers import local

        local._model = mock_model

        result = local.generate_embedding("test text")

        assert len(result) == 384

    def test_l2_normalized(self, mock_model):
        """Should return L2 normalized embeddings."""
        from valence.embeddings.providers import local

        local._model = mock_model

        result = local.generate_embedding("test text")

        # Check L2 norm is approximately 1
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 0.001, f"L2 norm should be 1.0, got {norm}"

    def test_calls_encode_with_normalize(self, mock_model):
        """Should call encode with normalize_embeddings=True."""
        from valence.embeddings.providers import local

        local._model = mock_model

        local.generate_embedding("test text")

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") is True


# ============================================================================
# Batch Embedding Tests
# ============================================================================


class TestGenerateEmbeddingsBatch:
    """Tests for generate_embeddings_batch function."""

    @pytest.fixture
    def mock_model(self):
        """Create mock model that returns proper batch embeddings."""
        model = MagicMock()

        def mock_encode(texts, **kwargs):
            # Return normalized vectors for each text
            n = len(texts) if isinstance(texts, list) else 1
            vecs = np.random.randn(n, 384).astype(np.float32)
            # Normalize each row
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / norms

        model.encode.side_effect = mock_encode
        model.get_sentence_embedding_dimension.return_value = 384
        return model

    def test_returns_list_of_embeddings(self, mock_model):
        """Should return list of embedding lists."""
        from valence.embeddings.providers import local

        local._model = mock_model

        texts = ["text 1", "text 2", "text 3"]
        result = local.generate_embeddings_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(emb, list) for emb in result)

    def test_each_embedding_384_dimensions(self, mock_model):
        """Each embedding should be 384 dimensions."""
        from valence.embeddings.providers import local

        local._model = mock_model

        texts = ["text 1", "text 2"]
        result = local.generate_embeddings_batch(texts)

        for emb in result:
            assert len(emb) == 384

    def test_respects_batch_size(self, mock_model):
        """Should pass batch_size to encode."""
        from valence.embeddings.providers import local

        local._model = mock_model

        texts = ["text"] * 10
        local.generate_embeddings_batch(texts, batch_size=5)

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("batch_size") == 5

    def test_shows_progress_for_large_batches(self, mock_model):
        """Should show progress bar for >100 texts by default."""
        from valence.embeddings.providers import local

        local._model = mock_model

        texts = ["text"] * 150
        local.generate_embeddings_batch(texts)

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("show_progress_bar") is True

    def test_no_progress_for_small_batches(self, mock_model):
        """Should not show progress bar for <=100 texts by default."""
        from valence.embeddings.providers import local

        local._model = mock_model

        texts = ["text"] * 50
        local.generate_embeddings_batch(texts)

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("show_progress_bar") is False

    def test_explicit_show_progress(self, mock_model):
        """Should respect explicit show_progress parameter."""
        from valence.embeddings.providers import local

        local._model = mock_model

        texts = ["text"] * 10
        local.generate_embeddings_batch(texts, show_progress=True)

        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("show_progress_bar") is True


# ============================================================================
# Integration with Service Tests
# ============================================================================


class TestServiceIntegration:
    """Tests for integration with embedding service."""

    def test_local_provider_default(self):
        """Local should be the default provider."""
        from valence.embeddings.service import EmbeddingProvider, get_embedding_provider

        # Clear env to test default
        with patch.dict(os.environ, {}, clear=True):
            # Need to clear VALENCE_EMBEDDING_PROVIDER specifically
            os.environ.pop("VALENCE_EMBEDDING_PROVIDER", None)
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.LOCAL
            finally:
                clear_config_cache()

    def test_openai_provider_override(self):
        """Should use OpenAI when explicitly configured."""
        from valence.embeddings.service import EmbeddingProvider, get_embedding_provider

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "openai"}):
            clear_config_cache()
            try:
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.OPENAI
            finally:
                clear_config_cache()

    def test_generate_embedding_uses_local(self):
        """generate_embedding should use local provider by default."""
        from valence.embeddings import service
        from valence.embeddings.providers import local

        # Mock the local provider
        mock_embedding = [0.1] * 384
        with patch.object(local, "_model", MagicMock()):
            with patch(
                "valence.embeddings.providers.local.generate_embedding",
                return_value=mock_embedding,
            ):
                with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "local"}):
                    result = service.generate_embedding("test", provider=service.EmbeddingProvider.LOCAL)

                    assert result == mock_embedding


# ============================================================================
# Reset Model Tests
# ============================================================================


class TestResetModel:
    """Tests for reset_model function."""

    def test_clears_cached_model(self):
        """Should clear the cached model."""
        from valence.embeddings.providers import local

        local._model = MagicMock()

        local.reset_model()

        assert local._model is None


# ============================================================================
# Offline/Custom Path Tests
# ============================================================================


class TestOfflineSupport:
    """Tests for offline installation and custom model paths."""

    def test_loads_from_local_path(self, tmp_path):
        """Should load model from local filesystem path."""
        from valence.embeddings.providers import local

        local.reset_model()

        # Create fake model directory
        model_dir = tmp_path / "my-model"
        model_dir.mkdir()

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": str(model_dir)}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()

                    # Should have loaded from the resolved path
                    mock_st.assert_called_once()
                    call_args = mock_st.call_args[0]
                    assert str(model_dir) in call_args[0]
            finally:
                clear_config_cache()

    def test_error_for_missing_local_path(self, tmp_path):
        """Should raise ModelLoadError with helpful message for missing local path."""
        from valence.embeddings.providers import local
        from valence.embeddings.providers.local import ModelLoadError

        local.reset_model()

        nonexistent_path = tmp_path / "does-not-exist"

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": str(nonexistent_path)}):
            clear_config_cache()
            try:
                with pytest.raises(ModelLoadError) as exc_info:
                    local.get_model()

                error_msg = str(exc_info.value)
                assert "not found" in error_msg.lower()
                assert "download_model.py" in error_msg
                assert "VALENCE_EMBEDDING_MODEL_PATH" in error_msg
            finally:
                clear_config_cache()

    def test_error_for_network_failure(self):
        """Should provide offline instructions when network fails."""
        from valence.embeddings.providers import local
        from valence.embeddings.providers.local import ModelLoadError

        local.reset_model()

        # Simulate network error
        network_error = OSError("Connection error: could not resolve hostname")

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": "BAAI/bge-small-en-v1.5"}):
            clear_config_cache()
            try:
                with patch(
                    "sentence_transformers.SentenceTransformer",
                    side_effect=network_error,
                ):
                    with pytest.raises(ModelLoadError) as exc_info:
                        local.get_model()

                    error_msg = str(exc_info.value)
                    assert "network unavailable" in error_msg.lower() or "cannot download" in error_msg.lower()
                    assert "download_model.py" in error_msg
                    assert "air-gapped" in error_msg.lower()
            finally:
                clear_config_cache()

    def test_model_load_error_is_exception(self):
        """ModelLoadError should be a proper exception."""
        from valence.embeddings.providers.local import ModelLoadError

        assert issubclass(ModelLoadError, Exception)

        error = ModelLoadError("test message")
        assert str(error) == "test message"

    def test_expands_home_in_path(self, tmp_path, monkeypatch):
        """Should expand ~ in model path."""
        from valence.embeddings.providers import local

        local.reset_model()

        # Create a model dir in a fake home
        fake_home = tmp_path / "fakehome"
        fake_home.mkdir()
        model_dir = fake_home / "models" / "bge"
        model_dir.mkdir(parents=True)

        monkeypatch.setenv("HOME", str(fake_home))

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": "~/models/bge"}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()

                    call_args = mock_st.call_args[0]
                    # Should have expanded ~ to actual path
                    assert "~" not in call_args[0]
                    assert "models/bge" in call_args[0]
            finally:
                clear_config_cache()

    def test_huggingface_name_not_treated_as_local(self):
        """HuggingFace model names should NOT be treated as local paths."""
        from valence.embeddings.providers import local

        local.reset_model()

        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # Standard HuggingFace model name
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_MODEL_PATH": "BAAI/bge-small-en-v1.5"}):
            clear_config_cache()
            try:
                with patch("sentence_transformers.SentenceTransformer", return_value=mock_model) as mock_st:
                    local.get_model()

                    # Should pass model name directly (not as resolved path)
                    call_args = mock_st.call_args[0]
                    assert call_args[0] == "BAAI/bge-small-en-v1.5"
            finally:
                clear_config_cache()


# ============================================================================
# Download Script Tests
# ============================================================================


class TestDownloadScript:
    """Tests for scripts/download_model.py"""

    def test_script_exists(self):
        """Download script should exist."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "download_model.py"
        assert script_path.exists(), f"Script not found at {script_path}"

    def test_script_is_executable_python(self):
        """Script should be valid Python."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "download_model.py"

        # Try to compile it
        with open(script_path) as f:
            source = f.read()

        # This will raise SyntaxError if invalid
        compile(source, str(script_path), "exec")

    def test_script_has_main_guard(self):
        """Script should have if __name__ == '__main__' guard."""
        script_path = Path(__file__).parent.parent.parent / "scripts" / "download_model.py"

        with open(script_path) as f:
            content = f.read()

        assert 'if __name__ == "__main__"' in content or "if __name__ == '__main__'" in content
