"""Tests for embedding provider configuration (Issue #26).

Verifies:
- VALENCE_EMBEDDING_PROVIDER environment variable support
- OpenAI provider (default)
- Local provider stub
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch, MagicMock

from valence.embeddings.service import (
    EmbeddingProvider,
    get_embedding_provider,
    generate_embedding,
    generate_local_embedding,
)


class TestEmbeddingProvider:
    """Test EmbeddingProvider enum."""

    def test_openai_provider(self):
        """OpenAI should be a valid provider."""
        assert EmbeddingProvider.OPENAI == "openai"

    def test_local_provider(self):
        """Local should be a valid provider."""
        assert EmbeddingProvider.LOCAL == "local"


class TestGetEmbeddingProvider:
    """Test provider detection from environment."""

    def test_default_is_openai(self):
        """Default provider should be OpenAI."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove VALENCE_EMBEDDING_PROVIDER if set
            os.environ.pop("VALENCE_EMBEDDING_PROVIDER", None)
            provider = get_embedding_provider()
            assert provider == EmbeddingProvider.OPENAI

    def test_openai_from_env(self):
        """Should detect OpenAI from env."""
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "openai"}):
            provider = get_embedding_provider()
            assert provider == EmbeddingProvider.OPENAI

    def test_local_from_env(self):
        """Should detect local from env."""
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "local"}):
            provider = get_embedding_provider()
            assert provider == EmbeddingProvider.LOCAL

    def test_case_insensitive(self):
        """Should handle case variations."""
        test_cases = ["LOCAL", "Local", "LOCAL"]
        for value in test_cases:
            with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": value}):
                provider = get_embedding_provider()
                assert provider == EmbeddingProvider.LOCAL

    def test_unknown_defaults_to_openai(self):
        """Unknown provider should default to OpenAI with warning."""
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "unknown-provider"}):
            provider = get_embedding_provider()
            assert provider == EmbeddingProvider.OPENAI


class TestLocalEmbedding:
    """Test local embedding stub."""

    def test_local_embedding_not_implemented(self):
        """Local embeddings should raise NotImplementedError (stub)."""
        with pytest.raises(NotImplementedError) as exc_info:
            generate_local_embedding("test text")
        
        assert "Local embeddings not yet implemented" in str(exc_info.value)

    def test_local_embedding_helpful_message(self):
        """Error message should guide users."""
        with pytest.raises(NotImplementedError) as exc_info:
            generate_local_embedding("test")
        
        error_msg = str(exc_info.value)
        assert "VALENCE_EMBEDDING_PROVIDER=openai" in error_msg


class TestGenerateEmbedding:
    """Test generate_embedding with provider routing."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client."""
        with patch("valence.embeddings.service.get_openai_client") as mock:
            client = MagicMock()
            mock.return_value = client
            
            # Mock response
            response = MagicMock()
            response.data = [MagicMock(embedding=[0.1] * 1536)]
            client.embeddings.create.return_value = response
            
            yield client

    def test_uses_openai_by_default(self, mock_openai):
        """Should use OpenAI when no provider specified."""
        with patch.dict(os.environ, {"VALENCE_EMBEDDING_PROVIDER": "openai", "OPENAI_API_KEY": "test"}):
            result = generate_embedding("test text")
            
            assert len(result) == 1536
            mock_openai.embeddings.create.assert_called_once()

    def test_explicit_openai_provider(self, mock_openai):
        """Should use OpenAI when explicitly specified."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            result = generate_embedding(
                "test text",
                provider=EmbeddingProvider.OPENAI
            )
            
            assert len(result) == 1536

    def test_explicit_local_provider_raises(self):
        """Should raise for local provider (not implemented)."""
        with pytest.raises(NotImplementedError):
            generate_embedding("test", provider=EmbeddingProvider.LOCAL)

    def test_truncates_long_text(self, mock_openai):
        """Should truncate text over 8000 chars."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            long_text = "a" * 10000
            generate_embedding(long_text, provider=EmbeddingProvider.OPENAI)
            
            # Check that truncated text was sent
            call_args = mock_openai.embeddings.create.call_args
            sent_text = call_args.kwargs.get("input") or call_args[1].get("input")
            assert len(sent_text) == 8000


class TestBeliefOptOut:
    """Test belief federation opt-out flag (Issue #26)."""

    def test_belief_create_accepts_opt_out(self):
        """belief_create should accept opt_out_federation parameter."""
        from valence.substrate.tools import belief_create
        
        # This tests the function signature accepts the parameter
        # Full integration test would require database
        import inspect
        sig = inspect.signature(belief_create)
        params = list(sig.parameters.keys())
        
        assert "opt_out_federation" in params

    def test_belief_create_schema_includes_opt_out(self):
        """Tool schema should include opt_out_federation."""
        from valence.substrate.tools import SUBSTRATE_TOOLS
        
        belief_create_tool = next(
            t for t in SUBSTRATE_TOOLS if t.name == "belief_create"
        )
        
        schema = belief_create_tool.inputSchema
        assert "opt_out_federation" in schema["properties"]
        assert schema["properties"]["opt_out_federation"]["type"] == "boolean"
