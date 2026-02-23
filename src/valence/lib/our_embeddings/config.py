"""Configuration for our-embeddings.

Reads embedding-related settings from environment variables.
Maintains backward compatibility with VALENCE_EMBEDDING_* env var names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Embedding configuration settings."""

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_model_path: str = "BAAI/bge-small-en-v1.5"  # Backward compat
    embedding_device: str = "cpu"
    openai_api_key: str = ""

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        """Create config from environment variables."""
        provider = os.environ.get("VALENCE_EMBEDDING_PROVIDER", "local")

        # Default model based on provider
        if provider == "openai":
            default_model = "text-embedding-3-small"
        else:
            default_model = "BAAI/bge-small-en-v1.5"

        model = os.environ.get("VALENCE_EMBEDDING_MODEL", default_model)

        return cls(
            embedding_provider=provider,
            embedding_model=model,
            embedding_model_path=os.environ.get("VALENCE_EMBEDDING_MODEL_PATH", model),
            embedding_device=os.environ.get("VALENCE_EMBEDDING_DEVICE", "cpu"),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        )


_config: EmbeddingConfig | None = None


def get_config() -> EmbeddingConfig:
    """Get the global embedding configuration instance."""
    global _config
    if _config is None:
        _config = EmbeddingConfig.from_env()
    return _config


def clear_config_cache() -> None:
    """Clear the config cache. Useful for testing."""
    global _config
    _config = None
