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
    embedding_model_path: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    openai_api_key: str = ""

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        """Create config from environment variables."""
        return cls(
            embedding_provider=os.environ.get("VALENCE_EMBEDDING_PROVIDER", "local"),
            embedding_model_path=os.environ.get("VALENCE_EMBEDDING_MODEL_PATH", "BAAI/bge-small-en-v1.5"),
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
