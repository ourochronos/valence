# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Embedding generation for Valence.

Uses OpenAI text-embedding-3-small by default.
Config via CoreSettings (get_config()).
"""

import logging
from typing import Any

import httpx

from valence.core.config import get_config

logger = logging.getLogger(__name__)


def get_embedding_config() -> dict[str, Any]:
    """Get embedding configuration from CoreSettings."""
    config = get_config()
    return {
        "provider": config.embedding_provider,
        "model": config.embedding_model,
        "dims": config.embedding_dims,
    }


def generate_embedding(text: str, model: str | None = None) -> list[float]:
    """Generate embedding vector for text.

    Args:
        text: Text to embed
        model: Model name (defaults to config)

    Returns:
        Embedding vector

    Raises:
        ValueError: If API key not set or provider not supported
        httpx.HTTPError: If API call fails
    """
    config = get_config()
    cfg = get_embedding_config()

    if cfg["provider"] != "openai":
        raise ValueError(f"Unsupported embedding provider: {cfg['provider']}")

    api_key = config.openai_api_key
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    if model is None:
        model = cfg["model"]

    # Truncate very long text
    if len(text) > 8000:
        logger.warning(f"Truncating text from {len(text)} to 8000 chars")
        text = text[:8000]

    try:
        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"input": text, "model": model, "dimensions": cfg["dims"]},
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except httpx.HTTPError as e:
        logger.error(f"Embedding generation failed: {e}")
        raise


def vector_to_pgvector(vector: list[float]) -> str:
    """Convert vector to pgvector format string.

    Args:
        vector: List of floats

    Returns:
        pgvector-formatted string like "[0.1,0.2,0.3]"
    """
    return "[" + ",".join(str(v) for v in vector) + "]"


def get_embedding_capability() -> dict[str, Any]:
    """Get this node's embedding capability for federation.

    Returns:
        Dict with model, dimensions, type_id, normalization
    """
    cfg = get_embedding_config()

    # Map model to type_id
    model = cfg["model"]
    dims = cfg["dims"]

    if "text-embedding-3-small" in model or dims == 1536:
        type_id = "openai_3_small"
    elif dims == 384:
        type_id = "bge_small_en_v15"
    elif dims == 768:
        type_id = "bge_base_en_v15"
    elif dims == 1024:
        type_id = "bge_large_en_v15"
    else:
        type_id = f"custom_{dims}d"

    return {
        "model": model,
        "dimensions": dims,
        "type_id": type_id,
        "normalization": "l2",
    }
