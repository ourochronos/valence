"""Embedding interoperability for federation (#356).

Ensures beliefs federate as text only — embeddings are never sent.
Received beliefs are re-embedded locally using the node's configured model.
Provides text-based similarity fallback for cross-model comparison.

Key principle: Each node embeds with its own model. Embedding vectors are
non-portable across models, so we strip them on send and regenerate on receive.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCapability:
    """Describes this node's embedding configuration for capability advertisement."""

    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = 384
    type_id: str = "bge_small_en_v15"
    normalization: str = "l2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "type_id": self.type_id,
            "normalization": self.normalization,
        }


def get_embedding_capability() -> EmbeddingCapability:
    """Get this node's embedding capability from config.

    Reads VALENCE_EMBEDDING_DIMS and related config, falling back to defaults.
    """
    import os

    provider = os.environ.get("VALENCE_EMBEDDING_PROVIDER", "local")

    # Default dimensions based on provider
    if provider == "openai":
        default_dims = 1536
    else:
        default_dims = 384

    dims = int(os.environ.get("VALENCE_EMBEDDING_DIMS", str(default_dims)))

    # Get configured model or infer from dimensions
    model = os.environ.get("VALENCE_EMBEDDING_MODEL", "")

    # Map common dimensions to known models
    dim_to_model = {
        384: ("BAAI/bge-small-en-v1.5", "bge_small_en_v15"),
        768: ("BAAI/bge-base-en-v1.5", "bge_base_en_v15"),
        1024: ("BAAI/bge-large-en-v1.5", "bge_large_en_v15"),
        1536: ("text-embedding-3-small", "openai_3_small"),
    }

    if model:
        # Use configured model, infer type_id
        if "text-embedding-3-small" in model or dims == 1536:
            type_id = "openai_3_small"
        elif "bge-small" in model or dims == 384:
            type_id = "bge_small_en_v15"
        elif "bge-base" in model or dims == 768:
            type_id = "bge_base_en_v15"
        elif "bge-large" in model or dims == 1024:
            type_id = "bge_large_en_v15"
        else:
            type_id = f"custom_{dims}d"
    else:
        # Infer from dimensions
        model, type_id = dim_to_model.get(dims, (f"unknown-{dims}d", f"custom_{dims}d"))

    return EmbeddingCapability(model=model, dimensions=dims, type_id=type_id)


def strip_embedding_for_federation(belief: dict[str, Any]) -> dict[str, Any]:
    """Remove embedding vectors from a belief before federation transmission.

    Beliefs should federate as text only. Each receiving node re-embeds
    using its own model.

    Args:
        belief: Belief dict (may contain 'embedding', 'embedding_model', etc.)

    Returns:
        New dict with embedding fields removed
    """
    stripped = {k: v for k, v in belief.items() if not k.startswith("embedding")}
    return stripped


def prepare_received_belief_for_embedding(belief: dict[str, Any]) -> str | None:
    """Extract the text content from a received federated belief for local embedding.

    Args:
        belief: Received belief dict from federation

    Returns:
        Text content suitable for embedding, or None if no content
    """
    content = belief.get("content")
    if not content:
        return None
    return content


def text_similarity(text_a: str, text_b: str) -> float:
    """Compute text-based similarity using TF-IDF cosine similarity.

    Fallback for cross-model comparison when vector similarity is unavailable.
    Uses simple term frequency with IDF approximation.

    Args:
        text_a: First text
        text_b: Second text

    Returns:
        Similarity score 0.0-1.0
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    # Build term frequency vectors
    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)

    # Get all unique terms
    all_terms = set(tf_a.keys()) | set(tf_b.keys())

    if not all_terms:
        return 0.0

    # Cosine similarity of TF vectors
    dot_product = sum(tf_a.get(t, 0) * tf_b.get(t, 0) for t in all_terms)
    mag_a = math.sqrt(sum(v * v for v in tf_a.values()))
    mag_b = math.sqrt(sum(v * v for v in tf_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot_product / (mag_a * mag_b)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer with lowercasing."""
    return [t for t in re.split(r"\W+", text.lower()) if t and len(t) > 1]


def build_embedding_capability_advertisement() -> dict[str, Any]:
    """Build embedding capability info for federation handshake.

    This should be included in the DID document's vfp:embedding field.

    Returns:
        Dict with embedding model info for capability advertisement
    """
    cap = get_embedding_capability()
    return {
        "vfp:embedding": cap.to_dict(),
    }
