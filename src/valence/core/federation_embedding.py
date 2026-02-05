"""Federation Embedding Standard - Shared configuration for VFP embeddings.

This module defines the canonical embedding format for the Valence Federation
Protocol (VFP). Both the embeddings and federation modules depend on this
shared standard.

Standard: BAAI/bge-small-en-v1.5
- 384 dimensions
- L2 normalization
- Excellent semantic similarity performance
- Efficient for federated networks

See: docs/federation/EMBEDDINGS.md for full specification.
"""

from __future__ import annotations

import math
from typing import Any


# =============================================================================
# FEDERATION EMBEDDING STANDARD
# =============================================================================

# The federation standard model - all nodes MUST support this for interoperability
FEDERATION_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FEDERATION_EMBEDDING_DIMS = 384

# Embedding type identifier used in database and protocol
FEDERATION_EMBEDDING_TYPE = "bge_small_en_v15"

# Standard version for protocol compatibility
FEDERATION_EMBEDDING_VERSION = "1.0"


def get_federation_standard() -> dict[str, Any]:
    """Get the federation embedding standard specification.

    Returns a dictionary describing the required embedding format
    for federation compatibility.

    Returns:
        Dictionary with model, dimensions, normalization, and version

    Example:
        >>> standard = get_federation_standard()
        >>> print(standard['model'])
        'BAAI/bge-small-en-v1.5'
        >>> print(standard['dimensions'])
        384
    """
    return {
        "model": FEDERATION_EMBEDDING_MODEL,
        "dimensions": FEDERATION_EMBEDDING_DIMS,
        "type": FEDERATION_EMBEDDING_TYPE,
        "normalization": "L2",
        "version": FEDERATION_EMBEDDING_VERSION,
    }


def is_federation_compatible(
    embedding_type: str | None,
    dimensions: int | None,
) -> bool:
    """Check if an embedding is federation compatible.

    Federation requires exact match on both model type and dimensions
    to ensure semantic similarity scores are comparable across nodes.

    Args:
        embedding_type: The embedding type identifier (e.g., 'bge_small_en_v15')
        dimensions: The embedding vector dimensions

    Returns:
        True if compatible with federation standard

    Example:
        >>> is_federation_compatible("bge_small_en_v15", 384)
        True
        >>> is_federation_compatible("text_embedding_3_small", 1536)
        False
    """
    return (
        embedding_type == FEDERATION_EMBEDDING_TYPE
        and dimensions == FEDERATION_EMBEDDING_DIMS
    )


def validate_federation_embedding(
    embedding: list[float] | None,
) -> tuple[bool, str | None]:
    """Validate an embedding vector for federation compatibility.

    Checks:
    1. Embedding is not None
    2. Correct dimensionality (384)
    3. Valid float values (no NaN/Inf)
    4. Approximately L2 normalized (magnitude ≈ 1.0)

    Args:
        embedding: The embedding vector to validate

    Returns:
        Tuple of (is_valid, error_message or None)

    Example:
        >>> valid, error = validate_federation_embedding([0.1] * 384)
        >>> print(valid, error)
        False 'Embedding not L2 normalized (magnitude=1.95)'
    """
    if embedding is None:
        return False, "Embedding is None"

    if not isinstance(embedding, (list, tuple)):
        return False, f"Embedding must be a list, got {type(embedding).__name__}"

    if len(embedding) != FEDERATION_EMBEDDING_DIMS:
        return (
            False,
            f"Embedding must have {FEDERATION_EMBEDDING_DIMS} dimensions, "
            f"got {len(embedding)}",
        )

    # Check for invalid values
    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            return False, f"Embedding[{i}] is not a number: {type(val).__name__}"
        if math.isnan(val) or math.isinf(val):
            return False, f"Embedding[{i}] contains invalid value: {val}"

    # Check L2 normalization (magnitude should be approximately 1.0)
    magnitude = math.sqrt(sum(v * v for v in embedding))
    if abs(magnitude - 1.0) > 0.01:  # Allow small floating point tolerance
        return False, f"Embedding not L2 normalized (magnitude={magnitude:.2f})"

    return True, None


def validate_incoming_belief_embedding(
    belief_data: dict[str, Any],
) -> tuple[bool, str | None]:
    """Validate embedding in an incoming federated belief.

    For federation sync, we require beliefs to include embeddings that
    match our federation standard. This ensures semantic queries work
    correctly across the network.

    Args:
        belief_data: The incoming belief dictionary from federation protocol

    Returns:
        Tuple of (is_valid, error_message or None)
    """
    # Check if embedding is provided
    embedding = belief_data.get("embedding")
    if embedding is None:
        # Embedding is optional - we can generate it locally
        return True, None

    # Validate model metadata if provided
    embedding_model = belief_data.get("embedding_model")
    embedding_dims = belief_data.get("embedding_dims")
    embedding_type = belief_data.get("embedding_type")

    # If metadata is provided, it must match federation standard
    if embedding_model and embedding_model != FEDERATION_EMBEDDING_MODEL:
        return False, (
            f"Embedding model mismatch: expected '{FEDERATION_EMBEDDING_MODEL}', "
            f"got '{embedding_model}'"
        )

    if embedding_dims and embedding_dims != FEDERATION_EMBEDDING_DIMS:
        return False, (
            f"Embedding dimensions mismatch: expected {FEDERATION_EMBEDDING_DIMS}, "
            f"got {embedding_dims}"
        )

    if embedding_type and embedding_type != FEDERATION_EMBEDDING_TYPE:
        return False, (
            f"Embedding type mismatch: expected '{FEDERATION_EMBEDDING_TYPE}', "
            f"got '{embedding_type}'"
        )

    # Validate the actual embedding vector
    return validate_federation_embedding(embedding)
