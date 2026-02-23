"""Federation embedding standard for cross-node compatibility.

Defines the canonical embedding format for Valence Federation Protocol (VFP).
All federation nodes MUST use this standard for cross-node semantic queries.

Standard: BAAI/bge-small-en-v1.5
- 384 dimensions
- L2 normalization
- Excellent semantic similarity performance
- Efficient for federated networks
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
    """
    return embedding_type == FEDERATION_EMBEDDING_TYPE and dimensions == FEDERATION_EMBEDDING_DIMS


def validate_federation_embedding(
    embedding: list[float] | None,
) -> tuple[bool, str | None]:
    """Validate an embedding vector for federation compatibility.

    Checks:
    1. Embedding is not None
    2. Correct dimensionality (384)
    3. Valid float values (no NaN/Inf)
    4. Approximately L2 normalized (magnitude ~ 1.0)

    Args:
        embedding: The embedding vector to validate

    Returns:
        Tuple of (is_valid, error_message or None)
    """
    if embedding is None:
        return False, "Embedding is None"

    if not isinstance(embedding, (list, tuple)):
        return False, f"Embedding must be a list, got {type(embedding).__name__}"

    if len(embedding) != FEDERATION_EMBEDDING_DIMS:
        return (
            False,
            f"Embedding must have {FEDERATION_EMBEDDING_DIMS} dimensions, got {len(embedding)}",
        )

    # Check for invalid values
    import math

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


# =============================================================================
# FEDERATION BELIEF PREPARATION
# =============================================================================


async def prepare_belief_for_federation(
    belief_id: str | UUID,
    include_embedding: bool = True,
    node_did: str | None = None,
) -> dict[str, Any]:
    """Prepare a belief for sharing via federation protocol.

    Ensures the belief has a federation-compatible embedding. If the belief
    doesn't have a compatible embedding, generates one using the local
    embedding provider.

    Args:
        belief_id: The belief's UUID (string or UUID object)
        include_embedding: Whether to include the embedding vector in output
        node_did: Optional node DID override. Falls back to VALENCE_NODE_DID env var,
                  then to a generated localhost DID.

    Returns:
        Dictionary with belief data formatted for federation protocol

    Raises:
        ValueError: If belief not found
        EmbeddingError: If embedding generation fails
    """
    from valence.lib.our_db import get_cursor

    from .providers.local import generate_embedding

    # Resolve node_did: parameter > env var > fallback
    if node_did is None:
        node_did = os.environ.get("VALENCE_NODE_DID") or "did:vkb:web:localhost:8420"

    # Normalize belief_id to string
    belief_id_str = str(belief_id)

    # Fetch belief from database
    with get_cursor() as cur:
        cur.execute(
            """
            SELECT
                b.id,
                b.content,
                b.confidence,
                b.domain_path,
                b.valid_from,
                b.valid_until,
                b.visibility,
                b.share_level,
                b.federation_id,
                b.created_at,
                b.modified_at,
                e.embedding_type,
                e.dimensions,
                e.vector
            FROM beliefs b
            LEFT JOIN belief_embeddings e ON b.id = e.belief_id
            WHERE b.id = %s
              AND b.status = 'active'
              AND b.is_local = TRUE
        """,
            (belief_id_str,),
        )
        row = cur.fetchone()

    if not row:
        raise ValueError(f"Belief not found or not shareable: {belief_id_str}")

    # Check visibility allows federation
    visibility = row.get("visibility", "private")
    if visibility == "private":
        raise ValueError(f"Belief {belief_id_str} has private visibility, cannot federate")

    # Determine embedding
    embedding = None
    embedding_type = row.get("embedding_type")
    dimensions = row.get("dimensions")

    if include_embedding:
        if is_federation_compatible(embedding_type, dimensions):
            # Use existing compatible embedding
            vector_data = row.get("vector")
            if vector_data:
                # Handle pgvector format
                if isinstance(vector_data, str):
                    # Parse "[0.1,0.2,...]" format
                    embedding = [float(v) for v in vector_data.strip("[]").split(",")]
                elif hasattr(vector_data, "tolist"):
                    embedding = vector_data.tolist()
                else:
                    embedding = list(vector_data)

                logger.debug(f"Using existing federation-compatible embedding for belief {belief_id_str}")

        if embedding is None:
            # Generate federation-standard embedding
            logger.info(f"Generating federation embedding for belief {belief_id_str}")
            embedding = generate_embedding(row["content"])

    # Build federation payload
    result: dict[str, Any] = {
        "belief_id": belief_id_str,
        "federation_id": str(row.get("federation_id") or row["id"]),
        "origin_node_did": node_did,
        "content": row["content"],
        "confidence": row["confidence"],
        "domain_path": row.get("domain_path", []),
        "visibility": visibility,
        "share_level": row.get("share_level", "belief_only"),
        "hop_count": 0,
        "federation_path": [],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
    }

    # Add temporal bounds if present
    if row.get("valid_from"):
        result["valid_from"] = row["valid_from"].isoformat()
    if row.get("valid_until"):
        result["valid_until"] = row["valid_until"].isoformat()

    # Add embedding metadata and vector
    if include_embedding and embedding:
        result["embedding"] = embedding
        result["embedding_model"] = FEDERATION_EMBEDDING_MODEL
        result["embedding_dims"] = FEDERATION_EMBEDDING_DIMS
        result["embedding_type"] = FEDERATION_EMBEDDING_TYPE

    return result


async def prepare_beliefs_batch_for_federation(
    belief_ids: list[str | UUID],
    include_embeddings: bool = True,
) -> list[dict[str, Any]]:
    """Prepare multiple beliefs for federation (batch operation).

    More efficient than calling prepare_belief_for_federation() repeatedly
    when sharing multiple beliefs.

    Args:
        belief_ids: List of belief UUIDs
        include_embeddings: Whether to include embedding vectors

    Returns:
        List of prepared belief dictionaries
    """
    results = []
    for belief_id in belief_ids:
        try:
            prepared = await prepare_belief_for_federation(belief_id, include_embedding=include_embeddings)
            results.append(prepared)
        except ValueError as e:
            logger.warning(f"Skipping belief {belief_id}: {e}")
        except Exception:
            logger.exception(f"Error preparing belief {belief_id} for federation")

    return results


# =============================================================================
# INCOMING EMBEDDING VALIDATION
# =============================================================================


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
        return False, (f"Embedding model mismatch: expected '{FEDERATION_EMBEDDING_MODEL}', got '{embedding_model}'")

    if embedding_dims and embedding_dims != FEDERATION_EMBEDDING_DIMS:
        return False, (f"Embedding dimensions mismatch: expected {FEDERATION_EMBEDDING_DIMS}, got {embedding_dims}")

    if embedding_type and embedding_type != FEDERATION_EMBEDDING_TYPE:
        return False, (f"Embedding type mismatch: expected '{FEDERATION_EMBEDDING_TYPE}', got '{embedding_type}'")

    # Validate the actual embedding vector
    return validate_federation_embedding(embedding)


def regenerate_embedding_if_needed(
    belief_data: dict[str, Any],
) -> list[float]:
    """Regenerate embedding for a belief if it's not federation-compatible.

    Use this when receiving beliefs via federation that either don't have
    embeddings or have incompatible embeddings.

    Args:
        belief_data: The belief dictionary containing at least 'content'

    Returns:
        Federation-compatible embedding vector (384 dims, L2 normalized)

    Raises:
        ValueError: If belief has no content
    """
    from .providers.local import generate_embedding

    content = belief_data.get("content")
    if not content:
        raise ValueError("Belief has no content to embed")

    return generate_embedding(content)
