"""our-embeddings - Vector embedding generation and similarity search."""

__version__ = "0.1.0"

from .federation import (
    FEDERATION_EMBEDDING_DIMS,
    FEDERATION_EMBEDDING_MODEL,
    FEDERATION_EMBEDDING_TYPE,
    get_federation_standard,
    is_federation_compatible,
    prepare_belief_for_federation,
    prepare_beliefs_batch_for_federation,
    regenerate_embedding_if_needed,
    validate_federation_embedding,
    validate_incoming_belief_embedding,
)
from .registry import (
    EmbeddingType,
    get_embedding_type,
    list_embedding_types,
    register_embedding_type,
)
from .service import (
    backfill_embeddings,
    embed_content,
    embed_content_async,
    search_similar,
    search_similar_async,
)

__all__ = [
    # Registry
    "EmbeddingType",
    "get_embedding_type",
    "list_embedding_types",
    "register_embedding_type",
    # Service
    "embed_content",
    "embed_content_async",
    "search_similar",
    "search_similar_async",
    "backfill_embeddings",
    # Federation
    "FEDERATION_EMBEDDING_MODEL",
    "FEDERATION_EMBEDDING_DIMS",
    "FEDERATION_EMBEDDING_TYPE",
    "get_federation_standard",
    "is_federation_compatible",
    "validate_federation_embedding",
    "prepare_belief_for_federation",
    "prepare_beliefs_batch_for_federation",
    "validate_incoming_belief_embedding",
    "regenerate_embedding_if_needed",
]
