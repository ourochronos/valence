"""Valence Embeddings - Vector generation and similarity search."""

from .registry import (
    EmbeddingType,
    get_embedding_type,
    list_embedding_types,
    register_embedding_type,
)
from .service import (
    embed_content,
    embed_content_async,
    search_similar,
    search_similar_async,
    backfill_embeddings,
)

__all__ = [
    "EmbeddingType",
    "get_embedding_type",
    "list_embedding_types",
    "register_embedding_type",
    "embed_content",
    "embed_content_async",
    "search_similar",
    "search_similar_async",
    "backfill_embeddings",
]
