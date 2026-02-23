"""Embedding service for generating and searching vectors.

Supports multiple embedding providers (Issue #26):
- openai: Uses OpenAI's text-embedding-3-small (default)
- local: Uses local embedding model (stub for future implementation)

Set VALENCE_EMBEDDING_PROVIDER environment variable to configure.

PRIVACY NOTE: When using 'openai' provider, belief content is sent
to OpenAI's API for embedding generation. See README for details.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import Any

from openai import OpenAI

from valence.lib.our_db import get_cursor
from valence.lib.our_db.exceptions import DatabaseError

from .config import get_config
from .exceptions import EmbeddingError
from .registry import ensure_default_type, get_embedding_type

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingProvider(StrEnum):
    """Available embedding providers."""

    OPENAI = "openai"
    LOCAL = "local"


def get_embedding_provider() -> EmbeddingProvider:
    """Get configured embedding provider from config.

    Defaults to 'local' for privacy and to avoid API costs.
    Set VALENCE_EMBEDDING_PROVIDER=openai to use OpenAI embeddings.
    """
    config = get_config()
    provider = config.embedding_provider.lower()
    try:
        return EmbeddingProvider(provider)
    except ValueError:
        logger.warning(f"Unknown embedding provider '{provider}', defaulting to 'local'")
        return EmbeddingProvider.LOCAL


# OpenAI client (lazy init)
_openai_client: OpenAI | None = None

# Thread pool for async operations
_executor = ThreadPoolExecutor(max_workers=4)


def get_openai_client() -> OpenAI:
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        config = get_config()
        api_key = config.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable required for OpenAI embeddings")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def generate_local_embedding(text: str) -> list[float]:
    """Generate embedding using local sentence-transformers model.

    Uses BAAI/bge-small-en-v1.5 by default, which produces 384-dimensional
    L2-normalized embeddings with excellent semantic similarity.

    Configure via environment variables:
    - VALENCE_EMBEDDING_MODEL_PATH: Model name or local path
    - VALENCE_EMBEDDING_DEVICE: Device to use (cpu|cuda)

    Args:
        text: Text to embed

    Returns:
        384-dimensional embedding vector (L2 normalized)
    """
    from .providers.local import generate_embedding as local_embed

    return local_embed(text)


def generate_embedding(
    text: str,
    model: str | None = None,
    provider: EmbeddingProvider | None = None,
) -> list[float]:
    """Generate embedding for text.

    Args:
        text: Text to embed
        model: Model name (defaults to config.embedding_model)
        provider: Embedding provider (defaults to env config)

    Returns:
        Embedding vector

    Raises:
        ValueError: If provider not configured correctly
        NotImplementedError: If local provider requested but not implemented
    """
    config = get_config()
    
    if provider is None:
        provider = get_embedding_provider()
    
    if model is None:
        model = config.embedding_model

    # Truncate very long text
    if len(text) > 8000:
        text = text[:8000]

    if provider == EmbeddingProvider.LOCAL:
        return generate_local_embedding(text)

    # Default: OpenAI
    client = get_openai_client()

    response = client.embeddings.create(
        model=model,
        input=text,
    )

    return response.data[0].embedding


def vector_to_pgvector(vector: list[float]) -> str:
    """Convert vector to pgvector format string."""
    return "[" + ",".join(str(v) for v in vector) + "]"


def embed_content(
    content_type: str,
    content_id: str,
    text: str,
    embedding_type_id: str | None = None,
) -> dict[str, Any]:
    """Generate and store embedding for content."""
    # Get embedding type
    emb_type = get_embedding_type(embedding_type_id) or ensure_default_type()

    # Generate embedding
    vector = generate_embedding(text, emb_type.model)
    vector_str = vector_to_pgvector(vector)

    with get_cursor() as cur:
        # Update the appropriate table based on content type
        if content_type == "belief":
            cur.execute(
                "UPDATE beliefs SET embedding = %s, modified_at = NOW() WHERE id = %s",
                (vector_str, content_id),
            )
        elif content_type == "exchange":
            cur.execute(
                "UPDATE vkb_exchanges SET embedding = %s WHERE id = %s",
                (vector_str, content_id),
            )
        elif content_type == "pattern":
            cur.execute(
                "UPDATE vkb_patterns SET embedding = %s WHERE id = %s",
                (vector_str, content_id),
            )

        # Track coverage
        cur.execute(
            """
            INSERT INTO embedding_coverage (content_type, content_id, embedding_type_id)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_type, content_id, embedding_type_id)
            DO UPDATE SET embedded_at = NOW()
            """,
            (content_type, content_id, emb_type.id),
        )

    logger.info(f"Embedded {content_type}:{content_id} with {emb_type.id}")

    return {
        "content_type": content_type,
        "content_id": content_id,
        "embedding_type": emb_type.id,
        "dimensions": len(vector),
    }


async def embed_content_async(
    content_type: str,
    content_id: str,
    text: str,
    embedding_type_id: str | None = None,
) -> dict[str, Any]:
    """Async wrapper for embed_content."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        embed_content,
        content_type,
        content_id,
        text,
        embedding_type_id,
    )


def search_similar(
    query: str,
    content_type: str | None = None,
    limit: int = 10,
    min_similarity: float = 0.5,
    embedding_type_id: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar content using embedding similarity."""
    # Get embedding type
    emb_type = get_embedding_type(embedding_type_id) or ensure_default_type()

    # Generate query embedding
    query_vector = generate_embedding(query, emb_type.model)
    query_str = vector_to_pgvector(query_vector)

    results = []

    with get_cursor() as cur:
        # Search each content type
        if content_type is None or content_type == "belief":
            cur.execute(
                """
                SELECT id, content, 1 - (embedding <=> %s::vector) as similarity
                FROM beliefs
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_str, query_str, min_similarity, query_str, limit),
            )
            for row in cur.fetchall():
                results.append(
                    {
                        "content_type": "belief",
                        "content_id": str(row["id"]),
                        "content": row["content"],
                        "similarity": float(row["similarity"]),
                    }
                )

        if content_type is None or content_type == "exchange":
            cur.execute(
                """
                SELECT id, session_id, content, 1 - (embedding <=> %s::vector) as similarity
                FROM vkb_exchanges
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_str, query_str, min_similarity, query_str, limit),
            )
            for row in cur.fetchall():
                results.append(
                    {
                        "content_type": "exchange",
                        "content_id": str(row["id"]),
                        "session_id": str(row["session_id"]),
                        "content": row["content"],
                        "similarity": float(row["similarity"]),
                    }
                )

        if content_type is None or content_type == "pattern":
            cur.execute(
                """
                SELECT id, type, description, 1 - (embedding <=> %s::vector) as similarity
                FROM vkb_patterns
                WHERE embedding IS NOT NULL
                AND 1 - (embedding <=> %s::vector) >= %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_str, query_str, min_similarity, query_str, limit),
            )
            for row in cur.fetchall():
                results.append(
                    {
                        "content_type": "pattern",
                        "content_id": str(row["id"]),
                        "pattern_type": row["type"],
                        "description": row["description"],
                        "similarity": float(row["similarity"]),
                    }
                )

    # Sort by similarity and limit
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]


async def search_similar_async(
    query: str,
    content_type: str | None = None,
    limit: int = 10,
    min_similarity: float = 0.5,
    embedding_type_id: str | None = None,
) -> list[dict[str, Any]]:
    """Async wrapper for search_similar."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        search_similar,
        query,
        content_type,
        limit,
        min_similarity,
        embedding_type_id,
    )


def backfill_embeddings(
    content_type: str,
    batch_size: int = 100,
    embedding_type_id: str | None = None,
) -> int:
    """Backfill embeddings for content that doesn't have them."""
    emb_type = get_embedding_type(embedding_type_id) or ensure_default_type()
    count = 0

    with get_cursor() as cur:
        if content_type == "belief":
            cur.execute(
                """
                SELECT id, content FROM beliefs
                WHERE embedding IS NULL
                AND status = 'active'
                LIMIT %s
                """,
                (batch_size,),
            )
        elif content_type == "exchange":
            cur.execute(
                """
                SELECT id, content FROM exchanges
                WHERE embedding IS NULL
                LIMIT %s
                """,
                (batch_size,),
            )
        elif content_type == "pattern":
            cur.execute(
                """
                SELECT id, description as content FROM patterns
                WHERE embedding IS NULL
                LIMIT %s
                """,
                (batch_size,),
            )
        else:
            return 0

        rows = cur.fetchall()

    for row in rows:
        try:
            embed_content(content_type, str(row["id"]), row["content"], emb_type.id)
            count += 1
        except (EmbeddingError, DatabaseError) as e:
            logger.error(f"Failed to embed {content_type} {row['id']}: {e}")

    logger.info(f"Backfilled {count} {content_type} embeddings")
    return count
