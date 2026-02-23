"""Embedding providers - pluggable embedding backends.

Available providers:
- local: sentence-transformers with bge-small-en-v1.5 (default, no API key needed)
- openai: OpenAI text-embedding-3-small (requires OPENAI_API_KEY)

Configure via VALENCE_EMBEDDING_PROVIDER environment variable.
"""

from .local import (
    EMBEDDING_DIMENSIONS as LOCAL_EMBEDDING_DIMENSIONS,
)
from .local import (
    MODEL_NAME as LOCAL_MODEL_NAME,
)
from .local import (
    generate_embedding as local_generate_embedding,
)
from .local import (
    generate_embeddings_batch as local_generate_embeddings_batch,
)
from .local import (
    get_model as get_local_model,
)
from .local import (
    reset_model as reset_local_model,
)

__all__ = [
    "local_generate_embedding",
    "local_generate_embeddings_batch",
    "get_local_model",
    "reset_local_model",
    "LOCAL_EMBEDDING_DIMENSIONS",
    "LOCAL_MODEL_NAME",
]
