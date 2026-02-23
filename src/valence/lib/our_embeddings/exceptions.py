"""Exception hierarchy for our-embeddings."""

from __future__ import annotations


class EmbeddingError(Exception):
    """Base exception for all embedding-related errors.

    Raised when:
    - Embedding generation fails
    - Embedding provider is unavailable
    - Vector dimension mismatch
    """

    def __init__(self, message: str, provider: str | None = None):
        self.message = message
        self.provider = provider
        super().__init__(message)
