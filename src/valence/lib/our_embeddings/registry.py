"""Embedding type registry for multi-model support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from valence.lib.our_db import get_cursor


@dataclass
class EmbeddingType:
    """Configuration for an embedding model."""

    id: str
    provider: str
    model: str
    dimensions: int
    is_default: bool = False
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "dimensions": self.dimensions,
            "is_default": self.is_default,
            "status": self.status,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EmbeddingType:
        """Create from database row."""
        return cls(
            id=row["id"],
            provider=row["provider"],
            model=row["model"],
            dimensions=row["dimensions"],
            is_default=row.get("is_default", False),
            status=row.get("status", "active"),
        )


def get_embedding_type(type_id: str | None = None) -> EmbeddingType | None:
    """Get an embedding type by ID, or the default if no ID provided."""
    with get_cursor() as cur:
        if type_id:
            cur.execute("SELECT * FROM embedding_types WHERE id = %s", (type_id,))
        else:
            cur.execute("SELECT * FROM embedding_types WHERE is_default = TRUE LIMIT 1")

        row = cur.fetchone()
        return EmbeddingType.from_row(dict(row)) if row else None


def list_embedding_types(status: str | None = None) -> list[EmbeddingType]:
    """List all embedding types."""
    with get_cursor() as cur:
        if status:
            cur.execute("SELECT * FROM embedding_types WHERE status = %s ORDER BY id", (status,))
        else:
            cur.execute("SELECT * FROM embedding_types ORDER BY id")

        rows = cur.fetchall()
        return [EmbeddingType.from_row(dict(row)) for row in rows]


def register_embedding_type(
    type_id: str,
    provider: str,
    model: str,
    dimensions: int,
    is_default: bool = False,
) -> EmbeddingType:
    """Register a new embedding type."""
    with get_cursor() as cur:
        # If setting as default, unset other defaults first
        if is_default:
            cur.execute("UPDATE embedding_types SET is_default = FALSE WHERE is_default = TRUE")

        cur.execute(
            """
            INSERT INTO embedding_types (id, provider, model, dimensions, is_default, status)
            VALUES (%s, %s, %s, %s, %s, 'active')
            ON CONFLICT (id) DO UPDATE
            SET provider = EXCLUDED.provider,
                model = EXCLUDED.model,
                dimensions = EXCLUDED.dimensions,
                is_default = EXCLUDED.is_default
            RETURNING *
            """,
            (type_id, provider, model, dimensions, is_default),
        )
        row = cur.fetchone()
        return EmbeddingType.from_row(dict(row))


# Known embedding configurations
KNOWN_EMBEDDINGS = {
    "openai_text3_small": {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    },
    "openai_text3_large": {
        "provider": "openai",
        "model": "text-embedding-3-large",
        "dimensions": 3072,
    },
    "openai_ada_002": {
        "provider": "openai",
        "model": "text-embedding-ada-002",
        "dimensions": 1536,
    },
}


def ensure_default_type() -> EmbeddingType:
    """Ensure the default embedding type exists."""
    existing = get_embedding_type()
    if existing:
        return existing

    # Register the default type
    return register_embedding_type(
        type_id="openai_text3_small",
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        is_default=True,
    )
