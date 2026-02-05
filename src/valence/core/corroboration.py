"""Corroboration tracking for Valence beliefs.

When beliefs arrive from federation peers, this module checks if similar
beliefs already exist and tracks corroboration from independent sources.

Corroboration boosts the `corroboration` dimension of 6D confidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import psycopg2

from .db import get_cursor
from .confidence import DimensionalConfidence, ConfidenceDimension

logger = logging.getLogger(__name__)

# Semantic similarity threshold for corroboration
CORROBORATION_THRESHOLD = 0.9

# Confidence boost formula: 1 - (1 / (1 + count * factor))
CORROBORATION_FACTOR = 0.3


@dataclass
class CorroborationResult:
    """Result of a corroboration check."""
    
    corroborated: bool
    existing_belief_id: UUID | None
    similarity: float
    source_did: str
    is_new_source: bool  # True if this source hadn't corroborated before
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "corroborated": self.corroborated,
            "existing_belief_id": str(self.existing_belief_id) if self.existing_belief_id else None,
            "similarity": self.similarity,
            "source_did": self.source_did,
            "is_new_source": self.is_new_source,
        }


@dataclass
class CorroborationInfo:
    """Corroboration details for a belief."""
    
    belief_id: UUID
    corroboration_count: int
    confidence_corroboration: float
    sources: list[dict[str, Any]]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": str(self.belief_id),
            "corroboration_count": self.corroboration_count,
            "confidence_corroboration": self.confidence_corroboration,
            "sources": self.sources,
        }


def calculate_corroboration_confidence(count: int) -> float:
    """Calculate corroboration confidence from source count.
    
    Uses asymptotic formula: 1 - (1 / (1 + count * factor))
    
    Examples:
        0 sources: 0.0
        1 source: 0.23
        2 sources: 0.38
        5 sources: 0.60
        10 sources: 0.75
    """
    if count <= 0:
        return 0.0
    return 1.0 - (1.0 / (1.0 + count * CORROBORATION_FACTOR))


def check_corroboration(
    content: str,
    source_did: str,
    content_embedding: list[float] | None = None,
) -> CorroborationResult | None:
    """Check if incoming belief content corroborates an existing belief.
    
    Args:
        content: The belief content to check
        source_did: DID of the source node
        content_embedding: Pre-computed embedding (optional, will generate if needed)
    
    Returns:
        CorroborationResult if a similar belief exists, None otherwise
    """
    try:
        # Generate embedding if not provided
        if content_embedding is None:
            from ..embeddings.service import generate_embedding, vector_to_pgvector
            content_embedding = generate_embedding(content)
        else:
            from ..embeddings.service import vector_to_pgvector
        
        query_vector = vector_to_pgvector(content_embedding)
        
        # Find most similar existing belief
        with get_cursor() as cur:
            cur.execute("""
                SELECT 
                    id, 
                    content,
                    corroborating_sources,
                    1 - (embedding <=> %s::vector) as similarity
                FROM beliefs
                WHERE embedding IS NOT NULL
                  AND status = 'active'
                  AND superseded_by_id IS NULL
                  AND is_local = TRUE
                ORDER BY embedding <=> %s::vector
                LIMIT 1
            """, (query_vector, query_vector))
            
            row = cur.fetchone()
            
            if not row:
                return None
            
            similarity = float(row["similarity"])
            
            if similarity < CORROBORATION_THRESHOLD:
                return None
            
            # Check if this source already corroborated
            existing_sources = row["corroborating_sources"] or []
            is_new_source = not any(
                s.get("source_did") == source_did 
                for s in existing_sources
            )
            
            return CorroborationResult(
                corroborated=True,
                existing_belief_id=row["id"],
                similarity=similarity,
                source_did=source_did,
                is_new_source=is_new_source,
            )
    
    except psycopg2.Error as e:
        logger.warning(f"Database error checking corroboration: {e}")
        return None


def add_corroboration(
    belief_id: UUID,
    source_did: str,
    similarity: float,
    boost_confidence: bool = True,
) -> bool:
    """Add a corroborating source to a belief.
    
    Args:
        belief_id: The belief being corroborated
        source_did: DID of the corroborating source
        similarity: Semantic similarity score
        boost_confidence: Whether to update confidence_corroboration
    
    Returns:
        True if corroboration was added (new source), False otherwise
    """
    try:
        with get_cursor() as cur:
            # Use the SQL function for atomic update
            cur.execute(
                "SELECT add_corroborating_source(%s, %s, %s, %s) as added",
                (belief_id, source_did, similarity, boost_confidence)
            )
            row = cur.fetchone()
            added = row["added"] if row else False
            
            if added:
                logger.info(
                    f"Added corroboration to belief {belief_id} from {source_did} "
                    f"(similarity={similarity:.3f})"
                )
            
            return added
    
    except psycopg2.Error as e:
        logger.warning(f"Database error adding corroboration: {e}")
        return False


def get_corroboration(belief_id: UUID) -> CorroborationInfo | None:
    """Get corroboration details for a belief.
    
    Args:
        belief_id: The belief UUID
    
    Returns:
        CorroborationInfo or None if belief not found
    """
    try:
        with get_cursor() as cur:
            cur.execute("""
                SELECT 
                    id,
                    corroboration_count,
                    confidence_corroboration,
                    corroborating_sources
                FROM beliefs
                WHERE id = %s
            """, (belief_id,))
            
            row = cur.fetchone()
            if not row:
                return None
            
            return CorroborationInfo(
                belief_id=row["id"],
                corroboration_count=row["corroboration_count"] or 0,
                confidence_corroboration=row["confidence_corroboration"] or 0.0,
                sources=row["corroborating_sources"] or [],
            )
    
    except psycopg2.Error as e:
        logger.warning(f"Database error getting corroboration: {e}")
        return None


def process_incoming_belief_corroboration(
    content: str,
    source_did: str,
    content_embedding: list[float] | None = None,
) -> CorroborationResult | None:
    """Process an incoming federated belief for corroboration.
    
    This is the main entry point for federation sync to call when
    receiving beliefs from peers.
    
    Args:
        content: Belief content
        source_did: DID of the source node
        content_embedding: Pre-computed embedding (optional)
    
    Returns:
        CorroborationResult if corroboration occurred, None otherwise
    """
    # Check if this corroborates an existing belief
    result = check_corroboration(content, source_did, content_embedding)
    
    if not result or not result.corroborated:
        return None
    
    # If it's a new source, add the corroboration
    if result.is_new_source and result.existing_belief_id:
        added = add_corroboration(
            belief_id=result.existing_belief_id,
            source_did=source_did,
            similarity=result.similarity,
            boost_confidence=True,
        )
        if not added:
            result.is_new_source = False
    
    return result


def get_most_corroborated_beliefs(
    limit: int = 10,
    min_count: int = 1,
    domain_filter: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Get beliefs with the most corroboration.
    
    Args:
        limit: Maximum beliefs to return
        min_count: Minimum corroboration count
        domain_filter: Optional domain filter
    
    Returns:
        List of belief dicts with corroboration info
    """
    try:
        conditions = [
            "corroboration_count >= %s",
            "status = 'active'",
            "superseded_by_id IS NULL",
        ]
        params: list[Any] = [min_count]
        
        if domain_filter:
            conditions.append("domain_path && %s")
            params.append(domain_filter)
        
        params.append(limit)
        
        with get_cursor() as cur:
            cur.execute(f"""
                SELECT 
                    id,
                    content,
                    corroboration_count,
                    confidence_corroboration,
                    corroborating_sources,
                    domain_path,
                    created_at
                FROM beliefs
                WHERE {' AND '.join(conditions)}
                ORDER BY corroboration_count DESC, created_at DESC
                LIMIT %s
            """, params)
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": str(row["id"]),
                    "content": row["content"],
                    "corroboration_count": row["corroboration_count"],
                    "confidence_corroboration": row["confidence_corroboration"],
                    "sources": row["corroborating_sources"] or [],
                    "domain_path": row["domain_path"] or [],
                    "created_at": row["created_at"].isoformat(),
                })
            
            return results
    
    except psycopg2.Error as e:
        logger.warning(f"Database error getting most corroborated beliefs: {e}")
        return []
