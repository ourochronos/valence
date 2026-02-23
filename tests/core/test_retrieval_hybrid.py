"""Unit tests for hybrid retrieval with RRF.

Tests the implementation of Issue #409: hybrid retrieval combining
vector KNN and full-text search via Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from valence.core.retrieval import _search_articles_sync, _search_ungrouped_sources_sync


class MockCursor:
    """Mock cursor for testing."""

    def __init__(self, fetchall_return=None):
        self.fetchall_return = fetchall_return or []
        self.execute_calls = []

    def execute(self, query, params=None):
        self.execute_calls.append((query, params))

    def fetchall(self):
        return self.fetchall_return

    def fetchone(self):
        return self.fetchall_return[0] if self.fetchall_return else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@patch("valence.core.embeddings.generate_embedding")
@patch("valence.core.retrieval._has_active_contentions")
@patch("valence.core.retrieval._build_provenance_summary")
@patch("valence.core.retrieval.get_cursor")
def test_hybrid_retrieval_vector_only_match(
    mock_get_cursor, mock_prov, mock_cont, mock_gen_embed
):
    """Test that vector-only matches are returned when text search finds nothing."""
    # Mock article found by vector but not by text
    mock_rows = [
        {
            "id": "123e4567-e89b-12d3-a456-426614174000",
            "content": "Discord bot setup guide",
            "title": "Discord Configuration",
            "status": "active",
            "superseded_by_id": None,
            "confidence_source": 0.8,
            "confidence_method": 0.9,
            "confidence_consistency": 1.0,
            "confidence_freshness": 1.0,
            "confidence_corroboration": 0.7,
            "confidence_applicability": 0.9,
            "created_at": datetime.now(UTC),
            "vec_rank": 1,
            "vec_score": 0.92,
            "text_rank": 1000,
            "text_score": 0.0,
            "rrf_score": 0.016666,  # 1/(60+1) + 1/(60+1000)
            "content_tsv": None,
            "embedding": None,
        }
    ]

    mock_cursor = MockCursor(fetchall_return=mock_rows)
    mock_get_cursor.return_value.__enter__.return_value = mock_cursor
    mock_gen_embed.return_value = [0.1] * 1536
    mock_prov.return_value = {"source_count": 1, "relationship_types": ["cited"]}
    mock_cont.return_value = False

    results = _search_articles_sync("discord configuration", limit=10)

    assert len(results) == 1
    assert results[0]["id"] == "123e4567-e89b-12d3-a456-426614174000"
    assert results[0]["type"] == "article"
    assert results[0]["vec_score"] > 0.9
    assert results[0]["text_score"] == 0.0
    assert results[0]["vec_rank"] == 1
    assert results[0]["text_rank"] == 1000


@patch("valence.core.embeddings.generate_embedding")
@patch("valence.core.retrieval._has_active_contentions")
@patch("valence.core.retrieval._build_provenance_summary")
@patch("valence.core.retrieval.get_cursor")
def test_hybrid_retrieval_text_only_match(
    mock_get_cursor, mock_prov, mock_cont, mock_gen_embed
):
    """Test that text-only matches are returned when embedding is missing."""
    # Mock article found by text but not by vector
    mock_rows = [
        {
            "id": "123e4567-e89b-12d3-a456-426614174001",
            "content": "Exact keyword match article",
            "title": "Delegation Compliance",
            "status": "active",
            "superseded_by_id": None,
            "confidence_source": 0.8,
            "confidence_method": 0.9,
            "confidence_consistency": 1.0,
            "confidence_freshness": 1.0,
            "confidence_corroboration": 0.7,
            "confidence_applicability": 0.9,
            "created_at": datetime.now(UTC),
            "vec_rank": 1000,
            "vec_score": 0.0,
            "text_rank": 1,
            "text_score": 0.85,
            "rrf_score": 0.016666,  # 1/(60+1000) + 1/(60+1)
            "content_tsv": None,
            "embedding": None,
        }
    ]

    mock_cursor = MockCursor(fetchall_return=mock_rows)
    mock_get_cursor.return_value.__enter__.return_value = mock_cursor
    mock_gen_embed.return_value = [0.1] * 1536
    mock_prov.return_value = {"source_count": 2, "relationship_types": ["cited"]}
    mock_cont.return_value = False

    results = _search_articles_sync("delegation compliance", limit=10)

    assert len(results) == 1
    assert results[0]["id"] == "123e4567-e89b-12d3-a456-426614174001"
    assert results[0]["type"] == "article"
    assert results[0]["vec_score"] == 0.0
    assert results[0]["text_score"] > 0.8
    assert results[0]["vec_rank"] == 1000
    assert results[0]["text_rank"] == 1


@patch("valence.core.embeddings.generate_embedding")
@patch("valence.core.retrieval._has_active_contentions")
@patch("valence.core.retrieval._build_provenance_summary")
@patch("valence.core.retrieval.get_cursor")
def test_hybrid_retrieval_rrf_combination(
    mock_get_cursor, mock_prov, mock_cont, mock_gen_embed
):
    """Test that RRF correctly ranks items with both signals highest."""
    # Mock: article with both signals vs article with only one signal
    mock_rows = [
        {
            "id": "123e4567-e89b-12d3-a456-426614174002",
            "content": "Both vector and text match",
            "title": "Best Match",
            "status": "active",
            "superseded_by_id": None,
            "confidence_source": 0.8,
            "confidence_method": 0.9,
            "confidence_consistency": 1.0,
            "confidence_freshness": 1.0,
            "confidence_corroboration": 0.7,
            "confidence_applicability": 0.9,
            "created_at": datetime.now(UTC),
            "vec_rank": 1,
            "vec_score": 0.95,
            "text_rank": 1,
            "text_score": 0.88,
            "rrf_score": 0.03278,  # 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03278
            "content_tsv": None,
            "embedding": None,
        },
        {
            "id": "123e4567-e89b-12d3-a456-426614174003",
            "content": "Only vector match",
            "title": "Vector Match",
            "status": "active",
            "superseded_by_id": None,
            "confidence_source": 0.8,
            "confidence_method": 0.9,
            "confidence_consistency": 1.0,
            "confidence_freshness": 1.0,
            "confidence_corroboration": 0.7,
            "confidence_applicability": 0.9,
            "created_at": datetime.now(UTC),
            "vec_rank": 2,
            "vec_score": 0.90,
            "text_rank": 1000,
            "text_score": 0.0,
            "rrf_score": 0.016129,  # 1/(60+2) + 1/(60+1000) ≈ 0.016129
            "content_tsv": None,
            "embedding": None,
        },
    ]

    mock_cursor = MockCursor(fetchall_return=mock_rows)
    mock_get_cursor.return_value.__enter__.return_value = mock_cursor
    mock_gen_embed.return_value = [0.1] * 1536
    mock_prov.return_value = {"source_count": 1, "relationship_types": ["cited"]}
    mock_cont.return_value = False

    results = _search_articles_sync("test query", limit=10)

    assert len(results) == 2
    # First result should have higher RRF score (both signals)
    assert results[0]["rrf_score"] > results[1]["rrf_score"]
    assert results[0]["id"] == "123e4567-e89b-12d3-a456-426614174002"
    # Similarity is normalized RRF score, should be in [0, 1]
    assert 0 <= results[0]["similarity"] <= 1
    assert 0 <= results[1]["similarity"] <= 1


@patch("valence.core.embeddings.generate_embedding")
@patch("valence.core.retrieval._has_active_contentions")
@patch("valence.core.retrieval._build_provenance_summary")
@patch("valence.core.retrieval.get_cursor")
def test_hybrid_retrieval_embedding_fallback(
    mock_get_cursor, mock_prov, mock_cont, mock_gen_embed
):
    """Test fallback to text-only when embedding generation fails."""
    mock_rows = [
        {
            "id": "123e4567-e89b-12d3-a456-426614174004",
            "content": "Text match article",
            "title": "Fallback Test",
            "status": "active",
            "superseded_by_id": None,
            "confidence_source": 0.8,
            "confidence_method": 0.9,
            "confidence_consistency": 1.0,
            "confidence_freshness": 1.0,
            "confidence_corroboration": 0.7,
            "confidence_applicability": 0.9,
            "created_at": datetime.now(UTC),
            "text_score": 0.75,
            "vec_score": 0.0,
            "text_rank": 1,
            "vec_rank": 1000,
            "rrf_score": 0.016393,  # 1/(60+1) ≈ 0.016393
            "content_tsv": None,
            "embedding": None,
        }
    ]

    mock_cursor = MockCursor(fetchall_return=mock_rows)
    mock_get_cursor.return_value.__enter__.return_value = mock_cursor
    # Simulate embedding generation failure
    mock_gen_embed.side_effect = Exception("OpenAI API error")
    mock_prov.return_value = {"source_count": 1, "relationship_types": ["cited"]}
    mock_cont.return_value = False

    results = _search_articles_sync("test query", limit=10)

    # Should still get results from text-only search
    assert len(results) == 1
    assert results[0]["id"] == "123e4567-e89b-12d3-a456-426614174004"
    assert results[0]["vec_score"] == 0.0
    assert results[0]["text_score"] > 0.7


@patch("valence.core.embeddings.generate_embedding")
@patch("valence.core.retrieval.get_cursor")
def test_sources_hybrid_retrieval(mock_get_cursor, mock_gen_embed):
    """Test hybrid retrieval for ungrouped sources."""
    mock_rows = [
        {
            "id": "223e4567-e89b-12d3-a456-426614174000",
            "type": "article",
            "title": "Ungrouped Source",
            "url": "https://example.com/article",
            "content": "Source content",
            "reliability": 0.8,
            "fingerprint": "abc123",
            "created_at": datetime.now(UTC),
            "vec_rank": 1,
            "vec_score": 0.89,
            "text_rank": 2,
            "text_score": 0.65,
            "rrf_score": 0.03226,  # 1/(60+1) + 1/(60+2)
        }
    ]

    mock_cursor = MockCursor(fetchall_return=mock_rows)
    mock_get_cursor.return_value.__enter__.return_value = mock_cursor
    mock_gen_embed.return_value = [0.1] * 1536

    results = _search_ungrouped_sources_sync("test query", limit=10)

    assert len(results) == 1
    assert results[0]["id"] == "223e4567-e89b-12d3-a456-426614174000"
    assert results[0]["type"] == "source"
    assert results[0]["reliability"] == 0.8
    assert results[0]["vec_score"] > 0.85
    assert results[0]["text_score"] > 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

