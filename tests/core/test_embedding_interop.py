"""Tests for embedding federation interop (#356).

Tests cover:
1. EmbeddingCapability defaults and serialization
2. strip_embedding_for_federation removes embedding fields
3. prepare_received_belief_for_embedding extracts content
4. text_similarity computes TF-IDF cosine similarity
5. text_similarity edge cases (empty, identical, disjoint)
6. build_embedding_capability_advertisement format
7. get_embedding_capability reads config
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from valence.core.embedding_interop import (
    EmbeddingCapability,
    build_embedding_capability_advertisement,
    get_embedding_capability,
    prepare_received_belief_for_embedding,
    strip_embedding_for_federation,
    text_similarity,
)


class TestEmbeddingCapability:
    """Test capability dataclass."""

    def test_defaults(self):
        cap = EmbeddingCapability()
        assert cap.model == "BAAI/bge-small-en-v1.5"
        assert cap.dimensions == 384

    def test_to_dict(self):
        cap = EmbeddingCapability()
        d = cap.to_dict()
        assert d["model"] == "BAAI/bge-small-en-v1.5"
        assert d["dimensions"] == 384
        assert d["type_id"] == "bge_small_en_v15"
        assert d["normalization"] == "l2"


class TestGetEmbeddingCapability:
    """Test reading capability from config."""

    def test_default_384(self, clean_env):
        """Test default dimensions when no provider is set (local default)."""
        cap = get_embedding_capability()
        assert cap.dimensions == 384
        assert cap.model == "BAAI/bge-small-en-v1.5"
    
    def test_openai_provider_default(self, clean_env, monkeypatch):
        """Test default dimensions when provider is set to openai."""
        monkeypatch.setenv("VALENCE_EMBEDDING_PROVIDER", "openai")
        cap = get_embedding_capability()
        assert cap.dimensions == 1536
        assert cap.model == "text-embedding-3-small"

    def test_custom_dims(self, clean_env, monkeypatch):
        monkeypatch.setenv("VALENCE_EMBEDDING_DIMS", "1536")
        cap = get_embedding_capability()
        assert cap.dimensions == 1536
        assert "openai" in cap.type_id


class TestStripEmbeddingForFederation:
    """Test embedding stripping."""

    def test_removes_embedding_fields(self):
        belief = {
            "id": "abc",
            "content": "Python is great",
            "embedding": [0.1, 0.2, 0.3],
            "embedding_model": "bge-small",
            "embedding_dims": 384,
            "confidence": {"overall": 0.8},
        }
        stripped = strip_embedding_for_federation(belief)
        assert "embedding" not in stripped
        assert "embedding_model" not in stripped
        assert "embedding_dims" not in stripped
        assert stripped["content"] == "Python is great"
        assert stripped["confidence"] == {"overall": 0.8}

    def test_no_embedding_fields_unchanged(self):
        belief = {"id": "abc", "content": "test"}
        stripped = strip_embedding_for_federation(belief)
        assert stripped == belief

    def test_returns_new_dict(self):
        belief = {"id": "abc", "embedding": [1, 2, 3]}
        stripped = strip_embedding_for_federation(belief)
        assert stripped is not belief


class TestPrepareReceivedBelief:
    """Test content extraction for re-embedding."""

    def test_extracts_content(self):
        belief = {"content": "Hello world", "confidence": {"overall": 0.8}}
        text = prepare_received_belief_for_embedding(belief)
        assert text == "Hello world"

    def test_none_for_missing_content(self):
        belief = {"confidence": {"overall": 0.8}}
        text = prepare_received_belief_for_embedding(belief)
        assert text is None

    def test_none_for_empty_content(self):
        belief = {"content": ""}
        text = prepare_received_belief_for_embedding(belief)
        assert text is None


class TestTextSimilarity:
    """Test TF-IDF text similarity fallback."""

    def test_identical_texts(self):
        score = text_similarity("hello world", "hello world")
        assert score == pytest.approx(1.0)

    def test_similar_texts(self):
        score = text_similarity(
            "Python is a programming language",
            "Python is a great programming language",
        )
        assert score > 0.8

    def test_disjoint_texts(self):
        score = text_similarity("hello world", "foo bar baz")
        assert score == 0.0

    def test_empty_texts(self):
        assert text_similarity("", "") == 0.0
        assert text_similarity("hello", "") == 0.0
        assert text_similarity("", "hello") == 0.0

    def test_case_insensitive(self):
        score1 = text_similarity("Hello World", "hello world")
        assert score1 == pytest.approx(1.0)

    def test_partial_overlap(self):
        score = text_similarity("the quick brown fox", "the slow brown dog")
        assert 0.0 < score < 1.0


class TestBuildCapabilityAdvertisement:
    """Test capability advertisement format."""

    def test_has_vfp_embedding_key(self):
        ad = build_embedding_capability_advertisement()
        assert "vfp:embedding" in ad
        assert "model" in ad["vfp:embedding"]
        assert "dimensions" in ad["vfp:embedding"]
