"""Tests for federation embedding standard.

Tests the canonical embedding format for VFP (Valence Federation Protocol).
"""

from __future__ import annotations

import math

import pytest

from valence.core.federation_embedding import (
    FEDERATION_EMBEDDING_DIMS,
    FEDERATION_EMBEDDING_MODEL,
    FEDERATION_EMBEDDING_TYPE,
    FEDERATION_EMBEDDING_VERSION,
    get_federation_standard,
    is_federation_compatible,
    validate_federation_embedding,
    validate_incoming_belief_embedding,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def valid_l2_embedding() -> list[float]:
    """Create a valid L2-normalized 384-dim embedding."""
    # Create a vector that will be L2 normalized
    raw = [1.0 / math.sqrt(FEDERATION_EMBEDDING_DIMS)] * FEDERATION_EMBEDDING_DIMS
    return raw


@pytest.fixture
def unnormalized_embedding() -> list[float]:
    """Create a non-normalized 384-dim embedding."""
    return [0.1] * FEDERATION_EMBEDDING_DIMS


@pytest.fixture
def wrong_dims_embedding() -> list[float]:
    """Create an embedding with wrong dimensions."""
    return [0.1] * 512  # OpenAI small is 1536, this is just wrong


# =============================================================================
# CONSTANTS TESTS
# =============================================================================


class TestConstants:
    """Test federation embedding constants."""

    def test_model_constant(self):
        """Verify federation model is BGE small."""
        assert FEDERATION_EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"

    def test_dimensions_constant(self):
        """Verify federation dimensions is 384."""
        assert FEDERATION_EMBEDDING_DIMS == 384

    def test_type_constant(self):
        """Verify federation type identifier."""
        assert FEDERATION_EMBEDDING_TYPE == "bge_small_en_v15"

    def test_version_constant(self):
        """Verify federation version."""
        assert FEDERATION_EMBEDDING_VERSION == "1.0"


# =============================================================================
# GET_FEDERATION_STANDARD TESTS
# =============================================================================


class TestGetFederationStandard:
    """Test get_federation_standard function."""

    def test_returns_dict(self):
        """Standard returns a dictionary."""
        result = get_federation_standard()
        assert isinstance(result, dict)

    def test_contains_model(self):
        """Standard includes model name."""
        result = get_federation_standard()
        assert result["model"] == FEDERATION_EMBEDDING_MODEL

    def test_contains_dimensions(self):
        """Standard includes dimensions."""
        result = get_federation_standard()
        assert result["dimensions"] == FEDERATION_EMBEDDING_DIMS

    def test_contains_type(self):
        """Standard includes type identifier."""
        result = get_federation_standard()
        assert result["type"] == FEDERATION_EMBEDDING_TYPE

    def test_contains_normalization(self):
        """Standard specifies L2 normalization."""
        result = get_federation_standard()
        assert result["normalization"] == "L2"

    def test_contains_version(self):
        """Standard includes version."""
        result = get_federation_standard()
        assert result["version"] == FEDERATION_EMBEDDING_VERSION

    def test_all_expected_keys(self):
        """Standard contains exactly the expected keys."""
        result = get_federation_standard()
        expected_keys = {"model", "dimensions", "type", "normalization", "version"}
        assert set(result.keys()) == expected_keys


# =============================================================================
# IS_FEDERATION_COMPATIBLE TESTS
# =============================================================================


class TestIsFederationCompatible:
    """Test is_federation_compatible function."""

    def test_compatible_exact_match(self):
        """Exact type and dimensions match is compatible."""
        assert is_federation_compatible(FEDERATION_EMBEDDING_TYPE, 384) is True

    def test_incompatible_wrong_type(self):
        """Wrong type is not compatible."""
        assert is_federation_compatible("text_embedding_3_small", 384) is False

    def test_incompatible_wrong_dimensions(self):
        """Wrong dimensions is not compatible."""
        assert is_federation_compatible(FEDERATION_EMBEDDING_TYPE, 1536) is False

    def test_incompatible_both_wrong(self):
        """Both wrong type and dimensions is not compatible."""
        assert is_federation_compatible("openai_ada", 1536) is False

    def test_none_type(self):
        """None type is not compatible."""
        assert is_federation_compatible(None, 384) is False

    def test_none_dimensions(self):
        """None dimensions is not compatible."""
        assert is_federation_compatible(FEDERATION_EMBEDDING_TYPE, None) is False

    def test_both_none(self):
        """Both None is not compatible."""
        assert is_federation_compatible(None, None) is False

    def test_empty_string_type(self):
        """Empty string type is not compatible."""
        assert is_federation_compatible("", 384) is False


# =============================================================================
# VALIDATE_FEDERATION_EMBEDDING TESTS
# =============================================================================


class TestValidateFederationEmbedding:
    """Test validate_federation_embedding function."""

    def test_valid_embedding(self, valid_l2_embedding):
        """Valid L2-normalized embedding passes validation."""
        valid, error = validate_federation_embedding(valid_l2_embedding)
        assert valid is True
        assert error is None

    def test_none_embedding(self):
        """None embedding fails validation."""
        valid, error = validate_federation_embedding(None)
        assert valid is False
        assert error == "Embedding is None"

    def test_not_a_list(self):
        """Non-list embedding fails validation."""
        valid, error = validate_federation_embedding("not a list")
        assert valid is False
        assert "must be a list" in error

    def test_dict_not_list(self):
        """Dict embedding fails validation."""
        valid, error = validate_federation_embedding({"key": "value"})
        assert valid is False
        assert "must be a list" in error

    def test_wrong_dimensions_too_few(self):
        """Embedding with too few dimensions fails."""
        embedding = [0.1] * 100
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "384 dimensions" in error
        assert "got 100" in error

    def test_wrong_dimensions_too_many(self):
        """Embedding with too many dimensions fails."""
        embedding = [0.1] * 1536
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "384 dimensions" in error
        assert "got 1536" in error

    def test_empty_embedding(self):
        """Empty embedding fails validation."""
        valid, error = validate_federation_embedding([])
        assert valid is False
        assert "384 dimensions" in error
        assert "got 0" in error

    def test_non_numeric_value(self):
        """Embedding with non-numeric value fails."""
        embedding = [0.1] * 383 + ["string"]
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "not a number" in error

    def test_nan_value(self):
        """Embedding with NaN fails."""
        embedding = [0.1] * 383 + [float("nan")]
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "invalid value" in error

    def test_inf_value(self):
        """Embedding with infinity fails."""
        embedding = [0.1] * 383 + [float("inf")]
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "invalid value" in error

    def test_negative_inf_value(self):
        """Embedding with negative infinity fails."""
        embedding = [0.1] * 383 + [float("-inf")]
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "invalid value" in error

    def test_unnormalized_embedding(self, unnormalized_embedding):
        """Non-L2-normalized embedding fails."""
        valid, error = validate_federation_embedding(unnormalized_embedding)
        assert valid is False
        assert "not L2 normalized" in error

    def test_zero_vector_fails(self):
        """Zero vector (magnitude 0) fails normalization check."""
        embedding = [0.0] * FEDERATION_EMBEDDING_DIMS
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "not L2 normalized" in error

    def test_tuple_is_valid(self, valid_l2_embedding):
        """Tuple is accepted as valid embedding container."""
        embedding_tuple = tuple(valid_l2_embedding)
        valid, error = validate_federation_embedding(embedding_tuple)
        assert valid is True
        assert error is None

    def test_integer_values_accepted(self):
        """Integer values in embedding are accepted."""
        # Create L2 normalized vector with some integers
        val = 1.0 / math.sqrt(FEDERATION_EMBEDDING_DIMS)
        embedding = [val] * FEDERATION_EMBEDDING_DIMS
        # Replace some with equivalent integers (0 is fine)
        embedding[0] = 0
        embedding[1] = 0
        # Recalculate to ensure still normalized
        remaining = FEDERATION_EMBEDDING_DIMS - 2
        new_val = 1.0 / math.sqrt(remaining)
        embedding = [0, 0] + [new_val] * remaining
        valid, error = validate_federation_embedding(embedding)
        assert valid is True

    def test_normalization_tolerance(self):
        """Embeddings within tolerance (±0.01) are valid."""
        # Magnitude of 1.005 should pass
        scale = 1.005
        val = scale / math.sqrt(FEDERATION_EMBEDDING_DIMS)
        embedding = [val] * FEDERATION_EMBEDDING_DIMS
        valid, error = validate_federation_embedding(embedding)
        assert valid is True

    def test_normalization_outside_tolerance(self):
        """Embeddings outside tolerance fail."""
        # Magnitude of 1.02 should fail
        scale = 1.02
        val = scale / math.sqrt(FEDERATION_EMBEDDING_DIMS)
        embedding = [val] * FEDERATION_EMBEDDING_DIMS
        valid, error = validate_federation_embedding(embedding)
        assert valid is False
        assert "not L2 normalized" in error


# =============================================================================
# VALIDATE_INCOMING_BELIEF_EMBEDDING TESTS
# =============================================================================


class TestValidateIncomingBeliefEmbedding:
    """Test validate_incoming_belief_embedding function."""

    def test_no_embedding_is_valid(self):
        """Belief without embedding is valid (optional field)."""
        belief_data = {"content": "Some belief", "source": "test"}
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is True
        assert error is None

    def test_none_embedding_is_valid(self):
        """Belief with None embedding is valid (optional field)."""
        belief_data = {"content": "Some belief", "embedding": None}
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is True
        assert error is None

    def test_valid_embedding_accepted(self, valid_l2_embedding):
        """Belief with valid embedding passes."""
        belief_data = {
            "content": "Test belief",
            "embedding": valid_l2_embedding,
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is True
        assert error is None

    def test_valid_embedding_with_metadata(self, valid_l2_embedding):
        """Belief with valid embedding and correct metadata passes."""
        belief_data = {
            "content": "Test belief",
            "embedding": valid_l2_embedding,
            "embedding_model": FEDERATION_EMBEDDING_MODEL,
            "embedding_dims": FEDERATION_EMBEDDING_DIMS,
            "embedding_type": FEDERATION_EMBEDDING_TYPE,
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is True
        assert error is None

    def test_wrong_model_rejected(self, valid_l2_embedding):
        """Belief with wrong embedding model is rejected."""
        belief_data = {
            "content": "Test belief",
            "embedding": valid_l2_embedding,
            "embedding_model": "text-embedding-3-small",
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is False
        assert "model mismatch" in error
        assert "text-embedding-3-small" in error

    def test_wrong_dims_rejected(self, valid_l2_embedding):
        """Belief with wrong embedding dimensions is rejected."""
        belief_data = {
            "content": "Test belief",
            "embedding": valid_l2_embedding,
            "embedding_dims": 1536,
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is False
        assert "dimensions mismatch" in error
        assert "1536" in error

    def test_wrong_type_rejected(self, valid_l2_embedding):
        """Belief with wrong embedding type is rejected."""
        belief_data = {
            "content": "Test belief",
            "embedding": valid_l2_embedding,
            "embedding_type": "openai_small",
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is False
        assert "type mismatch" in error
        assert "openai_small" in error

    def test_invalid_embedding_vector_rejected(self, unnormalized_embedding):
        """Belief with invalid embedding vector is rejected."""
        belief_data = {
            "content": "Test belief",
            "embedding": unnormalized_embedding,
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is False
        assert "not L2 normalized" in error

    def test_partial_metadata_valid(self, valid_l2_embedding):
        """Belief with only some correct metadata passes."""
        belief_data = {
            "content": "Test belief",
            "embedding": valid_l2_embedding,
            "embedding_model": FEDERATION_EMBEDDING_MODEL,
            # No dims or type provided
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is True

    def test_empty_dict_valid(self):
        """Empty belief dict is valid (embedding is optional)."""
        valid, error = validate_incoming_belief_embedding({})
        assert valid is True
        assert error is None

    def test_wrong_dims_embedding_rejected(self, wrong_dims_embedding):
        """Belief with wrong dimension embedding vector fails."""
        belief_data = {
            "content": "Test belief",
            "embedding": wrong_dims_embedding,
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is False
        assert "384 dimensions" in error

    def test_metadata_mismatch_checked_before_vector(self, valid_l2_embedding):
        """Model metadata mismatch is checked before vector validation."""
        belief_data = {
            "content": "Test belief",
            "embedding": valid_l2_embedding,
            "embedding_model": "wrong_model",
        }
        valid, error = validate_incoming_belief_embedding(belief_data)
        assert valid is False
        # Should fail on model mismatch, not vector validation
        assert "model mismatch" in error
