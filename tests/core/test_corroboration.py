"""Tests for valence.core.corroboration module."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch, call
from uuid import UUID, uuid4

import pytest

from valence.core.corroboration import (
    CORROBORATION_FACTOR,
    CORROBORATION_THRESHOLD,
    CorroborationInfo,
    CorroborationResult,
    add_corroboration,
    calculate_corroboration_confidence,
    check_corroboration,
    get_corroboration,
    get_most_corroborated_beliefs,
    process_incoming_belief_corroboration,
)


# ============================================================================
# Constants Tests
# ============================================================================

class TestConstants:
    """Tests for module constants."""

    def test_corroboration_threshold_is_high(self):
        """Corroboration threshold should be high (0.9) for semantic similarity."""
        assert CORROBORATION_THRESHOLD == 0.9

    def test_corroboration_factor_is_reasonable(self):
        """Corroboration factor should be positive and less than 1."""
        assert 0 < CORROBORATION_FACTOR < 1
        assert CORROBORATION_FACTOR == 0.3


# ============================================================================
# calculate_corroboration_confidence Tests
# ============================================================================

class TestCalculateCorroborationConfidence:
    """Tests for calculate_corroboration_confidence function."""

    def test_zero_sources_returns_zero(self):
        """Zero sources should give zero confidence."""
        assert calculate_corroboration_confidence(0) == 0.0

    def test_negative_sources_returns_zero(self):
        """Negative count should give zero confidence."""
        assert calculate_corroboration_confidence(-1) == 0.0
        assert calculate_corroboration_confidence(-100) == 0.0

    def test_one_source_confidence(self):
        """One source should give ~0.23 confidence."""
        result = calculate_corroboration_confidence(1)
        # 1 - (1 / (1 + 1 * 0.3)) = 1 - (1 / 1.3) ≈ 0.2308
        expected = 1.0 - (1.0 / (1.0 + 1 * CORROBORATION_FACTOR))
        assert abs(result - expected) < 0.001
        assert abs(result - 0.2308) < 0.01

    def test_two_sources_confidence(self):
        """Two sources should give ~0.38 confidence."""
        result = calculate_corroboration_confidence(2)
        expected = 1.0 - (1.0 / (1.0 + 2 * CORROBORATION_FACTOR))
        assert abs(result - expected) < 0.001
        assert abs(result - 0.375) < 0.01

    def test_five_sources_confidence(self):
        """Five sources should give ~0.60 confidence."""
        result = calculate_corroboration_confidence(5)
        expected = 1.0 - (1.0 / (1.0 + 5 * CORROBORATION_FACTOR))
        assert abs(result - expected) < 0.001
        assert abs(result - 0.60) < 0.01

    def test_ten_sources_confidence(self):
        """Ten sources should give ~0.75 confidence."""
        result = calculate_corroboration_confidence(10)
        expected = 1.0 - (1.0 / (1.0 + 10 * CORROBORATION_FACTOR))
        assert abs(result - expected) < 0.001
        assert abs(result - 0.75) < 0.01

    def test_confidence_is_monotonic(self):
        """More sources should always give higher confidence."""
        prev = 0.0
        for count in range(1, 100):
            current = calculate_corroboration_confidence(count)
            assert current > prev
            prev = current

    def test_confidence_approaches_one_asymptotically(self):
        """Confidence should approach 1.0 but never exceed it."""
        for count in [100, 1000, 10000]:
            result = calculate_corroboration_confidence(count)
            assert result < 1.0
            assert result > 0.95

    def test_formula_matches_documented(self):
        """Verify formula matches docstring: 1 - (1 / (1 + count * factor))."""
        for count in [0, 1, 2, 5, 10, 50, 100]:
            if count <= 0:
                assert calculate_corroboration_confidence(count) == 0.0
            else:
                expected = 1.0 - (1.0 / (1.0 + count * CORROBORATION_FACTOR))
                assert calculate_corroboration_confidence(count) == expected


# ============================================================================
# CorroborationResult Tests
# ============================================================================

class TestCorroborationResult:
    """Tests for CorroborationResult dataclass."""

    def test_create_corroborated_result(self):
        """Create a corroborated result with all fields."""
        belief_id = uuid4()
        result = CorroborationResult(
            corroborated=True,
            existing_belief_id=belief_id,
            similarity=0.95,
            source_did="did:example:source123",
            is_new_source=True,
        )
        
        assert result.corroborated is True
        assert result.existing_belief_id == belief_id
        assert result.similarity == 0.95
        assert result.source_did == "did:example:source123"
        assert result.is_new_source is True

    def test_create_non_corroborated_result(self):
        """Create a non-corroborated result."""
        result = CorroborationResult(
            corroborated=False,
            existing_belief_id=None,
            similarity=0.5,
            source_did="did:example:source456",
            is_new_source=False,
        )
        
        assert result.corroborated is False
        assert result.existing_belief_id is None
        assert result.similarity == 0.5

    def test_to_dict_with_belief_id(self):
        """to_dict should serialize UUID to string."""
        belief_id = uuid4()
        result = CorroborationResult(
            corroborated=True,
            existing_belief_id=belief_id,
            similarity=0.92,
            source_did="did:example:abc",
            is_new_source=True,
        )
        
        d = result.to_dict()
        
        assert d["corroborated"] is True
        assert d["existing_belief_id"] == str(belief_id)
        assert d["similarity"] == 0.92
        assert d["source_did"] == "did:example:abc"
        assert d["is_new_source"] is True

    def test_to_dict_without_belief_id(self):
        """to_dict should handle None belief_id."""
        result = CorroborationResult(
            corroborated=False,
            existing_belief_id=None,
            similarity=0.3,
            source_did="did:example:xyz",
            is_new_source=False,
        )
        
        d = result.to_dict()
        
        assert d["existing_belief_id"] is None

    def test_to_dict_is_json_serializable(self):
        """to_dict output should be JSON serializable."""
        import json
        
        result = CorroborationResult(
            corroborated=True,
            existing_belief_id=uuid4(),
            similarity=0.95,
            source_did="did:example:test",
            is_new_source=True,
        )
        
        # Should not raise
        json_str = json.dumps(result.to_dict())
        assert isinstance(json_str, str)


# ============================================================================
# CorroborationInfo Tests
# ============================================================================

class TestCorroborationInfo:
    """Tests for CorroborationInfo dataclass."""

    def test_create_corroboration_info(self):
        """Create corroboration info with all fields."""
        belief_id = uuid4()
        sources = [
            {"source_did": "did:a", "similarity": 0.95, "timestamp": "2024-01-01"},
            {"source_did": "did:b", "similarity": 0.92, "timestamp": "2024-01-02"},
        ]
        
        info = CorroborationInfo(
            belief_id=belief_id,
            corroboration_count=2,
            confidence_corroboration=0.375,
            sources=sources,
        )
        
        assert info.belief_id == belief_id
        assert info.corroboration_count == 2
        assert info.confidence_corroboration == 0.375
        assert len(info.sources) == 2
        assert info.sources[0]["source_did"] == "did:a"

    def test_create_empty_info(self):
        """Create info with no corroboration."""
        belief_id = uuid4()
        info = CorroborationInfo(
            belief_id=belief_id,
            corroboration_count=0,
            confidence_corroboration=0.0,
            sources=[],
        )
        
        assert info.corroboration_count == 0
        assert info.sources == []

    def test_to_dict(self):
        """to_dict should serialize properly."""
        belief_id = uuid4()
        sources = [{"source_did": "did:test", "similarity": 0.9}]
        
        info = CorroborationInfo(
            belief_id=belief_id,
            corroboration_count=1,
            confidence_corroboration=0.23,
            sources=sources,
        )
        
        d = info.to_dict()
        
        assert d["belief_id"] == str(belief_id)
        assert d["corroboration_count"] == 1
        assert d["confidence_corroboration"] == 0.23
        assert d["sources"] == sources

    def test_to_dict_is_json_serializable(self):
        """to_dict output should be JSON serializable."""
        import json
        
        info = CorroborationInfo(
            belief_id=uuid4(),
            corroboration_count=3,
            confidence_corroboration=0.5,
            sources=[{"did": "test"}],
        )
        
        json_str = json.dumps(info.to_dict())
        assert isinstance(json_str, str)


# ============================================================================
# check_corroboration Tests
# ============================================================================

class TestCheckCorroboration:
    """Tests for check_corroboration function."""

    @patch("valence.core.corroboration.get_cursor")
    @patch("valence.embeddings.service.generate_embedding")
    @patch("valence.embeddings.service.vector_to_pgvector")
    def test_generates_embedding_when_not_provided(
        self, mock_vector, mock_embed, mock_cursor
    ):
        """Should generate embedding if not provided."""
        mock_embed.return_value = [0.1] * 1536
        mock_vector.return_value = "[0.1,...]"
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        check_corroboration("test content", "did:source:1")
        
        mock_embed.assert_called_once_with("test content")

    @patch("valence.core.corroboration.get_cursor")
    @patch("valence.embeddings.service.vector_to_pgvector")
    def test_uses_provided_embedding(self, mock_vector, mock_cursor):
        """Should use provided embedding without generating."""
        embedding = [0.5] * 1536
        mock_vector.return_value = "[0.5,...]"
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        with patch("valence.embeddings.service.generate_embedding") as mock_gen:
            check_corroboration("test", "did:s", content_embedding=embedding)
            mock_gen.assert_not_called()

    @patch("valence.core.corroboration.get_cursor")
    @patch("valence.embeddings.service.vector_to_pgvector")
    def test_returns_none_when_no_beliefs_exist(self, mock_vector, mock_cursor):
        """Should return None if no beliefs in database."""
        mock_vector.return_value = "[0.1,...]"
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = check_corroboration("test", "did:s", content_embedding=[0.1] * 10)
        
        assert result is None

    @patch("valence.core.corroboration.get_cursor")
    @patch("valence.embeddings.service.vector_to_pgvector")
    def test_returns_none_when_similarity_below_threshold(
        self, mock_vector, mock_cursor
    ):
        """Should return None if similarity below threshold."""
        mock_vector.return_value = "[0.1,...]"
        
        belief_id = uuid4()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "content": "similar content",
            "corroborating_sources": [],
            "similarity": 0.85,  # Below 0.9 threshold
        }
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = check_corroboration("test", "did:s", content_embedding=[0.1] * 10)
        
        assert result is None

    @patch("valence.core.corroboration.get_cursor")
    @patch("valence.embeddings.service.vector_to_pgvector")
    def test_returns_result_when_similarity_above_threshold(
        self, mock_vector, mock_cursor
    ):
        """Should return CorroborationResult if similarity >= threshold."""
        mock_vector.return_value = "[0.1,...]"
        
        belief_id = uuid4()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "content": "matching content",
            "corroborating_sources": [],
            "similarity": 0.95,
        }
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = check_corroboration("test", "did:source:1", content_embedding=[0.1] * 10)
        
        assert result is not None
        assert result.corroborated is True
        assert result.existing_belief_id == belief_id
        assert result.similarity == 0.95
        assert result.source_did == "did:source:1"
        assert result.is_new_source is True

    @patch("valence.core.corroboration.get_cursor")
    @patch("valence.embeddings.service.vector_to_pgvector")
    def test_detects_existing_source(self, mock_vector, mock_cursor):
        """Should detect when source already corroborated."""
        mock_vector.return_value = "[0.1,...]"
        
        belief_id = uuid4()
        existing_sources = [
            {"source_did": "did:existing:source", "similarity": 0.92}
        ]
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "content": "content",
            "corroborating_sources": existing_sources,
            "similarity": 0.95,
        }
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = check_corroboration(
            "test", "did:existing:source", content_embedding=[0.1] * 10
        )
        
        assert result is not None
        assert result.is_new_source is False

    @patch("valence.core.corroboration.get_cursor")
    @patch("valence.embeddings.service.vector_to_pgvector")
    def test_handles_none_corroborating_sources(self, mock_vector, mock_cursor):
        """Should handle None corroborating_sources gracefully."""
        mock_vector.return_value = "[0.1,...]"
        
        belief_id = uuid4()
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "content": "content",
            "corroborating_sources": None,  # NULL in DB
            "similarity": 0.95,
        }
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = check_corroboration("test", "did:s", content_embedding=[0.1] * 10)
        
        assert result is not None
        assert result.is_new_source is True

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_none_on_exception(self, mock_cursor):
        """Should return None and log warning on exception."""
        mock_cursor.side_effect = Exception("Database error")
        
        with patch("valence.core.corroboration.logger") as mock_logger:
            result = check_corroboration("test", "did:s", content_embedding=[0.1] * 10)
            
            assert result is None
            mock_logger.warning.assert_called_once()


# ============================================================================
# add_corroboration Tests
# ============================================================================

class TestAddCorroboration:
    """Tests for add_corroboration function."""

    @patch("valence.core.corroboration.get_cursor")
    def test_adds_corroboration_successfully(self, mock_cursor):
        """Should add corroboration and return True."""
        belief_id = uuid4()
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"added": True}
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        with patch("valence.core.corroboration.logger"):
            result = add_corroboration(belief_id, "did:source:1", 0.95)
        
        assert result is True
        mock_cur.execute.assert_called_once()
        args = mock_cur.execute.call_args[0]
        assert "add_corroborating_source" in args[0]
        assert args[1] == (belief_id, "did:source:1", 0.95, True)

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_false_when_source_exists(self, mock_cursor):
        """Should return False if source already exists."""
        belief_id = uuid4()
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"added": False}
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = add_corroboration(belief_id, "did:s", 0.95)
        
        assert result is False

    @patch("valence.core.corroboration.get_cursor")
    def test_respects_boost_confidence_flag(self, mock_cursor):
        """Should pass boost_confidence to SQL function."""
        belief_id = uuid4()
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"added": True}
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        with patch("valence.core.corroboration.logger"):
            add_corroboration(belief_id, "did:s", 0.9, boost_confidence=False)
        
        args = mock_cur.execute.call_args[0]
        assert args[1][3] is False

    @patch("valence.core.corroboration.get_cursor")
    def test_handles_none_row_result(self, mock_cursor):
        """Should return False if fetchone returns None."""
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = add_corroboration(uuid4(), "did:s", 0.9)
        
        assert result is False

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_false_on_exception(self, mock_cursor):
        """Should return False and log warning on exception."""
        mock_cursor.side_effect = Exception("DB error")
        
        with patch("valence.core.corroboration.logger") as mock_logger:
            result = add_corroboration(uuid4(), "did:s", 0.9)
            
            assert result is False
            mock_logger.warning.assert_called_once()

    @patch("valence.core.corroboration.get_cursor")
    def test_logs_on_successful_add(self, mock_cursor):
        """Should log info when corroboration added."""
        belief_id = uuid4()
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"added": True}
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        with patch("valence.core.corroboration.logger") as mock_logger:
            add_corroboration(belief_id, "did:source:test", 0.95)
            
            mock_logger.info.assert_called_once()
            log_msg = mock_logger.info.call_args[0][0]
            assert str(belief_id) in log_msg
            assert "did:source:test" in log_msg
            assert "0.950" in log_msg


# ============================================================================
# get_corroboration Tests
# ============================================================================

class TestGetCorroboration:
    """Tests for get_corroboration function."""

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_corroboration_info(self, mock_cursor):
        """Should return CorroborationInfo for existing belief."""
        belief_id = uuid4()
        sources = [{"source_did": "did:a"}]
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "corroboration_count": 3,
            "confidence_corroboration": 0.5,
            "corroborating_sources": sources,
        }
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = get_corroboration(belief_id)
        
        assert result is not None
        assert result.belief_id == belief_id
        assert result.corroboration_count == 3
        assert result.confidence_corroboration == 0.5
        assert result.sources == sources

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_none_for_missing_belief(self, mock_cursor):
        """Should return None if belief doesn't exist."""
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = get_corroboration(uuid4())
        
        assert result is None

    @patch("valence.core.corroboration.get_cursor")
    def test_handles_null_values(self, mock_cursor):
        """Should handle NULL values from database."""
        belief_id = uuid4()
        
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "corroboration_count": None,
            "confidence_corroboration": None,
            "corroborating_sources": None,
        }
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = get_corroboration(belief_id)
        
        assert result is not None
        assert result.corroboration_count == 0
        assert result.confidence_corroboration == 0.0
        assert result.sources == []

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_none_on_exception(self, mock_cursor):
        """Should return None and log on exception."""
        mock_cursor.side_effect = Exception("DB error")
        
        with patch("valence.core.corroboration.logger") as mock_logger:
            result = get_corroboration(uuid4())
            
            assert result is None
            mock_logger.warning.assert_called_once()


# ============================================================================
# process_incoming_belief_corroboration Tests
# ============================================================================

class TestProcessIncomingBeliefCorroboration:
    """Tests for process_incoming_belief_corroboration function."""

    @patch("valence.core.corroboration.check_corroboration")
    def test_returns_none_when_no_corroboration(self, mock_check):
        """Should return None if no corroboration found."""
        mock_check.return_value = None
        
        result = process_incoming_belief_corroboration("content", "did:s")
        
        assert result is None
        mock_check.assert_called_once_with("content", "did:s", None)

    @patch("valence.core.corroboration.check_corroboration")
    def test_returns_none_when_not_corroborated(self, mock_check):
        """Should return None if result.corroborated is False."""
        mock_check.return_value = CorroborationResult(
            corroborated=False,
            existing_belief_id=None,
            similarity=0.5,
            source_did="did:s",
            is_new_source=False,
        )
        
        result = process_incoming_belief_corroboration("content", "did:s")
        
        assert result is None

    @patch("valence.core.corroboration.add_corroboration")
    @patch("valence.core.corroboration.check_corroboration")
    def test_adds_corroboration_for_new_source(self, mock_check, mock_add):
        """Should add corroboration when new source detected."""
        belief_id = uuid4()
        mock_check.return_value = CorroborationResult(
            corroborated=True,
            existing_belief_id=belief_id,
            similarity=0.95,
            source_did="did:new:source",
            is_new_source=True,
        )
        mock_add.return_value = True
        
        result = process_incoming_belief_corroboration("content", "did:new:source")
        
        assert result is not None
        assert result.corroborated is True
        mock_add.assert_called_once_with(
            belief_id=belief_id,
            source_did="did:new:source",
            similarity=0.95,
            boost_confidence=True,
        )

    @patch("valence.core.corroboration.add_corroboration")
    @patch("valence.core.corroboration.check_corroboration")
    def test_skips_add_for_existing_source(self, mock_check, mock_add):
        """Should not add corroboration for existing source."""
        mock_check.return_value = CorroborationResult(
            corroborated=True,
            existing_belief_id=uuid4(),
            similarity=0.95,
            source_did="did:existing",
            is_new_source=False,  # Already corroborated
        )
        
        result = process_incoming_belief_corroboration("content", "did:existing")
        
        assert result is not None
        mock_add.assert_not_called()

    @patch("valence.core.corroboration.add_corroboration")
    @patch("valence.core.corroboration.check_corroboration")
    def test_updates_is_new_source_on_add_failure(self, mock_check, mock_add):
        """Should set is_new_source=False if add fails."""
        belief_id = uuid4()
        mock_check.return_value = CorroborationResult(
            corroborated=True,
            existing_belief_id=belief_id,
            similarity=0.95,
            source_did="did:new",
            is_new_source=True,
        )
        mock_add.return_value = False  # Add failed
        
        result = process_incoming_belief_corroboration("content", "did:new")
        
        assert result is not None
        assert result.is_new_source is False  # Updated

    @patch("valence.core.corroboration.check_corroboration")
    def test_passes_embedding_to_check(self, mock_check):
        """Should pass embedding to check_corroboration."""
        mock_check.return_value = None
        embedding = [0.1, 0.2, 0.3]
        
        process_incoming_belief_corroboration("content", "did:s", embedding)
        
        mock_check.assert_called_once_with("content", "did:s", embedding)


# ============================================================================
# get_most_corroborated_beliefs Tests
# ============================================================================

class TestGetMostCorroboratedBeliefs:
    """Tests for get_most_corroborated_beliefs function."""

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_beliefs_list(self, mock_cursor):
        """Should return list of belief dicts."""
        belief_id = uuid4()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "id": belief_id,
                "content": "Test belief",
                "corroboration_count": 5,
                "confidence_corroboration": 0.6,
                "corroborating_sources": [{"did": "a"}],
                "domain_path": ["test", "domain"],
                "created_at": datetime(2024, 1, 15, 12, 0, 0),
            }
        ]
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = get_most_corroborated_beliefs()
        
        assert len(result) == 1
        assert result[0]["id"] == str(belief_id)
        assert result[0]["content"] == "Test belief"
        assert result[0]["corroboration_count"] == 5
        assert result[0]["confidence_corroboration"] == 0.6
        assert result[0]["sources"] == [{"did": "a"}]
        assert result[0]["domain_path"] == ["test", "domain"]
        assert result[0]["created_at"] == "2024-01-15T12:00:00"

    @patch("valence.core.corroboration.get_cursor")
    def test_respects_limit_parameter(self, mock_cursor):
        """Should pass limit to query."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        get_most_corroborated_beliefs(limit=5)
        
        query_params = mock_cur.execute.call_args[0][1]
        assert 5 in query_params

    @patch("valence.core.corroboration.get_cursor")
    def test_respects_min_count_parameter(self, mock_cursor):
        """Should filter by min_count."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        get_most_corroborated_beliefs(min_count=3)
        
        query_params = mock_cur.execute.call_args[0][1]
        assert 3 in query_params

    @patch("valence.core.corroboration.get_cursor")
    def test_applies_domain_filter(self, mock_cursor):
        """Should filter by domain when provided."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        get_most_corroborated_beliefs(domain_filter=["science", "physics"])
        
        query = mock_cur.execute.call_args[0][0]
        params = mock_cur.execute.call_args[0][1]
        
        assert "domain_path &&" in query
        assert ["science", "physics"] in params

    @patch("valence.core.corroboration.get_cursor")
    def test_handles_null_values_in_results(self, mock_cursor):
        """Should handle NULL values gracefully."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [
            {
                "id": uuid4(),
                "content": "Test",
                "corroboration_count": 1,
                "confidence_corroboration": 0.2,
                "corroborating_sources": None,  # NULL
                "domain_path": None,  # NULL
                "created_at": datetime(2024, 1, 1),
            }
        ]
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = get_most_corroborated_beliefs()
        
        assert result[0]["sources"] == []
        assert result[0]["domain_path"] == []

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_empty_list_on_no_results(self, mock_cursor):
        """Should return empty list when no beliefs found."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        result = get_most_corroborated_beliefs()
        
        assert result == []

    @patch("valence.core.corroboration.get_cursor")
    def test_returns_empty_list_on_exception(self, mock_cursor):
        """Should return empty list and log on exception."""
        mock_cursor.side_effect = Exception("DB error")
        
        with patch("valence.core.corroboration.logger") as mock_logger:
            result = get_most_corroborated_beliefs()
            
            assert result == []
            mock_logger.warning.assert_called_once()

    @patch("valence.core.corroboration.get_cursor")
    def test_orders_by_count_then_date(self, mock_cursor):
        """Should order by corroboration_count DESC, then created_at DESC."""
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = []
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=None)
        
        get_most_corroborated_beliefs()
        
        query = mock_cur.execute.call_args[0][0]
        assert "ORDER BY corroboration_count DESC, created_at DESC" in query


# ============================================================================
# Integration-style Tests (still mocked, but test workflows)
# ============================================================================

class TestCorroborationWorkflow:
    """Tests for complete corroboration workflows."""

    @patch("valence.core.corroboration.add_corroboration")
    @patch("valence.core.corroboration.check_corroboration")
    def test_full_corroboration_workflow(self, mock_check, mock_add):
        """Test the complete workflow of receiving and processing corroboration."""
        belief_id = uuid4()
        
        # Simulate finding a matching belief
        mock_check.return_value = CorroborationResult(
            corroborated=True,
            existing_belief_id=belief_id,
            similarity=0.92,
            source_did="did:peer:node123",
            is_new_source=True,
        )
        mock_add.return_value = True
        
        # Process the incoming belief
        result = process_incoming_belief_corroboration(
            content="The Earth orbits the Sun",
            source_did="did:peer:node123",
            content_embedding=[0.1] * 1536,
        )
        
        # Verify the workflow
        assert result is not None
        assert result.corroborated is True
        assert result.existing_belief_id == belief_id
        
        # Verify corroboration was added
        mock_add.assert_called_once_with(
            belief_id=belief_id,
            source_did="did:peer:node123",
            similarity=0.92,
            boost_confidence=True,
        )

    def test_confidence_increases_with_corroboration(self):
        """Confidence should increase as more sources corroborate."""
        confidences = []
        for count in range(0, 11):
            conf = calculate_corroboration_confidence(count)
            confidences.append(conf)
        
        # Verify monotonic increase
        for i in range(1, len(confidences)):
            if i > 0:
                assert confidences[i] > confidences[i-1]
        
        # Verify reasonable values
        assert confidences[0] == 0.0
        assert 0.2 < confidences[1] < 0.3
        assert 0.7 < confidences[10] < 0.8
