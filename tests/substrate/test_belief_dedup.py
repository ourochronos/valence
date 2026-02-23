"""Tests for belief deduplication and corroboration.

Covers:
- Exact duplicate detection via content_hash
- Reinforcement of existing beliefs on dedup
- Content hash determinism
- Corroboration count incrementing
- Confidence escalation ladder

NOTE (WU-11): Belief system has been replaced by the v2 articles/sources system.
These tests are deprecated and skipped. See tests/integration/test_v2_integration.py
for v2 equivalents.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "Deprecated: belief system replaced by v2 articles/sources (WU-11)",
    allow_module_level=True,
)

import hashlib
import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.substrate.tools import _content_hash, _reinforce_belief, belief_create


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_cursor():
    """Mock database cursor."""
    cursor = MagicMock()
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_get_cursor(mock_cursor):
    """Mock the get_cursor context manager."""

    @contextmanager
    def _mock_get_cursor(dict_cursor: bool = True) -> Generator:
        yield mock_cursor

    with patch("valence.substrate.tools._common.get_cursor", _mock_get_cursor):
        yield mock_cursor


def _make_belief_row(id=None, content="Test belief", confidence=None, **kwargs):
    now = datetime.now()
    return {
        "id": id or uuid4(),
        "content": content,
        "confidence": json.dumps(confidence or {"overall": 0.7}),
        "domain_path": kwargs.get("domain_path", ["test"]),
        "valid_from": None,
        "valid_until": None,
        "created_at": now,
        "modified_at": now,
        "source_id": None,
        "extraction_method": None,
        "supersedes_id": None,
        "superseded_by_id": None,
        "status": "active",
        "opt_out_federation": False,
        "content_hash": kwargs.get("content_hash"),
        "holder_id": None,
        "version": 1,
        "visibility": "private",
        "share_policy": None,
        "extraction_metadata": None,
        "embedding": None,
    }


# =============================================================================
# TESTS
# =============================================================================


class TestContentHash:
    def test_content_hash_is_deterministic(self):
        """Same input always produces same hash."""
        h1 = _content_hash("Hello world")
        h2 = _content_hash("Hello world")
        assert h1 == h2

    def test_content_hash_normalizes_case(self):
        """Hash is case-insensitive."""
        h1 = _content_hash("Hello World")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_content_hash_normalizes_whitespace(self):
        """Hash strips leading/trailing whitespace."""
        h1 = _content_hash("  Hello world  ")
        h2 = _content_hash("Hello world")
        assert h1 == h2

    def test_content_hash_is_sha256(self):
        """Hash uses SHA-256."""
        expected = hashlib.sha256("hello world".encode()).hexdigest()
        assert _content_hash("Hello World") == expected

    def test_content_hash_different_for_different_content(self):
        """Different content produces different hashes."""
        h1 = _content_hash("belief one")
        h2 = _content_hash("belief two")
        assert h1 != h2


class TestExactDuplicateReinforces:
    def test_exact_duplicate_reinforces_existing(self, mock_get_cursor):
        """When content_hash matches an active belief, reinforce instead of creating."""
        existing_id = uuid4()
        updated_row = _make_belief_row(id=existing_id, confidence={"overall": 0.65, "corroboration": 0.65})

        # fetchone calls in order:
        # 1. SELECT ... WHERE content_hash (hash match) → existing belief
        # 2. SELECT COUNT(*) corroborations → count
        # 3. UPDATE beliefs ... RETURNING * → updated belief row
        mock_get_cursor.fetchone.side_effect = [
            {"id": existing_id, "confidence": {"overall": 0.5}},  # hash match
            {"cnt": 1},  # corroboration count
            updated_row,  # UPDATE RETURNING (INSERT has no fetchone)
        ]

        result = belief_create("Test belief")

        assert result["success"] is True
        assert result["deduplicated"] is True
        assert result["action"] == "reinforced"
        assert result["corroboration_count"] == 2

    def test_novel_content_creates_new_belief(self, mock_get_cursor):
        """When no hash match and no fuzzy match, create a new belief normally."""
        new_id = uuid4()
        belief_row = _make_belief_row(id=new_id)

        # fetchone calls:
        # 1. SELECT ... WHERE content_hash → None (no hash match)
        # 2. SELECT ... cosine similarity → None (no fuzzy match)
        # 3. INSERT INTO beliefs ... RETURNING * → new belief row
        mock_get_cursor.fetchone.side_effect = [
            None,  # no hash match
            None,  # no fuzzy match (embedding cosine check)
            belief_row,  # INSERT RETURNING
        ]

        result = belief_create("Completely new belief")

        assert result["success"] is True
        assert "deduplicated" not in result


class TestCorroborationCount:
    def test_corroboration_count_increments(self, mock_get_cursor):
        """Each dedup bumps corroboration count."""
        existing_id = uuid4()
        updated_row = _make_belief_row(id=existing_id, confidence={"overall": 0.80, "corroboration": 0.80})

        # count is 2, so new count becomes 3
        mock_get_cursor.fetchone.side_effect = [
            {"id": existing_id, "confidence": {"overall": 0.65}},  # hash match
            {"cnt": 2},  # existing corroboration count
            updated_row,  # UPDATE RETURNING
        ]

        result = belief_create("Test belief")

        assert result["corroboration_count"] == 3


class TestConfidenceEscalation:
    def test_confidence_escalation_ladder(self):
        """Test the corroboration → confidence mapping."""
        from valence.core.curation import corroboration_confidence

        assert corroboration_confidence(0) == 0.50
        assert corroboration_confidence(1) == 0.50
        assert corroboration_confidence(2) == 0.65
        assert corroboration_confidence(3) == 0.80
        assert corroboration_confidence(5) == 0.80
        assert corroboration_confidence(100) == 0.80

    def test_reinforce_uses_escalation_ladder(self, mock_get_cursor):
        """_reinforce_belief updates confidence per the escalation ladder."""
        existing_id = uuid4()
        updated_row = _make_belief_row(id=existing_id, confidence={"overall": 0.65, "corroboration": 0.65})

        # _reinforce_belief fetchone calls:
        # 1. SELECT COUNT(*) → count
        # 2. UPDATE ... RETURNING * → updated row
        mock_get_cursor.fetchone.side_effect = [
            {"cnt": 1},  # existing count
            updated_row,  # UPDATE RETURNING
        ]

        result = _reinforce_belief(mock_get_cursor, existing_id, {"overall": 0.5})

        assert result["success"] is True
        assert result["deduplicated"] is True
        assert result["corroboration_count"] == 2
