"""Tests for trust registry CRUD operations.

Tests cover:
- get_node_trust / save_node_trust - node trust retrieval and persistence
- get_node - federation node retrieval
- get_user_trust_preference / set_user_preference - user preferences
- annotate_belief / get_belief_trust_adjustments - belief annotations
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from typing import Generator
from unittest.mock import MagicMock, patch
from uuid import uuid4, UUID

import pytest

from valence.federation.trust_registry import TrustRegistry
from valence.federation.models import (
    TrustPreference,
    AnnotationType,
)


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

    with patch("valence.federation.trust_registry.get_cursor", _mock_get_cursor):
        yield mock_cursor


@pytest.fixture
def trust_registry():
    """Create a TrustRegistry instance."""
    return TrustRegistry()


@pytest.fixture
def sample_node_trust_row():
    """Create a sample node trust row."""
    def _factory(
        id: UUID | None = None,
        node_id: UUID | None = None,
        **kwargs
    ):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "node_id": node_id or uuid4(),
            "trust": json.dumps(kwargs.get("trust", {"overall": 0.5})),
            "beliefs_received": kwargs.get("beliefs_received", 10),
            "beliefs_corroborated": kwargs.get("beliefs_corroborated", 5),
            "beliefs_disputed": kwargs.get("beliefs_disputed", 1),
            "sync_requests_served": kwargs.get("sync_requests_served", 20),
            "aggregation_participations": kwargs.get("aggregation_participations", 3),
            "endorsements_received": kwargs.get("endorsements_received", 2),
            "endorsements_given": kwargs.get("endorsements_given", 4),
            "relationship_started_at": kwargs.get("relationship_started_at", now),
            "last_interaction_at": kwargs.get("last_interaction_at", now),
            "manual_trust_adjustment": kwargs.get("manual_trust_adjustment", 0),
            "adjustment_reason": kwargs.get("adjustment_reason"),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
        }
    return _factory


@pytest.fixture
def sample_federation_node_row():
    """Create a sample federation node row."""
    def _factory(
        id: UUID | None = None,
        did: str | None = None,
        **kwargs
    ):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "did": did or f"did:vkb:web:node-{uuid4().hex[:8]}.example.com",
            "name": kwargs.get("name", "Test Node"),
            "endpoint_url": kwargs.get("endpoint_url", "https://node.example.com/api"),
            "public_key": kwargs.get("public_key", "test-public-key"),
            "status": kwargs.get("status", "active"),
            "capabilities": kwargs.get("capabilities", []),
            "last_seen_at": kwargs.get("last_seen_at", now),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
        }
    return _factory


@pytest.fixture
def sample_user_node_trust_row():
    """Create a sample user node trust row."""
    def _factory(
        id: UUID | None = None,
        node_id: UUID | None = None,
        **kwargs
    ):
        now = datetime.now()
        return {
            "id": id or uuid4(),
            "node_id": node_id or uuid4(),
            "trust_preference": kwargs.get("trust_preference", "automatic"),
            "manual_trust_score": kwargs.get("manual_trust_score"),
            "reason": kwargs.get("reason"),
            "domain_overrides": kwargs.get("domain_overrides", {}),
            "created_at": kwargs.get("created_at", now),
            "modified_at": kwargs.get("modified_at", now),
        }
    return _factory


@pytest.fixture
def sample_belief_annotation_row():
    """Create a sample belief trust annotation row."""
    def _factory(
        id: UUID | None = None,
        belief_id: UUID | None = None,
        **kwargs
    ):
        now = datetime.now()
        # Source node ID needs to be a string for from_row to parse it
        source_node_id = kwargs.get("source_node_id")
        if source_node_id is not None and isinstance(source_node_id, UUID):
            source_node_id = str(source_node_id)
        return {
            "id": id or uuid4(),
            "belief_id": belief_id or uuid4(),
            "type": kwargs.get("type", "corroboration"),
            "source_node_id": source_node_id,
            "confidence_delta": kwargs.get("confidence_delta", 0.1),
            "corroboration_attestation": kwargs.get("corroboration_attestation"),
            "expires_at": kwargs.get("expires_at"),
            "created_at": kwargs.get("created_at", now),
        }
    return _factory


# =============================================================================
# NODE TRUST RETRIEVAL TESTS
# =============================================================================


class TestGetNodeTrust:
    """Tests for get_node_trust method."""

    def test_get_node_trust_success(self, trust_registry, mock_get_cursor, sample_node_trust_row):
        """Test successful node trust retrieval."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_node_trust_row(node_id=node_id)

        result = trust_registry.get_node_trust(node_id)

        assert result is not None
        assert result.node_id == node_id
        mock_get_cursor.execute.assert_called_once()

    def test_get_node_trust_not_found(self, trust_registry, mock_get_cursor):
        """Test node trust not found."""
        mock_get_cursor.fetchone.return_value = None

        result = trust_registry.get_node_trust(uuid4())

        assert result is None

    def test_get_node_trust_db_error(self, trust_registry, mock_get_cursor):
        """Test database error handling."""
        mock_get_cursor.execute.side_effect = Exception("Database error")

        result = trust_registry.get_node_trust(uuid4())

        assert result is None


class TestGetNode:
    """Tests for get_node method."""

    def test_get_node_success(self, trust_registry, mock_get_cursor, sample_federation_node_row):
        """Test successful federation node retrieval."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_federation_node_row(id=node_id)

        result = trust_registry.get_node(node_id)

        assert result is not None
        assert result.id == node_id

    def test_get_node_not_found(self, trust_registry, mock_get_cursor):
        """Test node not found."""
        mock_get_cursor.fetchone.return_value = None

        result = trust_registry.get_node(uuid4())

        assert result is None

    def test_get_node_db_error(self, trust_registry, mock_get_cursor):
        """Test database error handling."""
        mock_get_cursor.execute.side_effect = Exception("Database error")

        result = trust_registry.get_node(uuid4())

        assert result is None


# =============================================================================
# NODE TRUST PERSISTENCE TESTS
# =============================================================================


class TestSaveNodeTrust:
    """Tests for save_node_trust method."""

    def test_save_node_trust_success(self, trust_registry, mock_get_cursor, sample_node_trust_row):
        """Test successful node trust save."""
        node_id = uuid4()
        row = sample_node_trust_row(node_id=node_id)
        mock_get_cursor.fetchone.return_value = row

        # Create a NodeTrust from the row
        from valence.federation.models import NodeTrust
        node_trust = NodeTrust.from_row(row)

        result = trust_registry.save_node_trust(node_trust)

        assert result is not None
        mock_get_cursor.execute.assert_called_once()

    def test_save_node_trust_not_found(self, trust_registry, mock_get_cursor, sample_node_trust_row):
        """Test save when node trust doesn't exist (returns None)."""
        mock_get_cursor.fetchone.return_value = None

        from valence.federation.models import NodeTrust
        node_trust = NodeTrust.from_row(sample_node_trust_row())

        result = trust_registry.save_node_trust(node_trust)

        assert result is None

    def test_save_node_trust_db_error(self, trust_registry, mock_get_cursor, sample_node_trust_row):
        """Test database error handling on save."""
        mock_get_cursor.execute.side_effect = Exception("Database error")

        from valence.federation.models import NodeTrust
        node_trust = NodeTrust.from_row(sample_node_trust_row())

        result = trust_registry.save_node_trust(node_trust)

        assert result is None


# =============================================================================
# USER TRUST PREFERENCE TESTS
# =============================================================================


class TestGetUserTrustPreference:
    """Tests for get_user_trust_preference method."""

    def test_get_user_trust_preference_success(
        self, trust_registry, mock_get_cursor, sample_user_node_trust_row
    ):
        """Test successful user preference retrieval."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_user_node_trust_row(node_id=node_id)

        result = trust_registry.get_user_trust_preference(node_id)

        assert result is not None
        assert result.node_id == node_id

    def test_get_user_trust_preference_not_found(self, trust_registry, mock_get_cursor):
        """Test preference not found."""
        mock_get_cursor.fetchone.return_value = None

        result = trust_registry.get_user_trust_preference(uuid4())

        assert result is None

    def test_get_user_trust_preference_db_error(self, trust_registry, mock_get_cursor):
        """Test database error handling."""
        mock_get_cursor.execute.side_effect = Exception("Database error")

        result = trust_registry.get_user_trust_preference(uuid4())

        assert result is None


class TestSetUserPreference:
    """Tests for set_user_preference method."""

    def test_set_user_preference_success(
        self, trust_registry, mock_get_cursor, sample_user_node_trust_row
    ):
        """Test successful user preference setting."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_user_node_trust_row(
            node_id=node_id,
            trust_preference="elevated"
        )

        result = trust_registry.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.ELEVATED,
            reason="Good track record"
        )

        assert result is not None
        mock_get_cursor.execute.assert_called_once()

    def test_set_user_preference_with_manual_score(
        self, trust_registry, mock_get_cursor, sample_user_node_trust_row
    ):
        """Test setting preference with manual score override."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_user_node_trust_row(
            node_id=node_id,
            manual_trust_score=0.9
        )

        result = trust_registry.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.ELEVATED,
            manual_score=0.9,
            reason="Manually verified"
        )

        assert result is not None

    def test_set_user_preference_with_domain_overrides(
        self, trust_registry, mock_get_cursor, sample_user_node_trust_row
    ):
        """Test setting preference with domain-specific overrides."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_user_node_trust_row(
            node_id=node_id,
            domain_overrides={"tech": "elevated", "finance": "reduced"}
        )

        result = trust_registry.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.AUTOMATIC,
            domain_overrides={"tech": "elevated", "finance": "reduced"}
        )

        assert result is not None

    def test_set_user_preference_blocked(
        self, trust_registry, mock_get_cursor, sample_user_node_trust_row
    ):
        """Test blocking a node."""
        node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_user_node_trust_row(
            node_id=node_id,
            trust_preference="blocked"
        )

        result = trust_registry.set_user_preference(
            node_id=node_id,
            preference=TrustPreference.BLOCKED,
            reason="Spam source"
        )

        assert result is not None

    def test_set_user_preference_db_error(self, trust_registry, mock_get_cursor):
        """Test database error handling."""
        mock_get_cursor.execute.side_effect = Exception("Database error")

        result = trust_registry.set_user_preference(
            node_id=uuid4(),
            preference=TrustPreference.ELEVATED
        )

        assert result is None


# =============================================================================
# BELIEF ANNOTATION TESTS
# =============================================================================


class TestAnnotateBelief:
    """Tests for annotate_belief method."""

    def test_annotate_belief_success(
        self, trust_registry, mock_get_cursor, sample_belief_annotation_row
    ):
        """Test successful belief annotation."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_belief_annotation_row(
            belief_id=belief_id
        )

        result = trust_registry.annotate_belief(
            belief_id=belief_id,
            annotation_type=AnnotationType.CORROBORATION,
            confidence_delta=0.1
        )

        assert result is not None
        mock_get_cursor.execute.assert_called_once()

    def test_annotate_belief_with_source_node(
        self, trust_registry, mock_get_cursor, sample_belief_annotation_row
    ):
        """Test annotation with source node."""
        belief_id = uuid4()
        source_node_id = uuid4()
        mock_get_cursor.fetchone.return_value = sample_belief_annotation_row(
            belief_id=belief_id,
            source_node_id=source_node_id
        )

        result = trust_registry.annotate_belief(
            belief_id=belief_id,
            annotation_type=AnnotationType.CORROBORATION,
            source_node_id=source_node_id,
            confidence_delta=0.15
        )

        assert result is not None

    def test_annotate_belief_with_attestation(
        self, trust_registry, mock_get_cursor, sample_belief_annotation_row
    ):
        """Test annotation with corroboration attestation."""
        belief_id = uuid4()
        attestation = {
            "source_did": "did:vkb:web:example.com",
            "similarity": 0.95,
            "corroborated_at": datetime.now().isoformat()
        }
        mock_get_cursor.fetchone.return_value = sample_belief_annotation_row(
            belief_id=belief_id,
            corroboration_attestation=attestation
        )

        result = trust_registry.annotate_belief(
            belief_id=belief_id,
            annotation_type=AnnotationType.CORROBORATION,
            attestation=attestation,
            confidence_delta=0.2
        )

        assert result is not None

    def test_annotate_belief_with_expiry(
        self, trust_registry, mock_get_cursor, sample_belief_annotation_row
    ):
        """Test annotation with expiry date."""
        belief_id = uuid4()
        expires_at = datetime.now()
        mock_get_cursor.fetchone.return_value = sample_belief_annotation_row(
            belief_id=belief_id,
            expires_at=expires_at
        )

        result = trust_registry.annotate_belief(
            belief_id=belief_id,
            annotation_type=AnnotationType.DISPUTE,
            confidence_delta=-0.1,
            expires_at=expires_at
        )

        assert result is not None

    def test_annotate_belief_db_error(self, trust_registry, mock_get_cursor):
        """Test database error handling."""
        mock_get_cursor.execute.side_effect = Exception("Database error")

        result = trust_registry.annotate_belief(
            belief_id=uuid4(),
            annotation_type=AnnotationType.CORROBORATION,
            confidence_delta=0.1
        )

        assert result is None


class TestGetBeliefTrustAdjustments:
    """Tests for get_belief_trust_adjustments method."""

    def test_get_belief_trust_adjustments_success(self, trust_registry, mock_get_cursor):
        """Test successful trust adjustment retrieval."""
        belief_id = uuid4()
        mock_get_cursor.fetchone.return_value = {"total_delta": 0.25}

        result = trust_registry.get_belief_trust_adjustments(belief_id)

        assert result == 0.25

    def test_get_belief_trust_adjustments_no_annotations(self, trust_registry, mock_get_cursor):
        """Test when no annotations exist."""
        mock_get_cursor.fetchone.return_value = {"total_delta": 0.0}

        result = trust_registry.get_belief_trust_adjustments(uuid4())

        assert result == 0.0

    def test_get_belief_trust_adjustments_null(self, trust_registry, mock_get_cursor):
        """Test when result is None."""
        mock_get_cursor.fetchone.return_value = None

        result = trust_registry.get_belief_trust_adjustments(uuid4())

        assert result == 0.0

    def test_get_belief_trust_adjustments_negative(self, trust_registry, mock_get_cursor):
        """Test negative adjustments (disputes)."""
        mock_get_cursor.fetchone.return_value = {"total_delta": -0.15}

        result = trust_registry.get_belief_trust_adjustments(uuid4())

        assert result == -0.15

    def test_get_belief_trust_adjustments_db_error(self, trust_registry, mock_get_cursor):
        """Test database error handling."""
        mock_get_cursor.execute.side_effect = Exception("Database error")

        result = trust_registry.get_belief_trust_adjustments(uuid4())

        assert result == 0.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestTrustRegistryIntegration:
    """Integration-style tests for TrustRegistry."""

    def test_trust_preference_enum_values(self):
        """Test that TrustPreference enum has expected values."""
        assert TrustPreference.BLOCKED.value == "blocked"
        assert TrustPreference.REDUCED.value == "reduced"
        assert TrustPreference.AUTOMATIC.value == "automatic"
        assert TrustPreference.ELEVATED.value == "elevated"
        assert TrustPreference.ANCHOR.value == "anchor"

    def test_annotation_type_enum_values(self):
        """Test that AnnotationType enum has expected values."""
        assert AnnotationType.CORROBORATION.value == "corroboration"
        assert AnnotationType.DISPUTE.value == "dispute"
        # Add more as defined in the enum

    def test_trust_registry_initialization(self):
        """Test that TrustRegistry can be instantiated."""
        registry = TrustRegistry()
        assert registry is not None

    def test_multiple_operations_same_registry(self, trust_registry, mock_get_cursor):
        """Test multiple operations on same registry instance."""
        mock_get_cursor.fetchone.return_value = None

        # Multiple calls should work
        result1 = trust_registry.get_node_trust(uuid4())
        result2 = trust_registry.get_node(uuid4())
        result3 = trust_registry.get_user_trust_preference(uuid4())

        # All should be None (not found)
        assert result1 is None
        assert result2 is None
        assert result3 is None

        # Should have made 3 queries
        assert mock_get_cursor.execute.call_count == 3
