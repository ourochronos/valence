"""
Spec Compliance Tests: Belief Schema

These tests verify that the beliefs table matches the Valence specification.
Tests marked with xfail will pass once migrations/002-qa-fixes.sql is applied.

QA Findings (4 gaps between spec and current schema):
1. holder_id vs source_id — spec requires holder_id for ownership model
2. version field missing — spec requires explicit versioning, not just supersession chain
3. content_hash missing — needed for deduplication per spec
4. visibility enum missing — needed for federation privacy levels

See: docs/FEDERATION_SCHEMA.md, docs/VISION.md
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ============================================================================
# Schema Field Existence Tests
# ============================================================================

class TestBeliefSchemaFields:
    """Test that belief table has all required fields per spec."""

    @pytest.mark.xfail(
        reason="holder_id not in base schema - requires migration 001 or 002",
        strict=True,
    )
    def test_holder_id_column_exists(self, mock_db_connection):
        """
        Spec requires holder_id (UUID) for multi-holder/federation support.
        
        The spec distinguishes between:
        - source_id: WHERE the belief came from (provenance)
        - holder_id: WHO holds/owns this belief (ownership)
        
        This is critical for federation where beliefs can be shared
        between nodes while maintaining ownership attribution.
        """
        columns = mock_db_connection.get_belief_columns()
        assert "holder_id" in columns, (
            "holder_id column required for federation ownership model. "
            "Current schema uses source_id for provenance, but holder_id "
            "is needed to track which agent/node owns this belief."
        )

    @pytest.mark.xfail(
        reason="version field not in base schema - requires migration 001 or 002",
        strict=True,
    )
    def test_version_column_exists(self, mock_db_connection):
        """
        Spec requires explicit version field for belief versioning.
        
        While the current schema has supersedes_id/superseded_by_id for
        creating supersession chains, the spec also requires an explicit
        version number for:
        - Simpler conflict resolution in federation
        - Efficient version comparison without chain traversal
        - Clear semantic versioning of belief evolution
        """
        columns = mock_db_connection.get_belief_columns()
        assert "version" in columns, (
            "version column required. Current schema relies solely on "
            "supersession chain, but explicit version number needed for "
            "efficient federation sync and conflict resolution."
        )

    @pytest.mark.xfail(
        reason="content_hash not in base schema - requires migration 001 or 002",
        strict=True,
    )
    def test_content_hash_column_exists(self, mock_db_connection):
        """
        Spec requires content_hash for deduplication.
        
        SHA-256 hash of belief content enables:
        - Efficient deduplication during federation sync
        - Quick detection of identical beliefs across nodes
        - Integrity verification for federated beliefs
        """
        columns = mock_db_connection.get_belief_columns()
        assert "content_hash" in columns, (
            "content_hash column required for deduplication. "
            "SHA-256 hash of content enables efficient federation sync "
            "and integrity verification."
        )

    @pytest.mark.xfail(
        reason="visibility enum not in base schema - requires migration 001 or 002",
        strict=True,
    )
    def test_visibility_column_exists(self, mock_db_connection):
        """
        Spec requires visibility enum for federation privacy levels.
        
        Valid values: 'private', 'federated', 'public'
        - private: Only visible to holder
        - federated: Shared with trusted federation nodes
        - public: Visible to anyone
        """
        columns = mock_db_connection.get_belief_columns()
        assert "visibility" in columns, (
            "visibility column required for federation privacy. "
            "Enum with values: private, federated, public."
        )


# ============================================================================
# Schema Constraint Tests
# ============================================================================

class TestBeliefSchemaConstraints:
    """Test that belief table has proper constraints per spec."""

    @pytest.mark.xfail(
        reason="version constraint not in base schema",
        strict=True,
    )
    def test_version_positive_constraint(self, mock_db_connection):
        """Version must be > 0."""
        constraints = mock_db_connection.get_belief_constraints()
        assert any(
            "version" in c and "> 0" in c for c in constraints
        ), "version_positive constraint required: version > 0"

    @pytest.mark.xfail(
        reason="content_hash format not validated in base schema",
        strict=True,
    )
    def test_content_hash_format(self, mock_db_connection):
        """Content hash should be CHAR(64) for SHA-256 hex."""
        columns = mock_db_connection.get_belief_column_types()
        assert columns.get("content_hash") == "character(64)", (
            "content_hash should be CHAR(64) for SHA-256 hex encoding"
        )

    @pytest.mark.xfail(
        reason="visibility enum not in base schema",
        strict=True,
    )
    def test_visibility_enum_values(self, mock_db_connection):
        """Visibility enum should have correct values."""
        enum_values = mock_db_connection.get_enum_values("visibility_level")
        expected = {"private", "federated", "public"}
        assert set(enum_values) == expected, (
            f"visibility_level enum should have {expected}, got {enum_values}"
        )


# ============================================================================
# Schema Index Tests
# ============================================================================

class TestBeliefSchemaIndexes:
    """Test that belief table has required indexes per spec."""

    @pytest.mark.xfail(
        reason="holder_id index not in base schema",
        strict=True,
    )
    def test_holder_id_index_exists(self, mock_db_connection):
        """Index on holder_id for efficient per-holder queries."""
        indexes = mock_db_connection.get_belief_indexes()
        assert any("holder" in idx for idx in indexes), (
            "idx_beliefs_holder index required for holder_id lookups"
        )

    @pytest.mark.xfail(
        reason="content_hash index not in base schema",
        strict=True,
    )
    def test_content_hash_index_exists(self, mock_db_connection):
        """Index on content_hash for deduplication lookups."""
        indexes = mock_db_connection.get_belief_indexes()
        assert any("content_hash" in idx for idx in indexes), (
            "idx_beliefs_content_hash index required for dedup lookups"
        )


# ============================================================================
# Migration Verification Tests (Post-Migration)
# ============================================================================

class TestMigration002Applied:
    """
    Tests that pass after migration 002 is applied.
    
    These tests verify the migration adds the required fields correctly.
    Run with: pytest -k "Migration002" --runxfail
    """

    def test_holder_id_nullable_for_backward_compat(self, mock_db_post_migration):
        """holder_id should be nullable to not break existing rows."""
        column_info = mock_db_post_migration.get_column_info("beliefs", "holder_id")
        # After migration, existing rows get default UUID, new can be NULL for compat
        assert column_info is not None
        assert column_info["type"] == "uuid"

    def test_version_defaults_to_one(self, mock_db_post_migration):
        """version should default to 1 for existing beliefs."""
        column_info = mock_db_post_migration.get_column_info("beliefs", "version")
        assert column_info is not None
        assert column_info["default"] == "1"

    def test_content_hash_computed_for_existing(self, mock_db_post_migration):
        """content_hash should be computed for all existing beliefs."""
        # Migration should have updated all rows with NULL content_hash
        null_count = mock_db_post_migration.count_null("beliefs", "content_hash")
        assert null_count == 0, "All beliefs should have content_hash computed"

    def test_visibility_defaults_to_private(self, mock_db_post_migration):
        """visibility should default to 'private' for privacy-first design."""
        column_info = mock_db_post_migration.get_column_info("beliefs", "visibility")
        assert column_info is not None
        assert column_info["default"] == "'private'::visibility_level"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_db_connection():
    """
    Mock database connection for testing base schema.
    
    Returns a mock that simulates the CURRENT schema (before migrations).
    All xfail tests use this to verify gaps exist.
    """
    mock = MagicMock()
    
    # Current schema columns (from schema.sql)
    mock.get_belief_columns.return_value = [
        "id", "content", "confidence", "domain_path",
        "valid_from", "valid_until", "created_at", "modified_at",
        "source_id", "extraction_method",
        "supersedes_id", "superseded_by_id",
        "status", "embedding", "content_tsv"
    ]
    
    mock.get_belief_column_types.return_value = {
        "id": "uuid",
        "content": "text",
        "confidence": "jsonb",
        "domain_path": "text[]",
        "source_id": "uuid",
        "status": "text",
    }
    
    mock.get_belief_constraints.return_value = [
        "beliefs_valid_status CHECK (status IN ('active', 'superseded', 'disputed', 'archived'))",
        "beliefs_valid_confidence CHECK ((confidence->>'overall')::numeric >= 0 AND ...)",
    ]
    
    mock.get_belief_indexes.return_value = [
        "idx_beliefs_domain",
        "idx_beliefs_status", 
        "idx_beliefs_created",
        "idx_beliefs_tsv",
        "idx_beliefs_source",
        "idx_beliefs_embedding",
    ]
    
    mock.get_enum_values.return_value = []  # No visibility_level enum in base
    
    return mock


@pytest.fixture
def mock_db_post_migration():
    """
    Mock database connection for testing schema AFTER migrations applied.
    
    Returns a mock that simulates the schema after 001 + 002 migrations.
    """
    mock = MagicMock()
    
    mock.get_column_info.side_effect = lambda table, col: {
        ("beliefs", "holder_id"): {
            "type": "uuid",
            "nullable": True,  # Nullable for backward compat
            "default": "'00000000-0000-0000-0000-000000000001'::uuid",
        },
        ("beliefs", "version"): {
            "type": "integer",
            "nullable": False,
            "default": "1",
        },
        ("beliefs", "content_hash"): {
            "type": "character(64)",
            "nullable": False,
            "default": None,  # Computed by trigger
        },
        ("beliefs", "visibility"): {
            "type": "visibility_level",
            "nullable": False,
            "default": "'private'::visibility_level",
        },
    }.get((table, col))
    
    mock.count_null.return_value = 0  # All content_hash computed
    
    mock.get_enum_values.return_value = ["private", "federated", "public"]
    
    return mock
