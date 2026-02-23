"""Tests for core maintenance module.

Tests cover:
1. RetentionConfig defaults and customization
2. apply_retention calls stored procedure correctly
3. archive_beliefs calls stored procedure correctly
4. cleanup_tombstones calls stored procedure correctly
5. run_full_maintenance executes operations in order
6. MaintenanceResult formatting
"""

from __future__ import annotations

from unittest.mock import MagicMock

from valence.core.maintenance import (
    ArchivalConfig,
    MaintenanceResult,
    RetentionConfig,
    apply_retention,
    archive_beliefs,
    cleanup_tombstones,
    run_full_maintenance,
)

# ============================================================================
# Config Tests
# ============================================================================


class TestRetentionConfig:
    """Test retention configuration defaults."""

    def test_defaults(self):
        config = RetentionConfig()
        assert config.belief_retrievals_days == 90
        assert config.sync_events_days == 90
        assert config.embedding_coverage_days is None

    def test_custom(self):
        config = RetentionConfig(belief_retrievals_days=30, sync_events_days=60, embedding_coverage_days=180)
        assert config.belief_retrievals_days == 30
        assert config.sync_events_days == 60
        assert config.embedding_coverage_days == 180


class TestArchivalConfig:
    """Test archival configuration defaults."""

    def test_defaults(self):
        config = ArchivalConfig()
        assert config.older_than_days == 180
        assert config.batch_size == 1000


class TestMaintenanceResult:
    """Test result formatting."""

    def test_str_normal(self):
        result = MaintenanceResult(operation="retention", details={"table": "beliefs", "deleted": 42})
        assert "retention" in str(result)
        assert "42" in str(result)

    def test_str_dry_run(self):
        result = MaintenanceResult(operation="archival", details={"archived": 10}, dry_run=True)
        assert "dry run" in str(result)


# ============================================================================
# apply_retention Tests
# ============================================================================


class TestApplyRetention:
    """Test retention policy application."""

    def test_calls_procedure_with_defaults(self):
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"table_name": "belief_retrievals", "deleted_count": 100},
            {"table_name": "sync_events", "deleted_count": 50},
            {"table_name": "audit_log", "deleted_count": 0},
        ]

        results = apply_retention(cur, dry_run=False)

        cur.execute.assert_called_once()
        args = cur.execute.call_args[0]
        assert "apply_retention_policies" in args[0]
        assert args[1] == (90, 90, None, False)
        assert len(results) == 3
        assert results[0].details["deleted"] == 100

    def test_calls_procedure_with_custom_config(self):
        cur = MagicMock()
        cur.fetchall.return_value = []
        config = RetentionConfig(belief_retrievals_days=30, sync_events_days=60, embedding_coverage_days=180)

        apply_retention(cur, config, dry_run=True)

        args = cur.execute.call_args[0]
        assert args[1] == (30, 60, 180, True)

    def test_dry_run_flag_set(self):
        cur = MagicMock()
        cur.fetchall.return_value = [{"table_name": "test", "deleted_count": 5}]

        results = apply_retention(cur, dry_run=True)

        assert results[0].dry_run is True


# ============================================================================
# archive_beliefs Tests
# ============================================================================


class TestArchiveBeliefs:
    """Test belief archival."""

    def test_calls_procedure_with_defaults(self):
        cur = MagicMock()
        cur.fetchone.return_value = {"archived_count": 25, "freed_embeddings": 20}

        result = archive_beliefs(cur)

        args = cur.execute.call_args[0]
        assert "archive_stale_beliefs" in args[0]
        assert args[1] == (180, 1000, False)
        assert result.details["archived"] == 25
        assert result.details["freed_embeddings"] == 20

    def test_custom_config(self):
        cur = MagicMock()
        cur.fetchone.return_value = {"archived_count": 0, "freed_embeddings": 0}
        config = ArchivalConfig(older_than_days=365, batch_size=500)

        archive_beliefs(cur, config, dry_run=True)

        args = cur.execute.call_args[0]
        assert args[1] == (365, 500, True)


# ============================================================================
# cleanup_tombstones Tests
# ============================================================================


class TestCleanupTombstones:
    """Test tombstone cleanup."""

    def test_calls_procedure(self):
        cur = MagicMock()
        cur.fetchone.return_value = {"count": 3}

        result = cleanup_tombstones(cur, dry_run=False)

        args = cur.execute.call_args[0]
        assert "cleanup_expired_tombstones" in args[0]
        assert result.details["removed"] == 3


# ============================================================================
# Full Maintenance Tests
# ============================================================================


class TestRunFullMaintenance:
    """Test full maintenance cycle."""

    def test_executes_all_operations(self):
        cur = MagicMock()
        # Retention results (fetchall call 1), compaction candidates (fetchall call 2), refresh_views (fetchall call 3)
        cur.fetchall.side_effect = [
            [
                {"table_name": "belief_retrievals", "deleted_count": 10},
                {"table_name": "audit_log", "deleted_count": 0},
            ],
            [],  # No compaction candidates
            [
                {"view_name": "beliefs_current_mat", "refreshed": True, "error_msg": None},
            ],
        ]
        # Archival and tombstone results
        cur.fetchone.side_effect = [
            {"archived_count": 5, "freed_embeddings": 3},
            {"count": 1},
        ]

        results = run_full_maintenance(cur, skip_vacuum=True)

        # retention (2) + archival (1) + tombstone (1) + compaction (1) + refresh_views (1) = 6
        assert len(results) == 6
        assert results[0].operation == "retention"
        assert results[2].operation == "archival"
        assert results[3].operation == "tombstone_cleanup"
        assert results[4].operation == "exchange_compaction"
        assert results[5].operation == "refresh_views"

    def test_dry_run_skips_vacuum_and_views(self):
        cur = MagicMock()
        cur.fetchall.return_value = []  # No retention rows, no compaction candidates
        cur.fetchone.side_effect = [
            {"archived_count": 0, "freed_embeddings": 0},
            {"count": 0},
        ]

        results = run_full_maintenance(cur, dry_run=True)

        # No vacuum or views in dry run
        assert all(r.operation != "vacuum_analyze" for r in results)
        assert all(r.operation != "refresh_views" for r in results)
        # Compaction still runs in dry_run (reports what would happen)
        assert any(r.operation == "exchange_compaction" for r in results)
