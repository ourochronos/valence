"""Tests for core maintenance module."""

from __future__ import annotations

from unittest.mock import MagicMock

from valence.core.maintenance import (
    MaintenanceResult,
    vacuum_analyze,
    run_full_maintenance,
    VACUUM_TABLES,
)


class TestMaintenanceResult:
    def test_summary(self):
        r = MaintenanceResult(operation="vacuum", details={"tables": 5})
        assert "vacuum" in r.summary

    def test_dry_run_flag(self):
        r = MaintenanceResult(operation="test", dry_run=True)
        assert r.dry_run is True


class TestVacuumAnalyze:
    def test_vacuums_all_tables(self):
        mock_cur = MagicMock()
        result = vacuum_analyze(mock_cur)
        assert result.operation == "vacuum_analyze"
        assert result.details["tables_vacuumed"] == len(VACUUM_TABLES)
        assert mock_cur.execute.call_count == len(VACUUM_TABLES)


class TestRunFullMaintenance:
    def test_dry_run_skips_everything(self):
        mock_cur = MagicMock()
        results = run_full_maintenance(mock_cur, dry_run=True)
        assert results == []

    def test_skip_flags(self):
        mock_cur = MagicMock()
        results = run_full_maintenance(mock_cur, skip_vacuum=True, skip_views=True)
        assert results == []
