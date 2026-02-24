"""Tests for core maintenance module."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from valence.core.maintenance import (
    VACUUM_TABLES,
    MaintenanceResult,
    check_and_run_maintenance,
    disable_maintenance_schedule,
    get_maintenance_schedule,
    run_full_maintenance,
    set_maintenance_schedule,
    vacuum_analyze,
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


class TestMaintenanceSchedule:
    def test_get_schedule_not_configured(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        result = get_maintenance_schedule(mock_cur)
        assert result is None

    def test_get_schedule_disabled(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"value": {"enabled": False}}
        result = get_maintenance_schedule(mock_cur)
        assert result is None

    def test_get_schedule_enabled(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"value": {"enabled": True, "interval_hours": 12, "last_run": None}}
        result = get_maintenance_schedule(mock_cur)
        assert result is not None
        assert result["interval_hours"] == 12

    def test_set_schedule_positive_interval(self):
        mock_cur = MagicMock()
        schedule = set_maintenance_schedule(mock_cur, 24)
        assert schedule["enabled"] is True
        assert schedule["interval_hours"] == 24
        assert schedule["last_run"] is None
        mock_cur.execute.assert_called_once()

    def test_set_schedule_invalid_interval(self):
        mock_cur = MagicMock()
        with pytest.raises(ValueError):
            set_maintenance_schedule(mock_cur, 0)

        with pytest.raises(ValueError):
            set_maintenance_schedule(mock_cur, -5)

    def test_disable_schedule(self):
        mock_cur = MagicMock()
        disable_maintenance_schedule(mock_cur)
        mock_cur.execute.assert_called_once()

        # Check that the call inserted disabled state
        call_args = mock_cur.execute.call_args
        inserted_json = call_args[0][1][0]
        inserted_data = json.loads(inserted_json)
        assert inserted_data["enabled"] is False


class TestCheckAndRunMaintenance:
    def test_no_schedule_returns_none(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = None
        result = check_and_run_maintenance(mock_cur)
        assert result is None

    def test_schedule_disabled_returns_none(self):
        mock_cur = MagicMock()
        mock_cur.fetchone.return_value = {"value": {"enabled": False}}
        result = check_and_run_maintenance(mock_cur)
        assert result is None

    def test_never_run_before_triggers_maintenance(self):
        mock_cur = MagicMock()

        # First call: get schedule
        # Second call: update schedule with last_run
        mock_cur.fetchone.side_effect = [
            {"value": {"enabled": True, "interval_hours": 24, "last_run": None}},
            {"matviewname": "test_view"},
        ]

        result = check_and_run_maintenance(mock_cur)

        assert result is not None
        assert result["maintenance_run"] is True
        assert "timestamp" in result
        assert "results" in result

        # Should have updated the schedule
        assert mock_cur.execute.call_count >= 2

    def test_elapsed_time_triggers_maintenance(self):
        mock_cur = MagicMock()

        # Last run was 25 hours ago
        last_run = datetime.now(UTC) - timedelta(hours=25)
        last_run_iso = last_run.isoformat()

        mock_cur.fetchone.side_effect = [
            {"value": {"enabled": True, "interval_hours": 24, "last_run": last_run_iso}},
            {"matviewname": "test_view"},
        ]

        result = check_and_run_maintenance(mock_cur)

        assert result is not None
        assert result["maintenance_run"] is True

    def test_not_enough_time_elapsed_skips(self):
        mock_cur = MagicMock()

        # Last run was 5 hours ago
        last_run = datetime.now(UTC) - timedelta(hours=5)
        last_run_iso = last_run.isoformat()

        mock_cur.fetchone.return_value = {"value": {"enabled": True, "interval_hours": 24, "last_run": last_run_iso}}

        result = check_and_run_maintenance(mock_cur)

        assert result is None
