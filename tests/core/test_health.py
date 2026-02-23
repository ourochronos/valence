"""Tests for valence.core.health module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ============================================================================
# check_env_vars Tests
# ============================================================================


class TestCheckEnvVars:
    """Tests for check_env_vars function."""

    def test_all_present(self, monkeypatch):
        """Should return True when all required vars are present."""
        from valence.core.health import check_env_vars

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")
        monkeypatch.setenv("VKB_DB_PASSWORD", "secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")

        ok, missing_required, missing_optional = check_env_vars()
        assert ok is True
        assert missing_required == []

    def test_missing_required(self, clean_env):
        """Should detect missing required vars."""
        from valence.core.health import check_env_vars

        ok, missing_required, missing_optional = check_env_vars()
        assert ok is False
        assert "VKB_DB_HOST" in missing_required
        assert "VKB_DB_NAME" in missing_required
        assert "VKB_DB_USER" in missing_required

    def test_missing_optional(self, monkeypatch):
        """Should detect missing optional vars."""
        from valence.core.health import check_env_vars

        # Set required vars
        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")
        # Don't set optional vars
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("VKB_DB_PASSWORD", raising=False)

        ok, missing_required, missing_optional = check_env_vars()
        assert ok is True  # Required are present
        assert "OPENAI_API_KEY" in missing_optional

    def test_partial_required(self, monkeypatch, clean_env):
        """Should fail if only some required vars are set."""
        from valence.core.health import check_env_vars

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        # Missing VKB_DB_NAME and VKB_DB_USER

        ok, missing_required, missing_optional = check_env_vars()
        assert ok is False
        assert "VKB_DB_HOST" not in missing_required
        assert "VKB_DB_NAME" in missing_required


# ============================================================================
# check_database_connection Tests
# ============================================================================


class TestCheckDatabaseConnection:
    """Tests for check_database_connection function."""

    def test_success(self, mock_psycopg2, env_with_db_vars):
        """Should return True on successful connection."""
        from valence.core.health import check_database_connection

        connected, error = check_database_connection()
        assert connected is True
        assert error is None

    def test_import_error(self, env_with_db_vars):
        """Should handle psycopg2 not installed."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "psycopg2":
                raise ImportError("No module named psycopg2")
            return real_import(name, *args, **kwargs)

        # Need to reload module to test import behavior
        import importlib

        import valence.core.health

        with patch.object(builtins, "__import__", mock_import):
            # Force re-import
            importlib.reload(valence.core.health)
            connected, error = valence.core.health.check_database_connection()
            assert connected is False
            assert "psycopg2 not installed" in error

        # Reload to restore normal behavior
        importlib.reload(valence.core.health)

    def test_operational_error(self, env_with_db_vars):
        """Should return False on OperationalError."""
        import psycopg2

        from valence.core.health import check_database_connection

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = psycopg2.OperationalError("Connection refused")
            connected, error = check_database_connection()
            assert connected is False
            assert "Connection failed" in error

    def test_generic_error(self, env_with_db_vars):
        """Should return False on generic Error."""
        import psycopg2

        from valence.core.health import check_database_connection

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = psycopg2.Error("Unknown error")
            connected, error = check_database_connection()
            assert connected is False
            assert "Database error" in error


# ============================================================================
# check_pgvector Tests
# ============================================================================


class TestCheckPgvector:
    """Tests for check_pgvector function."""

    def test_available(self, env_with_db_vars):
        """Should return True when pgvector is installed."""
        from valence.core.health import check_pgvector

        with patch("psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.fetchone.return_value = (True,)

            available, error = check_pgvector()
            assert available is True
            assert error is None

    def test_not_installed(self, env_with_db_vars):
        """Should return False when pgvector is not installed."""
        from valence.core.health import check_pgvector

        with patch("psycopg2.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
            mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
            mock_cursor.fetchone.return_value = (False,)

            available, error = check_pgvector()
            assert available is False
            assert "not installed" in error

    def test_database_error(self, env_with_db_vars):
        """Should return False on database error."""
        import psycopg2

        from valence.core.health import check_pgvector

        with patch("psycopg2.connect") as mock_connect:
            mock_connect.side_effect = psycopg2.Error("Connection failed")
            available, error = check_pgvector()
            assert available is False
            assert "Database error" in error


# ============================================================================
# check_schema Tests
# ============================================================================


class TestCheckSchema:
    """Tests for check_schema function."""

    def test_all_tables_present(self, mock_psycopg2, env_with_db_vars):
        """Should return True when all tables exist."""
        from valence.core.health import check_schema

        # Mock table_exists to return True for all tables
        mock_psycopg2["cursor"].fetchone.return_value = (True,)

        with patch("valence.core.health.table_exists", return_value=True):
            ok, missing = check_schema()
            assert ok is True
            assert missing == []

    def test_missing_tables(self, mock_psycopg2, env_with_db_vars):
        """Should detect missing tables."""
        from valence.core.health import check_schema

        with patch("valence.core.health.table_exists") as mock_exists:
            mock_exists.side_effect = lambda t: t != "articles"  # articles missing

            ok, missing = check_schema()
            assert ok is False
            assert "articles" in missing

    def test_database_exception_treated_as_missing(self, env_with_db_vars):
        """Should treat DatabaseException as missing table."""
        from valence.core.exceptions import DatabaseException
        from valence.core.health import check_schema

        with patch("valence.core.health.table_exists") as mock_exists:
            mock_exists.side_effect = DatabaseException("Connection failed")

            ok, missing = check_schema()
            assert ok is False
            # All tables should be considered missing
            assert len(missing) > 0


# ============================================================================
# HealthStatus Tests
# ============================================================================


class TestHealthStatus:
    """Tests for HealthStatus dataclass."""

    def test_default_values(self):
        """Should have safe default values."""
        from valence.core.health import HealthStatus

        status = HealthStatus()
        assert status.healthy is False
        assert status.database_connected is False
        assert status.schema_valid is False
        assert status.pgvector_available is False
        assert status.env_vars_present is False
        assert status.missing_env_vars == []
        assert status.missing_tables == []
        assert status.warnings == []
        assert status.stats == {}
        assert status.error is None

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        from valence.core.health import HealthStatus

        status = HealthStatus(
            healthy=True,
            database_connected=True,
            schema_valid=True,
            pgvector_available=True,
            env_vars_present=True,
            stats={"beliefs": 10},
        )
        d = status.to_dict()
        assert d["healthy"] is True
        assert d["database_connected"] is True
        assert d["stats"]["beliefs"] == 10


# ============================================================================
# run_health_check Tests
# ============================================================================


class TestRunHealthCheck:
    """Tests for run_health_check function."""

    def test_full_healthy(self, monkeypatch, mock_psycopg2, env_with_db_vars):
        """Should return healthy status when all checks pass."""
        from valence.core.health import run_health_check

        # Set all required env vars
        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx")

        with patch("valence.core.health.check_database_connection", return_value=(True, None)):
            with patch("valence.core.health.check_pgvector", return_value=(True, None)):
                with patch("valence.core.health.check_schema", return_value=(True, [])):
                    with patch("valence.core.health.DatabaseStats") as mock_stats:
                        mock_stats.collect.return_value = MagicMock(to_dict=lambda: {"beliefs": 10})
                        status = run_health_check()
                        assert status.healthy is True
                        assert status.database_connected is True
                        assert status.schema_valid is True

    def test_env_failure_stops_early(self, clean_env):
        """Should return early on env var failure."""
        from valence.core.health import run_health_check

        status = run_health_check()
        assert status.healthy is False
        assert status.env_vars_present is False
        assert "Missing required environment variables" in status.error
        # Other checks should not have run
        assert status.database_connected is False

    def test_db_failure_stops_early(self, monkeypatch):
        """Should return early on database failure."""
        from valence.core.health import run_health_check

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")

        with patch(
            "valence.core.health.check_database_connection",
            return_value=(False, "Connection refused"),
        ):
            status = run_health_check()
            assert status.healthy is False
            assert status.database_connected is False
            assert "Database connection failed" in status.error

    def test_schema_failure(self, monkeypatch):
        """Should return unhealthy on schema failure."""
        from valence.core.health import run_health_check

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")

        with patch("valence.core.health.check_database_connection", return_value=(True, None)):
            with patch("valence.core.health.check_pgvector", return_value=(True, None)):
                with patch(
                    "valence.core.health.check_schema",
                    return_value=(False, ["beliefs", "entities"]),
                ):
                    status = run_health_check()
                    assert status.healthy is False
                    assert "Missing required tables" in status.error
                    assert status.missing_tables == ["beliefs", "entities"]

    def test_pgvector_warning_not_fatal(self, monkeypatch):
        """pgvector unavailable should be a warning, not failure."""
        from valence.core.health import run_health_check

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")

        with patch("valence.core.health.check_database_connection", return_value=(True, None)):
            with patch(
                "valence.core.health.check_pgvector",
                return_value=(False, "Not installed"),
            ):
                with patch("valence.core.health.check_schema", return_value=(True, [])):
                    with patch("valence.core.health.DatabaseStats") as mock_stats:
                        mock_stats.collect.return_value = MagicMock(to_dict=lambda: {})
                        status = run_health_check()
                        assert status.healthy is True  # Still healthy
                        assert status.pgvector_available is False
                        assert any("pgvector" in w for w in status.warnings)


# ============================================================================
# startup_checks Tests
# ============================================================================


class TestStartupChecks:
    """Tests for startup_checks function."""

    def test_with_fail_fast_exits_on_failure(self, clean_env):
        """Should exit with code 1 on failure when fail_fast=True."""
        from valence.core.health import startup_checks

        with pytest.raises(SystemExit) as exc_info:
            startup_checks(fail_fast=True)
        assert exc_info.value.code == 1

    def test_without_fail_fast_returns_status(self, clean_env):
        """Should return status without exiting when fail_fast=False."""
        from valence.core.health import startup_checks

        # Should not raise
        status = startup_checks(fail_fast=False)
        assert status.healthy is False


# ============================================================================
# require_healthy Tests
# ============================================================================


class TestRequireHealthy:
    """Tests for require_healthy function."""

    def test_exits_on_unhealthy(self, clean_env):
        """Should exit when unhealthy and fail_fast=True."""
        from valence.core.health import require_healthy

        with pytest.raises(SystemExit):
            require_healthy(fail_fast=True)

    def test_returns_status_when_not_fail_fast(self, clean_env):
        """Should return status when fail_fast=False."""
        from valence.core.health import require_healthy

        status = require_healthy(fail_fast=False)
        assert status.healthy is False


# ============================================================================
# validate_environment Tests
# ============================================================================


class TestValidateEnvironment:
    """Tests for validate_environment function."""

    def test_raises_on_missing_vars(self, clean_env):
        """Should raise ConfigException when vars missing."""
        from valence.core.exceptions import ConfigException
        from valence.core.health import validate_environment

        with pytest.raises(ConfigException):
            validate_environment()

    def test_passes_when_all_present(self, monkeypatch):
        """Should pass when all required vars present."""
        from valence.core.health import validate_environment

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")

        # Should not raise
        validate_environment()


# ============================================================================
# validate_database Tests
# ============================================================================


class TestValidateDatabase:
    """Tests for validate_database function."""

    def test_raises_on_connection_failure(self, monkeypatch):
        """Should raise DatabaseException on connection failure."""
        from valence.core.exceptions import DatabaseException
        from valence.core.health import validate_database

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")

        with patch(
            "valence.core.health.check_database_connection",
            return_value=(False, "Failed"),
        ):
            with pytest.raises(DatabaseException, match="Database connection failed"):
                validate_database()

    def test_raises_on_schema_failure(self, monkeypatch):
        """Should raise DatabaseException on schema failure."""
        from valence.core.exceptions import DatabaseException
        from valence.core.health import validate_database

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")

        with patch("valence.core.health.check_database_connection", return_value=(True, None)):
            with patch("valence.core.health.check_schema", return_value=(False, ["beliefs"])):
                with pytest.raises(DatabaseException, match="schema invalid"):
                    validate_database()


# ============================================================================
# cli_health_check Tests
# ============================================================================


class TestCliHealthCheck:
    """Tests for cli_health_check function."""

    def test_returns_zero_on_healthy(self, monkeypatch, capsys):
        """Should return 0 when healthy."""
        from valence.core.health import cli_health_check

        monkeypatch.setenv("VKB_DB_HOST", "localhost")
        monkeypatch.setenv("VKB_DB_NAME", "valence")
        monkeypatch.setenv("VKB_DB_USER", "valence")

        with patch("valence.core.health.run_health_check") as mock_check:
            from valence.core.health import HealthStatus

            mock_check.return_value = HealthStatus(
                healthy=True,
                database_connected=True,
                schema_valid=True,
                pgvector_available=True,
                env_vars_present=True,
            )
            exit_code = cli_health_check()
            assert exit_code == 0

    def test_returns_one_on_unhealthy(self, clean_env, capsys):
        """Should return 1 when unhealthy."""
        from valence.core.health import cli_health_check

        exit_code = cli_health_check()
        assert exit_code == 1

        captured = capsys.readouterr()
        assert "Healthy: False" in captured.out

    def test_prints_status_information(self, monkeypatch, capsys):
        """Should print detailed status."""
        from valence.core.health import cli_health_check

        with patch("valence.core.health.run_health_check") as mock_check:
            from valence.core.health import HealthStatus

            mock_check.return_value = HealthStatus(
                healthy=False,
                missing_env_vars=["VKB_DB_HOST"],
                missing_tables=["beliefs"],
                error="Test error",
                warnings=["Warning 1"],
                stats={"beliefs": 10},
            )
            cli_health_check()

        captured = capsys.readouterr()
        assert "VKB_DB_HOST" in captured.out
        assert "beliefs" in captured.out
        assert "Test error" in captured.out
        assert "Warning 1" in captured.out
