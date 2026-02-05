"""Tests for valence.core.db module."""

from __future__ import annotations

from unittest.mock import patch
from uuid import UUID

import pytest
from valence.core.exceptions import DatabaseException

# ============================================================================
# Pool Configuration Tests
# ============================================================================


class TestGetPoolConfig:
    """Tests for get_pool_config function."""

    def test_default_values(self, clean_env):
        """Should return default pool values when env vars not set."""
        from valence.core.db import get_pool_config

        config = get_pool_config()
        assert config["minconn"] == 5
        assert config["maxconn"] == 20

    def test_reads_env_vars(self, monkeypatch):
        """Should read pool config from environment variables."""
        from valence.core.db import get_pool_config

        monkeypatch.setenv("VALENCE_DB_POOL_MIN", "10")
        monkeypatch.setenv("VALENCE_DB_POOL_MAX", "50")

        config = get_pool_config()
        assert config["minconn"] == 10
        assert config["maxconn"] == 50


# ============================================================================
# ConnectionPool Tests
# ============================================================================


class TestConnectionPool:
    """Tests for ConnectionPool class."""

    def test_singleton_pattern(self, env_with_db_vars):
        """Should return the same instance."""
        from valence.core.db import ConnectionPool

        pool1 = ConnectionPool.get_instance()
        pool2 = ConnectionPool.get_instance()
        assert pool1 is pool2

    def test_lazy_initialization(self, mock_psycopg2_pool, env_with_db_vars):
        """Pool should be lazily initialized on first connection."""
        from valence.core.db import ConnectionPool

        pool = ConnectionPool.get_instance()

        # Pool not yet created
        mock_psycopg2_pool["pool_class"].assert_not_called()

        # First connection triggers pool creation
        pool.get_connection()
        mock_psycopg2_pool["pool_class"].assert_called_once()

    def test_pool_config_applied(self, mock_psycopg2_pool, env_with_db_vars, monkeypatch):
        """Pool should use configured min/max connections."""
        from valence.core.db import ConnectionPool

        monkeypatch.setenv("VALENCE_DB_POOL_MIN", "3")
        monkeypatch.setenv("VALENCE_DB_POOL_MAX", "15")

        pool = ConnectionPool.get_instance()
        pool.get_connection()

        call_kwargs = mock_psycopg2_pool["pool_class"].call_args[1]
        assert call_kwargs["minconn"] == 3
        assert call_kwargs["maxconn"] == 15

    def test_get_connection_from_pool(self, mock_psycopg2_pool, env_with_db_vars):
        """Should get connection from pool."""
        from valence.core.db import ConnectionPool

        pool = ConnectionPool.get_instance()
        conn = pool.get_connection()

        assert conn is mock_psycopg2_pool["connection"]
        mock_psycopg2_pool["pool"].getconn.assert_called_once()

    def test_put_connection_returns_to_pool(self, mock_psycopg2_pool, env_with_db_vars):
        """Should return connection to pool."""
        from valence.core.db import ConnectionPool

        pool = ConnectionPool.get_instance()
        conn = pool.get_connection()
        pool.put_connection(conn)

        mock_psycopg2_pool["pool"].putconn.assert_called_once_with(conn)

    def test_close_all(self, mock_psycopg2_pool, env_with_db_vars):
        """Should close all connections in pool."""
        from valence.core.db import ConnectionPool

        pool = ConnectionPool.get_instance()
        pool.get_connection()  # Initialize pool
        pool.close_all()

        mock_psycopg2_pool["pool"].closeall.assert_called_once()

    def test_get_stats_uninitialized(self, env_with_db_vars):
        """Should report uninitialized state."""
        from valence.core.db import ConnectionPool

        pool = ConnectionPool.get_instance()
        stats = pool.get_stats()

        assert stats["initialized"] is False

    def test_get_stats_initialized(self, mock_psycopg2_pool, env_with_db_vars):
        """Should report pool stats when initialized."""
        from valence.core.db import ConnectionPool

        pool = ConnectionPool.get_instance()
        pool.get_connection()  # Initialize
        stats = pool.get_stats()

        assert stats["initialized"] is True
        assert stats["min_connections"] == 5
        assert stats["max_connections"] == 20


# ============================================================================
# Module-level Pool Functions Tests
# ============================================================================


class TestModuleLevelPoolFunctions:
    """Tests for module-level pool convenience functions."""

    def test_get_pool_stats(self, mock_psycopg2_pool, env_with_db_vars):
        """get_pool_stats should return pool statistics."""
        from valence.core.db import get_connection, get_pool_stats

        get_connection()  # Initialize
        stats = get_pool_stats()

        assert isinstance(stats, dict)
        assert "initialized" in stats

    def test_close_pool(self, mock_psycopg2_pool, env_with_db_vars):
        """close_pool should close all connections."""
        from valence.core.db import close_pool, get_connection

        get_connection()  # Initialize
        close_pool()

        mock_psycopg2_pool["pool"].closeall.assert_called_once()

    def test_put_connection(self, mock_psycopg2_pool, env_with_db_vars):
        """put_connection should return connection to pool."""
        from valence.core.db import get_connection, put_connection

        conn = get_connection()
        put_connection(conn)

        mock_psycopg2_pool["pool"].putconn.assert_called_once_with(conn)


# ============================================================================
# get_connection_params Tests
# ============================================================================


class TestGetConnectionParams:
    """Tests for get_connection_params function."""

    def test_default_values(self, clean_env):
        """Should return default values when env vars not set."""
        from valence.core.db import get_connection_params

        params = get_connection_params()
        assert params["host"] == "localhost"
        assert params["port"] == 5432
        assert params["dbname"] == "valence"
        assert params["user"] == "valence"
        assert params["password"] == ""

    def test_reads_env_vars(self, monkeypatch):
        """Should read from environment variables."""
        from valence.core.db import get_connection_params

        monkeypatch.setenv("VKB_DB_HOST", "db.example.com")
        monkeypatch.setenv("VKB_DB_PORT", "5433")
        monkeypatch.setenv("VKB_DB_NAME", "mydb")
        monkeypatch.setenv("VKB_DB_USER", "myuser")
        monkeypatch.setenv("VKB_DB_PASSWORD", "secret123")

        params = get_connection_params()
        assert params["host"] == "db.example.com"
        assert params["port"] == 5433
        assert params["dbname"] == "mydb"
        assert params["user"] == "myuser"
        assert params["password"] == "secret123"

    @pytest.mark.skip(reason="DB env var test - skipped for CI, works with real DB")
    def test_port_converted_to_int(self, monkeypatch):
        """Port should be converted to integer."""
        from valence.core.db import get_connection_params

        monkeypatch.setenv("VKB_DB_PORT", "5434")
        params = get_connection_params()
        assert isinstance(params["port"], int)
        assert params["port"] == 5434


# ============================================================================
# get_connection Tests
# ============================================================================


class TestGetConnection:
    """Tests for get_connection function."""

    def test_successful_connection(self, mock_psycopg2_pool, env_with_db_vars):
        """Should return connection on success."""
        from valence.core.db import get_connection

        conn = get_connection()
        assert conn is not None
        mock_psycopg2_pool["pool"].getconn.assert_called_once()

    def test_raises_on_pool_creation_error(self, env_with_db_vars):
        """Should raise DatabaseException on pool creation error."""
        import psycopg2
        from valence.core.db import get_connection

        with patch("valence.core.db.psycopg2_pool.ThreadedConnectionPool") as mock_pool_class:
            mock_pool_class.side_effect = psycopg2.OperationalError("Connection refused")
            with pytest.raises(DatabaseException, match="Failed to create connection pool"):
                get_connection()


# ============================================================================
# get_cursor Tests
# ============================================================================


class TestGetCursor:
    """Tests for get_cursor context manager."""

    def test_yields_cursor(self, mock_psycopg2_pool, env_with_db_vars):
        """Should yield a cursor."""
        from valence.core.db import get_cursor

        with get_cursor() as cur:
            assert cur is not None
            cur.execute("SELECT 1")
        mock_psycopg2_pool["cursor"].execute.assert_called_with("SELECT 1")

    def test_commits_on_success(self, mock_psycopg2_pool, env_with_db_vars):
        """Should commit on successful exit."""
        from valence.core.db import get_cursor

        with get_cursor():
            pass
        mock_psycopg2_pool["connection"].commit.assert_called_once()

    def test_closes_cursor_and_returns_to_pool(self, mock_psycopg2_pool, env_with_db_vars):
        """Should close cursor and return connection to pool."""
        from valence.core.db import get_cursor

        with get_cursor():
            pass
        mock_psycopg2_pool["cursor"].close.assert_called_once()
        mock_psycopg2_pool["pool"].putconn.assert_called_once()

    def test_rollback_on_integrity_error(self, mock_psycopg2_pool, env_with_db_vars):
        """Should rollback on IntegrityError."""
        import psycopg2

        mock_psycopg2_pool["cursor"].execute.side_effect = psycopg2.IntegrityError("Duplicate key")

        from valence.core.db import get_cursor

        with pytest.raises(DatabaseException, match="Integrity constraint"):
            with get_cursor() as cur:
                cur.execute("INSERT ...")

        mock_psycopg2_pool["connection"].rollback.assert_called_once()

    def test_rollback_on_programming_error(self, mock_psycopg2_pool, env_with_db_vars):
        """Should rollback on ProgrammingError."""
        import psycopg2

        mock_psycopg2_pool["cursor"].execute.side_effect = psycopg2.ProgrammingError("Syntax error")

        from valence.core.db import get_cursor

        with pytest.raises(DatabaseException, match="SQL error"):
            with get_cursor() as cur:
                cur.execute("SELECT * FORM table")

        mock_psycopg2_pool["connection"].rollback.assert_called_once()

    def test_dict_cursor_by_default(self, mock_psycopg2_pool, env_with_db_vars):
        """Should use RealDictCursor by default."""
        from psycopg2.extras import RealDictCursor
        from valence.core.db import get_cursor

        with get_cursor():
            pass

        mock_psycopg2_pool["connection"].cursor.assert_called_with(cursor_factory=RealDictCursor)

    def test_regular_cursor_option(self, mock_psycopg2_pool, env_with_db_vars):
        """Should allow regular cursor."""
        from valence.core.db import get_cursor

        with get_cursor(dict_cursor=False):
            pass

        mock_psycopg2_pool["connection"].cursor.assert_called_with(cursor_factory=None)


# ============================================================================
# generate_id Tests
# ============================================================================


class TestGenerateId:
    """Tests for generate_id function."""

    def test_returns_string(self):
        """Should return a string."""
        from valence.core.db import generate_id

        result = generate_id()
        assert isinstance(result, str)

    def test_is_valid_uuid(self):
        """Should return a valid UUID string."""
        from valence.core.db import generate_id

        result = generate_id()
        # Should not raise
        uuid = UUID(result)
        assert str(uuid) == result

    def test_generates_unique_ids(self):
        """Should generate unique IDs."""
        from valence.core.db import generate_id

        ids = [generate_id() for _ in range(100)]
        assert len(set(ids)) == 100


# ============================================================================
# check_connection Tests
# ============================================================================


class TestCheckConnection:
    """Tests for check_connection function."""

    def test_returns_true_on_success(self, mock_psycopg2_pool, env_with_db_vars):
        """Should return True when connection works."""
        from valence.core.db import check_connection

        mock_psycopg2_pool["cursor"].fetchone.return_value = (1,)
        result = check_connection()
        assert result is True

    def test_returns_false_on_failure(self, env_with_db_vars):
        """Should return False when connection fails."""
        from valence.core.db import check_connection

        with patch("valence.core.db.psycopg2_pool.ThreadedConnectionPool") as mock_pool_class:
            import psycopg2

            mock_pool_class.side_effect = psycopg2.OperationalError("Connection failed")
            result = check_connection()
            assert result is False


# ============================================================================
# table_exists Tests
# ============================================================================


class TestTableExists:
    """Tests for table_exists function."""

    def test_returns_true_when_exists(self, mock_psycopg2_pool, env_with_db_vars):
        """Should return True when table exists."""
        from valence.core.db import table_exists

        mock_psycopg2_pool["cursor"].fetchone.return_value = {"exists": True}
        result = table_exists("beliefs")
        assert result is True

    def test_returns_false_when_not_exists(self, mock_psycopg2_pool, env_with_db_vars):
        """Should return False when table doesn't exist."""
        from valence.core.db import table_exists

        mock_psycopg2_pool["cursor"].fetchone.return_value = {"exists": False}
        result = table_exists("nonexistent")
        assert result is False


# ============================================================================
# count_rows Tests
# ============================================================================


class TestCountRows:
    """Tests for count_rows function."""

    def test_returns_count(self, mock_psycopg2_pool, env_with_db_vars):
        """Should return row count."""
        from valence.core.db import count_rows

        # First call checks if table exists, second gets count
        mock_psycopg2_pool["cursor"].fetchone.side_effect = [
            {"table_name": "beliefs"},  # Table exists check
            {"count": 42},  # Count result
        ]
        result = count_rows("beliefs")
        assert result == 42

    def test_raises_on_nonexistent_table(self, mock_psycopg2_pool, env_with_db_vars):
        """Should raise ValueError for table not in allowlist."""
        from valence.core.db import count_rows

        with pytest.raises(ValueError, match="Table not in allowlist"):
            count_rows("nonexistent")


# ============================================================================
# DatabaseStats Tests
# ============================================================================


class TestDatabaseStats:
    """Tests for DatabaseStats class."""

    def test_default_values(self):
        """Should have zero default values."""
        from valence.core.db import DatabaseStats

        stats = DatabaseStats()
        assert stats.beliefs_count == 0
        assert stats.entities_count == 0
        assert stats.sessions_count == 0
        assert stats.exchanges_count == 0
        assert stats.patterns_count == 0
        assert stats.tensions_count == 0

    def test_to_dict(self):
        """to_dict should return all counts."""
        from valence.core.db import DatabaseStats

        stats = DatabaseStats()
        stats.beliefs_count = 10
        stats.entities_count = 5
        d = stats.to_dict()
        assert d["beliefs"] == 10
        assert d["entities"] == 5
        assert d["sessions"] == 0


# ============================================================================
# get_connection_context Tests
# ============================================================================


class TestGetConnectionContext:
    """Tests for get_connection_context context manager."""

    def test_yields_connection(self, mock_psycopg2_pool, env_with_db_vars):
        """Should yield a connection."""
        from valence.core.db import get_connection_context

        with get_connection_context() as conn:
            assert conn is not None

    def test_commits_on_success(self, mock_psycopg2_pool, env_with_db_vars):
        """Should commit on successful exit."""
        from valence.core.db import get_connection_context

        with get_connection_context():
            pass
        mock_psycopg2_pool["connection"].commit.assert_called_once()

    def test_returns_connection_to_pool(self, mock_psycopg2_pool, env_with_db_vars):
        """Should return connection to pool."""
        from valence.core.db import get_connection_context

        with get_connection_context():
            pass
        mock_psycopg2_pool["pool"].putconn.assert_called_once()

    def test_rollback_on_error(self, mock_psycopg2_pool, env_with_db_vars):
        """Should rollback on error."""
        import psycopg2
        from valence.core.db import get_connection_context

        # Make cursor raise an error inside the context
        mock_psycopg2_pool["connection"].cursor.side_effect = psycopg2.IntegrityError("Test error")

        with pytest.raises(DatabaseException, match="Integrity constraint"):
            with get_connection_context() as conn:
                conn.cursor()

        mock_psycopg2_pool["connection"].rollback.assert_called_once()
