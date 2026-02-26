"""Tests for valence.core.db module - database connection management.

Tests cover:
- DB config loading from environment
- Connection pool management
- Cursor context manager with commit/rollback
- Connection context manager
- Utility functions (generate_id, table_exists, count_rows, etc.)
- Schema management
- Row serialization
"""

from __future__ import annotations

import os
import uuid as _uuid
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestGetDbConfig:
    """Test database configuration loading from environment."""

    def test_defaults(self, monkeypatch):
        """Test default database configuration values."""
        # Clear all VALENCE_DB_* vars
        for key in list(os.environ.keys()):
            if key.startswith("VALENCE_DB_"):
                monkeypatch.delenv(key, raising=False)

        from valence.core.db import _get_db_config

        config = _get_db_config()
        assert config["host"] == "127.0.0.1"
        assert config["port"] == 5433
        assert config["dbname"] == "valence"
        assert config["user"] == "valence"
        assert config["password"] == "valence"

    def test_environment_overrides(self, monkeypatch):
        """Test environment variables override defaults."""
        monkeypatch.setenv("VALENCE_DB_HOST", "postgres.example.com")
        monkeypatch.setenv("VALENCE_DB_PORT", "5433")
        monkeypatch.setenv("VALENCE_DB_NAME", "testdb")
        monkeypatch.setenv("VALENCE_DB_USER", "testuser")
        monkeypatch.setenv("VALENCE_DB_PASSWORD", "secret123")

        from valence.core.db import _get_db_config

        config = _get_db_config()
        assert config["host"] == "postgres.example.com"
        assert config["port"] == 5433
        assert config["dbname"] == "testdb"
        assert config["user"] == "testuser"
        assert config["password"] == "secret123"


class TestConnectionPool:
    """Test connection pool creation and management."""

    @patch("valence.core.db.psycopg2_pool.ThreadedConnectionPool")
    def test_pool_creation(self, mock_pool_class):
        """Test pool is created with correct parameters."""
        from valence.core import db

        # Reset pool
        db._pool = None

        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        result = db._get_pool()

        assert result is mock_pool
        mock_pool_class.assert_called_once()
        call_kwargs = mock_pool_class.call_args[1]
        assert call_kwargs["minconn"] == 5
        assert call_kwargs["maxconn"] == 20

    @patch("valence.core.db.psycopg2_pool.ThreadedConnectionPool")
    def test_pool_singleton(self, mock_pool_class):
        """Test pool is created only once (singleton)."""
        from valence.core import db

        db._pool = None

        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        pool1 = db._get_pool()
        pool2 = db._get_pool()

        assert pool1 is pool2
        assert mock_pool_class.call_count == 1

    @patch("valence.core.db.psycopg2_pool.ThreadedConnectionPool")
    def test_pool_thread_safe(self, mock_pool_class):
        """Test pool creation is thread-safe with double-check locking."""
        import threading

        from valence.core import db

        db._pool = None

        call_count = 0
        first_thread_entered = threading.Event()
        second_thread_can_proceed = threading.Event()

        def delayed_init(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First thread: signal entry and wait
                first_thread_entered.set()
                second_thread_can_proceed.wait(timeout=1)
            mock_pool = MagicMock()
            return mock_pool

        mock_pool_class.side_effect = delayed_init

        pools = []

        def get_pool_delayed():
            pools.append(db._get_pool())

        def get_pool_fast():
            # Wait for first thread to enter lock
            first_thread_entered.wait(timeout=1)
            # Now try to get pool (should hit outer check with pool already initializing)
            pools.append(db._get_pool())
            # Allow first thread to complete
            second_thread_can_proceed.set()

        t1 = threading.Thread(target=get_pool_delayed)
        t2 = threading.Thread(target=get_pool_fast)

        t1.start()
        t2.start()

        t1.join(timeout=2)
        t2.join(timeout=2)

        # Should still only create pool once
        assert call_count == 1

    def test_close_pool(self):
        """Test pool can be closed."""
        from valence.core import db

        mock_pool = MagicMock()
        db._pool = mock_pool

        db.close_pool()

        mock_pool.closeall.assert_called_once()
        assert db._pool is None

    def test_close_pool_when_none(self):
        """Test closing pool when already None is safe."""
        from valence.core import db

        db._pool = None

        # Should not raise
        db.close_pool()

        assert db._pool is None


class TestGetCursor:
    """Test cursor context manager."""

    @patch("valence.core.db._get_healthy_connection")
    @patch("valence.core.db._get_pool")
    def test_cursor_commit_on_success(self, mock_get_pool, mock_get_healthy_conn):
        """Test cursor commits on successful execution."""
        from valence.core.db import get_cursor

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool
        mock_get_healthy_conn.return_value = mock_conn

        with get_cursor() as cur:
            assert cur is mock_cursor

        mock_conn.commit.assert_called_once()
        mock_conn.rollback.assert_not_called()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch("valence.core.db._get_healthy_connection")
    @patch("valence.core.db._get_pool")
    def test_cursor_rollback_on_error(self, mock_get_pool, mock_get_healthy_conn):
        """Test cursor rolls back on exception."""
        from valence.core.db import get_cursor

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(side_effect=ValueError("test error"))
        mock_conn.cursor.return_value = mock_cursor

        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool
        mock_get_healthy_conn.return_value = mock_conn

        with pytest.raises(ValueError):
            with get_cursor():
                raise ValueError("test error")

        mock_conn.rollback.assert_called_once()
        mock_conn.commit.assert_not_called()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch("valence.core.db._get_healthy_connection")
    @patch("valence.core.db._get_pool")
    def test_cursor_uses_realdict_factory(self, mock_get_pool, mock_get_healthy_conn):
        """Test cursor uses RealDictCursor factory."""
        from psycopg2.extras import RealDictCursor

        from valence.core.db import get_cursor

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool
        mock_get_healthy_conn.return_value = mock_conn

        with get_cursor():
            pass

        mock_conn.cursor.assert_called_once_with(cursor_factory=RealDictCursor)


class TestGetConnection:
    """Test connection context manager."""

    @patch("valence.core.db._get_healthy_connection")
    @patch("valence.core.db._get_pool")
    def test_connection_returned(self, mock_get_pool, mock_get_healthy_conn):
        """Test connection is returned to pool."""
        from valence.core.db import get_connection

        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool
        mock_get_healthy_conn.return_value = mock_conn

        with get_connection() as conn:
            assert conn is mock_conn

        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch("valence.core.db._get_healthy_connection")
    @patch("valence.core.db._get_pool")
    def test_connection_returned_on_error(self, mock_get_pool, mock_get_healthy_conn):
        """Test connection is returned even when error occurs."""
        from valence.core.db import get_connection

        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool
        mock_get_healthy_conn.return_value = mock_conn

        with pytest.raises(ValueError):
            with get_connection():
                raise ValueError("test error")

        mock_pool.putconn.assert_called_once_with(mock_conn)


class TestPutConnection:
    """Test put_connection helper."""

    @patch("valence.core.db._get_pool")
    def test_put_connection(self, mock_get_pool):
        """Test put_connection returns connection to pool."""
        from valence.core.db import put_connection

        mock_conn = MagicMock()
        mock_pool = MagicMock()
        mock_get_pool.return_value = mock_pool

        put_connection(mock_conn)

        mock_pool.putconn.assert_called_once_with(mock_conn)


class TestGenerateId:
    """Test ID generation."""

    def test_generate_id_returns_uuid_string(self):
        """Test generate_id returns valid UUID string."""
        from valence.core.db import generate_id

        result = generate_id()
        assert isinstance(result, str)
        # Should be parseable as UUID
        _uuid.UUID(result)

    def test_generate_id_unique(self):
        """Test generate_id returns unique values."""
        from valence.core.db import generate_id

        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestTableExists:
    """Test table_exists function."""

    @patch("valence.core.db.get_cursor")
    def test_table_exists_true(self, mock_get_cursor):
        """Test table_exists returns True when table exists."""
        from valence.core.db import table_exists

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"exists": True}
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = table_exists("sources")

        assert result is True
        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "information_schema.tables" in sql

    @patch("valence.core.db.get_cursor")
    def test_table_exists_false(self, mock_get_cursor):
        """Test table_exists returns False when table doesn't exist."""
        from valence.core.db import table_exists

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"exists": False}
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = table_exists("nonexistent")

        assert result is False

    @patch("valence.core.db.get_cursor")
    def test_table_exists_none_result(self, mock_get_cursor):
        """Test table_exists handles None result."""
        from valence.core.db import table_exists

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = table_exists("test")

        assert result is False


class TestCountRows:
    """Test count_rows function."""

    @patch("valence.core.db.get_cursor")
    def test_count_rows(self, mock_get_cursor):
        """Test count_rows returns correct count."""
        from valence.core.db import count_rows

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"count": 42}
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = count_rows("articles")

        assert result == 42

    @patch("valence.core.db.get_cursor")
    def test_count_rows_zero(self, mock_get_cursor):
        """Test count_rows returns zero for empty table."""
        from valence.core.db import count_rows

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"count": 0}
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = count_rows("empty_table")

        assert result == 0

    @patch("valence.core.db.get_cursor")
    def test_count_rows_none_result(self, mock_get_cursor):
        """Test count_rows handles None result."""
        from valence.core.db import count_rows

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = count_rows("test")

        assert result == 0


class TestGetSchemaVersion:
    """Test get_schema_version function."""

    @patch("valence.core.db.table_exists")
    @patch("valence.core.db.get_cursor")
    def test_get_schema_version_success(self, mock_get_cursor, mock_table_exists):
        """Test get_schema_version returns version string."""
        from valence.core.db import get_schema_version

        mock_table_exists.return_value = True

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = {"value": "2.0.0"}
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = get_schema_version()

        assert result == "2.0.0"

    @patch("valence.core.db.table_exists")
    def test_get_schema_version_no_table(self, mock_table_exists):
        """Test get_schema_version returns None when table doesn't exist."""
        from valence.core.db import get_schema_version

        mock_table_exists.return_value = False

        result = get_schema_version()

        assert result is None

    @patch("valence.core.db.table_exists")
    @patch("valence.core.db.get_cursor")
    def test_get_schema_version_no_row(self, mock_get_cursor, mock_table_exists):
        """Test get_schema_version returns None when no version row."""
        from valence.core.db import get_schema_version

        mock_table_exists.return_value = True

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = get_schema_version()

        assert result is None


class TestInitSchema:
    """Test init_schema function."""

    @patch("valence.core.db.get_connection")
    def test_init_schema_explicit_path(self, mock_get_connection):
        """Test init_schema with explicit schema path."""
        # Create a temp schema file
        import tempfile

        from valence.core.db import init_schema

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
            f.write("CREATE TABLE test (id UUID PRIMARY KEY);")
            schema_path = f.name

        try:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.__enter__ = Mock(return_value=mock_cursor)
            mock_cursor.__exit__ = Mock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_get_connection.return_value = mock_conn

            init_schema(schema_path)

            mock_cursor.execute.assert_called_once()
            sql = mock_cursor.execute.call_args[0][0]
            assert "CREATE TABLE test" in sql
        finally:
            os.unlink(schema_path)

    def test_init_schema_file_not_found(self):
        """Test init_schema raises when schema file not found."""
        from valence.core.db import init_schema

        with pytest.raises(FileNotFoundError):
            init_schema("/nonexistent/schema.sql")

    @patch("valence.core.db.get_connection")
    def test_init_schema_auto_detect(self, mock_get_connection):
        """Test init_schema with auto-detection of schema.sql."""
        # Create a temporary schema file in cwd
        from pathlib import Path

        from valence.core.db import init_schema

        cwd = Path.cwd()
        migrations_dir = cwd / "migrations"
        schema_file = migrations_dir / "schema.sql"
        original_content = schema_file.read_text() if schema_file.exists() else None
        try:
            migrations_dir.mkdir(exist_ok=True)
            schema_file.write_text("CREATE TABLE auto_test (id UUID);")

            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.__enter__ = Mock(return_value=mock_cursor)
            mock_cursor.__exit__ = Mock(return_value=False)
            mock_conn.cursor.return_value = mock_cursor
            mock_conn.__enter__ = Mock(return_value=mock_conn)
            mock_conn.__exit__ = Mock(return_value=False)
            mock_get_connection.return_value = mock_conn

            init_schema()

            mock_cursor.execute.assert_called_once()
            sql = mock_cursor.execute.call_args[0][0]
            assert "CREATE TABLE auto_test" in sql
        finally:
            if original_content is not None:
                schema_file.write_text(original_content)
            elif schema_file.exists():
                schema_file.unlink()

    def test_init_schema_auto_detect_not_found(self):
        """Test init_schema raises when auto-detect fails."""
        from pathlib import Path

        from valence.core.db import init_schema

        # Ensure no schema.sql exists in search paths
        cwd = Path.cwd()
        backups = []
        for candidate in [cwd / "schema.sql", cwd / "migrations" / "schema.sql"]:
            if candidate.exists():
                bak = candidate.with_suffix(".sql.bak")
                candidate.rename(bak)
                backups.append((bak, candidate))

        try:
            with pytest.raises(FileNotFoundError) as exc_info:
                init_schema()
            assert "schema.sql not found" in str(exc_info.value)
        finally:
            for bak, original in backups:
                if bak.exists():
                    bak.rename(original)


class TestCheckConnection:
    """Test check_connection function."""

    @patch("valence.core.db.get_cursor")
    def test_check_connection_success(self, mock_get_cursor):
        """Test check_connection returns True when connection works."""
        from valence.core.db import check_connection

        mock_cursor = MagicMock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_get_cursor.return_value = mock_cursor

        result = check_connection()

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")

    @patch("valence.core.db.get_cursor")
    def test_check_connection_failure(self, mock_get_cursor):
        """Test check_connection returns False on exception."""
        from valence.core.db import check_connection

        mock_get_cursor.side_effect = Exception("Connection failed")

        result = check_connection()

        assert result is False


class TestGetConnectionParams:
    """Test get_connection_params function."""

    def test_get_connection_params(self, monkeypatch):
        """Test get_connection_params returns config dict."""
        from valence.core.db import get_connection_params

        monkeypatch.setenv("VALENCE_DB_HOST", "testhost")

        params = get_connection_params()

        assert isinstance(params, dict)
        assert params["host"] == "testhost"


class TestSerializeRow:
    """Test serialize_row function."""

    def test_serialize_row_datetime(self):
        """Test serialize_row converts datetime to ISO string."""
        from valence.core.db import serialize_row

        dt = datetime(2024, 1, 15, 10, 30, 0)
        row = {"id": "123", "created_at": dt}

        result = serialize_row(row)

        assert result["created_at"] == "2024-01-15T10:30:00"

    def test_serialize_row_uuid(self):
        """Test serialize_row converts UUID to string."""
        from valence.core.db import serialize_row

        uid = _uuid.uuid4()
        row = {"id": uid, "name": "test"}

        result = serialize_row(row)

        assert result["id"] == str(uid)
        assert result["name"] == "test"

    def test_serialize_row_mixed(self):
        """Test serialize_row handles mixed types."""
        from valence.core.db import serialize_row

        uid = _uuid.uuid4()
        dt = datetime(2024, 1, 15, 10, 30, 0)
        row = {
            "id": uid,
            "created_at": dt,
            "name": "test",
            "count": 42,
            "active": True,
        }

        result = serialize_row(row)

        assert result["id"] == str(uid)
        assert result["created_at"] == "2024-01-15T10:30:00"
        assert result["name"] == "test"
        assert result["count"] == 42
        assert result["active"] is True

    def test_serialize_row_no_changes(self):
        """Test serialize_row doesn't modify primitives."""
        from valence.core.db import serialize_row

        row = {"id": "123", "name": "test", "count": 42}

        result = serialize_row(row)

        assert result == row
