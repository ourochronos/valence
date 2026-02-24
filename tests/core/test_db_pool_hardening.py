"""Tests for DB pool hardening features (#455, #457, #458).

Tests cover:
- Connection pool timeout (#455)
- Config-driven pool sizing (#457)
- Connection health checks (#458)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, Mock, patch

import pytest


class TestPoolTimeout:
    """Test connection pool timeout functionality (#455)."""

    def test_timeout_on_slow_pool(self):
        """Test timeout raises PoolError when pool.getconn() is slow."""
        from psycopg2.pool import PoolError

        from valence.core.db import _get_conn_with_timeout

        mock_pool = MagicMock()

        def slow_getconn():
            time.sleep(2)
            return MagicMock()

        mock_pool.getconn = slow_getconn

        with pytest.raises(PoolError) as exc_info:
            _get_conn_with_timeout(mock_pool, timeout=1)

        assert "timeout after 1 seconds" in str(exc_info.value)

    def test_timeout_success_fast_pool(self):
        """Test successful connection when pool.getconn() is fast."""
        from valence.core.db import _get_conn_with_timeout

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        result = _get_conn_with_timeout(mock_pool, timeout=5)

        assert result is mock_conn


class TestConnectionHealthCheck:
    """Test connection health check functionality (#458)."""

    def test_validate_connection_closed(self):
        """Test _validate_connection returns False for closed connection."""
        from valence.core.db import _validate_connection

        mock_conn = MagicMock()
        mock_conn.closed = True

        result = _validate_connection(mock_conn)

        assert result is False

    def test_validate_connection_healthy(self):
        """Test _validate_connection returns True for healthy connection."""
        from valence.core.db import _validate_connection

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = Mock(return_value=mock_cursor)
        mock_cursor.__exit__ = Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor

        result = _validate_connection(mock_conn)

        assert result is True
        mock_cursor.execute.assert_called_once_with("SELECT 1")
        mock_cursor.fetchone.assert_called_once()

    def test_validate_connection_exception(self):
        """Test _validate_connection returns False on exception."""
        from valence.core.db import _validate_connection

        mock_conn = MagicMock()
        mock_conn.closed = False
        mock_conn.cursor.side_effect = Exception("Connection error")

        result = _validate_connection(mock_conn)

        assert result is False

    @patch("valence.core.db._get_conn_with_timeout")
    @patch("valence.core.db._validate_connection")
    def test_get_healthy_connection_first_try(self, mock_validate, mock_get_timeout):
        """Test _get_healthy_connection succeeds on first try."""
        from valence.core.db import _get_healthy_connection

        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_get_timeout.return_value = mock_conn
        mock_validate.return_value = True

        result = _get_healthy_connection(mock_pool, timeout=30)

        assert result is mock_conn
        assert mock_get_timeout.call_count == 1
        assert mock_validate.call_count == 1

    @patch("valence.core.db._get_conn_with_timeout")
    @patch("valence.core.db._validate_connection")
    def test_get_healthy_connection_retry_stale(self, mock_validate, mock_get_timeout):
        """Test _get_healthy_connection retries on stale connection."""
        from psycopg2.pool import PoolError

        from valence.core.db import _get_healthy_connection

        mock_pool = MagicMock()
        mock_stale_conn = MagicMock()
        mock_get_timeout.return_value = mock_stale_conn
        # All attempts fail
        mock_validate.side_effect = [False, False, False]

        with pytest.raises(PoolError) as exc_info:
            _get_healthy_connection(mock_pool, timeout=30)

        assert "Failed to get healthy connection" in str(exc_info.value)
        assert mock_get_timeout.call_count == 3
        assert mock_validate.call_count == 3
        # Should close and putconn the stale connections
        assert mock_stale_conn.close.call_count == 3
        assert mock_pool.putconn.call_count == 3


class TestPoolConfigIntegration:
    """Test pool size configuration integration (#457)."""

    @patch("valence.core.db.psycopg2_pool.ThreadedConnectionPool")
    def test_pool_uses_config_values(self, mock_pool_class, monkeypatch):
        """Test pool uses values from CoreSettings."""
        from valence.core import db

        monkeypatch.setenv("VALENCE_DB_POOL_MIN", "3")
        monkeypatch.setenv("VALENCE_DB_POOL_MAX", "15")
        monkeypatch.setenv("VALENCE_DB_POOL_TIMEOUT", "45")

        # Clear config cache if it exists
        try:
            from valence.core.config import clear_config_cache

            clear_config_cache()
        except ImportError:
            pass

        db._pool = None

        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        result = db._get_pool()

        assert result is mock_pool
        call_kwargs = mock_pool_class.call_args[1]
        assert call_kwargs["minconn"] == 3
        assert call_kwargs["maxconn"] == 15
