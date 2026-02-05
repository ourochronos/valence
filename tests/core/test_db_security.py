"""Tests for database security - Issue #24 SQL injection prevention."""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock

from valence.core.db import count_rows, VALID_TABLES


class TestSQLInjectionPrevention:
    """Test SQL injection prevention in count_rows()."""

    def test_valid_table_in_allowlist(self):
        """count_rows should accept tables in the VALID_TABLES allowlist."""
        # All tables in allowlist should pass the allowlist check
        for table in VALID_TABLES:
            # Just check it doesn't raise ValueError for allowlist
            # (will fail at DB level if not connected, which is fine)
            try:
                count_rows(table)
            except ValueError as e:
                # Should NOT be an allowlist error
                assert "not in allowlist" not in str(e)
            except Exception:
                # Other errors (DB connection, etc.) are expected in test env
                pass

    def test_invalid_table_rejected_before_query(self):
        """count_rows should reject tables not in allowlist BEFORE any DB query."""
        with pytest.raises(ValueError) as exc_info:
            count_rows("nonexistent_table")

        assert "not in allowlist" in str(exc_info.value)
        assert "nonexistent_table" in str(exc_info.value)

    def test_sql_injection_attempt_rejected(self):
        """SQL injection attempts should be rejected by allowlist."""
        injection_attempts = [
            "beliefs; DROP TABLE beliefs;--",
            "beliefs UNION SELECT * FROM users--",
            "beliefs' OR '1'='1",
            "beliefs; DELETE FROM beliefs;--",
            "beliefs); DROP TABLE beliefs;--",
            "../../../etc/passwd",
            "beliefs\n; DROP TABLE beliefs;",
            "beliefs\x00; DROP TABLE beliefs;",
        ]

        for attempt in injection_attempts:
            with pytest.raises(ValueError) as exc_info:
                count_rows(attempt)

            assert "not in allowlist" in str(exc_info.value)

    def test_empty_table_name_rejected(self):
        """Empty table name should be rejected."""
        with pytest.raises(ValueError) as exc_info:
            count_rows("")

        assert "not in allowlist" in str(exc_info.value)

    def test_case_sensitivity(self):
        """Table names should be case-sensitive (lowercase only)."""
        # PostgreSQL lowercases unquoted identifiers, so our allowlist is lowercase
        with pytest.raises(ValueError):
            count_rows("BELIEFS")

        with pytest.raises(ValueError):
            count_rows("Beliefs")

    def test_allowlist_is_immutable(self):
        """VALID_TABLES should be immutable (frozenset)."""
        assert isinstance(VALID_TABLES, frozenset)

        # Should not be able to add to it
        with pytest.raises(AttributeError):
            VALID_TABLES.add("malicious_table")

    def test_valid_tables_contains_expected_tables(self):
        """VALID_TABLES should contain the expected system tables."""
        expected = {
            "beliefs",
            "entities",
            "vkb_sessions",
            "vkb_exchanges",
            "vkb_patterns",
            "tensions",
        }
        assert expected.issubset(VALID_TABLES)

    @patch("valence.core.db.get_cursor")
    def test_allowlist_check_happens_before_db_query(self, mock_get_cursor):
        """Allowlist check should happen BEFORE any database query."""
        # If we try an invalid table, get_cursor should never be called
        mock_cursor = MagicMock()
        mock_get_cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)

        with pytest.raises(ValueError) as exc_info:
            count_rows("malicious_table; DROP TABLE beliefs;--")

        # The cursor should never have been used
        mock_cursor.execute.assert_not_called()
        assert "not in allowlist" in str(exc_info.value)

    @patch("valence.core.db.get_cursor")
    def test_count_rows_uses_sql_identifier(self, mock_get_cursor):
        """count_rows should use sql.Identifier for safe table name interpolation.

        Issue #172: Use psycopg2's sql.Identifier for defense in depth,
        even though the allowlist provides primary protection.
        """
        from psycopg2 import sql

        mock_cursor = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_cursor)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_get_cursor.return_value = mock_ctx

        # Mock the table existence check
        mock_cursor.fetchone.side_effect = [
            {"table_name": "beliefs"},  # First call: table exists
            {"count": 42},  # Second call: the count result
        ]

        result = count_rows("beliefs")

        # Verify count_rows called execute twice
        assert mock_cursor.execute.call_count == 2

        # Second execute call should use sql.SQL/sql.Identifier (composed SQL)
        second_call_args = mock_cursor.execute.call_args_list[1][0]
        query = second_call_args[0]

        # sql.Composed or sql.SQL objects are used with sql.Identifier
        assert isinstance(query, (sql.SQL, sql.Composed)), \
            f"Expected sql.SQL or sql.Composed, got {type(query)}"

        assert result == 42
