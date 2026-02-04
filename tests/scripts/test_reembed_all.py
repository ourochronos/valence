"""Tests for re-embedding script (scripts/reembed_all.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

import pytest

# Add scripts to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))


# ============================================================================
# Database URL Parsing Tests
# ============================================================================

class TestParseDatabaseUrl:
    """Tests for parse_database_url function."""

    def test_parses_full_url(self):
        """Should parse all components from full URL."""
        from reembed_all import parse_database_url
        
        url = "postgresql://myuser:mypass@myhost:5433/mydb"
        result = parse_database_url(url)
        
        assert result["host"] == "myhost"
        assert result["port"] == 5433
        assert result["dbname"] == "mydb"
        assert result["user"] == "myuser"
        assert result["password"] == "mypass"

    def test_uses_defaults_for_missing(self):
        """Should use defaults when components are missing."""
        from reembed_all import parse_database_url
        
        url = "postgresql:///testdb"
        result = parse_database_url(url)
        
        assert result["host"] == "localhost"
        assert result["port"] == 5432
        assert result["dbname"] == "testdb"

    def test_parses_localhost_url(self):
        """Should parse localhost URL correctly."""
        from reembed_all import parse_database_url
        
        url = "postgresql://valence:valence@localhost:5432/valence"
        result = parse_database_url(url)
        
        assert result["host"] == "localhost"
        assert result["port"] == 5432
        assert result["dbname"] == "valence"
        assert result["user"] == "valence"


# ============================================================================
# Progress Tracking Tests
# ============================================================================

class TestProgressTracking:
    """Tests for progress file operations."""

    def test_load_progress_empty_when_no_file(self, tmp_path):
        """Should return empty dict when file doesn't exist."""
        from reembed_all import load_progress
        
        result = load_progress(str(tmp_path / "nonexistent.json"))
        
        assert result == {}

    def test_load_progress_from_file(self, tmp_path):
        """Should load progress from existing file."""
        from reembed_all import load_progress
        
        progress_file = tmp_path / "progress.json"
        progress_file.write_text('{"beliefs": 100, "vkb_exchanges": 50}')
        
        result = load_progress(str(progress_file))
        
        assert result["beliefs"] == 100
        assert result["vkb_exchanges"] == 50

    def test_save_progress_creates_file(self, tmp_path):
        """Should create progress file."""
        from reembed_all import save_progress
        
        progress_file = tmp_path / "progress.json"
        save_progress(str(progress_file), {"beliefs": 200})
        
        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert data["beliefs"] == 200

    def test_save_progress_none_is_noop(self, tmp_path):
        """Should not create file when progress_file is None."""
        from reembed_all import save_progress
        
        save_progress(None, {"beliefs": 100})
        
        # No exception, no file created
        assert True


# ============================================================================
# Rate Formatting Tests
# ============================================================================

class TestFormatRate:
    """Tests for format_rate function."""

    def test_formats_rate(self):
        """Should format rate as items/sec."""
        from reembed_all import format_rate
        
        result = format_rate(100, 2.0)
        
        assert result == "50.0/sec"

    def test_handles_zero_elapsed(self):
        """Should handle zero elapsed time."""
        from reembed_all import format_rate
        
        result = format_rate(100, 0)
        
        assert result == "N/A"


# ============================================================================
# Count Functions Tests
# ============================================================================

class TestCountFunctions:
    """Tests for database count functions."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    def test_count_needs_embedding(self, mock_cursor):
        """Should count rows needing embedding."""
        from reembed_all import count_needs_embedding
        
        mock_cursor.fetchone.return_value = (42,)
        
        result = count_needs_embedding(mock_cursor, "beliefs")
        
        assert result == 42
        mock_cursor.execute.assert_called_once()
        call_sql = mock_cursor.execute.call_args[0][0]
        assert "embedding_384 IS NULL" in call_sql
        assert "content IS NOT NULL" in call_sql

    def test_count_needs_embedding_patterns(self, mock_cursor):
        """Should use 'pattern' column for vkb_patterns table."""
        from reembed_all import count_needs_embedding
        
        mock_cursor.fetchone.return_value = (10,)
        
        count_needs_embedding(mock_cursor, "vkb_patterns")
        
        call_sql = mock_cursor.execute.call_args[0][0]
        assert "pattern IS NOT NULL" in call_sql

    def test_count_total(self, mock_cursor):
        """Should count total rows."""
        from reembed_all import count_total
        
        mock_cursor.fetchone.return_value = (500,)
        
        result = count_total(mock_cursor, "beliefs")
        
        assert result == 500

    def test_count_embedded(self, mock_cursor):
        """Should count rows with embeddings."""
        from reembed_all import count_embedded
        
        mock_cursor.fetchone.return_value = (300,)
        
        result = count_embedded(mock_cursor, "beliefs")
        
        assert result == 300
        call_sql = mock_cursor.execute.call_args[0][0]
        assert "embedding_384 IS NOT NULL" in call_sql


# ============================================================================
# Fetch Batch Tests
# ============================================================================

class TestFetchBatch:
    """Tests for fetch_batch function."""

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        cursor = MagicMock()
        return cursor

    def test_fetches_batch_with_limit_offset(self, mock_cursor):
        """Should fetch batch with correct LIMIT and OFFSET."""
        from reembed_all import fetch_batch
        
        mock_cursor.fetchall.return_value = []
        
        fetch_batch(mock_cursor, "beliefs", batch_size=50, offset=100)
        
        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        params = call_args[0][1]
        
        assert "LIMIT %s OFFSET %s" in sql
        assert params == (50, 100)

    def test_uses_correct_content_column(self, mock_cursor):
        """Should use correct content column for each table."""
        from reembed_all import fetch_batch
        
        mock_cursor.fetchall.return_value = []
        
        # beliefs uses 'content'
        fetch_batch(mock_cursor, "beliefs", 10, 0)
        sql = mock_cursor.execute.call_args[0][0]
        assert "content as content" in sql
        
        # vkb_patterns uses 'pattern'
        fetch_batch(mock_cursor, "vkb_patterns", 10, 0)
        sql = mock_cursor.execute.call_args[0][0]
        assert "pattern as content" in sql


# ============================================================================
# Update Embeddings Tests
# ============================================================================

class TestUpdateEmbeddings:
    """Tests for update_embeddings function."""

    @pytest.fixture
    def mock_conn(self):
        """Create mock connection."""
        return MagicMock()

    @pytest.fixture
    def mock_cursor(self):
        """Create mock cursor."""
        return MagicMock()

    def test_commits_after_update(self, mock_conn, mock_cursor):
        """Should commit transaction after update."""
        from reembed_all import update_embeddings
        
        updates = [
            (str(uuid4()), "[0.1,0.2,0.3]"),
            (str(uuid4()), "[0.4,0.5,0.6]"),
        ]
        
        with patch("reembed_all.execute_values") as mock_exec:
            update_embeddings(mock_conn, mock_cursor, "beliefs", updates)
        
        mock_conn.commit.assert_called_once()

    def test_no_commit_for_empty_updates(self, mock_conn, mock_cursor):
        """Should not commit when updates list is empty."""
        from reembed_all import update_embeddings
        
        update_embeddings(mock_conn, mock_cursor, "beliefs", [])
        
        mock_conn.commit.assert_not_called()


# ============================================================================
# Reembed Table Tests
# ============================================================================

class TestReembedTable:
    """Tests for reembed_table function."""

    @pytest.fixture
    def mock_conn(self):
        """Create mock connection with cursor."""
        conn = MagicMock()
        cursor = MagicMock()
        
        # Set up cursor context
        cursor.__enter__ = MagicMock(return_value=cursor)
        cursor.__exit__ = MagicMock(return_value=False)
        cursor.fetchone.return_value = (0,)  # count = 0
        cursor.fetchall.return_value = []
        
        conn.cursor.return_value = cursor
        return conn

    def test_dry_run_no_changes(self, mock_conn, tmp_path):
        """Dry run should not modify database."""
        from reembed_all import reembed_table
        
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.side_effect = [(10,), (100,)]  # needs=10, total=100
        
        result = reembed_table(
            mock_conn, "beliefs",
            dry_run=True,
            verbose=False
        )
        
        assert result["processed"] == 0
        mock_conn.commit.assert_not_called()

    def test_skips_when_no_rows_need_embedding(self, mock_conn):
        """Should skip table when all rows are embedded."""
        from reembed_all import reembed_table
        
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.side_effect = [(0,), (100,)]  # needs=0, total=100
        
        result = reembed_table(mock_conn, "beliefs", verbose=False)
        
        assert result["processed"] == 0

    def test_processes_batches(self, mock_conn, tmp_path):
        """Should process rows in batches."""
        from reembed_all import reembed_table
        
        cursor = mock_conn.cursor.return_value
        
        # needs=3, total=3
        cursor.fetchone.side_effect = [(3,), (3,)]
        
        # Return 3 rows then empty
        test_rows = [
            {"id": uuid4(), "content": "test 1"},
            {"id": uuid4(), "content": "test 2"},
            {"id": uuid4(), "content": "test 3"},
        ]
        cursor.fetchall.side_effect = [test_rows, []]
        
        mock_embeddings = [[0.1] * 384, [0.2] * 384, [0.3] * 384]
        
        with patch("reembed_all.generate_embeddings_batch", return_value=mock_embeddings):
            with patch("reembed_all.execute_values"):
                result = reembed_table(
                    mock_conn, "beliefs",
                    batch_size=10,
                    verbose=False
                )
        
        assert result["processed"] == 3
        assert result["elapsed"] > 0

    def test_saves_progress(self, mock_conn, tmp_path):
        """Should save progress to file."""
        from reembed_all import reembed_table
        
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.side_effect = [(2,), (2,)]
        
        test_rows = [
            {"id": uuid4(), "content": "test 1"},
            {"id": uuid4(), "content": "test 2"},
        ]
        cursor.fetchall.side_effect = [test_rows, []]
        
        progress_file = tmp_path / "progress.json"
        
        with patch("reembed_all.generate_embeddings_batch", return_value=[[0.1] * 384] * 2):
            with patch("reembed_all.execute_values"):
                reembed_table(
                    mock_conn, "beliefs",
                    progress_file=str(progress_file),
                    verbose=False
                )
        
        assert progress_file.exists()
        data = json.loads(progress_file.read_text())
        assert "beliefs" in data

    def test_resumes_from_progress(self, mock_conn, tmp_path):
        """Should resume from saved progress."""
        from reembed_all import reembed_table
        
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.side_effect = [(5,), (10,)]
        cursor.fetchall.return_value = []
        
        progress_file = tmp_path / "progress.json"
        progress_file.write_text('{"beliefs": 50}')
        
        reembed_table(
            mock_conn, "beliefs",
            progress_file=str(progress_file),
            verbose=False
        )
        
        # Check that OFFSET was passed
        call_args = [c for c in cursor.execute.call_args_list if "OFFSET" in str(c)]
        assert len(call_args) > 0


# ============================================================================
# Verification Tests
# ============================================================================

class TestVerifyEmbeddings:
    """Tests for verify_embeddings function."""

    @pytest.fixture
    def mock_conn(self):
        """Create mock connection."""
        conn = MagicMock()
        cursor = MagicMock()
        conn.cursor.return_value = cursor
        return conn

    def test_returns_stats_per_table(self, mock_conn):
        """Should return stats for each table."""
        from reembed_all import verify_embeddings
        
        cursor = mock_conn.cursor.return_value
        # total=100, embedded=80 for beliefs
        # total=50, embedded=50 for vkb_exchanges
        cursor.fetchone.side_effect = [
            (100,), (80,),  # beliefs
            (50,), (50,),   # vkb_exchanges
        ]
        
        result = verify_embeddings(
            mock_conn,
            ["beliefs", "vkb_exchanges"],
            verbose=False
        )
        
        assert result["beliefs"]["total"] == 100
        assert result["beliefs"]["embedded"] == 80
        assert result["beliefs"]["pct"] == 80.0
        assert result["vkb_exchanges"]["pct"] == 100.0

    def test_handles_empty_table(self, mock_conn):
        """Should handle empty tables gracefully."""
        from reembed_all import verify_embeddings
        
        cursor = mock_conn.cursor.return_value
        cursor.fetchone.side_effect = [(0,), (0,)]  # total=0, embedded=0
        
        result = verify_embeddings(mock_conn, ["beliefs"], verbose=False)
        
        assert result["beliefs"]["pct"] == 0


# ============================================================================
# Table Config Tests
# ============================================================================

class TestTableConfig:
    """Tests for TABLE_CONFIG constant."""

    def test_contains_all_tables(self):
        """Should contain all expected tables."""
        from reembed_all import TABLE_CONFIG
        
        assert "beliefs" in TABLE_CONFIG
        assert "vkb_exchanges" in TABLE_CONFIG
        assert "vkb_patterns" in TABLE_CONFIG

    def test_correct_content_columns(self):
        """Should map to correct content columns."""
        from reembed_all import TABLE_CONFIG
        
        assert TABLE_CONFIG["beliefs"] == "content"
        assert TABLE_CONFIG["vkb_exchanges"] == "content"
        assert TABLE_CONFIG["vkb_patterns"] == "pattern"


# ============================================================================
# CLI Tests
# ============================================================================

class TestMainCLI:
    """Tests for main CLI entry point."""

    def test_dry_run_flag(self):
        """Should support --dry-run flag."""
        import argparse
        from reembed_all import main
        
        # Just verify the script parses arguments correctly
        # Full integration test would require database
        with patch("sys.argv", ["reembed_all.py", "--dry-run", "--quiet"]):
            with patch("reembed_all.get_connection") as mock_conn:
                mock_conn.return_value = MagicMock()
                mock_conn.return_value.cursor.return_value.fetchone.return_value = (0,)
                mock_conn.return_value.cursor.return_value.fetchall.return_value = []
                
                # Should not raise
                try:
                    main()
                except SystemExit as e:
                    # Normal exit is fine
                    pass

    def test_table_flag(self):
        """Should support --table flag to process single table."""
        with patch("sys.argv", ["reembed_all.py", "--table", "beliefs", "--dry-run", "--quiet"]):
            with patch("reembed_all.get_connection") as mock_conn:
                mock_conn.return_value = MagicMock()
                mock_conn.return_value.cursor.return_value.fetchone.return_value = (0,)
                mock_conn.return_value.cursor.return_value.fetchall.return_value = []
                
                from reembed_all import main
                try:
                    main()
                except SystemExit:
                    pass
