"""Tests for the embeddings CLI commands.

Tests cover:
1. Argument parsing for `valence embeddings backfill`
2. Dry-run mode (no mutations, shows counts)
3. Content-type filtering
4. Force mode (re-embed all)
5. Normal backfill (delegates to service)
6. Error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.cli.main import app, cmd_embeddings

# ============================================================================
# Argument Parsing
# ============================================================================


class TestEmbeddingsArgParsing:
    """Test CLI argument parsing for embeddings commands."""

    def test_embeddings_backfill_defaults(self):
        """Parse embeddings backfill with defaults."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill"])
        assert args.command == "embeddings"
        assert args.embeddings_command == "backfill"
        assert args.batch_size == 100
        assert args.dry_run is False
        assert args.content_type is None
        assert args.force is False

    def test_embeddings_backfill_batch_size(self):
        """Parse --batch-size flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--batch-size", "50"])
        assert args.batch_size == 50

    def test_embeddings_backfill_batch_size_short(self):
        """Parse -b short flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "-b", "200"])
        assert args.batch_size == 200

    def test_embeddings_backfill_dry_run(self):
        """Parse --dry-run flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--dry-run"])
        assert args.dry_run is True

    def test_embeddings_backfill_content_type(self):
        """Parse --content-type flag."""
        parser = app()
        for ct in ("belief", "exchange", "pattern"):
            args = parser.parse_args(["embeddings", "backfill", "--content-type", ct])
            assert args.content_type == ct

    def test_embeddings_backfill_content_type_short(self):
        """Parse -t short flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "-t", "belief"])
        assert args.content_type == "belief"

    def test_embeddings_backfill_content_type_invalid(self):
        """Invalid content type raises error."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["embeddings", "backfill", "--content-type", "invalid"])

    def test_embeddings_backfill_force(self):
        """Parse --force flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--force"])
        assert args.force is True

    def test_embeddings_backfill_force_short(self):
        """Parse -f short flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "-f"])
        assert args.force is True

    def test_embeddings_backfill_all_flags(self):
        """Parse all flags together."""
        parser = app()
        args = parser.parse_args(
            [
                "embeddings",
                "backfill",
                "--batch-size",
                "25",
                "--dry-run",
                "--content-type",
                "belief",
                "--force",
            ]
        )
        assert args.batch_size == 25
        assert args.dry_run is True
        assert args.content_type == "belief"
        assert args.force is True

    def test_embeddings_requires_subcommand(self):
        """embeddings without subcommand raises error."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["embeddings"])


# ============================================================================
# Mock Database Fixture
# ============================================================================


@pytest.fixture
def mock_db():
    """Create a mock database connection and cursor."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    return mock_conn, mock_cur


# ============================================================================
# Dry Run Tests
# ============================================================================


class TestBackfillDryRun:
    """Test dry-run mode shows counts without mutating."""

    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_dry_run_shows_counts(self, mock_get_conn, mock_db, capsys):
        """Dry run shows missing counts per type."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        # Return counts for belief, exchange, pattern
        mock_cur.fetchone.side_effect = [
            {"count": 10},  # beliefs missing
            {"count": 5},  # exchanges missing
            {"count": 3},  # patterns missing
        ]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--dry-run"])
        result = cmd_embeddings(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Dry run" in captured.out
        assert "10" in captured.out
        assert "5" in captured.out
        assert "3" in captured.out
        assert "18" in captured.out  # total

    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_dry_run_single_type(self, mock_get_conn, mock_db, capsys):
        """Dry run with --content-type shows only that type."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchone.side_effect = [
            {"count": 15},  # beliefs missing
        ]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--dry-run", "-t", "belief"])
        result = cmd_embeddings(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Dry run" in captured.out
        assert "15" in captured.out

    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_dry_run_nothing_to_backfill(self, mock_get_conn, mock_db, capsys):
        """Dry run when nothing needs backfilling."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchone.side_effect = [
            {"count": 0},
            {"count": 0},
            {"count": 0},
        ]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--dry-run"])
        result = cmd_embeddings(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Nothing to backfill" in captured.out

    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_dry_run_force_counts_all(self, mock_get_conn, mock_db, capsys):
        """Dry run with --force counts all records, not just missing."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        # Force mode counts all active records
        mock_cur.fetchone.side_effect = [
            {"count": 100},  # all beliefs
            {"count": 50},  # all exchanges
            {"count": 25},  # all patterns
        ]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--dry-run", "--force"])
        result = cmd_embeddings(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Dry run" in captured.out
        assert "Force mode" in captured.out
        assert "175" in captured.out  # total


# ============================================================================
# Normal Backfill Tests
# ============================================================================


class TestBackfillNormal:
    """Test normal backfill delegates to service layer."""

    @patch("valence.embeddings.service.backfill_embeddings", return_value=5)
    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_backfill_calls_service(self, mock_get_conn, mock_backfill, mock_db, capsys):
        """Backfill delegates to backfill_embeddings service."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        # Counts
        mock_cur.fetchone.side_effect = [
            {"count": 5},  # beliefs missing
        ]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "-t", "belief"])
        result = cmd_embeddings(args)

        assert result == 0
        mock_backfill.assert_called_once_with("belief", batch_size=100)

    @patch("valence.embeddings.service.backfill_embeddings", side_effect=[3, 2, 1])
    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_backfill_all_types(self, mock_get_conn, mock_backfill, mock_db, capsys):
        """Backfill processes all types when no filter."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchone.side_effect = [
            {"count": 3},
            {"count": 2},
            {"count": 1},
        ]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill"])
        result = cmd_embeddings(args)

        assert result == 0
        assert mock_backfill.call_count == 3
        captured = capsys.readouterr()
        assert "complete" in captured.out.lower()

    @patch("valence.embeddings.service.backfill_embeddings", return_value=10)
    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_backfill_custom_batch_size(self, mock_get_conn, mock_backfill, mock_db, capsys):
        """Batch size is passed through to service."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchone.side_effect = [{"count": 10}]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "-t", "belief", "-b", "25"])
        result = cmd_embeddings(args)

        assert result == 0
        mock_backfill.assert_called_once_with("belief", batch_size=25)

    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_backfill_nothing_to_do(self, mock_get_conn, mock_db, capsys):
        """Backfill exits cleanly when nothing needs embedding."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchone.side_effect = [
            {"count": 0},
            {"count": 0},
            {"count": 0},
        ]

        parser = app()
        args = parser.parse_args(["embeddings", "backfill"])
        result = cmd_embeddings(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Nothing to backfill" in captured.out


# ============================================================================
# Force Mode Tests
# ============================================================================


class TestBackfillForce:
    """Test --force mode re-embeds existing records."""

    @patch("valence.embeddings.service.embed_content", return_value={})
    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_force_reembeds_all(self, mock_get_conn, mock_embed, mock_db, capsys):
        """Force mode fetches and re-embeds all records."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        belief_id = uuid4()

        # Count query returns total active beliefs
        mock_cur.fetchone.side_effect = [{"count": 1}]
        # Fetch records for force re-embed
        mock_cur.fetchall.return_value = [
            {"id": belief_id, "content": "Test belief"},
        ]

        parser = app()
        args = parser.parse_args(
            [
                "embeddings",
                "backfill",
                "--force",
                "-t",
                "belief",
            ]
        )
        result = cmd_embeddings(args)

        assert result == 0
        mock_embed.assert_called_once_with("belief", str(belief_id), "Test belief")
        captured = capsys.readouterr()
        assert "Force mode" in captured.out


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestBackfillErrors:
    """Test error handling in backfill."""

    @patch("valence.cli.commands.embeddings.get_db_connection")
    def test_db_error_returns_nonzero(self, mock_get_conn, capsys):
        """Database connection failure returns non-zero exit code."""
        mock_get_conn.side_effect = Exception("Connection refused")

        parser = app()
        args = parser.parse_args(["embeddings", "backfill"])
        result = cmd_embeddings(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "failed" in captured.err.lower()


# ============================================================================
# Command Routing
# ============================================================================


class TestEmbeddingsRouting:
    """Test embeddings command routing."""

    def test_main_dispatches_to_embeddings(self):
        """main() dispatches 'embeddings' to cmd_embeddings."""
        from valence.cli.main import main

        with patch("valence.cli.main.cmd_embeddings", return_value=0) as mock_cmd:
            with patch("sys.argv", ["valence", "embeddings", "backfill", "--dry-run"]):
                result = main()
                assert result == 0
                mock_cmd.assert_called_once()
