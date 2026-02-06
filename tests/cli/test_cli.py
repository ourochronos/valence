"""Tests for Valence CLI.

Tests cover:
1. CLI argument parsing
2. Command dispatch
3. Derivation chain visibility
4. Conflict detection
5. Init/add/query/list happy paths
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.cli.main import (
    app,
    cmd_add,
    cmd_conflicts,
    cmd_init,
    cmd_list,
    cmd_query,
    cmd_stats,
    compute_confidence_score,
    compute_recency_score,
    format_age,
    format_confidence,
    multi_signal_rank,
)

# ============================================================================
# Unit Tests - Pure Functions
# ============================================================================


class TestFormatConfidence:
    """Test confidence formatting."""

    def test_format_overall(self):
        """Format overall confidence."""
        assert format_confidence({"overall": 0.8}) == "80%"
        assert format_confidence({"overall": 0.95}) == "95%"
        assert format_confidence({"overall": 0.123}) == "12%"

    def test_format_empty(self):
        """Format empty confidence."""
        assert format_confidence({}) == "?"
        assert format_confidence(None) == "?"

    def test_format_non_numeric(self):
        """Format non-numeric overall."""
        # Should truncate to 5 chars
        result = format_confidence({"overall": "high"})
        assert len(result) <= 5


class TestFormatAge:
    """Test age formatting."""

    def test_format_recent(self):
        """Format very recent time."""
        now = datetime.now(UTC)
        assert format_age(now) == "now"
        assert format_age(now - timedelta(seconds=30)) == "now"

    def test_format_minutes(self):
        """Format minutes ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(minutes=5)) == "5m"
        assert format_age(now - timedelta(minutes=59)) == "59m"

    def test_format_hours(self):
        """Format hours ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(hours=3)) == "3h"
        assert format_age(now - timedelta(hours=23)) == "23h"

    def test_format_days(self):
        """Format days ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(days=5)) == "5d"
        assert format_age(now - timedelta(days=29)) == "29d"

    def test_format_months(self):
        """Format months ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(days=45)) == "1mo"
        assert format_age(now - timedelta(days=180)) == "6mo"

    def test_format_years(self):
        """Format years ago."""
        now = datetime.now(UTC)
        assert format_age(now - timedelta(days=400)) == "1y"
        assert format_age(now - timedelta(days=800)) == "2y"

    def test_format_none(self):
        """Format None datetime."""
        assert format_age(None) == "?"

    def test_format_naive_datetime(self):
        """Format naive datetime (no timezone) - gets treated as UTC."""
        # Note: naive datetime gets treated as UTC, so the result depends on local TZ
        now = datetime.now()
        result = format_age(now - timedelta(hours=2))
        # Just verify it returns something reasonable (not "?")
        assert result != "?"
        assert any(c in result for c in ["h", "m", "d", "y", "now", "mo"])


class TestArgumentParser:
    """Test CLI argument parsing."""

    def test_init_command(self):
        """Parse init command."""
        parser = app()
        args = parser.parse_args(["init"])
        assert args.command == "init"
        assert args.force is False

    def test_init_force(self):
        """Parse init with force flag."""
        parser = app()
        args = parser.parse_args(["init", "--force"])
        assert args.force is True

    def test_add_command(self):
        """Parse add command."""
        parser = app()
        args = parser.parse_args(["add", "Test belief content"])
        assert args.command == "add"
        assert args.content == "Test belief content"

    def test_add_with_options(self):
        """Parse add with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "add",
                "Test belief",
                "--confidence",
                "0.9",
                "--domain",
                "tech",
                "--domain",
                "python",
                "--derivation-type",
                "inference",
                "--derived-from",
                "12345678-1234-1234-1234-123456789abc",
                "--method",
                "Derived from observation",
            ]
        )
        assert args.content == "Test belief"
        assert args.confidence == "0.9"
        assert args.domain == ["tech", "python"]
        assert args.derivation_type == "inference"
        assert args.derived_from == "12345678-1234-1234-1234-123456789abc"
        assert args.method == "Derived from observation"

    def test_query_command(self):
        """Parse query command."""
        parser = app()
        args = parser.parse_args(["query", "search terms"])
        assert args.command == "query"
        assert args.query == "search terms"
        assert args.limit == 10  # default
        assert args.threshold == 0.3  # default

    def test_query_with_options(self):
        """Parse query with all options."""
        parser = app()
        args = parser.parse_args(
            [
                "query",
                "search terms",
                "--limit",
                "20",
                "--threshold",
                "0.5",
                "--domain",
                "tech",
                "--chain",
            ]
        )
        assert args.limit == 20
        assert args.threshold == 0.5
        assert args.domain == "tech"
        assert args.chain is True

    def test_list_command(self):
        """Parse list command."""
        parser = app()
        args = parser.parse_args(["list"])
        assert args.command == "list"
        assert args.limit == 10  # default

    def test_list_with_options(self):
        """Parse list with options."""
        parser = app()
        args = parser.parse_args(["list", "--limit", "50", "--domain", "tech"])
        assert args.limit == 50
        assert args.domain == "tech"

    def test_conflicts_command(self):
        """Parse conflicts command."""
        parser = app()
        args = parser.parse_args(["conflicts"])
        assert args.command == "conflicts"
        assert args.threshold == 0.85  # default
        assert args.auto_record is False

    def test_conflicts_with_options(self):
        """Parse conflicts with options."""
        parser = app()
        args = parser.parse_args(["conflicts", "--threshold", "0.9", "--auto-record"])
        assert args.threshold == 0.9
        assert args.auto_record is True

    def test_stats_command(self):
        """Parse stats command."""
        parser = app()
        args = parser.parse_args(["stats"])
        assert args.command == "stats"


# ============================================================================
# Integration Tests with Mocked Database
# ============================================================================


@pytest.fixture
def mock_db():
    """Create a mock database connection."""
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_conn.cursor.return_value = mock_cur
    mock_cur.__enter__ = MagicMock(return_value=mock_cur)
    mock_cur.__exit__ = MagicMock(return_value=False)
    return mock_conn, mock_cur


class TestInitCommand:
    """Test init command."""

    @patch("valence.cli.commands.beliefs.get_db_connection")
    def test_init_already_exists(self, mock_get_conn, mock_db):
        """Init when schema already exists."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        # Schema exists
        mock_cur.fetchone.side_effect = [
            {"exists": True},  # beliefs table exists
            {"count": 42},  # belief count
        ]

        parser = app()
        args = parser.parse_args(["init"])
        result = cmd_init(args)

        assert result == 0

    @patch("valence.cli.commands.beliefs.get_db_connection")
    def test_init_creates_schema(self, mock_get_conn, mock_db):
        """Init creates schema when not exists."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        # Schema doesn't exist, then creation succeeds
        mock_cur.fetchone.side_effect = [
            {"exists": False},  # beliefs table doesn't exist
        ]

        parser = app()
        args = parser.parse_args(["init"])
        result = cmd_init(args)

        # Should have executed CREATE TABLE statements
        assert mock_cur.execute.called
        assert result == 0


class TestAddCommand:
    """Test add command."""

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_add_basic(self, mock_embed, mock_get_conn, mock_db):
        """Add basic belief."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = None  # No embedding

        belief_id = uuid4()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "created_at": datetime.now(UTC),
        }

        parser = app()
        args = parser.parse_args(["add", "Test belief content"])
        result = cmd_add(args)

        assert result == 0
        # Verify INSERT was called
        insert_calls = [c for c in mock_cur.execute.call_args_list if "INSERT INTO beliefs" in str(c)]
        assert len(insert_calls) >= 1

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_add_with_derivation(self, mock_embed, mock_get_conn, mock_db):
        """Add belief with derivation info."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = [0.1] * 1536  # Mock embedding

        belief_id = uuid4()
        mock_cur.fetchone.return_value = {
            "id": belief_id,
            "created_at": datetime.now(UTC),
        }

        parser = app()
        args = parser.parse_args(
            [
                "add",
                "Derived belief",
                "--derivation-type",
                "inference",
                "--method",
                "Derived from logic",
            ]
        )
        result = cmd_add(args)

        assert result == 0
        # Check derivation insert
        insert_calls = [c for c in mock_cur.execute.call_args_list if "INSERT INTO belief_derivations" in str(c)]
        assert len(insert_calls) == 1


class TestQueryCommand:
    """Test query command with derivation chains."""

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_query_shows_derivation(self, mock_embed, mock_get_conn, mock_db, capsys):
        """Query results show derivation chains."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = [0.1] * 1536

        source_id = uuid4()
        belief_id = uuid4()

        # Query result with derivation info
        mock_cur.fetchall.return_value = [
            {
                "id": belief_id,
                "content": "Test belief content that was derived",
                "confidence": {"overall": 0.8},
                "domain_path": ["tech"],
                "created_at": datetime.now(UTC),
                "extraction_method": "inference",
                "supersedes_id": None,
                "similarity": 0.95,
                "derivation_type": "inference",
                "method_description": "Derived via logical deduction",
                "confidence_rationale": "Strong evidence",
                "derivation_sources": [
                    {
                        "source_belief_id": str(source_id),
                        "contribution_type": "primary",
                        "external_ref": None,
                    }
                ],
            }
        ]

        # Source belief lookup
        mock_cur.fetchone.return_value = {"content": "Original observation source"}

        parser = app()
        args = parser.parse_args(["query", "test"])
        result = cmd_query(args)

        assert result == 0

        captured = capsys.readouterr()
        # Verify derivation info is shown
        assert "Derivation: inference" in captured.out
        assert "Method: Derived via logical deduction" in captured.out
        assert "Derived from" in captured.out or "primary" in captured.out

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_query_no_results(self, mock_embed, mock_get_conn, mock_db, capsys):
        """Query with no results."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = None

        mock_cur.fetchall.return_value = []

        parser = app()
        args = parser.parse_args(["query", "nonexistent"])
        result = cmd_query(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No beliefs found" in captured.out


class TestConflictsCommand:
    """Test conflict detection."""

    @patch("valence.cli.commands.conflicts.get_db_connection")
    def test_conflicts_detects_negation(self, mock_get_conn, mock_db, capsys):
        """Detect conflicts with negation asymmetry."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        # Similar beliefs with opposite conclusions
        mock_cur.fetchall.return_value = [
            {
                "id_a": uuid4(),
                "content_a": "Python is good for data science",
                "confidence_a": {"overall": 0.8},
                "created_a": datetime.now(UTC),
                "id_b": uuid4(),
                "content_b": "Python is not good for data science",
                "confidence_b": {"overall": 0.7},
                "created_b": datetime.now(UTC),
                "similarity": 0.92,
            }
        ]

        parser = app()
        args = parser.parse_args(["conflicts"])
        result = cmd_conflicts(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "potential conflict" in captured.out.lower() or "Conflict" in captured.out

    @patch("valence.cli.commands.conflicts.get_db_connection")
    def test_conflicts_no_conflicts(self, mock_get_conn, mock_db, capsys):
        """No conflicts found."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchall.return_value = []

        parser = app()
        args = parser.parse_args(["conflicts"])
        result = cmd_conflicts(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No potential conflicts" in captured.out or "no" in captured.out.lower()

    @patch("valence.cli.commands.conflicts.get_db_connection")
    def test_conflicts_auto_record(self, mock_get_conn, mock_db):
        """Auto-record detected conflicts as tensions."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        conflict_pair = {
            "id_a": uuid4(),
            "content_a": "X is always true",
            "confidence_a": {"overall": 0.8},
            "created_a": datetime.now(UTC),
            "id_b": uuid4(),
            "content_b": "X is never true",
            "confidence_b": {"overall": 0.7},
            "created_b": datetime.now(UTC),
            "similarity": 0.90,
        }

        mock_cur.fetchall.return_value = [conflict_pair]
        mock_cur.fetchone.return_value = {"id": uuid4()}

        parser = app()
        args = parser.parse_args(["conflicts", "--auto-record"])
        result = cmd_conflicts(args)

        assert result == 0
        # Verify tension was inserted
        insert_calls = [c for c in mock_cur.execute.call_args_list if "INSERT INTO tensions" in str(c)]
        assert len(insert_calls) >= 1


class TestListCommand:
    """Test list command."""

    @patch("valence.cli.commands.beliefs.get_db_connection")
    def test_list_basic(self, mock_get_conn, mock_db, capsys):
        """List beliefs."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchall.return_value = [
            {
                "id": uuid4(),
                "content": "First belief",
                "confidence": {"overall": 0.9},
                "domain_path": ["tech"],
                "created_at": datetime.now(UTC),
                "derivation_type": "observation",
            },
            {
                "id": uuid4(),
                "content": "Second belief",
                "confidence": {"overall": 0.7},
                "domain_path": [],
                "created_at": datetime.now(UTC) - timedelta(hours=2),
                "derivation_type": "inference",
            },
        ]

        parser = app()
        args = parser.parse_args(["list"])
        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "First belief" in captured.out
        assert "Second belief" in captured.out


class TestStatsCommand:
    """Test stats command."""

    @patch("valence.cli.commands.stats.get_db_connection")
    def test_stats_basic(self, mock_get_conn, mock_db, capsys):
        """Show stats."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn

        mock_cur.fetchone.side_effect = [
            {"total": 100},
            {"active": 95},
            {"with_emb": 80},
            {"tensions": 3},
            {"count": 5},
            {"derivations": 50},
        ]

        parser = app()
        args = parser.parse_args(["stats"])
        result = cmd_stats(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "100" in captured.out  # total beliefs
        assert "Statistics" in captured.out


# ============================================================================
# Derivation Chain Tests
# ============================================================================


class TestDerivationChains:
    """Test derivation chain visibility."""

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_shows_external_ref(self, mock_embed, mock_get_conn, mock_db, capsys):
        """Show external references in derivation."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = [0.1] * 1536

        mock_cur.fetchall.return_value = [
            {
                "id": uuid4(),
                "content": "Belief from external source",
                "confidence": {"overall": 0.8},
                "domain_path": [],
                "created_at": datetime.now(UTC),
                "extraction_method": "hearsay",
                "supersedes_id": None,
                "similarity": 0.9,
                "derivation_type": "hearsay",
                "method_description": "Reported by trusted source",
                "confidence_rationale": None,
                "derivation_sources": [
                    {
                        "source_belief_id": None,
                        "contribution_type": "primary",
                        "external_ref": "https://example.com/doc",
                    }
                ],
            }
        ]

        parser = app()
        args = parser.parse_args(["query", "external"])
        result = cmd_query(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "External" in captured.out
        assert "example.com" in captured.out

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_shows_supersession_chain(self, mock_embed, mock_get_conn, mock_db, capsys):
        """Show supersession chain when --chain flag used."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = [0.1] * 1536

        old_id = uuid4()

        mock_cur.fetchall.return_value = [
            {
                "id": uuid4(),
                "content": "Updated belief",
                "confidence": {"overall": 0.9},
                "domain_path": [],
                "created_at": datetime.now(UTC),
                "extraction_method": "correction",
                "supersedes_id": old_id,
                "similarity": 0.95,
                "derivation_type": "correction",
                "method_description": "Corrected previous error",
                "confidence_rationale": None,
                "derivation_sources": [],
            }
        ]

        # Chain lookup
        mock_cur.fetchone.side_effect = [
            {
                "id": old_id,
                "content": "Original incorrect belief",
                "supersedes_id": None,
            },
        ]

        parser = app()
        args = parser.parse_args(["query", "updated", "--chain"])
        result = cmd_query(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Supersedes" in captured.out


# ============================================================================
# End-to-End Happy Path Test
# ============================================================================


class TestHappyPath:
    """Test the full happy path: pip install && valence init works."""

    def test_cli_module_imports(self):
        """Verify CLI module can be imported."""
        from valence.cli import app, main

        assert callable(main)
        assert callable(app)

    def test_help_output(self, capsys):
        """Verify help output works."""
        parser = app()

        # This should not raise
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])

        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "valence" in captured.out
        assert "init" in captured.out
        assert "add" in captured.out
        assert "query" in captured.out


# ============================================================================
# Multi-Signal Ranking Tests (Issue #13)
# ============================================================================


class TestComputeConfidenceScore:
    """Test confidence score computation."""

    def test_6d_confidence_geometric_mean(self):
        """Compute confidence from 6D vector using geometric mean."""
        belief = {
            "confidence_source": 0.9,
            "confidence_method": 0.8,
            "confidence_consistency": 0.9,
            "confidence_freshness": 1.0,
            "confidence_corroboration": 0.5,
            "confidence_applicability": 0.8,
        }
        score = compute_confidence_score(belief)
        # Geometric mean with weights: should be around 0.78
        assert 0.7 < score < 0.9

    def test_6d_penalizes_weak_dimension(self):
        """Geometric mean penalizes beliefs with one weak dimension."""
        # All strong except corroboration
        belief_weak = {
            "confidence_source": 0.9,
            "confidence_method": 0.9,
            "confidence_consistency": 0.9,
            "confidence_freshness": 0.9,
            "confidence_corroboration": 0.1,  # Very weak
            "confidence_applicability": 0.9,
        }

        # All moderate
        belief_moderate = {
            "confidence_source": 0.7,
            "confidence_method": 0.7,
            "confidence_consistency": 0.7,
            "confidence_freshness": 0.7,
            "confidence_corroboration": 0.7,
            "confidence_applicability": 0.7,
        }

        score_weak = compute_confidence_score(belief_weak)
        score_moderate = compute_confidence_score(belief_moderate)

        # Moderate should beat weak despite higher average
        assert score_moderate > score_weak

    def test_fallback_to_jsonb_overall(self):
        """Fall back to JSONB overall when 6D not populated."""
        belief = {
            "confidence": {"overall": 0.85},
        }
        score = compute_confidence_score(belief)
        assert score == 0.85

    def test_default_score(self):
        """Return default 0.5 when no confidence data."""
        belief = {}
        score = compute_confidence_score(belief)
        assert score == 0.5


class TestComputeRecencyScore:
    """Test recency score computation."""

    def test_recent_belief_high_score(self):
        """Recent beliefs get high recency score."""
        now = datetime.now(UTC)
        score = compute_recency_score(now)
        assert score > 0.99

    def test_old_belief_decays(self):
        """Old beliefs decay over time."""
        now = datetime.now(UTC)
        one_week_ago = now - timedelta(days=7)
        one_month_ago = now - timedelta(days=30)
        one_year_ago = now - timedelta(days=365)

        score_week = compute_recency_score(one_week_ago)
        score_month = compute_recency_score(one_month_ago)
        score_year = compute_recency_score(one_year_ago)

        # Scores should decrease with age
        assert score_week > score_month > score_year

        # With default decay (0.01), half-life ~69 days
        assert score_week > 0.9  # ~0.93
        assert 0.6 < score_month < 0.8  # ~0.74
        assert score_year < 0.05  # ~0.025

    def test_custom_decay_rate(self):
        """Custom decay rate adjusts half-life."""
        now = datetime.now(UTC)
        one_week_ago = now - timedelta(days=7)

        # High decay (news): half-life ~7 days
        score_high = compute_recency_score(one_week_ago, decay_rate=0.10)

        # Low decay (science): half-life ~346 days
        score_low = compute_recency_score(one_week_ago, decay_rate=0.002)

        assert score_low > score_high
        assert score_high < 0.6  # ~0.50
        assert score_low > 0.98  # ~0.986

    def test_none_returns_default(self):
        """None datetime returns default score."""
        score = compute_recency_score(None)
        assert score == 0.5

    def test_naive_datetime_handled(self):
        """Naive datetime (no timezone) is handled correctly."""
        naive_dt = datetime.now()
        score = compute_recency_score(naive_dt)
        # Should still return a valid score close to 1.0
        assert 0.9 < score <= 1.0


class TestMultiSignalRank:
    """Test multi-signal ranking algorithm."""

    def test_default_weights(self):
        """Default weights: semantic=0.50, confidence=0.35, recency=0.15."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.9,
                "confidence": {"overall": 0.5},
                "created_at": now - timedelta(days=30),
            },
            {
                "id": uuid4(),
                "similarity": 0.7,
                "confidence": {"overall": 0.95},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results)

        # Both should have final_score
        assert all("final_score" in r for r in ranked)

        # High semantic (0.9) vs High confidence+recency (0.95, recent)
        # First belief: 0.50×0.9 + 0.35×0.5 + 0.15×0.74 = 0.45 + 0.175 + 0.11 = 0.735
        # Second belief: 0.50×0.7 + 0.35×0.95 + 0.15×1.0 = 0.35 + 0.333 + 0.15 = 0.833
        # Second should rank higher with default weights
        assert ranked[0]["similarity"] == 0.7  # High confidence wins

    def test_high_recency_weight(self):
        """High recency weight prefers newer beliefs."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.95,
                "confidence": {"overall": 0.9},
                "created_at": now - timedelta(days=60),  # Old but great
            },
            {
                "id": uuid4(),
                "similarity": 0.7,
                "confidence": {"overall": 0.7},
                "created_at": now,  # New
            },
        ]

        # With high recency weight
        ranked = multi_signal_rank(results, recency_weight=0.5)

        # New belief should rank first with high recency weight
        assert ranked[0]["created_at"] == now

    def test_min_confidence_filter(self):
        """Filter beliefs below minimum confidence."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.9,
                "confidence": {"overall": 0.3},
                "created_at": now,
            },
            {
                "id": uuid4(),
                "similarity": 0.8,
                "confidence": {"overall": 0.8},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results, min_confidence=0.5)

        # Only high-confidence belief should remain
        assert len(ranked) == 1
        assert ranked[0]["confidence"]["overall"] == 0.8

    def test_explain_mode(self):
        """Explain mode includes score breakdown."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.85,
                "confidence": {"overall": 0.75},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results, explain=True)

        assert len(ranked) == 1
        assert "score_breakdown" in ranked[0]

        bd = ranked[0]["score_breakdown"]
        assert "semantic" in bd
        assert "confidence" in bd
        assert "recency" in bd
        assert "final" in bd

        # Verify breakdown structure
        assert "value" in bd["semantic"]
        assert "weight" in bd["semantic"]
        assert "contribution" in bd["semantic"]

        # Contributions should sum to final
        total = bd["semantic"]["contribution"] + bd["confidence"]["contribution"] + bd["recency"]["contribution"]
        assert abs(total - bd["final"]) < 0.001

    def test_weight_normalization(self):
        """Weights are normalized to sum to 1.0."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.8,
                "confidence": {"overall": 0.8},
                "created_at": now,
            },
        ]

        # Pass non-normalized weights
        ranked = multi_signal_rank(
            results,
            semantic_weight=1.0,
            confidence_weight=1.0,
            recency_weight=1.0,
            explain=True,
        )

        bd = ranked[0]["score_breakdown"]

        # Each weight should be 1/3 after normalization
        assert abs(bd["semantic"]["weight"] - 0.333) < 0.01
        assert abs(bd["confidence"]["weight"] - 0.333) < 0.01
        assert abs(bd["recency"]["weight"] - 0.333) < 0.01

    def test_empty_results(self):
        """Handle empty results gracefully."""
        ranked = multi_signal_rank([])
        assert ranked == []

    def test_results_sorted_by_final_score(self):
        """Results are sorted by final_score descending."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.5,
                "confidence": {"overall": 0.5},
                "created_at": now,
            },
            {
                "id": uuid4(),
                "similarity": 0.9,
                "confidence": {"overall": 0.9},
                "created_at": now,
            },
            {
                "id": uuid4(),
                "similarity": 0.7,
                "confidence": {"overall": 0.7},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results)

        scores = [r["final_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)


class TestQueryMultiSignalArgs:
    """Test query command with multi-signal ranking arguments."""

    def test_query_recency_weight_arg(self):
        """Parse recency weight argument."""
        parser = app()
        args = parser.parse_args(["query", "test", "--recency-weight", "0.4"])
        assert args.recency_weight == 0.4

    def test_query_min_confidence_arg(self):
        """Parse minimum confidence argument."""
        parser = app()
        args = parser.parse_args(["query", "test", "--min-confidence", "0.7"])
        assert args.min_confidence == 0.7

    def test_query_explain_arg(self):
        """Parse explain flag."""
        parser = app()
        args = parser.parse_args(["query", "test", "--explain"])
        assert args.explain is True

    def test_query_short_flags(self):
        """Parse short flag versions."""
        parser = app()
        args = parser.parse_args(["query", "test", "-r", "0.3", "-c", "0.6", "-e"])
        assert args.recency_weight == 0.3
        assert args.min_confidence == 0.6
        assert args.explain is True

    def test_query_defaults(self):
        """Default values for multi-signal args."""
        parser = app()
        args = parser.parse_args(["query", "test"])
        assert args.recency_weight == 0.15
        assert args.min_confidence is None
        assert args.explain is False

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_query_with_explain_output(self, mock_embed, mock_get_conn, mock_db, capsys):
        """Query with --explain shows score breakdown."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = [0.1] * 1536

        now = datetime.now(UTC)

        mock_cur.fetchall.return_value = [
            {
                "id": uuid4(),
                "content": "Test belief content",
                "confidence": {"overall": 0.85},
                "domain_path": [],
                "created_at": now,
                "extraction_method": "observation",
                "supersedes_id": None,
                "confidence_source": None,
                "confidence_method": None,
                "confidence_consistency": None,
                "confidence_freshness": None,
                "confidence_corroboration": None,
                "confidence_applicability": None,
                "similarity": 0.9,
                "derivation_type": "observation",
                "method_description": None,
                "confidence_rationale": None,
                "derivation_sources": None,
            }
        ]

        parser = app()
        args = parser.parse_args(["query", "test", "--explain"])
        result = cmd_query(args)

        assert result == 0
        captured = capsys.readouterr()

        # Should show score breakdown
        assert "Score Breakdown" in captured.out
        assert "Semantic:" in captured.out
        assert "Confidence:" in captured.out
        assert "Recency:" in captured.out

    @patch("valence.cli.commands.beliefs.get_db_connection")
    @patch("valence.cli.commands.beliefs.get_embedding")
    def test_query_min_confidence_filters(self, mock_embed, mock_get_conn, mock_db, capsys):
        """Query with --min-confidence filters low confidence beliefs."""
        mock_conn, mock_cur = mock_db
        mock_get_conn.return_value = mock_conn
        mock_embed.return_value = [0.1] * 1536

        now = datetime.now(UTC)

        # Return two beliefs, one below threshold
        mock_cur.fetchall.return_value = [
            {
                "id": uuid4(),
                "content": "Low confidence belief",
                "confidence": {"overall": 0.3},
                "domain_path": [],
                "created_at": now,
                "extraction_method": "hearsay",
                "supersedes_id": None,
                "confidence_source": None,
                "confidence_method": None,
                "confidence_consistency": None,
                "confidence_freshness": None,
                "confidence_corroboration": None,
                "confidence_applicability": None,
                "similarity": 0.95,
                "derivation_type": "hearsay",
                "method_description": None,
                "confidence_rationale": None,
                "derivation_sources": None,
            },
            {
                "id": uuid4(),
                "content": "High confidence belief",
                "confidence": {"overall": 0.9},
                "domain_path": [],
                "created_at": now,
                "extraction_method": "observation",
                "supersedes_id": None,
                "confidence_source": None,
                "confidence_method": None,
                "confidence_consistency": None,
                "confidence_freshness": None,
                "confidence_corroboration": None,
                "confidence_applicability": None,
                "similarity": 0.8,
                "derivation_type": "observation",
                "method_description": None,
                "confidence_rationale": None,
                "derivation_sources": None,
            },
        ]

        parser = app()
        args = parser.parse_args(["query", "test", "--min-confidence", "0.5"])
        result = cmd_query(args)

        assert result == 0
        captured = capsys.readouterr()

        # Should only show high confidence belief
        assert "High confidence belief" in captured.out
        assert "Low confidence belief" not in captured.out
        assert "Found 1 belief" in captured.out
