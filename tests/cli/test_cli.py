"""Tests for Valence CLI.

Tests cover:
1. CLI argument parsing
2. Command dispatch
3. Pure utility functions (format_confidence, format_age, ranking)
4. REST client commands (mocked HTTP)
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
        now = datetime.now()
        result = format_age(now - timedelta(hours=2))
        assert result != "?"
        assert any(c in result for c in ["h", "m", "d", "y", "now", "mo"])


class TestArgumentParser:
    """Test CLI argument parsing."""

    def test_init_command(self):
        """Parse init command."""
        parser = app()
        args = parser.parse_args(["init"])
        assert args.command == "init"

    def test_add_command(self):
        """Parse add command."""
        parser = app()
        args = parser.parse_args(["add", "Test belief content"])
        assert args.command == "add"
        assert args.content == "Test belief content"

    def test_add_with_options(self):
        """Parse add with options."""
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
            ]
        )
        assert args.content == "Test belief"
        assert args.confidence == "0.9"
        assert args.domain == ["tech", "python"]
        assert args.derivation_type == "inference"

    def test_query_command(self):
        """Parse query command."""
        parser = app()
        args = parser.parse_args(["query", "search terms"])
        assert args.command == "query"
        assert args.query == "search terms"
        assert args.limit == 10  # default

    def test_query_with_options(self):
        """Parse query with options."""
        parser = app()
        args = parser.parse_args(
            [
                "query",
                "search terms",
                "--limit",
                "20",
                "--domain",
                "tech",
                "--explain",
            ]
        )
        assert args.limit == 20
        assert args.domain == "tech"
        assert args.explain is True

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

    def test_global_flags(self):
        """Parse global flags."""
        parser = app()
        args = parser.parse_args(["--server", "http://example.com:8420", "--token", "vt_xxx", "--json", "stats"])
        assert args.server == "http://example.com:8420"
        assert args.token == "vt_xxx"
        assert args.output == "json"


# ============================================================================
# REST Client Command Tests
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset CLI config singleton for each test."""
    from valence.cli.config import reset_cli_config

    reset_cli_config()
    yield
    reset_cli_config()


class TestInitCommand:
    """Test init command (now delegates to server)."""

    @patch("valence.cli.commands.beliefs.get_client")
    def test_init_happy_path(self, mock_get_client):
        """Init calls POST /admin/migrate/up."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "applied": ["001_init"], "dry_run": False}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["init"])
        result = cmd_init(args)

        assert result == 0
        mock_client.post.assert_called_once_with("/admin/migrate/up", body={"dry_run": False})

    @patch("valence.cli.commands.beliefs.get_client")
    def test_init_connection_error(self, mock_get_client):
        """Init handles connection error."""
        from valence.cli.http_client import ValenceConnectionError

        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceConnectionError("http://127.0.0.1:8420")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["init"])
        result = cmd_init(args)

        assert result == 1


class TestAddCommand:
    """Test add command (now delegates to server)."""

    @patch("valence.cli.commands.beliefs.get_client")
    def test_add_basic(self, mock_get_client):
        """Add basic belief via REST."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "belief": {"id": "abc"}}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["add", "Test belief content"])
        result = cmd_add(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["content"] == "Test belief content"

    @patch("valence.cli.commands.beliefs.get_client")
    def test_add_with_options(self, mock_get_client):
        """Add belief with confidence, domains, derivation type."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "belief": {"id": "abc"}}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["add", "Test", "-c", "0.9", "-d", "tech", "-t", "inference"])
        result = cmd_add(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["confidence"] == {"overall": 0.9}
        assert call_body["domain_path"] == ["tech"]
        assert call_body["source_type"] == "inference"

    @patch("valence.cli.commands.beliefs.get_client")
    def test_add_api_error(self, mock_get_client):
        """Add handles API error."""
        from valence.cli.http_client import ValenceAPIError

        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceAPIError(400, "VALIDATION", "Content too short")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["add", "x"])
        result = cmd_add(args)

        assert result == 1


class TestQueryCommand:
    """Test query command (now delegates to server)."""

    @patch("valence.cli.commands.beliefs.get_client")
    def test_query_happy_path(self, mock_get_client):
        """Query calls GET /beliefs with params."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "beliefs": []}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["query", "test"])
        result = cmd_query(args)

        assert result == 0
        call_params = mock_client.get.call_args[1]["params"]
        assert call_params["query"] == "test"
        assert call_params["limit"] == "10"

    @patch("valence.cli.commands.beliefs.get_client")
    def test_query_with_domain(self, mock_get_client):
        """Query with domain filter."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "beliefs": []}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["query", "test", "-d", "tech"])
        result = cmd_query(args)

        assert result == 0
        call_params = mock_client.get.call_args[1]["params"]
        assert call_params["domain_filter"] == "tech"

    @patch("valence.cli.commands.beliefs.get_client")
    def test_query_with_ranking(self, mock_get_client):
        """Query with ranking params."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "beliefs": []}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["query", "test", "-r", "0.3", "-c", "0.7", "-e"])
        result = cmd_query(args)

        assert result == 0
        call_params = mock_client.get.call_args[1]["params"]
        assert "ranking" in call_params


class TestConflictsCommand:
    """Test conflict detection (now delegates to server)."""

    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_happy_path(self, mock_get_client):
        """Conflicts calls GET /beliefs/conflicts."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "conflicts": [], "count": 0}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["conflicts"])
        result = cmd_conflicts(args)

        assert result == 0

    @patch("valence.cli.commands.conflicts.get_client")
    def test_conflicts_with_options(self, mock_get_client):
        """Conflicts passes threshold and auto_record."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "conflicts": [], "count": 0}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["conflicts", "-t", "0.9", "--auto-record"])
        result = cmd_conflicts(args)

        assert result == 0
        call_params = mock_client.get.call_args[1]["params"]
        assert call_params["threshold"] == "0.9"
        assert call_params["auto_record"] == "true"


class TestListCommand:
    """Test list command (now delegates to server)."""

    @patch("valence.cli.commands.beliefs.get_client")
    def test_list_basic(self, mock_get_client):
        """List calls GET /beliefs with query=*."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "beliefs": []}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["list"])
        result = cmd_list(args)

        assert result == 0
        call_params = mock_client.get.call_args[1]["params"]
        assert call_params["query"] == "*"
        assert call_params["limit"] == "10"

    @patch("valence.cli.commands.beliefs.get_client")
    def test_list_with_domain(self, mock_get_client):
        """List with domain filter."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "beliefs": []}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["list", "-d", "tech", "-n", "20"])
        result = cmd_list(args)

        assert result == 0
        call_params = mock_client.get.call_args[1]["params"]
        assert call_params["domain_filter"] == "tech"
        assert call_params["limit"] == "20"


class TestStatsCommand:
    """Test stats command (now delegates to server)."""

    @patch("valence.cli.commands.stats.get_client")
    def test_stats_happy_path(self, mock_get_client):
        """Stats calls GET /stats."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "stats": {"total_beliefs": 100}}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["stats"])
        result = cmd_stats(args)

        assert result == 0
        mock_client.get.assert_called_once()
        assert "/stats" in mock_client.get.call_args[0][0]

    @patch("valence.cli.commands.stats.get_client")
    def test_stats_connection_error(self, mock_get_client):
        """Stats handles connection error."""
        from valence.cli.http_client import ValenceConnectionError

        mock_client = MagicMock()
        mock_client.get.side_effect = ValenceConnectionError("http://127.0.0.1:8420")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["stats"])
        result = cmd_stats(args)

        assert result == 1


# ============================================================================
# End-to-End Happy Path Test
# ============================================================================


class TestHappyPath:
    """Test the full happy path."""

    def test_cli_module_imports(self):
        """Verify CLI module can be imported."""
        from valence.cli import app, main

        assert callable(main)
        assert callable(app)

    def test_help_output(self, capsys):
        """Verify help output works."""
        parser = app()

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
        assert 0.7 < score < 0.9

    def test_6d_penalizes_weak_dimension(self):
        """Geometric mean penalizes beliefs with one weak dimension."""
        belief_weak = {
            "confidence_source": 0.9,
            "confidence_method": 0.9,
            "confidence_consistency": 0.9,
            "confidence_freshness": 0.9,
            "confidence_corroboration": 0.1,
            "confidence_applicability": 0.9,
        }
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
        assert score_moderate > score_weak

    def test_fallback_to_jsonb_overall(self):
        """Fall back to JSONB overall when 6D not populated."""
        belief = {"confidence": {"overall": 0.85}}
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

        assert score_week > score_month > score_year
        assert score_week > 0.9
        assert 0.6 < score_month < 0.8
        assert score_year < 0.05

    def test_custom_decay_rate(self):
        """Custom decay rate adjusts half-life."""
        now = datetime.now(UTC)
        one_week_ago = now - timedelta(days=7)

        score_high = compute_recency_score(one_week_ago, decay_rate=0.10)
        score_low = compute_recency_score(one_week_ago, decay_rate=0.002)

        assert score_low > score_high
        assert score_high < 0.6
        assert score_low > 0.98

    def test_none_returns_default(self):
        """None datetime returns default score."""
        score = compute_recency_score(None)
        assert score == 0.5

    def test_naive_datetime_handled(self):
        """Naive datetime (no timezone) is handled correctly."""
        naive_dt = datetime.now()
        score = compute_recency_score(naive_dt)
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

        assert all("final_score" in r for r in ranked)
        assert ranked[0]["similarity"] == 0.7

    def test_high_recency_weight(self):
        """High recency weight prefers newer beliefs."""
        now = datetime.now(UTC)

        results = [
            {
                "id": uuid4(),
                "similarity": 0.95,
                "confidence": {"overall": 0.9},
                "created_at": now - timedelta(days=60),
            },
            {
                "id": uuid4(),
                "similarity": 0.7,
                "confidence": {"overall": 0.7},
                "created_at": now,
            },
        ]

        ranked = multi_signal_rank(results, recency_weight=0.5)
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

        assert "value" in bd["semantic"]
        assert "weight" in bd["semantic"]
        assert "contribution" in bd["semantic"]

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

        ranked = multi_signal_rank(
            results,
            semantic_weight=1.0,
            confidence_weight=1.0,
            recency_weight=1.0,
            explain=True,
        )

        bd = ranked[0]["score_breakdown"]
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
            {"id": uuid4(), "similarity": 0.5, "confidence": {"overall": 0.5}, "created_at": now},
            {"id": uuid4(), "similarity": 0.9, "confidence": {"overall": 0.9}, "created_at": now},
            {"id": uuid4(), "similarity": 0.7, "confidence": {"overall": 0.7}, "created_at": now},
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
        assert args.recency_weight is None
        assert args.min_confidence is None
        assert args.explain is False
