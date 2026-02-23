"""Tests for exchange compaction (#359).

Tests cover:
1. CompactionConfig defaults
2. compact_exchanges finds candidates via SQL
3. compact_exchanges dry_run reports without modifying
4. compact_exchanges skips sessions with too few exchanges
5. compact_exchanges keeps first+last N, deletes middle
6. compact_exchanges stores summary on session
7. _build_compaction_summary extracts roles, tokens, tool_uses
8. _build_compaction_summary handles string tool_uses (JSON)
9. _build_compaction_summary handles empty exchanges
10. CLI --compact flag dispatches correctly
"""

from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, call, patch

import pytest

from valence.core.maintenance import (
    CompactionConfig,
    MaintenanceResult,
    _build_compaction_summary,
    compact_exchanges,
)


# ============================================================================
# Config Tests
# ============================================================================


class TestCompactionConfig:
    """Test compaction configuration defaults."""

    def test_defaults(self):
        config = CompactionConfig()
        assert config.keep_first == 5
        assert config.keep_last == 5
        assert config.min_exchanges == 15

    def test_custom(self):
        config = CompactionConfig(keep_first=3, keep_last=3, min_exchanges=10)
        assert config.keep_first == 3
        assert config.keep_last == 3
        assert config.min_exchanges == 10


# ============================================================================
# _build_compaction_summary Tests
# ============================================================================


class TestBuildCompactionSummary:
    """Test summary building from middle exchanges."""

    def test_counts_exchanges(self):
        exchanges = [
            {"role": "user", "tokens_approx": 10, "tool_uses": None},
            {"role": "assistant", "tokens_approx": 20, "tool_uses": None},
            {"role": "user", "tokens_approx": 15, "tool_uses": None},
        ]
        summary = _build_compaction_summary(exchanges)
        assert summary["exchange_count"] == 3

    def test_counts_roles(self):
        exchanges = [
            {"role": "user", "tokens_approx": 10, "tool_uses": None},
            {"role": "assistant", "tokens_approx": 20, "tool_uses": None},
            {"role": "user", "tokens_approx": 15, "tool_uses": None},
            {"role": "assistant", "tokens_approx": 25, "tool_uses": None},
        ]
        summary = _build_compaction_summary(exchanges)
        assert summary["roles"] == {"user": 2, "assistant": 2}

    def test_sums_tokens(self):
        exchanges = [
            {"role": "user", "tokens_approx": 100, "tool_uses": None},
            {"role": "assistant", "tokens_approx": 200, "tool_uses": None},
        ]
        summary = _build_compaction_summary(exchanges)
        assert summary["total_tokens"] == 300

    def test_handles_none_tokens(self):
        exchanges = [
            {"role": "user", "tokens_approx": None, "tool_uses": None},
            {"role": "assistant", "tokens_approx": 50, "tool_uses": None},
        ]
        summary = _build_compaction_summary(exchanges)
        assert summary["total_tokens"] == 50

    def test_counts_tool_uses_from_dicts(self):
        exchanges = [
            {"role": "assistant", "tokens_approx": 10, "tool_uses": [{"name": "belief_query"}, {"name": "belief_create"}]},
            {"role": "assistant", "tokens_approx": 10, "tool_uses": [{"name": "belief_query"}]},
        ]
        summary = _build_compaction_summary(exchanges)
        assert summary["tool_uses"]["belief_query"] == 2
        assert summary["tool_uses"]["belief_create"] == 1

    def test_counts_tool_uses_from_strings(self):
        exchanges = [
            {"role": "assistant", "tokens_approx": 10, "tool_uses": ["belief_query", "belief_create"]},
        ]
        summary = _build_compaction_summary(exchanges)
        assert summary["tool_uses"]["belief_query"] == 1

    def test_counts_tool_uses_from_json_string(self):
        exchanges = [
            {"role": "assistant", "tokens_approx": 10, "tool_uses": json.dumps([{"name": "entity_search"}])},
        ]
        summary = _build_compaction_summary(exchanges)
        assert summary["tool_uses"]["entity_search"] == 1

    def test_empty_exchanges(self):
        summary = _build_compaction_summary([])
        assert summary["exchange_count"] == 0
        assert summary["roles"] == {}
        assert summary["total_tokens"] == 0
        assert summary["tool_uses"] == {}


# ============================================================================
# compact_exchanges Tests
# ============================================================================


class TestCompactExchanges:
    """Test exchange compaction."""

    def test_no_candidates_returns_zero(self):
        cur = MagicMock()
        cur.fetchall.return_value = []  # No sessions to compact

        result = compact_exchanges(cur)

        assert result.operation == "exchange_compaction"
        assert result.details["sessions_compacted"] == 0
        assert result.details["exchanges_removed"] == 0

    def test_dry_run_reports_without_modifying(self):
        cur = MagicMock()
        # One session with 20 exchanges
        cur.fetchall.return_value = [{"id": "sess-1", "exchange_count": 20}]

        result = compact_exchanges(cur, dry_run=True)

        assert result.dry_run is True
        assert result.details["sessions_compacted"] == 1
        # 20 - 5 (first) - 5 (last) = 10 middle exchanges
        assert result.details["exchanges_removed"] == 10
        # Should NOT have queried individual exchanges
        assert cur.execute.call_count == 1  # Only the candidate query

    def test_dry_run_multiple_sessions(self):
        cur = MagicMock()
        cur.fetchall.return_value = [
            {"id": "sess-1", "exchange_count": 20},
            {"id": "sess-2", "exchange_count": 30},
        ]

        result = compact_exchanges(cur, dry_run=True)

        assert result.details["sessions_compacted"] == 2
        assert result.details["exchanges_removed"] == 10 + 20  # (20-10) + (30-10)

    def test_compacts_session(self):
        cur = MagicMock()

        # First call: find candidates
        candidates = [{"id": "sess-1", "exchange_count": 20}]

        # Second call: get all exchanges for sess-1 (20 exchanges)
        exchanges = []
        for i in range(20):
            exchanges.append(
                {
                    "id": f"ex-{i}",
                    "sequence": i,
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"message {i}",
                    "tokens_approx": 10,
                    "tool_uses": None,
                }
            )

        cur.fetchall.side_effect = [candidates, exchanges]

        result = compact_exchanges(cur)

        assert result.details["sessions_compacted"] == 1
        assert result.details["exchanges_removed"] == 10  # 20 - 5 - 5

        # Verify summary was stored on session
        update_call = None
        delete_call = None
        for c in cur.execute.call_args_list:
            sql = c[0][0]
            if "UPDATE vkb_sessions" in sql:
                update_call = c
            elif "DELETE FROM vkb_exchanges" in sql:
                delete_call = c

        assert update_call is not None, "Should have updated session with summary"
        assert delete_call is not None, "Should have deleted middle exchanges"

        # Check summary JSON was stored
        summary_json = update_call[0][1][0]
        summary = json.loads(summary_json)
        assert summary["exchange_count"] == 10
        assert summary["total_tokens"] == 100  # 10 exchanges * 10 tokens

        # Check correct middle exchange IDs were deleted
        deleted_ids = delete_call[0][1][0]
        assert len(deleted_ids) == 10
        assert deleted_ids[0] == "ex-5"  # First middle exchange
        assert deleted_ids[-1] == "ex-14"  # Last middle exchange

    def test_skips_session_with_too_few_after_fetch(self):
        """If actual exchange count <= keep_first + keep_last, skip."""
        cur = MagicMock()
        config = CompactionConfig(keep_first=5, keep_last=5, min_exchanges=8)

        # SQL says 9 exchanges, but after fetch there are exactly 10 (= keep_first + keep_last)
        candidates = [{"id": "sess-1", "exchange_count": 9}]
        exchanges = [{"id": f"ex-{i}", "sequence": i, "role": "user", "content": f"m{i}", "tokens_approx": 5, "tool_uses": None} for i in range(10)]
        cur.fetchall.side_effect = [candidates, exchanges]

        result = compact_exchanges(cur, config)

        # Should have fetched but not compacted
        assert result.details["sessions_compacted"] == 0
        assert result.details["exchanges_removed"] == 0

    def test_custom_config(self):
        cur = MagicMock()
        config = CompactionConfig(keep_first=2, keep_last=2, min_exchanges=5)

        # 8 exchanges, keep 2+2, compact 4
        candidates = [{"id": "sess-1", "exchange_count": 8}]
        exchanges = [{"id": f"ex-{i}", "sequence": i, "role": "user", "content": f"m{i}", "tokens_approx": 5, "tool_uses": None} for i in range(8)]
        cur.fetchall.side_effect = [candidates, exchanges]

        result = compact_exchanges(cur, config)

        assert result.details["sessions_compacted"] == 1
        assert result.details["exchanges_removed"] == 4  # 8 - 2 - 2

    def test_result_type(self):
        cur = MagicMock()
        cur.fetchall.return_value = []

        result = compact_exchanges(cur)

        assert isinstance(result, MaintenanceResult)
        assert result.operation == "exchange_compaction"


# ============================================================================
# CLI --compact flag
# ============================================================================


class TestCmdMaintenanceCompact:
    """Test --compact CLI flag."""

    @patch("valence.cli.commands.maintenance.get_client")
    def test_compact_flag(self, mock_get_client):
        from valence.cli.commands.maintenance import cmd_maintenance

        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "results": [], "count": 0, "dry_run": False}
        mock_get_client.return_value = mock_client

        args = argparse.Namespace(
            run_all=False,
            retention=False,
            archive=False,
            tombstones=False,
            compact=True,
            views=False,
            vacuum=False,
            dry_run=False,
        )
        result = cmd_maintenance(args)
        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["compact"] is True
