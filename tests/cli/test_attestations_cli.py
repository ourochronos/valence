"""Tests for attestations CLI commands.

Tests cover:
- Parser registration
- Command dispatch
- List, add, stats, trust subcommands
- JSON output
- Error handling (invalid UUIDs, nonexistent resources)

Part of Issue #271: Social — Usage attestations.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from uuid import uuid4

from valence.cli.commands.attestations import (
    cmd_attestations,
    cmd_attestations_add,
    cmd_attestations_list,
    cmd_attestations_stats,
    cmd_attestations_trust,
)
from valence.cli.main import app

# =============================================================================
# PARSER REGISTRATION TESTS
# =============================================================================


class TestParserRegistration:
    """Tests for attestations command registration."""

    def test_attestations_command_registered(self):
        """Attestations command is registered in the main parser."""
        parser = app()
        # Should parse without error
        args = parser.parse_args(["attestations", "list"])
        assert args.command == "attestations"
        assert args.attestations_command == "list"

    def test_attestations_add_args(self):
        """Add subcommand accepts expected arguments."""
        parser = app()
        rid = str(uuid4())
        args = parser.parse_args(
            [
                "attestations",
                "add",
                rid,
                "--user",
                "did:user1",
                "--failure",
                "--feedback",
                "Didn't work",
                "--json",
            ]
        )
        assert args.attestations_command == "add"
        assert args.resource_id == rid
        assert args.user == "did:user1"
        assert args.failure is True
        assert args.feedback == "Didn't work"
        assert args.json is True

    def test_attestations_list_args(self):
        """List subcommand accepts expected arguments."""
        parser = app()
        rid = str(uuid4())
        args = parser.parse_args(
            [
                "attestations",
                "list",
                "--resource",
                rid,
                "--user",
                "did:alice",
                "--success-only",
                "--limit",
                "10",
                "--json",
            ]
        )
        assert args.attestations_command == "list"
        assert args.resource == rid
        assert args.user == "did:alice"
        assert args.success_only is True
        assert args.limit == 10

    def test_attestations_stats_args(self):
        """Stats subcommand accepts expected arguments."""
        parser = app()
        rid = str(uuid4())
        args = parser.parse_args(["attestations", "stats", rid, "--json"])
        assert args.attestations_command == "stats"
        assert args.resource_id == rid

    def test_attestations_stats_no_resource(self):
        """Stats subcommand without resource ID (all resources)."""
        parser = app()
        args = parser.parse_args(["attestations", "stats"])
        assert args.attestations_command == "stats"
        assert args.resource_id is None

    def test_attestations_trust_args(self):
        """Trust subcommand accepts expected arguments."""
        parser = app()
        rid = str(uuid4())
        args = parser.parse_args(["attestations", "trust", rid, "--json"])
        assert args.attestations_command == "trust"
        assert args.resource_id == rid


# =============================================================================
# COMMAND DISPATCH TESTS
# =============================================================================


class TestCommandDispatch:
    """Tests for the attestations command dispatcher."""

    def test_dispatch_list(self):
        """Dispatch to list subcommand."""
        args = MagicMock()
        args.attestations_command = "list"
        args.resource = None
        args.user = None
        args.success_only = False
        args.failure_only = False
        args.limit = 50
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_svc.get_attestations.return_value = []
            mock_build.return_value = mock_svc
            result = cmd_attestations(args)
            assert result == 0

    def test_dispatch_add(self):
        """Dispatch to add subcommand."""
        rid = uuid4()
        args = MagicMock()
        args.attestations_command = "add"
        args.resource_id = str(rid)
        args.user = "did:vkb:local"
        args.failure = False
        args.feedback = None
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_att = MagicMock()
            mock_att.id = uuid4()
            mock_att.resource_id = rid
            mock_att.user_did = "did:vkb:local"
            mock_att.success = True
            mock_att.feedback = None
            mock_svc.add_attestation.return_value = mock_att
            mock_build.return_value = mock_svc
            result = cmd_attestations(args)
            assert result == 0

    def test_dispatch_unknown(self):
        """Unknown subcommand returns error."""
        args = MagicMock()
        args.attestations_command = "nonexistent"
        result = cmd_attestations(args)
        assert result == 1

    def test_dispatch_none(self):
        """No subcommand returns error."""
        args = MagicMock()
        args.attestations_command = None
        result = cmd_attestations(args)
        assert result == 1


# =============================================================================
# LIST COMMAND TESTS
# =============================================================================


class TestListCommand:
    """Tests for the attestations list command."""

    def test_list_empty(self, capsys):
        """List with no attestations shows empty message."""
        args = MagicMock()
        args.resource = None
        args.user = None
        args.success_only = False
        args.failure_only = False
        args.limit = 50
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_svc.get_attestations.return_value = []
            mock_build.return_value = mock_svc
            result = cmd_attestations_list(args)

        assert result == 0
        assert "No attestations" in capsys.readouterr().out

    def test_list_json_output(self, capsys):
        """List with --json outputs valid JSON."""
        args = MagicMock()
        args.resource = None
        args.user = None
        args.success_only = False
        args.failure_only = False
        args.limit = 50
        args.json = True

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_att = MagicMock()
            mock_att.to_dict.return_value = {
                "id": str(uuid4()),
                "resource_id": str(uuid4()),
                "user_did": "did:user1",
                "success": True,
                "feedback": None,
                "created_at": "2026-01-01T00:00:00",
            }
            mock_svc.get_attestations.return_value = [mock_att]
            mock_build.return_value = mock_svc
            result = cmd_attestations_list(args)

        assert result == 0
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert len(parsed) == 1

    def test_list_invalid_uuid(self, capsys):
        """Invalid resource UUID shows error."""
        args = MagicMock()
        args.resource = "not-a-uuid"
        args.user = None
        args.success_only = False
        args.failure_only = False
        args.limit = 50
        args.json = False

        result = cmd_attestations_list(args)
        assert result == 1
        assert "Invalid UUID" in capsys.readouterr().out


# =============================================================================
# ADD COMMAND TESTS
# =============================================================================


class TestAddCommand:
    """Tests for the attestations add command."""

    def test_add_success(self, capsys):
        """Successfully add an attestation."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.user = "did:user1"
        args.failure = False
        args.feedback = "Great!"
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_att = MagicMock()
            mock_att.id = uuid4()
            mock_att.resource_id = rid
            mock_att.user_did = "did:user1"
            mock_att.success = True
            mock_att.feedback = "Great!"
            mock_svc.add_attestation.return_value = mock_att
            mock_build.return_value = mock_svc
            result = cmd_attestations_add(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "Attestation recorded" in output
        assert "Great!" in output

    def test_add_failure_attestation(self, capsys):
        """Add a failed attestation."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.user = "did:user1"
        args.failure = True
        args.feedback = None
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_att = MagicMock()
            mock_att.id = uuid4()
            mock_att.resource_id = rid
            mock_att.user_did = "did:user1"
            mock_att.success = False
            mock_att.feedback = None
            mock_svc.add_attestation.return_value = mock_att
            mock_build.return_value = mock_svc
            result = cmd_attestations_add(args)

        assert result == 0

    def test_add_invalid_uuid(self, capsys):
        """Invalid resource UUID shows error."""
        args = MagicMock()
        args.resource_id = "bad-uuid"
        result = cmd_attestations_add(args)
        assert result == 1
        assert "Invalid UUID" in capsys.readouterr().out

    def test_add_json_output(self, capsys):
        """Add with --json outputs valid JSON."""
        rid = uuid4()
        att_id = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.user = "did:user1"
        args.failure = False
        args.feedback = None
        args.json = True

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_att = MagicMock()
            mock_att.to_dict.return_value = {
                "id": str(att_id),
                "resource_id": str(rid),
                "user_did": "did:user1",
                "success": True,
                "feedback": None,
                "created_at": "2026-01-01T00:00:00",
            }
            mock_svc.add_attestation.return_value = mock_att
            mock_build.return_value = mock_svc
            result = cmd_attestations_add(args)

        assert result == 0
        output = capsys.readouterr().out
        parsed = json.loads(output)
        assert parsed["success"] is True

    def test_add_service_error(self, capsys):
        """Service error shows error message."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.user = "did:user1"
        args.failure = False
        args.feedback = None
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            mock_svc = MagicMock()
            mock_svc.add_attestation.side_effect = Exception("Resource not found")
            mock_build.return_value = mock_svc
            result = cmd_attestations_add(args)

        assert result == 1
        assert "Resource not found" in capsys.readouterr().out


# =============================================================================
# STATS COMMAND TESTS
# =============================================================================


class TestStatsCommand:
    """Tests for the attestations stats command."""

    def test_stats_single_resource(self, capsys):
        """Show stats for a single resource."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            from valence.core.attestation_service import AttestationStats

            mock_svc = MagicMock()
            mock_svc.get_stats.return_value = AttestationStats(
                resource_id=rid,
                total=10,
                successes=8,
                failures=2,
                success_rate=0.8,
                unique_users=5,
            )
            mock_build.return_value = mock_svc
            result = cmd_attestations_stats(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "10" in output
        assert "80%" in output

    def test_stats_empty(self, capsys):
        """No attestations for resource shows empty message."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            from valence.core.attestation_service import AttestationStats

            mock_svc = MagicMock()
            mock_svc.get_stats.return_value = AttestationStats(resource_id=rid)
            mock_build.return_value = mock_svc
            result = cmd_attestations_stats(args)

        assert result == 0
        assert "No attestations" in capsys.readouterr().out

    def test_stats_all_resources(self, capsys):
        """Show stats for all resources."""
        args = MagicMock()
        args.resource_id = None
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            from valence.core.attestation_service import AttestationStats

            mock_svc = MagicMock()
            mock_svc.get_all_stats.return_value = [
                AttestationStats(resource_id=uuid4(), total=5, success_rate=1.0, unique_users=3),
                AttestationStats(resource_id=uuid4(), total=2, success_rate=0.5, unique_users=2),
            ]
            mock_build.return_value = mock_svc
            result = cmd_attestations_stats(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "2 resources" in output

    def test_stats_json_output(self, capsys):
        """Stats with --json outputs valid JSON."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.json = True

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            from valence.core.attestation_service import AttestationStats

            mock_svc = MagicMock()
            mock_svc.get_stats.return_value = AttestationStats(
                resource_id=rid,
                total=3,
                successes=2,
                failures=1,
                success_rate=0.667,
            )
            mock_build.return_value = mock_svc
            result = cmd_attestations_stats(args)

        assert result == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["total"] == 3

    def test_stats_invalid_uuid(self, capsys):
        """Invalid UUID shows error."""
        args = MagicMock()
        args.resource_id = "bad"
        args.json = False

        result = cmd_attestations_stats(args)
        assert result == 1
        assert "Invalid UUID" in capsys.readouterr().out


# =============================================================================
# TRUST COMMAND TESTS
# =============================================================================


class TestTrustCommand:
    """Tests for the attestations trust command."""

    def test_trust_signal(self, capsys):
        """Show trust signal for a resource."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            from valence.core.attestation_service import TrustSignal

            mock_svc = MagicMock()
            mock_svc.compute_trust_signal.return_value = TrustSignal(
                resource_id=rid,
                success_rate=0.9,
                diversity_score=0.6,
                volume_score=0.5,
                overall=0.72,
            )
            mock_build.return_value = mock_svc
            result = cmd_attestations_trust(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "Trust Signal" in output
        assert "90.0%" in output

    def test_trust_no_attestations(self, capsys):
        """No attestations shows empty message."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.json = False

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            from valence.core.attestation_service import TrustSignal

            mock_svc = MagicMock()
            mock_svc.compute_trust_signal.return_value = TrustSignal(resource_id=rid)
            mock_build.return_value = mock_svc
            result = cmd_attestations_trust(args)

        assert result == 0
        assert "No attestations" in capsys.readouterr().out

    def test_trust_json_output(self, capsys):
        """Trust with --json outputs valid JSON."""
        rid = uuid4()
        args = MagicMock()
        args.resource_id = str(rid)
        args.json = True

        with patch("valence.cli.commands.attestations._build_service") as mock_build:
            from valence.core.attestation_service import TrustSignal

            mock_svc = MagicMock()
            mock_svc.compute_trust_signal.return_value = TrustSignal(
                resource_id=rid,
                success_rate=0.8,
                diversity_score=0.4,
                volume_score=0.3,
                overall=0.56,
            )
            mock_build.return_value = mock_svc
            result = cmd_attestations_trust(args)

        assert result == 0
        parsed = json.loads(capsys.readouterr().out)
        assert parsed["overall"] == 0.56

    def test_trust_invalid_uuid(self, capsys):
        """Invalid UUID shows error."""
        args = MagicMock()
        args.resource_id = "not-uuid"
        args.json = False

        result = cmd_attestations_trust(args)
        assert result == 1
        assert "Invalid UUID" in capsys.readouterr().out
