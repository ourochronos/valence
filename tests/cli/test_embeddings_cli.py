"""Tests for the embeddings CLI commands.

Tests cover:
1. Argument parsing for `valence embeddings backfill`
2. REST client calls to server
3. Error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valence.cli.main import app, cmd_embeddings


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset CLI config singleton for each test."""
    from valence.cli.config import reset_cli_config

    reset_cli_config()
    yield
    reset_cli_config()


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
        assert args.batch_size == 50
        assert args.dry_run is False

    def test_embeddings_backfill_batch_size(self):
        """Parse --batch-size flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--batch-size", "100"])
        assert args.batch_size == 100

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

    def test_embeddings_requires_subcommand(self):
        """embeddings without subcommand raises error."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["embeddings"])


class TestMigrateArgParsing:
    """Test CLI argument parsing for embeddings migrate command."""

    def test_migrate_requires_model(self):
        """embeddings migrate requires --model."""
        parser = app()
        with pytest.raises(SystemExit):
            parser.parse_args(["embeddings", "migrate"])

    def test_migrate_known_model(self):
        """Parse known model."""
        parser = app()
        args = parser.parse_args(["embeddings", "migrate", "--model", "text-embedding-3-small"])
        assert args.model == "text-embedding-3-small"
        assert args.dims is None
        assert args.dry_run is False

    def test_migrate_custom_model_with_dims(self):
        """Parse custom model with explicit dimensions."""
        parser = app()
        args = parser.parse_args(["embeddings", "migrate", "--model", "my-model", "--dims", "768"])
        assert args.model == "my-model"
        assert args.dims == 768

    def test_migrate_dry_run(self):
        """Parse --dry-run flag."""
        parser = app()
        args = parser.parse_args(["embeddings", "migrate", "--model", "text-embedding-3-small", "--dry-run"])
        assert args.dry_run is True


class TestStatusArgParsing:
    """Test CLI argument parsing for embeddings status command."""

    def test_status_parses(self):
        """embeddings status parses with no extra args."""
        parser = app()
        args = parser.parse_args(["embeddings", "status"])
        assert args.embeddings_command == "status"


# ============================================================================
# REST Client Command Tests
# ============================================================================


class TestBackfillREST:
    """Test embeddings backfill via REST client."""

    @patch("valence.cli.commands.embeddings.get_client")
    def test_backfill_happy_path(self, mock_get_client):
        """Backfill calls POST /admin/embeddings/backfill."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "processed": 10, "errors": 0}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "backfill"])
        result = cmd_embeddings(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["batch_size"] == 50
        assert call_body["dry_run"] is False

    @patch("valence.cli.commands.embeddings.get_client")
    def test_backfill_dry_run(self, mock_get_client):
        """Dry run passes dry_run=True."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "would_process": 5, "dry_run": True}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--dry-run"])
        result = cmd_embeddings(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["dry_run"] is True

    @patch("valence.cli.commands.embeddings.get_client")
    def test_backfill_custom_batch_size(self, mock_get_client):
        """Custom batch size passed through."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "processed": 25}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "-b", "25"])
        result = cmd_embeddings(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["batch_size"] == 25

    @patch("valence.cli.commands.embeddings.get_client")
    def test_backfill_connection_error(self, mock_get_client):
        """Handles connection error."""
        from valence.cli.http_client import ValenceConnectionError

        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceConnectionError("http://127.0.0.1:8420")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "backfill"])
        result = cmd_embeddings(args)

        assert result == 1


class TestMigrateREST:
    """Test embeddings migrate via REST client."""

    @patch("valence.cli.commands.embeddings.get_client")
    def test_migrate_happy_path(self, mock_get_client):
        """Migrate calls POST /admin/embeddings/migrate."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "cleared_embeddings": 50}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "migrate", "--model", "text-embedding-3-small"])
        result = cmd_embeddings(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["model"] == "text-embedding-3-small"

    @patch("valence.cli.commands.embeddings.get_client")
    def test_migrate_with_dims(self, mock_get_client):
        """Migrate passes dims."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "cleared_embeddings": 50}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "migrate", "--model", "custom", "--dims", "768"])
        result = cmd_embeddings(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["model"] == "custom"
        assert call_body["dims"] == 768

    @patch("valence.cli.commands.embeddings.get_client")
    def test_migrate_dry_run(self, mock_get_client):
        """Migrate dry run."""
        mock_client = MagicMock()
        mock_client.post.return_value = {"success": True, "would_affect": 50, "dry_run": True}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "migrate", "--model", "text-embedding-3-small", "--dry-run"])
        result = cmd_embeddings(args)

        assert result == 0
        call_body = mock_client.post.call_args[1]["body"]
        assert call_body["dry_run"] is True

    @patch("valence.cli.commands.embeddings.get_client")
    def test_migrate_api_error(self, mock_get_client):
        """Handles API error."""
        from valence.cli.http_client import ValenceAPIError

        mock_client = MagicMock()
        mock_client.post.side_effect = ValenceAPIError(400, "MISSING_FIELD", "model or dims required")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "migrate", "--model", "x"])
        result = cmd_embeddings(args)

        assert result == 1


class TestStatusREST:
    """Test embeddings status via REST client."""

    @patch("valence.cli.commands.embeddings.get_client")
    def test_status_happy_path(self, mock_get_client):
        """Status calls GET /admin/embeddings/status."""
        mock_client = MagicMock()
        mock_client.get.return_value = {"success": True, "stats": {"total_beliefs": 100, "with_embeddings": 80}}
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "status"])
        result = cmd_embeddings(args)

        assert result == 0
        mock_client.get.assert_called_once()
        assert "/admin/embeddings/status" in mock_client.get.call_args[0][0]

    @patch("valence.cli.commands.embeddings.get_client")
    def test_status_connection_error(self, mock_get_client):
        """Handles connection error."""
        from valence.cli.http_client import ValenceConnectionError

        mock_client = MagicMock()
        mock_client.get.side_effect = ValenceConnectionError("http://127.0.0.1:8420")
        mock_get_client.return_value = mock_client

        parser = app()
        args = parser.parse_args(["embeddings", "status"])
        result = cmd_embeddings(args)

        assert result == 1


# ============================================================================
# Command Routing
# ============================================================================


class TestEmbeddingsRouting:
    """Test embeddings command routing."""

    def test_main_dispatches_to_embeddings(self):
        """main() dispatches 'embeddings' to cmd_embeddings via args.func."""
        parser = app()
        args = parser.parse_args(["embeddings", "backfill", "--dry-run"])

        assert hasattr(args, "func")
        assert args.func is cmd_embeddings
