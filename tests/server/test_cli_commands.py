"""Tests for server CLI commands (token management).

Covers:
- cmd_list: List tokens
- cmd_revoke: Revoke tokens by client-id, hash, or raw token
- cmd_verify: Verify token validity
- main(): Entry point and argument parsing
- Error handling paths
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from valence.server.auth import TokenStore, hash_token
from valence.server.cli import (
    cmd_create,
    cmd_list,
    cmd_revoke,
    cmd_verify,
    main,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def temp_token_file():
    """Create a temporary token file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        yield Path(f.name)
    # Cleanup handled by test if needed


@pytest.fixture
def populated_token_store(temp_token_file):
    """Create a token store with some test tokens."""
    store = TokenStore(temp_token_file)

    # Create a few test tokens
    token1 = store.create(
        client_id="test-client-1",
        description="First test token",
        scopes=["mcp:access"],
    )

    token2 = store.create(
        client_id="test-client-2",
        description="Second test token with longer description that might get truncated",
        scopes=["mcp:access", "mcp:admin"],
    )

    # Create an expired token
    expired_time = time.time() - 3600  # 1 hour ago
    token3 = store.create(
        client_id="expired-client",
        description="Expired token",
        expires_at=expired_time,
    )

    return store, {
        "token1": token1,
        "token2": token2,
        "token3": token3,
    }


class MockArgs:
    """Mock argparse.Namespace for testing."""

    pass


# ============================================================================
# Test cmd_list
# ============================================================================


class TestCmdList:
    """Test the list tokens command."""

    def test_list_empty_store(self, temp_token_file, capsys):
        """List with no tokens."""
        args = MockArgs()
        args.token_file = temp_token_file

        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No tokens found" in captured.out

    def test_list_with_tokens(self, temp_token_file, capsys):
        """List existing tokens."""
        # Create some tokens first
        store = TokenStore(temp_token_file)
        store.create(client_id="client-a", description="Test A")
        store.create(client_id="client-b", description="Test B")

        args = MockArgs()
        args.token_file = temp_token_file

        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "client-a" in captured.out
        assert "client-b" in captured.out
        assert "Test A" in captured.out
        assert "Test B" in captured.out
        assert "Total: 2 token(s)" in captured.out

    def test_list_shows_expiration(self, temp_token_file, capsys):
        """List shows expiration dates."""
        store = TokenStore(temp_token_file)
        # Non-expiring token
        store.create(client_id="permanent", description="Never expires")
        # Expiring token
        store.create(
            client_id="temporary",
            description="Expires soon",
            expires_at=time.time() + 86400,
        )

        args = MockArgs()
        args.token_file = temp_token_file

        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Never" in captured.out
        # Should have a date for the expiring token
        assert "20" in captured.out  # Year prefix for date

    def test_list_shows_expired_flag(self, temp_token_file, capsys):
        """List marks expired tokens."""
        store = TokenStore(temp_token_file)
        store.create(
            client_id="expired",
            description="Old token",
            expires_at=time.time() - 3600,  # 1 hour ago
        )

        args = MockArgs()
        args.token_file = temp_token_file

        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "(EXPIRED)" in captured.out

    def test_list_truncates_long_description(self, temp_token_file, capsys):
        """Long descriptions are truncated."""
        store = TokenStore(temp_token_file)
        long_desc = "This is a very long description that should be truncated in the output display"
        store.create(client_id="verbose", description=long_desc)

        args = MockArgs()
        args.token_file = temp_token_file

        result = cmd_list(args)

        assert result == 0
        captured = capsys.readouterr()
        # Should be truncated with "..."
        assert "..." in captured.out
        # Full description should NOT appear
        assert long_desc not in captured.out


# ============================================================================
# Test cmd_revoke
# ============================================================================


class TestCmdRevoke:
    """Test the revoke token command."""

    def test_revoke_by_client_id(self, temp_token_file, capsys):
        """Revoke tokens by client ID."""
        store = TokenStore(temp_token_file)
        store.create(client_id="revoke-me", description="To be revoked")

        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = "revoke-me"
        args.hash = None
        args.token = None

        result = cmd_revoke(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Revoked" in captured.out
        assert "revoke-me" in captured.out

        # Reload the store from file to verify token is gone
        store2 = TokenStore(temp_token_file)
        assert len(store2.get_by_client_id("revoke-me")) == 0

    def test_revoke_by_client_id_not_found(self, temp_token_file, capsys):
        """Revoke non-existent client ID."""
        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = "nonexistent"
        args.hash = None
        args.token = None

        result = cmd_revoke(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "No tokens found" in captured.out

    def test_revoke_by_hash(self, temp_token_file, capsys):
        """Revoke by token hash."""
        store = TokenStore(temp_token_file)
        raw_token = store.create(client_id="hash-test", description="Test")
        token_hash = hash_token(raw_token)

        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = None
        args.hash = token_hash
        args.token = None

        result = cmd_revoke(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "revoked" in captured.out.lower()

    def test_revoke_by_hash_not_found(self, temp_token_file, capsys):
        """Revoke by non-existent hash."""
        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = None
        args.hash = "nonexistent_hash_value_12345"
        args.token = None

        result = cmd_revoke(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_revoke_by_raw_token(self, temp_token_file, capsys):
        """Revoke by raw token string."""
        store = TokenStore(temp_token_file)
        raw_token = store.create(client_id="raw-test", description="Test")

        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = None
        args.hash = None
        args.token = raw_token

        result = cmd_revoke(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "revoked" in captured.out.lower()

    def test_revoke_by_raw_token_not_found(self, temp_token_file, capsys):
        """Revoke by invalid raw token."""
        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = None
        args.hash = None
        args.token = "invalid_token_12345"

        result = cmd_revoke(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "invalid" in captured.out.lower()

    def test_revoke_no_arguments(self, temp_token_file, capsys):
        """Revoke with no arguments fails."""
        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = None
        args.hash = None
        args.token = None

        result = cmd_revoke(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Must provide" in captured.out

    def test_revoke_multiple_tokens_for_client(self, temp_token_file, capsys):
        """Revoke all tokens for a client with multiple tokens."""
        store = TokenStore(temp_token_file)
        store.create(client_id="multi-client", description="Token 1")
        store.create(client_id="multi-client", description="Token 2")
        store.create(client_id="multi-client", description="Token 3")

        assert len(store.get_by_client_id("multi-client")) == 3

        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = "multi-client"
        args.hash = None
        args.token = None

        result = cmd_revoke(args)

        assert result == 0
        # Reload store to verify
        store = TokenStore(temp_token_file)
        assert len(store.get_by_client_id("multi-client")) == 0


# ============================================================================
# Test cmd_verify
# ============================================================================


class TestCmdVerify:
    """Test the verify token command."""

    def test_verify_valid_token(self, temp_token_file, capsys):
        """Verify a valid token."""
        store = TokenStore(temp_token_file)
        raw_token = store.create(
            client_id="verify-test",
            description="Verification test",
            scopes=["mcp:access", "mcp:admin"],
        )

        args = MockArgs()
        args.token_file = temp_token_file
        args.token = raw_token

        result = cmd_verify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "VALID" in captured.out
        assert "verify-test" in captured.out
        assert "mcp:access" in captured.out
        assert "mcp:admin" in captured.out

    def test_verify_invalid_token(self, temp_token_file, capsys):
        """Verify an invalid token."""
        args = MockArgs()
        args.token_file = temp_token_file
        args.token = "invalid_token_12345"

        result = cmd_verify(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "INVALID" in captured.out

    def test_verify_expired_token(self, temp_token_file, capsys):
        """Verify an expired token."""
        store = TokenStore(temp_token_file)
        raw_token = store.create(
            client_id="expired-verify",
            description="Expired",
            expires_at=time.time() - 3600,
        )

        args = MockArgs()
        args.token_file = temp_token_file
        args.token = raw_token

        result = cmd_verify(args)

        # Expired tokens should be invalid
        assert result == 1
        captured = capsys.readouterr()
        assert "INVALID" in captured.out

    def test_verify_shows_expiration(self, temp_token_file, capsys):
        """Verify shows expiration info."""
        store = TokenStore(temp_token_file)
        # Non-expiring token
        raw_token = store.create(
            client_id="no-expire",
            description="Never expires",
        )

        args = MockArgs()
        args.token_file = temp_token_file
        args.token = raw_token

        result = cmd_verify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Never" in captured.out

    def test_verify_shows_expiration_date(self, temp_token_file, capsys):
        """Verify shows expiration date for expiring tokens."""
        store = TokenStore(temp_token_file)
        future_time = time.time() + 86400 * 30  # 30 days
        raw_token = store.create(
            client_id="will-expire",
            description="Expires in 30 days",
            expires_at=future_time,
        )

        args = MockArgs()
        args.token_file = temp_token_file
        args.token = raw_token

        result = cmd_verify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Expires:" in captured.out
        # Should show a date, not "Never"
        assert "Never" not in captured.out


# ============================================================================
# Test main() entry point
# ============================================================================


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestMain:
    """Test the main CLI entry point."""

    def test_main_create_command(self, temp_token_file, capsys):
        """Main dispatches to create command."""
        with patch(
            "sys.argv",
            [
                "valence-token",
                "--token-file",
                str(temp_token_file),
                "create",
                "--client-id",
                "main-test",
                "--description",
                "Test token",
            ],
        ):
            with patch.object(Path, "home", return_value=temp_token_file.parent):
                result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Token created" in captured.out

    def test_main_list_command(self, temp_token_file, capsys):
        """Main dispatches to list command."""
        # Create a token first
        store = TokenStore(temp_token_file)
        store.create(client_id="list-test", description="Test")

        with patch(
            "sys.argv",
            [
                "valence-token",
                "--token-file",
                str(temp_token_file),
                "list",
            ],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "list-test" in captured.out

    def test_main_revoke_command(self, temp_token_file, capsys):
        """Main dispatches to revoke command."""
        store = TokenStore(temp_token_file)
        store.create(client_id="revoke-test", description="Test")

        with patch(
            "sys.argv",
            [
                "valence-token",
                "--token-file",
                str(temp_token_file),
                "revoke",
                "--client-id",
                "revoke-test",
            ],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "Revoked" in captured.out

    def test_main_verify_command(self, temp_token_file, capsys):
        """Main dispatches to verify command."""
        store = TokenStore(temp_token_file)
        raw_token = store.create(client_id="verify-test", description="Test")

        with patch(
            "sys.argv",
            [
                "valence-token",
                "--token-file",
                str(temp_token_file),
                "verify",
                raw_token,
            ],
        ):
            result = main()

        assert result == 0
        captured = capsys.readouterr()
        assert "VALID" in captured.out

    def test_main_no_command_exits(self):
        """Main with no command exits with error."""
        with patch("sys.argv", ["valence-token"]):
            with pytest.raises(SystemExit) as exc:
                main()
            # argparse exits with 2 for missing required args
            assert exc.value.code == 2

    def test_main_default_token_file(self, capsys):
        """Main uses default token file path."""
        # This tests that the default path is used when not specified
        with patch("sys.argv", ["valence-token", "list"]):
            # Mock the TokenStore to avoid file system issues
            with patch("valence.server.cli.TokenStore") as MockStore:  # noqa: N806
                mock_instance = MockStore.return_value
                mock_instance.list_tokens.return_value = []

                main()

        # Should use default path
        MockStore.assert_called_once()
        call_args = MockStore.call_args[0][0]
        assert "tokens.json" in str(call_args)


# ============================================================================
# Test argument validation
# ============================================================================


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
class TestArgumentValidation:
    """Test CLI argument validation."""

    def test_create_requires_client_id(self):
        """Create command requires --client-id."""
        with patch("sys.argv", ["valence-token", "create"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2

    def test_verify_requires_token_arg(self):
        """Verify command requires token positional arg."""
        with patch("sys.argv", ["valence-token", "verify"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 2

    def test_create_with_expires_days(self, temp_token_file, capsys):
        """Create with expiration days."""
        with patch(
            "sys.argv",
            [
                "valence-token",
                "--token-file",
                str(temp_token_file),
                "create",
                "--client-id",
                "expiring",
                "--expires-days",
                "30",
            ],
        ):
            with patch.object(Path, "home", return_value=temp_token_file.parent):
                result = main()

        assert result == 0

        # Verify the token has expiration
        store = TokenStore(temp_token_file)
        tokens = store.get_by_client_id("expiring")
        assert len(tokens) == 1
        assert tokens[0].expires_at is not None

    def test_create_with_custom_scopes(self, temp_token_file, capsys):
        """Create with custom scopes."""
        with patch(
            "sys.argv",
            [
                "valence-token",
                "--token-file",
                str(temp_token_file),
                "create",
                "--client-id",
                "scoped",
                "--scopes",
                "mcp:access,mcp:admin,custom:scope",
            ],
        ):
            with patch.object(Path, "home", return_value=temp_token_file.parent):
                result = main()

        assert result == 0

        # Verify the scopes
        store = TokenStore(temp_token_file)
        tokens = store.get_by_client_id("scoped")
        assert len(tokens) == 1
        assert set(tokens[0].scopes) == {"mcp:access", "mcp:admin", "custom:scope"}


# ============================================================================
# Test error handling
# ============================================================================


class TestErrorHandling:
    """Test error handling in CLI commands."""

    def test_create_handles_file_permission_error(self, temp_token_file, capsys):
        """Create handles file permission errors gracefully."""
        # This test verifies behavior when token file creation fails
        # We'll mock save_token_securely to raise an exception

        args = MockArgs()
        args.token_file = temp_token_file
        args.client_id = "error-test"
        args.description = "Test"
        args.scopes = "mcp:access"
        args.expires_days = None

        with patch("valence.server.cli.save_token_securely") as mock_save:
            mock_save.side_effect = PermissionError("Access denied")

            with pytest.raises(PermissionError):
                cmd_create(args)

    def test_list_handles_corrupt_token_file(self, temp_token_file, capsys):
        """List handles corrupt token file."""
        # Write invalid JSON
        with open(temp_token_file, "w") as f:
            f.write("not valid json {{{")

        args = MockArgs()
        args.token_file = temp_token_file

        # Should not crash, store loads empty on error
        result = cmd_list(args)
        assert result == 0

        captured = capsys.readouterr()
        assert "No tokens found" in captured.out

    def test_verify_empty_token(self, temp_token_file, capsys):
        """Verify empty token string."""
        args = MockArgs()
        args.token_file = temp_token_file
        args.token = ""

        result = cmd_verify(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "INVALID" in captured.out
