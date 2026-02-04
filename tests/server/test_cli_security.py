"""Tests for CLI security - Issue #28 secure token storage."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from valence.server.cli import (
    get_secure_token_dir,
    save_token_securely,
    cmd_create,
)


class TestSecureTokenStorage:
    """Test secure token file storage instead of console printing."""

    def test_get_secure_token_dir_creates_directory(self):
        """Token directory should be created with proper permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                token_dir = get_secure_token_dir()

                assert token_dir.exists()
                assert token_dir.is_dir()

                # Check permissions are 0700 (owner only)
                mode = token_dir.stat().st_mode
                assert mode & 0o777 == 0o700

    def test_save_token_creates_secure_file(self):
        """Tokens should be saved to files with 0600 permissions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                test_token = "test_token_12345"
                token_file = save_token_securely("test-client", test_token)

                # File should exist
                assert token_file.exists()

                # Check file permissions are 0600 (owner read/write only)
                mode = token_file.stat().st_mode
                assert mode & 0o777 == 0o600

                # Check content
                content = token_file.read_text().strip()
                assert content == test_token

    def test_save_token_sanitizes_client_id(self):
        """Client ID should be sanitized for use in filename."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                # Client ID with special characters / path traversal attempt
                test_token = "test_token"
                token_file = save_token_securely(
                    "test/client/../../../etc/passwd",
                    test_token,
                )

                # Key check: no path traversal should work
                # File should be in the token directory, not escaped
                assert ".." not in str(token_file)
                assert token_file.parent == Path(tmpdir) / ".valence" / "tokens"

                # Slashes should be converted to safe chars
                assert "/" not in token_file.name

                # File should still be created safely
                assert token_file.exists()
                assert token_file.read_text().strip() == test_token

    def test_cmd_create_does_not_print_token(self, capsys):
        """cmd_create should NOT print the raw token to console."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tf = Path(tmpdir) / "tokens.json"

            # Create a mock args object
            class MockArgs:
                pass

            args = MockArgs()
            args.token_file = tf
            args.client_id = "test-client"
            args.description = "Test token"
            args.scopes = "mcp:access"
            args.expires_days = None

            with patch.object(Path, "home", return_value=Path(tmpdir)):
                # Run cmd_create
                result = cmd_create(args)

                assert result == 0

                # Capture stdout
                captured = capsys.readouterr()

                # Token should NOT appear in output
                # The token format is typically a long random string
                # We check that no long alphanumeric strings that look like tokens appear
                lines = captured.out.split("\n")
                for line in lines:
                    # Skip lines that are about the file path
                    if "Token file:" in line or "cat " in line:
                        continue
                    # Check no line contains what looks like a raw token
                    # (long hex or base64-like strings)
                    words = line.split()
                    for word in words:
                        # Tokens are typically 32+ chars of hex/base64
                        if len(word) > 30 and word.isalnum():
                            # This might be a token - shouldn't be in output
                            pytest.fail(f"Possible token found in output: {word[:10]}...")

    def test_token_file_path_in_output(self, capsys):
        """cmd_create should print the path to the token file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tf = Path(tmpdir) / "tokens.json"

            class MockArgs:
                pass

            args = MockArgs()
            args.token_file = tf
            args.client_id = "test-client"
            args.description = "Test token"
            args.scopes = "mcp:access"
            args.expires_days = None

            with patch.object(Path, "home", return_value=Path(tmpdir)):
                cmd_create(args)

                captured = capsys.readouterr()

                # Should mention the token file location
                assert "Token file:" in captured.out or ".token" in captured.out

    def test_multiple_tokens_create_unique_files(self):
        """Multiple tokens for same client should create unique files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(Path, "home", return_value=Path(tmpdir)):
                file1 = save_token_securely("client", "token1")
                # No delay needed - random suffix ensures uniqueness
                file2 = save_token_securely("client", "token2")

                # Files should be different (random suffix ensures this)
                assert file1 != file2
                assert file1.read_text().strip() == "token1"
                assert file2.read_text().strip() == "token2"

    def test_token_dir_parent_created(self):
        """Parent directories should be created if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use a nested path that doesn't exist
            fake_home = Path(tmpdir) / "nonexistent" / "path"

            with patch.object(Path, "home", return_value=fake_home):
                token_dir = get_secure_token_dir()

                assert token_dir.exists()
                assert (fake_home / ".valence" / "tokens").exists()

    def test_permissions_on_existing_directory(self):
        """Permissions should be set correctly even if directory exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create with wrong permissions
            token_path = Path(tmpdir) / ".valence" / "tokens"
            token_path.mkdir(parents=True)
            os.chmod(token_path, 0o755)  # Wrong permissions

            with patch.object(Path, "home", return_value=Path(tmpdir)):
                token_dir = get_secure_token_dir()

                # Permissions should be corrected
                mode = token_dir.stat().st_mode
                assert mode & 0o777 == 0o700
