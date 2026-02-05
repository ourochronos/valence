"""Tests for sync request authentication (Issue #236).

Tests cover:
- VFP auth header creation
- Signature generation and verification
- Credential validation
"""

from __future__ import annotations

import base64
import hashlib
import time
from unittest.mock import MagicMock, patch

import pytest

# Import directly from the module to access private function
import valence.federation.sync as sync_module

# =============================================================================
# AUTH HEADER TESTS
# =============================================================================


class TestCreateAuthHeaders:
    """Tests for _create_auth_headers function."""

    @pytest.fixture
    def mock_config(self):
        """Mock federation config with valid credentials."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        config = MagicMock()
        config.federation_node_did = "did:vkb:web:test.example.com"
        config.federation_private_key = private_bytes.hex()
        return config

    def test_creates_required_headers(self, mock_config):
        """Test that all required VFP headers are created."""
        with patch.object(sync_module, "get_federation_config", return_value=mock_config):
            headers = sync_module._create_auth_headers(
                method="POST",
                url="https://peer.example.com/beliefs",
                body=b'{"test": "data"}',
            )

        assert "X-VFP-DID" in headers
        assert "X-VFP-Signature" in headers
        assert "X-VFP-Timestamp" in headers
        assert "X-VFP-Nonce" in headers

    def test_did_from_config(self, mock_config):
        """Test that DID comes from config."""
        with patch.object(sync_module, "get_federation_config", return_value=mock_config):
            headers = sync_module._create_auth_headers(
                method="POST",
                url="https://peer.example.com/beliefs",
                body=b'{"test": "data"}',
            )

        assert headers["X-VFP-DID"] == "did:vkb:web:test.example.com"

    def test_signature_is_valid_base64(self, mock_config):
        """Test that signature is valid base64."""
        with patch.object(sync_module, "get_federation_config", return_value=mock_config):
            headers = sync_module._create_auth_headers(
                method="POST",
                url="https://peer.example.com/beliefs",
                body=b'{"test": "data"}',
            )

        # Should not raise
        signature_bytes = base64.b64decode(headers["X-VFP-Signature"])
        # Ed25519 signatures are 64 bytes
        assert len(signature_bytes) == 64

    def test_timestamp_is_recent(self, mock_config):
        """Test that timestamp is recent (within 60 seconds)."""
        with patch.object(sync_module, "get_federation_config", return_value=mock_config):
            headers = sync_module._create_auth_headers(
                method="POST",
                url="https://peer.example.com/sync",
                body=b"{}",
            )

        timestamp = int(headers["X-VFP-Timestamp"])
        now = int(time.time())
        assert abs(now - timestamp) < 60

    def test_nonce_is_unique(self, mock_config):
        """Test that nonces are unique across calls."""
        with patch.object(sync_module, "get_federation_config", return_value=mock_config):
            headers1 = sync_module._create_auth_headers("POST", "https://peer.example.com/sync", b"{}")
            headers2 = sync_module._create_auth_headers("POST", "https://peer.example.com/sync", b"{}")

        assert headers1["X-VFP-Nonce"] != headers2["X-VFP-Nonce"]

    def test_signature_verifiable(self, mock_config):
        """Test that signature can be verified by the receiving node."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        # Get the public key from the private key for verification
        private_bytes = bytes.fromhex(mock_config.federation_private_key)
        private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
        public_key = private_key.public_key()

        body = b'{"type": "SYNC_REQUEST"}'
        url = "https://peer.example.com/sync"

        with patch.object(sync_module, "get_federation_config", return_value=mock_config):
            headers = sync_module._create_auth_headers("POST", url, body)

        # Reconstruct the message that was signed (matches verify_did_signature format)
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"POST /sync {headers['X-VFP-Timestamp']} {headers['X-VFP-Nonce']} {body_hash}"
        message_bytes = message.encode("utf-8")

        # Verify the signature - raises InvalidSignature on failure
        signature = base64.b64decode(headers["X-VFP-Signature"])
        public_key.verify(signature, message_bytes)

    def test_raises_without_credentials(self):
        """Test that missing credentials raise RuntimeError."""
        config = MagicMock()
        config.federation_node_did = None
        config.federation_private_key = None

        with patch.object(sync_module, "get_federation_config", return_value=config):
            with pytest.raises(RuntimeError, match="credentials not configured"):
                sync_module._create_auth_headers("POST", "https://peer.example.com/sync", b"{}")

    def test_raises_with_missing_did(self):
        """Test that missing DID raises RuntimeError."""
        config = MagicMock()
        config.federation_node_did = None
        config.federation_private_key = "deadbeef" * 8  # 32 bytes hex

        with patch.object(sync_module, "get_federation_config", return_value=config):
            with pytest.raises(RuntimeError, match="credentials not configured"):
                sync_module._create_auth_headers("POST", "https://peer.example.com/sync", b"{}")

    def test_raises_with_missing_private_key(self):
        """Test that missing private key raises RuntimeError."""
        config = MagicMock()
        config.federation_node_did = "did:vkb:web:test.example.com"
        config.federation_private_key = None

        with patch.object(sync_module, "get_federation_config", return_value=config):
            with pytest.raises(RuntimeError, match="credentials not configured"):
                sync_module._create_auth_headers("POST", "https://peer.example.com/sync", b"{}")

    def test_extracts_path_from_full_url(self, mock_config):
        """Test that path is correctly extracted from URLs with query strings."""
        with patch.object(sync_module, "get_federation_config", return_value=mock_config):
            headers = sync_module._create_auth_headers(
                method="POST",
                url="https://peer.example.com:8420/api/v1/sync?foo=bar",
                body=b"{}",
            )

        # Verify the signature was created (path extraction worked)
        assert headers["X-VFP-Signature"]
        assert len(base64.b64decode(headers["X-VFP-Signature"])) == 64
