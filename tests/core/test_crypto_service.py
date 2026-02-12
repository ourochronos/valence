"""Tests for crypto service (#340).

Tests cover:
1. encrypt_for_sharing returns plaintext envelope when no recipient key
2. encrypt_for_sharing returns x25519 envelope when key available
3. decrypt_envelope handles plaintext ("none") envelopes
4. decrypt_envelope handles x25519 envelopes
5. decrypt_envelope returns None for unknown algorithm
6. EncryptedEnvelope serialization round-trip
7. _plaintext_envelope format
8. Full encrypt → decrypt round-trip via our-crypto mock backend
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valence.core.crypto_service import (
    ALGORITHM_NONE,
    ALGORITHM_X25519,
    EncryptedEnvelope,
    decrypt_envelope,
    encrypt_for_sharing,
    _plaintext_envelope,
)


class TestEncryptedEnvelope:
    """Test envelope dataclass."""

    def test_to_dict(self):
        env = EncryptedEnvelope(algorithm="none", data={"content": "hello"})
        d = env.to_dict()
        assert d["algorithm"] == "none"
        assert d["content"] == "hello"

    def test_from_dict(self):
        d = {"algorithm": "x25519-ecies", "ciphertext_id": "abc", "encrypted_data": "def"}
        env = EncryptedEnvelope.from_dict(d)
        assert env.algorithm == "x25519-ecies"
        assert env.data["ciphertext_id"] == "abc"
        assert "algorithm" not in env.data

    def test_round_trip(self):
        original = {"algorithm": "none", "content": "test"}
        env = EncryptedEnvelope.from_dict(original)
        assert env.to_dict() == original


class TestPlaintextEnvelope:
    """Test plaintext envelope construction."""

    def test_format(self):
        env = _plaintext_envelope("hello world")
        assert env["algorithm"] == ALGORITHM_NONE
        assert env["content"] == "hello world"
        assert len(env) == 2

    def test_no_extra_fields(self):
        env = _plaintext_envelope("test")
        assert "note" not in env
        assert "encrypted_data" not in env


class TestEncryptForSharing:
    """Test encryption for sharing."""

    @patch("valence.core.crypto_service._resolve_recipient_public_key")
    def test_plaintext_when_no_key(self, mock_resolve):
        mock_resolve.return_value = None
        env = encrypt_for_sharing("hello", "did:valence:unknown")
        assert env["algorithm"] == ALGORITHM_NONE
        assert env["content"] == "hello"

    @patch("valence.core.crypto_service._resolve_recipient_public_key")
    @patch("valence.core.crypto_service._get_pre_backend")
    def test_x25519_when_key_available(self, mock_backend_fn, mock_resolve):
        from our_crypto import create_pre_backend

        backend = create_pre_backend("mock")
        mock_backend_fn.return_value = backend
        keypair = backend.generate_keypair(b"recipient")
        mock_resolve.return_value = keypair.public_key

        env = encrypt_for_sharing("secret belief", "did:valence:recipient")

        assert env["algorithm"] == ALGORITHM_X25519
        assert "ciphertext_id" in env
        assert "encrypted_data" in env
        assert "recipient_id" in env

    @patch("valence.core.crypto_service._resolve_recipient_public_key")
    def test_fallback_on_encryption_error(self, mock_resolve):
        """If encryption fails, fall back to plaintext."""
        mock_resolve.return_value = MagicMock()  # Not a real PREPublicKey

        env = encrypt_for_sharing("hello", "did:valence:bad")

        assert env["algorithm"] == ALGORITHM_NONE
        assert env["content"] == "hello"


class TestDecryptEnvelope:
    """Test envelope decryption."""

    def test_plaintext_envelope(self):
        env = {"algorithm": "none", "content": "hello"}
        result = decrypt_envelope(env)
        assert result == "hello"

    def test_plaintext_missing_content(self):
        env = {"algorithm": "none"}
        result = decrypt_envelope(env)
        assert result is None

    def test_unknown_algorithm(self):
        env = {"algorithm": "rsa-4096"}
        result = decrypt_envelope(env)
        assert result is None

    def test_default_algorithm_is_none(self):
        env = {"content": "legacy"}
        result = decrypt_envelope(env)
        assert result == "legacy"


class TestEncryptDecryptRoundTrip:
    """Test full encrypt → decrypt round trip with mock backend."""

    @patch("valence.core.crypto_service._resolve_recipient_public_key")
    @patch("valence.core.crypto_service._get_pre_backend")
    def test_round_trip_with_mock(self, mock_backend_fn, mock_resolve):
        from our_crypto import create_pre_backend

        backend = create_pre_backend("mock")
        mock_backend_fn.return_value = backend
        keypair = backend.generate_keypair(b"test-did")
        mock_resolve.return_value = keypair.public_key

        # Encrypt
        content = "Python is great for data science"
        env = encrypt_for_sharing(content, "did:valence:test")

        assert env["algorithm"] == ALGORITHM_X25519

        # Decrypt
        result = decrypt_envelope(env, private_key_bytes=keypair.private_key.key_bytes)

        assert result == content
