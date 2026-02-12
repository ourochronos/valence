"""Tests for identity service (#339).

Tests cover:
1. get_local_did falls back through config -> env -> default
2. sign_data with configured Ed25519 key produces real signature
3. sign_data without key produces placeholder
4. verify_signature accepts valid signature
5. verify_signature rejects invalid signature
6. build_consent_chain_message is deterministic
7. compute_chain_hash is SHA256 of signature
8. SignatureResult fields
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valence.core.identity_service import (
    SignatureResult,
    build_consent_chain_message,
    compute_chain_hash,
    sign_data,
    verify_signature,
)


class TestGetLocalDid:
    """Test DID resolution from config/env."""

    @patch("valence.core.config.get_config")
    def test_uses_config_did(self, mock_get_config):
        from valence.core.identity_service import get_local_did

        mock_config = MagicMock()
        mock_config.federation_did = "did:valence:from-config"
        mock_get_config.return_value = mock_config

        assert get_local_did() == "did:valence:from-config"

    @patch("valence.core.config.get_config")
    def test_falls_back_to_env(self, mock_get_config, monkeypatch):
        from valence.core.identity_service import get_local_did

        mock_config = MagicMock()
        mock_config.federation_did = None
        mock_get_config.return_value = mock_config
        monkeypatch.setenv("VALENCE_LOCAL_DID", "did:valence:from-env")

        assert get_local_did() == "did:valence:from-env"

    @patch("valence.core.config.get_config")
    def test_falls_back_to_default(self, mock_get_config, monkeypatch):
        from valence.core.identity_service import get_local_did

        mock_config = MagicMock()
        mock_config.federation_did = None
        mock_get_config.return_value = mock_config
        monkeypatch.delenv("VALENCE_LOCAL_DID", raising=False)

        assert get_local_did() == "did:valence:local"


class TestSignData:
    """Test Ed25519 signing and placeholder fallback."""

    @patch("valence.core.identity_service._get_private_key_bytes")
    @patch("valence.core.identity_service.get_local_did")
    def test_placeholder_when_no_key(self, mock_did, mock_key):
        mock_did.return_value = "did:valence:test"
        mock_key.return_value = None

        result = sign_data(b"hello world")

        assert result.is_placeholder is True
        assert result.signer_did == "did:valence:test"
        assert len(result.signature) == 32  # SHA256

    @patch("valence.core.identity_service._get_private_key_bytes")
    @patch("valence.core.identity_service.get_local_did")
    def test_real_signature_with_key(self, mock_did, mock_key):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.generate()
        mock_did.return_value = "did:valence:real"
        mock_key.return_value = key.private_bytes_raw()

        result = sign_data(b"hello world")

        assert result.is_placeholder is False
        assert result.signer_did == "did:valence:real"
        assert len(result.signature) == 64  # Ed25519

    @patch("valence.core.identity_service._get_private_key_bytes")
    @patch("valence.core.identity_service.get_local_did")
    def test_real_signature_is_verifiable(self, mock_did, mock_key):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.generate()
        pub_bytes = key.public_key().public_bytes_raw()
        mock_did.return_value = "did:valence:test"
        mock_key.return_value = key.private_bytes_raw()

        data = b"test data for signing"
        result = sign_data(data)

        assert verify_signature(data, result.signature, pub_bytes) is True

    @patch("valence.core.identity_service._get_private_key_bytes")
    @patch("valence.core.identity_service.get_local_did")
    def test_placeholder_is_deterministic(self, mock_did, mock_key):
        mock_did.return_value = "did:valence:test"
        mock_key.return_value = None

        result1 = sign_data(b"same data")
        result2 = sign_data(b"same data")

        assert result1.signature == result2.signature


class TestVerifySignature:
    """Test Ed25519 verification."""

    def test_valid_signature(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.generate()
        data = b"test message"
        sig = key.sign(data)
        pub = key.public_key().public_bytes_raw()

        assert verify_signature(data, sig, pub) is True

    def test_wrong_data_fails(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.generate()
        sig = key.sign(b"original message")
        pub = key.public_key().public_bytes_raw()

        assert verify_signature(b"different message", sig, pub) is False

    def test_wrong_key_fails(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key1 = Ed25519PrivateKey.generate()
        key2 = Ed25519PrivateKey.generate()
        data = b"test message"
        sig = key1.sign(data)
        pub2 = key2.public_key().public_bytes_raw()

        assert verify_signature(data, sig, pub2) is False

    def test_corrupted_signature_fails(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.generate()
        data = b"test message"
        sig = bytearray(key.sign(data))
        sig[0] ^= 0xFF  # Corrupt first byte
        pub = key.public_key().public_bytes_raw()

        assert verify_signature(data, bytes(sig), pub) is False


class TestBuildConsentChainMessage:
    """Test canonical message construction."""

    def test_deterministic(self):
        msg1 = build_consent_chain_message("belief-1", "did:a", "did:b", "know_me", '{"intent":"know_me"}')
        msg2 = build_consent_chain_message("belief-1", "did:a", "did:b", "know_me", '{"intent":"know_me"}')
        assert msg1 == msg2

    def test_different_inputs_different_messages(self):
        msg1 = build_consent_chain_message("belief-1", "did:a", "did:b", "know_me", '{}')
        msg2 = build_consent_chain_message("belief-2", "did:a", "did:b", "know_me", '{}')
        assert msg1 != msg2

    def test_returns_32_bytes(self):
        msg = build_consent_chain_message("b", "d1", "d2", "know_me", "{}")
        assert len(msg) == 32  # SHA256


class TestComputeChainHash:
    """Test chain hash computation."""

    def test_is_sha256_of_signature(self):
        import hashlib

        sig = b"x" * 64
        expected = hashlib.sha256(sig).digest()
        assert compute_chain_hash(sig) == expected

    def test_different_signatures_different_hashes(self):
        assert compute_chain_hash(b"a" * 64) != compute_chain_hash(b"b" * 64)

    def test_returns_32_bytes(self):
        assert len(compute_chain_hash(b"sig")) == 32
