"""Crypto service for PRE encryption/decryption (#340).

Wraps our-crypto's PRE backend for belief sharing. Encrypts belief content
before sharing, decrypts on receive. Falls back to unencrypted when:
- No recipient public key is available
- No local private key is configured
- The our-crypto library encounters errors

Old shares with {"algorithm": "none"} are always readable (backward compat).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Envelope algorithm identifiers
ALGORITHM_NONE = "none"
ALGORITHM_X25519 = "x25519-ecies"

# Backend type: "x25519" for production, "mock" for testing
_BACKEND_TYPE = "x25519"


def _get_pre_backend():
    """Get the PRE backend instance."""
    from our_crypto import create_pre_backend

    return create_pre_backend(_BACKEND_TYPE)


@dataclass
class EncryptedEnvelope:
    """Structured encrypted envelope for sharing."""

    algorithm: str
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"algorithm": self.algorithm, **self.data}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EncryptedEnvelope:
        algo = d.get("algorithm", ALGORITHM_NONE)
        data = {k: v for k, v in d.items() if k != "algorithm"}
        return cls(algorithm=algo, data=data)


def encrypt_for_sharing(content: str, recipient_did: str) -> dict[str, Any]:
    """Encrypt belief content for a recipient.

    Attempts X25519 ECIES encryption if the recipient's public key is
    available. Falls back to plaintext envelope otherwise.

    Args:
        content: The belief content to encrypt.
        recipient_did: The recipient's DID (used to look up public key).

    Returns:
        Envelope dict suitable for JSON storage in shares.encrypted_envelope.
    """
    pub_key = _resolve_recipient_public_key(recipient_did)

    if pub_key is None:
        logger.debug("No public key for %s — sharing unencrypted", recipient_did)
        return _plaintext_envelope(content)

    try:
        backend = _get_pre_backend()
        ciphertext = backend.encrypt(content.encode("utf-8"), pub_key)

        return {
            "algorithm": ALGORITHM_X25519,
            "ciphertext_id": ciphertext.ciphertext_id.hex(),
            "encrypted_data": ciphertext.encrypted_data.hex(),
            "recipient_id": ciphertext.recipient_id.hex(),
            "is_reencrypted": ciphertext.is_reencrypted,
            "metadata": ciphertext.metadata if ciphertext.metadata else {},
        }
    except Exception:
        logger.warning("PRE encryption failed for %s — falling back to plaintext", recipient_did, exc_info=True)
        return _plaintext_envelope(content)


def decrypt_envelope(envelope: dict[str, Any], private_key_bytes: bytes | None = None) -> str | None:
    """Decrypt an encrypted envelope.

    Handles both legacy plaintext and X25519 ECIES envelopes.

    Args:
        envelope: The envelope dict from shares.encrypted_envelope.
        private_key_bytes: The recipient's 32-byte private key. If None,
            attempts to load from config.

    Returns:
        The decrypted content string, or None if decryption fails.
    """
    algo = envelope.get("algorithm", ALGORITHM_NONE)

    if algo == ALGORITHM_NONE:
        return envelope.get("content")

    if algo == ALGORITHM_X25519:
        return _decrypt_x25519(envelope, private_key_bytes)

    logger.warning("Unknown envelope algorithm: %s", algo)
    return None


def _plaintext_envelope(content: str) -> dict[str, Any]:
    """Build a plaintext (unencrypted) envelope."""
    return {
        "algorithm": ALGORITHM_NONE,
        "content": content,
    }


def _resolve_recipient_public_key(recipient_did: str) -> Any:
    """Look up the recipient's PRE public key from their DID.

    Currently checks the local identity store. In federation, this would
    query the DID document from the recipient's node.

    Returns:
        PREPublicKey or None if not found.
    """
    try:
        from ..cli.commands.identity import _load_manager

        _mgr, store = _load_manager()
        node = store.get_node(recipient_did)
        if node is None or not node.is_active:
            return None

        # Ed25519 keys in identity store cannot be directly used for X25519 PRE.
        # TODO(#342): Store PRE public keys alongside Ed25519 keys when MLS groups land.
        return None
    except Exception:
        return None


def _decrypt_x25519(envelope: dict[str, Any], private_key_bytes: bytes | None = None) -> str | None:
    """Decrypt an X25519 ECIES envelope."""
    try:
        from our_crypto import PRECiphertext, PREPrivateKey

        if private_key_bytes is None:
            from .identity_service import _get_private_key_bytes

            private_key_bytes = _get_private_key_bytes()

        if private_key_bytes is None:
            logger.warning("No private key available for decryption")
            return None

        backend = _get_pre_backend()

        # Reconstruct ciphertext from envelope
        from datetime import datetime

        ciphertext = PRECiphertext(
            ciphertext_id=bytes.fromhex(envelope["ciphertext_id"]),
            encrypted_data=bytes.fromhex(envelope["encrypted_data"]),
            recipient_id=bytes.fromhex(envelope["recipient_id"]),
            is_reencrypted=envelope.get("is_reencrypted", False),
            created_at=datetime.now(),
            metadata=envelope.get("metadata", {}),
        )

        private_key = PREPrivateKey(
            key_id=ciphertext.recipient_id,
            key_bytes=private_key_bytes,
            created_at=datetime.now(),
            metadata={},
        )

        plaintext = backend.decrypt(ciphertext, private_key)
        return plaintext.decode("utf-8")
    except Exception:
        logger.warning("Failed to decrypt X25519 envelope", exc_info=True)
        return None
