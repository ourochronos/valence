"""Identity service for Ed25519 signing and verification (#339).

Provides a thin wrapper around the node's Ed25519 keypair from config.
Used by sharing, federation, and consent chain operations.

Falls back to placeholder signatures when no key is configured, supporting
backward compatibility during the transition from Cycle 1 placeholders.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SignatureResult:
    """Result of a signing operation."""

    signature: bytes
    signer_did: str
    is_placeholder: bool = False


def get_local_did() -> str:
    """Get the local node's DID from config.

    Falls back to VALENCE_LOCAL_DID env var, then "did:valence:local".
    """
    import os

    from .config import get_config

    config = get_config()
    if config.federation_did:
        return config.federation_did
    return os.environ.get("VALENCE_LOCAL_DID", "did:valence:local")


def _get_private_key_bytes() -> bytes | None:
    """Load the Ed25519 private key bytes from config."""
    from .config import get_config

    config = get_config()
    key_hex = config.federation_private_key
    if key_hex:
        return bytes.fromhex(key_hex)
    return None


def sign_data(data: bytes) -> SignatureResult:
    """Sign arbitrary data with the local node's Ed25519 private key.

    If no private key is configured, returns a SHA256 placeholder signature
    with is_placeholder=True for backward compatibility.

    Args:
        data: The bytes to sign.

    Returns:
        SignatureResult with the signature, signer DID, and placeholder flag.
    """
    local_did = get_local_did()
    key_bytes = _get_private_key_bytes()

    if key_bytes is None:
        # Placeholder mode: SHA256 hash (backward compat with Cycle 1)
        logger.debug("No Ed25519 key configured — using placeholder signature")
        placeholder = hashlib.sha256(data).digest()
        return SignatureResult(signature=placeholder, signer_did=local_did, is_placeholder=True)

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private_key = Ed25519PrivateKey.from_private_bytes(key_bytes)
    signature = private_key.sign(data)
    return SignatureResult(signature=signature, signer_did=local_did, is_placeholder=False)


def verify_signature(data: bytes, signature: bytes, public_key_bytes: bytes) -> bool:
    """Verify an Ed25519 signature.

    Args:
        data: The original signed data.
        signature: The 64-byte Ed25519 signature.
        public_key_bytes: The 32-byte Ed25519 public key.

    Returns:
        True if the signature is valid, False otherwise.
    """
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature, data)
        return True
    except Exception:
        return False


def build_consent_chain_message(
    belief_id: str,
    origin_did: str,
    recipient_did: str,
    intent: str,
    policy_json: str,
) -> bytes:
    """Build the canonical message for consent chain signing.

    The message is deterministic given the same inputs, so it can be
    reconstructed for verification.

    Args:
        belief_id: UUID of the belief being shared.
        origin_did: DID of the sharer.
        recipient_did: DID of the recipient.
        intent: Sharing intent (know_me, work_with_me, etc.).
        policy_json: JSON-serialized policy.

    Returns:
        SHA256 hash of the canonical message (32 bytes).
    """
    canonical = f"{belief_id}:{origin_did}:{recipient_did}:{intent}:{policy_json}"
    return hashlib.sha256(canonical.encode()).digest()


def compute_chain_hash(origin_signature: bytes) -> bytes:
    """Compute the chain hash from an origin signature.

    The chain hash is SHA256(origin_signature), used for quick
    integrity verification of the consent chain.
    """
    return hashlib.sha256(origin_signature).digest()
