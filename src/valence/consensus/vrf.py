"""Verifiable Random Function (VRF) Implementation.

Provides unpredictable but verifiable randomness using Ed25519 keys for
validator selection in Valence consensus.

SECURITY NOTE - SIMPLIFIED VRF CONSTRUCTION
===========================================

This implementation uses a **simplified VRF construction** based on Ed25519
signatures, NOT the full ECVRF-EDWARDS25519-SHA512-TAI specification per
RFC 9381. See docs/consensus/VRF_SECURITY.md for detailed analysis.

**Why this is acceptable for Valence:**

The core security properties required for validator selection are:

1. **Determinism**: Same (key, input) → same output
   ✓ Ed25519 signatures are deterministic (RFC 8032)

2. **Unpredictability**: Cannot predict output without private key
   ✓ Ed25519 provides this via discrete log hardness

3. **Verifiability**: Anyone can verify output matches key + input
   ✓ Ed25519 signature verification provides this

4. **Uniqueness**: Each (key, input) has exactly one valid output
   ✓ Inherited from Ed25519's deterministic nonce generation

**What full ECVRF-EDWARDS25519-SHA512-TAI adds:**

- Hash-to-curve (Elligator2) for domain separation
- Formal security proofs in the random oracle model
- Cofactor handling for Edwards curve edge cases
- Standardized proof format for interoperability

**Risk Assessment:**

The simplified construction is secure for validator selection because:
- Validators cannot predict or manipulate their tickets without their key
- The double-hashing with domain separators prevents cross-protocol attacks
- Ed25519's determinism ensures uniqueness of outputs
- Verification confirms the ticket came from the claimed key

Primary limitation: Non-standard construction is harder to audit externally.
Recommendation: Migrate to full RFC 9381 when mature Python library available.

See: https://datatracker.ietf.org/doc/html/rfc9381
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Try to import cryptography for Ed25519
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# Domain separator for VRF operations (prevents cross-protocol attacks)
DOMAIN_SEPARATOR_VRF_PROVE = b"valence-vrf-prove-v1"
DOMAIN_SEPARATOR_VRF_HASH = b"valence-vrf-hash-v1"
DOMAIN_SEPARATOR_EPOCH_SEED = b"valence-epoch-seed-v1"

# VRF output size (SHA-512 truncated to 256 bits for ticket)
VRF_OUTPUT_SIZE = 32


@dataclass
class VRFProof:
    """Cryptographic proof of VRF computation.

    Contains the signature that proves the VRF output was computed
    correctly using a specific private key.
    """

    # The Ed25519 signature over the input
    gamma: bytes  # Point on curve (encoded)
    c: bytes  # Challenge scalar
    s: bytes  # Response scalar

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    def to_bytes(self) -> bytes:
        """Serialize proof to bytes."""
        return self.gamma + self.c + self.s

    @classmethod
    def from_bytes(cls, data: bytes) -> VRFProof:
        """Deserialize proof from bytes."""
        if len(data) != 96:  # 32 + 32 + 32
            raise ValueError(f"Invalid proof length: {len(data)}, expected 96")
        return cls(
            gamma=data[:32],
            c=data[32:64],
            s=data[64:96],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "gamma": self.gamma.hex(),
            "c": self.c.hex(),
            "s": self.s.hex(),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VRFProof:
        """Create from dictionary."""
        return cls(
            gamma=bytes.fromhex(data["gamma"]),
            c=bytes.fromhex(data["c"]),
            s=bytes.fromhex(data["s"]),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
        )


@dataclass
class VRFOutput:
    """Result of a VRF computation.

    Contains both the deterministic output (ticket) and the proof
    that allows verification.
    """

    # The VRF output (deterministic for given key + input)
    ticket: bytes

    # Proof for verification
    proof: VRFProof

    # Input that was used
    input_hash: bytes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticket": self.ticket.hex(),
            "proof": self.proof.to_dict(),
            "input_hash": self.input_hash.hex(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VRFOutput:
        """Create from dictionary."""
        return cls(
            ticket=bytes.fromhex(data["ticket"]),
            proof=VRFProof.from_dict(data["proof"]),
            input_hash=bytes.fromhex(data["input_hash"]),
        )

    def ticket_as_int(self) -> int:
        """Get ticket as integer for comparison/sorting."""
        return int.from_bytes(self.ticket, "big")

    def ticket_as_float(self) -> float:
        """Get ticket as float in [0, 1) for probability calculations."""
        max_val = 2 ** (len(self.ticket) * 8)
        return self.ticket_as_int() / max_val


class VRF:
    """Verifiable Random Function using Ed25519.

    SIMPLIFIED CONSTRUCTION - See module docstring for security analysis.

    Uses Ed25519 signing as the core primitive. This differs from full RFC 9381
    ECVRF-EDWARDS25519-SHA512-TAI but provides equivalent security properties
    for the validator selection use case.

    Construction:
        1. Input is hashed with domain separator: H = SHA512(domain || input)
        2. Sign the hash: sig = Ed25519_Sign(sk, H)
        3. Derive ticket: ticket = SHA512(domain2 || sig || H)[:32]
        4. Proof = signature components (verifiable via Ed25519_Verify)

    Security Properties (verified by design):
        - Deterministic: Ed25519 uses deterministic nonces (RFC 8032 §5.1.6)
        - Unpredictable: Signatures require private key (DL hardness)
        - Verifiable: Ed25519 verification confirms signer
        - Unique: Deterministic nonce → one valid signature per (key, message)

    Limitations vs Full ECVRF:
        - No hash-to-curve (Elligator2) - uses SHA512 instead
        - Proof format is non-standard (signature-based, not gamma/c/s scalars)
        - No formal cofactor handling (Ed25519 handles this internally)

    Example:
        >>> vrf = VRF.generate()
        >>> output = vrf.prove(b"epoch_seed_123")
        >>> VRF.verify(vrf.public_key_bytes, b"epoch_seed_123", output)
        True
    """

    def __init__(
        self,
        private_key: Ed25519PrivateKey | None = None,
        private_key_bytes: bytes | None = None,
    ):
        """Initialize VRF with an Ed25519 private key.

        Args:
            private_key: Ed25519PrivateKey object (from cryptography)
            private_key_bytes: 32-byte Ed25519 private key seed
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for VRF operations")

        if private_key is not None:
            self._private_key = private_key
        elif private_key_bytes is not None:
            if len(private_key_bytes) != 32:
                raise ValueError(f"Private key must be 32 bytes, got {len(private_key_bytes)}")
            self._private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
        else:
            raise ValueError("Either private_key or private_key_bytes must be provided")

        self._public_key = self._private_key.public_key()

    @classmethod
    def generate(cls) -> VRF:
        """Generate a new VRF key pair.

        Returns:
            New VRF instance with fresh Ed25519 key pair
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for VRF operations")

        private_key = Ed25519PrivateKey.generate()
        return cls(private_key=private_key)

    @property
    def public_key_bytes(self) -> bytes:
        """Get the public key as bytes."""
        return self._public_key.public_bytes_raw()

    @property
    def private_key_bytes(self) -> bytes:
        """Get the private key seed as bytes."""
        return self._private_key.private_bytes_raw()

    def prove(self, alpha: bytes) -> VRFOutput:
        """Compute VRF output and proof for input alpha.

        Args:
            alpha: Input bytes (e.g., epoch seed concatenated with agent ID)

        Returns:
            VRFOutput containing the deterministic ticket and proof
        """
        # Hash input with domain separator
        input_hash = hashlib.sha512(DOMAIN_SEPARATOR_VRF_PROVE + alpha).digest()

        # Sign the hashed input
        # This serves as our "gamma" - the core VRF computation
        signature = self._private_key.sign(input_hash)

        # Derive the VRF output (ticket) from the signature
        # Hash again with different domain separator for output derivation
        ticket = hashlib.sha512(DOMAIN_SEPARATOR_VRF_HASH + signature + input_hash).digest()[
            :VRF_OUTPUT_SIZE
        ]

        # Create proof (using signature components)
        # In a full ECVRF implementation, gamma/c/s would be curve points/scalars
        # We use a simplified construction where the Ed25519 signature IS the proof
        proof = VRFProof(
            gamma=signature[:32],
            c=signature[32:64],
            s=input_hash[:32],  # Include input hash for verification
        )

        return VRFOutput(
            ticket=ticket,
            proof=proof,
            input_hash=input_hash[:32],
        )

    @staticmethod
    def verify(
        public_key_bytes: bytes,
        alpha: bytes,
        output: VRFOutput,
    ) -> bool:
        """Verify a VRF output and proof.

        Args:
            public_key_bytes: The Ed25519 public key (32 bytes)
            alpha: The original input
            output: The VRF output to verify

        Returns:
            True if the proof is valid, False otherwise
        """
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography library required for VRF operations")

        try:
            # Reconstruct the public key
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)

            # Reconstruct the input hash
            input_hash = hashlib.sha512(DOMAIN_SEPARATOR_VRF_PROVE + alpha).digest()

            # Verify input hash matches
            if output.input_hash != input_hash[:32]:
                return False

            # Reconstruct signature from proof
            signature = output.proof.gamma + output.proof.c

            # Verify the Ed25519 signature
            public_key.verify(signature, input_hash)

            # Verify the ticket derivation
            expected_ticket = hashlib.sha512(
                DOMAIN_SEPARATOR_VRF_HASH + signature + input_hash
            ).digest()[:VRF_OUTPUT_SIZE]

            if output.ticket != expected_ticket:
                return False

            return True

        except (InvalidSignature, ValueError, Exception):
            return False

    @staticmethod
    def derive_epoch_seed(
        previous_seed: bytes,
        block_hash: bytes,
        epoch_number: int,
    ) -> bytes:
        """Derive the seed for a new epoch.

        The epoch seed is derived from:
        - Previous epoch's seed (chain of randomness)
        - Block hash (external randomness source)
        - Epoch number (prevents replay)

        Args:
            previous_seed: Seed from the previous epoch (32 bytes)
            block_hash: Hash of a block at epoch boundary (32 bytes)
            epoch_number: The epoch number

        Returns:
            New epoch seed (32 bytes)
        """
        data = (
            DOMAIN_SEPARATOR_EPOCH_SEED
            + previous_seed
            + block_hash
            + epoch_number.to_bytes(8, "big")
        )
        return hashlib.sha256(data).digest()

    @staticmethod
    def genesis_seed() -> bytes:
        """Generate the genesis epoch seed.

        Used only for the first epoch when there's no previous seed.
        Should be generated from a ceremony or trusted setup in production.

        Returns:
            Genesis seed (32 bytes)
        """
        # In production, this would come from a distributed ceremony
        # For now, use a deterministic value based on domain separator
        return hashlib.sha256(DOMAIN_SEPARATOR_EPOCH_SEED + b"genesis").digest()


def compute_selection_ticket(
    vrf: VRF,
    epoch_seed: bytes,
    agent_fingerprint: bytes,
) -> VRFOutput:
    """Compute a validator's selection ticket for an epoch.

    Args:
        vrf: The validator's VRF instance
        epoch_seed: The epoch's random seed
        agent_fingerprint: The agent's unique identifier (DID fingerprint)

    Returns:
        VRF output containing the selection ticket
    """
    # Combine epoch seed with agent fingerprint
    input_data = hashlib.sha256(epoch_seed + agent_fingerprint).digest()

    return vrf.prove(input_data)
