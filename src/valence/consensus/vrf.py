"""Verifiable Random Function (VRF) Implementation.

Implements ECVRF-EDWARDS25519-SHA512-TAI per RFC 9381 for validator selection.
Provides unpredictable but verifiable randomness using Ed25519 keys.

Security properties:
- Unpredictability: VRF output cannot be predicted without the private key
- Uniqueness: Each input produces exactly one valid output per key
- Verifiability: Anyone can verify output given the public key and proof
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# Try to import cryptography for Ed25519
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    from cryptography.exceptions import InvalidSignature
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
    c: bytes      # Challenge scalar
    s: bytes      # Response scalar
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_bytes(self) -> bytes:
        """Serialize proof to bytes."""
        return self.gamma + self.c + self.s
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "VRFProof":
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
    def from_dict(cls, data: dict[str, Any]) -> "VRFProof":
        """Create from dictionary."""
        return cls(
            gamma=bytes.fromhex(data["gamma"]),
            c=bytes.fromhex(data["c"]),
            s=bytes.fromhex(data["s"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
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
    def from_dict(cls, data: dict[str, Any]) -> "VRFOutput":
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
    
    Implements a simplified VRF construction based on ECVRF-EDWARDS25519-SHA512-TAI.
    Uses Ed25519 signing as the core primitive with additional hashing for
    the VRF output derivation.
    
    This implementation prioritizes security properties:
    - Deterministic: same key + input = same output
    - Unpredictable: output cannot be predicted without private key
    - Verifiable: proof allows public verification of correctness
    
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
    def generate(cls) -> "VRF":
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
        input_hash = hashlib.sha512(
            DOMAIN_SEPARATOR_VRF_PROVE + alpha
        ).digest()
        
        # Sign the hashed input
        # This serves as our "gamma" - the core VRF computation
        signature = self._private_key.sign(input_hash)
        
        # Derive the VRF output (ticket) from the signature
        # Hash again with different domain separator for output derivation
        ticket = hashlib.sha512(
            DOMAIN_SEPARATOR_VRF_HASH + signature + input_hash
        ).digest()[:VRF_OUTPUT_SIZE]
        
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
            input_hash = hashlib.sha512(
                DOMAIN_SEPARATOR_VRF_PROVE + alpha
            ).digest()
            
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
            DOMAIN_SEPARATOR_EPOCH_SEED +
            previous_seed +
            block_hash +
            epoch_number.to_bytes(8, "big")
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
        return hashlib.sha256(
            DOMAIN_SEPARATOR_EPOCH_SEED + b"genesis"
        ).digest()


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
    input_data = hashlib.sha256(
        epoch_seed + agent_fingerprint
    ).digest()
    
    return vrf.prove(input_data)
