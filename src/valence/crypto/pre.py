"""Proxy Re-Encryption (PRE) Abstraction Layer.

Provides a Python interface for proxy re-encryption as needed for secure
federation aggregation. PRE allows a proxy to transform ciphertext encrypted
for one party into ciphertext decryptable by another party, without ever
seeing the plaintext.

This module defines the abstract interface that can be backed by different
implementations:
- MockPREBackend: For testing (simulated crypto)
- Future: UmbralPREBackend (using nucypher/pyUmbral or similar)

PRE provides:
- Unidirectional re-encryption (A -> B, not B -> A from same rekey)
- Proxy blindness (proxy cannot decrypt)
- Non-interactivity (delegatee doesn't need to be online)

Use case in Valence Federation:
- Instance A has encrypted beliefs
- Instance A wants to share with Instance B
- Instance A generates rekey(A_private, B_public)
- Proxy (federation aggregator) re-encrypts without seeing plaintext
- Instance B decrypts with B_private

Example:
    >>> backend = MockPREBackend()
    >>> alice = backend.generate_keypair(b"alice")
    >>> bob = backend.generate_keypair(b"bob")
    >>> ciphertext = backend.encrypt(b"secret data", alice.public_key)
    >>> rekey = backend.generate_rekey(alice.private_key, bob.public_key)
    >>> re_encrypted = backend.re_encrypt(ciphertext, rekey)
    >>> plaintext = backend.decrypt(re_encrypted, bob.private_key)
    >>> assert plaintext == b"secret data"
"""

from __future__ import annotations

import hashlib
import secrets as crypto_secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# =============================================================================
# Exceptions
# =============================================================================


class PREError(Exception):
    """Base exception for PRE operations."""
    pass


class PREKeyError(PREError):
    """Raised when a key operation fails."""
    pass


class PREEncryptionError(PREError):
    """Raised when encryption fails."""
    pass


class PREDecryptionError(PREError):
    """Raised when decryption fails."""
    pass


class PREReEncryptionError(PREError):
    """Raised when re-encryption fails."""
    pass


class PREInvalidCiphertextError(PREError):
    """Raised when ciphertext is malformed or invalid."""
    pass


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PREPublicKey:
    """A public key for proxy re-encryption.
    
    Attributes:
        key_id: Identifier for the key (e.g., owner DID)
        key_bytes: The actual public key material
        created_at: When the key was generated
        metadata: Optional metadata (algorithm, params, etc.)
    """
    
    key_id: bytes
    key_bytes: bytes
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "key_id": self.key_id.hex(),
            "key_bytes": self.key_bytes.hex(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PREPublicKey":
        """Create from dictionary."""
        return cls(
            key_id=bytes.fromhex(data["key_id"]),
            key_bytes=bytes.fromhex(data["key_bytes"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {}),
        )
    
    def __hash__(self) -> int:
        return hash((self.key_id, self.key_bytes))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PREPublicKey):
            return False
        return self.key_id == other.key_id and self.key_bytes == other.key_bytes


@dataclass
class PREPrivateKey:
    """A private key for proxy re-encryption.
    
    Attributes:
        key_id: Identifier for the key (matches public key)
        key_bytes: The actual private key material (SENSITIVE!)
        created_at: When the key was generated
        metadata: Optional metadata
    """
    
    key_id: bytes
    key_bytes: bytes
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.
        
        WARNING: This includes private key material. Handle with care.
        """
        return {
            "key_id": self.key_id.hex(),
            "key_bytes": self.key_bytes.hex(),
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PREPrivateKey":
        """Create from dictionary."""
        return cls(
            key_id=bytes.fromhex(data["key_id"]),
            key_bytes=bytes.fromhex(data["key_bytes"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {}),
        )
    
    def __hash__(self) -> int:
        return hash((self.key_id, self.key_bytes))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PREPrivateKey):
            return False
        return self.key_id == other.key_id and self.key_bytes == other.key_bytes


@dataclass
class PREKeyPair:
    """A public/private key pair for proxy re-encryption.
    
    Attributes:
        public_key: The public key
        private_key: The private key (SENSITIVE!)
    """
    
    public_key: PREPublicKey
    private_key: PREPrivateKey
    
    @property
    def key_id(self) -> bytes:
        """Get the key pair identifier."""
        return self.public_key.key_id
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.
        
        WARNING: This includes private key material. Handle with care.
        """
        return {
            "public_key": self.public_key.to_dict(),
            "private_key": self.private_key.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PREKeyPair":
        """Create from dictionary."""
        return cls(
            public_key=PREPublicKey.from_dict(data["public_key"]),
            private_key=PREPrivateKey.from_dict(data["private_key"]),
        )


@dataclass
class ReEncryptionKey:
    """A re-encryption key for transforming ciphertexts.
    
    This key allows a proxy to transform ciphertext encrypted for the
    delegator into ciphertext decryptable by the delegatee, without
    the proxy being able to decrypt either.
    
    Attributes:
        rekey_id: Unique identifier for this rekey
        delegator_id: Key ID of the original encryption key owner
        delegatee_id: Key ID of the intended recipient
        key_bytes: The re-encryption key material
        created_at: When the rekey was generated
        expires_at: Optional expiration time
        metadata: Optional metadata (conditions, policy, etc.)
    """
    
    rekey_id: bytes
    delegator_id: bytes
    delegatee_id: bytes
    key_bytes: bytes
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the rekey has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rekey_id": self.rekey_id.hex(),
            "delegator_id": self.delegator_id.hex(),
            "delegatee_id": self.delegatee_id.hex(),
            "key_bytes": self.key_bytes.hex(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReEncryptionKey":
        """Create from dictionary."""
        return cls(
            rekey_id=bytes.fromhex(data["rekey_id"]),
            delegator_id=bytes.fromhex(data["delegator_id"]),
            delegatee_id=bytes.fromhex(data["delegatee_id"]),
            key_bytes=bytes.fromhex(data["key_bytes"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class PRECiphertext:
    """Ciphertext that can be re-encrypted.
    
    Attributes:
        ciphertext_id: Unique identifier for this ciphertext
        encrypted_data: The encrypted payload
        recipient_id: Key ID this is encrypted for
        is_reencrypted: Whether this has been re-encrypted
        original_recipient_id: If re-encrypted, the original recipient
        created_at: When the ciphertext was created
        metadata: Optional metadata (algorithm params, etc.)
    """
    
    ciphertext_id: bytes
    encrypted_data: bytes
    recipient_id: bytes
    is_reencrypted: bool = False
    original_recipient_id: bytes | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ciphertext_id": self.ciphertext_id.hex(),
            "encrypted_data": self.encrypted_data.hex(),
            "recipient_id": self.recipient_id.hex(),
            "is_reencrypted": self.is_reencrypted,
            "original_recipient_id": self.original_recipient_id.hex() if self.original_recipient_id else None,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PRECiphertext":
        """Create from dictionary."""
        return cls(
            ciphertext_id=bytes.fromhex(data["ciphertext_id"]),
            encrypted_data=bytes.fromhex(data["encrypted_data"]),
            recipient_id=bytes.fromhex(data["recipient_id"]),
            is_reencrypted=data.get("is_reencrypted", False),
            original_recipient_id=bytes.fromhex(data["original_recipient_id"]) if data.get("original_recipient_id") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Abstract Backend
# =============================================================================


class PREBackend(ABC):
    """Abstract interface for Proxy Re-Encryption operations.
    
    This defines the contract that any PRE implementation must fulfill.
    Implementations include:
    - MockPREBackend: For testing (simulated crypto)
    - Future: UmbralPREBackend (using nucypher/pyUmbral)
    
    Security Model:
    - Delegator has keypair (pk_A, sk_A)
    - Delegatee has keypair (pk_B, sk_B)
    - Delegator generates rekey = generate_rekey(sk_A, pk_B)
    - Proxy can transform ciphertext_A -> ciphertext_B using rekey
    - Proxy CANNOT decrypt ciphertext_A or ciphertext_B
    - Only delegatee can decrypt ciphertext_B with sk_B
    """
    
    @abstractmethod
    def generate_keypair(self, key_id: bytes) -> PREKeyPair:
        """Generate a new key pair for PRE.
        
        Args:
            key_id: Identifier for the key pair (e.g., user DID)
        
        Returns:
            New PREKeyPair
        """
        pass
    
    @abstractmethod
    def generate_rekey(
        self,
        delegator_private_key: PREPrivateKey,
        delegatee_public_key: PREPublicKey,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReEncryptionKey:
        """Generate a re-encryption key.
        
        This creates a key that allows a proxy to transform ciphertext
        encrypted for the delegator into ciphertext decryptable by
        the delegatee.
        
        Args:
            delegator_private_key: Private key of the delegating party
            delegatee_public_key: Public key of the receiving party
            expires_at: Optional expiration time for the rekey
            metadata: Optional metadata (policy conditions, etc.)
        
        Returns:
            ReEncryptionKey for the proxy to use
        
        Raises:
            PREKeyError: If key generation fails
        """
        pass
    
    @abstractmethod
    def encrypt(
        self,
        plaintext: bytes,
        recipient_public_key: PREPublicKey,
        metadata: dict[str, Any] | None = None,
    ) -> PRECiphertext:
        """Encrypt data for a recipient.
        
        Args:
            plaintext: Data to encrypt
            recipient_public_key: Public key of the intended recipient
            metadata: Optional metadata
        
        Returns:
            PRECiphertext that can be decrypted or re-encrypted
        
        Raises:
            PREEncryptionError: If encryption fails
        """
        pass
    
    @abstractmethod
    def decrypt(
        self,
        ciphertext: PRECiphertext,
        recipient_private_key: PREPrivateKey,
    ) -> bytes:
        """Decrypt ciphertext.
        
        Args:
            ciphertext: The ciphertext to decrypt
            recipient_private_key: Private key of the recipient
        
        Returns:
            Decrypted plaintext
        
        Raises:
            PREDecryptionError: If decryption fails
            PREKeyError: If wrong key is used
        """
        pass
    
    @abstractmethod
    def re_encrypt(
        self,
        ciphertext: PRECiphertext,
        rekey: ReEncryptionKey,
    ) -> PRECiphertext:
        """Re-encrypt ciphertext for a new recipient.
        
        This transforms ciphertext encrypted for the delegator into
        ciphertext decryptable by the delegatee, without decrypting.
        
        Args:
            ciphertext: Original ciphertext (encrypted for delegator)
            rekey: Re-encryption key (delegator -> delegatee)
        
        Returns:
            New PRECiphertext (encrypted for delegatee)
        
        Raises:
            PREReEncryptionError: If re-encryption fails
            PREInvalidCiphertextError: If ciphertext is invalid
        """
        pass
    
    @abstractmethod
    def verify_ciphertext(self, ciphertext: PRECiphertext) -> bool:
        """Verify ciphertext integrity.
        
        Args:
            ciphertext: The ciphertext to verify
        
        Returns:
            True if ciphertext is valid, False otherwise
        """
        pass


# =============================================================================
# Mock Implementation (for testing)
# =============================================================================


class MockPREBackend(PREBackend):
    """Mock PRE implementation for testing.
    
    This provides a functional but NOT cryptographically secure implementation.
    It simulates PRE behavior for testing federation flows without requiring
    actual cryptographic libraries.
    
    SECURITY WARNING:
    This implementation uses XOR and hashing to simulate encryption.
    It is NOT secure and should NEVER be used in production.
    Use UmbralPREBackend (when implemented) for real cryptographic security.
    
    The mock maintains internal state to enable realistic testing:
    - Tracks which keys exist
    - Validates key relationships in re-encryption
    - Simulates failure modes
    """
    
    def __init__(self) -> None:
        """Initialize the mock backend."""
        # Track keypairs by key_id for validation
        self._keypairs: dict[bytes, PREKeyPair] = {}
        # Track rekeys for validation
        self._rekeys: dict[bytes, ReEncryptionKey] = {}
        # Track ciphertexts for validation
        self._ciphertexts: dict[bytes, tuple[bytes, bytes]] = {}  # id -> (plaintext, recipient_id)
    
    def generate_keypair(self, key_id: bytes) -> PREKeyPair:
        """Generate a mock key pair."""
        # Generate random key material
        public_bytes = crypto_secrets.token_bytes(32)
        private_bytes = crypto_secrets.token_bytes(32)
        
        now = datetime.now()
        
        public_key = PREPublicKey(
            key_id=key_id,
            key_bytes=public_bytes,
            created_at=now,
            metadata={"algorithm": "mock-pre", "version": "1.0"},
        )
        
        private_key = PREPrivateKey(
            key_id=key_id,
            key_bytes=private_bytes,
            created_at=now,
            metadata={"algorithm": "mock-pre", "version": "1.0"},
        )
        
        keypair = PREKeyPair(public_key=public_key, private_key=private_key)
        self._keypairs[key_id] = keypair
        
        return keypair
    
    def generate_rekey(
        self,
        delegator_private_key: PREPrivateKey,
        delegatee_public_key: PREPublicKey,
        expires_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReEncryptionKey:
        """Generate a mock re-encryption key."""
        # In real PRE, this would be computed from the keys
        # For mock, we just generate random bytes but track the relationship
        rekey_bytes = crypto_secrets.token_bytes(32)
        rekey_id = crypto_secrets.token_bytes(16)
        
        rekey = ReEncryptionKey(
            rekey_id=rekey_id,
            delegator_id=delegator_private_key.key_id,
            delegatee_id=delegatee_public_key.key_id,
            key_bytes=rekey_bytes,
            created_at=datetime.now(),
            expires_at=expires_at,
            metadata=metadata or {},
        )
        
        self._rekeys[rekey_id] = rekey
        
        return rekey
    
    def encrypt(
        self,
        plaintext: bytes,
        recipient_public_key: PREPublicKey,
        metadata: dict[str, Any] | None = None,
    ) -> PRECiphertext:
        """Mock encrypt data.
        
        Uses XOR with a derived key (NOT SECURE - for testing only).
        """
        ciphertext_id = crypto_secrets.token_bytes(16)
        
        # Derive a "key" from the public key (NOT SECURE)
        derived_key = hashlib.sha256(recipient_public_key.key_bytes).digest()
        
        # XOR encrypt (NOT SECURE - for testing only)
        encrypted = self._xor_bytes(plaintext, derived_key)
        
        ciphertext = PRECiphertext(
            ciphertext_id=ciphertext_id,
            encrypted_data=encrypted,
            recipient_id=recipient_public_key.key_id,
            is_reencrypted=False,
            original_recipient_id=None,
            created_at=datetime.now(),
            metadata=metadata or {},
        )
        
        # Track for validation
        self._ciphertexts[ciphertext_id] = (plaintext, recipient_public_key.key_id)
        
        return ciphertext
    
    def decrypt(
        self,
        ciphertext: PRECiphertext,
        recipient_private_key: PREPrivateKey,
    ) -> bytes:
        """Mock decrypt ciphertext."""
        # Validate recipient matches
        if ciphertext.recipient_id != recipient_private_key.key_id:
            raise PREDecryptionError(
                f"Key mismatch: ciphertext for {ciphertext.recipient_id.hex()}, "
                f"but got key {recipient_private_key.key_id.hex()}"
            )
        
        # Look up original plaintext (mock shortcut)
        if ciphertext.ciphertext_id in self._ciphertexts:
            plaintext, _ = self._ciphertexts[ciphertext.ciphertext_id]
            return plaintext
        
        # Fallback: try to decrypt with XOR
        if recipient_private_key.key_id not in self._keypairs:
            raise PREDecryptionError("Unknown private key")
        
        keypair = self._keypairs[recipient_private_key.key_id]
        derived_key = hashlib.sha256(keypair.public_key.key_bytes).digest()
        
        try:
            return self._xor_bytes(ciphertext.encrypted_data, derived_key)
        except Exception as e:
            raise PREDecryptionError(f"Decryption failed: {e}") from e
    
    def re_encrypt(
        self,
        ciphertext: PRECiphertext,
        rekey: ReEncryptionKey,
    ) -> PRECiphertext:
        """Mock re-encrypt ciphertext for new recipient."""
        # Validate rekey matches ciphertext recipient
        if ciphertext.recipient_id != rekey.delegator_id:
            raise PREReEncryptionError(
                f"Rekey delegator {rekey.delegator_id.hex()} doesn't match "
                f"ciphertext recipient {ciphertext.recipient_id.hex()}"
            )
        
        # Check expiration
        if rekey.is_expired:
            raise PREReEncryptionError("Re-encryption key has expired")
        
        # In real PRE, this would transform the ciphertext cryptographically
        # For mock, we create a new ciphertext pointing to same plaintext
        
        new_ciphertext_id = crypto_secrets.token_bytes(16)
        
        # Get delegatee public key for "encryption"
        if rekey.delegatee_id in self._keypairs:
            delegatee_keypair = self._keypairs[rekey.delegatee_id]
            derived_key = hashlib.sha256(delegatee_keypair.public_key.key_bytes).digest()
        else:
            # Use rekey bytes as fallback
            derived_key = hashlib.sha256(rekey.key_bytes).digest()
        
        # Get original plaintext and re-encrypt
        if ciphertext.ciphertext_id in self._ciphertexts:
            plaintext, _ = self._ciphertexts[ciphertext.ciphertext_id]
            encrypted = self._xor_bytes(plaintext, derived_key)
        else:
            # Just transform the bytes (won't decrypt correctly, but simulates)
            encrypted = self._xor_bytes(ciphertext.encrypted_data, rekey.key_bytes)
        
        new_ciphertext = PRECiphertext(
            ciphertext_id=new_ciphertext_id,
            encrypted_data=encrypted,
            recipient_id=rekey.delegatee_id,
            is_reencrypted=True,
            original_recipient_id=ciphertext.recipient_id,
            created_at=datetime.now(),
            metadata={
                **ciphertext.metadata,
                "reencrypted_from": ciphertext.ciphertext_id.hex(),
                "rekey_id": rekey.rekey_id.hex(),
            },
        )
        
        # Track the new ciphertext with same plaintext
        if ciphertext.ciphertext_id in self._ciphertexts:
            plaintext, _ = self._ciphertexts[ciphertext.ciphertext_id]
            self._ciphertexts[new_ciphertext_id] = (plaintext, rekey.delegatee_id)
        
        return new_ciphertext
    
    def verify_ciphertext(self, ciphertext: PRECiphertext) -> bool:
        """Verify ciphertext is tracked (mock validation)."""
        return ciphertext.ciphertext_id in self._ciphertexts
    
    @staticmethod
    def _xor_bytes(data: bytes, key: bytes) -> bytes:
        """XOR data with key (repeated if necessary).
        
        NOT SECURE - for testing only.
        """
        result = bytearray(len(data))
        for i, byte in enumerate(data):
            result[i] = byte ^ key[i % len(key)]
        return bytes(result)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_mock_backend() -> MockPREBackend:
    """Create a mock PRE backend for testing."""
    return MockPREBackend()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "PREError",
    "PREKeyError",
    "PREEncryptionError",
    "PREDecryptionError",
    "PREReEncryptionError",
    "PREInvalidCiphertextError",
    # Data classes
    "PREPublicKey",
    "PREPrivateKey",
    "PREKeyPair",
    "ReEncryptionKey",
    "PRECiphertext",
    # Abstract backend
    "PREBackend",
    # Implementations
    "MockPREBackend",
    # Convenience
    "create_mock_backend",
]
