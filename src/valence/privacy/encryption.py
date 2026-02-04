"""Encryption envelope for Valence belief sharing.

Implements envelope encryption using AES-256-GCM for content
and X25519 key exchange for DEK encryption.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import os
import base64

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization


@dataclass
class EncryptionEnvelope:
    """Envelope encryption for belief content.
    
    Uses hybrid encryption:
    - AES-256-GCM for content (symmetric, fast)
    - X25519 + HKDF for key exchange (asymmetric, secure)
    
    The Data Encryption Key (DEK) is encrypted with the recipient's
    public key, so only they can decrypt the content.
    """
    
    encrypted_content: bytes
    encrypted_dek: bytes  # DEK encrypted with recipient's public key
    nonce: bytes
    ephemeral_public_key: bytes  # Sender's ephemeral public key for X25519
    algorithm: str = "AES-256-GCM"
    recipient_key_id: Optional[str] = None
    
    @classmethod
    def encrypt(cls, content: bytes, recipient_public_key: bytes) -> "EncryptionEnvelope":
        """Encrypt content for a specific recipient.
        
        Args:
            content: The plaintext content to encrypt
            recipient_public_key: Recipient's X25519 public key (32 bytes)
            
        Returns:
            EncryptionEnvelope containing all data needed for decryption
        """
        # Generate ephemeral keypair for this message
        ephemeral_private = X25519PrivateKey.generate()
        ephemeral_public = ephemeral_private.public_key()
        
        # Load recipient's public key
        recipient_key = X25519PublicKey.from_public_bytes(recipient_public_key)
        
        # Perform X25519 key exchange
        shared_secret = ephemeral_private.exchange(recipient_key)
        
        # Derive DEK using HKDF
        dek = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"valence-belief-encryption",
        ).derive(shared_secret)
        
        # Generate nonce for AES-GCM
        nonce = os.urandom(12)
        
        # Encrypt content with DEK
        aesgcm = AESGCM(dek)
        encrypted_content = aesgcm.encrypt(nonce, content, None)
        
        # Get ephemeral public key bytes
        ephemeral_public_bytes = ephemeral_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        
        return cls(
            encrypted_content=encrypted_content,
            encrypted_dek=b"",  # DEK is derived, not transmitted
            nonce=nonce,
            ephemeral_public_key=ephemeral_public_bytes,
        )
    
    @classmethod
    def decrypt(cls, envelope: "EncryptionEnvelope", recipient_private_key: bytes) -> bytes:
        """Decrypt an envelope using the recipient's private key.
        
        Args:
            envelope: The EncryptionEnvelope to decrypt
            recipient_private_key: Recipient's X25519 private key (32 bytes)
            
        Returns:
            Decrypted content bytes
        """
        # Load keys
        private_key = X25519PrivateKey.from_private_bytes(recipient_private_key)
        ephemeral_public = X25519PublicKey.from_public_bytes(envelope.ephemeral_public_key)
        
        # Perform X25519 key exchange (reverse direction)
        shared_secret = private_key.exchange(ephemeral_public)
        
        # Derive same DEK using HKDF
        dek = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"valence-belief-encryption",
        ).derive(shared_secret)
        
        # Decrypt content
        aesgcm = AESGCM(dek)
        return aesgcm.decrypt(envelope.nonce, envelope.encrypted_content, None)
    
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "encrypted_content": base64.b64encode(self.encrypted_content).decode(),
            "encrypted_dek": base64.b64encode(self.encrypted_dek).decode(),
            "nonce": base64.b64encode(self.nonce).decode(),
            "ephemeral_public_key": base64.b64encode(self.ephemeral_public_key).decode(),
            "algorithm": self.algorithm,
            "recipient_key_id": self.recipient_key_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EncryptionEnvelope":
        """Deserialize from dictionary."""
        return cls(
            encrypted_content=base64.b64decode(data["encrypted_content"]),
            encrypted_dek=base64.b64decode(data["encrypted_dek"]),
            nonce=base64.b64decode(data["nonce"]),
            ephemeral_public_key=base64.b64decode(data["ephemeral_public_key"]),
            algorithm=data.get("algorithm", "AES-256-GCM"),
            recipient_key_id=data.get("recipient_key_id"),
        )


def generate_keypair() -> Tuple[bytes, bytes]:
    """Generate an X25519 keypair for encryption.
    
    Returns:
        Tuple of (private_key_bytes, public_key_bytes)
    """
    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()
    
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    
    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    
    return private_bytes, public_bytes
