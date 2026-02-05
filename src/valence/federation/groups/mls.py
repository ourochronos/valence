"""MLS Protocol Primitives for Valence Federation.

Implements MLS (Messaging Layer Security) protocol primitives for
group key management in federated knowledge sharing.

Key concepts:
- KeyPackage: Pre-key bundle for adding members without online interaction
- WelcomeMessage: Message allowing new members to join with group secrets
- CommitMessage: Epoch transition notification for existing members
- EpochSecrets: Cryptographic secrets for a specific epoch
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization


# =============================================================================
# CONSTANTS
# =============================================================================

# Protocol version
MLS_PROTOCOL_VERSION = "1.0"
DEFAULT_CIPHER_SUITE = "X25519-AES256GCM-SHA256"
MAX_GROUP_SIZE = 1000
EPOCH_SECRET_SIZE = 32
ENCRYPTION_KEY_SIZE = 32

# Key derivation contexts
KDF_INFO_EPOCH_SECRET = b"valence-mls-epoch-secret"
KDF_INFO_ENCRYPTION_KEY = b"valence-mls-encryption-key"
KDF_INFO_WELCOME_KEY = b"valence-mls-welcome-key"
KDF_INFO_MEMBER_SECRET = b"valence-mls-member-secret"

# Key sizes
AES_KEY_SIZE = 32  # 256 bits
NONCE_SIZE = 12    # 96 bits for GCM

# Epoch history limits
MAX_EPOCH_HISTORY = 100  # Maximum number of epochs to retain for recovery


# =============================================================================
# EXCEPTIONS
# =============================================================================


class MLSError(Exception):
    """Base exception for MLS-related errors."""
    pass


class GroupNotFoundError(MLSError):
    """Group does not exist."""
    pass


class MemberExistsError(MLSError):
    """Member is already in the group."""
    pass


class MemberNotFoundError(MLSError):
    """Member is not in the group."""
    pass


class InvalidKeyPackageError(MLSError):
    """KeyPackage is invalid or expired."""
    pass


class GroupFullError(MLSError):
    """Group has reached maximum capacity."""
    pass


class PermissionDeniedError(MLSError):
    """Member does not have required permission."""
    pass


class EpochMismatchError(MLSError):
    """Epoch does not match expected value."""
    pass


# =============================================================================
# KEY PACKAGE
# =============================================================================


@dataclass
class KeyPackage:
    """Pre-key bundle for adding a member to a group.
    
    Contains public keys needed to establish shared secrets with
    the new member. Must be uploaded by the member beforehand.
    
    In MLS terms, this is a LeafNode with init keys.
    """
    
    id: UUID
    member_did: str
    
    # HPKE keys for encryption to this member
    init_public_key: bytes      # X25519 public key for key exchange
    
    # Signing key for authenticating the member
    signature_public_key: bytes  # Ed25519 public key
    
    # Credentials
    credential_type: str = "basic"
    
    # Validity
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    
    # Signature over the package
    signature: bytes = b""
    
    def is_valid(self) -> bool:
        """Check if the KeyPackage is still valid."""
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        return True
    
    @classmethod
    def generate(
        cls,
        member_did: str,
        signing_private_key: bytes,
        expires_in: timedelta | None = None,
    ) -> tuple["KeyPackage", bytes]:
        """Generate a new KeyPackage for a member.
        
        Args:
            member_did: The member's DID
            signing_private_key: Ed25519 private key for signing
            expires_in: How long the package is valid
            
        Returns:
            Tuple of (KeyPackage, init_private_key)
        """
        # Generate X25519 keypair for init key
        init_private = X25519PrivateKey.generate()
        init_public = init_private.public_key()
        
        init_private_bytes = init_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        init_public_bytes = init_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        
        # Get signing public key
        signing_private = Ed25519PrivateKey.from_private_bytes(signing_private_key)
        signature_public_bytes = signing_private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        
        now = datetime.now()
        expires_at = now + expires_in if expires_in else None
        
        package = cls(
            id=uuid4(),
            member_did=member_did,
            init_public_key=init_public_bytes,
            signature_public_key=signature_public_bytes,
            created_at=now,
            expires_at=expires_at,
        )
        
        # Sign the package
        package.signature = package._sign(signing_private_key)
        
        return package, init_private_bytes
    
    def _sign(self, private_key: bytes) -> bytes:
        """Sign the KeyPackage content."""
        content = self._signable_content()
        signing_key = Ed25519PrivateKey.from_private_bytes(private_key)
        return signing_key.sign(content)
    
    def verify_signature(self) -> bool:
        """Verify the KeyPackage signature."""
        try:
            content = self._signable_content()
            public_key = Ed25519PublicKey.from_public_bytes(self.signature_public_key)
            public_key.verify(self.signature, content)
            return True
        except Exception:
            return False
    
    def _signable_content(self) -> bytes:
        """Get the content to be signed."""
        return json.dumps({
            "id": str(self.id),
            "member_did": self.member_did,
            "init_public_key": base64.b64encode(self.init_public_key).decode(),
            "signature_public_key": base64.b64encode(self.signature_public_key).decode(),
            "credential_type": self.credential_type,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }, sort_keys=True).encode()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "member_did": self.member_did,
            "init_public_key": base64.b64encode(self.init_public_key).decode(),
            "signature_public_key": base64.b64encode(self.signature_public_key).decode(),
            "credential_type": self.credential_type,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "signature": base64.b64encode(self.signature).decode(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KeyPackage":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            member_did=data["member_did"],
            init_public_key=base64.b64decode(data["init_public_key"]),
            signature_public_key=base64.b64decode(data["signature_public_key"]),
            credential_type=data.get("credential_type", "basic"),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            signature=base64.b64decode(data["signature"]),
        )


# =============================================================================
# EPOCH SECRETS
# =============================================================================


@dataclass
class EpochSecrets:
    """Cryptographic secrets for a specific epoch.
    
    Each epoch has its own set of derived keys to provide
    forward secrecy and post-compromise security.
    """
    
    epoch: int
    
    # Core secret for this epoch
    epoch_secret: bytes = b""
    
    # Derived keys
    encryption_key: bytes = b""   # For encrypting group content
    
    # Tree secret for deriving member-specific secrets
    tree_secret: bytes = b""
    
    # Confirmation key for verifying commits
    confirmation_key: bytes = b""
    
    @classmethod
    def derive(cls, epoch: int, init_secret: bytes, commit_secret: bytes | None = None) -> "EpochSecrets":
        """Derive epoch secrets from init secret and optional commit secret.
        
        Args:
            epoch: The epoch number
            init_secret: Random secret (for first epoch) or previous epoch's tree secret
            commit_secret: Fresh randomness from the commit (if any)
        """
        # Combine secrets if commit_secret provided
        if commit_secret:
            combined = init_secret + commit_secret
        else:
            combined = init_secret
        
        # Derive epoch secret
        epoch_secret = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=epoch.to_bytes(8, 'big'),
            info=KDF_INFO_EPOCH_SECRET,
        ).derive(combined)
        
        # Derive encryption key
        encryption_key = HKDF(
            algorithm=hashes.SHA256(),
            length=AES_KEY_SIZE,
            salt=None,
            info=KDF_INFO_ENCRYPTION_KEY,
        ).derive(epoch_secret)
        
        # Derive tree secret (for member secrets and next epoch)
        tree_secret = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"valence-mls-tree-secret",
        ).derive(epoch_secret)
        
        # Derive confirmation key
        confirmation_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"valence-mls-confirmation-key",
        ).derive(epoch_secret)
        
        return cls(
            epoch=epoch,
            epoch_secret=epoch_secret,
            encryption_key=encryption_key,
            tree_secret=tree_secret,
            confirmation_key=confirmation_key,
        )
    
    def derive_member_secret(self, leaf_index: int) -> bytes:
        """Derive a member-specific secret from the tree secret."""
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=leaf_index.to_bytes(4, 'big'),
            info=KDF_INFO_MEMBER_SECRET,
        ).derive(self.tree_secret)


# =============================================================================
# WELCOME MESSAGE
# =============================================================================


@dataclass
class WelcomeMessage:
    """Welcome message for a new group member.
    
    Contains encrypted group secrets and state, allowing the
    new member to decrypt group content from their join point.
    """
    
    id: UUID
    group_id: UUID
    
    # Who is being welcomed
    new_member_did: str
    
    # Encrypted group secrets (encrypted to new member's init key)
    encrypted_group_secrets: bytes = b""
    encrypted_group_secrets_nonce: bytes = b""
    
    # Ephemeral public key used for encryption
    ephemeral_public_key: bytes = b""
    
    # Current epoch info
    epoch: int = 0
    
    # Group info (encrypted)
    encrypted_group_info: bytes = b""
    encrypted_group_info_nonce: bytes = b""
    
    # Member roster at join time (encrypted)
    encrypted_roster: bytes = b""
    encrypted_roster_nonce: bytes = b""
    
    # Signature from adder
    adder_did: str = ""
    signature: bytes = b""
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None
    
    def encrypt_secrets(
        self,
        group_secrets: bytes,
        group_info: dict,
        roster: list[dict],
        recipient_init_key: bytes,
    ) -> None:
        """Encrypt group secrets for the new member.
        
        Uses X25519 key exchange + AES-GCM.
        """
        # Generate ephemeral keypair
        ephemeral_private = X25519PrivateKey.generate()
        ephemeral_public = ephemeral_private.public_key()
        
        self.ephemeral_public_key = ephemeral_public.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        
        # Derive shared secret
        recipient_key = X25519PublicKey.from_public_bytes(recipient_init_key)
        shared_secret = ephemeral_private.exchange(recipient_key)
        
        # Derive welcome key
        welcome_key = HKDF(
            algorithm=hashes.SHA256(),
            length=AES_KEY_SIZE,
            salt=None,
            info=KDF_INFO_WELCOME_KEY,
        ).derive(shared_secret)
        
        aesgcm = AESGCM(welcome_key)
        
        # Encrypt group secrets
        nonce1 = os.urandom(NONCE_SIZE)
        self.encrypted_group_secrets = aesgcm.encrypt(nonce1, group_secrets, None)
        self.encrypted_group_secrets_nonce = nonce1
        
        # Encrypt group info
        nonce2 = os.urandom(NONCE_SIZE)
        group_info_bytes = json.dumps(group_info, sort_keys=True).encode()
        self.encrypted_group_info = aesgcm.encrypt(nonce2, group_info_bytes, None)
        self.encrypted_group_info_nonce = nonce2
        
        # Encrypt roster
        nonce3 = os.urandom(NONCE_SIZE)
        roster_bytes = json.dumps(roster, sort_keys=True).encode()
        self.encrypted_roster = aesgcm.encrypt(nonce3, roster_bytes, None)
        self.encrypted_roster_nonce = nonce3
    
    def decrypt_secrets(self, init_private_key: bytes) -> tuple[bytes, dict, list]:
        """Decrypt the welcome message using the member's init private key.
        
        Returns:
            Tuple of (group_secrets, group_info, roster)
        """
        # Load keys
        private_key = X25519PrivateKey.from_private_bytes(init_private_key)
        ephemeral_public = X25519PublicKey.from_public_bytes(self.ephemeral_public_key)
        
        # Derive shared secret
        shared_secret = private_key.exchange(ephemeral_public)
        
        # Derive welcome key
        welcome_key = HKDF(
            algorithm=hashes.SHA256(),
            length=AES_KEY_SIZE,
            salt=None,
            info=KDF_INFO_WELCOME_KEY,
        ).derive(shared_secret)
        
        aesgcm = AESGCM(welcome_key)
        
        # Decrypt secrets
        group_secrets = aesgcm.decrypt(
            self.encrypted_group_secrets_nonce,
            self.encrypted_group_secrets,
            None,
        )
        
        group_info_bytes = aesgcm.decrypt(
            self.encrypted_group_info_nonce,
            self.encrypted_group_info,
            None,
        )
        group_info = json.loads(group_info_bytes.decode())
        
        roster_bytes = aesgcm.decrypt(
            self.encrypted_roster_nonce,
            self.encrypted_roster,
            None,
        )
        roster = json.loads(roster_bytes.decode())
        
        return group_secrets, group_info, roster
    
    def sign(self, adder_private_key: bytes) -> None:
        """Sign the welcome message."""
        content = self._signable_content()
        signing_key = Ed25519PrivateKey.from_private_bytes(adder_private_key)
        self.signature = signing_key.sign(content)
    
    def verify_signature(self, adder_public_key: bytes) -> bool:
        """Verify the adder's signature."""
        try:
            content = self._signable_content()
            public_key = Ed25519PublicKey.from_public_bytes(adder_public_key)
            public_key.verify(self.signature, content)
            return True
        except Exception:
            return False
    
    def _signable_content(self) -> bytes:
        """Get content to sign."""
        return json.dumps({
            "id": str(self.id),
            "group_id": str(self.group_id),
            "new_member_did": self.new_member_did,
            "epoch": self.epoch,
            "adder_did": self.adder_did,
            "created_at": self.created_at.isoformat(),
        }, sort_keys=True).encode()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "group_id": str(self.group_id),
            "new_member_did": self.new_member_did,
            "encrypted_group_secrets": base64.b64encode(self.encrypted_group_secrets).decode(),
            "encrypted_group_secrets_nonce": base64.b64encode(self.encrypted_group_secrets_nonce).decode(),
            "ephemeral_public_key": base64.b64encode(self.ephemeral_public_key).decode(),
            "epoch": self.epoch,
            "encrypted_group_info": base64.b64encode(self.encrypted_group_info).decode(),
            "encrypted_group_info_nonce": base64.b64encode(self.encrypted_group_info_nonce).decode(),
            "encrypted_roster": base64.b64encode(self.encrypted_roster).decode(),
            "encrypted_roster_nonce": base64.b64encode(self.encrypted_roster_nonce).decode(),
            "adder_did": self.adder_did,
            "signature": base64.b64encode(self.signature).decode(),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WelcomeMessage":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            group_id=UUID(data["group_id"]),
            new_member_did=data["new_member_did"],
            encrypted_group_secrets=base64.b64decode(data["encrypted_group_secrets"]),
            encrypted_group_secrets_nonce=base64.b64decode(data["encrypted_group_secrets_nonce"]),
            ephemeral_public_key=base64.b64decode(data["ephemeral_public_key"]),
            epoch=data["epoch"],
            encrypted_group_info=base64.b64decode(data["encrypted_group_info"]),
            encrypted_group_info_nonce=base64.b64decode(data["encrypted_group_info_nonce"]),
            encrypted_roster=base64.b64decode(data["encrypted_roster"]),
            encrypted_roster_nonce=base64.b64decode(data["encrypted_roster_nonce"]),
            adder_did=data["adder_did"],
            signature=base64.b64decode(data["signature"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
        )


# =============================================================================
# COMMIT MESSAGE
# =============================================================================


@dataclass
class CommitMessage:
    """A commit that changes group state (epoch transition).
    
    Commits are sent to existing members to notify them of
    membership changes and provide new epoch secrets.
    """
    
    id: UUID
    group_id: UUID
    
    # Epoch transition
    from_epoch: int
    to_epoch: int
    
    # What changed
    proposals: list[dict] = field(default_factory=list)
    
    # Commit secret (encrypted per member)
    # Maps member DID -> encrypted commit secret
    encrypted_commit_secrets: dict[str, dict] = field(default_factory=dict)
    
    # Committer info
    committer_did: str = ""
    signature: bytes = b""
    
    # Confirmation MAC
    confirmation_tag: bytes = b""
    
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "group_id": str(self.group_id),
            "from_epoch": self.from_epoch,
            "to_epoch": self.to_epoch,
            "proposals": self.proposals,
            "encrypted_commit_secrets": self.encrypted_commit_secrets,
            "committer_did": self.committer_did,
            "signature": base64.b64encode(self.signature).decode(),
            "confirmation_tag": base64.b64encode(self.confirmation_tag).decode(),
            "created_at": self.created_at.isoformat(),
        }


# =============================================================================
# GROUP ENCRYPTION UTILITIES
# =============================================================================


def encrypt_group_content(
    content: bytes,
    epoch_secrets: EpochSecrets,
    associated_data: bytes | None = None,
) -> tuple[bytes, bytes]:
    """Encrypt content for the group using current epoch key.
    
    Args:
        content: Plaintext to encrypt
        epoch_secrets: Current epoch's secrets
        associated_data: Optional AAD for authentication
        
    Returns:
        Tuple of (ciphertext, nonce)
    """
    nonce = os.urandom(NONCE_SIZE)
    aesgcm = AESGCM(epoch_secrets.encryption_key)
    ciphertext = aesgcm.encrypt(nonce, content, associated_data)
    return ciphertext, nonce


def decrypt_group_content(
    ciphertext: bytes,
    nonce: bytes,
    epoch_secrets: EpochSecrets,
    associated_data: bytes | None = None,
) -> bytes:
    """Decrypt group content using epoch key.
    
    Args:
        ciphertext: Encrypted content
        nonce: Nonce used for encryption
        epoch_secrets: Epoch secrets (must match encryption epoch)
        associated_data: Optional AAD for authentication
        
    Returns:
        Decrypted plaintext
    """
    aesgcm = AESGCM(epoch_secrets.encryption_key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data)
