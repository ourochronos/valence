"""MLS-style Group Encryption for Valence Federation.

Implements a simplified MLS (Messaging Layer Security) protocol for
group key management in federated knowledge sharing.

Key concepts:
- KeyPackage: Pre-key bundle for adding members without online interaction
- Welcome: Message allowing new members to join with group secrets
- Epoch: Group state version, incremented on membership changes
- Tree-based key derivation: Scalable group secret management

Security properties:
- Forward secrecy: Past messages protected when members leave
- Post-compromise security: Group recovers from key compromise
- Epoch isolation: Each epoch has independent encryption keys
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
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

# Epoch history limits
MAX_EPOCH_HISTORY = 100  # Maximum number of epochs to retain for recovery


# =============================================================================
# ENUMS
# =============================================================================


class GroupRole(str, Enum):
    """Role of a member in a group."""
    ADMIN = "admin"        # Can add/remove members, change settings
    MEMBER = "member"      # Can read/write content
    OBSERVER = "observer"  # Read-only access


# Alias for backward compatibility
MemberRole = GroupRole


class ProposalType(str, Enum):
    """Type of group change proposal."""
    ADD = "add"
    REMOVE = "remove"
    UPDATE = "update"
    REINIT = "reinit"


class MemberStatus(str, Enum):
    """Status of a group member."""
    PENDING = "pending"    # Invited but not yet joined
    ACTIVE = "active"      # Fully joined and active
    REMOVED = "removed"    # Removed from group
    LEFT = "left"          # Voluntarily left


class GroupStatus(str, Enum):
    """Status of a group."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DISSOLVED = "dissolved"


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
# GROUP MEMBER
# =============================================================================


@dataclass
class GroupMember:
    """A member of a federated group."""
    
    did: str
    role: GroupRole = GroupRole.MEMBER
    status: MemberStatus = MemberStatus.ACTIVE
    
    # Key material (for active members)
    init_public_key: bytes = b""
    signature_public_key: bytes = b""
    
    # Epoch when member joined
    joined_at_epoch: int = 0
    
    # Timestamps
    joined_at: datetime = field(default_factory=datetime.now)
    removed_at: datetime | None = None
    
    # Leaf index in the ratchet tree (for key derivation)
    leaf_index: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "did": self.did,
            "role": self.role.value,
            "status": self.status.value,
            "init_public_key": base64.b64encode(self.init_public_key).decode() if self.init_public_key else "",
            "signature_public_key": base64.b64encode(self.signature_public_key).decode() if self.signature_public_key else "",
            "joined_at_epoch": self.joined_at_epoch,
            "joined_at": self.joined_at.isoformat(),
            "removed_at": self.removed_at.isoformat() if self.removed_at else None,
            "leaf_index": self.leaf_index,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupMember":
        """Create from dictionary."""
        return cls(
            did=data["did"],
            role=GroupRole(data.get("role", "member")),
            status=MemberStatus(data.get("status", "active")),
            init_public_key=base64.b64decode(data["init_public_key"]) if data.get("init_public_key") else b"",
            signature_public_key=base64.b64decode(data["signature_public_key"]) if data.get("signature_public_key") else b"",
            joined_at_epoch=data.get("joined_at_epoch", 0),
            joined_at=datetime.fromisoformat(data["joined_at"]) if data.get("joined_at") else datetime.now(),
            removed_at=datetime.fromisoformat(data["removed_at"]) if data.get("removed_at") else None,
            leaf_index=data.get("leaf_index", 0),
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
# GROUP STATE
# =============================================================================


@dataclass
class GroupState:
    """Complete state of a federated group.
    
    Maintains membership roster, epoch secrets, and group configuration.
    """
    
    id: UUID
    name: str
    
    # Current state
    epoch: int = 0
    status: GroupStatus = GroupStatus.ACTIVE
    
    # Membership
    members: dict[str, GroupMember] = field(default_factory=dict)  # DID -> GroupMember
    pending_members: dict[str, KeyPackage] = field(default_factory=dict)  # DID -> KeyPackage
    
    # Secrets (only populated locally for this member)
    current_secrets: EpochSecrets | None = None
    
    # Init secret for first epoch (random)
    init_secret: bytes = b""
    
    # Configuration
    config: dict[str, Any] = field(default_factory=dict)
    
    # Audit
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Next available leaf index
    next_leaf_index: int = 0
    
    def get_active_members(self) -> list[GroupMember]:
        """Get all active members."""
        return [m for m in self.members.values() if m.status == MemberStatus.ACTIVE]
    
    def get_member(self, did: str) -> GroupMember | None:
        """Get a member by DID."""
        return self.members.get(did)
    
    def is_admin(self, did: str) -> bool:
        """Check if a DID has admin rights."""
        member = self.members.get(did)
        return member is not None and member.role == GroupRole.ADMIN
    
    def member_count(self) -> int:
        """Get count of active members."""
        return len(self.get_active_members())
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (without secrets)."""
        return {
            "id": str(self.id),
            "name": self.name,
            "epoch": self.epoch,
            "status": self.status.value,
            "members": {did: m.to_dict() for did, m in self.members.items()},
            "config": self.config,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "updated_at": self.updated_at.isoformat(),
        }
    
    def get_group_info(self) -> dict:
        """Get group info for welcome messages."""
        return {
            "id": str(self.id),
            "name": self.name,
            "epoch": self.epoch,
            "config": self.config,
            "created_by": self.created_by,
        }
    
    def get_roster(self) -> list[dict]:
        """Get current roster for welcome messages."""
        return [m.to_dict() for m in self.get_active_members()]


# =============================================================================
# GROUP OPERATIONS
# =============================================================================


def create_group(
    name: str,
    creator_did: str,
    creator_key_package: KeyPackage,
    config: dict[str, Any] | None = None,
) -> GroupState:
    """Create a new federated group.
    
    Args:
        name: Human-readable group name
        creator_did: DID of the group creator
        creator_key_package: KeyPackage of the creator
        config: Optional group configuration
        
    Returns:
        Initialized GroupState
    """
    group_id = uuid4()
    now = datetime.now()
    
    # Generate random init secret for first epoch
    init_secret = secrets.token_bytes(32)
    
    # Create initial epoch secrets
    initial_secrets = EpochSecrets.derive(epoch=0, init_secret=init_secret)
    
    # Create creator as admin member
    creator = GroupMember(
        did=creator_did,
        role=GroupRole.ADMIN,
        status=MemberStatus.ACTIVE,
        init_public_key=creator_key_package.init_public_key,
        signature_public_key=creator_key_package.signature_public_key,
        joined_at_epoch=0,
        joined_at=now,
        leaf_index=0,
    )
    
    group = GroupState(
        id=group_id,
        name=name,
        epoch=0,
        status=GroupStatus.ACTIVE,
        members={creator_did: creator},
        current_secrets=initial_secrets,
        init_secret=init_secret,
        config=config or {},
        created_at=now,
        created_by=creator_did,
        updated_at=now,
        next_leaf_index=1,
    )
    
    return group


def add_member(
    group: GroupState,
    new_member_did: str,
    new_member_key_package: KeyPackage,
    adder_did: str,
    adder_signing_key: bytes,
    role: GroupRole = GroupRole.MEMBER,
) -> tuple[GroupState, WelcomeMessage, CommitMessage]:
    """Add a new member to the group.
    
    This is the primary member onboarding function. It:
    1. Validates the adder has permission
    2. Validates the new member's KeyPackage
    3. Creates the new member entry
    4. Generates a Welcome message for the new member
    5. Generates a Commit message for existing members
    6. Updates the group epoch
    
    Args:
        group: Current group state
        new_member_did: DID of the member to add
        new_member_key_package: KeyPackage uploaded by new member
        adder_did: DID of the member adding (must have permission)
        adder_signing_key: Ed25519 private key of the adder
        role: Role to assign to new member
        
    Returns:
        Tuple of (updated_group, welcome_message, commit_message)
        
    Raises:
        PermissionError: If adder cannot add members
        ValueError: If KeyPackage is invalid or member already exists
    """
    # Validate adder has permission
    adder = group.get_member(adder_did)
    if adder is None:
        raise PermissionError(f"Adder {adder_did} is not a group member")
    if adder.role not in (GroupRole.ADMIN,):
        raise PermissionError(f"Adder {adder_did} does not have permission to add members")
    if adder.status != MemberStatus.ACTIVE:
        raise PermissionError(f"Adder {adder_did} is not an active member")
    
    # Validate new member not already in group
    if new_member_did in group.members:
        existing = group.members[new_member_did]
        if existing.status == MemberStatus.ACTIVE:
            raise ValueError(f"Member {new_member_did} is already in the group")
    
    # Validate KeyPackage
    if not new_member_key_package.is_valid():
        raise ValueError("KeyPackage has expired")
    if not new_member_key_package.verify_signature():
        raise ValueError("KeyPackage signature is invalid")
    if new_member_key_package.member_did != new_member_did:
        raise ValueError("KeyPackage DID does not match new member DID")
    
    now = datetime.now()
    new_epoch = group.epoch + 1
    
    # Generate commit secret (fresh randomness for the new epoch)
    commit_secret = secrets.token_bytes(32)
    
    # Derive new epoch secrets
    if group.current_secrets:
        new_secrets = EpochSecrets.derive(
            epoch=new_epoch,
            init_secret=group.current_secrets.tree_secret,
            commit_secret=commit_secret,
        )
    else:
        # Fallback if no current secrets
        new_secrets = EpochSecrets.derive(
            epoch=new_epoch,
            init_secret=group.init_secret,
            commit_secret=commit_secret,
        )
    
    # Create new member entry
    new_member = GroupMember(
        did=new_member_did,
        role=role,
        status=MemberStatus.ACTIVE,
        init_public_key=new_member_key_package.init_public_key,
        signature_public_key=new_member_key_package.signature_public_key,
        joined_at_epoch=new_epoch,
        joined_at=now,
        leaf_index=group.next_leaf_index,
    )
    
    # Create Welcome message
    welcome = WelcomeMessage(
        id=uuid4(),
        group_id=group.id,
        new_member_did=new_member_did,
        epoch=new_epoch,
        adder_did=adder_did,
        created_at=now,
        expires_at=now + timedelta(days=7),
    )
    
    # Encrypt group secrets for new member
    # The new member gets the new epoch's tree secret
    welcome.encrypt_secrets(
        group_secrets=new_secrets.tree_secret,
        group_info=group.get_group_info(),
        roster=group.get_roster() + [new_member.to_dict()],
        recipient_init_key=new_member_key_package.init_public_key,
    )
    
    # Sign the welcome
    welcome.sign(adder_signing_key)
    
    # Create Commit message for existing members
    commit = CommitMessage(
        id=uuid4(),
        group_id=group.id,
        from_epoch=group.epoch,
        to_epoch=new_epoch,
        proposals=[{
            "type": "add",
            "member_did": new_member_did,
            "role": role.value,
            "added_by": adder_did,
        }],
        committer_did=adder_did,
        created_at=now,
    )
    
    # Encrypt commit secret for each existing active member
    for member in group.get_active_members():
        if member.init_public_key:
            # Generate ephemeral key for this member
            ephemeral_private = X25519PrivateKey.generate()
            ephemeral_public = ephemeral_private.public_key()
            
            # Derive shared secret
            member_key = X25519PublicKey.from_public_bytes(member.init_public_key)
            shared_secret = ephemeral_private.exchange(member_key)
            
            # Derive encryption key
            enc_key = HKDF(
                algorithm=hashes.SHA256(),
                length=AES_KEY_SIZE,
                salt=None,
                info=b"valence-mls-commit-key",
            ).derive(shared_secret)
            
            # Encrypt commit secret
            nonce = os.urandom(NONCE_SIZE)
            aesgcm = AESGCM(enc_key)
            encrypted = aesgcm.encrypt(nonce, commit_secret, None)
            
            commit.encrypted_commit_secrets[member.did] = {
                "encrypted": base64.b64encode(encrypted).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "ephemeral_public_key": base64.b64encode(
                    ephemeral_public.public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ).decode(),
            }
    
    # Compute confirmation tag
    import hmac
    confirmation_content = json.dumps({
        "from_epoch": commit.from_epoch,
        "to_epoch": commit.to_epoch,
        "proposals": commit.proposals,
        "committer_did": commit.committer_did,
    }, sort_keys=True).encode()
    commit.confirmation_tag = hmac.digest(
        new_secrets.confirmation_key,
        confirmation_content,
        'sha256',
    )
    
    # Sign commit
    signing_key = Ed25519PrivateKey.from_private_bytes(adder_signing_key)
    commit_content = json.dumps({
        "id": str(commit.id),
        "group_id": str(commit.group_id),
        "from_epoch": commit.from_epoch,
        "to_epoch": commit.to_epoch,
        "proposals": commit.proposals,
    }, sort_keys=True).encode()
    commit.signature = signing_key.sign(commit_content)
    
    # Update group state
    updated_group = GroupState(
        id=group.id,
        name=group.name,
        epoch=new_epoch,
        status=group.status,
        members={**group.members, new_member_did: new_member},
        pending_members=group.pending_members,
        current_secrets=new_secrets,
        init_secret=group.init_secret,
        config=group.config,
        created_at=group.created_at,
        created_by=group.created_by,
        updated_at=now,
        next_leaf_index=group.next_leaf_index + 1,
    )
    
    return updated_group, welcome, commit


def process_welcome(
    welcome: WelcomeMessage,
    init_private_key: bytes,
    adder_public_key: bytes | None = None,
) -> tuple[bytes, dict, list[GroupMember]]:
    """Process a welcome message to join a group.
    
    Args:
        welcome: The WelcomeMessage to process
        init_private_key: The new member's init private key
        adder_public_key: Optional public key to verify adder's signature
        
    Returns:
        Tuple of (epoch_tree_secret, group_info, roster)
    """
    # Verify signature if public key provided
    if adder_public_key and not welcome.verify_signature(adder_public_key):
        raise ValueError("Invalid welcome message signature")
    
    # Decrypt the welcome
    tree_secret, group_info, roster_dicts = welcome.decrypt_secrets(init_private_key)
    
    # Parse roster
    roster = [GroupMember.from_dict(m) for m in roster_dicts]
    
    return tree_secret, group_info, roster


def process_commit(
    commit: CommitMessage,
    member_did: str,
    member_init_private_key: bytes,
    current_secrets: EpochSecrets,
) -> EpochSecrets:
    """Process a commit message to update epoch secrets.
    
    Args:
        commit: The CommitMessage to process
        member_did: This member's DID
        member_init_private_key: This member's init private key
        current_secrets: Current epoch secrets
        
    Returns:
        New EpochSecrets for the commit's target epoch
    """
    # Check this commit is for the next epoch
    if commit.from_epoch != current_secrets.epoch:
        raise ValueError(f"Commit is for wrong epoch (expected {current_secrets.epoch})")
    
    # Get encrypted commit secret for this member
    if member_did not in commit.encrypted_commit_secrets:
        raise ValueError(f"No encrypted commit secret for {member_did}")
    
    encrypted_data = commit.encrypted_commit_secrets[member_did]
    
    # Decrypt commit secret
    private_key = X25519PrivateKey.from_private_bytes(member_init_private_key)
    ephemeral_public = X25519PublicKey.from_public_bytes(
        base64.b64decode(encrypted_data["ephemeral_public_key"])
    )
    
    shared_secret = private_key.exchange(ephemeral_public)
    
    enc_key = HKDF(
        algorithm=hashes.SHA256(),
        length=AES_KEY_SIZE,
        salt=None,
        info=b"valence-mls-commit-key",
    ).derive(shared_secret)
    
    aesgcm = AESGCM(enc_key)
    commit_secret = aesgcm.decrypt(
        base64.b64decode(encrypted_data["nonce"]),
        base64.b64decode(encrypted_data["encrypted"]),
        None,
    )
    
    # Derive new epoch secrets
    new_secrets = EpochSecrets.derive(
        epoch=commit.to_epoch,
        init_secret=current_secrets.tree_secret,
        commit_secret=commit_secret,
    )
    
    # Verify confirmation tag
    import hmac
    confirmation_content = json.dumps({
        "from_epoch": commit.from_epoch,
        "to_epoch": commit.to_epoch,
        "proposals": commit.proposals,
        "committer_did": commit.committer_did,
    }, sort_keys=True).encode()
    expected_tag = hmac.digest(
        new_secrets.confirmation_key,
        confirmation_content,
        'sha256',
    )
    
    if not hmac.compare_digest(expected_tag, commit.confirmation_tag):
        raise ValueError("Commit confirmation tag verification failed")
    
    return new_secrets


# =============================================================================
# GROUP ENCRYPTION
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


# =============================================================================
# MEMBER REMOVAL (OFFBOARDING) - Issue #75
# =============================================================================


@dataclass
class RemovalAuditEntry:
    """Audit log entry for member removal.
    
    Provides accountability for offboarding decisions.
    """
    id: UUID
    group_id: UUID
    removed_did: str
    remover_did: str
    reason: str | None
    epoch_before: int
    epoch_after: int
    timestamp: datetime = field(default_factory=datetime.now)
    signature: bytes = b""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "group_id": str(self.group_id),
            "removed_did": self.removed_did,
            "remover_did": self.remover_did,
            "reason": self.reason,
            "epoch_before": self.epoch_before,
            "epoch_after": self.epoch_after,
            "timestamp": self.timestamp.isoformat(),
            "signature": base64.b64encode(self.signature).decode() if self.signature else "",
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RemovalAuditEntry":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            group_id=UUID(data["group_id"]),
            removed_did=data["removed_did"],
            remover_did=data["remover_did"],
            reason=data.get("reason"),
            epoch_before=data["epoch_before"],
            epoch_after=data["epoch_after"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else datetime.now(),
            signature=base64.b64decode(data["signature"]) if data.get("signature") else b"",
        )


def remove_member(
    group: GroupState,
    member_did: str,
    remover_did: str,
    remover_signing_key: bytes,
    reason: str | None = None,
) -> tuple[GroupState, CommitMessage, RemovalAuditEntry]:
    """Remove a member from the group and rotate keys.
    
    This is the primary member offboarding function. It:
    1. Validates the remover has permission (admin or self-removal)
    2. Marks the member as removed
    3. Advances to a new epoch
    4. Derives new group keys (removed member cannot access)
    5. Creates a Commit message for remaining members
    6. Logs the removal in the audit trail
    
    Forward secrecy is preserved because:
    - The removed member is not included in the new commit secret distribution
    - New epoch secrets are derived from the new member set
    - Without the new epoch secret, the removed member cannot decrypt future messages
    
    Args:
        group: Current group state
        member_did: DID of the member to remove
        remover_did: DID of the member performing the removal (must be admin or self)
        remover_signing_key: Ed25519 private key of the remover
        reason: Optional reason for removal
        
    Returns:
        Tuple of (updated_group, commit_message, audit_entry)
        
    Raises:
        PermissionError: If remover cannot remove members
        ValueError: If member not found or is last admin
    """
    import hmac
    
    # Validate remover has permission
    remover = group.get_member(remover_did)
    if remover is None:
        raise PermissionError(f"Remover {remover_did} is not a group member")
    if remover.status != MemberStatus.ACTIVE:
        raise PermissionError(f"Remover {remover_did} is not an active member")
    
    # Can remove if: admin, or removing yourself
    is_self_removal = remover_did == member_did
    is_admin = remover.role == GroupRole.ADMIN
    
    if not is_admin and not is_self_removal:
        raise PermissionError(f"Remover {remover_did} does not have permission to remove members")
    
    # Validate member to remove exists and is active
    member = group.get_member(member_did)
    if member is None:
        raise ValueError(f"Member {member_did} not found in group")
    if member.status != MemberStatus.ACTIVE:
        raise ValueError(f"Member {member_did} is not active (status: {member.status.value})")
    
    # Prevent removing last admin (unless self-removal)
    if member.role == GroupRole.ADMIN and not is_self_removal:
        admin_count = sum(1 for m in group.get_active_members() if m.role == GroupRole.ADMIN)
        if admin_count <= 1:
            raise ValueError("Cannot remove the last admin")
    
    now = datetime.now()
    epoch_before = group.epoch
    new_epoch = group.epoch + 1
    
    # Mark member as removed
    updated_member = GroupMember(
        did=member.did,
        role=member.role,
        status=MemberStatus.REMOVED,
        init_public_key=member.init_public_key,
        signature_public_key=member.signature_public_key,
        joined_at_epoch=member.joined_at_epoch,
        joined_at=member.joined_at,
        removed_at=now,
        leaf_index=member.leaf_index,
    )
    
    # Build updated members dict
    updated_members = dict(group.members)
    updated_members[member_did] = updated_member
    
    # Generate new commit secret (fresh randomness for the new epoch)
    commit_secret = secrets.token_bytes(32)
    
    # Derive new epoch secrets
    # CRITICAL: This is where forward secrecy happens - 
    # the removed member won't receive the commit_secret
    if group.current_secrets:
        new_secrets = EpochSecrets.derive(
            epoch=new_epoch,
            init_secret=group.current_secrets.tree_secret,
            commit_secret=commit_secret,
        )
    else:
        new_secrets = EpochSecrets.derive(
            epoch=new_epoch,
            init_secret=group.init_secret,
            commit_secret=commit_secret,
        )
    
    # Create Commit message for REMAINING members only
    commit = CommitMessage(
        id=uuid4(),
        group_id=group.id,
        from_epoch=epoch_before,
        to_epoch=new_epoch,
        proposals=[{
            "type": "remove",
            "member_did": member_did,
            "removed_by": remover_did,
            "reason": reason,
        }],
        committer_did=remover_did,
        created_at=now,
    )
    
    # Encrypt commit secret for each REMAINING active member
    # CRITICAL: Do NOT include the removed member
    remaining_members = [
        m for m in group.get_active_members() 
        if m.did != member_did
    ]
    
    for remaining_member in remaining_members:
        if remaining_member.init_public_key:
            # Generate ephemeral key for this member
            ephemeral_private = X25519PrivateKey.generate()
            ephemeral_public = ephemeral_private.public_key()
            
            # Derive shared secret
            member_key = X25519PublicKey.from_public_bytes(remaining_member.init_public_key)
            shared_secret = ephemeral_private.exchange(member_key)
            
            # Derive encryption key
            enc_key = HKDF(
                algorithm=hashes.SHA256(),
                length=AES_KEY_SIZE,
                salt=None,
                info=b"valence-mls-commit-key",
            ).derive(shared_secret)
            
            # Encrypt commit secret
            nonce = os.urandom(NONCE_SIZE)
            aesgcm = AESGCM(enc_key)
            encrypted = aesgcm.encrypt(nonce, commit_secret, None)
            
            commit.encrypted_commit_secrets[remaining_member.did] = {
                "encrypted": base64.b64encode(encrypted).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "ephemeral_public_key": base64.b64encode(
                    ephemeral_public.public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ).decode(),
            }
    
    # Compute confirmation tag
    confirmation_content = json.dumps({
        "from_epoch": commit.from_epoch,
        "to_epoch": commit.to_epoch,
        "proposals": commit.proposals,
        "committer_did": commit.committer_did,
    }, sort_keys=True).encode()
    commit.confirmation_tag = hmac.digest(
        new_secrets.confirmation_key,
        confirmation_content,
        'sha256',
    )
    
    # Sign commit
    signing_key = Ed25519PrivateKey.from_private_bytes(remover_signing_key)
    commit_content = json.dumps({
        "id": str(commit.id),
        "group_id": str(commit.group_id),
        "from_epoch": commit.from_epoch,
        "to_epoch": commit.to_epoch,
        "proposals": commit.proposals,
    }, sort_keys=True).encode()
    commit.signature = signing_key.sign(commit_content)
    
    # Create audit entry
    audit_entry = RemovalAuditEntry(
        id=uuid4(),
        group_id=group.id,
        removed_did=member_did,
        remover_did=remover_did,
        reason=reason,
        epoch_before=epoch_before,
        epoch_after=new_epoch,
        timestamp=now,
    )
    
    # Sign audit entry
    audit_content = json.dumps({
        "id": str(audit_entry.id),
        "group_id": str(audit_entry.group_id),
        "removed_did": audit_entry.removed_did,
        "remover_did": audit_entry.remover_did,
        "reason": audit_entry.reason,
        "epoch_before": audit_entry.epoch_before,
        "epoch_after": audit_entry.epoch_after,
        "timestamp": audit_entry.timestamp.isoformat(),
    }, sort_keys=True).encode()
    audit_entry.signature = signing_key.sign(audit_content)
    
    # Update group state
    updated_group = GroupState(
        id=group.id,
        name=group.name,
        epoch=new_epoch,
        status=group.status,
        members=updated_members,
        pending_members=group.pending_members,
        current_secrets=new_secrets,
        init_secret=group.init_secret,
        config=group.config,
        created_at=group.created_at,
        created_by=group.created_by,
        updated_at=now,
        next_leaf_index=group.next_leaf_index,
    )
    
    return updated_group, commit, audit_entry


def can_decrypt_at_epoch(
    group: GroupState,
    member_did: str,
    epoch: int,
) -> bool:
    """Check if a member can decrypt messages at a given epoch.
    
    A member can decrypt messages at epoch E if:
    - They joined at or before epoch E
    - They were not removed before epoch E
    
    This is useful for verifying forward secrecy properties.
    
    Args:
        group: The group state
        member_did: DID of the member
        epoch: The epoch to check
        
    Returns:
        True if the member can decrypt at that epoch
    """
    member = group.get_member(member_did)
    if member is None:
        return False
    
    # Must have joined at or before this epoch
    if member.joined_at_epoch > epoch:
        return False
    
    # If removed, check when
    if member.status in (MemberStatus.REMOVED, MemberStatus.LEFT):
        # Find the epoch when they were removed
        # If removed_at is set, they can't decrypt at epochs after removal
        # For simplicity, we assume removal happened at the epoch when status changed
        # A more complete implementation would track removal_epoch explicitly
        
        # For now, removed members can't decrypt current or future epochs
        if epoch >= group.epoch and member.status == MemberStatus.REMOVED:
            return False
    
    return True


def get_removal_history(group: GroupState) -> list[dict[str, Any]]:
    """Get history of member removals from the group.
    
    Reconstructs removal history from the members list.
    
    Args:
        group: The group state
        
    Returns:
        List of removal records with member info and timestamps
    """
    removals = []
    for member in group.members.values():
        if member.status in (MemberStatus.REMOVED, MemberStatus.LEFT):
            removals.append({
                "member_did": member.did,
                "role_at_removal": member.role.value,
                "joined_at_epoch": member.joined_at_epoch,
                "joined_at": member.joined_at.isoformat(),
                "removed_at": member.removed_at.isoformat() if member.removed_at else None,
                "status": member.status.value,
            })
    
    # Sort by removal time
    removals.sort(key=lambda r: r["removed_at"] or "")
    return removals


def rotate_keys(
    group: GroupState,
    rotator_did: str,
    rotator_signing_key: bytes,
    reason: str | None = None,
) -> tuple[GroupState, CommitMessage]:
    """Manually rotate group keys without membership changes.
    
    Creates a new epoch with fresh keys. Useful for:
    - Periodic key rotation
    - After suspected key compromise
    - Compliance requirements
    
    Args:
        group: Current group state
        rotator_did: DID of the member requesting rotation (must be admin)
        rotator_signing_key: Ed25519 private key of the rotator
        reason: Optional reason for rotation
        
    Returns:
        Tuple of (updated_group, commit_message)
        
    Raises:
        PermissionError: If rotator is not an admin
    """
    import hmac
    
    # Validate rotator has permission
    rotator = group.get_member(rotator_did)
    if rotator is None:
        raise PermissionError(f"Rotator {rotator_did} is not a group member")
    if rotator.role != GroupRole.ADMIN:
        raise PermissionError(f"Rotator {rotator_did} is not an admin")
    if rotator.status != MemberStatus.ACTIVE:
        raise PermissionError(f"Rotator {rotator_did} is not an active member")
    
    now = datetime.now()
    new_epoch = group.epoch + 1
    
    # Generate new commit secret
    commit_secret = secrets.token_bytes(32)
    
    # Derive new epoch secrets
    if group.current_secrets:
        new_secrets = EpochSecrets.derive(
            epoch=new_epoch,
            init_secret=group.current_secrets.tree_secret,
            commit_secret=commit_secret,
        )
    else:
        new_secrets = EpochSecrets.derive(
            epoch=new_epoch,
            init_secret=group.init_secret,
            commit_secret=commit_secret,
        )
    
    # Create Commit message
    commit = CommitMessage(
        id=uuid4(),
        group_id=group.id,
        from_epoch=group.epoch,
        to_epoch=new_epoch,
        proposals=[{
            "type": "update",
            "action": "key_rotation",
            "rotated_by": rotator_did,
            "reason": reason,
        }],
        committer_did=rotator_did,
        created_at=now,
    )
    
    # Encrypt commit secret for all active members
    for member in group.get_active_members():
        if member.init_public_key:
            ephemeral_private = X25519PrivateKey.generate()
            ephemeral_public = ephemeral_private.public_key()
            
            member_key = X25519PublicKey.from_public_bytes(member.init_public_key)
            shared_secret = ephemeral_private.exchange(member_key)
            
            enc_key = HKDF(
                algorithm=hashes.SHA256(),
                length=AES_KEY_SIZE,
                salt=None,
                info=b"valence-mls-commit-key",
            ).derive(shared_secret)
            
            nonce = os.urandom(NONCE_SIZE)
            aesgcm = AESGCM(enc_key)
            encrypted = aesgcm.encrypt(nonce, commit_secret, None)
            
            commit.encrypted_commit_secrets[member.did] = {
                "encrypted": base64.b64encode(encrypted).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "ephemeral_public_key": base64.b64encode(
                    ephemeral_public.public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ).decode(),
            }
    
    # Compute confirmation tag
    confirmation_content = json.dumps({
        "from_epoch": commit.from_epoch,
        "to_epoch": commit.to_epoch,
        "proposals": commit.proposals,
        "committer_did": commit.committer_did,
    }, sort_keys=True).encode()
    commit.confirmation_tag = hmac.digest(
        new_secrets.confirmation_key,
        confirmation_content,
        'sha256',
    )
    
    # Sign commit
    signing_key = Ed25519PrivateKey.from_private_bytes(rotator_signing_key)
    commit_content = json.dumps({
        "id": str(commit.id),
        "group_id": str(commit.group_id),
        "from_epoch": commit.from_epoch,
        "to_epoch": commit.to_epoch,
        "proposals": commit.proposals,
    }, sort_keys=True).encode()
    commit.signature = signing_key.sign(commit_content)
    
    # Update group state
    updated_group = GroupState(
        id=group.id,
        name=group.name,
        epoch=new_epoch,
        status=group.status,
        members=group.members,
        pending_members=group.pending_members,
        current_secrets=new_secrets,
        init_secret=group.init_secret,
        config=group.config,
        created_at=group.created_at,
        created_by=group.created_by,
        updated_at=now,
        next_leaf_index=group.next_leaf_index,
    )
    
    return updated_group, commit
