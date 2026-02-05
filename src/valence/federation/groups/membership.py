"""Group Membership Management for Valence Federation.

Handles group member management, state tracking, and membership operations
including member addition, removal, and group state transitions.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, serialization

from .mls import (
    KeyPackage,
    EpochSecrets,
    WelcomeMessage,
    CommitMessage,
    AES_KEY_SIZE,
    NONCE_SIZE,
)


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
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupState":
        """Restore from dictionary."""
        return cls(
            id=UUID(data["id"]),
            name=data["name"],
            epoch=data.get("epoch", 0),
            status=GroupStatus(data.get("status", "active")),
            members={did: GroupMember.from_dict(m) for did, m in data.get("members", {}).items()},
            config=data.get("config", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            created_by=data.get("created_by", ""),
            updated_at=datetime.fromisoformat(data["updated_at"]) if "updated_at" in data else datetime.now(),
        )
    
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
# REMOVAL AUDIT ENTRY
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
