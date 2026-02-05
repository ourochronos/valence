"""Federation Group Administration.

High-level federation group management linking federations to MLS groups.
Includes FederationGroup wrapper, storage, and administrative operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from .mls import KeyPackage
from .membership import (
    GroupState,
    GroupRole,
    GroupStatus,
    MemberStatus,
    GroupMember,
    create_group,
)


# =============================================================================
# FEDERATION GROUP
# =============================================================================


@dataclass
class FederationGroup:
    """Links a federation to its MLS group for end-to-end encryption.
    
    This is the main entry point for Issue #73, connecting the MLS group
    primitives (Issue #72) to federation-level concepts.
    
    Attributes:
        id: Unique identifier for this federation group
        federation_id: ID of the federation this group belongs to
        group_state: The underlying MLS GroupState
        allowed_domains: Optional list of belief domains permitted in this group
        metadata: Additional configuration and metadata
    """
    
    id: UUID
    federation_id: UUID
    group_state: GroupState
    
    # Optional domain restrictions
    allowed_domains: list[str] = field(default_factory=list)
    
    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    @property
    def epoch(self) -> int:
        """Get current epoch from underlying group state."""
        return self.group_state.epoch
    
    @property
    def name(self) -> str:
        """Get group name."""
        return self.group_state.name
    
    @property
    def creator_did(self) -> str:
        """Get creator DID."""
        return self.group_state.created_by
    
    @property
    def status(self) -> GroupStatus:
        """Get group status."""
        return self.group_state.status
    
    @property
    def member_count(self) -> int:
        """Get count of active members."""
        return len(self.group_state.get_active_members())
    
    @property
    def members(self) -> list[GroupMember]:
        """Get active members."""
        return self.group_state.get_active_members()
    
    def get_member(self, member_did: str) -> GroupMember | None:
        """Get member by DID."""
        return self.group_state.get_member(member_did)
    
    def has_member(self, member_did: str) -> bool:
        """Check if DID is an active member."""
        member = self.get_member(member_did)
        return member is not None and member.status == MemberStatus.ACTIVE
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "id": str(self.id),
            "federation_id": str(self.federation_id),
            "group_state": self.group_state.to_dict(),
            "allowed_domains": self.allowed_domains,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FederationGroup:
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]) if isinstance(data["id"], str) else data["id"],
            federation_id=UUID(data["federation_id"]) if isinstance(data["federation_id"], str) else data["federation_id"],
            group_state=GroupState.from_dict(data["group_state"]),
            allowed_domains=data.get("allowed_domains", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if isinstance(data.get("created_at"), str) else data.get("created_at", datetime.now()),
            modified_at=datetime.fromisoformat(data["modified_at"]) if isinstance(data.get("modified_at"), str) else data.get("modified_at", datetime.now()),
        )


# =============================================================================
# FEDERATION GROUP OPERATIONS
# =============================================================================


def create_federation_group(
    federation_id: UUID,
    creator_did: str,
    creator_signing_key: bytes | None = None,
    name: str = "",
    description: str = "",
    allowed_domains: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[FederationGroup, KeyPackage]:
    """Create a new MLS group for a federation.
    
    This is the main function for Issue #73, linking federation_id to an MLS group.
    The creator becomes the first member with admin role.
    
    Args:
        federation_id: ID of the federation this group belongs to
        creator_did: DID of the group creator (becomes admin)
        creator_signing_key: Optional Ed25519 signing key (generated if not provided)
        name: Optional human-readable name for the group
        description: Optional description
        allowed_domains: Optional list of allowed belief domains
        metadata: Optional additional metadata
    
    Returns:
        Tuple of (FederationGroup, creator's KeyPackage)
    
    Example:
        >>> from uuid import uuid4
        >>> fed_group, creator_kp = create_federation_group(
        ...     federation_id=uuid4(),
        ...     creator_did="did:vkb:web:example.com",
        ...     name="Example Federation Group",
        ... )
        >>> fed_group.member_count
        1
        >>> fed_group.members[0].role
        <GroupRole.ADMIN: 'admin'>
    """
    # Generate signing key if not provided
    if creator_signing_key is None:
        signing_key = Ed25519PrivateKey.generate()
        creator_signing_key = signing_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
    
    # Generate KeyPackage for creator from signing key
    creator_key_package, _ = KeyPackage.generate(
        member_did=creator_did,
        signing_private_key=creator_signing_key,
    )
    
    # Use existing create_group from membership module
    group_name = name or f"Federation {federation_id} Group"
    group_state = create_group(
        name=group_name,
        creator_did=creator_did,
        creator_key_package=creator_key_package,
        config=metadata,
    )
    
    # Wrap in FederationGroup
    federation_group = FederationGroup(
        id=uuid4(),
        federation_id=federation_id,
        group_state=group_state,
        allowed_domains=allowed_domains or [],
        metadata={
            "description": description,
            **(metadata or {}),
        },
        created_at=datetime.now(),
        modified_at=datetime.now(),
    )
    
    return federation_group, creator_key_package


def get_federation_group_info(group: FederationGroup) -> dict[str, Any]:
    """Get summary information about a federation group.
    
    Returns a dictionary with:
    - Basic group info (id, federation_id, name, status)
    - Current epoch
    - Member count
    - Admin list
    """
    admins = [
        m.did for m in group.members 
        if m.role == GroupRole.ADMIN and m.status == MemberStatus.ACTIVE
    ]
    
    return {
        "id": str(group.id),
        "federation_id": str(group.federation_id),
        "name": group.name,
        "description": group.metadata.get("description", ""),
        "creator_did": group.creator_did,
        "status": group.status.value,
        "epoch": group.epoch,
        "member_count": group.member_count,
        "admins": admins,
        "allowed_domains": group.allowed_domains,
        "created_at": group.created_at.isoformat(),
        "modified_at": group.modified_at.isoformat(),
    }


def list_federation_group_members(group: FederationGroup) -> list[dict[str, Any]]:
    """List all active members with their roles.
    
    Returns list of member info dicts (excludes sensitive key material).
    """
    return [
        {
            "member_did": m.did,
            "role": m.role.value,
            "status": m.status.value,
            "joined_at": m.joined_at.isoformat(),
            "joined_at_epoch": m.joined_at_epoch,
        }
        for m in group.members
        if m.status == MemberStatus.ACTIVE
    ]


def verify_federation_membership(group: FederationGroup, member_did: str) -> bool:
    """Verify if a DID is an active member of the federation group."""
    return group.has_member(member_did)


def get_federation_member_role(group: FederationGroup, member_did: str) -> GroupRole | None:
    """Get a member's role, or None if not an active member."""
    member = group.get_member(member_did)
    if member and member.status == MemberStatus.ACTIVE:
        return member.role
    return None


# =============================================================================
# IN-MEMORY STORAGE FOR FEDERATION GROUPS (MVP)
# =============================================================================

# Simple in-memory storage for federation groups
_federation_group_store: dict[UUID, FederationGroup] = {}
_federation_to_group: dict[UUID, UUID] = {}  # federation_id -> group_id


def store_federation_group(group: FederationGroup) -> None:
    """Store a federation group in the in-memory store."""
    _federation_group_store[group.id] = group
    _federation_to_group[group.federation_id] = group.id


def get_federation_group(group_id: UUID) -> FederationGroup | None:
    """Get a federation group by ID."""
    return _federation_group_store.get(group_id)


def get_group_by_federation_id(federation_id: UUID) -> FederationGroup | None:
    """Get the group for a federation."""
    group_id = _federation_to_group.get(federation_id)
    if group_id:
        return _federation_group_store.get(group_id)
    return None


def delete_federation_group(group_id: UUID) -> bool:
    """Delete a federation group."""
    group = _federation_group_store.get(group_id)
    if group:
        del _federation_group_store[group_id]
        if group.federation_id in _federation_to_group:
            del _federation_to_group[group.federation_id]
        return True
    return False


def list_federation_groups() -> list[FederationGroup]:
    """List all federation groups."""
    return list(_federation_group_store.values())


def clear_federation_store() -> None:
    """Clear the in-memory store (for testing)."""
    _federation_group_store.clear()
    _federation_to_group.clear()
