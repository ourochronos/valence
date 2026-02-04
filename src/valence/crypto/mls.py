"""MLS (Messaging Layer Security) Abstraction Layer.

Provides a Python interface for MLS group encryption as specified in RFC 9420.
This module defines the abstract interface that can be backed by different
implementations:
- MockMLSBackend: For testing (no real crypto)
- Future: OpenMLS FFI binding via PyO3/cffi

MLS provides:
- Asynchronous group key agreement
- Forward secrecy and post-compromise security
- Efficient key updates with tree-based key derivation

Security properties:
- Forward secrecy: Compromise of current keys doesn't expose past messages
- Post-compromise security: Group recovers security after member compromise
- Authentication: All group operations are authenticated

Example:
    >>> backend = MockMLSBackend()
    >>> group = backend.create_group(b"group-123", creator_id=b"alice")
    >>> group = backend.add_member(group.group_id, b"bob", b"bob-key-package")
    >>> secrets = backend.get_application_secrets(group.group_id)
"""

from __future__ import annotations

import hashlib
import secrets as crypto_secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any


# =============================================================================
# Exceptions
# =============================================================================


class MLSError(Exception):
    """Base exception for MLS operations."""
    pass


class MLSGroupNotFoundError(MLSError):
    """Raised when a group ID is not found."""
    pass


class MLSMemberNotFoundError(MLSError):
    """Raised when a member is not found in the group."""
    pass


class MLSEpochMismatchError(MLSError):
    """Raised when an operation references the wrong epoch."""
    pass


class MLSInvalidProposalError(MLSError):
    """Raised when a proposal is invalid."""
    pass


# =============================================================================
# Data Classes
# =============================================================================


class ProposalType(Enum):
    """Types of MLS proposals."""
    ADD = auto()
    REMOVE = auto()
    UPDATE = auto()
    REINIT = auto()


@dataclass
class MLSMember:
    """Represents a member in an MLS group.
    
    Attributes:
        member_id: Unique identifier for the member (e.g., DID)
        leaf_index: Position in the ratchet tree
        credential: Member's credential (e.g., X.509 cert or basic)
        key_package: Member's key package for adding to groups
        joined_at: When the member joined the group
        last_update: When the member last updated their keys
    """
    
    member_id: bytes
    leaf_index: int
    credential: bytes = b""
    key_package: bytes = b""
    joined_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "member_id": self.member_id.hex(),
            "leaf_index": self.leaf_index,
            "credential": self.credential.hex(),
            "key_package": self.key_package.hex(),
            "joined_at": self.joined_at.isoformat(),
            "last_update": self.last_update.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLSMember":
        """Create from dictionary."""
        return cls(
            member_id=bytes.fromhex(data["member_id"]),
            leaf_index=data["leaf_index"],
            credential=bytes.fromhex(data.get("credential", "")),
            key_package=bytes.fromhex(data.get("key_package", "")),
            joined_at=datetime.fromisoformat(data["joined_at"]) if data.get("joined_at") else datetime.now(),
            last_update=datetime.fromisoformat(data["last_update"]) if data.get("last_update") else datetime.now(),
        )


@dataclass
class MLSKeySchedule:
    """Key schedule derived from the MLS epoch secret.
    
    The key schedule derives multiple secrets from the epoch secret:
    - application_secret: For encrypting application messages
    - confirmation_key: For confirming commits
    - membership_key: For membership proofs
    - resumption_psk: For group resumption
    - exporter_secret: For deriving external secrets
    
    In a real implementation, these would be derived using HKDF.
    """
    
    epoch: int
    epoch_secret: bytes
    application_secret: bytes
    confirmation_key: bytes
    membership_key: bytes
    resumption_psk: bytes
    exporter_secret: bytes
    
    def derive_application_key(self, generation: int) -> bytes:
        """Derive an application key for a specific generation.
        
        Application keys are derived in a chain from the application secret,
        providing forward secrecy for individual messages.
        
        Args:
            generation: The message generation (increments per message)
        
        Returns:
            32-byte application key for encryption
        """
        # In real MLS, this uses HKDF with the application secret
        data = (
            b"valence-mls-app-key-v1" +
            self.application_secret +
            generation.to_bytes(8, "big")
        )
        return hashlib.sha256(data).digest()
    
    def derive_nonce(self, generation: int) -> bytes:
        """Derive a nonce for a specific generation.
        
        Args:
            generation: The message generation
        
        Returns:
            12-byte nonce for AEAD
        """
        data = (
            b"valence-mls-nonce-v1" +
            self.application_secret +
            generation.to_bytes(8, "big")
        )
        return hashlib.sha256(data).digest()[:12]
    
    def export_secret(self, label: bytes, context: bytes, length: int = 32) -> bytes:
        """Export a secret for external use.
        
        Args:
            label: Label for the exported secret
            context: Context binding for the secret
            length: Desired length in bytes
        
        Returns:
            Exported secret of the specified length
        """
        data = (
            b"valence-mls-export-v1" +
            self.exporter_secret +
            label +
            context +
            length.to_bytes(4, "big")
        )
        # Use SHA-512 and truncate for variable length
        full_hash = hashlib.sha512(data).digest()
        return full_hash[:length]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "epoch": self.epoch,
            "epoch_secret": self.epoch_secret.hex(),
            "application_secret": self.application_secret.hex(),
            "confirmation_key": self.confirmation_key.hex(),
            "membership_key": self.membership_key.hex(),
            "resumption_psk": self.resumption_psk.hex(),
            "exporter_secret": self.exporter_secret.hex(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLSKeySchedule":
        """Create from dictionary."""
        return cls(
            epoch=data["epoch"],
            epoch_secret=bytes.fromhex(data["epoch_secret"]),
            application_secret=bytes.fromhex(data["application_secret"]),
            confirmation_key=bytes.fromhex(data["confirmation_key"]),
            membership_key=bytes.fromhex(data["membership_key"]),
            resumption_psk=bytes.fromhex(data["resumption_psk"]),
            exporter_secret=bytes.fromhex(data["exporter_secret"]),
        )


@dataclass
class MLSProposal:
    """An MLS proposal for group modification.
    
    Proposals are collected and then committed together.
    """
    
    proposal_type: ProposalType
    sender: bytes
    epoch: int
    payload: bytes  # Type-specific payload
    proposal_ref: bytes = field(default_factory=lambda: crypto_secrets.token_bytes(16))
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "proposal_type": self.proposal_type.name,
            "sender": self.sender.hex(),
            "epoch": self.epoch,
            "payload": self.payload.hex(),
            "proposal_ref": self.proposal_ref.hex(),
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLSProposal":
        """Create from dictionary."""
        return cls(
            proposal_type=ProposalType[data["proposal_type"]],
            sender=bytes.fromhex(data["sender"]),
            epoch=data["epoch"],
            payload=bytes.fromhex(data["payload"]),
            proposal_ref=bytes.fromhex(data["proposal_ref"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


@dataclass
class MLSGroup:
    """Represents an MLS group state.
    
    The group maintains:
    - Group identity and configuration
    - Current epoch (increments on each commit)
    - Member list with their credentials
    - Pending proposals awaiting commit
    
    In a full implementation, this would also include the ratchet tree
    and transcript hash.
    
    Attributes:
        group_id: Unique identifier for the group
        epoch: Current epoch number (increments on commits)
        cipher_suite: The cipher suite for the group (e.g., MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519)
        members: List of current group members
        pending_proposals: Proposals awaiting commit
        created_at: When the group was created
        last_commit: When the last commit occurred
    """
    
    group_id: bytes
    epoch: int = 0
    cipher_suite: int = 0x0001  # MLS_128_DHKEMX25519_AES128GCM_SHA256_Ed25519
    members: list[MLSMember] = field(default_factory=list)
    pending_proposals: list[MLSProposal] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_commit: datetime = field(default_factory=datetime.now)
    
    # Internal state (not serialized)
    _tree_hash: bytes = field(default=b"", repr=False)
    _confirmed_transcript_hash: bytes = field(default=b"", repr=False)
    
    def get_member(self, member_id: bytes) -> MLSMember | None:
        """Get a member by their ID."""
        for member in self.members:
            if member.member_id == member_id:
                return member
        return None
    
    def get_member_by_index(self, leaf_index: int) -> MLSMember | None:
        """Get a member by their leaf index."""
        for member in self.members:
            if member.leaf_index == leaf_index:
                return member
        return None
    
    @property
    def member_count(self) -> int:
        """Number of members in the group."""
        return len(self.members)
    
    @property
    def member_ids(self) -> list[bytes]:
        """List of member IDs."""
        return [m.member_id for m in self.members]
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "group_id": self.group_id.hex(),
            "epoch": self.epoch,
            "cipher_suite": self.cipher_suite,
            "members": [m.to_dict() for m in self.members],
            "pending_proposals": [p.to_dict() for p in self.pending_proposals],
            "created_at": self.created_at.isoformat(),
            "last_commit": self.last_commit.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLSGroup":
        """Create from dictionary."""
        return cls(
            group_id=bytes.fromhex(data["group_id"]),
            epoch=data["epoch"],
            cipher_suite=data.get("cipher_suite", 0x0001),
            members=[MLSMember.from_dict(m) for m in data.get("members", [])],
            pending_proposals=[MLSProposal.from_dict(p) for p in data.get("pending_proposals", [])],
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            last_commit=datetime.fromisoformat(data["last_commit"]) if data.get("last_commit") else datetime.now(),
        )


@dataclass
class MLSCommit:
    """An MLS commit message.
    
    Commits apply pending proposals and advance the epoch.
    """
    
    group_id: bytes
    epoch: int
    proposals: list[bytes]  # Proposal references being committed
    committer: bytes
    commit_secret: bytes  # For deriving the new epoch secret
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "group_id": self.group_id.hex(),
            "epoch": self.epoch,
            "proposals": [p.hex() for p in self.proposals],
            "committer": self.committer.hex(),
            "commit_secret": self.commit_secret.hex(),
            "created_at": self.created_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MLSCommit":
        """Create from dictionary."""
        return cls(
            group_id=bytes.fromhex(data["group_id"]),
            epoch=data["epoch"],
            proposals=[bytes.fromhex(p) for p in data["proposals"]],
            committer=bytes.fromhex(data["committer"]),
            commit_secret=bytes.fromhex(data["commit_secret"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


# =============================================================================
# Abstract Backend
# =============================================================================


class MLSBackend(ABC):
    """Abstract interface for MLS operations.
    
    This defines the contract that any MLS implementation must fulfill.
    Implementations include:
    - MockMLSBackend: For testing
    - Future: OpenMLSBackend (via FFI)
    """
    
    @abstractmethod
    def create_group(
        self,
        group_id: bytes,
        creator_id: bytes,
        credential: bytes = b"",
        cipher_suite: int = 0x0001,
    ) -> MLSGroup:
        """Create a new MLS group.
        
        Args:
            group_id: Unique identifier for the group
            creator_id: ID of the group creator
            credential: Creator's credential
            cipher_suite: MLS cipher suite to use
        
        Returns:
            New MLSGroup instance
        """
        pass
    
    @abstractmethod
    def add_member(
        self,
        group_id: bytes,
        member_id: bytes,
        key_package: bytes,
        credential: bytes = b"",
    ) -> MLSGroup:
        """Add a member to the group.
        
        Creates an Add proposal and commits it immediately.
        For batched adds, use propose_add() + commit().
        
        Args:
            group_id: Group to add to
            member_id: New member's ID
            key_package: New member's key package
            credential: New member's credential
        
        Returns:
            Updated group state
        
        Raises:
            MLSGroupNotFoundError: If group doesn't exist
        """
        pass
    
    @abstractmethod
    def remove_member(
        self,
        group_id: bytes,
        member_id: bytes,
        remover_id: bytes,
    ) -> MLSGroup:
        """Remove a member from the group.
        
        Args:
            group_id: Group to remove from
            member_id: Member to remove
            remover_id: ID of member performing removal
        
        Returns:
            Updated group state
        
        Raises:
            MLSGroupNotFoundError: If group doesn't exist
            MLSMemberNotFoundError: If member not in group
        """
        pass
    
    @abstractmethod
    def update_keys(
        self,
        group_id: bytes,
        member_id: bytes,
    ) -> MLSGroup:
        """Update a member's keys (self-update).
        
        Provides post-compromise security by rotating keys.
        
        Args:
            group_id: Group to update in
            member_id: Member updating their keys
        
        Returns:
            Updated group state
        """
        pass
    
    @abstractmethod
    def get_group(self, group_id: bytes) -> MLSGroup | None:
        """Get a group by ID.
        
        Args:
            group_id: Group identifier
        
        Returns:
            MLSGroup if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_key_schedule(self, group_id: bytes) -> MLSKeySchedule:
        """Get the current key schedule for a group.
        
        Args:
            group_id: Group identifier
        
        Returns:
            Current MLSKeySchedule
        
        Raises:
            MLSGroupNotFoundError: If group doesn't exist
        """
        pass
    
    @abstractmethod
    def propose_add(
        self,
        group_id: bytes,
        proposer_id: bytes,
        member_id: bytes,
        key_package: bytes,
    ) -> MLSProposal:
        """Create an Add proposal without committing.
        
        Args:
            group_id: Target group
            proposer_id: Member creating the proposal
            member_id: Member to add
            key_package: New member's key package
        
        Returns:
            The proposal (added to pending)
        """
        pass
    
    @abstractmethod
    def propose_remove(
        self,
        group_id: bytes,
        proposer_id: bytes,
        member_id: bytes,
    ) -> MLSProposal:
        """Create a Remove proposal without committing.
        
        Args:
            group_id: Target group
            proposer_id: Member creating the proposal
            member_id: Member to remove
        
        Returns:
            The proposal (added to pending)
        """
        pass
    
    @abstractmethod
    def commit(
        self,
        group_id: bytes,
        committer_id: bytes,
        proposal_refs: list[bytes] | None = None,
    ) -> tuple[MLSGroup, MLSCommit]:
        """Commit pending proposals.
        
        Args:
            group_id: Target group
            committer_id: Member creating the commit
            proposal_refs: Specific proposals to commit (None = all pending)
        
        Returns:
            Tuple of (updated group, commit message)
        """
        pass
    
    @abstractmethod
    def process_commit(
        self,
        group_id: bytes,
        commit: MLSCommit,
    ) -> MLSGroup:
        """Process a received commit message.
        
        Args:
            group_id: Target group
            commit: The commit to process
        
        Returns:
            Updated group state
        """
        pass


# =============================================================================
# Mock Implementation
# =============================================================================


class MockMLSBackend(MLSBackend):
    """Mock MLS backend for testing.
    
    Provides a functional implementation without real cryptography.
    Useful for testing group management logic, protocol flows, and integration.
    
    Example:
        >>> backend = MockMLSBackend()
        >>> group = backend.create_group(b"test-group", b"alice")
        >>> group = backend.add_member(group.group_id, b"bob", b"bob-kp")
        >>> assert len(group.members) == 2
    """
    
    def __init__(self):
        """Initialize the mock backend."""
        self._groups: dict[bytes, MLSGroup] = {}
        self._key_schedules: dict[bytes, MLSKeySchedule] = {}
    
    def _derive_key_schedule(self, group_id: bytes, epoch: int) -> MLSKeySchedule:
        """Derive a mock key schedule for an epoch."""
        # In real MLS, this comes from the epoch secret via HKDF
        base = hashlib.sha256(group_id + epoch.to_bytes(8, "big")).digest()
        
        def derive(label: bytes) -> bytes:
            return hashlib.sha256(base + label).digest()
        
        return MLSKeySchedule(
            epoch=epoch,
            epoch_secret=derive(b"epoch"),
            application_secret=derive(b"application"),
            confirmation_key=derive(b"confirmation"),
            membership_key=derive(b"membership"),
            resumption_psk=derive(b"resumption"),
            exporter_secret=derive(b"exporter"),
        )
    
    def _next_leaf_index(self, group: MLSGroup) -> int:
        """Get the next available leaf index."""
        if not group.members:
            return 0
        return max(m.leaf_index for m in group.members) + 1
    
    def create_group(
        self,
        group_id: bytes,
        creator_id: bytes,
        credential: bytes = b"",
        cipher_suite: int = 0x0001,
    ) -> MLSGroup:
        """Create a new MLS group."""
        creator = MLSMember(
            member_id=creator_id,
            leaf_index=0,
            credential=credential,
            key_package=crypto_secrets.token_bytes(32),  # Mock key package
        )
        
        group = MLSGroup(
            group_id=group_id,
            epoch=0,
            cipher_suite=cipher_suite,
            members=[creator],
        )
        
        self._groups[group_id] = group
        self._key_schedules[group_id] = self._derive_key_schedule(group_id, 0)
        
        return group
    
    def add_member(
        self,
        group_id: bytes,
        member_id: bytes,
        key_package: bytes,
        credential: bytes = b"",
    ) -> MLSGroup:
        """Add a member to the group."""
        group = self._groups.get(group_id)
        if group is None:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        
        # Check if already a member
        if group.get_member(member_id) is not None:
            raise MLSError(f"Member already in group: {member_id.hex()}")
        
        # Add the member
        new_member = MLSMember(
            member_id=member_id,
            leaf_index=self._next_leaf_index(group),
            credential=credential,
            key_package=key_package,
        )
        group.members.append(new_member)
        
        # Advance epoch
        group.epoch += 1
        group.last_commit = datetime.now()
        
        # Update key schedule
        self._key_schedules[group_id] = self._derive_key_schedule(group_id, group.epoch)
        
        return group
    
    def remove_member(
        self,
        group_id: bytes,
        member_id: bytes,
        remover_id: bytes,
    ) -> MLSGroup:
        """Remove a member from the group."""
        group = self._groups.get(group_id)
        if group is None:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        
        member = group.get_member(member_id)
        if member is None:
            raise MLSMemberNotFoundError(f"Member not found: {member_id.hex()}")
        
        # Check remover is in the group
        if group.get_member(remover_id) is None:
            raise MLSMemberNotFoundError(f"Remover not in group: {remover_id.hex()}")
        
        # Remove the member
        group.members = [m for m in group.members if m.member_id != member_id]
        
        # Advance epoch
        group.epoch += 1
        group.last_commit = datetime.now()
        
        # Update key schedule (ensures removed member can't decrypt new messages)
        self._key_schedules[group_id] = self._derive_key_schedule(group_id, group.epoch)
        
        return group
    
    def update_keys(
        self,
        group_id: bytes,
        member_id: bytes,
    ) -> MLSGroup:
        """Update a member's keys."""
        group = self._groups.get(group_id)
        if group is None:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        
        member = group.get_member(member_id)
        if member is None:
            raise MLSMemberNotFoundError(f"Member not found: {member_id.hex()}")
        
        # Update the member's key package (mock)
        member.key_package = crypto_secrets.token_bytes(32)
        member.last_update = datetime.now()
        
        # Advance epoch
        group.epoch += 1
        group.last_commit = datetime.now()
        
        # Update key schedule
        self._key_schedules[group_id] = self._derive_key_schedule(group_id, group.epoch)
        
        return group
    
    def get_group(self, group_id: bytes) -> MLSGroup | None:
        """Get a group by ID."""
        return self._groups.get(group_id)
    
    def get_key_schedule(self, group_id: bytes) -> MLSKeySchedule:
        """Get the current key schedule."""
        if group_id not in self._groups:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        return self._key_schedules[group_id]
    
    def propose_add(
        self,
        group_id: bytes,
        proposer_id: bytes,
        member_id: bytes,
        key_package: bytes,
    ) -> MLSProposal:
        """Create an Add proposal."""
        group = self._groups.get(group_id)
        if group is None:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        
        if group.get_member(proposer_id) is None:
            raise MLSMemberNotFoundError(f"Proposer not in group: {proposer_id.hex()}")
        
        # Serialize payload: 4-byte length prefix for member_id + member_id + key_package
        member_id_len = len(member_id).to_bytes(4, "big")
        proposal = MLSProposal(
            proposal_type=ProposalType.ADD,
            sender=proposer_id,
            epoch=group.epoch,
            payload=member_id_len + member_id + key_package,
        )
        
        group.pending_proposals.append(proposal)
        return proposal
    
    def propose_remove(
        self,
        group_id: bytes,
        proposer_id: bytes,
        member_id: bytes,
    ) -> MLSProposal:
        """Create a Remove proposal."""
        group = self._groups.get(group_id)
        if group is None:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        
        if group.get_member(proposer_id) is None:
            raise MLSMemberNotFoundError(f"Proposer not in group: {proposer_id.hex()}")
        
        if group.get_member(member_id) is None:
            raise MLSMemberNotFoundError(f"Target member not in group: {member_id.hex()}")
        
        proposal = MLSProposal(
            proposal_type=ProposalType.REMOVE,
            sender=proposer_id,
            epoch=group.epoch,
            payload=member_id,
        )
        
        group.pending_proposals.append(proposal)
        return proposal
    
    def commit(
        self,
        group_id: bytes,
        committer_id: bytes,
        proposal_refs: list[bytes] | None = None,
    ) -> tuple[MLSGroup, MLSCommit]:
        """Commit pending proposals."""
        group = self._groups.get(group_id)
        if group is None:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        
        if group.get_member(committer_id) is None:
            raise MLSMemberNotFoundError(f"Committer not in group: {committer_id.hex()}")
        
        # Select proposals to commit
        if proposal_refs is None:
            proposals_to_commit = group.pending_proposals[:]
        else:
            proposals_to_commit = [
                p for p in group.pending_proposals
                if p.proposal_ref in proposal_refs
            ]
        
        # Verify all proposals are for current epoch
        for p in proposals_to_commit:
            if p.epoch != group.epoch:
                raise MLSEpochMismatchError(
                    f"Proposal epoch {p.epoch} != group epoch {group.epoch}"
                )
        
        # Apply proposals
        for proposal in proposals_to_commit:
            if proposal.proposal_type == ProposalType.ADD:
                # Extract member_id and key_package from payload
                # Format: 4-byte length prefix + member_id + key_package
                member_id_len = int.from_bytes(proposal.payload[:4], "big")
                member_id = proposal.payload[4:4 + member_id_len]
                key_package = proposal.payload[4 + member_id_len:]
                
                new_member = MLSMember(
                    member_id=member_id,
                    leaf_index=self._next_leaf_index(group),
                    key_package=key_package,
                )
                group.members.append(new_member)
                
            elif proposal.proposal_type == ProposalType.REMOVE:
                member_id = proposal.payload
                group.members = [m for m in group.members if m.member_id != member_id]
        
        # Clear committed proposals
        committed_refs = [p.proposal_ref for p in proposals_to_commit]
        group.pending_proposals = [
            p for p in group.pending_proposals
            if p.proposal_ref not in committed_refs
        ]
        
        # Create commit
        commit = MLSCommit(
            group_id=group_id,
            epoch=group.epoch,
            proposals=committed_refs,
            committer=committer_id,
            commit_secret=crypto_secrets.token_bytes(32),
        )
        
        # Advance epoch
        group.epoch += 1
        group.last_commit = datetime.now()
        
        # Update key schedule
        self._key_schedules[group_id] = self._derive_key_schedule(group_id, group.epoch)
        
        return group, commit
    
    def process_commit(
        self,
        group_id: bytes,
        commit: MLSCommit,
    ) -> MLSGroup:
        """Process a received commit.
        
        In a distributed system, non-committing members receive the commit
        and update their state accordingly.
        """
        group = self._groups.get(group_id)
        if group is None:
            raise MLSGroupNotFoundError(f"Group not found: {group_id.hex()}")
        
        if commit.epoch != group.epoch:
            raise MLSEpochMismatchError(
                f"Commit epoch {commit.epoch} != group epoch {group.epoch}"
            )
        
        # In a real implementation, we would:
        # 1. Verify the commit signature
        # 2. Decrypt the path secrets
        # 3. Update the ratchet tree
        # 4. Derive new key schedule
        
        # For mock, just advance the epoch
        group.epoch += 1
        group.last_commit = datetime.now()
        self._key_schedules[group_id] = self._derive_key_schedule(group_id, group.epoch)
        
        return group
    
    def clear(self) -> None:
        """Clear all groups (for testing)."""
        self._groups.clear()
        self._key_schedules.clear()
