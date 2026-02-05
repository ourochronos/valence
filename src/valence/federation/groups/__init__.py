"""MLS-style Group Encryption for Valence Federation.

This package implements a simplified MLS (Messaging Layer Security) protocol for
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

Submodules:
- mls: Core MLS protocol primitives (KeyPackage, EpochSecrets, etc.)
- membership: Group membership management (GroupState, add/remove members)
- admin: Federation group administration (FederationGroup, storage)
"""

# Re-export everything from submodules for backward compatibility
from .mls import (
    # Constants
    MLS_PROTOCOL_VERSION,
    DEFAULT_CIPHER_SUITE,
    MAX_GROUP_SIZE,
    EPOCH_SECRET_SIZE,
    ENCRYPTION_KEY_SIZE,
    KDF_INFO_EPOCH_SECRET,
    KDF_INFO_ENCRYPTION_KEY,
    KDF_INFO_WELCOME_KEY,
    KDF_INFO_MEMBER_SECRET,
    AES_KEY_SIZE,
    NONCE_SIZE,
    MAX_EPOCH_HISTORY,
    # Exceptions
    MLSError,
    GroupNotFoundError,
    MemberExistsError,
    MemberNotFoundError,
    InvalidKeyPackageError,
    GroupFullError,
    PermissionDeniedError,
    EpochMismatchError,
    # Classes
    KeyPackage,
    EpochSecrets,
    WelcomeMessage,
    CommitMessage,
    # Functions
    encrypt_group_content,
    decrypt_group_content,
)

from .membership import (
    # Enums
    GroupRole,
    MemberRole,  # Alias for backward compat
    ProposalType,
    MemberStatus,
    GroupStatus,
    # Classes
    GroupMember,
    GroupState,
    RemovalAuditEntry,
    # Functions
    create_group,
    add_member,
    process_welcome,
    process_commit,
    remove_member,
    can_decrypt_at_epoch,
    get_removal_history,
    rotate_keys,
)

from .admin import (
    # Classes
    FederationGroup,
    # Functions
    create_federation_group,
    get_federation_group_info,
    list_federation_group_members,
    verify_federation_membership,
    get_federation_member_role,
    # Storage functions
    store_federation_group,
    get_federation_group,
    get_group_by_federation_id,
    delete_federation_group,
    list_federation_groups,
    clear_federation_store,
)

__all__ = [
    # Constants
    "MLS_PROTOCOL_VERSION",
    "DEFAULT_CIPHER_SUITE",
    "MAX_GROUP_SIZE",
    "EPOCH_SECRET_SIZE",
    "ENCRYPTION_KEY_SIZE",
    "KDF_INFO_EPOCH_SECRET",
    "KDF_INFO_ENCRYPTION_KEY",
    "KDF_INFO_WELCOME_KEY",
    "KDF_INFO_MEMBER_SECRET",
    "AES_KEY_SIZE",
    "NONCE_SIZE",
    "MAX_EPOCH_HISTORY",
    # Exceptions
    "MLSError",
    "GroupNotFoundError",
    "MemberExistsError",
    "MemberNotFoundError",
    "InvalidKeyPackageError",
    "GroupFullError",
    "PermissionDeniedError",
    "EpochMismatchError",
    # Enums
    "GroupRole",
    "MemberRole",
    "ProposalType",
    "MemberStatus",
    "GroupStatus",
    # Classes
    "KeyPackage",
    "EpochSecrets",
    "WelcomeMessage",
    "CommitMessage",
    "GroupMember",
    "GroupState",
    "RemovalAuditEntry",
    "FederationGroup",
    # Group operations
    "create_group",
    "add_member",
    "process_welcome",
    "process_commit",
    "remove_member",
    "can_decrypt_at_epoch",
    "get_removal_history",
    "rotate_keys",
    # Encryption
    "encrypt_group_content",
    "decrypt_group_content",
    # Federation group operations
    "create_federation_group",
    "get_federation_group_info",
    "list_federation_group_members",
    "verify_federation_membership",
    "get_federation_member_role",
    # Storage
    "store_federation_group",
    "get_federation_group",
    "get_group_by_federation_id",
    "delete_federation_group",
    "list_federation_groups",
    "clear_federation_store",
]
