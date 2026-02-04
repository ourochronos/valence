"""Privacy module for Valence - share policies, encryption, and sharing service."""

from .types import ShareLevel, EnforcementType, PropagationRules, SharePolicy
from .encryption import EncryptionEnvelope
from .sharing import (
    ShareRequest,
    ShareResult,
    ConsentChainEntry,
    Share,
    SharingService,
)
from .migration import (
    migrate_visibility,
    migrate_all_beliefs,
    migrate_all_beliefs_sync,
    get_share_policy_json,
)
from .trust import (
    TrustEdge,
    TrustEdge4D,
    TrustGraphStore,
    TrustService,
    get_trust_graph_store,
    get_trust_service,
    DecayModel,
    grant_trust,
    revoke_trust,
    get_trust,
    list_trusted,
    list_trusters,
)
from .domains import (
    Domain,
    DomainMembership,
    DomainRole,
    DomainService,
    DomainError,
    DomainNotFoundError,
    DomainExistsError,
    MembershipExistsError,
    MembershipNotFoundError,
    PermissionDeniedError,
)

__all__ = [
    # Types
    "ShareLevel",
    "EnforcementType", 
    "PropagationRules",
    "SharePolicy",
    # Encryption
    "EncryptionEnvelope",
    # Sharing
    "ShareRequest",
    "ShareResult",
    "ConsentChainEntry",
    "Share",
    "SharingService",
    # Migration
    "migrate_visibility",
    "migrate_all_beliefs",
    "migrate_all_beliefs_sync",
    "get_share_policy_json",
    # Trust (Issue #58, #59)
    "TrustEdge",
    "TrustEdge4D",
    "TrustGraphStore",
    "TrustService",
    "get_trust_graph_store",
    "get_trust_service",
    "DecayModel",
    "grant_trust",
    "revoke_trust",
    "get_trust",
    "list_trusted",
    "list_trusters",
    # Domains
    "Domain",
    "DomainMembership",
    "DomainRole",
    "DomainService",
    "DomainError",
    "DomainNotFoundError",
    "DomainExistsError",
    "MembershipExistsError",
    "MembershipNotFoundError",
    "PermissionDeniedError",
]
