"""Trust graph for Valence privacy module.

Implements multi-dimensional trust edges between DIDs with database storage.
Trust has four dimensions:
- competence: Ability to perform tasks correctly
- integrity: Honesty and consistency
- confidentiality: Ability to keep secrets
- judgment: Ability to evaluate others (affects delegated trust)

The judgment dimension is special: it affects how much weight we give to
someone's trust recommendations. Low judgment means their trust in others
is weighted less in transitive trust calculations.

Trust Decay:
Trust can decay over time if not refreshed. This models the natural erosion
of trust when relationships aren't maintained. Decay is configurable via:
- decay_rate: Rate of decay per day (0.0 = no decay)
- decay_model: LINEAR (constant loss) or EXPONENTIAL (percentage loss)
- last_refreshed: When trust was last confirmed/refreshed

This package is organized into the following modules:
- edges: TrustEdge and DecayModel types
- computation: Trust computation algorithms (delegation, transitive trust)
- graph_store: Database persistence for trust edges
- service: High-level TrustService API
- federation: Federation trust relationships
"""

from __future__ import annotations

# Computation functions
from .computation import (
    compute_delegated_trust,
    compute_transitive_trust,
)

# Edge types
from .edges import (
    CLOCK_SKEW_TOLERANCE,
    DEFAULT_EPISTEMIC_WEIGHTS,
    EPISTEMIC_DIMENSIONS,
    TRUST_SCHEMA_CORE,
    TRUST_SCHEMA_EPISTEMIC,
    DecayModel,
    EpistemicTrustDimension,
    RelationshipType,
    TrustEdge,
    TrustEdge4D,
    compute_epistemic_trust,
)

# Federation types and functions
from .federation import (
    FEDERATION_PREFIX,
    FederationMembershipRegistry,
    FederationTrustEdge,
    get_did_federation,
    get_effective_trust_with_federation,
    get_federation_registry,
    get_federation_trust,
    register_federation_member,
    revoke_federation_trust,
    set_federation_trust,
    unregister_federation_member,
)

# Graph store
from .graph_store import (
    TrustGraphStore,
    get_trust_graph_store,
)

# Service and convenience functions
from .service import (
    TrustService,
    compute_delegated_trust_from_service,
    compute_weighted_trust,
    get_trust,
    get_trust_dimensions,
    get_trust_service,
    grant_trust,
    list_trusted,
    list_trusters,
    revoke_trust,
    set_distrust,
    set_ignore,
    set_trust_dimensions,
    unwatch,
    watch,
)

__all__ = [
    # Constants
    "CLOCK_SKEW_TOLERANCE",
    "DEFAULT_EPISTEMIC_WEIGHTS",
    "EPISTEMIC_DIMENSIONS",
    "FEDERATION_PREFIX",
    "TRUST_SCHEMA_CORE",
    "TRUST_SCHEMA_EPISTEMIC",
    # Edge types
    "DecayModel",
    "EpistemicTrustDimension",
    "RelationshipType",
    "TrustEdge",
    "TrustEdge4D",
    # Epistemic trust computation
    "compute_epistemic_trust",
    # Computation
    "compute_delegated_trust",
    "compute_transitive_trust",
    # Graph store
    "TrustGraphStore",
    "get_trust_graph_store",
    # Service
    "TrustService",
    "get_trust_service",
    "grant_trust",
    "revoke_trust",
    "get_trust",
    "list_trusted",
    "list_trusters",
    "watch",
    "unwatch",
    "set_distrust",
    "set_ignore",
    "compute_delegated_trust_from_service",
    "compute_weighted_trust",
    "get_trust_dimensions",
    "set_trust_dimensions",
    # Federation types
    "FederationTrustEdge",
    "FederationMembershipRegistry",
    # Federation functions
    "get_federation_registry",
    "set_federation_trust",
    "get_federation_trust",
    "revoke_federation_trust",
    "get_effective_trust_with_federation",
    "register_federation_member",
    "unregister_federation_member",
    "get_did_federation",
]
