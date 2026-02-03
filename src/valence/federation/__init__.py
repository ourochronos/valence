"""Valence Federation - Federated epistemic network for knowledge sharing.

This package implements the Valence Federation Protocol (VFP), enabling
sovereign knowledge bases to form trust networks, share intelligence,
and collectively resolve contradictions across the epistemic commons.

Key components:
- models: Data models for federation (FederationNode, FederatedBelief, NodeTrust)
- identity: DID:vkb identity system
- protocol: Federation protocol handlers
- discovery: Node discovery and registration
- sync: Belief synchronization between nodes
- trust: Trust computation and management
- privacy: Differential privacy for aggregation (coming in Phase 3)
"""

from .models import (
    # Enums
    NodeStatus,
    TrustPhase,
    Visibility,
    ShareLevel,
    SyncStatus,
    ThreatLevel,
    ResolutionProposal,
    ResolutionStatus,
    ConsensusMethod,
    Vote,
    TrustPreference,
    AnnotationType,
    # Dataclasses
    FederationNode,
    FederatedBelief,
    BeliefProvenance,
    NodeTrust,
    UserNodeTrust,
    BeliefTrustAnnotation,
    AggregatedBelief,
    AggregationSource,
    TensionResolution,
    ConsensusVote,
    SyncState,
    SyncEvent,
    SyncOutboundItem,
    PrivacyParameters,
    CorroborationAttestation,
    TrustAttestation,
    AggregationQuery,
    AggregationResult,
    LocalSummary,
)

from .identity import (
    DIDMethod,
    DID,
    DIDDocument,
    VerificationMethod,
    ServiceEndpoint,
    generate_keypair,
    create_web_did,
    create_key_did,
    create_user_did,
    parse_did,
    resolve_did,
)

from .protocol import (
    MessageType,
    ErrorCode,
    create_auth_challenge,
    verify_auth_challenge,
    handle_share_belief,
    handle_request_beliefs,
    handle_sync_request,
    parse_message,
    handle_message,
)

from .discovery import (
    discover_node,
    discover_node_sync,
    register_node,
    get_node_by_did,
    get_node_by_id,
    get_node_trust,
    update_node_status,
    mark_node_active,
    mark_node_unreachable,
    bootstrap_federation,
    bootstrap_federation_sync,
    check_node_health,
    check_all_nodes_health,
    list_nodes,
    list_active_nodes,
    list_nodes_with_trust,
    get_known_peers,
    exchange_peers,
)

from .sync import (
    SyncManager,
    get_sync_state,
    update_sync_state,
    queue_belief_for_sync,
    get_pending_sync_items,
    mark_sync_item_sent,
    mark_sync_item_failed,
    compare_vector_clocks,
    update_vector_clock,
    trigger_sync,
    get_sync_status,
)

from .trust import (
    TrustSignal,
    TrustManager,
    get_trust_manager,
    get_effective_trust,
    process_corroboration,
    process_dispute,
    assess_and_respond_to_threat,
    SIGNAL_WEIGHTS,
    DECAY_HALF_LIFE_DAYS,
    PHASE_TRANSITION,
    THREAT_THRESHOLDS,
    PREFERENCE_MULTIPLIERS,
)

from .tools import (
    FEDERATION_TOOLS,
    FEDERATION_TOOL_HANDLERS,
    handle_federation_tool,
)

__all__ = [
    # Enums
    "NodeStatus",
    "TrustPhase",
    "Visibility",
    "ShareLevel",
    "SyncStatus",
    "ThreatLevel",
    "ResolutionProposal",
    "ResolutionStatus",
    "ConsensusMethod",
    "Vote",
    "TrustPreference",
    "AnnotationType",
    # Model dataclasses
    "FederationNode",
    "FederatedBelief",
    "BeliefProvenance",
    "NodeTrust",
    "UserNodeTrust",
    "BeliefTrustAnnotation",
    "AggregatedBelief",
    "AggregationSource",
    "TensionResolution",
    "ConsensusVote",
    "SyncState",
    "SyncEvent",
    "SyncOutboundItem",
    "PrivacyParameters",
    "CorroborationAttestation",
    "TrustAttestation",
    "AggregationQuery",
    "AggregationResult",
    "LocalSummary",
    # Identity
    "DIDMethod",
    "DID",
    "DIDDocument",
    "VerificationMethod",
    "ServiceEndpoint",
    "generate_keypair",
    "create_web_did",
    "create_key_did",
    "create_user_did",
    "parse_did",
    "resolve_did",
    # Protocol
    "MessageType",
    "ErrorCode",
    "create_auth_challenge",
    "verify_auth_challenge",
    "handle_share_belief",
    "handle_request_beliefs",
    "handle_sync_request",
    "parse_message",
    "handle_message",
    # Discovery
    "discover_node",
    "discover_node_sync",
    "register_node",
    "get_node_by_did",
    "get_node_by_id",
    "get_node_trust",
    "update_node_status",
    "mark_node_active",
    "mark_node_unreachable",
    "bootstrap_federation",
    "bootstrap_federation_sync",
    "check_node_health",
    "check_all_nodes_health",
    "list_nodes",
    "list_active_nodes",
    "list_nodes_with_trust",
    "get_known_peers",
    "exchange_peers",
    # Sync
    "SyncManager",
    "get_sync_state",
    "update_sync_state",
    "queue_belief_for_sync",
    "get_pending_sync_items",
    "mark_sync_item_sent",
    "mark_sync_item_failed",
    "compare_vector_clocks",
    "update_vector_clock",
    "trigger_sync",
    "get_sync_status",
    # Trust
    "TrustSignal",
    "TrustManager",
    "get_trust_manager",
    "get_effective_trust",
    "process_corroboration",
    "process_dispute",
    "assess_and_respond_to_threat",
    "SIGNAL_WEIGHTS",
    "DECAY_HALF_LIFE_DAYS",
    "PHASE_TRANSITION",
    "THREAT_THRESHOLDS",
    "PREFERENCE_MULTIPLIERS",
    # Tools
    "FEDERATION_TOOLS",
    "FEDERATION_TOOL_HANDLERS",
    "handle_federation_tool",
]
