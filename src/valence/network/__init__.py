"""
Valence Network - E2E encrypted relay protocol.

This module provides end-to-end encryption for messages relayed through
router nodes, ensuring routers cannot read message content.
"""

from valence.network.crypto import (
    KeyPair,
    generate_identity_keypair,
    generate_encryption_keypair,
    encrypt_message,
    decrypt_message,
    # Circuit encryption (Issue #115)
    generate_circuit_keypair,
    derive_circuit_key,
    create_onion,
    peel_onion,
    encrypt_onion_layer,
    decrypt_onion_layer,
    encrypt_circuit_payload,
    decrypt_circuit_layer,
    encrypt_backward_payload,
    decrypt_backward_layers,
)
from valence.network.messages import (
    RelayMessage,
    DeliverPayload,
    # Circuit messages (Issue #115)
    Circuit,
    CircuitHop,
    CircuitCreateMessage,
    CircuitCreatedMessage,
    CircuitRelayMessage,
    CircuitDestroyMessage,
    CircuitExtendMessage,
)
from valence.network.router import (
    RouterNode,
    Connection,
    QueuedMessage,
    NodeConnectionHistory,
    # Circuit state (Issue #115)
    CircuitHopState,
    CircuitState,
)
from valence.network.seed import (
    SeedNode,
    RouterRecord,
    SeedConfig,
    HealthStatus,
    HealthState,
    HealthMonitor,
    # Regional routing utilities
    COUNTRY_TO_CONTINENT,
    get_continent,
    compute_region_score,
)
from valence.network.discovery import (
    DiscoveryClient,
    RouterInfo,
    DiscoveryError,
    NoSeedsAvailableError,
    SignatureVerificationError,
    create_discovery_client,
    discover_routers,
)
from valence.network.node import (
    NodeClient,
    RouterConnection,
    PendingMessage,
    PendingAck,
    FailoverState,
    ConnectionState,
    StateConflictError,
    StaleStateError,
    NodeError,
    NoRoutersAvailableError,
    create_node_client,
)
# Decomposed NodeClient components (Issue #128)
from valence.network.connection_manager import (
    ConnectionManager,
    ConnectionManagerConfig,
)
from valence.network.message_handler import (
    MessageHandler,
    MessageHandlerConfig,
)
from valence.network.router_client import (
    RouterClient,
    RouterClientConfig,
)
from valence.network.health_monitor import (
    HealthMonitor,
    HealthMonitorConfig,
)
from valence.network.config import (
    TrafficAnalysisMitigationConfig,
    PrivacyLevel,
    BatchingConfig,
    TimingJitterConfig,
    ConstantRateConfig,
    MixNetworkConfig,
    PRIVACY_LOW,
    PRIVACY_MEDIUM,
    PRIVACY_HIGH,
    PRIVACY_PARANOID,
    get_recommended_config,
)

__all__ = [
    # Crypto
    "KeyPair",
    "generate_identity_keypair",
    "generate_encryption_keypair",
    "encrypt_message",
    "decrypt_message",
    # Circuit encryption (Issue #115)
    "generate_circuit_keypair",
    "derive_circuit_key",
    "create_onion",
    "peel_onion",
    "encrypt_onion_layer",
    "decrypt_onion_layer",
    "encrypt_circuit_payload",
    "decrypt_circuit_layer",
    "encrypt_backward_payload",
    "decrypt_backward_layers",
    # Messages
    "RelayMessage",
    "DeliverPayload",
    # Circuit messages (Issue #115)
    "Circuit",
    "CircuitHop",
    "CircuitCreateMessage",
    "CircuitCreatedMessage",
    "CircuitRelayMessage",
    "CircuitDestroyMessage",
    "CircuitExtendMessage",
    # Router
    "RouterNode",
    "Connection",
    "QueuedMessage",
    "NodeConnectionHistory",
    # Circuit state (Issue #115)
    "CircuitHopState",
    "CircuitState",
    # Seed
    "SeedNode",
    "RouterRecord",
    "SeedConfig",
    # Health Monitoring
    "HealthStatus",
    "HealthState",
    "HealthMonitor",
    # Regional Routing
    "COUNTRY_TO_CONTINENT",
    "get_continent",
    "compute_region_score",
    # Discovery
    "DiscoveryClient",
    "RouterInfo",
    "DiscoveryError",
    "NoSeedsAvailableError",
    "SignatureVerificationError",
    "create_discovery_client",
    "discover_routers",
    # Node
    "NodeClient",
    "RouterConnection",
    "PendingMessage",
    "PendingAck",
    "FailoverState",
    "ConnectionState",
    "StateConflictError",
    "StaleStateError",
    "NodeError",
    "NoRoutersAvailableError",
    "create_node_client",
    # NodeClient components (Issue #128)
    "ConnectionManager",
    "ConnectionManagerConfig",
    "MessageHandler",
    "MessageHandlerConfig",
    "RouterClient",
    "RouterClientConfig",
    "HealthMonitor",
    "HealthMonitorConfig",
    # Traffic Analysis Mitigations (Issue #120)
    "TrafficAnalysisMitigationConfig",
    "PrivacyLevel",
    "BatchingConfig",
    "TimingJitterConfig",
    "ConstantRateConfig",
    "MixNetworkConfig",
    "PRIVACY_LOW",
    "PRIVACY_MEDIUM",
    "PRIVACY_HIGH",
    "PRIVACY_PARANOID",
    "get_recommended_config",
]
