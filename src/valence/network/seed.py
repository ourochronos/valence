"""
Valence Seed Node - The "phone book" for router discovery.

This module is a backward-compatibility shim. The implementation has been
refactored into the valence.network.seed package.

For new code, import directly from the package:
    from valence.network.seed import SeedNode, SeedConfig, create_seed_node
"""

# Re-export everything from the new package location for backward compatibility
from valence.network.seed import (
    COUNTRY_TO_CONTINENT,
    CorrelationDetector,
    HealthMonitor,
    HealthState,
    HealthStatus,
    RateLimiter,
    RegistrationEvent,
    ReputationManager,
    ReputationRecord,
    RouterRecord,
    SeedConfig,
    SeedNode,
    SeedPeerManager,
    SeedRevocationManager,
    SeedRevocationRecord,
    SybilResistance,
    _secure_random,
    compute_region_score,
    create_seed_node,
    get_continent,
    run_seed_node,
)

__all__ = [
    # Main classes
    "SeedNode",
    "SeedConfig",
    "RouterRecord",
    # Health monitoring
    "HealthStatus",
    "HealthState",
    "HealthMonitor",
    # Reputation / rate limiting
    "RegistrationEvent",
    "ReputationRecord",
    "RateLimiter",
    "ReputationManager",
    # Sybil resistance
    "CorrelationDetector",
    "SybilResistance",
    # Revocation
    "SeedRevocationRecord",
    "SeedRevocationManager",
    # Peering
    "SeedPeerManager",
    # Regional routing
    "COUNTRY_TO_CONTINENT",
    "get_continent",
    "compute_region_score",
    # Convenience functions
    "create_seed_node",
    "run_seed_node",
    # Internal (for security tests)
    "_secure_random",
]
