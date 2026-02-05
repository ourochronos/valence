"""
Valence Seed Node - The "phone book" for router discovery.

This package provides seed node functionality for the Valence network:
- Router registration and discovery
- Health monitoring and heartbeat tracking
- Sybil resistance (rate limiting, reputation, correlation detection)
- Seed-to-seed gossip for registry synchronization
- Seed revocation management

Usage:
    from valence.network.seed import SeedNode, SeedConfig, create_seed_node
    
    # Create and run a seed node
    node = create_seed_node(host="0.0.0.0", port=8470)
    await node.run_forever()
"""

# Re-export backward compatible _secure_random for tests
import secrets
_secure_random = secrets.SystemRandom()

# Configuration
from .config import (
    SeedConfig,
    COUNTRY_TO_CONTINENT,
    get_continent,
    compute_region_score,
)

# Health monitoring
from .health import (
    HealthStatus,
    HealthState,
    HealthMonitor,
)

# Sybil resistance
from .sybil import (
    RegistrationEvent,
    ReputationRecord,
    RateLimiter,
    ReputationManager,
    CorrelationDetector,
    SybilResistance,
)

# Registry
from .registry import RouterRecord

# Revocation
from .revocation import (
    SeedRevocationRecord,
    SeedRevocationManager,
)

# Gossip
from .gossip import SeedPeerManager

# Core
from .core import (
    SeedNode,
    create_seed_node,
    run_seed_node,
)


__all__ = [
    # Secure random (for tests)
    "_secure_random",
    
    # Configuration
    "SeedConfig",
    "COUNTRY_TO_CONTINENT",
    "get_continent",
    "compute_region_score",
    
    # Health monitoring
    "HealthStatus",
    "HealthState",
    "HealthMonitor",
    
    # Sybil resistance
    "RegistrationEvent",
    "ReputationRecord",
    "RateLimiter",
    "ReputationManager",
    "CorrelationDetector",
    "SybilResistance",
    
    # Registry
    "RouterRecord",
    
    # Revocation
    "SeedRevocationRecord",
    "SeedRevocationManager",
    
    # Gossip
    "SeedPeerManager",
    
    # Core
    "SeedNode",
    "create_seed_node",
    "run_seed_node",
]
