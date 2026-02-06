"""Identity management for Valence — multi-DID per user (node = DID).

Each node in the Valence network has its own DID. A user may operate multiple
nodes, grouped into an IdentityCluster. DIDs are linked via cryptographic
proofs so that compromise of one node does not endanger the others.

Key concepts:
- **DIDNode**: A single node identity with its own Ed25519 keypair.
- **IdentityCluster**: Groups multiple DIDNodes under one conceptual identity.
- **LinkProof**: Cryptographic proof that two DIDs belong to the same cluster.
- **DIDManager**: Service layer for creating, linking, revoking, and resolving DIDs.

Security properties:
- No master key / single point of failure.
- Revoking one DID does not affect others in the cluster.
- Link proofs are bidirectional (both nodes sign).
"""

from valence.identity.did_manager import DIDManager
from valence.identity.multi_did import (
    DIDNode,
    DIDStatus,
    IdentityCluster,
    LinkProof,
)

__all__ = [
    "DIDManager",
    "DIDNode",
    "DIDStatus",
    "IdentityCluster",
    "LinkProof",
]
