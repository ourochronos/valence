"""Multi-DID identity models for Valence.

Implements Issue #277: Identity — multi-DID per user (node = DID).

Each physical/logical node owns exactly one DID.  Multiple DIDs may belong to
the same *conceptual* user via an :class:`IdentityCluster`.  Cluster membership
is proven cryptographically using :class:`LinkProof` — both sides sign to
prevent unilateral claims.

Security design:
- **No master identity key.**  Every DID is independent.
- **Node compromise isolation.**  Revoking one DID leaves other cluster
  members unaffected.
- **Bilateral proofs.**  A link requires signatures from *both* DIDs,
  preventing impersonation.
"""

from __future__ import annotations

import enum
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# DID status
# ---------------------------------------------------------------------------


class DIDStatus(enum.StrEnum):
    """Lifecycle status of a DID."""

    ACTIVE = "active"
    REVOKED = "revoked"
    SUSPENDED = "suspended"


# ---------------------------------------------------------------------------
# DIDNode
# ---------------------------------------------------------------------------


@dataclass
class DIDNode:
    """A single node identity in the Valence network.

    Each node generates its own Ed25519 keypair.  The ``did`` field follows
    the ``did:valence:<fingerprint>`` scheme where the fingerprint is derived
    from the public key.

    Attributes:
        did: The decentralised identifier string (``did:valence:…``).
        public_key: Raw Ed25519 public key bytes.
        label: Human-readable label (e.g. ``"laptop"``, ``"phone"``).
        created_at: UNIX timestamp of creation.
        status: Current lifecycle status.
        metadata: Arbitrary key/value metadata (e.g. device info).
        cluster_id: ID of the :class:`IdentityCluster` this node belongs to,
            or ``None`` if unclustered.
    """

    did: str
    public_key: bytes
    label: str = ""
    created_at: float = field(default_factory=time.time)
    status: DIDStatus = DIDStatus.ACTIVE
    metadata: dict[str, Any] = field(default_factory=dict)
    cluster_id: str | None = None

    # -- helpers --

    @property
    def fingerprint(self) -> str:
        """Return the hex fingerprint portion of the DID."""
        # did:valence:<fingerprint>
        parts = self.did.split(":")
        return parts[-1] if len(parts) >= 3 else self.did

    @property
    def is_active(self) -> bool:
        return self.status == DIDStatus.ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "did": self.did,
            "public_key": self.public_key.hex(),
            "label": self.label,
            "created_at": self.created_at,
            "status": self.status.value,
            "metadata": self.metadata,
            "cluster_id": self.cluster_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DIDNode:
        return cls(
            did=data["did"],
            public_key=bytes.fromhex(data["public_key"]),
            label=data.get("label", ""),
            created_at=data.get("created_at", time.time()),
            status=DIDStatus(data.get("status", "active")),
            metadata=data.get("metadata", {}),
            cluster_id=data.get("cluster_id"),
        )


# ---------------------------------------------------------------------------
# LinkProof
# ---------------------------------------------------------------------------


@dataclass
class LinkProof:
    """Cryptographic proof that two DIDs belong to the same identity cluster.

    Both DIDs must sign the *link statement* — a canonical byte string encoding
    the two DIDs, the cluster ID, and a timestamp.  This prevents unilateral
    claims ("I belong to your cluster") and enables offline verification.

    Attributes:
        proof_id: Unique identifier for this proof.
        did_a: First DID in the link.
        did_b: Second DID in the link.
        cluster_id: The cluster both DIDs belong to.
        signature_a: Signature from ``did_a`` over the link statement.
        signature_b: Signature from ``did_b`` over the link statement.
        created_at: UNIX timestamp when the proof was created.
        nonce: Random nonce to prevent replay.
    """

    proof_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    did_a: str = ""
    did_b: str = ""
    cluster_id: str = ""
    signature_a: bytes = b""
    signature_b: bytes = b""
    created_at: float = field(default_factory=time.time)
    nonce: bytes = field(default_factory=lambda: uuid.uuid4().bytes)

    # -- canonical statement construction --

    def link_statement(self) -> bytes:
        """Build the canonical byte string that both parties sign.

        The statement is ``SHA-256(sorted_dids || cluster_id || nonce)``.
        Sorting the DIDs ensures order-independence.
        """
        dids_sorted = "|".join(sorted([self.did_a, self.did_b]))
        raw = f"{dids_sorted}|{self.cluster_id}|{self.nonce.hex()}".encode()
        return hashlib.sha256(raw).digest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "proof_id": self.proof_id,
            "did_a": self.did_a,
            "did_b": self.did_b,
            "cluster_id": self.cluster_id,
            "signature_a": self.signature_a.hex(),
            "signature_b": self.signature_b.hex(),
            "created_at": self.created_at,
            "nonce": self.nonce.hex(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LinkProof:
        return cls(
            proof_id=data["proof_id"],
            did_a=data["did_a"],
            did_b=data["did_b"],
            cluster_id=data["cluster_id"],
            signature_a=bytes.fromhex(data["signature_a"]),
            signature_b=bytes.fromhex(data["signature_b"]),
            created_at=data.get("created_at", time.time()),
            nonce=bytes.fromhex(data["nonce"]),
        )


# ---------------------------------------------------------------------------
# IdentityCluster
# ---------------------------------------------------------------------------


@dataclass
class IdentityCluster:
    """Groups multiple :class:`DIDNode` instances under one conceptual identity.

    The cluster itself has **no master key**.  Membership is determined solely
    by the set of :class:`LinkProof` objects connecting DIDs.  Removing a DID
    only requires revoking the proofs that reference it — remaining members
    are unaffected.

    Attributes:
        cluster_id: Unique identifier for this cluster.
        label: Human-readable label (e.g. ``"alice"``).
        nodes: Mapping of DID string → :class:`DIDNode`.
        proofs: List of :class:`LinkProof` objects in this cluster.
        created_at: UNIX timestamp of cluster creation.
        metadata: Arbitrary key/value metadata.
    """

    cluster_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    nodes: dict[str, DIDNode] = field(default_factory=dict)
    proofs: list[LinkProof] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- queries --

    @property
    def active_nodes(self) -> dict[str, DIDNode]:
        """Return only nodes with ACTIVE status."""
        return {did: n for did, n in self.nodes.items() if n.is_active}

    @property
    def revoked_nodes(self) -> dict[str, DIDNode]:
        """Return only nodes with REVOKED status."""
        return {did: n for did, n in self.nodes.items() if n.status == DIDStatus.REVOKED}

    def has_did(self, did: str) -> bool:
        return did in self.nodes

    def get_node(self, did: str) -> DIDNode | None:
        return self.nodes.get(did)

    # -- mutations --

    def add_node(self, node: DIDNode) -> None:
        """Add a node to the cluster and update its ``cluster_id``."""
        node.cluster_id = self.cluster_id
        self.nodes[node.did] = node

    def remove_node(self, did: str) -> DIDNode | None:
        """Remove a node from the cluster (does *not* revoke — just detaches)."""
        node = self.nodes.pop(did, None)
        if node is not None:
            node.cluster_id = None
        return node

    def add_proof(self, proof: LinkProof) -> None:
        self.proofs.append(proof)

    # -- serialisation --

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "nodes": {did: n.to_dict() for did, n in self.nodes.items()},
            "proofs": [p.to_dict() for p in self.proofs],
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityCluster:
        cluster = cls(
            cluster_id=data["cluster_id"],
            label=data.get("label", ""),
            created_at=data.get("created_at", time.time()),
            metadata=data.get("metadata", {}),
        )
        for did_str, node_data in data.get("nodes", {}).items():
            cluster.nodes[did_str] = DIDNode.from_dict(node_data)
        for proof_data in data.get("proofs", []):
            cluster.proofs.append(LinkProof.from_dict(proof_data))
        return cluster
