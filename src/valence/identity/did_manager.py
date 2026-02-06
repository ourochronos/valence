"""DID Manager — service layer for multi-DID identity operations.

Implements Issue #277: create, link, unlink, revoke, and resolve DIDs.

The manager uses Ed25519 signing (from ``cryptography``) to:
- Derive deterministic ``did:valence:<fingerprint>`` identifiers from public keys.
- Generate bilateral :class:`LinkProof` objects where *both* DIDs sign.
- Verify proofs offline without relying on a central authority.

Storage is pluggable: the default in-memory backend is suitable for tests;
persistent backends (e.g. SQLite, Postgres) can be injected.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Protocol

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
)

from valence.identity.multi_did import (
    DIDNode,
    DIDStatus,
    IdentityCluster,
    LinkProof,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DIDError(Exception):
    """Base exception for DID operations."""


class DIDNotFoundError(DIDError):
    """Raised when a DID is not found in the store."""


class DIDAlreadyExistsError(DIDError):
    """Raised when trying to create a DID that already exists."""


class DIDRevokedError(DIDError):
    """Raised when an operation targets a revoked DID."""


class ClusterNotFoundError(DIDError):
    """Raised when a cluster is not found."""


class LinkProofInvalidError(DIDError):
    """Raised when a link proof fails verification."""


# ---------------------------------------------------------------------------
# Storage protocol
# ---------------------------------------------------------------------------


class DIDStore(Protocol):
    """Abstract storage backend for DID data."""

    def save_node(self, node: DIDNode) -> None: ...
    def get_node(self, did: str) -> DIDNode | None: ...
    def list_nodes(self, cluster_id: str | None = None) -> list[DIDNode]: ...
    def save_cluster(self, cluster: IdentityCluster) -> None: ...
    def get_cluster(self, cluster_id: str) -> IdentityCluster | None: ...
    def list_clusters(self) -> list[IdentityCluster]: ...
    def save_proof(self, proof: LinkProof) -> None: ...
    def get_proofs_for_did(self, did: str) -> list[LinkProof]: ...


# ---------------------------------------------------------------------------
# In-memory store (default / tests)
# ---------------------------------------------------------------------------


class InMemoryDIDStore:
    """Simple in-memory implementation of :class:`DIDStore`."""

    def __init__(self) -> None:
        self._nodes: dict[str, DIDNode] = {}
        self._clusters: dict[str, IdentityCluster] = {}
        self._proofs: list[LinkProof] = []

    def save_node(self, node: DIDNode) -> None:
        self._nodes[node.did] = node

    def get_node(self, did: str) -> DIDNode | None:
        return self._nodes.get(did)

    def list_nodes(self, cluster_id: str | None = None) -> list[DIDNode]:
        nodes = list(self._nodes.values())
        if cluster_id is not None:
            nodes = [n for n in nodes if n.cluster_id == cluster_id]
        return nodes

    def save_cluster(self, cluster: IdentityCluster) -> None:
        self._clusters[cluster.cluster_id] = cluster

    def get_cluster(self, cluster_id: str) -> IdentityCluster | None:
        return self._clusters.get(cluster_id)

    def list_clusters(self) -> list[IdentityCluster]:
        return list(self._clusters.values())

    def save_proof(self, proof: LinkProof) -> None:
        self._proofs.append(proof)

    def get_proofs_for_did(self, did: str) -> list[LinkProof]:
        return [p for p in self._proofs if p.did_a == did or p.did_b == did]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _public_key_bytes(pub: Ed25519PublicKey) -> bytes:
    """Extract raw 32-byte public key."""
    return pub.public_bytes(Encoding.Raw, PublicFormat.Raw)


def _did_from_public_key(pub: Ed25519PublicKey) -> str:
    """Derive a ``did:valence:<fingerprint>`` from an Ed25519 public key."""
    raw = _public_key_bytes(pub)
    fingerprint = hashlib.sha256(raw).hexdigest()[:32]
    return f"did:valence:{fingerprint}"


# ---------------------------------------------------------------------------
# DIDManager
# ---------------------------------------------------------------------------


class DIDManager:
    """High-level service for multi-DID identity management.

    Typical workflow::

        mgr = DIDManager()

        # Create node DIDs
        node_a, key_a = mgr.create_node_did(label="laptop")
        node_b, key_b = mgr.create_node_did(label="phone")

        # Link them into a cluster
        proof = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)

        # Resolve identity
        cluster = mgr.resolve_identity(node_a.did)

        # Revoke compromised node
        mgr.revoke_did(node_b.did)
    """

    def __init__(self, store: DIDStore | None = None) -> None:
        self._store: Any = store or InMemoryDIDStore()

    # -- DID creation -------------------------------------------------------

    def create_node_did(
        self,
        label: str = "",
        metadata: dict[str, Any] | None = None,
        private_key: Ed25519PrivateKey | None = None,
    ) -> tuple[DIDNode, Ed25519PrivateKey]:
        """Create a new node DID with a fresh Ed25519 keypair.

        Args:
            label: Human-readable label for the node.
            metadata: Optional key-value metadata.
            private_key: Optionally supply an existing key (tests / import).

        Returns:
            Tuple of (DIDNode, private_key).

        Raises:
            DIDAlreadyExistsError: If the derived DID already exists.
        """
        if private_key is None:
            private_key = Ed25519PrivateKey.generate()
        pub = private_key.public_key()
        did = _did_from_public_key(pub)

        if self._store.get_node(did) is not None:
            raise DIDAlreadyExistsError(f"DID {did} already exists")

        node = DIDNode(
            did=did,
            public_key=_public_key_bytes(pub),
            label=label,
            created_at=time.time(),
            status=DIDStatus.ACTIVE,
            metadata=metadata or {},
        )
        self._store.save_node(node)
        return node, private_key

    # -- Linking ------------------------------------------------------------

    def link_dids(
        self,
        did_a: str,
        key_a: Ed25519PrivateKey,
        did_b: str,
        key_b: Ed25519PrivateKey,
        cluster_label: str = "",
    ) -> LinkProof:
        """Link two DIDs into the same identity cluster with bilateral signing.

        If neither DID is already in a cluster, a new cluster is created.
        If one is in a cluster, the other is added to that cluster.
        If both are in different clusters, the clusters are merged.

        Args:
            did_a: First DID string.
            key_a: Private key of DID A (for signing).
            did_b: Second DID string.
            key_b: Private key of DID B (for signing).
            cluster_label: Optional human-readable label for a new cluster.

        Returns:
            The :class:`LinkProof` attesting the link.

        Raises:
            DIDNotFoundError: If either DID is not registered.
            DIDRevokedError: If either DID is revoked.
        """
        node_a = self._store.get_node(did_a)
        node_b = self._store.get_node(did_b)

        if node_a is None:
            raise DIDNotFoundError(f"DID {did_a} not found")
        if node_b is None:
            raise DIDNotFoundError(f"DID {did_b} not found")
        if node_a.status == DIDStatus.REVOKED:
            raise DIDRevokedError(f"DID {did_a} is revoked")
        if node_b.status == DIDStatus.REVOKED:
            raise DIDRevokedError(f"DID {did_b} is revoked")

        # Determine / create cluster
        cluster = self._resolve_or_create_cluster(node_a, node_b, cluster_label)

        # Build proof
        proof = LinkProof(
            did_a=did_a,
            did_b=did_b,
            cluster_id=cluster.cluster_id,
        )
        statement = proof.link_statement()

        # Bilateral signing
        proof.signature_a = key_a.sign(statement)
        proof.signature_b = key_b.sign(statement)

        # Persist
        cluster.add_node(node_a)
        cluster.add_node(node_b)
        cluster.add_proof(proof)
        self._store.save_cluster(cluster)
        self._store.save_proof(proof)
        self._store.save_node(node_a)
        self._store.save_node(node_b)

        return proof

    def _resolve_or_create_cluster(
        self,
        node_a: DIDNode,
        node_b: DIDNode,
        label: str,
    ) -> IdentityCluster:
        """Find or create the cluster for a link operation."""
        cluster_a = self._store.get_cluster(node_a.cluster_id) if node_a.cluster_id else None
        cluster_b = self._store.get_cluster(node_b.cluster_id) if node_b.cluster_id else None

        if cluster_a and cluster_b:
            if cluster_a.cluster_id == cluster_b.cluster_id:
                return cluster_a
            # Merge B into A
            for did, node in cluster_b.nodes.items():
                cluster_a.add_node(node)
                self._store.save_node(node)
            for p in cluster_b.proofs:
                cluster_a.add_proof(p)
            self._store.save_cluster(cluster_a)
            return cluster_a
        elif cluster_a:
            return cluster_a
        elif cluster_b:
            return cluster_b
        else:
            cluster = IdentityCluster(label=label)
            self._store.save_cluster(cluster)
            return cluster

    # -- Unlinking ----------------------------------------------------------

    def unlink_did(self, did: str) -> DIDNode:
        """Remove a DID from its cluster without revoking it.

        The node remains active but is no longer part of any cluster.

        Args:
            did: The DID to unlink.

        Returns:
            The unlinked :class:`DIDNode`.

        Raises:
            DIDNotFoundError: If the DID is not registered.
        """
        node = self._store.get_node(did)
        if node is None:
            raise DIDNotFoundError(f"DID {did} not found")

        if node.cluster_id:
            cluster = self._store.get_cluster(node.cluster_id)
            if cluster:
                cluster.remove_node(did)
                # Remove proofs referencing this DID
                cluster.proofs = [p for p in cluster.proofs if p.did_a != did and p.did_b != did]
                self._store.save_cluster(cluster)

        node.cluster_id = None
        self._store.save_node(node)
        return node

    # -- Listing ------------------------------------------------------------

    def list_nodes(self, cluster_id: str | None = None) -> list[DIDNode]:
        """List all registered nodes, optionally filtered by cluster.

        Args:
            cluster_id: If given, only return nodes in this cluster.

        Returns:
            List of :class:`DIDNode` instances.
        """
        return self._store.list_nodes(cluster_id=cluster_id)

    # -- Resolution ---------------------------------------------------------

    def resolve_identity(self, did: str) -> IdentityCluster | None:
        """Find the identity cluster a DID belongs to.

        Args:
            did: The DID to look up.

        Returns:
            The :class:`IdentityCluster` or ``None`` if the DID is unclustered.

        Raises:
            DIDNotFoundError: If the DID is not registered.
        """
        node = self._store.get_node(did)
        if node is None:
            raise DIDNotFoundError(f"DID {did} not found")
        if node.cluster_id is None:
            return None
        return self._store.get_cluster(node.cluster_id)

    # -- Revocation ---------------------------------------------------------

    def revoke_did(self, did: str, reason: str = "") -> DIDNode:
        """Revoke a compromised DID.

        The DID is marked REVOKED and removed from its cluster.  Other
        cluster members remain unaffected.

        Args:
            did: The DID to revoke.
            reason: Optional human-readable reason.

        Returns:
            The revoked :class:`DIDNode`.

        Raises:
            DIDNotFoundError: If the DID is not registered.
        """
        node = self._store.get_node(did)
        if node is None:
            raise DIDNotFoundError(f"DID {did} not found")

        node.status = DIDStatus.REVOKED
        node.metadata["revoked_reason"] = reason
        node.metadata["revoked_at"] = time.time()

        # Remove from cluster
        if node.cluster_id:
            cluster = self._store.get_cluster(node.cluster_id)
            if cluster:
                cluster.remove_node(did)
                # Remove proofs referencing revoked DID
                cluster.proofs = [p for p in cluster.proofs if p.did_a != did and p.did_b != did]
                self._store.save_cluster(cluster)

        node.cluster_id = None
        self._store.save_node(node)
        return node

    # -- Verification -------------------------------------------------------

    def verify_link_proof(self, proof: LinkProof) -> bool:
        """Verify a :class:`LinkProof` by checking both signatures.

        Args:
            proof: The proof to verify.

        Returns:
            ``True`` if both signatures are valid, ``False`` otherwise.
        """
        node_a = self._store.get_node(proof.did_a)
        node_b = self._store.get_node(proof.did_b)
        if node_a is None or node_b is None:
            return False

        statement = proof.link_statement()

        try:
            pub_a = Ed25519PublicKey.from_public_bytes(node_a.public_key)
            pub_a.verify(proof.signature_a, statement)
        except Exception:
            return False

        try:
            pub_b = Ed25519PublicKey.from_public_bytes(node_b.public_key)
            pub_b.verify(proof.signature_b, statement)
        except Exception:
            return False

        return True

    # -- Cluster listing ----------------------------------------------------

    def list_clusters(self) -> list[IdentityCluster]:
        """Return all identity clusters."""
        return self._store.list_clusters()
