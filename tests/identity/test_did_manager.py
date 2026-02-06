"""Tests for the DIDManager service layer.

Issue #277: Identity — multi-DID per user (node = DID).

Tests cover:
- DID creation and deterministic derivation
- Bilateral link proof generation and verification
- Cluster creation, merging, unlinking
- Revocation with isolation (compromised node doesn't affect others)
- Error handling
"""

from __future__ import annotations

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from valence.identity.did_manager import (
    DIDAlreadyExistsError,
    DIDManager,
    DIDNotFoundError,
    DIDRevokedError,
    InMemoryDIDStore,
    _did_from_public_key,
)
from valence.identity.multi_did import DIDStatus, LinkProof

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mgr() -> DIDManager:
    return DIDManager()


@pytest.fixture()
def mgr_with_two_nodes(mgr: DIDManager):
    """Return (manager, node_a, key_a, node_b, key_b)."""
    node_a, key_a = mgr.create_node_did(label="laptop")
    node_b, key_b = mgr.create_node_did(label="phone")
    return mgr, node_a, key_a, node_b, key_b


# ---------------------------------------------------------------------------
# DID creation
# ---------------------------------------------------------------------------


class TestCreateNodeDID:
    def test_creates_valid_did(self, mgr: DIDManager):
        node, key = mgr.create_node_did(label="test")
        assert node.did.startswith("did:valence:")
        assert len(node.public_key) == 32
        assert node.label == "test"
        assert node.is_active
        assert node.cluster_id is None

    def test_deterministic_did_from_key(self, mgr: DIDManager):
        key = Ed25519PrivateKey.generate()
        node, _ = mgr.create_node_did(private_key=key)
        expected = _did_from_public_key(key.public_key())
        assert node.did == expected

    def test_duplicate_key_raises(self, mgr: DIDManager):
        key = Ed25519PrivateKey.generate()
        mgr.create_node_did(private_key=key)
        with pytest.raises(DIDAlreadyExistsError):
            mgr.create_node_did(private_key=key)

    def test_metadata_stored(self, mgr: DIDManager):
        node, _ = mgr.create_node_did(metadata={"os": "linux"})
        assert node.metadata["os"] == "linux"

    def test_different_keys_different_dids(self, mgr: DIDManager):
        n1, _ = mgr.create_node_did()
        n2, _ = mgr.create_node_did()
        assert n1.did != n2.did


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------


class TestLinkDIDs:
    def test_link_creates_cluster(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        proof = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        assert proof.did_a == node_a.did
        assert proof.did_b == node_b.did
        assert node_a.cluster_id == node_b.cluster_id
        assert node_a.cluster_id is not None

    def test_link_proof_is_verifiable(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        proof = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        assert mgr.verify_link_proof(proof) is True

    def test_link_adds_to_existing_cluster(self, mgr: DIDManager):
        n1, k1 = mgr.create_node_did(label="n1")
        n2, k2 = mgr.create_node_did(label="n2")
        n3, k3 = mgr.create_node_did(label="n3")

        mgr.link_dids(n1.did, k1, n2.did, k2)
        mgr.link_dids(n1.did, k1, n3.did, k3)

        # All three should be in the same cluster
        assert n1.cluster_id == n2.cluster_id == n3.cluster_id

        cluster = mgr.resolve_identity(n1.did)
        assert cluster is not None
        assert len(cluster.nodes) == 3

    def test_link_merges_clusters(self, mgr: DIDManager):
        n1, k1 = mgr.create_node_did(label="n1")
        n2, k2 = mgr.create_node_did(label="n2")
        n3, k3 = mgr.create_node_did(label="n3")
        n4, k4 = mgr.create_node_did(label="n4")

        # Create two separate clusters
        mgr.link_dids(n1.did, k1, n2.did, k2)
        mgr.link_dids(n3.did, k3, n4.did, k4)

        assert n1.cluster_id != n3.cluster_id

        # Now link n2 and n3 — clusters should merge
        mgr.link_dids(n2.did, k2, n3.did, k3)

        # All four should be in the same cluster
        assert n1.cluster_id == n2.cluster_id == n3.cluster_id == n4.cluster_id

    def test_link_already_same_cluster(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        # Linking again should not error
        proof2 = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        assert mgr.verify_link_proof(proof2)

    def test_link_unknown_did_raises(self, mgr: DIDManager):
        node, key = mgr.create_node_did()
        fake_key = Ed25519PrivateKey.generate()
        with pytest.raises(DIDNotFoundError):
            mgr.link_dids(node.did, key, "did:valence:nonexistent", fake_key)

    def test_link_revoked_did_raises(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        mgr.revoke_did(node_a.did)
        with pytest.raises(DIDRevokedError):
            mgr.link_dids(node_a.did, key_a, node_b.did, key_b)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerifyLinkProof:
    def test_valid_proof(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        proof = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        assert mgr.verify_link_proof(proof) is True

    def test_tampered_signature_fails(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        proof = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        proof.signature_a = b"\x00" * 64  # tamper
        assert mgr.verify_link_proof(proof) is False

    def test_unknown_did_fails(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        proof = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        proof.did_a = "did:valence:unknown"
        assert mgr.verify_link_proof(proof) is False


# ---------------------------------------------------------------------------
# Unlinking
# ---------------------------------------------------------------------------


class TestUnlinkDID:
    def test_unlink_removes_from_cluster(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        mgr.link_dids(node_a.did, key_a, node_b.did, key_b)

        mgr.unlink_did(node_a.did)
        assert node_a.cluster_id is None
        assert node_a.is_active  # still active, just unclustered

        cluster = mgr.resolve_identity(node_b.did)
        assert cluster is not None
        assert not cluster.has_did(node_a.did)

    def test_unlink_unknown_raises(self, mgr: DIDManager):
        with pytest.raises(DIDNotFoundError):
            mgr.unlink_did("did:valence:nope")

    def test_unlink_unclustered_is_noop(self, mgr: DIDManager):
        node, _ = mgr.create_node_did()
        result = mgr.unlink_did(node.did)
        assert result.cluster_id is None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


class TestResolveIdentity:
    def test_resolve_clustered(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        mgr.link_dids(node_a.did, key_a, node_b.did, key_b)

        cluster = mgr.resolve_identity(node_a.did)
        assert cluster is not None
        assert cluster.has_did(node_a.did)
        assert cluster.has_did(node_b.did)

    def test_resolve_unclustered_returns_none(self, mgr: DIDManager):
        node, _ = mgr.create_node_did()
        assert mgr.resolve_identity(node.did) is None

    def test_resolve_unknown_raises(self, mgr: DIDManager):
        with pytest.raises(DIDNotFoundError):
            mgr.resolve_identity("did:valence:unknown")


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


class TestRevokeDID:
    def test_revoke_marks_revoked(self, mgr: DIDManager):
        node, _ = mgr.create_node_did()
        revoked = mgr.revoke_did(node.did, reason="compromised")
        assert revoked.status == DIDStatus.REVOKED
        assert not revoked.is_active
        assert revoked.metadata["revoked_reason"] == "compromised"

    def test_revoke_removes_from_cluster(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        mgr.link_dids(node_a.did, key_a, node_b.did, key_b)

        mgr.revoke_did(node_a.did)

        # node_a is revoked and out of the cluster
        assert node_a.cluster_id is None
        assert node_a.status == DIDStatus.REVOKED

        # node_b is still active in the cluster
        cluster = mgr.resolve_identity(node_b.did)
        assert cluster is not None
        assert not cluster.has_did(node_a.did)
        assert cluster.has_did(node_b.did)
        assert node_b.is_active

    def test_revoke_isolates_compromised_node(self, mgr: DIDManager):
        """Security: revoking one DID must not affect others in the cluster."""
        nodes_keys = [mgr.create_node_did(label=f"n{i}") for i in range(4)]

        # Link all into one cluster
        for i in range(1, 4):
            mgr.link_dids(
                nodes_keys[0][0].did,
                nodes_keys[0][1],
                nodes_keys[i][0].did,
                nodes_keys[i][1],
            )

        # Revoke the second node
        mgr.revoke_did(nodes_keys[1][0].did)

        # The other 3 should still be clustered and active
        cluster = mgr.resolve_identity(nodes_keys[0][0].did)
        assert cluster is not None
        assert len(cluster.active_nodes) == 3
        assert not cluster.has_did(nodes_keys[1][0].did)

    def test_revoke_unknown_raises(self, mgr: DIDManager):
        with pytest.raises(DIDNotFoundError):
            mgr.revoke_did("did:valence:nope")


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


class TestListNodes:
    def test_list_all(self, mgr: DIDManager):
        mgr.create_node_did(label="a")
        mgr.create_node_did(label="b")
        nodes = mgr.list_nodes()
        assert len(nodes) == 2

    def test_list_by_cluster(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        # Create a third unclustered node
        mgr.create_node_did(label="stray")

        clustered = mgr.list_nodes(cluster_id=node_a.cluster_id)
        assert len(clustered) == 2

    def test_list_clusters(self, mgr_with_two_nodes):
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        mgr.link_dids(node_a.did, key_a, node_b.did, key_b)
        clusters = mgr.list_clusters()
        assert len(clusters) == 1


# ---------------------------------------------------------------------------
# InMemoryDIDStore
# ---------------------------------------------------------------------------


class TestInMemoryDIDStore:
    def test_basic_crud(self):
        from valence.identity.multi_did import DIDNode, IdentityCluster, LinkProof

        store = InMemoryDIDStore()

        node = DIDNode(did="did:valence:test", public_key=b"\x00" * 32)
        store.save_node(node)
        assert store.get_node("did:valence:test") is node
        assert store.get_node("did:valence:missing") is None

        cluster = IdentityCluster(cluster_id="c1")
        store.save_cluster(cluster)
        assert store.get_cluster("c1") is cluster
        assert store.get_cluster("c2") is None

        proof = LinkProof(did_a="did:valence:a", did_b="did:valence:b")
        store.save_proof(proof)
        assert len(store.get_proofs_for_did("did:valence:a")) == 1
        assert len(store.get_proofs_for_did("did:valence:b")) == 1
        assert len(store.get_proofs_for_did("did:valence:c")) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_no_master_key_single_point_of_failure(self, mgr: DIDManager):
        """Security: cluster has no master key.

        Each node signs independently. Losing any single node's key
        does not compromise the remaining nodes.
        """
        n1, k1 = mgr.create_node_did(label="primary")
        n2, k2 = mgr.create_node_did(label="secondary")
        n3, k3 = mgr.create_node_did(label="tertiary")

        mgr.link_dids(n1.did, k1, n2.did, k2)
        mgr.link_dids(n2.did, k2, n3.did, k3)

        # "Compromise" n1 — revoke it
        mgr.revoke_did(n1.did)

        # n2 and n3 are still linked and functional
        cluster = mgr.resolve_identity(n2.did)
        assert cluster is not None
        assert len(cluster.active_nodes) == 2
        assert cluster.has_did(n2.did)
        assert cluster.has_did(n3.did)

    def test_link_proof_bilateral(self, mgr_with_two_nodes):
        """Both parties must sign the proof (bilateral)."""
        mgr, node_a, key_a, node_b, key_b = mgr_with_two_nodes
        proof = mgr.link_dids(node_a.did, key_a, node_b.did, key_b)

        # Both signatures must be non-empty
        assert len(proof.signature_a) > 0
        assert len(proof.signature_b) > 0

        # Verify with correct keys succeeds
        assert mgr.verify_link_proof(proof) is True

        # Removing either signature should fail verification
        proof_copy_a = LinkProof(
            proof_id=proof.proof_id,
            did_a=proof.did_a,
            did_b=proof.did_b,
            cluster_id=proof.cluster_id,
            signature_a=b"\x00" * 64,  # bogus
            signature_b=proof.signature_b,
            nonce=proof.nonce,
        )
        assert mgr.verify_link_proof(proof_copy_a) is False
