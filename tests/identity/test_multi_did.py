"""Tests for multi_did models (DIDNode, IdentityCluster, LinkProof).

Issue #277: Identity — multi-DID per user (node = DID).
"""

from __future__ import annotations

import uuid

import pytest

from valence.identity.multi_did import (
    DIDNode,
    DIDStatus,
    IdentityCluster,
    LinkProof,
)

# ---------------------------------------------------------------------------
# DIDNode
# ---------------------------------------------------------------------------


class TestDIDNode:
    """Tests for the DIDNode model."""

    def test_create_basic(self):
        node = DIDNode(
            did="did:valence:abc123",
            public_key=b"\x01" * 32,
            label="laptop",
        )
        assert node.did == "did:valence:abc123"
        assert node.public_key == b"\x01" * 32
        assert node.label == "laptop"
        assert node.status == DIDStatus.ACTIVE
        assert node.is_active
        assert node.cluster_id is None

    def test_fingerprint(self):
        node = DIDNode(did="did:valence:deadbeef", public_key=b"\x00" * 32)
        assert node.fingerprint == "deadbeef"

    def test_fingerprint_fallback(self):
        node = DIDNode(did="nodid", public_key=b"\x00" * 32)
        assert node.fingerprint == "nodid"

    def test_is_active_false_when_revoked(self):
        node = DIDNode(
            did="did:valence:x",
            public_key=b"\x00" * 32,
            status=DIDStatus.REVOKED,
        )
        assert not node.is_active

    def test_serialisation_roundtrip(self):
        node = DIDNode(
            did="did:valence:abc",
            public_key=b"\xde\xad" * 16,
            label="phone",
            created_at=1000.0,
            status=DIDStatus.ACTIVE,
            metadata={"os": "linux"},
            cluster_id="cluster-1",
        )
        data = node.to_dict()
        restored = DIDNode.from_dict(data)
        assert restored.did == node.did
        assert restored.public_key == node.public_key
        assert restored.label == node.label
        assert restored.status == node.status
        assert restored.metadata == node.metadata
        assert restored.cluster_id == node.cluster_id

    def test_default_metadata_is_empty_dict(self):
        node = DIDNode(did="did:valence:x", public_key=b"\x00" * 32)
        assert node.metadata == {}

    def test_status_enum_values(self):
        assert DIDStatus.ACTIVE.value == "active"
        assert DIDStatus.REVOKED.value == "revoked"
        assert DIDStatus.SUSPENDED.value == "suspended"


# ---------------------------------------------------------------------------
# LinkProof
# ---------------------------------------------------------------------------


class TestLinkProof:
    """Tests for the LinkProof model."""

    def test_link_statement_is_deterministic(self):
        proof = LinkProof(
            did_a="did:valence:aaa",
            did_b="did:valence:bbb",
            cluster_id="c1",
            nonce=b"\x00" * 16,
        )
        s1 = proof.link_statement()
        s2 = proof.link_statement()
        assert s1 == s2
        assert len(s1) == 32  # SHA-256 digest

    def test_link_statement_order_independent(self):
        """Swapping did_a and did_b should produce the same statement."""
        nonce = uuid.uuid4().bytes
        proof1 = LinkProof(
            did_a="did:valence:aaa",
            did_b="did:valence:bbb",
            cluster_id="c1",
            nonce=nonce,
        )
        proof2 = LinkProof(
            did_a="did:valence:bbb",
            did_b="did:valence:aaa",
            cluster_id="c1",
            nonce=nonce,
        )
        assert proof1.link_statement() == proof2.link_statement()

    def test_different_nonce_changes_statement(self):
        base = dict(did_a="did:valence:a", did_b="did:valence:b", cluster_id="c1")
        p1 = LinkProof(**base, nonce=b"\x00" * 16)
        p2 = LinkProof(**base, nonce=b"\xff" * 16)
        assert p1.link_statement() != p2.link_statement()

    def test_serialisation_roundtrip(self):
        proof = LinkProof(
            proof_id="proof-1",
            did_a="did:valence:aaa",
            did_b="did:valence:bbb",
            cluster_id="c1",
            signature_a=b"\xaa" * 64,
            signature_b=b"\xbb" * 64,
            created_at=2000.0,
            nonce=b"\xcc" * 16,
        )
        data = proof.to_dict()
        restored = LinkProof.from_dict(data)
        assert restored.proof_id == proof.proof_id
        assert restored.did_a == proof.did_a
        assert restored.did_b == proof.did_b
        assert restored.cluster_id == proof.cluster_id
        assert restored.signature_a == proof.signature_a
        assert restored.signature_b == proof.signature_b
        assert restored.nonce == proof.nonce


# ---------------------------------------------------------------------------
# IdentityCluster
# ---------------------------------------------------------------------------


class TestIdentityCluster:
    """Tests for the IdentityCluster model."""

    @pytest.fixture()
    def cluster(self) -> IdentityCluster:
        return IdentityCluster(cluster_id="c1", label="alice")

    @pytest.fixture()
    def node_a(self) -> DIDNode:
        return DIDNode(did="did:valence:aaa", public_key=b"\x01" * 32, label="laptop")

    @pytest.fixture()
    def node_b(self) -> DIDNode:
        return DIDNode(did="did:valence:bbb", public_key=b"\x02" * 32, label="phone")

    def test_add_node(self, cluster: IdentityCluster, node_a: DIDNode):
        cluster.add_node(node_a)
        assert node_a.cluster_id == "c1"
        assert cluster.has_did("did:valence:aaa")
        assert cluster.get_node("did:valence:aaa") is node_a

    def test_active_nodes(self, cluster: IdentityCluster, node_a: DIDNode, node_b: DIDNode):
        node_b.status = DIDStatus.REVOKED
        cluster.add_node(node_a)
        cluster.add_node(node_b)
        active = cluster.active_nodes
        assert len(active) == 1
        assert "did:valence:aaa" in active

    def test_revoked_nodes(self, cluster: IdentityCluster, node_a: DIDNode, node_b: DIDNode):
        node_b.status = DIDStatus.REVOKED
        cluster.add_node(node_a)
        cluster.add_node(node_b)
        revoked = cluster.revoked_nodes
        assert len(revoked) == 1
        assert "did:valence:bbb" in revoked

    def test_remove_node(self, cluster: IdentityCluster, node_a: DIDNode):
        cluster.add_node(node_a)
        removed = cluster.remove_node("did:valence:aaa")
        assert removed is node_a
        assert removed.cluster_id is None
        assert not cluster.has_did("did:valence:aaa")

    def test_remove_nonexistent(self, cluster: IdentityCluster):
        assert cluster.remove_node("did:valence:nope") is None

    def test_add_proof(self, cluster: IdentityCluster):
        proof = LinkProof(did_a="a", did_b="b", cluster_id="c1")
        cluster.add_proof(proof)
        assert len(cluster.proofs) == 1

    def test_serialisation_roundtrip(self, cluster: IdentityCluster, node_a: DIDNode):
        cluster.add_node(node_a)
        proof = LinkProof(
            did_a="did:valence:aaa",
            did_b="did:valence:bbb",
            cluster_id="c1",
            signature_a=b"\xaa" * 64,
            signature_b=b"\xbb" * 64,
            nonce=b"\x00" * 16,
        )
        cluster.add_proof(proof)

        data = cluster.to_dict()
        restored = IdentityCluster.from_dict(data)
        assert restored.cluster_id == cluster.cluster_id
        assert restored.label == cluster.label
        assert len(restored.nodes) == 1
        assert len(restored.proofs) == 1
        assert restored.nodes["did:valence:aaa"].label == "laptop"
