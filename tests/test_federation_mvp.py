"""Tests for Federation MVP components."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from oro_federation.identity import (
    DIDMethod,
    create_key_did,
    generate_keypair,
    parse_did,
    sign_belief_content,
    verify_belief_signature,
)
from oro_federation.peers import PeerStore


class TestIdentity:
    """Tests for identity module."""

    def test_generate_keypair(self):
        """Test keypair generation."""
        keypair = generate_keypair()

        assert keypair.private_key_bytes is not None
        assert keypair.public_key_bytes is not None
        assert len(keypair.private_key_bytes) == 32
        assert len(keypair.public_key_bytes) == 32

    def test_keypair_multibase(self):
        """Test multibase encoding of public key."""
        keypair = generate_keypair()
        multibase = keypair.public_key_multibase

        # Should start with 'z' (base58btc)
        assert multibase.startswith("z")

    def test_create_key_did(self):
        """Test DID creation from keypair."""
        keypair = generate_keypair()
        did = create_key_did(keypair.public_key_multibase)

        assert did.method == DIDMethod.KEY
        assert did.full.startswith("did:vkb:key:z")

    def test_parse_key_did(self):
        """Test parsing a key DID."""
        keypair = generate_keypair()
        did = create_key_did(keypair.public_key_multibase)

        parsed = parse_did(did.full)

        assert parsed.method == DIDMethod.KEY
        assert parsed.identifier == keypair.public_key_multibase

    def test_sign_and_verify(self):
        """Test signing and verifying beliefs."""
        keypair = generate_keypair()

        content = {
            "id": "test-belief-123",
            "content": "This is a test belief",
            "confidence": 0.9,
            "domains": ["test"],
            "origin_did": "did:vkb:key:test",
        }

        # Sign
        signature = sign_belief_content(content, keypair.private_key_bytes)

        assert signature is not None
        assert len(signature) > 0

        # Verify
        is_valid = verify_belief_signature(content, signature, keypair.public_key_multibase)

        assert is_valid is True

    def test_verify_invalid_signature(self):
        """Test that invalid signatures are rejected."""
        keypair1 = generate_keypair()
        keypair2 = generate_keypair()

        content = {
            "id": "test-belief-123",
            "content": "This is a test belief",
            "confidence": 0.9,
            "domains": ["test"],
            "origin_did": "did:vkb:key:test",
        }

        # Sign with keypair1
        signature = sign_belief_content(content, keypair1.private_key_bytes)

        # Verify with keypair2's public key (should fail)
        is_valid = verify_belief_signature(content, signature, keypair2.public_key_multibase)

        assert is_valid is False

    def test_verify_tampered_content(self):
        """Test that tampered content is rejected."""
        keypair = generate_keypair()

        content = {
            "id": "test-belief-123",
            "content": "This is a test belief",
            "confidence": 0.9,
            "domains": ["test"],
            "origin_did": "did:vkb:key:test",
        }

        # Sign
        signature = sign_belief_content(content, keypair.private_key_bytes)

        # Tamper with content
        content["content"] = "This has been tampered with!"

        # Verify (should fail)
        is_valid = verify_belief_signature(content, signature, keypair.public_key_multibase)

        assert is_valid is False


class TestPeerStore:
    """Tests for peer storage."""

    def test_add_peer(self):
        """Test adding a peer."""
        store = PeerStore()

        peer = store.add_peer(
            did="did:vkb:key:z123",
            endpoint="http://localhost:8000",
            public_key_multibase="z123",
            name="Test Node",
        )

        assert peer.did == "did:vkb:key:z123"
        assert peer.name == "Test Node"
        assert peer.trust_score == 0.5

    def test_get_peer(self):
        """Test getting a peer."""
        store = PeerStore()

        store.add_peer(
            did="did:vkb:key:z123",
            endpoint="http://localhost:8000",
            public_key_multibase="z123",
        )

        peer = store.get_peer("did:vkb:key:z123")

        assert peer is not None
        assert peer.did == "did:vkb:key:z123"

    def test_list_peers(self):
        """Test listing peers."""
        store = PeerStore()

        store.add_peer(did="did:vkb:key:z1", endpoint="http://a", public_key_multibase="z1")
        store.add_peer(did="did:vkb:key:z2", endpoint="http://b", public_key_multibase="z2")
        store.add_peer(did="did:vkb:key:z3", endpoint="http://c", public_key_multibase="z3")

        peers = store.list_peers()

        assert len(peers) == 3

    def test_update_trust(self):
        """Test updating trust scores."""
        store = PeerStore()

        store.add_peer(
            did="did:vkb:key:z123",
            endpoint="http://localhost:8000",
            public_key_multibase="z123",
        )

        # Increase trust
        new_trust = store.update_trust("did:vkb:key:z123", 0.1)

        assert abs(new_trust - 0.6) < 0.001

        # Decrease trust
        new_trust = store.update_trust("did:vkb:key:z123", -0.2)

        assert abs(new_trust - 0.4) < 0.001

    def test_trust_bounds(self):
        """Test that trust is bounded to [0, 1]."""
        store = PeerStore()

        store.add_peer(
            did="did:vkb:key:z123",
            endpoint="http://localhost:8000",
            public_key_multibase="z123",
        )

        # Try to exceed 1.0
        new_trust = store.update_trust("did:vkb:key:z123", 1.0)
        assert new_trust == 1.0

        # Reset and try to go below 0.0
        store._peers["did:vkb:key:z123"].trust_score = 0.1
        new_trust = store.update_trust("did:vkb:key:z123", -0.5)
        assert new_trust == 0.0

    def test_record_beliefs(self):
        """Test recording belief exchanges."""
        store = PeerStore()

        store.add_peer(
            did="did:vkb:key:z123",
            endpoint="http://localhost:8000",
            public_key_multibase="z123",
        )

        store.record_belief_sent("did:vkb:key:z123")
        store.record_belief_sent("did:vkb:key:z123")
        store.record_belief_received("did:vkb:key:z123")

        peer = store.get_peer("did:vkb:key:z123")

        assert peer.beliefs_sent == 2
        assert peer.beliefs_received == 1
        # Trust should have increased from receiving a belief
        assert peer.trust_score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
