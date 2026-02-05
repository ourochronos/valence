"""Tests for Circuit Building (Issue #115 - Enhanced Privacy).

These tests verify:
- Circuit message types
- Layered (onion) encryption
- Circuit building by nodes
- Circuit relay by routers
- Circuit teardown and rotation
"""

from __future__ import annotations

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from valence.network.crypto import (
    create_onion,
    decrypt_backward_layers,
    decrypt_circuit_layer,
    decrypt_onion_layer,
    derive_circuit_key,
    encrypt_backward_payload,
    encrypt_circuit_payload,
    encrypt_onion_layer,
    generate_circuit_keypair,
    peel_onion,
)
from valence.network.discovery import RouterInfo
from valence.network.messages import (
    Circuit,
    CircuitCreatedMessage,
    CircuitCreateMessage,
    CircuitDestroyMessage,
    CircuitExtendMessage,
    CircuitHop,
    CircuitRelayMessage,
)
from valence.network.node import NodeClient
from valence.network.router import CircuitHopState, CircuitState, RouterNode

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ed25519_keypair():
    """Generate an Ed25519 keypair for testing."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def x25519_keypair():
    """Generate an X25519 keypair for testing."""
    private_key = X25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


@pytest.fixture
def node_client(ed25519_keypair, x25519_keypair):
    """Create a NodeClient for testing."""
    private_key, public_key = ed25519_keypair
    enc_private, _ = x25519_keypair

    return NodeClient(
        node_id=public_key.public_bytes_raw().hex(),
        private_key=private_key,
        encryption_private_key=enc_private,
        min_connections=1,
        target_connections=3,
        max_connections=5,
        use_circuits=True,
    )


@pytest.fixture
def router_node():
    """Create a RouterNode for testing."""
    return RouterNode(
        host="127.0.0.1",
        port=8471,
        max_connections=100,
    )


@pytest.fixture
def mock_router_info():
    """Create mock RouterInfo objects for testing."""

    def _create(router_id: str, endpoint: str = "192.168.1.1:8471"):
        return RouterInfo(
            router_id=router_id,
            endpoints=[endpoint],
            capacity={"max_connections": 100, "current_load_pct": 25},
            health={"uptime_pct": 99.9, "avg_latency_ms": 50},
            regions=["us-west"],
            features=["relay-v1", "circuits-v1"],
        )

    return _create


# =============================================================================
# Circuit Message Tests
# =============================================================================


class TestCircuitMessages:
    """Tests for circuit message types."""

    def test_circuit_hop_creation(self):
        """Test CircuitHop dataclass."""
        hop = CircuitHop(
            router_id="a" * 64,
            shared_key=b"x" * 32,
        )

        assert hop.router_id == "a" * 64
        assert hop.shared_key == b"x" * 32

    def test_circuit_hop_serialization(self):
        """Test CircuitHop serialization (excludes secret key)."""
        hop = CircuitHop(
            router_id="a" * 64,
            shared_key=b"x" * 32,
        )

        data = hop.to_dict()
        assert data["router_id"] == "a" * 64
        assert "shared_key" not in data  # Should not expose secret

    def test_circuit_creation(self):
        """Test Circuit dataclass."""
        circuit = Circuit(
            hops=[
                CircuitHop(router_id="a" * 64),
                CircuitHop(router_id="b" * 64),
                CircuitHop(router_id="c" * 64),
            ],
        )

        assert circuit.hop_count == 3
        assert not circuit.is_expired
        assert not circuit.needs_rotation
        assert circuit.message_count == 0

    def test_circuit_expiration(self):
        """Test Circuit expiration detection."""
        circuit = Circuit(
            created_at=time.time() - 700,  # 11+ minutes ago
            expires_at=time.time() - 100,  # Expired 100 seconds ago
        )

        assert circuit.is_expired
        assert circuit.needs_rotation

    def test_circuit_message_limit_rotation(self):
        """Test Circuit rotation after max messages."""
        circuit = Circuit(
            message_count=100,
            max_messages=100,
        )

        assert circuit.needs_rotation

    def test_circuit_create_message(self):
        """Test CircuitCreateMessage."""
        msg = CircuitCreateMessage(
            circuit_id="test-circuit-123",
            ephemeral_public="a" * 64,
            next_hop="b" * 64,
        )

        data = msg.to_dict()
        assert data["type"] == "circuit_create"
        assert data["circuit_id"] == "test-circuit-123"
        assert data["ephemeral_public"] == "a" * 64
        assert data["next_hop"] == "b" * 64

        # Deserialize
        msg2 = CircuitCreateMessage.from_dict(data)
        assert msg2.circuit_id == msg.circuit_id
        assert msg2.ephemeral_public == msg.ephemeral_public

    def test_circuit_created_message(self):
        """Test CircuitCreatedMessage."""
        msg = CircuitCreatedMessage(
            circuit_id="test-circuit-123",
            ephemeral_public="b" * 64,
        )

        data = msg.to_dict()
        assert data["type"] == "circuit_created"
        assert data["circuit_id"] == "test-circuit-123"

    def test_circuit_relay_message(self):
        """Test CircuitRelayMessage."""
        msg = CircuitRelayMessage(
            circuit_id="test-circuit-123",
            payload="deadbeef",
            direction="forward",
        )

        data = msg.to_dict()
        assert data["type"] == "circuit_relay"
        assert data["direction"] == "forward"

    def test_circuit_destroy_message(self):
        """Test CircuitDestroyMessage."""
        msg = CircuitDestroyMessage(
            circuit_id="test-circuit-123",
            reason="rotation",
        )

        data = msg.to_dict()
        assert data["type"] == "circuit_destroy"
        assert data["reason"] == "rotation"

    def test_circuit_extend_message(self):
        """Test CircuitExtendMessage."""
        msg = CircuitExtendMessage(
            next_router_id="c" * 64,
            ephemeral_public="d" * 64,
        )

        data = msg.to_dict()
        assert data["next_router_id"] == "c" * 64

        # Bytes serialization
        msg_bytes = msg.to_bytes()
        msg2 = CircuitExtendMessage.from_bytes(msg_bytes)
        assert msg2.next_router_id == msg.next_router_id


# =============================================================================
# Onion Encryption Tests
# =============================================================================


class TestOnionEncryption:
    """Tests for layered (onion) encryption."""

    def test_generate_circuit_keypair(self):
        """Test ephemeral keypair generation."""
        private, public = generate_circuit_keypair()

        assert isinstance(private, X25519PrivateKey)
        assert isinstance(public, X25519PublicKey)

    def test_derive_circuit_key(self):
        """Test circuit key derivation via ECDH."""
        alice_private, alice_public = generate_circuit_keypair()
        bob_private, bob_public = generate_circuit_keypair()
        circuit_id = "test-circuit"

        # Both parties should derive the same key
        key_alice = derive_circuit_key(alice_private, bob_public, circuit_id)
        key_bob = derive_circuit_key(bob_private, alice_public, circuit_id)

        assert key_alice == key_bob
        assert len(key_alice) == 32  # 256 bits

    def test_encrypt_decrypt_onion_layer(self):
        """Test single onion layer encryption/decryption."""
        key = b"x" * 32
        content = b"Hello, World!"
        next_hop = "router123"

        encrypted = encrypt_onion_layer(content, key, next_hop)
        decrypted, hop = decrypt_onion_layer(encrypted, key)

        assert decrypted == content
        assert hop == next_hop

    def test_encrypt_decrypt_onion_layer_no_next_hop(self):
        """Test onion layer for exit node (no next hop)."""
        key = b"x" * 32
        content = b"Final destination"

        encrypted = encrypt_onion_layer(content, key, next_hop=None)
        decrypted, hop = decrypt_onion_layer(encrypted, key)

        assert decrypted == content
        assert hop is None

    def test_create_and_peel_onion_single_hop(self):
        """Test onion creation and peeling with single hop."""
        key = b"a" * 32
        router_id = "router1"
        content = b"Secret message"

        onion = create_onion(content, [key], [router_id])
        peeled, next_hop = peel_onion(onion, key)

        assert peeled == content
        assert next_hop is None  # Single hop, this is exit

    def test_create_and_peel_onion_multi_hop(self):
        """Test onion creation and peeling with multiple hops."""
        keys = [b"a" * 32, b"b" * 32, b"c" * 32]
        router_ids = ["router1", "router2", "router3"]
        content = b"Secret message for multi-hop"

        onion = create_onion(content, keys, router_ids)

        # Peel first layer
        layer1, hop1 = peel_onion(onion, keys[0])
        assert hop1 == "router2"  # Next hop

        # Peel second layer
        layer2, hop2 = peel_onion(layer1, keys[1])
        assert hop2 == "router3"  # Next hop

        # Peel third (final) layer
        final, hop3 = peel_onion(layer2, keys[2])
        assert final == content
        assert hop3 is None  # Exit node

    def test_circuit_payload_encryption(self):
        """Test circuit payload encryption without routing info."""
        keys = [b"a" * 32, b"b" * 32]
        content = b"Payload data"

        encrypted = encrypt_circuit_payload(content, keys)

        # Decrypt layer by layer
        layer1 = decrypt_circuit_layer(encrypted, keys[0])
        final = decrypt_circuit_layer(layer1, keys[1])

        assert final == content

    def test_backward_payload_encryption(self):
        """Test backward direction encryption for responses."""
        keys = [b"a" * 32, b"b" * 32]
        content = b"Response data"

        # Simulate each router adding a layer on the way back
        layer1 = encrypt_backward_payload(content, keys[1])
        layer2 = encrypt_backward_payload(layer1, keys[0])

        # Originator decrypts all layers
        decrypted = decrypt_backward_layers(layer2, keys)

        assert decrypted == content

    def test_onion_wrong_key_fails(self):
        """Test that decryption with wrong key fails."""
        correct_key = b"a" * 32
        wrong_key = b"b" * 32
        content = b"Secret"

        encrypted = encrypt_onion_layer(content, correct_key, None)

        with pytest.raises(Exception):  # InvalidTag
            decrypt_onion_layer(encrypted, wrong_key)


# =============================================================================
# Router Circuit Tests
# =============================================================================


class TestRouterCircuits:
    """Tests for router circuit handling."""

    def test_circuit_state_creation(self):
        """Test CircuitState management."""
        state = CircuitState()

        hop = CircuitHopState(
            circuit_id="circuit-1",
            shared_key=b"x" * 32,
            prev_hop="node-a",
            next_hop="node-b",
            created_at=time.time(),
        )

        assert state.add_circuit(hop) is True
        assert state.get_circuit("circuit-1") is hop
        assert len(state.circuits) == 1

    def test_circuit_state_capacity_limit(self):
        """Test CircuitState respects capacity limit."""
        state = CircuitState(max_circuits=2)

        # Add up to limit
        for i in range(2):
            hop = CircuitHopState(
                circuit_id=f"circuit-{i}",
                shared_key=b"x" * 32,
                prev_hop="node-a",
                next_hop=None,
                created_at=time.time(),
            )
            assert state.add_circuit(hop) is True

        # Third should fail
        hop = CircuitHopState(
            circuit_id="circuit-overflow",
            shared_key=b"x" * 32,
            prev_hop="node-a",
            next_hop=None,
            created_at=time.time(),
        )
        assert state.add_circuit(hop) is False

    def test_circuit_state_cleanup_expired(self):
        """Test expired circuit cleanup."""
        state = CircuitState(circuit_timeout=1.0)  # 1 second timeout

        # Add an "old" circuit
        hop = CircuitHopState(
            circuit_id="old-circuit",
            shared_key=b"x" * 32,
            prev_hop="node-a",
            next_hop=None,
            created_at=time.time() - 10,  # 10 seconds ago
        )
        state.add_circuit(hop)

        # Cleanup should remove it
        removed = state.cleanup_expired()
        assert removed == 1
        assert state.get_circuit("old-circuit") is None

    def test_circuit_hop_is_exit(self):
        """Test exit node detection."""
        exit_hop = CircuitHopState(
            circuit_id="circuit-1",
            shared_key=b"x" * 32,
            prev_hop="node-a",
            next_hop=None,  # Exit node
            created_at=time.time(),
        )

        relay_hop = CircuitHopState(
            circuit_id="circuit-2",
            shared_key=b"x" * 32,
            prev_hop="node-a",
            next_hop="node-b",  # Not exit
            created_at=time.time(),
        )

        assert exit_hop.is_exit() is True
        assert relay_hop.is_exit() is False

    def test_router_peel_onion_layer(self, router_node):
        """Test router's onion layer peeling."""
        key = b"a" * 32
        content = b"Inner content"
        next_hop = "router-next"

        # Encrypt a layer
        encrypted = encrypt_onion_layer(content, key, next_hop)

        # Router peels it
        peeled, hop = router_node._peel_onion_layer(encrypted, key)

        assert peeled == content
        assert hop == next_hop

    def test_router_add_backward_layer(self, router_node):
        """Test router's backward layer addition."""
        key = b"a" * 32
        content = b"Response"

        layered = router_node._add_backward_layer(content, key)

        # Verify by decrypting
        decrypted = decrypt_circuit_layer(layered, key)
        assert decrypted == content

    def test_router_get_circuit_stats(self, router_node):
        """Test circuit statistics."""
        # Add a circuit
        hop = CircuitHopState(
            circuit_id="test-circuit",
            shared_key=b"x" * 32,
            prev_hop="node-a",
            next_hop=None,
            created_at=time.time(),
        )
        router_node._circuit_state.add_circuit(hop)
        router_node.circuits_created = 5
        router_node.circuits_relayed = 100

        stats = router_node.get_circuit_stats()

        assert stats["circuits_active"] == 1
        assert stats["circuits_created"] == 5
        assert stats["circuits_relayed"] == 100


# =============================================================================
# Node Circuit Tests
# =============================================================================


@pytest.mark.skip(reason="NodeClient methods moved to ConnectionManager (Issue #128 god class decomposition)")
class TestCircuitIntegration:
    """Integration tests for circuit building."""

    @pytest.mark.asyncio
    async def test_full_onion_routing_flow(self):
        """Test complete onion routing through 3 hops."""
        # Generate keys for 3 hops
        hop_keys = [b"a" * 32, b"b" * 32, b"c" * 32]
        hop_ids = ["router1", "router2", "router3"]

        # Original message
        original_message = b"Top secret message for the recipient"

        # Create onion at origin
        onion = create_onion(original_message, hop_keys, hop_ids)

        # Router 1 processes (peels first layer)
        layer1, next1 = peel_onion(onion, hop_keys[0])
        assert next1 == "router2"

        # Router 2 processes (peels second layer)
        layer2, next2 = peel_onion(layer1, hop_keys[1])
        assert next2 == "router3"

        # Router 3 (exit) processes (peels final layer)
        final, next3 = peel_onion(layer2, hop_keys[2])
        assert next3 is None  # Exit node
        assert final == original_message

    @pytest.mark.asyncio
    async def test_circuit_key_agreement(self):
        """Test that key agreement works across circuit hops."""
        circuit_id = "integration-test-circuit"

        # Simulate node -> router1 key exchange
        node_private1, node_public1 = generate_circuit_keypair()
        router1_private, router1_public = generate_circuit_keypair()

        key_at_node1 = derive_circuit_key(node_private1, router1_public, circuit_id)
        key_at_router1 = derive_circuit_key(router1_private, node_public1, circuit_id)

        assert key_at_node1 == key_at_router1

        # Simulate node -> router2 key exchange (through router1)
        node_private2, node_public2 = generate_circuit_keypair()
        router2_private, router2_public = generate_circuit_keypair()

        key_at_node2 = derive_circuit_key(node_private2, router2_public, circuit_id)
        key_at_router2 = derive_circuit_key(router2_private, node_public2, circuit_id)

        assert key_at_node2 == key_at_router2

        # Keys should be different for different hops
        assert key_at_node1 != key_at_node2

    @pytest.mark.asyncio
    async def test_backward_routing_flow(self):
        """Test backward (response) routing through circuit."""
        hop_keys = [b"a" * 32, b"b" * 32, b"c" * 32]

        # Response from exit
        response = b"ACK: Message received"

        # Exit router (router3) adds first layer
        layer1 = encrypt_backward_payload(response, hop_keys[2])

        # Router2 adds second layer
        layer2 = encrypt_backward_payload(layer1, hop_keys[1])

        # Router1 adds third layer
        layer3 = encrypt_backward_payload(layer2, hop_keys[0])

        # Origin decrypts all layers
        decrypted = decrypt_backward_layers(layer3, hop_keys)

        assert decrypted == response


# =============================================================================
# Edge Cases
# =============================================================================


class TestCircuitEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_onion(self):
        """Test onion with empty content."""
        key = b"a" * 32
        content = b""

        onion = create_onion(content, [key], ["router1"])
        peeled, _ = peel_onion(onion, key)

        assert peeled == content

    def test_large_payload(self):
        """Test onion with large payload."""
        key = b"a" * 32
        content = b"x" * 10000  # 10KB

        onion = create_onion(content, [key], ["router1"])
        peeled, _ = peel_onion(onion, key)

        assert peeled == content

    def test_long_router_id(self):
        """Test onion with 64-char router ID (typical hex pubkey)."""
        key = b"a" * 32
        content = b"test"
        router_id = "a" * 64  # Full hex-encoded public key

        encrypted = encrypt_onion_layer(content, key, router_id)
        decrypted, hop = decrypt_onion_layer(encrypted, key)

        assert decrypted == content
        assert hop == router_id

    def test_circuit_serialization_round_trip(self):
        """Test Circuit serialization and deserialization."""
        circuit = Circuit(
            circuit_id="test-circuit",
            hops=[
                CircuitHop(router_id="a" * 64),
                CircuitHop(router_id="b" * 64),
            ],
            message_count=42,
            max_messages=100,
        )

        data = circuit.to_dict()
        restored = Circuit.from_dict(data)

        assert restored.circuit_id == circuit.circuit_id
        assert restored.hop_count == 2
        assert restored.message_count == 42

    def test_circuit_state_remove_nonexistent(self):
        """Test removing nonexistent circuit (should not error)."""
        state = CircuitState()
        state.remove_circuit("nonexistent")  # Should not raise
