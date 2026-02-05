"""Tests for VRF (Verifiable Random Function) implementation.

Tests cover:
- VRF key generation
- VRF proof computation
- VRF verification
- Epoch seed derivation
- Determinism properties
- Security properties
"""

from __future__ import annotations

import hashlib

import pytest

# Check if cryptography is available
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,  # noqa: F401
    )

    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from valence.consensus.vrf import (
    VRF,
    VRF_OUTPUT_SIZE,
    VRFOutput,
    VRFProof,
    compute_selection_ticket,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def vrf_instance():
    """Create a VRF instance for testing."""
    if not CRYPTO_AVAILABLE:
        pytest.skip("cryptography library not available")
    return VRF.generate()


@pytest.fixture
def sample_seed():
    """Sample epoch seed."""
    return hashlib.sha256(b"test_epoch_seed_123").digest()


@pytest.fixture
def sample_fingerprint():
    """Sample agent fingerprint."""
    return hashlib.sha256(b"did:vkb:key:z6MkTestAgent").digest()


# =============================================================================
# VRF KEY GENERATION TESTS
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestVRFGeneration:
    """Tests for VRF key generation."""

    def test_generate_creates_valid_instance(self):
        """VRF.generate() should create a valid VRF instance."""
        vrf = VRF.generate()
        assert vrf is not None
        assert len(vrf.public_key_bytes) == 32
        assert len(vrf.private_key_bytes) == 32

    def test_generate_creates_unique_keys(self):
        """Each generated VRF should have unique keys."""
        vrf1 = VRF.generate()
        vrf2 = VRF.generate()
        assert vrf1.public_key_bytes != vrf2.public_key_bytes
        assert vrf1.private_key_bytes != vrf2.private_key_bytes

    def test_init_from_private_key_bytes(self):
        """VRF should be reconstructable from private key bytes."""
        vrf1 = VRF.generate()
        vrf2 = VRF(private_key_bytes=vrf1.private_key_bytes)
        assert vrf1.public_key_bytes == vrf2.public_key_bytes

    def test_init_rejects_invalid_key_length(self):
        """VRF should reject keys with invalid length."""
        with pytest.raises(ValueError, match="32 bytes"):
            VRF(private_key_bytes=b"too_short")

    def test_init_requires_key(self):
        """VRF should require either private_key or private_key_bytes."""
        with pytest.raises(ValueError, match="must be provided"):
            VRF()


# =============================================================================
# VRF PROOF TESTS
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestVRFProof:
    """Tests for VRF proof computation."""

    def test_prove_returns_vrf_output(self, vrf_instance):
        """VRF.prove() should return a VRFOutput."""
        output = vrf_instance.prove(b"test_input")
        assert isinstance(output, VRFOutput)
        assert isinstance(output.proof, VRFProof)

    def test_prove_produces_correct_ticket_length(self, vrf_instance):
        """Ticket should be VRF_OUTPUT_SIZE bytes."""
        output = vrf_instance.prove(b"test_input")
        assert len(output.ticket) == VRF_OUTPUT_SIZE

    def test_prove_is_deterministic(self, vrf_instance):
        """Same input should produce same output."""
        output1 = vrf_instance.prove(b"same_input")
        output2 = vrf_instance.prove(b"same_input")
        assert output1.ticket == output2.ticket

    def test_prove_different_inputs_produce_different_outputs(self, vrf_instance):
        """Different inputs should produce different outputs."""
        output1 = vrf_instance.prove(b"input_1")
        output2 = vrf_instance.prove(b"input_2")
        assert output1.ticket != output2.ticket

    def test_prove_different_keys_produce_different_outputs(self):
        """Different keys with same input should produce different outputs."""
        vrf1 = VRF.generate()
        vrf2 = VRF.generate()
        output1 = vrf1.prove(b"same_input")
        output2 = vrf2.prove(b"same_input")
        assert output1.ticket != output2.ticket


# =============================================================================
# VRF VERIFICATION TESTS
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestVRFVerification:
    """Tests for VRF verification."""

    def test_verify_valid_proof(self, vrf_instance):
        """Valid proof should verify successfully."""
        alpha = b"test_input"
        output = vrf_instance.prove(alpha)
        assert VRF.verify(vrf_instance.public_key_bytes, alpha, output)

    def test_verify_rejects_wrong_input(self, vrf_instance):
        """Proof for different input should fail verification."""
        output = vrf_instance.prove(b"original_input")
        assert not VRF.verify(vrf_instance.public_key_bytes, b"wrong_input", output)

    def test_verify_rejects_wrong_public_key(self, vrf_instance):
        """Proof verified with wrong key should fail."""
        output = vrf_instance.prove(b"test_input")
        wrong_vrf = VRF.generate()
        assert not VRF.verify(wrong_vrf.public_key_bytes, b"test_input", output)

    def test_verify_rejects_tampered_ticket(self, vrf_instance):
        """Tampered ticket should fail verification."""
        alpha = b"test_input"
        output = vrf_instance.prove(alpha)
        # Tamper with the ticket
        tampered_output = VRFOutput(
            ticket=bytes([b ^ 0xFF for b in output.ticket]),  # Flip all bits
            proof=output.proof,
            input_hash=output.input_hash,
        )
        assert not VRF.verify(vrf_instance.public_key_bytes, alpha, tampered_output)

    def test_verify_rejects_tampered_proof(self, vrf_instance):
        """Tampered proof should fail verification."""
        alpha = b"test_input"
        output = vrf_instance.prove(alpha)
        # Tamper with the proof
        tampered_proof = VRFProof(
            gamma=bytes([b ^ 0xFF for b in output.proof.gamma]),
            c=output.proof.c,
            s=output.proof.s,
        )
        tampered_output = VRFOutput(
            ticket=output.ticket,
            proof=tampered_proof,
            input_hash=output.input_hash,
        )
        assert not VRF.verify(vrf_instance.public_key_bytes, alpha, tampered_output)


# =============================================================================
# EPOCH SEED TESTS
# =============================================================================


class TestEpochSeed:
    """Tests for epoch seed derivation."""

    def test_derive_epoch_seed_is_deterministic(self):
        """Same inputs should produce same seed."""
        prev_seed = b"a" * 32
        block_hash = b"b" * 32
        epoch = 42

        seed1 = VRF.derive_epoch_seed(prev_seed, block_hash, epoch)
        seed2 = VRF.derive_epoch_seed(prev_seed, block_hash, epoch)
        assert seed1 == seed2

    def test_derive_epoch_seed_correct_length(self):
        """Epoch seed should be 32 bytes."""
        seed = VRF.derive_epoch_seed(b"a" * 32, b"b" * 32, 1)
        assert len(seed) == 32

    def test_derive_epoch_seed_different_epoch_numbers(self):
        """Different epoch numbers should produce different seeds."""
        prev_seed = b"a" * 32
        block_hash = b"b" * 32

        seed1 = VRF.derive_epoch_seed(prev_seed, block_hash, 1)
        seed2 = VRF.derive_epoch_seed(prev_seed, block_hash, 2)
        assert seed1 != seed2

    def test_derive_epoch_seed_different_previous_seeds(self):
        """Different previous seeds should produce different new seeds."""
        block_hash = b"b" * 32

        seed1 = VRF.derive_epoch_seed(b"a" * 32, block_hash, 1)
        seed2 = VRF.derive_epoch_seed(b"c" * 32, block_hash, 1)
        assert seed1 != seed2

    def test_derive_epoch_seed_different_block_hashes(self):
        """Different block hashes should produce different seeds."""
        prev_seed = b"a" * 32

        seed1 = VRF.derive_epoch_seed(prev_seed, b"b" * 32, 1)
        seed2 = VRF.derive_epoch_seed(prev_seed, b"c" * 32, 1)
        assert seed1 != seed2

    def test_genesis_seed_is_deterministic(self):
        """Genesis seed should be deterministic."""
        seed1 = VRF.genesis_seed()
        seed2 = VRF.genesis_seed()
        assert seed1 == seed2
        assert len(seed1) == 32


# =============================================================================
# SELECTION TICKET TESTS
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestSelectionTicket:
    """Tests for selection ticket computation."""

    def test_compute_selection_ticket_returns_output(self, vrf_instance, sample_seed, sample_fingerprint):
        """compute_selection_ticket should return VRFOutput."""
        output = compute_selection_ticket(vrf_instance, sample_seed, sample_fingerprint)
        assert isinstance(output, VRFOutput)

    def test_selection_ticket_is_deterministic(self, vrf_instance, sample_seed, sample_fingerprint):
        """Same inputs should produce same ticket."""
        output1 = compute_selection_ticket(vrf_instance, sample_seed, sample_fingerprint)
        output2 = compute_selection_ticket(vrf_instance, sample_seed, sample_fingerprint)
        assert output1.ticket == output2.ticket

    def test_selection_ticket_different_seeds(self, vrf_instance, sample_fingerprint):
        """Different seeds should produce different tickets."""
        seed1 = hashlib.sha256(b"seed_1").digest()
        seed2 = hashlib.sha256(b"seed_2").digest()

        output1 = compute_selection_ticket(vrf_instance, seed1, sample_fingerprint)
        output2 = compute_selection_ticket(vrf_instance, seed2, sample_fingerprint)
        assert output1.ticket != output2.ticket

    def test_selection_ticket_different_fingerprints(self, vrf_instance, sample_seed):
        """Different agent fingerprints should produce different tickets."""
        fp1 = hashlib.sha256(b"agent_1").digest()
        fp2 = hashlib.sha256(b"agent_2").digest()

        output1 = compute_selection_ticket(vrf_instance, sample_seed, fp1)
        output2 = compute_selection_ticket(vrf_instance, sample_seed, fp2)
        assert output1.ticket != output2.ticket


# =============================================================================
# VRF OUTPUT TESTS
# =============================================================================


class TestVRFOutput:
    """Tests for VRFOutput dataclass."""

    def test_ticket_as_int(self):
        """ticket_as_int should convert to integer correctly."""
        ticket = bytes([0x00, 0x00, 0x00, 0x01])  # 4 bytes = 1
        proof = VRFProof(gamma=b"a" * 32, c=b"b" * 32, s=b"c" * 32)
        output = VRFOutput(ticket=ticket, proof=proof, input_hash=b"d" * 32)
        assert output.ticket_as_int() == 1

    def test_ticket_as_float_range(self):
        """ticket_as_float should return value in [0, 1)."""
        proof = VRFProof(gamma=b"a" * 32, c=b"b" * 32, s=b"c" * 32)

        # Test with various ticket values
        for i in range(10):
            ticket = hashlib.sha256(f"test_{i}".encode()).digest()
            output = VRFOutput(ticket=ticket, proof=proof, input_hash=b"d" * 32)
            f = output.ticket_as_float()
            assert 0.0 <= f < 1.0

    def test_to_dict_and_from_dict(self):
        """Serialization should be reversible."""
        proof = VRFProof(gamma=b"a" * 32, c=b"b" * 32, s=b"c" * 32)
        output = VRFOutput(
            ticket=b"t" * 32,
            proof=proof,
            input_hash=b"i" * 32,
        )

        data = output.to_dict()
        restored = VRFOutput.from_dict(data)

        assert restored.ticket == output.ticket
        assert restored.proof.gamma == output.proof.gamma
        assert restored.input_hash == output.input_hash


# =============================================================================
# VRF PROOF SERIALIZATION TESTS
# =============================================================================


class TestVRFProofSerialization:
    """Tests for VRFProof serialization."""

    def test_to_bytes_length(self):
        """Serialized proof should be 96 bytes."""
        proof = VRFProof(gamma=b"a" * 32, c=b"b" * 32, s=b"c" * 32)
        assert len(proof.to_bytes()) == 96

    def test_from_bytes_round_trip(self):
        """Deserialization should reverse serialization."""
        original = VRFProof(gamma=b"g" * 32, c=b"c" * 32, s=b"s" * 32)
        data = original.to_bytes()
        restored = VRFProof.from_bytes(data)

        assert restored.gamma == original.gamma
        assert restored.c == original.c
        assert restored.s == original.s

    def test_from_bytes_rejects_invalid_length(self):
        """from_bytes should reject invalid length."""
        with pytest.raises(ValueError, match="Invalid proof length"):
            VRFProof.from_bytes(b"too_short")

    def test_to_dict_and_from_dict(self):
        """Dictionary serialization should be reversible."""
        original = VRFProof(gamma=b"g" * 32, c=b"c" * 32, s=b"s" * 32)
        data = original.to_dict()
        restored = VRFProof.from_dict(data)

        assert restored.gamma == original.gamma
        assert restored.c == original.c
        assert restored.s == original.s


# =============================================================================
# UNPREDICTABILITY TESTS
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestVRFUnpredictability:
    """Tests verifying unpredictability properties."""

    def test_ticket_distribution_appears_uniform(self):
        """VRF tickets should appear uniformly distributed."""
        vrf = VRF.generate()

        # Generate many tickets
        tickets = []
        for i in range(100):
            output = vrf.prove(f"input_{i}".encode())
            tickets.append(output.ticket_as_float())

        # Check distribution (crude uniformity test)
        # Divide into 4 quartiles, each should have ~25 values
        quartiles = [0, 0, 0, 0]
        for t in tickets:
            idx = min(int(t * 4), 3)
            quartiles[idx] += 1

        # Each quartile should have between 10 and 40 (25 ± 15)
        # Wider bounds to reduce flakiness from natural random variance
        for count in quartiles:
            assert 10 <= count <= 40, f"Non-uniform distribution: {quartiles}"

    def test_no_pattern_between_sequential_inputs(self):
        """Sequential inputs should not produce predictable outputs."""
        vrf = VRF.generate()

        outputs = [vrf.prove(f"seq_{i}".encode()) for i in range(10)]
        tickets = [o.ticket_as_int() for o in outputs]

        # Check that consecutive tickets aren't consistently increasing or decreasing
        increasing = sum(1 for i in range(len(tickets) - 1) if tickets[i + 1] > tickets[i])
        # Should be roughly 50% increasing (4-6 out of 9)
        assert 2 <= increasing <= 7, f"Pattern detected: {increasing}/9 increasing"
