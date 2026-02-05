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


# =============================================================================
# SECURITY PROPERTY TESTS
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestVRFSecurityProperties:
    """Tests verifying security properties of the simplified VRF construction.

    These tests verify the properties documented in docs/consensus/VRF_SECURITY.md:
    - Determinism (same key + input = same output)
    - Uniqueness (different keys = different outputs)
    - Domain separation (prevents cross-protocol attacks)
    - Proof binding (proof is cryptographically bound to ticket)
    """

    def test_determinism_across_instances(self):
        """VRF outputs should be deterministic even across VRF instances.

        Verifies: Same private key bytes + same input = same output,
        regardless of when/how the VRF object is instantiated.
        """
        # Generate a key and save the bytes
        vrf1 = VRF.generate()
        key_bytes = vrf1.private_key_bytes
        alpha = b"test_determinism_input"

        # Compute output with first instance
        output1 = vrf1.prove(alpha)

        # Create new instance from same key bytes
        vrf2 = VRF(private_key_bytes=key_bytes)
        output2 = vrf2.prove(alpha)

        # Create yet another instance
        vrf3 = VRF(private_key_bytes=key_bytes)
        output3 = vrf3.prove(alpha)

        # All outputs must be identical
        assert output1.ticket == output2.ticket == output3.ticket
        assert output1.proof.to_bytes() == output2.proof.to_bytes() == output3.proof.to_bytes()

    def test_uniqueness_guarantees_different_keys_different_outputs(self):
        """Different keys must produce different outputs for same input.

        Verifies: The VRF output is cryptographically bound to the key.
        """
        alpha = b"same_input_for_all"

        # Generate multiple keys
        vrfs = [VRF.generate() for _ in range(20)]

        # Compute tickets
        tickets = [vrf.prove(alpha).ticket for vrf in vrfs]

        # All tickets must be unique (collision probability is negligible)
        assert len(set(tickets)) == len(tickets), "Collision detected - extremely unlikely with secure VRF"

    def test_domain_separator_prevents_cross_use(self):
        """Domain separators should prevent signature reuse attacks.

        Verifies: Raw Ed25519 signatures cannot be used as VRF proofs.
        """
        vrf = VRF.generate()
        alpha = b"test_input"

        # Get VRF output
        vrf_output = vrf.prove(alpha)

        # Manually compute a raw Ed25519 signature (without domain separator)
        raw_signature = vrf._private_key.sign(alpha)

        # The raw signature should NOT equal the VRF proof components
        reconstructed_sig = vrf_output.proof.gamma + vrf_output.proof.c
        assert raw_signature != reconstructed_sig, "Domain separator not effective"

    def test_proof_is_bound_to_ticket(self):
        """Proof and ticket are cryptographically bound.

        Verifies: Cannot use proof from one input with ticket from another.
        """
        vrf = VRF.generate()

        output1 = vrf.prove(b"input_1")
        output2 = vrf.prove(b"input_2")

        # Try to create a franken-output with proof from output1 and ticket from output2
        franken_output = VRFOutput(
            ticket=output2.ticket,
            proof=output1.proof,  # Wrong proof!
            input_hash=output2.input_hash,
        )

        # This should fail verification
        assert not VRF.verify(vrf.public_key_bytes, b"input_2", franken_output)

    def test_input_hash_is_bound_to_output(self):
        """Input hash in output must match actual input.

        Verifies: Cannot claim output is for different input than it was computed for.
        """
        vrf = VRF.generate()

        output = vrf.prove(b"actual_input")

        # Try to verify with different input
        assert not VRF.verify(vrf.public_key_bytes, b"claimed_input", output)

    def test_public_key_binding(self):
        """Output is bound to specific public key.

        Verifies: Cannot claim output from key A belongs to key B.
        """
        vrf_a = VRF.generate()
        vrf_b = VRF.generate()
        alpha = b"test_input"

        # Compute with key A
        output_a = vrf_a.prove(alpha)

        # Should verify with A's public key
        assert VRF.verify(vrf_a.public_key_bytes, alpha, output_a)

        # Should NOT verify with B's public key
        assert not VRF.verify(vrf_b.public_key_bytes, alpha, output_a)

    def test_bit_flip_detection(self):
        """Single bit flips in proof or ticket should be detected.

        Verifies: Proof verification is sensitive to all bits.
        """
        vrf = VRF.generate()
        alpha = b"test_input"
        output = vrf.prove(alpha)

        # Test bit flips in ticket
        for byte_idx in [0, 15, 31]:  # Start, middle, end
            tampered_ticket = bytearray(output.ticket)
            tampered_ticket[byte_idx] ^= 0x01
            tampered = VRFOutput(
                ticket=bytes(tampered_ticket),
                proof=output.proof,
                input_hash=output.input_hash,
            )
            assert not VRF.verify(vrf.public_key_bytes, alpha, tampered), f"Bit flip at byte {byte_idx} not detected in ticket"

        # Test bit flips in proof gamma
        for byte_idx in [0, 15, 31]:
            tampered_gamma = bytearray(output.proof.gamma)
            tampered_gamma[byte_idx] ^= 0x01
            tampered_proof = VRFProof(
                gamma=bytes(tampered_gamma),
                c=output.proof.c,
                s=output.proof.s,
            )
            tampered = VRFOutput(
                ticket=output.ticket,
                proof=tampered_proof,
                input_hash=output.input_hash,
            )
            assert not VRF.verify(vrf.public_key_bytes, alpha, tampered), f"Bit flip at byte {byte_idx} not detected in gamma"

    def test_empty_input_handling(self):
        """VRF should handle empty input safely."""
        vrf = VRF.generate()

        # Empty input should work
        output = vrf.prove(b"")
        assert len(output.ticket) == 32
        assert VRF.verify(vrf.public_key_bytes, b"", output)

        # Empty input should produce different output than non-empty
        output2 = vrf.prove(b"x")
        assert output.ticket != output2.ticket

    def test_large_input_handling(self):
        """VRF should handle large inputs safely."""
        vrf = VRF.generate()

        # Large input (1MB)
        large_input = b"x" * (1024 * 1024)
        output = vrf.prove(large_input)

        assert len(output.ticket) == 32
        assert VRF.verify(vrf.public_key_bytes, large_input, output)


# =============================================================================
# VALIDATOR SELECTION SIMULATION TESTS
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography not available")
class TestValidatorSelectionSimulation:
    """Tests simulating actual validator selection scenarios.

    These tests verify the VRF behaves correctly in realistic
    validator selection scenarios.
    """

    def test_selection_fairness_simulation(self):
        """Simulate validator selection and verify fairness.

        With uniform stake, each validator should win approximately
        equally often.
        """
        num_validators = 10
        num_epochs = 100

        # Create validators
        validators = [VRF.generate() for _ in range(num_validators)]
        wins = [0] * num_validators

        # Simulate epochs
        for epoch in range(num_epochs):
            epoch_seed = hashlib.sha256(f"epoch_{epoch}".encode()).digest()

            # Compute tickets
            tickets = []
            for i, vrf in enumerate(validators):
                output = vrf.prove(epoch_seed)
                tickets.append((output.ticket_as_int(), i))

            # Winner has lowest ticket
            tickets.sort()
            winner = tickets[0][1]
            wins[winner] += 1

        # Each validator should win roughly 10% of the time (10 ± 7)
        # Using wide margin due to statistical variance
        for i, win_count in enumerate(wins):
            assert 3 <= win_count <= 25, f"Validator {i} won {win_count} times (expected ~10)"

    def test_epoch_isolation(self):
        """Tickets from one epoch cannot be used in another.

        Verifies: Epoch seed properly isolates selection rounds.
        """
        vrf = VRF.generate()

        seed_epoch_1 = hashlib.sha256(b"epoch_1").digest()
        seed_epoch_2 = hashlib.sha256(b"epoch_2").digest()

        output_e1 = vrf.prove(seed_epoch_1)
        output_e2 = vrf.prove(seed_epoch_2)

        # Tickets should be different
        assert output_e1.ticket != output_e2.ticket

        # Verification should fail if seeds are swapped
        assert VRF.verify(vrf.public_key_bytes, seed_epoch_1, output_e1)
        assert not VRF.verify(vrf.public_key_bytes, seed_epoch_2, output_e1)

    def test_ticket_ordering_consistency(self):
        """Ticket ordering should be consistent across verifications.

        Verifies: Selection order is deterministic and reproducible.
        """
        validators = [VRF.generate() for _ in range(5)]
        epoch_seed = hashlib.sha256(b"test_epoch").digest()

        # Compute and sort tickets
        def get_ordering():
            tickets = [(vrf.prove(epoch_seed).ticket_as_int(), i) for i, vrf in enumerate(validators)]
            tickets.sort()
            return [idx for _, idx in tickets]

        # Ordering should be identical every time
        ordering1 = get_ordering()
        ordering2 = get_ordering()
        ordering3 = get_ordering()

        assert ordering1 == ordering2 == ordering3
