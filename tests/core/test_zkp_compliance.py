"""Tests for ZKP compliance proofs (#343).

Tests cover:
1. generate_consent_proof creates valid proof
2. verify_consent_proof validates correct proof
3. generate_policy_proof creates proof
4. verify_policy_proof validates correct proof
5. add_dp_noise stays in bounds
6. add_dp_noise changes value
7. apply_dp_to_confidence preserves dimensions
8. ConsentProof serialization
9. PolicyProof serialization
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from valence.core.zkp_compliance import (
    ConsentProof,
    PolicyProof,
    add_dp_noise,
    apply_dp_to_confidence,
    generate_consent_proof,
    generate_policy_proof,
    verify_consent_proof,
    verify_policy_proof,
)


@pytest.fixture
def mock_zkp():
    """Mock the ZKP backend with realistic behavior."""
    backend = MagicMock()

    # Setup mock prover that returns proof-like objects
    mock_proof = MagicMock()
    mock_proof.proof_data = b"\x01" * 64
    mock_proof.public_inputs_hash = b"\x02" * 32

    mock_prover = MagicMock()
    mock_prover.prove.return_value = mock_proof
    backend.create_prover.return_value = mock_prover

    # Setup mock verifier
    mock_result = MagicMock()
    mock_result.valid = True
    mock_verifier = MagicMock()
    mock_verifier.verify.return_value = mock_result
    backend.create_verifier.return_value = mock_verifier

    with patch("valence.core.zkp_compliance._get_zkp_backend", return_value=backend):
        yield backend


class TestGenerateConsentProof:
    """Test consent proof generation."""

    def test_creates_proof(self, mock_zkp):
        proof = generate_consent_proof(
            belief_id="belief-1",
            prover_did="did:valence:alice",
            consent_record={"chain_id": "c1", "valid": True},
        )
        assert proof.belief_id == "belief-1"
        assert proof.prover_did == "did:valence:alice"
        assert proof.proof_type == "HAS_CONSENT"
        assert proof.is_valid is True
        assert proof.proof_data == b"\x01" * 64

    def test_calls_backend(self, mock_zkp):
        generate_consent_proof("b1", "did:alice", {"chain": "data"})
        mock_zkp.setup.assert_called_with("HAS_CONSENT")
        mock_zkp.create_prover.assert_called_with("HAS_CONSENT")


class TestVerifyConsentProof:
    """Test consent proof verification."""

    def test_verifies_valid(self, mock_zkp):
        proof = ConsentProof(
            id="p1", belief_id="b1", prover_did="did:alice",
            proof_type="HAS_CONSENT",
            proof_data=b"\x01" * 64,
            public_inputs_hash=b"\x02" * 32,
            is_valid=True,
        )
        result = verify_consent_proof(proof, "b1", "did:alice")
        assert result is True

    def test_rejects_invalid(self, mock_zkp):
        # Make verifier return invalid
        mock_zkp.create_verifier.return_value.verify.return_value.valid = False

        proof = ConsentProof(
            id="p1", belief_id="b1", prover_did="did:alice",
            proof_type="HAS_CONSENT",
            proof_data=b"\xff" * 64,
            public_inputs_hash=b"\x00" * 32,
        )
        result = verify_consent_proof(proof, "b1", "did:alice")
        assert result is False


class TestGeneratePolicyProof:
    """Test policy proof generation."""

    def test_creates_proof(self, mock_zkp):
        proof = generate_policy_proof(
            operation_type="reshare",
            policy={"max_hops": 2, "intent": "learn_from_me"},
            operation_details={"hop": 1, "recipient": "did:bob"},
        )
        assert proof.operation_type == "reshare"
        assert proof.is_valid is True
        assert len(proof.policy_hash) == 64  # SHA256 hex

    def test_deterministic_policy_hash(self, mock_zkp):
        policy = {"b": 2, "a": 1}
        p1 = generate_policy_proof("op", policy, {})
        p2 = generate_policy_proof("op", policy, {})
        assert p1.policy_hash == p2.policy_hash


class TestVerifyPolicyProof:
    """Test policy proof verification."""

    def test_verifies_valid(self, mock_zkp):
        proof = PolicyProof(
            id="p1", operation_type="reshare",
            policy_hash="abc" * 20 + "abcd",
            proof_data=b"\x01" * 64,
            public_inputs_hash=b"\x02" * 32,
            is_valid=True,
        )
        result = verify_policy_proof(proof, "reshare")
        assert result is True


class TestDifferentialPrivacy:
    """Test differential privacy noise addition."""

    def test_stays_in_bounds(self):
        for _ in range(100):
            noised = add_dp_noise(0.5)
            assert 0.0 <= noised <= 1.0

    def test_clips_low(self):
        # With very large negative noise, should clip to 0
        noised = add_dp_noise(0.0, epsilon=100.0, sensitivity=0.001)
        assert noised >= 0.0

    def test_clips_high(self):
        noised = add_dp_noise(1.0, epsilon=100.0, sensitivity=0.001)
        assert noised <= 1.0

    def test_adds_noise(self):
        # With enough trials, noise should cause some variation
        values = {add_dp_noise(0.5, epsilon=0.1) for _ in range(20)}
        assert len(values) > 1  # Not all identical

    def test_apply_to_confidence(self):
        confidence = {"overall": 0.8, "source_reliability": 0.7, "method_quality": 0.6}
        noised = apply_dp_to_confidence(confidence)
        assert set(noised.keys()) == set(confidence.keys())
        for dim in noised:
            assert 0.0 <= noised[dim] <= 1.0


class TestSerialization:
    """Test dataclass serialization."""

    def test_consent_proof_to_dict(self):
        proof = ConsentProof(
            id="p1", belief_id="b1", prover_did="did:alice",
            proof_type="HAS_CONSENT",
            proof_data=b"\x01" * 64,
            public_inputs_hash=b"\x02" * 32,
            is_valid=True,
        )
        d = proof.to_dict()
        assert d["proof_type"] == "HAS_CONSENT"
        assert d["is_valid"] is True

    def test_policy_proof_to_dict(self):
        proof = PolicyProof(
            id="p1", operation_type="reshare",
            policy_hash="abc123",
            proof_data=b"\x01" * 64,
            public_inputs_hash=b"\x02" * 32,
        )
        d = proof.to_dict()
        assert d["operation_type"] == "reshare"
        assert d["policy_hash"] == "abc123"
