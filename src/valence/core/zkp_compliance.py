"""ZKP compliance proofs for learn_from_me sharing (#343).

Generates and verifies zero-knowledge proofs for sharing intent
compliance. Allows a recipient to prove they have valid consent
or are operating within policy without revealing the underlying
consent records.

Proof types:
- HAS_CONSENT: Prove valid consent exists for a sharing operation
- WITHIN_POLICY: Prove an operation complies with the sharing policy

Differential privacy is applied to shared confidence scores.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from our_crypto import create_zkp_backend

logger = logging.getLogger(__name__)

DEFAULT_ZKP_BACKEND = "sigma"


@dataclass
class ConsentProof:
    """A ZKP proof that valid consent exists."""

    id: str
    belief_id: str
    prover_did: str
    proof_type: str  # HAS_CONSENT or WITHIN_POLICY
    proof_data: bytes
    public_inputs_hash: bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "belief_id": self.belief_id,
            "prover_did": self.prover_did,
            "proof_type": self.proof_type,
            "created_at": self.created_at.isoformat(),
            "is_valid": self.is_valid,
        }


@dataclass
class PolicyProof:
    """A ZKP proof that operation is within policy."""

    id: str
    operation_type: str
    policy_hash: str
    proof_data: bytes
    public_inputs_hash: bytes
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_valid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "operation_type": self.operation_type,
            "policy_hash": self.policy_hash,
            "created_at": self.created_at.isoformat(),
            "is_valid": self.is_valid,
        }


# Differential privacy noise for confidence scores
DP_EPSILON = 1.0  # Privacy budget
DP_SENSITIVITY = 0.1  # Max confidence change from one record


def _get_zkp_backend():
    """Get the ZKP backend. Separate function for test patching."""
    return create_zkp_backend(DEFAULT_ZKP_BACKEND)


def add_dp_noise(value: float, epsilon: float = DP_EPSILON, sensitivity: float = DP_SENSITIVITY) -> float:
    """Add Laplacian noise for differential privacy.

    Args:
        value: Original confidence value (0-1).
        epsilon: Privacy budget (lower = more private).
        sensitivity: Maximum influence of one record.

    Returns:
        Noised value clipped to [0, 1].
    """
    scale = sensitivity / epsilon
    noise = random.gauss(0, scale)
    return max(0.0, min(1.0, value + noise))


def generate_consent_proof(
    belief_id: str,
    prover_did: str,
    consent_record: dict[str, Any],
) -> ConsentProof:
    """Generate a ZKP proving valid consent exists.

    The proof demonstrates that the prover has a valid consent chain
    for accessing the belief without revealing the consent details.

    Args:
        belief_id: UUID of the belief.
        prover_did: DID of the prover.
        consent_record: Private consent chain data (hidden in proof).

    Returns:
        ConsentProof with zero-knowledge proof data.
    """
    backend = _get_zkp_backend()
    backend.setup("HAS_CONSENT")

    prover = backend.create_prover("HAS_CONSENT")
    proof = prover.prove(
        private_inputs=consent_record,
        public_inputs={"user_id": prover_did, "action": f"access:{belief_id}"},
    )

    return ConsentProof(
        id=str(uuid4()),
        belief_id=belief_id,
        prover_did=prover_did,
        proof_type="HAS_CONSENT",
        proof_data=proof.proof_data,
        public_inputs_hash=proof.public_inputs_hash,
        is_valid=True,
    )


def verify_consent_proof(
    proof: ConsentProof,
    belief_id: str,
    prover_did: str,
) -> bool:
    """Verify a consent proof.

    Args:
        proof: The ConsentProof to verify.
        belief_id: Expected belief ID.
        prover_did: Expected prover DID.

    Returns:
        True if proof is valid.
    """
    backend = _get_zkp_backend()
    backend.setup("HAS_CONSENT")

    verifier = backend.create_verifier("HAS_CONSENT")

    from our_crypto import ComplianceProof
    compliance_proof = ComplianceProof(
        proof_type="HAS_CONSENT",
        proof_data=proof.proof_data,
        public_inputs_hash=proof.public_inputs_hash,
    )

    result = verifier.verify(
        proof=compliance_proof,
        public_inputs={"user_id": prover_did, "action": f"access:{belief_id}"},
    )

    return result.valid


def generate_policy_proof(
    operation_type: str,
    policy: dict[str, Any],
    operation_details: dict[str, Any],
) -> PolicyProof:
    """Generate a ZKP proving operation complies with policy.

    Args:
        operation_type: Type of operation (e.g., "reshare", "access").
        policy: The policy to check against (private).
        operation_details: Details of the operation (private).

    Returns:
        PolicyProof with zero-knowledge proof data.
    """
    import hashlib
    policy_hash = hashlib.sha256(str(sorted(policy.items())).encode()).hexdigest()

    backend = _get_zkp_backend()
    backend.setup("WITHIN_POLICY")

    prover = backend.create_prover("WITHIN_POLICY")
    proof = prover.prove(
        private_inputs={"operation_details": operation_details, "policy_evaluation": True},
        public_inputs={"policy_hash": policy_hash, "operation_type": operation_type},
    )

    return PolicyProof(
        id=str(uuid4()),
        operation_type=operation_type,
        policy_hash=policy_hash,
        proof_data=proof.proof_data,
        public_inputs_hash=proof.public_inputs_hash,
        is_valid=True,
    )


def verify_policy_proof(
    proof: PolicyProof,
    operation_type: str,
) -> bool:
    """Verify a policy compliance proof.

    Args:
        proof: The PolicyProof to verify.
        operation_type: Expected operation type.

    Returns:
        True if proof is valid.
    """
    backend = _get_zkp_backend()
    backend.setup("WITHIN_POLICY")

    verifier = backend.create_verifier("WITHIN_POLICY")

    from our_crypto import ComplianceProof
    compliance_proof = ComplianceProof(
        proof_type="WITHIN_POLICY",
        proof_data=proof.proof_data,
        public_inputs_hash=proof.public_inputs_hash,
    )

    result = verifier.verify(
        proof=compliance_proof,
        public_inputs={"policy_hash": proof.policy_hash, "operation_type": operation_type},
    )

    return result.valid


def apply_dp_to_confidence(confidence: dict[str, float]) -> dict[str, float]:
    """Apply differential privacy to confidence dimensions before sharing.

    Args:
        confidence: Dict of confidence dimension name -> value.

    Returns:
        Noised confidence dict with values clipped to [0, 1].
    """
    return {dim: add_dp_noise(val) for dim, val in confidence.items()}
