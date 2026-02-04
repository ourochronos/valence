"""Cryptographic primitives for Valence.

This module provides cryptographic abstractions including:
- MLS (Messaging Layer Security) for group encryption
- ZKP (Zero-Knowledge Proofs) for compliance verification
- PRE (Proxy Re-Encryption) for federation aggregation
"""

from valence.crypto.mls import (
    MLSGroup,
    MLSMember,
    MLSKeySchedule,
    MLSBackend,
    MockMLSBackend,
    MLSError,
    MLSGroupNotFoundError,
    MLSMemberNotFoundError,
    MLSEpochMismatchError,
)

from valence.crypto.zkp import (
    # Exceptions
    ZKPError,
    ZKPInvalidProofError,
    ZKPCircuitNotFoundError,
    ZKPProvingError,
    ZKPVerificationError,
    ZKPInputError,
    # Types
    ComplianceProofType,
    PublicParameters,
    ComplianceProof,
    VerificationResult,
    # Abstract interfaces
    ZKPProver,
    ZKPVerifier,
    ZKPBackend,
    # Mock implementations
    MockZKPProver,
    MockZKPVerifier,
    MockZKPBackend,
    # Utilities
    hash_public_inputs,
    verify_proof,
)

from valence.crypto.pre import (
    # Exceptions
    PREError,
    PREKeyError,
    PREEncryptionError,
    PREDecryptionError,
    PREReEncryptionError,
    PREInvalidCiphertextError,
    # Data classes
    PREPublicKey,
    PREPrivateKey,
    PREKeyPair,
    ReEncryptionKey,
    PRECiphertext,
    # Abstract interface
    PREBackend,
    # Mock implementation
    MockPREBackend,
    # Utilities
    create_mock_backend,
)

__all__ = [
    # MLS
    "MLSGroup",
    "MLSMember",
    "MLSKeySchedule",
    "MLSBackend",
    "MockMLSBackend",
    "MLSError",
    "MLSGroupNotFoundError",
    "MLSMemberNotFoundError",
    "MLSEpochMismatchError",
    # ZKP Exceptions
    "ZKPError",
    "ZKPInvalidProofError",
    "ZKPCircuitNotFoundError",
    "ZKPProvingError",
    "ZKPVerificationError",
    "ZKPInputError",
    # ZKP Types
    "ComplianceProofType",
    "PublicParameters",
    "ComplianceProof",
    "VerificationResult",
    # ZKP Interfaces
    "ZKPProver",
    "ZKPVerifier",
    "ZKPBackend",
    # ZKP Mock Implementations
    "MockZKPProver",
    "MockZKPVerifier",
    "MockZKPBackend",
    # ZKP Utilities
    "hash_public_inputs",
    "verify_proof",
    # PRE Exceptions
    "PREError",
    "PREKeyError",
    "PREEncryptionError",
    "PREDecryptionError",
    "PREReEncryptionError",
    "PREInvalidCiphertextError",
    # PRE Data Classes
    "PREPublicKey",
    "PREPrivateKey",
    "PREKeyPair",
    "ReEncryptionKey",
    "PRECiphertext",
    # PRE Interfaces
    "PREBackend",
    # PRE Mock Implementation
    "MockPREBackend",
    # PRE Utilities
    "create_mock_backend",
]
