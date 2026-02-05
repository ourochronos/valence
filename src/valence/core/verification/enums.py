"""Enumerations for the verification protocol."""

from __future__ import annotations

from enum import StrEnum


class VerificationResult(StrEnum):
    """Result of a verification."""

    CONFIRMED = "confirmed"  # Evidence supports the belief
    CONTRADICTED = "contradicted"  # Evidence refutes the belief
    UNCERTAIN = "uncertain"  # Insufficient evidence either way
    PARTIAL = "partial"  # Partially correct, with qualifications


class VerificationStatus(StrEnum):
    """Status of a verification."""

    PENDING = "pending"  # Awaiting validation
    ACCEPTED = "accepted"  # Validated and accepted
    DISPUTED = "disputed"  # Under dispute
    OVERTURNED = "overturned"  # Dispute overturned the verification
    REJECTED = "rejected"  # Failed validation
    EXPIRED = "expired"  # Timed out during validation


class StakeType(StrEnum):
    """Type of stake."""

    STANDARD = "standard"  # Normal verification stake
    BOUNTY = "bounty"  # Claiming a discrepancy bounty
    CHALLENGE = "challenge"  # Challenging existing verification


class EvidenceType(StrEnum):
    """Type of evidence provided."""

    BELIEF = "belief"  # Reference to another Valence belief
    EXTERNAL = "external"  # External source (URL, paper, etc.)
    OBSERVATION = "observation"  # Verifier's direct observation
    DERIVATION = "derivation"  # Logical/mathematical proof
    TESTIMONY = "testimony"  # Statement from another agent


class EvidenceContribution(StrEnum):
    """How evidence contributes to verification."""

    SUPPORTS = "supports"  # Evidence for the belief
    CONTRADICTS = "contradicts"  # Evidence against the belief
    CONTEXT = "context"  # Adds relevant context
    QUALIFIES = "qualifies"  # Adds conditions/limitations


class ContradictionType(StrEnum):
    """Type of contradiction found."""

    FACTUALLY_FALSE = "factually_false"
    OUTDATED = "outdated"
    MISATTRIBUTED = "misattributed"
    OVERSTATED = "overstated"
    MISSING_CONTEXT = "missing_context"
    LOGICAL_ERROR = "logical_error"


class UncertaintyReason(StrEnum):
    """Reason for uncertain result."""

    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONFLICTING_SOURCES = "conflicting_sources"
    OUTSIDE_EXPERTISE = "outside_expertise"
    UNFALSIFIABLE = "unfalsifiable"
    REQUIRES_EXPERIMENT = "requires_experiment"


class DisputeType(StrEnum):
    """Type of dispute."""

    EVIDENCE_INVALID = "evidence_invalid"
    EVIDENCE_FABRICATED = "evidence_fabricated"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    REASONING_FLAWED = "reasoning_flawed"
    CONFLICT_OF_INTEREST = "conflict_of_interest"
    NEW_EVIDENCE = "new_evidence"


class DisputeOutcome(StrEnum):
    """Outcome of a dispute resolution."""

    UPHELD = "upheld"  # Original verification stands
    OVERTURNED = "overturned"  # Verification was wrong
    MODIFIED = "modified"  # Result changed
    DISMISSED = "dismissed"  # Dispute was frivolous


class DisputeStatus(StrEnum):
    """Status of a dispute."""

    PENDING = "pending"  # Awaiting resolution
    RESOLVED = "resolved"  # Resolution complete
    EXPIRED = "expired"  # Timed out


class ResolutionMethod(StrEnum):
    """Method used to resolve a dispute."""

    AUTOMATIC = "automatic"  # Algorithm decides
    JURY = "jury"  # Random selection of jurors
    EXPERT = "expert"  # Domain experts decide
    APPEAL = "appeal"  # Higher-level review
