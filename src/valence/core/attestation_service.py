"""Attestation service for querying, aggregating, and computing trust signals.

Builds on the UsageAttestation model (resources.py) and basic tracking
in ResourceSharingService (resource_sharing.py). This service adds:

- Query attestations across resources with filtering
- Aggregate statistics (success rates, usage patterns)
- Trust signal computation from attestation patterns

Part of Issue #271: Social — Usage attestations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from .resources import UsageAttestation


@dataclass
class AttestationStats:
    """Aggregated statistics for attestations on a resource."""

    resource_id: UUID
    total: int = 0
    successes: int = 0
    failures: int = 0
    success_rate: float | None = None
    unique_users: int = 0
    latest_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "resource_id": str(self.resource_id),
            "total": self.total,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.success_rate,
            "unique_users": self.unique_users,
            "latest_at": self.latest_at.isoformat() if self.latest_at else None,
        }


@dataclass
class TrustSignal:
    """Trust signal derived from attestation patterns.

    Combines multiple signals into a single quality indicator:
    - success_rate: raw success/total ratio
    - diversity_score: how many unique users attested (more = more trustworthy)
    - volume_score: how many attestations exist (more = more confident)
    - overall: weighted combination of all signals
    """

    resource_id: UUID
    success_rate: float = 0.0
    diversity_score: float = 0.0
    volume_score: float = 0.0
    overall: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "resource_id": str(self.resource_id),
            "success_rate": round(self.success_rate, 3),
            "diversity_score": round(self.diversity_score, 3),
            "volume_score": round(self.volume_score, 3),
            "overall": round(self.overall, 3),
        }


@dataclass
class AttestationFilter:
    """Filter criteria for querying attestations."""

    resource_id: UUID | None = None
    user_did: str | None = None
    success: bool | None = None
    since: datetime | None = None
    limit: int = 50


class AttestationService:
    """Service for querying, aggregating, and analyzing usage attestations.

    This service does NOT own attestation storage — it delegates to
    ResourceSharingService for the actual data. It adds query, aggregation,
    and trust signal computation on top.
    """

    # Weights for trust signal computation
    SUCCESS_WEIGHT: float = 0.5
    DIVERSITY_WEIGHT: float = 0.3
    VOLUME_WEIGHT: float = 0.2

    # Volume scoring: at this many attestations, volume_score = 1.0
    VOLUME_SATURATION: int = 20

    def __init__(self, sharing_service: object) -> None:
        """Initialize with a ResourceSharingService instance.

        Args:
            sharing_service: A ResourceSharingService (or compatible object)
                that provides attest_usage(), get_attestations(), list_resources().
        """
        self._sharing = sharing_service

    def add_attestation(
        self,
        resource_id: UUID,
        user_did: str,
        success: bool = True,
        feedback: str | None = None,
    ) -> UsageAttestation:
        """Add a usage attestation (delegates to ResourceSharingService).

        Args:
            resource_id: ID of the resource used.
            user_did: DID of the user attesting.
            success: Whether the usage was successful.
            feedback: Optional feedback text.

        Returns:
            The created UsageAttestation.
        """
        return self._sharing.attest_usage(
            resource_id=resource_id,
            user_did=user_did,
            success=success,
            feedback=feedback,
        )

    def get_attestations(
        self,
        filt: AttestationFilter | None = None,
    ) -> list[UsageAttestation]:
        """Query attestations with optional filtering.

        When resource_id is provided, returns attestations for that resource.
        Additional filters (user_did, success, since) are applied in-memory.

        When no resource_id is provided, scans all resources.

        Args:
            filt: Filter criteria. If None, returns all attestations.

        Returns:
            List of matching UsageAttestation objects.
        """
        if filt is None:
            filt = AttestationFilter()

        attestations: list[UsageAttestation] = []

        if filt.resource_id is not None:
            # Single resource — direct lookup
            attestations = list(self._sharing.get_attestations(filt.resource_id))
        else:
            # Scan all resources
            resources = self._sharing.list_resources()
            for resource in resources:
                attestations.extend(self._sharing.get_attestations(resource.id))

        # Apply filters
        if filt.user_did is not None:
            attestations = [a for a in attestations if a.user_did == filt.user_did]

        if filt.success is not None:
            attestations = [a for a in attestations if a.success == filt.success]

        if filt.since is not None:
            attestations = [a for a in attestations if a.created_at >= filt.since]

        # Sort by created_at descending (newest first)
        attestations.sort(key=lambda a: a.created_at, reverse=True)

        # Apply limit
        return attestations[: filt.limit]

    def get_stats(self, resource_id: UUID) -> AttestationStats:
        """Compute aggregate statistics for a resource's attestations.

        Args:
            resource_id: ID of the resource.

        Returns:
            AttestationStats with counts, rates, and user diversity.
        """
        attestations = self._sharing.get_attestations(resource_id)

        if not attestations:
            return AttestationStats(resource_id=resource_id)

        successes = sum(1 for a in attestations if a.success)
        total = len(attestations)
        unique_users = len({a.user_did for a in attestations})
        latest = max(a.created_at for a in attestations)

        return AttestationStats(
            resource_id=resource_id,
            total=total,
            successes=successes,
            failures=total - successes,
            success_rate=successes / total if total > 0 else None,
            unique_users=unique_users,
            latest_at=latest,
        )

    def compute_trust_signal(self, resource_id: UUID) -> TrustSignal:
        """Compute a trust signal from attestation patterns.

        Combines:
        - success_rate: proportion of successful attestations
        - diversity_score: normalized unique user count (log scale)
        - volume_score: how many attestations, saturating at VOLUME_SATURATION

        Overall = weighted combination of all three.

        Args:
            resource_id: ID of the resource.

        Returns:
            TrustSignal with component scores and overall.
        """
        stats = self.get_stats(resource_id)

        if stats.total == 0:
            return TrustSignal(resource_id=resource_id)

        # Success rate: direct from stats
        success_rate = stats.success_rate or 0.0

        # Diversity: more unique users = more trustworthy
        # Scale: 1 user = 0.2, 2 = 0.4, 5 = 0.8, 10+ = 1.0
        diversity = min(stats.unique_users / 5.0, 1.0)

        # Volume: more attestations = more confident
        volume = min(stats.total / self.VOLUME_SATURATION, 1.0)

        # Weighted overall
        overall = self.SUCCESS_WEIGHT * success_rate + self.DIVERSITY_WEIGHT * diversity + self.VOLUME_WEIGHT * volume

        return TrustSignal(
            resource_id=resource_id,
            success_rate=success_rate,
            diversity_score=diversity,
            volume_score=volume,
            overall=overall,
        )

    def get_all_stats(self) -> list[AttestationStats]:
        """Get attestation stats for all resources that have attestations.

        Returns:
            List of AttestationStats, sorted by total attestations descending.
        """
        resources = self._sharing.list_resources()
        all_stats = []

        for resource in resources:
            stats = self.get_stats(resource.id)
            if stats.total > 0:
                all_stats.append(stats)

        all_stats.sort(key=lambda s: s.total, reverse=True)
        return all_stats
