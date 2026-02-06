"""Tests for the attestation service.

Tests cover:
- Adding attestations (delegates to ResourceSharingService)
- Querying attestations with filters
- Aggregate statistics computation
- Trust signal computation from attestation patterns
- Edge cases (no attestations, single user, etc.)

Part of Issue #271: Social — Usage attestations.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from valence.core.attestation_service import (
    AttestationFilter,
    AttestationService,
    AttestationStats,
    TrustSignal,
)
from valence.core.resource_sharing import (
    DefaultTrustProvider,
    InMemoryResourceStore,
    ResourceSharingService,
)
from valence.core.resources import Resource, ResourceType

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def trust_provider() -> DefaultTrustProvider:
    return DefaultTrustProvider(default_level=0.7)


@pytest.fixture
def store() -> InMemoryResourceStore:
    return InMemoryResourceStore()


@pytest.fixture
def sharing_service(store, trust_provider) -> ResourceSharingService:
    return ResourceSharingService(store=store, trust_provider=trust_provider)


@pytest.fixture
def service(sharing_service) -> AttestationService:
    return AttestationService(sharing_service)


@pytest.fixture
def sample_resource(sharing_service) -> Resource:
    """A shared resource ready for attestation."""
    resource = Resource(
        id=uuid4(),
        type=ResourceType.PROMPT,
        content="You are a helpful assistant. Respond clearly.",
        author_did="did:vkb:web:alice.example.com",
        name="helpful-assistant",
    )
    sharing_service.share_resource(resource)
    return resource


@pytest.fixture
def multi_resource_setup(sharing_service) -> list[Resource]:
    """Multiple shared resources for cross-resource queries."""
    resources = []
    for i in range(3):
        r = Resource(
            id=uuid4(),
            type=ResourceType.PROMPT,
            content=f"Prompt template {i}",
            author_did="did:vkb:web:alice.example.com",
            name=f"prompt-{i}",
        )
        sharing_service.share_resource(r)
        resources.append(r)
    return resources


# =============================================================================
# ADD ATTESTATION TESTS
# =============================================================================


class TestAddAttestation:
    """Tests for adding attestations through the service."""

    def test_add_success(self, service, sample_resource):
        """Add a successful attestation."""
        att = service.add_attestation(
            resource_id=sample_resource.id,
            user_did="did:user1",
            success=True,
            feedback="Works great!",
        )
        assert att.resource_id == sample_resource.id
        assert att.user_did == "did:user1"
        assert att.success is True
        assert att.feedback == "Works great!"

    def test_add_failure(self, service, sample_resource):
        """Add a failed attestation."""
        att = service.add_attestation(
            resource_id=sample_resource.id,
            user_did="did:user1",
            success=False,
            feedback="Didn't work for my use case",
        )
        assert att.success is False

    def test_add_no_feedback(self, service, sample_resource):
        """Add an attestation without feedback."""
        att = service.add_attestation(
            resource_id=sample_resource.id,
            user_did="did:user1",
        )
        assert att.feedback is None

    def test_add_nonexistent_resource(self, service):
        """Adding to a nonexistent resource raises error."""
        from valence.core.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            service.add_attestation(uuid4(), "did:user1")


# =============================================================================
# QUERY ATTESTATION TESTS
# =============================================================================


class TestGetAttestations:
    """Tests for querying attestations with filters."""

    def test_get_all_empty(self, service):
        """Empty service returns empty list."""
        result = service.get_attestations()
        assert result == []

    def test_get_all_for_resource(self, service, sample_resource):
        """Get all attestations for a specific resource."""
        service.add_attestation(sample_resource.id, "did:user1", True)
        service.add_attestation(sample_resource.id, "did:user2", False)
        service.add_attestation(sample_resource.id, "did:user3", True)

        filt = AttestationFilter(resource_id=sample_resource.id)
        result = service.get_attestations(filt)
        assert len(result) == 3

    def test_filter_by_user(self, service, sample_resource):
        """Filter attestations by user DID."""
        service.add_attestation(sample_resource.id, "did:alice", True)
        service.add_attestation(sample_resource.id, "did:bob", False)
        service.add_attestation(sample_resource.id, "did:alice", True)

        filt = AttestationFilter(resource_id=sample_resource.id, user_did="did:alice")
        result = service.get_attestations(filt)
        assert len(result) == 2
        assert all(a.user_did == "did:alice" for a in result)

    def test_filter_by_success(self, service, sample_resource):
        """Filter attestations by success status."""
        service.add_attestation(sample_resource.id, "did:user1", True)
        service.add_attestation(sample_resource.id, "did:user2", False)
        service.add_attestation(sample_resource.id, "did:user3", True)

        filt = AttestationFilter(resource_id=sample_resource.id, success=True)
        result = service.get_attestations(filt)
        assert len(result) == 2
        assert all(a.success for a in result)

    def test_filter_by_failure(self, service, sample_resource):
        """Filter attestations by failure status."""
        service.add_attestation(sample_resource.id, "did:user1", True)
        service.add_attestation(sample_resource.id, "did:user2", False)

        filt = AttestationFilter(resource_id=sample_resource.id, success=False)
        result = service.get_attestations(filt)
        assert len(result) == 1
        assert not result[0].success

    def test_filter_limit(self, service, sample_resource):
        """Limit number of results."""
        for i in range(10):
            service.add_attestation(sample_resource.id, f"did:user{i}", True)

        filt = AttestationFilter(resource_id=sample_resource.id, limit=3)
        result = service.get_attestations(filt)
        assert len(result) == 3

    def test_sorted_newest_first(self, service, sample_resource):
        """Results are sorted newest first."""
        service.add_attestation(sample_resource.id, "did:first", True)
        service.add_attestation(sample_resource.id, "did:second", True)
        service.add_attestation(sample_resource.id, "did:third", True)

        result = service.get_attestations(AttestationFilter(resource_id=sample_resource.id))
        # Newest first — third should be first
        assert result[0].user_did == "did:third"

    def test_cross_resource_query(self, service, multi_resource_setup):
        """Query attestations across all resources."""
        for r in multi_resource_setup:
            service.add_attestation(r.id, "did:user1", True)

        # No resource_id filter = scan all
        result = service.get_attestations(AttestationFilter())
        assert len(result) == 3

    def test_cross_resource_user_filter(self, service, multi_resource_setup):
        """Filter by user across all resources."""
        for r in multi_resource_setup:
            service.add_attestation(r.id, "did:alice", True)
            service.add_attestation(r.id, "did:bob", False)

        filt = AttestationFilter(user_did="did:alice")
        result = service.get_attestations(filt)
        assert len(result) == 3
        assert all(a.user_did == "did:alice" for a in result)

    def test_default_filter(self, service, sample_resource):
        """Passing None uses default filter."""
        service.add_attestation(sample_resource.id, "did:user1", True)
        result = service.get_attestations(None)
        assert len(result) == 1


# =============================================================================
# STATS TESTS
# =============================================================================


class TestAttestationStats:
    """Tests for aggregate statistics computation."""

    def test_stats_empty(self, service, sample_resource):
        """No attestations returns zero stats."""
        stats = service.get_stats(sample_resource.id)
        assert stats.total == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.success_rate is None
        assert stats.unique_users == 0
        assert stats.latest_at is None

    def test_stats_basic(self, service, sample_resource):
        """Basic stats computation."""
        service.add_attestation(sample_resource.id, "did:user1", True)
        service.add_attestation(sample_resource.id, "did:user2", False)
        service.add_attestation(sample_resource.id, "did:user3", True)

        stats = service.get_stats(sample_resource.id)
        assert stats.total == 3
        assert stats.successes == 2
        assert stats.failures == 1
        assert stats.success_rate == pytest.approx(2 / 3)
        assert stats.unique_users == 3
        assert stats.latest_at is not None

    def test_stats_all_success(self, service, sample_resource):
        """100% success rate."""
        for i in range(5):
            service.add_attestation(sample_resource.id, f"did:user{i}", True)

        stats = service.get_stats(sample_resource.id)
        assert stats.success_rate == 1.0

    def test_stats_all_failure(self, service, sample_resource):
        """0% success rate."""
        for i in range(3):
            service.add_attestation(sample_resource.id, f"did:user{i}", False)

        stats = service.get_stats(sample_resource.id)
        assert stats.success_rate == 0.0

    def test_stats_duplicate_users(self, service, sample_resource):
        """Same user attesting multiple times counts as one unique user."""
        service.add_attestation(sample_resource.id, "did:user1", True)
        service.add_attestation(sample_resource.id, "did:user1", True)
        service.add_attestation(sample_resource.id, "did:user1", False)

        stats = service.get_stats(sample_resource.id)
        assert stats.total == 3
        assert stats.unique_users == 1

    def test_stats_to_dict(self, service, sample_resource):
        """Stats can be serialized to dict."""
        service.add_attestation(sample_resource.id, "did:user1", True)

        stats = service.get_stats(sample_resource.id)
        d = stats.to_dict()
        assert d["resource_id"] == str(sample_resource.id)
        assert d["total"] == 1
        assert d["successes"] == 1
        assert d["success_rate"] == 1.0

    def test_get_all_stats(self, service, multi_resource_setup):
        """Get stats across all resources."""
        # Only add attestations to first two resources
        service.add_attestation(multi_resource_setup[0].id, "did:user1", True)
        service.add_attestation(multi_resource_setup[0].id, "did:user2", True)
        service.add_attestation(multi_resource_setup[1].id, "did:user1", False)

        all_stats = service.get_all_stats()
        # Only resources with attestations
        assert len(all_stats) == 2
        # Sorted by total descending
        assert all_stats[0].total == 2
        assert all_stats[1].total == 1

    def test_get_all_stats_empty(self, service):
        """No attestations returns empty list."""
        assert service.get_all_stats() == []


# =============================================================================
# TRUST SIGNAL TESTS
# =============================================================================


class TestTrustSignal:
    """Tests for trust signal computation."""

    def test_no_attestations(self, service, sample_resource):
        """No attestations produces zero trust signal."""
        signal = service.compute_trust_signal(sample_resource.id)
        assert signal.overall == 0.0
        assert signal.success_rate == 0.0
        assert signal.diversity_score == 0.0
        assert signal.volume_score == 0.0

    def test_single_success(self, service, sample_resource):
        """Single successful attestation."""
        service.add_attestation(sample_resource.id, "did:user1", True)

        signal = service.compute_trust_signal(sample_resource.id)
        assert signal.success_rate == 1.0
        assert signal.diversity_score == pytest.approx(1 / 5)  # 1 user / 5
        assert signal.volume_score == pytest.approx(1 / 20)  # 1 / VOLUME_SATURATION
        assert signal.overall > 0

    def test_perfect_resource(self, service, sample_resource):
        """A well-attested resource gets high trust."""
        # 20 attestations from 10 unique users, all successful
        for i in range(20):
            service.add_attestation(sample_resource.id, f"did:user{i % 10}", True)

        signal = service.compute_trust_signal(sample_resource.id)
        assert signal.success_rate == 1.0
        assert signal.volume_score == 1.0  # At saturation
        assert signal.diversity_score == 1.0  # 10 users >= 5
        assert signal.overall == pytest.approx(1.0)

    def test_mixed_results(self, service, sample_resource):
        """Mixed success/failure attestations."""
        service.add_attestation(sample_resource.id, "did:user1", True)
        service.add_attestation(sample_resource.id, "did:user2", False)

        signal = service.compute_trust_signal(sample_resource.id)
        assert signal.success_rate == 0.5
        assert signal.diversity_score == pytest.approx(2 / 5)
        assert signal.volume_score == pytest.approx(2 / 20)
        assert 0 < signal.overall < 1

    def test_all_failures(self, service, sample_resource):
        """All failed attestations produce low trust."""
        for i in range(5):
            service.add_attestation(sample_resource.id, f"did:user{i}", False)

        signal = service.compute_trust_signal(sample_resource.id)
        assert signal.success_rate == 0.0
        # Still has diversity and volume, but success_rate=0 drags overall down
        assert signal.overall < 0.5

    def test_signal_to_dict(self, service, sample_resource):
        """Trust signal can be serialized to dict."""
        service.add_attestation(sample_resource.id, "did:user1", True)

        signal = service.compute_trust_signal(sample_resource.id)
        d = signal.to_dict()
        assert d["resource_id"] == str(sample_resource.id)
        assert "success_rate" in d
        assert "diversity_score" in d
        assert "volume_score" in d
        assert "overall" in d

    def test_weights_sum_to_one(self, service):
        """Verify weight invariant."""
        total = AttestationService.SUCCESS_WEIGHT + AttestationService.DIVERSITY_WEIGHT + AttestationService.VOLUME_WEIGHT
        assert total == pytest.approx(1.0)

    def test_single_user_many_attestations(self, service, sample_resource):
        """Single user with many attestations: high volume, low diversity."""
        for _ in range(20):
            service.add_attestation(sample_resource.id, "did:same-user", True)

        signal = service.compute_trust_signal(sample_resource.id)
        assert signal.success_rate == 1.0
        assert signal.volume_score == 1.0
        assert signal.diversity_score == pytest.approx(1 / 5)  # Only 1 user
        # Overall less than perfect because of low diversity
        assert signal.overall < 1.0


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestDataclasses:
    """Tests for AttestationStats and TrustSignal dataclasses."""

    def test_attestation_stats_defaults(self):
        """Default values for AttestationStats."""
        stats = AttestationStats(resource_id=uuid4())
        assert stats.total == 0
        assert stats.successes == 0
        assert stats.failures == 0
        assert stats.success_rate is None
        assert stats.unique_users == 0
        assert stats.latest_at is None

    def test_trust_signal_defaults(self):
        """Default values for TrustSignal."""
        signal = TrustSignal(resource_id=uuid4())
        assert signal.success_rate == 0.0
        assert signal.diversity_score == 0.0
        assert signal.volume_score == 0.0
        assert signal.overall == 0.0

    def test_attestation_filter_defaults(self):
        """Default values for AttestationFilter."""
        filt = AttestationFilter()
        assert filt.resource_id is None
        assert filt.user_did is None
        assert filt.success is None
        assert filt.since is None
        assert filt.limit == 50
