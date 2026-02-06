"""Tests for Watch vs Trust distinction (Issue #269).

Tests the separation of 'seeing content' from 'giving reputation weight':

| Relationship | See content? | Reputation boost? | Affects worldview? |
|---|---|---|---|
| Trust     | ✅ | ✅ | ✅ |
| Watch     | ✅ | ❌ | ❌ |
| Distrust  | Optional | Negative | Inverse |
| Ignore    | ❌ | ❌ | ❌ |
"""

from __future__ import annotations

from valence.privacy.trust import (
    RelationshipType,
    TrustEdge,
    TrustService,
)

# ===========================================================================
# RelationshipType enum tests
# ===========================================================================


class TestRelationshipType:
    """Tests for the RelationshipType enum."""

    def test_enum_values(self):
        """All four relationship types exist with expected string values."""
        assert RelationshipType.TRUST.value == "trust"
        assert RelationshipType.WATCH.value == "watch"
        assert RelationshipType.DISTRUST.value == "distrust"
        assert RelationshipType.IGNORE.value == "ignore"

    def test_from_string(self):
        """Case-insensitive parsing with TRUST default."""
        assert RelationshipType.from_string("trust") == RelationshipType.TRUST
        assert RelationshipType.from_string("WATCH") == RelationshipType.WATCH
        assert RelationshipType.from_string("Distrust") == RelationshipType.DISTRUST
        assert RelationshipType.from_string("ignore") == RelationshipType.IGNORE
        # Unknown → default to TRUST
        assert RelationshipType.from_string("unknown") == RelationshipType.TRUST

    def test_shows_content(self):
        """TRUST and WATCH show content; DISTRUST and IGNORE do not."""
        assert RelationshipType.TRUST.shows_content is True
        assert RelationshipType.WATCH.shows_content is True
        assert RelationshipType.DISTRUST.shows_content is False
        assert RelationshipType.IGNORE.shows_content is False

    def test_affects_reputation(self):
        """TRUST and DISTRUST affect reputation; WATCH and IGNORE do not."""
        assert RelationshipType.TRUST.affects_reputation is True
        assert RelationshipType.WATCH.affects_reputation is False
        assert RelationshipType.DISTRUST.affects_reputation is True
        assert RelationshipType.IGNORE.affects_reputation is False

    def test_reputation_sign(self):
        """TRUST is positive, DISTRUST is negative, others are zero."""
        assert RelationshipType.TRUST.reputation_sign == 1
        assert RelationshipType.WATCH.reputation_sign == 0
        assert RelationshipType.DISTRUST.reputation_sign == -1
        assert RelationshipType.IGNORE.reputation_sign == 0

    def test_affects_worldview(self):
        """TRUST and DISTRUST affect worldview; WATCH and IGNORE do not."""
        assert RelationshipType.TRUST.affects_worldview is True
        assert RelationshipType.WATCH.affects_worldview is False
        assert RelationshipType.DISTRUST.affects_worldview is True
        assert RelationshipType.IGNORE.affects_worldview is False


# ===========================================================================
# TrustEdge with relationship_type tests
# ===========================================================================


class TestTrustEdgeRelationshipType:
    """Tests for TrustEdge with the relationship_type field."""

    def test_default_is_trust(self):
        """New edges default to TRUST relationship for backward compat."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
        )
        assert edge.relationship_type == RelationshipType.TRUST

    def test_explicit_relationship_type(self):
        """Can create edges with explicit relationship types."""
        for rt in RelationshipType:
            edge = TrustEdge(
                source_did="did:key:alice",
                target_did="did:key:bob",
                relationship_type=rt,
            )
            assert edge.relationship_type == rt

    def test_string_conversion(self):
        """String values are auto-converted to enum in __post_init__."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            relationship_type="watch",  # type: ignore[arg-type]
        )
        assert edge.relationship_type == RelationshipType.WATCH

    def test_to_dict_includes_relationship_type(self):
        """Serialization includes the relationship_type field."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            relationship_type=RelationshipType.WATCH,
        )
        d = edge.to_dict()
        assert d["relationship_type"] == "watch"

    def test_from_dict_parses_relationship_type(self):
        """Deserialization handles relationship_type correctly."""
        data = {
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "relationship_type": "distrust",
        }
        edge = TrustEdge.from_dict(data)
        assert edge.relationship_type == RelationshipType.DISTRUST

    def test_from_dict_defaults_to_trust(self):
        """Deserialization defaults to TRUST when field is missing (backward compat)."""
        data = {
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
        }
        edge = TrustEdge.from_dict(data)
        assert edge.relationship_type == RelationshipType.TRUST

    def test_with_decay_preserves_relationship_type(self):
        """with_decay() preserves the relationship_type."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            relationship_type=RelationshipType.WATCH,
        )
        decayed = edge.with_decay(0.1)
        assert decayed.relationship_type == RelationshipType.WATCH

    def test_with_delegation_preserves_relationship_type(self):
        """with_delegation() preserves the relationship_type."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            relationship_type=RelationshipType.DISTRUST,
        )
        delegated = edge.with_delegation(can_delegate=True)
        assert delegated.relationship_type == RelationshipType.DISTRUST


# ===========================================================================
# TrustService watch/unwatch/distrust/ignore tests
# ===========================================================================


class TestTrustServiceWatch:
    """Tests for TrustService watch/unwatch convenience methods."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_watch_creates_watch_edge(self):
        """watch() creates a WATCH relationship with zero trust scores."""
        edge = self.service.watch("did:key:alice", "did:key:bob")
        assert edge.relationship_type == RelationshipType.WATCH
        assert edge.competence == 0.0
        assert edge.integrity == 0.0
        assert edge.confidentiality == 0.0
        assert edge.judgment == 0.0

    def test_watch_with_domain(self):
        """watch() supports domain-specific watching."""
        edge = self.service.watch("did:key:alice", "did:key:bob", domain="crypto")
        assert edge.relationship_type == RelationshipType.WATCH
        assert edge.domain == "crypto"

    def test_unwatch_removes_watch_edge(self):
        """unwatch() removes a WATCH edge."""
        self.service.watch("did:key:alice", "did:key:bob")
        removed = self.service.unwatch("did:key:alice", "did:key:bob")
        assert removed is True
        # Edge is gone
        edge = self.service.get_trust("did:key:alice", "did:key:bob")
        assert edge is None

    def test_unwatch_does_not_remove_trust_edge(self):
        """unwatch() leaves TRUST edges intact."""
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        removed = self.service.unwatch("did:key:alice", "did:key:bob")
        assert removed is False
        # Edge still exists
        edge = self.service.get_trust("did:key:alice", "did:key:bob")
        assert edge is not None
        assert edge.relationship_type == RelationshipType.TRUST

    def test_unwatch_nonexistent(self):
        """unwatch() returns False when no edge exists."""
        removed = self.service.unwatch("did:key:alice", "did:key:bob")
        assert removed is False

    def test_list_watched(self):
        """list_watched() returns only WATCH edges."""
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        self.service.watch("did:key:alice", "did:key:carol")
        self.service.watch("did:key:alice", "did:key:dave")

        watched = self.service.list_watched("did:key:alice")
        assert len(watched) == 2
        dids = {e.target_did for e in watched}
        assert dids == {"did:key:carol", "did:key:dave"}


class TestTrustServiceDistrust:
    """Tests for TrustService distrust method."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_distrust_creates_distrust_edge(self):
        """distrust() creates a DISTRUST relationship."""
        edge = self.service.distrust("did:key:alice", "did:key:evil")
        assert edge.relationship_type == RelationshipType.DISTRUST

    def test_distrust_with_custom_scores(self):
        """distrust() accepts custom dimension scores."""
        edge = self.service.distrust(
            "did:key:alice",
            "did:key:evil",
            competence=0.9,
            integrity=0.1,
        )
        assert edge.competence == 0.9
        assert edge.integrity == 0.1


class TestTrustServiceIgnore:
    """Tests for TrustService ignore method."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_ignore_creates_ignore_edge(self):
        """ignore() creates an IGNORE relationship with zero scores."""
        edge = self.service.ignore("did:key:alice", "did:key:spammer")
        assert edge.relationship_type == RelationshipType.IGNORE
        assert edge.competence == 0.0
        assert edge.integrity == 0.0


# ===========================================================================
# Content visibility tests
# ===========================================================================


class TestContentVisibility:
    """Tests for content visibility based on relationship type."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_trust_shows_content(self):
        """TRUST relationship makes content visible."""
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        assert self.service.is_content_visible("did:key:alice", "did:key:bob") is True

    def test_watch_shows_content(self):
        """WATCH relationship makes content visible."""
        self.service.watch("did:key:alice", "did:key:bob")
        assert self.service.is_content_visible("did:key:alice", "did:key:bob") is True

    def test_distrust_hides_content(self):
        """DISTRUST relationship hides content by default."""
        self.service.distrust("did:key:alice", "did:key:evil")
        assert self.service.is_content_visible("did:key:alice", "did:key:evil") is False

    def test_ignore_hides_content(self):
        """IGNORE relationship hides content."""
        self.service.ignore("did:key:alice", "did:key:spammer")
        assert self.service.is_content_visible("did:key:alice", "did:key:spammer") is False

    def test_no_relationship_hides_content(self):
        """No relationship means content is not visible."""
        assert self.service.is_content_visible("did:key:alice", "did:key:stranger") is False


# ===========================================================================
# Reputation weight tests
# ===========================================================================


class TestReputationWeight:
    """Tests for reputation weight based on relationship type."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_trust_positive_reputation(self):
        """TRUST gives positive reputation weight."""
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        weight = self.service.get_reputation_weight("did:key:alice", "did:key:bob")
        assert weight > 0.0

    def test_watch_zero_reputation(self):
        """WATCH gives zero reputation weight."""
        self.service.watch("did:key:alice", "did:key:bob")
        weight = self.service.get_reputation_weight("did:key:alice", "did:key:bob")
        assert weight == 0.0

    def test_distrust_negative_reputation(self):
        """DISTRUST gives negative reputation weight."""
        self.service.distrust(
            "did:key:alice",
            "did:key:evil",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
        )
        weight = self.service.get_reputation_weight("did:key:alice", "did:key:evil")
        assert weight < 0.0

    def test_ignore_zero_reputation(self):
        """IGNORE gives zero reputation weight."""
        self.service.ignore("did:key:alice", "did:key:spammer")
        weight = self.service.get_reputation_weight("did:key:alice", "did:key:spammer")
        assert weight == 0.0

    def test_no_relationship_zero_reputation(self):
        """No relationship gives zero reputation weight."""
        weight = self.service.get_reputation_weight("did:key:alice", "did:key:stranger")
        assert weight == 0.0


# ===========================================================================
# Relationship type query tests
# ===========================================================================


class TestRelationshipTypeQuery:
    """Tests for querying relationship types."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_get_relationship_type(self):
        """get_relationship_type() returns the correct type."""
        self.service.watch("did:key:alice", "did:key:bob")
        rt = self.service.get_relationship_type("did:key:alice", "did:key:bob")
        assert rt == RelationshipType.WATCH

    def test_get_relationship_type_none(self):
        """get_relationship_type() returns None for no relationship."""
        rt = self.service.get_relationship_type("did:key:alice", "did:key:stranger")
        assert rt is None

    def test_list_by_relationship(self):
        """list_by_relationship() filters by relationship type."""
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        self.service.watch("did:key:alice", "did:key:carol")
        self.service.distrust("did:key:alice", "did:key:evil")
        self.service.ignore("did:key:alice", "did:key:spammer")

        trust_edges = self.service.list_by_relationship("did:key:alice", RelationshipType.TRUST)
        assert len(trust_edges) == 1
        assert trust_edges[0].target_did == "did:key:bob"

        watch_edges = self.service.list_by_relationship("did:key:alice", RelationshipType.WATCH)
        assert len(watch_edges) == 1
        assert watch_edges[0].target_did == "did:key:carol"

        distrust_edges = self.service.list_by_relationship("did:key:alice", RelationshipType.DISTRUST)
        assert len(distrust_edges) == 1
        assert distrust_edges[0].target_did == "did:key:evil"

        ignore_edges = self.service.list_by_relationship("did:key:alice", RelationshipType.IGNORE)
        assert len(ignore_edges) == 1
        assert ignore_edges[0].target_did == "did:key:spammer"


# ===========================================================================
# Relationship upgrade/downgrade tests
# ===========================================================================


class TestRelationshipTransitions:
    """Tests for transitioning between relationship types."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_watch_to_trust(self):
        """Can upgrade from WATCH to TRUST."""
        self.service.watch("did:key:alice", "did:key:bob")
        edge = self.service.get_trust("did:key:alice", "did:key:bob")
        assert edge.relationship_type == RelationshipType.WATCH

        # Upgrade to trust
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        edge = self.service.get_trust("did:key:alice", "did:key:bob")
        assert edge.relationship_type == RelationshipType.TRUST

    def test_trust_to_watch(self):
        """Can downgrade from TRUST to WATCH."""
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        # Downgrade to watch
        self.service.watch("did:key:alice", "did:key:bob")
        edge = self.service.get_trust("did:key:alice", "did:key:bob")
        assert edge.relationship_type == RelationshipType.WATCH

    def test_trust_to_distrust(self):
        """Can change from TRUST to DISTRUST."""
        self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            relationship_type=RelationshipType.TRUST,
        )
        self.service.distrust("did:key:alice", "did:key:bob")
        edge = self.service.get_trust("did:key:alice", "did:key:bob")
        assert edge.relationship_type == RelationshipType.DISTRUST

    def test_watch_to_ignore(self):
        """Can change from WATCH to IGNORE."""
        self.service.watch("did:key:alice", "did:key:bob")
        self.service.ignore("did:key:alice", "did:key:bob")
        edge = self.service.get_trust("did:key:alice", "did:key:bob")
        assert edge.relationship_type == RelationshipType.IGNORE


# ===========================================================================
# Backward compatibility tests
# ===========================================================================


class TestBackwardCompatibility:
    """Ensure existing trust operations work unchanged."""

    def setup_method(self):
        self.service = TrustService(use_memory=True)

    def test_grant_trust_defaults_to_trust_type(self):
        """grant_trust() without relationship_type defaults to TRUST."""
        edge = self.service.grant_trust(
            "did:key:alice",
            "did:key:bob",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
        )
        assert edge.relationship_type == RelationshipType.TRUST

    def test_existing_edges_deserialize_as_trust(self):
        """Edges without relationship_type field deserialize as TRUST."""
        data = {
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "competence": 0.8,
            "integrity": 0.7,
            "confidentiality": 0.6,
            "judgment": 0.3,
        }
        edge = TrustEdge.from_dict(data)
        assert edge.relationship_type == RelationshipType.TRUST

    def test_revoke_trust_works_for_all_types(self):
        """revoke_trust() works regardless of relationship type."""
        for rt in RelationshipType:
            self.service.grant_trust(
                "did:key:alice",
                f"did:key:{rt.value}",
                competence=0.5,
                integrity=0.5,
                confidentiality=0.5,
                relationship_type=rt,
            )
            revoked = self.service.revoke_trust("did:key:alice", f"did:key:{rt.value}")
            assert revoked is True
