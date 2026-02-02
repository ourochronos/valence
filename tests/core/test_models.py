"""Tests for valence.core.models module."""

from __future__ import annotations

import json
from datetime import datetime
from uuid import UUID, uuid4

import pytest

from valence.core.models import (
    Belief,
    BeliefEntity,
    BeliefStatus,
    Entity,
    EntityRole,
    EntityType,
    Exchange,
    ExchangeRole,
    Pattern,
    PatternStatus,
    Platform,
    Session,
    SessionInsight,
    SessionStatus,
    Source,
    Tension,
    TensionSeverity,
    TensionStatus,
    TensionType,
)
from valence.core.confidence import DimensionalConfidence


# ============================================================================
# Enum Tests
# ============================================================================

class TestBeliefStatus:
    """Tests for BeliefStatus enum."""

    def test_all_values_exist(self):
        """All expected status values should exist."""
        expected = {"ACTIVE", "SUPERSEDED", "DISPUTED", "ARCHIVED"}
        actual = {s.name for s in BeliefStatus}
        assert actual == expected

    def test_values_are_lowercase(self):
        """All values should be lowercase."""
        for status in BeliefStatus:
            assert status.value == status.value.lower()

    def test_string_behavior(self):
        """Should behave as string via value."""
        assert BeliefStatus.ACTIVE == "active"
        assert BeliefStatus.ACTIVE.value == "active"


class TestEntityType:
    """Tests for EntityType enum."""

    def test_all_values_exist(self):
        """All expected type values should exist."""
        expected = {
            "PERSON", "ORGANIZATION", "TOOL", "CONCEPT",
            "PROJECT", "LOCATION", "SERVICE"
        }
        actual = {t.name for t in EntityType}
        assert actual == expected


class TestEntityRole:
    """Tests for EntityRole enum."""

    def test_all_values_exist(self):
        """All expected role values should exist."""
        expected = {"SUBJECT", "OBJECT", "CONTEXT", "SOURCE"}
        actual = {r.name for r in EntityRole}
        assert actual == expected


class TestSessionStatus:
    """Tests for SessionStatus enum."""

    def test_all_values_exist(self):
        """All expected status values should exist."""
        expected = {"ACTIVE", "COMPLETED", "ABANDONED"}
        actual = {s.name for s in SessionStatus}
        assert actual == expected


class TestPlatform:
    """Tests for Platform enum."""

    def test_all_values_exist(self):
        """All expected platform values should exist."""
        expected = {"CLAUDE_CODE", "MATRIX", "API", "SLACK"}
        actual = {p.name for p in Platform}
        assert actual == expected

    def test_claude_code_value(self):
        """CLAUDE_CODE should have hyphenated value."""
        assert Platform.CLAUDE_CODE.value == "claude-code"


class TestExchangeRole:
    """Tests for ExchangeRole enum."""

    def test_all_values_exist(self):
        """All expected role values should exist."""
        expected = {"USER", "ASSISTANT", "SYSTEM"}
        actual = {r.name for r in ExchangeRole}
        assert actual == expected


class TestPatternStatus:
    """Tests for PatternStatus enum."""

    def test_all_values_exist(self):
        """All expected status values should exist."""
        expected = {"EMERGING", "ESTABLISHED", "FADING", "ARCHIVED"}
        actual = {s.name for s in PatternStatus}
        assert actual == expected


class TestTensionType:
    """Tests for TensionType enum."""

    def test_all_values_exist(self):
        """All expected type values should exist."""
        expected = {
            "CONTRADICTION", "TEMPORAL_CONFLICT",
            "SCOPE_CONFLICT", "PARTIAL_OVERLAP"
        }
        actual = {t.name for t in TensionType}
        assert actual == expected


class TestTensionSeverity:
    """Tests for TensionSeverity enum."""

    def test_all_values_exist(self):
        """All expected severity values should exist."""
        expected = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        actual = {s.name for s in TensionSeverity}
        assert actual == expected


class TestTensionStatus:
    """Tests for TensionStatus enum."""

    def test_all_values_exist(self):
        """All expected status values should exist."""
        expected = {"DETECTED", "INVESTIGATING", "RESOLVED", "ACCEPTED"}
        actual = {s.name for s in TensionStatus}
        assert actual == expected


# ============================================================================
# Source Model Tests
# ============================================================================

class TestSource:
    """Tests for Source dataclass."""

    def test_create_minimal(self):
        """Create source with minimal fields."""
        source_id = uuid4()
        source = Source(id=source_id, type="conversation")
        assert source.id == source_id
        assert source.type == "conversation"
        assert source.title is None
        assert source.url is None

    def test_create_full(self):
        """Create source with all fields."""
        source_id = uuid4()
        session_id = uuid4()
        now = datetime.now()
        source = Source(
            id=source_id,
            type="document",
            title="Test Document",
            url="https://example.com/doc",
            content_hash="abc123",
            session_id=session_id,
            metadata={"key": "value"},
            created_at=now,
        )
        assert source.id == source_id
        assert source.type == "document"
        assert source.title == "Test Document"
        assert source.url == "https://example.com/doc"
        assert source.content_hash == "abc123"
        assert source.session_id == session_id
        assert source.metadata == {"key": "value"}
        assert source.created_at == now

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        source_id = uuid4()
        session_id = uuid4()
        now = datetime.now()
        source = Source(
            id=source_id,
            type="document",
            title="Test",
            session_id=session_id,
            created_at=now,
        )
        d = source.to_dict()
        assert d["id"] == str(source_id)
        assert d["type"] == "document"
        assert d["title"] == "Test"
        assert d["session_id"] == str(session_id)
        assert d["created_at"] == now.isoformat()

    def test_from_row(self, source_row_factory):
        """from_row should parse database row."""
        row = source_row_factory(type="document", title="Test Doc")
        source = Source.from_row(row)
        assert source.type == "document"
        assert source.title == "Test Doc"

    def test_from_row_with_string_id(self, source_row_factory):
        """from_row should handle string IDs."""
        source_id = uuid4()
        row = source_row_factory(id=str(source_id))
        source = Source.from_row(row)
        assert source.id == source_id


# ============================================================================
# Entity Model Tests
# ============================================================================

class TestEntity:
    """Tests for Entity dataclass."""

    def test_create_minimal(self):
        """Create entity with minimal fields."""
        entity_id = uuid4()
        entity = Entity(id=entity_id, name="Test Entity", type=EntityType.CONCEPT)
        assert entity.id == entity_id
        assert entity.name == "Test Entity"
        assert entity.type == EntityType.CONCEPT
        assert entity.aliases == []

    def test_create_full(self):
        """Create entity with all fields."""
        entity_id = uuid4()
        canonical_id = uuid4()
        now = datetime.now()
        entity = Entity(
            id=entity_id,
            name="Claude",
            type=EntityType.TOOL,
            description="An AI assistant",
            aliases=["Claude AI", "Anthropic Claude"],
            canonical_id=canonical_id,
            created_at=now,
            modified_at=now,
        )
        assert entity.name == "Claude"
        assert entity.type == EntityType.TOOL
        assert "Claude AI" in entity.aliases
        assert entity.canonical_id == canonical_id

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        entity_id = uuid4()
        now = datetime.now()
        entity = Entity(
            id=entity_id,
            name="Test",
            type=EntityType.PERSON,
            created_at=now,
            modified_at=now,
        )
        d = entity.to_dict()
        assert d["id"] == str(entity_id)
        assert d["name"] == "Test"
        assert d["type"] == "person"
        assert d["created_at"] == now.isoformat()

    def test_from_row(self, entity_row_factory):
        """from_row should parse database row."""
        row = entity_row_factory(name="Test Person", type="person")
        entity = Entity.from_row(row)
        assert entity.name == "Test Person"
        assert entity.type == EntityType.PERSON


# ============================================================================
# Belief Model Tests
# ============================================================================

class TestBelief:
    """Tests for Belief dataclass."""

    def test_create_minimal(self):
        """Create belief with minimal fields."""
        belief_id = uuid4()
        confidence = DimensionalConfidence(overall=0.8)
        belief = Belief(id=belief_id, content="Test belief", confidence=confidence)
        assert belief.id == belief_id
        assert belief.content == "Test belief"
        assert belief.confidence.overall == 0.8
        assert belief.status == BeliefStatus.ACTIVE

    def test_create_full(self):
        """Create belief with all fields."""
        belief_id = uuid4()
        source_id = uuid4()
        supersedes_id = uuid4()
        confidence = DimensionalConfidence(overall=0.9)
        now = datetime.now()
        belief = Belief(
            id=belief_id,
            content="Full belief",
            confidence=confidence,
            domain_path=["tech", "python"],
            valid_from=now,
            valid_until=None,
            created_at=now,
            modified_at=now,
            source_id=source_id,
            extraction_method="manual",
            supersedes_id=supersedes_id,
            superseded_by_id=None,
            status=BeliefStatus.ACTIVE,
        )
        assert belief.domain_path == ["tech", "python"]
        assert belief.source_id == source_id
        assert belief.supersedes_id == supersedes_id

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        belief_id = uuid4()
        source_id = uuid4()
        now = datetime.now()
        confidence = DimensionalConfidence(overall=0.8)
        belief = Belief(
            id=belief_id,
            content="Test",
            confidence=confidence,
            source_id=source_id,
            created_at=now,
            modified_at=now,
        )
        d = belief.to_dict()
        assert d["id"] == str(belief_id)
        assert d["content"] == "Test"
        assert d["confidence"]["overall"] == 0.8
        assert d["source_id"] == str(source_id)
        assert d["status"] == "active"

    def test_to_dict_with_entities(self):
        """to_dict should include entities if loaded."""
        belief_id = uuid4()
        entity_id = uuid4()
        now = datetime.now()
        entity = Entity(
            id=entity_id,
            name="Test",
            type=EntityType.CONCEPT,
            created_at=now,
            modified_at=now,
        )
        belief = Belief(
            id=belief_id,
            content="Test",
            confidence=DimensionalConfidence(overall=0.8),
            created_at=now,
            modified_at=now,
            entities=[(entity, EntityRole.SUBJECT)],
        )
        d = belief.to_dict()
        assert "entities" in d
        assert len(d["entities"]) == 1
        assert d["entities"][0]["role"] == "subject"

    def test_from_row(self, belief_row_factory):
        """from_row should parse database row."""
        row = belief_row_factory(
            content="Test content",
            confidence={"overall": 0.75},
            status="active",
        )
        belief = Belief.from_row(row)
        assert belief.content == "Test content"
        assert belief.confidence.overall == 0.75
        assert belief.status == BeliefStatus.ACTIVE

    def test_from_row_with_json_string_confidence(self, belief_row_factory):
        """from_row should handle JSON string confidence."""
        row = belief_row_factory(
            confidence=json.dumps({"overall": 0.85}),
        )
        # The factory already does json.dumps, so we need raw string
        row["confidence"] = '{"overall": 0.85}'
        belief = Belief.from_row(row)
        assert belief.confidence.overall == 0.85


# ============================================================================
# BeliefEntity Model Tests
# ============================================================================

class TestBeliefEntity:
    """Tests for BeliefEntity dataclass."""

    def test_create(self):
        """Create belief entity link."""
        belief_id = uuid4()
        entity_id = uuid4()
        now = datetime.now()
        be = BeliefEntity(
            belief_id=belief_id,
            entity_id=entity_id,
            role=EntityRole.SUBJECT,
            created_at=now,
        )
        assert be.belief_id == belief_id
        assert be.entity_id == entity_id
        assert be.role == EntityRole.SUBJECT

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        belief_id = uuid4()
        entity_id = uuid4()
        now = datetime.now()
        be = BeliefEntity(
            belief_id=belief_id,
            entity_id=entity_id,
            role=EntityRole.OBJECT,
            created_at=now,
        )
        d = be.to_dict()
        assert d["belief_id"] == str(belief_id)
        assert d["entity_id"] == str(entity_id)
        assert d["role"] == "object"


# ============================================================================
# Tension Model Tests
# ============================================================================

class TestTension:
    """Tests for Tension dataclass."""

    def test_create_minimal(self):
        """Create tension with minimal fields."""
        tension_id = uuid4()
        belief_a = uuid4()
        belief_b = uuid4()
        tension = Tension(id=tension_id, belief_a_id=belief_a, belief_b_id=belief_b)
        assert tension.id == tension_id
        assert tension.belief_a_id == belief_a
        assert tension.belief_b_id == belief_b
        assert tension.type == TensionType.CONTRADICTION
        assert tension.severity == TensionSeverity.MEDIUM
        assert tension.status == TensionStatus.DETECTED

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        tension_id = uuid4()
        belief_a = uuid4()
        belief_b = uuid4()
        now = datetime.now()
        tension = Tension(
            id=tension_id,
            belief_a_id=belief_a,
            belief_b_id=belief_b,
            description="Conflicting statements",
            detected_at=now,
        )
        d = tension.to_dict()
        assert d["id"] == str(tension_id)
        assert d["belief_a_id"] == str(belief_a)
        assert d["type"] == "contradiction"
        assert d["severity"] == "medium"

    def test_from_row(self, tension_row_factory):
        """from_row should parse database row."""
        row = tension_row_factory(
            type="temporal_conflict",
            severity="high",
            status="investigating",
        )
        tension = Tension.from_row(row)
        assert tension.type == TensionType.TEMPORAL_CONFLICT
        assert tension.severity == TensionSeverity.HIGH
        assert tension.status == TensionStatus.INVESTIGATING


# ============================================================================
# Session Model Tests
# ============================================================================

class TestSession:
    """Tests for Session dataclass."""

    def test_create_minimal(self):
        """Create session with minimal fields."""
        session_id = uuid4()
        session = Session(id=session_id, platform=Platform.CLAUDE_CODE)
        assert session.id == session_id
        assert session.platform == Platform.CLAUDE_CODE
        assert session.status == SessionStatus.ACTIVE

    def test_create_full(self):
        """Create session with all fields."""
        session_id = uuid4()
        now = datetime.now()
        session = Session(
            id=session_id,
            platform=Platform.MATRIX,
            project_context="test-project",
            status=SessionStatus.COMPLETED,
            summary="A test session",
            themes=["testing", "development"],
            started_at=now,
            ended_at=now,
            claude_session_id="claude-123",
            external_room_id="!room:matrix.org",
            metadata={"key": "value"},
            exchange_count=10,
            insight_count=2,
        )
        assert session.project_context == "test-project"
        assert session.themes == ["testing", "development"]
        assert session.external_room_id == "!room:matrix.org"
        assert session.exchange_count == 10

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        session_id = uuid4()
        now = datetime.now()
        session = Session(
            id=session_id,
            platform=Platform.API,
            started_at=now,
        )
        d = session.to_dict()
        assert d["id"] == str(session_id)
        assert d["platform"] == "api"
        assert d["status"] == "active"

    def test_from_row(self, session_row_factory):
        """from_row should parse database row."""
        row = session_row_factory(
            platform="matrix",
            status="completed",
            themes=["theme1", "theme2"],
        )
        session = Session.from_row(row)
        assert session.platform == Platform.MATRIX
        assert session.status == SessionStatus.COMPLETED
        assert session.themes == ["theme1", "theme2"]


# ============================================================================
# Exchange Model Tests
# ============================================================================

class TestExchange:
    """Tests for Exchange dataclass."""

    def test_create_minimal(self):
        """Create exchange with minimal fields."""
        exchange_id = uuid4()
        session_id = uuid4()
        exchange = Exchange(
            id=exchange_id,
            session_id=session_id,
            sequence=1,
            role=ExchangeRole.USER,
            content="Hello",
        )
        assert exchange.id == exchange_id
        assert exchange.sequence == 1
        assert exchange.role == ExchangeRole.USER
        assert exchange.content == "Hello"

    def test_create_full(self):
        """Create exchange with all fields."""
        exchange_id = uuid4()
        session_id = uuid4()
        now = datetime.now()
        exchange = Exchange(
            id=exchange_id,
            session_id=session_id,
            sequence=5,
            role=ExchangeRole.ASSISTANT,
            content="Here's the answer",
            created_at=now,
            tokens_approx=150,
            tool_uses=[{"tool": "read", "path": "/file.py"}],
        )
        assert exchange.tokens_approx == 150
        assert len(exchange.tool_uses) == 1

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        exchange_id = uuid4()
        session_id = uuid4()
        now = datetime.now()
        exchange = Exchange(
            id=exchange_id,
            session_id=session_id,
            sequence=1,
            role=ExchangeRole.SYSTEM,
            content="System message",
            created_at=now,
        )
        d = exchange.to_dict()
        assert d["id"] == str(exchange_id)
        assert d["session_id"] == str(session_id)
        assert d["role"] == "system"

    def test_from_row(self, exchange_row_factory):
        """from_row should parse database row."""
        row = exchange_row_factory(
            sequence=3,
            role="assistant",
            content="Test response",
        )
        exchange = Exchange.from_row(row)
        assert exchange.sequence == 3
        assert exchange.role == ExchangeRole.ASSISTANT
        assert exchange.content == "Test response"


# ============================================================================
# Pattern Model Tests
# ============================================================================

class TestPattern:
    """Tests for Pattern dataclass."""

    def test_create_minimal(self):
        """Create pattern with minimal fields."""
        pattern_id = uuid4()
        pattern = Pattern(
            id=pattern_id,
            type="preference",
            description="User prefers dark mode",
        )
        assert pattern.id == pattern_id
        assert pattern.type == "preference"
        assert pattern.occurrence_count == 1
        assert pattern.confidence == 0.5
        assert pattern.status == PatternStatus.EMERGING

    def test_create_full(self):
        """Create pattern with all fields."""
        pattern_id = uuid4()
        session_ids = [uuid4(), uuid4()]
        now = datetime.now()
        pattern = Pattern(
            id=pattern_id,
            type="topic_recurrence",
            description="Frequently discusses testing",
            evidence=session_ids,
            occurrence_count=5,
            confidence=0.8,
            status=PatternStatus.ESTABLISHED,
            first_observed=now,
            last_observed=now,
        )
        assert len(pattern.evidence) == 2
        assert pattern.occurrence_count == 5
        assert pattern.status == PatternStatus.ESTABLISHED

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        pattern_id = uuid4()
        session_id = uuid4()
        now = datetime.now()
        pattern = Pattern(
            id=pattern_id,
            type="working_style",
            description="Iterative development",
            evidence=[session_id],
            first_observed=now,
            last_observed=now,
        )
        d = pattern.to_dict()
        assert d["id"] == str(pattern_id)
        assert d["type"] == "working_style"
        assert d["evidence"] == [str(session_id)]
        assert d["status"] == "emerging"

    def test_from_row(self, pattern_row_factory):
        """from_row should parse database row."""
        session_ids = [str(uuid4()), str(uuid4())]
        row = pattern_row_factory(
            type="preference",
            description="Test pattern",
            evidence=session_ids,
            confidence=0.75,
            status="established",
        )
        pattern = Pattern.from_row(row)
        assert pattern.type == "preference"
        assert pattern.confidence == 0.75
        assert pattern.status == PatternStatus.ESTABLISHED
        assert len(pattern.evidence) == 2

    def test_from_row_with_uuid_evidence(self, pattern_row_factory):
        """from_row should handle UUID evidence list."""
        session_ids = [uuid4(), uuid4()]
        row = pattern_row_factory(evidence=session_ids)
        pattern = Pattern.from_row(row)
        assert pattern.evidence == session_ids


# ============================================================================
# SessionInsight Model Tests
# ============================================================================

class TestSessionInsight:
    """Tests for SessionInsight dataclass."""

    def test_create(self):
        """Create session insight."""
        insight_id = uuid4()
        session_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()
        insight = SessionInsight(
            id=insight_id,
            session_id=session_id,
            belief_id=belief_id,
            extraction_method="llm_extraction",
            extracted_at=now,
        )
        assert insight.id == insight_id
        assert insight.session_id == session_id
        assert insight.belief_id == belief_id
        assert insight.extraction_method == "llm_extraction"

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        insight_id = uuid4()
        session_id = uuid4()
        belief_id = uuid4()
        now = datetime.now()
        insight = SessionInsight(
            id=insight_id,
            session_id=session_id,
            belief_id=belief_id,
            extracted_at=now,
        )
        d = insight.to_dict()
        assert d["id"] == str(insight_id)
        assert d["session_id"] == str(session_id)
        assert d["belief_id"] == str(belief_id)
        assert d["extraction_method"] == "manual"


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestModelEdgeCases:
    """Edge case tests for models."""

    def test_belief_with_empty_domain_path(self):
        """Belief should handle empty domain path."""
        belief = Belief(
            id=uuid4(),
            content="Test",
            confidence=DimensionalConfidence(overall=0.7),
            domain_path=[],
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        d = belief.to_dict()
        assert d["domain_path"] == []

    def test_session_with_empty_themes(self):
        """Session should handle empty themes."""
        session = Session(
            id=uuid4(),
            platform=Platform.API,
            themes=[],
            started_at=datetime.now(),
        )
        d = session.to_dict()
        assert d["themes"] == []

    def test_exchange_with_empty_tool_uses(self):
        """Exchange should handle empty tool uses."""
        exchange = Exchange(
            id=uuid4(),
            session_id=uuid4(),
            sequence=1,
            role=ExchangeRole.USER,
            content="Test",
            tool_uses=[],
            created_at=datetime.now(),
        )
        d = exchange.to_dict()
        assert d["tool_uses"] == []

    def test_entity_with_empty_aliases(self):
        """Entity should handle empty aliases."""
        entity = Entity(
            id=uuid4(),
            name="Test",
            type=EntityType.CONCEPT,
            aliases=[],
            created_at=datetime.now(),
            modified_at=datetime.now(),
        )
        d = entity.to_dict()
        assert d["aliases"] == []
