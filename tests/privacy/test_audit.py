"""Tests for audit event logging - AuditEvent and AuditLogger."""

import pytest
import tempfile
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from valence.privacy.audit import (
    AuditEventType,
    AuditEvent,
    AuditLogger,
    InMemoryAuditBackend,
    FileAuditBackend,
    get_audit_logger,
    set_audit_logger,
)


class TestAuditEventType:
    """Tests for AuditEventType enum."""
    
    def test_sharing_events_exist(self):
        """Verify sharing event types are defined."""
        assert AuditEventType.SHARE.value == "share"
        assert AuditEventType.RECEIVE.value == "receive"
        assert AuditEventType.REVOKE.value == "revoke"
    
    def test_trust_events_exist(self):
        """Verify trust event types are defined."""
        assert AuditEventType.GRANT_TRUST.value == "grant_trust"
        assert AuditEventType.REVOKE_TRUST.value == "revoke_trust"
        assert AuditEventType.UPDATE_TRUST.value == "update_trust"
    
    def test_domain_events_exist(self):
        """Verify domain event types are defined."""
        assert AuditEventType.DOMAIN_CREATE.value == "domain_create"
        assert AuditEventType.MEMBER_ADD.value == "member_add"
        assert AuditEventType.ROLE_CHANGE.value == "role_change"
    
    def test_access_events_exist(self):
        """Verify access event types are defined."""
        assert AuditEventType.ACCESS_GRANTED.value == "access_granted"
        assert AuditEventType.ACCESS_DENIED.value == "access_denied"
    
    def test_type_from_string(self):
        """Test creating type from string value."""
        assert AuditEventType("share") == AuditEventType.SHARE
        assert AuditEventType("grant_trust") == AuditEventType.GRANT_TRUST


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""
    
    def test_create_event(self):
        """Test creating an audit event."""
        event = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            target_did="did:key:bob",
            resource="belief:123",
            action="share_belief",
            success=True,
        )
        
        assert event.event_type == AuditEventType.SHARE
        assert event.actor_did == "did:key:alice"
        assert event.target_did == "did:key:bob"
        assert event.resource == "belief:123"
        assert event.action == "share_belief"
        assert event.success is True
        assert event.event_id is not None
        assert event.timestamp is not None
        assert event.metadata == {}
    
    def test_auto_generated_fields(self):
        """Test that event_id and timestamp are auto-generated."""
        event1 = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:1",
            action="share",
            success=True,
        )
        event2 = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:2",
            action="share",
            success=True,
        )
        
        # Each event gets unique ID
        assert event1.event_id != event2.event_id
        # Timestamps should be very close
        assert abs((event2.timestamp - event1.timestamp).total_seconds()) < 1
    
    def test_metadata(self):
        """Test event with metadata."""
        event = AuditEvent(
            event_type=AuditEventType.GRANT_TRUST,
            actor_did="did:key:alice",
            target_did="did:key:bob",
            resource="trust:alice:bob",
            action="grant_trust",
            success=True,
            metadata={"trust_level": 0.8, "domain": "medical"},
        )
        
        assert event.metadata["trust_level"] == 0.8
        assert event.metadata["domain"] == "medical"
    
    def test_to_dict(self):
        """Test serialization to dict."""
        timestamp = datetime(2026, 2, 4, 12, 0, 0, tzinfo=timezone.utc)
        event = AuditEvent(
            event_id="test-event-123",
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            target_did="did:key:bob",
            resource="belief:456",
            action="share_belief",
            timestamp=timestamp,
            success=True,
            metadata={"hop_count": 1},
        )
        
        data = event.to_dict()
        assert data["event_id"] == "test-event-123"
        assert data["event_type"] == "share"
        assert data["actor_did"] == "did:key:alice"
        assert data["target_did"] == "did:key:bob"
        assert data["resource"] == "belief:456"
        assert data["action"] == "share_belief"
        assert data["success"] is True
        assert data["metadata"] == {"hop_count": 1}
        assert "2026-02-04" in data["timestamp"]
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "event_id": "evt-789",
            "event_type": "revoke",
            "actor_did": "did:key:alice",
            "target_did": "did:key:bob",
            "resource": "belief:100",
            "action": "revoke_share",
            "timestamp": "2026-02-04T15:30:00+00:00",
            "success": False,
            "metadata": {"reason": "expired"},
        }
        
        event = AuditEvent.from_dict(data)
        assert event.event_id == "evt-789"
        assert event.event_type == AuditEventType.REVOKE
        assert event.actor_did == "did:key:alice"
        assert event.target_did == "did:key:bob"
        assert event.resource == "belief:100"
        assert event.success is False
        assert event.metadata["reason"] == "expired"
    
    def test_from_dict_minimal(self):
        """Test deserialization with minimal data."""
        data = {
            "event_type": "share",
            "actor_did": "did:key:alice",
            "resource": "belief:1",
            "action": "share",
        }
        
        event = AuditEvent.from_dict(data)
        assert event.event_type == AuditEventType.SHARE
        assert event.actor_did == "did:key:alice"
        assert event.target_did is None
        assert event.success is True  # default
        assert event.metadata == {}  # default
    
    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            actor_did="did:key:mallory",
            target_did="did:key:alice",
            resource="belief:secret",
            action="access_denied",
            success=False,
            metadata={"reason": "insufficient_trust", "required": 0.8},
        )
        
        restored = AuditEvent.from_dict(original.to_dict())
        assert restored.event_id == original.event_id
        assert restored.event_type == original.event_type
        assert restored.actor_did == original.actor_did
        assert restored.target_did == original.target_did
        assert restored.resource == original.resource
        assert restored.action == original.action
        assert restored.success == original.success
        assert restored.metadata == original.metadata
    
    def test_to_json(self):
        """Test JSON serialization."""
        event = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:1",
            action="share",
            success=True,
        )
        
        json_str = event.to_json()
        data = json.loads(json_str)
        assert data["event_type"] == "share"
        assert data["actor_did"] == "did:key:alice"
    
    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"event_type": "share", "actor_did": "did:key:bob", "resource": "belief:2", "action": "share", "success": true}'
        event = AuditEvent.from_json(json_str)
        assert event.event_type == AuditEventType.SHARE
        assert event.actor_did == "did:key:bob"


class TestInMemoryAuditBackend:
    """Tests for InMemoryAuditBackend."""
    
    def test_write_and_query(self):
        """Test writing and querying events."""
        backend = InMemoryAuditBackend()
        
        event = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            target_did="did:key:bob",
            resource="belief:1",
            action="share",
            success=True,
        )
        backend.write(event)
        
        results = backend.query()
        assert len(results) == 1
        assert results[0].event_id == event.event_id
    
    def test_query_by_event_type(self):
        """Test filtering by event type."""
        backend = InMemoryAuditBackend()
        
        backend.write(AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:1",
            action="share",
            success=True,
        ))
        backend.write(AuditEvent(
            event_type=AuditEventType.GRANT_TRUST,
            actor_did="did:key:alice",
            resource="trust:1",
            action="grant",
            success=True,
        ))
        
        results = backend.query(event_type=AuditEventType.SHARE)
        assert len(results) == 1
        assert results[0].event_type == AuditEventType.SHARE
    
    def test_query_by_actor(self):
        """Test filtering by actor DID."""
        backend = InMemoryAuditBackend()
        
        backend.write(AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:1",
            action="share",
            success=True,
        ))
        backend.write(AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:bob",
            resource="belief:2",
            action="share",
            success=True,
        ))
        
        results = backend.query(actor_did="did:key:alice")
        assert len(results) == 1
        assert results[0].actor_did == "did:key:alice"
    
    def test_query_by_time_range(self):
        """Test filtering by time range."""
        backend = InMemoryAuditBackend()
        
        now = datetime.now(timezone.utc)
        old_event = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:old",
            action="share",
            success=True,
            timestamp=now - timedelta(days=7),
        )
        new_event = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:new",
            action="share",
            success=True,
            timestamp=now,
        )
        
        backend.write(old_event)
        backend.write(new_event)
        
        # Query last 3 days
        results = backend.query(start_time=now - timedelta(days=3))
        assert len(results) == 1
        assert results[0].resource == "belief:new"
    
    def test_query_limit(self):
        """Test query result limit."""
        backend = InMemoryAuditBackend()
        
        for i in range(10):
            backend.write(AuditEvent(
                event_type=AuditEventType.SHARE,
                actor_did="did:key:alice",
                resource=f"belief:{i}",
                action="share",
                success=True,
            ))
        
        results = backend.query(limit=3)
        assert len(results) == 3
    
    def test_count(self):
        """Test counting events."""
        backend = InMemoryAuditBackend()
        
        for i in range(5):
            backend.write(AuditEvent(
                event_type=AuditEventType.SHARE,
                actor_did="did:key:alice",
                resource=f"belief:{i}",
                action="share",
                success=i % 2 == 0,  # alternating success
            ))
        
        assert backend.count() == 5
        assert backend.count(event_type=AuditEventType.SHARE) == 5
        assert backend.count(success=True) == 3
        assert backend.count(success=False) == 2
    
    def test_max_events_limit(self):
        """Test that backend respects max events limit."""
        backend = InMemoryAuditBackend(max_events=5)
        
        for i in range(10):
            backend.write(AuditEvent(
                event_type=AuditEventType.SHARE,
                actor_did="did:key:alice",
                resource=f"belief:{i}",
                action="share",
                success=True,
            ))
        
        events = backend.all_events()
        assert len(events) == 5
        # Should have the most recent 5 events
        assert events[0].resource == "belief:5"
        assert events[4].resource == "belief:9"
    
    def test_clear(self):
        """Test clearing all events."""
        backend = InMemoryAuditBackend()
        
        backend.write(AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:1",
            action="share",
            success=True,
        ))
        
        assert backend.count() == 1
        backend.clear()
        assert backend.count() == 0


class TestFileAuditBackend:
    """Tests for FileAuditBackend."""
    
    def test_write_and_query(self):
        """Test writing and querying events."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            backend = FileAuditBackend(log_path)
            
            event = AuditEvent(
                event_type=AuditEventType.SHARE,
                actor_did="did:key:alice",
                target_did="did:key:bob",
                resource="belief:1",
                action="share",
                success=True,
            )
            backend.write(event)
            
            results = backend.query()
            assert len(results) == 1
            assert results[0].event_id == event.event_id
    
    def test_persistence(self):
        """Test that events persist across backend instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            
            # Write with first backend
            backend1 = FileAuditBackend(log_path)
            backend1.write(AuditEvent(
                event_type=AuditEventType.SHARE,
                actor_did="did:key:alice",
                resource="belief:1",
                action="share",
                success=True,
            ))
            
            # Read with second backend
            backend2 = FileAuditBackend(log_path)
            results = backend2.query()
            assert len(results) == 1
    
    def test_query_filters(self):
        """Test query filters on file backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            backend = FileAuditBackend(log_path)
            
            backend.write(AuditEvent(
                event_type=AuditEventType.SHARE,
                actor_did="did:key:alice",
                resource="belief:1",
                action="share",
                success=True,
            ))
            backend.write(AuditEvent(
                event_type=AuditEventType.GRANT_TRUST,
                actor_did="did:key:bob",
                resource="trust:1",
                action="grant",
                success=True,
            ))
            
            results = backend.query(event_type=AuditEventType.SHARE)
            assert len(results) == 1
            assert results[0].event_type == AuditEventType.SHARE
            
            results = backend.query(actor_did="did:key:bob")
            assert len(results) == 1
            assert results[0].actor_did == "did:key:bob"
    
    def test_count(self):
        """Test counting events in file backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            backend = FileAuditBackend(log_path)
            
            for i in range(5):
                backend.write(AuditEvent(
                    event_type=AuditEventType.SHARE,
                    actor_did="did:key:alice",
                    resource=f"belief:{i}",
                    action="share",
                    success=i % 2 == 0,
                ))
            
            assert backend.count() == 5
            assert backend.count(success=True) == 3
    
    def test_empty_file_query(self):
        """Test querying non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nonexistent.jsonl"
            backend = FileAuditBackend(log_path)
            
            results = backend.query()
            assert results == []
            assert backend.count() == 0
    
    def test_creates_parent_directories(self):
        """Test that parent directories are created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nested" / "dir" / "audit.jsonl"
            backend = FileAuditBackend(log_path)
            
            backend.write(AuditEvent(
                event_type=AuditEventType.SHARE,
                actor_did="did:key:alice",
                resource="belief:1",
                action="share",
                success=True,
            ))
            
            assert log_path.exists()


class TestAuditLogger:
    """Tests for AuditLogger service."""
    
    def test_log_event(self):
        """Test logging an event."""
        logger = AuditLogger()
        
        event = AuditEvent(
            event_type=AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:1",
            action="share",
            success=True,
        )
        logger.log_event(event)
        
        results = logger.query()
        assert len(results) == 1
    
    def test_log_convenience_method(self):
        """Test the log() convenience method."""
        logger = AuditLogger()
        
        event = logger.log(
            AuditEventType.SHARE,
            actor_did="did:key:alice",
            resource="belief:1",
            action="share",
            success=True,
            hop_count=1,
        )
        
        assert event.event_type == AuditEventType.SHARE
        assert event.metadata["hop_count"] == 1
        
        results = logger.query()
        assert len(results) == 1
    
    def test_log_share(self):
        """Test log_share convenience method."""
        logger = AuditLogger()
        
        event = logger.log_share(
            actor_did="did:key:alice",
            target_did="did:key:bob",
            resource="belief:123",
            share_level="bounded",
        )
        
        assert event.event_type == AuditEventType.SHARE
        assert event.actor_did == "did:key:alice"
        assert event.target_did == "did:key:bob"
        assert event.metadata["share_level"] == "bounded"
    
    def test_log_receive(self):
        """Test log_receive convenience method."""
        logger = AuditLogger()
        
        event = logger.log_receive(
            actor_did="did:key:bob",
            source_did="did:key:alice",
            resource="belief:123",
        )
        
        assert event.event_type == AuditEventType.RECEIVE
        assert event.actor_did == "did:key:bob"
        assert event.target_did == "did:key:alice"
    
    def test_log_revoke(self):
        """Test log_revoke convenience method."""
        logger = AuditLogger()
        
        event = logger.log_revoke(
            actor_did="did:key:alice",
            target_did="did:key:bob",
            resource="belief:123",
            reason="expired",
        )
        
        assert event.event_type == AuditEventType.REVOKE
        assert event.metadata["reason"] == "expired"
    
    def test_log_grant_trust(self):
        """Test log_grant_trust convenience method."""
        logger = AuditLogger()
        
        event = logger.log_grant_trust(
            actor_did="did:key:alice",
            target_did="did:key:bob",
            trust_level=0.8,
            domain="medical",
        )
        
        assert event.event_type == AuditEventType.GRANT_TRUST
        assert event.metadata["trust_level"] == 0.8
        assert event.metadata["domain"] == "medical"
        assert event.resource == "trust:did:key:alice:did:key:bob"
    
    def test_log_revoke_trust(self):
        """Test log_revoke_trust convenience method."""
        logger = AuditLogger()
        
        event = logger.log_revoke_trust(
            actor_did="did:key:alice",
            target_did="did:key:bob",
        )
        
        assert event.event_type == AuditEventType.REVOKE_TRUST
    
    def test_log_access_denied(self):
        """Test log_access_denied convenience method."""
        logger = AuditLogger()
        
        event = logger.log_access_denied(
            actor_did="did:key:mallory",
            resource="belief:secret",
            reason="insufficient_trust",
        )
        
        assert event.event_type == AuditEventType.ACCESS_DENIED
        assert event.success is False
        assert event.metadata["reason"] == "insufficient_trust"
    
    def test_query_and_count(self):
        """Test query and count methods."""
        logger = AuditLogger()
        
        logger.log_share(
            actor_did="did:key:alice",
            target_did="did:key:bob",
            resource="belief:1",
        )
        logger.log_grant_trust(
            actor_did="did:key:alice",
            target_did="did:key:bob",
            trust_level=0.8,
        )
        
        assert logger.count() == 2
        assert logger.count(event_type=AuditEventType.SHARE) == 1
        
        results = logger.query(event_type=AuditEventType.GRANT_TRUST)
        assert len(results) == 1
    
    def test_with_file_backend(self):
        """Test logger with file backend."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            backend = FileAuditBackend(log_path)
            logger = AuditLogger(backend)
            
            logger.log_share(
                actor_did="did:key:alice",
                target_did="did:key:bob",
                resource="belief:1",
            )
            
            # Verify persistence
            logger2 = AuditLogger(FileAuditBackend(log_path))
            assert logger2.count() == 1


class TestModuleSingleton:
    """Tests for module-level singleton functions."""
    
    def test_get_default_logger(self):
        """Test getting default logger."""
        logger = get_audit_logger()
        assert logger is not None
        assert isinstance(logger, AuditLogger)
    
    def test_set_default_logger(self):
        """Test setting custom default logger."""
        custom_backend = InMemoryAuditBackend()
        custom_logger = AuditLogger(custom_backend)
        
        set_audit_logger(custom_logger)
        
        retrieved = get_audit_logger()
        assert retrieved is custom_logger
