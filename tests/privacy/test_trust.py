"""Tests for trust graph storage - TrustEdge and TrustGraphStore."""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

from valence.privacy.trust import (
    TrustEdge,
    TrustGraphStore,
    get_trust_graph_store,
)


class TestTrustEdge:
    """Tests for TrustEdge dataclass."""
    
    def test_create_basic_edge(self):
        """Test creating a basic trust edge."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
        )
        
        assert edge.source_did == "did:key:alice"
        assert edge.target_did == "did:key:bob"
        assert edge.competence == 0.5  # default
        assert edge.integrity == 0.5
        assert edge.confidentiality == 0.5
        assert edge.judgment == 0.5
        assert edge.domain is None
        assert edge.id is None
    
    def test_create_edge_with_scores(self):
        """Test creating edge with custom trust scores."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.9,
            integrity=0.8,
            confidentiality=0.7,
            judgment=0.6,
            domain="medical",
        )
        
        assert edge.competence == 0.9
        assert edge.integrity == 0.8
        assert edge.confidentiality == 0.7
        assert edge.judgment == 0.6
        assert edge.domain == "medical"
    
    def test_overall_trust_calculation(self):
        """Test geometric mean calculation for overall trust."""
        # All same value -> overall should be that value
        edge = TrustEdge(
            source_did="did:key:a",
            target_did="did:key:b",
            competence=0.8,
            integrity=0.8,
            confidentiality=0.8,
            judgment=0.8,
        )
        assert abs(edge.overall_trust - 0.8) < 0.001
        
        # Mixed values
        edge2 = TrustEdge(
            source_did="did:key:a",
            target_did="did:key:b",
            competence=1.0,
            integrity=1.0,
            confidentiality=1.0,
            judgment=0.0,
        )
        # geometric mean of (1, 1, 1, 0) = 0
        assert edge2.overall_trust == 0.0
    
    def test_invalid_score_too_high(self):
        """Test that scores > 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="competence must be between"):
            TrustEdge(
                source_did="did:key:a",
                target_did="did:key:b",
                competence=1.5,
            )
    
    def test_invalid_score_negative(self):
        """Test that scores < 0.0 raise ValueError."""
        with pytest.raises(ValueError, match="integrity must be between"):
            TrustEdge(
                source_did="did:key:a",
                target_did="did:key:b",
                integrity=-0.1,
            )
    
    def test_no_self_trust(self):
        """Test that self-trust edges are rejected."""
        with pytest.raises(ValueError, match="Cannot create trust edge to self"):
            TrustEdge(
                source_did="did:key:alice",
                target_did="did:key:alice",
            )
    
    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiry set."""
        edge = TrustEdge(
            source_did="did:key:a",
            target_did="did:key:b",
        )
        assert not edge.is_expired()
    
    def test_is_expired_future(self):
        """Test is_expired with future expiry."""
        edge = TrustEdge(
            source_did="did:key:a",
            target_did="did:key:b",
            expires_at=datetime.now(timezone.utc) + timedelta(days=1),
        )
        assert not edge.is_expired()
    
    def test_is_expired_past(self):
        """Test is_expired with past expiry."""
        edge = TrustEdge(
            source_did="did:key:a",
            target_did="did:key:b",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert edge.is_expired()
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.9,
            integrity=0.8,
            confidentiality=0.7,
            judgment=0.6,
            domain="research",
        )
        
        data = edge.to_dict()
        assert data["source_did"] == "did:key:alice"
        assert data["target_did"] == "did:key:bob"
        assert data["competence"] == 0.9
        assert data["integrity"] == 0.8
        assert data["confidentiality"] == 0.7
        assert data["judgment"] == 0.6
        assert data["domain"] == "research"
        assert "overall_trust" in data
    
    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "competence": 0.9,
            "integrity": 0.8,
            "domain": "test",
        }
        
        edge = TrustEdge.from_dict(data)
        assert edge.source_did == "did:key:alice"
        assert edge.target_did == "did:key:bob"
        assert edge.competence == 0.9
        assert edge.integrity == 0.8
        assert edge.confidentiality == 0.5  # default
        assert edge.judgment == 0.5  # default
        assert edge.domain == "test"
    
    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.95,
            integrity=0.85,
            confidentiality=0.75,
            judgment=0.65,
            domain="finance",
        )
        
        restored = TrustEdge.from_dict(original.to_dict())
        assert restored.source_did == original.source_did
        assert restored.target_did == original.target_did
        assert restored.competence == original.competence
        assert restored.integrity == original.integrity
        assert restored.confidentiality == original.confidentiality
        assert restored.judgment == original.judgment
        assert restored.domain == original.domain


class TestTrustGraphStore:
    """Tests for TrustGraphStore database operations."""
    
    @pytest.fixture
    def mock_cursor(self):
        """Create a mock cursor for database tests."""
        with patch("valence.privacy.trust.get_cursor") as mock_get_cursor:
            mock_cur = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=mock_cur)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_get_cursor.return_value = mock_ctx
            yield mock_cur
    
    @pytest.fixture
    def store(self):
        """Create a TrustGraphStore instance."""
        return TrustGraphStore()
    
    def test_add_edge_new(self, store, mock_cursor):
        """Test adding a new trust edge."""
        mock_cursor.fetchone.return_value = {
            "id": uuid4(),
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }
        
        edge = TrustEdge(
            source_did="did:key:alice",
            target_did="did:key:bob",
            competence=0.9,
        )
        
        result = store.add_edge(edge)
        
        assert result.id is not None
        assert mock_cursor.execute.called
        # Check that upsert query was used
        call_args = mock_cursor.execute.call_args[0][0]
        assert "INSERT INTO trust_edges" in call_args
        assert "ON CONFLICT" in call_args
    
    def test_get_edge_found(self, store, mock_cursor):
        """Test getting an existing trust edge."""
        mock_cursor.fetchone.return_value = {
            "id": uuid4(),
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "competence": 0.9,
            "integrity": 0.8,
            "confidentiality": 0.7,
            "judgment": 0.6,
            "domain": None,
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "expires_at": None,
        }
        
        edge = store.get_edge("did:key:alice", "did:key:bob")
        
        assert edge is not None
        assert edge.source_did == "did:key:alice"
        assert edge.target_did == "did:key:bob"
        assert edge.competence == 0.9
    
    def test_get_edge_not_found(self, store, mock_cursor):
        """Test getting a non-existent trust edge."""
        mock_cursor.fetchone.return_value = None
        
        edge = store.get_edge("did:key:alice", "did:key:nobody")
        
        assert edge is None
    
    def test_get_edge_with_domain(self, store, mock_cursor):
        """Test getting edge with domain filter."""
        mock_cursor.fetchone.return_value = {
            "id": uuid4(),
            "source_did": "did:key:alice",
            "target_did": "did:key:bob",
            "competence": 0.9,
            "integrity": 0.8,
            "confidentiality": 0.7,
            "judgment": 0.6,
            "domain": "medical",
            "created_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
            "expires_at": None,
        }
        
        edge = store.get_edge("did:key:alice", "did:key:bob", domain="medical")
        
        assert edge is not None
        assert edge.domain == "medical"
        # Verify domain was passed to query
        call_args = mock_cursor.execute.call_args
        assert "medical" in call_args[0][1]
    
    def test_get_edges_from(self, store, mock_cursor):
        """Test getting all edges from a DID."""
        mock_cursor.fetchall.return_value = [
            {
                "id": uuid4(),
                "source_did": "did:key:alice",
                "target_did": "did:key:bob",
                "competence": 0.9,
                "integrity": 0.8,
                "confidentiality": 0.7,
                "judgment": 0.6,
                "domain": None,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "expires_at": None,
            },
            {
                "id": uuid4(),
                "source_did": "did:key:alice",
                "target_did": "did:key:carol",
                "competence": 0.7,
                "integrity": 0.7,
                "confidentiality": 0.7,
                "judgment": 0.7,
                "domain": None,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "expires_at": None,
            },
        ]
        
        edges = store.get_edges_from("did:key:alice")
        
        assert len(edges) == 2
        assert all(e.source_did == "did:key:alice" for e in edges)
    
    def test_get_edges_from_with_domain(self, store, mock_cursor):
        """Test getting edges with domain filter."""
        mock_cursor.fetchall.return_value = []
        
        store.get_edges_from("did:key:alice", domain="medical")
        
        call_args = mock_cursor.execute.call_args[0][0]
        assert "domain = %s" in call_args
    
    def test_get_edges_from_include_expired(self, store, mock_cursor):
        """Test including expired edges."""
        mock_cursor.fetchall.return_value = []
        
        # Without include_expired (default)
        store.get_edges_from("did:key:alice")
        call_args = mock_cursor.execute.call_args[0][0]
        assert "expires_at" in call_args
        
        # With include_expired
        store.get_edges_from("did:key:alice", include_expired=True)
        call_args = mock_cursor.execute.call_args[0][0]
        # The expiry filter should not have an extra clause
    
    def test_get_edges_to(self, store, mock_cursor):
        """Test getting all edges to a DID."""
        mock_cursor.fetchall.return_value = [
            {
                "id": uuid4(),
                "source_did": "did:key:bob",
                "target_did": "did:key:alice",
                "competence": 0.8,
                "integrity": 0.8,
                "confidentiality": 0.8,
                "judgment": 0.8,
                "domain": None,
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "expires_at": None,
            },
        ]
        
        edges = store.get_edges_to("did:key:alice")
        
        assert len(edges) == 1
        assert edges[0].target_did == "did:key:alice"
    
    def test_delete_edge_found(self, store, mock_cursor):
        """Test deleting an existing edge."""
        mock_cursor.fetchone.return_value = {"id": uuid4()}
        
        result = store.delete_edge("did:key:alice", "did:key:bob")
        
        assert result is True
        call_args = mock_cursor.execute.call_args[0][0]
        assert "DELETE FROM trust_edges" in call_args
    
    def test_delete_edge_not_found(self, store, mock_cursor):
        """Test deleting a non-existent edge."""
        mock_cursor.fetchone.return_value = None
        
        result = store.delete_edge("did:key:alice", "did:key:nobody")
        
        assert result is False
    
    def test_delete_edge_with_domain(self, store, mock_cursor):
        """Test deleting edge with domain."""
        mock_cursor.fetchone.return_value = {"id": uuid4()}
        
        store.delete_edge("did:key:alice", "did:key:bob", domain="medical")
        
        call_args = mock_cursor.execute.call_args
        assert "medical" in call_args[0][1]
    
    def test_delete_edges_from(self, store, mock_cursor):
        """Test deleting all edges from a DID."""
        mock_cursor.fetchall.return_value = [{"id": uuid4()}, {"id": uuid4()}]
        
        count = store.delete_edges_from("did:key:alice")
        
        assert count == 2
    
    def test_delete_edges_to(self, store, mock_cursor):
        """Test deleting all edges to a DID."""
        mock_cursor.fetchall.return_value = [{"id": uuid4()}]
        
        count = store.delete_edges_to("did:key:bob")
        
        assert count == 1
    
    def test_cleanup_expired(self, store, mock_cursor):
        """Test cleaning up expired edges."""
        mock_cursor.fetchall.return_value = [{"id": uuid4()}, {"id": uuid4()}, {"id": uuid4()}]
        
        count = store.cleanup_expired()
        
        assert count == 3
        call_args = mock_cursor.execute.call_args[0][0]
        assert "expires_at IS NOT NULL" in call_args
        assert "expires_at < NOW()" in call_args
    
    def test_count_edges_all(self, store, mock_cursor):
        """Test counting all edges."""
        mock_cursor.fetchone.return_value = {"count": 42}
        
        count = store.count_edges()
        
        assert count == 42
    
    def test_count_edges_filtered(self, store, mock_cursor):
        """Test counting edges with filters."""
        mock_cursor.fetchone.return_value = {"count": 5}
        
        count = store.count_edges(source_did="did:key:alice")
        
        assert count == 5
        call_args = mock_cursor.execute.call_args
        assert "did:key:alice" in call_args[0][1]


class TestModuleFunctions:
    """Tests for module-level convenience functions."""
    
    def test_get_trust_graph_store_singleton(self):
        """Test that get_trust_graph_store returns singleton."""
        # Reset module state
        import valence.privacy.trust as trust_module
        trust_module._default_store = None
        
        store1 = get_trust_graph_store()
        store2 = get_trust_graph_store()
        
        assert store1 is store2


# Integration tests (require database)
@pytest.mark.integration
class TestTrustGraphStoreIntegration:
    """Integration tests that require a real database.
    
    Run with: pytest -m integration
    """
    
    @pytest.fixture
    def store(self):
        """Create store and clean up after test."""
        store = TrustGraphStore()
        yield store
        # Cleanup test data
        try:
            store.delete_edges_from("did:test:alice")
            store.delete_edges_from("did:test:bob")
            store.delete_edges_from("did:test:carol")
        except Exception:
            pass
    
    def test_full_crud_cycle(self, store):
        """Test create, read, update, delete cycle."""
        # Create
        edge = TrustEdge(
            source_did="did:test:alice",
            target_did="did:test:bob",
            competence=0.9,
            integrity=0.8,
            confidentiality=0.7,
            judgment=0.6,
        )
        created = store.add_edge(edge)
        assert created.id is not None
        
        # Read
        retrieved = store.get_edge("did:test:alice", "did:test:bob")
        assert retrieved is not None
        assert retrieved.competence == 0.9
        
        # Update
        edge.competence = 0.95
        updated = store.add_edge(edge)
        assert updated.competence == 0.95
        
        # Verify update
        retrieved2 = store.get_edge("did:test:alice", "did:test:bob")
        assert retrieved2.competence == 0.95
        
        # Delete
        deleted = store.delete_edge("did:test:alice", "did:test:bob")
        assert deleted is True
        
        # Verify deletion
        gone = store.get_edge("did:test:alice", "did:test:bob")
        assert gone is None
    
    def test_graph_queries(self, store):
        """Test graph traversal queries."""
        # Create a small graph: alice -> bob, alice -> carol, bob -> carol
        store.add_edge(TrustEdge(
            source_did="did:test:alice",
            target_did="did:test:bob",
            competence=0.9,
        ))
        store.add_edge(TrustEdge(
            source_did="did:test:alice",
            target_did="did:test:carol",
            competence=0.8,
        ))
        store.add_edge(TrustEdge(
            source_did="did:test:bob",
            target_did="did:test:carol",
            competence=0.7,
        ))
        
        # Who does alice trust?
        from_alice = store.get_edges_from("did:test:alice")
        assert len(from_alice) == 2
        
        # Who trusts carol?
        to_carol = store.get_edges_to("did:test:carol")
        assert len(to_carol) == 2
    
    def test_domain_scoped_trust(self, store):
        """Test domain-specific trust edges."""
        # General trust
        store.add_edge(TrustEdge(
            source_did="did:test:alice",
            target_did="did:test:bob",
            competence=0.5,
        ))
        
        # Domain-specific trust (higher)
        store.add_edge(TrustEdge(
            source_did="did:test:alice",
            target_did="did:test:bob",
            competence=0.9,
            domain="medical",
        ))
        
        # Should have both edges
        all_edges = store.get_edges_from("did:test:alice")
        assert len(all_edges) == 2
        
        # Filter by domain
        medical_edges = store.get_edges_from("did:test:alice", domain="medical")
        assert len(medical_edges) == 1
        assert medical_edges[0].competence == 0.9
