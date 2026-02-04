"""Tests for Peer Sync (Week 2 Federation).

Tests cover:
1. Trust Registry - add/remove/list peers
2. Export Package - serialize beliefs for sharing
3. Import - receive beliefs with trust weighting
4. Federated Query - source attribution
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from valence.federation.peer_sync import (
    TrustRegistry,
    TrustedPeer,
    ExportPackage,
    ExportedBelief,
    ImportResult,
    export_beliefs,
    import_beliefs,
    query_federated,
    get_trust_registry,
)


# ============================================================================
# Trust Registry Tests
# ============================================================================

class TestTrustRegistry:
    """Test the trust registry for managing peers."""
    
    def test_add_peer(self, tmp_path):
        """Add a new peer to the registry."""
        registry = TrustRegistry(tmp_path / "trust.json")
        
        peer = registry.add_peer(
            did="did:vkb:web:alice.example.com",
            trust_level=0.8,
            name="Alice",
            notes="My friend",
        )
        
        assert peer.did == "did:vkb:web:alice.example.com"
        assert peer.trust_level == 0.8
        assert peer.name == "Alice"
        assert peer.notes == "My friend"
    
    def test_update_peer_trust(self, tmp_path):
        """Update trust level of existing peer."""
        registry = TrustRegistry(tmp_path / "trust.json")
        
        registry.add_peer("did:vkb:web:alice.example.com", trust_level=0.5)
        registry.add_peer("did:vkb:web:alice.example.com", trust_level=0.9)
        
        peer = registry.get_peer("did:vkb:web:alice.example.com")
        assert peer is not None
        assert peer.trust_level == 0.9
    
    def test_invalid_trust_level(self, tmp_path):
        """Reject invalid trust levels."""
        registry = TrustRegistry(tmp_path / "trust.json")
        
        with pytest.raises(ValueError):
            registry.add_peer("did:vkb:web:test", trust_level=1.5)
        
        with pytest.raises(ValueError):
            registry.add_peer("did:vkb:web:test", trust_level=-0.1)
    
    def test_list_peers_sorted(self, tmp_path):
        """List peers sorted by trust level (highest first)."""
        registry = TrustRegistry(tmp_path / "trust.json")
        
        registry.add_peer("did:vkb:web:low", trust_level=0.3)
        registry.add_peer("did:vkb:web:high", trust_level=0.9)
        registry.add_peer("did:vkb:web:mid", trust_level=0.6)
        
        peers = registry.list_peers()
        
        assert len(peers) == 3
        assert peers[0].did == "did:vkb:web:high"
        assert peers[1].did == "did:vkb:web:mid"
        assert peers[2].did == "did:vkb:web:low"
    
    def test_remove_peer(self, tmp_path):
        """Remove a peer from the registry."""
        registry = TrustRegistry(tmp_path / "trust.json")
        
        registry.add_peer("did:vkb:web:alice", trust_level=0.5)
        assert registry.get_peer("did:vkb:web:alice") is not None
        
        removed = registry.remove_peer("did:vkb:web:alice")
        assert removed is True
        assert registry.get_peer("did:vkb:web:alice") is None
        
        # Removing non-existent returns False
        removed = registry.remove_peer("did:vkb:web:nonexistent")
        assert removed is False
    
    def test_get_trust_level(self, tmp_path):
        """Get trust level for a peer."""
        registry = TrustRegistry(tmp_path / "trust.json")
        
        registry.add_peer("did:vkb:web:alice", trust_level=0.75)
        
        assert registry.get_trust_level("did:vkb:web:alice") == 0.75
        assert registry.get_trust_level("did:vkb:web:unknown") == 0.0
    
    def test_persistence(self, tmp_path):
        """Registry persists to disk and reloads."""
        path = tmp_path / "trust.json"
        
        # Create and populate
        registry1 = TrustRegistry(path)
        registry1.add_peer("did:vkb:web:alice", trust_level=0.8, name="Alice")
        registry1.set_local_did("did:vkb:web:me")
        
        # Reload from disk
        registry2 = TrustRegistry(path)
        
        assert registry2.local_did == "did:vkb:web:me"
        peer = registry2.get_peer("did:vkb:web:alice")
        assert peer is not None
        assert peer.trust_level == 0.8
        assert peer.name == "Alice"
    
    def test_record_sync(self, tmp_path):
        """Record sync interaction updates stats."""
        registry = TrustRegistry(tmp_path / "trust.json")
        registry.add_peer("did:vkb:web:alice", trust_level=0.8)
        
        registry.record_sync("did:vkb:web:alice", beliefs_received=10, beliefs_sent=5)
        
        peer = registry.get_peer("did:vkb:web:alice")
        assert peer.beliefs_received == 10
        assert peer.beliefs_sent == 5
        assert peer.last_sync_at is not None


# ============================================================================
# Export Package Tests
# ============================================================================

class TestExportPackage:
    """Test export package serialization."""
    
    def test_export_package_serialization(self):
        """Export package can be serialized and deserialized."""
        belief = ExportedBelief(
            federation_id=str(uuid4()),
            content="Test belief content",
            confidence={"overall": 0.8},
            domain_path=["tech", "python"],
            origin_did="did:vkb:web:me",
            created_at=datetime.now().isoformat(),
            content_hash="abc123",
        )
        
        package = ExportPackage(
            exporter_did="did:vkb:web:me",
            recipient_did="did:vkb:web:alice",
            created_at=datetime.now().isoformat(),
            beliefs=[belief],
            domain_summary={"tech": 1, "python": 1},
        )
        
        # Serialize
        json_str = package.to_json()
        
        # Deserialize
        loaded = ExportPackage.from_json(json_str)
        
        assert loaded.exporter_did == "did:vkb:web:me"
        assert loaded.recipient_did == "did:vkb:web:alice"
        assert len(loaded.beliefs) == 1
        assert loaded.beliefs[0].content == "Test belief content"
        assert loaded.beliefs[0].domain_path == ["tech", "python"]
    
    def test_export_package_format(self):
        """Export package has correct structure."""
        package = ExportPackage(
            exporter_did="did:vkb:web:me",
            created_at="2025-01-01T00:00:00",
            beliefs=[],
        )
        
        d = package.to_dict()
        
        assert "format_version" in d
        assert d["format_version"] == "1.0"
        assert "exporter_did" in d
        assert "beliefs" in d
        assert isinstance(d["beliefs"], list)


# ============================================================================
# Export Function Tests
# ============================================================================

class TestExportBeliefs:
    """Test belief export functionality."""
    
    @patch('valence.core.db.get_cursor')
    @patch('valence.federation.peer_sync.get_trust_registry')
    def test_export_beliefs_basic(self, mock_registry, mock_cursor):
        """Export returns properly formatted package."""
        # Mock registry
        mock_reg = MagicMock()
        mock_reg.local_did = "did:vkb:web:me"
        mock_registry.return_value = mock_reg
        
        # Mock database results
        mock_cur = MagicMock()
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=False)
        
        belief_id = uuid4()
        mock_cur.fetchall.return_value = [
            {
                "id": belief_id,
                "content": "Test belief",
                "confidence": {"overall": 0.8},
                "domain_path": ["test"],
                "created_at": datetime.now(),
                "valid_from": None,
                "valid_until": None,
                "extraction_method": "observation",
                "content_hash": "abc123",
            }
        ]
        
        package = export_beliefs(recipient_did="did:vkb:web:alice")
        
        assert package.exporter_did == "did:vkb:web:me"
        assert package.recipient_did == "did:vkb:web:alice"
        assert len(package.beliefs) == 1
        assert package.beliefs[0].content == "Test belief"


# ============================================================================
# Import Function Tests
# ============================================================================

class TestImportBeliefs:
    """Test belief import functionality."""
    
    @patch('valence.core.db.get_cursor')
    @patch('valence.federation.peer_sync.get_trust_registry')
    def test_import_applies_trust_weighting(self, mock_registry, mock_cursor):
        """Imported beliefs have confidence weighted by trust."""
        # Mock registry with 50% trust
        mock_reg = MagicMock()
        mock_reg.get_trust_level.return_value = 0.5
        mock_registry.return_value = mock_reg
        
        # Mock database
        mock_cur = MagicMock()
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=False)
        
        # No duplicates
        mock_cur.fetchone.return_value = None
        
        # Create package with 80% confident belief
        belief = ExportedBelief(
            federation_id=str(uuid4()),
            content="Test belief",
            confidence={"overall": 0.8},
            domain_path=["test"],
            origin_did="did:vkb:web:alice",
            created_at=datetime.now().isoformat(),
            content_hash="abc123",
        )
        package = ExportPackage(
            exporter_did="did:vkb:web:alice",
            beliefs=[belief],
        )
        
        result = import_beliefs(package, from_did="did:vkb:web:alice")
        
        # Should have imported 1 belief
        assert result.imported == 1
        assert result.trust_level_applied == 0.5
        
        # Verify INSERT was called with weighted confidence
        calls = mock_cur.execute.call_args_list
        insert_call = [c for c in calls if "INSERT INTO beliefs" in str(c[0][0])][0]
        inserted_confidence = json.loads(insert_call[0][1][1])
        
        # Original 0.8 * trust 0.5 = 0.4
        assert inserted_confidence["overall"] == 0.4
        assert inserted_confidence["_original_overall"] == 0.8
        assert inserted_confidence["_peer_trust"] == 0.5
    
    @patch('valence.core.db.get_cursor')
    @patch('valence.federation.peer_sync.get_trust_registry')
    def test_import_skips_duplicates(self, mock_registry, mock_cursor):
        """Import skips beliefs that already exist."""
        mock_reg = MagicMock()
        mock_reg.get_trust_level.return_value = 0.5
        mock_registry.return_value = mock_reg
        
        mock_cur = MagicMock()
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=False)
        
        # Duplicate found
        mock_cur.fetchone.return_value = {"id": uuid4()}
        
        belief = ExportedBelief(
            federation_id=str(uuid4()),
            content="Existing belief",
            confidence={"overall": 0.8},
            domain_path=[],
            origin_did="did:vkb:web:alice",
            created_at=datetime.now().isoformat(),
            content_hash="abc123",
        )
        package = ExportPackage(beliefs=[belief])
        
        result = import_beliefs(package, from_did="did:vkb:web:alice")
        
        assert result.imported == 0
        assert result.skipped_duplicate == 1
    
    @patch('valence.federation.peer_sync.get_trust_registry')
    def test_import_rejects_untrusted(self, mock_registry):
        """Import rejects beliefs from peers with no trust."""
        mock_reg = MagicMock()
        mock_reg.get_trust_level.return_value = 0.0
        mock_registry.return_value = mock_reg
        
        belief = ExportedBelief(
            federation_id=str(uuid4()),
            content="Untrusted belief",
            confidence={"overall": 0.9},
            domain_path=[],
            origin_did="did:vkb:web:stranger",
            created_at=datetime.now().isoformat(),
            content_hash="abc123",
        )
        package = ExportPackage(beliefs=[belief])
        
        result = import_beliefs(package, from_did="did:vkb:web:stranger")
        
        assert result.imported == 0
        assert result.skipped_low_trust == 1


# ============================================================================
# Federated Query Tests
# ============================================================================

class TestFederatedQuery:
    """Test federated query with source attribution."""
    
    @patch('openai.OpenAI')
    @patch('valence.core.db.get_cursor')
    def test_query_returns_source_attribution(self, mock_cursor, mock_openai):
        """Query results include source attribution."""
        mock_cur = MagicMock()
        mock_cursor.return_value.__enter__ = MagicMock(return_value=mock_cur)
        mock_cursor.return_value.__exit__ = MagicMock(return_value=False)
        
        # Mock OpenAI
        mock_client = MagicMock()
        mock_openai.return_value = mock_client
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_client.embeddings.create.return_value = mock_response
        
        mock_cur.fetchall.return_value = [
            {
                "id": uuid4(),
                "content": "Local belief",
                "confidence": {"overall": 0.8},
                "domain_path": ["test"],
                "created_at": datetime.now(),
                "is_local": True,
                "origin_node_did": None,
                "origin_node_trust": None,
                "similarity": 0.9,
            },
            {
                "id": uuid4(),
                "content": "Peer belief",
                "confidence": {"overall": 0.6, "_original_overall": 0.8, "_peer_trust": 0.75},
                "domain_path": ["test"],
                "created_at": datetime.now(),
                "is_local": False,
                "origin_node_did": "did:vkb:web:alice",
                "origin_node_trust": 0.75,
                "similarity": 0.85,
            },
        ]
        
        results = query_federated("test query", scope="federated")
        
        assert len(results) == 2
        
        # First result should be local
        local = [r for r in results if r.is_local][0]
        assert local.is_local is True
        assert local.origin_did is None
        
        # Second result should be from peer
        peer = [r for r in results if not r.is_local][0]
        assert peer.is_local is False
        assert peer.origin_did == "did:vkb:web:alice"
        assert peer.origin_trust == 0.75


# ============================================================================
# CLI Integration Tests
# ============================================================================

class TestCLICommands:
    """Test CLI command handlers."""
    
    def test_peer_add_command(self, tmp_path):
        """Test peer add CLI command."""
        from valence.cli.main import cmd_peer_add
        
        # Patch registry to use temp path
        with patch('valence.federation.peer_sync.get_trust_registry') as mock_get_registry:
            registry = TrustRegistry(tmp_path / "trust.json")
            mock_get_registry.return_value = registry
            
            args = MagicMock()
            args.did = "did:vkb:web:test"
            args.trust = 0.8
            args.name = "Test"
            args.notes = None
            
            result = cmd_peer_add(args)
            
            assert result == 0
            assert registry.get_peer("did:vkb:web:test") is not None
    
    def test_peer_list_command(self, tmp_path, capsys):
        """Test peer list CLI command."""
        from valence.cli.main import cmd_peer_list
        
        with patch('valence.federation.peer_sync.get_trust_registry') as mock_get_registry:
            registry = TrustRegistry(tmp_path / "trust.json")
            registry.add_peer("did:vkb:web:alice", trust_level=0.9, name="Alice")
            mock_get_registry.return_value = registry
            
            args = MagicMock()
            result = cmd_peer_list(args)
            
            assert result == 0
            captured = capsys.readouterr()
            assert "alice" in captured.out.lower()
            assert "90%" in captured.out


# ============================================================================
# End-to-End Flow Tests
# ============================================================================

class TestEndToEndFlow:
    """Test complete export/import flow."""
    
    def test_export_import_roundtrip(self, tmp_path):
        """Test full export -> import -> query flow."""
        # This is a conceptual test - in reality you'd need a DB
        # Shows the expected flow
        
        # 1. Alice creates an export package
        alice_belief = ExportedBelief(
            federation_id="belief-1",
            content="Python is great for data science",
            confidence={"overall": 0.9},
            domain_path=["tech", "python"],
            origin_did="did:vkb:web:alice",
            created_at=datetime.now().isoformat(),
            content_hash="abc123",
        )
        
        export_package = ExportPackage(
            exporter_did="did:vkb:web:alice",
            recipient_did="did:vkb:web:bob",
            created_at=datetime.now().isoformat(),
            beliefs=[alice_belief],
            domain_summary={"tech": 1, "python": 1},
        )
        
        # 2. Save to file
        export_file = tmp_path / "alice_export.json"
        export_file.write_text(export_package.to_json())
        
        # 3. Bob loads the file
        loaded_package = ExportPackage.from_json(export_file.read_text())
        
        # 4. Verify package integrity
        assert loaded_package.exporter_did == "did:vkb:web:alice"
        assert len(loaded_package.beliefs) == 1
        assert loaded_package.beliefs[0].content == "Python is great for data science"
        assert loaded_package.beliefs[0].confidence["overall"] == 0.9


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_beliefs():
    """Sample beliefs for testing."""
    return [
        ExportedBelief(
            federation_id=str(uuid4()),
            content="The sky is blue",
            confidence={"overall": 0.9},
            domain_path=["science", "astronomy"],
            origin_did="did:vkb:web:alice",
            created_at=datetime.now().isoformat(),
            content_hash="sky1",
        ),
        ExportedBelief(
            federation_id=str(uuid4()),
            content="Water boils at 100°C",
            confidence={"overall": 0.95},
            domain_path=["science", "physics"],
            origin_did="did:vkb:web:alice",
            created_at=datetime.now().isoformat(),
            content_hash="water1",
        ),
    ]
