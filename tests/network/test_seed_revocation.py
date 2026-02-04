"""
Tests for Seed Revocation (Issue #121).

Tests cover:
- SeedRevocation message type creation and serialization
- SeedRevocationList for out-of-band distribution
- SeedRevocationManager for seed nodes
- Gossip propagation of revocations between seeds
- DiscoveryClient honoring revocations
- Ed25519 signature verification for revocations
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from valence.network.messages import (
    SeedRevocation,
    SeedRevocationList,
    RevocationReason,
)
from valence.network.seed import (
    SeedConfig,
    SeedNode,
    SeedRevocationManager,
    SeedRevocationRecord,
)
from valence.network.discovery import DiscoveryClient


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def test_config():
    """Create a test seed config with revocation enabled."""
    return SeedConfig(
        host="127.0.0.1",
        port=28470,
        seed_id="test-seed-revocation",
        verify_signatures=False,  # Disable for unit tests
        verify_pow=False,
        probe_endpoints=False,
        seed_revocation_enabled=True,
        seed_revocation_verify_signatures=False,  # Disable for unit tests
        seed_revocation_gossip_enabled=True,
        seed_revocation_max_age_seconds=86400.0 * 30,
        gossip_enabled=False,  # Disable gossip for unit tests
    )


@pytest.fixture
def revocation_manager(test_config):
    """Create a revocation manager instance."""
    return SeedRevocationManager(test_config)


@pytest.fixture
def seed_node(test_config):
    """Create a test seed node with revocation enabled."""
    return SeedNode(config=test_config)


@pytest.fixture
def discovery_client():
    """Create a discovery client for testing."""
    return DiscoveryClient(
        verify_signatures=False,
        default_seeds=["https://seed1.test:8470", "https://seed2.test:8470"],
    )


@pytest.fixture
def sample_revocation():
    """Create a sample revocation for testing."""
    return SeedRevocation(
        seed_id="abcd1234" * 8,  # 64 hex chars (32 bytes)
        reason=RevocationReason.KEY_COMPROMISE,
        reason_detail="Private key exposed in public repository",
        timestamp=time.time(),
        effective_at=time.time(),
        issuer_id="abcd1234" * 8,  # Self-signed
    )


# =============================================================================
# SEED REVOCATION MESSAGE TESTS
# =============================================================================


class TestSeedRevocationMessage:
    """Tests for SeedRevocation message type."""
    
    def test_create_revocation(self):
        """Test creating a revocation message."""
        now = time.time()
        rev = SeedRevocation(
            seed_id="test-seed-id",
            reason=RevocationReason.KEY_COMPROMISE,
            reason_detail="Test reason",
            timestamp=now,
            issuer_id="test-seed-id",
        )
        
        assert rev.type == "seed_revocation"
        assert rev.seed_id == "test-seed-id"
        assert rev.reason == RevocationReason.KEY_COMPROMISE
        assert rev.timestamp == now
        assert rev.effective_at == now  # Defaults to timestamp
        assert rev.issuer_id == "test-seed-id"
    
    def test_revocation_with_delayed_effective(self):
        """Test creating a revocation with delayed effectiveness."""
        now = time.time()
        future = now + 3600  # 1 hour from now
        
        rev = SeedRevocation(
            seed_id="test-seed-id",
            reason=RevocationReason.RETIRED,
            timestamp=now,
            effective_at=future,
            issuer_id="test-seed-id",
        )
        
        assert rev.timestamp == now
        assert rev.effective_at == future
        assert not rev.is_effective  # Not yet effective
    
    def test_revocation_is_effective(self):
        """Test is_effective property."""
        past = time.time() - 3600
        
        rev = SeedRevocation(
            seed_id="test-seed-id",
            reason=RevocationReason.RETIRED,
            timestamp=past,
            effective_at=past,
            issuer_id="test-seed-id",
        )
        
        assert rev.is_effective
    
    def test_revocation_serialization(self, sample_revocation):
        """Test serializing revocation to dict and JSON."""
        data = sample_revocation.to_dict()
        
        assert data["type"] == "seed_revocation"
        assert data["seed_id"] == sample_revocation.seed_id
        assert data["reason"] == RevocationReason.KEY_COMPROMISE
        assert "revocation_id" in data
        
        # Test JSON serialization
        json_str = sample_revocation.to_json()
        assert isinstance(json_str, str)
        
        # Parse it back
        parsed = json.loads(json_str)
        assert parsed["seed_id"] == sample_revocation.seed_id
    
    def test_revocation_deserialization(self, sample_revocation):
        """Test deserializing revocation from dict."""
        data = sample_revocation.to_dict()
        
        restored = SeedRevocation.from_dict(data)
        
        assert restored.seed_id == sample_revocation.seed_id
        assert restored.reason == sample_revocation.reason
        assert restored.revocation_id == sample_revocation.revocation_id
        assert restored.timestamp == sample_revocation.timestamp
    
    def test_revocation_signable_data(self, sample_revocation):
        """Test getting signable data for signatures."""
        signable = sample_revocation.get_signable_data()
        
        # Should not include signature
        assert "signature" not in signable
        
        # Should include all security-relevant fields
        assert signable["seed_id"] == sample_revocation.seed_id
        assert signable["timestamp"] == sample_revocation.timestamp
        assert signable["issuer_id"] == sample_revocation.issuer_id
    
    def test_revocation_signable_bytes_deterministic(self, sample_revocation):
        """Test that signable bytes are deterministic."""
        bytes1 = sample_revocation.get_signable_bytes()
        bytes2 = sample_revocation.get_signable_bytes()
        
        assert bytes1 == bytes2
        
        # Should be valid JSON
        parsed = json.loads(bytes1.decode())
        assert parsed["seed_id"] == sample_revocation.seed_id


class TestSeedRevocationList:
    """Tests for SeedRevocationList (out-of-band distribution)."""
    
    def test_create_revocation_list(self, sample_revocation):
        """Test creating a revocation list."""
        rev_list = SeedRevocationList(
            version=1,
            revocations=[sample_revocation],
            authority_id="authority-key-hex",
        )
        
        assert rev_list.version == 1
        assert len(rev_list.revocations) == 1
        assert rev_list.authority_id == "authority-key-hex"
    
    def test_revocation_list_serialization(self, sample_revocation):
        """Test serializing revocation list."""
        rev_list = SeedRevocationList(
            version=2,
            revocations=[sample_revocation],
            authority_id="authority-key-hex",
            signature="signature-hex",
        )
        
        data = rev_list.to_dict()
        
        assert data["version"] == 2
        assert len(data["revocations"]) == 1
        assert data["authority_id"] == "authority-key-hex"
        assert data["signature"] == "signature-hex"
        
        # Test JSON
        json_str = rev_list.to_json()
        parsed = json.loads(json_str)
        assert parsed["version"] == 2
    
    def test_revocation_list_deserialization(self, sample_revocation):
        """Test deserializing revocation list."""
        rev_list = SeedRevocationList(
            version=3,
            revocations=[sample_revocation],
            authority_id="authority-key-hex",
        )
        
        json_str = rev_list.to_json()
        restored = SeedRevocationList.from_json(json_str)
        
        assert restored.version == 3
        assert len(restored.revocations) == 1
        assert restored.revocations[0].seed_id == sample_revocation.seed_id
    
    def test_get_revoked_seed_ids(self, sample_revocation):
        """Test getting set of revoked seed IDs."""
        # Create revocations with different effectiveness
        now = time.time()
        
        effective_rev = SeedRevocation(
            seed_id="effective-seed",
            reason=RevocationReason.KEY_COMPROMISE,
            timestamp=now - 3600,
            effective_at=now - 3600,
            issuer_id="effective-seed",
        )
        
        future_rev = SeedRevocation(
            seed_id="future-seed",
            reason=RevocationReason.RETIRED,
            timestamp=now,
            effective_at=now + 3600,  # Future
            issuer_id="future-seed",
        )
        
        rev_list = SeedRevocationList(
            version=1,
            revocations=[effective_rev, future_rev],
        )
        
        revoked = rev_list.get_revoked_seed_ids()
        
        assert "effective-seed" in revoked
        assert "future-seed" not in revoked  # Not yet effective
    
    def test_is_seed_revoked(self, sample_revocation):
        """Test checking if specific seed is revoked."""
        rev_list = SeedRevocationList(
            version=1,
            revocations=[sample_revocation],
        )
        
        assert rev_list.is_seed_revoked(sample_revocation.seed_id)
        assert not rev_list.is_seed_revoked("unknown-seed")


# =============================================================================
# SEED REVOCATION MANAGER TESTS
# =============================================================================


class TestSeedRevocationManager:
    """Tests for SeedRevocationManager."""
    
    def test_add_revocation(self, revocation_manager, sample_revocation):
        """Test adding a revocation."""
        data = sample_revocation.to_dict()
        
        success, error = revocation_manager.add_revocation(data)
        
        assert success is True
        assert error is None
        assert revocation_manager.is_seed_revoked(sample_revocation.seed_id)
    
    def test_revocation_storage(self, revocation_manager, sample_revocation):
        """Test that revocations are stored correctly."""
        data = sample_revocation.to_dict()
        revocation_manager.add_revocation(data)
        
        record = revocation_manager.get_revocation(sample_revocation.seed_id)
        
        assert record is not None
        assert record.seed_id == sample_revocation.seed_id
        assert record.reason == sample_revocation.reason
    
    def test_duplicate_revocation(self, revocation_manager, sample_revocation):
        """Test adding duplicate revocation."""
        data = sample_revocation.to_dict()
        
        success1, _ = revocation_manager.add_revocation(data)
        success2, _ = revocation_manager.add_revocation(data)  # Duplicate
        
        assert success1 is True
        assert success2 is True  # Duplicates are accepted but not re-added
        assert len(revocation_manager.get_all_revocations()) == 1
    
    def test_newer_revocation_replaces_older(self, revocation_manager):
        """Test that newer revocation replaces older one for same seed."""
        seed_id = "test-seed-123"
        now = time.time()
        
        old_rev = {
            "revocation_id": "old-rev",
            "seed_id": seed_id,
            "reason": RevocationReason.RETIRED,
            "timestamp": now - 3600,
            "effective_at": now - 3600,
            "issuer_id": seed_id,
        }
        
        new_rev = {
            "revocation_id": "new-rev",
            "seed_id": seed_id,
            "reason": RevocationReason.KEY_COMPROMISE,
            "timestamp": now,
            "effective_at": now,
            "issuer_id": seed_id,
        }
        
        revocation_manager.add_revocation(old_rev)
        revocation_manager.add_revocation(new_rev)
        
        record = revocation_manager.get_revocation(seed_id)
        assert record.revocation_id == "new-rev"
        assert record.reason == RevocationReason.KEY_COMPROMISE
    
    def test_old_revocation_rejected_if_newer_exists(self, revocation_manager):
        """Test that old revocation doesn't replace newer one."""
        seed_id = "test-seed-456"
        now = time.time()
        
        new_rev = {
            "revocation_id": "new-rev",
            "seed_id": seed_id,
            "reason": RevocationReason.KEY_COMPROMISE,
            "timestamp": now,
            "effective_at": now,
            "issuer_id": seed_id,
        }
        
        old_rev = {
            "revocation_id": "old-rev",
            "seed_id": seed_id,
            "reason": RevocationReason.RETIRED,
            "timestamp": now - 3600,
            "effective_at": now - 3600,
            "issuer_id": seed_id,
        }
        
        revocation_manager.add_revocation(new_rev)
        revocation_manager.add_revocation(old_rev)  # Older, should be ignored
        
        record = revocation_manager.get_revocation(seed_id)
        assert record.revocation_id == "new-rev"
    
    def test_revocation_too_old_rejected(self, revocation_manager):
        """Test that very old revocations are rejected."""
        old_timestamp = time.time() - (86400 * 60)  # 60 days ago (max is 30)
        
        old_rev = {
            "revocation_id": "ancient-rev",
            "seed_id": "ancient-seed",
            "reason": RevocationReason.RETIRED,
            "timestamp": old_timestamp,
            "effective_at": old_timestamp,
            "issuer_id": "ancient-seed",
        }
        
        success, error = revocation_manager.add_revocation(old_rev)
        
        assert success is False
        assert "revocation_too_old" in error
    
    def test_missing_seed_id_rejected(self, revocation_manager):
        """Test that revocation without seed_id is rejected."""
        invalid_rev = {
            "revocation_id": "rev-123",
            "reason": RevocationReason.RETIRED,
            "timestamp": time.time(),
        }
        
        success, error = revocation_manager.add_revocation(invalid_rev)
        
        assert success is False
        assert error == "missing_seed_id"
    
    def test_get_all_revocations(self, revocation_manager):
        """Test getting all revocations."""
        now = time.time()
        
        for i in range(3):
            rev = {
                "revocation_id": f"rev-{i}",
                "seed_id": f"seed-{i}",
                "reason": RevocationReason.RETIRED,
                "timestamp": now,
                "effective_at": now,
                "issuer_id": f"seed-{i}",
            }
            revocation_manager.add_revocation(rev)
        
        all_revs = revocation_manager.get_all_revocations()
        assert len(all_revs) == 3
    
    def test_get_revoked_seed_ids(self, revocation_manager):
        """Test getting set of revoked seed IDs."""
        now = time.time()
        
        # Add effective revocation
        revocation_manager.add_revocation({
            "revocation_id": "rev-1",
            "seed_id": "effective-seed",
            "reason": RevocationReason.RETIRED,
            "timestamp": now - 100,
            "effective_at": now - 100,
            "issuer_id": "effective-seed",
        })
        
        # Add future revocation
        revocation_manager.add_revocation({
            "revocation_id": "rev-2",
            "seed_id": "future-seed",
            "reason": RevocationReason.RETIRED,
            "timestamp": now,
            "effective_at": now + 3600,
            "issuer_id": "future-seed",
        })
        
        revoked = revocation_manager.get_revoked_seed_ids()
        
        assert "effective-seed" in revoked
        assert "future-seed" not in revoked
    
    def test_load_revocation_list_from_file(self, revocation_manager, sample_revocation):
        """Test loading revocations from file."""
        rev_list = SeedRevocationList(
            version=1,
            revocations=[sample_revocation],
        )
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(rev_list.to_json())
            temp_path = f.name
        
        try:
            loaded, errors = revocation_manager.load_revocation_list_from_file(temp_path)
            
            assert loaded == 1
            assert len(errors) == 0
            assert revocation_manager.is_seed_revoked(sample_revocation.seed_id)
        finally:
            os.unlink(temp_path)
    
    def test_load_missing_file(self, revocation_manager):
        """Test loading from non-existent file."""
        loaded, errors = revocation_manager.load_revocation_list_from_file("/nonexistent/file.json")
        
        assert loaded == 0
        assert any("file_not_found" in e for e in errors)
    
    def test_load_invalid_json(self, revocation_manager):
        """Test loading from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("not valid json {{{")
            temp_path = f.name
        
        try:
            loaded, errors = revocation_manager.load_revocation_list_from_file(temp_path)
            
            assert loaded == 0
            assert any("invalid_json" in e for e in errors)
        finally:
            os.unlink(temp_path)
    
    def test_get_revocations_for_gossip(self, revocation_manager, sample_revocation):
        """Test getting revocations for gossip propagation."""
        revocation_manager.add_revocation(sample_revocation.to_dict())
        
        gossip_revs = revocation_manager.get_revocations_for_gossip()
        
        assert len(gossip_revs) == 1
        assert gossip_revs[0]["seed_id"] == sample_revocation.seed_id
    
    def test_process_gossip_revocations(self, revocation_manager):
        """Test processing revocations from gossip."""
        now = time.time()
        
        gossip_revs = [
            {
                "revocation_id": "gossip-rev-1",
                "seed_id": "gossip-seed-1",
                "reason": RevocationReason.KEY_COMPROMISE,
                "timestamp": now,
                "effective_at": now,
                "issuer_id": "gossip-seed-1",
            },
            {
                "revocation_id": "gossip-rev-2",
                "seed_id": "gossip-seed-2",
                "reason": RevocationReason.MALICIOUS_BEHAVIOR,
                "timestamp": now,
                "effective_at": now,
                "issuer_id": "gossip-seed-2",
            },
        ]
        
        added = revocation_manager.process_gossip_revocations(gossip_revs)
        
        assert added == 2
        assert revocation_manager.is_seed_revoked("gossip-seed-1")
        assert revocation_manager.is_seed_revoked("gossip-seed-2")
    
    def test_get_stats(self, revocation_manager, sample_revocation):
        """Test getting revocation statistics."""
        revocation_manager.add_revocation(sample_revocation.to_dict())
        
        stats = revocation_manager.get_stats()
        
        assert stats["enabled"] is True
        assert stats["total_revocations"] == 1
        assert stats["effective_revocations"] == 1
        assert sample_revocation.seed_id in stats["revoked_seeds"]


# =============================================================================
# DISCOVERY CLIENT REVOCATION TESTS
# =============================================================================


class TestDiscoveryClientRevocation:
    """Tests for DiscoveryClient honoring revocations."""
    
    def test_add_revoked_seed(self, discovery_client):
        """Test adding a seed to revocation set."""
        discovery_client.add_revoked_seed("revoked-seed-id")
        
        assert discovery_client.is_seed_revoked("revoked-seed-id")
    
    def test_remove_revoked_seed(self, discovery_client):
        """Test removing a seed from revocation set."""
        discovery_client.add_revoked_seed("revoked-seed-id")
        discovery_client.remove_revoked_seed("revoked-seed-id")
        
        assert not discovery_client.is_seed_revoked("revoked-seed-id")
    
    def test_revoked_seed_filtered_from_list(self, discovery_client):
        """Test that revoked seeds are filtered from seed list."""
        # Add some custom seeds
        discovery_client.add_seed("https://good-seed.test:8470")
        discovery_client.add_seed("https://bad-seed.test:8470")
        
        # Revoke one of them
        discovery_client.add_revoked_seed("https://bad-seed.test:8470")
        
        # Get seed list
        seeds = discovery_client._get_seed_list()
        
        assert "https://good-seed.test:8470" in seeds
        assert "https://bad-seed.test:8470" not in seeds
    
    def test_revoked_last_successful_seed_cleared(self, discovery_client):
        """Test that revoked last_successful_seed is cleared."""
        discovery_client.last_successful_seed = "https://revoked.test:8470"
        discovery_client.add_revoked_seed("https://revoked.test:8470")
        
        seeds = discovery_client._get_seed_list()
        
        assert discovery_client.last_successful_seed is None
        assert "https://revoked.test:8470" not in seeds
    
    def test_get_revoked_seeds(self, discovery_client):
        """Test getting set of revoked seeds."""
        discovery_client.add_revoked_seed("seed-1")
        discovery_client.add_revoked_seed("seed-2")
        
        revoked = discovery_client.get_revoked_seeds()
        
        assert "seed-1" in revoked
        assert "seed-2" in revoked
    
    def test_clear_revocations(self, discovery_client):
        """Test clearing all revocations."""
        discovery_client.add_revoked_seed("seed-1")
        discovery_client.add_revoked_seed("seed-2")
        discovery_client.clear_revocations()
        
        assert len(discovery_client.get_revoked_seeds()) == 0
    
    def test_load_revocation_list_from_file(self, discovery_client, sample_revocation):
        """Test loading revocations from file."""
        rev_list = SeedRevocationList(
            version=1,
            revocations=[sample_revocation],
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(rev_list.to_json())
            temp_path = f.name
        
        try:
            loaded, errors = discovery_client.load_revocation_list(temp_path)
            
            assert loaded == 1
            assert len(errors) == 0
            assert discovery_client.is_seed_revoked(sample_revocation.seed_id)
        finally:
            os.unlink(temp_path)
    
    def test_revocation_stats(self, discovery_client):
        """Test getting revocation statistics."""
        discovery_client.add_revoked_seed("seed-1")
        
        # Trigger a stat increment by getting seed list with revoked seed
        discovery_client.add_seed("seed-1")  # Add as custom seed
        discovery_client._get_seed_list()  # Should skip and increment stat
        
        stats = discovery_client.get_revocation_stats()
        
        assert stats["revoked_count"] >= 1
    
    def test_stats_include_revoked_seeds_skipped(self, discovery_client):
        """Test that stats track revoked seeds skipped."""
        discovery_client.add_seed("https://revoked.test:8470")
        discovery_client.add_revoked_seed("https://revoked.test:8470")
        
        # Get seed list multiple times
        discovery_client._get_seed_list()
        discovery_client._get_seed_list()
        
        assert discovery_client._stats["revoked_seeds_skipped"] >= 2


# =============================================================================
# SEED NODE INTEGRATION TESTS
# =============================================================================


class TestSeedNodeRevocation:
    """Integration tests for seed node revocation handling."""
    
    def test_seed_node_has_revocation_manager(self, seed_node):
        """Test that seed node has revocation manager."""
        assert seed_node.revocation_manager is not None
        assert isinstance(seed_node.revocation_manager, SeedRevocationManager)
    
    def test_is_seed_revoked(self, seed_node, sample_revocation):
        """Test checking if seed is revoked via seed node."""
        seed_node.revocation_manager.add_revocation(sample_revocation.to_dict())
        
        assert seed_node.is_seed_revoked(sample_revocation.seed_id)
        assert not seed_node.is_seed_revoked("unknown-seed")
    
    @pytest.mark.asyncio
    async def test_handle_revoke_seed(self, seed_node, sample_revocation):
        """Test the revoke_seed HTTP endpoint."""
        from aiohttp.test_utils import make_mocked_request
        from aiohttp import web
        
        # Create a mock request
        request = MagicMock()
        request.json = AsyncMock(return_value=sample_revocation.to_dict())
        
        response = await seed_node.handle_revoke_seed(request)
        
        assert isinstance(response, web.Response)
        data = json.loads(response.body)
        assert data["status"] == "accepted"
        assert data["seed_id"] == sample_revocation.seed_id
    
    @pytest.mark.asyncio
    async def test_handle_get_revocations(self, seed_node, sample_revocation):
        """Test the get_revocations HTTP endpoint."""
        from aiohttp import web
        
        # Add a revocation
        seed_node.revocation_manager.add_revocation(sample_revocation.to_dict())
        
        # Create a mock request without query params
        request = MagicMock()
        request.query = {}
        
        response = await seed_node.handle_get_revocations(request)
        
        assert isinstance(response, web.Response)
        data = json.loads(response.body)
        assert data["total"] == 1
        assert sample_revocation.seed_id in data["revoked_seed_ids"]
    
    @pytest.mark.asyncio
    async def test_handle_get_revocations_by_seed_id(self, seed_node, sample_revocation):
        """Test the get_revocations endpoint with seed_id query param."""
        from aiohttp import web
        
        # Add a revocation
        seed_node.revocation_manager.add_revocation(sample_revocation.to_dict())
        
        # Query for specific seed
        request = MagicMock()
        request.query = {"seed_id": sample_revocation.seed_id}
        
        response = await seed_node.handle_get_revocations(request)
        
        data = json.loads(response.body)
        assert data["revoked"] is True
        assert data["seed_id"] == sample_revocation.seed_id
    
    @pytest.mark.asyncio
    async def test_status_includes_revocation_stats(self, seed_node, sample_revocation):
        """Test that /status endpoint includes revocation stats."""
        from aiohttp import web
        
        # Add a revocation
        seed_node.revocation_manager.add_revocation(sample_revocation.to_dict())
        
        # Create mock request
        request = MagicMock()
        
        response = await seed_node.handle_status(request)
        
        data = json.loads(response.body)
        assert "revocations" in data
        assert data["revocations"]["total_revocations"] == 1


# =============================================================================
# SIGNATURE VERIFICATION TESTS
# =============================================================================


class TestRevocationSignatureVerification:
    """Tests for Ed25519 signature verification on revocations."""
    
    def test_self_signed_revocation(self):
        """Test verification of self-signed revocation."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        
        # Generate a key pair
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        public_key_hex = public_key.public_bytes_raw().hex()
        
        # Create a revocation
        rev = SeedRevocation(
            seed_id=public_key_hex,
            reason=RevocationReason.KEY_COMPROMISE,
            timestamp=time.time(),
            issuer_id=public_key_hex,  # Self-signed
        )
        
        # Sign it
        message = rev.get_signable_bytes()
        signature = private_key.sign(message)
        rev.signature = signature.hex()
        
        # Create manager with signature verification enabled
        config = SeedConfig(
            seed_revocation_enabled=True,
            seed_revocation_verify_signatures=True,
        )
        manager = SeedRevocationManager(config)
        
        # Should pass verification
        success, error = manager.add_revocation(rev.to_dict())
        
        assert success is True, f"Verification failed: {error}"
    
    def test_authority_signed_revocation(self):
        """Test verification of authority-signed revocation."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        
        # Generate authority key pair
        authority_key = Ed25519PrivateKey.generate()
        authority_public = authority_key.public_key()
        authority_id = authority_public.public_bytes_raw().hex()
        
        # Generate seed key (different from authority)
        seed_id = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
        
        # Create a revocation issued by authority
        rev = SeedRevocation(
            seed_id=seed_id,
            reason=RevocationReason.MALICIOUS_BEHAVIOR,
            timestamp=time.time(),
            issuer_id=authority_id,  # Authority signed
        )
        
        # Sign with authority key
        message = rev.get_signable_bytes()
        signature = authority_key.sign(message)
        rev.signature = signature.hex()
        
        # Create manager with signature verification and trusted authority
        config = SeedConfig(
            seed_revocation_enabled=True,
            seed_revocation_verify_signatures=True,
            seed_revocation_trusted_authorities=[authority_id],
        )
        manager = SeedRevocationManager(config)
        
        # Should pass verification
        success, error = manager.add_revocation(rev.to_dict())
        
        assert success is True, f"Verification failed: {error}"
    
    def test_untrusted_issuer_rejected(self):
        """Test that revocation from untrusted issuer is rejected."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        
        # Generate an untrusted key
        untrusted_key = Ed25519PrivateKey.generate()
        untrusted_id = untrusted_key.public_key().public_bytes_raw().hex()
        
        # Create revocation with untrusted issuer
        seed_id = "some-other-seed-id-hex" * 2  # Different from issuer
        
        rev = SeedRevocation(
            seed_id=seed_id,
            reason=RevocationReason.ADMIN_ACTION,
            timestamp=time.time(),
            issuer_id=untrusted_id,
        )
        
        # Sign it
        message = rev.get_signable_bytes()
        signature = untrusted_key.sign(message)
        rev.signature = signature.hex()
        
        # Create manager with NO trusted authorities
        config = SeedConfig(
            seed_revocation_enabled=True,
            seed_revocation_verify_signatures=True,
            seed_revocation_trusted_authorities=[],  # No trusted authorities
        )
        manager = SeedRevocationManager(config)
        
        # Should be rejected (issuer != seed_id and not trusted authority)
        success, error = manager.add_revocation(rev.to_dict())
        
        assert success is False
        assert "untrusted_issuer" in error
    
    def test_invalid_signature_rejected(self):
        """Test that invalid signature is rejected."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        
        # Generate a key pair
        private_key = Ed25519PrivateKey.generate()
        public_key_hex = private_key.public_key().public_bytes_raw().hex()
        
        # Create a revocation
        rev = SeedRevocation(
            seed_id=public_key_hex,
            reason=RevocationReason.KEY_COMPROMISE,
            timestamp=time.time(),
            issuer_id=public_key_hex,
        )
        
        # Use a WRONG signature (sign different data)
        wrong_message = b"wrong data"
        signature = private_key.sign(wrong_message)
        rev.signature = signature.hex()
        
        # Create manager with signature verification
        config = SeedConfig(
            seed_revocation_enabled=True,
            seed_revocation_verify_signatures=True,
        )
        manager = SeedRevocationManager(config)
        
        # Should fail verification
        success, error = manager.add_revocation(rev.to_dict())
        
        assert success is False
        assert "signature_verification_failed" in error
