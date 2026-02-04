"""Tests for privacy types - SharePolicy serialization and behavior."""

import pytest
from datetime import datetime, timedelta, timezone

from valence.privacy.types import (
    ShareLevel,
    EnforcementType,
    PropagationRules,
    SharePolicy,
)


class TestShareLevel:
    """Tests for ShareLevel enum."""
    
    def test_all_levels_exist(self):
        """Verify all share levels are defined."""
        assert ShareLevel.PRIVATE.value == "private"
        assert ShareLevel.DIRECT.value == "direct"
        assert ShareLevel.BOUNDED.value == "bounded"
        assert ShareLevel.CASCADING.value == "cascading"
        assert ShareLevel.PUBLIC.value == "public"
    
    def test_level_from_string(self):
        """Test creating level from string value."""
        assert ShareLevel("private") == ShareLevel.PRIVATE
        assert ShareLevel("public") == ShareLevel.PUBLIC


class TestEnforcementType:
    """Tests for EnforcementType enum."""
    
    def test_all_types_exist(self):
        """Verify all enforcement types are defined."""
        assert EnforcementType.CRYPTOGRAPHIC.value == "cryptographic"
        assert EnforcementType.POLICY.value == "policy"
        assert EnforcementType.HONOR.value == "honor"


class TestPropagationRules:
    """Tests for PropagationRules."""
    
    def test_defaults(self):
        """Test default values."""
        rules = PropagationRules()
        assert rules.max_hops is None
        assert rules.allowed_domains is None
        assert rules.min_trust_to_receive is None
        assert rules.strip_on_forward is None
        assert rules.expires_at is None
    
    def test_to_dict(self):
        """Test serialization to dict."""
        expires = datetime(2025, 12, 31, 23, 59, 59)
        rules = PropagationRules(
            max_hops=3,
            allowed_domains=["medical", "research"],
            min_trust_to_receive=0.7,
            strip_on_forward=["raw_content"],
            expires_at=expires,
        )
        
        data = rules.to_dict()
        assert data["max_hops"] == 3
        assert data["allowed_domains"] == ["medical", "research"]
        assert data["min_trust_to_receive"] == 0.7
        assert data["strip_on_forward"] == ["raw_content"]
        assert data["expires_at"] == "2025-12-31T23:59:59"
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "max_hops": 2,
            "allowed_domains": ["finance"],
            "min_trust_to_receive": 0.5,
            "strip_on_forward": None,
            "expires_at": "2025-06-15T12:00:00",
        }
        
        rules = PropagationRules.from_dict(data)
        assert rules.max_hops == 2
        assert rules.allowed_domains == ["finance"]
        assert rules.min_trust_to_receive == 0.5
        assert rules.expires_at == datetime(2025, 6, 15, 12, 0, 0)
    
    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = PropagationRules(
            max_hops=5,
            allowed_domains=["test"],
            expires_at=datetime(2026, 1, 1),
        )
        
        restored = PropagationRules.from_dict(original.to_dict())
        assert restored.max_hops == original.max_hops
        assert restored.allowed_domains == original.allowed_domains
        assert restored.expires_at == original.expires_at


class TestSharePolicy:
    """Tests for SharePolicy."""
    
    def test_default_enforcement(self):
        """Test that default enforcement is POLICY."""
        policy = SharePolicy(level=ShareLevel.BOUNDED)
        assert policy.enforcement == EnforcementType.POLICY
    
    def test_private_factory(self):
        """Test private() factory method."""
        policy = SharePolicy.private()
        assert policy.level == ShareLevel.PRIVATE
        assert policy.enforcement == EnforcementType.CRYPTOGRAPHIC
    
    def test_public_factory(self):
        """Test public() factory method."""
        policy = SharePolicy.public()
        assert policy.level == ShareLevel.PUBLIC
        assert policy.enforcement == EnforcementType.HONOR
    
    def test_direct_factory(self):
        """Test direct() factory method."""
        recipients = ["did:key:alice", "did:key:bob"]
        policy = SharePolicy.direct(recipients)
        assert policy.level == ShareLevel.DIRECT
        assert policy.enforcement == EnforcementType.CRYPTOGRAPHIC
        assert policy.recipients == recipients
    
    def test_bounded_factory(self):
        """Test bounded() factory method."""
        policy = SharePolicy.bounded(max_hops=3, allowed_domains=["medical"])
        assert policy.level == ShareLevel.BOUNDED
        assert policy.propagation.max_hops == 3
        assert policy.propagation.allowed_domains == ["medical"]
    
    def test_to_dict(self):
        """Test serialization to dict."""
        policy = SharePolicy(
            level=ShareLevel.DIRECT,
            enforcement=EnforcementType.CRYPTOGRAPHIC,
            recipients=["did:key:alice"],
        )
        
        data = policy.to_dict()
        assert data["level"] == "direct"
        assert data["enforcement"] == "cryptographic"
        assert data["recipients"] == ["did:key:alice"]
        assert data["propagation"] is None
    
    def test_to_dict_with_propagation(self):
        """Test serialization with propagation rules."""
        policy = SharePolicy.bounded(max_hops=2)
        data = policy.to_dict()
        
        assert data["propagation"] is not None
        assert data["propagation"]["max_hops"] == 2
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "level": "cascading",
            "enforcement": "policy",
            "recipients": None,
            "propagation": {
                "max_hops": 10,
                "allowed_domains": None,
                "min_trust_to_receive": 0.3,
                "strip_on_forward": None,
                "expires_at": None,
            },
        }
        
        policy = SharePolicy.from_dict(data)
        assert policy.level == ShareLevel.CASCADING
        assert policy.enforcement == EnforcementType.POLICY
        assert policy.propagation.max_hops == 10
        assert policy.propagation.min_trust_to_receive == 0.3
    
    def test_from_dict_defaults(self):
        """Test deserialization with minimal data."""
        data = {"level": "private"}
        policy = SharePolicy.from_dict(data)
        
        assert policy.level == ShareLevel.PRIVATE
        assert policy.enforcement == EnforcementType.POLICY  # default
        assert policy.recipients is None
        assert policy.propagation is None
    
    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=5,
                min_trust_to_receive=0.6,
            ),
        )
        
        restored = SharePolicy.from_dict(original.to_dict())
        assert restored.level == original.level
        assert restored.enforcement == original.enforcement
        assert restored.propagation.max_hops == original.propagation.max_hops
        assert restored.propagation.min_trust_to_receive == original.propagation.min_trust_to_receive
    
    def test_allows_sharing_private(self):
        """Test that private policy blocks all sharing."""
        policy = SharePolicy.private()
        assert not policy.allows_sharing_to("did:key:anyone")
    
    def test_allows_sharing_public(self):
        """Test that public policy allows all sharing."""
        policy = SharePolicy.public()
        assert policy.allows_sharing_to("did:key:anyone")
    
    def test_allows_sharing_direct(self):
        """Test that direct policy only allows listed recipients."""
        policy = SharePolicy.direct(["did:key:alice", "did:key:bob"])
        assert policy.allows_sharing_to("did:key:alice")
        assert policy.allows_sharing_to("did:key:bob")
        assert not policy.allows_sharing_to("did:key:charlie")
    
    def test_is_expired_no_expiry(self):
        """Test is_expired when no expiry is set."""
        policy = SharePolicy.public()
        assert not policy.is_expired()
    
    def test_is_expired_future(self):
        """Test is_expired with future expiry."""
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            propagation=PropagationRules(
                expires_at=datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=30),
            ),
        )
        assert not policy.is_expired()
    
    def test_is_expired_past(self):
        """Test is_expired with past expiry."""
        policy = SharePolicy(
            level=ShareLevel.BOUNDED,
            propagation=PropagationRules(
                expires_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1),
            ),
        )
        assert policy.is_expired()
