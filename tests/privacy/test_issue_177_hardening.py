"""Tests for Issue #177: Privacy hardening batch.

Tests all the security improvements from the hardening batch:
- Medium severity: sensitive domain classification, audit PII sanitization,
  watermark secret_key warnings, budget consumption for failed queries,
  aggregated sync logging
- Low severity: capability TTL reduction, consent store LRU limits,
  canary strip_canaries warning, extraction rate limiting
"""

import pytest
import re
import secrets
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

# =============================================================================
# MEDIUM SEVERITY TESTS
# =============================================================================


class TestSensitiveDomainClassification:
    """Test structured classification for sensitive domains."""

    def test_exact_match_works(self):
        """Sensitive domains detected with exact token match."""
        from valence.federation.privacy import is_sensitive_domain
        
        # Exact matches should work
        assert is_sensitive_domain(["health"]) is True
        assert is_sensitive_domain(["medical"]) is True
        assert is_sensitive_domain(["finance"]) is True
        assert is_sensitive_domain(["legal"]) is True
    
    def test_path_tokens_detected(self):
        """Domains in paths are detected."""
        from valence.federation.privacy import is_sensitive_domain
        
        # Path separators should be split
        assert is_sensitive_domain(["personal/health"]) is True
        assert is_sensitive_domain(["data/medical/records"]) is True
        assert is_sensitive_domain(["user_finance"]) is True
        assert is_sensitive_domain(["legal-documents"]) is True
    
    def test_no_false_positives_from_substring(self):
        """Substring matching no longer causes false positives."""
        from valence.federation.privacy import is_sensitive_domain
        
        # These should NOT match because we use token matching, not substring
        assert is_sensitive_domain(["weather"]) is False  # contains "the" 
        assert is_sensitive_domain(["healthcare_news"]) is True  # health is a token
        assert is_sensitive_domain(["definitely"]) is False  # contains "fin"
    
    def test_non_sensitive_domains(self):
        """Non-sensitive domains return False."""
        from valence.federation.privacy import is_sensitive_domain
        
        assert is_sensitive_domain(["science"]) is False
        assert is_sensitive_domain(["sports"]) is False
        assert is_sensitive_domain(["entertainment"]) is False
        assert is_sensitive_domain(["general"]) is False
    
    def test_get_sensitive_category(self):
        """Category detection works."""
        from valence.federation.privacy import get_sensitive_category
        
        assert get_sensitive_category("health") == "health"
        assert get_sensitive_category("finance") == "finance"
        assert get_sensitive_category("immigration") == "immigration"
        assert get_sensitive_category("sports") is None


class TestAuditPIISanitization:
    """Test PII sanitization in audit metadata."""

    def test_email_sanitized(self):
        """Email addresses are redacted."""
        from valence.privacy.audit import sanitize_metadata
        
        result = sanitize_metadata({"contact": "user@example.com"})
        assert "[PII_REDACTED]" in result["contact"]
        assert "user@example.com" not in result["contact"]
    
    def test_phone_sanitized(self):
        """Phone numbers are redacted."""
        from valence.privacy.audit import sanitize_metadata
        
        result = sanitize_metadata({"phone": "555-123-4567"})
        assert "[PII_REDACTED]" in result["phone"]
    
    def test_ssn_sanitized(self):
        """SSN patterns are redacted."""
        from valence.privacy.audit import sanitize_metadata
        
        result = sanitize_metadata({"ssn": "123-45-6789"})
        assert "[PII_REDACTED]" in result["ssn"]
    
    def test_sensitive_keys_fully_redacted(self):
        """Keys named after sensitive data are fully redacted."""
        from valence.privacy.audit import sanitize_metadata
        
        result = sanitize_metadata({
            "password": "secret123",
            "api_key": "sk-abc123",
            "token": "bearer_xyz",
        })
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
    
    def test_non_sensitive_preserved(self):
        """Non-sensitive data is preserved."""
        from valence.privacy.audit import sanitize_metadata
        
        result = sanitize_metadata({
            "action": "login",
            "timestamp": "2024-01-01T00:00:00Z",
            "resource_id": "belief_123",
        })
        assert result["action"] == "login"
        assert result["timestamp"] == "2024-01-01T00:00:00Z"
        assert result["resource_id"] == "belief_123"
    
    def test_audit_event_auto_sanitizes(self):
        """AuditEvent automatically sanitizes metadata on creation."""
        from valence.privacy.audit import AuditEvent, AuditEventType
        
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_DENIED,
            actor_did="did:example:alice",
            resource="belief_123",
            action="read",
            success=False,
            metadata={"email": "alice@example.com", "reason": "unauthorized"},
        )
        
        # Email should be sanitized
        assert "[PII_REDACTED]" in event.metadata["email"]
        # Non-PII preserved
        assert event.metadata["reason"] == "unauthorized"


class TestWatermarkSecurityWarnings:
    """Test WatermarkRegistry security documentation."""
    
    def test_minimum_key_length_enforced(self):
        """Short secret keys are rejected."""
        from valence.privacy.watermark import WatermarkRegistry
        
        with pytest.raises(ValueError, match="at least 16 bytes"):
            WatermarkRegistry(secret_key=b"short")
    
    def test_valid_key_accepted(self):
        """Valid length keys work."""
        from valence.privacy.watermark import WatermarkRegistry
        
        registry = WatermarkRegistry(secret_key=secrets.token_bytes(32))
        assert registry.secret_key is not None
    
    def test_repr_hides_secret(self):
        """__repr__ doesn't expose the secret key."""
        from valence.privacy.watermark import WatermarkRegistry
        
        key = secrets.token_bytes(32)
        registry = WatermarkRegistry(secret_key=key)
        repr_str = repr(registry)
        
        # Should not contain the key bytes
        assert key.hex() not in repr_str
        assert "secret" not in repr_str.lower() or "WatermarkRegistry" in repr_str


class TestFailedQueryBudgetConsumption:
    """Test that failed k-anonymity queries consume budget."""
    
    def test_failed_query_consumes_budget(self):
        """Queries failing k-anonymity still consume epsilon."""
        from valence.federation.privacy import (
            execute_private_query,
            PrivacyConfig,
            PrivacyBudget,
            FAILED_QUERY_EPSILON_COST,
        )
        
        config = PrivacyConfig(min_contributors=5)
        budget = PrivacyBudget(federation_id=uuid4())
        
        initial_epsilon = budget.spent_epsilon
        
        # Query with insufficient contributors
        result = execute_private_query(
            confidences=[0.5, 0.6],  # Only 2, need 5
            config=config,
            budget=budget,
            topic_hash="test_topic",
        )
        
        # Query should fail
        assert result.success is False
        assert "Insufficient contributors" in result.failure_reason
        
        # But budget should be consumed
        assert result.budget_consumed is True
        assert result.epsilon_consumed == FAILED_QUERY_EPSILON_COST
        assert budget.spent_epsilon == initial_epsilon + FAILED_QUERY_EPSILON_COST
    
    def test_successful_query_consumes_full_budget(self):
        """Successful queries consume full epsilon."""
        from valence.federation.privacy import (
            execute_private_query,
            PrivacyConfig,
            PrivacyBudget,
        )
        
        config = PrivacyConfig(epsilon=1.0, min_contributors=3)
        budget = PrivacyBudget(federation_id=uuid4())
        
        # Query with sufficient contributors
        result = execute_private_query(
            confidences=[0.5, 0.6, 0.7, 0.8, 0.9],
            config=config,
            budget=budget,
            topic_hash="test_topic",
        )
        
        assert result.success is True
        assert result.epsilon_consumed == 1.0


# =============================================================================
# LOW SEVERITY TESTS
# =============================================================================


class TestCapabilityTTL:
    """Test reduced default capability TTL."""
    
    def test_default_ttl_is_15_minutes(self):
        """Default TTL should be 15 minutes, not 1 hour."""
        from valence.privacy.capabilities import DEFAULT_TTL_SECONDS
        
        assert DEFAULT_TTL_SECONDS == 900  # 15 * 60
    
    def test_capability_uses_new_default(self):
        """New capabilities use the reduced TTL."""
        from valence.privacy.capabilities import Capability
        
        cap = Capability.create(
            issuer_did="did:example:issuer",
            holder_did="did:example:holder",
            resource="test_resource",
            actions=["read"],
        )
        
        # Should expire in ~15 minutes, not 1 hour
        ttl = cap.ttl_seconds
        assert ttl <= 900
        assert ttl > 850  # Allow some clock drift


class TestConsentStoreLRU:
    """Test LRU eviction in InMemoryConsentChainStore."""
    
    @pytest.mark.asyncio
    async def test_chains_evicted_at_capacity(self):
        """Old chains are evicted when max capacity reached."""
        from valence.federation.consent import (
            InMemoryConsentChainStore,
            CrossFederationConsentChain,
        )
        
        store = InMemoryConsentChainStore(max_chains=3, max_revocations=10)
        
        # Add 4 chains (1 more than max)
        chains = []
        for i in range(4):
            chain = CrossFederationConsentChain(
                id=f"chain_{i}",
                original_chain_id=f"orig_{i}",
                origin_federation_id="fed_a",
                origin_gateway_id="gw_a",
            )
            chains.append(chain)
            await store.store_cross_chain(chain)
        
        # First chain should be evicted
        assert await store.get_cross_chain("chain_0") is None
        assert await store.get_cross_chain("chain_1") is not None
        assert await store.get_cross_chain("chain_2") is not None
        assert await store.get_cross_chain("chain_3") is not None
    
    @pytest.mark.asyncio
    async def test_accessed_chains_not_evicted(self):
        """Accessing a chain moves it to end of LRU queue."""
        from valence.federation.consent import (
            InMemoryConsentChainStore,
            CrossFederationConsentChain,
        )
        
        store = InMemoryConsentChainStore(max_chains=3, max_revocations=10)
        
        # Add 3 chains
        for i in range(3):
            chain = CrossFederationConsentChain(
                id=f"chain_{i}",
                original_chain_id=f"orig_{i}",
                origin_federation_id="fed_a",
                origin_gateway_id="gw_a",
            )
            await store.store_cross_chain(chain)
        
        # Access chain_0, making it recently used
        await store.get_cross_chain("chain_0")
        
        # Add another chain - should evict chain_1 (oldest non-accessed)
        chain = CrossFederationConsentChain(
            id="chain_3",
            original_chain_id="orig_3",
            origin_federation_id="fed_a",
            origin_gateway_id="gw_a",
        )
        await store.store_cross_chain(chain)
        
        # chain_0 should still exist (was accessed), chain_1 evicted
        assert await store.get_cross_chain("chain_0") is not None
        assert await store.get_cross_chain("chain_1") is None


class TestCanaryStripWarning:
    """Test that strip_canaries has security documentation."""
    
    def test_docstring_contains_warning(self):
        """strip_canaries docstring contains security warning."""
        from valence.privacy.canary import strip_canaries
        
        docstring = strip_canaries.__doc__
        assert "WARNING" in docstring or "⚠️" in docstring
        assert "ADVERSARIAL" in docstring
    
    def test_strip_canaries_still_works(self):
        """Function still strips canaries despite warnings."""
        from valence.privacy.canary import (
            strip_canaries, 
            embed_canary,
            CanaryToken,
            EmbeddingStrategy,
        )
        
        secret = secrets.token_bytes(32)
        token = CanaryToken.generate(secret, "test_recipient")
        
        content = "This is test content."
        watermarked = embed_canary(content, token, EmbeddingStrategy.VISIBLE)
        
        # Verify canary is present
        assert "CANARY" in watermarked
        
        # Strip should remove it
        stripped = strip_canaries(watermarked)
        assert "CANARY" not in stripped


class TestExtractionRateLimiting:
    """Test per-DID rate limiting in ExtractionService."""
    
    def test_rate_limit_enforced(self):
        """Requests beyond limit are rejected."""
        from valence.privacy.extraction import (
            ExtractionService,
            ExtractionLevel,
            RateLimitConfig,
            RateLimitExceededError,
        )
        
        # Very low limit for testing
        config = RateLimitConfig(max_extractions_per_window=2, window_seconds=3600)
        service = ExtractionService(rate_limit_config=config)
        
        # First two should succeed
        service.extract("content 1", ExtractionLevel.THEMES, "src_1", "did:test:alice")
        service.extract("content 2", ExtractionLevel.THEMES, "src_2", "did:test:alice")
        
        # Third should fail
        with pytest.raises(RateLimitExceededError) as exc_info:
            service.extract("content 3", ExtractionLevel.THEMES, "src_3", "did:test:alice")
        
        assert exc_info.value.did == "did:test:alice"
        assert exc_info.value.limit == 2
    
    def test_different_dids_have_separate_limits(self):
        """Different DIDs have independent rate limits."""
        from valence.privacy.extraction import (
            ExtractionService,
            ExtractionLevel,
            RateLimitConfig,
            RateLimitExceededError,
        )
        
        config = RateLimitConfig(max_extractions_per_window=1, window_seconds=3600)
        service = ExtractionService(rate_limit_config=config)
        
        # Alice uses her limit
        service.extract("content 1", ExtractionLevel.THEMES, "src_1", "did:test:alice")
        
        # Bob can still extract
        service.extract("content 2", ExtractionLevel.THEMES, "src_2", "did:test:bob")
        
        # Alice is blocked
        with pytest.raises(RateLimitExceededError):
            service.extract("content 3", ExtractionLevel.THEMES, "src_3", "did:test:alice")
    
    def test_rate_limit_status_reporting(self):
        """Rate limit status is correctly reported."""
        from valence.privacy.extraction import (
            ExtractionService,
            ExtractionLevel,
            RateLimitConfig,
        )
        
        config = RateLimitConfig(max_extractions_per_window=5, window_seconds=3600)
        service = ExtractionService(rate_limit_config=config)
        
        # Before any extractions
        status = service.get_rate_limit_status("did:test:alice")
        assert status["remaining"] == 5
        assert status["used"] == 0
        
        # After one extraction
        service.extract("content", ExtractionLevel.THEMES, "src", "did:test:alice")
        status = service.get_rate_limit_status("did:test:alice")
        assert status["remaining"] == 4
        assert status["used"] == 1
    
    def test_no_did_skips_rate_limit(self):
        """Requests without DID skip rate limiting."""
        from valence.privacy.extraction import (
            ExtractionService,
            ExtractionLevel,
            RateLimitConfig,
        )
        
        config = RateLimitConfig(max_extractions_per_window=1, window_seconds=3600)
        service = ExtractionService(rate_limit_config=config)
        
        # Multiple requests without DID should all succeed
        for i in range(5):
            service.extract(f"content {i}", ExtractionLevel.THEMES, f"src_{i}")
