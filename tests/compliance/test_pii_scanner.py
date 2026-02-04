"""Tests for PII scanner (Issue #35).

Verifies:
- Email, phone, SSN pattern detection
- Classification levels (L0-L4)
- Federation blocking for L3+ content
- Force override functionality
"""

from __future__ import annotations

import pytest

from valence.compliance.pii_scanner import (
    PIIScanner,
    PIIMatch,
    PIIType,
    ClassificationLevel,
    ScanResult,
    scan_for_pii,
    check_federation_allowed,
)


class TestPIITypes:
    """Test detection of different PII types."""

    def test_email_detection(self):
        """Should detect email addresses."""
        scanner = PIIScanner()
        result = scanner.scan("Contact me at john.doe@example.com for details")
        
        assert result.contains_pii
        assert len(result.matches) == 1
        assert result.matches[0].pii_type == PIIType.EMAIL
        assert result.matches[0].value == "john.doe@example.com"
        assert result.max_classification == ClassificationLevel.L3_PERSONAL

    def test_multiple_emails(self):
        """Should detect multiple email addresses."""
        scanner = PIIScanner()
        result = scanner.scan("Send to alice@test.org and bob@company.co.uk")
        
        assert result.contains_pii
        assert len(result.matches) == 2
        emails = {m.value for m in result.matches}
        assert "alice@test.org" in emails
        assert "bob@company.co.uk" in emails

    def test_us_phone_formats(self):
        """Should detect various US phone number formats."""
        scanner = PIIScanner()
        
        test_cases = [
            ("Call 555-123-4567 today", "555-123-4567"),
            ("Phone: (555) 123-4567", "(555) 123-4567"),
            ("Dial 555.123.4567", "555.123.4567"),
            ("Reach us at +1-555-123-4567", "+1-555-123-4567"),
        ]
        
        for text, expected in test_cases:
            result = scanner.scan(text)
            assert result.contains_pii, f"Failed to detect phone in: {text}"
            # May detect as US or INTL depending on format
            phone_matches = [m for m in result.matches if "phone" in m.pii_type.value]
            assert len(phone_matches) >= 1, f"No phone match for: {text}"

    def test_ssn_detection(self):
        """Should detect SSN patterns."""
        scanner = PIIScanner()
        result = scanner.scan("SSN: 123-45-6789")
        
        assert result.contains_pii
        ssn_matches = [m for m in result.matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) == 1
        assert result.max_classification == ClassificationLevel.L4_PROHIBITED

    def test_ssn_invalid_formats(self):
        """Should not flag invalid SSN patterns."""
        scanner = PIIScanner()
        
        # Invalid SSNs (starting with 000, 666, or 9xx)
        invalid = [
            "000-12-3456",  # Invalid area number
            "666-12-3456",  # Invalid area number
            "900-12-3456",  # Invalid area number (9xx series)
        ]
        
        for ssn in invalid:
            result = scanner.scan(f"Number: {ssn}")
            ssn_matches = [m for m in result.matches if m.pii_type == PIIType.SSN]
            assert len(ssn_matches) == 0, f"Should not detect invalid SSN: {ssn}"

    def test_credit_card_detection(self):
        """Should detect credit card numbers."""
        scanner = PIIScanner()
        
        test_cases = [
            ("Visa: 4111111111111111", "4111111111111111"),
            ("MC: 5500000000000004", "5500000000000004"),
            ("Amex: 378282246310005", "378282246310005"),
        ]
        
        for text, expected in test_cases:
            result = scanner.scan(text)
            assert result.contains_pii, f"Failed to detect CC in: {text}"
            cc_matches = [m for m in result.matches if m.pii_type == PIIType.CREDIT_CARD]
            assert len(cc_matches) >= 1, f"No CC match for: {text}"
            assert result.max_classification == ClassificationLevel.L4_PROHIBITED

    def test_ip_address_detection(self):
        """Should detect IP addresses."""
        scanner = PIIScanner()
        result = scanner.scan("Server IP: 192.168.1.100")
        
        assert result.contains_pii
        ip_matches = [m for m in result.matches if m.pii_type == PIIType.IP_ADDRESS]
        assert len(ip_matches) == 1
        assert ip_matches[0].value == "192.168.1.100"
        assert result.max_classification == ClassificationLevel.L2_SENSITIVE

    def test_no_pii(self):
        """Should handle text without PII."""
        scanner = PIIScanner()
        result = scanner.scan("PostgreSQL scales better for read-heavy workloads")
        
        assert not result.contains_pii
        assert len(result.matches) == 0
        assert result.max_classification == ClassificationLevel.L0_PUBLIC
        assert result.can_federate


class TestClassificationLevels:
    """Test classification level assignment and behavior."""

    def test_l0_public(self):
        """L0 content should federate freely."""
        result = scan_for_pii("This is public information")
        
        assert result.max_classification == ClassificationLevel.L0_PUBLIC
        assert result.can_federate
        assert not result.requires_consent
        assert not result.hard_blocked

    def test_l2_sensitive(self):
        """L2 content requires consent but can federate."""
        # IP addresses are L2
        result = scan_for_pii("Server at 10.0.0.1 handles requests")
        
        assert result.max_classification == ClassificationLevel.L2_SENSITIVE
        assert result.can_federate
        assert result.requires_consent
        assert not result.hard_blocked

    def test_l3_personal(self):
        """L3 content blocked from auto-federation."""
        # Emails are L3
        result = scan_for_pii("Contact john@example.com")
        
        assert result.max_classification == ClassificationLevel.L3_PERSONAL
        assert not result.can_federate
        assert result.requires_consent
        assert not result.hard_blocked

    def test_l4_prohibited(self):
        """L4 content is hard blocked."""
        # SSNs are L4
        result = scan_for_pii("SSN: 123-45-6789")
        
        assert result.max_classification == ClassificationLevel.L4_PROHIBITED
        assert not result.can_federate
        assert result.hard_blocked


class TestFederationBlocking:
    """Test federation blocking logic (Issue #35 core requirement)."""

    def test_allows_l0_content(self):
        """Should allow L0 (public) content."""
        allowed, result = check_federation_allowed("Public knowledge")
        assert allowed
        assert result.can_federate

    def test_allows_l2_content(self):
        """Should allow L2 (sensitive) content with warning."""
        allowed, result = check_federation_allowed("Server IP: 192.168.1.1")
        assert allowed
        assert result.requires_consent

    def test_blocks_l3_without_force(self):
        """Should soft-block L3 content without force flag."""
        allowed, result = check_federation_allowed("Email: test@example.com")
        
        assert not allowed
        assert result.max_classification == ClassificationLevel.L3_PERSONAL
        assert not result.hard_blocked  # Soft block, not hard

    def test_allows_l3_with_force(self):
        """Should allow L3 content with force flag (explicit confirmation)."""
        allowed, result = check_federation_allowed(
            "Email: test@example.com",
            force=True
        )
        
        assert allowed  # Force overrides soft block
        assert result.max_classification == ClassificationLevel.L3_PERSONAL

    def test_blocks_l4_always(self):
        """Should hard-block L4 content even with force flag."""
        allowed, result = check_federation_allowed(
            "SSN: 123-45-6789",
            force=True
        )
        
        assert not allowed  # L4 is always blocked
        assert result.hard_blocked
        assert result.max_classification == ClassificationLevel.L4_PROHIBITED


class TestPIIRedaction:
    """Test PII redaction functionality."""

    def test_email_redaction(self):
        """Should redact emails showing partial info."""
        scanner = PIIScanner()
        result = scanner.scan("Contact john.doe@example.com")
        
        assert len(result.matches) == 1
        redacted = result.matches[0].redacted_value
        assert "@example.com" in redacted
        assert "john.doe" not in redacted

    def test_phone_redaction(self):
        """Should redact phones showing last 4 digits."""
        scanner = PIIScanner()
        result = scanner.scan("Call 555-123-4567")
        
        phone_matches = [m for m in result.matches if "phone" in m.pii_type.value]
        if phone_matches:
            redacted = phone_matches[0].redacted_value
            assert "4567" in redacted
            assert "555" not in redacted

    def test_ssn_full_redaction(self):
        """Should fully redact SSNs."""
        scanner = PIIScanner()
        result = scanner.scan("SSN: 123-45-6789")
        
        ssn_matches = [m for m in result.matches if m.pii_type == PIIType.SSN]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].redacted_value == "***-**-****"

    def test_redact_text_preserves_structure(self):
        """Should redact text while preserving non-PII content."""
        scanner = PIIScanner()
        original = "Contact john@example.com for the PostgreSQL guide"
        redacted = scanner.redact_text(original)
        
        assert "PostgreSQL guide" in redacted
        assert "john@example.com" not in redacted
        assert "@example.com" in redacted  # Domain preserved


class TestScannerConfiguration:
    """Test scanner configuration options."""

    def test_custom_enabled_types(self):
        """Should only scan for enabled types."""
        scanner = PIIScanner(enabled_types={PIIType.EMAIL})
        
        # Should detect email
        result1 = scanner.scan("Email: test@example.com")
        assert result1.contains_pii
        
        # Should NOT detect phone when not enabled
        result2 = scanner.scan("Phone: 555-123-4567 Email: none")
        phone_matches = [m for m in result2.matches if "phone" in m.pii_type.value]
        assert len(phone_matches) == 0

    def test_empty_text(self):
        """Should handle empty text."""
        result = scan_for_pii("")
        assert not result.contains_pii
        assert len(result.matches) == 0

    def test_unicode_text(self):
        """Should handle unicode text."""
        result = scan_for_pii("联系方式: test@example.com")
        assert result.contains_pii
        assert len(result.matches) == 1


class TestScanResultSerialization:
    """Test ScanResult serialization for API responses."""

    def test_to_dict_complete(self):
        """Should serialize all fields."""
        result = scan_for_pii("Contact john@example.com, SSN: 123-45-6789")
        
        data = result.to_dict()
        
        assert "contains_pii" in data
        assert "max_classification_level" in data
        assert "max_classification_name" in data
        assert "match_count" in data
        assert "matches" in data
        assert "can_federate" in data
        assert "requires_consent" in data
        assert "hard_blocked" in data

    def test_match_to_dict(self):
        """Should serialize match with redacted value."""
        result = scan_for_pii("Email: test@example.com")
        
        assert len(result.matches) == 1
        match_data = result.matches[0].to_dict()
        
        assert match_data["type"] == "email"
        assert "value" in match_data  # Should be redacted
        assert "test@example.com" not in match_data["value"]  # Not full email
        assert "start" in match_data
        assert "end" in match_data
        assert "classification_level" in match_data
