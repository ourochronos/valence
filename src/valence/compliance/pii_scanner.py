"""PII Scanner for content classification before federation.

Implements PII detection per COMPLIANCE.md §1:
- Email, phone, SSN patterns
- Blocks L3+ content from auto-federation
- Supports --force override for manual review

Reference: spec/compliance/COMPLIANCE.md §1 "PII Detection"
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class PIIType(StrEnum):
    """Types of PII that can be detected."""
    
    EMAIL = "email"
    PHONE_US = "phone_us"
    PHONE_INTL = "phone_intl"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    # Future: Named entity recognition for names, addresses


class ClassificationLevel(IntEnum):
    """Data classification levels per COMPLIANCE.md §1."""
    
    L0_PUBLIC = 0       # Public - can federate freely
    L1_SHARED = 1       # Shared - with consent
    L2_SENSITIVE = 2    # Sensitive - restricted
    L3_PERSONAL = 3     # Personal - never auto-federate
    L4_PROHIBITED = 4   # Prohibited - hard block


# PII types and their classification levels
PII_CLASSIFICATION: dict[PIIType, ClassificationLevel] = {
    PIIType.EMAIL: ClassificationLevel.L3_PERSONAL,
    PIIType.PHONE_US: ClassificationLevel.L3_PERSONAL,
    PIIType.PHONE_INTL: ClassificationLevel.L3_PERSONAL,
    PIIType.SSN: ClassificationLevel.L4_PROHIBITED,
    PIIType.CREDIT_CARD: ClassificationLevel.L4_PROHIBITED,
    PIIType.IP_ADDRESS: ClassificationLevel.L2_SENSITIVE,
}


@dataclass
class PIIMatch:
    """A detected PII occurrence in text."""
    
    pii_type: PIIType
    value: str
    start: int
    end: int
    classification: ClassificationLevel
    redacted_value: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "type": self.pii_type.value,
            "value": self.redacted_value or "[REDACTED]",
            "start": self.start,
            "end": self.end,
            "classification_level": self.classification.value,
            "classification_name": self.classification.name,
        }


@dataclass
class ScanResult:
    """Result of PII scanning."""
    
    contains_pii: bool
    max_classification: ClassificationLevel
    matches: list[PIIMatch] = field(default_factory=list)
    can_federate: bool = True
    requires_consent: bool = False
    hard_blocked: bool = False
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize for API response."""
        return {
            "contains_pii": self.contains_pii,
            "max_classification_level": self.max_classification.value,
            "max_classification_name": self.max_classification.name,
            "match_count": len(self.matches),
            "matches": [m.to_dict() for m in self.matches],
            "can_federate": self.can_federate,
            "requires_consent": self.requires_consent,
            "hard_blocked": self.hard_blocked,
        }


class PIIScanner:
    """Scanner for detecting PII in text content.
    
    Uses regex patterns for common PII types. For production use,
    consider adding NER (Named Entity Recognition) for names and addresses.
    """
    
    # Compiled regex patterns
    PATTERNS: dict[PIIType, re.Pattern[str]] = {
        # Email: standard format
        PIIType.EMAIL: re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            re.IGNORECASE
        ),
        
        # US Phone: (xxx) xxx-xxxx, xxx-xxx-xxxx, xxx.xxx.xxxx, etc.
        PIIType.PHONE_US: re.compile(
            r'\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'
        ),
        
        # International phone: +XX format with various separators
        PIIType.PHONE_INTL: re.compile(
            r'\b\+(?!1\s)(?:[0-9][-.\s]?){7,14}[0-9]\b'
        ),
        
        # SSN: xxx-xx-xxxx format (US Social Security Number)
        PIIType.SSN: re.compile(
            r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
        ),
        
        # Credit card: 13-19 digits with optional separators
        # Covers Visa, MasterCard, Amex, Discover, etc.
        PIIType.CREDIT_CARD: re.compile(
            r'\b(?:4[0-9]{12}(?:[0-9]{3})?|'  # Visa
            r'5[1-5][0-9]{14}|'                # MasterCard
            r'3[47][0-9]{13}|'                 # Amex
            r'6(?:011|5[0-9]{2})[0-9]{12}|'   # Discover
            r'(?:2131|1800|35\d{3})\d{11})\b' # JCB
        ),
        
        # IPv4 address
        PIIType.IP_ADDRESS: re.compile(
            r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        ),
    }
    
    def __init__(self, enabled_types: set[PIIType] | None = None):
        """Initialize scanner.
        
        Args:
            enabled_types: Set of PII types to scan for.
                          If None, scans for all types.
        """
        self.enabled_types = enabled_types or set(PIIType)
    
    def scan(self, text: str) -> ScanResult:
        """Scan text for PII.
        
        Args:
            text: Content to scan
            
        Returns:
            ScanResult with all matches and classification
        """
        matches: list[PIIMatch] = []
        max_classification = ClassificationLevel.L0_PUBLIC
        
        for pii_type in self.enabled_types:
            pattern = self.PATTERNS.get(pii_type)
            if not pattern:
                continue
            
            for match in pattern.finditer(text):
                classification = PII_CLASSIFICATION.get(
                    pii_type, 
                    ClassificationLevel.L2_SENSITIVE
                )
                
                pii_match = PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    classification=classification,
                    redacted_value=self._redact(pii_type, match.group()),
                )
                matches.append(pii_match)
                
                if classification > max_classification:
                    max_classification = classification
        
        # Determine federation eligibility
        can_federate = max_classification < ClassificationLevel.L3_PERSONAL
        requires_consent = max_classification >= ClassificationLevel.L2_SENSITIVE
        hard_blocked = max_classification >= ClassificationLevel.L4_PROHIBITED
        
        return ScanResult(
            contains_pii=len(matches) > 0,
            max_classification=max_classification,
            matches=matches,
            can_federate=can_federate,
            requires_consent=requires_consent,
            hard_blocked=hard_blocked,
        )
    
    def _redact(self, pii_type: PIIType, value: str) -> str:
        """Create redacted version of PII value.
        
        Shows partial info for identification without full exposure.
        """
        if pii_type == PIIType.EMAIL:
            parts = value.split('@')
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                redacted_local = local[0] + '*' * (len(local) - 1)
                return f"{redacted_local}@{domain}"
            return value[:2] + '***'
        
        elif pii_type in (PIIType.PHONE_US, PIIType.PHONE_INTL):
            # Show last 4 digits
            digits = re.sub(r'[^\d]', '', value)
            return f"***-***-{digits[-4:]}" if len(digits) >= 4 else "***"
        
        elif pii_type == PIIType.SSN:
            return "***-**-****"
        
        elif pii_type == PIIType.CREDIT_CARD:
            digits = re.sub(r'[^\d]', '', value)
            return f"****-****-****-{digits[-4:]}" if len(digits) >= 4 else "****"
        
        elif pii_type == PIIType.IP_ADDRESS:
            parts = value.split('.')
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.***.***"
            return "***.***.***"
        
        return "***"
    
    def redact_text(self, text: str) -> str:
        """Return text with all PII redacted.
        
        Args:
            text: Original text
            
        Returns:
            Text with PII replaced by redacted versions
        """
        result = self.scan(text)
        
        if not result.matches:
            return text
        
        # Sort matches by start position (reverse) for safe replacement
        sorted_matches = sorted(result.matches, key=lambda m: m.start, reverse=True)
        
        redacted = text
        for match in sorted_matches:
            redacted = (
                redacted[:match.start] + 
                (match.redacted_value or "[REDACTED]") + 
                redacted[match.end:]
            )
        
        return redacted


# Global scanner instance
_default_scanner: PIIScanner | None = None


def get_scanner() -> PIIScanner:
    """Get the default PII scanner."""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = PIIScanner()
    return _default_scanner


def scan_for_pii(text: str) -> ScanResult:
    """Convenience function to scan text for PII.
    
    Args:
        text: Content to scan
        
    Returns:
        ScanResult with matches and classification
    """
    return get_scanner().scan(text)


def check_federation_allowed(
    content: str,
    force: bool = False,
) -> tuple[bool, ScanResult]:
    """Check if content can be federated.
    
    Implements the classification enforcement from COMPLIANCE.md §1:
    - L4: HARD BLOCK, log for review
    - L3: SOFT BLOCK, require explicit confirmation (force flag)
    - L2: WARNING, require consent acknowledgment
    
    Args:
        content: Content to check
        force: Override L3 soft blocks (requires explicit confirmation)
        
    Returns:
        Tuple of (allowed, scan_result)
    """
    result = scan_for_pii(content)
    
    # L4 is always blocked
    if result.hard_blocked:
        logger.warning(
            f"Federation HARD BLOCKED: L4 content detected "
            f"({len(result.matches)} PII matches)"
        )
        return False, result
    
    # L3 requires force flag
    if result.max_classification >= ClassificationLevel.L3_PERSONAL:
        if not force:
            logger.info(
                f"Federation SOFT BLOCKED: L3 content detected "
                f"(use --force to override)"
            )
            return False, result
        else:
            logger.warning(
                f"Federation forced for L3 content "
                f"({len(result.matches)} PII matches)"
            )
    
    return True, result
