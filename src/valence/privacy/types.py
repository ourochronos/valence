"""Privacy types for Valence belief sharing.

Implements SharePolicy with graduated sharing levels and enforcement types.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone


class ShareLevel(Enum):
    """Graduated levels of belief sharing permissions."""
    
    PRIVATE = "private"      # Never leaves node
    DIRECT = "direct"        # Specific recipient, no reshare
    BOUNDED = "bounded"      # Can reshare within scope
    CASCADING = "cascading"  # Propagates with restrictions
    PUBLIC = "public"        # Open


class EnforcementType(Enum):
    """How sharing policies are enforced."""
    
    CRYPTOGRAPHIC = "cryptographic"  # Math enforced (encryption, signatures)
    POLICY = "policy"                # Protocol enforced (software checks)
    HONOR = "honor"                  # Trust-based (no technical enforcement)


@dataclass
class PropagationRules:
    """Rules governing how beliefs can propagate through the network."""
    
    max_hops: Optional[int] = None
    allowed_domains: Optional[List[str]] = None
    min_trust_to_receive: Optional[float] = None
    strip_on_forward: Optional[List[str]] = None  # Fields to remove on forward
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "max_hops": self.max_hops,
            "allowed_domains": self.allowed_domains,
            "min_trust_to_receive": self.min_trust_to_receive,
            "strip_on_forward": self.strip_on_forward,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropagationRules":
        """Deserialize from dictionary."""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])
        
        return cls(
            max_hops=data.get("max_hops"),
            allowed_domains=data.get("allowed_domains"),
            min_trust_to_receive=data.get("min_trust_to_receive"),
            strip_on_forward=data.get("strip_on_forward"),
            expires_at=expires_at,
        )


@dataclass
class SharePolicy:
    """Policy controlling how a belief can be shared.
    
    Combines a ShareLevel with enforcement type and optional propagation rules.
    """
    
    level: ShareLevel
    enforcement: EnforcementType = EnforcementType.POLICY
    recipients: Optional[List[str]] = None  # DIDs for DIRECT sharing
    propagation: Optional[PropagationRules] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "level": self.level.value,
            "enforcement": self.enforcement.value,
            "recipients": self.recipients,
            "propagation": self.propagation.to_dict() if self.propagation else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharePolicy":
        """Deserialize from dictionary."""
        return cls(
            level=ShareLevel(data["level"]),
            enforcement=EnforcementType(data.get("enforcement", "policy")),
            recipients=data.get("recipients"),
            propagation=PropagationRules.from_dict(data["propagation"]) if data.get("propagation") else None,
        )
    
    @classmethod
    def private(cls) -> "SharePolicy":
        """Create a private policy - belief never leaves the node."""
        return cls(level=ShareLevel.PRIVATE, enforcement=EnforcementType.CRYPTOGRAPHIC)
    
    @classmethod
    def public(cls) -> "SharePolicy":
        """Create a public policy - belief is openly shareable."""
        return cls(level=ShareLevel.PUBLIC, enforcement=EnforcementType.HONOR)
    
    @classmethod
    def direct(cls, recipients: List[str]) -> "SharePolicy":
        """Create a direct policy - share only with specific recipients."""
        return cls(
            level=ShareLevel.DIRECT,
            enforcement=EnforcementType.CRYPTOGRAPHIC,
            recipients=recipients,
        )
    
    @classmethod
    def bounded(cls, max_hops: int = 2, allowed_domains: Optional[List[str]] = None) -> "SharePolicy":
        """Create a bounded policy - can reshare within scope."""
        return cls(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.POLICY,
            propagation=PropagationRules(
                max_hops=max_hops,
                allowed_domains=allowed_domains,
            ),
        )
    
    def allows_sharing_to(self, recipient_did: str) -> bool:
        """Check if this policy allows sharing to a specific recipient."""
        if self.level == ShareLevel.PRIVATE:
            return False
        if self.level == ShareLevel.PUBLIC:
            return True
        if self.level == ShareLevel.DIRECT:
            return self.recipients is not None and recipient_did in self.recipients
        # BOUNDED and CASCADING require additional context (hop count, trust, etc.)
        return True
    
    def is_expired(self) -> bool:
        """Check if the policy has expired."""
        if self.propagation and self.propagation.expires_at:
            now = datetime.now(timezone.utc).replace(tzinfo=None)
            return now > self.propagation.expires_at
        return False
