"""Canary token system for detecting unauthorized content sharing.

Embeds unique, traceable markers in shared content that can be detected
if the content is leaked or shared without authorization.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set
import hashlib
import hmac
import re
import secrets
import uuid


class EmbeddingStrategy(Enum):
    """Strategy for embedding canary tokens in content."""
    
    VISIBLE = "visible"  # Human-readable marker
    INVISIBLE = "invisible"  # Zero-width unicode characters


# Zero-width characters for invisible encoding
ZERO_WIDTH_SPACE = "\u200b"  # Binary 0
ZERO_WIDTH_NON_JOINER = "\u200c"  # Binary 1
ZERO_WIDTH_JOINER = "\u200d"  # Delimiter


@dataclass
class CanaryToken:
    """A unique, verifiable marker for tracking content sharing.
    
    Each token has:
    - A unique ID for tracking
    - An HMAC signature for verification (prevents forgery)
    - Metadata about when/why it was created
    """
    
    token_id: str
    signature: str  # HMAC-SHA256 of token_id with secret key
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    recipient_id: Optional[str] = None  # Who this content was shared with
    content_hash: Optional[str] = None  # Hash of original content for correlation
    metadata: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def generate(
        cls,
        secret_key: bytes,
        recipient_id: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "CanaryToken":
        """Generate a new canary token with HMAC signature.
        
        Args:
            secret_key: Secret key for HMAC signature (should be kept secret)
            recipient_id: Optional identifier for the content recipient
            content: Optional content to hash for correlation
            metadata: Optional additional metadata
            
        Returns:
            A new CanaryToken with unique ID and valid signature
        """
        # Generate unique token ID
        token_id = f"canary_{uuid.uuid4().hex[:16]}_{secrets.token_hex(8)}"
        
        # Create HMAC signature
        signature = hmac.new(
            secret_key,
            token_id.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        
        # Hash content if provided
        content_hash = None
        if content:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        
        return cls(
            token_id=token_id,
            signature=signature,
            recipient_id=recipient_id,
            content_hash=content_hash,
            metadata=metadata or {},
        )
    
    def verify(self, secret_key: bytes) -> bool:
        """Verify this token's signature is valid.
        
        Args:
            secret_key: The secret key used to generate the signature
            
        Returns:
            True if signature is valid, False otherwise
        """
        expected_signature = hmac.new(
            secret_key,
            self.token_id.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(self.signature, expected_signature)
    
    def to_marker(self) -> str:
        """Convert token to a compact marker string for embedding."""
        return f"{self.token_id}:{self.signature[:16]}"
    
    @classmethod
    def from_marker(cls, marker: str, secret_key: bytes) -> Optional["CanaryToken"]:
        """Reconstruct a token from a marker string.
        
        Args:
            marker: The marker string (token_id:signature_prefix)
            secret_key: Secret key to regenerate full signature
            
        Returns:
            CanaryToken if valid, None if marker is malformed
        """
        try:
            parts = marker.split(":")
            if len(parts) != 2:
                return None
            
            token_id, sig_prefix = parts
            
            # Regenerate full signature
            full_signature = hmac.new(
                secret_key,
                token_id.encode("utf-8"),
                hashlib.sha256
            ).hexdigest()
            
            # Verify prefix matches
            if not full_signature.startswith(sig_prefix):
                return None
            
            return cls(
                token_id=token_id,
                signature=full_signature,
            )
        except Exception:
            return None


def _encode_to_zero_width(data: str) -> str:
    """Encode a string to zero-width unicode characters.
    
    Uses binary encoding where:
    - ZERO_WIDTH_SPACE = 0
    - ZERO_WIDTH_NON_JOINER = 1
    - ZERO_WIDTH_JOINER = byte delimiter
    """
    result = []
    for byte in data.encode("utf-8"):
        bits = format(byte, "08b")
        for bit in bits:
            if bit == "0":
                result.append(ZERO_WIDTH_SPACE)
            else:
                result.append(ZERO_WIDTH_NON_JOINER)
        result.append(ZERO_WIDTH_JOINER)  # Byte delimiter
    return "".join(result)


def _decode_from_zero_width(encoded: str) -> Optional[str]:
    """Decode zero-width unicode characters back to original string."""
    try:
        # Split by byte delimiter
        bytes_data: list[int] = []
        current_bits: list[str] = []
        
        for char in encoded:
            if char == ZERO_WIDTH_JOINER:
                if len(current_bits) == 8:
                    byte_val = int("".join(current_bits), 2)
                    bytes_data.append(byte_val)
                current_bits = []
            elif char == ZERO_WIDTH_SPACE:
                current_bits.append("0")
            elif char == ZERO_WIDTH_NON_JOINER:
                current_bits.append("1")
        
        if not bytes_data:
            return None
            
        return bytes(bytes_data).decode("utf-8")
    except Exception:
        return None


def embed_canary(
    content: str,
    token: CanaryToken,
    strategy: EmbeddingStrategy = EmbeddingStrategy.INVISIBLE,
    position: str = "end",  # "start", "end", or "distributed"
) -> str:
    """Embed a canary token into content.
    
    Args:
        content: The content to embed the token in
        token: The canary token to embed
        strategy: How to embed (visible or invisible)
        position: Where to embed ("start", "end", or "distributed")
        
    Returns:
        Content with embedded canary token
    """
    marker = token.to_marker()
    
    if strategy == EmbeddingStrategy.VISIBLE:
        # Human-readable marker (useful for debugging or explicit tracking)
        visible_marker = f"[CANARY:{marker}]"
        if position == "start":
            return f"{visible_marker}\n{content}"
        elif position == "distributed":
            # Split marker across content at natural break points
            lines = content.split("\n")
            if len(lines) >= 3:
                mid = len(lines) // 2
                lines.insert(mid, f"<!-- {visible_marker} -->")
                return "\n".join(lines)
            return f"{content}\n{visible_marker}"
        else:  # end
            return f"{content}\n{visible_marker}"
    
    else:  # INVISIBLE
        # Zero-width unicode encoding
        invisible_marker = _encode_to_zero_width(f"CANARY:{marker}")
        
        if position == "start":
            return f"{invisible_marker}{content}"
        elif position == "distributed":
            # Distribute across word boundaries
            words = content.split(" ")
            if len(words) >= 4:
                # Insert marker chunks between words
                marker_parts = [invisible_marker[i:i+20] for i in range(0, len(invisible_marker), 20)]
                step = max(1, len(words) // (len(marker_parts) + 1))
                for i, part in enumerate(marker_parts):
                    insert_pos = min((i + 1) * step, len(words) - 1)
                    words[insert_pos] = part + words[insert_pos]
                return " ".join(words)
            return f"{content}{invisible_marker}"
        else:  # end
            return f"{content}{invisible_marker}"


def detect_canaries(content: str, secret_key: Optional[bytes] = None) -> List[CanaryToken]:
    """Detect canary tokens in content.
    
    Args:
        content: Content to scan for canary tokens
        secret_key: Optional secret key to verify detected tokens
        
    Returns:
        List of detected CanaryToken objects
    """
    detected: List[CanaryToken] = []
    
    # Track seen token IDs to avoid duplicates
    seen_token_ids: Set[str] = set()
    
    # Pattern 1: HTML comment markers <!-- [CANARY:...] --> (check first to avoid duplication)
    comment_pattern = r"<!--\s*\[CANARY:(canary_[a-f0-9_]+:[a-f0-9]+)\]\s*-->"
    for match in re.finditer(comment_pattern, content):
        marker = match.group(1)
        parts = marker.split(":")
        if len(parts) == 2:
            token_id = parts[0]
            if token_id in seen_token_ids:
                continue
            if secret_key:
                token = CanaryToken.from_marker(marker, secret_key)
                if token:
                    detected.append(token)
                    seen_token_ids.add(token_id)
            else:
                detected.append(CanaryToken(
                    token_id=token_id,
                    signature=parts[1],
                ))
                seen_token_ids.add(token_id)
    
    # Pattern 2: Visible markers [CANARY:token_id:signature] (not in HTML comments)
    visible_pattern = r"(?<!<!--\s)\[CANARY:(canary_[a-f0-9_]+:[a-f0-9]+)\](?!\s*-->)"
    for match in re.finditer(visible_pattern, content):
        marker = match.group(1)
        parts = marker.split(":")
        if len(parts) == 2:
            token_id = parts[0]
            if token_id in seen_token_ids:
                continue
            if secret_key:
                token = CanaryToken.from_marker(marker, secret_key)
                if token:
                    detected.append(token)
                    seen_token_ids.add(token_id)
            else:
                detected.append(CanaryToken(
                    token_id=token_id,
                    signature=parts[1],
                ))
                seen_token_ids.add(token_id)
    
    # Pattern 3: Invisible zero-width markers
    # Extract all zero-width character sequences
    zero_width_pattern = f"[{ZERO_WIDTH_SPACE}{ZERO_WIDTH_NON_JOINER}{ZERO_WIDTH_JOINER}]+"
    for match in re.finditer(zero_width_pattern, content):
        encoded = match.group(0)
        decoded = _decode_from_zero_width(encoded)
        if decoded and decoded.startswith("CANARY:"):
            marker = decoded[7:]  # Remove "CANARY:" prefix
            if secret_key:
                token = CanaryToken.from_marker(marker, secret_key)
                if token:
                    detected.append(token)
            else:
                parts = marker.split(":")
                if len(parts) == 2:
                    detected.append(CanaryToken(
                        token_id=parts[0],
                        signature=parts[1],
                    ))
    
    return detected


def strip_canaries(content: str) -> str:
    """Remove all canary tokens from content.
    
    Args:
        content: Content to strip canaries from
        
    Returns:
        Content with all canary markers removed
    """
    # Remove HTML comment markers first (more specific)
    content = re.sub(r"<!--\s*\[CANARY:canary_[a-f0-9_]+:[a-f0-9]+\]\s*-->\n?", "", content)
    
    # Remove visible markers
    content = re.sub(r"\[CANARY:canary_[a-f0-9_]+:[a-f0-9]+\]\n?", "", content)
    
    # Remove invisible markers
    zero_width_pattern = f"[{ZERO_WIDTH_SPACE}{ZERO_WIDTH_NON_JOINER}{ZERO_WIDTH_JOINER}]+"
    content = re.sub(zero_width_pattern, "", content)
    
    return content


@dataclass
class LeakReport:
    """Report of a detected canary token leak."""
    
    token: CanaryToken
    detected_at: datetime
    source: str  # Where the leak was detected
    context: Optional[str] = None  # Surrounding content
    verified: bool = False  # Whether signature was verified


class CanaryRegistry:
    """Registry for tracking issued canary tokens and detecting leaks.
    
    Maintains a record of all issued tokens and provides methods
    for detecting and reporting leaks.
    """
    
    def __init__(self, secret_key: Optional[bytes] = None):
        """Initialize the registry.
        
        Args:
            secret_key: Secret key for generating and verifying tokens.
                       If not provided, a random key is generated.
        """
        self._secret_key = secret_key or secrets.token_bytes(32)
        self._tokens: Dict[str, CanaryToken] = {}
        self._leaks: List[LeakReport] = []
    
    @property
    def secret_key(self) -> bytes:
        """Get the secret key (for external storage/loading)."""
        return self._secret_key
    
    def issue_token(
        self,
        recipient_id: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> CanaryToken:
        """Issue a new canary token and register it.
        
        Args:
            recipient_id: Identifier for who will receive the content
            content: Optional content to associate with this token
            metadata: Optional additional metadata
            
        Returns:
            A new registered CanaryToken
        """
        token = CanaryToken.generate(
            secret_key=self._secret_key,
            recipient_id=recipient_id,
            content=content,
            metadata=metadata,
        )
        self._tokens[token.token_id] = token
        return token
    
    def get_token(self, token_id: str) -> Optional[CanaryToken]:
        """Retrieve a registered token by ID."""
        return self._tokens.get(token_id)
    
    def list_tokens(
        self,
        recipient_id: Optional[str] = None,
    ) -> List[CanaryToken]:
        """List registered tokens, optionally filtered by recipient."""
        tokens = list(self._tokens.values())
        if recipient_id:
            tokens = [t for t in tokens if t.recipient_id == recipient_id]
        return tokens
    
    def scan_for_leaks(
        self,
        content: str,
        source: str,
        context_chars: int = 100,
    ) -> List[LeakReport]:
        """Scan content for leaked canary tokens.
        
        Args:
            content: Content to scan
            source: Description of where this content was found
            context_chars: Characters of context to include in report
            
        Returns:
            List of LeakReport for any detected tokens
        """
        detected = detect_canaries(content, self._secret_key)
        reports = []
        
        for token in detected:
            # Check if this is one of our registered tokens
            registered = self._tokens.get(token.token_id)
            
            # Extract context around the detection
            # For simplicity, just use first 100 chars of content
            context = content[:context_chars] if len(content) > context_chars else content
            
            report = LeakReport(
                token=registered or token,
                detected_at=datetime.now(timezone.utc),
                source=source,
                context=context,
                verified=registered is not None,
            )
            reports.append(report)
            self._leaks.append(report)
        
        return reports
    
    def get_leaks(
        self,
        token_id: Optional[str] = None,
        recipient_id: Optional[str] = None,
    ) -> List[LeakReport]:
        """Get recorded leak reports, optionally filtered."""
        leaks = self._leaks
        
        if token_id:
            leaks = [l for l in leaks if l.token.token_id == token_id]
        
        if recipient_id:
            leaks = [l for l in leaks if l.token.recipient_id == recipient_id]
        
        return leaks
    
    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token (remove from registry).
        
        Note: This doesn't remove the token from content already shared,
        but it will no longer be recognized as a valid registered token.
        
        Returns:
            True if token was found and revoked, False otherwise
        """
        if token_id in self._tokens:
            del self._tokens[token_id]
            return True
        return False
    
    def export_state(self) -> Dict:
        """Export registry state for persistence."""
        return {
            "tokens": {
                tid: {
                    "token_id": t.token_id,
                    "signature": t.signature,
                    "created_at": t.created_at.isoformat(),
                    "recipient_id": t.recipient_id,
                    "content_hash": t.content_hash,
                    "metadata": t.metadata,
                }
                for tid, t in self._tokens.items()
            },
            "leaks": [
                {
                    "token_id": l.token.token_id,
                    "detected_at": l.detected_at.isoformat(),
                    "source": l.source,
                    "context": l.context,
                    "verified": l.verified,
                }
                for l in self._leaks
            ],
        }
    
    @classmethod
    def from_state(cls, state: Dict, secret_key: bytes) -> "CanaryRegistry":
        """Restore registry from exported state."""
        registry = cls(secret_key=secret_key)
        
        for token_data in state.get("tokens", {}).values():
            token = CanaryToken(
                token_id=token_data["token_id"],
                signature=token_data["signature"],
                created_at=datetime.fromisoformat(token_data["created_at"]),
                recipient_id=token_data.get("recipient_id"),
                content_hash=token_data.get("content_hash"),
                metadata=token_data.get("metadata", {}),
            )
            registry._tokens[token.token_id] = token
        
        for leak_data in state.get("leaks", []):
            token_or_none = registry._tokens.get(leak_data["token_id"])
            token: CanaryToken
            if token_or_none is None:
                token = CanaryToken(
                    token_id=leak_data["token_id"],
                    signature="",
                )
            else:
                token = token_or_none
            report = LeakReport(
                token=token,
                detected_at=datetime.fromisoformat(leak_data["detected_at"]),
                source=leak_data["source"],
                context=leak_data.get("context"),
                verified=leak_data.get("verified", False),
            )
            registry._leaks.append(report)
        
        return registry
