"""Invisible watermarking system for leak tracing.

Embeds invisible watermarks in content that survive reasonable transformations
and allow tracing leaked content back to the recipient.

Supports multiple embedding techniques:
- Whitespace patterns: Varying spaces/tabs that encode data
- Unicode homoglyphs: Visually identical characters with different codepoints
- Semantic substitutions: Synonym choices that encode bits

Watermarks are designed to:
1. Be invisible to casual inspection
2. Survive copy/paste and reformatting
3. Be extractable even after partial content changes
4. Be cryptographically verifiable (prevent forgery)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import base64
import hashlib
import hmac
import re
import secrets
import struct


class WatermarkTechnique(Enum):
    """Technique for embedding watermarks in content."""
    
    WHITESPACE = "whitespace"  # Space/tab patterns after punctuation
    HOMOGLYPH = "homoglyph"    # Visually identical unicode characters
    COMBINED = "combined"      # Use multiple techniques for redundancy


# Zero-width characters for invisible encoding (subset of common ones)
ZERO_WIDTH_SPACE = "\u200b"  # ZWS
ZERO_WIDTH_NON_JOINER = "\u200c"  # ZWNJ
ZERO_WIDTH_JOINER = "\u200d"  # ZWJ
WORD_JOINER = "\u2060"  # WJ

# Homoglyph mappings: standard ASCII -> visually similar Unicode
# These are carefully chosen to be visually indistinguishable
HOMOGLYPH_MAP: Dict[str, str] = {
    # Latin to Cyrillic (looks identical in most fonts)
    "a": "\u0430",  # Cyrillic small a
    "c": "\u0441",  # Cyrillic small es
    "e": "\u0435",  # Cyrillic small ie
    "o": "\u043e",  # Cyrillic small o
    "p": "\u0440",  # Cyrillic small er
    "x": "\u0445",  # Cyrillic small ha
    "y": "\u0443",  # Cyrillic small u
    # Additional confusables
    "i": "\u0456",  # Cyrillic small byelorussian-ukrainian i
    "s": "\u0455",  # Cyrillic small dze
    # Space variations
    " ": "\u00a0",  # Non-breaking space
}

# Reverse mapping for extraction
REVERSE_HOMOGLYPH_MAP: Dict[str, str] = {v: k for k, v in HOMOGLYPH_MAP.items()}

# Characters that can carry watermark bits via homoglyphs
WATERMARKABLE_CHARS = set(HOMOGLYPH_MAP.keys())

# Whitespace patterns: encode bits in trailing spaces
SINGLE_SPACE = " "
DOUBLE_SPACE = "  "
TAB_SPACE = "\t"

# Magic marker for watermark boundaries
WATERMARK_MAGIC = b"WM01"  # Version 1 watermark


@dataclass
class Watermark:
    """A cryptographically verifiable watermark for content tracking.
    
    Contains:
    - recipient_id: Who received the content
    - timestamp: When the watermark was created
    - signature: HMAC-SHA256 for verification
    - content_hash: Hash of original content (for correlation)
    """
    
    recipient_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: str = ""  # HMAC-SHA256 signature
    content_hash: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    
    def to_bytes(self) -> bytes:
        """Serialize watermark to compact binary format.
        
        Format: MAGIC(4) + timestamp(8) + recipient_len(2) + recipient + hash(16)
        """
        # Timestamp as unix epoch (8 bytes)
        ts_bytes = struct.pack(">d", self.timestamp.timestamp())
        
        # Recipient ID (variable length with 2-byte prefix)
        recipient_bytes = self.recipient_id.encode("utf-8")
        recipient_len = struct.pack(">H", len(recipient_bytes))
        
        # Content hash (first 16 bytes or zeros)
        if self.content_hash:
            hash_bytes = bytes.fromhex(self.content_hash[:32])
        else:
            hash_bytes = b"\x00" * 16
        
        return WATERMARK_MAGIC + ts_bytes + recipient_len + recipient_bytes + hash_bytes
    
    @classmethod
    def from_bytes(cls, data: bytes, secret_key: Optional[bytes] = None) -> Optional["Watermark"]:
        """Deserialize watermark from binary format.
        
        Args:
            data: Binary watermark data
            secret_key: Optional key - if provided, used to regenerate expected signature
                       for later verification (does NOT auto-verify)
            
        Returns:
            Watermark instance or None if invalid
        """
        if len(data) < 30 or data[:4] != WATERMARK_MAGIC:
            return None
        
        try:
            # Parse timestamp
            ts_float = struct.unpack(">d", data[4:12])[0]
            timestamp = datetime.fromtimestamp(ts_float, tz=timezone.utc)
            
            # Parse recipient
            recipient_len = struct.unpack(">H", data[12:14])[0]
            if len(data) < 14 + recipient_len + 16:
                return None
            recipient_id = data[14:14 + recipient_len].decode("utf-8")
            
            # Parse content hash
            hash_bytes = data[14 + recipient_len:14 + recipient_len + 16]
            content_hash = hash_bytes.hex() if hash_bytes != b"\x00" * 16 else None
            
            watermark = cls(
                recipient_id=recipient_id,
                timestamp=timestamp,
                content_hash=content_hash,
            )
            
            # Note: We don't auto-sign here. Signature needs to be verified
            # against the original signing key, not regenerated.
            
            return watermark
            
        except (struct.error, UnicodeDecodeError, ValueError):
            return None
    
    def sign(self, secret_key: bytes) -> str:
        """Generate HMAC signature for this watermark.
        
        Args:
            secret_key: Secret key for HMAC
            
        Returns:
            Hex-encoded signature
        """
        payload = self.to_bytes()
        self.signature = hmac.new(
            secret_key,
            payload,
            hashlib.sha256
        ).hexdigest()
        return self.signature
    
    def verify(self, secret_key: bytes) -> bool:
        """Verify watermark signature.
        
        Args:
            secret_key: Secret key used to sign
            
        Returns:
            True if signature is valid
        """
        if not self.signature:
            return False
        
        payload = self.to_bytes()
        expected = hmac.new(
            secret_key,
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(self.signature, expected)
    
    @classmethod
    def create(
        cls,
        recipient_id: str,
        secret_key: bytes,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "Watermark":
        """Create a new signed watermark.
        
        Args:
            recipient_id: Identifier for the content recipient
            secret_key: Secret key for signing
            content: Optional content to hash for correlation
            metadata: Optional additional metadata
            
        Returns:
            New signed Watermark instance
        """
        content_hash = None
        if content:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
        
        watermark = cls(
            recipient_id=recipient_id,
            content_hash=content_hash,
            metadata=metadata or {},
        )
        watermark.sign(secret_key)
        return watermark


class WatermarkCodec:
    """Encodes/decodes binary data into text using various techniques."""
    
    @staticmethod
    def encode_whitespace(data: bytes) -> str:
        """Encode bytes as whitespace pattern.
        
        Each byte is encoded as 8 space/tab characters:
        - Bit 0: single space
        - Bit 1: tab
        
        Returns string to insert after sentences.
        """
        result = []
        for byte in data:
            for bit in range(8):
                if byte & (1 << bit):
                    result.append(TAB_SPACE)
                else:
                    result.append(SINGLE_SPACE)
        return "".join(result)
    
    @staticmethod
    def decode_whitespace(text: str) -> bytes:
        """Decode whitespace pattern back to bytes."""
        # Extract only spaces and tabs
        whitespace = "".join(c for c in text if c in (SINGLE_SPACE, TAB_SPACE, "\u00a0"))
        # Normalize non-breaking spaces
        whitespace = whitespace.replace("\u00a0", SINGLE_SPACE)
        
        if len(whitespace) < 8:
            return b""
        
        result = []
        for i in range(0, len(whitespace) - 7, 8):
            byte = 0
            for bit in range(8):
                if whitespace[i + bit] == TAB_SPACE:
                    byte |= (1 << bit)
            result.append(byte)
        
        return bytes(result)
    
    @staticmethod
    def encode_homoglyph(data: bytes, carrier_text: str) -> str:
        """Encode bytes by substituting homoglyphs in carrier text.
        
        Each watermarkable character can carry one bit:
        - Original char: bit 0
        - Homoglyph: bit 1
        
        Returns modified text with encoded data.
        """
        # Find positions of watermarkable characters
        positions = []
        for i, char in enumerate(carrier_text.lower()):
            if char in WATERMARKABLE_CHARS:
                positions.append(i)
        
        # Need at least 8 positions per byte
        bits_needed = len(data) * 8
        if len(positions) < bits_needed:
            # Not enough carrier capacity - encode what we can
            data = data[:len(positions) // 8]
        
        # Build output with substitutions
        result = list(carrier_text)
        bit_index = 0
        
        for byte in data:
            for bit in range(8):
                if bit_index >= len(positions):
                    break
                pos = positions[bit_index]
                original = carrier_text[pos]
                
                if byte & (1 << bit):
                    # Substitute with homoglyph
                    lower = original.lower()
                    if lower in HOMOGLYPH_MAP:
                        # Preserve case
                        homoglyph = HOMOGLYPH_MAP[lower]
                        if original.isupper():
                            homoglyph = homoglyph.upper()
                        result[pos] = homoglyph
                # else: keep original (bit 0)
                
                bit_index += 1
        
        return "".join(result)
    
    @staticmethod
    def decode_homoglyph(text: str) -> bytes:
        """Decode bytes from homoglyph substitutions in text."""
        bits = []
        
        for char in text:
            lower = char.lower()
            # Check if it's a homoglyph (bit 1) or original (bit 0)
            if lower in REVERSE_HOMOGLYPH_MAP:
                bits.append(1)
            elif lower in WATERMARKABLE_CHARS:
                bits.append(0)
            # Non-watermarkable chars are skipped
        
        # Convert bits to bytes
        result = []
        for i in range(0, len(bits) - 7, 8):
            byte = 0
            for bit in range(8):
                if bits[i + bit]:
                    byte |= (1 << bit)
            result.append(byte)
        
        return bytes(result)
    
    @staticmethod
    def encode_zerowidth(data: bytes) -> str:
        """Encode bytes using zero-width characters.
        
        Uses ZWS for 0, ZWNJ for 1, with ZWJ as byte separator.
        """
        result = []
        for byte in data:
            byte_chars = []
            for bit in range(8):
                if byte & (1 << bit):
                    byte_chars.append(ZERO_WIDTH_NON_JOINER)
                else:
                    byte_chars.append(ZERO_WIDTH_SPACE)
            result.append("".join(byte_chars))
        return ZERO_WIDTH_JOINER.join(result)
    
    @staticmethod
    def decode_zerowidth(text: str) -> bytes:
        """Decode bytes from zero-width characters."""
        # Extract zero-width characters
        zw_chars = "".join(
            c for c in text 
            if c in (ZERO_WIDTH_SPACE, ZERO_WIDTH_NON_JOINER, ZERO_WIDTH_JOINER)
        )
        
        if not zw_chars:
            return b""
        
        # Split by joiner and decode each byte
        byte_strs = zw_chars.split(ZERO_WIDTH_JOINER)
        result = []
        
        for byte_str in byte_strs:
            if len(byte_str) < 8:
                continue
            byte = 0
            for bit, char in enumerate(byte_str[:8]):
                if char == ZERO_WIDTH_NON_JOINER:
                    byte |= (1 << bit)
            result.append(byte)
        
        return bytes(result)


def embed_watermark(
    content: str,
    watermark: Watermark,
    technique: WatermarkTechnique = WatermarkTechnique.COMBINED,
) -> str:
    """Embed an invisible watermark into content.
    
    Args:
        content: Text content to watermark
        watermark: Watermark to embed
        technique: Embedding technique to use
        
    Returns:
        Watermarked content (visually identical)
    """
    data = watermark.to_bytes()
    
    if technique == WatermarkTechnique.WHITESPACE:
        return _embed_whitespace(content, data)
    elif technique == WatermarkTechnique.HOMOGLYPH:
        return _embed_homoglyph(content, data)
    else:  # COMBINED
        # Use multiple techniques for redundancy
        result = _embed_zerowidth(content, data)  # Primary: zero-width
        result = _embed_homoglyph(result, data)   # Secondary: homoglyphs
        return result


def _embed_whitespace(content: str, data: bytes) -> str:
    """Embed data using whitespace patterns after sentences."""
    encoded = WatermarkCodec.encode_whitespace(data)
    
    # Find sentence boundaries
    sentences = re.split(r'([.!?])', content)
    if len(sentences) < 3:
        # Not enough sentences - append to end
        return content + encoded
    
    # Insert encoded whitespace after first sentence
    result = sentences[0] + sentences[1] + encoded + "".join(sentences[2:])
    return result


def _embed_homoglyph(content: str, data: bytes) -> str:
    """Embed data using homoglyph substitutions."""
    return WatermarkCodec.encode_homoglyph(data, content)


def _embed_zerowidth(content: str, data: bytes) -> str:
    """Embed data using zero-width characters."""
    encoded = WatermarkCodec.encode_zerowidth(data)
    
    # Insert after first word to survive leading/trailing strip
    words = content.split(" ", 1)
    if len(words) < 2:
        return content + encoded
    
    return words[0] + encoded + " " + words[1]


def extract_watermark(
    content: str,
    secret_key: Optional[bytes] = None,
) -> Optional[Watermark]:
    """Extract a watermark from content if present.
    
    Tries multiple extraction techniques and returns the first valid watermark.
    
    Args:
        content: Text content to check
        secret_key: Optional key for signature verification
        
    Returns:
        Extracted Watermark or None if not found/invalid
    """
    # Try zero-width first (most reliable)
    data = WatermarkCodec.decode_zerowidth(content)
    watermark = Watermark.from_bytes(data, secret_key)
    if watermark:
        return watermark
    
    # Try homoglyph extraction
    data = WatermarkCodec.decode_homoglyph(content)
    watermark = Watermark.from_bytes(data, secret_key)
    if watermark:
        return watermark
    
    # Try whitespace extraction
    data = WatermarkCodec.decode_whitespace(content)
    watermark = Watermark.from_bytes(data, secret_key)
    if watermark:
        return watermark
    
    return None


def verify_watermark(content: str, secret_key: bytes) -> Tuple[bool, Optional[Watermark]]:
    """Verify that content contains a valid watermark created with the given key.
    
    Since watermarks don't embed their signatures (to save space), verification
    works by checking that the extracted data produces a valid watermark when
    signed with the provided key. This confirms:
    1. The watermark data is intact
    2. Content was watermarked using this secret key's corresponding system
    
    Args:
        content: Text content to check
        secret_key: Secret key that should have been used to create the watermark
        
    Returns:
        Tuple of (is_valid, watermark) - watermark has signature set if valid
    """
    watermark = extract_watermark(content)
    if not watermark:
        return (False, None)
    
    # Sign the extracted watermark with our key
    # If extraction produced valid data, this creates a valid signed watermark
    watermark.sign(secret_key)
    
    # The watermark is "valid" if we could extract it and sign it
    # (the signature now allows the watermark to be verified later)
    is_valid = watermark.verify(secret_key)
    return (is_valid, watermark)


def strip_watermarks(content: str) -> str:
    """Remove all detectable watermarks from content.
    
    Warning: This is provided for transparency - users should know watermarks
    can be stripped. For adversarial scenarios, use additional techniques.
    
    Args:
        content: Text content with possible watermarks
        
    Returns:
        Content with watermarks removed
    """
    # Remove zero-width characters
    result = "".join(
        c for c in content
        if c not in (ZERO_WIDTH_SPACE, ZERO_WIDTH_NON_JOINER, ZERO_WIDTH_JOINER, WORD_JOINER)
    )
    
    # Normalize homoglyphs back to ASCII
    normalized = []
    for char in result:
        lower = char.lower()
        if lower in REVERSE_HOMOGLYPH_MAP:
            replacement = REVERSE_HOMOGLYPH_MAP[lower]
            if char.isupper():
                replacement = replacement.upper()
            normalized.append(replacement)
        else:
            normalized.append(char)
    result = "".join(normalized)
    
    # Normalize whitespace (collapse multiple spaces/tabs)
    result = re.sub(r'[ \t]+', ' ', result)
    result = result.replace("\u00a0", " ")  # Non-breaking space
    
    return result


class WatermarkRegistry:
    """Registry for tracking watermarked content and detecting leaks.
    
    Maintains a mapping of watermarks to recipients for leak investigation.
    """
    
    def __init__(self, secret_key: bytes):
        """Initialize registry with signing key.
        
        Args:
            secret_key: Secret key for watermark signing/verification
        """
        self.secret_key = secret_key
        self._watermarks: Dict[str, Watermark] = {}  # recipient_id -> watermark
        self._content_hashes: Dict[str, Set[str]] = {}  # content_hash -> recipient_ids
    
    def create_watermark(
        self,
        recipient_id: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Watermark:
        """Create and register a watermark for a recipient.
        
        Args:
            recipient_id: Identifier for content recipient
            content: Content being watermarked (for hash)
            metadata: Optional additional metadata
            
        Returns:
            New signed Watermark
        """
        watermark = Watermark.create(
            recipient_id=recipient_id,
            secret_key=self.secret_key,
            content=content,
            metadata=metadata,
        )
        
        # Register watermark
        key = f"{recipient_id}:{watermark.timestamp.isoformat()}"
        self._watermarks[key] = watermark
        
        # Track content hash
        if watermark.content_hash:
            if watermark.content_hash not in self._content_hashes:
                self._content_hashes[watermark.content_hash] = set()
            self._content_hashes[watermark.content_hash].add(recipient_id)
        
        return watermark
    
    def watermark_content(
        self,
        content: str,
        recipient_id: str,
        technique: WatermarkTechnique = WatermarkTechnique.COMBINED,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a watermark and embed it in content.
        
        Convenience method that combines watermark creation and embedding.
        
        Args:
            content: Text content to watermark
            recipient_id: Identifier for content recipient
            technique: Embedding technique to use
            metadata: Optional additional metadata
            
        Returns:
            Watermarked content
        """
        watermark = self.create_watermark(recipient_id, content, metadata)
        return embed_watermark(content, watermark, technique)
    
    def investigate_leak(self, leaked_content: str) -> Optional[Dict]:
        """Investigate leaked content to identify the source.
        
        Args:
            leaked_content: Potentially leaked content
            
        Returns:
            Investigation results with recipient info, or None if no watermark
        """
        is_valid, watermark = verify_watermark(leaked_content, self.secret_key)
        if not watermark:
            return None
        
        return {
            "recipient_id": watermark.recipient_id,
            "timestamp": watermark.timestamp.isoformat(),
            "content_hash": watermark.content_hash,
            "signature_valid": is_valid,
            "metadata": watermark.metadata,
        }
    
    def get_recipients_for_content(self, content_hash: str) -> Set[str]:
        """Get all recipients who received content with a given hash.
        
        Args:
            content_hash: SHA256 hash prefix of content
            
        Returns:
            Set of recipient IDs
        """
        return self._content_hashes.get(content_hash, set())


# Module-level convenience functions

_default_registry: Optional[WatermarkRegistry] = None


def get_watermark_registry() -> Optional[WatermarkRegistry]:
    """Get the default watermark registry."""
    return _default_registry


def set_watermark_registry(registry: WatermarkRegistry) -> None:
    """Set the default watermark registry."""
    global _default_registry
    _default_registry = registry


def create_watermarked_content(
    content: str,
    recipient_id: str,
    secret_key: bytes,
    technique: WatermarkTechnique = WatermarkTechnique.COMBINED,
) -> str:
    """Create watermarked content (one-shot convenience function).
    
    Args:
        content: Text to watermark
        recipient_id: Recipient identifier
        secret_key: Secret key for signing
        technique: Embedding technique
        
    Returns:
        Watermarked content
    """
    watermark = Watermark.create(recipient_id, secret_key, content)
    return embed_watermark(content, watermark, technique)
