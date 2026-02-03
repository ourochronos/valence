"""DID:vkb Identity System for Valence Federation.

Implements the did:vkb DID method for node and user identity.

DID Formats:
- Node (web): did:vkb:web:<domain>
- Node (key): did:vkb:key:<multibase-encoded-public-key>
- User: did:vkb:user:<node-method>:<node-id>:<username>

Examples:
- did:vkb:web:valence.example.com
- did:vkb:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
- did:vkb:user:web:valence.example.com:alice
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

# Try to import cryptography for Ed25519, fall back to pure Python if needed
try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
    from cryptography.hazmat.primitives import serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


# =============================================================================
# CONSTANTS
# =============================================================================

DID_METHOD = "vkb"
DID_PREFIX = f"did:{DID_METHOD}:"

# Multibase prefix for base58btc encoding
MULTIBASE_BASE58BTC = "z"

# Multicodec prefix for Ed25519 public key (0xed01)
MULTICODEC_ED25519_PUB = bytes([0xed, 0x01])

# Well-known endpoint paths
WELL_KNOWN_NODE_METADATA = "/.well-known/vfp-node-metadata"
WELL_KNOWN_TRUST_ANCHORS = "/.well-known/vfp-trust-anchors"


# =============================================================================
# ENUMS
# =============================================================================


class DIDMethod(str, Enum):
    """DID method variants for did:vkb."""
    WEB = "web"      # Domain-verified
    KEY = "key"      # Self-sovereign (key-based)
    USER = "user"    # User identity


# =============================================================================
# BASE58 ENCODING (simplified, no external dependency)
# =============================================================================

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def base58_encode(data: bytes) -> str:
    """Encode bytes to base58 string."""
    num = int.from_bytes(data, "big")
    result = ""
    while num > 0:
        num, remainder = divmod(num, 58)
        result = BASE58_ALPHABET[remainder] + result

    # Handle leading zeros
    for byte in data:
        if byte == 0:
            result = BASE58_ALPHABET[0] + result
        else:
            break

    return result or BASE58_ALPHABET[0]


def base58_decode(string: str) -> bytes:
    """Decode base58 string to bytes."""
    num = 0
    for char in string:
        num = num * 58 + BASE58_ALPHABET.index(char)

    # Convert to bytes
    result = []
    while num > 0:
        num, remainder = divmod(num, 256)
        result.insert(0, remainder)

    # Handle leading zeros
    for char in string:
        if char == BASE58_ALPHABET[0]:
            result.insert(0, 0)
        else:
            break

    return bytes(result)


def multibase_encode(data: bytes) -> str:
    """Encode bytes to multibase (base58btc)."""
    return MULTIBASE_BASE58BTC + base58_encode(data)


def multibase_decode(string: str) -> bytes:
    """Decode multibase string to bytes."""
    if not string.startswith(MULTIBASE_BASE58BTC):
        raise ValueError(f"Unsupported multibase encoding: {string[0]}")
    return base58_decode(string[1:])


# =============================================================================
# KEY GENERATION
# =============================================================================


@dataclass
class KeyPair:
    """Ed25519 key pair for node identity."""
    private_key_bytes: bytes
    public_key_bytes: bytes

    @property
    def public_key_multibase(self) -> str:
        """Get public key in multibase format (with multicodec prefix)."""
        # Prepend multicodec prefix for Ed25519 public key
        prefixed = MULTICODEC_ED25519_PUB + self.public_key_bytes
        return multibase_encode(prefixed)

    @property
    def private_key_hex(self) -> str:
        """Get private key as hex string (for secure storage)."""
        return self.private_key_bytes.hex()

    @classmethod
    def from_private_key_hex(cls, hex_string: str) -> KeyPair:
        """Create KeyPair from stored private key hex."""
        private_bytes = bytes.fromhex(hex_string)
        if CRYPTO_AVAILABLE:
            private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
            public_bytes = private_key.public_key().public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
        else:
            # Fallback: assume 32-byte seed, derive public key manually
            # This is a simplified version - real implementation needs full Ed25519
            raise NotImplementedError("Ed25519 key derivation requires cryptography library")

        return cls(
            private_key_bytes=private_bytes,
            public_key_bytes=public_bytes,
        )


def generate_keypair() -> KeyPair:
    """Generate a new Ed25519 key pair."""
    if CRYPTO_AVAILABLE:
        private_key = Ed25519PrivateKey.generate()
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    else:
        # Fallback: generate random bytes (won't work for signing)
        private_bytes = secrets.token_bytes(32)
        public_bytes = secrets.token_bytes(32)
        # In production, cryptography library should be required

    return KeyPair(
        private_key_bytes=private_bytes,
        public_key_bytes=public_bytes,
    )


def public_key_from_multibase(multibase: str) -> bytes:
    """Extract raw public key bytes from multibase-encoded key."""
    decoded = multibase_decode(multibase)
    # Remove multicodec prefix
    if decoded[:2] == MULTICODEC_ED25519_PUB:
        return decoded[2:]
    return decoded


# =============================================================================
# DID
# =============================================================================


@dataclass
class DID:
    """Parsed Decentralized Identifier."""

    method: DIDMethod
    identifier: str

    # For user DIDs
    node_method: DIDMethod | None = None
    node_identifier: str | None = None
    username: str | None = None

    @property
    def full(self) -> str:
        """Get full DID string."""
        if self.method == DIDMethod.USER:
            return f"did:vkb:user:{self.node_method.value}:{self.node_identifier}:{self.username}"
        return f"did:vkb:{self.method.value}:{self.identifier}"

    @property
    def node_did(self) -> str | None:
        """Get the node DID for a user DID."""
        if self.method == DIDMethod.USER:
            return f"did:vkb:{self.node_method.value}:{self.node_identifier}"
        return None

    def __str__(self) -> str:
        return self.full

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DID):
            return self.full == other.full
        if isinstance(other, str):
            return self.full == other
        return False

    def __hash__(self) -> int:
        return hash(self.full)


def parse_did(did_string: str) -> DID:
    """Parse a DID string into a DID object.

    Raises ValueError if the DID is invalid.
    """
    if not did_string.startswith(DID_PREFIX):
        raise ValueError(f"Invalid DID: must start with '{DID_PREFIX}'")

    parts = did_string[len(DID_PREFIX):].split(":")

    if len(parts) < 2:
        raise ValueError(f"Invalid DID: missing method or identifier")

    method_str = parts[0]

    try:
        method = DIDMethod(method_str)
    except ValueError:
        raise ValueError(f"Invalid DID method: {method_str}")

    if method == DIDMethod.USER:
        # User DID: did:vkb:user:<node-method>:<node-id>:<username>
        if len(parts) < 4:
            raise ValueError("Invalid user DID: must have node-method, node-id, and username")

        try:
            node_method = DIDMethod(parts[1])
        except ValueError:
            raise ValueError(f"Invalid node method in user DID: {parts[1]}")

        return DID(
            method=DIDMethod.USER,
            identifier=":".join(parts[1:]),  # Full identifier includes all parts
            node_method=node_method,
            node_identifier=parts[2],
            username=":".join(parts[3:]),  # Username might contain colons
        )

    # Node DID: did:vkb:web:<domain> or did:vkb:key:<key>
    identifier = ":".join(parts[1:])

    if method == DIDMethod.WEB:
        # Validate domain format
        if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$", identifier):
            raise ValueError(f"Invalid domain in web DID: {identifier}")

    elif method == DIDMethod.KEY:
        # Validate multibase format (should start with z for base58btc)
        if not identifier.startswith("z"):
            raise ValueError(f"Invalid key DID: must use base58btc encoding (start with 'z')")
        try:
            multibase_decode(identifier)
        except Exception as e:
            raise ValueError(f"Invalid multibase encoding in key DID: {e}")

    return DID(
        method=method,
        identifier=identifier,
    )


def create_web_did(domain: str) -> DID:
    """Create a web-based DID for a domain.

    Example: create_web_did("valence.example.com")
    Returns: DID for did:vkb:web:valence.example.com
    """
    # Normalize domain (lowercase, no trailing dots)
    domain = domain.lower().rstrip(".")

    # Validate domain
    if not re.match(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)*$", domain):
        raise ValueError(f"Invalid domain: {domain}")

    return DID(
        method=DIDMethod.WEB,
        identifier=domain,
    )


def create_key_did(public_key_multibase: str) -> DID:
    """Create a key-based DID from a multibase-encoded public key.

    Example: create_key_did("z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK")
    Returns: DID for did:vkb:key:z6Mk...
    """
    # Validate multibase format
    if not public_key_multibase.startswith("z"):
        raise ValueError("Public key must be in multibase base58btc format (start with 'z')")

    try:
        decoded = multibase_decode(public_key_multibase)
        # Should have multicodec prefix + 32-byte key
        if len(decoded) < 34:  # 2 bytes prefix + 32 bytes key
            raise ValueError("Public key too short")
    except Exception as e:
        raise ValueError(f"Invalid multibase encoding: {e}")

    return DID(
        method=DIDMethod.KEY,
        identifier=public_key_multibase,
    )


def create_user_did(node_did: str | DID, username: str) -> DID:
    """Create a user DID under a node.

    Example: create_user_did("did:vkb:web:valence.example.com", "alice")
    Returns: DID for did:vkb:user:web:valence.example.com:alice
    """
    if isinstance(node_did, str):
        node = parse_did(node_did)
    else:
        node = node_did

    if node.method == DIDMethod.USER:
        raise ValueError("Cannot create user DID under another user DID")

    # Validate username (alphanumeric, hyphens, underscores)
    if not re.match(r"^[a-zA-Z0-9_-]+$", username):
        raise ValueError(f"Invalid username: {username}")

    return DID(
        method=DIDMethod.USER,
        identifier=f"{node.method.value}:{node.identifier}:{username}",
        node_method=node.method,
        node_identifier=node.identifier,
        username=username,
    )


# =============================================================================
# DID DOCUMENT
# =============================================================================


@dataclass
class VerificationMethod:
    """Verification method in a DID document."""

    id: str
    type: str = "Ed25519VerificationKey2020"
    controller: str = ""
    public_key_multibase: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "controller": self.controller,
            "publicKeyMultibase": self.public_key_multibase,
        }


@dataclass
class ServiceEndpoint:
    """Service endpoint in a DID document."""

    id: str
    type: str
    service_endpoint: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "serviceEndpoint": self.service_endpoint,
        }


@dataclass
class DIDDocument:
    """DID Document for a Valence Federation node."""

    id: str  # The DID
    controller: str | None = None

    # Verification
    verification_methods: list[VerificationMethod] = field(default_factory=list)
    authentication: list[str] = field(default_factory=list)
    assertion_method: list[str] = field(default_factory=list)

    # Services
    services: list[ServiceEndpoint] = field(default_factory=list)

    # VFP-specific
    capabilities: list[str] = field(default_factory=list)
    profile: dict[str, Any] = field(default_factory=dict)
    protocol_version: str = "1.0"

    # Metadata
    created: datetime | None = None
    updated: datetime | None = None

    @property
    def did(self) -> DID:
        """Parse and return the DID."""
        return parse_did(self.id)

    @property
    def primary_verification_method(self) -> VerificationMethod | None:
        """Get the primary verification method."""
        if self.verification_methods:
            return self.verification_methods[0]
        return None

    @property
    def public_key_multibase(self) -> str | None:
        """Get the primary public key in multibase format."""
        vm = self.primary_verification_method
        return vm.public_key_multibase if vm else None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (JSON-LD format)."""
        doc = {
            "@context": [
                "https://www.w3.org/ns/did/v1",
                "https://valence.dev/ns/vfp/v1",
            ],
            "id": self.id,
        }

        if self.controller:
            doc["controller"] = self.controller

        if self.verification_methods:
            doc["verificationMethod"] = [vm.to_dict() for vm in self.verification_methods]

        if self.authentication:
            doc["authentication"] = self.authentication

        if self.assertion_method:
            doc["assertionMethod"] = self.assertion_method

        if self.services:
            doc["service"] = [s.to_dict() for s in self.services]

        if self.capabilities:
            doc["vfp:capabilities"] = self.capabilities

        if self.profile:
            doc["vfp:profile"] = self.profile

        doc["vfp:protocolVersion"] = self.protocol_version

        if self.created:
            doc["created"] = self.created.isoformat()
        if self.updated:
            doc["updated"] = self.updated.isoformat()

        return doc

    def to_json(self, indent: int | None = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DIDDocument:
        """Create from dictionary."""
        verification_methods = []
        for vm in data.get("verificationMethod", []):
            verification_methods.append(VerificationMethod(
                id=vm["id"],
                type=vm.get("type", "Ed25519VerificationKey2020"),
                controller=vm.get("controller", ""),
                public_key_multibase=vm.get("publicKeyMultibase", ""),
            ))

        services = []
        for s in data.get("service", []):
            services.append(ServiceEndpoint(
                id=s["id"],
                type=s["type"],
                service_endpoint=s["serviceEndpoint"],
            ))

        return cls(
            id=data["id"],
            controller=data.get("controller"),
            verification_methods=verification_methods,
            authentication=data.get("authentication", []),
            assertion_method=data.get("assertionMethod", []),
            services=services,
            capabilities=data.get("vfp:capabilities", []),
            profile=data.get("vfp:profile", {}),
            protocol_version=data.get("vfp:protocolVersion", "1.0"),
            created=datetime.fromisoformat(data["created"]) if data.get("created") else None,
            updated=datetime.fromisoformat(data["updated"]) if data.get("updated") else None,
        )


def create_did_document(
    did: str | DID,
    public_key_multibase: str,
    federation_endpoint: str | None = None,
    mcp_endpoint: str | None = None,
    capabilities: list[str] | None = None,
    name: str | None = None,
    domains: list[str] | None = None,
) -> DIDDocument:
    """Create a DID Document for a node.

    Args:
        did: The node's DID (string or DID object)
        public_key_multibase: Ed25519 public key in multibase format
        federation_endpoint: URL for federation protocol
        mcp_endpoint: URL for MCP protocol
        capabilities: List of capabilities (belief_sync, aggregation_participate, etc.)
        name: Human-readable node name
        domains: Knowledge domains this node specializes in
    """
    if isinstance(did, DID):
        did_str = did.full
    else:
        did_str = did
        did = parse_did(did_str)

    now = datetime.now()

    # Create verification method
    vm_id = f"{did_str}#keys-1"
    vm = VerificationMethod(
        id=vm_id,
        type="Ed25519VerificationKey2020",
        controller=did_str,
        public_key_multibase=public_key_multibase,
    )

    # Create services
    services = []
    if federation_endpoint:
        services.append(ServiceEndpoint(
            id=f"{did_str}#vfp",
            type="ValenceFederationProtocol",
            service_endpoint=federation_endpoint,
        ))
    if mcp_endpoint:
        services.append(ServiceEndpoint(
            id=f"{did_str}#mcp",
            type="ModelContextProtocol",
            service_endpoint=mcp_endpoint,
        ))

    # Build profile
    profile = {}
    if name:
        profile["name"] = name
    if domains:
        profile["domains"] = domains

    return DIDDocument(
        id=did_str,
        verification_methods=[vm],
        authentication=[vm_id],
        assertion_method=[vm_id],
        services=services,
        capabilities=capabilities or ["belief_sync"],
        profile=profile,
        protocol_version="1.0",
        created=now,
        updated=now,
    )


# =============================================================================
# DID RESOLUTION
# =============================================================================


async def resolve_did(did_string: str) -> DIDDocument | None:
    """Resolve a DID to its DID Document.

    For web DIDs, fetches from the well-known endpoint.
    For key DIDs, generates a minimal document from the key.

    Note: This is an async function for network operations.
    """
    did = parse_did(did_string)

    if did.method == DIDMethod.KEY:
        # Key DIDs are self-describing
        return _resolve_key_did(did)

    elif did.method == DIDMethod.WEB:
        # Web DIDs require network fetch
        return await _resolve_web_did(did)

    elif did.method == DIDMethod.USER:
        # User DIDs resolve to their node's document
        if did.node_did:
            return await resolve_did(did.node_did)
        return None

    return None


def _resolve_key_did(did: DID) -> DIDDocument:
    """Resolve a key-based DID (no network required)."""
    # The key is embedded in the DID
    public_key_multibase = did.identifier

    vm_id = f"{did.full}#keys-1"
    vm = VerificationMethod(
        id=vm_id,
        type="Ed25519VerificationKey2020",
        controller=did.full,
        public_key_multibase=public_key_multibase,
    )

    return DIDDocument(
        id=did.full,
        verification_methods=[vm],
        authentication=[vm_id],
        assertion_method=[vm_id],
        capabilities=["belief_sync"],  # Minimal capabilities
        protocol_version="1.0",
    )


async def _resolve_web_did(did: DID) -> DIDDocument | None:
    """Resolve a web-based DID by fetching from well-known endpoint."""
    import aiohttp

    url = f"https://{did.identifier}{WELL_KNOWN_NODE_METADATA}"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    return DIDDocument.from_dict(data)
                else:
                    return None
    except Exception:
        return None


def resolve_did_sync(did_string: str) -> DIDDocument | None:
    """Synchronous version of resolve_did (for key DIDs only).

    For web DIDs, use the async version or resolve manually.
    """
    did = parse_did(did_string)

    if did.method == DIDMethod.KEY:
        return _resolve_key_did(did)

    # For other methods, return None (need async for network)
    return None


# =============================================================================
# SIGNING AND VERIFICATION
# =============================================================================


def sign_message(message: bytes, private_key_bytes: bytes) -> bytes:
    """Sign a message with Ed25519 private key."""
    if not CRYPTO_AVAILABLE:
        raise NotImplementedError("Signing requires cryptography library")

    private_key = Ed25519PrivateKey.from_private_bytes(private_key_bytes)
    return private_key.sign(message)


def verify_signature(
    message: bytes,
    signature: bytes,
    public_key_multibase: str,
) -> bool:
    """Verify an Ed25519 signature.

    Args:
        message: The original message bytes
        signature: The signature bytes
        public_key_multibase: Public key in multibase format

    Returns:
        True if signature is valid, False otherwise
    """
    if not CRYPTO_AVAILABLE:
        raise NotImplementedError("Verification requires cryptography library")

    try:
        public_key_bytes = public_key_from_multibase(public_key_multibase)
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
        public_key.verify(signature, message)
        return True
    except Exception:
        return False


def canonical_json(obj: Any) -> bytes:
    """Convert object to canonical JSON for signing.

    Uses RFC 8785 JSON Canonicalization Scheme:
    - Keys sorted lexicographically
    - No whitespace
    - UTF-8 encoding
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sign_belief_content(
    content: dict[str, Any],
    private_key_bytes: bytes,
) -> str:
    """Sign belief content and return base64-encoded signature."""
    message = canonical_json(content)
    signature = sign_message(message, private_key_bytes)
    return base64.b64encode(signature).decode("ascii")


def verify_belief_signature(
    content: dict[str, Any],
    signature_b64: str,
    public_key_multibase: str,
) -> bool:
    """Verify a belief signature.

    Args:
        content: The belief content dictionary
        signature_b64: Base64-encoded signature
        public_key_multibase: Signer's public key in multibase format

    Returns:
        True if signature is valid
    """
    try:
        message = canonical_json(content)
        signature = base64.b64decode(signature_b64)
        return verify_signature(message, signature, public_key_multibase)
    except Exception:
        return False
