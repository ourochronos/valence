"""Tests for DID:vkb Identity System.

Covers:
- Base58 and multibase encoding/decoding
- KeyPair generation and restoration
- DID parsing and creation (web, key, user)
- DID Document creation and serialization
- DID resolution (sync for key DIDs)
- Signing and verification
"""

import base64
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from valence.federation.identity import (
    # Constants
    DID_METHOD,
    DID_PREFIX,
    MULTIBASE_BASE58BTC,
    MULTICODEC_ED25519_PUB,
    WELL_KNOWN_NODE_METADATA,
    WELL_KNOWN_TRUST_ANCHORS,
    CRYPTO_AVAILABLE,
    # Enums
    DIDMethod,
    # Base58
    base58_encode,
    base58_decode,
    multibase_encode,
    multibase_decode,
    # Keys
    KeyPair,
    generate_keypair,
    public_key_from_multibase,
    # DID
    DID,
    parse_did,
    create_web_did,
    create_key_did,
    create_user_did,
    # DID Document
    VerificationMethod,
    ServiceEndpoint,
    DIDDocument,
    create_did_document,
    # Resolution
    resolve_did,
    resolve_did_sync,
    _resolve_key_did,
    _resolve_web_did,
    # Signing
    sign_message,
    verify_signature,
    canonical_json,
    sign_belief_content,
    verify_belief_signature,
)


# =============================================================================
# TEST CONSTANTS
# =============================================================================


class TestConstants:
    """Test module constants."""

    def test_did_method(self):
        """DID method should be 'vkb'."""
        assert DID_METHOD == "vkb"

    def test_did_prefix(self):
        """DID prefix should be 'did:vkb:'."""
        assert DID_PREFIX == "did:vkb:"

    def test_multibase_prefix(self):
        """Multibase prefix should be 'z' for base58btc."""
        assert MULTIBASE_BASE58BTC == "z"

    def test_multicodec_prefix(self):
        """Multicodec prefix should be 0xed01 for Ed25519."""
        assert MULTICODEC_ED25519_PUB == bytes([0xed, 0x01])

    def test_well_known_paths(self):
        """Well-known paths should be correct."""
        assert WELL_KNOWN_NODE_METADATA == "/.well-known/vfp-node-metadata"
        assert WELL_KNOWN_TRUST_ANCHORS == "/.well-known/vfp-trust-anchors"


# =============================================================================
# TEST ENUMS
# =============================================================================


class TestDIDMethod:
    """Test DIDMethod enum."""

    def test_web_value(self):
        """WEB should have value 'web'."""
        assert DIDMethod.WEB.value == "web"

    def test_key_value(self):
        """KEY should have value 'key'."""
        assert DIDMethod.KEY.value == "key"

    def test_user_value(self):
        """USER should have value 'user'."""
        assert DIDMethod.USER.value == "user"

    def test_enum_from_string(self):
        """Should create enum from string value."""
        assert DIDMethod("web") == DIDMethod.WEB
        assert DIDMethod("key") == DIDMethod.KEY
        assert DIDMethod("user") == DIDMethod.USER

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with pytest.raises(ValueError):
            DIDMethod("invalid")


# =============================================================================
# TEST BASE58 ENCODING
# =============================================================================


class TestBase58:
    """Test base58 encoding/decoding."""

    def test_encode_empty(self):
        """Empty bytes should encode to '1'."""
        result = base58_encode(b"")
        assert result == "1"

    def test_encode_zero(self):
        """Zero byte should encode to '1'."""
        result = base58_encode(b"\x00")
        assert result == "1"

    def test_encode_multiple_zeros(self):
        """Multiple zero bytes should encode to multiple '1's."""
        result = base58_encode(b"\x00\x00\x00")
        assert result == "111"

    def test_encode_simple(self):
        """Simple bytes should encode correctly."""
        # 'a' = 0x61 = 97 in base10
        result = base58_encode(b"a")
        assert result == "2g"

    def test_encode_hello(self):
        """'Hello' should encode to known value."""
        result = base58_encode(b"Hello")
        assert result == "9Ajdvzr"

    def test_decode_empty_result(self):
        """'1' should decode to zero byte."""
        result = base58_decode("1")
        assert result == b"\x00"

    def test_decode_multiple_ones(self):
        """Multiple '1's should decode to multiple zero bytes."""
        result = base58_decode("111")
        assert result == b"\x00\x00\x00"

    def test_decode_simple(self):
        """Simple string should decode correctly."""
        result = base58_decode("2g")
        assert result == b"a"

    def test_decode_hello(self):
        """Known value should decode to 'Hello'."""
        result = base58_decode("9Ajdvzr")
        assert result == b"Hello"

    def test_roundtrip(self):
        """Encode then decode should return original."""
        original = b"Test data for roundtrip"
        encoded = base58_encode(original)
        decoded = base58_decode(encoded)
        assert decoded == original

    def test_roundtrip_with_zeros(self):
        """Roundtrip should preserve leading zeros."""
        original = b"\x00\x00Test"
        encoded = base58_encode(original)
        decoded = base58_decode(encoded)
        assert decoded == original

    def test_decode_invalid_char(self):
        """Invalid character should raise error."""
        with pytest.raises(ValueError):
            base58_decode("0")  # '0' is not in base58 alphabet

    def test_decode_invalid_char_O(self):
        """'O' is not in base58 alphabet."""
        with pytest.raises(ValueError):
            base58_decode("O")


class TestMultibase:
    """Test multibase encoding/decoding."""

    def test_encode_adds_prefix(self):
        """Multibase encode should add 'z' prefix."""
        result = multibase_encode(b"test")
        assert result.startswith("z")

    def test_encode_is_base58(self):
        """After prefix, should be valid base58."""
        result = multibase_encode(b"test")
        # Remove prefix and decode
        decoded = base58_decode(result[1:])
        assert decoded == b"test"

    def test_decode_removes_prefix(self):
        """Multibase decode should strip prefix."""
        # Known encoding
        encoded = multibase_encode(b"test")
        decoded = multibase_decode(encoded)
        assert decoded == b"test"

    def test_decode_invalid_prefix_raises(self):
        """Non-z prefix should raise error."""
        with pytest.raises(ValueError, match="Unsupported multibase encoding"):
            multibase_decode("a123456")

    def test_decode_m_prefix_raises(self):
        """'m' prefix (base64) should raise error."""
        with pytest.raises(ValueError, match="Unsupported multibase encoding"):
            multibase_decode("mYWJj")

    def test_roundtrip(self):
        """Multibase roundtrip should work."""
        original = b"roundtrip test data"
        encoded = multibase_encode(original)
        decoded = multibase_decode(encoded)
        assert decoded == original


# =============================================================================
# TEST KEY PAIR
# =============================================================================


class TestKeyPair:
    """Test KeyPair class."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_generate_keypair(self):
        """Should generate valid keypair."""
        kp = generate_keypair()
        assert len(kp.private_key_bytes) == 32
        assert len(kp.public_key_bytes) == 32

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_public_key_multibase(self):
        """Public key multibase should have correct format."""
        kp = generate_keypair()
        mb = kp.public_key_multibase
        # Should start with 'z'
        assert mb.startswith("z")
        # Should decode to multicodec prefix + 32 bytes
        decoded = multibase_decode(mb)
        assert decoded[:2] == MULTICODEC_ED25519_PUB
        assert len(decoded) == 34  # 2 prefix + 32 key

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_private_key_hex(self):
        """Private key hex should be valid hex string."""
        kp = generate_keypair()
        hex_str = kp.private_key_hex
        # Should be 64 hex characters (32 bytes)
        assert len(hex_str) == 64
        # Should be valid hex
        assert bytes.fromhex(hex_str) == kp.private_key_bytes

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_from_private_key_hex(self):
        """Should restore keypair from hex."""
        kp1 = generate_keypair()
        hex_str = kp1.private_key_hex

        kp2 = KeyPair.from_private_key_hex(hex_str)
        assert kp2.private_key_bytes == kp1.private_key_bytes
        assert kp2.public_key_bytes == kp1.public_key_bytes

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_keypairs_unique(self):
        """Each generated keypair should be unique."""
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        assert kp1.private_key_bytes != kp2.private_key_bytes
        assert kp1.public_key_bytes != kp2.public_key_bytes


class TestPublicKeyFromMultibase:
    """Test public_key_from_multibase function."""

    def test_extracts_key_bytes(self):
        """Should extract raw key bytes from multibase."""
        # Create a known multibase-encoded key
        raw_key = b"a" * 32  # 32 bytes of 'a'
        prefixed = MULTICODEC_ED25519_PUB + raw_key
        mb = multibase_encode(prefixed)

        result = public_key_from_multibase(mb)
        assert result == raw_key

    def test_without_multicodec_prefix(self):
        """Should return decoded bytes if no multicodec prefix."""
        raw_data = b"x" * 32
        mb = multibase_encode(raw_data)

        result = public_key_from_multibase(mb)
        assert result == raw_data


# =============================================================================
# TEST DID CLASS
# =============================================================================


class TestDID:
    """Test DID dataclass."""

    def test_full_web_did(self):
        """Web DID full string should be correct."""
        did = DID(method=DIDMethod.WEB, identifier="example.com")
        assert did.full == "did:vkb:web:example.com"

    def test_full_key_did(self):
        """Key DID full string should be correct."""
        did = DID(method=DIDMethod.KEY, identifier="z6MkTest")
        assert did.full == "did:vkb:key:z6MkTest"

    def test_full_user_did(self):
        """User DID full string should be correct."""
        did = DID(
            method=DIDMethod.USER,
            identifier="web:example.com:alice",
            node_method=DIDMethod.WEB,
            node_identifier="example.com",
            username="alice",
        )
        assert did.full == "did:vkb:user:web:example.com:alice"

    def test_node_did_for_user(self):
        """User DID should return node_did."""
        did = DID(
            method=DIDMethod.USER,
            identifier="web:example.com:alice",
            node_method=DIDMethod.WEB,
            node_identifier="example.com",
            username="alice",
        )
        assert did.node_did == "did:vkb:web:example.com"

    def test_node_did_for_web(self):
        """Web DID should return None for node_did."""
        did = DID(method=DIDMethod.WEB, identifier="example.com")
        assert did.node_did is None

    def test_str(self):
        """str() should return full DID."""
        did = DID(method=DIDMethod.WEB, identifier="example.com")
        assert str(did) == "did:vkb:web:example.com"

    def test_eq_with_did(self):
        """Should equal another DID with same full string."""
        did1 = DID(method=DIDMethod.WEB, identifier="example.com")
        did2 = DID(method=DIDMethod.WEB, identifier="example.com")
        assert did1 == did2

    def test_eq_with_string(self):
        """Should equal string representation."""
        did = DID(method=DIDMethod.WEB, identifier="example.com")
        assert did == "did:vkb:web:example.com"

    def test_eq_not_equal_different_identifier(self):
        """Should not equal DID with different identifier."""
        did1 = DID(method=DIDMethod.WEB, identifier="example.com")
        did2 = DID(method=DIDMethod.WEB, identifier="other.com")
        assert did1 != did2

    def test_eq_not_equal_other_type(self):
        """Should not equal non-DID, non-string."""
        did = DID(method=DIDMethod.WEB, identifier="example.com")
        assert did != 123
        assert did != None

    def test_hash(self):
        """Should be hashable."""
        did1 = DID(method=DIDMethod.WEB, identifier="example.com")
        did2 = DID(method=DIDMethod.WEB, identifier="example.com")
        assert hash(did1) == hash(did2)
        # Should work in sets
        did_set = {did1, did2}
        assert len(did_set) == 1


# =============================================================================
# TEST DID PARSING
# =============================================================================


class TestParseDID:
    """Test parse_did function."""

    def test_parse_web_did(self):
        """Should parse web DID."""
        did = parse_did("did:vkb:web:example.com")
        assert did.method == DIDMethod.WEB
        assert did.identifier == "example.com"

    def test_parse_web_did_subdomain(self):
        """Should parse web DID with subdomain."""
        did = parse_did("did:vkb:web:valence.example.com")
        assert did.method == DIDMethod.WEB
        assert did.identifier == "valence.example.com"

    def test_parse_key_did(self):
        """Should parse key DID."""
        did = parse_did("did:vkb:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK")
        assert did.method == DIDMethod.KEY
        assert did.identifier == "z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK"

    def test_parse_user_did(self):
        """Should parse user DID."""
        did = parse_did("did:vkb:user:web:example.com:alice")
        assert did.method == DIDMethod.USER
        assert did.node_method == DIDMethod.WEB
        assert did.node_identifier == "example.com"
        assert did.username == "alice"

    def test_parse_user_did_key_based_node(self):
        """Should parse user DID with key-based node."""
        did = parse_did("did:vkb:user:key:z6MkTest:bob")
        assert did.method == DIDMethod.USER
        assert did.node_method == DIDMethod.KEY
        assert did.node_identifier == "z6MkTest"
        assert did.username == "bob"

    def test_parse_user_did_username_with_colons(self):
        """Should handle username with colons."""
        did = parse_did("did:vkb:user:web:example.com:alice:smith")
        assert did.username == "alice:smith"

    def test_invalid_prefix_raises(self):
        """Should raise for non-did:vkb: prefix."""
        with pytest.raises(ValueError, match="must start with"):
            parse_did("did:web:example.com")

    def test_missing_method_raises(self):
        """Should raise for missing method."""
        with pytest.raises(ValueError, match="missing method"):
            parse_did("did:vkb:")

    def test_invalid_method_raises(self):
        """Should raise for invalid method."""
        with pytest.raises(ValueError, match="Invalid DID method"):
            parse_did("did:vkb:invalid:test")

    def test_user_did_missing_parts_raises(self):
        """Should raise for incomplete user DID."""
        with pytest.raises(ValueError, match="must have node-method"):
            parse_did("did:vkb:user:web")

    def test_user_did_invalid_node_method_raises(self):
        """Should raise for invalid node method in user DID."""
        with pytest.raises(ValueError, match="Invalid node method"):
            parse_did("did:vkb:user:invalid:example.com:alice")

    def test_invalid_domain_raises(self):
        """Should raise for invalid domain format."""
        with pytest.raises(ValueError, match="Invalid domain"):
            parse_did("did:vkb:web:-invalid.com")

    def test_key_did_invalid_encoding_raises(self):
        """Should raise for non-base58btc key."""
        with pytest.raises(ValueError, match="must use base58btc"):
            parse_did("did:vkb:key:abc123")

    def test_key_did_bad_multibase_raises(self):
        """Should raise for invalid multibase content."""
        with pytest.raises(ValueError, match="Invalid multibase"):
            parse_did("did:vkb:key:z0OI")  # Contains invalid chars


# =============================================================================
# TEST DID CREATION
# =============================================================================


class TestCreateWebDID:
    """Test create_web_did function."""

    def test_creates_web_did(self):
        """Should create web DID from domain."""
        did = create_web_did("example.com")
        assert did.method == DIDMethod.WEB
        assert did.identifier == "example.com"
        assert did.full == "did:vkb:web:example.com"

    def test_normalizes_case(self):
        """Should lowercase domain."""
        did = create_web_did("Example.COM")
        assert did.identifier == "example.com"

    def test_strips_trailing_dot(self):
        """Should strip trailing dot from domain."""
        did = create_web_did("example.com.")
        assert did.identifier == "example.com"

    def test_invalid_domain_raises(self):
        """Should raise for invalid domain."""
        with pytest.raises(ValueError, match="Invalid domain"):
            create_web_did("invalid..domain")

    def test_domain_starting_with_hyphen_raises(self):
        """Should raise for domain starting with hyphen."""
        with pytest.raises(ValueError, match="Invalid domain"):
            create_web_did("-example.com")


class TestCreateKeyDID:
    """Test create_key_did function."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_creates_key_did(self):
        """Should create key DID from multibase key."""
        kp = generate_keypair()
        did = create_key_did(kp.public_key_multibase)
        assert did.method == DIDMethod.KEY
        assert did.identifier == kp.public_key_multibase

    def test_invalid_prefix_raises(self):
        """Should raise for non-z prefix."""
        with pytest.raises(ValueError, match="must be in multibase base58btc"):
            create_key_did("abc123")

    def test_invalid_multibase_raises(self):
        """Should raise for invalid multibase content."""
        with pytest.raises(ValueError, match="Invalid multibase"):
            create_key_did("zBad0OI")  # Invalid base58 chars

    def test_short_key_raises(self):
        """Should raise for too-short key."""
        # Encode just a few bytes
        short = multibase_encode(b"short")
        with pytest.raises(ValueError, match="too short"):
            create_key_did(short)


class TestCreateUserDID:
    """Test create_user_did function."""

    def test_creates_user_did_from_string(self):
        """Should create user DID from node DID string."""
        did = create_user_did("did:vkb:web:example.com", "alice")
        assert did.method == DIDMethod.USER
        assert did.node_method == DIDMethod.WEB
        assert did.node_identifier == "example.com"
        assert did.username == "alice"
        assert did.full == "did:vkb:user:web:example.com:alice"

    def test_creates_user_did_from_did_object(self):
        """Should create user DID from DID object."""
        node = create_web_did("example.com")
        did = create_user_did(node, "bob")
        assert did.method == DIDMethod.USER
        assert did.username == "bob"

    def test_user_under_user_raises(self):
        """Should raise when creating user under another user."""
        user_did = "did:vkb:user:web:example.com:alice"
        with pytest.raises(ValueError, match="Cannot create user DID under another user"):
            create_user_did(user_did, "bob")

    def test_invalid_username_raises(self):
        """Should raise for invalid username characters."""
        with pytest.raises(ValueError, match="Invalid username"):
            create_user_did("did:vkb:web:example.com", "alice@bob")

    def test_username_with_spaces_raises(self):
        """Should raise for username with spaces."""
        with pytest.raises(ValueError, match="Invalid username"):
            create_user_did("did:vkb:web:example.com", "alice bob")

    def test_valid_username_with_underscore(self):
        """Should accept underscore in username."""
        did = create_user_did("did:vkb:web:example.com", "alice_bob")
        assert did.username == "alice_bob"

    def test_valid_username_with_hyphen(self):
        """Should accept hyphen in username."""
        did = create_user_did("did:vkb:web:example.com", "alice-bob")
        assert did.username == "alice-bob"


# =============================================================================
# TEST VERIFICATION METHOD
# =============================================================================


class TestVerificationMethod:
    """Test VerificationMethod dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        vm = VerificationMethod(
            id="did:vkb:web:example.com#keys-1",
            type="Ed25519VerificationKey2020",
            controller="did:vkb:web:example.com",
            public_key_multibase="z6MkTest",
        )
        d = vm.to_dict()
        assert d["id"] == "did:vkb:web:example.com#keys-1"
        assert d["type"] == "Ed25519VerificationKey2020"
        assert d["controller"] == "did:vkb:web:example.com"
        assert d["publicKeyMultibase"] == "z6MkTest"


class TestServiceEndpoint:
    """Test ServiceEndpoint dataclass."""

    def test_to_dict(self):
        """Should convert to dictionary."""
        se = ServiceEndpoint(
            id="did:vkb:web:example.com#vfp",
            type="ValenceFederationProtocol",
            service_endpoint="https://example.com/vfp",
        )
        d = se.to_dict()
        assert d["id"] == "did:vkb:web:example.com#vfp"
        assert d["type"] == "ValenceFederationProtocol"
        assert d["serviceEndpoint"] == "https://example.com/vfp"


# =============================================================================
# TEST DID DOCUMENT
# =============================================================================


class TestDIDDocument:
    """Test DIDDocument dataclass."""

    def test_did_property(self):
        """Should parse id to DID."""
        doc = DIDDocument(id="did:vkb:web:example.com")
        assert doc.did.method == DIDMethod.WEB
        assert doc.did.identifier == "example.com"

    def test_primary_verification_method_exists(self):
        """Should return first verification method."""
        vm = VerificationMethod(id="test", public_key_multibase="z6MkTest")
        doc = DIDDocument(id="did:vkb:web:example.com", verification_methods=[vm])
        assert doc.primary_verification_method == vm

    def test_primary_verification_method_none(self):
        """Should return None if no verification methods."""
        doc = DIDDocument(id="did:vkb:web:example.com")
        assert doc.primary_verification_method is None

    def test_public_key_multibase(self):
        """Should return primary key's multibase."""
        vm = VerificationMethod(id="test", public_key_multibase="z6MkTest")
        doc = DIDDocument(id="did:vkb:web:example.com", verification_methods=[vm])
        assert doc.public_key_multibase == "z6MkTest"

    def test_public_key_multibase_none(self):
        """Should return None if no verification method."""
        doc = DIDDocument(id="did:vkb:web:example.com")
        assert doc.public_key_multibase is None

    def test_to_dict_minimal(self):
        """Should convert minimal doc to dict."""
        doc = DIDDocument(id="did:vkb:web:example.com")
        d = doc.to_dict()
        assert "@context" in d
        assert d["id"] == "did:vkb:web:example.com"
        assert d["vfp:protocolVersion"] == "1.0"

    def test_to_dict_full(self):
        """Should convert full doc to dict."""
        vm = VerificationMethod(
            id="did:vkb:web:example.com#keys-1",
            controller="did:vkb:web:example.com",
            public_key_multibase="z6MkTest",
        )
        svc = ServiceEndpoint(
            id="did:vkb:web:example.com#vfp",
            type="ValenceFederationProtocol",
            service_endpoint="https://example.com/vfp",
        )
        now = datetime(2024, 1, 1, 12, 0, 0)
        doc = DIDDocument(
            id="did:vkb:web:example.com",
            controller="did:vkb:key:z6MkController",
            verification_methods=[vm],
            authentication=["did:vkb:web:example.com#keys-1"],
            assertion_method=["did:vkb:web:example.com#keys-1"],
            services=[svc],
            capabilities=["belief_sync", "aggregation_participate"],
            profile={"name": "Test Node"},
            protocol_version="1.0",
            created=now,
            updated=now,
        )
        d = doc.to_dict()
        assert d["controller"] == "did:vkb:key:z6MkController"
        assert len(d["verificationMethod"]) == 1
        assert d["authentication"] == ["did:vkb:web:example.com#keys-1"]
        assert d["assertionMethod"] == ["did:vkb:web:example.com#keys-1"]
        assert len(d["service"]) == 1
        assert d["vfp:capabilities"] == ["belief_sync", "aggregation_participate"]
        assert d["vfp:profile"]["name"] == "Test Node"
        assert d["created"] == "2024-01-01T12:00:00"
        assert d["updated"] == "2024-01-01T12:00:00"

    def test_to_json(self):
        """Should convert to JSON string."""
        doc = DIDDocument(id="did:vkb:web:example.com")
        j = doc.to_json()
        parsed = json.loads(j)
        assert parsed["id"] == "did:vkb:web:example.com"

    def test_from_dict(self):
        """Should create from dictionary."""
        data = {
            "id": "did:vkb:web:example.com",
            "controller": "did:vkb:key:z6MkController",
            "verificationMethod": [
                {
                    "id": "did:vkb:web:example.com#keys-1",
                    "type": "Ed25519VerificationKey2020",
                    "controller": "did:vkb:web:example.com",
                    "publicKeyMultibase": "z6MkTest",
                }
            ],
            "authentication": ["did:vkb:web:example.com#keys-1"],
            "assertionMethod": ["did:vkb:web:example.com#keys-1"],
            "service": [
                {
                    "id": "did:vkb:web:example.com#vfp",
                    "type": "ValenceFederationProtocol",
                    "serviceEndpoint": "https://example.com/vfp",
                }
            ],
            "vfp:capabilities": ["belief_sync"],
            "vfp:profile": {"name": "Test"},
            "vfp:protocolVersion": "1.0",
            "created": "2024-01-01T12:00:00",
            "updated": "2024-01-01T12:00:00",
        }
        doc = DIDDocument.from_dict(data)
        assert doc.id == "did:vkb:web:example.com"
        assert doc.controller == "did:vkb:key:z6MkController"
        assert len(doc.verification_methods) == 1
        assert doc.verification_methods[0].public_key_multibase == "z6MkTest"
        assert len(doc.services) == 1
        assert doc.capabilities == ["belief_sync"]
        assert doc.profile["name"] == "Test"
        assert doc.created == datetime(2024, 1, 1, 12, 0, 0)

    def test_from_dict_minimal(self):
        """Should handle minimal dictionary."""
        data = {"id": "did:vkb:web:example.com"}
        doc = DIDDocument.from_dict(data)
        assert doc.id == "did:vkb:web:example.com"
        assert doc.verification_methods == []
        assert doc.services == []

    def test_roundtrip(self):
        """to_dict then from_dict should preserve data."""
        vm = VerificationMethod(
            id="did:vkb:web:example.com#keys-1",
            controller="did:vkb:web:example.com",
            public_key_multibase="z6MkTest",
        )
        original = DIDDocument(
            id="did:vkb:web:example.com",
            verification_methods=[vm],
            capabilities=["belief_sync"],
        )
        d = original.to_dict()
        restored = DIDDocument.from_dict(d)
        assert restored.id == original.id
        assert restored.capabilities == original.capabilities
        assert restored.verification_methods[0].id == vm.id


# =============================================================================
# TEST CREATE DID DOCUMENT
# =============================================================================


class TestCreateDIDDocument:
    """Test create_did_document function."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_creates_document(self):
        """Should create DID document with all fields."""
        kp = generate_keypair()
        doc = create_did_document(
            did="did:vkb:web:example.com",
            public_key_multibase=kp.public_key_multibase,
            federation_endpoint="https://example.com/vfp",
            mcp_endpoint="https://example.com/mcp",
            capabilities=["belief_sync", "aggregation_participate"],
            name="Test Node",
            domains=["testing", "development"],
        )
        assert doc.id == "did:vkb:web:example.com"
        assert len(doc.verification_methods) == 1
        assert doc.verification_methods[0].public_key_multibase == kp.public_key_multibase
        assert doc.verification_methods[0].controller == "did:vkb:web:example.com"
        assert len(doc.services) == 2
        assert doc.capabilities == ["belief_sync", "aggregation_participate"]
        assert doc.profile["name"] == "Test Node"
        assert doc.profile["domains"] == ["testing", "development"]
        assert doc.created is not None
        assert doc.updated is not None

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_creates_with_did_object(self):
        """Should accept DID object."""
        kp = generate_keypair()
        did_obj = create_web_did("example.com")
        doc = create_did_document(
            did=did_obj,
            public_key_multibase=kp.public_key_multibase,
        )
        assert doc.id == "did:vkb:web:example.com"

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_minimal_document(self):
        """Should create minimal document with defaults."""
        kp = generate_keypair()
        doc = create_did_document(
            did="did:vkb:web:example.com",
            public_key_multibase=kp.public_key_multibase,
        )
        assert doc.services == []
        assert doc.capabilities == ["belief_sync"]  # default
        assert doc.profile == {}


# =============================================================================
# TEST DID RESOLUTION
# =============================================================================


class TestResolveKeyDID:
    """Test _resolve_key_did function."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_resolves_key_did(self):
        """Should resolve key DID to document."""
        kp = generate_keypair()
        did = create_key_did(kp.public_key_multibase)
        doc = _resolve_key_did(did)

        assert doc.id == did.full
        assert len(doc.verification_methods) == 1
        assert doc.verification_methods[0].public_key_multibase == kp.public_key_multibase
        assert doc.capabilities == ["belief_sync"]


class TestResolveDIDSync:
    """Test resolve_did_sync function."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_resolves_key_did(self):
        """Should resolve key DID synchronously."""
        kp = generate_keypair()
        did_str = f"did:vkb:key:{kp.public_key_multibase}"
        doc = resolve_did_sync(did_str)

        assert doc is not None
        assert doc.id == did_str

    def test_returns_none_for_web_did(self):
        """Should return None for web DID (needs async)."""
        doc = resolve_did_sync("did:vkb:web:example.com")
        assert doc is None

    def test_returns_none_for_user_did(self):
        """Should return None for user DID (needs async)."""
        doc = resolve_did_sync("did:vkb:user:web:example.com:alice")
        assert doc is None


@pytest.mark.asyncio
class TestResolveDIDAsync:
    """Test async resolve_did function."""

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    async def test_resolves_key_did(self):
        """Should resolve key DID asynchronously."""
        kp = generate_keypair()
        did_str = f"did:vkb:key:{kp.public_key_multibase}"
        doc = await resolve_did(did_str)

        assert doc is not None
        assert doc.id == did_str

    async def test_user_did_resolves_to_node(self):
        """User DID should resolve to node's document."""
        # Mock _resolve_web_did to return a document
        mock_doc = DIDDocument(id="did:vkb:web:example.com")
        with patch("valence.federation.identity._resolve_web_did", return_value=mock_doc) as mock:
            doc = await resolve_did("did:vkb:user:web:example.com:alice")
            mock.assert_called_once()
            assert doc == mock_doc


@pytest.mark.asyncio
class TestResolveWebDID:
    """Test _resolve_web_did function."""

    async def test_fetches_from_well_known(self):
        """Should fetch from well-known endpoint."""
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={"id": "did:vkb:web:example.com"}
        )

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            did = parse_did("did:vkb:web:example.com")
            doc = await _resolve_web_did(did)

            assert doc is not None
            assert doc.id == "did:vkb:web:example.com"

    async def test_returns_none_on_404(self):
        """Should return None on 404."""
        mock_response = MagicMock()
        mock_response.status = 404

        mock_session = MagicMock()
        mock_session.get = MagicMock(
            return_value=MagicMock(__aenter__=AsyncMock(return_value=mock_response), __aexit__=AsyncMock())
        )

        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_client.return_value.__aexit__ = AsyncMock()

            did = parse_did("did:vkb:web:example.com")
            doc = await _resolve_web_did(did)

            assert doc is None

    async def test_returns_none_on_exception(self):
        """Should return None on network error."""
        with patch("aiohttp.ClientSession") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(side_effect=Exception("Network error"))

            did = parse_did("did:vkb:web:example.com")
            doc = await _resolve_web_did(did)

            assert doc is None


# =============================================================================
# TEST SIGNING AND VERIFICATION
# =============================================================================


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
class TestSigning:
    """Test signing functions."""

    def test_sign_message(self):
        """Should sign message with private key."""
        kp = generate_keypair()
        message = b"Hello, World!"
        signature = sign_message(message, kp.private_key_bytes)

        assert len(signature) == 64  # Ed25519 signatures are 64 bytes

    def test_verify_signature_valid(self):
        """Should verify valid signature."""
        kp = generate_keypair()
        message = b"Hello, World!"
        signature = sign_message(message, kp.private_key_bytes)

        assert verify_signature(message, signature, kp.public_key_multibase)

    def test_verify_signature_invalid_message(self):
        """Should reject signature for wrong message."""
        kp = generate_keypair()
        message = b"Hello, World!"
        signature = sign_message(message, kp.private_key_bytes)

        assert not verify_signature(b"Different message", signature, kp.public_key_multibase)

    def test_verify_signature_invalid_key(self):
        """Should reject signature for wrong key."""
        kp1 = generate_keypair()
        kp2 = generate_keypair()
        message = b"Hello, World!"
        signature = sign_message(message, kp1.private_key_bytes)

        assert not verify_signature(message, signature, kp2.public_key_multibase)

    def test_verify_signature_invalid_signature(self):
        """Should reject corrupted signature."""
        kp = generate_keypair()
        message = b"Hello, World!"

        bad_signature = b"\x00" * 64
        assert not verify_signature(message, bad_signature, kp.public_key_multibase)


class TestCanonicalJSON:
    """Test canonical_json function."""

    def test_sorts_keys(self):
        """Should sort keys lexicographically."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(obj)
        assert result == b'{"a":2,"m":3,"z":1}'

    def test_no_whitespace(self):
        """Should have no whitespace."""
        obj = {"key": "value"}
        result = canonical_json(obj)
        assert b" " not in result
        assert b"\n" not in result

    def test_nested_objects(self):
        """Should sort nested object keys."""
        obj = {"b": {"z": 1, "a": 2}, "a": 1}
        result = canonical_json(obj)
        assert result == b'{"a":1,"b":{"a":2,"z":1}}'

    def test_unicode(self):
        """Should handle unicode correctly."""
        obj = {"key": "日本語"}
        result = canonical_json(obj)
        assert "日本語" in result.decode("utf-8")

    def test_returns_bytes(self):
        """Should return UTF-8 encoded bytes."""
        obj = {"key": "value"}
        result = canonical_json(obj)
        assert isinstance(result, bytes)


@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
class TestBeliefSigning:
    """Test belief-specific signing functions."""

    def test_sign_belief_content(self):
        """Should sign belief content and return base64."""
        kp = generate_keypair()
        content = {"statement": "The sky is blue", "confidence": 0.9}
        signature = sign_belief_content(content, kp.private_key_bytes)

        # Should be valid base64
        decoded = base64.b64decode(signature)
        assert len(decoded) == 64  # Ed25519 signature

    def test_verify_belief_signature_valid(self):
        """Should verify valid belief signature."""
        kp = generate_keypair()
        content = {"statement": "The sky is blue", "confidence": 0.9}
        signature = sign_belief_content(content, kp.private_key_bytes)

        assert verify_belief_signature(content, signature, kp.public_key_multibase)

    def test_verify_belief_signature_invalid_content(self):
        """Should reject signature for modified content."""
        kp = generate_keypair()
        content = {"statement": "The sky is blue", "confidence": 0.9}
        signature = sign_belief_content(content, kp.private_key_bytes)

        modified = {"statement": "The sky is red", "confidence": 0.9}
        assert not verify_belief_signature(modified, signature, kp.public_key_multibase)

    def test_verify_belief_signature_key_order_irrelevant(self):
        """Signature should be valid regardless of key order in verification."""
        kp = generate_keypair()
        # Sign with one key order
        content1 = {"statement": "Test", "confidence": 0.9}
        signature = sign_belief_content(content1, kp.private_key_bytes)

        # Verify with different key order (should still work due to canonicalization)
        content2 = {"confidence": 0.9, "statement": "Test"}
        assert verify_belief_signature(content2, signature, kp.public_key_multibase)

    def test_verify_belief_signature_invalid_base64(self):
        """Should return False for invalid base64 signature."""
        kp = generate_keypair()
        content = {"statement": "Test"}

        assert not verify_belief_signature(content, "not-valid-base64!!!", kp.public_key_multibase)

    def test_verify_belief_signature_invalid_key(self):
        """Should return False for invalid public key."""
        kp = generate_keypair()
        content = {"statement": "Test"}
        signature = sign_belief_content(content, kp.private_key_bytes)

        assert not verify_belief_signature(content, signature, "zinvalid-key")


# =============================================================================
# TEST CRYPTO NOT AVAILABLE
# =============================================================================


class TestCryptoNotAvailable:
    """Test behavior when cryptography library is not available."""

    def test_sign_message_raises(self):
        """sign_message should raise without cryptography."""
        with patch("valence.federation.identity.CRYPTO_AVAILABLE", False):
            with pytest.raises(NotImplementedError, match="requires cryptography"):
                sign_message(b"test", b"\x00" * 32)

    def test_verify_signature_raises(self):
        """verify_signature should raise without cryptography."""
        with patch("valence.federation.identity.CRYPTO_AVAILABLE", False):
            with pytest.raises(NotImplementedError, match="requires cryptography"):
                verify_signature(b"test", b"\x00" * 64, "z6MkTest")

    def test_keypair_from_hex_raises(self):
        """KeyPair.from_private_key_hex should raise without cryptography."""
        with patch("valence.federation.identity.CRYPTO_AVAILABLE", False):
            with pytest.raises(NotImplementedError, match="requires cryptography"):
                KeyPair.from_private_key_hex("00" * 32)
