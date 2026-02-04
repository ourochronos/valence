"""Tests for Proxy Re-Encryption (PRE) abstraction layer.

These tests verify the PRE interface and mock implementation work correctly.
"""

import pytest
from datetime import datetime, timedelta

from valence.crypto.pre import (
    # Exceptions
    PREError,
    PREKeyError,
    PREEncryptionError,
    PREDecryptionError,
    PREReEncryptionError,
    PREInvalidCiphertextError,
    # Data classes
    PREPublicKey,
    PREPrivateKey,
    PREKeyPair,
    ReEncryptionKey,
    PRECiphertext,
    # Backend
    PREBackend,
    MockPREBackend,
    create_mock_backend,
)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestPREPublicKey:
    """Tests for PREPublicKey dataclass."""
    
    def test_create_public_key(self):
        """Test creating a public key."""
        key = PREPublicKey(
            key_id=b"alice",
            key_bytes=b"public-key-material",
        )
        
        assert key.key_id == b"alice"
        assert key.key_bytes == b"public-key-material"
        assert isinstance(key.created_at, datetime)
        assert key.metadata == {}
    
    def test_public_key_serialization(self):
        """Test public key to_dict/from_dict roundtrip."""
        key = PREPublicKey(
            key_id=b"bob",
            key_bytes=b"bob-public-key",
            metadata={"algorithm": "test"},
        )
        
        data = key.to_dict()
        restored = PREPublicKey.from_dict(data)
        
        assert restored.key_id == key.key_id
        assert restored.key_bytes == key.key_bytes
        assert restored.metadata == key.metadata
    
    def test_public_key_equality(self):
        """Test public key equality comparison."""
        key1 = PREPublicKey(key_id=b"alice", key_bytes=b"key")
        key2 = PREPublicKey(key_id=b"alice", key_bytes=b"key")
        key3 = PREPublicKey(key_id=b"bob", key_bytes=b"key")
        
        assert key1 == key2
        assert key1 != key3
    
    def test_public_key_hashable(self):
        """Test public keys can be used in sets/dicts."""
        key1 = PREPublicKey(key_id=b"alice", key_bytes=b"key")
        key2 = PREPublicKey(key_id=b"bob", key_bytes=b"key")
        
        key_set = {key1, key2}
        assert len(key_set) == 2


class TestPREPrivateKey:
    """Tests for PREPrivateKey dataclass."""
    
    def test_create_private_key(self):
        """Test creating a private key."""
        key = PREPrivateKey(
            key_id=b"alice",
            key_bytes=b"secret-key-material",
        )
        
        assert key.key_id == b"alice"
        assert key.key_bytes == b"secret-key-material"
    
    def test_private_key_serialization(self):
        """Test private key to_dict/from_dict roundtrip."""
        key = PREPrivateKey(
            key_id=b"carol",
            key_bytes=b"carol-secret",
            metadata={"version": "1"},
        )
        
        data = key.to_dict()
        restored = PREPrivateKey.from_dict(data)
        
        assert restored.key_id == key.key_id
        assert restored.key_bytes == key.key_bytes
        assert restored.metadata == key.metadata


class TestPREKeyPair:
    """Tests for PREKeyPair dataclass."""
    
    @pytest.fixture
    def keypair(self):
        """Create a test keypair."""
        return PREKeyPair(
            public_key=PREPublicKey(key_id=b"alice", key_bytes=b"pub"),
            private_key=PREPrivateKey(key_id=b"alice", key_bytes=b"priv"),
        )
    
    def test_keypair_key_id(self, keypair):
        """Test key_id property."""
        assert keypair.key_id == b"alice"
    
    def test_keypair_serialization(self, keypair):
        """Test keypair to_dict/from_dict roundtrip."""
        data = keypair.to_dict()
        restored = PREKeyPair.from_dict(data)
        
        assert restored.public_key.key_id == keypair.public_key.key_id
        assert restored.private_key.key_id == keypair.private_key.key_id


class TestReEncryptionKey:
    """Tests for ReEncryptionKey dataclass."""
    
    def test_create_rekey(self):
        """Test creating a re-encryption key."""
        rekey = ReEncryptionKey(
            rekey_id=b"rekey-123",
            delegator_id=b"alice",
            delegatee_id=b"bob",
            key_bytes=b"rekey-material",
        )
        
        assert rekey.delegator_id == b"alice"
        assert rekey.delegatee_id == b"bob"
        assert not rekey.is_expired
    
    def test_rekey_expiration(self):
        """Test re-encryption key expiration."""
        # Not expired
        future = datetime.now() + timedelta(hours=1)
        rekey = ReEncryptionKey(
            rekey_id=b"rekey-1",
            delegator_id=b"a",
            delegatee_id=b"b",
            key_bytes=b"k",
            expires_at=future,
        )
        assert not rekey.is_expired
        
        # Expired
        past = datetime.now() - timedelta(hours=1)
        expired_rekey = ReEncryptionKey(
            rekey_id=b"rekey-2",
            delegator_id=b"a",
            delegatee_id=b"b",
            key_bytes=b"k",
            expires_at=past,
        )
        assert expired_rekey.is_expired
    
    def test_rekey_serialization(self):
        """Test rekey to_dict/from_dict roundtrip."""
        rekey = ReEncryptionKey(
            rekey_id=b"rekey-test",
            delegator_id=b"alice",
            delegatee_id=b"bob",
            key_bytes=b"material",
            expires_at=datetime.now() + timedelta(days=1),
            metadata={"policy": "sharing"},
        )
        
        data = rekey.to_dict()
        restored = ReEncryptionKey.from_dict(data)
        
        assert restored.rekey_id == rekey.rekey_id
        assert restored.delegator_id == rekey.delegator_id
        assert restored.delegatee_id == rekey.delegatee_id
        assert restored.metadata == rekey.metadata


class TestPRECiphertext:
    """Tests for PRECiphertext dataclass."""
    
    def test_create_ciphertext(self):
        """Test creating a ciphertext."""
        ct = PRECiphertext(
            ciphertext_id=b"ct-123",
            encrypted_data=b"encrypted-stuff",
            recipient_id=b"alice",
        )
        
        assert ct.encrypted_data == b"encrypted-stuff"
        assert ct.recipient_id == b"alice"
        assert not ct.is_reencrypted
        assert ct.original_recipient_id is None
    
    def test_reencrypted_ciphertext(self):
        """Test ciphertext that has been re-encrypted."""
        ct = PRECiphertext(
            ciphertext_id=b"ct-456",
            encrypted_data=b"reencrypted-stuff",
            recipient_id=b"bob",
            is_reencrypted=True,
            original_recipient_id=b"alice",
        )
        
        assert ct.is_reencrypted
        assert ct.original_recipient_id == b"alice"
    
    def test_ciphertext_serialization(self):
        """Test ciphertext to_dict/from_dict roundtrip."""
        ct = PRECiphertext(
            ciphertext_id=b"ct-test",
            encrypted_data=b"data",
            recipient_id=b"carol",
            is_reencrypted=True,
            original_recipient_id=b"alice",
            metadata={"type": "belief"},
        )
        
        data = ct.to_dict()
        restored = PRECiphertext.from_dict(data)
        
        assert restored.ciphertext_id == ct.ciphertext_id
        assert restored.recipient_id == ct.recipient_id
        assert restored.is_reencrypted == ct.is_reencrypted
        assert restored.original_recipient_id == ct.original_recipient_id


# =============================================================================
# Mock Backend Tests
# =============================================================================


class TestMockPREBackend:
    """Tests for MockPREBackend implementation."""
    
    @pytest.fixture
    def backend(self):
        """Create a fresh mock backend."""
        return MockPREBackend()
    
    def test_generate_keypair(self, backend):
        """Test key pair generation."""
        keypair = backend.generate_keypair(b"alice")
        
        assert keypair.key_id == b"alice"
        assert len(keypair.public_key.key_bytes) == 32
        assert len(keypair.private_key.key_bytes) == 32
        assert keypair.public_key.key_id == keypair.private_key.key_id
    
    def test_generate_multiple_keypairs(self, backend):
        """Test generating multiple different keypairs."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        
        assert alice.public_key.key_bytes != bob.public_key.key_bytes
        assert alice.private_key.key_bytes != bob.private_key.key_bytes
    
    def test_generate_rekey(self, backend):
        """Test re-encryption key generation."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        
        rekey = backend.generate_rekey(
            delegator_private_key=alice.private_key,
            delegatee_public_key=bob.public_key,
        )
        
        assert rekey.delegator_id == b"alice"
        assert rekey.delegatee_id == b"bob"
        assert len(rekey.key_bytes) == 32
    
    def test_generate_rekey_with_expiration(self, backend):
        """Test re-encryption key with expiration."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        
        expires = datetime.now() + timedelta(hours=24)
        rekey = backend.generate_rekey(
            delegator_private_key=alice.private_key,
            delegatee_public_key=bob.public_key,
            expires_at=expires,
        )
        
        assert rekey.expires_at == expires
        assert not rekey.is_expired
    
    def test_encrypt_decrypt_roundtrip(self, backend):
        """Test basic encrypt/decrypt roundtrip."""
        alice = backend.generate_keypair(b"alice")
        plaintext = b"Hello, this is a secret message!"
        
        ciphertext = backend.encrypt(plaintext, alice.public_key)
        decrypted = backend.decrypt(ciphertext, alice.private_key)
        
        assert decrypted == plaintext
    
    def test_encrypt_creates_ciphertext(self, backend):
        """Test encryption produces proper ciphertext object."""
        alice = backend.generate_keypair(b"alice")
        plaintext = b"Secret data"
        
        ciphertext = backend.encrypt(
            plaintext,
            alice.public_key,
            metadata={"type": "belief"},
        )
        
        assert ciphertext.recipient_id == b"alice"
        assert not ciphertext.is_reencrypted
        assert ciphertext.metadata["type"] == "belief"
        assert len(ciphertext.ciphertext_id) == 16
    
    def test_decrypt_wrong_key_fails(self, backend):
        """Test decryption with wrong key fails."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        
        ciphertext = backend.encrypt(b"secret", alice.public_key)
        
        with pytest.raises(PREDecryptionError):
            backend.decrypt(ciphertext, bob.private_key)
    
    def test_re_encrypt_basic(self, backend):
        """Test basic re-encryption flow."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        plaintext = b"Shared secret for federation"
        
        # Alice encrypts
        ciphertext = backend.encrypt(plaintext, alice.public_key)
        
        # Alice generates rekey for Bob
        rekey = backend.generate_rekey(
            alice.private_key,
            bob.public_key,
        )
        
        # Proxy re-encrypts
        re_encrypted = backend.re_encrypt(ciphertext, rekey)
        
        # Verify re-encrypted properties
        assert re_encrypted.recipient_id == b"bob"
        assert re_encrypted.is_reencrypted
        assert re_encrypted.original_recipient_id == b"alice"
        
        # Bob decrypts
        decrypted = backend.decrypt(re_encrypted, bob.private_key)
        assert decrypted == plaintext
    
    def test_re_encrypt_chain(self, backend):
        """Test data can flow through multiple re-encryptions."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        carol = backend.generate_keypair(b"carol")
        plaintext = b"Multi-hop federation data"
        
        # Alice -> Bob -> Carol
        ct_alice = backend.encrypt(plaintext, alice.public_key)
        
        rekey_ab = backend.generate_rekey(alice.private_key, bob.public_key)
        ct_bob = backend.re_encrypt(ct_alice, rekey_ab)
        
        rekey_bc = backend.generate_rekey(bob.private_key, carol.public_key)
        ct_carol = backend.re_encrypt(ct_bob, rekey_bc)
        
        # Carol can decrypt
        decrypted = backend.decrypt(ct_carol, carol.private_key)
        assert decrypted == plaintext
    
    def test_re_encrypt_wrong_rekey_fails(self, backend):
        """Test re-encryption with mismatched rekey fails."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        carol = backend.generate_keypair(b"carol")
        
        # Encrypt for Alice
        ciphertext = backend.encrypt(b"secret", alice.public_key)
        
        # Create rekey Bob -> Carol (not Alice -> anyone)
        wrong_rekey = backend.generate_rekey(bob.private_key, carol.public_key)
        
        # Should fail - ciphertext is for Alice, rekey is from Bob
        with pytest.raises(PREReEncryptionError):
            backend.re_encrypt(ciphertext, wrong_rekey)
    
    def test_re_encrypt_expired_rekey_fails(self, backend):
        """Test re-encryption with expired rekey fails."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        
        ciphertext = backend.encrypt(b"secret", alice.public_key)
        
        # Create expired rekey
        expired = datetime.now() - timedelta(hours=1)
        rekey = backend.generate_rekey(
            alice.private_key,
            bob.public_key,
            expires_at=expired,
        )
        
        with pytest.raises(PREReEncryptionError):
            backend.re_encrypt(ciphertext, rekey)
    
    def test_verify_ciphertext(self, backend):
        """Test ciphertext verification."""
        alice = backend.generate_keypair(b"alice")
        
        ciphertext = backend.encrypt(b"data", alice.public_key)
        assert backend.verify_ciphertext(ciphertext)
        
        # Unknown ciphertext
        fake_ct = PRECiphertext(
            ciphertext_id=b"unknown",
            encrypted_data=b"fake",
            recipient_id=b"alice",
        )
        assert not backend.verify_ciphertext(fake_ct)
    
    def test_large_plaintext(self, backend):
        """Test encryption of larger data."""
        alice = backend.generate_keypair(b"alice")
        plaintext = b"x" * 10000  # 10KB
        
        ciphertext = backend.encrypt(plaintext, alice.public_key)
        decrypted = backend.decrypt(ciphertext, alice.private_key)
        
        assert decrypted == plaintext
    
    def test_empty_plaintext(self, backend):
        """Test encryption of empty data."""
        alice = backend.generate_keypair(b"alice")
        plaintext = b""
        
        ciphertext = backend.encrypt(plaintext, alice.public_key)
        decrypted = backend.decrypt(ciphertext, alice.private_key)
        
        assert decrypted == plaintext


# =============================================================================
# Federation Use Case Tests
# =============================================================================


class TestFederationScenarios:
    """Tests simulating federation aggregation scenarios."""
    
    @pytest.fixture
    def backend(self):
        """Create a fresh mock backend."""
        return MockPREBackend()
    
    def test_belief_sharing_between_instances(self, backend):
        """Test sharing encrypted beliefs between federation instances."""
        # Instance A and B have their own keypairs
        instance_a = backend.generate_keypair(b"instance-a")
        instance_b = backend.generate_keypair(b"instance-b")
        
        # Instance A has an encrypted belief
        belief_data = b'{"content": "The sky is blue", "confidence": 0.95}'
        encrypted_belief = backend.encrypt(belief_data, instance_a.public_key)
        
        # Instance A decides to share with Instance B
        rekey = backend.generate_rekey(
            instance_a.private_key,
            instance_b.public_key,
            metadata={"shared_at": datetime.now().isoformat()},
        )
        
        # Federation proxy re-encrypts without seeing the belief
        shared_belief = backend.re_encrypt(encrypted_belief, rekey)
        
        # Instance B can now decrypt
        received = backend.decrypt(shared_belief, instance_b.private_key)
        assert received == belief_data
        
        # Verify proxy couldn't decrypt
        # (In real implementation, proxy wouldn't have any keys)
        assert shared_belief.is_reencrypted
    
    def test_revocable_sharing(self, backend):
        """Test that sharing can be time-limited."""
        instance_a = backend.generate_keypair(b"instance-a")
        instance_b = backend.generate_keypair(b"instance-b")
        
        # Share with 1-hour window
        rekey = backend.generate_rekey(
            instance_a.private_key,
            instance_b.public_key,
            expires_at=datetime.now() + timedelta(hours=1),
        )
        
        # Works while valid
        belief = backend.encrypt(b"shared data", instance_a.public_key)
        shared = backend.re_encrypt(belief, rekey)
        assert backend.decrypt(shared, instance_b.private_key) == b"shared data"
        
        # After expiration, no new re-encryptions possible
        # (Would need to generate new rekey)
    
    def test_unidirectional_delegation(self, backend):
        """Test that delegation is unidirectional."""
        alice = backend.generate_keypair(b"alice")
        bob = backend.generate_keypair(b"bob")
        
        # Alice delegates to Bob
        rekey_ab = backend.generate_rekey(alice.private_key, bob.public_key)
        
        # Bob cannot use this rekey to share Alice's other data
        # Bob would need his own rekey to share his data
        alice_secret = backend.encrypt(b"alice data", alice.public_key)
        bob_secret = backend.encrypt(b"bob data", bob.public_key)
        
        # Alice's data can be shared with Bob
        shared = backend.re_encrypt(alice_secret, rekey_ab)
        assert backend.decrypt(shared, bob.private_key) == b"alice data"
        
        # Bob's data cannot be shared using rekey_ab
        with pytest.raises(PREReEncryptionError):
            backend.re_encrypt(bob_secret, rekey_ab)
    
    def test_multi_recipient_sharing(self, backend):
        """Test sharing same data with multiple recipients."""
        source = backend.generate_keypair(b"source")
        dest1 = backend.generate_keypair(b"dest1")
        dest2 = backend.generate_keypair(b"dest2")
        dest3 = backend.generate_keypair(b"dest3")
        
        # Source has encrypted data
        data = b"Federation knowledge graph update"
        encrypted = backend.encrypt(data, source.public_key)
        
        # Create rekeys for each destination
        rekey1 = backend.generate_rekey(source.private_key, dest1.public_key)
        rekey2 = backend.generate_rekey(source.private_key, dest2.public_key)
        rekey3 = backend.generate_rekey(source.private_key, dest3.public_key)
        
        # Re-encrypt for each destination
        for dest, rekey in [(dest1, rekey1), (dest2, rekey2), (dest3, rekey3)]:
            shared = backend.re_encrypt(encrypted, rekey)
            assert backend.decrypt(shared, dest.private_key) == data


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""
    
    def test_create_mock_backend(self):
        """Test create_mock_backend convenience function."""
        backend = create_mock_backend()
        
        assert isinstance(backend, MockPREBackend)
        assert isinstance(backend, PREBackend)


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for exception hierarchy."""
    
    def test_exception_hierarchy(self):
        """Test all exceptions inherit from PREError."""
        assert issubclass(PREKeyError, PREError)
        assert issubclass(PREEncryptionError, PREError)
        assert issubclass(PREDecryptionError, PREError)
        assert issubclass(PREReEncryptionError, PREError)
        assert issubclass(PREInvalidCiphertextError, PREError)
    
    def test_exceptions_catchable(self):
        """Test exceptions can be caught by base class."""
        try:
            raise PREDecryptionError("test")
        except PREError as e:
            assert str(e) == "test"
