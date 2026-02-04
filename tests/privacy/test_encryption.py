"""Tests for encryption envelope - AES-256-GCM with X25519 key exchange."""

import pytest

from valence.privacy.encryption import EncryptionEnvelope, generate_keypair


class TestGenerateKeypair:
    """Tests for keypair generation."""
    
    def test_generates_valid_lengths(self):
        """Test that keypairs have correct lengths."""
        private_key, public_key = generate_keypair()
        assert len(private_key) == 32
        assert len(public_key) == 32
    
    def test_generates_unique_keys(self):
        """Test that each call generates unique keys."""
        key1_priv, key1_pub = generate_keypair()
        key2_priv, key2_pub = generate_keypair()
        
        assert key1_priv != key2_priv
        assert key1_pub != key2_pub


class TestEncryptionEnvelope:
    """Tests for EncryptionEnvelope."""
    
    def test_encrypt_decrypt_roundtrip(self):
        """Test basic encrypt/decrypt roundtrip."""
        # Generate recipient keypair
        recipient_private, recipient_public = generate_keypair()
        
        # Original content
        plaintext = b"This is a secret belief that should be encrypted."
        
        # Encrypt
        envelope = EncryptionEnvelope.encrypt(plaintext, recipient_public)
        
        # Verify envelope structure
        assert envelope.encrypted_content != plaintext
        assert len(envelope.nonce) == 12
        assert len(envelope.ephemeral_public_key) == 32
        assert envelope.algorithm == "AES-256-GCM"
        
        # Decrypt
        decrypted = EncryptionEnvelope.decrypt(envelope, recipient_private)
        assert decrypted == plaintext
    
    def test_different_recipients_different_ciphertext(self):
        """Test that same content encrypted for different recipients differs."""
        _, pub1 = generate_keypair()
        _, pub2 = generate_keypair()
        
        plaintext = b"Same content"
        
        env1 = EncryptionEnvelope.encrypt(plaintext, pub1)
        env2 = EncryptionEnvelope.encrypt(plaintext, pub2)
        
        # Different ephemeral keys mean different ciphertext
        assert env1.encrypted_content != env2.encrypted_content
        assert env1.ephemeral_public_key != env2.ephemeral_public_key
    
    def test_wrong_key_fails_decryption(self):
        """Test that decryption with wrong key fails."""
        _, recipient_public = generate_keypair()
        wrong_private, _ = generate_keypair()
        
        plaintext = b"Secret content"
        envelope = EncryptionEnvelope.encrypt(plaintext, recipient_public)
        
        # Decryption with wrong key should fail (InvalidTag from AES-GCM)
        with pytest.raises(Exception):  # cryptography raises InvalidTag
            EncryptionEnvelope.decrypt(envelope, wrong_private)
    
    def test_empty_content(self):
        """Test encryption of empty content."""
        private, public = generate_keypair()
        
        envelope = EncryptionEnvelope.encrypt(b"", public)
        decrypted = EncryptionEnvelope.decrypt(envelope, private)
        
        assert decrypted == b""
    
    def test_large_content(self):
        """Test encryption of large content."""
        private, public = generate_keypair()
        
        # 1MB of data
        plaintext = b"x" * (1024 * 1024)
        
        envelope = EncryptionEnvelope.encrypt(plaintext, public)
        decrypted = EncryptionEnvelope.decrypt(envelope, private)
        
        assert decrypted == plaintext
    
    def test_unicode_content(self):
        """Test encryption of unicode content."""
        private, public = generate_keypair()
        
        plaintext = "Hello, 世界! 🎉".encode("utf-8")
        
        envelope = EncryptionEnvelope.encrypt(plaintext, public)
        decrypted = EncryptionEnvelope.decrypt(envelope, private)
        
        assert decrypted == plaintext
        assert decrypted.decode("utf-8") == "Hello, 世界! 🎉"
    
    def test_to_dict(self):
        """Test serialization to dict."""
        private, public = generate_keypair()
        
        envelope = EncryptionEnvelope.encrypt(b"test", public)
        data = envelope.to_dict()
        
        assert "encrypted_content" in data
        assert "nonce" in data
        assert "ephemeral_public_key" in data
        assert data["algorithm"] == "AES-256-GCM"
        
        # Should be base64 encoded strings
        assert isinstance(data["encrypted_content"], str)
        assert isinstance(data["nonce"], str)
    
    def test_from_dict(self):
        """Test deserialization from dict."""
        private, public = generate_keypair()
        
        original = EncryptionEnvelope.encrypt(b"test content", public)
        data = original.to_dict()
        
        restored = EncryptionEnvelope.from_dict(data)
        
        assert restored.encrypted_content == original.encrypted_content
        assert restored.nonce == original.nonce
        assert restored.ephemeral_public_key == original.ephemeral_public_key
        assert restored.algorithm == original.algorithm
    
    def test_serialization_roundtrip_with_decrypt(self):
        """Test that serialized envelope can still be decrypted."""
        private, public = generate_keypair()
        plaintext = b"Serialization test content"
        
        # Encrypt
        envelope = EncryptionEnvelope.encrypt(plaintext, public)
        
        # Serialize and deserialize
        data = envelope.to_dict()
        restored = EncryptionEnvelope.from_dict(data)
        
        # Decrypt restored envelope
        decrypted = EncryptionEnvelope.decrypt(restored, private)
        assert decrypted == plaintext
    
    def test_recipient_key_id(self):
        """Test optional recipient_key_id field."""
        private, public = generate_keypair()
        
        envelope = EncryptionEnvelope.encrypt(b"test", public)
        envelope.recipient_key_id = "did:key:alice#key-1"
        
        data = envelope.to_dict()
        assert data["recipient_key_id"] == "did:key:alice#key-1"
        
        restored = EncryptionEnvelope.from_dict(data)
        assert restored.recipient_key_id == "did:key:alice#key-1"


class TestEncryptionSecurity:
    """Security-focused tests for encryption."""
    
    def test_nonce_uniqueness(self):
        """Test that each encryption uses unique nonce."""
        private, public = generate_keypair()
        
        nonces = set()
        for _ in range(100):
            envelope = EncryptionEnvelope.encrypt(b"same content", public)
            nonces.add(envelope.nonce)
        
        # All 100 should be unique
        assert len(nonces) == 100
    
    def test_ephemeral_key_uniqueness(self):
        """Test that each encryption uses unique ephemeral key."""
        private, public = generate_keypair()
        
        keys = set()
        for _ in range(100):
            envelope = EncryptionEnvelope.encrypt(b"same content", public)
            keys.add(envelope.ephemeral_public_key)
        
        # All 100 should be unique
        assert len(keys) == 100
    
    def test_tampered_ciphertext_fails(self):
        """Test that tampering with ciphertext is detected."""
        private, public = generate_keypair()
        
        envelope = EncryptionEnvelope.encrypt(b"original content", public)
        
        # Tamper with ciphertext
        tampered = bytearray(envelope.encrypted_content)
        tampered[0] ^= 0xFF  # Flip bits
        envelope.encrypted_content = bytes(tampered)
        
        # Should fail authentication
        with pytest.raises(Exception):
            EncryptionEnvelope.decrypt(envelope, private)
    
    def test_tampered_nonce_fails(self):
        """Test that tampering with nonce is detected."""
        private, public = generate_keypair()
        
        envelope = EncryptionEnvelope.encrypt(b"original content", public)
        
        # Tamper with nonce
        tampered = bytearray(envelope.nonce)
        tampered[0] ^= 0xFF
        envelope.nonce = bytes(tampered)
        
        # Should fail
        with pytest.raises(Exception):
            EncryptionEnvelope.decrypt(envelope, private)
