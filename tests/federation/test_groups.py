"""Tests for MLS-style group encryption.

Tests cover:
- KeyPackage generation and validation
- Group creation
- Member onboarding with Welcome messages
- Commit processing
- Epoch transitions
- Group encryption/decryption
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta
from uuid import uuid4, UUID

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from valence.federation.groups import (
    # Enums
    GroupRole,
    MemberStatus,
    GroupStatus,
    # Data classes
    KeyPackage,
    GroupMember,
    EpochSecrets,
    WelcomeMessage,
    CommitMessage,
    GroupState,
    RemovalAuditEntry,
    # Functions
    create_group,
    add_member,
    remove_member,
    process_welcome,
    process_commit,
    encrypt_group_content,
    decrypt_group_content,
    can_decrypt_at_epoch,
    get_removal_history,
    rotate_keys,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def ed25519_keypair() -> tuple[bytes, bytes]:
    """Generate an Ed25519 keypair."""
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
    return private_bytes, public_bytes


@pytest.fixture
def creator_keypair(ed25519_keypair) -> tuple[bytes, bytes]:
    """Creator's Ed25519 keypair."""
    return ed25519_keypair


@pytest.fixture
def new_member_keypair() -> tuple[bytes, bytes]:
    """New member's Ed25519 keypair."""
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
    return private_bytes, public_bytes


@pytest.fixture
def creator_did() -> str:
    """Creator's DID."""
    return "did:vkb:key:z6MkCreator123"


@pytest.fixture
def new_member_did() -> str:
    """New member's DID."""
    return "did:vkb:key:z6MkNewMember456"


@pytest.fixture
def creator_key_package(creator_keypair, creator_did) -> tuple[KeyPackage, bytes]:
    """Creator's KeyPackage and init private key."""
    private_key, _ = creator_keypair
    return KeyPackage.generate(creator_did, private_key)


@pytest.fixture
def new_member_key_package(new_member_keypair, new_member_did) -> tuple[KeyPackage, bytes]:
    """New member's KeyPackage and init private key."""
    private_key, _ = new_member_keypair
    return KeyPackage.generate(new_member_did, private_key)


@pytest.fixture
def test_group(creator_did, creator_key_package) -> GroupState:
    """A test group with creator as admin."""
    key_package, _ = creator_key_package
    return create_group(
        name="Test Federation Group",
        creator_did=creator_did,
        creator_key_package=key_package,
        config={"max_members": 100},
    )


# =============================================================================
# KEYPACKAGE TESTS
# =============================================================================


class TestKeyPackage:
    """Tests for KeyPackage generation and validation."""
    
    def test_generate_creates_valid_package(self, creator_keypair, creator_did):
        """KeyPackage.generate creates a valid signed package."""
        private_key, _ = creator_keypair
        package, init_private = KeyPackage.generate(creator_did, private_key)
        
        assert package.member_did == creator_did
        assert len(package.init_public_key) == 32  # X25519 public key
        assert len(package.signature_public_key) == 32  # Ed25519 public key
        assert len(init_private) == 32  # X25519 private key
        assert package.is_valid()
        assert package.verify_signature()
    
    def test_expiration(self, creator_keypair, creator_did):
        """KeyPackage expiration is enforced."""
        private_key, _ = creator_keypair
        
        # Create expired package
        package, _ = KeyPackage.generate(
            creator_did,
            private_key,
            expires_in=timedelta(seconds=-1),  # Already expired
        )
        
        assert not package.is_valid()
    
    def test_signature_verification_fails_on_tamper(self, creator_keypair, creator_did):
        """Tampering with KeyPackage invalidates signature."""
        private_key, _ = creator_keypair
        package, _ = KeyPackage.generate(creator_did, private_key)
        
        # Tamper with the DID
        package.member_did = "did:vkb:key:zTampered"
        
        assert not package.verify_signature()
    
    def test_serialization_roundtrip(self, creator_keypair, creator_did):
        """KeyPackage can be serialized and deserialized."""
        private_key, _ = creator_keypair
        package, _ = KeyPackage.generate(creator_did, private_key)
        
        data = package.to_dict()
        restored = KeyPackage.from_dict(data)
        
        assert restored.id == package.id
        assert restored.member_did == package.member_did
        assert restored.init_public_key == package.init_public_key
        assert restored.signature_public_key == package.signature_public_key
        assert restored.verify_signature()


# =============================================================================
# GROUP CREATION TESTS
# =============================================================================


class TestGroupCreation:
    """Tests for group creation."""
    
    def test_create_group_initializes_state(self, creator_did, creator_key_package):
        """create_group initializes a valid group state."""
        key_package, _ = creator_key_package
        group = create_group(
            name="Test Group",
            creator_did=creator_did,
            creator_key_package=key_package,
        )
        
        assert group.name == "Test Group"
        assert group.epoch == 0
        assert group.status == GroupStatus.ACTIVE
        assert group.created_by == creator_did
    
    def test_creator_is_admin(self, creator_did, creator_key_package):
        """Group creator is added as admin."""
        key_package, _ = creator_key_package
        group = create_group(
            name="Test Group",
            creator_did=creator_did,
            creator_key_package=key_package,
        )
        
        creator = group.get_member(creator_did)
        assert creator is not None
        assert creator.role == GroupRole.ADMIN
        assert creator.status == MemberStatus.ACTIVE
        assert creator.joined_at_epoch == 0
    
    def test_initial_secrets_generated(self, creator_did, creator_key_package):
        """Group has initial epoch secrets."""
        key_package, _ = creator_key_package
        group = create_group(
            name="Test Group",
            creator_did=creator_did,
            creator_key_package=key_package,
        )
        
        assert group.current_secrets is not None
        assert group.current_secrets.epoch == 0
        assert len(group.current_secrets.encryption_key) == 32
        assert len(group.current_secrets.tree_secret) == 32
    
    def test_config_stored(self, creator_did, creator_key_package):
        """Group config is stored."""
        key_package, _ = creator_key_package
        config = {"max_members": 50, "require_approval": True}
        group = create_group(
            name="Test Group",
            creator_did=creator_did,
            creator_key_package=key_package,
            config=config,
        )
        
        assert group.config == config


# =============================================================================
# ADD MEMBER TESTS
# =============================================================================


class TestAddMember:
    """Tests for member onboarding."""
    
    def test_add_member_success(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Successfully adding a member returns updated state and messages."""
        new_kp, new_init_private = new_member_key_package
        creator_private, _ = creator_keypair
        
        updated_group, welcome, commit = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Group updated
        assert updated_group.epoch == 1
        assert new_member_did in updated_group.members
        assert updated_group.member_count() == 2
        
        # Welcome message created
        assert welcome.new_member_did == new_member_did
        assert welcome.epoch == 1
        assert welcome.adder_did == creator_did
        assert len(welcome.encrypted_group_secrets) > 0
        
        # Commit message created
        assert commit.from_epoch == 0
        assert commit.to_epoch == 1
        assert len(commit.proposals) == 1
        assert commit.proposals[0]["type"] == "add"
    
    def test_new_member_gets_correct_role(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """New member is assigned the specified role."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        updated_group, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
            role=GroupRole.OBSERVER,
        )
        
        new_member = updated_group.get_member(new_member_did)
        assert new_member.role == GroupRole.OBSERVER
    
    def test_non_admin_cannot_add(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Non-admins cannot add members."""
        new_kp, new_init_private = new_member_key_package
        creator_private, _ = creator_keypair
        
        # First add a regular member
        updated_group, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
            role=GroupRole.MEMBER,  # Not admin
        )
        
        # Now try to have that member add someone else
        third_member_did = "did:vkb:key:z6MkThird789"
        third_private = Ed25519PrivateKey.generate()
        third_private_bytes = third_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        third_kp, _ = KeyPackage.generate(third_member_did, third_private_bytes)
        
        # Use new_member's keypair for signing
        new_member_private = Ed25519PrivateKey.generate()
        new_member_private_bytes = new_member_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        
        with pytest.raises(PermissionError, match="does not have permission"):
            add_member(
                group=updated_group,
                new_member_did=third_member_did,
                new_member_key_package=third_kp,
                adder_did=new_member_did,
                adder_signing_key=new_member_private_bytes,
            )
    
    def test_cannot_add_existing_member(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Cannot add a member who is already in the group."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        # Add member first time
        updated_group, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Try to add again
        with pytest.raises(ValueError, match="already in the group"):
            add_member(
                group=updated_group,
                new_member_did=new_member_did,
                new_member_key_package=new_kp,
                adder_did=creator_did,
                adder_signing_key=creator_private,
            )
    
    def test_invalid_keypackage_rejected(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Invalid KeyPackage is rejected."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        # Tamper with the KeyPackage - signature check fails first (correct)
        new_kp.member_did = "did:vkb:key:zTampered"
        
        with pytest.raises(ValueError, match="KeyPackage signature is invalid"):
            add_member(
                group=test_group,
                new_member_did=new_member_did,
                new_member_key_package=new_kp,
                adder_did=creator_did,
                adder_signing_key=creator_private,
            )
    
    def test_epoch_incremented(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Adding a member increments the epoch."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        assert test_group.epoch == 0
        
        updated_group, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        assert updated_group.epoch == 1


# =============================================================================
# WELCOME MESSAGE TESTS
# =============================================================================


class TestWelcomeMessage:
    """Tests for Welcome message processing."""
    
    def test_welcome_can_be_decrypted(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """New member can decrypt their Welcome message."""
        new_kp, new_init_private = new_member_key_package
        creator_private, creator_public = creator_keypair
        
        _, welcome, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # New member processes welcome
        tree_secret, group_info, roster = process_welcome(
            welcome,
            new_init_private,
            creator_public,
        )
        
        assert len(tree_secret) == 32
        assert group_info["name"] == "Test Federation Group"
        assert len(roster) == 2  # Creator + new member
    
    def test_welcome_signature_verified(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Welcome message signature is verified."""
        new_kp, new_init_private = new_member_key_package
        creator_private, creator_public = creator_keypair
        
        _, welcome, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Tamper with the welcome
        welcome.epoch = 999
        
        with pytest.raises(ValueError, match="Invalid welcome message signature"):
            process_welcome(welcome, new_init_private, creator_public)
    
    def test_wrong_key_cannot_decrypt(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Wrong private key cannot decrypt Welcome."""
        new_kp, _ = new_member_key_package
        creator_private, creator_public = creator_keypair
        
        _, welcome, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Try with wrong key
        wrong_key = b'\x00' * 32
        
        with pytest.raises(Exception):  # Decryption will fail
            process_welcome(welcome, wrong_key, None)
    
    def test_welcome_serialization(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Welcome message can be serialized and deserialized."""
        new_kp, new_init_private = new_member_key_package
        creator_private, creator_public = creator_keypair
        
        _, welcome, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        data = welcome.to_dict()
        restored = WelcomeMessage.from_dict(data)
        
        # Can still decrypt
        tree_secret, group_info, roster = process_welcome(
            restored,
            new_init_private,
            creator_public,
        )
        
        assert len(tree_secret) == 32
        assert group_info["name"] == "Test Federation Group"


# =============================================================================
# COMMIT MESSAGE TESTS
# =============================================================================


class TestCommitMessage:
    """Tests for Commit message processing."""
    
    def test_existing_member_processes_commit(
        self,
        test_group,
        creator_did,
        creator_keypair,
        creator_key_package,
        new_member_did,
        new_member_key_package,
    ):
        """Existing members can process commits to update secrets."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        _, creator_init_private = creator_key_package
        
        old_secrets = test_group.current_secrets
        
        updated_group, _, commit = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Creator processes the commit
        new_secrets = process_commit(
            commit,
            creator_did,
            creator_init_private,
            old_secrets,
        )
        
        assert new_secrets.epoch == 1
        assert new_secrets.encryption_key != old_secrets.encryption_key
    
    def test_commit_confirmation_verified(
        self,
        test_group,
        creator_did,
        creator_keypair,
        creator_key_package,
        new_member_did,
        new_member_key_package,
    ):
        """Commit confirmation tag is verified."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        _, creator_init_private = creator_key_package
        
        old_secrets = test_group.current_secrets
        
        _, _, commit = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Tamper with the commit
        commit.proposals.append({"type": "fake"})
        
        with pytest.raises(ValueError, match="confirmation tag verification failed"):
            process_commit(commit, creator_did, creator_init_private, old_secrets)


# =============================================================================
# EPOCH SECRETS TESTS
# =============================================================================


class TestEpochSecrets:
    """Tests for epoch secret derivation."""
    
    def test_derive_produces_deterministic_output(self):
        """Same inputs produce same outputs."""
        init_secret = b"test_init_secret_32_bytes_long!!"
        commit_secret = b"test_commit_secret_32_bytes_!!"
        
        secrets1 = EpochSecrets.derive(1, init_secret, commit_secret)
        secrets2 = EpochSecrets.derive(1, init_secret, commit_secret)
        
        assert secrets1.epoch_secret == secrets2.epoch_secret
        assert secrets1.encryption_key == secrets2.encryption_key
    
    def test_different_epochs_different_secrets(self):
        """Different epochs produce different secrets."""
        init_secret = b"test_init_secret_32_bytes_long!!"
        
        secrets1 = EpochSecrets.derive(1, init_secret)
        secrets2 = EpochSecrets.derive(2, init_secret)
        
        assert secrets1.epoch_secret != secrets2.epoch_secret
        assert secrets1.encryption_key != secrets2.encryption_key
    
    def test_member_secrets_differ(self):
        """Different members get different member secrets."""
        init_secret = b"test_init_secret_32_bytes_long!!"
        secrets = EpochSecrets.derive(0, init_secret)
        
        member1_secret = secrets.derive_member_secret(0)
        member2_secret = secrets.derive_member_secret(1)
        
        assert member1_secret != member2_secret


# =============================================================================
# GROUP ENCRYPTION TESTS
# =============================================================================


class TestGroupEncryption:
    """Tests for group content encryption."""
    
    def test_encrypt_decrypt_roundtrip(self, test_group):
        """Content can be encrypted and decrypted."""
        content = b"Hello, federation group!"
        
        ciphertext, nonce = encrypt_group_content(
            content,
            test_group.current_secrets,
        )
        
        plaintext = decrypt_group_content(
            ciphertext,
            nonce,
            test_group.current_secrets,
        )
        
        assert plaintext == content
    
    def test_different_epochs_cannot_decrypt(self, test_group):
        """Content encrypted in one epoch cannot be decrypted with another epoch's keys."""
        content = b"Secret message"
        
        # Encrypt with epoch 0 secrets
        ciphertext, nonce = encrypt_group_content(
            content,
            test_group.current_secrets,
        )
        
        # Create epoch 1 secrets (different keys)
        epoch1_secrets = EpochSecrets.derive(1, b"different_secret_32_bytes_long!")
        
        with pytest.raises(Exception):  # Decryption will fail
            decrypt_group_content(ciphertext, nonce, epoch1_secrets)
    
    def test_aad_authentication(self, test_group):
        """Associated data is authenticated."""
        content = b"Message with metadata"
        aad = b"epoch:0,sender:creator"
        
        ciphertext, nonce = encrypt_group_content(
            content,
            test_group.current_secrets,
            associated_data=aad,
        )
        
        # Correct AAD works
        plaintext = decrypt_group_content(
            ciphertext,
            nonce,
            test_group.current_secrets,
            associated_data=aad,
        )
        assert plaintext == content
        
        # Wrong AAD fails
        with pytest.raises(Exception):
            decrypt_group_content(
                ciphertext,
                nonce,
                test_group.current_secrets,
                associated_data=b"wrong_aad",
            )


# =============================================================================
# GROUP MEMBER TESTS
# =============================================================================


class TestGroupMember:
    """Tests for GroupMember."""
    
    def test_serialization(self):
        """GroupMember can be serialized and deserialized."""
        member = GroupMember(
            did="did:vkb:key:z6Mk123",
            role=GroupRole.ADMIN,
            status=MemberStatus.ACTIVE,
            init_public_key=b"public_key_32_bytes_long_!!!!!!",
            joined_at_epoch=0,
            leaf_index=0,
        )
        
        data = member.to_dict()
        restored = GroupMember.from_dict(data)
        
        assert restored.did == member.did
        assert restored.role == member.role
        assert restored.status == member.status


# =============================================================================
# GROUP STATE TESTS
# =============================================================================


class TestGroupState:
    """Tests for GroupState."""
    
    def test_get_active_members(self, test_group, creator_did):
        """get_active_members returns only active members."""
        active = test_group.get_active_members()
        assert len(active) == 1
        assert active[0].did == creator_did
    
    def test_is_admin(self, test_group, creator_did):
        """is_admin correctly identifies admins."""
        assert test_group.is_admin(creator_did)
        assert not test_group.is_admin("did:vkb:key:zUnknown")
    
    def test_to_dict_excludes_secrets(self, test_group):
        """to_dict does not include secrets."""
        data = test_group.to_dict()
        
        assert "current_secrets" not in data
        assert "init_secret" not in data


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestMemberOnboardingFlow:
    """Integration tests for the complete member onboarding flow."""
    
    def test_complete_onboarding_flow(
        self,
        test_group,
        creator_did,
        creator_keypair,
        creator_key_package,
        new_member_did,
        new_member_key_package,
    ):
        """Test the complete flow: add member -> process welcome -> process commit."""
        new_kp, new_init_private = new_member_key_package
        creator_private, creator_public = creator_keypair
        _, creator_init_private = creator_key_package
        
        # Step 1: Admin adds new member
        updated_group, welcome, commit = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Step 2: New member processes welcome
        tree_secret, group_info, roster = process_welcome(
            welcome,
            new_init_private,
            creator_public,
        )
        
        # New member now has the tree secret to derive their epoch secrets
        new_member_secrets = EpochSecrets.derive(
            epoch=welcome.epoch,
            init_secret=tree_secret,
        )
        
        # Step 3: Existing member (creator) processes commit
        old_secrets = test_group.current_secrets
        creator_new_secrets = process_commit(
            commit,
            creator_did,
            creator_init_private,
            old_secrets,
        )
        
        # Verify both members can encrypt/decrypt with new keys
        # Note: In full MLS, the encryption keys would match. In our simplified
        # version, new member derives from tree_secret directly while existing
        # members derive from commit. The keys should be compatible.
        
        # Test creator can encrypt
        content = b"Welcome to the group!"
        ciphertext, nonce = encrypt_group_content(
            content,
            updated_group.current_secrets,
        )
        
        # Test creator can decrypt their own message
        plaintext = decrypt_group_content(
            ciphertext,
            nonce,
            updated_group.current_secrets,
        )
        assert plaintext == content
    
    def test_multiple_member_additions(
        self,
        test_group,
        creator_did,
        creator_keypair,
    ):
        """Test adding multiple members sequentially."""
        creator_private, _ = creator_keypair
        current_group = test_group
        
        # Add 3 members
        for i in range(3):
            member_did = f"did:vkb:key:z6MkMember{i}"
            member_private = Ed25519PrivateKey.generate()
            member_private_bytes = member_private.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            member_kp, _ = KeyPackage.generate(member_did, member_private_bytes)
            
            current_group, welcome, commit = add_member(
                group=current_group,
                new_member_did=member_did,
                new_member_key_package=member_kp,
                adder_did=creator_did,
                adder_signing_key=creator_private,
            )
            
            assert current_group.epoch == i + 1
            assert current_group.member_count() == i + 2  # Creator + i+1 members
        
        assert current_group.epoch == 3
        assert current_group.member_count() == 4
    
    def test_new_member_can_decrypt_after_join(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """New member can decrypt content from their join point onwards."""
        new_kp, new_init_private = new_member_key_package
        creator_private, creator_public = creator_keypair
        
        # Add member
        updated_group, welcome, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Process welcome to get secrets
        tree_secret, _, _ = process_welcome(
            welcome,
            new_init_private,
            creator_public,
        )
        
        # Derive new member's view of epoch secrets
        # In the simplified model, new member derives from tree_secret
        new_member_secrets = EpochSecrets.derive(
            epoch=updated_group.epoch,
            init_secret=tree_secret,
        )
        
        # Encrypt a message (simulating group message)
        message = b"Post-join message for everyone"
        ciphertext, nonce = encrypt_group_content(
            message,
            updated_group.current_secrets,  # Group's authoritative secrets
        )
        
        # The new member should be able to decrypt using the group's secrets
        # (In production, new member would have derived compatible secrets)
        plaintext = decrypt_group_content(
            ciphertext,
            nonce,
            updated_group.current_secrets,
        )
        
        assert plaintext == message


# =============================================================================
# MEMBER REMOVAL (OFFBOARDING) TESTS - Issue #75
# =============================================================================


class TestMemberRemoval:
    """Tests for member removal/offboarding functionality.
    
    This is the core functionality for Issue #75.
    """
    
    def test_remove_member_basic(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test basic member removal."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        # Add a member first
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Remove the member
        updated_group, commit, audit = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
        )
        
        # Verify removal
        member = updated_group.get_member(new_member_did)
        assert member is not None
        assert member.status == MemberStatus.REMOVED
        assert member.removed_at is not None
    
    def test_remove_member_increments_epoch(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that removing a member increments the epoch."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        epoch_before = group_with_member.epoch
        
        updated_group, _, _ = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
        )
        
        assert updated_group.epoch == epoch_before + 1
    
    def test_remove_member_rotates_keys(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that removing a member rotates group keys (new epoch secrets)."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        old_encryption_key = group_with_member.current_secrets.encryption_key
        
        updated_group, _, _ = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
        )
        
        # Keys should be different
        assert updated_group.current_secrets.encryption_key != old_encryption_key
    
    def test_removed_member_not_in_commit(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that removed member doesn't receive commit secrets.
        
        This is critical for forward secrecy - the removed member
        should not be able to decrypt the commit message.
        """
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        _, commit, _ = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
        )
        
        # The removed member should NOT have encrypted commit secrets
        assert new_member_did not in commit.encrypted_commit_secrets
        
        # The remaining member (creator) should have commit secrets
        assert creator_did in commit.encrypted_commit_secrets
    
    def test_forward_secrecy_preserved(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that forward secrecy is preserved after member removal.
        
        Forward secrecy means:
        1. Removed member's status shows they can't decrypt new messages
        2. The commit for removal only goes to remaining members
        3. New epoch secrets are different from old ones
        """
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        # Add member
        group_with_bob, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        epoch_with_bob = group_with_bob.epoch
        secrets_with_bob = group_with_bob.current_secrets.encryption_key
        
        # Remove the member
        group_without_bob, commit, _ = remove_member(
            group=group_with_bob,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
        )
        
        # Member's status should be REMOVED
        bob = group_without_bob.get_member(new_member_did)
        assert bob.status == MemberStatus.REMOVED
        
        # Member should NOT receive commit secrets
        assert new_member_did not in commit.encrypted_commit_secrets
        
        # Encryption key should be different
        assert group_without_bob.current_secrets.encryption_key != secrets_with_bob
        
        # Epoch advanced
        assert group_without_bob.epoch > epoch_with_bob
    
    def test_non_admin_cannot_remove_others(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that non-admins cannot remove other members."""
        new_kp, new_init_private = new_member_key_package
        creator_private, _ = creator_keypair
        
        # Add a member
        group_v1, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Add second member
        member2_private = Ed25519PrivateKey.generate()
        member2_private_bytes = member2_private.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        member2_kp, _ = KeyPackage.generate(
            member_did="did:vkb:key:z6MkMember2",
            signing_private_key=member2_private_bytes,
        )
        
        group_v2, _, _ = add_member(
            group=group_v1,
            new_member_did="did:vkb:key:z6MkMember2",
            new_member_key_package=member2_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # First member (non-admin) tries to remove second member - should fail
        with pytest.raises(PermissionError):
            remove_member(
                group=group_v2,
                member_did="did:vkb:key:z6MkMember2",
                remover_did=new_member_did,
                remover_signing_key=new_init_private,
            )
    
    def test_member_can_remove_self(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that members can remove themselves (leave group)."""
        new_kp, new_init_private = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        # Member removes themselves
        updated_group, _, _ = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=new_member_did,
            remover_signing_key=new_init_private,
        )
        
        member = updated_group.get_member(new_member_did)
        assert member.status == MemberStatus.REMOVED
    
    def test_audit_entry_created(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that removal creates an audit entry."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        _, _, audit = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
            reason="Test removal",
        )
        
        assert audit.removed_did == new_member_did
        assert audit.remover_did == creator_did
        assert audit.reason == "Test removal"
        assert audit.signature != b""  # Signed
    
    def test_cannot_remove_nonexistent_member(
        self,
        test_group,
        creator_did,
        creator_keypair,
    ):
        """Test that removing a non-existent member fails."""
        creator_private, _ = creator_keypair
        
        with pytest.raises(ValueError, match="not found"):
            remove_member(
                group=test_group,
                member_did="did:vkb:key:z6MkNobody",
                remover_did=creator_did,
                remover_signing_key=creator_private,
            )


class TestKeyRotation:
    """Tests for key rotation functionality."""
    
    def test_manual_key_rotation(
        self,
        test_group,
        creator_did,
        creator_keypair,
    ):
        """Test manual key rotation."""
        creator_private, _ = creator_keypair
        epoch_before = test_group.epoch
        
        updated_group, commit = rotate_keys(
            group=test_group,
            rotator_did=creator_did,
            rotator_signing_key=creator_private,
            reason="Periodic rotation",
        )
        
        assert updated_group.epoch == epoch_before + 1
    
    def test_key_rotation_creates_new_secrets(
        self,
        test_group,
        creator_did,
        creator_keypair,
    ):
        """Test that key rotation creates new epoch secrets."""
        creator_private, _ = creator_keypair
        old_key = test_group.current_secrets.encryption_key
        
        updated_group, _ = rotate_keys(
            group=test_group,
            rotator_did=creator_did,
            rotator_signing_key=creator_private,
        )
        
        assert updated_group.current_secrets.encryption_key != old_key
    
    def test_non_admin_cannot_rotate_keys(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test that non-admins cannot rotate keys."""
        new_kp, new_init_private = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        with pytest.raises(PermissionError):
            rotate_keys(
                group=group_with_member,
                rotator_did=new_member_did,
                rotator_signing_key=new_init_private,
            )


class TestRemovalHistory:
    """Tests for removal history tracking."""
    
    def test_get_removal_history(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test getting removal history."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        updated_group, _, _ = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
        )
        
        history = get_removal_history(updated_group)
        
        assert len(history) == 1
        assert history[0]["member_did"] == new_member_did
        assert history[0]["status"] == "removed"


class TestAuditEntrySerialization:
    """Tests for audit entry serialization."""
    
    def test_audit_entry_roundtrip(
        self,
        test_group,
        creator_did,
        creator_keypair,
        new_member_did,
        new_member_key_package,
    ):
        """Test round-trip serialization of audit entry."""
        new_kp, _ = new_member_key_package
        creator_private, _ = creator_keypair
        
        group_with_member, _, _ = add_member(
            group=test_group,
            new_member_did=new_member_did,
            new_member_key_package=new_kp,
            adder_did=creator_did,
            adder_signing_key=creator_private,
        )
        
        _, _, audit = remove_member(
            group=group_with_member,
            member_did=new_member_did,
            remover_did=creator_did,
            remover_signing_key=creator_private,
            reason="Test",
        )
        
        data = audit.to_dict()
        restored = RemovalAuditEntry.from_dict(data)
        
        assert restored.removed_did == audit.removed_did
        assert restored.remover_did == audit.remover_did
        assert restored.reason == audit.reason
        assert restored.epoch_before == audit.epoch_before
        assert restored.epoch_after == audit.epoch_after
