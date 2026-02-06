"""Tests for gateway key rotation mechanism.

Tests cover:
- KeyVersion dataclass (creation, expiry, signing, verification)
- KeyRotationConfig defaults and customization
- KeyRotationManager (generation, rotation, transition period, verification)
- GatewayNode integration (sign/verify messages, auto-rotation, stats)
- Edge cases (revocation, expired keys, unknown versions)

Issue #253: Gateway key rotation mechanism
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from valence.federation.gateway import (
    DEFAULT_KEY_SIZE,
    DEFAULT_ROTATION_INTERVAL_HOURS,
    DEFAULT_TRANSITION_PERIOD_HOURS,
    MAX_KEY_HISTORY,
    AuditEventType,
    GatewayNode,
    GatewayStatus,
    KeyRotationConfig,
    KeyRotationException,
    KeyRotationManager,
    KeyVersion,
    create_gateway,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def key_version() -> KeyVersion:
    """Create a test key version."""
    return KeyVersion(
        version_id="v1-test",
        key_material=b"test-key-material-32-bytes-long!",
    )


@pytest.fixture
def rotation_config() -> KeyRotationConfig:
    """Create a test rotation config with short intervals."""
    return KeyRotationConfig(
        rotation_interval=timedelta(seconds=10),
        transition_period=timedelta(seconds=5),
        max_key_history=3,
    )


@pytest.fixture
def manager(rotation_config: KeyRotationConfig) -> KeyRotationManager:
    """Create a KeyRotationManager with test config."""
    return KeyRotationManager(config=rotation_config)


@pytest.fixture
def gateway() -> GatewayNode:
    """Create a GatewayNode for testing key rotation integration."""
    return GatewayNode(
        federation_id=uuid4(),
        endpoint="https://gateway.test.example.com/v1",
        key_rotation_config=KeyRotationConfig(
            rotation_interval=timedelta(seconds=10),
            transition_period=timedelta(seconds=5),
        ),
    )


# =============================================================================
# KEY VERSION TESTS
# =============================================================================


class TestKeyVersion:
    """Tests for the KeyVersion dataclass."""

    def test_creation_defaults(self, key_version: KeyVersion) -> None:
        """KeyVersion has correct defaults on creation."""
        assert key_version.version_id == "v1-test"
        assert key_version.key_material == b"test-key-material-32-bytes-long!"
        assert key_version.expires_at is None
        assert key_version.revoked is False

    def test_is_expired_no_expiry(self, key_version: KeyVersion) -> None:
        """Key without expiry is never expired."""
        assert key_version.is_expired is False

    def test_is_expired_future(self, key_version: KeyVersion) -> None:
        """Key with future expiry is not expired."""
        key_version.expires_at = datetime.now(UTC) + timedelta(hours=1)
        assert key_version.is_expired is False

    def test_is_expired_past(self, key_version: KeyVersion) -> None:
        """Key with past expiry is expired."""
        key_version.expires_at = datetime.now(UTC) - timedelta(hours=1)
        assert key_version.is_expired is True

    def test_is_usable_fresh(self, key_version: KeyVersion) -> None:
        """Fresh key is usable."""
        assert key_version.is_usable is True

    def test_is_usable_revoked(self, key_version: KeyVersion) -> None:
        """Revoked key is not usable."""
        key_version.revoked = True
        assert key_version.is_usable is False

    def test_is_usable_expired(self, key_version: KeyVersion) -> None:
        """Expired key is not usable."""
        key_version.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert key_version.is_usable is False

    def test_sign_produces_hex(self, key_version: KeyVersion) -> None:
        """Sign returns a hex string."""
        sig = key_version.sign(b"hello world")
        assert isinstance(sig, str)
        # HMAC-SHA256 produces 64 hex chars
        assert len(sig) == 64

    def test_sign_deterministic(self, key_version: KeyVersion) -> None:
        """Same data + key produces same signature."""
        sig1 = key_version.sign(b"hello")
        sig2 = key_version.sign(b"hello")
        assert sig1 == sig2

    def test_sign_different_data(self, key_version: KeyVersion) -> None:
        """Different data produces different signatures."""
        sig1 = key_version.sign(b"hello")
        sig2 = key_version.sign(b"world")
        assert sig1 != sig2

    def test_verify_valid(self, key_version: KeyVersion) -> None:
        """Verify returns True for valid signature."""
        sig = key_version.sign(b"test data")
        assert key_version.verify(b"test data", sig) is True

    def test_verify_invalid(self, key_version: KeyVersion) -> None:
        """Verify returns False for invalid signature."""
        assert key_version.verify(b"test data", "bad-signature") is False

    def test_verify_wrong_data(self, key_version: KeyVersion) -> None:
        """Verify returns False when data doesn't match."""
        sig = key_version.sign(b"original")
        assert key_version.verify(b"tampered", sig) is False

    def test_different_keys_different_sigs(self) -> None:
        """Different keys produce different signatures for same data."""
        k1 = KeyVersion(version_id="v1", key_material=b"key-material-one-32-bytes-long!!")
        k2 = KeyVersion(version_id="v2", key_material=b"key-material-two-32-bytes-long!!")
        data = b"same data"
        assert k1.sign(data) != k2.sign(data)

    def test_to_dict_excludes_key_material(self, key_version: KeyVersion) -> None:
        """to_dict does not expose key material."""
        d = key_version.to_dict()
        assert "key_material" not in d
        assert d["version_id"] == "v1-test"
        assert d["revoked"] is False
        assert d["is_usable"] is True
        assert d["is_expired"] is False

    def test_to_dict_with_expiry(self, key_version: KeyVersion) -> None:
        """to_dict includes expiry when set."""
        key_version.expires_at = datetime(2030, 1, 1, tzinfo=UTC)
        d = key_version.to_dict()
        assert d["expires_at"] == "2030-01-01T00:00:00+00:00"


# =============================================================================
# KEY ROTATION CONFIG TESTS
# =============================================================================


class TestKeyRotationConfig:
    """Tests for KeyRotationConfig."""

    def test_defaults(self) -> None:
        """Default config uses module-level constants."""
        cfg = KeyRotationConfig()
        assert cfg.key_size == DEFAULT_KEY_SIZE
        assert cfg.rotation_interval == timedelta(hours=DEFAULT_ROTATION_INTERVAL_HOURS)
        assert cfg.transition_period == timedelta(hours=DEFAULT_TRANSITION_PERIOD_HOURS)
        assert cfg.max_key_history == MAX_KEY_HISTORY
        assert cfg.auto_rotate is True

    def test_custom_values(self) -> None:
        """Custom config values are respected."""
        cfg = KeyRotationConfig(
            key_size=64,
            rotation_interval=timedelta(hours=12),
            transition_period=timedelta(hours=1),
            max_key_history=10,
            auto_rotate=False,
        )
        assert cfg.key_size == 64
        assert cfg.rotation_interval == timedelta(hours=12)
        assert cfg.transition_period == timedelta(hours=1)
        assert cfg.max_key_history == 10
        assert cfg.auto_rotate is False


# =============================================================================
# KEY ROTATION MANAGER TESTS
# =============================================================================


class TestKeyRotationManager:
    """Tests for KeyRotationManager."""

    def test_init_generates_key(self) -> None:
        """Manager generates an initial key on creation."""
        mgr = KeyRotationManager()
        assert mgr.active_key is not None
        assert mgr.active_key.is_usable is True

    def test_init_with_initial_key(self, key_version: KeyVersion) -> None:
        """Manager accepts a provided initial key."""
        mgr = KeyRotationManager(initial_key=key_version)
        assert mgr.active_key is key_version

    def test_active_key_property(self, manager: KeyRotationManager) -> None:
        """active_key returns the current key."""
        key = manager.active_key
        assert isinstance(key, KeyVersion)
        assert key.is_usable

    def test_previous_keys_initially_empty(self, manager: KeyRotationManager) -> None:
        """No previous keys before first rotation."""
        assert manager.previous_keys == []

    def test_all_valid_keys_single(self, manager: KeyRotationManager) -> None:
        """Only active key when no rotation has occurred."""
        keys = manager.all_valid_keys
        assert len(keys) == 1
        assert keys[0] is manager.active_key

    def test_rotate_creates_new_key(self, manager: KeyRotationManager) -> None:
        """Rotation generates a new active key."""
        old_key = manager.active_key
        new_key = manager.rotate()
        assert new_key is not old_key
        assert manager.active_key is new_key
        assert new_key.is_usable

    def test_rotate_moves_old_to_previous(self, manager: KeyRotationManager) -> None:
        """Old key moves to previous keys on rotation."""
        old_key = manager.active_key
        manager.rotate()
        assert old_key in manager.previous_keys

    def test_rotate_sets_expiry_on_old_key(self, manager: KeyRotationManager, rotation_config: KeyRotationConfig) -> None:
        """Old key gets transition period expiry after rotation."""
        old_key = manager.active_key
        manager.rotate()
        assert old_key.expires_at is not None
        # Should expire within transition period (± small delta)
        expected = datetime.now(UTC) + rotation_config.transition_period
        delta = abs((old_key.expires_at - expected).total_seconds())
        assert delta < 2  # within 2 seconds

    def test_rotate_preserves_key_history(self, manager: KeyRotationManager) -> None:
        """Multiple rotations preserve keys up to max_key_history."""
        keys = []
        for _ in range(3):
            keys.append(manager.active_key)
            manager.rotate()
        # All 3 should be in previous keys (config max is 3)
        for k in keys:
            assert k in manager._previous_keys

    def test_rotate_prunes_excess_keys(self, manager: KeyRotationManager) -> None:
        """Keys beyond max_key_history are pruned."""
        # Config max is 3
        for _ in range(5):
            manager.rotate()
        assert len(manager._previous_keys) <= 3

    def test_needs_rotation_fresh_key(self, manager: KeyRotationManager) -> None:
        """Fresh key does not need rotation."""
        assert manager.needs_rotation is False

    def test_needs_rotation_expired_interval(self) -> None:
        """Key past rotation interval needs rotation."""
        config = KeyRotationConfig(
            rotation_interval=timedelta(seconds=0),  # immediate
            auto_rotate=True,
        )
        mgr = KeyRotationManager(config=config)
        # Backdate the key
        mgr._active_key.created_at = datetime.now(UTC) - timedelta(seconds=1)
        assert mgr.needs_rotation is True

    def test_needs_rotation_auto_disabled(self) -> None:
        """No rotation needed when auto_rotate is disabled."""
        config = KeyRotationConfig(
            rotation_interval=timedelta(seconds=0),
            auto_rotate=False,
        )
        mgr = KeyRotationManager(config=config)
        mgr._active_key.created_at = datetime.now(UTC) - timedelta(hours=100)
        assert mgr.needs_rotation is False

    def test_rotate_if_needed_no_rotation(self, manager: KeyRotationManager) -> None:
        """rotate_if_needed returns None when no rotation needed."""
        result = manager.rotate_if_needed()
        assert result is None

    def test_rotate_if_needed_triggers_rotation(self) -> None:
        """rotate_if_needed rotates when interval exceeded."""
        config = KeyRotationConfig(
            rotation_interval=timedelta(seconds=0),
            transition_period=timedelta(seconds=5),
            auto_rotate=True,
        )
        mgr = KeyRotationManager(config=config)
        mgr._active_key.created_at = datetime.now(UTC) - timedelta(seconds=1)
        old_key = mgr.active_key
        new_key = mgr.rotate_if_needed()
        assert new_key is not None
        assert new_key is not old_key
        assert mgr.active_key is new_key

    def test_sign_returns_version_and_sig(self, manager: KeyRotationManager) -> None:
        """sign returns (version_id, signature) tuple."""
        version_id, sig = manager.sign(b"data")
        assert version_id == manager.active_key.version_id
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_verify_with_active_key(self, manager: KeyRotationManager) -> None:
        """Verify succeeds with signature from active key."""
        _, sig = manager.sign(b"test")
        assert manager.verify(b"test", sig) is True

    def test_verify_with_previous_key(self, manager: KeyRotationManager) -> None:
        """Verify succeeds with signature from previous key during transition."""
        _, sig = manager.sign(b"test")
        old_version = manager.active_key.version_id
        manager.rotate()
        # Old key should still verify during transition
        assert manager.verify(b"test", sig, key_version=old_version) is True

    def test_verify_with_explicit_version(self, manager: KeyRotationManager) -> None:
        """Verify with explicit key version tries only that key."""
        _, sig = manager.sign(b"test")
        version = manager.active_key.version_id
        assert manager.verify(b"test", sig, key_version=version) is True

    def test_verify_unknown_version(self, manager: KeyRotationManager) -> None:
        """Verify with unknown version returns False."""
        assert manager.verify(b"test", "any-sig", key_version="v-unknown") is False

    def test_verify_expired_key_fails(self) -> None:
        """Verify fails for expired previous key."""
        config = KeyRotationConfig(
            transition_period=timedelta(seconds=0),
        )
        mgr = KeyRotationManager(config=config)
        _, sig = mgr.sign(b"test")
        old_version = mgr.active_key.version_id
        mgr.rotate()
        # Expire the old key immediately
        for k in mgr._previous_keys:
            if k.version_id == old_version:
                k.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert mgr.verify(b"test", sig, key_version=old_version) is False

    def test_verify_fallback_to_previous(self, manager: KeyRotationManager) -> None:
        """Verify without version tries active then previous keys."""
        _, sig = manager.sign(b"test")
        manager.rotate()
        # Without specifying version, should still find it via fallback
        assert manager.verify(b"test", sig) is True

    def test_verify_invalid_signature(self, manager: KeyRotationManager) -> None:
        """Verify returns False for invalid signature."""
        assert manager.verify(b"test", "0" * 64) is False

    def test_get_key_by_version_active(self, manager: KeyRotationManager) -> None:
        """get_key_by_version finds the active key."""
        key = manager.get_key_by_version(manager.active_key.version_id)
        assert key is manager.active_key

    def test_get_key_by_version_previous(self, manager: KeyRotationManager) -> None:
        """get_key_by_version finds previous keys."""
        old_version = manager.active_key.version_id
        manager.rotate()
        key = manager.get_key_by_version(old_version)
        assert key is not None
        assert key.version_id == old_version

    def test_get_key_by_version_unknown(self, manager: KeyRotationManager) -> None:
        """get_key_by_version returns None for unknown version."""
        assert manager.get_key_by_version("nonexistent") is None

    def test_revoke_key(self, manager: KeyRotationManager) -> None:
        """Revoke marks a key as revoked."""
        old_version = manager.active_key.version_id
        manager.rotate()
        result = manager.revoke_key(old_version)
        assert result is True
        key = manager.get_key_by_version(old_version)
        assert key is not None
        assert key.revoked is True

    def test_revoke_unknown_key(self, manager: KeyRotationManager) -> None:
        """Revoking an unknown key returns False."""
        assert manager.revoke_key("nonexistent") is False

    def test_revoke_active_key_triggers_rotation(self, manager: KeyRotationManager) -> None:
        """Revoking the active key triggers immediate rotation."""
        old_version = manager.active_key.version_id
        manager.revoke_key(old_version)
        assert manager.active_key.version_id != old_version
        assert manager.active_key.is_usable

    def test_revoked_key_not_in_valid(self, manager: KeyRotationManager) -> None:
        """Revoked previous key is excluded from valid keys."""
        old_version = manager.active_key.version_id
        manager.rotate()
        manager.revoke_key(old_version)
        valid = manager.all_valid_keys
        version_ids = [k.version_id for k in valid]
        assert old_version not in version_ids

    def test_rotation_callback(self, manager: KeyRotationManager) -> None:
        """Rotation callbacks are invoked."""
        called_with: list[tuple] = []
        manager.on_rotation(lambda new, old: called_with.append((new, old)))
        old_key = manager.active_key
        new_key = manager.rotate()
        assert len(called_with) == 1
        assert called_with[0] == (new_key, old_key)

    def test_rotation_callback_error_handled(self, manager: KeyRotationManager) -> None:
        """Failing callback doesn't prevent rotation."""

        def bad_callback(new: KeyVersion, old: KeyVersion | None) -> None:
            raise RuntimeError("callback error")

        manager.on_rotation(bad_callback)
        # Should not raise
        new_key = manager.rotate()
        assert new_key is not None

    def test_get_status(self, manager: KeyRotationManager) -> None:
        """get_status returns expected fields."""
        status = manager.get_status()
        assert "active_key" in status
        assert "previous_keys" in status
        assert "needs_rotation" in status
        assert "rotation_count" in status
        assert status["rotation_count"] == 0

    def test_get_status_after_rotation(self, manager: KeyRotationManager) -> None:
        """get_status updates after rotation."""
        manager.rotate()
        status = manager.get_status()
        assert status["rotation_count"] == 1
        assert status["valid_previous_keys"] >= 1

    def test_generated_keys_have_unique_versions(self) -> None:
        """Generated keys have unique version IDs."""
        mgr = KeyRotationManager()
        versions = {mgr.active_key.version_id}
        for _ in range(5):
            new_key = mgr.rotate()
            assert new_key.version_id not in versions
            versions.add(new_key.version_id)

    def test_generated_keys_have_correct_size(self) -> None:
        """Generated key material matches configured size."""
        config = KeyRotationConfig(key_size=64)
        mgr = KeyRotationManager(config=config)
        assert len(mgr.active_key.key_material) == 64


# =============================================================================
# GATEWAY NODE KEY ROTATION INTEGRATION TESTS
# =============================================================================


class TestGatewayKeyRotationIntegration:
    """Tests for key rotation integrated into GatewayNode."""

    def test_gateway_has_key_manager(self, gateway: GatewayNode) -> None:
        """GatewayNode initializes with a KeyRotationManager."""
        assert gateway.key_rotation_manager is not None
        assert isinstance(gateway.key_rotation_manager, KeyRotationManager)

    def test_gateway_sign_message(self, gateway: GatewayNode) -> None:
        """GatewayNode can sign messages."""
        version_id, sig = gateway.sign_message(b"test data")
        assert isinstance(version_id, str)
        assert isinstance(sig, str)
        assert len(sig) == 64

    def test_gateway_verify_message(self, gateway: GatewayNode) -> None:
        """GatewayNode can verify messages it signed."""
        version_id, sig = gateway.sign_message(b"test data")
        assert gateway.verify_message(b"test data", sig) is True
        assert gateway.verify_message(b"test data", sig, key_version=version_id) is True

    def test_gateway_verify_invalid_message(self, gateway: GatewayNode) -> None:
        """GatewayNode rejects invalid signatures."""
        assert gateway.verify_message(b"test", "bad") is False

    def test_gateway_rotate_keys(self, gateway: GatewayNode) -> None:
        """GatewayNode.rotate_keys triggers key rotation."""
        old_version = gateway.key_rotation_manager.active_key.version_id
        new_key = gateway.rotate_keys()
        assert new_key.version_id != old_version
        assert gateway.key_rotation_manager.active_key is new_key

    def test_gateway_rotate_keys_creates_audit_entry(self, gateway: GatewayNode) -> None:
        """Key rotation creates an audit entry."""
        gateway.rotate_keys()
        audit = gateway.get_audit_log(event_type=AuditEventType.KEY_ROTATED)
        assert len(audit) >= 1
        entry = audit[0]
        assert entry.event_type == AuditEventType.KEY_ROTATED
        assert "new_version" in entry.metadata

    def test_gateway_verify_after_rotation(self, gateway: GatewayNode) -> None:
        """Messages signed before rotation still verify during transition."""
        version_id, sig = gateway.sign_message(b"pre-rotation")
        gateway.rotate_keys()
        # Should still verify via fallback
        assert gateway.verify_message(b"pre-rotation", sig) is True
        # Should still verify with explicit version
        assert gateway.verify_message(b"pre-rotation", sig, key_version=version_id) is True

    def test_gateway_auto_rotation_on_sign(self, gateway: GatewayNode) -> None:
        """sign_message auto-rotates when interval exceeded."""
        old_key = gateway.key_rotation_manager.active_key
        old_version = old_key.version_id
        # Backdate the key and set a very short rotation interval
        old_key.created_at = datetime.now(UTC) - timedelta(hours=1)
        gateway.key_rotation_manager.config.rotation_interval = timedelta(seconds=0)

        version_id, sig = gateway.sign_message(b"data")
        # Should have auto-rotated
        assert version_id != old_version

    def test_gateway_stats_include_key_rotation(self, gateway: GatewayNode) -> None:
        """get_stats includes key rotation information."""
        gateway.status = GatewayStatus.ACTIVE
        stats = gateway.get_stats()
        assert "key_rotation" in stats
        kr = stats["key_rotation"]
        assert "active_key" in kr
        assert "needs_rotation" in kr
        assert "rotation_count" in kr

    def test_gateway_to_dict_includes_key_version(self, gateway: GatewayNode) -> None:
        """to_dict includes active key version ID."""
        d = gateway.to_dict()
        assert "active_key_version" in d
        assert d["active_key_version"] == gateway.key_rotation_manager.active_key.version_id

    def test_create_gateway_with_key_config(self) -> None:
        """create_gateway accepts key_rotation_config."""
        config = KeyRotationConfig(
            rotation_interval=timedelta(hours=12),
        )
        gw = create_gateway(
            federation_id=uuid4(),
            endpoint="https://test.example.com/v1",
            key_rotation_config=config,
            register=False,
        )
        assert gw.key_rotation_manager.config.rotation_interval == timedelta(hours=12)

    def test_cross_gateway_verification(self) -> None:
        """Signature from one gateway can be verified if key is shared."""
        gw1 = GatewayNode(
            federation_id=uuid4(),
            endpoint="https://gw1.example.com/v1",
        )
        # In real usage, key material would be exchanged via the protocol.
        # Here we simulate by giving gw2 a copy of gw1's active key.
        gw1_key = gw1.key_rotation_manager.active_key
        gw2 = GatewayNode(
            federation_id=uuid4(),
            endpoint="https://gw2.example.com/v1",
            key_rotation_config=KeyRotationConfig(),
        )
        # Inject gw1's key into gw2's manager for verification
        gw2.key_rotation_manager._previous_keys.append(
            KeyVersion(
                version_id=gw1_key.version_id,
                key_material=gw1_key.key_material,
            )
        )

        version_id, sig = gw1.sign_message(b"cross-gateway-data")
        assert gw2.verify_message(b"cross-gateway-data", sig, key_version=version_id) is True

    def test_multiple_rotations_transition(self, gateway: GatewayNode) -> None:
        """Signatures from multiple previous keys verify during transition."""
        sigs = []
        for i in range(3):
            version_id, sig = gateway.sign_message(f"message-{i}".encode())
            sigs.append((version_id, sig, f"message-{i}".encode()))
            gateway.rotate_keys()

        # All should still verify (within transition period)
        for version_id, sig, data in sigs:
            assert gateway.verify_message(data, sig, key_version=version_id) is True


# =============================================================================
# KEY ROTATION EXCEPTION TESTS
# =============================================================================


class TestKeyRotationException:
    """Tests for KeyRotationException."""

    def test_basic_exception(self) -> None:
        """KeyRotationException stores message and version."""
        exc = KeyRotationException("test error", key_version="v1")
        assert str(exc) == "test error"
        assert exc.key_version == "v1"

    def test_exception_without_version(self) -> None:
        """KeyRotationException works without key version."""
        exc = KeyRotationException("generic error")
        assert exc.key_version is None
