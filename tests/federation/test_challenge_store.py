"""Tests for auth challenge store backends.

Tests both MemoryAuthChallengeStore and RedisAuthChallengeStore.
Redis tests use unittest.mock to avoid requiring a running Redis instance.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from valence.federation.challenge_store import (
    _REDIS_KEY_PREFIX,
    AuthChallengeStore,
    MemoryAuthChallengeStore,
    RedisAuthChallengeStore,
    get_auth_challenge_store,
    reset_auth_challenge_store,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_store():
    """Reset global store between tests."""
    reset_auth_challenge_store()
    yield
    reset_auth_challenge_store()


@pytest.fixture
def memory_store():
    return MemoryAuthChallengeStore()


def _make_mock_redis_client():
    """Create a mock Redis client that behaves like a dict-backed store."""
    client = MagicMock()
    storage: dict[str, tuple[str, int | None]] = {}

    def mock_setex(key, ttl, value):
        storage[key] = (value, ttl)

    def mock_get(key):
        entry = storage.get(key)
        return entry[0] if entry else None

    def mock_delete(*keys):
        for key in keys:
            storage.pop(key, None)

    def mock_ping():
        return True

    def mock_scan(cursor, match=None, count=100):
        prefix = match.replace("*", "") if match else ""
        matched = [k for k in storage if k.startswith(prefix)]
        return (0, matched)

    client.setex.side_effect = mock_setex
    client.get.side_effect = mock_get
    client.delete.side_effect = mock_delete
    client.ping.side_effect = mock_ping
    client.scan.side_effect = mock_scan

    return client, storage


@pytest.fixture
def mock_redis_client():
    return _make_mock_redis_client()


@pytest.fixture
def redis_store(mock_redis_client):
    """Create a RedisAuthChallengeStore with a mocked Redis client."""
    client, _ = mock_redis_client
    store = RedisAuthChallengeStore.__new__(RedisAuthChallengeStore)
    store._client = client
    return store


# =============================================================================
# MEMORY STORE TESTS
# =============================================================================


class TestMemoryAuthChallengeStore:
    """Tests for in-memory challenge store."""

    def test_store_and_retrieve(self, memory_store):
        did = "did:vkb:web:test.example.com"
        challenge = "abc123"
        expires = datetime.now() + timedelta(minutes=5)

        memory_store.store_challenge(did, challenge, expires)
        result = memory_store.get_challenge(did)

        assert result is not None
        assert result[0] == challenge
        assert result[1] == expires

    def test_get_nonexistent(self, memory_store):
        result = memory_store.get_challenge("did:vkb:web:nobody.example.com")
        assert result is None

    def test_delete_challenge(self, memory_store):
        did = "did:vkb:web:test.example.com"
        memory_store.store_challenge(did, "abc", datetime.now() + timedelta(minutes=5))

        memory_store.delete_challenge(did)
        assert memory_store.get_challenge(did) is None

    def test_delete_nonexistent(self, memory_store):
        # Should not raise
        memory_store.delete_challenge("did:vkb:web:nobody.example.com")

    def test_overwrite_challenge(self, memory_store):
        did = "did:vkb:web:test.example.com"
        memory_store.store_challenge(did, "first", datetime.now() + timedelta(minutes=5))
        memory_store.store_challenge(did, "second", datetime.now() + timedelta(minutes=5))

        result = memory_store.get_challenge(did)
        assert result is not None
        assert result[0] == "second"

    def test_cleanup_expired(self, memory_store):
        did_expired = "did:vkb:web:expired.example.com"
        did_valid = "did:vkb:web:valid.example.com"

        memory_store.store_challenge(did_expired, "old", datetime.now() - timedelta(minutes=1))
        memory_store.store_challenge(did_valid, "new", datetime.now() + timedelta(minutes=5))

        removed = memory_store.cleanup_expired()
        assert removed == 1
        assert memory_store.get_challenge(did_expired) is None
        assert memory_store.get_challenge(did_valid) is not None

    def test_cleanup_all_expired(self, memory_store):
        for i in range(5):
            memory_store.store_challenge(
                f"did:vkb:web:{i}.example.com",
                f"c{i}",
                datetime.now() - timedelta(minutes=1),
            )

        removed = memory_store.cleanup_expired()
        assert removed == 5

    def test_cleanup_none_expired(self, memory_store):
        memory_store.store_challenge("did:vkb:web:a.example.com", "c", datetime.now() + timedelta(minutes=5))
        removed = memory_store.cleanup_expired()
        assert removed == 0

    def test_contains(self, memory_store):
        did = "did:vkb:web:test.example.com"
        assert did not in memory_store
        memory_store.store_challenge(did, "c", datetime.now() + timedelta(minutes=5))
        assert did in memory_store

    def test_clear(self, memory_store):
        memory_store.store_challenge("did:vkb:web:a.example.com", "a", datetime.now() + timedelta(minutes=5))
        memory_store.store_challenge("did:vkb:web:b.example.com", "b", datetime.now() + timedelta(minutes=5))

        memory_store.clear()
        assert memory_store.get_challenge("did:vkb:web:a.example.com") is None
        assert memory_store.get_challenge("did:vkb:web:b.example.com") is None

    def test_is_abstract_subclass(self, memory_store):
        assert isinstance(memory_store, AuthChallengeStore)


# =============================================================================
# REDIS STORE TESTS (MOCKED)
# =============================================================================


class TestRedisAuthChallengeStore:
    """Tests for Redis-backed challenge store with mocked Redis client."""

    def test_store_and_retrieve(self, redis_store):
        did = "did:vkb:web:test.example.com"
        challenge = "abc123"
        expires = datetime.now() + timedelta(minutes=5)

        redis_store.store_challenge(did, challenge, expires)
        result = redis_store.get_challenge(did)

        assert result is not None
        assert result[0] == challenge
        # Datetime roundtrips through ISO format
        assert result[1].isoformat() == expires.isoformat()

    def test_get_nonexistent(self, redis_store):
        result = redis_store.get_challenge("did:vkb:web:nobody.example.com")
        assert result is None

    def test_delete_challenge(self, redis_store):
        did = "did:vkb:web:test.example.com"
        redis_store.store_challenge(did, "abc", datetime.now() + timedelta(minutes=5))

        redis_store.delete_challenge(did)
        assert redis_store.get_challenge(did) is None

    def test_key_prefix(self, redis_store):
        did = "did:vkb:web:test.example.com"
        expected_key = f"{_REDIS_KEY_PREFIX}{did}"

        redis_store.store_challenge(did, "abc", datetime.now() + timedelta(minutes=5))

        call_args = redis_store._client.setex.call_args
        assert call_args[0][0] == expected_key

    def test_ttl_calculation(self, redis_store):
        did = "did:vkb:web:test.example.com"
        expires = datetime.now() + timedelta(minutes=5)

        redis_store.store_challenge(did, "abc", expires)

        call_args = redis_store._client.setex.call_args
        ttl = call_args[0][1]
        # TTL should be approximately 300 seconds (5 minutes)
        assert 295 <= ttl <= 305

    def test_json_serialization(self, redis_store):
        did = "did:vkb:web:test.example.com"
        challenge = "abc123"
        expires = datetime.now() + timedelta(minutes=5)

        redis_store.store_challenge(did, challenge, expires)

        call_args = redis_store._client.setex.call_args
        stored_json = call_args[0][2]
        data = json.loads(stored_json)
        assert data["challenge"] == challenge
        assert data["expires_at"] == expires.isoformat()

    def test_overwrite_challenge(self, redis_store):
        did = "did:vkb:web:test.example.com"
        redis_store.store_challenge(did, "first", datetime.now() + timedelta(minutes=5))
        redis_store.store_challenge(did, "second", datetime.now() + timedelta(minutes=5))

        result = redis_store.get_challenge(did)
        assert result is not None
        assert result[0] == "second"

    def test_is_abstract_subclass(self, redis_store):
        assert isinstance(redis_store, AuthChallengeStore)

    def test_import_error_without_redis(self):
        """Test that a helpful error is raised if redis is not installed."""
        with patch("valence.federation.challenge_store.redis", None):
            with pytest.raises(ImportError, match="redis package is required"):
                RedisAuthChallengeStore()

    def test_cleanup_expired_noop(self, redis_store):
        """Redis TTL handles expiry; cleanup returns 0 for non-expired."""
        did = "did:vkb:web:test.example.com"
        expires = datetime.now() + timedelta(minutes=5)
        redis_store.store_challenge(did, "abc", expires)

        removed = redis_store.cleanup_expired()
        assert removed == 0

    def test_cleanup_expired_removes_past(self, redis_store):
        """cleanup_expired should remove entries past their expires_at."""
        did = "did:vkb:web:expired.example.com"
        expires = datetime.now() - timedelta(minutes=1)
        redis_store.store_challenge(did, "old", expires)

        removed = redis_store.cleanup_expired()
        assert removed == 1


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestGetAuthChallengeStore:
    """Tests for the factory function."""

    def test_default_is_memory(self):
        os.environ.pop("VALENCE_CHALLENGE_STORE", None)
        reset_auth_challenge_store()

        store = get_auth_challenge_store()
        assert isinstance(store, MemoryAuthChallengeStore)

    def test_explicit_memory(self):
        with patch.dict(os.environ, {"VALENCE_CHALLENGE_STORE": "memory"}):
            reset_auth_challenge_store()
            store = get_auth_challenge_store()
            assert isinstance(store, MemoryAuthChallengeStore)

    def test_unknown_backend_falls_back_to_memory(self):
        with patch.dict(os.environ, {"VALENCE_CHALLENGE_STORE": "postgres"}):
            reset_auth_challenge_store()
            store = get_auth_challenge_store()
            assert isinstance(store, MemoryAuthChallengeStore)

    def test_redis_backend(self):
        """Test that redis backend is instantiated when configured."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch.dict(os.environ, {"VALENCE_CHALLENGE_STORE": "redis"}):
            reset_auth_challenge_store()
            with patch("valence.federation.challenge_store.redis") as mock_redis_mod:
                mock_redis_mod.Redis.from_url.return_value = mock_client
                mock_redis_mod.ConnectionError = ConnectionError

                store = get_auth_challenge_store()
                assert isinstance(store, RedisAuthChallengeStore)

    def test_singleton_behavior(self):
        store1 = get_auth_challenge_store()
        store2 = get_auth_challenge_store()
        assert store1 is store2

    def test_reset_clears_singleton(self):
        store1 = get_auth_challenge_store()
        reset_auth_challenge_store()
        store2 = get_auth_challenge_store()
        assert store1 is not store2

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"VALENCE_CHALLENGE_STORE": "MEMORY"}):
            reset_auth_challenge_store()
            store = get_auth_challenge_store()
            assert isinstance(store, MemoryAuthChallengeStore)
