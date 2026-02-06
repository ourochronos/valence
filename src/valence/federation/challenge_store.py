"""Challenge Store backends for auth challenge persistence.

Provides pluggable storage for pending authentication challenges.
Default is in-memory; Redis backend available for production use
where server restarts must not lose pending challenges.

Configure via environment variables:
    VALENCE_CHALLENGE_STORE=memory|redis  (default: memory)
    VALENCE_REDIS_URL=redis://localhost:6379  (default)

See: protocol.py for usage in auth challenge/verify flow.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime

try:
    import redis
except ImportError:  # pragma: no cover
    redis = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Key prefix for Redis to avoid collisions
_REDIS_KEY_PREFIX = "valence:auth_challenge:"


class AuthChallengeStore(ABC):
    """Abstract interface for auth challenge storage.

    Stores pending authentication challenges keyed by client DID.
    Each challenge has a nonce string and an expiration time.
    """

    @abstractmethod
    def store_challenge(
        self,
        client_did: str,
        challenge: str,
        expires_at: datetime,
    ) -> None:
        """Store a pending auth challenge.

        Args:
            client_did: The DID of the client.
            challenge: The challenge nonce string.
            expires_at: When the challenge expires.
        """
        ...

    @abstractmethod
    def get_challenge(self, client_did: str) -> tuple[str, datetime] | None:
        """Retrieve a pending challenge for a client DID.

        Args:
            client_did: The DID to look up.

        Returns:
            Tuple of (challenge_nonce, expires_at) or None if not found.
        """
        ...

    @abstractmethod
    def delete_challenge(self, client_did: str) -> None:
        """Remove a pending challenge.

        Args:
            client_did: The DID whose challenge to remove.
        """
        ...

    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove expired challenges.

        Returns:
            Number of challenges removed.
        """
        ...


class MemoryAuthChallengeStore(AuthChallengeStore):
    """In-memory auth challenge store.

    Suitable for development and single-process deployments.
    Challenges are lost on server restart.
    """

    def __init__(self) -> None:
        self._challenges: dict[str, tuple[str, datetime]] = {}

    def store_challenge(
        self,
        client_did: str,
        challenge: str,
        expires_at: datetime,
    ) -> None:
        self._challenges[client_did] = (challenge, expires_at)

    def get_challenge(self, client_did: str) -> tuple[str, datetime] | None:
        entry = self._challenges.get(client_did)
        if entry is None:
            return None
        return entry

    def delete_challenge(self, client_did: str) -> None:
        self._challenges.pop(client_did, None)

    def cleanup_expired(self) -> int:
        now = datetime.now()
        expired = [did for did, (_, exp) in self._challenges.items() if now > exp]
        for did in expired:
            del self._challenges[did]
        return len(expired)

    def __contains__(self, client_did: str) -> bool:
        """Support 'did in store' syntax for convenience."""
        return client_did in self._challenges

    def clear(self) -> None:
        """Clear all challenges (useful for testing)."""
        self._challenges.clear()


class RedisAuthChallengeStore(AuthChallengeStore):
    """Redis-backed auth challenge store.

    Uses Redis TTL for automatic expiry. Survives server restarts.
    Requires redis-py: ``pip install valence[redis]``

    Connection configured via VALENCE_REDIS_URL environment variable
    (default: redis://localhost:6379).
    """

    def __init__(self, redis_url: str | None = None) -> None:
        if redis is None:
            raise ImportError("redis package is required for RedisAuthChallengeStore. Install with: pip install valence[redis]")

        url = redis_url or os.environ.get("VALENCE_REDIS_URL", "redis://localhost:6379")
        self._client = redis.Redis.from_url(url, decode_responses=True)
        # Verify connection
        try:
            self._client.ping()
        except redis.ConnectionError:
            logger.warning("Redis connection failed at init — will retry on use")

    def _key(self, client_did: str) -> str:
        return f"{_REDIS_KEY_PREFIX}{client_did}"

    def store_challenge(
        self,
        client_did: str,
        challenge: str,
        expires_at: datetime,
    ) -> None:
        key = self._key(client_did)
        data = json.dumps(
            {
                "challenge": challenge,
                "expires_at": expires_at.isoformat(),
            }
        )
        # Compute TTL in seconds from now
        ttl_seconds = max(1, int((expires_at - datetime.now()).total_seconds()))
        self._client.setex(key, ttl_seconds, data)

    def get_challenge(self, client_did: str) -> tuple[str, datetime] | None:
        key = self._key(client_did)
        raw = self._client.get(key)
        if raw is None:
            return None
        data = json.loads(raw)
        return (data["challenge"], datetime.fromisoformat(data["expires_at"]))

    def delete_challenge(self, client_did: str) -> None:
        key = self._key(client_did)
        self._client.delete(key)

    def cleanup_expired(self) -> int:
        # Redis TTL handles expiry automatically; this is a no-op.
        # We scan for any keys that might be past expiry but still present
        # (shouldn't happen with TTL, but belt-and-suspenders).
        count = 0
        cursor = 0
        while True:
            cursor, keys = self._client.scan(cursor, match=f"{_REDIS_KEY_PREFIX}*", count=100)
            for key in keys:
                raw = self._client.get(key)
                if raw is not None:
                    data = json.loads(raw)
                    exp = datetime.fromisoformat(data["expires_at"])
                    if datetime.now() > exp:
                        self._client.delete(key)
                        count += 1
            if cursor == 0:
                break
        return count


# =============================================================================
# FACTORY
# =============================================================================

_store_instance: AuthChallengeStore | None = None


def get_auth_challenge_store() -> AuthChallengeStore:
    """Get or create the global auth challenge store.

    Reads VALENCE_CHALLENGE_STORE environment variable:
        - "memory" (default): In-memory store
        - "redis": Redis-backed store

    Returns:
        The configured AuthChallengeStore instance.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance

    backend = os.environ.get("VALENCE_CHALLENGE_STORE", "memory").lower()

    if backend == "redis":
        logger.info("Using Redis auth challenge store")
        _store_instance = RedisAuthChallengeStore()
    elif backend == "memory":
        logger.info("Using in-memory auth challenge store")
        _store_instance = MemoryAuthChallengeStore()
    else:
        logger.warning(f"Unknown challenge store backend '{backend}', falling back to memory")
        _store_instance = MemoryAuthChallengeStore()

    return _store_instance


def reset_auth_challenge_store() -> None:
    """Reset the global store instance (for testing)."""
    global _store_instance
    _store_instance = None
