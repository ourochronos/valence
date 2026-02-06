"""Belief-level nonce tracking for replay protection.

Provides defense-in-depth against replay attacks at the belief layer.
Transport-level protections (TLS, session tokens) are the primary defense;
this module adds an additional layer by tracking per-origin nonces with TTL.

Usage:
    tracker = get_nonce_tracker()

    # On receive
    if tracker.is_seen(origin_node_did, belief.nonce):
        reject("Replayed belief")
    tracker.record_nonce(origin_node_did, belief.nonce)

    # On send
    belief.nonce = generate_nonce()
"""

from __future__ import annotations

import logging
import secrets
import threading
import time

logger = logging.getLogger(__name__)

# Default TTL for nonces (1 hour)
DEFAULT_NONCE_TTL_SECONDS = 3600

# Default max nonces per origin before forced cleanup
DEFAULT_MAX_NONCES_PER_ORIGIN = 10000


def generate_nonce() -> str:
    """Generate a cryptographically random nonce for belief replay protection.

    Returns:
        A 32-character hex string (128 bits of entropy).
    """
    return secrets.token_hex(16)


class NonceTracker:
    """Tracks seen nonces per origin node with TTL-based expiry.

    Thread-safe in-memory nonce store. Each nonce is associated with
    an origin node ID and expires after a configurable TTL.

    Args:
        ttl_seconds: Time-to-live for nonces in seconds. Nonces older
            than this are eligible for cleanup.
        max_per_origin: Maximum nonces stored per origin before forced
            cleanup is triggered.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_NONCE_TTL_SECONDS,
        max_per_origin: int = DEFAULT_MAX_NONCES_PER_ORIGIN,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_per_origin = max_per_origin
        # origin_node_id -> {nonce -> timestamp}
        self._seen: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    @property
    def ttl_seconds(self) -> int:
        """Return the configured TTL in seconds."""
        return self._ttl

    def record_nonce(self, origin_node_id: str, nonce: str) -> None:
        """Record a nonce as seen for a given origin node.

        Args:
            origin_node_id: DID or identifier of the origin node.
            nonce: The nonce string to record.
        """
        with self._lock:
            if origin_node_id not in self._seen:
                self._seen[origin_node_id] = {}

            self._seen[origin_node_id][nonce] = time.monotonic()

            # If we've exceeded max per origin, do a cleanup pass
            if len(self._seen[origin_node_id]) > self._max_per_origin:
                self._cleanup_origin(origin_node_id)

    def is_seen(self, origin_node_id: str, nonce: str) -> bool:
        """Check if a nonce has already been seen for a given origin.

        Expired nonces are treated as unseen. This method does NOT
        remove expired entries — use cleanup() for that.

        Args:
            origin_node_id: DID or identifier of the origin node.
            nonce: The nonce string to check.

        Returns:
            True if the nonce was previously recorded and has not expired.
        """
        with self._lock:
            origin_nonces = self._seen.get(origin_node_id)
            if origin_nonces is None:
                return False

            timestamp = origin_nonces.get(nonce)
            if timestamp is None:
                return False

            # Check if expired
            if time.monotonic() - timestamp > self._ttl:
                # Expired — remove and return False
                del origin_nonces[nonce]
                if not origin_nonces:
                    del self._seen[origin_node_id]
                return False

            return True

    def cleanup(self) -> int:
        """Remove all expired nonces across all origins.

        Returns:
            Number of nonces removed.
        """
        removed = 0
        now = time.monotonic()

        with self._lock:
            empty_origins = []

            for origin_id, nonces in self._seen.items():
                expired_keys = [nonce for nonce, ts in nonces.items() if now - ts > self._ttl]
                for key in expired_keys:
                    del nonces[key]
                    removed += 1

                if not nonces:
                    empty_origins.append(origin_id)

            for origin_id in empty_origins:
                del self._seen[origin_id]

        if removed > 0:
            logger.debug(f"Nonce cleanup: removed {removed} expired nonces")

        return removed

    def _cleanup_origin(self, origin_node_id: str) -> None:
        """Remove expired nonces for a single origin (caller must hold lock)."""
        nonces = self._seen.get(origin_node_id)
        if nonces is None:
            return

        now = time.monotonic()
        expired_keys = [nonce for nonce, ts in nonces.items() if now - ts > self._ttl]
        for key in expired_keys:
            del nonces[key]

        if not nonces:
            del self._seen[origin_node_id]

    def origin_count(self) -> int:
        """Return the number of tracked origins."""
        with self._lock:
            return len(self._seen)

    def nonce_count(self, origin_node_id: str | None = None) -> int:
        """Return the number of tracked nonces.

        Args:
            origin_node_id: If provided, count only nonces for this origin.
                Otherwise, count all nonces across all origins.

        Returns:
            Number of tracked (non-expired, but not cleaned) nonces.
        """
        with self._lock:
            if origin_node_id is not None:
                nonces = self._seen.get(origin_node_id, {})
                return len(nonces)
            return sum(len(n) for n in self._seen.values())

    def clear(self) -> None:
        """Remove all tracked nonces."""
        with self._lock:
            self._seen.clear()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_nonce_tracker: NonceTracker | None = None


def get_nonce_tracker() -> NonceTracker:
    """Get the global NonceTracker instance.

    Returns a module-level singleton. Use reset_nonce_tracker() in tests.
    """
    global _nonce_tracker
    if _nonce_tracker is None:
        _nonce_tracker = NonceTracker()
    return _nonce_tracker


def reset_nonce_tracker() -> None:
    """Reset the global NonceTracker (for testing)."""
    global _nonce_tracker
    _nonce_tracker = None
