"""LRU (Least Recently Used) cache utilities for bounded in-memory caching.

Issue #147: Prevent unbounded memory growth in in-memory caches.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Optional, TypeVar

# Default max size for LRU caches (configurable via VALENCE_CACHE_MAX_SIZE)
DEFAULT_CACHE_MAX_SIZE = 1000

K = TypeVar("K")
V = TypeVar("V")


def get_cache_max_size() -> int:
    """Get the configured cache max size from config."""
    from .config import get_config
    try:
        return get_config().cache_max_size
    except Exception:
        return DEFAULT_CACHE_MAX_SIZE


class LRUDict(dict[K, V]):
    """
    A dictionary with LRU (Least Recently Used) eviction policy.

    When the cache exceeds max_size, the least recently accessed items
    are evicted to maintain the size limit.

    Thread-safe for concurrent access.

    Example:
        cache = LRUDict(max_size=100)
        cache["key1"] = "value1"  # Adds item
        cache["key1"]  # Accessing moves key1 to most recent
        # When cache exceeds 100 items, oldest items are evicted
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items. If None, uses VALENCE_CACHE_MAX_SIZE
                      env var or DEFAULT_CACHE_MAX_SIZE.
        """
        super().__init__()
        self._max_size = max_size if max_size is not None else get_cache_max_size()
        self._order: OrderedDict[K, None] = OrderedDict()
        self._lock = threading.RLock()

        # Initialize from args/kwargs if provided
        if args or kwargs:
            initial_data = dict(*args, **kwargs)
            for k, v in initial_data.items():
                self[k] = v

    @property
    def max_size(self) -> int:
        """Maximum cache size."""
        return self._max_size

    def __setitem__(self, key: K, value: V) -> None:
        """Set item and update access order."""
        with self._lock:
            # If key exists, remove from order tracking
            if key in self._order:
                self._order.move_to_end(key)
            else:
                self._order[key] = None

            super().__setitem__(key, value)
            self._evict_if_needed()

    def __getitem__(self, key: K) -> V:
        """Get item and mark as recently used."""
        with self._lock:
            value = super().__getitem__(key)
            # Move to end (most recently used)
            if key in self._order:
                self._order.move_to_end(key)
            return value

    def __delitem__(self, key: K) -> None:
        """Delete item."""
        with self._lock:
            super().__delitem__(key)
            self._order.pop(key, None)

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get item without updating access order (peek)."""
        with self._lock:
            return super().get(key, default)

    def peek(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get item without updating access order."""
        return self.get(key, default)

    def pop(self, key: K, *args: Any) -> V:
        """Remove and return item."""
        with self._lock:
            self._order.pop(key, None)
            return super().pop(key, *args)

    def clear(self) -> None:
        """Clear all items."""
        with self._lock:
            super().clear()
            self._order.clear()

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update from dict/iterable."""
        with self._lock:
            if args:
                other = dict(args[0]) if hasattr(args[0], "keys") else dict(args[0])
                for k, v in other.items():
                    self[k] = v
            for k, v in kwargs.items():
                self[k] = v

    def _evict_if_needed(self) -> None:
        """Evict oldest items if over max size."""
        while len(self._order) > self._max_size:
            # Pop oldest (first) item
            oldest_key = next(iter(self._order))
            self._order.pop(oldest_key)
            super().pop(oldest_key, None)

    def __iter__(self) -> Iterator[K]:
        """Iterate over keys in order (oldest to newest)."""
        with self._lock:
            return iter(list(self._order.keys()))

    def keys(self) -> Any:
        """Return keys in LRU order."""
        with self._lock:
            return list(self._order.keys())

    def values(self) -> Any:
        """Return values in LRU order."""
        with self._lock:
            return [super(LRUDict, self).get(k) for k in self._order.keys()]

    def items(self) -> Any:
        """Return items in LRU order."""
        with self._lock:
            return [(k, super(LRUDict, self).get(k)) for k in self._order.keys()]

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            return {
                "size": len(self),
                "max_size": self._max_size,
                "utilization": len(self) / self._max_size if self._max_size > 0 else 0,
            }


class BoundedList(list):
    """
    A list with a maximum size limit.

    When items are appended and the list exceeds max_size,
    the oldest items (from the front) are removed.

    Thread-safe for concurrent access.

    Example:
        events = BoundedList(max_size=100)
        events.append(event)  # Adds event
        # When list exceeds 100 items, oldest items are removed
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        initial: Optional[list] = None,
    ) -> None:
        """
        Initialize bounded list.

        Args:
            max_size: Maximum number of items. If None, uses VALENCE_CACHE_MAX_SIZE
                      env var or DEFAULT_CACHE_MAX_SIZE.
            initial: Optional initial list of items.
        """
        super().__init__(initial or [])
        self._max_size = max_size if max_size is not None else get_cache_max_size()
        self._lock = threading.RLock()
        self._trim_if_needed()

    @property
    def max_size(self) -> int:
        """Maximum list size."""
        return self._max_size

    def append(self, item: Any) -> None:
        """Append item and trim if needed."""
        with self._lock:
            super().append(item)
            self._trim_if_needed()

    def extend(self, items: Any) -> None:
        """Extend with items and trim if needed."""
        with self._lock:
            super().extend(items)
            self._trim_if_needed()

    def insert(self, index: int, item: Any) -> None:
        """Insert item and trim if needed."""
        with self._lock:
            super().insert(index, item)
            self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        """Remove oldest items if over max size."""
        while len(self) > self._max_size:
            super().pop(0)

    def stats(self) -> dict[str, Any]:
        """Return list statistics."""
        with self._lock:
            return {
                "size": len(self),
                "max_size": self._max_size,
                "utilization": len(self) / self._max_size if self._max_size > 0 else 0,
            }
