"""Tests for LRU cache utilities (Issue #147)."""

import os
import threading
import time
from unittest.mock import patch

from valence.core.lru_cache import (
    DEFAULT_CACHE_MAX_SIZE,
    BoundedList,
    LRUDict,
    get_cache_max_size,
)


class TestGetCacheMaxSize:
    """Tests for get_cache_max_size()."""

    def test_default_value(self):
        """Should return default when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear VALENCE_CACHE_MAX_SIZE if present
            os.environ.pop("VALENCE_CACHE_MAX_SIZE", None)
            assert get_cache_max_size() == DEFAULT_CACHE_MAX_SIZE

    def test_env_var_override(self):
        """Should return env var value when set."""
        with patch.dict(os.environ, {"VALENCE_CACHE_MAX_SIZE": "500"}):
            assert get_cache_max_size() == 500

    def test_invalid_env_var(self):
        """Should return default for invalid env var."""
        with patch.dict(os.environ, {"VALENCE_CACHE_MAX_SIZE": "not_a_number"}):
            assert get_cache_max_size() == DEFAULT_CACHE_MAX_SIZE


class TestLRUDict:
    """Tests for LRUDict."""

    def test_basic_set_get(self):
        """Basic set and get operations."""
        cache = LRUDict(max_size=10)
        cache["key1"] = "value1"
        assert cache["key1"] == "value1"

    def test_eviction_on_overflow(self):
        """Should evict oldest items when exceeding max_size."""
        cache = LRUDict(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        assert len(cache) == 3

        # Adding 4th item should evict oldest ("a")
        cache["d"] = 4

        assert len(cache) == 3
        assert "a" not in cache
        assert "b" in cache
        assert "c" in cache
        assert "d" in cache

    def test_access_updates_lru_order(self):
        """Accessing an item should make it most recently used."""
        cache = LRUDict(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Access "a" to make it most recent
        _ = cache["a"]

        # Adding new item should evict "b" (now oldest)
        cache["d"] = 4

        assert "a" in cache  # Was accessed, so kept
        assert "b" not in cache  # Oldest, evicted
        assert "c" in cache
        assert "d" in cache

    def test_update_existing_key(self):
        """Updating existing key should make it most recent."""
        cache = LRUDict(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Update "a"
        cache["a"] = 100

        # Adding new item should evict "b"
        cache["d"] = 4

        assert cache["a"] == 100
        assert "b" not in cache

    def test_delete(self):
        """Deleting items."""
        cache = LRUDict(max_size=10)
        cache["a"] = 1
        cache["b"] = 2

        del cache["a"]

        assert "a" not in cache
        assert "b" in cache
        assert len(cache) == 1

    def test_pop(self):
        """Pop should remove and return item."""
        cache = LRUDict(max_size=10)
        cache["a"] = 1

        val = cache.pop("a")

        assert val == 1
        assert "a" not in cache

    def test_clear(self):
        """Clear should remove all items."""
        cache = LRUDict(max_size=10)
        cache["a"] = 1
        cache["b"] = 2

        cache.clear()

        assert len(cache) == 0

    def test_peek_does_not_update_order(self):
        """peek/get should not update LRU order."""
        cache = LRUDict(max_size=3)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Peek at "a" without updating order
        val = cache.peek("a")
        assert val == 1

        # Adding new item should still evict "a" (still oldest)
        cache["d"] = 4

        assert "a" not in cache

    def test_stats(self):
        """Stats should return correct info."""
        cache = LRUDict(max_size=10)
        cache["a"] = 1
        cache["b"] = 2

        stats = cache.stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["utilization"] == 0.2

    def test_iteration_order(self):
        """Iteration should be in LRU order (oldest first)."""
        cache = LRUDict(max_size=10)
        cache["a"] = 1
        cache["b"] = 2
        cache["c"] = 3

        # Access "a" to make it newest
        _ = cache["a"]

        keys = list(cache.keys())
        assert keys == ["b", "c", "a"]

    def test_thread_safety(self):
        """Should be thread-safe for concurrent access."""
        cache = LRUDict(max_size=100)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 50):
                    cache[f"key{i}"] = i
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    _ = list(cache.keys())
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(0,)),
            threading.Thread(target=writer, args=(50,)),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_init_with_data(self):
        """Should initialize with existing data."""
        cache = LRUDict(max_size=10, a=1, b=2)

        assert cache["a"] == 1
        assert cache["b"] == 2


class TestBoundedList:
    """Tests for BoundedList."""

    def test_basic_append(self):
        """Basic append operation."""
        lst = BoundedList(max_size=10)
        lst.append(1)
        lst.append(2)

        assert len(lst) == 2
        assert lst[0] == 1
        assert lst[1] == 2

    def test_eviction_on_overflow(self):
        """Should remove oldest items when exceeding max_size."""
        lst = BoundedList(max_size=3)
        lst.append(1)
        lst.append(2)
        lst.append(3)

        assert len(lst) == 3

        # Adding 4th item should remove oldest (1)
        lst.append(4)

        assert len(lst) == 3
        assert 1 not in lst
        assert list(lst) == [2, 3, 4]

    def test_extend(self):
        """Extend should add multiple items and trim."""
        lst = BoundedList(max_size=3)
        lst.extend([1, 2, 3, 4, 5])

        assert len(lst) == 3
        assert list(lst) == [3, 4, 5]

    def test_insert(self):
        """Insert should add item and trim if needed."""
        lst = BoundedList(max_size=3, initial=[1, 2, 3])
        lst.insert(0, 0)

        assert len(lst) == 3
        # Oldest (0) was inserted at front, then trimmed
        assert list(lst) == [1, 2, 3]

    def test_stats(self):
        """Stats should return correct info."""
        lst = BoundedList(max_size=10)
        lst.append(1)
        lst.append(2)

        stats = lst.stats()

        assert stats["size"] == 2
        assert stats["max_size"] == 10
        assert stats["utilization"] == 0.2

    def test_init_with_data(self):
        """Should initialize with existing data and trim."""
        lst = BoundedList(max_size=3, initial=[1, 2, 3, 4, 5])

        assert len(lst) == 3
        assert list(lst) == [3, 4, 5]

    def test_thread_safety(self):
        """Should be thread-safe for concurrent access."""
        lst = BoundedList(max_size=100)
        errors = []

        def appender(start):
            try:
                for i in range(start, start + 50):
                    lst.append(i)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    _ = list(lst)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=appender, args=(0,)),
            threading.Thread(target=appender, args=(50,)),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


class TestIntegration:
    """Integration tests for cache usage."""

    def test_router_cache_pattern(self):
        """Test pattern used by router_cache in discovery.py."""
        cache = LRUDict(max_size=5)

        # Simulate router discovery
        for i in range(10):
            router_id = f"router_{i}"
            cache[router_id] = {"id": router_id, "latency": i * 10}

        # Should only keep last 5
        assert len(cache) == 5
        assert "router_0" not in cache
        assert "router_9" in cache

        # Accessing router_5 should keep it alive
        _ = cache["router_5"]

        # Add more routers
        for i in range(10, 15):
            router_id = f"router_{i}"
            cache[router_id] = {"id": router_id}

        # router_5 should still be there (was accessed)
        assert "router_5" in cache

    def test_failure_events_pattern(self):
        """Test pattern used by _failure_events in node.py."""
        events = BoundedList(max_size=100)

        # Simulate failure events over time
        for i in range(150):
            events.append(
                {
                    "router_id": f"router_{i % 10}",
                    "failure_type": "timeout",
                    "timestamp": time.time(),
                }
            )

        # Should only keep last 100
        assert len(events) == 100

        # Can still iterate and filter
        timeout_events = [e for e in events if e["failure_type"] == "timeout"]
        assert len(timeout_events) == 100
