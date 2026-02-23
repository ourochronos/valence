"""Tests for valence.core.temporal — temporal validity and supersession chains."""

from __future__ import annotations

from datetime import datetime, timedelta

from valence.core.temporal import (
    SupersessionChain,
    TemporalValidity,
    calculate_freshness,
    freshness_label,
)

# =============================================================================
# TEMPORAL VALIDITY TESTS
# =============================================================================


class TestTemporalValidity:
    """Tests for TemporalValidity dataclass."""

    def test_always_valid(self):
        tv = TemporalValidity.always_valid()
        assert tv.valid_from is None
        assert tv.valid_until is None
        assert tv.is_current() is True
        assert tv.is_expired() is False

    def test_from_now(self):
        tv = TemporalValidity.from_now()
        assert tv.valid_from is not None
        assert tv.valid_until is None
        assert tv.is_current() is True

    def test_until(self):
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity.until(future)
        assert tv.valid_from is None
        assert tv.valid_until == future
        assert tv.is_current() is True
        assert tv.is_expired() is False

    def test_until_past(self):
        past = datetime.now() - timedelta(days=1)
        tv = TemporalValidity.until(past)
        assert tv.is_current() is False
        assert tv.is_expired() is True

    def test_range(self):
        start = datetime.now() - timedelta(days=1)
        end = datetime.now() + timedelta(days=1)
        tv = TemporalValidity.range(start, end)
        assert tv.is_current() is True
        assert tv.is_expired() is False

    def test_for_duration(self):
        tv = TemporalValidity.for_duration(timedelta(hours=1))
        assert tv.valid_from is not None
        assert tv.valid_until is not None
        assert tv.is_current() is True
        delta = tv.valid_until - tv.valid_from
        assert abs(delta.total_seconds() - 3600) < 1

    def test_is_valid_at_specific_time(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 12, 31)
        tv = TemporalValidity.range(start, end)

        assert tv.is_valid_at(datetime(2025, 6, 15)) is True
        assert tv.is_valid_at(datetime(2024, 6, 15)) is False
        assert tv.is_valid_at(datetime(2026, 6, 15)) is False

    def test_is_future(self):
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity(valid_from=future)
        assert tv.is_future() is True
        assert tv.is_current() is False

    def test_is_future_no_start(self):
        tv = TemporalValidity.always_valid()
        assert tv.is_future() is False

    def test_overlaps_both_unbounded(self):
        a = TemporalValidity.always_valid()
        b = TemporalValidity.always_valid()
        assert a.overlaps(b) is True

    def test_overlaps_one_unbounded(self):
        a = TemporalValidity.always_valid()
        b = TemporalValidity.range(datetime(2025, 1, 1), datetime(2025, 12, 31))
        assert a.overlaps(b) is True
        assert b.overlaps(a) is True

    def test_overlaps_overlapping_ranges(self):
        a = TemporalValidity.range(datetime(2025, 1, 1), datetime(2025, 6, 30))
        b = TemporalValidity.range(datetime(2025, 3, 1), datetime(2025, 12, 31))
        assert a.overlaps(b) is True
        assert b.overlaps(a) is True

    def test_overlaps_non_overlapping(self):
        a = TemporalValidity.range(datetime(2025, 1, 1), datetime(2025, 3, 31))
        b = TemporalValidity.range(datetime(2025, 7, 1), datetime(2025, 12, 31))
        assert a.overlaps(b) is False
        assert b.overlaps(a) is False

    def test_contains(self):
        tv = TemporalValidity.range(datetime(2025, 1, 1), datetime(2025, 12, 31))
        assert tv.contains(datetime(2025, 6, 15)) is True
        assert tv.contains(datetime(2024, 6, 15)) is False

    def test_duration(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        tv = TemporalValidity.range(start, end)
        assert tv.duration() == timedelta(days=30)

    def test_duration_unbounded(self):
        tv = TemporalValidity.always_valid()
        assert tv.duration() is None

    def test_remaining(self):
        future = datetime.now() + timedelta(hours=2)
        tv = TemporalValidity.until(future)
        remaining = tv.remaining()
        assert remaining is not None
        assert remaining.total_seconds() > 0
        assert remaining.total_seconds() < 7201

    def test_remaining_no_end(self):
        tv = TemporalValidity.always_valid()
        assert tv.remaining() is None

    def test_remaining_expired(self):
        past = datetime.now() - timedelta(hours=1)
        tv = TemporalValidity.until(past)
        remaining = tv.remaining()
        assert remaining == timedelta(0)

    def test_expire_now(self):
        tv = TemporalValidity.from_now()
        expired = tv.expire_now()
        assert expired.valid_until is not None
        assert expired.valid_from == tv.valid_from

    def test_extend(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 1, 31)
        tv = TemporalValidity.range(start, end)
        extended = tv.extend(timedelta(days=30))
        assert extended.valid_until == datetime(2025, 3, 2)

    def test_extend_unbounded(self):
        tv = TemporalValidity.always_valid()
        extended = tv.extend(timedelta(days=30))
        assert extended is tv  # Same object, unbounded stays unbounded

    def test_to_dict(self):
        start = datetime(2025, 1, 1)
        end = datetime(2025, 12, 31)
        tv = TemporalValidity.range(start, end)
        d = tv.to_dict()
        assert d["valid_from"] == start.isoformat()
        assert d["valid_until"] == end.isoformat()

    def test_to_dict_null(self):
        tv = TemporalValidity.always_valid()
        d = tv.to_dict()
        assert d["valid_from"] is None
        assert d["valid_until"] is None

    def test_from_dict(self):
        data = {
            "valid_from": "2025-01-01T00:00:00",
            "valid_until": "2025-12-31T00:00:00",
        }
        tv = TemporalValidity.from_dict(data)
        assert tv.valid_from == datetime(2025, 1, 1)
        assert tv.valid_until == datetime(2025, 12, 31)

    def test_from_dict_null(self):
        tv = TemporalValidity.from_dict({})
        assert tv.valid_from is None
        assert tv.valid_until is None

    def test_from_dict_datetime_objects(self):
        now = datetime.now()
        tv = TemporalValidity.from_dict({"valid_from": now, "valid_until": now})
        assert tv.valid_from == now
        assert tv.valid_until == now

    def test_str_always_valid(self):
        assert str(TemporalValidity.always_valid()) == "always valid"

    def test_str_until(self):
        end = datetime(2025, 12, 31)
        s = str(TemporalValidity.until(end))
        assert "valid until" in s
        assert "2025-12-31" in s

    def test_str_from(self):
        start = datetime(2025, 1, 1)
        s = str(TemporalValidity(valid_from=start))
        assert "valid from" in s
        assert "2025-01-01" in s

    def test_str_range(self):
        s = str(TemporalValidity.range(datetime(2025, 1, 1), datetime(2025, 12, 31)))
        assert "valid" in s
        assert "2025-01-01" in s
        assert "2025-12-31" in s


# =============================================================================
# SUPERSESSION CHAIN TESTS
# =============================================================================


class TestSupersessionChain:
    """Tests for SupersessionChain."""

    def test_basic_chain(self):
        chain = SupersessionChain(
            belief_ids=["a", "b", "c"],
            reasons=["initial", "updated", "corrected"],
            timestamps=[datetime(2025, 1, 1), datetime(2025, 6, 1), datetime(2025, 12, 1)],
        )
        assert chain.original_id == "a"
        assert chain.current_id == "c"
        assert chain.length == 3
        assert chain.revision_count == 2

    def test_single_belief_chain(self):
        chain = SupersessionChain(
            belief_ids=["a"],
            reasons=["initial"],
            timestamps=[datetime(2025, 1, 1)],
        )
        assert chain.original_id == "a"
        assert chain.current_id == "a"
        assert chain.revision_count == 0

    def test_get_at_time(self):
        chain = SupersessionChain(
            belief_ids=["a", "b", "c"],
            reasons=["initial", "updated", "corrected"],
            timestamps=[datetime(2025, 1, 1), datetime(2025, 6, 1), datetime(2025, 12, 1)],
        )
        # Algorithm: find first i where when < timestamps[i], return belief_ids[i] (or None if i==0)
        # Semantically: timestamps[i] is when belief_ids[i] was superseded
        assert chain.get_at_time(datetime(2025, 3, 1)) == "b"  # when < ts[1], return ids[1]
        assert chain.get_at_time(datetime(2025, 8, 1)) == "c"  # when < ts[2], return ids[2]
        assert chain.get_at_time(datetime(2026, 1, 1)) == "c"  # no match, return current

    def test_get_at_time_before_first(self):
        chain = SupersessionChain(
            belief_ids=["a", "b"],
            reasons=["initial", "updated"],
            timestamps=[datetime(2025, 6, 1), datetime(2025, 12, 1)],
        )
        assert chain.get_at_time(datetime(2025, 1, 1)) is None

    def test_to_dict(self):
        chain = SupersessionChain(
            belief_ids=["a", "b"],
            reasons=["initial", "fixed"],
            timestamps=[datetime(2025, 1, 1), datetime(2025, 6, 1)],
        )
        d = chain.to_dict()
        assert d["belief_ids"] == ["a", "b"]
        assert d["reasons"] == ["initial", "fixed"]
        assert d["length"] == 2
        assert d["revision_count"] == 1


# =============================================================================
# FRESHNESS TESTS
# =============================================================================


class TestCalculateFreshness:
    """Tests for calculate_freshness."""

    def test_just_created(self):
        freshness = calculate_freshness(datetime.now())
        assert freshness > 0.99

    def test_half_life(self):
        half_life = 30.0
        created = datetime.now() - timedelta(days=30)
        freshness = calculate_freshness(created, half_life_days=half_life)
        assert abs(freshness - 0.5) < 0.01

    def test_very_old(self):
        created = datetime.now() - timedelta(days=365)
        freshness = calculate_freshness(created)
        assert freshness < 0.01

    def test_custom_half_life(self):
        created = datetime.now() - timedelta(days=7)
        freshness = calculate_freshness(created, half_life_days=7.0)
        assert abs(freshness - 0.5) < 0.01

    def test_bounded_0_1(self):
        freshness = calculate_freshness(datetime.now() - timedelta(days=10000))
        assert 0.0 <= freshness <= 1.0

        freshness = calculate_freshness(datetime.now())
        assert 0.0 <= freshness <= 1.0


class TestFreshnessLabel:
    """Tests for freshness_label."""

    def test_very_fresh(self):
        assert freshness_label(0.95) == "very fresh"

    def test_fresh(self):
        assert freshness_label(0.75) == "fresh"

    def test_moderately_fresh(self):
        assert freshness_label(0.55) == "moderately fresh"

    def test_aging(self):
        assert freshness_label(0.35) == "aging"

    def test_stale(self):
        assert freshness_label(0.15) == "stale"

    def test_very_stale(self):
        assert freshness_label(0.05) == "very stale"
