"""Tests for valence.core.temporal module."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from valence.core.temporal import (
    SupersessionChain,
    TemporalValidity,
    calculate_freshness,
    freshness_label,
)


# ============================================================================
# TemporalValidity Tests
# ============================================================================

class TestTemporalValidityBasic:
    """Basic tests for TemporalValidity dataclass."""

    def test_default_values(self):
        """Test default values (always valid)."""
        tv = TemporalValidity()
        assert tv.valid_from is None
        assert tv.valid_until is None

    def test_create_with_valid_from(self):
        """Test creating with valid_from."""
        now = datetime.now()
        tv = TemporalValidity(valid_from=now)
        assert tv.valid_from == now
        assert tv.valid_until is None

    def test_create_with_valid_until(self):
        """Test creating with valid_until."""
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity(valid_until=future)
        assert tv.valid_from is None
        assert tv.valid_until == future

    def test_create_with_both_bounds(self):
        """Test creating with both bounds."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        tv = TemporalValidity(valid_from=start, valid_until=end)
        assert tv.valid_from == start
        assert tv.valid_until == end


class TestTemporalValidityFactoryMethods:
    """Tests for TemporalValidity factory methods."""

    def test_always_valid(self):
        """always_valid() should create unbounded validity."""
        tv = TemporalValidity.always_valid()
        assert tv.valid_from is None
        assert tv.valid_until is None

    def test_from_now(self):
        """from_now() should set valid_from to current time."""
        before = datetime.now()
        tv = TemporalValidity.from_now()
        after = datetime.now()
        assert tv.valid_from is not None
        assert before <= tv.valid_from <= after
        assert tv.valid_until is None

    def test_until(self):
        """until() should set valid_until."""
        end = datetime(2025, 1, 1)
        tv = TemporalValidity.until(end)
        assert tv.valid_from is None
        assert tv.valid_until == end

    def test_range(self):
        """range() should set both bounds."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        tv = TemporalValidity.range(start, end)
        assert tv.valid_from == start
        assert tv.valid_until == end

    def test_for_duration(self):
        """for_duration() should create range from now."""
        duration = timedelta(days=7)
        before = datetime.now()
        tv = TemporalValidity.for_duration(duration)
        after = datetime.now()

        assert tv.valid_from is not None
        assert tv.valid_until is not None
        assert before <= tv.valid_from <= after
        # valid_until should be approximately valid_from + duration
        expected_end = tv.valid_from + duration
        assert abs((tv.valid_until - expected_end).total_seconds()) < 1


class TestTemporalValidityIsValidAt:
    """Tests for TemporalValidity.is_valid_at()."""

    def test_always_valid_is_valid_anytime(self):
        """Always valid should be valid at any time."""
        tv = TemporalValidity.always_valid()
        assert tv.is_valid_at(datetime(1900, 1, 1)) is True
        assert tv.is_valid_at(datetime(2100, 1, 1)) is True
        assert tv.is_valid_at() is True  # Default is now

    def test_valid_from_past(self):
        """valid_from in the past should be valid now."""
        past = datetime.now() - timedelta(days=30)
        tv = TemporalValidity(valid_from=past)
        assert tv.is_valid_at() is True

    def test_valid_from_future(self):
        """valid_from in the future should not be valid now."""
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity(valid_from=future)
        assert tv.is_valid_at() is False

    def test_valid_until_future(self):
        """valid_until in the future should be valid now."""
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity(valid_until=future)
        assert tv.is_valid_at() is True

    def test_valid_until_past(self):
        """valid_until in the past should not be valid now."""
        past = datetime.now() - timedelta(days=30)
        tv = TemporalValidity(valid_until=past)
        assert tv.is_valid_at() is False

    def test_range_contains_time(self):
        """Time within range should be valid."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        tv = TemporalValidity.range(start, end)
        assert tv.is_valid_at(datetime(2024, 6, 15)) is True

    def test_range_before_start(self):
        """Time before range should not be valid."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        tv = TemporalValidity.range(start, end)
        assert tv.is_valid_at(datetime(2023, 12, 31)) is False

    def test_range_after_end(self):
        """Time after range should not be valid."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        tv = TemporalValidity.range(start, end)
        assert tv.is_valid_at(datetime(2025, 1, 1)) is False

    def test_exact_boundaries(self):
        """Exact boundaries should be valid."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        tv = TemporalValidity.range(start, end)
        assert tv.is_valid_at(start) is True
        assert tv.is_valid_at(end) is True


class TestTemporalValidityIsCurrent:
    """Tests for TemporalValidity.is_current()."""

    def test_always_valid_is_current(self):
        """Always valid should be current."""
        tv = TemporalValidity.always_valid()
        assert tv.is_current() is True

    def test_from_now_is_current(self):
        """from_now() should be current."""
        tv = TemporalValidity.from_now()
        assert tv.is_current() is True

    def test_future_validity_not_current(self):
        """Future validity should not be current."""
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity(valid_from=future)
        assert tv.is_current() is False

    def test_expired_not_current(self):
        """Expired validity should not be current."""
        past = datetime.now() - timedelta(days=30)
        tv = TemporalValidity(valid_until=past)
        assert tv.is_current() is False


class TestTemporalValidityIsExpired:
    """Tests for TemporalValidity.is_expired()."""

    def test_always_valid_not_expired(self):
        """Always valid should never be expired."""
        tv = TemporalValidity.always_valid()
        assert tv.is_expired() is False

    def test_no_end_not_expired(self):
        """No valid_until means never expired."""
        tv = TemporalValidity(valid_from=datetime.now())
        assert tv.is_expired() is False

    def test_future_end_not_expired(self):
        """Future valid_until should not be expired."""
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity(valid_until=future)
        assert tv.is_expired() is False

    def test_past_end_is_expired(self):
        """Past valid_until should be expired."""
        past = datetime.now() - timedelta(days=30)
        tv = TemporalValidity(valid_until=past)
        assert tv.is_expired() is True


class TestTemporalValidityIsFuture:
    """Tests for TemporalValidity.is_future()."""

    def test_always_valid_not_future(self):
        """Always valid should not be future."""
        tv = TemporalValidity.always_valid()
        assert tv.is_future() is False

    def test_no_start_not_future(self):
        """No valid_from means not future."""
        tv = TemporalValidity(valid_until=datetime.now() + timedelta(days=30))
        assert tv.is_future() is False

    def test_past_start_not_future(self):
        """Past valid_from should not be future."""
        past = datetime.now() - timedelta(days=30)
        tv = TemporalValidity(valid_from=past)
        assert tv.is_future() is False

    def test_future_start_is_future(self):
        """Future valid_from should be future."""
        future = datetime.now() + timedelta(days=30)
        tv = TemporalValidity(valid_from=future)
        assert tv.is_future() is True


class TestTemporalValidityOverlaps:
    """Tests for TemporalValidity.overlaps()."""

    def test_always_valid_overlaps_all(self):
        """Always valid should overlap with anything."""
        tv1 = TemporalValidity.always_valid()
        tv2 = TemporalValidity.range(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert tv1.overlaps(tv2) is True
        assert tv2.overlaps(tv1) is True

    def test_two_always_valid_overlap(self):
        """Two always valid should overlap."""
        tv1 = TemporalValidity.always_valid()
        tv2 = TemporalValidity.always_valid()
        assert tv1.overlaps(tv2) is True

    def test_overlapping_ranges(self):
        """Overlapping ranges should report overlap."""
        tv1 = TemporalValidity.range(datetime(2024, 1, 1), datetime(2024, 6, 30))
        tv2 = TemporalValidity.range(datetime(2024, 3, 1), datetime(2024, 12, 31))
        assert tv1.overlaps(tv2) is True
        assert tv2.overlaps(tv1) is True

    def test_non_overlapping_ranges(self):
        """Non-overlapping ranges should not report overlap."""
        tv1 = TemporalValidity.range(datetime(2024, 1, 1), datetime(2024, 3, 31))
        tv2 = TemporalValidity.range(datetime(2024, 7, 1), datetime(2024, 12, 31))
        assert tv1.overlaps(tv2) is False
        assert tv2.overlaps(tv1) is False

    def test_adjacent_ranges_overlap(self):
        """Adjacent ranges (touching) should overlap."""
        tv1 = TemporalValidity.range(datetime(2024, 1, 1), datetime(2024, 6, 30))
        tv2 = TemporalValidity.range(datetime(2024, 6, 30), datetime(2024, 12, 31))
        assert tv1.overlaps(tv2) is True

    def test_half_bounded_overlaps(self):
        """Half-bounded ranges should handle overlap correctly."""
        tv1 = TemporalValidity(valid_from=datetime(2024, 6, 1))  # Open ended
        tv2 = TemporalValidity(valid_until=datetime(2024, 12, 31))  # Open start
        assert tv1.overlaps(tv2) is True


class TestTemporalValidityContains:
    """Tests for TemporalValidity.contains()."""

    def test_contains_delegates_to_is_valid_at(self):
        """contains() should delegate to is_valid_at()."""
        tv = TemporalValidity.range(datetime(2024, 1, 1), datetime(2024, 12, 31))
        when = datetime(2024, 6, 15)
        assert tv.contains(when) == tv.is_valid_at(when)


class TestTemporalValidityDuration:
    """Tests for TemporalValidity.duration()."""

    def test_duration_unbounded_is_none(self):
        """Unbounded validity should have None duration."""
        tv = TemporalValidity.always_valid()
        assert tv.duration() is None

    def test_duration_open_start_is_none(self):
        """Open start should have None duration."""
        tv = TemporalValidity(valid_until=datetime.now())
        assert tv.duration() is None

    def test_duration_open_end_is_none(self):
        """Open end should have None duration."""
        tv = TemporalValidity(valid_from=datetime.now())
        assert tv.duration() is None

    def test_duration_bounded(self):
        """Bounded range should return duration."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 8)
        tv = TemporalValidity.range(start, end)
        assert tv.duration() == timedelta(days=7)


class TestTemporalValidityRemaining:
    """Tests for TemporalValidity.remaining()."""

    def test_remaining_no_end_is_none(self):
        """No valid_until should return None remaining."""
        tv = TemporalValidity(valid_from=datetime.now())
        assert tv.remaining() is None

    def test_remaining_future_end(self):
        """Future valid_until should return positive remaining."""
        future = datetime.now() + timedelta(days=7)
        tv = TemporalValidity(valid_until=future)
        remaining = tv.remaining()
        assert remaining is not None
        assert remaining > timedelta(days=6)
        assert remaining < timedelta(days=8)

    def test_remaining_past_end_is_zero(self):
        """Past valid_until should return zero remaining."""
        past = datetime.now() - timedelta(days=7)
        tv = TemporalValidity(valid_until=past)
        remaining = tv.remaining()
        assert remaining == timedelta(0)


class TestTemporalValidityExpireNow:
    """Tests for TemporalValidity.expire_now()."""

    def test_expire_now_sets_valid_until(self):
        """expire_now() should set valid_until to now."""
        tv = TemporalValidity(valid_from=datetime(2024, 1, 1))
        before = datetime.now()
        expired = tv.expire_now()
        after = datetime.now()

        assert expired.valid_from == tv.valid_from
        assert expired.valid_until is not None
        assert before <= expired.valid_until <= after

    def test_expire_now_returns_new_instance(self):
        """expire_now() should return new instance."""
        original = TemporalValidity(valid_from=datetime.now())
        expired = original.expire_now()
        assert expired is not original

    def test_expire_now_preserves_valid_from(self):
        """expire_now() should preserve valid_from."""
        start = datetime(2024, 1, 1)
        original = TemporalValidity(valid_from=start)
        expired = original.expire_now()
        assert expired.valid_from == start


class TestTemporalValidityExtend:
    """Tests for TemporalValidity.extend()."""

    def test_extend_adds_duration(self):
        """extend() should add duration to valid_until."""
        end = datetime(2024, 6, 30)
        tv = TemporalValidity(valid_until=end)
        extended = tv.extend(timedelta(days=30))
        assert extended.valid_until == end + timedelta(days=30)

    def test_extend_unbounded_returns_self(self):
        """extend() on unbounded should return self."""
        tv = TemporalValidity(valid_from=datetime.now())  # No end
        extended = tv.extend(timedelta(days=30))
        assert extended is tv

    def test_extend_preserves_valid_from(self):
        """extend() should preserve valid_from."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 6, 30)
        tv = TemporalValidity.range(start, end)
        extended = tv.extend(timedelta(days=30))
        assert extended.valid_from == start


class TestTemporalValidityToDict:
    """Tests for TemporalValidity.to_dict()."""

    def test_to_dict_always_valid(self):
        """to_dict() for always valid should have nulls."""
        tv = TemporalValidity.always_valid()
        d = tv.to_dict()
        assert d["valid_from"] is None
        assert d["valid_until"] is None

    def test_to_dict_with_dates(self):
        """to_dict() should convert datetimes to ISO strings."""
        start = datetime(2024, 1, 1, 10, 30, 0)
        end = datetime(2024, 12, 31, 23, 59, 59)
        tv = TemporalValidity.range(start, end)
        d = tv.to_dict()
        assert d["valid_from"] == start.isoformat()
        assert d["valid_until"] == end.isoformat()


class TestTemporalValidityFromDict:
    """Tests for TemporalValidity.from_dict()."""

    def test_from_dict_empty(self):
        """from_dict() with empty dict should be always valid."""
        tv = TemporalValidity.from_dict({})
        assert tv.valid_from is None
        assert tv.valid_until is None

    def test_from_dict_with_strings(self):
        """from_dict() should parse ISO strings."""
        d = {
            "valid_from": "2024-01-01T10:30:00",
            "valid_until": "2024-12-31T23:59:59",
        }
        tv = TemporalValidity.from_dict(d)
        assert tv.valid_from == datetime(2024, 1, 1, 10, 30, 0)
        assert tv.valid_until == datetime(2024, 12, 31, 23, 59, 59)

    def test_from_dict_with_datetime_objects(self):
        """from_dict() should accept datetime objects."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        d = {"valid_from": start, "valid_until": end}
        tv = TemporalValidity.from_dict(d)
        assert tv.valid_from == start
        assert tv.valid_until == end

    def test_from_dict_roundtrip(self):
        """to_dict and from_dict should roundtrip."""
        original = TemporalValidity.range(
            datetime(2024, 1, 1, 10, 0, 0),
            datetime(2024, 12, 31, 23, 0, 0),
        )
        roundtripped = TemporalValidity.from_dict(original.to_dict())
        assert roundtripped.valid_from == original.valid_from
        assert roundtripped.valid_until == original.valid_until


class TestTemporalValidityStr:
    """Tests for TemporalValidity.__str__()."""

    def test_str_always_valid(self):
        """__str__() for always valid."""
        tv = TemporalValidity.always_valid()
        assert str(tv) == "always valid"

    def test_str_valid_from_only(self):
        """__str__() with only valid_from."""
        start = datetime(2024, 1, 1)
        tv = TemporalValidity(valid_from=start)
        s = str(tv)
        assert "valid from" in s
        assert start.isoformat() in s

    def test_str_valid_until_only(self):
        """__str__() with only valid_until."""
        end = datetime(2024, 12, 31)
        tv = TemporalValidity(valid_until=end)
        s = str(tv)
        assert "valid until" in s
        assert end.isoformat() in s

    def test_str_range(self):
        """__str__() with both bounds."""
        start = datetime(2024, 1, 1)
        end = datetime(2024, 12, 31)
        tv = TemporalValidity.range(start, end)
        s = str(tv)
        assert "valid" in s
        assert "to" in s


# ============================================================================
# SupersessionChain Tests
# ============================================================================

class TestSupersessionChain:
    """Tests for SupersessionChain dataclass."""

    def test_basic_creation(self):
        """Test basic chain creation."""
        ids = ["id1", "id2", "id3"]
        reasons = ["Initial", "Update 1", "Update 2"]
        times = [datetime(2024, 1, 1), datetime(2024, 2, 1), datetime(2024, 3, 1)]
        chain = SupersessionChain(belief_ids=ids, reasons=reasons, timestamps=times)

        assert chain.belief_ids == ids
        assert chain.reasons == reasons
        assert chain.timestamps == times

    def test_original_id(self):
        """original_id should return first ID."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2", "id3"],
            reasons=["a", "b", "c"],
            timestamps=[datetime.now()] * 3,
        )
        assert chain.original_id == "id1"

    def test_current_id(self):
        """current_id should return last ID."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2", "id3"],
            reasons=["a", "b", "c"],
            timestamps=[datetime.now()] * 3,
        )
        assert chain.current_id == "id3"

    def test_length(self):
        """length should return number of beliefs."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2", "id3"],
            reasons=["a", "b", "c"],
            timestamps=[datetime.now()] * 3,
        )
        assert chain.length == 3

    def test_revision_count(self):
        """revision_count should be length - 1."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2", "id3"],
            reasons=["a", "b", "c"],
            timestamps=[datetime.now()] * 3,
        )
        assert chain.revision_count == 2

    def test_single_belief_chain(self):
        """Single belief chain should have revision_count of 0."""
        chain = SupersessionChain(
            belief_ids=["id1"],
            reasons=["Initial"],
            timestamps=[datetime.now()],
        )
        assert chain.length == 1
        assert chain.revision_count == 0
        assert chain.original_id == "id1"
        assert chain.current_id == "id1"


class TestSupersessionChainGetAtTime:
    """Tests for SupersessionChain.get_at_time()."""

    def test_get_at_time_before_first(self):
        """Time before first timestamp should return None."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2"],
            reasons=["a", "b"],
            timestamps=[datetime(2024, 2, 1), datetime(2024, 3, 1)],
        )
        result = chain.get_at_time(datetime(2024, 1, 1))
        assert result is None

    def test_get_at_time_at_first(self):
        """Time at first timestamp returns next belief in chain."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2"],
            reasons=["a", "b"],
            timestamps=[datetime(2024, 2, 1), datetime(2024, 3, 1)],
        )
        # At exactly the first timestamp (when supersession happened),
        # we're at the superseded belief, so it returns the next in chain
        result = chain.get_at_time(datetime(2024, 2, 1))
        # Since when not < ts[0] and when < ts[1], returns belief_ids[1]
        assert result == "id2"

    def test_get_at_time_between(self):
        """Time between timestamps should return appropriate ID."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2", "id3"],
            reasons=["a", "b", "c"],
            timestamps=[datetime(2024, 1, 1), datetime(2024, 3, 1), datetime(2024, 5, 1)],
        )
        # 2024-02-01 is >= ts[0] but < ts[1], so returns belief_ids[1]
        result = chain.get_at_time(datetime(2024, 2, 1))
        assert result == "id2"

    def test_get_at_time_after_last(self):
        """Time after last timestamp should return current ID."""
        chain = SupersessionChain(
            belief_ids=["id1", "id2"],
            reasons=["a", "b"],
            timestamps=[datetime(2024, 2, 1), datetime(2024, 3, 1)],
        )
        result = chain.get_at_time(datetime(2024, 12, 1))
        assert result == "id2"


class TestSupersessionChainToDict:
    """Tests for SupersessionChain.to_dict()."""

    def test_to_dict(self):
        """to_dict should include all fields."""
        times = [datetime(2024, 1, 1), datetime(2024, 2, 1)]
        chain = SupersessionChain(
            belief_ids=["id1", "id2"],
            reasons=["Initial", "Updated"],
            timestamps=times,
        )
        d = chain.to_dict()

        assert d["belief_ids"] == ["id1", "id2"]
        assert d["reasons"] == ["Initial", "Updated"]
        assert d["timestamps"] == [t.isoformat() for t in times]
        assert d["length"] == 2
        assert d["revision_count"] == 1


# ============================================================================
# calculate_freshness Tests
# ============================================================================

class TestCalculateFreshness:
    """Tests for calculate_freshness function."""

    def test_brand_new_is_fresh(self):
        """Just created should have freshness near 1.0."""
        freshness = calculate_freshness(datetime.now())
        assert freshness > 0.99

    def test_at_half_life(self):
        """At half-life, freshness should be 0.5."""
        created = datetime.now() - timedelta(days=30)
        freshness = calculate_freshness(created, half_life_days=30)
        assert abs(freshness - 0.5) < 0.01

    def test_at_two_half_lives(self):
        """At two half-lives, freshness should be 0.25."""
        created = datetime.now() - timedelta(days=60)
        freshness = calculate_freshness(created, half_life_days=30)
        assert abs(freshness - 0.25) < 0.01

    def test_very_old_is_stale(self):
        """Very old content should have low freshness."""
        created = datetime.now() - timedelta(days=365)
        freshness = calculate_freshness(created, half_life_days=30)
        assert freshness < 0.01

    def test_freshness_bounded_at_zero(self):
        """Freshness should not go below 0."""
        created = datetime.now() - timedelta(days=10000)
        freshness = calculate_freshness(created)
        assert freshness >= 0

    def test_freshness_bounded_at_one(self):
        """Freshness should not exceed 1."""
        freshness = calculate_freshness(datetime.now())
        assert freshness <= 1.0

    def test_custom_half_life(self):
        """Custom half-life should work."""
        created = datetime.now() - timedelta(days=7)
        freshness = calculate_freshness(created, half_life_days=7)
        assert abs(freshness - 0.5) < 0.01


# ============================================================================
# freshness_label Tests
# ============================================================================

class TestFreshnessLabel:
    """Tests for freshness_label function."""

    @pytest.mark.parametrize("value,expected", [
        (0.95, "very fresh"),
        (0.90, "very fresh"),
        (0.89, "fresh"),
        (0.70, "fresh"),
        (0.69, "moderately fresh"),
        (0.50, "moderately fresh"),
        (0.49, "aging"),
        (0.30, "aging"),
        (0.29, "stale"),
        (0.10, "stale"),
        (0.09, "very stale"),
        (0.0, "very stale"),
    ])
    def test_freshness_label_boundaries(self, value, expected):
        """Test freshness label at various boundaries."""
        assert freshness_label(value) == expected

    def test_freshness_label_exact_boundary(self):
        """Test exact boundary value."""
        assert freshness_label(0.9) == "very fresh"
        assert freshness_label(0.7) == "fresh"
        assert freshness_label(0.5) == "moderately fresh"
        assert freshness_label(0.3) == "aging"
        assert freshness_label(0.1) == "stale"
