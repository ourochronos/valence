# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

"""Tests for epistemic_type ranking (#70) and cold-start boost (#71)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from valence.core.ranking import detect_query_intent, multi_signal_rank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_article(
    similarity: float = 0.5,
    confidence: float = 0.7,
    epistemic_type: str | None = None,
    created_at: datetime | str | None = None,
    **kwargs,
) -> dict:
    """Build a minimal article dict for ranking tests."""
    if created_at is None:
        created_at = (datetime.now(UTC) - timedelta(days=365)).isoformat()
    elif isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    return {
        "id": "test-id",
        "similarity": similarity,
        "confidence": {"overall": confidence},
        "epistemic_type": epistemic_type,
        "created_at": created_at,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# detect_query_intent tests
# ---------------------------------------------------------------------------


class TestDetectQueryIntent:
    def test_procedural_how_to(self):
        assert detect_query_intent("how to deploy the service") == "procedural"

    def test_procedural_how_should(self):
        assert detect_query_intent("how should I set up the database") == "procedural"

    def test_procedural_steps_to(self):
        assert detect_query_intent("steps to configure nginx") == "procedural"

    def test_procedural_process_for(self):
        assert detect_query_intent("process for onboarding new users") == "procedural"

    def test_procedural_whats_the_process(self):
        assert detect_query_intent("what's the process for releasing") == "procedural"

    def test_procedural_checklist(self):
        assert detect_query_intent("deployment checklist items") == "procedural"

    def test_procedural_workflow(self):
        assert detect_query_intent("CI/CD workflow overview") == "procedural"

    def test_procedural_procedure(self):
        assert detect_query_intent("backup procedure details") == "procedural"

    def test_episodic_what_happened(self):
        assert detect_query_intent("what happened during the outage") == "episodic"

    def test_episodic_when_did(self):
        assert detect_query_intent("when did we deploy version 2") == "episodic"

    def test_episodic_session(self):
        assert detect_query_intent("notes from last session") == "episodic"

    def test_episodic_last_time(self):
        assert detect_query_intent("what did we discuss last time") == "episodic"

    def test_episodic_date_pattern(self):
        assert detect_query_intent("changes made on 2025-11-01") == "episodic"

    def test_general_default(self):
        assert detect_query_intent("python async patterns") == "general"

    def test_general_empty(self):
        assert detect_query_intent("database indexing strategies") == "general"

    def test_case_insensitive(self):
        assert detect_query_intent("How To deploy the service") == "procedural"


# ---------------------------------------------------------------------------
# Epistemic type awareness tests (#70)
# ---------------------------------------------------------------------------


class TestEpistemicTypeAwareness:
    def test_procedural_article_boosted_for_procedural_query(self):
        article = make_article(similarity=0.5, epistemic_type="procedural")
        results = multi_signal_rank([article], query_intent="procedural", cold_start_boost=False)
        # Without boost, base score would be around 0.5*0.5 + 0.35*0.7 + 0.15*... ≈ some value
        # With 1.3x we just check it's larger than unmodified
        base_article = make_article(similarity=0.5, epistemic_type="semantic")
        base_results = multi_signal_rank([base_article], query_intent="procedural", cold_start_boost=False)
        assert results[0]["final_score"] > base_results[0]["final_score"]

    def test_episodic_article_penalized_for_procedural_query(self):
        procedural_article = make_article(similarity=0.5, epistemic_type="procedural")
        episodic_article = make_article(similarity=0.5, epistemic_type="episodic")
        proc_results = multi_signal_rank([procedural_article], query_intent="procedural", cold_start_boost=False)
        epis_results = multi_signal_rank([episodic_article], query_intent="procedural", cold_start_boost=False)
        assert proc_results[0]["final_score"] > epis_results[0]["final_score"]

    def test_no_adjustment_for_general_intent(self):
        article_proc = make_article(similarity=0.5, epistemic_type="procedural")
        article_epis = make_article(similarity=0.5, epistemic_type="episodic")
        article_sem = make_article(similarity=0.5, epistemic_type="semantic")
        r_proc = multi_signal_rank([article_proc], query_intent="general", cold_start_boost=False)
        r_epis = multi_signal_rank([article_epis], query_intent="general", cold_start_boost=False)
        r_sem = multi_signal_rank([article_sem], query_intent="general", cold_start_boost=False)
        # All should have same base score since no adjustment
        assert abs(r_proc[0]["final_score"] - r_epis[0]["final_score"]) < 1e-9
        assert abs(r_proc[0]["final_score"] - r_sem[0]["final_score"]) < 1e-9

    def test_semantic_type_no_adjustment(self):
        """Articles with epistemic_type=semantic are not adjusted even for procedural intent."""
        article = make_article(similarity=0.5, epistemic_type="semantic")
        no_intent = multi_signal_rank([make_article(similarity=0.5, epistemic_type="semantic")], query_intent=None, cold_start_boost=False)
        with_intent = multi_signal_rank([article], query_intent="procedural", cold_start_boost=False)
        assert abs(with_intent[0]["final_score"] - no_intent[0]["final_score"]) < 1e-9

    def test_boost_multiplier_is_1_3x(self):
        article_match = make_article(similarity=0.5, epistemic_type="procedural")
        article_none = make_article(similarity=0.5, epistemic_type=None)
        r_match = multi_signal_rank([article_match], query_intent="procedural", cold_start_boost=False)
        r_none = multi_signal_rank([article_none], query_intent="procedural", cold_start_boost=False)
        assert abs(r_match[0]["final_score"] / r_none[0]["final_score"] - 1.3) < 1e-6

    def test_penalty_multiplier_is_0_85x(self):
        article_conflict = make_article(similarity=0.5, epistemic_type="episodic")
        article_none = make_article(similarity=0.5, epistemic_type=None)
        r_conflict = multi_signal_rank([article_conflict], query_intent="procedural", cold_start_boost=False)
        r_none = multi_signal_rank([article_none], query_intent="procedural", cold_start_boost=False)
        assert abs(r_conflict[0]["final_score"] / r_none[0]["final_score"] - 0.85) < 1e-6


# ---------------------------------------------------------------------------
# Cold-start boost tests (#71)
# ---------------------------------------------------------------------------


class TestColdStartBoost:
    def test_fresh_high_confidence_gets_floor(self):
        """Articles < 24h old with confidence >= 0.7 get floor of 0.3."""
        fresh = make_article(
            similarity=0.0,  # very low semantic match
            confidence=0.8,
            created_at=datetime.now(UTC) - timedelta(hours=1),
        )
        results = multi_signal_rank([fresh], cold_start_boost=True)
        assert results[0]["final_score"] >= 0.3

    def test_cold_start_decays_at_48h(self):
        """Articles 25-48h old get a lower floor of 0.15 but not 0.3."""
        # Use very low confidence to keep base score minimal, then verify floor is 0.15 not 0.3
        # At 36h: floor = 0.15; at <24h: floor = 0.3
        article_36h = make_article(
            similarity=0.0,
            confidence=0.7,  # just above threshold
            created_at=datetime.now(UTC) - timedelta(hours=36),
        )
        article_2h = make_article(
            similarity=0.0,
            confidence=0.7,
            created_at=datetime.now(UTC) - timedelta(hours=2),
        )
        results_36h = multi_signal_rank([article_36h], cold_start_boost=True)
        results_2h = multi_signal_rank([article_2h], cold_start_boost=True)
        # 36h article gets the lower 0.15 floor, 2h article gets 0.3 floor
        # Both should be >= their respective floors
        assert results_36h[0]["final_score"] >= 0.15
        # 2h article has a higher floor applied
        assert results_2h[0]["final_score"] >= 0.3
        # At 36h, the floor is 0.15, not 0.3; so if we force base below 0.3, it won't reach 0.3
        # We can verify floor difference by checking 2h > 36h when base scores are same
        assert results_2h[0]["final_score"] >= results_36h[0]["final_score"]

    def test_cold_start_no_boost_for_old_articles(self):
        """Articles older than 48h do not get a score floor applied."""
        # Compare: fresh vs old with identical signals — fresh should get boosted floor, old not.
        fresh_article = make_article(
            similarity=0.0,
            confidence=0.75,
            created_at=datetime.now(UTC) - timedelta(hours=2),
        )
        old_article = make_article(
            similarity=0.0,
            confidence=0.75,
            created_at=datetime.now(UTC) - timedelta(hours=72),
        )
        results_fresh = multi_signal_rank([fresh_article], cold_start_boost=True)
        results_old = multi_signal_rank([old_article], cold_start_boost=True)
        # Fresh article gets a 0.3 floor; old article gets no floor
        assert results_fresh[0]["final_score"] >= 0.3
        # Old article has same base similarity=0, so without cold-start floor it's lower
        assert results_old[0]["final_score"] < results_fresh[0]["final_score"]

    def test_cold_start_no_boost_for_low_confidence(self):
        """Fresh articles with confidence < 0.7 do not get the cold-start score floor."""
        # Compare low-confidence vs high-confidence fresh articles at similarity=0
        fresh_low_conf = make_article(
            similarity=0.0,
            confidence=0.5,  # below threshold
            created_at=datetime.now(UTC) - timedelta(hours=1),
        )
        fresh_high_conf = make_article(
            similarity=0.0,
            confidence=0.8,  # above threshold
            created_at=datetime.now(UTC) - timedelta(hours=1),
        )
        results_low = multi_signal_rank([fresh_low_conf], cold_start_boost=True)
        results_high = multi_signal_rank([fresh_high_conf], cold_start_boost=True)
        # High-confidence gets the 0.3 floor
        assert results_high[0]["final_score"] >= 0.3
        # Low-confidence does NOT get the floor, so it scores based on natural signals only
        # With sim=0, confidence=0.5 and high recency, natural score is modest
        # The key is: low-conf score < high-conf score (no artificial floor applied to low)
        assert results_low[0]["final_score"] < results_high[0]["final_score"]

    def test_cold_start_boost_false_disables_feature(self):
        """cold_start_boost=False disables the floor entirely."""

        # Use compiled_at (old) to get low recency; created_at (fresh) for cold-start eligibility
        # This gives a base score below 0.3, but cold-start floor would push it up
        def _make() -> dict:
            return {
                "id": "cs-test",
                "similarity": 0.0,
                "confidence": {"overall": 0.7},
                "epistemic_type": None,
                "created_at": (datetime.now(UTC) - timedelta(hours=1)).isoformat(),
                "compiled_at": (datetime.now(UTC) - timedelta(days=365)).isoformat(),
            }

        results_on = multi_signal_rank([_make()], cold_start_boost=True)
        results_off = multi_signal_rank([_make()], cold_start_boost=False)
        # With boost=True, floor raises score to at least 0.3
        assert results_on[0]["final_score"] >= 0.3
        # With boost=False, natural score (low recency, zero similarity) is below 0.3
        assert results_off[0]["final_score"] < 0.3

    def test_cold_start_floor_does_not_lower_high_scores(self):
        """If a fresh article already scores above 0.3, floor doesn't change it."""
        fresh_high = make_article(
            similarity=0.9,
            confidence=0.9,
            created_at=datetime.now(UTC) - timedelta(hours=1),
        )
        results = multi_signal_rank([fresh_high], cold_start_boost=True)
        # Score should be well above 0.3
        assert results[0]["final_score"] > 0.3


# ---------------------------------------------------------------------------
# Combined test
# ---------------------------------------------------------------------------


class TestCombined:
    def test_fresh_procedural_beats_old_episodic_for_how_to(self):
        """Fresh procedural article should outrank an old episodic article for a 'how to' query."""
        fresh_procedural = make_article(
            similarity=0.5,
            confidence=0.8,
            epistemic_type="procedural",
            created_at=datetime.now(UTC) - timedelta(hours=2),
        )
        old_episodic = make_article(
            similarity=0.6,  # slightly higher semantic match
            confidence=0.9,
            epistemic_type="episodic",
            created_at=datetime.now(UTC) - timedelta(days=180),
        )
        results = multi_signal_rank(
            [old_episodic, fresh_procedural],
            query_intent="procedural",
            cold_start_boost=True,
        )
        # Fresh procedural should be ranked first
        assert results[0]["epistemic_type"] == "procedural"
