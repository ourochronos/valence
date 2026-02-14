"""Confidence and corroboration tool implementations.

Functions:
    confidence_explain, belief_corroboration, _corroboration_label
"""

from __future__ import annotations

from typing import Any

from . import _common
from ._common import DEFAULT_WEIGHTS, Belief, ConfidenceDimension, confidence_label, logger


def belief_corroboration(belief_id: str) -> dict[str, Any]:
    """Get corroboration details for a belief.

    Shows how many independent sources confirm this belief and who they are.
    """
    from uuid import UUID

    from our_federation.corroboration import get_corroboration

    try:
        belief_uuid = UUID(belief_id)
    except ValueError:
        return {"success": False, "error": f"Invalid belief ID: {belief_id}"}

    corroboration = get_corroboration(belief_uuid)

    if not corroboration:
        return {"success": False, "error": f"Belief not found: {belief_id}"}

    return {
        "success": True,
        "belief_id": str(corroboration.belief_id),
        "corroboration_count": corroboration.corroboration_count,
        "confidence_corroboration": corroboration.confidence_corroboration,
        "corroborating_sources": corroboration.sources,
        "confidence_label": _corroboration_label(corroboration.corroboration_count),
    }


def _corroboration_label(count: int) -> str:
    """Human-readable label for corroboration level."""
    if count == 0:
        return "uncorroborated"
    elif count == 1:
        return "single corroboration"
    elif count <= 3:
        return "moderately corroborated"
    elif count <= 6:
        return "well corroborated"
    else:
        return "highly corroborated"


def confidence_explain(belief_id: str) -> dict[str, Any]:
    """Explain confidence score for a belief."""
    with _common.get_cursor() as cur:
        cur.execute("SELECT * FROM beliefs WHERE id = %s", (belief_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Belief not found: {belief_id}"}

        belief = Belief.from_row(dict(row))
        conf = belief.confidence

        # Build explanation
        dimensions_dict: dict[str, Any] = {}
        weights_used: dict[str, float] = {}
        explanation: dict[str, Any] = {
            "success": True,
            "belief_id": belief_id,
            "content_preview": (belief.content[:100] + "..." if len(belief.content) > 100 else belief.content),
            "overall_confidence": conf.overall,
            "overall_label": confidence_label(conf.overall),
            "dimensions": dimensions_dict,
            "computation_method": "weighted_geometric_mean",
            "weights_used": weights_used,
        }

        # Document each dimension
        dimension_explanations = {
            "source_reliability": "How trustworthy is the information source? Higher for verified sources, lower for hearsay.",
            "method_quality": "How rigorous was the method of acquiring this knowledge? Higher for systematic analysis, lower for casual observation.",
            "internal_consistency": "Does this belief align with other beliefs? Higher if consistent, lower if it contradicts known facts.",
            "temporal_freshness": "How recent is this information? Higher for fresh data, decays over time.",
            "corroboration": "Is this supported by multiple independent sources? Higher with more confirmation.",
            "domain_applicability": "How relevant is this to the current context/domain? Higher if directly applicable.",
        }

        for dim in ConfidenceDimension:
            if dim == ConfidenceDimension.OVERALL:
                continue

            value = getattr(conf, dim.value, None)
            if value is not None:
                weight = DEFAULT_WEIGHTS.get(dim, 0)
                dimensions_dict[dim.value] = {
                    "value": value,
                    "label": confidence_label(value),
                    "weight": weight,
                    "explanation": dimension_explanations.get(dim.value, ""),
                }
                weights_used[dim.value] = weight

        # Add recommendations
        recommendations = []
        if conf.source_reliability is not None and conf.source_reliability < 0.5:
            recommendations.append("Consider verifying the source or finding corroborating evidence")
        if conf.corroboration is not None and conf.corroboration < 0.3:
            recommendations.append("This belief has low corroboration - seek additional sources")
        if conf.temporal_freshness is not None and conf.temporal_freshness < 0.5:
            recommendations.append("This information may be outdated - consider refreshing")
        if conf.internal_consistency is not None and conf.internal_consistency < 0.5:
            recommendations.append("This belief may conflict with other knowledge - review tensions")

        if recommendations:
            explanation["recommendations"] = recommendations
        else:
            explanation["recommendations"] = ["Confidence dimensions are balanced - no immediate concerns"]

        # Check for trust annotations
        try:
            cur.execute(
                """
                SELECT type, confidence_delta, created_at
                FROM belief_trust_annotations
                WHERE belief_id = %s
                AND (expires_at IS NULL OR expires_at > NOW())
                ORDER BY created_at DESC
                """,
                (belief_id,),
            )
            annotations = cur.fetchall()
            if annotations:
                explanation["trust_annotations"] = [
                    {
                        "type": a["type"],
                        "confidence_delta": float(a["confidence_delta"]),
                        "created_at": a["created_at"].isoformat(),
                    }
                    for a in annotations
                ]
        except Exception as e:
            logger.debug(f"Trust annotations table may not exist: {e}")

        return explanation
