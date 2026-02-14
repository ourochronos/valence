"""Consensus mechanism MCP tool implementations.

Handlers for consensus status, corroboration, challenges.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from valence.core.consensus import (
    calculate_independence,
)
from valence.core.consensus import (
    get_challenge as db_get_challenge,
)
from valence.core.consensus import (
    get_challenges_for_belief as db_get_challenges,
)
from valence.core.consensus import (
    get_consensus_status as db_get_consensus_status,
)
from valence.core.consensus import (
    get_corroborations as db_get_corroborations,
)
from valence.core.consensus import (
    resolve_challenge as db_resolve_challenge,
)
from valence.core.consensus import (
    submit_challenge as db_submit_challenge,
)
from valence.core.consensus import (
    submit_corroboration as db_submit_corroboration,
)
from valence.core.exceptions import NotFoundError, ValidationException

logger = logging.getLogger(__name__)


def _parse_uuid(value: str, name: str) -> UUID | dict[str, Any]:
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return {"success": False, "error": f"Invalid UUID for {name}: {value}"}


# ============================================================================
# Consensus Status
# ============================================================================


def consensus_status(belief_id: str, **_: Any) -> dict[str, Any]:
    """Get consensus status for a belief."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    status = db_get_consensus_status(bid)
    if not status:
        return {"success": True, "status": None, "message": "No consensus status found (L1 personal belief)"}

    return {
        "success": True,
        "status": {
            "belief_id": str(status.belief_id),
            "current_layer": status.current_layer.value,
            "corroboration_count": status.corroboration_count,
            "total_corroboration_weight": status.total_corroboration_weight,
            "finality": status.finality.value,
            "last_challenge_at": status.last_challenge_at.isoformat() if status.last_challenge_at else None,
            "elevated_at": status.elevated_at.isoformat() if status.elevated_at else None,
        },
    }


# ============================================================================
# Corroboration
# ============================================================================


def corroboration_submit(
    primary_belief_id: str,
    corroborating_belief_id: str,
    primary_holder: str,
    corroborator: str,
    semantic_similarity: float,
    evidence_sources_a: list[str] | None = None,
    evidence_sources_b: list[str] | None = None,
    method_a: str | None = None,
    method_b: str | None = None,
    corroborator_reputation: float = 0.5,
    **_: Any,
) -> dict[str, Any]:
    """Submit a corroboration between two beliefs."""
    pbid = _parse_uuid(primary_belief_id, "primary_belief_id")
    if isinstance(pbid, dict):
        return pbid
    cbid = _parse_uuid(corroborating_belief_id, "corroborating_belief_id")
    if isinstance(cbid, dict):
        return cbid

    independence = calculate_independence(
        evidence_sources_a or [],
        evidence_sources_b or [],
        method_a,
        method_b,
    )

    try:
        corr = db_submit_corroboration(
            primary_belief_id=pbid,
            corroborating_belief_id=cbid,
            primary_holder=primary_holder,
            corroborator=corroborator,
            semantic_similarity=semantic_similarity,
            independence=independence,
            corroborator_reputation=corroborator_reputation,
        )
    except ValidationException as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "corroboration": {
            "id": str(corr.id),
            "effective_weight": corr.effective_weight,
            "independence": corr.independence.to_dict(),
        },
    }


def corroboration_list(belief_id: str, **_: Any) -> dict[str, Any]:
    """List corroborations for a belief."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    corrs = db_get_corroborations(bid)
    return {
        "success": True,
        "corroborations": [
            {
                "id": str(c.id),
                "corroborating_belief_id": str(c.corroborating_belief_id),
                "corroborator": c.corroborator,
                "semantic_similarity": c.semantic_similarity,
                "effective_weight": c.effective_weight,
                "independence_overall": c.independence.overall,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in corrs
        ],
        "total": len(corrs),
    }


# ============================================================================
# Challenges
# ============================================================================


def challenge_submit(
    belief_id: str,
    challenger_id: str,
    reasoning: str,
    evidence: list[dict[str, Any]] | None = None,
    stake_amount: float = 0.0,
    **_: Any,
) -> dict[str, Any]:
    """Submit a challenge to a belief's consensus status."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    try:
        challenge = db_submit_challenge(bid, challenger_id, reasoning, evidence, stake_amount)
    except (NotFoundError, ValidationException) as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "challenge": {
            "id": str(challenge.id),
            "target_layer": challenge.target_layer.value,
            "status": challenge.status.value,
            "created_at": challenge.created_at.isoformat() if challenge.created_at else None,
        },
    }


def challenge_resolve(
    challenge_id: str,
    upheld: bool,
    resolution_reasoning: str,
    **_: Any,
) -> dict[str, Any]:
    """Resolve a pending challenge."""
    cid = _parse_uuid(challenge_id, "challenge_id")
    if isinstance(cid, dict):
        return cid

    try:
        challenge = db_resolve_challenge(cid, upheld, resolution_reasoning)
    except (NotFoundError, ValidationException) as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "challenge": {
            "id": str(challenge.id),
            "status": challenge.status.value,
            "resolved_at": challenge.resolved_at.isoformat() if challenge.resolved_at else None,
        },
    }


def challenge_get(challenge_id: str, **_: Any) -> dict[str, Any]:
    """Get a challenge by ID."""
    cid = _parse_uuid(challenge_id, "challenge_id")
    if isinstance(cid, dict):
        return cid

    challenge = db_get_challenge(cid)
    if not challenge:
        return {"success": False, "error": f"Challenge {challenge_id} not found"}

    return {
        "success": True,
        "challenge": {
            "id": str(challenge.id),
            "belief_id": str(challenge.belief_id),
            "challenger_id": challenge.challenger_id,
            "target_layer": challenge.target_layer.value,
            "reasoning": challenge.reasoning,
            "status": challenge.status.value,
            "resolution_reasoning": challenge.resolution_reasoning,
            "stake_amount": challenge.stake_amount,
            "created_at": challenge.created_at.isoformat() if challenge.created_at else None,
            "resolved_at": challenge.resolved_at.isoformat() if challenge.resolved_at else None,
        },
    }


def challenges_list(belief_id: str, **_: Any) -> dict[str, Any]:
    """List challenges for a belief."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    challenges = db_get_challenges(bid)
    return {
        "success": True,
        "challenges": [
            {
                "id": str(c.id),
                "challenger_id": c.challenger_id,
                "target_layer": c.target_layer.value,
                "status": c.status.value,
                "stake_amount": c.stake_amount,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in challenges
        ],
        "total": len(challenges),
    }
