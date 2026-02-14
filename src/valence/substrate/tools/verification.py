"""Verification protocol MCP tool implementations.

Handlers for verification submission, acceptance, disputes, reputation
queries, and bounty listing. Each function receives kwargs from the
MCP dispatch layer and returns a {success: bool, ...} dict.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from valence.core.exceptions import NotFoundError, ValidationException
from valence.core.verification.db import (
    accept_verification as db_accept_verification,
)
from valence.core.verification.db import (
    get_bounty as db_get_bounty,
)
from valence.core.verification.db import (
    get_dispute as db_get_dispute,
)
from valence.core.verification.db import (
    get_reputation,
)
from valence.core.verification.db import (
    get_reputation_events as db_get_reputation_events,
)
from valence.core.verification.db import (
    get_verification as db_get_verification,
)
from valence.core.verification.db import (
    get_verification_summary as db_get_verification_summary,
)
from valence.core.verification.db import (
    get_verifications_for_belief as db_get_verifications_for_belief,
)
from valence.core.verification.db import (
    list_bounties as db_list_bounties,
)
from valence.core.verification.db import (
    resolve_dispute as db_resolve_dispute,
)
from valence.core.verification.db import (
    submit_dispute as db_submit_dispute,
)
from valence.core.verification.db import (
    submit_verification as db_submit_verification,
)
from valence.core.verification.enums import (
    DisputeOutcome,
    DisputeType,
    EvidenceContribution,
    EvidenceType,
    ResolutionMethod,
    VerificationResult,
)
from valence.core.verification.evidence import Evidence
from valence.core.verification.results import ResultDetails

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================


def _parse_uuid(value: str, name: str) -> UUID | dict[str, Any]:
    """Parse a UUID string, returning error dict on failure."""
    try:
        return UUID(value)
    except (ValueError, AttributeError):
        return {"success": False, "error": f"Invalid UUID for {name}: {value}"}


def _parse_evidence(raw: list[dict[str, Any]]) -> list[Evidence]:
    """Convert list of dicts into Evidence objects."""
    from uuid import uuid4

    result = []
    for e in raw:
        result.append(
            Evidence(
                id=UUID(e["id"]) if "id" in e else uuid4(),
                type=EvidenceType(e.get("type", "external")),
                relevance=float(e.get("relevance", 0.5)),
                contribution=EvidenceContribution(e.get("contribution", "supports")),
            )
        )
    return result


# ============================================================================
# Verification Tools
# ============================================================================


def verification_submit(
    belief_id: str,
    verifier_id: str,
    result: str,
    evidence: list[dict[str, Any]],
    stake_amount: float,
    reasoning: str | None = None,
    result_details: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Submit a verification for a belief."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    try:
        vresult = VerificationResult(result)
    except ValueError:
        return {"success": False, "error": f"Invalid result '{result}'. Must be one of: {', '.join(r.value for r in VerificationResult)}"}

    parsed_evidence = _parse_evidence(evidence)
    details = None
    if result_details:
        details = ResultDetails(
            accuracy_estimate=result_details.get("accuracy_estimate"),
            confirmed_aspects=result_details.get("confirmed_aspects"),
            contradicted_aspects=result_details.get("contradicted_aspects"),
        )

    # We need belief_info — fetch from DB
    from ._common import get_cursor

    belief_info: dict[str, Any] = {}
    with get_cursor() as cur:
        cur.execute("SELECT source_id, confidence, domain_path FROM beliefs WHERE id = %s", (belief_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Belief {belief_id} not found"}

        import json

        conf = row.get("confidence", {})
        if isinstance(conf, str):
            conf = json.loads(conf)

        belief_info = {
            "holder_id": row.get("source_id", "unknown"),
            "confidence": conf,
            "domain_path": row.get("domain_path", []),
        }

    try:
        v = db_submit_verification(
            belief_id=bid,
            belief_info=belief_info,
            verifier_id=verifier_id,
            result=vresult,
            evidence=parsed_evidence,
            stake_amount=stake_amount,
            reasoning=reasoning,
            result_details=details,
        )
    except ValidationException as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "verification": {
            "id": str(v.id),
            "belief_id": str(v.belief_id),
            "verifier_id": v.verifier_id,
            "result": v.result.value,
            "status": v.status.value,
            "stake_amount": v.stake.amount,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        },
    }


def verification_accept(verification_id: str, **_: Any) -> dict[str, Any]:
    """Accept a pending verification."""
    vid = _parse_uuid(verification_id, "verification_id")
    if isinstance(vid, dict):
        return vid

    try:
        v = db_accept_verification(vid)
    except (NotFoundError, ValidationException) as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "verification": {
            "id": str(v.id),
            "status": v.status.value,
            "accepted_at": v.accepted_at.isoformat() if v.accepted_at else None,
        },
    }


def verification_get(verification_id: str, **_: Any) -> dict[str, Any]:
    """Get a verification by ID."""
    vid = _parse_uuid(verification_id, "verification_id")
    if isinstance(vid, dict):
        return vid

    v = db_get_verification(vid)
    if not v:
        return {"success": False, "error": f"Verification {verification_id} not found"}

    return {
        "success": True,
        "verification": {
            "id": str(v.id),
            "belief_id": str(v.belief_id),
            "verifier_id": v.verifier_id,
            "holder_id": v.holder_id,
            "result": v.result.value,
            "status": v.status.value,
            "reasoning": v.reasoning,
            "stake_amount": v.stake.amount,
            "evidence_count": len(v.evidence),
            "dispute_id": str(v.dispute_id) if v.dispute_id else None,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "accepted_at": v.accepted_at.isoformat() if v.accepted_at else None,
        },
    }


def verification_list(belief_id: str, **_: Any) -> dict[str, Any]:
    """List verifications for a belief."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    verifications = db_get_verifications_for_belief(bid)
    return {
        "success": True,
        "verifications": [
            {
                "id": str(v.id),
                "verifier_id": v.verifier_id,
                "result": v.result.value,
                "status": v.status.value,
                "stake_amount": v.stake.amount,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in verifications
        ],
        "total": len(verifications),
    }


def verification_summary(belief_id: str, **_: Any) -> dict[str, Any]:
    """Get verification summary for a belief."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    summary = db_get_verification_summary(bid)
    return {"success": True, **summary}


# ============================================================================
# Dispute Tools
# ============================================================================


def dispute_submit(
    verification_id: str,
    disputer_id: str,
    counter_evidence: list[dict[str, Any]],
    stake_amount: float,
    dispute_type: str,
    reasoning: str,
    proposed_result: str | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Submit a dispute against a verification."""
    vid = _parse_uuid(verification_id, "verification_id")
    if isinstance(vid, dict):
        return vid

    try:
        dtype = DisputeType(dispute_type)
    except ValueError:
        return {"success": False, "error": f"Invalid dispute_type '{dispute_type}'. Must be one of: {', '.join(d.value for d in DisputeType)}"}

    parsed_evidence = _parse_evidence(counter_evidence)
    proposed = VerificationResult(proposed_result) if proposed_result else None

    try:
        d = db_submit_dispute(
            verification_id=vid,
            disputer_id=disputer_id,
            counter_evidence=parsed_evidence,
            stake_amount=stake_amount,
            dispute_type=dtype,
            reasoning=reasoning,
            proposed_result=proposed,
        )
    except (NotFoundError, ValidationException) as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "dispute": {
            "id": str(d.id),
            "verification_id": str(d.verification_id),
            "disputer_id": d.disputer_id,
            "dispute_type": d.dispute_type.value,
            "status": d.status.value,
            "stake_amount": d.stake.amount,
            "created_at": d.created_at.isoformat() if d.created_at else None,
        },
    }


def dispute_resolve(
    dispute_id: str,
    outcome: str,
    resolution_reasoning: str,
    resolution_method: str = "automatic",
    **_: Any,
) -> dict[str, Any]:
    """Resolve a dispute."""
    did = _parse_uuid(dispute_id, "dispute_id")
    if isinstance(did, dict):
        return did

    try:
        doutcome = DisputeOutcome(outcome)
    except ValueError:
        return {"success": False, "error": f"Invalid outcome '{outcome}'. Must be one of: {', '.join(o.value for o in DisputeOutcome)}"}

    try:
        method = ResolutionMethod(resolution_method)
    except ValueError:
        method = ResolutionMethod.AUTOMATIC

    try:
        d = db_resolve_dispute(did, doutcome, resolution_reasoning, method)
    except (NotFoundError, ValidationException) as e:
        return {"success": False, "error": str(e)}

    return {
        "success": True,
        "dispute": {
            "id": str(d.id),
            "outcome": d.outcome.value if d.outcome else None,
            "status": d.status.value,
            "resolved_at": d.resolved_at.isoformat() if d.resolved_at else None,
        },
    }


def dispute_get(dispute_id: str, **_: Any) -> dict[str, Any]:
    """Get a dispute by ID."""
    did = _parse_uuid(dispute_id, "dispute_id")
    if isinstance(did, dict):
        return did

    d = db_get_dispute(did)
    if not d:
        return {"success": False, "error": f"Dispute {dispute_id} not found"}

    return {
        "success": True,
        "dispute": {
            "id": str(d.id),
            "verification_id": str(d.verification_id),
            "disputer_id": d.disputer_id,
            "dispute_type": d.dispute_type.value,
            "reasoning": d.reasoning,
            "status": d.status.value,
            "outcome": d.outcome.value if d.outcome else None,
            "resolution_reasoning": d.resolution_reasoning,
            "stake_amount": d.stake.amount,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "resolved_at": d.resolved_at.isoformat() if d.resolved_at else None,
        },
    }


# ============================================================================
# Reputation Tools
# ============================================================================


def reputation_get(identity_id: str, **_: Any) -> dict[str, Any]:
    """Get reputation for an identity."""
    rep = get_reputation(identity_id)
    if not rep:
        return {"success": False, "error": f"No reputation found for {identity_id}"}

    return {
        "success": True,
        "reputation": {
            "identity_id": rep.identity_id,
            "overall": rep.overall,
            "by_domain": rep.by_domain,
            "verification_count": rep.verification_count,
            "discrepancy_finds": rep.discrepancy_finds,
            "stake_at_risk": rep.stake_at_risk,
        },
    }


def reputation_events(identity_id: str, limit: int = 50, **_: Any) -> dict[str, Any]:
    """Get reputation event history for an identity."""
    events = db_get_reputation_events(identity_id, limit=limit)
    return {
        "success": True,
        "events": [
            {
                "id": str(e.id),
                "delta": e.delta,
                "old_value": e.old_value,
                "new_value": e.new_value,
                "reason": e.reason,
                "dimension": e.dimension,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
        "total": len(events),
    }


# ============================================================================
# Bounty Tools
# ============================================================================


def bounty_get(belief_id: str, **_: Any) -> dict[str, Any]:
    """Get discrepancy bounty for a belief."""
    bid = _parse_uuid(belief_id, "belief_id")
    if isinstance(bid, dict):
        return bid

    b = db_get_bounty(bid)
    if not b:
        return {"success": False, "error": f"No bounty found for belief {belief_id}"}

    return {
        "success": True,
        "bounty": {
            "belief_id": str(b.belief_id),
            "holder_id": b.holder_id,
            "total_bounty": b.total_bounty,
            "base_amount": b.base_amount,
            "confidence_premium": b.confidence_premium,
            "age_factor": b.age_factor,
            "claimed": b.claimed,
            "claimed_by": b.claimed_by,
            "created_at": b.created_at.isoformat() if b.created_at else None,
            "expires_at": b.expires_at.isoformat() if b.expires_at else None,
        },
    }


def bounty_list(unclaimed_only: bool = True, limit: int = 20, **_: Any) -> dict[str, Any]:
    """List discrepancy bounties."""
    bounties = db_list_bounties(unclaimed_only=unclaimed_only, limit=limit)
    return {
        "success": True,
        "bounties": [
            {
                "belief_id": str(b.belief_id),
                "holder_id": b.holder_id,
                "total_bounty": b.total_bounty,
                "claimed": b.claimed,
                "created_at": b.created_at.isoformat() if b.created_at else None,
            }
            for b in bounties
        ],
        "total": len(bounties),
    }
