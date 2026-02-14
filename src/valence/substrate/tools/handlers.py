"""Substrate tool dispatch and handler mapping.

Provides:
    SUBSTRATE_HANDLERS -- tool name to handler function mapping
    handle_substrate_tool -- dispatch function
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .backup import (
    backup_create,
    backup_get,
    backup_list,
    backup_verify,
)
from .beliefs import (
    belief_create,
    belief_get,
    belief_query,
    belief_search,
    belief_supersede,
)
from .confidence import belief_corroboration, confidence_explain
from .consensus import (
    challenge_get,
    challenge_resolve,
    challenge_submit,
    challenges_list,
    consensus_status,
    corroboration_list,
    corroboration_submit,
)
from .entities import entity_get, entity_search
from .incentives import (
    calibration_history,
    calibration_run,
    reward_claim,
    rewards_claim_all,
    rewards_pending,
    transfer_history,
    velocity_status,
)
from .sharing import belief_share, belief_share_revoke, belief_shares_list
from .tensions import tension_list, tension_resolve
from .trust import trust_check
from .verification import (
    bounty_get,
    bounty_list,
    dispute_get,
    dispute_resolve,
    dispute_submit,
    reputation_events,
    reputation_get,
    verification_accept,
    verification_get,
    verification_list,
    verification_submit,
    verification_summary,
)

# Tool name to handler mapping
SUBSTRATE_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    "belief_query": belief_query,
    "belief_create": belief_create,
    "belief_supersede": belief_supersede,
    "belief_get": belief_get,
    "entity_get": entity_get,
    "entity_search": entity_search,
    "tension_list": tension_list,
    "tension_resolve": tension_resolve,
    "belief_corroboration": belief_corroboration,
    "belief_search": belief_search,
    "trust_check": trust_check,
    "confidence_explain": confidence_explain,
    "belief_share": belief_share,
    "belief_shares_list": belief_shares_list,
    "belief_share_revoke": belief_share_revoke,
    # Verification protocol
    "verification_submit": verification_submit,
    "verification_accept": verification_accept,
    "verification_get": verification_get,
    "verification_list": verification_list,
    "verification_summary": verification_summary,
    "dispute_submit": dispute_submit,
    "dispute_resolve": dispute_resolve,
    "dispute_get": dispute_get,
    "reputation_get": reputation_get,
    "reputation_events": reputation_events,
    "bounty_get": bounty_get,
    "bounty_list": bounty_list,
    # Incentive system
    "calibration_run": calibration_run,
    "calibration_history": calibration_history,
    "rewards_pending": rewards_pending,
    "reward_claim": reward_claim,
    "rewards_claim_all": rewards_claim_all,
    "transfer_history": transfer_history,
    "velocity_status": velocity_status,
    # Consensus mechanism
    "consensus_status": consensus_status,
    "corroboration_submit": corroboration_submit,
    "corroboration_list": corroboration_list,
    "challenge_submit": challenge_submit,
    "challenge_resolve": challenge_resolve,
    "challenge_get": challenge_get,
    "challenges_list": challenges_list,
    # Resilient storage
    "backup_create": backup_create,
    "backup_verify": backup_verify,
    "backup_list": backup_list,
    "backup_get": backup_get,
}


def handle_substrate_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Handle a substrate tool call.

    Args:
        name: Tool name
        arguments: Tool arguments

    Returns:
        Tool result dictionary
    """
    handler = SUBSTRATE_HANDLERS.get(name)
    if handler is None:
        return {"success": False, "error": f"Unknown substrate tool: {name}"}

    return handler(**arguments)
