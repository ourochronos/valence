"""Substrate tool definitions and implementations.

Tool implementations extracted from mcp_server.py for use in the unified HTTP server.
Descriptions include behavioral conditioning for proactive Claude usage.

This package re-exports all public names so that existing imports like
``from valence.substrate.tools import belief_query`` continue to work.
"""

from __future__ import annotations

# Re-export shared utilities (also ensures patch target backward compat)
from ._common import _validate_enum, get_cursor  # noqa: F401
from .backup import (  # noqa: F401
    backup_create,
    backup_get,
    backup_list,
    backup_verify,
)

# Re-export all tool implementations
from .beliefs import (  # noqa: F401
    _content_hash,
    _log_retrievals,
    _reinforce_belief,
    belief_create,
    belief_get,
    belief_query,
    belief_search,
    belief_supersede,
)
from .confidence import (  # noqa: F401
    _corroboration_label,
    belief_corroboration,
    confidence_explain,
)
from .consensus import (  # noqa: F401
    challenge_get,
    challenge_resolve,
    challenge_submit,
    challenges_list,
    consensus_status,
    corroboration_list,
    corroboration_submit,
)

# Re-export tool definitions
from .definitions import SUBSTRATE_TOOLS  # noqa: F401
from .entities import entity_get, entity_search  # noqa: F401

# Re-export handler dispatch
from .handlers import SUBSTRATE_HANDLERS, handle_substrate_tool  # noqa: F401
from .incentives import (  # noqa: F401
    calibration_history,
    calibration_run,
    reward_claim,
    rewards_claim_all,
    rewards_pending,
    transfer_history,
    velocity_status,
)
from .sharing import (  # noqa: F401
    _get_local_did,
    belief_share,
    belief_share_revoke,
    belief_shares_list,
)
from .tensions import tension_list, tension_resolve  # noqa: F401
from .trust import trust_check  # noqa: F401
from .verification import (  # noqa: F401
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

__all__ = [
    # Definitions
    "SUBSTRATE_TOOLS",
    # Handlers
    "SUBSTRATE_HANDLERS",
    "handle_substrate_tool",
    # Beliefs
    "belief_query",
    "belief_create",
    "belief_supersede",
    "belief_get",
    "belief_search",
    "_content_hash",
    "_reinforce_belief",
    "_log_retrievals",
    # Entities
    "entity_get",
    "entity_search",
    # Tensions
    "tension_list",
    "tension_resolve",
    # Confidence
    "confidence_explain",
    "belief_corroboration",
    "_corroboration_label",
    # Trust
    "trust_check",
    # Sharing
    "belief_share",
    "belief_shares_list",
    "belief_share_revoke",
    "_get_local_did",
    "_validate_enum",
    # Shared (for patching)
    "get_cursor",
    # Verification protocol
    "verification_submit",
    "verification_accept",
    "verification_get",
    "verification_list",
    "verification_summary",
    "dispute_submit",
    "dispute_resolve",
    "dispute_get",
    "reputation_get",
    "reputation_events",
    "bounty_get",
    "bounty_list",
    # Incentive system
    "calibration_run",
    "calibration_history",
    "rewards_pending",
    "reward_claim",
    "rewards_claim_all",
    "transfer_history",
    "velocity_status",
    # Consensus mechanism
    "consensus_status",
    "corroboration_submit",
    "corroboration_list",
    "challenge_submit",
    "challenge_resolve",
    "challenge_get",
    "challenges_list",
    # Resilient storage
    "backup_create",
    "backup_verify",
    "backup_list",
    "backup_get",
]
