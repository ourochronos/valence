"""Reputation and stake calculation functions.

Contains all functions for calculating stakes, rewards, penalties,
and bounties in the verification protocol.
"""

from __future__ import annotations

import math

from .constants import ReputationConstants


# ============================================================================
# Stake Calculation Functions
# ============================================================================

def calculate_min_stake(
    belief_confidence: float,
    verifier_domain_reputation: float = 0.5,
) -> float:
    """Calculate minimum stake required for verification.
    
    Formula: min_stake = base_stake × confidence_multiplier × domain_multiplier
    
    Args:
        belief_confidence: Overall confidence of the belief (0.0-1.0)
        verifier_domain_reputation: Verifier's reputation in the belief's domain
    
    Returns:
        Minimum stake amount
    """
    base_stake = 0.01  # 1% of neutral reputation
    confidence_multiplier = belief_confidence
    domain_multiplier = 1.0 + (verifier_domain_reputation * 0.5)
    
    return base_stake * confidence_multiplier * domain_multiplier


def calculate_max_stake(verifier_reputation: float) -> float:
    """Calculate maximum stake a verifier can make.
    
    Capped at 20% of verifier's current overall reputation.
    
    Args:
        verifier_reputation: Verifier's overall reputation
    
    Returns:
        Maximum stake amount
    """
    return verifier_reputation * ReputationConstants.MAX_STAKE_RATIO


def calculate_dispute_min_stake(
    verification_stake: float,
    is_holder: bool = False,
) -> float:
    """Calculate minimum stake for filing a dispute.
    
    Args:
        verification_stake: Stake of the original verification
        is_holder: True if disputer is the belief holder
    
    Returns:
        Minimum dispute stake
    """
    multiplier = 1.0 if is_holder else 1.5
    return verification_stake * multiplier


def calculate_bounty(
    holder_stake: float,
    belief_confidence: float,
    days_since_creation: int,
    domain_importance: float = 1.0,
) -> float:
    """Calculate discrepancy bounty for a belief.
    
    Formula: bounty = holder_stake × confidence_premium × age_factor × domain_multiplier
    
    Args:
        holder_stake: Stake the holder put on the belief
        belief_confidence: Overall confidence of the belief
        days_since_creation: Days since belief was created
        domain_importance: Importance weight of the belief's domain
    
    Returns:
        Total bounty amount
    """
    confidence_premium = belief_confidence ** 2
    age_factor = min(2.0, 1.0 + days_since_creation / 30)
    
    return holder_stake * ReputationConstants.BOUNTY_MULTIPLIER * confidence_premium * age_factor * domain_importance


# ============================================================================
# Reputation Update Functions
# ============================================================================

def calculate_confirmation_reward(
    stake: float,
    min_stake: float,
    belief_confidence: float,
    existing_confirmations: int,
) -> float:
    """Calculate reward for a CONFIRMED verification.
    
    Formula: base_reward × stake_multiplier × confidence_factor × diminishing_factor
    
    Args:
        stake: Amount staked by verifier
        min_stake: Minimum required stake
        belief_confidence: Confidence of the verified belief
        existing_confirmations: Number of existing confirmations
    
    Returns:
        Reputation reward
    """
    base_reward = ReputationConstants.CONFIRMATION_BASE
    stake_multiplier = min(2.0, stake / min_stake) if min_stake > 0 else 1.0
    confidence_factor = belief_confidence
    diminishing_factor = 1.0 / math.sqrt(existing_confirmations + 1)
    
    return base_reward * stake_multiplier * confidence_factor * diminishing_factor


def calculate_contradiction_reward(
    stake: float,
    min_stake: float,
    belief_confidence: float,
    is_first_contradiction: bool,
    existing_contradictions: int = 0,
) -> float:
    """Calculate reward for a CONTRADICTED verification.
    
    Formula: base_bounty × stake_multiplier × confidence_premium × novelty_bonus
    
    Args:
        stake: Amount staked by verifier
        min_stake: Minimum required stake
        belief_confidence: Confidence of the verified belief
        is_first_contradiction: Whether this is the first contradiction
        existing_contradictions: Number of existing contradictions
    
    Returns:
        Reputation reward
    """
    base_bounty = ReputationConstants.CONTRADICTION_BASE
    stake_multiplier = min(3.0, stake / min_stake) if min_stake > 0 else 1.0
    confidence_premium = belief_confidence ** 2
    
    if is_first_contradiction:
        novelty_bonus = ReputationConstants.FIRST_FINDER_BONUS
    else:
        novelty_bonus = 1.0 / math.sqrt(existing_contradictions + 1)
    
    return base_bounty * stake_multiplier * confidence_premium * novelty_bonus


def calculate_holder_confirmation_bonus(
    verifier_reputation: float,
    stake: float,
    min_stake: float,
) -> float:
    """Calculate holder's bonus when their belief is confirmed.
    
    Args:
        verifier_reputation: Reputation of the verifier
        stake: Stake amount
        min_stake: Minimum required stake
    
    Returns:
        Reputation bonus for holder
    """
    return 0.0005 * verifier_reputation * math.sqrt(stake / min_stake) if min_stake > 0 else 0.0


def calculate_holder_contradiction_penalty(
    belief_confidence: float,
    verifier_reputation: float,
) -> float:
    """Calculate holder's penalty when their belief is contradicted.
    
    Formula: base_penalty × overconfidence_multiplier × verifier_weight
    
    Args:
        belief_confidence: Confidence of the contradicted belief
        verifier_reputation: Reputation of the verifier
    
    Returns:
        Reputation penalty for holder (positive number)
    """
    base_penalty = ReputationConstants.CONTRADICTION_PENALTY_BASE
    overconfidence_multiplier = belief_confidence ** 2
    verifier_weight = verifier_reputation
    
    return base_penalty * overconfidence_multiplier * verifier_weight


def calculate_partial_reward(
    accuracy_estimate: float,
    stake: float,
    min_stake: float,
    belief_confidence: float,
    existing_confirmations: int,
    existing_contradictions: int,
) -> float:
    """Calculate reward for a PARTIAL verification.
    
    Proportional credit based on accuracy estimate.
    
    Args:
        accuracy_estimate: Portion of belief that's accurate (0.0-1.0)
        stake: Amount staked
        min_stake: Minimum required stake
        belief_confidence: Confidence of the belief
        existing_confirmations: Number of existing confirmations
        existing_contradictions: Number of existing contradictions
    
    Returns:
        Reputation reward
    """
    confirm_portion = calculate_confirmation_reward(
        stake, min_stake, belief_confidence, existing_confirmations
    ) * accuracy_estimate
    
    contradict_portion = calculate_contradiction_reward(
        stake, min_stake, belief_confidence, 
        existing_contradictions == 0, existing_contradictions
    ) * (1 - accuracy_estimate)
    
    return confirm_portion + contradict_portion
