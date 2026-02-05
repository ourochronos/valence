#!/usr/bin/env python3
"""Example 02: Trust Graph - Basic trust operations.

This example demonstrates Valence's trust system:
1. Understanding trust signals and how they affect reputation
2. Trust phases (probation → established → trusted)
3. Trust propagation across the network
4. Detecting trust concentration (anti-sybil)

Requirements:
    - PostgreSQL with pgvector running
    - `pip install valence` or run from source

Usage:
    python examples/02_trust_graph.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path when running from source
src_path = Path(__file__).parent.parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from valence.federation import (
    # Trust management
    TrustSignal,
    TrustManager,
    get_trust_manager,
    get_effective_trust,
    process_corroboration,
    check_trust_concentration,
    # Trust propagation
    TrustPropagation,
    get_trust_propagation,
    compute_transitive_trust,
    # Ring detection (anti-sybil)
    calculate_ring_coefficient,
    # Constants
    SIGNAL_WEIGHTS,
    PHASE_TRANSITION,
    CONCENTRATION_THRESHOLDS,
)
from valence.federation.models import NodeTrust, TrustPhase


def print_header(text: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def print_step(step: int, text: str) -> None:
    """Print a step description."""
    print(f"\n[Step {step}] {text}")
    print("-" * 40)


def main() -> None:
    """Run the trust graph example."""
    print_header("Valence Example 02: Trust Graph")

    print("""
Valence uses a multi-signal trust system where reputation is earned
through positive interactions, not claimed. Trust decays over time,
requiring ongoing good behavior to maintain.

Key concepts:
- Trust signals: Actions that increase or decrease trust
- Trust phases: Probation → Established → Trusted
- Trust propagation: Transitive trust through the network
- Ring detection: Identifying coordinated manipulation
""")

    # =========================================================================
    # Step 1: Understanding trust signals
    # =========================================================================
    print_step(1, "Trust signals and their weights")

    print("Trust is built through these signals:\n")
    for signal, weight in SIGNAL_WEIGHTS.items():
        direction = "+" if weight > 0 else ""
        print(f"  {signal.name:<25} {direction}{weight:.2f}")

    print("\nPositive signals (build trust):")
    print("  - BELIEF_CORROBORATED: Your belief was confirmed by others")
    print("  - QUERY_HELPFUL: Your query response was useful")
    print("  - CHALLENGE_VALID: You correctly challenged bad information")

    print("\nNegative signals (reduce trust):")
    print("  - BELIEF_CONTRADICTED: Your belief was shown to be wrong")
    print("  - SPAM_DETECTED: You sent low-quality or repetitive content")
    print("  - CHALLENGE_INVALID: You incorrectly challenged good information")

    # =========================================================================
    # Step 2: Trust phases
    # =========================================================================
    print_step(2, "Trust phases and transitions")

    print("Nodes progress through trust phases:\n")
    print("  PROBATION → ESTABLISHED → TRUSTED")
    print()
    print(f"  Probation → Established: {PHASE_TRANSITION['probation_to_established']:.2f} trust score")
    print(f"  Established → Trusted:   {PHASE_TRANSITION['established_to_trusted']:.2f} trust score")
    print(f"  Demotion threshold:      {PHASE_TRANSITION['demotion_threshold']:.2f} trust score")

    print("\nPhase benefits:")
    print("  PROBATION:    Limited sharing, beliefs weighted lower")
    print("  ESTABLISHED:  Normal participation, standard weights")
    print("  TRUSTED:      Priority in consensus, can vouch for others")

    # =========================================================================
    # Step 3: Simulating trust accumulation
    # =========================================================================
    print_step(3, "Simulating trust accumulation")

    # Create a mock node trust record
    node_did = "did:vkb:example-node-12345"
    
    print(f"Node: {node_did}")
    print("\nStarting trust: 0.50 (new node)")
    
    # Simulate a series of interactions
    interactions = [
        (TrustSignal.BELIEF_CORROBORATED, "belief confirmed by 2 peers"),
        (TrustSignal.QUERY_HELPFUL, "answered query accurately"),
        (TrustSignal.BELIEF_CORROBORATED, "another belief confirmed"),
        (TrustSignal.BELIEF_CONTRADICTED, "one belief was wrong"),
        (TrustSignal.QUERY_HELPFUL, "helpful response"),
        (TrustSignal.CHALLENGE_VALID, "correctly identified misinformation"),
    ]
    
    trust = 0.50  # Starting trust
    print("\nInteraction history:")
    
    for signal, description in interactions:
        weight = SIGNAL_WEIGHTS[signal]
        trust = max(0.0, min(1.0, trust + weight))  # Clamp 0-1
        direction = "+" if weight > 0 else ""
        print(f"  {signal.name:<25} ({direction}{weight:.2f}) → {trust:.2%} : {description}")
    
    # Determine phase
    if trust >= PHASE_TRANSITION["established_to_trusted"]:
        phase = "TRUSTED"
    elif trust >= PHASE_TRANSITION["probation_to_established"]:
        phase = "ESTABLISHED"
    else:
        phase = "PROBATION"
    
    print(f"\nFinal trust: {trust:.2%}")
    print(f"Trust phase: {phase}")

    # =========================================================================
    # Step 4: Trust propagation
    # =========================================================================
    print_step(4, "Trust propagation through the network")

    print("""
Valence supports transitive trust: if you trust Alice, and Alice
trusts Bob, you have some (reduced) trust in Bob.

Trust decays with each hop:
  - Direct trust: 100% of trust score
  - 1 hop away:   ~70% of trust score (decay factor)
  - 2 hops away:  ~49% of trust score
  - 3+ hops:      Rapidly diminishing
""")

    # Demonstrate transitive trust calculation
    print("Example trust chain:")
    print("  You → Alice (0.90) → Bob (0.80) → Carol (0.70)")
    print()
    
    decay = 0.7  # Default decay factor
    direct_alice = 0.90
    alice_to_bob = 0.80
    bob_to_carol = 0.70
    
    your_trust_bob = direct_alice * alice_to_bob * decay
    your_trust_carol = direct_alice * alice_to_bob * bob_to_carol * (decay ** 2)
    
    print(f"  Your trust in Alice (direct):   {direct_alice:.2%}")
    print(f"  Your trust in Bob (1 hop):      {your_trust_bob:.2%}")
    print(f"  Your trust in Carol (2 hops):   {your_trust_carol:.2%}")

    # =========================================================================
    # Step 5: Ring coefficient (anti-sybil)
    # =========================================================================
    print_step(5, "Ring coefficient (anti-sybil protection)")

    print("""
Valence detects coordinated trust manipulation through ring analysis.
If a group of nodes only trust each other (forming a "ring"), their
influence is dampened.

Ring coefficient: 0.0 (isolated ring) to 1.0 (well-connected)
""")

    print("Example scenarios:")
    print()
    print("  Healthy network (diverse connections):")
    print("    A → B, A → C, A → D")
    print("    B → A, B → E, B → F")
    print("    Ring coefficient: ~0.95 (high - good!)")
    print()
    print("  Suspicious ring (mutual endorsements only):")
    print("    A ↔ B ↔ C ↔ A")
    print("    Ring coefficient: ~0.30 (low - dampened)")
    print()
    print("  Sybil attack (fake nodes boosting one):")
    print("    Sybil1 → Target")
    print("    Sybil2 → Target")
    print("    Sybil3 → Target")
    print("    Ring coefficient: ~0.10 (very low - blocked)")

    print("\nConcentration thresholds:")
    for level, threshold in CONCENTRATION_THRESHOLDS.items():
        print(f"  {level:<10}: {threshold:.0%} from single source triggers warning")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Summary")

    print("""
What we demonstrated:

✓ Trust signals: How actions affect reputation
✓ Trust phases: Progressive trust levels
✓ Trust propagation: Transitive trust through the network  
✓ Ring detection: Anti-sybil protection

Key principles:
- Trust is earned through helpful behavior, not claimed
- Bad actors are attenuated, not banned (graceful degradation)
- Coordinated manipulation is detected and dampened
- Trust decays over time, requiring ongoing good behavior

Next steps:
- See 03_federation.py for peer-to-peer sharing with trust
""")


if __name__ == "__main__":
    # Check for database connection
    if not os.environ.get("VKB_DB_PASSWORD"):
        print("Warning: VKB_DB_PASSWORD not set. Using default 'valence'.")
        os.environ["VKB_DB_PASSWORD"] = "valence"

    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nThis example can run without a database connection.")
        print("If you see import errors, make sure valence is installed:")
        print("  pip install valence")
        sys.exit(1)
