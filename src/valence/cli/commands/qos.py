"""QoS management CLI commands.

Provides ``valence qos status`` and ``valence qos score`` subcommands
for inspecting contribution-based QoS state.

Issue #276: Network: Contribution-based QoS with dynamic curve.
"""

from __future__ import annotations

import argparse
import json


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register QoS commands on the CLI parser."""
    qos_parser = subparsers.add_parser("qos", help="Contribution-based QoS management")
    qos_subparsers = qos_parser.add_subparsers(dest="qos_command", required=True)

    # qos status
    qos_status_parser = qos_subparsers.add_parser("status", help="Show QoS system status")
    qos_status_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # qos score
    qos_score_parser = qos_subparsers.add_parser("score", help="Show contribution score for a node")
    qos_score_parser.add_argument("node_id", nargs="?", default=None, help="Node ID (default: self)")
    qos_score_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    qos_score_parser.add_argument(
        "--routing-capacity",
        type=float,
        default=None,
        dest="routing_capacity",
        help="Set routing_capacity score (0.0-1.0)",
    )
    qos_score_parser.add_argument(
        "--uptime-reliability",
        type=float,
        default=None,
        dest="uptime_reliability",
        help="Set uptime_reliability score (0.0-1.0)",
    )
    qos_score_parser.add_argument(
        "--belief-quality",
        type=float,
        default=None,
        dest="belief_quality",
        help="Set belief_quality score (0.0-1.0)",
    )
    qos_score_parser.add_argument(
        "--resource-sharing",
        type=float,
        default=None,
        dest="resource_sharing",
        help="Set resource_sharing score (0.0-1.0)",
    )
    qos_score_parser.add_argument(
        "--trust-received",
        type=float,
        default=None,
        dest="trust_received",
        help="Set trust_received score (0.0-1.0)",
    )

    qos_parser.set_defaults(func=cmd_qos)


def cmd_qos_status(args: argparse.Namespace) -> int:
    """Show QoS system status including policy, load, and tier summary."""
    from oro_network.qos import QoSPolicy
    from oro_network.qos_manager import QoSManager

    try:
        manager = QoSManager(policy=QoSPolicy())
        status = manager.get_status()

        if getattr(args, "json", False):
            print(json.dumps(status, indent=2))
            return 0

        # Human-readable output
        print("📊 QoS System Status")
        print("─" * 50)

        print("\n⚙️  Policy:")
        policy = status["policy"]
        print(f"   Min service floor:    {policy['min_service_floor']:.1%}")
        print(f"   Steepness range:      {policy['min_steepness']:.1f} – {policy['max_steepness']:.1f}")
        print(f"   New user grace:       {policy['new_user_grace_period']:.0f}s")
        print(
            f"   Tier thresholds:      high≥{policy['high_tier_threshold']:.0%}  normal≥{policy['normal_tier_threshold']:.0%}  low≥{policy['low_tier_threshold']:.0%}"
        )

        print("\n📈 Load:")
        load = status["load"]
        print(f"   Load factor:          {load['load_factor']:.1%}")
        print(f"   Connections:          {load['active_connections']}/{load['max_connections']}")
        print(f"   Avg queue depth:      {load['avg_queue_depth']:.1f}")
        print(f"   Avg latency:          {load['avg_latency_ms']:.1f}ms")
        print(f"   Current steepness:    {status['current_steepness']:.2f}")

        print(f"\n🔢 Tracked Nodes: {status['node_count']}")
        tiers = status["tier_summary"]
        for tier_name, count in tiers.items():
            bar = "█" * min(count, 40)
            print(f"   {tier_name:10s} {count:5d}  {bar}")

        print()
        return 0

    except Exception as e:
        print(f"❌ QoS status failed: {e}")
        return 1


def cmd_qos_score(args: argparse.Namespace) -> int:
    """Show contribution score for a node (or simulated default)."""
    from oro_network.qos import ContributionDimension, ContributionScore, QoSPolicy

    try:
        node_id = getattr(args, "node_id", None) or "self"
        score = ContributionScore(node_id=node_id)

        # If dimension values were provided, apply them.
        for dim in ContributionDimension:
            val = getattr(args, dim.value, None)
            if val is not None:
                score.set_dimension(dim, val)

        policy = QoSPolicy()

        if getattr(args, "json", False):
            data = score.to_dict()
            data["tier"] = policy.assign_tier(score.overall).value
            data["priority_at_load_0"] = round(policy.compute_priority(score.overall, 0.0), 4)
            data["priority_at_load_50"] = round(policy.compute_priority(score.overall, 0.5), 4)
            data["priority_at_load_100"] = round(policy.compute_priority(score.overall, 1.0), 4)
            print(json.dumps(data, indent=2))
            return 0

        # Human-readable output
        print(f"📊 Contribution Score: {node_id}")
        print("─" * 50)

        print("\n📐 Dimensions:")
        for dim in ContributionDimension:
            val = score.dimensions.get(dim, 0.0)
            weight = score.weights.get(dim, 0.0)
            bar_len = int(val * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"   {dim.value:20s}  {bar}  {val:.2f}  (w={weight:.2f})")

        overall = score.overall
        tier = policy.assign_tier(overall)
        print(f"\n   {'Overall':20s}  {'':20s}  {overall:.2f}")
        print(f"   Tier: {tier.value}")

        print("\n📈 Priority at different loads:")
        for load_pct in [0, 25, 50, 75, 100]:
            load = load_pct / 100.0
            pri = policy.compute_priority(overall, load)
            steep = policy.compute_steepness(load)
            print(f"   Load {load_pct:3d}%: priority={pri:.3f}  steepness={steep:.2f}")

        print()
        return 0

    except Exception as e:
        print(f"❌ QoS score failed: {e}")
        return 1


def cmd_qos(args: argparse.Namespace) -> int:
    """Dispatch QoS subcommands."""
    qos_command = getattr(args, "qos_command", None)
    if qos_command == "status":
        return cmd_qos_status(args)
    elif qos_command == "score":
        return cmd_qos_score(args)
    else:
        print("Usage: valence qos {status|score}")
        return 1
