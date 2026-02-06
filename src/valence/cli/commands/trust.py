"""Trust management commands."""

from __future__ import annotations

import argparse
import json


def cmd_trust_check(args: argparse.Namespace) -> int:
    """Check for trust concentration issues in the federation network."""
    from ...federation.trust import check_trust_concentration
    from ...federation.trust_policy import CONCENTRATION_THRESHOLDS

    # Build custom thresholds if provided
    thresholds = dict(CONCENTRATION_THRESHOLDS)  # Copy defaults
    if args.single_threshold is not None:
        thresholds["single_node_warning"] = args.single_threshold
    if args.top3_threshold is not None:
        thresholds["top_3_warning"] = args.top3_threshold
    if args.min_sources is not None:
        thresholds["min_trusted_sources"] = args.min_sources

    try:
        report = check_trust_concentration(thresholds)

        # JSON output
        if args.json:
            print(json.dumps(report.to_dict(), indent=2))
            return 0 if not report.has_critical_warnings else 1

        # Human-readable output
        print("🔍 Trust Concentration Analysis")
        print("─" * 50)

        # Network metrics
        print("\n📊 Network Metrics:")
        print(f"   Total nodes:      {report.total_nodes}")
        print(f"   Active nodes:     {report.active_nodes}")
        print(f"   Trusted sources:  {report.trusted_sources}")
        print(f"   Total trust:      {report.total_trust:.2f}")
        print(f"   Top node share:   {report.top_node_share:.1%}")
        print(f"   Top 3 share:      {report.top_3_share:.1%}")
        if report.gini_coefficient is not None:
            gini_desc = "equal" if report.gini_coefficient < 0.3 else ("moderate" if report.gini_coefficient < 0.5 else "concentrated")
            print(f"   Gini coefficient: {report.gini_coefficient:.2f} ({gini_desc})")

        # Warnings
        if report.warnings:
            print(f"\n⚠️  Warnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"\n   {warning}")
                if warning.node_name:
                    print(f"      Node: {warning.node_name}")
                if warning.recommendation:
                    print(f"      💡 {warning.recommendation}")
        else:
            print("\n✅ No trust concentration issues detected")

        print()

        return 0 if not report.has_critical_warnings else 1

    except Exception as e:
        print(f"❌ Trust check failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_trust(args: argparse.Namespace) -> int:
    """Dispatch trust subcommands."""
    if args.trust_command == "check":
        return cmd_trust_check(args)
    else:
        print(f"Unknown trust command: {args.trust_command}")
        return 1
