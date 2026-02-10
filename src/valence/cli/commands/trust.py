"""Trust management commands."""

from __future__ import annotations

import argparse
import json


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register trust commands on the CLI parser."""
    trust_parser = subparsers.add_parser("trust", help="Trust network management")
    trust_subparsers = trust_parser.add_subparsers(dest="trust_command", required=True)

    # trust check
    trust_check_parser = trust_subparsers.add_parser("check", help="Check for trust concentration issues")
    trust_check_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    trust_check_parser.add_argument(
        "--single-threshold",
        type=float,
        default=None,
        help="Custom threshold for single node dominance (default: 30%%)",
    )
    trust_check_parser.add_argument(
        "--top3-threshold",
        type=float,
        default=None,
        help="Custom threshold for top 3 nodes dominance (default: 50%%)",
    )
    trust_check_parser.add_argument(
        "--min-sources",
        type=int,
        default=None,
        help="Minimum trusted sources (default: 3)",
    )

    # trust watch
    trust_watch_parser = trust_subparsers.add_parser("watch", help="Watch an entity (see content without reputation boost)")
    trust_watch_parser.add_argument("entity", help="DID of entity to watch")
    trust_watch_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # trust unwatch
    trust_unwatch_parser = trust_subparsers.add_parser("unwatch", help="Remove a watch relationship")
    trust_unwatch_parser.add_argument("entity", help="DID of entity to unwatch")
    trust_unwatch_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # trust distrust
    trust_distrust_parser = trust_subparsers.add_parser("distrust", help="Mark an entity as distrusted (negative reputation)")
    trust_distrust_parser.add_argument("entity", help="DID of entity to distrust")
    trust_distrust_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # trust ignore
    trust_ignore_parser = trust_subparsers.add_parser("ignore", help="Ignore an entity (block content)")
    trust_ignore_parser.add_argument("entity", help="DID of entity to ignore")
    trust_ignore_parser.add_argument("--domain", "-d", help="Optional domain scope")

    # trust set (#268)
    trust_set_parser = trust_subparsers.add_parser(
        "set",
        help="Set multi-dimensional epistemic trust on an entity",
    )
    trust_set_parser.add_argument("entity", help="Target entity DID")
    trust_set_parser.add_argument("--source", default=None, help="Source DID (default: self)")
    trust_set_parser.add_argument("--domain", default=None, help="Domain scope")
    trust_set_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    trust_set_parser.add_argument("--conclusions", type=float, default=None, help="Trust in their conclusions (0-1)")
    trust_set_parser.add_argument("--reasoning", type=float, default=None, help="Trust in their reasoning (0-1)")
    trust_set_parser.add_argument("--perspective", type=float, default=None, help="Trust in their perspective (0-1)")
    trust_set_parser.add_argument("--honesty", type=float, default=None, help="Trust in their honesty (0-1)")
    trust_set_parser.add_argument("--methodology", type=float, default=None, help="Trust in their methodology (0-1)")
    trust_set_parser.add_argument("--predictive", type=float, default=None, help="Trust in their predictions (0-1)")
    trust_set_parser.add_argument("--competence", type=float, default=None, help="Core competence score (0-1)")
    trust_set_parser.add_argument("--integrity", type=float, default=None, help="Core integrity score (0-1)")
    trust_set_parser.add_argument("--confidentiality", type=float, default=None, help="Core confidentiality score (0-1)")
    trust_set_parser.add_argument("--judgment", type=float, default=None, help="Core judgment score (0-1)")

    # trust show (#268)
    trust_show_parser = trust_subparsers.add_parser(
        "show",
        help="Show trust dimensions for an entity",
    )
    trust_show_parser.add_argument("entity", help="Target entity DID")
    trust_show_parser.add_argument("--source", default=None, help="Source DID (default: self)")
    trust_show_parser.add_argument("--domain", default=None, help="Domain scope")
    trust_show_parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    trust_parser.set_defaults(func=cmd_trust)


def cmd_trust_set(args: argparse.Namespace) -> int:
    """Set multi-dimensional epistemic trust on an entity (#268).

    Usage:
        valence trust set <entity> --conclusions 0.8 --reasoning 0.9 --honesty 0.7
    """
    from oro_privacy.trust import EPISTEMIC_DIMENSIONS, TrustService

    service = TrustService(use_memory=False)

    # Collect epistemic dimensions from args
    dimensions: dict[str, float] = {}
    for dim in EPISTEMIC_DIMENSIONS:
        value = getattr(args, dim, None)
        if value is not None:
            dimensions[dim] = value

    # Collect core 4D overrides
    competence = getattr(args, "competence", None)
    integrity = getattr(args, "integrity", None)
    confidentiality = getattr(args, "confidentiality", None)
    judgment = getattr(args, "judgment", None)

    if not dimensions and all(v is None for v in [competence, integrity, confidentiality, judgment]):
        print("❌ No trust dimensions specified. Use --conclusions, --reasoning, etc.")
        return 1

    try:
        edge = service.set_trust_dimensions(
            source_did=args.source or "self",
            target_did=args.entity,
            dimensions=dimensions,
            domain=getattr(args, "domain", None),
            competence=competence,
            integrity=integrity,
            confidentiality=confidentiality,
            judgment=judgment,
        )

        if args.json:
            print(json.dumps(edge.to_dict(), indent=2, default=str))
        else:
            print(f"✅ Trust updated: → {args.entity}")
            if dimensions:
                print("   Epistemic dimensions:")
                for dim, val in sorted(dimensions.items()):
                    print(f"     {dim}: {val:.2f}")
            if edge.epistemic_trust is not None:
                print(f"   Epistemic overall: {edge.epistemic_trust:.2f}")
            print(f"   Core overall: {edge.overall_trust:.2f}")

        return 0

    except Exception as e:
        print(f"❌ Failed to set trust: {e}")
        return 1


def cmd_trust_show(args: argparse.Namespace) -> int:
    """Show trust dimensions for an entity (#268)."""
    from oro_privacy.trust import TrustService

    service = TrustService(use_memory=False)

    try:
        edge = service.get_trust(
            source_did=args.source or "self",
            target_did=args.entity,
            domain=getattr(args, "domain", None),
        )

        if edge is None:
            print(f"No trust edge found for {args.entity}")
            return 1

        if args.json:
            print(json.dumps(edge.to_dict(), indent=2, default=str))
        else:
            print(f"🔍 Trust: → {args.entity}")
            print(f"   Schema: {edge.schema}")
            print("\n   Core dimensions:")
            print(f"     competence:      {edge.competence:.2f}")
            print(f"     integrity:       {edge.integrity:.2f}")
            print(f"     confidentiality: {edge.confidentiality:.2f}")
            print(f"     judgment:        {edge.judgment:.2f}")
            print(f"     overall:         {edge.overall_trust:.2f}")

            if edge.dimensions:
                print("\n   Epistemic dimensions:")
                for dim, val in sorted(edge.dimensions.items()):
                    print(f"     {dim}: {val:.2f}")
                eps = edge.epistemic_trust
                if eps is not None:
                    print(f"     epistemic overall: {eps:.2f}")

        return 0

    except Exception as e:
        print(f"❌ Failed to get trust: {e}")
        return 1


def cmd_trust_check(args: argparse.Namespace) -> int:
    """Check for trust concentration issues in the federation network."""
    from oro_federation.trust import check_trust_concentration
    from oro_federation.trust_policy import CONCENTRATION_THRESHOLDS

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


def cmd_trust_watch(args: argparse.Namespace) -> int:
    """Watch an entity (see content without giving reputation weight)."""
    from oro_privacy.trust import TrustService

    try:
        service = TrustService(use_memory=False)
        source_did = _get_own_did()
        target_did = args.entity

        edge = service.watch(
            source_did=source_did,
            target_did=target_did,
            domain=getattr(args, "domain", None),
        )
        print(f"👁️  Now watching {target_did}")
        if edge.domain:
            print(f"   Domain: {edge.domain}")
        print("   Content visible: ✅  Reputation boost: ❌  Worldview effect: ❌")
        return 0
    except Exception as e:
        print(f"❌ Watch failed: {e}")
        return 1


def cmd_trust_unwatch(args: argparse.Namespace) -> int:
    """Remove a watch relationship."""
    from oro_privacy.trust import TrustService

    try:
        service = TrustService(use_memory=False)
        source_did = _get_own_did()
        target_did = args.entity

        removed = service.unwatch(
            source_did=source_did,
            target_did=target_did,
            domain=getattr(args, "domain", None),
        )
        if removed:
            print(f"👁️  Unwatched {target_did}")
        else:
            print(f"ℹ️  No watch relationship found with {target_did}")
        return 0
    except Exception as e:
        print(f"❌ Unwatch failed: {e}")
        return 1


def cmd_trust_distrust(args: argparse.Namespace) -> int:
    """Mark an entity as distrusted (negative reputation weight)."""
    from oro_privacy.trust import TrustService

    try:
        service = TrustService(use_memory=False)
        source_did = _get_own_did()
        target_did = args.entity

        edge = service.distrust(
            source_did=source_did,
            target_did=target_did,
            domain=getattr(args, "domain", None),
        )
        print(f"⚠️  Now distrusting {target_did}")
        if edge.domain:
            print(f"   Domain: {edge.domain}")
        print("   Content visible: optional  Reputation: negative  Worldview: inverse")
        return 0
    except Exception as e:
        print(f"❌ Distrust failed: {e}")
        return 1


def cmd_trust_ignore(args: argparse.Namespace) -> int:
    """Ignore an entity (block content, no reputation effect)."""
    from oro_privacy.trust import TrustService

    try:
        service = TrustService(use_memory=False)
        source_did = _get_own_did()
        target_did = args.entity

        edge = service.ignore(
            source_did=source_did,
            target_did=target_did,
            domain=getattr(args, "domain", None),
        )
        print(f"🔇 Now ignoring {target_did}")
        if edge.domain:
            print(f"   Domain: {edge.domain}")
        print("   Content visible: ❌  Reputation boost: ❌  Worldview effect: ❌")
        return 0
    except Exception as e:
        print(f"❌ Ignore failed: {e}")
        return 1


def _get_own_did() -> str:
    """Get the user's own DID for CLI operations."""
    import os

    # Try environment first, then config
    did = os.environ.get("VALENCE_DID")
    if did:
        return did

    # Fall back to a reasonable default
    try:
        from ...core.config import get_config

        config = get_config()
        return getattr(config, "did", "did:key:local")
    except Exception:
        return "did:key:local"


def cmd_trust(args: argparse.Namespace) -> int:
    """Dispatch trust subcommands."""
    if args.trust_command == "check":
        return cmd_trust_check(args)
    elif args.trust_command == "set":
        return cmd_trust_set(args)
    elif args.trust_command == "show":
        return cmd_trust_show(args)
    elif args.trust_command == "watch":
        return cmd_trust_watch(args)
    elif args.trust_command == "unwatch":
        return cmd_trust_unwatch(args)
    elif args.trust_command == "distrust":
        return cmd_trust_distrust(args)
    elif args.trust_command == "ignore":
        return cmd_trust_ignore(args)
    else:
        print(f"Unknown trust command: {args.trust_command}")
        return 1
