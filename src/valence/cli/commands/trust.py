"""Trust and peer management CLI commands.

Commands for trust network management:
- peer add: Add or update trusted peers
- peer list: List trusted peers
- peer remove: Remove peers
- trust check: Check trust concentration
"""

from __future__ import annotations

import argparse
import json

from .beliefs import format_age


# ============================================================================
# PEER Commands
# ============================================================================

def cmd_peer_add(args: argparse.Namespace) -> int:
    """Add a trusted peer to the local registry."""
    from ...federation.peer_sync import get_trust_registry
    
    try:
        registry = get_trust_registry()
        peer = registry.add_peer(
            did=args.did,
            trust_level=args.trust,
            name=args.name,
            notes=args.notes,
        )
        
        print(f"✅ Peer added/updated")
        print(f"   DID:   {peer.did}")
        print(f"   Trust: {peer.trust_level:.0%}")
        if peer.name:
            print(f"   Name:  {peer.name}")
        return 0
        
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return 1
    except Exception as e:
        print(f"❌ Failed to add peer: {e}")
        return 1


def cmd_peer_list(args: argparse.Namespace) -> int:
    """List trusted peers."""
    from ...federation.peer_sync import get_trust_registry
    
    try:
        registry = get_trust_registry()
        peers = registry.list_peers()
        
        if not peers:
            print("📭 No trusted peers")
            print("\n💡 Add a peer with: valence peer add <did> --trust 0.8")
            return 0
        
        print(f"👥 {len(peers)} trusted peer(s)\n")
        
        for p in peers:
            trust_bar = '█' * int(p.trust_level * 10) + '░' * (10 - int(p.trust_level * 10))
            name_str = f" ({p.name})" if p.name else ""
            last_sync = format_age(p.last_sync_at) if p.last_sync_at else "never"
            
            print(f"  {p.did}{name_str}")
            print(f"    Trust: [{trust_bar}] {p.trust_level:.0%}")
            print(f"    Stats: ↓{p.beliefs_received} received, ↑{p.beliefs_sent} sent")
            print(f"    Synced: {last_sync}")
            print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to list peers: {e}")
        return 1


def cmd_peer_remove(args: argparse.Namespace) -> int:
    """Remove a peer from the trust registry."""
    from ...federation.peer_sync import get_trust_registry
    
    try:
        registry = get_trust_registry()
        
        if registry.remove_peer(args.did):
            print(f"✅ Removed peer: {args.did}")
            return 0
        else:
            print(f"⚠️  Peer not found: {args.did}")
            return 1
        
    except Exception as e:
        print(f"❌ Failed to remove peer: {e}")
        return 1


def cmd_peer(args: argparse.Namespace) -> int:
    """Dispatch peer subcommands."""
    if args.peer_command == 'add':
        return cmd_peer_add(args)
    elif args.peer_command == 'list':
        return cmd_peer_list(args)
    elif args.peer_command == 'remove':
        return cmd_peer_remove(args)
    else:
        print(f"Unknown peer command: {args.peer_command}")
        return 1


# ============================================================================
# TRUST Commands
# ============================================================================

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
        print(f"\n📊 Network Metrics:")
        print(f"   Total nodes:      {report.total_nodes}")
        print(f"   Active nodes:     {report.active_nodes}")
        print(f"   Trusted sources:  {report.trusted_sources}")
        print(f"   Total trust:      {report.total_trust:.2f}")
        print(f"   Top node share:   {report.top_node_share:.1%}")
        print(f"   Top 3 share:      {report.top_3_share:.1%}")
        if report.gini_coefficient is not None:
            gini_desc = "equal" if report.gini_coefficient < 0.3 else (
                "moderate" if report.gini_coefficient < 0.5 else "concentrated")
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
            print(f"\n✅ No trust concentration issues detected")
        
        print()
        
        return 0 if not report.has_critical_warnings else 1
        
    except Exception as e:
        print(f"❌ Trust check failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_trust(args: argparse.Namespace) -> int:
    """Dispatch trust subcommands."""
    if args.trust_command == 'check':
        return cmd_trust_check(args)
    else:
        print(f"Unknown trust command: {args.trust_command}")
        return 1
