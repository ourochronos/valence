"""Federation CLI commands.

Commands for federated knowledge sharing:
- export: Export beliefs for sharing
- import: Import beliefs from peers
- query (federated): Search across federated beliefs
"""

from __future__ import annotations

import argparse
import json


# ============================================================================
# FEDERATED QUERY
# ============================================================================

def cmd_query_federated(args: argparse.Namespace) -> int:
    """Search beliefs with source attribution (federated scope)."""
    from ...federation.peer_sync import query_federated
    
    try:
        results = query_federated(
            query_text=args.query,
            scope=getattr(args, 'scope', 'local'),
            domain_filter=[args.domain] if args.domain else None,
            limit=args.limit,
            threshold=args.threshold,
        )
        
        if not results:
            print(f"🔍 No beliefs found for: {args.query}")
            return 0
        
        print(f"🔍 Found {len(results)} belief(s) for: {args.query}\n")
        
        for i, r in enumerate(results, 1):
            sim_pct = f"{r.similarity:.0%}"
            conf_pct = f"{r.effective_confidence:.0%}"
            
            # Header with source attribution
            print(f"{'─' * 60}")
            print(f"[{i}] {r.content[:70]}{'...' if len(r.content) > 70 else ''}")
            print(f"    ID: {r.id[:8]}  Confidence: {conf_pct}  Similarity: {sim_pct}")
            
            if r.domain_path:
                print(f"    Domains: {', '.join(r.domain_path)}")
            
            # Source attribution
            if r.is_local:
                print(f"    📍 Source: LOCAL")
            else:
                trust_pct = f"{r.origin_trust:.0%}" if r.origin_trust else "?"
                print(f"    🔗 Source: {r.origin_did} (trust: {trust_pct})")
                
                # Show original vs weighted confidence
                original = r.confidence.get('_original_overall')
                if original:
                    print(f"       Original confidence: {original:.0%} → weighted: {r.effective_confidence:.0%}")
        
        print(f"{'─' * 60}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
# EXPORT Command
# ============================================================================

def cmd_export(args: argparse.Namespace) -> int:
    """Export beliefs for sharing with a peer."""
    from ...federation.peer_sync import export_beliefs
    
    try:
        package = export_beliefs(
            recipient_did=args.to,
            domain_filter=args.domain,
            min_confidence=args.min_confidence,
            limit=args.limit,
            include_federated=args.include_federated,
        )
        
        if not package.beliefs:
            print("📭 No beliefs to export")
            return 0
        
        # Output to file or stdout
        json_output = package.to_json()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            print(f"✅ Exported {len(package.beliefs)} beliefs to {args.output}")
        else:
            print(json_output)
        
        # Show summary unless quiet
        if args.output:
            print(f"\n📦 Export Summary:")
            print(f"   Beliefs: {len(package.beliefs)}")
            print(f"   From:    {package.exporter_did}")
            if package.recipient_did:
                print(f"   To:      {package.recipient_did}")
            if package.domain_summary:
                print(f"   Domains: {', '.join(f'{k}({v})' for k,v in package.domain_summary.items())}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


# ============================================================================
# IMPORT Command  
# ============================================================================

def cmd_import(args: argparse.Namespace) -> int:
    """Import beliefs from a peer."""
    import sys
    from ...federation.peer_sync import ExportPackage, import_beliefs, get_trust_registry
    
    try:
        # Read input file
        if args.file == '-':
            json_str = sys.stdin.read()
        else:
            with open(args.file) as f:
                json_str = f.read()
        
        # Parse package
        try:
            package = ExportPackage.from_json(json_str)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON: {e}")
            return 1
        
        # Determine source DID
        from_did = args.source or package.exporter_did
        if not from_did:
            print("❌ Source DID required (use --from or ensure exporter_did in package)")
            return 1
        
        # Check trust
        registry = get_trust_registry()
        peer = registry.get_peer(from_did)
        
        if not peer and not args.trust:
            print(f"⚠️  Unknown peer: {from_did}")
            print(f"   Add with: valence peer add {from_did} --trust 0.5")
            print(f"   Or use --trust flag to import without adding to registry")
            return 1
        
        # Import
        result = import_beliefs(
            package=package,
            from_did=from_did,
            trust_override=args.trust,
        )
        
        # Show results
        print(f"📥 Import Results:")
        print(f"   Total in package: {result.total_in_package}")
        print(f"   ✅ Imported:      {result.imported}")
        print(f"   ⏭️  Duplicates:    {result.skipped_duplicate}")
        if result.skipped_low_trust:
            print(f"   🚫 Low trust:     {result.skipped_low_trust}")
        if result.skipped_error:
            print(f"   ❌ Errors:        {result.skipped_error}")
        print(f"   Trust applied:   {result.trust_level_applied:.0%}")
        
        if result.errors:
            print(f"\n⚠️  Errors:")
            for err in result.errors[:5]:
                print(f"   - {err}")
        
        return 0 if result.imported > 0 or result.skipped_duplicate > 0 else 1
        
    except FileNotFoundError:
        print(f"❌ File not found: {args.file}")
        return 1
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
