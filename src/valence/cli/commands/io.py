"""Import/export commands."""

from __future__ import annotations

import argparse
import json


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
            with open(args.output, "w") as f:
                f.write(json_output)
            print(f"✅ Exported {len(package.beliefs)} beliefs to {args.output}")
        else:
            print(json_output)

        # Show summary unless quiet
        if args.output:
            print("\n📦 Export Summary:")
            print(f"   Beliefs: {len(package.beliefs)}")
            print(f"   From:    {package.exporter_did}")
            if package.recipient_did:
                print(f"   To:      {package.recipient_did}")
            if package.domain_summary:
                print(f"   Domains: {', '.join(f'{k}({v})' for k, v in package.domain_summary.items())}")

        return 0

    except Exception as e:
        print(f"❌ Export failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def cmd_import(args: argparse.Namespace) -> int:
    """Import beliefs from a peer."""
    import sys

    from ...federation.peer_sync import ExportPackage, get_trust_registry, import_beliefs

    try:
        # Read input file
        if args.file == "-":
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
            print("   Or use --trust flag to import without adding to registry")
            return 1

        # Import
        result = import_beliefs(
            package=package,
            from_did=from_did,
            trust_override=args.trust,
        )

        # Show results
        print("📥 Import Results:")
        print(f"   Total in package: {result.total_in_package}")
        print(f"   ✅ Imported:      {result.imported}")
        print(f"   ⏭️  Duplicates:    {result.skipped_duplicate}")
        if result.skipped_low_trust:
            print(f"   🚫 Low trust:     {result.skipped_low_trust}")
        if result.skipped_error:
            print(f"   ❌ Errors:        {result.skipped_error}")
        print(f"   Trust applied:   {result.trust_level_applied:.0%}")

        if result.errors:
            print("\n⚠️  Errors:")
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
