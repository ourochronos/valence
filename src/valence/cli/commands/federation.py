"""Federated query command."""

from __future__ import annotations

import argparse


def cmd_query_federated(args: argparse.Namespace) -> int:
    """Search beliefs with source attribution (federated scope)."""
    from ...federation.peer_sync import query_federated

    try:
        results = query_federated(
            query_text=args.query,
            scope=getattr(args, "scope", "local"),
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
                print("    📍 Source: LOCAL")
            else:
                trust_pct = f"{r.origin_trust:.0%}" if r.origin_trust else "?"
                print(f"    🔗 Source: {r.origin_did} (trust: {trust_pct})")

                # Show original vs weighted confidence
                original = r.confidence.get("_original_overall")
                if original:
                    print(f"       Original confidence: {original:.0%} → weighted: {r.effective_confidence:.0%}")

        print(f"{'─' * 60}")

        return 0

    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
