"""Administrative CLI commands.

Commands for administrative operations:
- discover: Discover network routers
- conflicts: Detect contradicting beliefs
- migrate-visibility: Migrate visibility to SharePolicy
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .beliefs import get_db_connection, format_confidence


# ============================================================================
# DISCOVER Command (Network Bootstrap)
# ============================================================================

def cmd_discover(args: argparse.Namespace) -> int:
    """Discover network routers via seed nodes."""
    
    async def run_discovery():
        from ...network.discovery import (
            DiscoveryClient,
            DiscoveryError,
            NoSeedsAvailableError,
        )
        
        # Create client with custom settings
        client = DiscoveryClient(
            verify_signatures=not args.no_verify,
        )
        
        # Add custom seeds
        if args.seeds:
            for seed in args.seeds:
                client.add_seed(seed)
        
        # Build preferences
        preferences = {}
        if args.region:
            preferences["region"] = args.region
        if args.features:
            preferences["features"] = args.features
        
        try:
            routers = await client.discover_routers(
                count=args.count,
                preferences=preferences if preferences else None,
                force_refresh=args.refresh,
            )
        except NoSeedsAvailableError as e:
            print(f"❌ No seeds available: {e}", file=sys.stderr)
            return 1
        except DiscoveryError as e:
            print(f"❌ Discovery failed: {e}", file=sys.stderr)
            return 1
        
        # JSON output
        if args.json:
            output = {
                "routers": [r.to_dict() for r in routers],
                "count": len(routers),
            }
            if args.stats:
                output["stats"] = client.get_stats()
            print(json.dumps(output, indent=2))
            return 0
        
        # Human-readable output
        if not routers:
            print("📭 No routers discovered")
            return 0
        
        print(f"📡 Discovered {len(routers)} router(s)\n")
        
        for i, router in enumerate(routers, 1):
            router_id = router.router_id
            if len(router_id) > 30:
                router_id = router_id[:27] + "..."
            
            endpoints = ", ".join(router.endpoints[:2])
            if len(router.endpoints) > 2:
                endpoints += f" (+{len(router.endpoints) - 2} more)"
            
            regions = ", ".join(router.regions) if router.regions else "unspecified"
            features = ", ".join(router.features) if router.features else "none"
            
            load = router.capacity.get("current_load_pct", "?")
            uptime = router.health.get("uptime_pct", "?")
            
            print(f"{i}. {router_id}")
            print(f"   Endpoints: {endpoints}")
            print(f"   Regions:   {regions}")
            print(f"   Features:  {features}")
            print(f"   Load: {load}% | Uptime: {uptime}%")
            print()
        
        # Show stats if requested
        if args.stats:
            stats = client.get_stats()
            print("─" * 40)
            print("📊 Discovery Statistics:")
            print(f"   Queries:            {stats['queries']}")
            print(f"   Cache hits:         {stats['cache_hits']}")
            print(f"   Seed failures:      {stats['seed_failures']}")
            print(f"   Signature failures: {stats['signature_failures']}")
        
        return 0
    
    return asyncio.run(run_discovery())


# ============================================================================
# CONFLICTS Command
# ============================================================================

def cmd_conflicts(args: argparse.Namespace) -> int:
    """Detect beliefs that may contradict each other.
    
    Uses semantic similarity > threshold combined with:
    - Opposite sentiment signals (negation words)
    - Different conclusions about same entities
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        threshold = args.threshold
        
        print(f"🔍 Scanning for conflicts (similarity > {threshold:.0%})...\n")
        
        # Find pairs of beliefs with high semantic similarity
        # that might be contradictions
        cur.execute("""
            WITH belief_pairs AS (
                SELECT 
                    b1.id as id_a,
                    b1.content as content_a,
                    b1.confidence as confidence_a,
                    b1.created_at as created_a,
                    b2.id as id_b,
                    b2.content as content_b,
                    b2.confidence as confidence_b,
                    b2.created_at as created_b,
                    1 - (b1.embedding <=> b2.embedding) as similarity
                FROM beliefs b1
                CROSS JOIN beliefs b2
                WHERE b1.id < b2.id
                  AND b1.embedding IS NOT NULL
                  AND b2.embedding IS NOT NULL
                  AND b1.status = 'active'
                  AND b2.status = 'active'
                  AND b1.superseded_by_id IS NULL
                  AND b2.superseded_by_id IS NULL
                  AND 1 - (b1.embedding <=> b2.embedding) > %s
                ORDER BY similarity DESC
                LIMIT 50
            )
            SELECT * FROM belief_pairs
            WHERE NOT EXISTS (
                SELECT 1 FROM tensions t 
                WHERE (t.belief_a_id = belief_pairs.id_a AND t.belief_b_id = belief_pairs.id_b)
                   OR (t.belief_a_id = belief_pairs.id_b AND t.belief_b_id = belief_pairs.id_a)
            )
        """, (threshold,))
        
        pairs = cur.fetchall()
        
        if not pairs:
            print("✅ No potential conflicts detected")
            return 0
        
        # Analyze each pair for contradiction signals
        conflicts = []
        negation_words = {'not', 'never', 'no', "n't", 'cannot', 'without', 'neither', 
                         'none', 'nobody', 'nothing', 'nowhere', 'false', 'incorrect',
                         'wrong', 'fail', 'reject', 'deny', 'refuse', 'avoid'}
        
        for pair in pairs:
            content_a = pair['content_a'].lower()
            content_b = pair['content_b'].lower()
            
            words_a = set(content_a.split())
            words_b = set(content_b.split())
            
            # Check for negation asymmetry
            neg_a = bool(words_a & negation_words)
            neg_b = bool(words_b & negation_words)
            
            # Higher conflict score if one has negation and other doesn't
            conflict_signal = 0.0
            reason = []
            
            if neg_a != neg_b:
                conflict_signal += 0.4
                reason.append("negation asymmetry")
            
            # Check for opposite conclusions (simple heuristic)
            # e.g., "X is good" vs "X is bad"
            opposites = [
                ('good', 'bad'), ('right', 'wrong'), ('true', 'false'),
                ('should', 'should not'), ('always', 'never'), ('prefer', 'avoid'),
                ('like', 'dislike'), ('works', 'fails'), ('correct', 'incorrect'),
            ]
            
            for pos, neg in opposites:
                if (pos in content_a and neg in content_b) or (neg in content_a and pos in content_b):
                    conflict_signal += 0.3
                    reason.append(f"opposite: {pos}/{neg}")
                    break
            
            # High similarity + some conflict signal = potential contradiction
            if conflict_signal > 0.2 or pair['similarity'] > 0.92:
                conflicts.append({
                    **pair,
                    'conflict_score': conflict_signal + (pair['similarity'] - threshold) * 0.5,
                    'reason': ', '.join(reason) if reason else 'high similarity'
                })
        
        if not conflicts:
            print("✅ Found similar beliefs but no likely contradictions")
            return 0
        
        # Sort by conflict score
        conflicts.sort(key=lambda x: x['conflict_score'], reverse=True)
        
        print(f"⚠️  Found {len(conflicts)} potential conflict(s):\n")
        
        for i, c in enumerate(conflicts[:10], 1):
            print(f"{'═' * 60}")
            print(f"Conflict #{i} (similarity: {c['similarity']:.1%}, signal: {c['conflict_score']:.2f})")
            print(f"Reason: {c['reason']}")
            print()
            print(f"  A [{str(c['id_a'])[:8]}] {c['content_a'][:70]}...")
            print(f"  B [{str(c['id_b'])[:8]}] {c['content_b'][:70]}...")
            print()
            
            if args.auto_record:
                # Record as tension
                cur.execute("""
                    INSERT INTO tensions (belief_a_id, belief_b_id, type, description, severity)
                    VALUES (%s, %s, 'contradiction', %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                """, (
                    c['id_a'], 
                    c['id_b'], 
                    f"Auto-detected: {c['reason']}",
                    'high' if c['conflict_score'] > 0.5 else 'medium'
                ))
                tension_row = cur.fetchone()
                if tension_row:
                    print(f"  📝 Recorded as tension: {str(tension_row['id'])[:8]}")
        
        if args.auto_record:
            conn.commit()
        
        print(f"{'═' * 60}")
        print(f"\n💡 Use 'valence tension resolve <id>' to resolve conflicts")
        
        return 0
        
    except Exception as e:
        print(f"❌ Conflict detection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cur.close()
        conn.close()


# ============================================================================
# MIGRATE-VISIBILITY Command
# ============================================================================

def cmd_migrate_visibility(args: argparse.Namespace) -> int:
    """Migrate existing beliefs from old visibility to SharePolicy."""
    from ...privacy.migration import migrate_all_beliefs_sync
    
    print("🔄 Migrating visibility to SharePolicy...")
    
    try:
        conn = get_db_connection()
        
        # Check if share_policy column exists
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'beliefs' AND column_name = 'share_policy'
            )
        """)
        has_column = cur.fetchone()['exists']
        
        if not has_column:
            print("⚠️  share_policy column not found. Adding it...")
            cur.execute("""
                ALTER TABLE beliefs 
                ADD COLUMN IF NOT EXISTS share_policy JSONB
            """)
            conn.commit()
            print("✅ share_policy column added")
        
        cur.close()
        
        # Run migration
        result = migrate_all_beliefs_sync(conn)
        
        print(f"\n📊 Migration Results:")
        print(f"   Total beliefs:     {result['total']}")
        print(f"   Needed migration:  {result['needed_migration']}")
        print(f"   Migrated:          {result['migrated']}")
        
        if result['needed_migration'] == 0:
            print("\n✅ All beliefs already have share_policy set")
        else:
            print(f"\n✅ Successfully migrated {result['needed_migration']} beliefs")
        
        conn.close()
        return 0
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
