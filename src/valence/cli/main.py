#!/usr/bin/env python3
"""
Valence CLI - Personal knowledge substrate for AI agents.

Commands:
  valence init              Initialize database (creates schema)
  valence add <content>     Add a new belief
  valence query <text>      Search beliefs with derivation chains
  valence list              List recent beliefs
  valence conflicts         Detect contradicting beliefs
  valence stats             Show database statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Try to load .env from common locations
for env_path in [Path.cwd() / '.env', Path.home() / '.valence' / '.env']:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ.setdefault(key.strip(), value.strip())
        break


def get_db_connection():
    """Get database connection using config."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from ..core.config import get_config
    
    config = get_config()
    return psycopg2.connect(
        host=config.db_host,
        port=config.db_port,
        dbname=config.db_name,
        user=config.db_user,
        password=config.db_password,
        cursor_factory=RealDictCursor,
    )


def get_embedding(text: str) -> list[float] | None:
    """Generate embedding using OpenAI."""
    from ..core.config import get_config
    config = get_config()
    api_key = config.openai_api_key
    if not api_key:
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"⚠️  Embedding failed: {e}", file=sys.stderr)
        return None


def format_confidence(conf: dict) -> str:
    """Format confidence for display."""
    if not conf:
        return "?"
    overall = conf.get('overall', 0)
    if isinstance(overall, (int, float)):
        return f"{overall:.0%}"
    return str(overall)[:5]


# ============================================================================
# Multi-Signal Ranking (Valence Query Protocol)
# ============================================================================

def compute_confidence_score(belief: dict) -> float:
    """
    Compute aggregated confidence score from 6D confidence vector.
    
    Uses geometric mean to penalize beliefs with any weak dimension.
    Falls back to JSONB 'overall' field for backward compatibility.
    """
    # Try 6D confidence columns first
    src = belief.get('confidence_source', 0.5)
    meth = belief.get('confidence_method', 0.5)
    cons = belief.get('confidence_consistency', 1.0)
    fresh = belief.get('confidence_freshness', 1.0)
    corr = belief.get('confidence_corroboration', 0.1)
    app = belief.get('confidence_applicability', 0.8)
    
    # Check if 6D columns are populated (not default placeholder)
    has_6d = any([
        belief.get('confidence_source') is not None,
        belief.get('confidence_method') is not None,
    ])
    
    if has_6d:
        # Geometric mean with spec weights
        # w_sr=0.25, w_mq=0.20, w_ic=0.15, w_tf=0.15, w_cor=0.15, w_da=0.10
        import math
        try:
            score = (
                (src ** 0.25) *
                (meth ** 0.20) *
                (cons ** 0.15) *
                (fresh ** 0.15) *
                (corr ** 0.15) *
                (app ** 0.10)
            )
            return min(1.0, max(0.0, score))
        except (ValueError, ZeroDivisionError):
            pass
    
    # Fallback to JSONB overall
    conf = belief.get('confidence', {})
    if isinstance(conf, dict):
        overall = conf.get('overall', 0.5)
        if isinstance(overall, (int, float)):
            return min(1.0, max(0.0, float(overall)))
    
    return 0.5  # Default


def compute_recency_score(created_at: datetime, decay_rate: float = 0.01) -> float:
    """
    Compute recency score with exponential decay.
    
    Default decay_rate=0.01 gives a half-life of ~69 days.
    """
    import math
    
    if not created_at:
        return 0.5
    
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    age_days = (now - created_at).total_seconds() / 86400
    
    # Exponential decay: e^(-λ × age)
    recency = math.exp(-decay_rate * age_days)
    return min(1.0, max(0.0, recency))


def multi_signal_rank(
    results: list[dict],
    semantic_weight: float = 0.50,
    confidence_weight: float = 0.35,
    recency_weight: float = 0.15,
    min_confidence: float | None = None,
    explain: bool = False,
) -> list[dict]:
    """
    Apply multi-signal ranking to query results.
    
    Formula: final_score = w_semantic × semantic + w_confidence × confidence + w_recency × recency
    
    Args:
        results: List of belief dicts with 'similarity' (semantic score)
        semantic_weight: Weight for semantic similarity (default 0.50)
        confidence_weight: Weight for confidence score (default 0.35)
        recency_weight: Weight for recency score (default 0.15)
        min_confidence: Filter out beliefs below this confidence (optional)
        explain: Include score breakdown in results
    
    Returns:
        Sorted results with 'final_score' and optional 'score_breakdown'
    """
    # Normalize weights to sum to 1.0
    total_weight = semantic_weight + confidence_weight + recency_weight
    if total_weight > 0:
        semantic_weight /= total_weight
        confidence_weight /= total_weight
        recency_weight /= total_weight
    
    ranked = []
    for r in results:
        # Semantic score (already computed from embedding similarity)
        semantic = r.get('similarity', 0.0)
        if isinstance(semantic, (int, float)):
            semantic = min(1.0, max(0.0, float(semantic)))
        else:
            semantic = 0.0
        
        # Confidence score
        confidence = compute_confidence_score(r)
        
        # Filter by minimum confidence if specified
        if min_confidence is not None and confidence < min_confidence:
            continue
        
        # Recency score
        created_at = r.get('created_at')
        recency = compute_recency_score(created_at) if created_at else 0.5
        
        # Final score
        final_score = (
            semantic_weight * semantic +
            confidence_weight * confidence +
            recency_weight * recency
        )
        
        r['final_score'] = final_score
        
        if explain:
            r['score_breakdown'] = {
                'semantic': {'value': semantic, 'weight': semantic_weight, 'contribution': semantic_weight * semantic},
                'confidence': {'value': confidence, 'weight': confidence_weight, 'contribution': confidence_weight * confidence},
                'recency': {'value': recency, 'weight': recency_weight, 'contribution': recency_weight * recency},
                'final': final_score,
            }
        
        ranked.append(r)
    
    # Sort by final score descending
    ranked.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    return ranked


def format_age(dt: datetime) -> str:
    """Format datetime as human-readable age."""
    if not dt:
        return "?"
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    delta = now - dt
    
    if delta.days > 365:
        return f"{delta.days // 365}y"
    elif delta.days > 30:
        return f"{delta.days // 30}mo"
    elif delta.days > 0:
        return f"{delta.days}d"
    elif delta.seconds > 3600:
        return f"{delta.seconds // 3600}h"
    elif delta.seconds > 60:
        return f"{delta.seconds // 60}m"
    else:
        return "now"


# ============================================================================
# INIT Command
# ============================================================================

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize valence database."""
    import psycopg2
    from ..core.config import get_config
    
    print("🔧 Initializing Valence database...")
    
    # Check if we need to create the database
    config = get_config()
    db_name = config.db_name
    db_host = config.db_host
    db_port = str(config.db_port)
    db_user = config.db_user
    db_pass = config.db_password
    
    # First, try to connect to the database
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if beliefs table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'beliefs'
            )
        """)
        exists = cur.fetchone()['exists']
        
        if exists and not args.force:
            print(f"✅ Database already initialized at {db_host}:{db_port}/{db_name}")
            cur.execute("SELECT COUNT(*) as count FROM beliefs")
            count = cur.fetchone()['count']
            print(f"   📊 {count} beliefs stored")
            conn.close()
            return 0
        
        conn.close()
    except psycopg2.OperationalError as e:
        print(f"⚠️  Cannot connect to database: {e}")
        print(f"\nTo create a new database, run:")
        print(f"  createdb -h {db_host} -p {db_port} -U {db_user} {db_name}")
        print(f"  # Or with Docker:")
        print(f"  docker run -d --name valence-db -e POSTGRES_USER={db_user} \\")
        print(f"    -e POSTGRES_PASSWORD={db_pass or 'valence'} -e POSTGRES_DB={db_name} \\")
        print(f"    -p {db_port}:5432 pgvector/pgvector:pg16")
        return 1
    
    print("🔨 Creating schema...")
    
    # Read and execute schema
    schema_dir = Path(__file__).parent.parent / 'substrate'
    migrations_dir = Path(__file__).parent.parent.parent.parent / 'migrations'
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Enable required extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        
        # Create core tables if they don't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                type TEXT NOT NULL,
                title TEXT,
                url TEXT,
                content_hash TEXT,
                session_id UUID,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                content_hash CHAR(64),
                embedding vector(1536),
                confidence JSONB DEFAULT '{"overall": 0.7}',
                domain_path TEXT[] DEFAULT '{}',
                source_id UUID REFERENCES sources(id),
                extraction_method TEXT,
                valid_from TIMESTAMPTZ,
                valid_until TIMESTAMPTZ,
                supersedes_id UUID REFERENCES beliefs(id),
                superseded_by_id UUID REFERENCES beliefs(id),
                status TEXT DEFAULT 'active',
                holder_id UUID DEFAULT '00000000-0000-0000-0000-000000000001',
                visibility TEXT DEFAULT 'private',
                version INTEGER DEFAULT 1,
                is_local BOOLEAN DEFAULT TRUE,
                corroboration_count INTEGER DEFAULT 0,
                corroborating_sources JSONB DEFAULT '[]',
                confidence_source REAL DEFAULT 0.5,
                confidence_method REAL DEFAULT 0.5,
                confidence_consistency REAL DEFAULT 1.0,
                confidence_freshness REAL DEFAULT 1.0,
                confidence_corroboration REAL DEFAULT 0.1,
                confidence_applicability REAL DEFAULT 0.8,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                modified_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                aliases TEXT[] DEFAULT '{}',
                canonical_id UUID REFERENCES entities(id),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                modified_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        # Partial unique index for non-aliased entities
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_type_unique 
            ON entities(name, type) WHERE canonical_id IS NULL
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS belief_entities (
                belief_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
                entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
                role TEXT DEFAULT 'subject',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (belief_id, entity_id)
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tensions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                belief_a_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
                belief_b_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
                type TEXT DEFAULT 'contradiction',
                description TEXT,
                severity TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'detected',
                resolution TEXT,
                resolved_at TIMESTAMPTZ,
                detected_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS belief_derivations (
                belief_id UUID PRIMARY KEY REFERENCES beliefs(id) ON DELETE CASCADE,
                derivation_type TEXT NOT NULL DEFAULT 'assumption',
                method_description TEXT,
                confidence_rationale TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS derivation_sources (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
                source_belief_id UUID REFERENCES beliefs(id),
                external_ref TEXT,
                contribution_type TEXT DEFAULT 'primary',
                weight REAL DEFAULT 1.0
            )
        """)
        
        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_embedding ON beliefs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_domain ON beliefs USING GIN(domain_path)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_created ON beliefs(created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_derivation_sources_belief ON derivation_sources(belief_id)")
        
        # Create text search
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE beliefs ADD COLUMN content_tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_tsv ON beliefs USING GIN(content_tsv)")
        
        conn.commit()
        print("✅ Schema created successfully!")
        
        # Show connection info
        print(f"\n📍 Connected to: {db_host}:{db_port}/{db_name}")
        print("\n💡 Quick start:")
        print("   valence add 'Your first belief here' -d general")
        print("   valence query 'search terms'")
        print("   valence list")
        
        return 0
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Schema creation failed: {e}")
        return 1
    finally:
        cur.close()
        conn.close()


# ============================================================================
# ADD Command
# ============================================================================

def cmd_add(args: argparse.Namespace) -> int:
    """Add a new belief."""
    content = args.content
    
    if not content:
        print("❌ Content is required")
        return 1
    
    # Parse confidence
    confidence = {"overall": 0.7}
    if args.confidence:
        try:
            confidence = json.loads(args.confidence)
        except json.JSONDecodeError:
            # Try parsing as simple float
            try:
                confidence = {"overall": float(args.confidence)}
            except ValueError:
                print(f"❌ Invalid confidence: {args.confidence}")
                return 1
    
    # Parse domains
    domains = args.domain or []
    
    # Generate embedding
    embedding = get_embedding(content)
    embedding_str = None
    if embedding:
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"
    
    # Parse derivation info
    derivation_type = args.derivation_type or 'observation'
    derived_from = args.derived_from  # UUID of source belief
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Insert belief
        cur.execute("""
            INSERT INTO beliefs (content, confidence, domain_path, embedding, extraction_method)
            VALUES (%s, %s, %s, %s::vector, %s)
            RETURNING id, created_at
        """, (content, json.dumps(confidence), domains, embedding_str, derivation_type))
        
        row = cur.fetchone()
        belief_id = row['id']
        
        # Add derivation record
        cur.execute("""
            INSERT INTO belief_derivations (belief_id, derivation_type, method_description)
            VALUES (%s, %s, %s)
        """, (belief_id, derivation_type, args.method or None))
        
        # Link derived_from if provided
        if derived_from:
            cur.execute("""
                INSERT INTO derivation_sources (belief_id, source_belief_id, contribution_type)
                VALUES (%s, %s, 'primary')
            """, (belief_id, derived_from))
        
        conn.commit()
        
        print(f"✅ Belief added: {str(belief_id)[:8]}...")
        print(f"   📝 {content[:60]}{'...' if len(content) > 60 else ''}")
        print(f"   🎯 Confidence: {format_confidence(confidence)}")
        if domains:
            print(f"   📁 Domains: {', '.join(domains)}")
        if embedding:
            print(f"   🧠 Embedding: generated")
        else:
            print(f"   ⚠️  No embedding (set OPENAI_API_KEY)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to add belief: {e}")
        return 1
    finally:
        cur.close()
        conn.close()


# ============================================================================
# QUERY Command
# ============================================================================

def cmd_query(args: argparse.Namespace) -> int:
    """Search beliefs with multi-signal ranking and derivation chains."""
    query = args.query
    
    # Use federated query if scope is federated
    if hasattr(args, 'scope') and args.scope == 'federated':
        return cmd_query_federated(args)
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Try semantic search first if we can get an embedding
        embedding = get_embedding(query)
        
        results = []
        
        # Get ranking weights from args (with defaults)
        recency_weight = getattr(args, 'recency_weight', 0.15) or 0.15
        min_confidence = getattr(args, 'min_confidence', None)
        explain = getattr(args, 'explain', False)
        
        # Semantic weight adjusts based on recency weight
        # Default: semantic=0.50, confidence=0.35, recency=0.15
        semantic_weight = 0.50
        confidence_weight = 0.35
        
        # If recency_weight is overridden, scale others proportionally
        if recency_weight != 0.15:
            remaining = 1.0 - recency_weight
            ratio = remaining / 0.85  # 0.85 = default semantic + confidence
            semantic_weight = 0.50 * ratio
            confidence_weight = 0.35 * ratio
        
        if embedding:
            embedding_str = f"[{','.join(str(x) for x in embedding)}]"
            
            # Semantic search with derivation info AND 6D confidence columns
            cur.execute("""
                WITH ranked AS (
                    SELECT 
                        b.id,
                        b.content,
                        b.confidence,
                        b.domain_path,
                        b.created_at,
                        b.extraction_method,
                        b.supersedes_id,
                        b.confidence_source,
                        b.confidence_method,
                        b.confidence_consistency,
                        b.confidence_freshness,
                        b.confidence_corroboration,
                        b.confidence_applicability,
                        1 - (b.embedding <=> %s::vector) as similarity,
                        d.derivation_type,
                        d.method_description,
                        d.confidence_rationale
                    FROM beliefs b
                    LEFT JOIN belief_derivations d ON d.belief_id = b.id
                    WHERE b.embedding IS NOT NULL
                      AND b.status = 'active'
                      AND b.superseded_by_id IS NULL
                    ORDER BY b.embedding <=> %s::vector
                    LIMIT %s
                )
                SELECT r.*,
                    (SELECT json_agg(json_build_object(
                        'source_belief_id', ds.source_belief_id,
                        'external_ref', ds.external_ref,
                        'contribution_type', ds.contribution_type
                    ))
                    FROM derivation_sources ds 
                    WHERE ds.belief_id = r.id) as derivation_sources
                FROM ranked r
                WHERE r.similarity >= %s
            """, (embedding_str, embedding_str, args.limit * 2, args.threshold))  # Get extra for filtering
            
            results = cur.fetchall()
        
        # Fallback to text search if no embedding or no results
        if not results:
            cur.execute("""
                SELECT 
                    b.id,
                    b.content,
                    b.confidence,
                    b.domain_path,
                    b.created_at,
                    b.extraction_method,
                    b.supersedes_id,
                    b.confidence_source,
                    b.confidence_method,
                    b.confidence_consistency,
                    b.confidence_freshness,
                    b.confidence_corroboration,
                    b.confidence_applicability,
                    ts_rank(b.content_tsv, websearch_to_tsquery('english', %s)) as similarity,
                    d.derivation_type,
                    d.method_description,
                    d.confidence_rationale,
                    (SELECT json_agg(json_build_object(
                        'source_belief_id', ds.source_belief_id,
                        'external_ref', ds.external_ref,
                        'contribution_type', ds.contribution_type
                    ))
                    FROM derivation_sources ds 
                    WHERE ds.belief_id = b.id) as derivation_sources
                FROM beliefs b
                LEFT JOIN belief_derivations d ON d.belief_id = b.id
                WHERE b.content_tsv @@ websearch_to_tsquery('english', %s)
                  AND b.status = 'active'
                  AND b.superseded_by_id IS NULL
                ORDER BY ts_rank(b.content_tsv, websearch_to_tsquery('english', %s)) DESC
                LIMIT %s
            """, (query, query, query, args.limit * 2))
            
            results = cur.fetchall()
        
        # Convert to list of dicts for processing
        results = [dict(r) for r in results]
        
        # Domain filter
        if args.domain:
            results = [r for r in results if args.domain in (r.get('domain_path') or [])]
        
        # Apply multi-signal ranking
        results = multi_signal_rank(
            results,
            semantic_weight=semantic_weight,
            confidence_weight=confidence_weight,
            recency_weight=recency_weight,
            min_confidence=min_confidence,
            explain=explain,
        )
        
        # Limit results after ranking
        results = results[:args.limit]
        
        if not results:
            print(f"🔍 No beliefs found for: {query}")
            return 0
        
        print(f"🔍 Found {len(results)} belief(s) for: {query}")
        if explain:
            print(f"   Weights: semantic={semantic_weight:.2f}, confidence={confidence_weight:.2f}, recency={recency_weight:.2f}")
        print()
        
        for i, r in enumerate(results, 1):
            final_score = r.get('final_score', 0)
            sim = r.get('similarity', 0)
            
            # Header
            print(f"{'─' * 60}")
            print(f"[{i}] {r['content'][:70]}{'...' if len(r['content']) > 70 else ''}")
            
            # Show final score and components
            conf_score = compute_confidence_score(r)
            created_at = r.get('created_at')
            age_str = format_age(created_at) if created_at else "unknown"
            print(f"    ID: {str(r['id'])[:8]}  Score: {final_score:.0%}  Confidence: {conf_score:.0%}  Semantic: {sim:.0%}  Age: {age_str}")
            
            if r.get('domain_path'):
                print(f"    Domains: {', '.join(r['domain_path'])}")
            
            # Show score breakdown if explain mode
            if explain and r.get('score_breakdown'):
                bd = r['score_breakdown']
                print(f"    ┌─ Score Breakdown:")
                print(f"    │  Semantic:   {bd['semantic']['value']:.2f} × {bd['semantic']['weight']:.2f} = {bd['semantic']['contribution']:.3f}")
                print(f"    │  Confidence: {bd['confidence']['value']:.2f} × {bd['confidence']['weight']:.2f} = {bd['confidence']['contribution']:.3f}")
                print(f"    │  Recency:    {bd['recency']['value']:.2f} × {bd['recency']['weight']:.2f} = {bd['recency']['contribution']:.3f}")
                print(f"    │  Final:      {bd['final']:.3f}")
                print(f"    └─")
            
            # === DERIVATION CHAIN ===
            derivation_type = r.get('derivation_type') or r.get('extraction_method') or 'unknown'
            print(f"    ┌─ Derivation: {derivation_type}")
            
            if r.get('method_description'):
                print(f"    │  Method: {r['method_description']}")
            
            if r.get('confidence_rationale'):
                print(f"    │  Rationale: {r['confidence_rationale']}")
            
            # Show source beliefs
            sources = r.get('derivation_sources') or []
            if sources:
                for src in sources:
                    if src.get('source_belief_id'):
                        # Fetch source belief content
                        cur.execute("SELECT content FROM beliefs WHERE id = %s", (src['source_belief_id'],))
                        src_row = cur.fetchone()
                        src_content = src_row['content'][:50] if src_row else '?'
                        print(f"    │  ← Derived from ({src.get('contribution_type', 'primary')}): {src_content}...")
                    elif src.get('external_ref'):
                        print(f"    │  ← External: {src['external_ref']}")
            
            # Show supersession chain if exists
            if r.get('supersedes_id'):
                print(f"    │  ⟳ Supersedes: {str(r['supersedes_id'])[:8]}...")
                # Optionally walk the chain
                if args.chain:
                    chain = []
                    current = r['supersedes_id']
                    depth = 0
                    while current and depth < 5:
                        cur.execute("SELECT id, content, supersedes_id FROM beliefs WHERE id = %s", (current,))
                        chain_row = cur.fetchone()
                        if chain_row:
                            chain.append(f"{chain_row['content'][:40]}...")
                            current = chain_row['supersedes_id']
                            depth += 1
                        else:
                            break
                    for j, c in enumerate(chain):
                        print(f"    │    {'└' if j == len(chain)-1 else '├'}─ {c}")
            
            print(f"    └─")
        
        print(f"{'─' * 60}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        cur.close()
        conn.close()


# ============================================================================
# FEDERATED QUERY
# ============================================================================

def cmd_query_federated(args: argparse.Namespace) -> int:
    """Search beliefs with source attribution (federated scope)."""
    from ..federation.peer_sync import query_federated
    
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
# LIST Command
# ============================================================================

def cmd_list(args: argparse.Namespace) -> int:
    """List recent beliefs."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        sql = """
            SELECT 
                b.id,
                b.content,
                b.confidence,
                b.domain_path,
                b.created_at,
                d.derivation_type
            FROM beliefs b
            LEFT JOIN belief_derivations d ON d.belief_id = b.id
            WHERE b.status = 'active'
              AND b.superseded_by_id IS NULL
        """
        params = []
        
        if args.domain:
            sql += " AND %s = ANY(b.domain_path)"
            params.append(args.domain)
        
        sql += " ORDER BY b.created_at DESC LIMIT %s"
        params.append(args.limit)
        
        cur.execute(sql, params)
        results = cur.fetchall()
        
        if not results:
            print("📭 No beliefs found")
            return 0
        
        print(f"📚 {len(results)} belief(s)" + (f" in domain '{args.domain}'" if args.domain else "") + "\n")
        
        for r in results:
            conf = format_confidence(r.get('confidence', {}))
            age = format_age(r.get('created_at'))
            deriv_raw = r.get('derivation_type') or '?'
            deriv = deriv_raw[:6] if deriv_raw else '?'
            content = r['content'][:55] + '...' if len(r['content']) > 55 else r['content']
            
            print(f"  {str(r['id'])[:8]}  [{conf:>4}] [{age:>3}] [{deriv:>6}]  {content}")
        
        return 0
        
    except Exception as e:
        print(f"❌ List failed: {e}")
        return 1
    finally:
        cur.close()
        conn.close()


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
# STATS Command
# ============================================================================

def cmd_stats(args: argparse.Namespace) -> int:
    """Show database statistics."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) as total FROM beliefs")
        total = cur.fetchone()['total']
        
        cur.execute("SELECT COUNT(*) as active FROM beliefs WHERE status = 'active' AND superseded_by_id IS NULL")
        active = cur.fetchone()['active']
        
        cur.execute("SELECT COUNT(*) as with_emb FROM beliefs WHERE embedding IS NOT NULL")
        with_embedding = cur.fetchone()['with_emb']
        
        cur.execute("SELECT COUNT(*) as tensions FROM tensions WHERE status = 'detected'")
        tensions = cur.fetchone()['tensions']
        
        try:
            cur.execute("SELECT COUNT(DISTINCT d) as count FROM beliefs, LATERAL unnest(domain_path) as d")
            domains = cur.fetchone()['count']
        except (Exception,) as e:
            logger.debug(f"Could not count domains (column may not exist): {e}")
            domains = 0
        
        cur.execute("SELECT COUNT(*) as derivations FROM belief_derivations")
        derivations = cur.fetchone()['derivations']
        
        # Count federated beliefs
        try:
            cur.execute("SELECT COUNT(*) as federated FROM beliefs WHERE is_local = FALSE")
            federated = cur.fetchone()['federated']
        except (Exception,) as e:
            logger.debug(f"Could not count federated beliefs (column may not exist): {e}")
            federated = 0
        
        print("📊 Valence Statistics")
        print("─" * 30)
        print(f"  Total beliefs:      {total}")
        print(f"  Active beliefs:     {active}")
        print(f"  Local beliefs:      {active - federated}")
        print(f"  Federated beliefs:  {federated}")
        print(f"  With embeddings:    {with_embedding}")
        print(f"  Derivation records: {derivations}")
        print(f"  Unique domains:     {domains}")
        print(f"  Unresolved tensions:{tensions}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Stats failed: {e}")
        return 1
    finally:
        cur.close()
        conn.close()


# ============================================================================
# PEER Commands
# ============================================================================

def cmd_peer_add(args: argparse.Namespace) -> int:
    """Add a trusted peer to the local registry."""
    from ..federation.peer_sync import get_trust_registry
    
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
    from ..federation.peer_sync import get_trust_registry
    
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
    from ..federation.peer_sync import get_trust_registry
    
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
# EXPORT Command
# ============================================================================

def cmd_export(args: argparse.Namespace) -> int:
    """Export beliefs for sharing with a peer."""
    from ..federation.peer_sync import export_beliefs
    
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
    from ..federation.peer_sync import ExportPackage, import_beliefs, get_trust_registry
    
    try:
        # Read input file
        if args.file == '-':
            import sys
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


# ============================================================================
# Main Entry Point
# ============================================================================

def app() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog='valence',
        description='Personal knowledge substrate for AI agents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  valence init                            Initialize database
  valence add "Fact here" -d tech         Add belief with domain
  valence query "search terms"            Search beliefs
  valence query "terms" --scope federated Include peer beliefs
  valence list -n 20                      List recent beliefs
  valence conflicts                       Detect contradictions
  valence stats                           Show statistics
  
Network:
  valence discover                        Discover network routers
  valence discover --seed <url>           Use custom seed

Federation (Week 2):
  valence peer add <did> --trust 0.8      Add trusted peer
  valence peer list                       Show trusted peers
  valence export --to <did> -o file.json  Export beliefs for peer
  valence import file.json --from <did>   Import from peer
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # init
    init_parser = subparsers.add_parser('init', help='Initialize database schema')
    init_parser.add_argument('--force', '-f', action='store_true', help='Recreate schema even if exists')
    
    # add
    add_parser = subparsers.add_parser('add', help='Add a new belief')
    add_parser.add_argument('content', help='Belief content')
    add_parser.add_argument('--confidence', '-c', help='Confidence (JSON or float 0-1)')
    add_parser.add_argument('--domain', '-d', action='append', help='Domain tag (repeatable)')
    add_parser.add_argument('--derivation-type', '-t', 
                          choices=['observation', 'inference', 'aggregation', 'hearsay', 'assumption', 'correction', 'synthesis'],
                          default='observation', help='How this belief was derived')
    add_parser.add_argument('--derived-from', help='UUID of source belief this was derived from')
    add_parser.add_argument('--method', '-m', help='Method description for derivation')
    
    # query
    query_parser = subparsers.add_parser('query', help='Search beliefs with multi-signal ranking')
    query_parser.add_argument('query', help='Search query')
    query_parser.add_argument('--limit', '-n', type=int, default=10, help='Max results')
    query_parser.add_argument('--threshold', '-t', type=float, default=0.3, help='Min semantic similarity')
    query_parser.add_argument('--domain', '-d', help='Filter by domain')
    query_parser.add_argument('--chain', action='store_true', help='Show full supersession chains')
    query_parser.add_argument('--scope', '-s', choices=['local', 'federated'], default='local',
                            help='Search scope: local (default) or federated (include peer beliefs)')
    # Multi-signal ranking options (Valence Query Protocol)
    query_parser.add_argument('--recency-weight', '-r', type=float, default=0.15,
                            help='Recency weight 0.0-1.0 (default 0.15). Higher = prefer newer beliefs')
    query_parser.add_argument('--min-confidence', '-c', type=float, default=None,
                            help='Filter beliefs below this confidence threshold (0.0-1.0)')
    query_parser.add_argument('--explain', '-e', action='store_true',
                            help='Show detailed score breakdown per result')
    
    # list
    list_parser = subparsers.add_parser('list', help='List recent beliefs')
    list_parser.add_argument('--limit', '-n', type=int, default=10, help='Max results')
    list_parser.add_argument('--domain', '-d', help='Filter by domain')
    
    # conflicts
    conflicts_parser = subparsers.add_parser('conflicts', help='Detect contradicting beliefs')
    conflicts_parser.add_argument('--threshold', '-t', type=float, default=0.85, 
                                 help='Similarity threshold for conflict detection')
    conflicts_parser.add_argument('--auto-record', '-r', action='store_true',
                                 help='Automatically record detected conflicts as tensions')
    
    # stats
    subparsers.add_parser('stats', help='Show database statistics')
    
    # ========================================================================
    # DISCOVER command (Network bootstrap)
    # ========================================================================
    
    discover_parser = subparsers.add_parser('discover', help='Discover network routers via seeds')
    discover_parser.add_argument('--seed', '-s', action='append', dest='seeds',
                                help='Custom seed URL (repeatable)')
    discover_parser.add_argument('--count', '-n', type=int, default=5,
                                help='Number of routers to request (default: 5)')
    discover_parser.add_argument('--region', '-r', help='Preferred region')
    discover_parser.add_argument('--feature', '-f', action='append', dest='features',
                                help='Required feature (repeatable)')
    discover_parser.add_argument('--refresh', action='store_true',
                                help='Force refresh (bypass cache)')
    discover_parser.add_argument('--no-verify', action='store_true',
                                help='Skip router signature verification')
    discover_parser.add_argument('--json', '-j', action='store_true',
                                help='Output as JSON')
    discover_parser.add_argument('--stats', action='store_true',
                                help='Show discovery statistics')
    
    # ========================================================================
    # PEER commands (Week 2 Federation)
    # ========================================================================
    
    peer_parser = subparsers.add_parser('peer', help='Manage trusted peers')
    peer_subparsers = peer_parser.add_subparsers(dest='peer_command', required=True)
    
    # peer add
    peer_add_parser = peer_subparsers.add_parser('add', help='Add or update a trusted peer')
    peer_add_parser.add_argument('did', help='Peer DID (e.g., did:vkb:web:alice.example.com)')
    peer_add_parser.add_argument('--trust', type=float, required=True, 
                                help='Trust level 0.0-1.0 (e.g., 0.8 for 80%% trust)')
    peer_add_parser.add_argument('--name', help='Human-readable name for this peer')
    peer_add_parser.add_argument('--notes', help='Notes about this peer')
    
    # peer list
    peer_subparsers.add_parser('list', help='List trusted peers')
    
    # peer remove
    peer_remove_parser = peer_subparsers.add_parser('remove', help='Remove a peer')
    peer_remove_parser.add_argument('did', help='Peer DID to remove')
    
    # ========================================================================
    # EXPORT command
    # ========================================================================
    
    export_parser = subparsers.add_parser('export', help='Export beliefs for sharing')
    export_parser.add_argument('--to', dest='to', help='Recipient DID (for filtering)')
    export_parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    export_parser.add_argument('--domain', '-d', action='append', help='Filter by domain')
    export_parser.add_argument('--min-confidence', type=float, default=0.0,
                              help='Minimum confidence threshold')
    export_parser.add_argument('--limit', '-n', type=int, default=1000, help='Max beliefs')
    export_parser.add_argument('--include-federated', action='store_true',
                              help='Include beliefs received from other peers')
    
    # ========================================================================
    # IMPORT command
    # ========================================================================
    
    import_parser = subparsers.add_parser('import', help='Import beliefs from a peer')
    import_parser.add_argument('file', help='Import file (JSON) or - for stdin')
    import_parser.add_argument('--from', dest='source', help='Source peer DID (overrides package)')
    import_parser.add_argument('--trust', type=float, 
                              help='Override trust level (otherwise uses registry)')
    
    # ========================================================================
    # TRUST commands
    # ========================================================================
    
    trust_parser = subparsers.add_parser('trust', help='Trust network management')
    trust_subparsers = trust_parser.add_subparsers(dest='trust_command', required=True)
    
    # trust check
    trust_check_parser = trust_subparsers.add_parser('check', help='Check for trust concentration issues')
    trust_check_parser.add_argument('--json', '-j', action='store_true', 
                                   help='Output as JSON')
    trust_check_parser.add_argument('--single-threshold', type=float, default=None,
                                   help='Custom threshold for single node dominance (default: 30%%)')
    trust_check_parser.add_argument('--top3-threshold', type=float, default=None,
                                   help='Custom threshold for top 3 nodes dominance (default: 50%%)')
    trust_check_parser.add_argument('--min-sources', type=int, default=None,
                                   help='Minimum trusted sources (default: 3)')
    
    # ========================================================================
    # MIGRATE-VISIBILITY command
    # ========================================================================
    
    subparsers.add_parser('migrate-visibility', 
                         help='Migrate existing beliefs from visibility to SharePolicy')
    
    return parser


# ============================================================================
# DISCOVER Command (Network Bootstrap)
# ============================================================================

def cmd_discover(args: argparse.Namespace) -> int:
    """Discover network routers via seed nodes."""
    import asyncio
    
    async def run_discovery():
        from ..network.discovery import (
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
            import json
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
# TRUST Commands
# ============================================================================

def cmd_trust_check(args: argparse.Namespace) -> int:
    """Check for trust concentration issues in the federation network."""
    from ..federation.trust import check_trust_concentration
    from ..federation.trust_policy import CONCENTRATION_THRESHOLDS
    
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


# ============================================================================
# MIGRATE-VISIBILITY Command
# ============================================================================

def cmd_migrate_visibility(args: argparse.Namespace) -> int:
    """Migrate existing beliefs from old visibility to SharePolicy."""
    from ..privacy.migration import migrate_all_beliefs_sync
    
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


def main() -> int:
    """Main entry point."""
    parser = app()
    args = parser.parse_args()
    
    commands = {
        'init': cmd_init,
        'add': cmd_add,
        'query': cmd_query,
        'list': cmd_list,
        'conflicts': cmd_conflicts,
        'stats': cmd_stats,
        'discover': cmd_discover,
        'peer': cmd_peer,
        'export': cmd_export,
        'import': cmd_import,
        'trust': cmd_trust,
        'migrate-visibility': cmd_migrate_visibility,
    }
    
    handler = commands.get(args.command)
    if handler:
        return handler(args)
    
    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
