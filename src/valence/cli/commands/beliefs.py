"""Belief management CLI commands.

Commands for core belief operations:
- init: Initialize database
- add: Add new beliefs
- query: Search beliefs with multi-signal ranking
- list: List recent beliefs
- stats: Show database statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Database Helpers
# ============================================================================

def get_db_connection():
    """Get database connection using config."""
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from ...core.config import get_config
    
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
    from ...core.config import get_config
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


# ============================================================================
# INIT Command
# ============================================================================

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize valence database."""
    import psycopg2
    from ...core.config import get_config
    
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
    
    conn = None
    cur = None
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
        if cur:
            cur.close()
        if conn:
            conn.close()


# ============================================================================
# QUERY Command
# ============================================================================

def cmd_query(args: argparse.Namespace) -> int:
    """Search beliefs with multi-signal ranking and derivation chains."""
    query = args.query
    
    # Use federated query if scope is federated
    if hasattr(args, 'scope') and args.scope == 'federated':
        from .federation import cmd_query_federated
        return cmd_query_federated(args)
    
    conn = None
    cur = None
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
        if cur:
            cur.close()
        if conn:
            conn.close()


# ============================================================================
# LIST Command
# ============================================================================

def cmd_list(args: argparse.Namespace) -> int:
    """List recent beliefs."""
    conn = None
    cur = None
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
        if cur:
            cur.close()
        if conn:
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
