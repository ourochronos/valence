"""Belief management commands: init, add, query, list."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..utils import (
    compute_confidence_score,
    format_age,
    format_confidence,
    get_db_connection,
    get_embedding,
    multi_signal_rank,
)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register belief commands (init, add, query, list) on the CLI parser."""
    # init
    init_parser = subparsers.add_parser("init", help="Initialize database schema")
    init_parser.add_argument("--force", "-f", action="store_true", help="Recreate schema even if exists")
    init_parser.set_defaults(func=cmd_init)

    # add
    add_parser = subparsers.add_parser("add", help="Add a new belief")
    add_parser.add_argument("content", help="Belief content")
    add_parser.add_argument("--confidence", "-c", help="Confidence (JSON or float 0-1)")
    add_parser.add_argument("--domain", "-d", action="append", help="Domain tag (repeatable)")
    add_parser.add_argument(
        "--derivation-type",
        "-t",
        choices=[
            "observation",
            "inference",
            "aggregation",
            "hearsay",
            "assumption",
            "correction",
            "synthesis",
        ],
        default="observation",
        help="How this belief was derived",
    )
    add_parser.add_argument("--derived-from", help="UUID of source belief this was derived from")
    add_parser.add_argument("--method", "-m", help="Method description for derivation")
    add_parser.set_defaults(func=cmd_add)

    # query
    query_parser = subparsers.add_parser("query", help="Search beliefs with multi-signal ranking")
    query_parser.add_argument("query", help="Search query")
    query_parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    query_parser.add_argument("--threshold", "-t", type=float, default=0.3, help="Min semantic similarity")
    query_parser.add_argument("--domain", "-d", help="Filter by domain")
    query_parser.add_argument("--chain", action="store_true", help="Show full supersession chains")
    query_parser.add_argument(
        "--scope",
        "-s",
        choices=["local", "federated"],
        default="local",
        help="Search scope: local (default) or federated (include peer beliefs)",
    )
    query_parser.add_argument(
        "--recency-weight",
        "-r",
        type=float,
        default=0.15,
        help="Recency weight 0.0-1.0 (default 0.15). Higher = prefer newer beliefs",
    )
    query_parser.add_argument(
        "--min-confidence",
        "-c",
        type=float,
        default=None,
        help="Filter beliefs below this confidence threshold (0.0-1.0)",
    )
    query_parser.add_argument(
        "--explain",
        "-e",
        action="store_true",
        help="Show detailed score breakdown per result",
    )
    query_parser.set_defaults(func=cmd_query)

    # list
    list_parser = subparsers.add_parser("list", help="List recent beliefs")
    list_parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
    list_parser.add_argument("--domain", "-d", help="Filter by domain")
    list_parser.set_defaults(func=cmd_list)

    # verify-chains
    verify_parser = subparsers.add_parser("verify-chains", help="Verify supersession chain integrity")
    verify_parser.add_argument("--limit", "-n", type=int, default=None, help="Max chains to verify (default: all)")
    verify_parser.add_argument("--json", action="store_true", dest="output_json", help="Output as JSON")
    verify_parser.set_defaults(func=cmd_verify_chains)


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
        cur.execute(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'beliefs'
            )
        """
        )
        exists = cur.fetchone()["exists"]

        if exists and not args.force:
            print(f"✅ Database already initialized at {db_host}:{db_port}/{db_name}")
            cur.execute("SELECT COUNT(*) as count FROM beliefs")
            count = cur.fetchone()["count"]
            print(f"   📊 {count} beliefs stored")
            conn.close()
            return 0

        conn.close()
    except psycopg2.OperationalError as e:
        print(f"⚠️  Cannot connect to database: {e}")
        print("\nTo create a new database, run:")
        print(f"  createdb -h {db_host} -p {db_port} -U {db_user} {db_name}")
        print("  # Or with Docker:")
        print(f"  docker run -d --name valence-db -e POSTGRES_USER={db_user} \\")
        print(f"    -e POSTGRES_PASSWORD={db_pass or 'valence'} -e POSTGRES_DB={db_name} \\")
        print(f"    -p {db_port}:5432 pgvector/pgvector:pg16")
        return 1

    print("🔨 Creating schema...")

    # Read and execute schema
    Path(__file__).parent.parent.parent / "substrate"
    Path(__file__).parent.parent.parent.parent / "migrations"

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Enable required extensions
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

        # Create core tables if they don't exist
        cur.execute(
            """
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
        """
        )

        cur.execute(
            """
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
        """
        )

        cur.execute(
            """
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
        """
        )

        # Partial unique index for non-aliased entities
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_name_type_unique
            ON entities(name, type) WHERE canonical_id IS NULL
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS belief_entities (
                belief_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
                entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
                role TEXT DEFAULT 'subject',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (belief_id, entity_id)
            )
        """
        )

        cur.execute(
            """
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
        """
        )

        # Create indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_embedding ON beliefs USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_domain ON beliefs USING GIN(domain_path)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_created ON beliefs(created_at DESC)")

        # Create text search
        cur.execute(
            """
            DO $$ BEGIN
                ALTER TABLE beliefs ADD COLUMN content_tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """
        )
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
    derivation_type = args.derivation_type or "observation"

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Insert belief
        cur.execute(
            """
            INSERT INTO beliefs (content, confidence, domain_path, embedding, extraction_method)
            VALUES (%s, %s, %s, %s::vector, %s)
            RETURNING id, created_at
        """,
            (content, json.dumps(confidence), domains, embedding_str, derivation_type),
        )

        row = cur.fetchone()
        belief_id = row["id"]

        conn.commit()

        print(f"✅ Belief added: {str(belief_id)[:8]}...")
        print(f"   📝 {content[:60]}{'...' if len(content) > 60 else ''}")
        print(f"   🎯 Confidence: {format_confidence(confidence)}")
        if domains:
            print(f"   📁 Domains: {', '.join(domains)}")
        if embedding:
            print("   🧠 Embedding: generated")
        else:
            print("   ⚠️  No embedding (set OPENAI_API_KEY)")

        return 0

    except Exception as e:
        print(f"❌ Failed to add belief: {e}")
        return 1
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def cmd_query(args: argparse.Namespace) -> int:
    """Search beliefs with multi-signal ranking and derivation chains."""
    from .federation import cmd_query_federated

    query = args.query

    # Use federated query if scope is federated
    if hasattr(args, "scope") and args.scope == "federated":
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
        recency_weight = getattr(args, "recency_weight", 0.15) or 0.15
        min_confidence = getattr(args, "min_confidence", None)
        explain = getattr(args, "explain", False)

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

            # Semantic search with 6D confidence columns
            cur.execute(
                """
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
                    1 - (b.embedding <=> %s::vector) as similarity
                FROM beliefs b
                WHERE b.embedding IS NOT NULL
                  AND b.status = 'active'
                  AND b.superseded_by_id IS NULL
                  AND 1 - (b.embedding <=> %s::vector) >= %s
                ORDER BY b.embedding <=> %s::vector
                LIMIT %s
            """,
                (embedding_str, embedding_str, args.threshold, embedding_str, args.limit * 2),
            )  # Get extra for filtering

            results = cur.fetchall()

        # Fallback to text search if no embedding or no results
        if not results:
            cur.execute(
                """
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
                    ts_rank(b.content_tsv, websearch_to_tsquery('english', %s)) as similarity
                FROM beliefs b
                WHERE b.content_tsv @@ websearch_to_tsquery('english', %s)
                  AND b.status = 'active'
                  AND b.superseded_by_id IS NULL
                ORDER BY ts_rank(b.content_tsv, websearch_to_tsquery('english', %s)) DESC
                LIMIT %s
            """,
                (query, query, query, args.limit * 2),
            )

            results = cur.fetchall()

        # Convert to list of dicts for processing
        results = [dict(r) for r in results]

        # Domain filter
        if args.domain:
            results = [r for r in results if args.domain in (r.get("domain_path") or [])]

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
        results = results[: args.limit]

        if not results:
            print(f"🔍 No beliefs found for: {query}")
            return 0

        print(f"🔍 Found {len(results)} belief(s) for: {query}")
        if explain:
            print(f"   Weights: semantic={semantic_weight:.2f}, confidence={confidence_weight:.2f}, recency={recency_weight:.2f}")
        print()

        for i, r in enumerate(results, 1):
            final_score = r.get("final_score", 0)
            sim = r.get("similarity", 0)

            # Header
            print(f"{'─' * 60}")
            print(f"[{i}] {r['content'][:70]}{'...' if len(r['content']) > 70 else ''}")

            # Show final score and components
            conf_score = compute_confidence_score(r)
            created_at = r.get("created_at")
            age_str = format_age(created_at) if created_at else "unknown"
            print(f"    ID: {str(r['id'])[:8]}  Score: {final_score:.0%}  Confidence: {conf_score:.0%}  Semantic: {sim:.0%}  Age: {age_str}")

            if r.get("domain_path"):
                print(f"    Domains: {', '.join(r['domain_path'])}")

            # Show score breakdown if explain mode
            if explain and r.get("score_breakdown"):
                bd = r["score_breakdown"]
                print("    ┌─ Score Breakdown:")
                print(f"    │  Semantic:   {bd['semantic']['value']:.2f} × {bd['semantic']['weight']:.2f} = {bd['semantic']['contribution']:.3f}")
                print(
                    f"    │  Confidence: {bd['confidence']['value']:.2f} × {bd['confidence']['weight']:.2f} = {bd['confidence']['contribution']:.3f}"
                )
                print(f"    │  Recency:    {bd['recency']['value']:.2f} × {bd['recency']['weight']:.2f} = {bd['recency']['contribution']:.3f}")
                print(f"    │  Final:      {bd['final']:.3f}")
                print("    └─")

            # === EXTRACTION METHOD ===
            extraction_method = r.get("extraction_method") or "unknown"
            print(f"    ┌─ Method: {extraction_method}")

            # Show supersession chain if exists
            if r.get("supersedes_id"):
                print(f"    │  ⟳ Supersedes: {str(r['supersedes_id'])[:8]}...")
                # Optionally walk the chain
                if args.chain:
                    chain = []
                    current = r["supersedes_id"]
                    depth = 0
                    while current and depth < 5:
                        cur.execute(
                            "SELECT id, content, supersedes_id FROM beliefs WHERE id = %s",
                            (current,),
                        )
                        chain_row = cur.fetchone()
                        if chain_row:
                            chain.append(f"{chain_row['content'][:40]}...")
                            current = chain_row["supersedes_id"]
                            depth += 1
                        else:
                            break
                    for j, c in enumerate(chain):
                        print(f"    │    {'└' if j == len(chain) - 1 else '├'}─ {c}")

            print("    └─")

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
                b.extraction_method
            FROM beliefs b
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
            conf = format_confidence(r.get("confidence", {}))
            age = format_age(r.get("created_at"))
            deriv_raw = r.get("extraction_method") or "?"
            deriv = deriv_raw[:6] if deriv_raw else "?"
            content = r["content"][:55] + "..." if len(r["content"]) > 55 else r["content"]

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


def cmd_verify_chains(args: argparse.Namespace) -> int:
    """Verify supersession chain integrity."""
    import json as json_mod

    from ...core.verification.chain_integrity import verify_chains

    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        report = verify_chains(cur, limit=args.limit)

        if args.output_json:
            print(json_mod.dumps(report.to_dict(), indent=2))
        else:
            print(report)
            if report.issues:
                print()
                for issue in report.issues:
                    ids = ", ".join(bid[:8] for bid in issue.belief_ids)
                    print(f"  [{issue.issue_type}] {issue.description}")
                    if ids:
                        print(f"    beliefs: {ids}")

        return 0 if not report.issues else 1

    except Exception as e:
        print(f"Failed to verify chains: {e}")
        import traceback

        traceback.print_exc()
        return 1
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
