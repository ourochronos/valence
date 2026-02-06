"""Migration 001: Initial schema bootstrap.

Creates the core Valence schema from scratch. This represents the full
schema as of the migration system introduction, consolidating all
previously-archived migrations (v0.2.x 001–014) and substrate migrations
(001–003) into a single baseline.

For fresh installs, this is the starting point.
For existing databases, mark this as applied without running:
    INSERT INTO _migrations (version, description, checksum)
    VALUES ('001', 'initial_schema', '<checksum>');
"""

version = "001"
description = "initial_schema"


def up(conn) -> None:
    """Create the full Valence schema."""
    cur = conn.cursor()
    try:
        # Extensions
        cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
        cur.execute('CREATE EXTENSION IF NOT EXISTS "vector"')

        # ------------------------------------------------------------------
        # Sources
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                type TEXT NOT NULL,
                title TEXT,
                url TEXT,
                content_hash TEXT,
                session_id UUID,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash) WHERE content_hash IS NOT NULL")

        # ------------------------------------------------------------------
        # Beliefs
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content TEXT NOT NULL,
                confidence JSONB NOT NULL DEFAULT '{"overall": 0.7}',
                domain_path TEXT[] NOT NULL DEFAULT '{}',
                valid_from TIMESTAMPTZ,
                valid_until TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                source_id UUID REFERENCES sources(id) ON DELETE SET NULL,
                extraction_method TEXT,
                supersedes_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,
                superseded_by_id UUID REFERENCES beliefs(id) ON DELETE SET NULL,
                status TEXT NOT NULL DEFAULT 'active',
                embedding VECTOR(384),
                content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                -- Federation columns
                origin_node_id UUID,
                is_local BOOLEAN NOT NULL DEFAULT TRUE,
                federation_id UUID,
                visibility TEXT NOT NULL DEFAULT 'private',
                share_level TEXT NOT NULL DEFAULT 'belief_only',
                share_policy JSONB,
                CONSTRAINT beliefs_valid_status CHECK (status IN ('active', 'superseded', 'disputed', 'archived')),
                CONSTRAINT beliefs_valid_confidence CHECK (
                    (confidence->>'overall')::numeric >= 0 AND
                    (confidence->>'overall')::numeric <= 1
                ),
                CONSTRAINT beliefs_valid_visibility CHECK (
                    visibility IN ('private', 'trusted', 'federated', 'public')
                ),
                CONSTRAINT beliefs_valid_share_level CHECK (
                    share_level IN ('belief_only', 'with_provenance', 'full')
                )
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_domain ON beliefs USING GIN (domain_path)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_status ON beliefs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_created ON beliefs(created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_tsv ON beliefs USING GIN (content_tsv)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_source ON beliefs(source_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_federation ON beliefs(federation_id) WHERE federation_id IS NOT NULL")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_beliefs_origin ON beliefs(origin_node_id) WHERE origin_node_id IS NOT NULL")

        # ------------------------------------------------------------------
        # Entities
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                description TEXT,
                properties JSONB DEFAULT '{}',
                embedding VECTOR(384),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                modified_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)")

        # ------------------------------------------------------------------
        # Entity-Belief links
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entity_beliefs (
                entity_id UUID REFERENCES entities(id) ON DELETE CASCADE,
                belief_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
                relationship TEXT NOT NULL DEFAULT 'related',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (entity_id, belief_id)
            )
        """)

        # ------------------------------------------------------------------
        # Tensions (contradictions)
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tensions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                belief_a_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
                belief_b_id UUID REFERENCES beliefs(id) ON DELETE CASCADE,
                similarity FLOAT,
                tension_type TEXT NOT NULL DEFAULT 'contradiction',
                status TEXT NOT NULL DEFAULT 'active',
                resolution TEXT,
                resolved_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT tensions_unique_pair UNIQUE (belief_a_id, belief_b_id)
            )
        """)

        # ------------------------------------------------------------------
        # VKB Session tracking
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vkb_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                agent_id TEXT,
                context JSONB DEFAULT '{}',
                started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                ended_at TIMESTAMPTZ
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS vkb_exchanges (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES vkb_sessions(id) ON DELETE CASCADE,
                tool_name TEXT NOT NULL,
                input_data JSONB DEFAULT '{}',
                output_data JSONB DEFAULT '{}',
                duration_ms INTEGER,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS vkb_patterns (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pattern_type TEXT NOT NULL,
                description TEXT,
                frequency INTEGER DEFAULT 1,
                last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            )
        """)

        # ------------------------------------------------------------------
        # Federation tables
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS federation_nodes (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                did TEXT NOT NULL UNIQUE,
                display_name TEXT,
                endpoint_url TEXT,
                public_key TEXT,
                capabilities JSONB DEFAULT '[]',
                status TEXT NOT NULL DEFAULT 'active',
                last_seen TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT valid_node_status CHECK (status IN ('active', 'inactive', 'blocked'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS node_trust (
                node_id UUID REFERENCES federation_nodes(id) ON DELETE CASCADE,
                topic TEXT NOT NULL DEFAULT '*',
                trust_level FLOAT NOT NULL DEFAULT 0.5,
                evidence JSONB DEFAULT '{}',
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (node_id, topic),
                CONSTRAINT valid_trust_level CHECK (trust_level >= 0.0 AND trust_level <= 1.0)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_state (
                node_id UUID REFERENCES federation_nodes(id) ON DELETE CASCADE,
                last_sync TIMESTAMPTZ,
                last_hash TEXT,
                sync_status TEXT NOT NULL DEFAULT 'idle',
                error_count INTEGER DEFAULT 0,
                PRIMARY KEY (node_id)
            )
        """)

        # ------------------------------------------------------------------
        # Consent chains (sharing)
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS consent_chains (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                belief_id UUID NOT NULL REFERENCES beliefs(id) ON DELETE CASCADE,
                origin_sharer TEXT NOT NULL,
                origin_timestamp TIMESTAMPTZ NOT NULL,
                origin_policy JSONB NOT NULL,
                origin_signature BYTEA NOT NULL,
                chain_links JSONB DEFAULT '[]',
                is_valid BOOLEAN NOT NULL DEFAULT TRUE,
                revoked_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS encrypted_shares (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                consent_chain_id UUID NOT NULL REFERENCES consent_chains(id) ON DELETE CASCADE,
                encrypted_content BYTEA NOT NULL,
                encryption_algorithm TEXT NOT NULL DEFAULT 'xchacha20-poly1305',
                recipient_did TEXT NOT NULL,
                key_agreement JSONB NOT NULL,
                nonce BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # ------------------------------------------------------------------
        # User-level trust (delegation)
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_node_trust (
                user_id TEXT NOT NULL,
                node_id UUID NOT NULL REFERENCES federation_nodes(id) ON DELETE CASCADE,
                topic TEXT NOT NULL DEFAULT '*',
                trust_level FLOAT NOT NULL DEFAULT 0.5,
                notes TEXT,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (user_id, node_id, topic),
                CONSTRAINT valid_user_trust CHECK (trust_level >= 0.0 AND trust_level <= 1.0)
            )
        """)

        # ------------------------------------------------------------------
        # Schema version (legacy — kept for backward compat)
        # ------------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

    finally:
        cur.close()


def down(conn) -> None:
    """Drop all Valence tables (DESTRUCTIVE — use with care)."""
    cur = conn.cursor()
    try:
        tables = [
            "encrypted_shares",
            "consent_chains",
            "sync_state",
            "node_trust",
            "user_node_trust",
            "federation_nodes",
            "vkb_patterns",
            "vkb_exchanges",
            "vkb_sessions",
            "tensions",
            "entity_beliefs",
            "entities",
            "beliefs",
            "sources",
            "schema_version",
        ]
        for table in tables:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    finally:
        cur.close()
