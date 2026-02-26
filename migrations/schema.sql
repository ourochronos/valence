--
-- Valence v2.0 Schema
-- Consolidated schema replacing 27 migration files
--

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Extensions
--

CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public;
CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA public;

SET default_tablespace = '';
SET default_table_access_method = heap;

-- ============================================================================
-- Core Knowledge Tables
-- ============================================================================

--
-- sources: Immutable evidence (append-only with redaction support)
--
CREATE TABLE public.sources (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    type text NOT NULL,
    title text,
    url text,
    content text,
    fingerprint text,
    reliability numeric(3,2) DEFAULT 0.5 NOT NULL,
    content_hash text,
    session_id uuid,
    metadata jsonb DEFAULT '{}'::jsonb,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    embedding public.vector(1536),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, COALESCE(content, ''::text))) STORED,
    redacted_at timestamp with time zone,
    redacted_by text,
    supersedes_id uuid
);

COMMENT ON COLUMN public.sources.reliability IS 'Initial reliability: document=0.8, code=0.8, web=0.6, conversation=0.5, observation=0.4, tool_output=0.7, user_input=0.75';
COMMENT ON COLUMN public.sources.redacted_at IS 'Privacy redaction timestamp. When set, content and embedding are NULL.';
COMMENT ON COLUMN public.sources.redacted_by IS 'Reason for redaction (e.g., "PII removal per user request").';

ALTER TABLE public.sources OWNER TO valence;

--
-- articles: Compiled knowledge units
--
CREATE TABLE public.articles (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    content text NOT NULL,
    title text,
    author_type text DEFAULT 'system'::text NOT NULL,
    pinned boolean DEFAULT false NOT NULL,
    epistemic_type text DEFAULT 'semantic' NOT NULL,
    size_tokens integer,
    compiled_at timestamp with time zone,
    usage_score numeric(8,4) DEFAULT 0 NOT NULL,
    confidence jsonb DEFAULT '{"overall": 0.7}'::jsonb NOT NULL,
    domain_path text[] DEFAULT '{}'::text[] NOT NULL,
    valid_from timestamp with time zone,
    valid_until timestamp with time zone,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    modified_at timestamp with time zone DEFAULT now() NOT NULL,
    source_id uuid,
    extraction_method text,
    extraction_metadata jsonb,
    supersedes_id uuid,
    superseded_by_id uuid,
    holder_id uuid,
    version integer DEFAULT 1 NOT NULL,
    content_hash character(64),
    status text DEFAULT 'active'::text NOT NULL,
    archived_at timestamp with time zone,
    corroboration_count integer DEFAULT 0 NOT NULL,
    corroborating_sources jsonb DEFAULT '[]'::jsonb NOT NULL,
    confidence_source real DEFAULT 0.5,
    confidence_method real DEFAULT 0.5,
    confidence_consistency real DEFAULT 1.0,
    confidence_freshness real DEFAULT 1.0,
    confidence_corroboration real DEFAULT 0.1,
    confidence_applicability real DEFAULT 0.8,
    embedding public.vector(1536),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english'::regconfig, content)) STORED,
    degraded boolean DEFAULT false,
    CONSTRAINT articles_author_type_check CHECK ((author_type = ANY (ARRAY['system'::text, 'operator'::text, 'agent'::text]))),
    CONSTRAINT articles_valid_confidence CHECK (((((confidence ->> 'overall'::text))::numeric >= (0)::numeric) AND (((confidence ->> 'overall'::text))::numeric <= (1)::numeric))),
    CONSTRAINT articles_valid_status CHECK ((status = ANY (ARRAY['active'::text, 'superseded'::text, 'disputed'::text, 'archived'::text]))),
    CONSTRAINT articles_version_positive CHECK ((version > 0))
);

ALTER TABLE public.articles OWNER TO valence;

--
-- article_sources: Provenance links (many-to-many with relationship type)
--
CREATE TABLE public.article_sources (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    article_id uuid NOT NULL,
    source_id uuid NOT NULL,
    relationship text NOT NULL,
    added_at timestamp with time zone DEFAULT now() NOT NULL,
    notes text,
    CONSTRAINT article_sources_relationship_check CHECK ((relationship = ANY (ARRAY['originates'::text, 'confirms'::text, 'supersedes'::text, 'contradicts'::text, 'contends'::text])))
);

ALTER TABLE public.article_sources OWNER TO valence;

--
-- usage_traces: Retrieval signals for decay/prioritization
--
CREATE TABLE public.usage_traces (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    article_id uuid NOT NULL,
    query_text text,
    tool_name text NOT NULL,
    retrieved_at timestamp with time zone DEFAULT now() NOT NULL,
    final_score numeric,
    session_id uuid,
    source_id uuid
);

ALTER TABLE public.usage_traces OWNER TO valence;

--
-- contentions: Flagged disagreements between sources
--
CREATE TABLE public.contentions (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    article_id uuid NOT NULL,
    related_article_id uuid,
    type text DEFAULT 'contradiction'::text NOT NULL,
    description text,
    severity text DEFAULT 'medium'::text NOT NULL,
    status text DEFAULT 'detected'::text NOT NULL,
    resolution text,
    resolved_at timestamp with time zone,
    detected_at timestamp with time zone DEFAULT now() NOT NULL,
    materiality numeric(3,2) DEFAULT 0.5,
    opt_out_federation boolean DEFAULT false NOT NULL,
    share_policy jsonb,
    extraction_metadata jsonb,
    degraded boolean DEFAULT false,
    CONSTRAINT contentions_different_articles CHECK ((article_id <> related_article_id)),
    CONSTRAINT contentions_valid_severity CHECK ((severity = ANY (ARRAY['low'::text, 'medium'::text, 'high'::text, 'critical'::text]))),
    CONSTRAINT contentions_valid_status CHECK ((status = ANY (ARRAY['detected'::text, 'investigating'::text, 'resolved'::text, 'accepted'::text]))),
    CONSTRAINT contentions_valid_type CHECK ((type = ANY (ARRAY['contradiction'::text, 'temporal_conflict'::text, 'scope_conflict'::text, 'partial_overlap'::text])))
);

ALTER TABLE public.contentions OWNER TO valence;

--
-- entities: Extracted entities from sources/articles
--
CREATE TABLE public.entities (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    name text NOT NULL,
    type text NOT NULL,
    description text,
    aliases text[] DEFAULT '{}'::text[],
    canonical_id uuid,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    modified_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT entities_valid_type CHECK ((type = ANY (ARRAY['person'::text, 'organization'::text, 'tool'::text, 'concept'::text, 'project'::text, 'location'::text, 'service'::text])))
);

ALTER TABLE public.entities OWNER TO valence;

--
-- article_entities: Entity-article links
--
CREATE TABLE public.article_entities (
    article_id uuid NOT NULL,
    entity_id uuid NOT NULL,
    role text DEFAULT 'subject'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT article_entities_valid_role CHECK ((role = ANY (ARRAY['subject'::text, 'object'::text, 'context'::text, 'source'::text])))
);

ALTER TABLE public.article_entities OWNER TO valence;

--
-- system_config: Runtime configuration
--
CREATE TABLE public.system_config (
    key text NOT NULL,
    value jsonb NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL
);

ALTER TABLE public.system_config OWNER TO valence;

--
-- article_mutations: Article edit history (audit trail)
--
CREATE TABLE public.article_mutations (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    mutation_type text NOT NULL,
    article_id uuid NOT NULL,
    related_article_id uuid,
    trigger_source_id uuid,
    summary text,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT article_mutations_mutation_type_check CHECK ((mutation_type = ANY (ARRAY['created'::text, 'updated'::text, 'split'::text, 'merged'::text, 'archived'::text])))
);

ALTER TABLE public.article_mutations OWNER TO valence;

--
-- mutation_queue: Pending async mutations (background compilation)
--
CREATE TABLE public.mutation_queue (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    operation text NOT NULL,
    article_id uuid NOT NULL,
    priority integer DEFAULT 5 NOT NULL,
    payload jsonb DEFAULT '{}'::jsonb,
    status text DEFAULT 'pending'::text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    processed_at timestamp with time zone,
    CONSTRAINT mutation_queue_operation_check CHECK ((operation = ANY (ARRAY['split'::text, 'merge_candidate'::text, 'recompile'::text, 'decay_check'::text, 'recompile_degraded'::text]))),
    CONSTRAINT mutation_queue_status_check CHECK ((status = ANY (ARRAY['pending'::text, 'processing'::text, 'completed'::text, 'failed'::text])))
);

ALTER TABLE public.mutation_queue OWNER TO valence;

--
-- compilation_queue: Queued compilations when inference unavailable (#491)
--
CREATE TABLE public.compilation_queue (
    id uuid DEFAULT gen_random_uuid() NOT NULL,
    source_ids text[] NOT NULL,
    title_hint text,
    queued_at timestamp with time zone DEFAULT now() NOT NULL,
    attempts integer DEFAULT 0 NOT NULL,
    last_attempt timestamp with time zone,
    status text DEFAULT 'pending'::text NOT NULL,
    CONSTRAINT compilation_queue_status_check CHECK ((status = ANY (ARRAY['pending'::text, 'processing'::text, 'failed'::text])))
);

ALTER TABLE public.compilation_queue OWNER TO valence;

-- ============================================================================
-- Views
-- ============================================================================

--
-- article_usage: View of usage traces for articles
--
CREATE VIEW public.article_usage AS
SELECT 
    article_id,
    query_text,
    tool_name,
    retrieved_at,
    final_score,
    session_id
FROM public.usage_traces
WHERE article_id IS NOT NULL;

ALTER VIEW public.article_usage OWNER TO valence;

--
-- articles_current: View of active, non-superseded articles
--
CREATE VIEW public.articles_current AS
SELECT 
    id,
    content,
    title,
    author_type,
    pinned,
    size_tokens,
    compiled_at,
    usage_score,
    confidence,
    domain_path,
    valid_from,
    valid_until,
    created_at,
    modified_at,
    source_id,
    extraction_method,
    extraction_metadata,
    supersedes_id,
    superseded_by_id,
    holder_id,
    version,
    content_hash,
    status,
    archived_at,
    corroboration_count,
    corroborating_sources,
    confidence_source,
    confidence_method,
    confidence_consistency,
    confidence_freshness,
    confidence_corroboration,
    confidence_applicability,
    embedding,
    content_tsv
FROM public.articles
WHERE status = 'active' AND superseded_by_id IS NULL;

ALTER VIEW public.articles_current OWNER TO valence;

--
-- articles_with_sources: View of articles with source relationships
--
CREATE VIEW public.articles_with_sources AS
SELECT 
    a.id,
    a.content,
    a.title,
    a.author_type,
    a.pinned,
    a.size_tokens,
    a.compiled_at,
    a.usage_score,
    a.confidence,
    a.domain_path,
    a.valid_from,
    a.valid_until,
    a.created_at,
    a.modified_at,
    a.source_id,
    a.extraction_method,
    a.extraction_metadata,
    a.supersedes_id,
    a.superseded_by_id,
    a.holder_id,
    a.version,
    a.content_hash,
    a.status,
    a.archived_at,
    a.corroboration_count,
    a.corroborating_sources,
    a.confidence_source,
    a.confidence_method,
    a.confidence_consistency,
    a.confidence_freshness,
    a.confidence_corroboration,
    a.confidence_applicability,
    a.embedding,
    a.content_tsv,
    COUNT(DISTINCT asrc.source_id) AS source_count,
    ARRAY_AGG(DISTINCT asrc.relationship) FILTER (WHERE asrc.relationship IS NOT NULL) AS relationship_types,
    BOOL_OR((asrc.relationship = 'contradicts' OR asrc.relationship = 'contends')) AS has_contention
FROM public.articles a
LEFT JOIN public.article_sources asrc ON a.id = asrc.article_id
WHERE a.status = 'active'
GROUP BY a.id;

ALTER VIEW public.articles_with_sources OWNER TO valence;

-- ============================================================================
-- Primary Keys
-- ============================================================================

ALTER TABLE ONLY public.sources
    ADD CONSTRAINT sources_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.article_sources
    ADD CONSTRAINT article_sources_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.article_sources
    ADD CONSTRAINT article_sources_article_id_source_id_relationship_key UNIQUE (article_id, source_id, relationship);

ALTER TABLE ONLY public.usage_traces
    ADD CONSTRAINT usage_traces_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.contentions
    ADD CONSTRAINT contentions_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.article_entities
    ADD CONSTRAINT article_entities_pkey PRIMARY KEY (article_id, entity_id, role);

ALTER TABLE ONLY public.system_config
    ADD CONSTRAINT system_config_pkey PRIMARY KEY (key);

ALTER TABLE ONLY public.article_mutations
    ADD CONSTRAINT article_mutations_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.mutation_queue
    ADD CONSTRAINT mutation_queue_pkey PRIMARY KEY (id);

ALTER TABLE ONLY public.compilation_queue
    ADD CONSTRAINT compilation_queue_pkey PRIMARY KEY (id);

-- ============================================================================
-- Indexes
-- ============================================================================

-- sources
CREATE UNIQUE INDEX idx_sources_fingerprint ON public.sources USING btree (fingerprint) WHERE (fingerprint IS NOT NULL);
CREATE INDEX idx_sources_type ON public.sources USING btree (type);
CREATE INDEX idx_sources_created ON public.sources USING btree (created_at DESC);
CREATE INDEX idx_sources_hash ON public.sources USING btree (content_hash) WHERE (content_hash IS NOT NULL);
CREATE INDEX idx_sources_tsv ON public.sources USING gin (content_tsv);
CREATE INDEX idx_sources_embedding ON public.sources USING hnsw (embedding public.vector_cosine_ops) WITH (m='16', ef_construction='200');
CREATE INDEX idx_sources_supersedes ON public.sources USING btree (supersedes_id) WHERE (supersedes_id IS NOT NULL);

-- articles
CREATE INDEX idx_articles_status ON public.articles USING btree (status);
CREATE INDEX idx_articles_created ON public.articles USING btree (created_at DESC);
CREATE INDEX idx_articles_domain ON public.articles USING gin (domain_path);
CREATE INDEX idx_articles_source ON public.articles USING btree (source_id);
CREATE INDEX idx_articles_holder ON public.articles USING btree (holder_id) WHERE (holder_id IS NOT NULL);
CREATE INDEX idx_articles_content_hash ON public.articles USING btree (content_hash) WHERE (content_hash IS NOT NULL);
CREATE INDEX idx_articles_archived ON public.articles USING btree (archived_at DESC) WHERE (status = 'archived');
CREATE INDEX idx_articles_archival_candidates ON public.articles USING btree (modified_at) WHERE (status = 'superseded');
CREATE INDEX idx_articles_tsv ON public.articles USING gin (content_tsv);
CREATE INDEX idx_articles_embedding ON public.articles USING hnsw (embedding public.vector_cosine_ops) WITH (m='16', ef_construction='200');

-- article_sources
CREATE INDEX idx_article_sources_article ON public.article_sources USING btree (article_id);
CREATE INDEX idx_article_sources_source ON public.article_sources USING btree (source_id);
CREATE INDEX idx_article_sources_rel ON public.article_sources USING btree (relationship);

-- usage_traces
CREATE INDEX idx_usage_traces_article ON public.usage_traces USING btree (article_id);
CREATE INDEX idx_usage_traces_time ON public.usage_traces USING btree (retrieved_at DESC);

-- contentions
CREATE INDEX idx_contentions_article ON public.contentions USING btree (article_id);
CREATE INDEX idx_contentions_related_article ON public.contentions USING btree (related_article_id);
CREATE INDEX idx_contentions_status ON public.contentions USING btree (status);
CREATE INDEX idx_contentions_severity ON public.contentions USING btree (severity);

-- entities
CREATE INDEX idx_entities_name ON public.entities USING btree (name);
CREATE INDEX idx_entities_type ON public.entities USING btree (type);
CREATE INDEX idx_entities_aliases ON public.entities USING gin (aliases);
CREATE INDEX idx_entities_canonical ON public.entities USING btree (canonical_id) WHERE (canonical_id IS NOT NULL);
CREATE UNIQUE INDEX idx_entities_unique_canonical ON public.entities USING btree (lower(name), type) WHERE (canonical_id IS NULL);

-- article_entities
CREATE INDEX idx_article_entities_entity ON public.article_entities USING btree (entity_id);

-- article_mutations
CREATE INDEX idx_article_mutations_article ON public.article_mutations USING btree (article_id);
CREATE INDEX idx_article_mutations_type ON public.article_mutations USING btree (mutation_type);

-- mutation_queue
CREATE INDEX idx_mutation_queue_status ON public.mutation_queue USING btree (status, priority);
CREATE INDEX idx_mutation_queue_article ON public.mutation_queue USING btree (article_id);

-- compilation_queue
CREATE INDEX idx_compilation_queue_status ON public.compilation_queue USING btree (status, queued_at);

-- ============================================================================
-- Foreign Keys
-- ============================================================================

-- article_sources
ALTER TABLE ONLY public.article_sources
    ADD CONSTRAINT article_sources_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.article_sources
    ADD CONSTRAINT article_sources_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.sources(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.sources
    ADD CONSTRAINT sources_supersedes_id_fkey FOREIGN KEY (supersedes_id) REFERENCES public.sources(id) ON DELETE SET NULL;

-- articles
ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.sources(id) ON DELETE SET NULL;

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_supersedes_id_fkey FOREIGN KEY (supersedes_id) REFERENCES public.articles(id) ON DELETE SET NULL;

ALTER TABLE ONLY public.articles
    ADD CONSTRAINT articles_superseded_by_id_fkey FOREIGN KEY (superseded_by_id) REFERENCES public.articles(id) ON DELETE SET NULL;

-- usage_traces
ALTER TABLE ONLY public.usage_traces
    ADD CONSTRAINT usage_traces_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.usage_traces
    ADD CONSTRAINT usage_traces_source_id_fkey FOREIGN KEY (source_id) REFERENCES public.sources(id) ON DELETE SET NULL;

-- contentions
ALTER TABLE ONLY public.contentions
    ADD CONSTRAINT contentions_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.contentions
    ADD CONSTRAINT contentions_related_article_id_fkey FOREIGN KEY (related_article_id) REFERENCES public.articles(id) ON DELETE CASCADE;

-- entities
ALTER TABLE ONLY public.entities
    ADD CONSTRAINT entities_canonical_id_fkey FOREIGN KEY (canonical_id) REFERENCES public.entities(id) ON DELETE SET NULL;

-- article_entities
ALTER TABLE ONLY public.article_entities
    ADD CONSTRAINT article_entities_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.article_entities
    ADD CONSTRAINT article_entities_entity_id_fkey FOREIGN KEY (entity_id) REFERENCES public.entities(id) ON DELETE CASCADE;

-- article_mutations
ALTER TABLE ONLY public.article_mutations
    ADD CONSTRAINT article_mutations_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id) ON DELETE CASCADE;

ALTER TABLE ONLY public.article_mutations
    ADD CONSTRAINT article_mutations_related_article_id_fkey FOREIGN KEY (related_article_id) REFERENCES public.articles(id) ON DELETE SET NULL;

ALTER TABLE ONLY public.article_mutations
    ADD CONSTRAINT article_mutations_trigger_source_id_fkey FOREIGN KEY (trigger_source_id) REFERENCES public.sources(id) ON DELETE SET NULL;

-- mutation_queue
ALTER TABLE ONLY public.mutation_queue
    ADD CONSTRAINT mutation_queue_article_id_fkey FOREIGN KEY (article_id) REFERENCES public.articles(id) ON DELETE CASCADE;

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Migrate embedding_types config to system_config
INSERT INTO public.system_config (key, value, updated_at)
VALUES ('embedding_config', '{"provider": "openai", "model": "text-embedding-3-small", "dimensions": 1536, "is_default": true}', now())
ON CONFLICT (key) DO NOTHING;
-- Migration 004: Session ingestion tables
-- Issue #469: Add sessions and session_messages tables
-- Date: 2026-02-24

CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    channel TEXT,
    participants TEXT[] DEFAULT '{}',
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB DEFAULT '{}',
    parent_session_id TEXT REFERENCES sessions(session_id),
    subagent_label TEXT,
    subagent_model TEXT,
    subagent_task TEXT,
    current_chunk_index INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE session_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    chunk_index INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    speaker TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    flushed_at TIMESTAMPTZ
);

CREATE INDEX idx_session_messages_unflushed ON session_messages (session_id) WHERE flushed_at IS NULL;
CREATE INDEX idx_sessions_stale ON sessions (last_activity_at) WHERE status = 'active';
CREATE INDEX idx_sessions_parent ON sessions (parent_session_id) WHERE parent_session_id IS NOT NULL;
