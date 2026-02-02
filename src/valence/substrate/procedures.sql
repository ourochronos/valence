-- Valence Stored Procedures
-- PostgreSQL functions for common operations

-- ============================================================================
-- BELIEF OPERATIONS
-- ============================================================================

-- Create a belief with optional entity links (atomic operation)
CREATE OR REPLACE FUNCTION belief_create_with_entities(
    p_content TEXT,
    p_confidence JSONB DEFAULT '{"overall": 0.7}',
    p_domain_path TEXT[] DEFAULT '{}',
    p_source_id UUID DEFAULT NULL,
    p_extraction_method TEXT DEFAULT NULL,
    p_valid_from TIMESTAMPTZ DEFAULT NULL,
    p_entities JSONB DEFAULT '[]'
    -- entities format: [{"name": "...", "type": "...", "role": "subject"}, ...]
)
RETURNS UUID AS $$
DECLARE
    v_belief_id UUID;
    v_entity JSONB;
    v_entity_id UUID;
BEGIN
    -- Create the belief
    INSERT INTO beliefs (content, confidence, domain_path, source_id, extraction_method, valid_from)
    VALUES (p_content, p_confidence, p_domain_path, p_source_id, p_extraction_method, p_valid_from)
    RETURNING id INTO v_belief_id;

    -- Link entities
    FOR v_entity IN SELECT * FROM jsonb_array_elements(p_entities)
    LOOP
        -- Find or create entity
        INSERT INTO entities (name, type)
        VALUES (
            v_entity->>'name',
            COALESCE(v_entity->>'type', 'concept')
        )
        ON CONFLICT (name, type) WHERE canonical_id IS NULL
        DO UPDATE SET modified_at = NOW()
        RETURNING id INTO v_entity_id;

        -- Link to belief
        INSERT INTO belief_entities (belief_id, entity_id, role)
        VALUES (
            v_belief_id,
            v_entity_id,
            COALESCE(v_entity->>'role', 'subject')
        )
        ON CONFLICT DO NOTHING;
    END LOOP;

    RETURN v_belief_id;
END;
$$ LANGUAGE plpgsql;


-- Supersede a belief with a new one
CREATE OR REPLACE FUNCTION belief_supersede(
    p_old_belief_id UUID,
    p_new_content TEXT,
    p_reason TEXT,
    p_confidence JSONB DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_old_belief beliefs%ROWTYPE;
    v_new_belief_id UUID;
    v_new_confidence JSONB;
BEGIN
    -- Get old belief
    SELECT * INTO v_old_belief FROM beliefs WHERE id = p_old_belief_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Belief not found: %', p_old_belief_id;
    END IF;

    -- Determine confidence for new belief
    v_new_confidence := COALESCE(p_confidence, v_old_belief.confidence);

    -- Create new belief
    INSERT INTO beliefs (
        content, confidence, domain_path, source_id,
        extraction_method, supersedes_id, valid_from
    )
    VALUES (
        p_new_content,
        v_new_confidence,
        v_old_belief.domain_path,
        v_old_belief.source_id,
        'supersession: ' || p_reason,
        p_old_belief_id,
        NOW()
    )
    RETURNING id INTO v_new_belief_id;

    -- Update old belief
    UPDATE beliefs
    SET status = 'superseded',
        superseded_by_id = v_new_belief_id,
        valid_until = NOW(),
        modified_at = NOW()
    WHERE id = p_old_belief_id;

    -- Copy entity links to new belief
    INSERT INTO belief_entities (belief_id, entity_id, role)
    SELECT v_new_belief_id, entity_id, role
    FROM belief_entities
    WHERE belief_id = p_old_belief_id;

    RETURN v_new_belief_id;
END;
$$ LANGUAGE plpgsql;


-- Search beliefs with hybrid search (keyword + semantic)
CREATE OR REPLACE FUNCTION belief_search(
    p_query TEXT,
    p_query_embedding VECTOR(1536) DEFAULT NULL,
    p_domain_filter TEXT[] DEFAULT NULL,
    p_entity_id UUID DEFAULT NULL,
    p_include_superseded BOOLEAN DEFAULT FALSE,
    p_limit INTEGER DEFAULT 20,
    p_semantic_weight FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    id UUID,
    content TEXT,
    confidence JSONB,
    domain_path TEXT[],
    created_at TIMESTAMPTZ,
    source_id UUID,
    status TEXT,
    keyword_score FLOAT,
    semantic_score FLOAT,
    combined_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH keyword_matches AS (
        SELECT
            b.id,
            ts_rank(b.content_tsv, websearch_to_tsquery('english', p_query)) as score
        FROM beliefs b
        WHERE b.content_tsv @@ websearch_to_tsquery('english', p_query)
    ),
    semantic_matches AS (
        SELECT
            b.id,
            1 - (b.embedding <=> p_query_embedding) as score
        FROM beliefs b
        WHERE b.embedding IS NOT NULL
          AND p_query_embedding IS NOT NULL
    ),
    combined AS (
        SELECT
            b.*,
            COALESCE(km.score, 0) as kw_score,
            COALESCE(sm.score, 0) as sem_score,
            (1 - p_semantic_weight) * COALESCE(km.score, 0) +
            p_semantic_weight * COALESCE(sm.score, 0) as comb_score
        FROM beliefs b
        LEFT JOIN keyword_matches km ON b.id = km.id
        LEFT JOIN semantic_matches sm ON b.id = sm.id
        WHERE (km.id IS NOT NULL OR sm.id IS NOT NULL)
          AND (p_include_superseded OR (b.status = 'active' AND b.superseded_by_id IS NULL))
          AND (p_domain_filter IS NULL OR b.domain_path && p_domain_filter)
          AND (p_entity_id IS NULL OR EXISTS (
              SELECT 1 FROM belief_entities be WHERE be.belief_id = b.id AND be.entity_id = p_entity_id
          ))
    )
    SELECT
        c.id,
        c.content,
        c.confidence,
        c.domain_path,
        c.created_at,
        c.source_id,
        c.status,
        c.kw_score::FLOAT,
        c.sem_score::FLOAT,
        c.comb_score::FLOAT
    FROM combined c
    ORDER BY c.comb_score DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- ENTITY OPERATIONS
-- ============================================================================

-- Find or create an entity
CREATE OR REPLACE FUNCTION entity_find_or_create(
    p_name TEXT,
    p_type TEXT DEFAULT 'concept',
    p_description TEXT DEFAULT NULL,
    p_aliases TEXT[] DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    v_entity_id UUID;
BEGIN
    -- Try to find existing entity (including by alias)
    SELECT id INTO v_entity_id
    FROM entities
    WHERE (name = p_name OR p_name = ANY(aliases))
      AND type = p_type
      AND canonical_id IS NULL;

    IF FOUND THEN
        -- Update aliases if new ones provided
        IF array_length(p_aliases, 1) > 0 THEN
            UPDATE entities
            SET aliases = aliases || p_aliases,
                modified_at = NOW()
            WHERE id = v_entity_id;
        END IF;
        RETURN v_entity_id;
    END IF;

    -- Create new entity
    INSERT INTO entities (name, type, description, aliases)
    VALUES (p_name, p_type, p_description, p_aliases)
    RETURNING id INTO v_entity_id;

    RETURN v_entity_id;
END;
$$ LANGUAGE plpgsql;


-- Merge two entities (redirect from_entity to to_entity)
CREATE OR REPLACE FUNCTION entity_merge(
    p_from_entity_id UUID,
    p_to_entity_id UUID
)
RETURNS VOID AS $$
DECLARE
    v_from_entity entities%ROWTYPE;
BEGIN
    -- Get source entity
    SELECT * INTO v_from_entity FROM entities WHERE id = p_from_entity_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Source entity not found: %', p_from_entity_id;
    END IF;

    -- Update target entity with source aliases
    UPDATE entities
    SET aliases = aliases || ARRAY[v_from_entity.name] || v_from_entity.aliases,
        modified_at = NOW()
    WHERE id = p_to_entity_id;

    -- Delete belief_entities that would cause duplicates after merge
    DELETE FROM belief_entities be1
    WHERE be1.entity_id = p_from_entity_id
      AND EXISTS (
          SELECT 1 FROM belief_entities be2
          WHERE be2.entity_id = p_to_entity_id
            AND be2.belief_id = be1.belief_id
            AND be2.role = be1.role
      );

    -- Update remaining belief_entities to point to target
    UPDATE belief_entities
    SET entity_id = p_to_entity_id
    WHERE entity_id = p_from_entity_id;

    -- Mark source as merged
    UPDATE entities
    SET canonical_id = p_to_entity_id,
        modified_at = NOW()
    WHERE id = p_from_entity_id;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- TENSION OPERATIONS
-- ============================================================================

-- Detect potential tensions in recent beliefs
CREATE OR REPLACE FUNCTION tension_detect(
    p_since TIMESTAMPTZ DEFAULT NOW() - INTERVAL '7 days',
    p_similarity_threshold FLOAT DEFAULT 0.85
)
RETURNS TABLE (
    belief_a_id UUID,
    belief_b_id UUID,
    similarity FLOAT
) AS $$
BEGIN
    -- Find pairs of recent beliefs with high embedding similarity
    -- that might be contradictions (same topic, different claims)
    RETURN QUERY
    SELECT DISTINCT
        b1.id as belief_a_id,
        b2.id as belief_b_id,
        (1 - (b1.embedding <=> b2.embedding))::FLOAT as similarity
    FROM beliefs b1
    JOIN beliefs b2 ON b1.id < b2.id  -- Avoid duplicates
    WHERE b1.created_at > p_since
      AND b2.created_at > p_since
      AND b1.status = 'active'
      AND b2.status = 'active'
      AND b1.superseded_by_id IS NULL
      AND b2.superseded_by_id IS NULL
      AND b1.embedding IS NOT NULL
      AND b2.embedding IS NOT NULL
      AND (1 - (b1.embedding <=> b2.embedding)) > p_similarity_threshold
      -- Exclude already recorded tensions
      AND NOT EXISTS (
          SELECT 1 FROM tensions t
          WHERE (t.belief_a_id = b1.id AND t.belief_b_id = b2.id)
             OR (t.belief_a_id = b2.id AND t.belief_b_id = b1.id)
      );
END;
$$ LANGUAGE plpgsql;


-- Record a new tension
CREATE OR REPLACE FUNCTION tension_record(
    p_belief_a_id UUID,
    p_belief_b_id UUID,
    p_type TEXT DEFAULT 'contradiction',
    p_description TEXT DEFAULT NULL,
    p_severity TEXT DEFAULT 'medium'
)
RETURNS UUID AS $$
DECLARE
    v_tension_id UUID;
BEGIN
    INSERT INTO tensions (belief_a_id, belief_b_id, type, description, severity)
    VALUES (p_belief_a_id, p_belief_b_id, p_type, p_description, p_severity)
    RETURNING id INTO v_tension_id;

    RETURN v_tension_id;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SESSION OPERATIONS
-- ============================================================================

-- Start a new session
CREATE OR REPLACE FUNCTION session_start(
    p_platform TEXT,
    p_project_context TEXT DEFAULT NULL,
    p_external_room_id TEXT DEFAULT NULL,
    p_claude_session_id TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS UUID AS $$
DECLARE
    v_session_id UUID;
BEGIN
    INSERT INTO vkb_sessions (platform, project_context, external_room_id, claude_session_id, metadata)
    VALUES (p_platform, p_project_context, p_external_room_id, p_claude_session_id, p_metadata)
    RETURNING id INTO v_session_id;

    -- Create source record for this session
    INSERT INTO sources (type, session_id, title)
    VALUES ('conversation', v_session_id, 'Session: ' || v_session_id::TEXT);

    RETURN v_session_id;
END;
$$ LANGUAGE plpgsql;


-- End a session with summary
CREATE OR REPLACE FUNCTION session_end(
    p_session_id UUID,
    p_summary TEXT DEFAULT NULL,
    p_themes TEXT[] DEFAULT NULL,
    p_status TEXT DEFAULT 'completed'
)
RETURNS VOID AS $$
BEGIN
    UPDATE vkb_sessions
    SET status = p_status,
        summary = COALESCE(p_summary, summary),
        themes = COALESCE(p_themes, themes),
        ended_at = NOW()
    WHERE id = p_session_id;
END;
$$ LANGUAGE plpgsql;


-- Add an exchange to a session
CREATE OR REPLACE FUNCTION exchange_add(
    p_session_id UUID,
    p_role TEXT,
    p_content TEXT,
    p_tokens_approx INTEGER DEFAULT NULL,
    p_tool_uses JSONB DEFAULT '[]'
)
RETURNS UUID AS $$
DECLARE
    v_exchange_id UUID;
    v_sequence INTEGER;
BEGIN
    -- Get next sequence number
    SELECT COALESCE(MAX(sequence), 0) + 1
    INTO v_sequence
    FROM vkb_exchanges
    WHERE session_id = p_session_id;

    INSERT INTO vkb_exchanges (session_id, sequence, role, content, tokens_approx, tool_uses)
    VALUES (p_session_id, v_sequence, p_role, p_content, p_tokens_approx, p_tool_uses)
    RETURNING id INTO v_exchange_id;

    RETURN v_exchange_id;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- PATTERN OPERATIONS
-- ============================================================================

-- Record a new pattern
CREATE OR REPLACE FUNCTION pattern_record(
    p_type TEXT,
    p_description TEXT,
    p_evidence UUID[] DEFAULT '{}',
    p_confidence NUMERIC DEFAULT 0.5
)
RETURNS UUID AS $$
DECLARE
    v_pattern_id UUID;
BEGIN
    INSERT INTO vkb_patterns (type, description, evidence, confidence)
    VALUES (p_type, p_description, p_evidence, p_confidence)
    RETURNING id INTO v_pattern_id;

    RETURN v_pattern_id;
END;
$$ LANGUAGE plpgsql;


-- Reinforce an existing pattern
CREATE OR REPLACE FUNCTION pattern_reinforce(
    p_pattern_id UUID,
    p_session_id UUID DEFAULT NULL
)
RETURNS VOID AS $$
DECLARE
    v_current_evidence UUID[];
    v_new_confidence NUMERIC;
BEGIN
    -- Get current pattern
    SELECT evidence, confidence
    INTO v_current_evidence, v_new_confidence
    FROM vkb_patterns
    WHERE id = p_pattern_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Pattern not found: %', p_pattern_id;
    END IF;

    -- Add session to evidence if not already present
    IF p_session_id IS NOT NULL AND NOT (p_session_id = ANY(v_current_evidence)) THEN
        v_current_evidence := v_current_evidence || p_session_id;
    END IF;

    -- Increase confidence (asymptotic to 1.0)
    v_new_confidence := LEAST(0.99, v_new_confidence + (1 - v_new_confidence) * 0.1);

    UPDATE vkb_patterns
    SET evidence = v_current_evidence,
        occurrence_count = occurrence_count + 1,
        confidence = v_new_confidence,
        last_observed = NOW(),
        status = CASE
            WHEN occurrence_count >= 5 AND status = 'emerging' THEN 'established'
            ELSE status
        END
    WHERE id = p_pattern_id;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- INSIGHT OPERATIONS
-- ============================================================================

-- Extract an insight from a session and create a belief
CREATE OR REPLACE FUNCTION insight_extract(
    p_session_id UUID,
    p_content TEXT,
    p_domain_path TEXT[] DEFAULT '{}',
    p_confidence JSONB DEFAULT '{"overall": 0.8}',
    p_entities JSONB DEFAULT '[]'
)
RETURNS UUID AS $$
DECLARE
    v_belief_id UUID;
    v_source_id UUID;
BEGIN
    -- Get source for this session
    SELECT id INTO v_source_id
    FROM sources
    WHERE session_id = p_session_id
    LIMIT 1;

    -- Create belief
    v_belief_id := belief_create_with_entities(
        p_content,
        p_confidence,
        p_domain_path,
        v_source_id,
        'conversation_extraction',
        NOW(),
        p_entities
    );

    -- Link to session
    INSERT INTO vkb_session_insights (session_id, belief_id, extraction_method)
    VALUES (p_session_id, v_belief_id, 'manual')
    ON CONFLICT DO NOTHING;

    RETURN v_belief_id;
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- EMBEDDING OPERATIONS
-- ============================================================================

-- Update embedding for a belief
CREATE OR REPLACE FUNCTION belief_set_embedding(
    p_belief_id UUID,
    p_embedding VECTOR(1536),
    p_embedding_type TEXT DEFAULT 'openai_text3_small'
)
RETURNS VOID AS $$
BEGIN
    UPDATE beliefs
    SET embedding = p_embedding,
        modified_at = NOW()
    WHERE id = p_belief_id;

    -- Track coverage
    INSERT INTO embedding_coverage (content_type, content_id, embedding_type_id)
    VALUES ('belief', p_belief_id, p_embedding_type)
    ON CONFLICT (content_type, content_id, embedding_type_id)
    DO UPDATE SET embedded_at = NOW();
END;
$$ LANGUAGE plpgsql;


-- Update embedding for an exchange
CREATE OR REPLACE FUNCTION exchange_set_embedding(
    p_exchange_id UUID,
    p_embedding VECTOR(1536),
    p_embedding_type TEXT DEFAULT 'openai_text3_small'
)
RETURNS VOID AS $$
BEGIN
    UPDATE vkb_exchanges
    SET embedding = p_embedding
    WHERE id = p_exchange_id;

    INSERT INTO embedding_coverage (content_type, content_id, embedding_type_id)
    VALUES ('exchange', p_exchange_id, p_embedding_type)
    ON CONFLICT (content_type, content_id, embedding_type_id)
    DO UPDATE SET embedded_at = NOW();
END;
$$ LANGUAGE plpgsql;


-- Get content needing embeddings
CREATE OR REPLACE FUNCTION embedding_backfill_queue(
    p_content_type TEXT,
    p_embedding_type TEXT DEFAULT 'openai_text3_small',
    p_limit INTEGER DEFAULT 100
)
RETURNS TABLE (
    content_id UUID,
    content TEXT
) AS $$
BEGIN
    IF p_content_type = 'belief' THEN
        RETURN QUERY
        SELECT b.id, b.content
        FROM beliefs b
        LEFT JOIN embedding_coverage ec
            ON ec.content_type = 'belief'
            AND ec.content_id = b.id
            AND ec.embedding_type_id = p_embedding_type
        WHERE ec.content_id IS NULL
          AND b.status = 'active'
        ORDER BY b.created_at DESC
        LIMIT p_limit;
    ELSIF p_content_type = 'exchange' THEN
        RETURN QUERY
        SELECT e.id, e.content
        FROM vkb_exchanges e
        LEFT JOIN embedding_coverage ec
            ON ec.content_type = 'exchange'
            AND ec.content_id = e.id
            AND ec.embedding_type_id = p_embedding_type
        WHERE ec.content_id IS NULL
        ORDER BY e.created_at DESC
        LIMIT p_limit;
    ELSIF p_content_type = 'pattern' THEN
        RETURN QUERY
        SELECT p.id, p.description
        FROM vkb_patterns p
        LEFT JOIN embedding_coverage ec
            ON ec.content_type = 'pattern'
            AND ec.content_id = p.id
            AND ec.embedding_type_id = p_embedding_type
        WHERE ec.content_id IS NULL
        ORDER BY p.created_at DESC
        LIMIT p_limit;
    END IF;
END;
$$ LANGUAGE plpgsql;
