"""Belief tool implementations."""

from __future__ import annotations

from typing import Any

from . import _common
from ._common import Belief, DimensionalConfidence, Entity, Tension, _validate_enum, hashlib, json, logger


def _log_retrievals(beliefs: list[dict], query: str, tool_name: str) -> None:
    """Log belief retrievals for the feedback loop. Non-fatal on error."""
    if not beliefs:
        return
    try:
        with _common.get_cursor() as cur:
            for b in beliefs[:20]:  # Cap logging to top 20
                cur.execute(
                    """
                    INSERT INTO belief_retrievals (belief_id, query_text, tool_name, final_score)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (b.get("id"), query[:500] if query else None, tool_name, b.get("final_score")),
                )
    except Exception:
        pass  # Retrieval logging is never fatal


def belief_query(
    query: str,
    domain_filter: list[str] | None = None,
    entity_id: str | None = None,
    include_superseded: bool = False,
    include_revoked: bool = False,
    include_archived: bool = False,
    include_expired: bool = False,
    limit: int = 20,
    user_did: str | None = None,
    ranking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Search beliefs with revocation, archival, and temporal validity filtering."""
    # Audit log when accessing revoked content
    if include_revoked:
        logger.info(f"Query includes revoked content: user={user_did or 'unknown'}, query={query[:100]}{'...' if len(query) > 100 else ''}")

    with _common.get_cursor() as cur:
        sql = """
            SELECT b.*, ts_rank(b.content_tsv, websearch_to_tsquery('english', %s)) as relevance
            FROM beliefs b
            WHERE b.content_tsv @@ websearch_to_tsquery('english', %s)
        """
        params: list[Any] = [query, query]

        if not include_superseded:
            sql += " AND b.superseded_by_id IS NULL"
            if include_archived:
                sql += " AND b.status IN ('active', 'archived')"
            else:
                sql += " AND b.status = 'active'"

        # Filter by temporal validity window
        if not include_expired:
            sql += " AND (b.valid_from IS NULL OR b.valid_from <= NOW())"
            sql += " AND (b.valid_until IS NULL OR b.valid_until >= NOW())"

        # Filter out beliefs with revoked consent chains by default
        if not include_revoked:
            sql += """
                AND b.id NOT IN (
                    SELECT DISTINCT cc.belief_id
                    FROM consent_chains cc
                    WHERE cc.revoked = true
                )
            """

        if domain_filter:
            sql += " AND b.domain_path && %s"
            params.append(domain_filter)

        if entity_id:
            sql += " AND EXISTS (SELECT 1 FROM belief_entities be WHERE be.belief_id = b.id AND be.entity_id = %s)"
            params.append(entity_id)

        sql += " ORDER BY relevance DESC LIMIT %s"
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()

        beliefs = []
        # Normalize relevance scores to 0-1 for ranking
        max_relevance = max((float(row.get("relevance", 0)) for row in rows), default=1.0) or 1.0
        for row in rows:
            belief = Belief.from_row(dict(row))
            belief_dict = belief.to_dict()
            raw_relevance = float(row.get("relevance", 0))
            belief_dict["relevance_score"] = raw_relevance
            belief_dict["similarity"] = raw_relevance / max_relevance  # Normalized for ranking
            beliefs.append(belief_dict)

        # Apply multi-signal ranking if requested or by default
        if ranking or beliefs:
            from ...core.ranking import multi_signal_rank

            rank_params = ranking or {}
            beliefs = multi_signal_rank(
                beliefs,
                semantic_weight=rank_params.get("semantic_weight", 0.50),
                confidence_weight=rank_params.get("confidence_weight", 0.35),
                recency_weight=rank_params.get("recency_weight", 0.15),
                explain=rank_params.get("explain", False),
            )

        _log_retrievals(beliefs, query, "belief_query")

        return {
            "success": True,
            "beliefs": beliefs,
            "total_count": len(beliefs),
            "include_revoked": include_revoked,
            "include_archived": include_archived,
        }


def _content_hash(content: str) -> str:
    """Compute deterministic hash for belief content deduplication."""
    return hashlib.sha256(content.strip().lower().encode()).hexdigest()


def _reinforce_belief(cur: Any, belief_id: Any, confidence_json: dict, source_ref: str | None = None) -> dict[str, Any]:
    """Reinforce an existing belief: bump corroboration count and update confidence."""
    from ...core.curation import corroboration_confidence

    cur.execute("SELECT COUNT(*) as cnt FROM belief_corroborations WHERE belief_id = %s", (belief_id,))
    count = cur.fetchone()["cnt"]
    cur.execute(
        "INSERT INTO belief_corroborations (belief_id, source_type, similarity_score) VALUES (%s, 'session', 1.0)",
        (belief_id,),
    )
    new_count = count + 1
    new_overall = corroboration_confidence(new_count)
    confidence_json["overall"] = max(confidence_json.get("overall", 0.5), new_overall)
    confidence_json["corroboration"] = new_overall
    cur.execute(
        "UPDATE beliefs SET confidence = %s, modified_at = NOW() WHERE id = %s RETURNING *",
        (json.dumps(confidence_json), belief_id),
    )
    updated_row = cur.fetchone()
    belief = Belief.from_row(dict(updated_row))
    return {
        "success": True, "deduplicated": True, "action": "reinforced",
        "corroboration_count": new_count, "belief": belief.to_dict(),
    }


def belief_create(
    content: str,
    confidence: dict[str, Any] | None = None,
    domain_path: list[str] | None = None,
    source_type: str | None = None,
    source_ref: str | None = None,
    opt_out_federation: bool = False,
    entities: list[dict[str, str]] | None = None,
    visibility: str = "private",
    sharing_intent: str | None = None,
) -> dict[str, Any]:
    """Create a new belief, or reinforce an existing one if duplicate detected.

    Dedup checks: (1) exact content hash, (2) cosine > 0.90 if embeddings available.
    Duplicates get reinforced (corroboration bump) instead of creating a new belief.
    """
    # --- Input validation ---
    valid_visibilities = ["private", "federated", "public"]
    if err := _validate_enum(visibility, valid_visibilities, "visibility"):
        return err

    valid_intents = ["know_me", "work_with_me", "learn_from_me", "use_this"]
    if sharing_intent:
        if err := _validate_enum(sharing_intent, valid_intents, "sharing_intent"):
            return err
        if sharing_intent == "know_me":
            return {"success": False, "error": "know_me intent requires a recipient — use belief_share instead"}

    confidence_obj = DimensionalConfidence.from_dict(confidence or {"overall": 0.7})
    content_hash_val = _content_hash(content)

    with _common.get_cursor() as cur:
        # --- Dedup check: exact content hash match ---
        cur.execute(
            "SELECT id, confidence FROM beliefs WHERE content_hash = %s AND status = 'active' AND superseded_by_id IS NULL",
            (content_hash_val,),
        )
        exact_match = cur.fetchone()
        if exact_match:
            existing_conf = exact_match["confidence"] if isinstance(exact_match["confidence"], dict) else json.loads(exact_match["confidence"])
            return _reinforce_belief(cur, exact_match["id"], existing_conf, source_ref)

        # --- Dedup check: fuzzy semantic match (cosine > 0.90) ---
        embedding_str = None  # Reused for inline storage if no duplicate found
        try:
            from our_embeddings.service import generate_embedding, vector_to_pgvector

            query_vector = generate_embedding(content)
            embedding_str = vector_to_pgvector(query_vector)

            cur.execute(
                """SELECT id, confidence, 1 - (embedding <=> %s::vector) as similarity
                FROM beliefs
                WHERE embedding IS NOT NULL AND status = 'active' AND superseded_by_id IS NULL
                AND 1 - (embedding <=> %s::vector) > 0.90
                ORDER BY embedding <=> %s::vector
                LIMIT 1""",
                (embedding_str, embedding_str, embedding_str),
            )
            fuzzy_match = cur.fetchone()
            if fuzzy_match:
                existing_conf = fuzzy_match["confidence"] if isinstance(fuzzy_match["confidence"], dict) else json.loads(fuzzy_match["confidence"])
                return _reinforce_belief(cur, fuzzy_match["id"], existing_conf, source_ref)
        except Exception:
            pass  # Embedding unavailable — skip fuzzy check

        # --- No duplicate found: create new belief ---
        source_id = None
        if source_type:
            cur.execute(
                "INSERT INTO sources (type, url) VALUES (%s, %s) RETURNING id",
                (source_type, source_ref),
            )
            source_id = cur.fetchone()["id"]

        # Build share_policy from sharing_intent if provided
        share_policy_json = None
        if sharing_intent:
            from our_privacy.types import IntentConfig, SharingIntent

            try:
                intent_config = IntentConfig(intent=SharingIntent(sharing_intent))
                share_policy_json = json.dumps(intent_config.to_dict())
            except ValueError as e:
                return {"success": False, "error": f"Invalid sharing intent configuration: {e}"}

        cur.execute(
            """
            INSERT INTO beliefs (content, confidence, domain_path, source_id, opt_out_federation, content_hash, visibility, share_policy, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            RETURNING *
            """,
            (
                content,
                json.dumps(confidence_obj.to_dict()),
                domain_path or [],
                source_id,
                opt_out_federation,
                content_hash_val,
                visibility,
                share_policy_json,
                embedding_str,
            ),
        )
        belief_row = cur.fetchone()
        belief_id = belief_row["id"]

        # Link entities
        if entities:
            for entity in entities:
                cur.execute(
                    """
                    INSERT INTO entities (name, type)
                    VALUES (%s, %s)
                    ON CONFLICT (LOWER(name), type) WHERE canonical_id IS NULL
                    DO UPDATE SET modified_at = NOW()
                    RETURNING id
                    """,
                    (entity["name"], entity.get("type", "concept")),
                )
                entity_id = cur.fetchone()["id"]

                cur.execute(
                    """
                    INSERT INTO belief_entities (belief_id, entity_id, role)
                    VALUES (%s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (belief_id, entity_id, entity.get("role", "subject")),
                )

        belief = Belief.from_row(dict(belief_row))
        return {
            "success": True,
            "belief": belief.to_dict(),
        }


def belief_supersede(
    old_belief_id: str,
    new_content: str,
    reason: str,
    confidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Supersede an existing belief."""
    with _common.get_cursor() as cur:
        # Get old belief
        cur.execute("SELECT * FROM beliefs WHERE id = %s", (old_belief_id,))
        old_row = cur.fetchone()
        if not old_row:
            return {"success": False, "error": f"Belief not found: {old_belief_id}"}

        old_belief = Belief.from_row(dict(old_row))

        # Determine new confidence
        new_confidence = DimensionalConfidence.from_dict(confidence or old_belief.confidence.to_dict())

        # Generate embedding for the new belief
        new_embedding_str = None
        try:
            from our_embeddings.service import generate_embedding, vector_to_pgvector

            new_embedding_str = vector_to_pgvector(generate_embedding(new_content))
        except Exception:
            pass  # Embedding unavailable — will be backfilled later

        # Create new belief
        cur.execute(
            """
            INSERT INTO beliefs (content, confidence, domain_path, source_id, extraction_method, supersedes_id, valid_from, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, NOW(), %s::vector)
            RETURNING *
            """,
            (
                new_content,
                json.dumps(new_confidence.to_dict()),
                old_belief.domain_path,
                str(old_belief.source_id) if old_belief.source_id else None,
                f"supersession: {reason}",
                old_belief_id,
                new_embedding_str,
            ),
        )
        new_row = cur.fetchone()
        new_belief_id = new_row["id"]

        # Update old belief
        cur.execute(
            """
            UPDATE beliefs
            SET status = 'superseded', superseded_by_id = %s, valid_until = NOW(), modified_at = NOW()
            WHERE id = %s
            """,
            (new_belief_id, old_belief_id),
        )

        # Copy entity links
        cur.execute(
            """
            INSERT INTO belief_entities (belief_id, entity_id, role)
            SELECT %s, entity_id, role FROM belief_entities WHERE belief_id = %s
            """,
            (new_belief_id, old_belief_id),
        )

        new_belief = Belief.from_row(dict(new_row))
        return {
            "success": True,
            "old_belief_id": old_belief_id,
            "new_belief": new_belief.to_dict(),
            "reason": reason,
        }


def belief_get(
    belief_id: str,
    include_history: bool = False,
    include_tensions: bool = False,
) -> dict[str, Any]:
    """Get a belief by ID."""
    with _common.get_cursor() as cur:
        cur.execute("SELECT * FROM beliefs WHERE id = %s", (belief_id,))
        row = cur.fetchone()
        if not row:
            return {"success": False, "error": f"Belief not found: {belief_id}"}

        belief = Belief.from_row(dict(row))
        result: dict[str, Any] = {
            "success": True,
            "belief": belief.to_dict(),
        }

        # Load source
        if belief.source_id:
            cur.execute("SELECT * FROM sources WHERE id = %s", (str(belief.source_id),))
            source_row = cur.fetchone()
            if source_row:
                result["belief"]["source"] = dict(source_row)

        # Load entities
        cur.execute(
            "SELECT e.*, be.role FROM entities e "
            "JOIN belief_entities be ON e.id = be.entity_id WHERE be.belief_id = %s",
            (belief_id,),
        )
        entity_rows = cur.fetchall()
        result["belief"]["entities"] = [{"entity": Entity.from_row(dict(r)).to_dict(), "role": r["role"]} for r in entity_rows]

        if include_history:
            history: list[dict[str, Any]] = []
            current_id: str | None = belief_id
            while current_id:
                cur.execute("SELECT id, supersedes_id, created_at, extraction_method FROM beliefs WHERE id = %s", (current_id,))
                hist_row = cur.fetchone()
                if not hist_row:
                    break
                history.append({
                    "id": str(hist_row["id"]),
                    "created_at": hist_row["created_at"].isoformat(),
                    "reason": hist_row.get("extraction_method"),
                })
                current_id = str(hist_row["supersedes_id"]) if hist_row["supersedes_id"] else None
            result["history"] = list(reversed(history))

        if include_tensions:
            cur.execute(
                "SELECT * FROM tensions WHERE (belief_a_id = %s OR belief_b_id = %s) AND status != 'resolved'",
                (belief_id, belief_id),
            )
            tension_rows = cur.fetchall()
            result["tensions"] = [Tension.from_row(dict(r)).to_dict() for r in tension_rows]

        return result


def belief_search(
    query: str,
    min_similarity: float = 0.5,
    min_confidence: float | None = None,
    domain_filter: list[str] | None = None,
    include_archived: bool = False,
    limit: int = 10,
    ranking: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Semantic search for beliefs using embeddings."""
    try:
        from our_embeddings.service import generate_embedding, vector_to_pgvector
    except ImportError:
        return {"success": False, "error": "Embeddings service not available. Install openai package."}

    try:
        query_vector = generate_embedding(query)
        query_str = vector_to_pgvector(query_vector)
    except Exception as e:
        return {"success": False, "error": f"Failed to generate embedding: {str(e)}"}

    with _common.get_cursor() as cur:
        # Build query with similarity filter
        status_filter = "b.status IN ('active', 'archived')" if include_archived else "b.status = 'active'"
        sql = f"""
            SELECT b.*, 1 - (b.embedding <=> %s::vector) as similarity
            FROM beliefs b
            WHERE b.embedding IS NOT NULL
            AND {status_filter}
            AND 1 - (b.embedding <=> %s::vector) >= %s
        """
        params: list[Any] = [query_str, query_str, min_similarity]

        if min_confidence is not None:
            sql += " AND (b.confidence->>'overall')::numeric >= %s"
            params.append(min_confidence)

        if domain_filter:
            sql += " AND b.domain_path && %s"
            params.append(domain_filter)

        sql += " ORDER BY b.embedding <=> %s::vector LIMIT %s"
        params.extend([query_str, limit])

        cur.execute(sql, params)
        rows = cur.fetchall()

        beliefs = []
        for row in rows:
            belief = Belief.from_row(dict(row))
            belief_dict = belief.to_dict()
            belief_dict["similarity"] = float(row["similarity"])
            beliefs.append(belief_dict)

        # Apply multi-signal ranking (similarity already 0-1 from cosine)
        if ranking or beliefs:
            from ...core.ranking import multi_signal_rank

            rank_params = ranking or {}
            beliefs = multi_signal_rank(
                beliefs,
                semantic_weight=rank_params.get("semantic_weight", 0.50),
                confidence_weight=rank_params.get("confidence_weight", 0.35),
                recency_weight=rank_params.get("recency_weight", 0.15),
                explain=rank_params.get("explain", False),
            )

        _log_retrievals(beliefs, query, "belief_search")
        return {"success": True, "beliefs": beliefs, "total_count": len(beliefs), "query_embedded": True}
