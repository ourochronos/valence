"""Migration utilities for converting old visibility to SharePolicy.

Provides functions to migrate existing beliefs from the 3-level visibility
enum (private/federated/public) to the new SharePolicy schema.
"""

from typing import Dict, Any, Optional
import logging

from .types import SharePolicy, ShareLevel, EnforcementType, PropagationRules

logger = logging.getLogger(__name__)


def migrate_visibility(old_visibility: str) -> Dict[str, Any]:
    """Convert old visibility enum to new SharePolicy.
    
    Mapping:
        - private/PRIVATE → ShareLevel.PRIVATE with CRYPTOGRAPHIC enforcement
        - federated/FEDERATED → ShareLevel.BOUNDED with CRYPTOGRAPHIC enforcement
          and propagation rules for federation domain
        - public/PUBLIC → ShareLevel.PUBLIC with HONOR enforcement
        - Unknown values → Default to PRIVATE with CRYPTOGRAPHIC
    
    Args:
        old_visibility: The old visibility string (private/federated/public)
    
    Returns:
        Dictionary representation of SharePolicy for JSON storage
    """
    mapping = {
        "private": SharePolicy(
            level=ShareLevel.PRIVATE,
            enforcement=EnforcementType.CRYPTOGRAPHIC
        ),
        "PRIVATE": SharePolicy(
            level=ShareLevel.PRIVATE,
            enforcement=EnforcementType.CRYPTOGRAPHIC
        ),
        "federated": SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.CRYPTOGRAPHIC,
            propagation=PropagationRules(
                allowed_domains=["federation"]  # Current federation scope
            )
        ),
        "FEDERATED": SharePolicy(
            level=ShareLevel.BOUNDED,
            enforcement=EnforcementType.CRYPTOGRAPHIC,
            propagation=PropagationRules(
                allowed_domains=["federation"]
            )
        ),
        "public": SharePolicy(
            level=ShareLevel.PUBLIC,
            enforcement=EnforcementType.HONOR
        ),
        "PUBLIC": SharePolicy(
            level=ShareLevel.PUBLIC,
            enforcement=EnforcementType.HONOR
        ),
    }
    
    policy = mapping.get(old_visibility)
    if not policy:
        # Default to private for unknown values
        logger.warning(f"Unknown visibility '{old_visibility}', defaulting to private")
        policy = SharePolicy(
            level=ShareLevel.PRIVATE,
            enforcement=EnforcementType.CRYPTOGRAPHIC
        )
    
    return policy.to_dict()


def get_share_policy_json(visibility: str) -> str:
    """Get the JSON string representation for a visibility level.
    
    Used for SQL CASE statements to avoid Python serialization overhead.
    
    Args:
        visibility: The old visibility string
    
    Returns:
        JSON string suitable for PostgreSQL jsonb column
    """
    json_mapping = {
        "private": '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}',
        "PRIVATE": '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}',
        "federated": '{"level": "bounded", "enforcement": "cryptographic", "recipients": null, "propagation": {"max_hops": null, "allowed_domains": ["federation"], "min_trust_to_receive": null, "strip_on_forward": null, "expires_at": null}}',
        "FEDERATED": '{"level": "bounded", "enforcement": "cryptographic", "recipients": null, "propagation": {"max_hops": null, "allowed_domains": ["federation"], "min_trust_to_receive": null, "strip_on_forward": null, "expires_at": null}}',
        "public": '{"level": "public", "enforcement": "honor", "recipients": null, "propagation": null}',
        "PUBLIC": '{"level": "public", "enforcement": "honor", "recipients": null, "propagation": null}',
    }
    return json_mapping.get(
        visibility,
        '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}'
    )


async def migrate_all_beliefs(db_connection) -> Dict[str, Any]:
    """Migrate all beliefs to new share_policy format.
    
    Performs a batch UPDATE using PostgreSQL CASE statement for efficiency.
    Only updates beliefs where share_policy IS NULL.
    
    Args:
        db_connection: asyncpg database connection
    
    Returns:
        Dictionary with migration statistics:
            - total: Total number of beliefs in database
            - needed_migration: Number that needed migration
            - migrated: Number successfully migrated
    """
    # Count before migration
    total = await db_connection.fetchval("SELECT COUNT(*) FROM beliefs")
    needs_migration = await db_connection.fetchval(
        "SELECT COUNT(*) FROM beliefs WHERE share_policy IS NULL"
    )
    
    if needs_migration == 0:
        logger.info("No beliefs need migration")
        return {
            "total": total,
            "needed_migration": 0,
            "migrated": 0
        }
    
    # Batch migrate using CASE statement
    # This is more efficient than row-by-row updates
    await db_connection.execute("""
        UPDATE beliefs 
        SET share_policy = CASE visibility
            WHEN 'private' THEN '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}'::jsonb
            WHEN 'PRIVATE' THEN '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}'::jsonb
            WHEN 'federated' THEN '{"level": "bounded", "enforcement": "cryptographic", "recipients": null, "propagation": {"max_hops": null, "allowed_domains": ["federation"], "min_trust_to_receive": null, "strip_on_forward": null, "expires_at": null}}'::jsonb
            WHEN 'FEDERATED' THEN '{"level": "bounded", "enforcement": "cryptographic", "recipients": null, "propagation": {"max_hops": null, "allowed_domains": ["federation"], "min_trust_to_receive": null, "strip_on_forward": null, "expires_at": null}}'::jsonb
            WHEN 'public' THEN '{"level": "public", "enforcement": "honor", "recipients": null, "propagation": null}'::jsonb
            WHEN 'PUBLIC' THEN '{"level": "public", "enforcement": "honor", "recipients": null, "propagation": null}'::jsonb
            ELSE '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}'::jsonb
        END
        WHERE share_policy IS NULL
    """)
    
    # Count after migration
    migrated = await db_connection.fetchval(
        "SELECT COUNT(*) FROM beliefs WHERE share_policy IS NOT NULL"
    )
    
    logger.info(f"Migration complete: {needs_migration} beliefs migrated")
    
    return {
        "total": total,
        "needed_migration": needs_migration,
        "migrated": migrated
    }


def migrate_all_beliefs_sync(db_connection) -> Dict[str, Any]:
    """Synchronous version of migrate_all_beliefs for psycopg2.
    
    Args:
        db_connection: psycopg2 database connection
    
    Returns:
        Dictionary with migration statistics
    """
    cur = db_connection.cursor()
    
    try:
        # Count before migration
        cur.execute("SELECT COUNT(*) FROM beliefs")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM beliefs WHERE share_policy IS NULL")
        needs_migration = cur.fetchone()[0]
        
        if needs_migration == 0:
            logger.info("No beliefs need migration")
            return {
                "total": total,
                "needed_migration": 0,
                "migrated": 0
            }
        
        # Batch migrate
        cur.execute("""
            UPDATE beliefs 
            SET share_policy = CASE visibility
                WHEN 'private' THEN '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}'::jsonb
                WHEN 'PRIVATE' THEN '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}'::jsonb
                WHEN 'federated' THEN '{"level": "bounded", "enforcement": "cryptographic", "recipients": null, "propagation": {"max_hops": null, "allowed_domains": ["federation"], "min_trust_to_receive": null, "strip_on_forward": null, "expires_at": null}}'::jsonb
                WHEN 'FEDERATED' THEN '{"level": "bounded", "enforcement": "cryptographic", "recipients": null, "propagation": {"max_hops": null, "allowed_domains": ["federation"], "min_trust_to_receive": null, "strip_on_forward": null, "expires_at": null}}'::jsonb
                WHEN 'public' THEN '{"level": "public", "enforcement": "honor", "recipients": null, "propagation": null}'::jsonb
                WHEN 'PUBLIC' THEN '{"level": "public", "enforcement": "honor", "recipients": null, "propagation": null}'::jsonb
                ELSE '{"level": "private", "enforcement": "cryptographic", "recipients": null, "propagation": null}'::jsonb
            END
            WHERE share_policy IS NULL
        """)
        
        db_connection.commit()
        
        # Count after migration
        cur.execute("SELECT COUNT(*) FROM beliefs WHERE share_policy IS NOT NULL")
        migrated = cur.fetchone()[0]
        
        logger.info(f"Migration complete: {needs_migration} beliefs migrated")
        
        return {
            "total": total,
            "needed_migration": needs_migration,
            "migrated": migrated
        }
    finally:
        cur.close()
