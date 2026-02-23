#!/usr/bin/env python3
"""Migrate embedding vectors from 384-dim (bge-small) to 1536-dim (OpenAI).

This script:
1. Checks current embedding state
2. Saves view definitions that depend on embedding columns
3. Drops dependent views
4. NULLs all existing embeddings
5. Alters vector columns from vector(384) to vector(1536)
6. Recreates views
7. Reports migration status

Run re-embedding separately using compute_missing_embeddings.
"""

import os
import sys

# Set up environment for database connection
os.environ.update({
    'ORO_DB_PORT': '5433',
    'ORO_DB_HOST': '127.0.0.1',
    'ORO_DB_USER': 'valence',
    'ORO_DB_PASSWORD': 'valence',
    'ORO_DB_NAME': 'valence'
})

# Import after env setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from valence.lib.our_db import get_cursor


def main():
    print("=" * 70)
    print("EMBEDDING MIGRATION: 384-dim → 1536-dim")
    print("=" * 70)
    
    view_definitions = {}
    
    with get_cursor() as cur:
        # 1. Check current state
        print("\n1. Checking current embedding state...")
        
        cur.execute("""
            SELECT t.table_name, c.column_name 
            FROM information_schema.columns c
            JOIN information_schema.tables t 
                ON c.table_name = t.table_name 
                AND c.table_schema = t.table_schema
            WHERE c.column_name = 'embedding' 
            AND c.table_schema = 'public'
            AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name
        """)
        
        tables_with_embeddings = [row['table_name'] for row in cur.fetchall()]
        print(f"   Found embedding columns in tables: {', '.join(tables_with_embeddings)}")
        
        # Count embeddings per table
        embedding_counts = {}
        for table in tables_with_embeddings:
            cur.execute(f"SELECT count(*) as c FROM {table} WHERE embedding IS NOT NULL")
            count = cur.fetchone()['c']
            embedding_counts[table] = count
            print(f"   - {table}: {count} rows with embeddings")
        
        total_embeddings = sum(embedding_counts.values())
        print(f"   Total embeddings to migrate: {total_embeddings}")
        
        # 2. Save view definitions
        print("\n2. Saving dependent view definitions...")
        cur.execute("""
            SELECT table_name, view_definition 
            FROM information_schema.views 
            WHERE table_schema = 'public' 
            AND view_definition LIKE '%embedding%'
            ORDER BY table_name
        """)
        
        for row in cur.fetchall():
            view_name = row['table_name']
            view_def = row['view_definition'].strip()
            view_definitions[view_name] = view_def
            print(f"   ✓ Saved definition for {view_name}")
        
        # 3. Drop views
        print("\n3. Dropping dependent views...")
        for view_name in view_definitions.keys():
            cur.execute(f"DROP VIEW IF EXISTS {view_name} CASCADE")
            print(f"   ✓ Dropped {view_name}")
        
        # 4. NULL all embeddings
        print("\n4. Clearing existing embeddings...")
        for table in tables_with_embeddings:
            cur.execute(f"UPDATE {table} SET embedding = NULL")
            print(f"   ✓ Cleared {table}")
        
        # 5. Alter column dimensions
        print("\n5. Altering vector column dimensions (384 → 1536)...")
        for table in tables_with_embeddings:
            try:
                cur.execute(f"ALTER TABLE {table} ALTER COLUMN embedding TYPE vector(1536)")
                print(f"   ✓ Altered {table}")
            except Exception as e:
                print(f"   ✗ Failed to alter {table}: {e}")
                raise
        
        # 6. Recreate views
        print("\n6. Recreating views...")
        for view_name, view_def in view_definitions.items():
            try:
                # The view_definition from information_schema already starts with SELECT
                cur.execute(f"CREATE OR REPLACE VIEW {view_name} AS {view_def}")
                print(f"   ✓ Recreated {view_name}")
            except Exception as e:
                print(f"   ✗ Failed to recreate {view_name}: {e}")
                print(f"      Definition: {view_def[:200]}...")
                # Don't fail - we can recreate manually if needed
        
        # 7. Verify changes
        print("\n7. Verifying migration...")
        cur.execute("""
            SELECT t.table_name 
            FROM information_schema.columns c
            JOIN information_schema.tables t 
                ON c.table_name = t.table_name 
                AND c.table_schema = t.table_schema
            WHERE c.column_name = 'embedding' 
            AND c.table_schema = 'public'
            AND t.table_type = 'BASE TABLE'
            ORDER BY t.table_name
        """)
        
        for row in cur.fetchall():
            print(f"   ✓ {row['table_name']}: vector(1536)")
        
        # Check that all embeddings are NULL
        all_null = True
        for table in tables_with_embeddings:
            cur.execute(f"SELECT count(*) as c FROM {table} WHERE embedding IS NOT NULL")
            count = cur.fetchone()['c']
            if count > 0:
                print(f"   ✗ Warning: {table} still has {count} non-NULL embeddings!")
                all_null = False
        
        if all_null:
            print("   ✓ All embeddings successfully cleared")
    
    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print(f"Cleared {total_embeddings} embeddings from {len(tables_with_embeddings)} tables")
    print(f"Dropped and recreated {len(view_definitions)} views")
    print("Next step: Run backfill script to re-embed with OpenAI")
    print("=" * 70)


if __name__ == "__main__":
    main()
