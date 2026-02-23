#!/usr/bin/env python3
"""Backfill embeddings using OpenAI after migration.

Re-embeds all articles and sources that are missing embeddings after
the migration from local bge-small (384-dim) to OpenAI (1536-dim).

Prerequisites:
- Database columns migrated to vector(1536)
- OPENAI_API_KEY set in environment or .env file
- VALENCE_EMBEDDING_PROVIDER=openai
- VALENCE_EMBEDDING_MODEL=text-embedding-3-small
- VALENCE_EMBEDDING_DIMS=1536
"""

import asyncio
import os
import sys
from pathlib import Path

# Load .env if present
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    print(f"Loading environment from {env_file}")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                # Don't override existing env vars
                if key not in os.environ:
                    os.environ[key] = value

# Set up database connection
os.environ.setdefault('ORO_DB_PORT', '5433')
os.environ.setdefault('ORO_DB_HOST', '127.0.0.1')
os.environ.setdefault('ORO_DB_USER', 'valence')
os.environ.setdefault('ORO_DB_PASSWORD', 'valence')
os.environ.setdefault('ORO_DB_NAME', 'valence')

# Verify OpenAI configuration
required_vars = {
    'OPENAI_API_KEY': 'OpenAI API key',
    'VALENCE_EMBEDDING_PROVIDER': 'Embedding provider (should be "openai")',
    'VALENCE_EMBEDDING_MODEL': 'Embedding model (should be "text-embedding-3-small")',
    'VALENCE_EMBEDDING_DIMS': 'Embedding dimensions (should be "1536")',
}

print("=" * 70)
print("OPENAI EMBEDDING BACKFILL")
print("=" * 70)
print("\nConfiguration check:")

missing_vars = []
for var, desc in required_vars.items():
    value = os.environ.get(var)
    if not value or value.startswith('sk-YOUR-'):
        missing_vars.append((var, desc))
        print(f"  ✗ {var}: NOT SET ({desc})")
    else:
        # Mask API key
        if 'KEY' in var:
            display_value = value[:10] + '...' if len(value) > 10 else '***'
        else:
            display_value = value
        print(f"  ✓ {var}: {display_value}")

if missing_vars:
    print("\n❌ Configuration incomplete!")
    print("\nMissing variables:")
    for var, desc in missing_vars:
        print(f"  - {var}: {desc}")
    print("\nPlease update ~/projects/valence/.env with:")
    print("  OPENAI_API_KEY=sk-your-actual-key-here")
    print("  VALENCE_EMBEDDING_PROVIDER=openai")
    print("  VALENCE_EMBEDDING_MODEL=text-embedding-3-small")
    print("  VALENCE_EMBEDDING_DIMS=1536")
    print("\nGet your OpenAI API key from: https://platform.openai.com/api-keys")
    sys.exit(1)

print("\n✓ Configuration valid, proceeding with backfill...")

# Import after env setup
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from valence.core.deferred_embeddings import compute_missing_embeddings
from valence.lib.our_db import get_cursor


async def count_missing(table: str) -> tuple[int, int]:
    """Count total rows and rows missing embeddings."""
    def _sync_count():
        with get_cursor() as cur:
            cur.execute(f"SELECT count(*) as total FROM {table}")
            total = cur.fetchone()['total']
            
            cur.execute(f"SELECT count(*) as missing FROM {table} WHERE embedding IS NULL")
            missing = cur.fetchone()['missing']
            
            return total, missing
    
    return await asyncio.to_thread(_sync_count)


async def backfill_table(table: str, batch_size: int = 50) -> int:
    """Backfill embeddings for a table in batches."""
    print(f"\n{'='*70}")
    print(f"Backfilling {table}")
    print('='*70)
    
    total, missing = await count_missing(table)
    print(f"Total rows: {total}")
    print(f"Missing embeddings: {missing}")
    
    if missing == 0:
        print("✓ No embeddings to compute")
        return 0
    
    total_computed = 0
    batch_num = 1
    
    while True:
        print(f"\nBatch {batch_num}: Computing up to {batch_size} embeddings...")
        
        computed = await compute_missing_embeddings(table, batch_size=batch_size)
        total_computed += computed
        
        print(f"  ✓ Computed {computed} embeddings")
        print(f"  Progress: {total_computed}/{missing} ({100*total_computed/missing:.1f}%)")
        
        if computed < batch_size:
            # Done with this table
            break
        
        batch_num += 1
        
        # Brief pause to avoid rate limits
        await asyncio.sleep(0.5)
    
    # Verify
    _, still_missing = await count_missing(table)
    print(f"\n✓ Backfill complete for {table}")
    print(f"  Computed: {total_computed}")
    print(f"  Remaining: {still_missing}")
    
    return total_computed


async def verify_embeddings():
    """Verify embedding dimensions and sample similarity search."""
    print(f"\n{'='*70}")
    print("Verification")
    print('='*70)
    
    def _sync_verify():
        with get_cursor() as cur:
            # Check dimensions
            cur.execute("""
                SELECT array_length(embedding::real[], 1) as dims 
                FROM articles 
                WHERE embedding IS NOT NULL 
                LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                dims = row['dims']
                print(f"\n✓ Embedding dimensions: {dims}")
                if dims != 1536:
                    print(f"  ⚠️  Warning: Expected 1536, got {dims}")
            else:
                print("\n✗ No embeddings found in articles table")
            
            # Sample similarity search
            cur.execute("""
                SELECT id, title, content 
                FROM articles 
                WHERE embedding IS NOT NULL 
                LIMIT 1
            """)
            sample = cur.fetchone()
            
            if sample:
                print(f"\n✓ Sample article: {sample['title']}")
                
                # Simple cosine similarity test
                cur.execute("""
                    SELECT 
                        id, 
                        title, 
                        1 - (embedding <=> (SELECT embedding FROM articles WHERE id = %s)) as similarity
                    FROM articles 
                    WHERE id != %s 
                    AND embedding IS NOT NULL
                    ORDER BY embedding <=> (SELECT embedding FROM articles WHERE id = %s)
                    LIMIT 3
                """, (sample['id'], sample['id'], sample['id']))
                
                similar = cur.fetchall()
                if similar:
                    print("\n✓ Similar articles (cosine similarity):")
                    for art in similar:
                        print(f"  - {art['title']}: {art['similarity']:.3f}")
    
    await asyncio.to_thread(_sync_verify)


async def main():
    """Main backfill workflow."""
    # Backfill both tables
    articles_count = await backfill_table('articles', batch_size=50)
    sources_count = await backfill_table('sources', batch_size=100)
    
    # Verify
    await verify_embeddings()
    
    # Summary
    print(f"\n{'='*70}")
    print("BACKFILL COMPLETE")
    print('='*70)
    print(f"Articles re-embedded: {articles_count}")
    print(f"Sources re-embedded: {sources_count}")
    print(f"Total: {articles_count + sources_count}")
    print('='*70)


if __name__ == "__main__":
    asyncio.run(main())
