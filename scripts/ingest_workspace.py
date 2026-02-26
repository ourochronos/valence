#!/usr/bin/env python3
"""Batch ingest workspace files into Valence.

Ingests files from the OpenClaw workspace into Valence as sources.
Uses fingerprinting to skip already-ingested files (dedup by content hash).
Designed to run periodically (cron or manual) to keep Valence current.

Usage:
    python scripts/ingest_workspace.py [--workspace PATH] [--compile] [--dry-run]
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path

# Set DB config before imports
for key in ['HOST', 'PORT', 'NAME', 'USER', 'PASSWORD']:
    val = os.environ.get(f'VALENCE_DB_{key}') or {
        'HOST': '127.0.0.1', 'PORT': '5433', 'NAME': 'valence',
        'USER': 'valence', 'PASSWORD': 'valence'
    }[key]
    os.environ[f'VALENCE_DB_{key}'] = val

from valence.core import sources, compilation
from valence.core.inference import provider
from valence.core.backends.gemini_cli import create_gemini_backend


# File patterns to ingest
INGEST_PATTERNS = {
    # Workspace root files
    'workspace': {
        'glob': '*.md',
        'source_type': 'document',
        'recursive': False,
    },
    # Memory files
    'memory': {
        'glob': 'memory/*.md',
        'source_type': 'document',
        'recursive': False,
    },
    # Daily logs
    'daily_logs': {
        'glob': 'memory/2*.md',
        'source_type': 'conversation',
        'recursive': False,
    },
    # Knowledge files (non-recursive — top level only, subdirs are too large)
    'knowledge_top': {
        'glob': 'knowledge/*.md',
        'source_type': 'document',
        'recursive': False,
    },
}

# Files to skip
SKIP_FILES = {'BOOTSTRAP.md', '.DS_Store'}

# Max content size (10KB — larger files get truncated)
MAX_CONTENT = 10000


async def ingest_file(filepath: Path, source_type: str, dry_run: bool = False) -> dict:
    """Ingest a single file. Returns result dict."""
    title = filepath.name
    try:
        content = filepath.read_text(encoding='utf-8')
    except Exception as e:
        return {'file': str(filepath), 'status': 'error', 'error': str(e)}

    if not content.strip():
        return {'file': str(filepath), 'status': 'skipped', 'reason': 'empty'}

    if len(content) > MAX_CONTENT:
        content = content[:MAX_CONTENT] + f'\n\n[Truncated at {MAX_CONTENT} chars]'

    if dry_run:
        return {'file': str(filepath), 'status': 'dry_run', 'size': len(content)}

    result = await sources.ingest_source(
        content=content,
        source_type=source_type,
        title=title,
        metadata={'path': str(filepath), 'workspace': True}
    )

    if result.success:
        return {'file': str(filepath), 'status': 'ingested', 'id': result.data['id']}
    elif 'duplicate' in str(result.error).lower() or 'conflict' in str(result.error).lower():
        return {'file': str(filepath), 'status': 'exists', 'reason': 'duplicate fingerprint'}
    else:
        return {'file': str(filepath), 'status': 'error', 'error': result.error}


async def compile_ungrouped(limit: int = 5) -> list:
    """Find ungrouped sources and compile them into articles."""
    # Get sources that aren't linked to any article
    all_sources = await sources.list_sources(limit=200)
    sdata = all_sources.data if hasattr(all_sources, 'data') else all_sources

    # Group by rough topic using title keywords
    groups = {}
    for s in sdata:
        title = s.get('title', '').lower()
        # Simple grouping by first meaningful word
        for keyword in ['jane', 'identity', 'soul', 'infrastructure', 'model', 'spec', 'project', 'agent', 'valence', 'memory', 'chris']:
            if keyword in title:
                groups.setdefault(keyword, []).append(s)
                break
        else:
            groups.setdefault('misc', []).append(s)

    compiled = []
    for topic, topic_sources in list(groups.items())[:limit]:
        if len(topic_sources) < 2:
            continue
        source_ids = [s['id'] for s in topic_sources[:5]]
        result = await compilation.compile_article(
            source_ids=source_ids,
            title_hint=f'{topic.title()} Knowledge'
        )
        if result.success:
            compiled.append({
                'topic': topic,
                'title': result.data.get('title', 'Unknown'),
                'sources': len(source_ids),
            })
            print(f'  ✓ Compiled: {result.data.get("title")} ({len(source_ids)} sources)')
        else:
            print(f'  ✗ {topic}: {result.error}')

    return compiled


async def main():
    parser = argparse.ArgumentParser(description='Ingest workspace files into Valence')
    parser.add_argument('--workspace', default=os.path.expanduser('~/.openclaw/workspace'),
                        help='Workspace path')
    parser.add_argument('--compile', action='store_true',
                        help='Compile ungrouped sources into articles after ingestion')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be ingested without doing it')
    parser.add_argument('--model', default='gemini-2.5-flash',
                        help='Gemini model for compilation')
    args = parser.parse_args()

    workspace = Path(args.workspace)
    if not workspace.exists():
        print(f'Workspace not found: {workspace}')
        sys.exit(1)

    if not args.dry_run:
        provider.configure(create_gemini_backend(args.model))

    stats = {'ingested': 0, 'exists': 0, 'skipped': 0, 'errors': 0}

    for category, config in INGEST_PATTERNS.items():
        pattern = config['glob']
        source_type = config['source_type']

        files = sorted(workspace.glob(pattern))
        files = [f for f in files if f.is_file() and f.name not in SKIP_FILES]

        if not files:
            continue

        print(f'\n--- {category} ({len(files)} files) ---')
        for filepath in files:
            result = await ingest_file(filepath, source_type, args.dry_run)
            status = result['status']
            stats[status] = stats.get(status, 0) + 1

            icon = {'ingested': '✓', 'exists': '·', 'skipped': '-', 'error': '✗', 'dry_run': '?'}
            print(f'  {icon.get(status, "?")} {filepath.name} [{status}]')

    print(f'\n--- Summary ---')
    print(f'  Ingested: {stats.get("ingested", 0)}')
    print(f'  Already exists: {stats.get("exists", 0)}')
    print(f'  Skipped: {stats.get("skipped", 0)}')
    print(f'  Errors: {stats.get("errors", 0)}')

    if args.compile and not args.dry_run:
        print(f'\n--- Compiling articles ---')
        compiled = await compile_ungrouped()
        print(f'  Compiled {len(compiled)} articles')


if __name__ == '__main__':
    asyncio.run(main())
