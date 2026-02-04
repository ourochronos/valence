# Migration & Onboarding Extension

## Overview

Migration is adoption. The ability to import existing knowledge with intelligent confidence inference removes the "cold start" problem that kills most knowledge tools.

## Design Principles

### 1. Zero-Friction Import
- Point at a folder, get beliefs
- No manual tagging required (but available)
- Incremental — import what you want, when you want

### 2. Intelligent Inference
During import, we infer:
- **Domains**: From folder structure, tags, content
- **Confidence vectors**: From source type, age, structure
- **Temporal validity**: From file dates, content cues
- **Relationships**: From links, references, proximity

### 3. Non-Destructive
- Original files untouched
- Provenance preserved (link back to source)
- Can re-sync if source changes

### 4. Immediate Value
- Query works instantly post-import
- No "setup phase" before utility
- Review/refinement is optional enhancement

## Supported Formats (Priority Order)

### Phase 1: Common Knowledge Bases
| Format | Source | Complexity |
|--------|--------|------------|
| Markdown folders | Obsidian, plain files | Low |
| JSON/JSONL | Custom exports, APIs | Low |
| Notion export | Notion → Markdown/CSV | Medium |
| Roam JSON | Roam Research | Medium |
| Logseq EDN | Logseq | Medium |

### Phase 2: Documents & Notes
| Format | Source | Complexity |
|--------|--------|------------|
| Google Docs | Takeout export | Medium |
| Apple Notes | Export or API | Medium |
| Evernote ENEX | Evernote export | Medium |
| OneNote | Export packages | High |

### Phase 3: Communication Archives
| Format | Source | Complexity |
|--------|--------|------------|
| Discord export | DiscordChatExporter | Medium |
| Slack export | Workspace export | Medium |
| Email (mbox/eml) | Thunderbird, Takeout | High |
| iMessage | Chat exports | High |

### Phase 4: Web & Bookmarks
| Format | Source | Complexity |
|--------|--------|------------|
| Browser bookmarks | Chrome, Firefox JSON | Low |
| Pocket export | Pocket HTML | Low |
| Instapaper | CSV export | Low |
| Hypothes.is | API export | Medium |

## Confidence Inference Heuristics

### Source-Based Defaults

```yaml
# Base confidence by source type
personal_notes:
  source_reliability: 0.6  # Your own observations
  method_quality: 0.5      # Informal capture
  internal_consistency: 0.7
  temporal_freshness: from_date
  corroboration: 0.3       # Uncorroborated
  domain_applicability: 0.8

bookmarked_article:
  source_reliability: 0.5  # Unknown author
  method_quality: 0.6      # Published content
  internal_consistency: 0.6
  temporal_freshness: from_date
  corroboration: 0.4
  domain_applicability: 0.7

academic_paper:
  source_reliability: 0.8  # Peer review
  method_quality: 0.8      # Formal methodology
  internal_consistency: 0.8
  temporal_freshness: from_date
  corroboration: 0.7       # Citations
  domain_applicability: 0.9
```

### Content-Based Adjustments

- **Links to sources** → +0.1 source_reliability
- **Dated observations** → temporal validity extracted
- **"I think" / hedging language** → -0.1 overall
- **Citations/references** → +0.15 corroboration
- **Technical specificity** → +0.1 method_quality

### Structure-Based Signals

- **Folder depth** → domain hierarchy
- **File naming conventions** → domain hints
- **Tags/frontmatter** → explicit domains
- **Backlinks** → relationship inference

## Import Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│ Source File │────▶│   Parser     │────▶│  Normalizer   │
└─────────────┘     └──────────────┘     └───────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│   Belief    │◀────│  Confidence  │◀────│   Content     │
│   Store     │     │  Inference   │     │   Chunker     │
└─────────────┘     └──────────────┘     └───────────────┘
```

### 1. Parser
Format-specific extraction of:
- Content (text, structure)
- Metadata (dates, tags, authors)
- Relationships (links, hierarchy)

### 2. Normalizer
Convert to common intermediate format:
```typescript
interface ImportedContent {
  id: string;
  source_path: string;
  source_type: string;
  content: string;
  created_at?: Date;
  modified_at?: Date;
  tags?: string[];
  links?: string[];
  metadata?: Record<string, unknown>;
}
```

### 3. Content Chunker
Break into belief-sized units:
- Respect semantic boundaries (paragraphs, sections)
- Maintain context (headings, hierarchy)
- Link related chunks

### 4. Confidence Inference
Apply heuristics to generate initial vectors:
- Source-based defaults
- Content-based adjustments
- Relationship-based signals

### 5. Belief Store
Create beliefs with:
- Inferred confidence vectors
- Source provenance (path, type, date)
- Extracted domains
- Embeddings for retrieval

## CLI Interface

```bash
# Import a folder
valence import ./my-notes --format=markdown

# Import with domain prefix
valence import ./work-notes --domain=work

# Dry run (preview what would be imported)
valence import ./notes --dry-run

# Import specific file types
valence import ./docs --include="*.md,*.txt"

# Re-sync changed files
valence import ./notes --sync

# Import from specific source
valence import-notion ./notion-export
valence import-roam ./roam-export.json
valence import-obsidian ./vault
```

## Review Interface

Post-import review is optional but valuable:

```bash
# List recently imported beliefs
valence review --imported-after="2024-01-01"

# Review by domain
valence review --domain=work

# Bulk adjust confidence
valence review --low-confidence --adjust
```

### Review UI Concepts

- Show belief with inferred confidence
- One-click accept/adjust/delete
- Bulk operations for similar content
- Learning from adjustments (improve inference)

## Incremental Adoption

### Phase 1: Shadow Mode
- Import existing KB
- Query through Valence
- Original system unchanged
- Learn what works

### Phase 2: Hybrid
- New content in Valence
- Old content imported/synced
- Gradual transition

### Phase 3: Primary
- Valence is source of truth
- Old system archived or linked
- Full feature utilization

## Sync Strategies

### One-Time Import
- Snapshot of existing knowledge
- No ongoing relationship
- Simplest approach

### Periodic Sync
- Re-import changed files
- Update existing beliefs
- Handle deletions gracefully

### Live Sync (Future)
- Watch for changes
- Real-time import
- Bi-directional (Valence → source)

## Error Handling

### Graceful Degradation
- Unparseable files → skip with warning
- Missing metadata → use defaults
- Encoding issues → attempt recovery

### Import Report
```
Import Complete
───────────────
Files processed: 1,247
Beliefs created: 3,892
Skipped (binary): 45
Skipped (error): 3
Domains inferred: 12

Low confidence beliefs: 234 (consider review)
Duplicate candidates: 17 (consider merge)
```

## Privacy Considerations

- All processing local
- No content sent externally (except embeddings if using external API)
- Sensitive content detection (optional warning)
- Selective import (exclude patterns)

## Future: Learning from Corrections

Track when users adjust inferred confidence:
- Build corpus of corrections
- Improve inference heuristics
- Eventually: fine-tuned confidence model

This creates a virtuous cycle where migration improves over time.
