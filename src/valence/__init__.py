# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Ourochronos Contributors

# See docs/deployment.md for setup and configuration guide.

"""Valence v2 - Knowledge system for AI agents.

Valence ingests information from diverse sources, compiles it into articles
through use-driven promotion, and maintains those articles as living documents
— always current, always traceable to their origins, always queryable.

Architecture:
  Sources (immutable, typed by origin)
    → Articles (compiled on demand, not eagerly)
    → Provenance (typed relationships: originates/confirms/supersedes/contradicts/contends)
    → Retrieval (ranked by relevance × confidence × freshness)

Key design principles (see docs/architecture.md):
  - Lazy by design: embedding, compilation, and relationship analysis are
    triggered by use, not ingestion.
  - One-level atomic mutations: each operation is independently debuggable;
    cascades are queued, never nested.
  - Explicit degraded mode: outputs produced without full inference are flagged
    and automatically requeued when inference becomes available.
  - CLI is the primary interface (not MCP).
  - All inference routes through a single configured provider (gemini, cerebras,
    or ollama). Data sovereignty: US-only inference.

CLI entry point: ``valence``  (see docs/cli-reference.md)
Behavioral spec: SPEC.md
"""

__version__ = "2.0.0"

from . import (
    core as core,
)
