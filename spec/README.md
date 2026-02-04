# Valence Technical Specifications

This directory contains the detailed technical specifications for implementing Valence. For the philosophical foundation and architectural overview, see [docs/](../docs/).

## Overview

Valence is a distributed epistemic infrastructure for AI agents. These specs define how beliefs, confidence, trust, and knowledge flow through the network.

## Components

The system is built from 11 interconnected components:

### Foundation Layer
| Component | Description | Status |
|-----------|-------------|--------|
| [belief-schema](components/belief-schema/) | Core data model for beliefs | ✅ Specified |
| [confidence-vectors](components/confidence-vectors/) | 6-dimensional confidence model | ✅ Specified |
| [identity-crypto](components/identity-crypto/) | Cryptographic identity & signatures | ✅ Specified |
| [trust-graph](components/trust-graph/) | Agent trust relationships | ✅ Specified |

### Protocol Layer
| Component | Description | Status |
|-----------|-------------|--------|
| [query-protocol](components/query-protocol/) | Multi-signal belief retrieval | ✅ Specified |
| [verification-protocol](components/verification-protocol/) | Claim verification & reputation | ✅ Specified |
| [federation-layer](components/federation-layer/) | Cross-node belief sharing | ✅ Specified |

### Network Layer
| Component | Description | Status |
|-----------|-------------|--------|
| [consensus-mechanism](components/consensus-mechanism/) | Network-wide belief consensus | ✅ Specified |
| [incentive-system](components/incentive-system/) | Reputation-based incentives | ✅ Specified |
| [api-integration](components/api-integration/) | External API & MCP bridge | ✅ Specified |

### Resilience Layer
| Component | Description | Status |
|-----------|-------------|--------|
| [resilient-storage](components/resilient-storage/) | Backup, DR, post-quantum crypto | ✅ Specified |

## Extensions

| Extension | Description |
|-----------|-------------|
| [migration](extensions/migration/) | Import from existing knowledge bases |

## Key Documents

| Document | Purpose |
|----------|---------|
| [MANIFESTO.md](MANIFESTO.md) | Why Valence exists — the movement |
| [ADOPTION.md](ADOPTION.md) | Phase-by-phase adoption path |
| [SOCIAL-LAYER.md](SOCIAL-LAYER.md) | Trust-weighted social features |
| [INFORMATION-ECONOMY.md](INFORMATION-ECONOMY.md) | Post-capitalism knowledge model |
| [CONTEXT-WINDOW.md](CONTEXT-WINDOW.md) | LLM context as epistemic retrieval |
| [STATUS.md](STATUS.md) | Current development status |

## Architecture Principles

1. **Beliefs, not facts** — Everything has confidence vectors, provenance, temporal validity
2. **Trust is earned** — Reputation from behavior, not credentials
3. **Temporal validity** — When something was true matters as much as what was true
4. **Local-first** — Your data, your control, federation is opt-in
5. **Anti-fragile** — Design against adversaries, welcome scrutiny

## Implementation Status

See [STATUS.md](STATUS.md) for detailed progress tracking.

**Current phase:** Specification complete, beginning implementation

## Getting Started

For implementation:
1. Start with [belief-schema](components/belief-schema/) — the core data model
2. Add [confidence-vectors](components/confidence-vectors/) — the epistemic engine
3. Implement [query-protocol](components/query-protocol/) — how retrieval works
4. Build up from there

For migration from existing systems:
- See [extensions/migration](extensions/migration/) for import patterns
