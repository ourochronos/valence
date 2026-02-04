# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Nothing yet

---

## [0.1.0-alpha] - 2026-02-03

### 🎉 Initial Alpha Release

This release marks the completion of Valence's comprehensive technical specification
and the foundation of its Python implementation.

### Added

#### Specifications (~850KB across 41 documents)

**Core Components (Wave 1)**
- `belief-schema` — Belief data model with PGVector storage
- `confidence-vectors` — Six-dimensional confidence scoring
- `identity-crypto` — Ed25519/X25519 cryptographic identity
- `trust-graph` — Relationship trust propagation algorithms

**Protocol Components (Wave 2)**
- `query-protocol` — Privacy-preserving semantic queries
- `verification-protocol` — Claim verification and reputation
- `federation-layer` — Node federation with differential privacy

**Network Components (Wave 3)**
- `consensus-mechanism` — Byzantine fault-tolerant consensus
- `incentive-system` — Token economics and stake mechanics
- `api-integration` — REST API and MCP integration specs

**Resilience Components (Wave 4)**
- `resilient-storage` — Post-quantum encryption, erasure coding

**Extensions**
- Migration & onboarding specification
- MCP bridge for AI agent integration
- SDK specification for client libraries

**Community Documents**
- `MANIFESTO.md` — The philosophical foundation
- `ADOPTION.md` — Phase-by-phase adoption path
- `SOCIAL-LAYER.md` — Trust-weighted social features
- `INFORMATION-ECONOMY.md` — Post-capitalism knowledge model

#### Implementation (Python Package)

- `valence.substrate` — Knowledge substrate with PGVector
- `valence.vkb` — Conversation tracking (sessions, exchanges, patterns)
- `valence.embeddings` — Multi-provider embedding architecture
- `valence.server` — HTTP API server with JWT auth
- `valence.agents` — Matrix bot integration
- `valence.federation` — Federation layer foundation

**MCP Servers**
- `valence-substrate` — Belief management tools for AI agents
- `valence-vkb` — Conversation tracking tools

**Infrastructure**
- Ansible IaC for pod deployment
- Docker Compose configuration
- E2E deployment testing

### Status

- **Specifications**: Complete (production-ready design)
- **Implementation**: Alpha (functional core, not production-ready)
- **Documentation**: Complete philosophy and architecture docs
- **Tests**: Basic coverage, expanding

### Known Limitations

- Federation layer is specified but not implemented
- Consensus mechanism exists in spec only
- Token economics are designed, not deployed
- No mobile clients yet

---

[Unreleased]: https://github.com/orobobos/valence/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/orobobos/valence/releases/tag/v0.1.0-alpha
