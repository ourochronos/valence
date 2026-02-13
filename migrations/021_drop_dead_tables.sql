-- Migration 021: Drop dead tables (belief_derivations, derivation_sources)
-- These were only used by the legacy CLI init command, never part of the MCP API.
-- The extraction_method column on beliefs replaces derivation_type.

DROP TABLE IF EXISTS derivation_sources CASCADE;
DROP TABLE IF EXISTS belief_derivations CASCADE;
