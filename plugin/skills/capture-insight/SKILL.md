---
name: capture-insight
description: Capture an important insight or decision to the knowledge base
user_invocable: true
args:
  - name: content
    description: The insight to capture
    required: true
  - name: domain
    description: Domain classification (e.g., "tech/architecture", "preferences")
    required: false
  - name: confidence
    description: Confidence level 0-1 (default 0.8)
    required: false
---

# Capture Insight

Store an important piece of information in the Valence knowledge base.

## Instructions

1. First, search for similar existing beliefs using `mcp__valence_substrate__belief_query`
2. If a very similar belief exists (>90% match), consider:
   - Using `mcp__valence_substrate__belief_supersede` to update it
   - Boosting its confidence through corroboration
   - Noting any differences as a potential tension
3. If this is new information, use `mcp__valence_substrate__belief_create` with:
   - The content as provided
   - Appropriate confidence level (default 0.8 for user-stated facts)
   - Domain path parsed from the domain argument (split on "/")
   - Current session as source (use VALENCE_SESSION_ID if available)
4. Extract and link relevant entities (people, tools, projects, concepts)
5. Confirm what was captured to the user

## Content to Capture
{{ content }}

{% if domain %}
## Domain
{{ domain }}
{% endif %}

{% if confidence %}
## Confidence
{{ confidence }}
{% endif %}

## Execution

1. Search for duplicates first
2. Parse domain path (e.g., "tech/architecture" -> ["tech", "architecture"])
3. Identify entities in the content (look for proper nouns, tool names, etc.)
4. Create the belief with appropriate metadata
5. Report what was stored and any entities linked
