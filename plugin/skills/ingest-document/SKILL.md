---
name: ingest-document
description: Ingest a document into the knowledge substrate, extracting beliefs
user_invocable: true
args:
  - name: path
    description: Path to the document to ingest
    required: true
  - name: domain
    description: Base domain for extracted beliefs (optional)
    required: false
---

# Ingest Document

Process a document and extract beliefs from it into the knowledge substrate.

## Instructions

1. Read the document at the given path using the Read tool
2. Analyze the content to identify:
   - Factual claims and assertions
   - Decisions that were made
   - Preferences expressed
   - Technical specifications
   - Key entities (people, tools, projects)
3. For each extracted belief:
   - Assess confidence based on how clearly stated it is
   - Determine appropriate domain path
   - Identify related entities
   - Create the belief using `mcp__valence_substrate__belief_create`
4. Link the document as a source
5. Report a summary of what was extracted

## Document Path
{{ path }}

{% if domain %}
## Base Domain
{{ domain }}
{% endif %}

## Execution Steps

1. **Read the document**
   - Use the Read tool to get the content
   - Note the document type (markdown, code, config, etc.)

2. **Extract beliefs**
   For each significant piece of information:
   - Formulate it as a clear belief statement
   - Assign confidence:
     - 0.9+ for explicit, unambiguous statements
     - 0.7-0.9 for clear implications
     - 0.5-0.7 for inferences or uncertain claims
   - Classify into domain

3. **Extract entities**
   - People mentioned
   - Tools, libraries, frameworks
   - Projects or codebases
   - Concepts and patterns

4. **Create beliefs**
   Use belief_create for each, linking:
   - The document as source
   - Relevant entities
   - Appropriate domain path

5. **Report results**
   - Number of beliefs extracted
   - Key entities found
   - Any uncertainties or items needing review

## Quality Guidelines

- Don't over-extract: focus on significant, reusable knowledge
- Preserve context: beliefs should be understandable standalone
- Link entities: this enables cross-referencing later
- Note uncertainty: lower confidence for inferred information
- Avoid duplicates: check if similar beliefs already exist
