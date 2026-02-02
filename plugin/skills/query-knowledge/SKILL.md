---
name: query-knowledge
description: Search the Valence knowledge base for relevant information
user_invocable: true
args:
  - name: query
    description: What to search for
    required: true
  - name: domain
    description: Domain filter (optional, e.g., "tech", "preferences")
    required: false
---

# Query Knowledge

Search the Valence knowledge substrate for relevant beliefs.

## Instructions

1. Use `mcp__valence_substrate__belief_query` with the provided query
2. If a domain filter is provided, include it in the search
3. Present results in a clear, organized format showing:
   - The belief content
   - Confidence level (interpret: 0.9+ = very high, 0.7+ = high, 0.5+ = moderate)
   - When it was created
   - Source if available
4. Highlight any tensions (contradictions) if present
5. Suggest related queries if results seem incomplete

## Query: {{ query }}
{% if domain %}
## Domain Filter: {{ domain }}
{% endif %}

## Execution

Search for beliefs matching this query. Format the results clearly for the user.

If no results are found, suggest:
- Alternative phrasings of the query
- Broader domain searches
- Whether to capture new information on this topic

If results are found but seem incomplete, offer to:
- Search with different terms
- Check related entities
- Look for patterns on this topic
