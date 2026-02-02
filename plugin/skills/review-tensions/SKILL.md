---
name: review-tensions
description: Review and resolve contradictions in the knowledge base
user_invocable: true
args:
  - name: severity
    description: Minimum severity to show (low, medium, high, critical)
    required: false
  - name: limit
    description: Maximum number of tensions to review
    required: false
---

# Review Tensions

Review unresolved contradictions between beliefs in the knowledge base.

## Instructions

1. Use `mcp__valence_substrate__tension_list` to get unresolved tensions
2. For each tension, present clearly:
   - The two conflicting beliefs
   - Their confidence levels
   - Their sources and creation dates
   - The type and severity of conflict
3. For each tension, suggest possible resolutions:
   - **Supersede A with B**: B is more recent/accurate
   - **Supersede B with A**: A is more recent/accurate
   - **Keep both**: They're actually compatible (mark as accepted)
   - **Archive both**: Both are outdated
4. Ask the user how to resolve, or auto-resolve if clear
5. Use `mcp__valence_substrate__tension_resolve` to record resolution

{% if severity %}
## Minimum Severity
{{ severity }}
{% endif %}

{% if limit %}
## Limit
{{ limit }}
{% else %}
## Limit
10
{% endif %}

## Execution Steps

1. **List tensions**
   ```
   tension_list(status="detected", severity="{{ severity | default: 'low' }}", limit={{ limit | default: 10 }})
   ```

2. **Present each tension**
   Format:
   ```
   ## Tension #N ({{ severity }})

   **Belief A** (confidence: X, created: DATE)
   > [content of belief A]
   Source: [source info]

   **Belief B** (confidence: Y, created: DATE)
   > [content of belief B]
   Source: [source info]

   **Conflict type**: [contradiction/temporal_conflict/etc.]

   **Suggested resolution**: [your recommendation]
   ```

3. **Get user decision** (unless auto-resolvable)
   Ask: "How would you like to resolve this?"
   Options:
   - Keep A (supersede B)
   - Keep B (supersede A)
   - Keep both (they're compatible)
   - Archive both (outdated)

4. **Apply resolution**
   ```
   tension_resolve(tension_id, resolution="user explanation", action="chosen_action")
   ```

## Auto-Resolution Guidelines

You may auto-resolve (with explanation) when:
- One belief is clearly more recent AND has higher confidence
- One belief explicitly contradicts documented facts
- The "conflict" is actually just different phrasing (mark as accepted)

Always explain your reasoning if auto-resolving.

## When to Escalate

Ask the user when:
- Both beliefs have similar confidence/recency
- The conflict involves preferences or opinions
- Resolution requires domain expertise
- The stakes seem high (architectural decisions, etc.)
