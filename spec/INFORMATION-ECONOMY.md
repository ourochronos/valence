# The Information Economy

*What comes after capitalism for knowledge work?*

---

## The Question

Capitalism rewards:
- Producing goods people buy
- Services people pay for
- Attention sold to advertisers

But what about **information that should be free**?
- Journalism (corrupted by engagement metrics)
- Local reporting (no business model)
- Infrastructure observation (unpaid civic duty)
- Scientific verification (grants + prestige, poorly aligned)

How do we reward people for contributing *truth* to the commons?

---

## The Valence Answer

**Reputation is the currency of the information economy.**

Not tokens. Not money. Not followers.

Reputation that:
- Is earned through verified contributions
- Cannot be bought, only demonstrated
- Is domain-specific (trusted for X, not for Y)
- Translates to influence and visibility
- Decays without continued contribution

---

## Independent Reporters

Anyone can be a reporter. Your beat is whatever you observe.

### The Flow

```
1. OBSERVE
   See something noteworthy
   - Infrastructure problem
   - Local event
   - Price change
   - Policy violation
   - Weather condition
   
2. REPORT
   Store as belief with:
   - Content: "Pothole at 5th and Main, ~2ft diameter"
   - Confidence: 0.9 (I saw it myself)
   - Evidence: Photo, timestamp, location
   - Domains: [infrastructure, roads, downtown]
   
3. VERIFY
   Others in the area can:
   - Confirm (saw it too → corroboration rises)
   - Contradict (it's been fixed → you update or lose rep)
   - Add detail (it's actually 3ft → refinement)
   
4. AGGREGATE
   Network now knows:
   - "Pothole at 5th and Main"
   - Confidence: 0.95 (multiple independent reports)
   - First reporter: @LocalEyes (reputation boost)
   - Status: Unresolved
   
5. ACT
   City queries: "Infrastructure issues in downtown?"
   → Gets prioritized list by confidence + severity
   → Can subscribe to updates
   → Resolution gets verified too
```

### Incentive Alignment

| Action | Reputation Effect |
|--------|-------------------|
| First accurate report | +++ (discovery bonus) |
| Corroborating report | + (verification reward) |
| False report | -- (proportional to claimed confidence) |
| Finding error in existing report | ++ (discrepancy bounty) |
| Updating with resolution | + (closing the loop) |

**Spam is expensive. Accuracy pays.**

---

## Beyond Potholes

This model works for:

### Local Journalism
- Meeting coverage ("City council voted X")
- Business changes ("New restaurant opening at Y")
- Community events
- Crime/safety observations

### Specialized Domains
- Tech: Bug reports, security vulnerabilities, performance issues
- Science: Experimental replications, data observations
- Markets: Price reports, availability, quality assessments
- Environment: Wildlife sightings, pollution, weather

### Institutional Accountability
- Policy tracking ("Agency said X, reality is Y")
- Promise verification ("Politician committed to X in 2020")
- Spending oversight ("Contract awarded to Z")

### Personal Expertise
- Professional knowledge sharing
- Skill demonstration through accurate predictions
- Building reputation in your domain

---

## The Feed

Not algorithmic. Not chronological. **Trust-weighted and query-driven.**

### Your Feed Options

**Following Mode**
"Show me beliefs from agents I trust, sorted by recency"
→ Like Twitter, but trust-filtered

**Domain Mode**
"Show me high-confidence beliefs about [topic] from trusted sources"
→ Like a specialized news feed

**Discovery Mode**
"Show me rising beliefs being verified by multiple sources"
→ Like trending, but truth-seeking

**Local Mode**
"Show me beliefs within 5 miles, last 24 hours"
→ Hyperlocal awareness

**Alert Mode**
"Notify me when beliefs about [X] change significantly"
→ Subscriptions to knowledge changes

### Feed Algorithm (Transparent)

```
score = (
  semantic_relevance × 
  source_trust × 
  confidence_score × 
  recency_factor × 
  corroboration_bonus
) / seen_penalty

You control the weights.
You see why each item ranked where it did.
```

---

## Economic Model

### How Reputation Becomes Value

1. **Influence**: High-rep voices shape what network believes
2. **Access**: Query detailed data, not just summaries
3. **Governance**: Participate in protocol decisions
4. **Priority**: Your verifications processed faster
5. **Invitations**: Exclusive federations want high-rep members

### What Reputation Replaces

| Old Model | Valence Model |
|-----------|---------------|
| Ad revenue | Reputation accumulation |
| Subscriptions | Query access tiers |
| Tips/donations | Verification rewards |
| Employment | Domain expertise building |
| Credentials | Demonstrated accuracy |

### Bootstrap

New participants can:
- Start in low-stakes domains (less reputation required)
- Verify others' claims (easier than originating)
- Join federations that vouch for members
- Build slowly through consistent accuracy

No cold start problem — value from day one through verification work.

---

## Implementation

This needs:

1. **Location-aware beliefs** (add geo to belief schema)
2. **Media attachments** (photos/video as evidence)
3. **Subscription system** (already in query-protocol)
4. **Feed UI** (ranked query results)
5. **Mobile client** (report from anywhere)

The core infrastructure already supports this. It's application layer on top of what we spec'd.

---

## Why This Wins

**Current information economy:**
- Journalists beholden to outlets/advertisers
- Citizen journalism uncompensated
- Verification is cost center, not profit center
- Truth competes with engagement

**Valence information economy:**
- Anyone can report and earn reputation
- Verification is how you earn
- Truth-seeking is the profitable strategy
- No gatekeepers, but earned credibility

**The incentives finally point toward truth.**

---

*"What comes after capitalism? An economy where contributing accurate information to the commons is directly rewarded."*
