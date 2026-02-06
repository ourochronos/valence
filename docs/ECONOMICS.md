# Economic Sustainability Models for Valence

*Discussion document for issue #274 — informing a decision, not making one.*

---

## 1. Problem Statement

Valence is infrastructure for sovereign knowledge. It must persist. Infrastructure that disappears betrays the people who built on it — and for a system whose core promise is "structurally incapable of betrayal," economic failure is a form of structural failure.

The sustainability problem has four dimensions:

### 1.1 Infrastructure Costs

Running Valence requires compute. Even in a federated model where users host their own nodes, someone must maintain:

- **Protocol coordination**: Discovery services, schema registries, trust anchor bootstrap
- **Development infrastructure**: CI/CD, testing, security audits, package distribution
- **Reference implementations**: At minimum one well-maintained node implementation
- **Documentation and onboarding**: The barrier to running a node must stay low

These costs are real and ongoing. They don't disappear because the protocol is decentralized — they just distribute differently.

### 1.2 Development Resources

Valence is not a weekend project. It requires sustained, high-quality engineering across:

- Core protocol evolution (federation, aggregation, privacy)
- Security (cryptographic review, vulnerability response)
- Ecosystem tooling (SDKs, integrations, migration tools)
- UX (if adoption matters, usability matters)

Volunteer-only development is possible but creates fragility. A single burned-out maintainer is a single point of failure that no amount of architectural decentralization can fix.

### 1.3 Incentive Alignment

The deepest problem. Whoever funds development exerts gravitational pull on what gets built. This isn't corruption — it's physics. A VC-funded Valence would build features VCs value (growth metrics, monetization hooks). A grant-funded Valence would build what grant committees value (research novelty, institutional legibility). A user-funded Valence would build what users value.

The question is: **which gravitational field do we want to operate in?**

### 1.4 Avoiding Capture

From PRINCIPLES.md: *"Structure prevents capture. PBC or equivalent. The entity cannot be sold, pivoted, or hollowed out."*

This constrains the solution space. Any funding model that creates a pathway to capture — an acqui-hire incentive, a runway-dependent board, a token with speculative dynamics — is architecturally incompatible, regardless of how much capital it could raise.

---

## 2. Model Analysis

### 2.1 Open Source + Services

**Description**: Core protocol and reference implementation are free and open source. Revenue comes from hosted services: managed node hosting, enterprise support contracts, SLA-backed federation endpoints, consulting.

**Examples**: Red Hat, GitLab, Elasticsearch (before license change), PostHog, Canonical.

**Pros**:
- Proven model with many successful precedents
- Aligns development with operational expertise (you eat your own cooking)
- Enterprise customers provide stable, recurring revenue
- Open core attracts contributors and builds trust
- Revenue scales with adoption

**Cons**:
- Creates pressure to withhold features from the open source version ("open core" drift)
- Enterprise focus can pull development away from individual user needs
- Requires building and maintaining commercial operations (sales, support, SLAs) — a very different competency from protocol development
- Hosting a "managed Valence node" creates exactly the centralization the architecture is designed to prevent

**Risks**:
- **Feature gating temptation**: The most natural way to monetize is to make the hosted version better than the self-hosted version. This directly conflicts with user sovereignty.
- **Elastic/MongoDB trap**: Competitive pressure from cloud providers can push toward license restrictions that undermine openness.
- **Centralization creep**: If most users end up on the hosted service (because self-hosting is hard), you've recreated the platform problem with extra steps.

**Compatibility with principles**:
- ⚠️ **Tension** with "structurally incapable of betrayal" — the commercial entity has an incentive to make self-hosting harder, even if it resists that incentive.
- ⚠️ **Tension** with "designed to survive being stolen" — if the business model depends on being the best host, a competitor forking and hosting better is a threat, not a success.
- ✅ **Compatible** with openness, if the core remains genuinely open.

### 2.2 Foundation Model

**Description**: A nonprofit foundation (or PBC) holds the project. Funding comes from grants, donations, corporate sponsorships, and potentially government funding for digital infrastructure.

**Examples**: Mozilla Foundation, Apache Software Foundation, Linux Foundation, Signal Foundation, Let's Encrypt (ISRG).

**Pros**:
- Mission alignment is structural (nonprofit charter, PBC bylaws)
- No equity, no acquisition pathway — capture-resistant by design
- Grants can fund long-term research that no market would pay for (privacy-preserving aggregation, trust models)
- Foundation governance can include community representation
- Tax advantages for donors in many jurisdictions

**Cons**:
- Grant funding is cyclical, competitive, and often tied to deliverables that may not match project priorities
- Corporate sponsorship creates softer capture — you don't build features that anger your sponsors
- Fundraising is a skill and a full-time job; engineering organizations are often bad at it
- Foundations can become bureaucratic, slow, and self-perpetuating
- Salary competitiveness is a real problem — you lose people to better-paying opportunities

**Risks**:
- **Mozilla problem**: Large enough to sustain itself, not aligned enough to stay focused. Revenue diversification (search deals) creates dependency on the entities you're supposed to be an alternative to.
- **Grant dependency**: If 60% of funding comes from one grantor, that grantor has soft veto power over direction.
- **Institutional drift**: Foundations can become more focused on their own survival than their mission. The organization outlives its purpose and becomes a jobs program.

**Compatibility with principles**:
- ✅ **Strong compatibility** with mission permanence and capture resistance
- ✅ **Compatible** with openness — no commercial reason to restrict
- ⚠️ **Tension** with "aggregation serves users" — grant-funded priorities may diverge from user needs
- ⚠️ **Risk** of institutional capture replacing corporate capture (slower, subtler, equally corrosive)

### 2.3 Protocol Fees

**Description**: The protocol itself collects tiny fees on certain operations — federation queries, trust attestations, aggregation requests. Fees flow to a treasury that funds development.

**Examples**: Ethereum gas fees, Filecoin storage fees, ENS registration fees. More modestly: DNS registration fees funding ICANN.

**Pros**:
- Revenue directly proportional to usage — the more valuable the protocol is, the more it earns
- No need for a separate commercial entity
- Can fund development without external dependencies
- Creates a natural mechanism for resource allocation (what's worth paying for?)

**Cons**:
- **Introduces a toll booth into a protocol designed to be free and open.** This is the fundamental problem.
- Requires some form of fee collection infrastructure — which is a centralization point
- Fee governance (who sets rates, who controls the treasury) recreates the governance problems you're trying to avoid
- Token-based fee mechanisms introduce speculation, volatility, and attract the wrong community
- Even tiny fees create barriers to adoption, especially in contexts where Valence should be most accessible (low-resource communities, developing economies)

**Risks**:
- **Centralization magnet**: Fee collection requires either a central collector (centralization) or a consensus mechanism (complexity, energy, blockchain baggage)
- **Tokenization pressure**: "We have protocol fees" → "We should have a token" → "The token should be tradeable" → "Now we're a crypto project." This pipeline is well-documented and nearly gravitational.
- **Exclusion**: Any fee, however tiny, excludes someone. For infrastructure meant to be a commons, this is a design failure, not a rounding error.
- **Perverse incentives**: Fee-funded development is incentivized to make the protocol chattier, not more efficient.

**Compatibility with principles**:
- ❌ **Conflicts** with "structurally incapable of betrayal" — fee control is a lever of power
- ❌ **Conflicts** with "designed to survive being stolen" — a fork without fees is strictly better for users
- ⚠️ **Tension** with user sovereignty — users must pay to use their own knowledge infrastructure
- ⚠️ **Tension** with openness — fee enforcement requires gatekeeping at some layer

### 2.4 Cooperative Model

**Description**: Users collectively fund the infrastructure they depend on. This could be a formal cooperative (member-owned entity), a membership organization, or a community funding pool (like OpenCollective but with governance).

**Examples**: REI, Mondragon, AP (Associated Press), rural electric cooperatives, consumer credit unions. In tech: Stocksy, Open Food Network.

**Pros**:
- Maximum incentive alignment — the funders are the users
- Democratic governance is structurally compatible with user sovereignty
- No external capture pathway — members can't be bought out by a third party
- Scales naturally with the user base
- Members have skin in the game and voice in governance

**Cons**:
- Cooperatives are slow to make decisions (democratic governance has overhead)
- Requires a critical mass of paying members before becoming self-sustaining — the chicken-and-egg problem
- Membership fees create the same accessibility concerns as protocol fees (though scholarships/sliding scale can mitigate)
- Cooperative governance can be captured by motivated minorities (whoever shows up to meetings)
- Legal structures for digital cooperatives are immature in most jurisdictions
- Difficult to fund speculative R&D — members fund what they need now, not what the ecosystem needs in five years

**Risks**:
- **Governance gridlock**: 10,000 members with 10,000 opinions about what to build next
- **Free rider problem**: The protocol is open, so non-members benefit equally. Why pay?
- **Tyranny of the present**: Cooperatives are good at maintaining existing value, less good at taking big swings on uncertain future value
- **Scale ceiling**: Very few tech cooperatives have scaled past modest size. The model may work for sustaining but not for growing.

**Compatibility with principles**:
- ✅ **Strong compatibility** with user sovereignty — users govern what serves them
- ✅ **Compatible** with mission permanence — cooperative charter can be constitutional
- ✅ **Compatible** with capture resistance — no equity to acquire
- ⚠️ **Tension** with "designed to survive being stolen" — if a fork is free and the cooperative charges dues, the fork wins on price (though not on governance or quality)

### 2.5 Lightweight Volunteer Model

**Description**: No formal entity, no revenue model. Development is done by volunteers. Infrastructure costs are kept minimal through aggressive architectural choices (no central services, users host everything). Coordination through GitHub, Discord, and shared norms.

**Examples**: Early Linux, SQLite (partially — Hwaci provides some funding), many successful IETF protocols, most of the internet's foundational infrastructure.

**Pros**:
- Zero capture risk — there's nothing to capture
- Maximum alignment with "designed to survive being stolen" — it already is, in a sense
- No governance overhead beyond code review norms
- Forces architectural decisions that minimize operational dependency
- Attracts contributors motivated by the mission, not compensation
- Can coexist with any of the above models as a starting phase

**Cons**:
- **Bus factor**: If key maintainers burn out or move on, the project stalls
- Security response is slow or nonexistent — volunteers have day jobs
- No one is responsible for boring-but-necessary work (documentation, CVE triage, dependency updates)
- Hard to attract non-developer contributions (design, UX, documentation, community management)
- Quality and velocity are unpredictable
- Users who depend on the protocol have no recourse when things break

**Risks**:
- **Maintainer burnout**: The most common cause of death for volunteer open source projects. Not dramatic — just a slow fade.
- **Tragedy of the commons**: Everyone benefits, no one invests. Infrastructure degrades.
- **Implicit hierarchy**: Without formal governance, power concentrates in whoever has the most time and commit access. This is less democratic than it appears.
- **Security liability**: A protocol handling personal knowledge substrates cannot afford the security posture of a hobby project.

**Compatibility with principles**:
- ✅ **Strong compatibility** with openness and resilience
- ✅ **Compatible** with capture resistance
- ❌ **Conflicts** with mission permanence — volunteer projects are fragile, and fragility is a form of structural failure
- ⚠️ **Tension** with "structurally incapable of betrayal" — under-maintained infrastructure betrays users through neglect, not malice

---

## 3. Architecture Implications

Each funding model exerts pressure on technical decisions. These pressures are worth naming explicitly, because they operate whether or not anyone intends them.

### 3.1 Centralization Pressure

| Model | Centralization Pressure | Mechanism |
|-------|------------------------|-----------|
| Open Source + Services | **High** | Hosted service must be better than self-hosting to justify revenue |
| Foundation | **Low-Medium** | Grants may require measurable infrastructure (dashboards, analytics) |
| Protocol Fees | **High** | Fee collection requires a coordination point |
| Cooperative | **Low** | Members want the system to work for them, wherever they run it |
| Volunteer | **Very Low** | No one wants to maintain central infrastructure for free |

### 3.2 Feature Gating Pressure

| Model | Feature Gating Risk | Mechanism |
|-------|---------------------|-----------|
| Open Source + Services | **High** | "Premium" features fund the business |
| Foundation | **Low** | No commercial reason to gate features |
| Protocol Fees | **Medium** | Fee-free features vs. fee-requiring features creates a de facto tier |
| Cooperative | **Low** | Members want everything available to members; non-members may get less |
| Volunteer | **None** | No one gates features for free |

### 3.3 Data Ownership Impact

| Model | Data Ownership Risk | Mechanism |
|-------|---------------------|-----------|
| Open Source + Services | **Medium-High** | Hosted users' data lives on company servers |
| Foundation | **Low** | No business reason to hold user data |
| Protocol Fees | **Medium** | Fee infrastructure may require transaction logs |
| Cooperative | **Low** | Members govern data policies |
| Volunteer | **Very Low** | No infrastructure to hold data |

### 3.4 Development Priority Distortion

| Model | Priority Distortion | Toward |
|-------|---------------------|--------|
| Open Source + Services | Enterprise features, admin dashboards, SSO | Paying customers |
| Foundation | Publishable results, grant deliverables | Institutional priorities |
| Protocol Fees | Protocol chattiness, transaction throughput | Fee-generating activity |
| Cooperative | Current member needs, incremental improvement | Present over future |
| Volunteer | What's interesting to work on | Contributor interests |

---

## 4. Compatibility Assessment

Evaluating each model against Valence's non-negotiable principles:

### 4.1 Scoring Matrix

| Principle | Services | Foundation | Protocol Fees | Cooperative | Volunteer |
|-----------|----------|------------|---------------|-------------|-----------|
| User sovereignty | ⚠️ | ✅ | ❌ | ✅ | ✅ |
| Structurally incapable of betrayal | ⚠️ | ✅ | ❌ | ✅ | ⚠️ |
| Aggregation serves users | ⚠️ | ⚠️ | ⚠️ | ✅ | ✅ |
| Designed to survive being stolen | ❌ | ✅ | ❌ | ⚠️ | ✅ |
| Mission permanence | ✅ | ✅ | ⚠️ | ✅ | ❌ |
| Openness as resilience | ⚠️ | ✅ | ⚠️ | ✅ | ✅ |

**Legend**: ✅ = compatible, ⚠️ = tension exists, ❌ = structural conflict

### 4.2 Key Observations

**Protocol fees are the least compatible model.** Every mechanism required to collect fees (central treasury, fee enforcement, transaction tracking) creates exactly the leverage points Valence's architecture is designed to eliminate. The model also fails the "survive being stolen" test catastrophically — a fee-free fork is always more attractive.

**The volunteer model is the most ideologically pure but the least durable.** It scores well on every principle except mission permanence, which is arguably the meta-principle that enables all the others. A project that can't sustain itself can't keep any of its promises.

**Foundation and cooperative models score best overall**, but with different strengths. Foundations excel at long-term research and institutional legitimacy. Cooperatives excel at user alignment and democratic governance. Neither is perfect alone.

**Open source + services is viable but requires constant vigilance** against the structural incentives it creates. Every successful open-core company eventually faces the question: "Should this feature be free or paid?" That question, asked enough times, reshapes the project.

### 4.3 The Hybrid Possibility

These models are not mutually exclusive. A plausible structure might combine:

- **Foundation** as the legal home and steward of the protocol specification
- **Cooperative** as the governance body for operational decisions
- **Volunteer contributions** as the cultural norm and primary development model
- **Services** offered by independent entities (not the foundation) for users who want managed hosting

This separates the protocol steward (foundation) from the commercial ecosystem (independent service providers), while giving users collective governance (cooperative) and keeping the door open for unpaid contribution (volunteer).

Whether this complexity is justified — or whether it creates more problems than it solves — is the decision this document is meant to inform.

---

## 5. Open Questions

These questions don't have obvious answers. They need discussion.

1. **What's the actual cost floor?** Before choosing a model, we need a realistic estimate of minimum viable infrastructure and development costs. The answer determines whether the volunteer model is sufficient or whether funded development is required.

2. **How many users constitute sustainability?** If cooperative, what membership fee at what membership count covers costs? Is this realistic given Valence's current trajectory?

3. **Can a foundation avoid the Mozilla problem?** Mozilla's cautionary tale is well-known. What structural choices would keep a Valence foundation focused?

4. **Is there a way to offer services without centralization pressure?** Could a "hosting certification" or "compatible provider" model create a service ecosystem without a single provider becoming dominant?

5. **What happens when a major funder disagrees with the community?** Every model except pure volunteer has this problem. How is it resolved structurally, not interpersonally?

6. **What legal jurisdictions are compatible?** Cooperative law, foundation law, and PBC law vary enormously. The choice of legal structure constrains the choice of jurisdiction and vice versa.

7. **Is there precedent for a federated protocol sustaining itself economically?** Email, IRC, XMPP — all federated, all sustained, but none by a single entity. Is "the protocol sustains itself through the ecosystem" a viable model, or a way of saying "someone else's problem"?

---

## 6. Further Reading

- Nadia Eghbal, *Working in Public: The Making and Maintenance of Open Source Software* (2020)
- Yochai Benkler, *The Wealth of Networks* (2006)
- Trebor Scholz, *Platform Cooperativism* (2016)
- Mozilla Foundation financial reports and governance critiques
- Signal Foundation funding model and sustainability discussions
- Hintjens, P., *Social Architecture* (2016) — on community-driven development
- Open Source Initiative, "Open Source Business Models" (ongoing)

---

*This document is a starting point for discussion, not a conclusion. The right model will emerge from community deliberation, not from analysis alone. But analysis can narrow the space and name the tradeoffs clearly.*

*Relates to: [PRINCIPLES.md](./PRINCIPLES.md) | [VISION.md](./VISION.md) | [TRUST_MODEL.md](./TRUST_MODEL.md)*
