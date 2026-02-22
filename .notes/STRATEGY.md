# Accrue Strategy & Go-to-Market

**Last Updated:** 2026-02-03

---

## TL;DR

- **Open source (Apache 2.0)** the core library for distribution and credibility
- **Verticalize the paid product** - start with Sales Intelligence, not general-purpose
- **Monetize via:** Cloud-hosted SaaS ($29-299/mo) + Consulting ($200-500/hr)
- **Timeline:** $10k MRR in 6-12 months with consistent execution

---

## Market Position

### The Gap

| Type | Examples | Gap |
|------|----------|-----|
| **SaaS Platforms** | Clay, Apollo, ZoomInfo | $149-800/mo, no-code, not developer-friendly |
| **Open Source** | Beton | Web UI only, not a Python library |
| **LLM Libraries** | Instructor, LangChain | Not DataFrame/enrichment focused |

**Accrue fills:** A developer-friendly Python library for LLM-powered DataFrame enrichment. "Clay for developers."

### Competitive Advantage

1. **Real market gap** - No OSS Clay alternative as a Python library
2. **Growing market** - LLM costs dropping, enrichment becoming mainstream
3. **Developer-first** - vs. no-code competitors
4. **Deep expertise** - You built it, consulting credibility

---

## Strategy: Open Core + Vertical Products

```
OPEN SOURCE (Apache 2.0)
├── Accrue Core Library
├── Maximum distribution, all use cases
└── Builds brand and credibility

PAID PRODUCTS (Vertical)
├── Product #1: Sales Intelligence ($99/mo)
│   └── "Conversational Clay" for lead scoring
├── Product #2: Due Diligence ($299/mo) [Later]
│   └── VC/PE market research
└── Product #3: Competitive Intel ($199/mo) [Later]
```

### Why Vertical > Horizontal

| Factor | Vertical | Horizontal |
|--------|----------|------------|
| Marketing | "Score your leads with AI" (clear) | "Analyze data with AI" (vague) |
| Competition | Clay, Apollo (beatable) | ChatGPT (unbeatable) |
| Pricing power | High (solves specific pain) | Low (commodity) |
| Sales cycle | Short (self-select) | Long (education needed) |

**General analyst agent = vitamin.** People might want it.
**Lead scoring agent = painkiller.** People need it and will pay.

---

## Go-to-Market Phases

### Phase 1: Distribution (Months 1-3)

**Goal:** 1,000 GitHub stars, 10k+ PyPI downloads, 500 email subscribers

**Week 1-2: Pre-Launch**
- [ ] Fix release blockers (LICENSE done, CLI, README)
- [ ] Create demo GIF (30 seconds)
- [ ] Write launch post: "I Built an Open-Source Alternative to Clay"
- [ ] Landing page with email capture

**Week 3: Soft Launch**
- [ ] Publish to PyPI
- [ ] dev.to, Reddit (r/Python, r/datascience, r/MachineLearning)

**Week 4: Hacker News**
- Post: "Show HN: Accrue - Enrich your data with LLMs in 4 lines of Python"
- Monday/Tuesday 8-10 AM Pacific
- Respond to ALL comments for 6+ hours

**Weeks 5-12:** Ship weekly, engage communities, guest on podcasts

### Phase 2: First Revenue (Months 3-6)

**Goal:** $2k-5k MRR, 50 paying customers

**Cloud-Hosted Launch**
- `app.accrue.dev` or similar
- User brings own API key initially
- Pricing: Free (100 rows) → $29 (5k rows) → $99 (25k rows)

**Consulting Soft Launch**
- "Need help implementing? [Book a call]" in README
- Start at $200-300/hr, raise to $400-500/hr after testimonials

### Phase 3: Scale (Months 6-12)

**Goal:** $10k+ MRR

**If cloud winning:** Add team features, templates, integrations
**If consulting winning:** Productized services, courses, agency model
**If exceptional traction:** Consider small raise or YC

---

## Pricing

### Cloud-Hosted

| Tier | Price | Rows/Month | Target |
|------|-------|------------|--------|
| Free | $0 | 100 | Evaluation |
| Starter | $29 | 5,000 | Side projects |
| Pro | $99 | 25,000 | Growth companies |
| Business | $299 | 100,000 + team | Scaling |
| Enterprise | Custom | Unlimited | Large orgs |

**Value prop:** 10x cheaper than Clay ($149+), with full Python flexibility

### Consulting

| Service | Price |
|---------|-------|
| Strategy Call (1hr) | $300 |
| Implementation Sprint (1 week) | $5,000 |
| Retainer (10 hrs/month) | $3,000 |
| Custom Development | $250/hr |

---

## Paid Product: Analyst Agent

```
PAID PRODUCT
┌───────────────────────────────────────────┐
│         Conversational Interface          │
│  "Score these 200 leads for our product"  │
│  "Which companies just raised funding?"   │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│          Agent Reasoning Layer            │
│  Query planning, tool selection, reports  │
└─────────────────┬─────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│         ACCRUE CORE (OSS)                 │
│  Enrichment, Graph, Provenance, Cross-Row │
└───────────────────────────────────────────┘
```

**Positioning:** "Your AI research analyst that actually understands your data"

**Example conversation:**
```
User: Score these 200 leads for our enterprise security product

Agent: [Enriching...] [Analyzing buying signals...]

       Hot (32 leads) - Strong fit, recent signals
       - 18 hired CISO in last 90 days
       - 9 mentioned "SOC 2" in job posts

       Top 5 to contact first:
       1. Acme Corp (Score: 94) - New CISO + budget cycle
       ...
```

---

## Success Metrics

### Phase 1 (Months 1-3)
- [ ] 1,000 GitHub stars
- [ ] 10,000 monthly PyPI downloads
- [ ] 500 email subscribers
- [ ] 1 front-page HN post

### Phase 2 (Months 3-6)
- [ ] 3,000 GitHub stars
- [ ] 50 paying customers
- [ ] $3,000 MRR
- [ ] 10 testimonials

### Phase 3 (Months 6-12)
- [ ] 5,000+ GitHub stars
- [ ] $10,000 MRR
- [ ] Clear path to $20k MRR

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| No launch traction | Multiple venues, iterate messaging |
| Big player copies | Move fast, community moat |
| Burnout as solo dev | Boundaries, automate support |
| Someone forks | Community and brand are moat |

---

## The Bottom Line

Open source gets distribution and credibility that would cost $100k+ in marketing. Cloud SaaS and consulting are monetization layers. You're not selling code—you're selling convenience (cloud) and expertise (consulting).

**Expected timeline to $10k/month:** 6-12 months with consistent execution.

---

*Internal planning document*
