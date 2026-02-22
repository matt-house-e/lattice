# 013: Rename Library from Lattice to Accrue

**Status:** Accepted
**Date:** 2026-02-22

**Context:** The library has been developed under the name "Lattice" through Phases 1-5 and into Phase 6B. With Phase 6A (PyPI publish) approaching, a name audit revealed significant conflicts: "Lattice" collides with Lattice HQ (HR SaaS, $3B+ valuation, dominant SEO), the `lattice` PyPI package (taken — published as `lattice-enrichment` as workaround), and multiple open-source projects (Lattice cryptography libraries, Kubernetes Lattice). These conflicts create brand confusion, SEO competition, and a non-ideal PyPI package name.

**Decision:** Rename the library from **Lattice** to **Accrue**. The name "Accrue" was selected after evaluating candidates across five criteria: PyPI availability, Python SEO clarity, brand/trademark conflicts, domain availability, and metaphor fit.

Key findings for "Accrue":
- **PyPI**: `accrue` is available (confirmed via `https://pypi.org/pypi/accrue/json` returning 404)
- **SEO**: "python accrue" returns zero competing libraries — clean namespace
- **Brand**: No trademark conflicts in software classes; "accrue" is a common English verb
- **Domain**: `accrue.dev` appears registrable
- **Metaphor**: "Value accrues through enrichment steps" — strong fit for a data enrichment pipeline where each step adds incremental value

The rename is a documentation + tracking decision now; the actual code rename (imports, package directory, pyproject.toml, tests, docs) will be executed as a separate epic.

**Alternatives considered:**

| Candidate | PyPI | SEO | Brand risk | Domain | Metaphor | Verdict |
|-----------|------|-----|------------|--------|----------|---------|
| **Accrue** | Available | Clean | None | accrue.dev likely available | "Value accrues" — strong | **Selected** |
| Inlay | Available | Clean | Low | inlay.dev likely available | "Inlaying data into rows" — moderate | Runner-up; weaker metaphor |
| Stipple | Available | Clean | Low | stipple.dev likely available | Art technique of adding dots — moderate | Too obscure for non-artists |
| Burnish | Available | Clean | Low | burnish.dev likely available | "Polishing data" — moderate | Slightly generic |
| Lattice | Taken (`lattice-enrichment` workaround) | Dominated by Lattice HQ | High (Lattice HQ, $3B) | lattice.dev taken | "Lattice of interconnected data" | **Rejected** — SEO/brand conflicts |

**Consequences:**
- All code, tests, docs, examples, and configuration must be updated (tracked in GitHub Epic)
- PyPI package name changes from `lattice-enrichment` to `accrue`
- Import path changes from `from lattice import ...` to `from accrue import ...`
- No backwards-compatibility shim — pre-publish, no external users to migrate
- GitHub repo rename to be evaluated (can redirect)
