# GitHub Standards and Best Practices

> Version 2.0 - Accrue
> Based on: Conventional Commits v1.0, SemVer 2.0, and GitHub best practices

## Table of Contents
1. [Why These Standards Matter](#why-these-standards-matter)
2. [Issue Management](#issue-management)
3. [Label Taxonomy](#label-taxonomy)
4. [Pull Request Standards](#pull-request-standards)
5. [Branch Strategy](#branch-strategy)
6. [Commit Message Convention](#commit-message-convention)
7. [Claude Code Integration](#claude-code-integration)

---

## Why These Standards Matter

### The Problem We're Solving

Without standardized workflows, AI assistants and human developers create
inconsistent issues, PRs, and commits. This leads to:

- **Poor searchability**: Can't find related issues or track feature progress
- **Unclear priorities**: No way to quickly identify critical vs. nice-to-have
  items
- **Automation failures**: CI/CD can't parse inconsistent formats
- **Knowledge silos**: New team members struggle to understand workflow
- **AI hallucination**: Claude Code creates different formats each time

### The Solution: Rigid Standards with Clear Rationale

By implementing these standards, we achieve:

- **Predictable AI behavior**: Claude Code follows exact templates every time
- **Automated versioning**: Commit types directly trigger version bumps
- **Visual hierarchy**: Color-coded labels provide instant context
- **Searchable history**: Consistent prefixes enable powerful queries
- **Reduced cognitive load**: Developers know exactly where things belong

---

## Issue Management

### Issue Types (Based on Agile Best Practices)

We use a **hierarchical system** matching industry-standard Agile methodologies:

#### 1. **Epic** (Large Initiative)
- **Purpose**: Multi-sprint features or major architectural changes
- **Scope**: 2+ weeks of work, multiple developers
- **Example**: "Implement Knowledge Base Integration"
- **Prefix**: `epic/`

#### 2. **Story** (User-Facing Feature)
- **Purpose**: Delivers direct value to end users
- **Scope**: 1-5 days of work
- **Example**: "As a user, I want to upload screenshots"
- **Prefix**: `story/`

#### 3. **Task** (Technical Work)
- **Purpose**: Non-user-facing technical improvements
- **Scope**: < 1 day of work
- **Example**: "Refactor LLM service for better testing"
- **Prefix**: `task/`

#### 4. **Bug** (Defect)
- **Purpose**: Fix broken functionality
- **Scope**: Variable
- **Example**: "Conversation node fails on empty input"
- **Prefix**: `bug/`

#### 5. **Spike** (Research/Investigation)
- **Purpose**: Time-boxed research to reduce uncertainty
- **Scope**: Max 4 hours
- **Example**: "Investigate JSM Cloud API rate limits"
- **Prefix**: `spike/`

### Issue Title Format

```
[Type]: [Component] Description (max 60 chars)
```

!!! info "Why this format?"

    - **Sorting**: Issues group naturally in alphabetical lists
    - **Scanning**: Developers can quickly identify relevant issues
    - **Automation**: Bots can parse and categorize automatically

#### Component Prefixes

Based on the accrue package structure:

- `[Core]` - Config, processors, enricher, exceptions
- `[VectorStore]` - Vector store and document management
- `[Chains]` - LLM chains and prompts
- `[Data]` - Data models and fields
- `[Test]` - Testing infrastructure
- `[Docs]` - Documentation
- `[Infra]` - CI/CD and deployment

### Issue Body Template

```markdown
## Context
Why does this issue exist? What problem does it solve?

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Technical Notes
Implementation hints, constraints, or dependencies.
```

---

## Label Taxonomy

We use a **prefix-based, color-coded system**:

### Type Labels

| Label | Color | Description |
|-------|-------|-------------|
| `type:epic` | `#0052CC` | Multi-sprint initiative |
| `type:story` | `#0075CA` | User-facing feature |
| `type:task` | `#4C9AFF` | Technical work |
| `type:bug` | `#FF5630` | Defect |
| `type:spike` | `#6554C0` | Research/investigation |

### Priority Labels

| Label | Color | Description |
|-------|-------|-------------|
| `priority:critical` | `#FF5630` | Production down, data loss risk |
| `priority:high` | `#FF7452` | Blocking other work |
| `priority:medium` | `#FFAB00` | Important but not blocking |
| `priority:low` | `#FFC400` | Nice to have |

### Component Labels

| Label | Color | Description |
|-------|-------|-------------|
| `component:core` | `#00B8D9` | Config, processors, enricher, exceptions |
| `component:vector-store` | `#00B8D9` | Vector store and document management |
| `component:chains` | `#00B8D9` | LLM chains and prompts |
| `component:data` | `#00B8D9` | Data models and fields |
| `component:testing` | `#00B8D9` | Test infrastructure |
| `component:docs` | `#00B8D9` | Documentation |
| `component:infra` | `#00B8D9` | CI/CD, deployment configs |

---

## Pull Request Standards

### Branch Naming Convention

```
[type]/[issue-number]-[brief-description]
```

**Examples:**

- `feature/45-add-file-upload`
- `fix/89-conversation-node-crash`
- `docs/34-api-documentation`
- `spike/67-jsm-investigation`

### PR Title Format

Must follow **Conventional Commits** specification:

```
<type>(<scope>): <description> (#issue-number)
```

**Types** (from Conventional Commits v1.0):

- `feat`: New feature (→ MINOR version)
- `fix`: Bug fix (→ PATCH version)
- `docs`: Documentation only
- `style`: Formatting, no logic change
- `refactor`: Code restructure, no behavior change
- `perf`: Performance improvement
- `test`: Add/fix tests
- `build`: Build system changes
- `ci`: CI/CD changes
- `chore`: Maintenance tasks
- `revert`: Revert previous commit

**BREAKING CHANGE**: Add `!` after type or include in footer (→ MAJOR version)

**Examples:**

```
feat(workflow): add retry logic to conversation node (#123)
fix(llm): handle OpenAI timeout errors (#456)
docs(api): update endpoint documentation (#789)
feat(ui)!: redesign chat interface with new theme (#234)
```

### PR Body Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Tests pass
- [ ] Linting/formatting passes
- [ ] Self-reviewed
- [ ] Documentation updated (if applicable)

Closes #<issue-number>
```

---

## Branch Strategy

### GitHub Flow (Simplified Single-Branch Workflow)

We use **GitHub Flow** - a lightweight, branch-based workflow with a single
protected `main` branch:

```
main (production-ready)
  ├── feature/[issue]-[description]
  ├── fix/[issue]-[description]
  ├── docs/[issue]-[description]
  └── spike/[issue]-[description]
```

**Key Principles:**

- **Single main branch**: Always deployable, protected
- **Short-lived feature branches**: Merge directly to `main` via PR
- **Squash merging**: Favor squashing commits when merging PRs to keep `main`
  history clean
- **Continuous deployment**: Every merge to `main` can go to production
- **Simple and fast**: No complex branching hierarchy

**Why GitHub Flow over GitFlow?**

- Faster iteration for small teams (2 developers)
- Reduces merge conflicts and integration issues
- Simpler mental model - no `develop` branch to manage
- Better for continuous deployment practices
- Clean linear history with squash merges

### Branch Protection Rules

**main branch:**

- Require PR reviews (1+ approvals)
- Require status checks (CI must pass)
- Require up-to-date branches before merging
- Include administrators in restrictions
- No force pushes allowed
- Require linear history (prefer squash merging)

---

## Commit Message Convention

### Format (Conventional Commits v1.0)

```
<type>(<scope>): <subject>

[optional body]

[optional footer(s)]
```

### Rules

1. **Subject line**: 50 chars max, imperative mood, no period
2. **Body**: Wrap at 72 chars, explain what and why
3. **Footer**: Breaking changes, issue references

### Examples

```
feat(workflow): add information completeness scoring

Implement LLM-based scoring system to evaluate when enough
information has been gathered. Uses GPT-4o structured output
to return confidence scores.

Closes #123
```

```
fix(llm): handle rate limit errors gracefully

Add exponential backoff retry logic for OpenAI API calls.
Maximum 3 retries with 2x backoff starting at 1 second.

Bug reported by users experiencing timeout errors during
high load periods.

Fixes #456
```

---

## Claude Code Integration

AI assistants should follow these standards when creating issues, PRs, or commits.

### For Issues

- Use title format: `[Type]: [Component] Description`
- Apply labels: 1 type + 1 priority + component(s)
- Fill in Context, Acceptance Criteria, and Technical Notes

### For Pull Requests

- Branch: `[type]/[issue]-[description]`
- Title: Follow Conventional Commits
- Always include `Closes #<issue-number>` in body
- No AI signatures or "Generated by" messages

---

## Appendix: Quick Reference

### Priority Matrix
- **Critical**: Production down, data loss risk
- **High**: Blocking other work, sprint commitment
- **Medium**: Important but not blocking
- **Low**: Nice to have, technical debt

### Component Mapping
- `core` → `accrue/core/`
- `vector-store` → `accrue/vector_store/`
- `chains` → `accrue/chains/`
- `data` → `accrue/data/`
- `testing` → `tests/`
- `infra` → CI/CD, deployment configs
- `docs` → `docs/`, README, CLAUDE.md

---

## References

- [Conventional Commits v1.0](https://www.conventionalcommits.org/)
- [Semantic Versioning 2.0](https://semver.org/)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Python PEP 8](https://peps.python.org/pep-0008/)
- [Agile Alliance Glossary](https://www.agilealliance.org/agile101/agile-glossary/)