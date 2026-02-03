---
description: Create a GitHub issue following project standards
accepts_args: true
---

Create a GitHub issue based on the following description: $ARGUMENTS

Follow these steps:

1. **Analyze the description** to determine:
   - **Type**: epic (multi-sprint), story (user-facing), task (technical), bug (defect), or spike (research)
   - **Component**: core, vector-store, chains, data, testing, docs, or infra
   - **Priority**: critical, high, medium, or low (default to medium if unclear)

2. **Format the title** as: `[Type]: [Component] Description` (max 60 chars)
   - Type should be capitalized: Epic, Story, Task, Bug, Spike
   - Component prefixes: [Core], [VectorStore], [Chains], [Data], [Test], [Docs], [Infra]

3. **Determine labels** (apply all that match):
   - Type label: `type:epic`, `type:story`, `type:task`, `type:bug`, or `type:spike`
   - Priority label: `priority:critical`, `priority:high`, `priority:medium`, or `priority:low`
   - Component label: `component:core`, `component:vector-store`, `component:chains`, `component:data`, `component:testing`, `component:docs`, or `component:infra`

4. **Create the issue** using `gh issue create` with a HEREDOC body:

```bash
gh issue create --title "[Type]: [Component] Description" \
  --label "type:xxx,priority:xxx,component:xxx" \
  --body "$(cat <<'EOF'
## Context
[Explain why this issue exists and what problem it solves]

## Acceptance Criteria
- [ ] [Specific, measurable criterion 1]
- [ ] [Specific, measurable criterion 2]

## Technical Notes
[Implementation hints, constraints, or dependencies]
EOF
)"
```

5. **Output the issue URL** after creation.
