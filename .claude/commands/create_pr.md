---
description: Create a pull request following project standards
accepts_args: true
---

Create a pull request for issue #$ARGUMENTS (or use current branch context if no issue provided).

Follow these steps:

1. **If an issue number is provided**, fetch issue details:
   ```bash
   gh issue view $ARGUMENTS --json title,labels,body
   ```

2. **Determine the commit type** based on issue type:
   - `type:bug` → `fix`
   - `type:story` → `feat`
   - `type:task` → `chore`
   - `type:epic` → `feat`
   - `type:spike` → `docs` or `chore`
   - Default → `feat`

3. **Determine scope** from the component label or branch name:
   - `component:core` → `core`
   - `component:vector-store` → `vector-store`
   - `component:chains` → `chains`
   - `component:data` → `data`
   - `component:testing` → `test`
   - `component:docs` → `docs`
   - `component:infra` → `infra`

4. **Format the PR title** following Conventional Commits:
   ```
   <type>(<scope>): <description> (#issue-number)
   ```
   Example: `feat(core): add retry logic to enricher (#123)`

5. **Review changes** to summarize:
   ```bash
   git diff main...HEAD --stat
   git log main..HEAD --oneline
   ```

6. **Create the PR** using `gh pr create` with HEREDOC body:
   ```bash
   gh pr create --title "<type>(<scope>): <description> (#issue)" \
     --body "$(cat <<'EOF'
   ## Summary
   [Brief description of what this PR does]

   ## Changes
   - [Change 1]
   - [Change 2]

   ## Testing
   [How was this tested?]

   ## Checklist
   - [ ] Tests pass
   - [ ] Linting/formatting passes
   - [ ] Self-reviewed
   - [ ] Documentation updated (if applicable)

   Closes #<issue-number>
   EOF
   )"
   ```

7. **Output the PR URL** after creation.
