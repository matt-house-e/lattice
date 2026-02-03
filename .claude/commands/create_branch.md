---
description: Create a branch from a GitHub issue
accepts_args: true
---

Create a git branch for issue #$ARGUMENTS.

Follow these steps:

1. **Fetch the issue details**:
   ```bash
   gh issue view $ARGUMENTS --json title,labels,number
   ```

2. **Parse the issue type** from the title prefix or labels:
   - Title starts with `[Epic]:` or has `type:epic` label → `epic/`
   - Title starts with `[Story]:` or has `type:story` label → `feat/`
   - Title starts with `[Task]:` or has `type:task` label → `task/`
   - Title starts with `[Bug]:` or has `type:bug` label → `fix/`
   - Title starts with `[Spike]:` or has `type:spike` label → `spike/`
   - Default → `feat/`

3. **Extract description** from the title:
   - Remove the `[Type]: [Component]` prefix
   - Convert to lowercase
   - Replace spaces with hyphens
   - Remove special characters
   - Truncate to ~40 chars if needed

4. **Format the branch name**:
   ```
   [type]/[issue-number]-[brief-description]
   ```
   Examples:
   - `feat/45-add-file-upload`
   - `fix/89-conversation-node-crash`
   - `task/67-refactor-llm-service`

5. **Create and checkout the branch**:
   ```bash
   git checkout -b <branch-name>
   ```

6. **Output confirmation** with the branch name.
