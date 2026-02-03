# Git Workflow Guide

Simple, practical Git workflow for this project.

## Branch Strategy

- **`main`** - Production-ready code
- **`feature/description`** - Feature branches for new capabilities

Keep it simple. No develop branch needed.

## Commit Message Format

```
type: Brief description of change

- Specific change 1
- Specific change 2
- Why this matters/benefit

🤖 Generated with [Claude Code](https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:**
- `feat` - New features
- `fix` - Bug fixes  
- `docs` - Documentation only
- `refactor` - Code restructuring
- `test` - Adding tests

## Pull Request Process

1. **Create feature branch:**
   ```bash
   git checkout -b feature/web-search
   ```

2. **Work and commit:**
   ```bash
   git add .
   git commit -m "feat: add web search integration"
   ```

3. **Push and create PR:**
   ```bash
   git push origin feature/web-search
   ```

4. **Manual review and merge** - No automated gates

## Local Testing Before PR

Basic checks (run if helpful):
```bash
# Test functionality
python examples/test_web_enrichment.py --quick

# Check imports work
python -c "import lattice; print('✅ Package imports')"
```

## What Gets Committed

✅ **Always commit:**
- Source code (`lattice/`)
- Tests (`tests/`)
- Examples (`examples/`)
- Documentation (`.md` files)

❌ **Never commit:**
- Generated data files (`data/`)
- Environment files (`.env`)
- IDE files (`.vscode/`, `.idea/`)

## Release Process

1. **Update version** in `lattice/__init__.py` and `pyproject.toml`
2. **Test locally** - run examples
3. **Commit and tag:**
   ```bash
   git commit -m "release: v0.2.1"
   git tag v0.2.1
   git push origin main --tags
   ```

Manual releases only. No automation until needed.

## GitHub Issue Standards

**ALWAYS include labels when creating issues.** Use this format:

```bash
gh issue create --title "[Type]: [Component] Description" \
  --label "type:task,priority:medium,component:core" \
  --body "..."
```

### Required Labels (pick one from each category)

**Type** (required):
- `type:epic` - Multi-sprint initiative
- `type:story` - User-facing feature
- `type:task` - Technical work
- `type:bug` - Defect
- `type:spike` - Research/investigation

**Priority** (required):
- `priority:critical` - Production down
- `priority:high` - Blocking other work
- `priority:medium` - Important but not blocking
- `priority:low` - Nice to have

**Component** (1+ required):
- `component:core` - Config, processors, enricher
- `component:chains` - LLM chains and prompts
- `component:data` - Data models and fields
- `component:vector-store` - Vector store
- `component:testing` - Test infrastructure
- `component:docs` - Documentation
- `component:infra` - CI/CD

### Issue Title Format

```
[Type]: [Component] Description (max 60 chars)
```

Examples:
- `Task: [Chains] Migrate to Pydantic structured outputs`
- `Bug: [Core] Checkpoint resume fails on empty DataFrame`
- `Story: [Data] Add CSV field validation`

See `docs/github-standards.md` for full details.