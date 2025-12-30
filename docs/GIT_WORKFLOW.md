# Git Workflow & Branching Strategy

## Branches

| Branch | Purpose | Who Deploys |
|--------|---------|-------------|
| `main` | Production-ready code | CI/CD auto-deploys |
| `develop` | Integration branch | Manual testing |
| `feature/*` | New features | Merged to develop |
| `fix/*` | Bug fixes | Merged to develop/main |

## Workflow

```
feature/add-new-rule    fix/ocr-crash
        │                    │
        └──────┬─────────────┘
               ▼
           develop ──────────► PR Review ──────► main ──► Deploy
```

## Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Adding tests
- `refactor:` Code refactoring
- `chore:` Maintenance

Example:
```
feat: add allergen detection for tree nuts
fix: OCR failing on blurry images
docs: update deployment guide
```

## Pull Request Rules

1. **All tests must pass** - CI blocks merge on failure
2. **Medical rule tests required** - Extra scrutiny for safety
3. **Minimum 1 review** - For main branch
4. **No force push to main** - Preserve history

## Quick Commands

```bash
# Create feature branch
git checkout -b feature/my-feature develop

# Push and create PR
git push -u origin feature/my-feature

# Update from develop
git fetch origin
git rebase origin/develop

# Merge to develop (after PR approval)
git checkout develop
git merge --no-ff feature/my-feature
```
