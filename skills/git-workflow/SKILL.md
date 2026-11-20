---
name: git-workflow
description: "Safe Git operational protocol: branch hygiene, atomic commits, diff inspection, and conflict resolution."
version: 1.0.0
author: M.A.D.D.Y.
tags: [git, version-control, branches, commits, workflow]
---

# Safe Git Version Control Workflow

Guidelines for managing source control safely, ensuring atomic commits, and preventing accidental data loss.

## Core Rules
1. Never commit with generic messages (e.g. "update", "fix"). State what was changed and why.
2. Never force-push (`git push --force`) to shared branches (`main`, `master`, `prod`).
3. Always inspect `git status` and `git diff` before staging and committing.

## Standard Development Flow

### 1. Working Directory Check
- Verify current branch and clean working tree:
  `git status`
- Create a feature branch for non-trivial modifications:
  `git checkout -b feature/<feature-name>`

### 2. Pre-Commit Review
- Review exact modified lines:
  `git diff`
- Run the test suite before committing to ensure tests pass.

### 3. Atomic Commits
- Stage only relevant files:
  `git add <files>`
- Write a clear, imperative commit message:
  `git commit -m "Add feature X supporting capability Y"`
