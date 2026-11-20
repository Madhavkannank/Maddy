---
name: codebase-inspection
description: "Inspect codebases: LOC, directory architecture, file composition, and dependency trees."
version: 1.0.0
author: M.A.D.D.Y.
tags: [codebase, analysis, architecture, loc, inspection]
---

# Codebase Architecture & Inspection

Guidelines and procedures for analyzing unfamiliar codebases, auditing architecture, and summarizing project composition.

## When to Use
- User asks about the structure or layout of a repository.
- Auditing file count, lines of code, or programming languages in a project.
- Identifying core entrypoints, configuration files, and key abstractions before modifying code.

## Systematic Inspection Protocol

### 1. High-Level Inventory
- Check `README.md`, `pyproject.toml`, `package.json`, or root config files.
- Inspect top-level directory structure using `filesystem_search` or directory listing.
- Identify the project type: CLI, Web Service (FastAPI/Flask/Express), Frontend (React/Vite), Library, or ML Pipeline.

### 2. Dependency Audit
- Check pinned dependencies in `pyproject.toml`, `requirements.txt`, or `package.json`.
- Verify runtime constraints (Python version, CUDA requirements, native C/C++ libraries).

### 3. Architecture & Entrypoint Mapping
- Locate the main system entrypoints (e.g. `main.py`, `app.py`, `cli.py`, `index.ts`).
- Trace data flow from user entrypoint to business logic and data persistence.
- Record key architectural invariants in working memory before writing code.
