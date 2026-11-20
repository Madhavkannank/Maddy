---
name: code-review
description: "Static code review: defect analysis, security vulnerability scan, and style check."
version: 1.0.0
author: M.A.D.D.Y.
platforms: [linux, windows]
category: coding
tags: [code, review, security, quality, pep8, architecture]
dependencies: [read_file]
permissions: [SAFE]
---

# Code Review & Quality Assurance Protocol

Structured procedure for conducting static code reviews, finding software defects, and enforcing security invariants.

## When to Use
- Reviewing pull requests, git diffs, or newly implemented functions.
- Auditing code for OWASP security vulnerabilities (SQL injection, path traversal, shell injection).
- Ensuring compliance with project conventions and PEP 8 standards.

## Review Protocol

### 1. Invariant & Boundary Verification
- Check function argument boundaries and null/None handling.
- Verify that resources (files, database connections, sockets) are wrapped in context managers (`with` statements).
- Check that error handling does not swallow unexpected exceptions with bare `except: pass`.

### 2. Security & Sandboxing Audit
- Ensure user-provided paths are resolved using `.resolve()` and checked against allowed directory bounds.
- Ensure terminal commands never concatenate unsanitized user strings into `shell=True`.
- Verify secrets, credentials, and API keys are redacted and never logged in plain text.

### 3. Architecture & Dependency Check
- Verify that changes adhere to the project's layered execution flow (`Plan → Act → Observe → Verify → Replan`).
- Ensure no unexpected third-party frameworks or unapproved dependencies are introduced.

### 4. Constructive Reporting
- Rank verified findings by severity: Critical, High, Medium, Low.
- Provide concrete line references and proposed minimal diff fixes.
