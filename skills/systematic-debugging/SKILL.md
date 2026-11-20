---
name: systematic-debugging
description: "Root-cause debugging protocol: reproduction, fault isolation, hypothesis testing, and verification."
version: 1.0.0
author: M.A.D.D.Y.
tags: [debugging, troubleshooting, errors, traceback, diagnosis]
---

# Systematic Debugging Protocol

Structured method for diagnosing bugs, crashes, memory leaks, and unexpected behaviors without random trial-and-error.

## 4-Phase Diagnostic Cycle

### Phase 1: Minimal Reproduction
- Identify the exact command, payload, or action that triggers the fault.
- Extract the raw error traceback, exit code, and failing assertion.
- Check recent changes (`git diff`, recent file modifications) to identify regression points.

### Phase 2: Fault Isolation & Scope Bounding
- Isolate whether the error stems from:
  1. Environment / Dependency mismatch (e.g. missing package, CUDA mismatch).
  2. Input / Parameter schema violation.
  3. Logic bug (off-by-one, NoneType dereference, key error).
  4. Concurrency / Race condition / File lock contention.
- Never guess or modify multiple decoupled components at once.

### Phase 3: Falsifiable Hypothesis Formulation
- State clearly: *"If X is the root cause, then modifying Y will produce Z."*
- Inspect variables and state at the failure site using isolated scratch scripts or focused print logs.

### Phase 4: Patch & Regression Verification
- Apply the minimal targeted fix.
- Re-run the reproduction test to verify the fix works.
- Run the full test suite to ensure no unexpected regressions were introduced.
