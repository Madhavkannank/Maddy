---
name: grounded-research
description: "Web research and literature review protocol: source evaluation, factual grounding, and claim verification."
version: 1.0.0
author: M.A.D.D.Y.
tags: [research, web, arxiv, citations, verification]
---

# Grounded Research & Verification Protocol

Protocol for conducting web and paper research with zero hallucination, strict factual grounding, and provenance tracking.

## Core Rules
1. Never assert a factual claim without grounding it in retrieved tool output (`browser_search`, `arxiv_search`, `browser_extract`).
2. Distinguish between verified empirical facts and speculative claims.
3. Keep spoken summaries concise and free of raw URLs, citations, or markdown artifacts.

## Research Workflow

### 1. Broad Search & Triangulation
- Search for the query using precise technical keywords.
- Compare at least two distinct sources when verifying API changes, benchmark scores, or release dates.

### 2. Primary Source Inspection
- If documentation or paper abstracts are available, prefer the author's primary source over blog summaries.
- Extract concrete data points: exact package names, versions, benchmark numbers, and function signatures.

### 3. Synthesis & Presentation
- Package results into an executive summary followed by key findings and actionable next steps.
- Present spoken answers with natural, direct language.
