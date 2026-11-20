---
name: web-research
description: "Web research protocol: multi-source triangulation, factual verification, and citation."
version: 1.0.0
author: M.A.D.D.Y.
platforms: [linux, windows]
category: research
tags: [web, research, citations, verification, factcheck]
dependencies: [weather, calculator]
permissions: [SAFE]
---

# Web Research & Fact Verification Protocol

Systematic procedure for researching external technical information, validating claims, and summarizing findings.

## When to Use
- User asks for factual information, technical documentation, or real-world events.
- Synthesizing answers across multiple independent web sources.
- Verifying library APIs, software releases, or benchmark numbers.

## Research Protocol

### 1. Query Formulation
- Identify key entities, version numbers, and technical terms.
- Formulate specific search queries rather than conversational sentences.

### 2. Multi-Source Triangulation
- Never rely on a single unverified blog post or discussion thread.
- Cross-reference facts across official documentation, GitHub repositories, or authoritative specifications.

### 3. Factual Grounding & Anti-Hallucination
- Quote exact version numbers, dates, and metric values from retrieved tool outputs.
- Explicitly flag any uncertainty or unverified inferences with "[unverified]".

### 4. Direct Executive Summary
- Present answers in a clear, executive structure: direct conclusion first, followed by supporting evidence and source references.
