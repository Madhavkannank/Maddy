"""Repetition Guard for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Detects degenerate repetition loops in model outputs before they consume the token budget.
"""
from __future__ import annotations

import math
import re
from typing import Optional

MIN_FRAGMENT_LENGTH = 300
_REPEAT_WINDOW = 50
_MIN_REPEAT_COUNT = 4
_DOMINANCE_RATIO = 0.5

REPETITION_LOOP_INTERRUPTED = "[the reply degenerated into a repetition loop and was interrupted]"


def is_repetition_dominated(text: str) -> bool:
    """
    True when a contiguous run of repeated text covers at least half of the text,
    indicating an LLM generation failure or pathological loop.
    """
    if not isinstance(text, str):
        return False
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return False

    # 1. Line repetition check (the most common echo pattern)
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) >= 15]
    if len(lines) >= 4:
        from collections import Counter
        counts = Counter(lines)
        most_common_line, count = counts.most_common(1)[0]
        if count >= 3 and (count * len(most_common_line)) >= (n * _DOMINANCE_RATIO):
            return True

    # 2. Window count check (detects recurring identical chunks)
    window = _REPEAT_WINDOW
    needed = max(_MIN_REPEAT_COUNT, math.ceil(n * _DOMINANCE_RATIO / window))
    counts_map: dict[str, int] = {}
    for i in range(n - window + 1):
        key = text[i : i + window]
        c = counts_map.get(key, 0) + 1
        if c >= needed:
            return True
        counts_map[key] = c

    return False


def clean_repetition_if_dominated(text: str) -> str:
    """
    If the text is repetition dominated, truncates at the first detected repetition
    and appends a clean notification tag.
    """
    if not is_repetition_dominated(text):
        return text

    # Truncate to reasonable first-half content
    safe_cut = min(len(text), max(100, int(len(text) * 0.35)))
    truncated = text[:safe_cut].rsplit(".", 1)[0]
    if not truncated:
        truncated = text[:safe_cut]
    return f"{truncated.strip()}\n\n{REPETITION_LOOP_INTERRUPTED}"
