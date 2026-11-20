"""Degenerate Loop Guard for M.A.D.D.Y. Agent Core.
(Modular Autonomous Digital Daemon for You).

Detects and halts infinite re-execution loops, consecutive identical calls, ping-pong cycles,
and model-output runaway text repetition loops.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.errors import DegenerateLoopError

# Repetition guard constants
_REPEAT_WINDOW = 60
_MIN_REPEAT_COUNT = 5
_DOMINANCE_RATIO = 0.5
MIN_REPETITION_FRAGMENT_LENGTH = 300


def normalize_arguments(obj: Any) -> Any:
    """Recursively normalize dictionary/list arguments, stripping trivial formatting mutations.

    Normalizes whitespace in strings, collapses redundant spaces, and sorts dictionary keys.
    """
    if isinstance(obj, dict):
        return {str(k): normalize_arguments(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [normalize_arguments(v) for v in obj]
    elif isinstance(obj, str):
        # Collapse multi-whitespace and strip leading/trailing blanks
        clean = re.sub(r"\s+", " ", obj.strip())
        # Normalize relative path prefixes e.g. "./file.py" -> "file.py"
        if clean.startswith("./") and len(clean) > 2:
            clean = clean[2:]
        return clean
    return obj


def is_runaway_repetition(text: str) -> bool:
    """Detect if model text has degenerated into a runaway repetition loop.

    Triggers when:
    1. A single normalized line repeats >= 5 times and covers >= 50% of text.
    2. A 60-character substring recurs often enough to cover >= 50% of text.
    """
    if not isinstance(text, str) or len(text) < MIN_REPETITION_FRAGMENT_LENGTH:
        return False

    n = len(text)

    # 1. Fast line-based repetition check
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        counts = Counter(lines)
        for line, count in counts.items():
            if count >= _MIN_REPEAT_COUNT and (count * len(line)) >= (n * _DOMINANCE_RATIO):
                return True

    # 2. Sliding window repetition check
    window = _REPEAT_WINDOW
    if n >= window:
        needed = max(_MIN_REPEAT_COUNT, math.ceil(n * _DOMINANCE_RATIO / window))
        window_counts: Dict[str, int] = {}
        for i in range(n - window + 1):
            chunk = text[i : i + window]
            c = window_counts.get(chunk, 0) + 1
            if c >= needed:
                return True
            window_counts[chunk] = c

    return False


@dataclass
class ToolCallSignature:
    """Represents a unique tool call invocation signature for loop guard analysis."""
    tool_name: str
    arguments: Dict[str, Any]
    call_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize tool call signature to a dictionary."""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "call_hash": self.call_hash or LoopGuard.compute_call_hash(self.tool_name, self.arguments),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ToolCallSignature:
        """Deserialize tool call signature from a dictionary."""
        return cls(
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            call_hash=data.get("call_hash"),
        )


class LoopGuard:
    """Detects and halts repetitive tool cycles and degenerate execution loops."""

    def __init__(
        self,
        max_identical_calls: int = 3,
        max_cycle_period: int = 4,
        history_window_size: int = 32,
    ):
        self.max_identical_calls = max_identical_calls
        self.max_cycle_period = max_cycle_period
        self.history_window_size = history_window_size

        # Streak tracking for consecutive identical calls
        self._last_call_hash: Optional[str] = None
        self._identical_streak: int = 0

        # Call history deque for cyclic pattern detection
        self._call_history: deque[str] = deque(maxlen=history_window_size)
        self._call_signatures: deque[str] = deque(maxlen=history_window_size)

    @staticmethod
    def compute_call_hash(tool_name: str, arguments: Dict[str, Any]) -> str:
        """Compute deterministic SHA-256 hash of a tool call with normalized arguments."""
        normalized = normalize_arguments(arguments or {})
        canonical_args = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        raw = f"{tool_name}:{canonical_args}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def check_model_text(self, text: str) -> None:
        """Check if model output has degraded into runaway repetition loop.

        Raises:
            DegenerateLoopError: If runaway repetition is detected.
        """
        if is_runaway_repetition(text):
            raise DegenerateLoopError(
                message="Degenerate model output halted: runaway token/line repetition loop detected.",
                pattern="runaway_text_repetition",
                call_hash=None,
                repeat_count=_MIN_REPEAT_COUNT,
            )

    def record_and_check(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Register a planned tool call and verify it does not violate anti-loop policies.

        Raises:
            DegenerateLoopError: If identical calls repeat >= max_identical_calls or a cycle is detected.
        """
        call_hash = self.compute_call_hash(tool_name, arguments)
        signature = f"{tool_name}({','.join(sorted(arguments.keys())) if arguments else ''})"

        # 1. Consecutive identical call check
        if call_hash == self._last_call_hash:
            self._identical_streak += 1
            if self._identical_streak >= self.max_identical_calls:
                raise DegenerateLoopError(
                    message=f"Degenerate loop halted: tool '{tool_name}' invoked with identical arguments {self._identical_streak} times consecutively.",
                    pattern="consecutive_identical",
                    call_hash=call_hash,
                    repeat_count=self._identical_streak,
                )
        else:
            self._last_call_hash = call_hash
            self._identical_streak = 1

        # 2. Cycle detection across history window (periods 2, 3, 4)
        history = list(self._call_history)
        history.append(call_hash)

        for period in range(2, self.max_cycle_period + 1):
            if len(history) >= period * 2:
                tail1 = history[-period:]
                tail2 = history[-2 * period : -period]
                if tail1 == tail2:
                    # Detected cycle repetition
                    pattern_names = list(self._call_signatures)[-period:]
                    pattern_str = " -> ".join(pattern_names)
                    raise DegenerateLoopError(
                        message=f"Degenerate cycle detected: repeating sequence [{pattern_str}] across {period}-step boundary.",
                        pattern=f"cycle_period_{period}",
                        call_hash=call_hash,
                        repeat_count=2,
                    )

        # Update history
        self._call_history.append(call_hash)
        self._call_signatures.append(signature)

    @property
    def streak_count(self) -> int:
        """Consecutive identical tool call streak count."""
        return self._identical_streak

    @streak_count.setter
    def streak_count(self, value: int) -> None:
        self._identical_streak = value

    def to_dict(self) -> Dict[str, Any]:
        """Serialize LoopGuard state to a JSON-compatible dictionary."""
        return {
            "max_identical_calls": self.max_identical_calls,
            "max_cycle_period": self.max_cycle_period,
            "history_window_size": self.history_window_size,
            "_last_call_hash": self._last_call_hash,
            "last_call_hash": self._last_call_hash,
            "_identical_streak": self._identical_streak,
            "streak_count": self._identical_streak,
            "_call_history": list(self._call_history),
            "_call_signatures": list(self._call_signatures),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LoopGuard:
        """Deserialize LoopGuard state from a dictionary."""
        if not data:
            return cls()
        guard = cls(
            max_identical_calls=data.get("max_identical_calls", 3),
            max_cycle_period=data.get("max_cycle_period", 4),
            history_window_size=data.get("history_window_size", 32),
        )
        guard._last_call_hash = data.get("_last_call_hash", data.get("last_call_hash"))
        streak = data.get("_identical_streak")
        if streak is None:
            streak = data.get("identical_streak", data.get("_streak_count", data.get("streak_count", 0)))
        guard._identical_streak = int(streak)

        call_hist = data.get("_call_history", data.get("call_history", []))
        guard._call_history = deque(call_hist, maxlen=guard.history_window_size)

        call_sigs = data.get("_call_signatures", data.get("call_signatures", []))
        guard._call_signatures = deque(call_sigs, maxlen=guard.history_window_size)

        return guard

    def reset_streak_on_progress(self) -> None:
        """Invoked when a mutating tool succeeds, indicating genuine state progress."""
        self._identical_streak = 0
