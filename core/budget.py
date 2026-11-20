"""Per-turn iteration and resource budgeting for M.A.D.D.Y. Agent Core.
(Modular Autonomous Digital Daemon for You)

Fulfills CLAUDE.md Engineering Rule 8:
"Every execution loop must have step, time, and cost budgets."
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from core.errors import BudgetExceededError, ToolTimeoutError


class IterationBudget:
    """Thread-safe resource budget governing turn iterations, timeouts, and steps."""

    def __init__(
        self,
        max_iterations: int = 4,
        max_tool_timeout_seconds: float = 10.0,
        max_turn_timeout_seconds: float = 60.0,
    ):
        self.max_iterations = max_iterations
        self.max_tool_timeout_seconds = max_tool_timeout_seconds
        self.max_turn_timeout_seconds = max_turn_timeout_seconds

        self._used_iterations: int = 0
        self._turn_start_time: float = time.monotonic()
        self._lock = threading.Lock()

    def consume(self) -> int:
        """Consume one iteration step. Raises BudgetExceededError if limit reached."""
        with self._lock:
            # Check turn timeout
            elapsed = time.monotonic() - self._turn_start_time
            if elapsed > self.max_turn_timeout_seconds:
                raise BudgetExceededError(
                    message=f"Turn timeout exceeded: {elapsed:.2f}s > {self.max_turn_timeout_seconds}s limit.",
                    used_iterations=self._used_iterations,
                    max_iterations=self.max_iterations,
                )

            if self._used_iterations >= self.max_iterations:
                raise BudgetExceededError(
                    message=f"Iteration budget exhausted: {self._used_iterations}/{self.max_iterations} steps consumed.",
                    used_iterations=self._used_iterations,
                    max_iterations=self.max_iterations,
                )

            self._used_iterations += 1
            return self._used_iterations

    def refund(self) -> None:
        """Refund an iteration (e.g. for pure housekeeping or programmatic chaining)."""
        with self._lock:
            if self._used_iterations > 0:
                self._used_iterations -= 1

    @property
    def used(self) -> int:
        with self._lock:
            return self._used_iterations

    @property
    def remaining(self) -> int:
        with self._lock:
            return max(0, self.max_iterations - self._used_iterations)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._turn_start_time
