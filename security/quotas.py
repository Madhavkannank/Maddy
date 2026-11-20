"""
Resource Quotas and Execution Safeguards Engine for M.A.D.D.Y.
Enforces resource limits to protect disk, memory, process table, and CPU
from runaway tasks, concurrent worker exhaustion, and denial-of-service.
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ResourceQuotaConfig:
    """Configurable limits for resource consumption."""
    max_file_write_bytes: int = 50 * 1024 * 1024       # 50 MB per file write
    max_task_write_bytes: int = 200 * 1024 * 1024      # 200 MB aggregate per task
    max_concurrent_subprocesses: int = 5               # Max child processes
    max_tool_calls_per_minute: int = 120               # Rate limit per minute
    max_execution_timeout_seconds: float = 60.0        # Max execution time


@dataclass
class QuotaCheckResult:
    allowed: bool
    resource_type: str
    current_usage: float
    limit: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "resource_type": self.resource_type,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "reason": self.reason,
        }


class ResourceQuotaManager:
    """
    Thread-safe quota enforcement engine tracking per-task and aggregate system consumption.
    """

    def __init__(self, config: Optional[ResourceQuotaConfig] = None):
        self.config = config or ResourceQuotaConfig()
        self._lock = threading.RLock()

        # Task write accounting: task_id -> total_bytes_written
        self._task_write_usage: Dict[str, int] = {}

        # Subprocess accounting: task_id -> current_process_count
        self._active_subprocesses: Dict[str, int] = {}

        # Rate limiting: list of (timestamp, task_id)
        self._tool_call_history: List[float] = []

    def check_file_write(self, bytes_to_write: int, task_id: Optional[str] = None) -> QuotaCheckResult:
        """
        Verify that writing `bytes_to_write` respects both per-operation and per-task aggregate limits.
        """
        with self._lock:
            # 1. Per-operation limit
            if bytes_to_write > self.config.max_file_write_bytes:
                return QuotaCheckResult(
                    allowed=False,
                    resource_type="file_write_size",
                    current_usage=bytes_to_write,
                    limit=self.config.max_file_write_bytes,
                    reason=f"File write of {bytes_to_write} bytes exceeds max single-file limit of {self.config.max_file_write_bytes} bytes",
                )

            # 2. Per-task aggregate limit
            tid = task_id or "default"
            current_task_usage = self._task_write_usage.get(tid, 0)
            projected = current_task_usage + bytes_to_write

            if projected > self.config.max_task_write_bytes:
                return QuotaCheckResult(
                    allowed=False,
                    resource_type="task_aggregate_write",
                    current_usage=current_task_usage,
                    limit=self.config.max_task_write_bytes,
                    reason=f"Write of {bytes_to_write} bytes would exceed aggregate task limit of {self.config.max_task_write_bytes} bytes (current: {current_task_usage} bytes)",
                )

            return QuotaCheckResult(
                allowed=True,
                resource_type="file_write",
                current_usage=current_task_usage,
                limit=self.config.max_task_write_bytes,
                reason="File write within quota limits",
            )

    def record_file_write(self, bytes_written: int, task_id: Optional[str] = None) -> None:
        """Update task write counter upon successful file write."""
        with self._lock:
            tid = task_id or "default"
            self._task_write_usage[tid] = self._task_write_usage.get(tid, 0) + max(0, bytes_written)

    def acquire_subprocess_slot(self, task_id: Optional[str] = None) -> QuotaCheckResult:
        """
        Reserve a subprocess slot.
        """
        with self._lock:
            total_active = sum(self._active_subprocesses.values())
            if total_active >= self.config.max_concurrent_subprocesses:
                return QuotaCheckResult(
                    allowed=False,
                    resource_type="concurrent_subprocesses",
                    current_usage=total_active,
                    limit=self.config.max_concurrent_subprocesses,
                    reason=f"Concurrent subprocess count ({total_active}) has reached maximum quota ({self.config.max_concurrent_subprocesses})",
                )

            tid = task_id or "default"
            self._active_subprocesses[tid] = self._active_subprocesses.get(tid, 0) + 1
            return QuotaCheckResult(
                allowed=True,
                resource_type="concurrent_subprocesses",
                current_usage=total_active + 1,
                limit=self.config.max_concurrent_subprocesses,
                reason="Subprocess slot acquired",
            )

    def release_subprocess_slot(self, task_id: Optional[str] = None) -> None:
        """Release a subprocess slot when process terminates."""
        with self._lock:
            tid = task_id or "default"
            curr = self._active_subprocesses.get(tid, 0)
            if curr > 0:
                self._active_subprocesses[tid] = curr - 1

    def check_rate_limit(self) -> QuotaCheckResult:
        """
        Enforce tool invocation rate limit per minute.
        """
        with self._lock:
            now = time.time()
            cutoff = now - 60.0
            self._tool_call_history = [t for t in self._tool_call_history if t >= cutoff]

            if len(self._tool_call_history) >= self.config.max_tool_calls_per_minute:
                return QuotaCheckResult(
                    allowed=False,
                    resource_type="tool_rate_limit",
                    current_usage=len(self._tool_call_history),
                    limit=self.config.max_tool_calls_per_minute,
                    reason=f"Tool call rate limit exceeded ({len(self._tool_call_history)} calls in past 60s, limit {self.config.max_tool_calls_per_minute})",
                )

            self._tool_call_history.append(now)
            return QuotaCheckResult(
                allowed=True,
                resource_type="tool_rate_limit",
                current_usage=len(self._tool_call_history),
                limit=self.config.max_tool_calls_per_minute,
                reason="Rate limit acceptable",
            )

    def reset_task(self, task_id: str) -> None:
        """Reset quota accounting for a finished task."""
        with self._lock:
            self._task_write_usage.pop(task_id, None)
            self._active_subprocesses.pop(task_id, None)
