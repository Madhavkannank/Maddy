"""Classified and structured error taxonomy for M.A.D.D.Y. Agent Core.
(Modular Autonomous Digital Daemon for You)

Provides structured classification, failover reasons, and recovery hints so the dynamic
loop can make deterministic recovery decisions rather than matching raw string traces.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class FailoverReason(str, enum.Enum):
    """Why an operation or model call failed — determines loop recovery strategy."""
    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONTEXT_OVERFLOW = "context_overflow"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    FORMAT_ERROR = "format_error"
    MODEL_NOT_FOUND = "model_not_found"
    PERMISSION_DENIED = "permission_denied"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    DEGENERATE_LOOP = "degenerate_loop"
    BUDGET_EXCEEDED = "budget_exceeded"
    ROLE_ALTERNATION = "role_alternation"
    MALFORMED_TOOL_CALL = "malformed_tool_call"
    SKILL_ERROR = "skill_error"
    SKILL_VALIDATION = "skill_validation"
    DELEGATION_FAILED = "delegation_failed"
    UNKNOWN = "unknown"


class MaddyError(Exception):
    """Base exception for all M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ClassifiedError(MaddyError):
    """Structured classification of an error with recovery hints for the agent loop."""
    reason: FailoverReason = FailoverReason.UNKNOWN
    status_code: Optional[int] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    message: str = ""
    retryable: bool = False
    cooldown_seconds: float = 0.0
    recovery_hint: str = ""
    error_context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        super().__init__(self.message, details=self.error_context)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "reason": self.reason.value,
            "status_code": self.status_code,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
            "retryable": self.retryable,
            "cooldown_seconds": self.cooldown_seconds,
            "recovery_hint": self.recovery_hint,
            "error_context": self.error_context,
        }


class PermissionDeniedError(ClassifiedError):
    """Raised when an operation or tool violates deterministic permission checks."""
    def __init__(self, message: str, tool_name: str, tier: str, arguments: Optional[Dict[str, Any]] = None):
        super().__init__(
            reason=FailoverReason.PERMISSION_DENIED,
            message=message,
            retryable=False,
            recovery_hint="Request operator approval or alter operation to reside within safe tier parameters.",
            error_context={"tool_name": tool_name, "tier": tier, "arguments": arguments or {}}
        )


class ToolExecutionError(ClassifiedError):
    """Raised when a tool encounters an execution defect, non-zero exit code, or runtime exception."""
    def __init__(self, message: str, tool_name: str, exit_code: Optional[int] = None, stderr: str = "", stdout: str = ""):
        super().__init__(
            reason=FailoverReason.TOOL_EXECUTION_FAILED,
            message=message,
            retryable=True,
            recovery_hint="Inspect standard error output, correct invocation arguments, and retry.",
            error_context={
                "tool_name": tool_name,
                "exit_code": exit_code,
                "stderr": stderr,
                "stdout": stdout,
            }
        )


class ToolTimeoutError(ClassifiedError):
    """Raised when a tool exceeds its deterministic execution timeout budget."""
    def __init__(self, message: str, tool_name: str, timeout_seconds: float):
        super().__init__(
            reason=FailoverReason.TIMEOUT,
            message=message,
            retryable=False,
            recovery_hint="Optimize command to execute within the allotted budget or dispatch to background.",
            error_context={"tool_name": tool_name, "timeout_seconds": timeout_seconds}
        )


class DatabaseError(ClassifiedError):
    """Raised when an operation on SQLite or trace/memory persistence fails."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            reason=FailoverReason.UNKNOWN,
            message=message,
            retryable=False,
            recovery_hint="Check SQLite WAL lock, permissions, or database file integrity.",
            error_context=details or {},
        )


class DegenerateLoopError(ClassifiedError):
    """Raised when LoopGuard detects infinite tool cycle or repeating identical calls."""
    def __init__(self, message: str, pattern: str, call_hash: str, repeat_count: int):
        super().__init__(
            reason=FailoverReason.DEGENERATE_LOOP,
            message=message,
            retryable=False,
            recovery_hint="Halt repeating calls, synthesize failure explanation, or reformulate plan.",
            error_context={"pattern": pattern, "call_hash": call_hash, "repeat_count": repeat_count}
        )


class BudgetExceededError(ClassifiedError):
    """Raised when turn iteration budget or token budget is exhausted."""
    def __init__(self, message: str, used_iterations: int, max_iterations: int):
        super().__init__(
            reason=FailoverReason.BUDGET_EXCEEDED,
            message=message,
            retryable=False,
            recovery_hint="Synthesize partial results obtained so far and report state to operator.",
            error_context={"used_iterations": used_iterations, "max_iterations": max_iterations}
        )


class ModelApiError(ClassifiedError):
    """Raised when the Cloud LLM API returns an HTTP error or unreachable endpoint."""
    def __init__(self, message: str, status_code: Optional[int] = None, provider: str = "", model: str = "", retryable: bool = True):
        reason = FailoverReason.RATE_LIMIT if status_code == 429 else (
            FailoverReason.AUTH if status_code in (401, 403) else (
                FailoverReason.CONTEXT_OVERFLOW if status_code == 413 else FailoverReason.UNKNOWN
            )
        )
        super().__init__(
            reason=reason,
            status_code=status_code,
            provider=provider,
            model=model,
            message=message,
            retryable=retryable,
            recovery_hint="Wait for rate limit window, rotate credentials, or compress conversation context.",
            error_context={"status_code": status_code, "provider": provider, "model": model}
        )


class SkillError(ClassifiedError):
    """Raised when a skill operation or lifecycle execution fails."""
    def __init__(self, message: str, skill_name: str = "", error_context: Optional[Dict[str, Any]] = None):
        ctx = error_context or {}
        ctx["skill_name"] = skill_name
        super().__init__(
            reason=FailoverReason.SKILL_ERROR,
            message=message,
            retryable=False,
            recovery_hint="Inspect skill schema, verified dependencies, and instructions.",
            error_context=ctx,
        )


class SkillValidationError(ClassifiedError):
    """Raised when a skill fails frontmatter or schema validation."""
    def __init__(self, message: str, skill_name: str = "", errors: Optional[List[str]] = None):
        super().__init__(
            reason=FailoverReason.SKILL_VALIDATION,
            message=message,
            retryable=False,
            recovery_hint="Fix required YAML frontmatter fields: name, description, version.",
            error_context={"skill_name": skill_name, "validation_errors": errors or []},
        )


class SkillSecurityError(ClassifiedError):
    """Raised when a skill contains malicious payloads, dangerous shell patterns, or privilege escalation attempts."""
    def __init__(self, message: str, skill_name: str = "", security_findings: Optional[List[str]] = None):
        super().__init__(
            reason=FailoverReason.PERMISSION_DENIED,
            message=message,
            retryable=False,
            recovery_hint="Review skill file for dangerous shell pipes, secret harvesting, or permission bypass patterns.",
            error_context={"skill_name": skill_name, "security_findings": security_findings or []},
        )


class DelegationError(ClassifiedError):
    """Raised when subagent delegation or parallel task execution fails."""
    def __init__(self, message: str, task_id: str = "", error_context: Optional[Dict[str, Any]] = None):
        ctx = error_context or {}
        ctx["task_id"] = task_id
        super().__init__(
            reason=FailoverReason.DELEGATION_FAILED,
            message=message,
            retryable=True,
            recovery_hint="Inspect subagent logs, refine task scope, or adjust execution timeout.",
            error_context=ctx,
        )


class DelegationRecursionError(ClassifiedError):
    """Raised when delegation depth exceeds maximum allowed recursion limit."""
    def __init__(self, message: str, current_depth: int, max_depth: int):
        super().__init__(
            reason=FailoverReason.PERMISSION_DENIED,
            message=message,
            retryable=False,
            recovery_hint="Refactor workflow into flat parallel tasks or raise max_spawn_depth configuration.",
            error_context={"current_depth": current_depth, "max_depth": max_depth},
        )


class DelegationTimeoutError(ClassifiedError):
    """Raised when a delegated subagent exceeds its execution deadline."""
    def __init__(self, message: str, task_id: str, timeout_seconds: float):
        super().__init__(
            reason=FailoverReason.TIMEOUT,
            message=message,
            retryable=False,
            recovery_hint="Subagent exceeded time limit. Consider breaking into smaller sub-tasks or extending timeout.",
            error_context={"task_id": task_id, "timeout_seconds": timeout_seconds},
        )


class DependencyCycleError(ClassifiedError):
    """Raised when task graph contains circular dependencies."""
    def __init__(self, message: str, cycle: Optional[List[str]] = None):
        super().__init__(
            reason=FailoverReason.DELEGATION_FAILED,
            message=message,
            retryable=False,
            recovery_hint="Eliminate cyclic dependency chain in delegated task graph definition.",
            error_context={"cycle": cycle or []},
        )


ValidationError = SkillValidationError

