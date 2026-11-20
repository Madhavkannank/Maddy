"""Data models and configurations for M.A.D.D.Y. Delegation and Subagent Orchestration.

Provides typed definitions for subagents, task graph nodes, and aggregated execution results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SubagentRole(str, Enum):
    """Execution role and capability scope for a subagent."""
    LEAF = "leaf"            # Stripped of delegation tools; pure execution worker
    SPECIALIST = "specialist"  # Domain-specific executor (e.g. research, coding)
    COORDINATOR = "coordinator" # Allowed to further spawn subagents up to MAX_SPAWN_DEPTH


class SubagentStatus(str, Enum):
    """Lifecycle execution status of a delegated subagent task."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class SubagentTask:
    """Specification of an isolated subagent execution unit."""
    id: str
    description: str
    context: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    role: SubagentRole = SubagentRole.LEAF
    timeout_seconds: float = 120.0
    required_tools: Optional[List[str]] = None
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_id: str = "") -> SubagentTask:
        """Construct SubagentTask from dictionary representation (e.g. tool call args)."""
        task_id = str(data.get("id") or default_id or f"subtask_{int(time.time()*1000)}")
        description = str(data.get("task") or data.get("description", "")).strip()
        context = data.get("context")
        deps = data.get("dependencies", [])
        if isinstance(deps, str):
            deps = [deps]
        elif not isinstance(deps, list):
            deps = []

        raw_role = str(data.get("role", "leaf")).lower().strip()
        role = SubagentRole.LEAF
        for r in SubagentRole:
            if r.value == raw_role:
                role = r
                break

        timeout = float(data.get("timeout") or data.get("timeout_seconds") or 120.0)
        req_tools = data.get("required_tools")
        if isinstance(req_tools, str):
            req_tools = [req_tools]

        return cls(
            id=task_id,
            description=description,
            context=str(context) if context else None,
            dependencies=[str(d) for d in deps],
            role=role,
            timeout_seconds=timeout,
            required_tools=req_tools,
            priority=int(data.get("priority", 1)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "context": self.context,
            "dependencies": self.dependencies,
            "role": self.role.value,
            "timeout_seconds": self.timeout_seconds,
            "required_tools": self.required_tools,
            "priority": self.priority,
            "metadata": self.metadata,
        }


@dataclass
class SubagentResult:
    """Structured outcome returned by a delegated subagent execution."""
    task_id: str
    status: SubagentStatus
    summary: str = ""
    total_steps: int = 0
    duration_seconds: float = 0.0
    error: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    output_data: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status == SubagentStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "summary": self.summary,
            "total_steps": self.total_steps,
            "duration_seconds": round(self.duration_seconds, 3),
            "error": self.error,
            "tool_calls": self.tool_calls,
            "output_data": self.output_data,
        }


@dataclass
class DelegationConfig:
    """Operational parameters and guardrails for delegation coordinator."""
    max_concurrent_workers: int = 4
    max_spawn_depth: int = 3
    default_timeout_seconds: float = 120.0
    max_subagent_steps: int = 8
    truncate_max_chars: int = 2000
    truncate_head_chars: int = 1200
    truncate_tail_chars: int = 500
