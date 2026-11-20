"""
Task data structures and state definitions for M.A.D.D.Y. Task Graph Engine.
Supports durable long-running task execution, deterministic state transitions,
execution identity, and crash recovery.
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

class TaskState(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    WAITING = "waiting"
    VERIFYING = "verifying"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"
    BLOCKED = "blocked"
    PAUSED = "paused"
    INTERRUPTED = "interrupted"
    TIMED_OUT = "timed_out"

# Terminal vs Non-terminal state sets
TERMINAL_STATES: Set[TaskState] = {
    TaskState.SUCCESS,
    TaskState.FAILED,
    TaskState.CANCELLED,
    TaskState.BLOCKED,
}

NON_TERMINAL_STATES: Set[TaskState] = {
    TaskState.PENDING,
    TaskState.READY,
    TaskState.RUNNING,
    TaskState.WAITING,
    TaskState.VERIFYING,
    TaskState.RETRYING,
    TaskState.PAUSED,
    TaskState.INTERRUPTED,
    TaskState.TIMED_OUT,
}

# Deterministic State Transition Matrix
VALID_TRANSITIONS: Dict[TaskState, Set[TaskState]] = {
    TaskState.PENDING: {
        TaskState.READY,
        TaskState.RUNNING,
        TaskState.SUCCESS,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.BLOCKED,
        TaskState.PAUSED,
    },
    TaskState.READY: {
        TaskState.RUNNING,
        TaskState.SUCCESS,
        TaskState.FAILED,
        TaskState.PAUSED,
        TaskState.CANCELLED,
        TaskState.BLOCKED,
    },
    TaskState.RUNNING: {
        TaskState.VERIFYING,
        TaskState.SUCCESS,
        TaskState.FAILED,
        TaskState.PAUSED,
        TaskState.INTERRUPTED,
        TaskState.TIMED_OUT,
        TaskState.CANCELLED,
        TaskState.RETRYING,
    },
    TaskState.WAITING: {
        TaskState.READY,
        TaskState.RUNNING,
        TaskState.CANCELLED,
        TaskState.FAILED,
        TaskState.TIMED_OUT,
        TaskState.PAUSED,
    },
    TaskState.VERIFYING: {
        TaskState.SUCCESS,
        TaskState.FAILED,
        TaskState.RETRYING,
        TaskState.INTERRUPTED,
        TaskState.TIMED_OUT,
        TaskState.PAUSED,
        TaskState.CANCELLED,
    },
    TaskState.RETRYING: {
        TaskState.READY,
        TaskState.RUNNING,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.PAUSED: {
        TaskState.READY,
        TaskState.PENDING,
        TaskState.RUNNING,
        TaskState.CANCELLED,
        TaskState.FAILED,
    },
    TaskState.INTERRUPTED: {
        TaskState.READY,
        TaskState.RUNNING,
        TaskState.VERIFYING,
        TaskState.FAILED,
        TaskState.CANCELLED,
        TaskState.RETRYING,
        TaskState.PAUSED,
    },
    TaskState.TIMED_OUT: {
        TaskState.RETRYING,
        TaskState.READY,
        TaskState.RUNNING,
        TaskState.FAILED,
        TaskState.CANCELLED,
    },
    TaskState.BLOCKED: {
        TaskState.READY,
        TaskState.PENDING,
        TaskState.CANCELLED,
        TaskState.FAILED,
    },
    TaskState.FAILED: {
        TaskState.READY,
        TaskState.RETRYING,
        TaskState.PENDING,
    },
    TaskState.CANCELLED: {
        TaskState.READY,
        TaskState.PENDING,
    },
    TaskState.SUCCESS: {
        TaskState.READY,
    },
}

@dataclass
class Task:
    """
    Representation of a single task node in the Task Graph with full execution state
    and durability properties for crash resumption.
    """
    id: str
    description: str
    parent_id: Optional[str] = None
    status: TaskState = TaskState.PENDING
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    assigned_worker: Optional[str] = None
    required_tools: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    last_progress_at: Optional[float] = None
    timeout_seconds: Optional[float] = None
    current_step_id: Optional[str] = None
    current_action_id: Optional[str] = None
    current_attempt_id: Optional[str] = None
    verification_state: Dict[str, Any] = field(default_factory=dict)
    recovery_state: Dict[str, Any] = field(default_factory=dict)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_terminal(self) -> bool:
        """Returns True if the task is in a terminal state."""
        return self.status in TERMINAL_STATES

    def is_timed_out(self, current_time: Optional[float] = None, default_timeout: Optional[float] = None) -> bool:
        """
        Check if the task has exceeded its timeout deadline.
        Respects task-specific timeout_seconds if defined, otherwise falls back to default_timeout.
        Checks against started_at or last_progress_at.
        """
        if self.is_terminal():
            return False

        timeout = self.timeout_seconds if self.timeout_seconds is not None else default_timeout
        if timeout is None or timeout <= 0:
            return False

        now = current_time if current_time is not None else time.time()
        # Measure from last_progress_at if available, else started_at
        baseline = self.last_progress_at or self.started_at
        if baseline is None:
            return False

        return (now - baseline) > timeout

    def can_transition_to(self, target_state: TaskState) -> bool:
        """Checks if transitioning from current status to target_state is valid."""
        allowed = VALID_TRANSITIONS.get(self.status, set())
        return target_state in allowed

    def transition_to(self, target_state: TaskState, force: bool = False) -> None:
        """
        Transitions task to target_state with deterministic validation.
        Raises ValueError if the transition is illegal unless force=True.
        """
        if not force and not self.can_transition_to(target_state) and target_state != self.status:
            raise ValueError(
                f"Invalid TaskState transition: Cannot transition task '{self.id}' "
                f"from {self.status.value} to {target_state.value}"
            )
        now = time.time()
        self.status = target_state
        self.updated_at = now

        if target_state == TaskState.RUNNING and self.started_at is None:
            self.started_at = now
            self.last_progress_at = now
        elif target_state in TERMINAL_STATES and self.completed_at is None:
            self.completed_at = now

    def mark_progress(self, step_id: Optional[str] = None, action_id: Optional[str] = None) -> None:
        """Updates the last_progress_at timestamp and active step/action IDs."""
        now = time.time()
        self.last_progress_at = now
        self.updated_at = now
        if step_id is not None:
            self.current_step_id = step_id
        if action_id is not None:
            self.current_action_id = action_id

    def to_dict(self) -> Dict[str, Any]:
        """
        Lossless serialization preserving all execution, timing, and recovery state.
        Maintains complete backward compatibility with existing tests and code.
        """
        return {
            "id": self.id,
            "description": self.description,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "priority": self.priority,
            "assigned_worker": self.assigned_worker,
            "required_tools": list(self.required_tools),
            "context": dict(self.context),
            "result": self.result,
            "error": self.error,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "last_progress_at": self.last_progress_at,
            "timeout_seconds": self.timeout_seconds,
            "current_step_id": self.current_step_id,
            "current_action_id": self.current_action_id,
            "current_attempt_id": self.current_attempt_id,
            "verification_state": dict(self.verification_state),
            "recovery_state": dict(self.recovery_state),
            "checkpoint_data": dict(self.checkpoint_data),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """
        Reconstructs a Task from dictionary representation with full state rehydration.
        Handles missing fields gracefully for backward compatibility.
        """
        # Resolve status string to TaskState enum
        raw_status = data.get("status", TaskState.PENDING.value)
        try:
            status = TaskState(raw_status)
        except ValueError:
            status = TaskState.PENDING

        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            parent_id=data.get("parent_id"),
            status=status,
            dependencies=[str(d) for d in data.get("dependencies", [])],
            priority=int(data.get("priority", 1)),
            assigned_worker=data.get("assigned_worker"),
            required_tools=[str(t) for t in data.get("required_tools", [])],
            context=dict(data.get("context", {})),
            result=data.get("result"),
            error=data.get("error"),
            retry_count=int(data.get("retry_count", 0)),
            max_retries=int(data.get("max_retries", 2)),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            last_progress_at=data.get("last_progress_at"),
            timeout_seconds=data.get("timeout_seconds"),
            current_step_id=data.get("current_step_id"),
            current_action_id=data.get("current_action_id"),
            current_attempt_id=data.get("current_attempt_id"),
            verification_state=dict(data.get("verification_state", {})),
            recovery_state=dict(data.get("recovery_state", {})),
            checkpoint_data=dict(data.get("checkpoint_data", {})),
            metadata=dict(data.get("metadata", {})),
        )
