"""
Execution Identity and Action Idempotency Protocol for M.A.D.D.Y.
Tracks stable identifiers across task -> step -> action -> attempt hierarchy
to ensure safe resumption and prevent duplicate side effects after crashes.
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, Optional

class ActionState(Enum):
    """Execution state of a specific side-effecting action."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    NOT_COMPLETED = "not_completed"
    UNKNOWN = "unknown"
    FAILED = "failed"

class VerificationState(Enum):
    """External verification status of an action or task."""
    UNVERIFIED = "unverified"
    VERIFYING = "verifying"
    VERIFIED_SUCCESS = "verified_success"
    VERIFIED_FAILURE = "verified_failure"
    INDETERMINATE = "indeterminate"

@dataclass
class ExecutionIdentity:
    """
    Stable multi-level execution identity for an action within a long-running task.
    Conceptually: TASK -> STEP -> ACTION -> ATTEMPT
    """
    task_id: str
    step_id: str
    action_id: str
    attempt_id: str
    tool_name: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        task_id: str,
        step_number: int,
        action_index: int = 1,
        attempt_number: int = 1,
        tool_name: Optional[str] = None,
    ) -> "ExecutionIdentity":
        """Deterministic identity factory from ordinal counters."""
        step_id = f"S{step_number:02d}"
        action_id = f"A{action_index:02d}"
        attempt_id = f"ATTEMPT{attempt_number:02d}"
        return cls(
            task_id=task_id,
            step_id=step_id,
            action_id=action_id,
            attempt_id=attempt_id,
            tool_name=tool_name,
        )

    @property
    def composite_id(self) -> str:
        """Returns stable formatted composite ID: e.g. T1:S02:A01:ATTEMPT01."""
        return f"{self.task_id}:{self.step_id}:{self.action_id}:{self.attempt_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "step_id": self.step_id,
            "action_id": self.action_id,
            "attempt_id": self.attempt_id,
            "tool_name": self.tool_name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionIdentity":
        return cls(
            task_id=str(data.get("task_id", "")),
            step_id=str(data.get("step_id", "")),
            action_id=str(data.get("action_id", "")),
            attempt_id=str(data.get("attempt_id", "")),
            tool_name=data.get("tool_name"),
            created_at=float(data.get("created_at", time.time())),
        )

@dataclass
class ActionRecord:
    """
    Durable record of a side-effecting action and its verification state.
    """
    identity: ExecutionIdentity
    arguments: Dict[str, Any] = field(default_factory=dict)
    state: ActionState = ActionState.PENDING
    verification: VerificationState = VerificationState.UNVERIFIED
    observation: Optional[Any] = None
    verification_evidence: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "arguments": self.arguments,
            "state": self.state.value,
            "verification": self.verification.value,
            "observation": self.observation,
            "verification_evidence": self.verification_evidence,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        identity_data = data.get("identity", {})
        identity = ExecutionIdentity.from_dict(identity_data) if identity_data else ExecutionIdentity.create("unknown", 1)
        return cls(
            identity=identity,
            arguments=data.get("arguments", {}),
            state=ActionState(data.get("state", ActionState.PENDING.value)),
            verification=VerificationState(data.get("verification", VerificationState.UNVERIFIED.value)),
            observation=data.get("observation"),
            verification_evidence=data.get("verification_evidence"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
        )
