"""
Recovery Journal for M.A.D.D.Y. Execution Auditability.
Maintains a structured, deterministic record of execution state transitions,
actions, observations, verification outcomes, interruptions, and recovery decisions.
This is an execution audit trail — NOT a model chain-of-thought scratchpad.
"""
from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional

class JournalEventType(Enum):
    TASK_INITIALIZED = "TASK_INITIALIZED"
    TASK_RUNNING = "TASK_RUNNING"
    ACTION_START = "ACTION_START"
    ACTION_RESULT = "ACTION_RESULT"
    VERIFYING = "VERIFYING"
    VERIFICATION_SUCCESS = "VERIFICATION_SUCCESS"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    CHECKPOINT_SAVED = "CHECKPOINT_SAVED"
    CHECKPOINT_LOADED = "CHECKPOINT_LOADED"
    PROCESS_CRASH = "PROCESS_CRASH"
    TASK_INTERRUPTED = "TASK_INTERRUPTED"
    TASK_TIMED_OUT = "TASK_TIMED_OUT"
    TASK_PAUSED = "TASK_PAUSED"
    TASK_CANCELLED = "TASK_CANCELLED"
    EXTERNAL_STATE_UNKNOWN = "EXTERNAL_STATE_UNKNOWN"
    EXTERNAL_STATE_VERIFIED = "EXTERNAL_STATE_VERIFIED"
    RECOVERY_ANALYSIS = "RECOVERY_ANALYSIS"
    RECOVERY_DECISION = "RECOVERY_DECISION"
    RETRY_ATTEMPT = "RETRY_ATTEMPT"
    TASK_RESUMED = "TASK_RESUMED"
    TASK_SUCCESS = "TASK_SUCCESS"
    TASK_FAILED = "TASK_FAILED"

@dataclass
class JournalEntry:
    event_type: JournalEventType
    task_id: str
    message: str
    step_id: Optional[str] = None
    action_id: Optional[str] = None
    attempt_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "task_id": self.task_id,
            "message": self.message,
            "step_id": self.step_id,
            "action_id": self.action_id,
            "attempt_id": self.attempt_id,
            "payload": dict(self.payload),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JournalEntry":
        try:
            event_type = JournalEventType(data["event_type"])
        except (ValueError, KeyError):
            event_type = JournalEventType.ACTION_RESULT

        return cls(
            event_type=event_type,
            task_id=str(data.get("task_id", "")),
            message=str(data.get("message", "")),
            step_id=data.get("step_id"),
            action_id=data.get("action_id"),
            attempt_id=data.get("attempt_id"),
            payload=dict(data.get("payload", {})),
            timestamp=float(data.get("timestamp", time.time())),
        )

    def formatted_log_line(self) -> str:
        """Returns readable timestamped log line: 'HH:MM:SS [TASK] EVENT: message'."""
        timestr = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        scope = f"[{self.task_id}"
        if self.step_id:
            scope += f":{self.step_id}"
        if self.action_id:
            scope += f":{self.action_id}"
        scope += "]"
        return f"{timestr} {scope} {self.event_type.value} — {self.message}"

class RecoveryJournal:
    """
    Manages an in-memory and serializable sequence of journal entries for a task.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self._entries: List[JournalEntry] = []

    def record(
        self,
        event_type: JournalEventType,
        message: str,
        task_id: Optional[str] = None,
        step_id: Optional[str] = None,
        action_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> JournalEntry:
        entry = JournalEntry(
            event_type=event_type,
            task_id=task_id or self.task_id,
            message=message,
            step_id=step_id,
            action_id=action_id,
            attempt_id=attempt_id,
            payload=payload or {},
        )
        self._entries.append(entry)
        return entry

    def list_entries(self) -> List[JournalEntry]:
        return list(self._entries)

    def to_dict_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_dict_list(cls, task_id: str, entries_data: List[Dict[str, Any]]) -> "RecoveryJournal":
        journal = cls(task_id=task_id)
        for d in entries_data:
            journal._entries.append(JournalEntry.from_dict(d))
        return journal

    def get_audit_trail(self) -> List[str]:
        """Return formatted human-readable lines of the audit trail."""
        return [e.formatted_log_line() for e in self._entries]
