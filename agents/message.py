"""
Structured inter-agent messaging protocol for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Defines message schemas for communication between Supervisor and Dynamic Workers.
"""
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

class MessageType(Enum):
    TASK_ASSIGN = "task_assign"
    TASK_PROGRESS = "task_progress"
    TASK_RESULT = "task_result"
    TASK_FAILURE = "task_failure"

@dataclass
class AgentMessage:
    """
    Structured message payload passed between agents over the MessageBus.
    """
    sender: str
    receiver: str
    msg_type: MessageType
    task_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Converts message to dictionary representation."""
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "msg_type": self.msg_type.value,
            "task_id": self.task_id,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        """Instantiates AgentMessage from dictionary data."""
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            msg_type=MessageType(data["msg_type"]),
            task_id=data["task_id"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
        )
