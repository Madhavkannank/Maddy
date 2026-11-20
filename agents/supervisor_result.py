"""
SupervisorResult data structures.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class SupervisorResult:
    """
    Structured outcome of a Supervisor request orchestration.
    """
    status: str  # "SUCCESS", "PARTIAL_SUCCESS", "FAILED"
    user_request: str
    final_response: str
    completed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    failed_tasks: List[Dict[str, Any]] = field(default_factory=list)
    blocked_tasks: List[Dict[str, Any]] = field(default_factory=list)
    total_tasks: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "user_request": self.user_request,
            "final_response": self.final_response,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "blocked_tasks": self.blocked_tasks,
            "total_tasks": self.total_tasks,
        }
