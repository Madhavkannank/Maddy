"""
Data structures and status Enums for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) Core Agent Loop.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from tools.call import ToolCall
from tools.result import ToolResult

class AgentStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"

    # Status aliases
    ERROR = "failed"
    MAX_STEPS = "max_iterations_reached"

@dataclass
class AgentStep:
    """
    Record of a single step in the agent reasoning and action loop.
    """
    step_number: int
    thought: str = ""
    tool_call: Optional[ToolCall] = None
    tool_result: Optional[ToolResult] = None
    observation: Optional[str] = None
    final_answer: Optional[str] = None

@dataclass
class AgentResult:
    """
    Final outcome of an agent goal execution.
    """
    status: AgentStatus
    goal: str = ""
    final_answer: Optional[str] = None
    steps: List[AgentStep] = field(default_factory=list)
    total_steps: int = 0
    error_message: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.error and not self.error_message:
            self.error_message = self.error
        elif self.error_message and not self.error:
            self.error = self.error_message
