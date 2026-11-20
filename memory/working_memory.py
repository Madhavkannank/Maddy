"""
Working Memory implementation for transient step-by-step scratchpad tracking.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ScratchpadItem:
    key: str
    value: Any
    category: str = "general"

class WorkingMemory:
    """
    In-memory scratchpad tracking current active task steps, tool observations, and key-value states.
    """

    def __init__(self, goal: str = ""):
        self.goal = goal
        self._scratchpad: Dict[str, Any] = {}
        self._observations: List[Dict[str, Any]] = []

    def set(self, key: str, value: Any) -> None:
        """Store a key-value item in working memory."""
        self._scratchpad[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a key-value item from working memory."""
        return self._scratchpad.get(key, default)

    def add_observation(self, step_number: int, tool_name: str, result_data: Any) -> None:
        """Record a tool execution observation."""
        self._observations.append({
            "step": step_number,
            "tool": tool_name,
            "data": result_data,
        })

    def get_observations(self) -> List[Dict[str, Any]]:
        """Return all recorded observations."""
        return list(self._observations)

    def clear(self) -> None:
        """Clear transient working memory."""
        self._scratchpad.clear()
        self._observations.clear()
