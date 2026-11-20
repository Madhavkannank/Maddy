"""
Base classes and data structures for M.A.D.D.Y. Tool System (ACI).
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class Tool:
    """
    Structured Tool definition for Agent-Computer Interface (ACI).
    """
    name: str
    description: str
    parameter_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    risk_level: RiskLevel = RiskLevel.LOW
    timeout: float = 30.0
    supported_platforms: List[str] = field(default_factory=lambda: ["windows", "linux"])

    def to_schema_dict(self) -> Dict[str, Any]:
        """Format tool definition into JSON Schema format for LLM context."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameter_schema,
            },
            "risk_level": self.risk_level.value,
        }
