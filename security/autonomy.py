"""
Autonomy Progression Policy Engine for M.A.D.D.Y.
Regulates gradual autonomy expansion from human-in-the-loop observation to policy-bounded execution.
"""
from enum import IntEnum
from typing import Any, Dict, Optional
from tools.base import RiskLevel, Tool

class AutonomyLevel(IntEnum):
    OBSERVE = 1
    SUGGEST = 2
    ASK_PERMISSION = 3
    EXECUTE_TRUSTED = 4
    FULL_AUTONOMY = 5

class AutonomyEngine:
    """
    Evaluates whether an action should be automatically executed, suggested, or submitted for user approval
    based on the system's current AutonomyLevel and tool RiskLevel.
    Enforces Rule 30 (Gradual Autonomy Progression).
    """

    def __init__(self, current_level: AutonomyLevel = AutonomyLevel.EXECUTE_TRUSTED):
        self.current_level = current_level

    def evaluate_autonomy(self, tool: Tool, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evaluate tool execution authority under the current autonomy level.
        """
        auto_execute = self.should_auto_execute(tool)

        return {
            "autonomy_level": self.current_level.name,
            "tool_name": tool.name,
            "risk_level": tool.risk_level.value,
            "auto_execute": auto_execute,
            "requires_user_confirmation": not auto_execute,
        }

    def should_auto_execute(self, tool: Tool) -> bool:
        """
        Check if tool can be automatically executed without user interaction.
        """
        if self.current_level == AutonomyLevel.OBSERVE or self.current_level == AutonomyLevel.SUGGEST:
            return False
        
        if self.current_level == AutonomyLevel.ASK_PERMISSION:
            # Requires explicit approval for all non-trivial actions
            return False

        if self.current_level == AutonomyLevel.EXECUTE_TRUSTED:
            # Auto-executes LOW risk tools; requires permission for MEDIUM/HIGH/CRITICAL
            return tool.risk_level == RiskLevel.LOW

        if self.current_level == AutonomyLevel.FULL_AUTONOMY:
            # Auto-executes LOW and MEDIUM risk tools; requires approval for HIGH/CRITICAL
            return tool.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

        return False
