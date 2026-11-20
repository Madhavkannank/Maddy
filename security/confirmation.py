"""
Confirmation Handlers for soliciting user authorization for high-risk operations.
"""
from abc import ABC, abstractmethod
from typing import Callable, Optional
from tools.base import Tool
from tools.call import ToolCall
from security.evaluator import RiskEvaluation

class ConfirmationHandler(ABC):

    @abstractmethod
    def confirm(self, tool_call: ToolCall, tool: Tool, evaluation: RiskEvaluation) -> bool:
        """
        Solicit confirmation for a high-risk tool operation.
        Returns True if authorized by user, False if denied.
        """
        pass

class AutoDenyConfirmationHandler(ConfirmationHandler):
    """
    Default handler for non-interactive/headless execution.
    Automatically denies any action requiring user confirmation.
    """

    def confirm(self, tool_call: ToolCall, tool: Tool, evaluation: RiskEvaluation) -> bool:
        return False

class AutoApproveLowRiskConfirmationHandler(ConfirmationHandler):
    """
    Handler that automatically approves LOW and MEDIUM risk actions,
    but denies HIGH and CRITICAL risk actions.
    """

    def confirm(self, tool_call: ToolCall, tool: Tool, evaluation: RiskEvaluation) -> bool:
        from tools.base import RiskLevel
        return evaluation.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)

class InteractiveConfirmationHandler(ConfirmationHandler):
    """
    Handler using a user-provided callback function for confirmation prompts.
    """

    def __init__(self, callback: Callable[[ToolCall, Tool, RiskEvaluation], bool]):
        self.callback = callback

    def confirm(self, tool_call: ToolCall, tool: Tool, evaluation: RiskEvaluation) -> bool:
        return self.callback(tool_call, tool, evaluation)
