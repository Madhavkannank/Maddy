"""
Failure Recovery and Replanning Engine for M.A.D.D.Y.
Handles error classification, fallback strategy evaluation, bounded retries,
interrupted task analysis, external state verification, and failure explanations.
"""
from dataclasses import dataclass
from enum import Enum
import os
from typing import Any, Dict, Optional, Tuple
from tasks.identity import ActionState, VerificationState
from tasks.task import Task
from tools.call import ToolCall
from tools.result import ToolResult

class ErrorType(Enum):
    APPLICATION_NOT_FOUND = "APPLICATION_NOT_FOUND"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    NAVIGATION_ERROR = "NAVIGATION_ERROR"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    PROCESS_CRASH = "PROCESS_CRASH"
    WORKER_FAILURE = "WORKER_FAILURE"
    CHECKPOINT_CORRUPTION = "CHECKPOINT_CORRUPTION"
    INTERRUPTED_EXECUTION = "INTERRUPTED_EXECUTION"
    VERIFICATION_FAILURE = "VERIFICATION_FAILURE"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"

class RecoveryAction(Enum):
    RETRY = "RETRY"
    VERIFY_FIRST = "VERIFY_FIRST"
    REPLAN = "REPLAN"
    FALLBACK = "FALLBACK"
    EXTEND_DEADLINE = "EXTEND_DEADLINE"
    ASK_USER = "ASK_USER"
    FAIL = "FAIL"

@dataclass
class RecoveryStrategy:
    """Strategy recommendation for recovering from an execution or tool failure."""
    error_type: ErrorType
    can_retry: bool
    retry_count: int
    max_retries: int
    action: RecoveryAction = RecoveryAction.RETRY
    alternative_tool: Optional[str] = None
    suggested_action: str = ""
    user_explanation: str = ""
    requires_verification: bool = False

class RecoveryEngine:
    """
    Evaluates execution failures, classifies errors, and suggests recovery strategies or explanations.
    Enforces Rule 17 (Bounded Retries), Rule 18 (Failure Recovery), and Rule 31 (No Fake Completion).
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def classify_error(self, error_info: Any) -> ErrorType:
        """Classify error structure, exception, or text into a standard ErrorType."""
        if not error_info:
            return ErrorType.UNKNOWN_ERROR

        # Structured dict or object inspection
        if isinstance(error_info, dict):
            raw_type = error_info.get("type") or error_info.get("error_type")
            if raw_type:
                for member in ErrorType:
                    if member.value == str(raw_type) or member.name == str(raw_type):
                        return member

        err_str = str(error_info).lower()
        if ("app" in err_str or "application" in err_str) and ("not found" in err_str or "launch" in err_str or "close" in err_str):
            return ErrorType.APPLICATION_NOT_FOUND
        elif "file" in err_str and ("not found" in err_str or "no such file" in err_str or "does not exist" in err_str):
            return ErrorType.FILE_NOT_FOUND
        elif "navigate" in err_str or "browser" in err_str or "http" in err_str or "url" in err_str:
            return ErrorType.NAVIGATION_ERROR
        elif "permission" in err_str or "denied" in err_str or "blocked" in err_str:
            return ErrorType.PERMISSION_DENIED
        elif "validation" in err_str or "argument" in err_str or "schema" in err_str:
            return ErrorType.VALIDATION_ERROR
        elif "checkpoint" in err_str and ("corrupt" in err_str or "damage" in err_str or "malformed" in err_str):
            return ErrorType.CHECKPOINT_CORRUPTION
        elif "interrupted" in err_str or "sigkill" in err_str or "unexpectedly terminated" in err_str:
            return ErrorType.INTERRUPTED_EXECUTION
        elif "process" in err_str and ("crash" in err_str or "exit code" in err_str or "terminated" in err_str):
            return ErrorType.PROCESS_CRASH
        elif "worker" in err_str and ("failure" in err_str or "crashed" in err_str or "stopped" in err_str):
            return ErrorType.WORKER_FAILURE
        elif "verification" in err_str and ("fail" in err_str or "indeterminate" in err_str or "mismatch" in err_str):
            return ErrorType.VERIFICATION_FAILURE
        elif "task" in err_str and ("timed out" in err_str or "timeout" in err_str or "deadline" in err_str):
            return ErrorType.TASK_TIMEOUT
        elif "timeout" in err_str or "timed out" in err_str:
            return ErrorType.TIMEOUT_ERROR
        elif "crash" in err_str:
            return ErrorType.PROCESS_CRASH
        else:
            return ErrorType.UNKNOWN_ERROR

    def evaluate_recovery(
        self,
        tool_call: ToolCall,
        tool_result: ToolResult,
        current_retry_count: int = 0,
    ) -> RecoveryStrategy:
        """
        Evaluate tool failure and return a recommended RecoveryStrategy.
        """
        err_msg = json_extract_error(tool_result)
        error_type = self.classify_error(err_msg)

        can_retry = current_retry_count < self.max_retries
        alt_tool = self._suggest_alternative_tool(tool_call.name, error_type)

        rec_action = RecoveryAction.RETRY if can_retry else RecoveryAction.FAIL

        if error_type == ErrorType.APPLICATION_NOT_FOUND:
            explanation = f"I couldn't open application '{tool_call.arguments.get('app_name')}' because it was not found on the system."
            action = "Try searching for the application path or using browser alternative."
            if alt_tool:
                rec_action = RecoveryAction.FALLBACK
        elif error_type == ErrorType.FILE_NOT_FOUND:
            explanation = f"I couldn't read file '{tool_call.arguments.get('path')}' because the file does not exist."
            action = "Check if the file path is correct or list files in directory."
        elif error_type == ErrorType.PERMISSION_DENIED:
            explanation = f"Action '{tool_call.name}' was blocked by security permission policy."
            action = "Request explicit user approval before retrying."
            can_retry = False
            rec_action = RecoveryAction.ASK_USER
        elif error_type == ErrorType.NAVIGATION_ERROR:
            explanation = f"Browser navigation to '{tool_call.arguments.get('url')}' failed."
            action = "Verify internet connection or try searching the topic via browser.search."
            if alt_tool:
                rec_action = RecoveryAction.FALLBACK
        elif error_type in (ErrorType.TIMEOUT_ERROR, ErrorType.TASK_TIMEOUT):
            explanation = f"Operation '{tool_call.name}' exceeded timeout deadline."
            action = "Check network responsiveness or simplify request arguments."
            rec_action = RecoveryAction.RETRY if can_retry else RecoveryAction.FAIL
        else:
            explanation = f"Tool '{tool_call.name}' failed with error: {err_msg}"
            action = "Retry operation or check tool arguments."

        return RecoveryStrategy(
            error_type=error_type,
            can_retry=can_retry and (error_type != ErrorType.PERMISSION_DENIED),
            retry_count=current_retry_count,
            max_retries=self.max_retries,
            action=rec_action,
            alternative_tool=alt_tool,
            suggested_action=action,
            user_explanation=explanation,
        )

    def analyze_interrupted_task(self, task: Task) -> RecoveryStrategy:
        """
        Analyze an interrupted task to prevent duplicate side effects after a crash.
        Inspects task verification state and last known action.
        """
        verif = task.verification_state or {}
        v_state = verif.get("state", VerificationState.UNVERIFIED.value)

        # Case 1: Already verified success
        if v_state == VerificationState.VERIFIED_SUCCESS.value:
            return RecoveryStrategy(
                error_type=ErrorType.INTERRUPTED_EXECUTION,
                can_retry=False,
                retry_count=task.retry_count,
                max_retries=task.max_retries,
                action=RecoveryAction.RETRY,
                suggested_action="Action already completed and verified. Transition task to SUCCESS without repeating side effect.",
                user_explanation="Previous action was already verified as completed before interruption.",
                requires_verification=False,
            )

        # Case 2: Side-effecting action was running when crashed -> must verify first
        if task.current_action_id or task.checkpoint_data.get("last_tool_call"):
            return RecoveryStrategy(
                error_type=ErrorType.INTERRUPTED_EXECUTION,
                can_retry=task.retry_count < task.max_retries,
                retry_count=task.retry_count,
                max_retries=task.max_retries,
                action=RecoveryAction.VERIFY_FIRST,
                suggested_action="Verify external system state (file, process, download) before re-executing action.",
                user_explanation="Execution was interrupted during a side effect. Verifying external state first to avoid duplication.",
                requires_verification=True,
            )

        # Case 3: Safe to retry normal computation
        can_retry = task.retry_count < task.max_retries
        return RecoveryStrategy(
            error_type=ErrorType.INTERRUPTED_EXECUTION,
            can_retry=can_retry,
            retry_count=task.retry_count,
            max_retries=task.max_retries,
            action=RecoveryAction.RETRY if can_retry else RecoveryAction.FAIL,
            suggested_action="Resume task from last verified step.",
            user_explanation="Resuming interrupted task from checkpoint.",
            requires_verification=False,
        )

    def verify_external_action(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[ActionState, str]:
        """
        Verify if an external side-effecting action has actually completed on the system.
        Used before retrying interrupted tasks to avoid duplicate operations.
        """
        try:
            # 1. File system write/create operations
            if tool_name in ("filesystem.write", "filesystem_write"):
                path = arguments.get("path") or arguments.get("file_path")
                if path and os.path.exists(path):
                    size = os.path.getsize(path)
                    expected_content = arguments.get("content")
                    if expected_content and size > 0:
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                actual_content = f.read()
                            if actual_content == expected_content:
                                return ActionState.COMPLETED, f"File '{path}' exists with exact matching content."
                        except Exception:
                            pass
                    return ActionState.COMPLETED, f"File '{path}' exists (size: {size} bytes)."
                return ActionState.NOT_COMPLETED, f"File '{path}' does not exist on disk."

            # 2. File delete operations
            elif tool_name in ("filesystem.delete", "filesystem_delete"):
                path = arguments.get("path")
                if path and not os.path.exists(path):
                    return ActionState.COMPLETED, f"Target file '{path}' is confirmed deleted."
                return ActionState.NOT_COMPLETED, f"Target file '{path}' still exists on disk."

            # 3. Default for other tools
            return ActionState.UNKNOWN, f"Cannot automatically verify external state for tool '{tool_name}'."
        except Exception as ex:
            return ActionState.UNKNOWN, f"Verification failed with exception: {ex}"

    def _suggest_alternative_tool(self, tool_name: str, error_type: ErrorType) -> Optional[str]:
        """Suggest alternative higher/lower level tool fallback based on hierarchy."""
        if tool_name == "application.open" and error_type == ErrorType.APPLICATION_NOT_FOUND:
            return "browser.navigate"
        elif tool_name == "browser.navigate" and error_type == ErrorType.NAVIGATION_ERROR:
            return "browser.search"
        elif tool_name == "computer.click" and error_type == ErrorType.UNKNOWN_ERROR:
            return "vision.locate_element"
        return None

def json_extract_error(result_data: Any) -> str:
    """Helper to extract error message from result object or dictionary if present."""
    if hasattr(result_data, "error") and result_data.error:
        err = result_data.error
        if isinstance(err, dict):
            return err.get("message", str(err))
        return str(err)
    if isinstance(result_data, dict):
        if "error" in result_data:
            err = result_data["error"]
            if isinstance(err, dict):
                return err.get("message", str(err))
            return str(err)
    return str(result_data or "")
