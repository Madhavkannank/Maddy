"""
Risk Evaluator for classifying tool calls and inspecting arguments for risk markers.
Integrates PathSandbox and CommandSandbox to provide deep contextual risk evaluation.
"""
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from security.command_sandbox import CommandAssessment, CommandRiskCategory, CommandSandbox
from security.sandbox import PathOperation, PathSandbox, PathSandboxResult, SandboxViolationType
from tools.base import RiskLevel, Tool
from tools.call import ToolCall


@dataclass
class RiskEvaluation:
    risk_level: RiskLevel
    requires_confirmation: bool
    reason: str
    sandbox_result: Optional[PathSandboxResult] = None
    command_assessment: Optional[CommandAssessment] = None


class RiskEvaluator:
    """
    Evaluates tool calls against tool metadata, argument sandboxing, and command risks.
    """

    HIGH_RISK_PATH_MARKERS = [
        "/etc", "/usr", "/boot", "/sys", "/proc", "/var",
        "C:\\Windows", "C:\\Program Files", "C:\\System32",
    ]

    _default_sandbox: Optional[PathSandbox] = None
    _default_command_sandbox: Optional[CommandSandbox] = None

    @classmethod
    def get_default_sandbox(cls) -> PathSandbox:
        if cls._default_sandbox is None:
            cls._default_sandbox = PathSandbox()
        return cls._default_sandbox

    @classmethod
    def get_default_command_sandbox(cls) -> CommandSandbox:
        if cls._default_command_sandbox is None:
            cls._default_command_sandbox = CommandSandbox()
        return cls._default_command_sandbox

    @classmethod
    def evaluate(
        cls,
        tool_call: ToolCall,
        tool: Tool,
        path_sandbox: Optional[PathSandbox] = None,
        command_sandbox: Optional[CommandSandbox] = None,
    ) -> RiskEvaluation:
        base_risk = tool.risk_level
        args = tool_call.arguments
        reasons: List[str] = []
        effective_risk = base_risk
        sandbox_res: Optional[PathSandboxResult] = None
        cmd_assess: Optional[CommandAssessment] = None

        psandbox = path_sandbox or cls.get_default_sandbox()
        csandbox = command_sandbox or cls.get_default_command_sandbox()

        # 1. Inspect path arguments for filesystem security
        path_arg = args.get("path") or args.get("directory") or args.get("target_path")
        if isinstance(path_arg, str) and path_arg.strip():
            # Determine operation
            tool_lower = tool.name.lower()
            if "delete" in tool_lower or "remove" in tool_lower:
                op = PathOperation.DELETE
            elif "write" in tool_lower or "create" in tool_lower or "save" in tool_lower:
                op = PathOperation.WRITE
            elif "search" in tool_lower or "list" in tool_lower:
                op = PathOperation.LIST
            else:
                op = PathOperation.READ

            sandbox_res = psandbox.validate_path(path_arg, operation=op)
            if not sandbox_res.allowed:
                if sandbox_res.violation_type == SandboxViolationType.SENSITIVE_CREDENTIAL_PATH:
                    effective_risk = RiskLevel.CRITICAL
                    reasons.append(sandbox_res.reason)
                elif sandbox_res.violation_type in (
                    SandboxViolationType.FORBIDDEN_SYSTEM_PATH,
                    SandboxViolationType.TRAVERSAL_ATTEMPT,
                    SandboxViolationType.RESERVED_DEVICE_NAME,
                ):
                    effective_risk = max_risk(effective_risk, RiskLevel.HIGH)
                    reasons.append(sandbox_res.reason)
                else:
                    effective_risk = max_risk(effective_risk, RiskLevel.HIGH)
                    reasons.append(sandbox_res.reason)

            # Preserve backwards compatibility with legacy HIGH_RISK_PATH_MARKERS
            clean_path = path_arg.strip().replace("\\", "/")
            abs_path = os.path.abspath(path_arg)
            for marker in cls.HIGH_RISK_PATH_MARKERS:
                clean_marker = marker.replace("\\", "/")
                if clean_path.startswith(clean_marker) or abs_path.lower().startswith(marker.lower()):
                    effective_risk = max_risk(effective_risk, RiskLevel.HIGH)
                    marker_msg = f"Target path '{path_arg}' accesses system directory '{marker}'"
                    if marker_msg not in reasons:
                        reasons.append(marker_msg)
                    break

        # 2. Inspect command arguments
        cmd_arg = args.get("command") or args.get("cmd") or args.get("script")
        if isinstance(cmd_arg, str) and cmd_arg.strip():
            cmd_assess = csandbox.analyze_command(cmd_arg)
            effective_risk = max_risk(effective_risk, cmd_assess.effective_risk_level)
            reasons.append(cmd_assess.reason)

        if not reasons:
            reasons.append(f"Base tool risk level is '{base_risk.value}'")

        requires_confirmation = (effective_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL))

        return RiskEvaluation(
            risk_level=effective_risk,
            requires_confirmation=requires_confirmation,
            reason="; ".join(reasons),
            sandbox_result=sandbox_res,
            command_assessment=cmd_assess,
        )


def max_risk(r1: RiskLevel, r2: RiskLevel) -> RiskLevel:
    """Helper to pick the higher of two RiskLevel values."""
    order = [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    idx1 = order.index(r1) if r1 in order else 0
    idx2 = order.index(r2) if r2 in order else 0
    return order[max(idx1, idx2)]
