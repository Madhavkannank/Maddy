"""
Permission Engine for enforcing risk policies, sandbox boundaries, capability elevation leases,
resource quotas, and tamper-evident audit logging.
"""
from typing import Any, Dict, Optional

from security.audit import AuditLogger
from security.audit_ledger import AuditLedger
from security.command_sandbox import CommandSandbox
from security.confirmation import AutoDenyConfirmationHandler, ConfirmationHandler
from security.elevation import ElevationLease, ElevationManager
from security.evaluator import RiskEvaluation, RiskEvaluator
from security.policy import PermissionDecision, PolicyMode
from security.quotas import QuotaCheckResult, ResourceQuotaManager
from security.sandbox import PathOperation, PathSandbox, SandboxViolationType
from tools.base import RiskLevel, Tool
from tools.call import ToolCall


class PermissionEngine:
    """
    Central gatekeeper orchestrating sandbox validation, resource limits,
    capability leases, risk policies, confirmation handlers, and tamper-evident audit logging.
    """

    def __init__(
        self,
        policy_mode: PolicyMode = PolicyMode.BALANCED,
        confirmation_handler: Optional[ConfirmationHandler] = None,
        audit_logger: Optional[Any] = None,
        path_sandbox: Optional[PathSandbox] = None,
        command_sandbox: Optional[CommandSandbox] = None,
        elevation_manager: Optional[ElevationManager] = None,
        quota_manager: Optional[ResourceQuotaManager] = None,
    ):
        self.policy_mode = policy_mode
        self.confirmation_handler = confirmation_handler or AutoDenyConfirmationHandler()
        
        # Backward-compatible audit ledger initialization
        if audit_logger is not None and isinstance(audit_logger, AuditLedger):
            self.audit_ledger = audit_logger
        elif audit_logger is not None:
            self.audit_ledger = audit_logger
        else:
            self.audit_ledger = AuditLedger()

        # Alias for existing code expecting audit_logger
        self.audit_logger = self.audit_ledger

        self.path_sandbox = path_sandbox or PathSandbox()
        self.command_sandbox = command_sandbox or CommandSandbox()
        self.elevation_manager = elevation_manager or ElevationManager()
        self.quota_manager = quota_manager or ResourceQuotaManager()

    def check_permission(
        self,
        tool_call: ToolCall,
        tool: Tool,
        task_id: Optional[str] = None,
    ) -> PermissionDecision:
        """
        Comprehensive security gate enforcing:
        1. Quota Check
        2. Sandbox & Risk Evaluation
        3. Capability Elevation Leases
        4. Policy & User Confirmation
        5. Tamper-Evident Hash-Chained Audit Logging
        """
        # 1. Quota Checks
        rate_res = self.quota_manager.check_rate_limit()
        if not rate_res.allowed:
            decision = PermissionDecision(
                allowed=False,
                requires_confirmation=False,
                reason=rate_res.reason,
                decision_code="QUOTA_EXCEEDED",
                risk_level=tool.risk_level.value,
                quota_result=rate_res.to_dict(),
            )
            self._record_audit(tool.name, tool_call.arguments, tool.risk_level, decision)
            return decision

        # Check write size quota if writing a file
        if tool.name.lower() in ("filesystem.write", "file.write"):
            content_arg = tool_call.arguments.get("content", "")
            bytes_to_write = len(content_arg.encode("utf-8")) if isinstance(content_arg, str) else 0
            write_quota = self.quota_manager.check_file_write(bytes_to_write, task_id=task_id)
            if not write_quota.allowed:
                decision = PermissionDecision(
                    allowed=False,
                    requires_confirmation=False,
                    reason=write_quota.reason,
                    decision_code="QUOTA_EXCEEDED",
                    risk_level=tool.risk_level.value,
                    quota_result=write_quota.to_dict(),
                )
                self._record_audit(tool.name, tool_call.arguments, tool.risk_level, decision)
                return decision

        # 2. Risk & Sandbox Evaluation
        evaluation = RiskEvaluator.evaluate(
            tool_call=tool_call,
            tool=tool,
            path_sandbox=self.path_sandbox,
            command_sandbox=self.command_sandbox,
        )
        risk = evaluation.risk_level
        sandbox_dict = evaluation.sandbox_result.to_dict() if evaluation.sandbox_result else None

        target_path = (
            tool_call.arguments.get("path")
            or tool_call.arguments.get("directory")
            or tool_call.arguments.get("target_path")
        )

        # 3. Check for matching Active Elevation Lease
        active_lease = self.elevation_manager.find_matching_lease(tool.name, target_path)
        if active_lease:
            consumed = self.elevation_manager.consume_lease(active_lease.lease_id, tool.name, target_path)
            if consumed:
                decision = PermissionDecision(
                    allowed=True,
                    requires_confirmation=False,
                    reason=f"Action authorized under active elevation lease '{active_lease.lease_id[:8]}' ({active_lease.originating_authorization})",
                    decision_code="ALLOWED",
                    risk_level=risk.value,
                    sandbox_result=sandbox_dict,
                    lease_id=active_lease.lease_id,
                )
                self._record_audit(tool.name, tool_call.arguments, risk, decision, lease_id=active_lease.lease_id)
                return decision

        # 4. Handle DRY_RUN mode
        if self.policy_mode == PolicyMode.DRY_RUN:
            if risk in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
                decision = PermissionDecision(
                    allowed=False,
                    requires_confirmation=True,
                    reason=f"Action blocked by DRY_RUN mode (risk: {risk.value})",
                    decision_code="DRY_RUN_BLOCKED",
                    risk_level=risk.value,
                    sandbox_result=sandbox_dict,
                )
                self._record_audit(tool.name, tool_call.arguments, risk, decision)
                return decision

        # 5. Determine if action requires confirmation
        needs_confirm = evaluation.requires_confirmation
        if self.policy_mode == PolicyMode.STRICT and risk in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
            needs_confirm = True

        # Fail-closed for sensitive credential targets without explicit elevation lease
        if evaluation.sandbox_result and evaluation.sandbox_result.violation_type == SandboxViolationType.SENSITIVE_CREDENTIAL_PATH:
            needs_confirm = True

        # 6. Solicit confirmation if required
        if needs_confirm:
            confirmed = self.confirmation_handler.confirm(tool_call, tool, evaluation)
            if confirmed:
                decision = PermissionDecision(
                    allowed=True,
                    requires_confirmation=True,
                    reason=f"Action authorized by user confirmation ({evaluation.reason})",
                    decision_code="ALLOWED",
                    risk_level=risk.value,
                    sandbox_result=sandbox_dict,
                )
            else:
                decision = PermissionDecision(
                    allowed=False,
                    requires_confirmation=True,
                    reason=f"Permission denied: action requires confirmation but was denied or unconfirmed ({evaluation.reason})",
                    decision_code="REQUIRES_USER_CONFIRMATION",
                    risk_level=risk.value,
                    sandbox_result=sandbox_dict,
                )
            self._record_audit(tool.name, tool_call.arguments, risk, decision)
            return decision

        # 7. Action allowed directly by policy
        decision = PermissionDecision(
            allowed=True,
            requires_confirmation=False,
            reason=f"Allowed by policy '{self.policy_mode.value}' ({evaluation.reason})",
            decision_code="ALLOWED",
            risk_level=risk.value,
            sandbox_result=sandbox_dict,
        )
        self._record_audit(tool.name, tool_call.arguments, risk, decision)
        return decision

    def explain(self, tool_call: ToolCall, tool: Tool) -> Dict[str, Any]:
        """
        Explain what decision would be made without executing side effects (DRY RUN inspection).
        """
        evaluation = RiskEvaluator.evaluate(
            tool_call=tool_call,
            tool=tool,
            path_sandbox=self.path_sandbox,
            command_sandbox=self.command_sandbox,
        )
        target_path = (
            tool_call.arguments.get("path")
            or tool_call.arguments.get("directory")
            or tool_call.arguments.get("target_path")
        )
        active_lease = self.elevation_manager.find_matching_lease(tool.name, target_path)

        needs_confirm = evaluation.requires_confirmation
        if self.policy_mode == PolicyMode.STRICT and evaluation.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL):
            needs_confirm = True

        return {
            "tool_name": tool.name,
            "arguments": tool_call.arguments,
            "risk_level": evaluation.risk_level.value,
            "policy_mode": self.policy_mode.value,
            "would_require_confirmation": needs_confirm if not active_lease else False,
            "matching_lease_id": active_lease.lease_id if active_lease else None,
            "sandbox_result": evaluation.sandbox_result.to_dict() if evaluation.sandbox_result else None,
            "command_assessment": evaluation.command_assessment.to_dict() if evaluation.command_assessment else None,
            "reason": evaluation.reason,
        }

    def _record_audit(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        risk_level: RiskLevel,
        decision: PermissionDecision,
        lease_id: Optional[str] = None,
    ) -> None:
        if hasattr(self.audit_ledger, "record"):
            try:
                self.audit_ledger.record(
                    tool_name=tool_name,
                    arguments=arguments,
                    risk_level=risk_level,
                    decision=decision,
                    lease_id=lease_id,
                )
            except TypeError:
                # In case an older AuditLogger without lease_id parameter is used
                self.audit_ledger.record(
                    tool_name=tool_name,
                    arguments=arguments,
                    risk_level=risk_level,
                    decision=decision,
                )
