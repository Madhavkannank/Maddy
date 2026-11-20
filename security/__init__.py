"""
Security module for M.A.D.D.Y. Agent-Computer Interface (ACI).
Provides sandboxing, risk evaluation, permission policies, capability leases,
resource quotas, and tamper-evident auditing.
"""
from typing import Optional

from security.audit import AuditLogger, PermissionAuditRecord
from security.audit_ledger import AuditLedger, AuditLedgerEntry
from security.autonomy import AutonomyEngine, AutonomyLevel
from security.command_sandbox import (
    CommandAssessment,
    CommandDecision,
    CommandRiskCategory,
    CommandSandbox,
)
from security.confirmation import (
    AutoApproveLowRiskConfirmationHandler,
    AutoDenyConfirmationHandler,
    ConfirmationHandler,
    InteractiveConfirmationHandler,
)
from security.elevation import ElevationLease, ElevationManager
from security.evaluator import RiskEvaluation, RiskEvaluator
from security.permission_engine import PermissionEngine
from security.policy import PermissionDecision, PolicyMode
from security.quotas import (
    QuotaCheckResult,
    ResourceQuotaConfig,
    ResourceQuotaManager,
)
from security.sandbox import (
    PathOperation,
    PathSandbox,
    PathSandboxResult,
    SandboxViolationType,
)


def get_permission_engine(
    policy_mode: PolicyMode = PolicyMode.BALANCED,
    confirmation_handler: Optional[ConfirmationHandler] = None,
    path_sandbox: Optional[PathSandbox] = None,
    command_sandbox: Optional[CommandSandbox] = None,
    elevation_manager: Optional[ElevationManager] = None,
    quota_manager: Optional[ResourceQuotaManager] = None,
    audit_ledger: Optional[AuditLedger] = None,
) -> PermissionEngine:
    """
    Factory function returning a fully-configured PermissionEngine instance.
    """
    return PermissionEngine(
        policy_mode=policy_mode,
        confirmation_handler=confirmation_handler,
        path_sandbox=path_sandbox,
        command_sandbox=command_sandbox,
        elevation_manager=elevation_manager,
        quota_manager=quota_manager,
        audit_logger=audit_ledger,
    )


__all__ = [
    "PolicyMode",
    "PermissionDecision",
    "RiskEvaluation",
    "RiskEvaluator",
    "ConfirmationHandler",
    "AutoDenyConfirmationHandler",
    "AutoApproveLowRiskConfirmationHandler",
    "InteractiveConfirmationHandler",
    "PermissionAuditRecord",
    "AuditLogger",
    "AuditLedger",
    "AuditLedgerEntry",
    "PermissionEngine",
    "get_permission_engine",
    "AutonomyEngine",
    "AutonomyLevel",
    "PathSandbox",
    "PathOperation",
    "PathSandboxResult",
    "SandboxViolationType",
    "CommandSandbox",
    "CommandAssessment",
    "CommandRiskCategory",
    "CommandDecision",
    "ElevationLease",
    "ElevationManager",
    "ResourceQuotaConfig",
    "QuotaCheckResult",
    "ResourceQuotaManager",
]
