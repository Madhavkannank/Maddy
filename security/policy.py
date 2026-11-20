"""
Security Policy definitions and Permission Decision structures for M.A.D.D.Y.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class PolicyMode(Enum):
    STRICT = "strict"        # Requires confirmation for MEDIUM, HIGH, CRITICAL risks
    BALANCED = "balanced"    # LOW allowed, MEDIUM policy checked, HIGH/CRITICAL require confirmation
    PERMISSIVE = "permissive"# Auto-approve up to HIGH risk, CRITICAL requires confirmation
    DRY_RUN = "dry_run"      # Simulates permission decisions, blocks write/high-risk handlers from executing


@dataclass
class PermissionDecision:
    """
    Structured result of a permission engine check.
    Captures sandbox, quota, capability elevation lease, and audit tracking context.
    """
    allowed: bool
    requires_confirmation: bool
    reason: str
    decision_code: str  # ALLOWED, DENIED_RISK_POLICY, REQUIRES_USER_CONFIRMATION, DRY_RUN_BLOCKED, QUOTA_EXCEEDED, SANDBOX_VIOLATION
    risk_level: Optional[str] = None
    sandbox_result: Optional[Dict[str, Any]] = None
    quota_result: Optional[Dict[str, Any]] = None
    lease_id: Optional[str] = None
    audit_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "requires_confirmation": self.requires_confirmation,
            "reason": self.reason,
            "decision_code": self.decision_code,
            "risk_level": self.risk_level,
            "sandbox_result": self.sandbox_result,
            "quota_result": self.quota_result,
            "lease_id": self.lease_id,
            "audit_id": self.audit_id,
            "metadata": self.metadata,
        }
