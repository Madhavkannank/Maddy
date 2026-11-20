"""
Audit Logger for recording security permission decisions.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List
from security.policy import PermissionDecision
from tools.base import RiskLevel

@dataclass
class PermissionAuditRecord:
    timestamp: float
    tool_name: str
    arguments: Dict[str, Any]
    risk_level: str
    allowed: bool
    requires_confirmation: bool
    decision_code: str
    reason: str

class AuditLogger:
    """
    Maintains structured in-memory and configurable security audit logs.
    """

    def __init__(self, max_records: int = 1000):
        self.max_records = max_records
        self._records: List[PermissionAuditRecord] = []

    def record(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        risk_level: RiskLevel,
        decision: PermissionDecision,
    ) -> PermissionAuditRecord:
        rec = PermissionAuditRecord(
            timestamp=time.time(),
            tool_name=tool_name,
            arguments=arguments,
            risk_level=risk_level.value,
            allowed=decision.allowed,
            requires_confirmation=decision.requires_confirmation,
            decision_code=decision.decision_code,
            reason=decision.reason,
        )

        self._records.append(rec)
        if len(self._records) > self.max_records:
            self._records.pop(0)

        return rec

    def get_records(self) -> List[PermissionAuditRecord]:
        return list(self._records)

    def clear(self) -> None:
        self._records.clear()
