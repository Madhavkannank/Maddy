"""Autonomy Policy and Authority Gate for Background Task Execution in M.A.D.D.Y.

Enforces policy ceilings to prevent autonomous permission laundering
and unauthorized dangerous operations during background unattended execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from core.permissions import PermissionGate, PermissionTier
from core.tools.base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class AutonomyDecision:
    """Outcome of autonomy policy evaluation."""
    allowed: bool
    reason: str
    requires_approval: bool = False
    policy_tier: str = "SAFE"


class AutonomyPolicyGate:
    """Enforces execution boundaries for autonomous, unattended background tasks."""

    def __init__(
        self,
        permission_gate: Optional[PermissionGate] = None,
        allow_standard_tier: bool = True,
        disallowed_tools: Optional[List[str]] = None,
    ):
        self.permission_gate = permission_gate or PermissionGate()
        self.allow_standard_tier = allow_standard_tier
        self.disallowed_tools = set(disallowed_tools or [
            "execute_raw_sudo",
            "format_disk",
            "reboot_system",
        ])

    def evaluate(self, tool: BaseTool, arguments: Dict[str, Any]) -> AutonomyDecision:
        """Evaluate if a tool call is permissible in an autonomous background context.

        Rules:
        - PermissionTier.SAFE: Auto-approved.
        - PermissionTier.STANDARD: Approved if allow_standard_tier is True and arguments satisfy PermissionGate.
        - PermissionTier.DANGEROUS: Strictly rejected in background mode (cannot bypass permission dialogs).
        """
        # 1. Check explicit tool blacklist
        if tool.name in self.disallowed_tools:
            return AutonomyDecision(
                allowed=False,
                reason=f"Tool '{tool.name}' is strictly prohibited in autonomous background mode.",
                policy_tier="BLOCKED",
            )

        # 2. Tier-based enforcement
        tier = getattr(tool, "tier", PermissionTier.SAFE)

        if tier == PermissionTier.SAFE:
            return AutonomyDecision(
                allowed=True,
                reason="SAFE tier tool permitted for autonomous background execution.",
                policy_tier="SAFE",
            )

        if tier == PermissionTier.SYSTEM_IMPACTING:
            if not self.allow_standard_tier:
                return AutonomyDecision(
                    allowed=False,
                    reason="SYSTEM_IMPACTING tier tools are disabled by autonomy policy.",
                    requires_approval=True,
                    policy_tier="SYSTEM_IMPACTING",
                )

            # Check sandbox containment via PermissionGate
            perm_check = self.permission_gate.check_tool_execution(tool.name, arguments)
            if perm_check.is_allowed:
                return AutonomyDecision(
                    allowed=True,
                    reason="SYSTEM_IMPACTING tier tool verified and allowed within sandbox boundaries.",
                    policy_tier="SYSTEM_IMPACTING",
                )
            else:
                return AutonomyDecision(
                    allowed=False,
                    reason=f"Autonomous system-impacting check failed: {perm_check.reason}",
                    requires_approval=True,
                    policy_tier="SYSTEM_IMPACTING",
                )

        return AutonomyDecision(
            allowed=False,
            reason=f"Unrecognized permission tier '{tier}' for tool '{tool.name}'.",
            policy_tier="UNKNOWN",
        )
