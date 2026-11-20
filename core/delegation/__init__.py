"""Delegation and Subagent Orchestration module for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Provides capability-filtered subagents, bounded recursion trees, and dependency DAG coordination.
"""

from core.delegation.coordinator import DelegationCoordinator
from core.delegation.models import (
    DelegationConfig,
    SubagentResult,
    SubagentRole,
    SubagentStatus,
    SubagentTask,
)
from core.delegation.subagent import SubagentHarness, truncate_subagent_output

__all__ = [
    "DelegationConfig",
    "DelegationCoordinator",
    "SubagentHarness",
    "SubagentResult",
    "SubagentRole",
    "SubagentStatus",
    "SubagentTask",
    "truncate_subagent_output",
]
