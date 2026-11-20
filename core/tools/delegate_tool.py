"""Native Typed Model Tool for Task Delegation in M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Enables MaddyAgentCore to spawn isolated subagents for single goals, parallel execution,
or complex dependency-directed acyclic task graphs (DAGs).
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.delegation.coordinator import DelegationCoordinator
from core.delegation.models import DelegationConfig, SubagentRole, SubagentTask
from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from core.llm_client import CloudLLMClient
    from core.permissions import PermissionGate
    from core.tools.base import ToolRegistry
    from core.trace_store import TraceStore

logger = logging.getLogger(__name__)


class DelegateTool(BaseTool):
    """Tool enabling M.A.D.D.Y. to delegate subtasks to isolated child agent instances."""

    name = "delegate_task"
    description = (
        "Delegate one or more tasks to isolated subagent(s). "
        "The subagent executes autonomously in a fresh conversation and returns a concise summary. "
        "Use 'task' for a single goal, or 'tasks' for parallel execution and dependency-aware graphs."
    )
    tier = PermissionTier.SAFE

    def __init__(
        self,
        coordinator: Optional[DelegationCoordinator] = None,
        llm_client: Optional[CloudLLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        permission_gate: Optional[PermissionGate] = None,
        trace_store: Optional[TraceStore] = None,
        current_depth: int = 1,
        config: Optional[DelegationConfig] = None,
    ):
        if coordinator:
            self.coordinator = coordinator
        else:
            self.coordinator = DelegationCoordinator(
                llm_client=llm_client,
                tool_registry=tool_registry,
                permission_gate=permission_gate,
                trace_store=trace_store,
                config=config,
                current_depth=current_depth,
            )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Single task goal or objective to delegate to an isolated subagent.",
                },
                "tasks": {
                    "type": "array",
                    "description": "List of subtasks for batch parallel execution or dependency-directed DAG.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "Unique identifier for this subtask (e.g. 'task_1', 'fetch_data').",
                            },
                            "task": {
                                "type": "string",
                                "description": "Objective or instruction for this specific subtask.",
                            },
                            "context": {
                                "type": "string",
                                "description": "Optional background or input context for this subtask.",
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of task IDs that must complete before this task can execute.",
                            },
                            "role": {
                                "type": "string",
                                "enum": ["leaf", "specialist"],
                                "description": "Subagent role. Leaf subagents cannot further delegate.",
                            },
                            "timeout": {
                                "type": "number",
                                "description": "Per-task timeout in seconds (default 120.0).",
                            },
                        },
                        "required": ["task"],
                    },
                },
                "context": {
                    "type": "string",
                    "description": "Shared background context or parameters provided to all subagents.",
                },
                "role": {
                    "type": "string",
                    "enum": ["leaf", "specialist"],
                    "description": "Role for the subagent(s). Defaults to 'leaf'.",
                },
                "timeout": {
                    "type": "number",
                    "description": "Maximum execution duration in seconds for subagent(s).",
                },
            },
            "required": [],
        }

    def execute(
        self,
        task: Optional[str] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        role: str = "leaf",
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Execute delegation request."""
        # Case A: Batch / DAG multi-task execution
        if tasks and isinstance(tasks, list):
            if len(tasks) == 0:
                return ToolResult(
                    success=False,
                    output="Error: Empty tasks list provided for delegation.",
                    error="Empty tasks list",
                )

            # Convert task dicts into typed SubagentTask objects
            subagent_tasks: List[SubagentTask] = []
            has_dependencies = False

            for idx, t_dict in enumerate(tasks):
                task_id = str(t_dict.get("id") or f"subtask_{idx+1}")
                t_ctx = t_dict.get("context") or context
                t_role = t_dict.get("role") or role
                t_timeout = t_dict.get("timeout") or timeout

                subtask = SubagentTask.from_dict(
                    {
                        **t_dict,
                        "id": task_id,
                        "context": t_ctx,
                        "role": t_role,
                        "timeout": t_timeout,
                    },
                    default_id=task_id,
                )
                if subtask.dependencies:
                    has_dependencies = True
                subagent_tasks.append(subtask)

            # Check if any dependencies exist -> DAG mode, otherwise parallel batch mode
            if has_dependencies:
                logger.info("Executing dependency-directed DAG with %d tasks.", len(subagent_tasks))
                try:
                    dag_report = self.coordinator.delegate_dag(subagent_tasks)
                except Exception as exc:
                    return ToolResult(
                        success=False,
                        output=f"DAG validation or execution failed: {str(exc)}",
                        error=str(exc),
                    )

                summary_lines = [
                    f"Dependency DAG Execution {'SUCCEEDED' if dag_report['success'] else 'FAILED'} "
                    f"({dag_report['total_tasks']} tasks in {dag_report['duration_seconds']}s):"
                ]
                for r in dag_report["results"]:
                    summary_lines.append(f"- Task '{r['task_id']}' [{r['status']}]: {r.get('summary') or r.get('error')}")

                return ToolResult(
                    success=dag_report["success"],
                    output="\n".join(summary_lines),
                    metadata=dag_report,
                )
            else:
                logger.info("Executing flat parallel batch with %d tasks.", len(subagent_tasks))
                batch_report = self.coordinator.delegate_batch(subagent_tasks)
                summary_lines = [
                    f"Parallel Batch Execution {'SUCCEEDED' if batch_report['success'] else 'FAILED'} "
                    f"({batch_report['batch_size']} tasks in {batch_report['duration_seconds']}s):"
                ]
                for r in batch_report["results"]:
                    summary_lines.append(f"- Task '{r['task_id']}' [{r['status']}]: {r.get('summary') or r.get('error')}")

                return ToolResult(
                    success=batch_report["success"],
                    output="\n".join(summary_lines),
                    metadata=batch_report,
                )

        # Case B: Single task delegation
        if task and isinstance(task, str):
            single_task = SubagentTask(
                id=f"single_delegation_{int(time.time()*1000)}",
                description=task.strip(),
                context=context,
                role=SubagentRole.LEAF if role == "leaf" else SubagentRole.SPECIALIST,
                timeout_seconds=float(timeout) if timeout else 120.0,
            )
            result = self.coordinator.delegate_single(single_task)
            output_msg = (
                f"Subagent Execution {result.status.value} ({result.duration_seconds:.2f}s):\n"
                f"{result.summary if result.is_success else result.error}"
            )
            return ToolResult(
                success=result.is_success,
                output=output_msg,
                error=result.error if not result.is_success else None,
                metadata=result.to_dict(),
            )

        return ToolResult(
            success=False,
            output="Error: Either 'task' (string) or 'tasks' (list) must be provided for delegation.",
            error="Missing required task/tasks parameter",
        )
