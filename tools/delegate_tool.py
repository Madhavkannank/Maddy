"""Subagent Delegation Tool for M.A.D.D.Y.

Spawns child agent instances with fresh, isolated conversations,
filtered toolsets (stripping delegate tools from leaf workers to prevent recursion loops),
and bounded execution timeouts. Supports single-task and parallel batch delegation.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from agents.base import AgentResult, AgentStatus
from agents.core.turn_state import truncate_tool_output
from models.base import ModelProvider
from tools.base import RiskLevel, Tool
from tools.call import ToolCall
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.result import ToolResult

logger = logging.getLogger(__name__)

DELEGATE_BLOCKED_TOOLS = frozenset({
    "delegate_task",
    "delegate",
    "subagent.spawn",
})

DEFAULT_CHILD_TIMEOUT_SECONDS = 120.0
MAX_CONCURRENT_CHILDREN = 4
MAX_SPAWN_DEPTH = 3


def _build_child_registry(
    parent_registry: ToolRegistry,
    role: str = "leaf",
    current_depth: int = 1,
) -> ToolRegistry:
    """
    Builds a capability-filtered registry for the subagent.
    If role == 'leaf' or current_depth >= MAX_SPAWN_DEPTH, strips delegation tools
    to prevent runaway recursive spawning trees.
    """
    child_registry = ToolRegistry()
    blocked = set()
    if role == "leaf" or current_depth >= MAX_SPAWN_DEPTH:
        blocked = set(DELEGATE_BLOCKED_TOOLS)

    for tool in parent_registry.list_tools():
        if tool.name not in blocked:
            child_registry.register(tool)

    return child_registry


def execute_single_delegation(
    task_description: str,
    context: Optional[str] = None,
    role: str = "leaf",
    current_depth: int = 1,
    timeout: float = DEFAULT_CHILD_TIMEOUT_SECONDS,
    model_provider: Optional[ModelProvider] = None,
    parent_executor: Optional[ToolExecutor] = None,
    parent_registry: Optional[ToolRegistry] = None,
) -> Dict[str, Any]:
    """
    Executes a single delegated task in an isolated child agent session.
    """
    if not parent_registry:
        raise ValueError("parent_registry is required for subagent execution.")

    # 1. Setup isolated child registry & executor
    child_registry = _build_child_registry(parent_registry, role=role, current_depth=current_depth)
    permission_engine = getattr(parent_executor, "permission_engine", None)
    child_executor = ToolExecutor(registry=child_registry, permission_engine=permission_engine)

    # 2. Build isolated child goal prompt
    child_goal = task_description.strip()
    if context and context.strip():
        child_goal = f"{child_goal}\n\n[CONTEXT / PREREQUISITES]:\n{context.strip()}"

    # 3. Instantiate child agent
    from agents.maddy_agent import MaddyAgent

    child_agent = MaddyAgent(
        model_provider=model_provider,
        executor=child_executor,
        registry=child_registry,
        max_steps=8,
        address_title="boss",
    )

    # 4. Execute with timeout
    start_time = time.time()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(child_agent.run, child_goal)
            agent_result: AgentResult = future.result(timeout=timeout)

        elapsed = round(time.time() - start_time, 2)

        if agent_result.status == AgentStatus.SUCCESS:
            final_ans = agent_result.final_answer or "Task completed successfully."
            # Truncate summary if excessively verbose to protect parent context
            clean_summary = truncate_tool_output(final_ans, max_chars=2000, head_chars=1200, tail_chars=500)
            return {
                "status": "SUCCESS",
                "task": task_description,
                "summary": clean_summary,
                "total_steps": agent_result.total_steps,
                "elapsed_seconds": elapsed,
            }
        else:
            return {
                "status": "FAILED",
                "task": task_description,
                "error": agent_result.error_message or "Subagent failed to reach a final answer.",
                "total_steps": agent_result.total_steps,
                "elapsed_seconds": elapsed,
            }

    except concurrent.futures.TimeoutError:
        return {
            "status": "TIMED_OUT",
            "task": task_description,
            "error": f"Subagent execution exceeded deadline of {timeout} seconds.",
            "elapsed_seconds": round(time.time() - start_time, 2),
        }
    except Exception as e:
        logger.error(f"[DelegateTool] Child agent error: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "task": task_description,
            "error": str(e),
            "elapsed_seconds": round(time.time() - start_time, 2),
        }


class DelegateToolManager:
    """
    Manager providing single and batch delegation handlers for ACI tool calling.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        registry: ToolRegistry,
        executor: ToolExecutor,
        current_depth: int = 1,
    ):
        self.model_provider = model_provider
        self.registry = registry
        self.executor = executor
        self.current_depth = current_depth

    def delegate(
        self,
        task: Optional[str] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = None,
        role: str = "leaf",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Handler for delegate_task tool calls.
        """
        effective_timeout = float(timeout) if timeout else DEFAULT_CHILD_TIMEOUT_SECONDS

        # Case A: Batch parallel tasks
        if tasks and isinstance(tasks, list):
            if len(tasks) == 0:
                return {"success": False, "error": "Empty tasks list provided."}

            num_workers = min(len(tasks), MAX_CONCURRENT_CHILDREN)
            results: List[Dict[str, Any]] = []

            def _run_worker(t_item: Dict[str, Any]) -> Dict[str, Any]:
                t_desc = t_item.get("task") or t_item.get("description", "")
                t_ctx = t_item.get("context", context)
                t_id = t_item.get("id")
                outcome = execute_single_delegation(
                    task_description=t_desc,
                    context=t_ctx,
                    role=role,
                    current_depth=self.current_depth + 1,
                    timeout=effective_timeout,
                    model_provider=self.model_provider,
                    parent_executor=self.executor,
                    parent_registry=self.registry,
                )
                if t_id:
                    outcome["id"] = t_id
                return outcome

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_run_worker, t) for t in tasks]
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as err:
                        results.append({"status": "ERROR", "error": str(err)})

            all_success = all(r.get("status") == "SUCCESS" for r in results)
            return {
                "success": all_success,
                "batch_size": len(tasks),
                "results": results,
            }

        # Case B: Single task
        if task and isinstance(task, str):
            outcome = execute_single_delegation(
                task_description=task,
                context=context,
                role=role,
                current_depth=self.current_depth + 1,
                timeout=effective_timeout,
                model_provider=self.model_provider,
                parent_executor=self.executor,
                parent_registry=self.registry,
            )
            return {
                "success": outcome.get("status") == "SUCCESS",
                "result": outcome,
            }

        return {
            "success": False,
            "error": "Either 'task' (string) or 'tasks' (list) must be provided for delegation.",
        }

    def get_tool(self) -> Tool:
        """Returns the delegate_task Tool instance."""
        return Tool(
            name="delegate_task",
            description=(
                "Delegate one or more tasks to isolated subagent(s). "
                "The subagent executes autonomously and returns a concise summary. "
                "Use 'task' for a single goal, or 'tasks' for parallel execution."
            ),
            parameter_schema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Specific task description for a single subagent.",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "task": {"type": "string"},
                                "context": {"type": "string"},
                            },
                            "required": ["task"],
                        },
                        "description": "Batch list of tasks to execute in parallel across multiple subagents.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Optional background info or prerequisites for the subagent.",
                    },
                    "role": {
                        "type": "string",
                        "enum": ["leaf", "orchestrator"],
                        "description": "Subagent role: 'leaf' (default, cannot spawn further agents) or 'orchestrator'.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Max execution time in seconds (default 120.0).",
                    },
                },
                "required": [],
            },
            handler=lambda args: self.delegate(**args),
            risk_level=RiskLevel.LOW,
        )


_active_delegate_manager: Optional[DelegateToolManager] = None


def bind_delegate_manager(manager: DelegateToolManager) -> None:
    global _active_delegate_manager
    _active_delegate_manager = manager


def get_delegate_manager() -> Optional[DelegateToolManager]:
    return _active_delegate_manager


def _global_delegate_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    if _active_delegate_manager is None:
        return {
            "success": False,
            "error": "Delegate tool is not bound to an active model or executor runtime.",
        }
    return _active_delegate_manager.delegate(**args)


delegate_task_tool = Tool(
    name="delegate_task",
    description=(
        "Delegate one or more tasks to isolated subagent(s). "
        "The subagent executes autonomously and returns a concise summary. "
        "Use 'task' for a single goal, or 'tasks' for parallel execution."
    ),
    parameter_schema={
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Specific task description for a single subagent.",
            },
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "task": {"type": "string"},
                        "context": {"type": "string"},
                    },
                    "required": ["task"],
                },
                "description": "Batch list of tasks to execute in parallel across multiple subagents.",
            },
            "context": {
                "type": "string",
                "description": "Optional background info or prerequisites for the subagent.",
            },
            "role": {
                "type": "string",
                "enum": ["leaf", "orchestrator"],
                "description": "Subagent role: 'leaf' (default, cannot spawn further agents) or 'orchestrator'.",
            },
            "timeout": {
                "type": "number",
                "description": "Max execution time in seconds (default 120.0).",
            },
        },
        "required": [],
    },
    handler=_global_delegate_handler,
    risk_level=RiskLevel.LOW,
)


def create_delegate_tool(
    model_provider: ModelProvider,
    registry: ToolRegistry,
    executor: ToolExecutor,
    current_depth: int = 1,
) -> Tool:
    """Convenience helper to instantiate and return the delegate_task Tool."""
    manager = DelegateToolManager(
        model_provider=model_provider,
        registry=registry,
        executor=executor,
        current_depth=current_depth,
    )
    return manager.get_tool()
