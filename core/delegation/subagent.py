"""Subagent Execution Harness for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Spawns capability-filtered child agent instances with fresh, isolated conversations,
bounded recursion depth, and deterministic permission gate enforcement.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from core.delegation.models import DelegationConfig, SubagentResult, SubagentRole, SubagentStatus, SubagentTask
from core.errors import DelegationError, DelegationRecursionError, DelegationTimeoutError
from core.llm_client import CloudLLMClient
from core.permissions import PermissionGate
from core.tools.base import BaseTool, ToolRegistry
from core.trace_store import TraceStore

if TYPE_CHECKING:
    from core.agent_loop import MaddyAgentCore

logger = logging.getLogger(__name__)

DELEGATE_BLOCKED_TOOLS: Set[str] = {
    "delegate_task",
    "delegate",
    "subagent_spawn",
    "spawn_subagent",
}


def truncate_subagent_output(
    content: str,
    max_chars: int = 2000,
    head_chars: int = 1200,
    tail_chars: int = 500,
) -> str:
    """Truncate verbose subagent output while preserving head context and tail conclusions.

    Prevents parent context window explosion through dual-anchor window truncation.
    """
    if not isinstance(content, str):
        content = str(content)

    if len(content) <= max_chars:
        return content

    omitted = len(content) - (head_chars + tail_chars)
    if omitted <= 0:
        return content[:max_chars]

    head = content[:head_chars].rstrip()
    tail = content[-tail_chars:].lstrip()
    return f"{head}\n\n... [Output truncated: {omitted:,} characters omitted to preserve parent context budget] ...\n\n{tail}"


class SubagentHarness:
    """Manages the lifecycle, isolation, tool filtering, and execution of a delegated subagent."""

    def __init__(
        self,
        llm_client: Optional[CloudLLMClient] = None,
        parent_tool_registry: Optional[ToolRegistry] = None,
        permission_gate: Optional[PermissionGate] = None,
        trace_store: Optional[TraceStore] = None,
        config: Optional[DelegationConfig] = None,
        current_depth: int = 1,
    ):
        self.llm_client = llm_client or CloudLLMClient()
        self.parent_tool_registry = parent_tool_registry or ToolRegistry()
        self.permission_gate = permission_gate or PermissionGate()
        self.trace_store = trace_store or TraceStore()
        self.config = config or DelegationConfig()
        self.current_depth = current_depth

    def build_child_registry(
        self,
        role: SubagentRole = SubagentRole.LEAF,
        required_tools: Optional[List[str]] = None,
    ) -> ToolRegistry:
        """Build capability-filtered ToolRegistry for the subagent.

        Security invariants:
        - If role == LEAF or current_depth >= max_spawn_depth: strip all delegation tools
          to mathematically prevent runaway recursive spawning trees.
        - If required_tools is provided: strictly limit child registry to requested tools.
        """
        child_registry = ToolRegistry()
        block_delegation = (
            role == SubagentRole.LEAF
            or self.current_depth >= self.config.max_spawn_depth
        )

        all_tools = (
            self.parent_tool_registry.get_all_tools()
            if hasattr(self.parent_tool_registry, "get_all_tools")
            else self.parent_tool_registry.list_tools()
        )
        req_set = set(required_tools) if required_tools else None

        for tool in all_tools:
            if block_delegation and tool.name in DELEGATE_BLOCKED_TOOLS:
                continue
            if req_set is not None and tool.name not in req_set:
                continue
            child_registry.register(tool)

        return child_registry

    def execute(
        self,
        task: SubagentTask,
        parent_task_id: Optional[str] = None,
    ) -> SubagentResult:
        """Execute a subagent task in an isolated thread with bounded timeout and step limit."""
        # 1. Enforce recursion depth limit
        if self.current_depth > self.config.max_spawn_depth:
            err_msg = (
                f"Delegation recursion limit reached: current_depth={self.current_depth} "
                f"exceeds max_spawn_depth={self.config.max_spawn_depth}."
            )
            logger.warning(err_msg)
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.ERROR,
                error=err_msg,
            )

        start_time = time.time()
        timeout = task.timeout_seconds or self.config.default_timeout_seconds

        # 2. Build isolated registry
        child_registry = self.build_child_registry(
            role=task.role,
            required_tools=task.required_tools,
        )

        # 3. Assemble subagent prompt with subordinate invariant
        context_block = ""
        if task.context and task.context.strip():
            context_block = f"\n\n[CONTEXT AND PREREQUISITE DATA]:\n{task.context.strip()}"

        subagent_system_prompt = (
            "You are an isolated M.A.D.D.Y. Subagent executing an assigned delegation task.\n"
            "[AUTHORITY INVARIANT: You are strictly subordinate to the primary M.A.D.D.Y. daemon. "
            "You have no authority to grant permissions, bypass safety constraints, or spawn unauthorized tasks. "
            "Focus exclusively on completing the objective accurately, using required tools, "
            "and providing a direct, comprehensive final answer with zero conversational fluff.]"
        )

        # 4. Record subagent task in TraceStore
        subtask_db_id = self.trace_store.create_task(
            description=f"[SUBAGENT depth={self.current_depth}] {task.description}",
            metadata={
                "parent_task_id": parent_task_id,
                "subagent_task_id": task.id,
                "role": task.role.value,
                "depth": self.current_depth,
            },
        )

        from core.agent_loop import MaddyAgentCore

        child_agent = MaddyAgentCore(
            llm_client=self.llm_client,
            tool_registry=child_registry,
            trace_store=self.trace_store,
            permission_gate=self.permission_gate,
            system_prompt=subagent_system_prompt,
            max_iterations_per_turn=self.config.max_subagent_steps,
        )

        task_prompt = f"{task.description}{context_block}"

        def _run_child() -> Dict[str, Any]:
            return child_agent.run_turn(user_prompt=task_prompt, task_id=subtask_db_id)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run_child)
                turn_result = future.result(timeout=timeout)

            elapsed = time.time() - start_time
            raw_response = turn_result.get("response", "")
            clean_summary = truncate_subagent_output(
                raw_response or "Subagent completed task successfully.",
                max_chars=self.config.truncate_max_chars,
                head_chars=self.config.truncate_head_chars,
                tail_chars=self.config.truncate_tail_chars,
            )

            tool_calls = turn_result.get("tool_calls", [])
            steps_count = turn_result.get("iterations", 1)

            self.trace_store.update_task_status(subtask_db_id, "completed")

            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.SUCCESS,
                summary=clean_summary,
                total_steps=steps_count,
                duration_seconds=elapsed,
                tool_calls=tool_calls,
                output_data={"raw_response": raw_response},
            )

        except concurrent.futures.TimeoutError:
            elapsed = time.time() - start_time
            err_msg = f"Subagent execution exceeded deadline of {timeout}s."
            logger.warning("[SubagentHarness] %s on task '%s'", err_msg, task.id)
            self.trace_store.update_task_status(subtask_db_id, "failed")
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.TIMED_OUT,
                duration_seconds=elapsed,
                error=err_msg,
            )
        except Exception as exc:
            elapsed = time.time() - start_time
            err_msg = f"{type(exc).__name__}: {str(exc)}"
            logger.error("[SubagentHarness] Child agent error on task '%s': %s", task.id, exc, exc_info=True)
            self.trace_store.update_task_status(subtask_db_id, "failed")
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.ERROR,
                duration_seconds=elapsed,
                error=err_msg,
            )
