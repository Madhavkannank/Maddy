"""Delegation Coordinator and Dependency DAG Orchestrator for M.A.D.D.Y.

Coordinates single subagent dispatches, parallel worker pools, and dependency-aware DAG workflows.
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.delegation.models import DelegationConfig, SubagentResult, SubagentRole, SubagentStatus, SubagentTask
from core.delegation.subagent import SubagentHarness
from core.errors import DelegationError, DependencyCycleError
from core.llm_client import CloudLLMClient
from core.permissions import PermissionGate
from core.tools.base import ToolRegistry
from core.trace_store import TraceStore

logger = logging.getLogger(__name__)


class DelegationCoordinator:
    """Coordinates subagent spawning, parallel execution, and dependency-aware DAG resolution."""

    def __init__(
        self,
        llm_client: Optional[CloudLLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        permission_gate: Optional[PermissionGate] = None,
        trace_store: Optional[TraceStore] = None,
        config: Optional[DelegationConfig] = None,
        current_depth: int = 1,
    ):
        self.llm_client = llm_client or CloudLLMClient()
        self.tool_registry = tool_registry or ToolRegistry()
        self.permission_gate = permission_gate or PermissionGate()
        self.trace_store = trace_store or TraceStore()
        self.config = config or DelegationConfig()
        self.current_depth = current_depth

    def _create_harness(self, role: SubagentRole = SubagentRole.LEAF) -> SubagentHarness:
        """Instantiate a SubagentHarness configured with incremented recursion depth."""
        return SubagentHarness(
            llm_client=self.llm_client,
            parent_tool_registry=self.tool_registry,
            permission_gate=self.permission_gate,
            trace_store=self.trace_store,
            config=self.config,
            current_depth=self.current_depth + 1,
        )

    def delegate_single(
        self,
        task: SubagentTask,
        parent_task_id: Optional[str] = None,
        harness: Optional[SubagentHarness] = None,
    ) -> SubagentResult:
        """Dispatch a single subagent task synchronously."""
        active_harness = harness or self._create_harness(role=task.role)
        return active_harness.execute(task, parent_task_id=parent_task_id)

    def delegate_batch(
        self,
        tasks: List[SubagentTask],
        parent_task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute multiple independent subagent tasks concurrently in a bounded thread pool."""
        if not tasks:
            return {"success": True, "batch_size": 0, "results": [], "duration_seconds": 0.0}

        start_time = time.time()
        num_workers = min(len(tasks), self.config.max_concurrent_workers)
        results: List[SubagentResult] = []

        def _worker_fn(t: SubagentTask) -> SubagentResult:
            harness = self._create_harness(role=t.role)
            return harness.execute(t, parent_task_id=parent_task_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_worker_fn, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    res = fut.result()
                    results.append(res)
                except Exception as exc:
                    original_task = futures[fut]
                    logger.error("Error in parallel worker for task %s: %s", original_task.id, exc, exc_info=True)
                    results.append(
                        SubagentResult(
                            task_id=original_task.id,
                            status=SubagentStatus.ERROR,
                            error=str(exc),
                        )
                    )

        # Maintain original order of tasks
        result_map = {r.task_id: r for r in results}
        ordered_results = [result_map.get(t.id, SubagentResult(task_id=t.id, status=SubagentStatus.ERROR, error="Missing result")) for t in tasks]

        all_success = all(r.is_success for r in ordered_results)
        elapsed = time.time() - start_time

        return {
            "success": all_success,
            "batch_size": len(tasks),
            "results": [r.to_dict() for r in ordered_results],
            "duration_seconds": round(elapsed, 3),
        }

    def validate_dag(self, tasks: List[SubagentTask]) -> None:
        """Validate task graph dependencies: detect cycles, self-dependencies, and missing references."""
        task_ids = {t.id for t in tasks}

        # 1. Check self-dependencies and existence
        for t in tasks:
            if t.id in t.dependencies:
                raise DependencyCycleError(f"Task '{t.id}' cannot depend on itself.", cycle=[t.id, t.id])
            for dep in t.dependencies:
                if dep not in task_ids:
                    raise DelegationError(
                        f"Task '{t.id}' depends on non-existent task '{dep}'.",
                        task_id=t.id,
                    )

        # 2. Cycle detection via Kahn's algorithm / in-degree tracking
        in_degree: Dict[str, int] = {t.id: 0 for t in tasks}
        adj_list: Dict[str, List[str]] = defaultdict(list)

        for t in tasks:
            for dep in t.dependencies:
                adj_list[dep].append(t.id)
                in_degree[t.id] += 1

        queue = deque([t_id for t_id, deg in in_degree.items() if deg == 0])
        visited_count = 0

        while queue:
            curr = queue.popleft()
            visited_count += 1
            for neighbor in adj_list[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(tasks):
            # Find elements involved in cycle
            cycle_nodes = [t_id for t_id, deg in in_degree.items() if deg > 0]
            raise DependencyCycleError(
                f"Cyclic dependency detected in task graph among nodes: {cycle_nodes}",
                cycle=cycle_nodes,
            )

    def delegate_dag(
        self,
        tasks: List[SubagentTask],
        parent_task_id: Optional[str] = None,
        custom_executor: Optional[Callable[[SubagentTask, Optional[str]], SubagentResult]] = None,
    ) -> Dict[str, Any]:
        """Execute a dependency-directed acyclic graph (DAG) of subagent tasks.

        - Dispatches ready tasks concurrently in topological waves.
        - Injects upstream dependency outputs into downstream task context.
        - Automatically marks downstream tasks as SKIPPED if their upstream dependencies fail.
        """
        if not tasks:
            return {"success": True, "total_tasks": 0, "results": [], "duration_seconds": 0.0}

        # 1. Validate graph integrity
        self.validate_dag(tasks)

        start_time = time.time()
        completed_results: Dict[str, SubagentResult] = {}
        pending_tasks: Dict[str, SubagentTask] = {t.id: t for t in tasks}

        while pending_tasks:
            # 2. Find ready tasks (all dependencies met and SUCCESS)
            ready_tasks: List[SubagentTask] = []
            skipped_tasks: List[SubagentTask] = []

            for t_id, task in list(pending_tasks.items()):
                # Check dependencies status
                deps_satisfied = True
                dep_failed = False
                failed_dep_id = ""

                for dep_id in task.dependencies:
                    dep_res = completed_results.get(dep_id)
                    if not dep_res:
                        deps_satisfied = False
                        break
                    if not dep_res.is_success:
                        dep_failed = True
                        failed_dep_id = dep_id
                        break

                if dep_failed:
                    skipped_tasks.append(task)
                elif deps_satisfied:
                    ready_tasks.append(task)

            # 3. Propagate failure / skip blocked tasks
            for skipped_task in skipped_tasks:
                failed_dep = completed_results.get(skipped_task.dependencies[0])
                reason = f"Upstream dependency failed: '{failed_dep.task_id if failed_dep else 'unknown'}' (status: {failed_dep.status.value if failed_dep else 'failed'})."
                logger.warning("Skipping task '%s': %s", skipped_task.id, reason)
                res = SubagentResult(
                    task_id=skipped_task.id,
                    status=SubagentStatus.SKIPPED,
                    error=reason,
                )
                completed_results[skipped_task.id] = res
                del pending_tasks[skipped_task.id]

            if not ready_tasks:
                if pending_tasks:
                    # Deadlock detected: pending tasks remain but none are ready or skippable
                    for t_id in list(pending_tasks.keys()):
                        completed_results[t_id] = SubagentResult(
                            task_id=t_id,
                            status=SubagentStatus.ERROR,
                            error="Graph execution stalled: unresolvable dependency deadlock.",
                        )
                        del pending_tasks[t_id]
                break

            # 4. Context propagation: Inject upstream results into ready task context
            for task in ready_tasks:
                prereq_blocks = []
                for dep_id in task.dependencies:
                    dep_res = completed_results[dep_id]
                    prereq_blocks.append(
                        f"--- Upstream Result from Task '{dep_id}' ---\n{dep_res.summary}"
                    )
                if prereq_blocks:
                    injected_context = "\n\n".join(prereq_blocks)
                    if task.context:
                        task.context = f"{task.context}\n\n[PREREQUISITE OUTPUTS]:\n{injected_context}"
                    else:
                        task.context = f"[PREREQUISITE OUTPUTS]:\n{injected_context}"

            # 5. Execute current wave of ready tasks concurrently
            num_workers = min(len(ready_tasks), self.config.max_concurrent_workers)

            def _execute_single(t: SubagentTask) -> SubagentResult:
                if custom_executor:
                    return custom_executor(t, parent_task_id)
                harness = self._create_harness(role=t.role)
                return harness.execute(t, parent_task_id=parent_task_id)

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(_execute_single, t): t for t in ready_tasks}
                for fut in concurrent.futures.as_completed(futures):
                    t_orig = futures[fut]
                    try:
                        res = fut.result()
                        completed_results[t_orig.id] = res
                    except Exception as exc:
                        logger.error("Error executing DAG task %s: %s", t_orig.id, exc, exc_info=True)
                        completed_results[t_orig.id] = SubagentResult(
                            task_id=t_orig.id,
                            status=SubagentStatus.ERROR,
                            error=str(exc),
                        )
                    del pending_tasks[t_orig.id]

        elapsed = time.time() - start_time
        all_ordered = [completed_results[t.id] for t in tasks]
        all_success = all(r.is_success for r in all_ordered)

        return {
            "success": all_success,
            "total_tasks": len(tasks),
            "results": [r.to_dict() for r in all_ordered],
            "duration_seconds": round(elapsed, 3),
        }
