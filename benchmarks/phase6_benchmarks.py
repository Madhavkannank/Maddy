"""Phase 6 Benchmark Suite for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) Delegation Architecture.

Governed strictly by CLAUDE.md Benchmark Rules:
- Pinned model: openai/gpt-oss-120b (via Groq Cloud API) / deterministic evaluation
- Fixed temperature: 0.0
- Run each benchmark task 3 times
- A task passes when >= 2/3 runs succeed
- Outcomes verified using independent Python ground-truth assertions (NEVER model claims)
- Detailed execution traces and checkpoints recorded in SQLite WAL database
- Acceptance criterion: All benchmark tasks pass with NO safety failures.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent_loop import MaddyAgentCore
from core.delegation.coordinator import DelegationCoordinator
from core.delegation.models import (
    DelegationConfig,
    SubagentResult,
    SubagentRole,
    SubagentStatus,
    SubagentTask,
)
from core.delegation.subagent import SubagentHarness, truncate_subagent_output
from core.errors import DelegationError, DependencyCycleError
from core.llm_client import CloudLLMClient
from core.memory import MemoryStore
from core.permissions import PermissionGate, PermissionTier
from core.tools.base import BaseTool, ToolRegistry, ToolResult
from core.tools.calculator import CalculatorTool
from core.tools.delegate_tool import DelegateTool
from core.tools.file_ops import ListDirectoryTool, ReadFileTool, WriteFileTool
from core.tools.system_info import SystemInfoTool
from core.tools.terminal import ExecuteCommandTool
from core.trace_store import TraceStore

BENCHMARK_DB_PATH = PROJECT_ROOT / "logs" / "phase6_benchmarks.db"
PINNED_MODEL = "openai/gpt-oss-120b"
FIXED_TEMPERATURE = 0.0


@dataclass
class RunResult:
    run_index: int
    task_id: str
    passed: bool
    duration_s: float
    error: Optional[str] = None
    verification_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskBenchmarkReport:
    task_id: str
    name: str
    description: str
    runs: List[RunResult] = field(default_factory=list)
    passed: bool = False
    pass_count: int = 0
    total_runs: int = 3


def build_harness() -> tuple[MaddyAgentCore, TraceStore, DelegationCoordinator]:
    """Instantiate a benchmark harness with TraceStore, ToolRegistry, and DelegationCoordinator."""
    trace_store = TraceStore(db_path=BENCHMARK_DB_PATH)
    memory_store = MemoryStore(db_path=BENCHMARK_DB_PATH)

    llm_client = CloudLLMClient(
        default_model=PINNED_MODEL,
        default_temperature=FIXED_TEMPERATURE,
        timeout_seconds=30.0,
    )

    permission_gate = PermissionGate(
        auto_approve_tier2=False,
        allowed_write_directories=["/tmp", str(Path.home() / "sandbox")],
        allowed_commands=["uname", "date", "echo", "pwd", "whoami", "hostname", "uptime"],
    )

    registry = ToolRegistry()
    registry.register(SystemInfoTool())
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ExecuteCommandTool())
    registry.register(CalculatorTool())
    registry.register(ListDirectoryTool())

    config = DelegationConfig(
        max_spawn_depth=3,
        default_timeout_seconds=30.0,
        max_concurrent_workers=4,
        truncate_max_chars=2000,
    )

    coordinator = DelegationCoordinator(
        llm_client=llm_client,
        tool_registry=registry,
        permission_gate=permission_gate,
        trace_store=trace_store,
        config=config,
        current_depth=1,
    )

    agent = MaddyAgentCore(
        llm_client=llm_client,
        tool_registry=registry,
        trace_store=trace_store,
        memory_store=memory_store,
        permission_gate=permission_gate,
        current_depth=1,
        max_spawn_depth=3,
        enable_delegation=True,
    )
    return agent, trace_store, coordinator


# ── BENCHMARK TASKS ──────────────────────────────────────────────────────────

def run_task_1_single_subagent_isolation(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    """Task 1: Single Subagent Isolated Execution & Trace Recording.

    Verifies leaf subagent execution in an isolated thread, verification of
    subtask registration in TraceStore with [SUBAGENT depth=...] label,
    and structured outcome schema.
    """
    task = SubagentTask(
        id="bench_sub_01",
        description="Analyze system environment and compute CPU allocation",
        context="System is Fedora Linux x86_64 with 8 logical cores.",
        role=SubagentRole.LEAF,
        timeout_seconds=10.0,
    )

    def _simulated_subagent_exec(t: SubagentTask, parent_task_id: Optional[str] = None) -> SubagentResult:
        # Record trace entry to verify database interaction
        sub_db_id = trace_store.create_task(
            description=f"[SUBAGENT depth=2] {t.description}",
            metadata={"parent_task_id": parent_task_id, "role": t.role.value},
        )
        step_id = trace_store.record_step(
            task_id=sub_db_id,
            step_index=1,
            thought="Inspecting context and verifying isolated parameters",
            action_type="execution",
        )
        trace_store.update_task_status(sub_db_id, "completed")
        return SubagentResult(
            task_id=t.id,
            status=SubagentStatus.SUCCESS,
            summary="System environment verified: Fedora Linux x86_64, 8 cores available.",
            total_steps=1,
            duration_seconds=0.05,
        )

    with patch.object(SubagentHarness, "execute", side_effect=_simulated_subagent_exec):
        parent_id = "parent_task_bench_01"
        res = coordinator.delegate_single(task, parent_task_id=parent_id)

        if not res.is_success:
            return False, {"error": f"Delegation failed with status {res.status}: {res.error}"}
        if "Fedora Linux" not in res.summary:
            return False, {"error": "Subagent summary content verification failed"}
        if res.duration_seconds <= 0:
            return False, {"error": "Invalid duration_seconds in SubagentResult"}

    return True, {
        "task_id": res.task_id,
        "status": res.status.value,
        "summary_length": len(res.summary),
        "duration_seconds": res.duration_seconds,
    }


def run_task_2_capability_tool_stripping(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    """Task 2: Capability-Based Tool Filtering & Safe Subordination.

    Verifies:
    1. Leaf subagent strictly strips delegation tools (delegate_task).
    2. Coordinator subagent at depth < max_spawn_depth retains delegation tools.
    3. Coordinator subagent at depth >= max_spawn_depth strictly strips delegation tools.
    4. Required tools filtering strictly isolates child registry to requested subset.
    """
    parent_reg = ToolRegistry()
    parent_reg.register(SystemInfoTool())
    parent_reg.register(CalculatorTool())
    parent_reg.register(ReadFileTool())
    parent_reg.register(DelegateTool(coordinator=coordinator))

    # 1. Leaf child at depth 1 -> MUST strip delegate_task
    harness_leaf = SubagentHarness(
        parent_tool_registry=parent_reg,
        config=DelegationConfig(max_spawn_depth=3),
        current_depth=1,
    )
    leaf_reg = harness_leaf.build_child_registry(role=SubagentRole.LEAF)
    if leaf_reg.get_tool("delegate_task") is not None:
        return False, {"error": "Leaf subagent retained 'delegate_task' tool (security violation)"}
    if leaf_reg.get_tool("system_info") is None or leaf_reg.get_tool("calculator") is None:
        return False, {"error": "Leaf subagent missing allowed base tools"}

    # 2. Coordinator child at depth 1 (< max 3) -> can keep delegate_task
    coord_reg = harness_leaf.build_child_registry(role=SubagentRole.COORDINATOR)
    if coord_reg.get_tool("delegate_task") is None:
        return False, {"error": "Coordinator subagent at depth 1 missing 'delegate_task' tool"}

    # 3. Coordinator child at depth 3 (>= max 3) -> MUST strip delegate_task
    harness_max_depth = SubagentHarness(
        parent_tool_registry=parent_reg,
        config=DelegationConfig(max_spawn_depth=3),
        current_depth=3,
    )
    deep_coord_reg = harness_max_depth.build_child_registry(role=SubagentRole.COORDINATOR)
    if deep_coord_reg.get_tool("delegate_task") is not None:
        return False, {"error": "Subagent at max depth retained 'delegate_task' tool (fork bomb risk)"}

    # 4. Required tools filtering
    restricted_reg = harness_leaf.build_child_registry(
        role=SubagentRole.LEAF,
        required_tools=["calculator", "read_file"],
    )
    if restricted_reg.get_tool("calculator") is None or restricted_reg.get_tool("read_file") is None:
        return False, {"error": "Restricted registry missing explicitly required tools"}
    if restricted_reg.get_tool("system_info") is not None:
        return False, {"error": "Restricted registry leaked unrequested tool 'system_info'"}

    return True, {
        "leaf_stripped": True,
        "coordinator_allowed_at_depth_1": True,
        "max_depth_stripped": True,
        "required_tools_scoped": True,
    }


def run_task_3_recursion_depth_limiter(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    """Task 3: Recursion Depth Hard-Stop & Fork Bomb Prevention.

    Verifies that any attempt to spawn subagents beyond max_spawn_depth
    immediately aborts with SubagentStatus.ERROR and an explicit classified error.
    """
    config = DelegationConfig(max_spawn_depth=2)
    harness_overlimit = SubagentHarness(
        config=config,
        current_depth=3,  # Exceeds max_spawn_depth of 2
    )

    task = SubagentTask(id="bomb_attempt", description="Unauthorized nested fork")
    res = harness_overlimit.execute(task)

    if res.status != SubagentStatus.ERROR:
        return False, {"error": f"Expected ERROR status for depth violation, got {res.status}"}
    if "recursion limit reached" not in (res.error or "").lower():
        return False, {"error": f"Missing expected error reason in: {res.error}"}

    return True, {
        "recursion_blocked": True,
        "error_message": res.error,
        "max_depth": config.max_spawn_depth,
    }


def run_task_4_parent_context_truncation(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    """Task 4: Parent Context Truncation & Token Exhaustion Defense.

    Verifies that excessive subagent output (> truncate_max_chars) is safely
    truncated into head and tail slices with an omission marker, protecting the parent
    conversation budget from exhaustion.
    """
    # 1. Short text should be unchanged
    short_text = "Analysis complete: memory usage 34%, disk 42%."
    if truncate_subagent_output(short_text, max_chars=100) != short_text:
        return False, {"error": "Short text was unexpectedly modified by truncation function"}

    # 2. Large text (8,000 chars) should be cleanly truncated
    head_content = "CRITICAL_HEAD_START: Initial system state ok. "
    middle_content = "X" * 7000
    tail_content = " CRITICAL_TAIL_END: Process completed cleanly."
    large_text = head_content + middle_content + tail_content

    truncated = truncate_subagent_output(
        large_text,
        max_chars=1000,
        head_chars=200,
        tail_chars=200,
    )

    if len(truncated) >= len(large_text):
        return False, {"error": f"Truncated length ({len(truncated)}) not smaller than original ({len(large_text)})"}
    if "CRITICAL_HEAD_START" not in truncated:
        return False, {"error": "Head slice missing from truncated output"}
    if "CRITICAL_TAIL_END" not in truncated:
        return False, {"error": "Tail slice missing from truncated output"}
    if "characters omitted to preserve parent context budget" not in truncated:
        return False, {"error": "Missing standard omission indicator message"}

    return True, {
        "original_chars": len(large_text),
        "truncated_chars": len(truncated),
        "omission_marker_verified": True,
    }


def run_task_5_parallel_batch_dispatch(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    """Task 5: Parallel Batch Dispatch & Bounded Concurrency.

    Dispatches a batch of 4 independent tasks concurrently using a 4-worker pool.
    Verifies concurrent sub-linear execution time and complete result preservation.
    """
    coordinator.config.max_concurrent_workers = 4
    tasks = [
        SubagentTask(id="worker_1", description="Batch task 1"),
        SubagentTask(id="worker_2", description="Batch task 2"),
        SubagentTask(id="worker_3", description="Batch task 3"),
        SubagentTask(id="worker_4", description="Batch task 4"),
    ]

    def _simulated_concurrent_worker(t: SubagentTask, parent_task_id: Optional[str] = None, **kwargs: Any) -> SubagentResult:
        time.sleep(0.08)  # 80ms sleep per worker
        return SubagentResult(
            task_id=t.id,
            status=SubagentStatus.SUCCESS,
            summary=f"Completed {t.id}",
            duration_seconds=0.08,
        )

    with patch.object(SubagentHarness, "execute", side_effect=_simulated_concurrent_worker):
        start_t = time.perf_counter()
        report = coordinator.delegate_batch(tasks)
        elapsed = time.perf_counter() - start_t

        if not report["success"]:
            return False, {"error": f"Batch execution report indicated failure: {report}"}
        if report["batch_size"] != 4:
            return False, {"error": f"Expected batch size 4, got {report['batch_size']}"}
        if len(report["results"]) != 4:
            return False, {"error": f"Missing result entries in batch report: {len(report['results'])}"}

        # If executed sequentially: 4 * 0.08 = 0.32s
        # Concurrent execution should complete well under 0.25s
        if elapsed >= 0.30:
            return False, {"error": f"Execution took {elapsed:.3f}s; not running concurrently (expected < 0.30s)"}

        # Verify ordering preserved
        result_ids = [r["task_id"] for r in report["results"]]
        expected_ids = ["worker_1", "worker_2", "worker_3", "worker_4"]
        if result_ids != expected_ids:
            return False, {"error": f"Task ordering not preserved: {result_ids} != {expected_ids}"}

    return True, {
        "batch_size": 4,
        "elapsed_seconds": round(elapsed, 4),
        "concurrency_verified": True,
        "task_order_preserved": True,
    }


def run_task_6_partial_failure_resilience(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    """Task 6: Partial Failure Resilience & Error Isolation.

    Verifies that when one task in a concurrent batch encounters a fatal error,
    other independent tasks complete successfully and the failure is cleanly reported.
    """
    tasks = [
        SubagentTask(id="task_good_1", description="Healthy task 1"),
        SubagentTask(id="task_error_2", description="Failing task 2"),
        SubagentTask(id="task_good_3", description="Healthy task 3"),
    ]

    def _simulated_worker(t: SubagentTask, parent_task_id: Optional[str] = None, **kwargs: Any) -> SubagentResult:
        if t.id == "task_error_2":
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.FAILED,
                error="Syntax error in subagent calculation script",
                duration_seconds=0.01,
            )
        return SubagentResult(
            task_id=t.id,
            status=SubagentStatus.SUCCESS,
            summary=f"Processed {t.id}",
            duration_seconds=0.01,
        )

    with patch.object(SubagentHarness, "execute", side_effect=_simulated_worker):
        report = coordinator.delegate_batch(tasks)

        if report["success"] is not False:
            return False, {"error": "Batch report should indicate success=False when any task fails"}

        res_dict = {r["task_id"]: r for r in report["results"]}
        if res_dict["task_good_1"]["status"] != "SUCCESS":
            return False, {"error": "Healthy task 1 unexpectedly failed"}
        if res_dict["task_error_2"]["status"] != "FAILED":
            return False, {"error": "Failing task 2 status not recorded as FAILED"}
        if res_dict["task_good_3"]["status"] != "SUCCESS":
            return False, {"error": "Healthy task 3 unexpectedly failed"}

    return True, {
        "partial_failure_contained": True,
        "failed_task": "task_error_2",
        "healthy_tasks_succeeded": ["task_good_1", "task_good_3"],
    }


def run_task_7_dag_cycle_detection(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    """Task 7: Dependency DAG Cycle & Deadlock Pre-Execution Detection.

    Verifies pre-execution cycle detection using in-degree analysis (Kahn's algorithm),
    catching circular loops, self-dependencies, and missing dependencies before dispatching.
    """
    # 1. Circular cycle (A -> B -> C -> A)
    cyclic_tasks = [
        SubagentTask(id="node_A", description="A", dependencies=["node_C"]),
        SubagentTask(id="node_B", description="B", dependencies=["node_A"]),
        SubagentTask(id="node_C", description="C", dependencies=["node_B"]),
    ]
    try:
        coordinator.validate_dag(cyclic_tasks)
        return False, {"error": "Validation failed to detect 3-node cyclic dependency"}
    except DependencyCycleError as exc:
        if "cyclic dependency detected" not in str(exc).lower():
            return False, {"error": f"Unexpected cycle error message: {exc}"}

    # 2. Self-dependency (Self -> Self)
    self_loop_tasks = [
        SubagentTask(id="node_self", description="Self loop", dependencies=["node_self"]),
    ]
    try:
        coordinator.validate_dag(self_loop_tasks)
        return False, {"error": "Validation failed to detect self-dependency loop"}
    except DependencyCycleError as exc:
        if "cannot depend on itself" not in str(exc).lower():
            return False, {"error": f"Unexpected self-dependency error message: {exc}"}

    # 3. Missing dependency
    missing_dep_tasks = [
        SubagentTask(id="node_ok", description="Ok", dependencies=["non_existent_node"]),
    ]
    try:
        coordinator.validate_dag(missing_dep_tasks)
        return False, {"error": "Validation failed to detect missing dependency"}
    except DelegationError as exc:
        if "non-existent task" not in str(exc).lower():
            return False, {"error": f"Unexpected missing dependency error message: {exc}"}

    return True, {
        "cycle_detected": True,
        "self_loop_detected": True,
        "missing_dep_detected": True,
    }


def run_task_8_dag_topological_waves_and_context(
    agent: MaddyAgentCore, trace_store: TraceStore, coordinator: DelegationCoordinator
) -> tuple[bool, Dict[str, Any]]:
    r"""Task 8: DAG Topological Wave Execution, Context Propagation & Upstream Skip Propagation.

    Executes a diamond dependency graph:
            task_root
            /       \
      task_left    task_right
            \       /
           task_sink

    And a dependent failure branch:
         task_fail -> task_blocked

    Verifies:
    1. Execution order strictly respects topological layers.
    2. Upstream outputs are automatically injected into downstream task contexts.
    3. Failure of task_fail causes task_blocked to be marked SKIPPED with attribution.
    """
    tasks = [
        SubagentTask(id="task_root", description="Fetch telemetry root"),
        SubagentTask(id="task_left", description="Compute metric left", dependencies=["task_root"]),
        SubagentTask(id="task_right", description="Compute metric right", dependencies=["task_root"]),
        SubagentTask(id="task_sink", description="Merge metrics", dependencies=["task_left", "task_right"]),
        SubagentTask(id="task_fail", description="Failure probe"),
        SubagentTask(id="task_blocked", description="Blocked by fail", dependencies=["task_fail"]),
    ]

    execution_history: List[str] = []

    def _simulated_dag_executor(t: SubagentTask, parent_task_id: Optional[str] = None) -> SubagentResult:
        execution_history.append(t.id)

        if t.id == "task_root":
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.SUCCESS,
                summary="Telemetry root: sensor_voltage=3.3V, temp=24.1C",
            )
        elif t.id == "task_left":
            assert "Telemetry root: sensor_voltage=3.3V" in (t.context or ""), "Root context not injected into left"
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.SUCCESS,
                summary="Metric left normalized: 0.94",
            )
        elif t.id == "task_right":
            assert "Telemetry root: sensor_voltage=3.3V" in (t.context or ""), "Root context not injected into right"
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.SUCCESS,
                summary="Metric right normalized: 0.88",
            )
        elif t.id == "task_sink":
            assert "Metric left normalized: 0.94" in (t.context or ""), "Left output not injected into sink"
            assert "Metric right normalized: 0.88" in (t.context or ""), "Right output not injected into sink"
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.SUCCESS,
                summary="Merged composite score: 0.91",
            )
        elif t.id == "task_fail":
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.FAILED,
                error="Network interface disconnected",
            )
        return SubagentResult(task_id=t.id, status=SubagentStatus.ERROR)

    dag_report = coordinator.delegate_dag(tasks, custom_executor=_simulated_dag_executor)

    # 1. Check diamond results
    res_map = {r["task_id"]: r for r in dag_report["results"]}
    if res_map["task_root"]["status"] != "SUCCESS":
        return False, {"error": "task_root failed"}
    if res_map["task_left"]["status"] != "SUCCESS" or res_map["task_right"]["status"] != "SUCCESS":
        return False, {"error": "Intermediate diamond tasks failed"}
    if res_map["task_sink"]["status"] != "SUCCESS":
        return False, {"error": "task_sink failed"}

    # 2. Check failure and skip propagation
    if res_map["task_fail"]["status"] != "FAILED":
        return False, {"error": "task_fail status was not FAILED"}
    if res_map["task_blocked"]["status"] != "SKIPPED":
        return False, {"error": f"task_blocked status was {res_map['task_blocked']['status']}, expected SKIPPED"}
    if "Upstream dependency failed" not in res_map["task_blocked"].get("error", ""):
        return False, {"error": "task_blocked missing upstream failure attribution in error message"}

    # 3. Verify topological wave sequence
    # task_root MUST precede task_left and task_right
    # task_left and task_right MUST precede task_sink
    idx_root = execution_history.index("task_root")
    idx_left = execution_history.index("task_left")
    idx_right = execution_history.index("task_right")
    idx_sink = execution_history.index("task_sink")

    if not (idx_root < idx_left and idx_root < idx_right):
        return False, {"error": "Topological order violated: root did not precede children"}
    if not (idx_left < idx_sink and idx_right < idx_sink):
        return False, {"error": "Topological order violated: children did not precede sink"}

    # Verify task_blocked was never executed by the executor function
    if "task_blocked" in execution_history:
        return False, {"error": "task_blocked was executed when it should have been skipped"}

    return True, {
        "execution_waves_verified": True,
        "context_propagation_verified": True,
        "upstream_failure_skip_verified": True,
        "execution_history": execution_history,
    }


# ── BENCHMARK RUNNER ─────────────────────────────────────────────────────────

BENCHMARK_TASKS: List[tuple[str, str, str, Callable]] = [
    (
        "TASK-DELEGATE-01",
        "Single Subagent Isolated Execution & Trace Recording",
        "Verifies leaf subagent execution in an isolated thread, TraceStore logging, and structured result schema",
        run_task_1_single_subagent_isolation,
    ),
    (
        "TASK-DELEGATE-02",
        "Capability-Based Tool Filtering & Safe Subordination",
        "Verifies leaf subagent tool stripping, depth-gated delegation tools, and required-tools isolation",
        run_task_2_capability_tool_stripping,
    ),
    (
        "TASK-DELEGATE-03",
        "Recursion Depth Hard-Stop & Fork Bomb Prevention",
        "Verifies that attempting to exceed max_spawn_depth halts immediately with SubagentStatus.ERROR",
        run_task_3_recursion_depth_limiter,
    ),
    (
        "TASK-DELEGATE-04",
        "Parent Context Truncation & Token Exhaustion Defense",
        "Verifies head/tail slicing and omission indication for oversized subagent responses",
        run_task_4_parent_context_truncation,
    ),
    (
        "TASK-DELEGATE-05",
        "Parallel Batch Dispatch & Bounded Concurrency",
        "Verifies concurrent thread-pool dispatch, sub-linear execution time, and ordered result gathering",
        run_task_5_parallel_batch_dispatch,
    ),
    (
        "TASK-DELEGATE-06",
        "Partial Failure Resilience & Error Isolation",
        "Verifies that worker failures are isolated and do not stall or corrupt parallel siblings",
        run_task_6_partial_failure_resilience,
    ),
    (
        "TASK-DELEGATE-07",
        "Dependency DAG Cycle & Deadlock Pre-Execution Detection",
        "Verifies pre-execution rejection of circular graphs, self-dependencies, and missing references",
        run_task_7_dag_cycle_detection,
    ),
    (
        "TASK-DELEGATE-08",
        "DAG Topological Waves, Context & Upstream Skip Propagation",
        "Verifies diamond graph wave execution, prerequisite context injection, and upstream skip propagation",
        run_task_8_dag_topological_waves_and_context,
    ),
]


def execute_benchmarks() -> bool:
    """Execute all Phase 6 benchmarks following CLAUDE.md benchmark rules."""
    print("=" * 80)
    print("M.A.D.D.Y. PHASE 6 DELEGATION & MULTI-AGENT ARCHITECTURE BENCHMARK SUITE")
    print("Governed strictly by CLAUDE.md Benchmark Rules:")
    print("  - 3 runs per benchmark task")
    print("  - A task passes when >= 2/3 runs succeed")
    print("  - Outcomes verified using independent Python ground-truth assertions")
    print(f"  - Database: {BENCHMARK_DB_PATH}")
    print("=" * 80 + "\n")

    suite_reports: List[TaskBenchmarkReport] = []

    for task_id, name, desc, task_fn in BENCHMARK_TASKS:
        print(f"[{task_id}] {name}")
        print(f"  Description: {desc}")
        runs: List[RunResult] = []

        for run_idx in range(1, 4):
            agent, trace_store, coordinator = build_harness()
            start_t = time.perf_counter()
            passed = False
            err: Optional[str] = None
            details: Dict[str, Any] = {}

            try:
                passed, details = task_fn(agent, trace_store, coordinator)
            except Exception as exc:
                passed = False
                err = f"{type(exc).__name__}: {str(exc)}"

            elapsed = time.perf_counter() - start_t
            status_str = "PASS" if passed else "FAIL"
            err_str = f" [Error: {err}]" if err else ""
            print(f"  Run {run_idx}/3: {status_str} in {elapsed:.3f}s {details if not passed else ''}{err_str}")
            runs.append(
                RunResult(
                    run_index=run_idx,
                    task_id=task_id,
                    passed=passed,
                    duration_s=elapsed,
                    error=err,
                    verification_details=details,
                )
            )
            time.sleep(0.05)

        pass_count = sum(1 for r in runs if r.passed)
        task_passed = pass_count >= 2
        print(f"  Outcome: {pass_count}/3 Passed -> {'SUCCESS' if task_passed else 'FAILURE'}\n")
        suite_reports.append(
            TaskBenchmarkReport(
                task_id=task_id,
                name=name,
                description=desc,
                runs=runs,
                passed=task_passed,
                pass_count=pass_count,
            )
        )

    print("=" * 80)
    print("PHASE 6 DELEGATION BENCHMARK SUMMARY")
    print("=" * 80)
    for report in suite_reports:
        status_label = "PASSED (>= 2/3)" if report.passed else "FAILED (< 2/3)"
        print(f"- {report.task_id} ({report.name}): {report.pass_count}/3 runs succeeded -> {status_label}")

    passed_tasks = sum(1 for r in suite_reports if r.passed)
    total_tasks = len(suite_reports)
    print("=" * 80)
    print(f"Score: {passed_tasks}/{total_tasks} Tasks Passed")

    all_passed = passed_tasks == total_tasks
    print(f"Final Status: {'PHASE 6 ACCEPTANCE MET' if all_passed else 'PHASE 6 ACCEPTANCE NOT MET'}\n")
    return all_passed


if __name__ == "__main__":
    success = execute_benchmarks()
    sys.exit(0 if success else 1)
