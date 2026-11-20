"""Unit and Failure-Mode Test Suite for M.A.D.D.Y. Delegation and Subagent Orchestration.

Tests:
1. SubagentTask and SubagentResult serialization
2. Capability-based tool filtering (leaf vs coordinator, depth bounding)
3. Recursion depth limiter (preventing fork bombs)
4. Subagent output truncation for parent context protection
5. Single delegation execution and trace recording
6. Parallel batch delegation and concurrency
7. Parallel batch handling of partial failures
8. DAG cycle detection (A -> B -> A)
9. DAG self-dependency rejection
10. DAG missing dependency rejection
11. DAG sequential execution and context propagation
12. DAG diamond dependency graph resolution
13. DAG upstream failure propagation (skipping dependents)
14. Native DelegateTool model interface execution
15. MaddyAgentCore delegation auto-registration and depth prevention
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

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
from core.permissions import PermissionGate, PermissionTier
from core.tools.base import BaseTool, ToolRegistry, ToolResult
from core.tools.calculator import CalculatorTool
from core.tools.delegate_tool import DelegateTool
from core.tools.system_info import SystemInfoTool
from core.trace_store import TraceStore


class MockTool(BaseTool):
    name = "mock_echo"
    description = "Echo tool"
    tier = PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"msg": {"type": "string"}}}

    def execute(self, msg: str = "", **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=f"echo: {msg}")


# ── TEST CASES ───────────────────────────────────────────────────────────────

def test_subagent_task_creation_and_serialization():
    data = {
        "id": "t1",
        "task": "Perform telemetry analysis",
        "context": "System running Fedora Linux",
        "dependencies": ["t0"],
        "role": "leaf",
        "timeout": 45.0,
        "priority": 2,
    }
    task = SubagentTask.from_dict(data)
    assert task.id == "t1"
    assert task.description == "Perform telemetry analysis"
    assert task.context == "System running Fedora Linux"
    assert task.dependencies == ["t0"]
    assert task.role == SubagentRole.LEAF
    assert task.timeout_seconds == 45.0
    assert task.priority == 2

    dump = task.to_dict()
    assert dump["id"] == "t1"
    assert dump["role"] == "leaf"


def test_subagent_result_dataclass():
    res_success = SubagentResult(
        task_id="t1",
        status=SubagentStatus.SUCCESS,
        summary="Done",
        duration_seconds=1.2345,
    )
    assert res_success.is_success is True
    assert res_success.to_dict()["status"] == "SUCCESS"
    assert res_success.to_dict()["duration_seconds"] == round(1.2345, 3)

    res_fail = SubagentResult(
        task_id="t2",
        status=SubagentStatus.FAILED,
        error="Command failed",
    )
    assert res_fail.is_success is False


def test_tool_filtering_leaf_vs_coordinator():
    parent_reg = ToolRegistry()
    parent_reg.register(SystemInfoTool())
    parent_reg.register(CalculatorTool())
    parent_reg.register(MockTool())
    # Add delegation tool to parent
    parent_reg.register(DelegateTool(tool_registry=parent_reg))

    assert parent_reg.get_tool("delegate_task") is not None

    harness = SubagentHarness(
        parent_tool_registry=parent_reg,
        config=DelegationConfig(max_spawn_depth=3),
        current_depth=1,
    )

    # 1. Leaf child: delegate_task MUST be stripped
    leaf_reg = harness.build_child_registry(role=SubagentRole.LEAF)
    assert leaf_reg.get_tool("delegate_task") is None
    assert leaf_reg.get_tool("system_info") is not None

    # 2. Coordinator child at depth 1: delegate_task allowed
    coord_reg = harness.build_child_registry(role=SubagentRole.COORDINATOR)
    assert coord_reg.get_tool("delegate_task") is not None

    # 3. Child at depth 3: delegate_task MUST be stripped regardless of role
    harness_deep = SubagentHarness(
        parent_tool_registry=parent_reg,
        config=DelegationConfig(max_spawn_depth=3),
        current_depth=3,
    )
    deep_reg = harness_deep.build_child_registry(role=SubagentRole.COORDINATOR)
    assert deep_reg.get_tool("delegate_task") is None

    # 4. Required tools filtering
    filtered_reg = harness.build_child_registry(
        role=SubagentRole.LEAF,
        required_tools=["calculator"],
    )
    assert filtered_reg.get_tool("calculator") is not None
    assert filtered_reg.get_tool("system_info") is None


def test_recursion_depth_limiter():
    harness = SubagentHarness(
        config=DelegationConfig(max_spawn_depth=2),
        current_depth=3,  # Exceeds max_spawn_depth
    )
    task = SubagentTask(id="deep_task", description="Should fail")
    res = harness.execute(task)

    assert res.status == SubagentStatus.ERROR
    assert "recursion limit" in (res.error or "").lower()


def test_subagent_output_truncation():
    short_text = "Analysis complete: CPU usage 12%."
    assert truncate_subagent_output(short_text, max_chars=100) == short_text

    long_text = "HEAD_" + ("x" * 5000) + "_TAIL"
    truncated = truncate_subagent_output(
        long_text,
        max_chars=200,
        head_chars=50,
        tail_chars=50,
    )
    assert "HEAD_" in truncated
    assert "_TAIL" in truncated
    assert "characters omitted to preserve parent context budget" in truncated
    assert len(truncated) < len(long_text)


def test_single_delegation_execution(tmp_path):
    trace_store = TraceStore(db_path=tmp_path / "traces.db")
    reg = ToolRegistry()
    reg.register(SystemInfoTool())

    coordinator = DelegationCoordinator(
        tool_registry=reg,
        trace_store=trace_store,
    )

    task = SubagentTask(
        id="unit_diag",
        description="Run system info diagnostics",
        timeout_seconds=5.0,
    )

    # Use custom executor simulation
    with patch.object(SubagentHarness, "execute") as mock_exec:
        mock_exec.return_value = SubagentResult(
            task_id="unit_diag",
            status=SubagentStatus.SUCCESS,
            summary="System OS is Linux, 8 cores available.",
            duration_seconds=0.1,
            total_steps=1,
        )
        res = coordinator.delegate_single(task)
        assert res.is_success is True
        assert "System OS is Linux" in res.summary


def test_parallel_batch_delegation(tmp_path):
    trace_store = TraceStore(db_path=tmp_path / "traces.db")
    coordinator = DelegationCoordinator(
        trace_store=trace_store,
        config=DelegationConfig(max_concurrent_workers=4),
    )

    tasks = [
        SubagentTask(id="sub_1", description="Task 1"),
        SubagentTask(id="sub_2", description="Task 2"),
        SubagentTask(id="sub_3", description="Task 3"),
    ]

    def _mock_worker(task: SubagentTask, parent_task_id: Optional[str] = None, **kwargs: Any) -> SubagentResult:
        time.sleep(0.01)
        return SubagentResult(
            task_id=task.id,
            status=SubagentStatus.SUCCESS,
            summary=f"Result of {task.id}",
            duration_seconds=0.01,
        )

    with patch.object(SubagentHarness, "execute", side_effect=_mock_worker):
        batch_report = coordinator.delegate_batch(tasks)
        assert batch_report["success"] is True
        assert batch_report["batch_size"] == 3
        assert len(batch_report["results"]) == 3
        assert [r["task_id"] for r in batch_report["results"]] == ["sub_1", "sub_2", "sub_3"]


def test_parallel_batch_partial_failure():
    coordinator = DelegationCoordinator()
    tasks = [
        SubagentTask(id="sub_ok_1", description="Task 1"),
        SubagentTask(id="sub_fail_2", description="Task 2"),
        SubagentTask(id="sub_ok_3", description="Task 3"),
    ]

    def _mock_worker(task: SubagentTask, parent_task_id: Optional[str] = None, **kwargs: Any) -> SubagentResult:
        if task.id == "sub_fail_2":
            return SubagentResult(
                task_id=task.id,
                status=SubagentStatus.FAILED,
                error="Compilation error in worker 2",
            )
        return SubagentResult(
            task_id=task.id,
            status=SubagentStatus.SUCCESS,
            summary="Success",
        )

    with patch.object(SubagentHarness, "execute", side_effect=_mock_worker):
        batch_report = coordinator.delegate_batch(tasks)
        assert batch_report["success"] is False
        results = {r["task_id"]: r for r in batch_report["results"]}
        assert results["sub_ok_1"]["status"] == "SUCCESS"
        assert results["sub_fail_2"]["status"] == "FAILED"
        assert results["sub_ok_3"]["status"] == "SUCCESS"


def test_dag_cycle_detection():
    coordinator = DelegationCoordinator()
    # Cycle: t1 -> t2 -> t3 -> t1
    tasks = [
        SubagentTask(id="t1", description="T1", dependencies=["t3"]),
        SubagentTask(id="t2", description="T2", dependencies=["t1"]),
        SubagentTask(id="t3", description="T3", dependencies=["t2"]),
    ]

    with pytest.raises(DependencyCycleError) as exc_info:
        coordinator.validate_dag(tasks)
    assert "Cyclic dependency detected" in str(exc_info.value)


def test_dag_self_dependency_rejection():
    coordinator = DelegationCoordinator()
    tasks = [
        SubagentTask(id="loop_task", description="Loop", dependencies=["loop_task"]),
    ]
    with pytest.raises(DependencyCycleError) as exc_info:
        coordinator.validate_dag(tasks)
    assert "cannot depend on itself" in str(exc_info.value)


def test_dag_missing_dependency_rejection():
    coordinator = DelegationCoordinator()
    tasks = [
        SubagentTask(id="t1", description="T1", dependencies=["ghost_task"]),
    ]
    with pytest.raises(DelegationError) as exc_info:
        coordinator.validate_dag(tasks)
    assert "non-existent task 'ghost_task'" in str(exc_info.value)


def test_dag_sequential_execution_and_context_propagation():
    coordinator = DelegationCoordinator()

    tasks = [
        SubagentTask(id="fetch_data", description="Fetch sensor data"),
        SubagentTask(id="analyze_data", description="Analyze data", dependencies=["fetch_data"]),
    ]

    def _custom_executor(t: SubagentTask, p_id: Optional[str]) -> SubagentResult:
        if t.id == "fetch_data":
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.SUCCESS,
                summary="Sensor reading: temperature=22.5C, pressure=101.3kPa",
            )
        elif t.id == "analyze_data":
            # Verify context propagation from fetch_data
            assert "Sensor reading: temperature=22.5C" in (t.context or "")
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.SUCCESS,
                summary="Analysis: System nominal.",
            )
        return SubagentResult(task_id=t.id, status=SubagentStatus.ERROR)

    dag_report = coordinator.delegate_dag(tasks, custom_executor=_custom_executor)
    assert dag_report["success"] is True
    assert dag_report["total_tasks"] == 2


def test_dag_diamond_dependency_resolution():
    r"""Diamond graph:
         A
       /   \
      B     C
       \   /
         D
    """
    coordinator = DelegationCoordinator()
    tasks = [
        SubagentTask(id="task_A", description="Root task"),
        SubagentTask(id="task_B", description="Branch B", dependencies=["task_A"]),
        SubagentTask(id="task_C", description="Branch C", dependencies=["task_A"]),
        SubagentTask(id="task_D", description="Merge D", dependencies=["task_B", "task_C"]),
    ]

    execution_order = []

    def _custom_executor(t: SubagentTask, p_id: Optional[str]) -> SubagentResult:
        execution_order.append(t.id)
        if t.id == "task_D":
            assert "Upstream Result from Task 'task_B'" in (t.context or "")
            assert "Upstream Result from Task 'task_C'" in (t.context or "")
        return SubagentResult(
            task_id=t.id,
            status=SubagentStatus.SUCCESS,
            summary=f"Finished {t.id}",
        )

    dag_report = coordinator.delegate_dag(tasks, custom_executor=_custom_executor)
    assert dag_report["success"] is True
    assert execution_order[0] == "task_A"
    assert set(execution_order[1:3]) == {"task_B", "task_C"}
    assert execution_order[3] == "task_D"


def test_dag_upstream_failure_propagation():
    """If task A fails, dependent task B must be SKIPPED."""
    coordinator = DelegationCoordinator()
    tasks = [
        SubagentTask(id="task_fail", description="Will fail"),
        SubagentTask(id="task_dependent", description="Depends on failure", dependencies=["task_fail"]),
        SubagentTask(id="task_independent", description="Independent task"),
    ]

    def _custom_executor(t: SubagentTask, p_id: Optional[str]) -> SubagentResult:
        if t.id == "task_fail":
            return SubagentResult(
                task_id=t.id,
                status=SubagentStatus.FAILED,
                error="Fatal sensor disconnect",
            )
        return SubagentResult(
            task_id=t.id,
            status=SubagentStatus.SUCCESS,
            summary="Independent success",
        )

    dag_report = coordinator.delegate_dag(tasks, custom_executor=_custom_executor)
    assert dag_report["success"] is False
    results = {r["task_id"]: r for r in dag_report["results"]}

    assert results["task_fail"]["status"] == "FAILED"
    assert results["task_dependent"]["status"] == "SKIPPED"
    assert "Upstream dependency failed" in results["task_dependent"]["error"]
    assert results["task_independent"]["status"] == "SUCCESS"


def test_delegate_tool_model_interface():
    coordinator = DelegationCoordinator()
    tool = DelegateTool(coordinator=coordinator)

    # 1. Single task delegation via tool interface
    with patch.object(coordinator, "delegate_single") as mock_single:
        mock_single.return_value = SubagentResult(
            task_id="single",
            status=SubagentStatus.SUCCESS,
            summary="Calculated 42.",
            duration_seconds=0.5,
        )
        res = tool.execute(task="Calculate 6 * 7")
        assert res.success is True
        assert "Calculated 42." in res.output

    # 2. Parallel batch delegation via tool interface
    with patch.object(coordinator, "delegate_batch") as mock_batch:
        mock_batch.return_value = {
            "success": True,
            "batch_size": 2,
            "results": [
                {"task_id": "t1", "status": "SUCCESS", "summary": "Done 1"},
                {"task_id": "t2", "status": "SUCCESS", "summary": "Done 2"},
            ],
            "duration_seconds": 0.2,
        }
        res_batch = tool.execute(tasks=[{"id": "t1", "task": "A"}, {"id": "t2", "task": "B"}])
        assert res_batch.success is True
        assert "Parallel Batch Execution SUCCEEDED" in res_batch.output

    # 3. DAG delegation via tool interface
    with patch.object(coordinator, "delegate_dag") as mock_dag:
        mock_dag.return_value = {
            "success": True,
            "total_tasks": 2,
            "results": [
                {"task_id": "t1", "status": "SUCCESS", "summary": "Fetched"},
                {"task_id": "t2", "status": "SUCCESS", "summary": "Analyzed"},
            ],
            "duration_seconds": 0.3,
        }
        res_dag = tool.execute(
            tasks=[
                {"id": "t1", "task": "Fetch"},
                {"id": "t2", "task": "Analyze", "dependencies": ["t1"]},
            ]
        )
        assert res_dag.success is True
        assert "Dependency DAG Execution SUCCEEDED" in res_dag.output


def test_agent_core_delegation_tool_registration():
    from core.agent_loop import MaddyAgentCore

    reg = ToolRegistry()
    # Agent at depth 1 (parent): delegate_task is registered
    agent_parent = MaddyAgentCore(
        tool_registry=reg,
        current_depth=1,
        max_spawn_depth=3,
        enable_delegation=True,
    )
    assert agent_parent.tool_registry.get_tool("delegate_task") is not None

    # Agent at depth 3 (leaf limit): delegate_task is NOT registered
    reg_leaf = ToolRegistry()
    agent_leaf = MaddyAgentCore(
        tool_registry=reg_leaf,
        current_depth=3,
        max_spawn_depth=3,
        enable_delegation=True,
    )
    assert agent_leaf.tool_registry.get_tool("delegate_task") is None
