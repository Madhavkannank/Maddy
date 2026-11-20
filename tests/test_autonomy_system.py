"""Unit and Failure-Mode Test Suite for M.A.D.D.Y. Background Autonomy Architecture.

Tests:
1. ScheduledJob trigger calculations (interval, daily_time, one_shot, cron_expression)
2. SQLite JobRepository CRUD operations
3. SQLite JobRepository execution audit trail logging
4. LifecycleGuard crash recovery & orphan zombie job reconciliation
5. CircuitBreaker failure counting, exponential backoff, and suspension tripping
6. AutonomyPolicyGate tier enforcement (SAFE allowed, STANDARD sandboxed, DANGEROUS rejected)
7. AutonomyScheduler tick dispatch and non-overlapping execution locks
8. Dynamic job control (pause, resume, remove, trigger_now)
9. One-shot auto-completion lifecycle
10. CronJobManageTool typed model tool interface execution
11. MaddyAgentCore autonomy tool auto-registration
"""

import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from core.autonomy.circuit_breaker import CircuitBreaker
from core.autonomy.lifecycle import LifecycleGuard
from core.autonomy.models import (
    JobExecutionRecord,
    JobStatus,
    JobTriggerType,
    ScheduledJob,
)
from core.autonomy.policy import AutonomyPolicyGate
from core.autonomy.repository import JobRepository
from core.autonomy.scheduler import AutonomyScheduler
from core.permissions import PermissionGate, PermissionTier
from core.tools.base import BaseTool, ToolRegistry, ToolResult
from core.tools.cron_tool import CronJobManageTool


class MockSafeTool(BaseTool):
    name = "mock_status"
    description = "Read-only mock status"
    tier = PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="Status OK")


class MockSystemImpactingTool(BaseTool):
    name = "mock_write"
    description = "Mutating write tool"
    tier = PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {"path": {"type": "string"}}}

    def execute(self, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output="Written")


# ── TEST CASES ───────────────────────────────────────────────────────────────

def test_job_trigger_calculation():
    now = 1700000000.0  # Reference epoch

    # 1. Interval
    job_interval = ScheduledJob(
        id="j_int",
        name="Interval Job",
        goal="Check GPU temperature",
        trigger_type=JobTriggerType.INTERVAL,
        interval_seconds=300.0,
    )
    next_run = job_interval.calculate_next_run(from_time=now)
    assert next_run == now + 300.0

    # 2. Daily time
    job_daily = ScheduledJob(
        id="j_daily",
        name="Daily Job",
        goal="Send morning briefing",
        trigger_type=JobTriggerType.DAILY_TIME,
        time_of_day="08:30",
    )
    next_daily = job_daily.calculate_next_run(from_time=now)
    assert next_daily > now

    # 3. One-shot
    target = now + 7200.0
    job_oneshot = ScheduledJob(
        id="j_once",
        name="One-shot Job",
        goal="Delayed reminder",
        trigger_type=JobTriggerType.ONE_SHOT,
        target_epoch=target,
    )
    assert job_oneshot.calculate_next_run(from_time=now) == target

    # 4. Cron expression */15 * * * *
    job_cron = ScheduledJob(
        id="j_cron",
        name="Cron Job",
        goal="Check logs",
        trigger_type=JobTriggerType.CRON_EXPRESSION,
        cron_expression="*/15 * * * *",
    )
    assert job_cron.calculate_next_run(from_time=now) == now + 900.0


def test_repository_crud(tmp_path):
    repo = JobRepository(db_path=tmp_path / "autonomy.db")
    job = ScheduledJob(
        id="job_crud_1",
        name="Clean Caches",
        goal="Clear temp directories",
        trigger_type=JobTriggerType.INTERVAL,
        interval_seconds=600.0,
        metadata={"priority": "low"},
    )
    repo.save_job(job)

    # Fetch
    fetched = repo.get_job("job_crud_1")
    assert fetched is not None
    assert fetched.id == "job_crud_1"
    assert fetched.name == "Clean Caches"
    assert fetched.interval_seconds == 600.0
    assert fetched.metadata.get("priority") == "low"

    # List
    all_jobs = repo.list_jobs()
    assert len(all_jobs) == 1

    # Delete
    assert repo.delete_job("job_crud_1") is True
    assert repo.get_job("job_crud_1") is None
    assert len(repo.list_jobs()) == 0


def test_repository_execution_history(tmp_path):
    repo = JobRepository(db_path=tmp_path / "autonomy.db")
    job = ScheduledJob(id="job_hist_1", name="Telemetry", goal="Record stats")
    repo.save_job(job)

    exec_rec = repo.record_execution_start(
        execution_id="exec_001",
        job_id="job_hist_1",
        started_at=100.0,
        metadata={"worker": "test"},
    )
    assert exec_rec.status == "RUNNING"

    repo.record_execution_finish(
        execution_id="exec_001",
        completed_at=102.5,
        status="COMPLETED",
        result_summary="All stats captured cleanly.",
        total_steps=2,
    )

    history = repo.get_executions_for_job("job_hist_1")
    assert len(history) == 1
    assert history[0].execution_id == "exec_001"
    assert history[0].duration_seconds == 2.5
    assert history[0].status == "COMPLETED"
    assert "All stats captured cleanly" in (history[0].result_summary or "")


def test_lifecycle_guard_orphan_reconciliation(tmp_path):
    repo = JobRepository(db_path=tmp_path / "autonomy.db")

    # Simulate a job left in RUNNING status due to a process crash
    stranded_job = ScheduledJob(
        id="stranded_1",
        name="Stranded Job",
        goal="Heavy aggregation",
        status=JobStatus.RUNNING,
    )
    repo.save_job(stranded_job)
    repo.record_execution_start(
        execution_id="exec_stranded",
        job_id="stranded_1",
        started_at=time.time() - 50.0,
    )

    guard = LifecycleGuard(repo)
    recovered = guard.reconcile_orphans()

    assert len(recovered) == 1
    assert recovered[0].id == "stranded_1"
    assert recovered[0].status == JobStatus.RECOVERED
    assert recovered[0].last_status == "CRASHED_RECOVERED"

    # Verify execution record finalized
    exec_history = repo.get_executions_for_job("stranded_1")
    assert exec_history[0].status == "CRASHED_RECOVERED"


def test_circuit_breaker_quarantine():
    breaker = CircuitBreaker(base_backoff_seconds=10.0)
    job = ScheduledJob(
        id="job_cb",
        name="Failing Service",
        goal="Hit broken endpoint",
        failure_threshold=3,
    )

    # Failure 1
    tripped, delay = breaker.record_failure(job, "Connection timeout")
    assert tripped is False
    assert job.consecutive_failures == 1
    assert delay == 10.0

    # Failure 2
    tripped, delay = breaker.record_failure(job, "Connection timeout")
    assert tripped is False
    assert job.consecutive_failures == 2
    assert delay == 20.0

    # Failure 3 -> Trips breaker!
    tripped, delay = breaker.record_failure(job, "Connection timeout")
    assert tripped is True
    assert job.consecutive_failures == 3
    assert job.status == JobStatus.SUSPENDED
    assert "circuit_breaker_tripped_at" in job.metadata

    # Success restores breaker
    breaker.record_success(job)
    assert job.consecutive_failures == 0
    assert job.status == JobStatus.IDLE


def test_autonomy_policy_gate():
    safe_tool = MockSafeTool()
    impacting_tool = MockSystemImpactingTool()

    # 1. SAFE tool must be allowed
    policy_default = AutonomyPolicyGate(allow_standard_tier=True)
    decision_safe = policy_default.evaluate(safe_tool, {})
    assert decision_safe.allowed is True
    assert decision_safe.policy_tier == "SAFE"

    # 2. Blacklisted tool must be strictly blocked
    blocked_tool = MockSafeTool()
    blocked_tool.name = "execute_raw_sudo"
    decision_blocked = policy_default.evaluate(blocked_tool, {})
    assert decision_blocked.allowed is False
    assert decision_blocked.policy_tier == "BLOCKED"

    # 3. Disallowed system-impacting tier
    policy_restricted = AutonomyPolicyGate(allow_standard_tier=False)
    decision_restricted = policy_restricted.evaluate(impacting_tool, {"path": "/tmp/test"})
    assert decision_restricted.allowed is False
    assert decision_restricted.requires_approval is True
    assert decision_restricted.policy_tier == "SYSTEM_IMPACTING"


def test_scheduler_non_overlapping_execution_locks(tmp_path):
    repo = JobRepository(db_path=tmp_path / "autonomy.db")
    executed_count = 0
    execution_lock = threading.Lock()

    def _slow_runner(j: ScheduledJob) -> str:
        nonlocal executed_count
        with execution_lock:
            executed_count += 1
        time.sleep(0.15)
        return "Slow runner completed"

    scheduler = AutonomyScheduler(
        repository=repo,
        tick_interval=0.02,
        runner_fn=_slow_runner,
    )
    scheduler.start()

    try:
        job = scheduler.add_job(
            name="Frequent Task",
            goal="Rapid checks",
            interval_seconds=0.02,  # Very short interval to test overlap lock
        )
        # Force due immediately
        scheduler.trigger_now(job.id)
        time.sleep(0.2)

        # Due to 0.15s sleep, second tick during active execution MUST be skipped
        with execution_lock:
            # Over 0.2s with a 0.15s runtime, only 1 (or max 2) executions can have finished
            assert executed_count <= 2
    finally:
        scheduler.stop(timeout=1.0)


def test_scheduler_dynamic_lifecycle_control(tmp_path):
    repo = JobRepository(db_path=tmp_path / "autonomy.db")
    executed_jobs: List[str] = []

    def _tracking_runner(j: ScheduledJob) -> str:
        executed_jobs.append(j.id)
        return f"Ran {j.id}"

    scheduler = AutonomyScheduler(
        repository=repo,
        tick_interval=0.02,
        runner_fn=_tracking_runner,
    )
    scheduler.start()

    try:
        job = scheduler.add_job(
            name="Controllable Task",
            goal="Do work",
            interval_seconds=3600.0,
        )

        # 1. Immediate trigger
        scheduler.trigger_now(job.id)
        time.sleep(0.08)
        assert job.id in executed_jobs

        # 2. Pause
        scheduler.pause_job(job.id)
        reloaded = scheduler.get_job(job.id)
        assert reloaded.enabled is False
        assert reloaded.status == JobStatus.SUSPENDED

        # 3. Resume
        scheduler.resume_job(job.id)
        reloaded_resumed = scheduler.get_job(job.id)
        assert reloaded_resumed.enabled is True
        assert reloaded_resumed.status == JobStatus.SCHEDULED

        # 4. Remove
        scheduler.remove_job(job.id)
        assert scheduler.get_job(job.id) is None
    finally:
        scheduler.stop(timeout=1.0)


def test_scheduler_oneshot_completion(tmp_path):
    repo = JobRepository(db_path=tmp_path / "autonomy.db")
    completed = []

    def _oneshot_runner(j: ScheduledJob) -> str:
        completed.append(j.id)
        return "One-shot done"

    scheduler = AutonomyScheduler(
        repository=repo,
        tick_interval=0.02,
        runner_fn=_oneshot_runner,
    )
    scheduler.start()

    try:
        job = scheduler.add_job(
            name="One-shot Task",
            goal="Single alert",
            trigger_type=JobTriggerType.ONE_SHOT,
            target_epoch=time.time() - 1.0,  # Ready immediately
        )
        time.sleep(0.15)

        reloaded = scheduler.get_job(job.id)
        assert reloaded.status == JobStatus.COMPLETED
        assert reloaded.enabled is False
        assert reloaded.next_run_at is None
        assert job.id in completed
    finally:
        scheduler.stop(timeout=1.0)


def test_cron_tool_model_interface(tmp_path):
    repo = JobRepository(db_path=tmp_path / "autonomy.db")
    scheduler = AutonomyScheduler(repository=repo, tick_interval=0.05)
    tool = CronJobManageTool(scheduler=scheduler)

    # 1. Action: ADD
    res_add = tool.execute(
        action="add",
        name="Hourly Health Check",
        goal="Inspect system memory and swap usage",
        interval_seconds=3600.0,
    )
    assert res_add.success is True
    assert "Successfully scheduled task" in res_add.output
    job_id = res_add.metadata["id"]

    # 2. Action: LIST
    res_list = tool.execute(action="list")
    assert res_list.success is True
    assert "Hourly Health Check" in res_list.output
    assert res_list.metadata["total_jobs"] == 1

    # 3. Action: PAUSE
    res_pause = tool.execute(action="pause", job_id=job_id)
    assert res_pause.success is True
    assert "has been paused" in res_pause.output

    # 4. Action: RESUME
    res_resume = tool.execute(action="resume", job_id=job_id)
    assert res_resume.success is True
    assert "has been resumed" in res_resume.output

    # 5. Action: RUN (immediate trigger)
    res_run = tool.execute(action="run", job_id=job_id)
    assert res_run.success is True
    assert "triggered for immediate execution" in res_run.output

    # 6. Action: REMOVE
    res_remove = tool.execute(action="remove", job_id=job_id)
    assert res_remove.success is True
    assert "permanently deleted" in res_remove.output


def test_agent_core_autonomy_tool_registration():
    from core.agent_loop import MaddyAgentCore

    reg = ToolRegistry()
    agent = MaddyAgentCore(
        tool_registry=reg,
        enable_delegation=False,
    )
    # cronjob_manage must be auto-registered in active toolset
    assert agent.tool_registry.get_tool("cronjob_manage") is not None
