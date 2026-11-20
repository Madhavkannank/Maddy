"""
Comprehensive Verification and Integration Tests for Security Audit Remediation:
Issue 1: Verification Engine outcomes (VERIFIED, UNVERIFIED, FAILED, VERIFICATION_ERROR) & no silent failure
Issue 2: Background autonomy default runner fails explicitly when no backend is configured
Issue 3: Autonomy permission interface integration (PermissionGate.check_tool_execution)
Issue 4: Scheduled task timeout deadline enforcement and lifecycle tracking
Issue 5: Resumed execution unification with verification engine pipeline
Issue 6: CronJobManageTool dynamic operation_tier authorization granularity
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import unittest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.autonomy.models import JobStatus, JobTriggerType, ScheduledJob
from core.autonomy.policy import AutonomyPolicyGate
from core.autonomy.repository import JobRepository
from core.autonomy.scheduler import AutonomyScheduler
from core.permissions import PermissionCheckResult, PermissionGate, PermissionTier
from core.tools.base import BaseTool, ToolRegistry, ToolResult
from core.tools.cron_tool import CronJobManageTool
from tools.call import ToolCall
from verification.engine import VerificationEngine
from verification.schema import (
    Assertion,
    AssertionType,
    VerificationOutcome,
    VerificationReport,
)


class DummyTool(BaseTool):
    name = "write_file"
    description = "Test writing tool."
    tier = PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        }

    def execute(self, file_path: str, **kwargs: Any) -> ToolResult:
        return ToolResult(success=True, output=f"Wrote to {file_path}")


class TestSecurityAuditRemediation(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Issue 1: Verification Engine Outcomes & Silent Failure Prevention
    # -------------------------------------------------------------------------
    def test_issue1_verification_engine_outcomes_and_error_handling(self):
        """Verify explicit outcomes: VERIFIED, UNVERIFIED, FAILED, VERIFICATION_ERROR."""
        # 1. Enum members exist
        self.assertEqual(VerificationOutcome.VERIFIED.value, "VERIFIED")
        self.assertEqual(VerificationOutcome.UNVERIFIED.value, "UNVERIFIED")
        self.assertEqual(VerificationOutcome.FAILED.value, "FAILED")
        self.assertEqual(VerificationOutcome.VERIFICATION_ERROR.value, "VERIFICATION_ERROR")

        # Backward compatibility aliases
        self.assertEqual(VerificationOutcome.VERIFIED_SUCCESS, VerificationOutcome.VERIFIED)
        self.assertEqual(VerificationOutcome.VERIFIED_FAILURE, VerificationOutcome.FAILED)
        self.assertEqual(VerificationOutcome.INDETERMINATE, VerificationOutcome.UNVERIFIED)

        verifier = VerificationEngine()

        # 2. When tool has no assertions defined -> outcome is UNVERIFIED
        tc = ToolCall(name="unsupported_custom_tool", arguments={"foo": "bar"})
        tr = ToolResult(success=True, output="Done")
        res_tr, report = verifier.verify_tool_execution(tc, tr)
        self.assertEqual(report.outcome, VerificationOutcome.UNVERIFIED)
        self.assertTrue(res_tr.success)

        # 3. When filesystem assertion passes -> outcome is VERIFIED
        test_file = Path(self.tmp_dir) / "test_verify.txt"
        test_file.write_text("ground truth content")
        tc_write = ToolCall(name="filesystem.write", arguments={"path": str(test_file), "content": "ground truth content"})
        res_tr_pass, report_pass = verifier.verify_tool_execution(tc_write, ToolResult(success=True, output="Saved"))
        self.assertEqual(report_pass.outcome, VerificationOutcome.VERIFIED)
        self.assertTrue(report_pass.can_proceed)

        # 4. When assertion fails -> outcome is FAILED and overrides tool success
        missing_file = Path(self.tmp_dir) / "non_existent_file.txt"
        tc_missing = ToolCall(name="filesystem.write", arguments={"path": str(missing_file)})
        res_tr_fail, report_fail = verifier.verify_tool_execution(tc_missing, ToolResult(success=True, output="Saved"))
        self.assertEqual(report_fail.outcome, VerificationOutcome.FAILED)
        self.assertFalse(res_tr_fail.success)
        self.assertEqual(res_tr_fail.error.get("type"), "VERIFICATION_FAILURE")

        # 5. When verification engine itself encounters an exception -> VERIFICATION_ERROR
        # Simulate an assertion evaluation crash
        def faulty_generate(call):
            raise RuntimeError("Corrupted verification driver")

        verifier.generate_tool_assertions = faulty_generate
        res_tr_err, report_err = verifier.verify_tool_execution(tc, ToolResult(success=True, output="Done"))
        self.assertEqual(report_err.outcome, VerificationOutcome.VERIFICATION_ERROR)
        self.assertFalse(res_tr_err.success)
        self.assertEqual(res_tr_err.error.get("type"), "VERIFICATION_ERROR")

    # -------------------------------------------------------------------------
    # Issue 2: Background Autonomy Default Runner Fails Explicitly
    # -------------------------------------------------------------------------
    def test_issue2_scheduler_default_runner_fails_explicitly(self):
        """Verify scheduler produces an explicit failure when no backend runner is configured."""
        db_path = Path(self.tmp_dir) / "autonomy.db"
        repo = JobRepository(db_path=db_path)
        # Instantiate scheduler without custom runner_fn
        scheduler = AutonomyScheduler(repository=repo, tick_interval=0.02, runner_fn=None)
        scheduler.start()

        try:
            job = scheduler.add_job(
                name="Unconfigured Task",
                goal="Execute unattended backup",
                interval_seconds=3600.0,
            )
            # Trigger immediate execution
            scheduler.trigger_now(job.id)
            time.sleep(0.15)

            reloaded = scheduler.get_job(job.id)
            # Status must be FAILED, not SUCCESS, and last_error must detail missing runner
            self.assertEqual(reloaded.last_status, "FAILED")
            self.assertIn("No autonomous execution backend configured", reloaded.last_error)
            self.assertIn("backend runner unavailable", reloaded.last_error)

            # Audit executions table
            execs = repo.get_executions_for_job(job.id)
            self.assertTrue(len(execs) >= 1)
            self.assertEqual(execs[0].status, "FAILED")
            self.assertIn("backend runner unavailable", execs[0].error)
        finally:
            scheduler.stop(timeout=1.0)

    # -------------------------------------------------------------------------
    # Issue 3: Autonomy Permission Interface Integration
    # -------------------------------------------------------------------------
    def test_issue3_autonomy_permission_gate_integration(self):
        """Verify PermissionGate.check_tool_execution works and integrates with AutonomyPolicyGate."""
        sandbox_dir = Path(self.tmp_dir) / "allowed_sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        gate = PermissionGate(
            auto_approve_tier2=False,
            allowed_write_directories=[str(sandbox_dir)],
        )

        # 1. Direct check_tool_execution interface check
        res_safe = gate.check_tool_execution("uname", {"command": "uname"}, tier=PermissionTier.SAFE)
        self.assertIsInstance(res_safe, PermissionCheckResult)
        self.assertTrue(res_safe.is_allowed)

        # Sandboxed write is allowed
        sandboxed_file = str(sandbox_dir / "out.txt")
        res_sandboxed = gate.check_tool_execution("write_file", {"file_path": sandboxed_file})
        self.assertTrue(res_sandboxed.is_allowed)

        # Unsandboxed write is rejected cleanly with reason
        unsandboxed_file = "/etc/shadow"
        res_blocked = gate.check_tool_execution("write_file", {"file_path": unsandboxed_file})
        self.assertFalse(res_blocked.is_allowed)
        self.assertIn("denied", res_blocked.reason.lower())

        # 2. Integration with AutonomyPolicyGate
        policy = AutonomyPolicyGate(permission_gate=gate, allow_standard_tier=True)

        tool = DummyTool()
        # Evaluating dummy_writer inside sandbox
        dec_allowed = policy.evaluate(tool, {"file_path": sandboxed_file})
        self.assertTrue(dec_allowed.allowed)
        self.assertEqual(dec_allowed.policy_tier, "SYSTEM_IMPACTING")

        # Evaluating dummy_writer outside sandbox
        dec_denied = policy.evaluate(tool, {"file_path": unsandboxed_file})
        self.assertFalse(dec_allowed_denied := dec_denied.allowed)
        self.assertTrue(dec_denied.requires_approval)
        self.assertIn("failed", dec_denied.reason.lower())

    # -------------------------------------------------------------------------
    # Issue 4: Scheduled Task Timeout Deadline Enforcement
    # -------------------------------------------------------------------------
    def test_issue4_scheduled_task_timeout_deadline_enforcement(self):
        """Verify scheduler enforces job.timeout_seconds and records TIMED_OUT."""
        db_path = Path(self.tmp_dir) / "autonomy.db"
        repo = JobRepository(db_path=db_path)

        def _stalling_runner(j: ScheduledJob) -> str:
            time.sleep(0.3)
            return "Should not finish"

        scheduler = AutonomyScheduler(
            repository=repo,
            tick_interval=0.02,
            runner_fn=_stalling_runner,
        )
        scheduler.start()

        try:
            # Set job with a strict 0.05 second timeout
            job = scheduler.add_job(
                name="Timeout Task",
                goal="Long calculation",
                interval_seconds=3600.0,
                timeout_seconds=0.05,
            )
            scheduler.trigger_now(job.id)
            time.sleep(0.18)

            reloaded = scheduler.get_job(job.id)
            self.assertEqual(reloaded.last_status, "TIMED_OUT")
            self.assertIn("exceeded execution deadline", reloaded.last_error)

            # Audit record
            execs = repo.get_executions_for_job(job.id)
            self.assertTrue(len(execs) >= 1)
            self.assertEqual(execs[0].status, "TIMED_OUT")
            self.assertIn("timed out", execs[0].error.lower())
        finally:
            scheduler.stop(timeout=1.0)

    # -------------------------------------------------------------------------
    # Issue 5: Resumed Execution Verification Engine Unification
    # -------------------------------------------------------------------------
    def test_issue5_resumed_execution_invokes_verification_engine(self):
        """Verify resumed execution runs tools through the verification engine pipeline."""
        from core.agent_loop import MaddyAgentCore
        from core.trace_store import TraceStore
        from tools.call import ToolCall as VToolCall

        trace_db = Path(self.tmp_dir) / "traces.db"
        trace_store = TraceStore(db_path=trace_db)
        registry = ToolRegistry()

        class WritingTool(BaseTool):
            name = "filesystem.write"
            description = "Write file."
            tier = PermissionTier.SYSTEM_IMPACTING
            def get_parameters_schema(self):
                return {"type": "object", "properties": {"path": {"type": "string"}}}
            def execute(self, path: str, **kwargs):
                # Deliberately fake success without creating file
                return ToolResult(success=True, output="Saved successfully")

        registry.register_tool(WritingTool())
        gate = PermissionGate(auto_approve_tier2=True)
        verifier = VerificationEngine()

        agent = MaddyAgentCore(
            tool_registry=registry,
            permission_gate=gate,
            trace_store=trace_store,
            verification_engine=verifier,
            enable_delegation=False,
        )

        target_file = str(Path(self.tmp_dir) / "non_existent_target.txt")

        # Mock tool call in resumed execution
        class MockToolCallInfo:
            id = "call_mock_1"
            name = "filesystem.write"
            arguments = {"path": target_file}

        from core.loop_guard import LoopGuard
        loop_guard = LoopGuard()
        messages = []

        # Execute single tool directly through unified pipeline
        failed = agent._execute_single_tool(
            task_id="resumed_task_1",
            tc=MockToolCallInfo(),
            step_id="step_1",
            step_index=1,
            loop_guard=loop_guard,
            messages=messages,
        )

        # File does not exist on disk, so verification engine MUST detect false completion
        self.assertTrue(failed, "Unified pipeline must catch false completion via verification engine")
        tool_records = trace_store.get_task_tool_calls("resumed_task_1")
        self.assertEqual(len(tool_records), 1)
        self.assertEqual(tool_records[0]["status"], "failed")
        self.assertIn("Verification Failure", tool_records[0]["result_output"])

    # -------------------------------------------------------------------------
    # Issue 6: CronJobManageTool Dynamic Operation Tier Granularity
    # -------------------------------------------------------------------------
    def test_issue6_cronjob_manage_dynamic_operation_tier(self):
        """Verify CronJobManageTool classifies read-only as SAFE and mutations as SYSTEM_IMPACTING."""
        tool = CronJobManageTool()

        # Class default tier must be SYSTEM_IMPACTING (fail closed)
        self.assertEqual(tool.tier, PermissionTier.SYSTEM_IMPACTING)

        # Read-only actions must be SAFE
        self.assertEqual(tool.operation_tier({"action": "list"}), PermissionTier.SAFE)
        self.assertEqual(tool.operation_tier({"action": "history"}), PermissionTier.SAFE)
        self.assertEqual(tool.operation_tier({"action": "status"}), PermissionTier.SAFE)

        # Mutating or scheduling actions must be SYSTEM_IMPACTING
        self.assertEqual(tool.operation_tier({"action": "add", "goal": "Run check"}), PermissionTier.SYSTEM_IMPACTING)
        self.assertEqual(tool.operation_tier({"action": "run", "job_id": "job_1"}), PermissionTier.SYSTEM_IMPACTING)
        self.assertEqual(tool.operation_tier({"action": "remove", "job_id": "job_1"}), PermissionTier.SYSTEM_IMPACTING)
        self.assertEqual(tool.operation_tier({"action": "pause", "job_id": "job_1"}), PermissionTier.SYSTEM_IMPACTING)
        self.assertEqual(tool.operation_tier({"action": "resume", "job_id": "job_1"}), PermissionTier.SYSTEM_IMPACTING)

        # Verify PermissionGate enforces this granularity
        gate = PermissionGate(auto_approve_tier2=False)
        # 'list' should pass without error
        gate.check_permission(tool.name, tool.operation_tier({"action": "list"}), {"action": "list"})

        # 'add' should raise PermissionDeniedError when auto_approve_tier2 is False
        from core.errors import PermissionDeniedError
        with self.assertRaises(PermissionDeniedError):
            gate.check_permission(
                tool.name,
                tool.operation_tier({"action": "add", "goal": "Persistent task"}),
                {"action": "add", "goal": "Persistent task"},
            )


if __name__ == "__main__":
    unittest.main()
