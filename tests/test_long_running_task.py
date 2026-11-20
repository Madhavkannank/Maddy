"""
Comprehensive unit and integration test suite for Milestone 2:
Long-Running Task Execution, Checkpointing, Recovery, and Task Graph Resumption.
"""
import json
import os
import shutil
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional

from agents.supervisor import Supervisor
from memory.task_memory import CURRENT_CHECKPOINT_VERSION, TaskMemory
from models.base import ModelProvider, ModelResponse
from recovery.engine import (
    ActionState,
    ErrorType,
    RecoveryAction,
    RecoveryEngine,
    RecoveryStrategy,
)
from recovery.journal import JournalEventType, RecoveryJournal
from security.confirmation import AutoApproveLowRiskConfirmationHandler
from security.permission_engine import PermissionEngine
from tasks.graph import TaskGraph, TaskGraphError
from tasks.identity import ActionRecord, ExecutionIdentity, VerificationState
from tasks.task import NON_TERMINAL_STATES, TERMINAL_STATES, Task, TaskState
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class MockLongRunningModel(ModelProvider):
    """
    Mock model provider that returns predefined task decomposition and worker responses.
    """
    def __init__(self, decomposition_json: str, worker_responses: Optional[List[str]] = None):
        self.decomposition_json = decomposition_json
        self.worker_responses = list(worker_responses) if worker_responses else []
        self.call_history: List[str] = []

    def generate(self, prompt: str, system_prompt: Optional[str] = None, messages=None, options=None) -> ModelResponse:
        self.call_history.append(prompt)
        if "Supervisor Orchestrator" in (system_prompt or ""):
            return ModelResponse(content=self.decomposition_json)

        # Worker step responses
        if self.worker_responses:
            resp = self.worker_responses.pop(0)
            return ModelResponse(content=resp)
        return ModelResponse(content=json.dumps({"thought": "Default step", "final_answer": "Default finished"}))

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)

class TestLongRunningTask(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine(confirmation_handler=AutoApproveLowRiskConfirmationHandler())
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)
        self.recovery_engine = RecoveryEngine(max_retries=2)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Task Serialization Round-Trip
    # ------------------------------------------------------------------
    def test_task_serialization_round_trip(self):
        t1 = Task(
            id="T1",
            description="Download dataset",
            parent_id="P0",
            status=TaskState.RUNNING,
            dependencies=["T0"],
            priority=2,
            assigned_worker="worker_T1",
            required_tools=["filesystem.write"],
            context={"file": "data.csv"},
            result="Partial download",
            error="None",
            retry_count=1,
            max_retries=3,
            timeout_seconds=45.0,
            current_step_id="S02",
            current_action_id="A01",
            verification_state={"state": "verifying"},
            checkpoint_data={"bytes_read": 1024},
            metadata={"source": "api"},
        )
        t_dict = t1.to_dict()
        t_reconstructed = Task.from_dict(t_dict)

        self.assertEqual(t1.id, t_reconstructed.id)
        self.assertEqual(t1.description, t_reconstructed.description)
        self.assertEqual(t1.status, t_reconstructed.status)
        self.assertEqual(t1.dependencies, t_reconstructed.dependencies)
        self.assertEqual(t1.timeout_seconds, t_reconstructed.timeout_seconds)
        self.assertEqual(t1.current_step_id, t_reconstructed.current_step_id)
        self.assertEqual(t1.checkpoint_data, t_reconstructed.checkpoint_data)
        self.assertEqual(t1.verification_state, t_reconstructed.verification_state)

    # ------------------------------------------------------------------
    # 2 & 3. TaskGraph Serialization & DAG Dependency Preservation
    # ------------------------------------------------------------------
    def test_task_graph_serialization_round_trip(self):
        graph = TaskGraph(graph_id="graph_complex")
        t1 = Task(id="T1", description="Fetch data", priority=2)
        t2 = Task(id="T2", description="Process data", dependencies=["T1"], priority=3)
        t3 = Task(id="T3", description="Generate report", dependencies=["T2"])
        graph.add_task(t1)
        graph.add_task(t2)
        graph.add_task(t3)

        graph.update_status("T1", TaskState.SUCCESS, result="Data 100%")

        g_dict = graph.to_dict()
        graph_reconstructed = TaskGraph.from_dict(g_dict)

        self.assertEqual(graph_reconstructed.graph_id, "graph_complex")
        self.assertEqual(len(graph_reconstructed.list_tasks()), 3)
        self.assertEqual(graph_reconstructed.get_task("T1").status, TaskState.SUCCESS)
        self.assertEqual(graph_reconstructed.get_task("T2").dependencies, ["T1"])
        self.assertFalse(graph_reconstructed.detect_cycles())

        ready = graph_reconstructed.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].id, "T2")

    # ------------------------------------------------------------------
    # 4. State Transition Validation
    # ------------------------------------------------------------------
    def test_state_transition_validation(self):
        task = Task(id="T1", description="Transition validation")
        self.assertEqual(task.status, TaskState.PENDING)

        # Valid transitions
        task.transition_to(TaskState.READY)
        self.assertEqual(task.status, TaskState.READY)

        task.transition_to(TaskState.RUNNING)
        self.assertEqual(task.status, TaskState.RUNNING)
        self.assertIsNotNone(task.started_at)

        # Illegal transition directly from RUNNING to PENDING without force
        with self.assertRaises(ValueError):
            task.transition_to(TaskState.PENDING)

        # Force transition bypasses validation if needed by supervisor
        task.transition_to(TaskState.INTERRUPTED, force=True)
        self.assertEqual(task.status, TaskState.INTERRUPTED)

    # ------------------------------------------------------------------
    # 5. Graph Pause and Resume
    # ------------------------------------------------------------------
    def test_graph_pause_and_resume(self):
        graph = TaskGraph()
        t1 = Task(id="T1", description="Task 1")
        t2 = Task(id="T2", description="Task 2", dependencies=["T1"])
        graph.add_task(t1)
        graph.add_task(t2)

        # Put T1 into RUNNING
        graph.update_status("T1", TaskState.RUNNING)
        graph.pause()
        self.assertEqual(graph.get_task("T1").status, TaskState.PAUSED)

        # Resume graph
        graph.resume()
        # Dependencies of T1 are empty, so it should resume to READY
        self.assertEqual(graph.get_task("T1").status, TaskState.READY)

    # ------------------------------------------------------------------
    # 6. Graph Cancellation
    # ------------------------------------------------------------------
    def test_graph_cancellation_all(self):
        graph = TaskGraph()
        t1 = Task(id="T1", description="Task 1")
        t2 = Task(id="T2", description="Task 2", dependencies=["T1"])
        graph.add_task(t1)
        graph.add_task(t2)

        graph.cancel_all(reason="Emergency user abort")
        self.assertEqual(graph.get_task("T1").status, TaskState.CANCELLED)
        self.assertEqual(graph.get_task("T2").status, TaskState.CANCELLED)
        self.assertTrue(graph.is_complete())

    # ------------------------------------------------------------------
    # 7. Interrupted Task Handling & Execution Identity
    # ------------------------------------------------------------------
    def test_interrupted_task_recovery_analysis(self):
        exec_id = ExecutionIdentity.create(task_id="T1", step_number=2, action_index=1)
        self.assertEqual(exec_id.composite_id, "T1:S02:A01:ATTEMPT01")

        task = Task(
            id="T1",
            description="Interrupted side-effect task",
            status=TaskState.INTERRUPTED,
            current_action_id=exec_id.action_id,
            checkpoint_data={"tool_name": "filesystem.write", "arguments": {"path": "/tmp/dummy.txt"}},
        )

        strategy = self.recovery_engine.analyze_interrupted_task(task)
        self.assertEqual(strategy.action, RecoveryAction.VERIFY_FIRST)
        self.assertTrue(strategy.requires_verification)

    # ------------------------------------------------------------------
    # 8 & 9. Stuck Task Detection & Timeout Distinction
    # ------------------------------------------------------------------
    def test_stuck_task_detection(self):
        graph = TaskGraph()
        # T1 has a 10-second timeout
        t1 = Task(id="T1", description="Fast timeout task", timeout_seconds=10.0)
        # T2 has no specific timeout (falls back to default 120s)
        t2 = Task(id="T2", description="Long timeout task", timeout_seconds=200.0)
        graph.add_task(t1)
        graph.add_task(t2)

        graph.update_status("T1", TaskState.RUNNING)
        graph.update_status("T2", TaskState.RUNNING)

        # Simulate 15 seconds passed
        simulated_now = time.time() + 15.0
        stuck = graph.find_stuck_tasks(default_timeout_seconds=60.0, current_time=simulated_now)

        self.assertEqual(len(stuck), 1)
        self.assertEqual(stuck[0].id, "T1")

    # ------------------------------------------------------------------
    # 10, 11 & 12. Checkpoint Atomic Creation, Reload, Versioning
    # ------------------------------------------------------------------
    def test_atomic_checkpoint_creation_and_versioning(self):
        mem = TaskMemory(task_id="task_checkpoint_test", base_dir=self.temp_dir)
        graph = TaskGraph(graph_id="graph_v1")
        t1 = Task(id="T1", description="Work")
        graph.add_task(t1)

        saved_path = mem.save_checkpoint(graph)
        self.assertTrue(os.path.exists(saved_path))

        cp_data = mem.load_checkpoint()
        self.assertIsNotNone(cp_data)
        self.assertEqual(cp_data["checkpoint_version"], CURRENT_CHECKPOINT_VERSION)
        self.assertEqual(cp_data["task_id"], "task_checkpoint_test")

        reloaded_graph = mem.load_graph_checkpoint()
        self.assertIsNotNone(reloaded_graph)
        self.assertEqual(reloaded_graph.graph_id, "graph_v1")

    # ------------------------------------------------------------------
    # 13. Corrupt Checkpoint Recovery
    # ------------------------------------------------------------------
    def test_corrupt_checkpoint_backup_recovery(self):
        mem = TaskMemory(task_id="task_corrupt_recovery", base_dir=self.temp_dir)
        graph = TaskGraph(graph_id="graph_state_1")
        t1 = Task(id="T1", description="First state")
        graph.add_task(t1)
        mem.save_checkpoint(graph)

        # Second valid checkpoint creates backup of first
        t2 = Task(id="T2", description="Second state")
        graph.add_task(t2)
        mem.save_checkpoint(graph)

        # Corrupt the primary checkpoint file deliberately
        cp_path = os.path.join(mem.task_dir, "checkpoint.json")
        with open(cp_path, "w", encoding="utf-8") as f:
            f.write("{ INVALID JSON NOT TERMINATED ...")

        recovered = mem.load_checkpoint()
        self.assertIsNotNone(recovered)
        self.assertTrue(recovered.get("_recovered_from_backup"))

        reloaded_graph = mem.load_graph_checkpoint()
        self.assertIsNotNone(reloaded_graph)
        self.assertEqual(reloaded_graph.get_task("T1").id, "T1")

    # ------------------------------------------------------------------
    # 14. Resumable Task Discovery
    # ------------------------------------------------------------------
    def test_resumable_task_discovery(self):
        # 1. Active/unfinished task
        mem1 = TaskMemory(task_id="task_unfinished", base_dir=self.temp_dir)
        g1 = TaskGraph()
        g1.add_task(Task(id="T1", description="Ongoing", status=TaskState.RUNNING))
        mem1.save_goal("Build bridge")
        mem1.save_checkpoint(g1)

        # 2. Completed task
        mem2 = TaskMemory(task_id="task_completed", base_dir=self.temp_dir)
        g2 = TaskGraph()
        g2.add_task(Task(id="T1", description="Finished", status=TaskState.SUCCESS))
        mem2.save_goal("Clean room")
        mem2.save_checkpoint(g2)

        resumable = TaskMemory.list_resumable_tasks(base_dir=self.temp_dir)
        task_ids = [r["task_id"] for r in resumable]
        self.assertIn("task_unfinished", task_ids)
        self.assertNotIn("task_completed", task_ids)

    # ------------------------------------------------------------------
    # 15. Supervisor Incremental Checkpointing Integration
    # ------------------------------------------------------------------
    def test_supervisor_incremental_checkpointing(self):
        decomp = json.dumps({
            "thought": "Two-step task",
            "tasks": [
                {"id": "T1", "description": "Step 1", "dependencies": []},
                {"id": "T2", "description": "Step 2", "dependencies": ["T1"]},
            ],
        })
        mock_model = MockLongRunningModel(decomp, [
            json.dumps({"thought": "T1 done", "final_answer": "Result 1"}),
            json.dumps({"thought": "T2 done", "final_answer": "Result 2"}),
        ])

        mem = TaskMemory(task_id="task_sup_checkpoint", base_dir=self.temp_dir)
        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            task_memory=mem,
        )

        result = supervisor.run("Execute two step task")
        self.assertEqual(result.status, "SUCCESS")

        # Verify that checkpoint was saved and contains complete graph
        cp = mem.load_checkpoint()
        self.assertIsNotNone(cp)
        self.assertTrue(cp["graph"]["is_complete"])
        self.assertEqual(cp["graph"]["completed_tasks"], 2)

    # ------------------------------------------------------------------
    # 16. Multi-Step Crash / Restart Simulation: T1 -> T2 -> T3
    # ------------------------------------------------------------------
    def test_crash_restart_simulation_t1_t2_t3(self):
        """
        Scenario:
        T1 -> T2 -> T3
        T1 completes to SUCCESS.
        M.A.D.D.Y. crashes while T2 is RUNNING.
        M.A.D.D.Y. restarts via resume_task().
        Expected:
        - T1 is NOT re-executed!
        - T2 is safely resumed and completes.
        - T3 executes after T2 succeeds.
        - Final graph reaches SUCCESS.
        """
        task_id = "task_crash_sim"
        mem = TaskMemory(task_id=task_id, base_dir=self.temp_dir)
        mem.save_goal("Process multi-step workflow")

        # 1. Setup initial graph state before crash: T1=SUCCESS, T2=RUNNING, T3=PENDING
        g = TaskGraph(graph_id="crash_graph")
        t1 = Task(id="T1", description="Download data", status=TaskState.SUCCESS, result="Data v1")
        t2 = Task(id="T2", description="Transform data", dependencies=["T1"], status=TaskState.RUNNING)
        t3 = Task(id="T3", description="Publish report", dependencies=["T2"], status=TaskState.PENDING)
        g.add_task(t1)
        g.add_task(t2)
        g.add_task(t3)

        # Journal before crash
        journal = RecoveryJournal(task_id=task_id)
        journal.record(JournalEventType.TASK_SUCCESS, "T1 finished", task_id="T1")
        journal.record(JournalEventType.TASK_RUNNING, "T2 started", task_id="T2")

        # Save checkpoint simulating crash right after T2 started
        mem.save_checkpoint(g, journal_entries=journal.to_dict_list())

        # 2. Restart M.A.D.D.Y. with a fresh Supervisor
        mock_model = MockLongRunningModel(
            decomposition_json="{}", # Not called because graph rehydrates from checkpoint!
            worker_responses=[
                json.dumps({"thought": "T2 resumed & finished", "final_answer": "Transformed data"}),
                json.dumps({"thought": "T3 finished", "final_answer": "Published report"}),
            ],
        )

        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            task_memory=mem,
        )

        # Resume task from disk
        resumed_result = supervisor.resume_task(task_id=task_id, base_dir=self.temp_dir)

        self.assertEqual(resumed_result.status, "SUCCESS")
        self.assertEqual(resumed_result.total_tasks, 3)
        self.assertEqual(len(resumed_result.completed_tasks), 3)

        # Assert T1 result was preserved from pre-crash
        t1_completed = next(t for t in resumed_result.completed_tasks if t["id"] == "T1")
        self.assertEqual(t1_completed["result"], "Data v1")

        # Verify mock model was called ONLY for T2 and T3 (T1 was NOT re-executed)
        self.assertEqual(len(mock_model.call_history), 2)

    # ------------------------------------------------------------------
    # 17. External Side-Effect Idempotency Protection on Crash
    # ------------------------------------------------------------------
    def test_side_effect_idempotency_protection_on_crash(self):
        """
        Scenario:
        T1 performs a side-effect (writes file target.txt).
        The action completes on disk (file exists with content).
        BUT M.A.D.D.Y. crashes before recording SUCCESS in the graph!
        On restart:
        M.A.D.D.Y. detects the external file already exists,
        verifies it, marks T1 as SUCCESS, and DOES NOT call the tool again!
        """
        task_id = "task_side_effect_idempotency"
        mem = TaskMemory(task_id=task_id, base_dir=self.temp_dir)
        mem.save_goal("Write config file")

        target_file = os.path.join(self.temp_dir, "output.txt")
        # Pre-create the file to simulate side-effect that completed before crash
        with open(target_file, "w", encoding="utf-8") as f:
            f.write("Expected content")

        # Graph at time of crash: T1 was RUNNING with checkpoint_data pointing to the tool call
        g = TaskGraph(graph_id="side_effect_graph")
        t1 = Task(
            id="T1",
            description="Write output file",
            status=TaskState.RUNNING,
            current_action_id="A01",
            checkpoint_data={
                "tool_name": "filesystem.write",
                "arguments": {"path": target_file, "content": "Expected content"},
            },
        )
        g.add_task(t1)
        mem.save_checkpoint(g)

        # Supervisor on restart
        mock_model = MockLongRunningModel(decomposition_json="{}", worker_responses=[])
        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            task_memory=mem,
        )

        result = supervisor.resume_task(task_id=task_id, base_dir=self.temp_dir)

        # Verification must succeed without model or worker calls
        self.assertEqual(result.status, "SUCCESS")
        t1_task = supervisor._current_graph.get_task("T1")
        self.assertEqual(t1_task.status, TaskState.SUCCESS)
        self.assertEqual(t1_task.verification_state.get("state"), "verified_success")
        # Worker was not even called to re-execute!
        self.assertEqual(len(mock_model.call_history), 0)

    # ------------------------------------------------------------------
    # 18. Supervisor Pause & Resume
    # ------------------------------------------------------------------
    def test_supervisor_pause_and_resume(self):
        mem = TaskMemory(task_id="task_sup_pause", base_dir=self.temp_dir)
        mem.save_goal("Pause and resume task")

        g = TaskGraph(graph_id="pause_graph")
        t1 = Task(id="T1", description="Work unit", status=TaskState.RUNNING)
        g.add_task(t1)
        mem.save_checkpoint(g)

        mock_model = MockLongRunningModel(
            decomposition_json="{}",
            worker_responses=[json.dumps({"thought": "T1 finished", "final_answer": "Resumed success"})],
        )
        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            task_memory=mem,
        )
        supervisor._current_graph = g
        supervisor.journal = RecoveryJournal(task_id="task_sup_pause")

        # Pause supervisor
        supervisor.pause_task("task_sup_pause")
        self.assertEqual(g.get_task("T1").status, TaskState.PAUSED)

        # Resume task
        resumed = supervisor.resume_task("task_sup_pause", base_dir=self.temp_dir)
        self.assertEqual(resumed.status, "SUCCESS")
        self.assertEqual(len(resumed.completed_tasks), 1)

    # ------------------------------------------------------------------
    # 19. Supervisor Cancellation
    # ------------------------------------------------------------------
    def test_supervisor_cancel_task(self):
        mem = TaskMemory(task_id="task_sup_cancel", base_dir=self.temp_dir)
        mem.save_goal("Cancel task")

        g = TaskGraph(graph_id="cancel_graph")
        t1 = Task(id="T1", description="Work 1", status=TaskState.RUNNING)
        t2 = Task(id="T2", description="Work 2", dependencies=["T1"], status=TaskState.PENDING)
        g.add_task(t1)
        g.add_task(t2)
        mem.save_checkpoint(g)

        mock_model = MockLongRunningModel(decomposition_json="{}", worker_responses=[])
        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            task_memory=mem,
        )
        supervisor._current_graph = g
        supervisor.journal = RecoveryJournal(task_id="task_sup_cancel")

        supervisor.cancel_task("task_sup_cancel", reason="User stopped execution")
        self.assertEqual(g.get_task("T1").status, TaskState.CANCELLED)
        self.assertEqual(g.get_task("T2").status, TaskState.CANCELLED)
        self.assertTrue(g.is_complete())

    # ------------------------------------------------------------------
    # 20. Recovery Journal Formatting and Audit Trail
    # ------------------------------------------------------------------
    def test_recovery_journal_audit_trail(self):
        journal = RecoveryJournal(task_id="T_AUDIT")
        journal.record(JournalEventType.TASK_INITIALIZED, "Task started")
        journal.record(JournalEventType.ACTION_START, "Executing filesystem.write", step_id="S01", action_id="A01")
        journal.record(JournalEventType.VERIFICATION_SUCCESS, "File written and verified", step_id="S01", action_id="A01")

        trail = journal.get_audit_trail()
        self.assertEqual(len(trail), 3)
        self.assertIn("TASK_INITIALIZED", trail[0])
        self.assertIn("ACTION_START", trail[1])
        self.assertIn("VERIFICATION_SUCCESS", trail[2])

if __name__ == "__main__":
    unittest.main()

