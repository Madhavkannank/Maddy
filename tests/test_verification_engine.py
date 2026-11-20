"""
Comprehensive Unit and Integration Test Suite for Milestone 3:
Verification Engine & Environment Grounding.
Validates multi-domain assertions, false completion overrides, and truthful reporting.
"""
import json
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional

from agents.single_agent import SingleAgent
from agents.supervisor import Supervisor
from memory.task_memory import TaskMemory
from models.base import ModelProvider, ModelResponse
from platform_layer.interface import PlatformInterface
from security.confirmation import AutoApproveLowRiskConfirmationHandler
from security.permission_engine import PermissionEngine
from tasks.task import Task, TaskState
from tools import register_default_tools
from tools.call import ToolCall
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.result import ToolResult
from verification.engine import VerificationEngine
from verification.schema import (
    Assertion,
    AssertionResult,
    AssertionType,
    VerificationOutcome,
    VerificationReport,
)

class MockPlatform(PlatformInterface):
    """Mock platform interface for deterministic process and window assertions."""

    def __init__(self, running_apps: Optional[List[str]] = None, windows: Optional[List[str]] = None):
        self.running_apps = list(running_apps or ["chrome", "spotify"])
        self.windows = list(windows or ["Google Chrome - New Tab", "Spotify Free"])

    def get_os_name(self) -> str:
        return "windows"

    def open_application(self, app_name: str) -> None:
        self.running_apps.append(app_name.lower())

    def close_application(self, app_name: str) -> None:
        self.running_apps = [a for a in self.running_apps if a != app_name.lower()]

    def is_application_running(self, app_name: str) -> bool:
        return app_name.lower() in [a.lower() for a in self.running_apps]

    def get_open_windows(self) -> list:
        return list(self.windows)

    def speak(self, text: str) -> None:
        pass

class MockVerificationModel(ModelProvider):
    """Mock model provider returning specific tool calls or answers."""

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.call_history: List[str] = []

    def generate(self, prompt: str, system_prompt: Optional[str] = None, messages=None, options=None) -> ModelResponse:
        self.call_history.append(prompt)
        if self.responses:
            return ModelResponse(content=self.responses.pop(0))
        return ModelResponse(content=json.dumps({"thought": "Default", "final_answer": "Finished"}))

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)

class TestVerificationEngine(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_platform = MockPlatform()
        self.verifier = VerificationEngine(platform_driver=self.mock_platform)
        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine(confirmation_handler=AutoApproveLowRiskConfirmationHandler())
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. Filesystem Assertions
    # ------------------------------------------------------------------
    def test_filesystem_assertions(self):
        test_file = os.path.join(self.temp_dir, "sample.txt")

        # 1. File does not exist yet
        res_not_exists = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.FILE_NOT_EXISTS,
            target=test_file,
        ))
        self.assertTrue(res_not_exists.passed)

        # 2. Write file
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Hello M.A.D.D.Y. verification!")

        # 3. File exists assertion
        res_exists = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.FILE_EXISTS,
            target=test_file,
        ))
        self.assertTrue(res_exists.passed)
        self.assertIn("exists", res_exists.evidence)

        # 4. File minimum size assertion
        res_size = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.FILE_SIZE_MIN,
            target=test_file,
            expected_value=10,
        ))
        self.assertTrue(res_size.passed)

        # 5. File content assertion
        res_content = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.FILE_CONTAINS,
            target=test_file,
            expected_value="M.A.D.D.Y.",
        ))
        self.assertTrue(res_content.passed)

        # 6. Directory exists assertion
        res_dir = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.DIR_EXISTS,
            target=self.temp_dir,
        ))
        self.assertTrue(res_dir.passed)

    # ------------------------------------------------------------------
    # 2. Process & Window Assertions
    # ------------------------------------------------------------------
    def test_process_and_window_assertions(self):
        # 1. Running app check
        res_running = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.PROCESS_RUNNING,
            target="chrome",
        ))
        self.assertTrue(res_running.passed)

        # 2. Non-running app check
        res_not_running = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.PROCESS_NOT_RUNNING,
            target="notepad",
        ))
        self.assertTrue(res_not_running.passed)

        # 3. Window title check
        res_window = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.WINDOW_TITLE_EXISTS,
            target="Google Chrome",
        ))
        self.assertTrue(res_window.passed)

    # ------------------------------------------------------------------
    # 3. Visual Assertions
    # ------------------------------------------------------------------
    def test_visual_assertions(self):
        img1 = os.path.join(self.temp_dir, "img1.png")
        img2 = os.path.join(self.temp_dir, "img2.png")

        with open(img1, "wb") as f:
            f.write(b"header_bytes_1234567890_original")
        with open(img2, "wb") as f:
            f.write(b"header_bytes_1234567890_modified_screen_state_differs_substantially")

        res_diff = self.verifier.evaluate_assertion(Assertion(
            assertion_type=AssertionType.SCREEN_CHANGED,
            target=img1,
            expected_value=img2,
        ))
        self.assertTrue(res_diff.passed)

    # ------------------------------------------------------------------
    # 4. False Completion Override (Rules 16 & 32)
    # ------------------------------------------------------------------
    def test_false_completion_override(self):
        """
        Critical Test:
        A tool claims success: {"success": True, "path": "phantom.txt"}
        However, phantom.txt does NOT exist in the environment!
        VerificationEngine must intercept, detect false completion,
        and override tool_result.success = False!
        """
        phantom_path = os.path.join(self.temp_dir, "phantom.txt")
        tool_call = ToolCall(
            call_id="call_phantom",
            name="filesystem.write",
            arguments={"path": phantom_path, "content": "Ghost data"},
        )
        # Simulate tool claiming success even though nothing was written
        fake_success_result = ToolResult.ok(
            data={"bytes_written": 10},
            tool_name="filesystem.write",
        )

        verified_result, report = self.verifier.verify_tool_execution(tool_call, fake_success_result)

        # Overridden to False!
        self.assertFalse(verified_result.success)
        self.assertEqual(report.outcome, VerificationOutcome.VERIFIED_FAILURE)
        self.assertIn("does NOT exist", report.evidence_summary)
        self.assertEqual(verified_result.error.get("type"), "VERIFICATION_FAILURE")

    # ------------------------------------------------------------------
    # 5. Tool Execution with Valid Environment Grounding
    # ------------------------------------------------------------------
    def test_valid_tool_execution_verification(self):
        real_path = os.path.join(self.temp_dir, "real.txt")
        tool_call = ToolCall(
            call_id="call_real",
            name="filesystem.write",
            arguments={"path": real_path, "content": "Real data"},
        )
        # Actually execute tool through executor
        tool_result = self.executor.execute(tool_call)
        self.assertTrue(tool_result.success)

        # Verify through verification engine
        verified_result, report = self.verifier.verify_tool_execution(tool_call, tool_result)
        self.assertTrue(verified_result.success)
        self.assertEqual(report.outcome, VerificationOutcome.VERIFIED_SUCCESS)
        self.assertTrue(report.can_proceed)

    # ------------------------------------------------------------------
    # 6. Automatic Tool Assertion Generation
    # ------------------------------------------------------------------
    def test_automatic_tool_assertion_generation(self):
        write_call = ToolCall(name="filesystem.write", arguments={"path": "/tmp/a.txt", "content": "hello"})
        write_assertions = self.verifier.generate_tool_assertions(write_call)
        types = [a.assertion_type for a in write_assertions]
        self.assertIn(AssertionType.FILE_EXISTS, types)
        self.assertIn(AssertionType.FILE_SIZE_MIN, types)

        del_call = ToolCall(name="filesystem.delete", arguments={"path": "/tmp/b.txt"})
        del_assertions = self.verifier.generate_tool_assertions(del_call)
        self.assertEqual(del_assertions[0].assertion_type, AssertionType.FILE_NOT_EXISTS)

        app_call = ToolCall(name="application.open", arguments={"app_name": "Calculator"})
        app_assertions = self.verifier.generate_tool_assertions(app_call)
        self.assertEqual(app_assertions[0].assertion_type, AssertionType.PROCESS_RUNNING)

    # ------------------------------------------------------------------
    # 7. SingleAgent Integration with Verification Engine
    # ------------------------------------------------------------------
    def test_single_agent_verification_integration(self):
        """
        When SingleAgent runs, verification engine catches environment realities.
        """
        valid_file = os.path.join(self.temp_dir, "agent_test.txt")
        model = MockVerificationModel([
            json.dumps({
                "thought": "Write to file",
                "tool_call": {
                    "name": "filesystem.write",
                    "arguments": {"path": valid_file, "content": "Agent verified data"},
                },
            }),
            json.dumps({"thought": "File verified on disk", "final_answer": "Done!"}),
        ])

        agent = SingleAgent(
            model_provider=model,
            executor=self.executor,
            registry=self.registry,
            verification_engine=self.verifier,
            max_steps=5,
        )

        result = agent.run(f"Write {valid_file}")
        self.assertEqual(result.status.value, "success")
        self.assertTrue(os.path.exists(valid_file))

    # ------------------------------------------------------------------
    # 8. Supervisor Honest Verification Reporting (Rule 19 & Rule 32)
    # ------------------------------------------------------------------
    def test_supervisor_honest_indeterminate_reporting(self):
        """
        When a task result is unverified or cannot be verified,
        Supervisor must append honest disclaimer:
        'I attempted [X], but I could not verify whether it succeeded.'
        """
        decomp = json.dumps({
            "thought": "Unverifiable remote task",
            "tasks": [{"id": "T1", "description": "Send external telemetry packet", "dependencies": []}],
        })
        mock_model = MockVerificationModel([
            decomp,
            json.dumps({
                "thought": "Sent packet",
                "final_answer": "Packet was transmitted over unverified channel (could not verify remote ack).",
            }),
        ])

        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            verification_engine=self.verifier,
        )

        result = supervisor.run("Send telemetry packet")
        self.assertIn("could not verify whether it succeeded", result.final_response)

if __name__ == "__main__":
    unittest.main()
