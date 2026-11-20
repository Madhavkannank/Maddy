"""
End-to-end integration test: Supervisor -> TaskGraph -> SingleAgent -> ToolExecutor -> PermissionEngine.
"""
import os
import json
import tempfile
import unittest

from agents import Supervisor
from models.base import ModelProvider, ModelResponse
from security import PermissionEngine, PolicyMode
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class FullStackMockModel(ModelProvider):

    def __init__(self, target_path):
        self.target_path = target_path

    def generate(self, prompt, system_prompt=None, messages=None, options=None):
        sys_p = system_prompt or ""

        # 1. Supervisor decomposition
        if "Supervisor Orchestrator" in sys_p:
            decomp = {
                "thought": "Decompose into write and verify tasks",
                "tasks": [
                    {
                        "id": "T1",
                        "description": f"Write test file to {self.target_path}",
                        "dependencies": [],
                        "required_tools": ["filesystem.write"],
                    },
                    {
                        "id": "T2",
                        "description": f"Read and verify test file from {self.target_path}",
                        "dependencies": ["T1"],
                        "required_tools": ["filesystem.read"],
                    },
                ],
            }
            return ModelResponse(content=json.dumps(decomp))

        # Check observation results first
        if "bytes_written" in prompt:
            return ModelResponse(content=json.dumps({"thought": "File write complete", "final_answer": "File written successfully."}))

        if "Full stack integration verified" in prompt:
            return ModelResponse(content=json.dumps({"thought": "Verification complete", "final_answer": "File verified."}))

        # Task initial steps
        if f"Write test file to {self.target_path}" in prompt:
            step = {
                "thought": "Writing test file content",
                "tool_call": {
                    "name": "filesystem.write",
                    "arguments": {"path": self.target_path, "content": "Full stack integration verified"},
                },
            }
            return ModelResponse(content=json.dumps(step))

        if f"Read and verify test file from {self.target_path}" in prompt:
            step = {
                "thought": "Reading test file content",
                "tool_call": {
                    "name": "filesystem.read",
                    "arguments": {"path": self.target_path},
                },
            }
            return ModelResponse(content=json.dumps(step))

        return ModelResponse(content=json.dumps({"thought": "Default answer", "final_answer": "Done"}))

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)

class TestSupervisorIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.target_path = os.path.join(self.temp_dir.name, "integration.txt")

        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine(policy_mode=PolicyMode.BALANCED)
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)
        self.mock_model = FullStackMockModel(self.target_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_supervisor_orchestration_flow(self):
        supervisor = Supervisor(
            model_provider=self.mock_model,
            executor=self.executor,
            registry=self.registry,
        )

        result = supervisor.run(f"Write and verify integration file at {self.target_path}")

        # Verify SupervisorResult
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.total_tasks, 2)
        self.assertEqual(len(result.completed_tasks), 2)
        self.assertEqual(len(result.failed_tasks), 0)
        self.assertEqual(len(result.blocked_tasks), 0)

        # Verify actual file state on disk
        self.assertTrue(os.path.exists(self.target_path))
        with open(self.target_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "Full stack integration verified")

        # Verify security audit logs recorded
        records = self.perm_engine.audit_logger.get_records()
        self.assertGreaterEqual(len(records), 2)
        tool_names = [r.tool_name for r in records]
        self.assertIn("filesystem.write", tool_names)
        self.assertIn("filesystem.read", tool_names)

if __name__ == "__main__":
    unittest.main()
