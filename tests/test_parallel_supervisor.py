"""
Unit tests verifying single-model GPU inference queuing and parallel task execution with context isolation.
"""
import json
import os
import tempfile
import threading
import time
import unittest

from agents import Supervisor
from models.base import ModelMessage, ModelProvider, ModelResponse
from models.queue import ModelInferenceQueue
from security import PermissionEngine, PolicyMode
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class MultiTaskParallelModel(ModelProvider):

    def __init__(self, file_path1: str, file_path2: str):
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.active_threads = 0
        self.max_concurrent_llm_calls = 0
        self.lock = threading.Lock()

    def generate(self, prompt, system_prompt=None, messages=None, options=None):
        with self.lock:
            self.active_threads += 1
            if self.active_threads > self.max_concurrent_llm_calls:
                self.max_concurrent_llm_calls = self.active_threads

        # Simulate small inference delay
        time.sleep(0.01)

        sys_p = system_prompt or ""

        try:
            # 1. Supervisor decomposition into 2 independent tasks (T1 and T2)
            if "Supervisor Orchestrator" in sys_p:
                decomp = {
                    "thought": "Decompose into 2 independent parallel write tasks",
                    "tasks": [
                        {
                            "id": "T1",
                            "description": f"Write to file1 at {self.file_path1}",
                            "dependencies": [],
                            "required_tools": ["filesystem.write"],
                        },
                        {
                            "id": "T2",
                            "description": f"Write to file2 at {self.file_path2}",
                            "dependencies": [],
                            "required_tools": ["filesystem.write"],
                        },
                    ],
                }
                return ModelResponse(content=json.dumps(decomp))

            # Task 1 actions
            if f"Write to file1 at {self.file_path1}" in prompt:
                if "bytes_written" in prompt:
                    return ModelResponse(content=json.dumps({"thought": "File 1 written", "final_answer": "File 1 complete."}))
                return ModelResponse(content=json.dumps({
                    "thought": "Writing file 1",
                    "tool_call": {"name": "filesystem.write", "arguments": {"path": self.file_path1, "content": "Parallel Data 1"}}
                }))

            # Task 2 actions
            if f"Write to file2 at {self.file_path2}" in prompt:
                if "bytes_written" in prompt:
                    return ModelResponse(content=json.dumps({"thought": "File 2 written", "final_answer": "File 2 complete."}))
                return ModelResponse(content=json.dumps({
                    "thought": "Writing file 2",
                    "tool_call": {"name": "filesystem.write", "arguments": {"path": self.file_path2, "content": "Parallel Data 2"}}
                }))

            return ModelResponse(content=json.dumps({"thought": "Done", "final_answer": "Completed"}))
        finally:
            with self.lock:
                self.active_threads -= 1

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return {}

class TestParallelSupervisor(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file1 = os.path.join(self.temp_dir.name, "task1.txt")
        self.file2 = os.path.join(self.temp_dir.name, "task2.txt")

        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine(policy_mode=PolicyMode.BALANCED)
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)
        self.mock_model = MultiTaskParallelModel(self.file1, self.file2)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parallel_independent_tasks_with_gpu_inference_queue(self):
        supervisor = Supervisor(
            model_provider=self.mock_model,
            executor=self.executor,
            registry=self.registry,
        )

        result = supervisor.run("Run parallel file writing tasks")

        # 1. Verify both tasks completed successfully
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.total_tasks, 2)
        self.assertEqual(len(result.completed_tasks), 2)

        # 2. Verify files created independently on disk
        self.assertTrue(os.path.exists(self.file1))
        self.assertTrue(os.path.exists(self.file2))

        with open(self.file1, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Parallel Data 1")

        with open(self.file2, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), "Parallel Data 2")

        # 3. Verify ModelInferenceQueue serialized LLM calls (max concurrent LLM calls = 1)
        self.assertEqual(self.mock_model.max_concurrent_llm_calls, 1)

if __name__ == "__main__":
    unittest.main()
