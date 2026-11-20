"""
Unit tests for SingleAgent core execution loop.
"""
import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock

from agents import AgentStatus, SingleAgent
from models.base import ModelMessage, ModelProvider, ModelResponse
from security.permission_engine import PermissionEngine
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class MockModelProvider(ModelProvider):

    def __init__(self, response_sequence):
        self.response_sequence = response_sequence
        self.call_count = 0

    def generate(self, prompt, system_prompt=None, messages=None, options=None):
        if self.call_count < len(self.response_sequence):
            resp_content = self.response_sequence[self.call_count]
        else:
            resp_content = self.response_sequence[-1]
        self.call_count += 1
        return ModelResponse(content=resp_content)

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)

class TestSingleAgent(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine()
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_single_step_final_answer(self):
        mock_model = MockModelProvider([
            json.dumps({
                "thought": "I know the answer directly.",
                "final_answer": "The capital of France is Paris.",
            })
        ])

        agent = SingleAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            max_steps=5,
        )

        result = agent.run("What is the capital of France?")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.final_answer, "The capital of France is Paris.")
        self.assertEqual(result.total_steps, 1)

    def test_multi_step_tool_execution(self):
        target_path = os.path.join(self.temp_dir.name, "notes.txt")
        
        step1_response = json.dumps({
            "thought": "I need to write notes to a file.",
            "tool_call": {
                "name": "filesystem.write",
                "arguments": {"path": target_path, "content": "M.A.D.D.Y. step 1 notes"},
            },
        })
        step2_response = json.dumps({
            "thought": "Now I will read the file back to verify.",
            "tool_call": {
                "name": "filesystem.read",
                "arguments": {"path": target_path},
            },
        })
        step3_response = json.dumps({
            "thought": "The file has been written and verified.",
            "final_answer": "Successfully created and read notes.txt",
        })

        mock_model = MockModelProvider([step1_response, step2_response, step3_response])
        agent = SingleAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            max_steps=5,
        )

        result = agent.run(f"Write notes to {target_path} and verify")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.total_steps, 3)
        self.assertEqual(result.final_answer, "Successfully created and read notes.txt")

    def test_max_iterations_reached(self):
        infinite_tool_call = json.dumps({
            "thought": "I will loop forever calling tool.",
            "tool_call": {
                "name": "filesystem.read",
                "arguments": {"path": "nonexistent.txt"},
            },
        })
        mock_model = MockModelProvider([infinite_tool_call])
        agent = SingleAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            max_steps=3,
        )

        result = agent.run("Infinite goal")
        self.assertEqual(result.status, AgentStatus.MAX_ITERATIONS_REACHED)
        self.assertEqual(result.total_steps, 3)
        self.assertIsNone(result.final_answer)

    def test_tool_error_recovery(self):
        step1_invalid_call = json.dumps({
            "thought": "I will call an unknown tool.",
            "tool_call": {
                "name": "unknown.tool",
                "arguments": {},
            },
        })
        step2_recovery = json.dumps({
            "thought": "Unknown tool failed, so I will return final answer.",
            "final_answer": "Recovered from unknown tool error.",
        })

        mock_model = MockModelProvider([step1_invalid_call, step2_recovery])
        agent = SingleAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            max_steps=5,
        )

        result = agent.run("Try unknown tool")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.total_steps, 2)
        self.assertEqual(result.final_answer, "Recovered from unknown tool error.")

if __name__ == "__main__":
    unittest.main()
