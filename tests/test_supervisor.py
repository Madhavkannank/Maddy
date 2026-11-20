"""
Unit tests for Supervisor orchestration engine.
"""
import os
import json
import tempfile
import unittest

from agents import Supervisor
from models.base import ModelProvider, ModelResponse
from security.permission_engine import PermissionEngine
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class MockSupervisorModel(ModelProvider):

    def __init__(self, decomposition_json, agent_answers):
        self.decomposition_json = decomposition_json
        self.agent_answers = agent_answers
        self.call_count = 0

    def generate(self, prompt, system_prompt=None, messages=None, options=None):
        if "Supervisor Orchestrator" in (system_prompt or ""):
            return ModelResponse(content=self.decomposition_json)
        
        # SingleAgent prompt calls
        answer = self.agent_answers.pop(0) if self.agent_answers else "Done"
        return ModelResponse(content=json.dumps({"thought": "Executing task", "final_answer": answer}))

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)

class TestSupervisor(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine()
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)

    def test_single_task_decomposition_and_execution(self):
        decomp = json.dumps({
            "thought": "Single step request",
            "tasks": [{"id": "T1", "description": "Answer user question", "dependencies": []}],
        })
        mock_model = MockSupervisorModel(decomp, ["Paris is the capital of France."])

        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
        )

        result = supervisor.run("What is the capital of France?")
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.total_tasks, 1)
        self.assertIn("Paris is the capital of France.", result.final_response)

    def test_compound_task_decomposition_and_execution(self):
        decomp = json.dumps({
            "thought": "Compound request",
            "tasks": [
                {"id": "T1", "description": "Gather info", "dependencies": []},
                {"id": "T2", "description": "Format report", "dependencies": ["T1"]},
            ],
        })
        mock_model = MockSupervisorModel(decomp, ["Info gathered.", "Report formatted."])

        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
        )

        result = supervisor.run("Gather info and format report")
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.total_tasks, 2)
        self.assertEqual(len(result.completed_tasks), 2)

    def test_fallback_on_malformed_decomposition_plan(self):
        malformed_decomp = "Not valid JSON plan!"
        mock_model = MockSupervisorModel(malformed_decomp, ["Fallback answer"])

        supervisor = Supervisor(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
        )

        result = supervisor.run("Execute fallback request")
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.total_tasks, 1)
        self.assertEqual(result.completed_tasks[0]["id"], "T1")

if __name__ == "__main__":
    unittest.main()
