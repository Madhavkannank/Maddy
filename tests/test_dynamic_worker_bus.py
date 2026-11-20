"""
Unit tests for MessageBus, AgentMessage, and DynamicWorkerAgent messaging protocol.
"""
import unittest

from agents.bus import MessageBus, MessageBusError
from agents.dynamic_worker import DynamicWorkerAgent
from agents.message import AgentMessage, MessageType
from models.base import ModelMessage, ModelProvider, ModelResponse
from security.confirmation import AutoApproveLowRiskConfirmationHandler
from security.permission_engine import PermissionEngine
from tasks.task import Task
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class MockWorkerModel(ModelProvider):

    def generate(self, prompt, system_prompt=None, messages=None, options=None):
        if "Observation:" in prompt:
            return ModelResponse(content='{"thought": "File read complete", "final_answer": "Content is Mock file content"}')
        return ModelResponse(
            content='{"thought": "Mock worker action", "tool_call": {"name": "filesystem.read", "arguments": {"path": "/tmp/test.txt"}}}'
        )

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return {}

class TestDynamicWorkerBus(unittest.TestCase):

    def setUp(self):
        self.bus = MessageBus()
        self.bus.register_agent("supervisor")
        self.bus.register_agent("worker_T1")

        self.registry = ToolRegistry()
        register_default_tools(self.registry)

        self.perm_engine = PermissionEngine(confirmation_handler=AutoApproveLowRiskConfirmationHandler())
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)
        self.mock_model = MockWorkerModel()

        # Create dummy file for mock tool call
        with open("/tmp/test.txt", "w", encoding="utf-8") as f:
            f.write("Mock file content")

    def test_message_bus_publish_and_receive(self):
        msg = AgentMessage(
            sender="supervisor",
            receiver="worker_T1",
            msg_type=MessageType.TASK_ASSIGN,
            task_id="T1",
            payload={"goal": "Do work"},
        )
        self.bus.publish(msg)

        pending = self.bus.receive("worker_T1")
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].msg_type, MessageType.TASK_ASSIGN)
        self.assertEqual(pending[0].payload["goal"], "Do work")

        # After receive, queue should be empty
        self.assertEqual(len(self.bus.receive("worker_T1")), 0)

    def test_dynamic_worker_messaging_flow(self):
        worker = DynamicWorkerAgent(
            worker_id="worker_T1",
            model_provider=self.mock_model,
            executor=self.executor,
            registry=self.registry,
            message_bus=self.bus,
            max_steps=5,
        )

        task = Task(id="T1", description="Read test file", required_tools=["filesystem.read"])
        result = worker.execute_task(task=task, task_goal="Read test file", supervisor_id="supervisor")

        # Verify messages sent to supervisor over bus
        supervisor_messages = self.bus.receive("supervisor")
        self.assertGreaterEqual(len(supervisor_messages), 2)

        types = [m.msg_type for m in supervisor_messages]
        self.assertIn(MessageType.TASK_PROGRESS, types)
        self.assertIn(MessageType.TASK_RESULT, types)

    def test_dynamic_worker_tool_isolation(self):
        worker = DynamicWorkerAgent(
            worker_id="worker_T1",
            model_provider=self.mock_model,
            executor=self.executor,
            registry=self.registry,
            message_bus=self.bus,
        )

        isolated_registry = worker._build_isolated_registry(required_tools=["filesystem.read"])
        tool_names = [t.name for t in isolated_registry.list_tools()]
        self.assertIn("filesystem.read", tool_names)
        self.assertNotIn("application.open", tool_names)

if __name__ == "__main__":
    unittest.main()
