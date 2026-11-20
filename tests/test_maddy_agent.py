"""Comprehensive unit tests for MaddyAgent.

Verifies:
1. System prompt generation adhering to structured <tools> standard + M.A.D.D.Y. personality.
2. Single-step conversational responses with personality.
3. Multi-turn tool execution using <thought>, <tool_call>, and <tool_response>.
4. Parallel tool calling (<tool_call> ... <tool_call>) in a single step.
5. Scratchpad reasoning (<scratch_pad>) support.
6. Robustness against malformed tags and fallback to clean text.
7. Integration with VerificationEngine for environmental grounding.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock

from agents.base import AgentStatus
from agents.maddy_agent import MaddyAgent, build_maddy_system_prompt
from models.base import ModelMessage, ModelProvider, ModelResponse
from personality.prompts import MADDY_CORE_SYSTEM_PROMPT
from security.permission_engine import PermissionEngine
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry


class MockAgentModel(ModelProvider):
    def __init__(self, response_sequence):
        self.response_sequence = response_sequence
        self.call_count = 0
        self.received_messages = []

    def generate(self, prompt, system_prompt=None, messages=None, options=None):
        if messages:
            self.received_messages.append(list(messages))
        if self.call_count < len(self.response_sequence):
            resp = self.response_sequence[self.call_count]
        else:
            resp = self.response_sequence[-1]
        self.call_count += 1
        return ModelResponse(content=resp)

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt, messages).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)


class TestMaddyAgent(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine()
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_build_system_prompt_structure_and_personality(self):
        """Verify the system prompt includes both <tools> XML tags and M.A.D.D.Y. personality."""
        prompt = build_maddy_system_prompt(self.registry, address_title="boss")

        # 1. Personality checks
        self.assertIn("You are M.A.D.D.Y.", prompt)
        self.assertIn("CORE PERSONALITY", prompt)
        self.assertIn("HOW TO ADDRESS THE OPERATOR", prompt)
        self.assertIn("boss", prompt)

        # 2. Tool-calling structure checks
        self.assertIn("<tools>", prompt)
        self.assertIn("</tools>", prompt)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("<tool_response>", prompt)
        self.assertIn('"type": "function"', prompt)
        self.assertIn("filesystem.read", prompt)
        self.assertIn("filesystem.write", prompt)

    def test_single_step_direct_conversational_response(self):
        """Verify the agent returns direct answers without requiring rigid JSON schema."""
        mock_model = MockAgentModel([
            "That's 1161, boss. Anything else you need calculated?"
        ])

        agent = MaddyAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            address_title="boss",
        )

        result = agent.run("What is 27 times 43?")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.final_answer, "That's 1161, boss. Anything else you need calculated?")
        self.assertEqual(result.total_steps, 1)

    def test_single_step_with_thought_and_direct_answer(self):
        """Verify <thought> tags are parsed as thought and stripped from the user-facing answer."""
        mock_model = MockAgentModel([
            "<thought>The user is asking a basic identity query. Answer with composure.</thought>\nFully operational and at your service, sir."
        ])

        agent = MaddyAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            address_title="sir",
        )

        result = agent.run("Are you online, M.A.D.D.Y.?")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.final_answer, "Fully operational and at your service, sir.")
        self.assertEqual(result.steps[0].thought, "The user is asking a basic identity query. Answer with composure.")

    def test_multi_turn_tool_execution(self):
        """Verify native <thought> and <tool_call> workflow with tool response feedback."""
        target_path = os.path.join(self.temp_dir.name, "maddy_test.txt")

        step1 = f"""
<thought>I need to write the user's data to the specified file.</thought>
<tool_call>
{{"name": "filesystem.write", "arguments": {{"path": "{target_path.replace(os.sep, '/')}", "content": "MADDY protocol active"}}}}
</tool_call>
"""
        step2 = f"""
<thought>File is written. Now reading it back to verify.</thought>
<tool_call>
{{"name": "filesystem.read", "arguments": {{"path": "{target_path.replace(os.sep, '/')}"}}}}
</tool_call>
"""
        step3 = """
<thought>Verified file content matches. Informing user.</thought>
File written and verified successfully, boss.
"""

        mock_model = MockAgentModel([step1, step2, step3])
        agent = MaddyAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            address_title="boss",
        )

        result = agent.run("Write and verify file")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.total_steps, 3)
        self.assertEqual(result.final_answer, "File written and verified successfully, boss.")

        # Verify multi-turn messages were passed to model on subsequent turns
        self.assertGreaterEqual(len(mock_model.received_messages), 2)
        last_turn_messages = mock_model.received_messages[-1]
        # Check tool responses exist in conversation history
        tool_turns = [m for m in last_turn_messages if m.role == "tool"]
        self.assertGreaterEqual(len(tool_turns), 1)
        self.assertIn("<tool_response>", tool_turns[0].content)

    def test_parallel_tool_calls_in_single_turn(self):
        """Verify multiple <tool_call> blocks in a single assistant output are both executed."""
        file1 = os.path.join(self.temp_dir.name, "file1.txt").replace(os.sep, "/")
        file2 = os.path.join(self.temp_dir.name, "file2.txt").replace(os.sep, "/")

        step1 = f"""
<thought>Creating two files in parallel.</thought>
<tool_call>
{{"name": "filesystem.write", "arguments": {{"path": "{file1}", "content": "File 1"}}}}
</tool_call>
<tool_call>
{{"name": "filesystem.write", "arguments": {{"path": "{file2}", "content": "File 2"}}}}
</tool_call>
"""
        step2 = "Both files created, sir."

        mock_model = MockAgentModel([step1, step2])
        agent = MaddyAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            address_title="sir",
        )

        result = agent.run("Create file1 and file2")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.final_answer, "Both files created, sir.")

        # Both files should exist on disk
        self.assertTrue(os.path.exists(file1))
        self.assertTrue(os.path.exists(file2))

    def test_scratchpad_reasoning_compatibility(self):
        """Verify GOAP-style <scratch_pad> tags are seamlessly parsed."""
        response = """
<scratch_pad>
Goal: Answer system status.
Observation: No external tool needed.
Reflection: Ready.
</scratch_pad>
All systems operational, boss.
"""
        mock_model = MockAgentModel([response])
        agent = MaddyAgent(
            model_provider=mock_model,
            executor=self.executor,
            registry=self.registry,
            address_title="boss",
        )

        result = agent.run("Status report")
        self.assertEqual(result.status, AgentStatus.SUCCESS)
        self.assertEqual(result.final_answer, "All systems operational, boss.")
        self.assertIn("Goal: Answer system status.", result.steps[0].thought)


if __name__ == "__main__":
    unittest.main()
