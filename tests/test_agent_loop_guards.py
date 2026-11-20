"""Comprehensive Tests for Conversation Loop & Execution Safety Guards in M.A.D.D.Y.

Verifies repetition detection, tool output truncation, empty response recovery,
unclosed tool tag parsing, and runaway tool call prevention.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from agents.base import AgentStatus
from agents.core.conversation_loop import MaddyConversationLoop
from agents.core.repetition_guard import (
    MIN_FRAGMENT_LENGTH,
    REPETITION_LOOP_INTERRUPTED,
    clean_repetition_if_dominated,
    is_repetition_dominated,
)
from agents.core.turn_state import (
    DEFAULT_TOOL_OUTPUT_MAX_CHARS,
    TurnState,
    truncate_tool_output,
)
from models.base import ModelMessage, ModelProvider, ModelResponse
from tools.base import Tool
from tools.call import ToolCall
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.result import ToolResult


def create_echo_tool() -> Tool:
    return Tool(
        name="echo_tool",
        description="Echoes back the input text.",
        parameter_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        handler=lambda args: {"echo": args.get("text", "")},
    )


class MockProvider(ModelProvider):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.call_count = 0
        self.prompts_received = []
        self.messages_received = []

    def generate(self, prompt: str, system_prompt: str = "", messages=None, options=None) -> ModelResponse:
        self.call_count += 1
        self.prompts_received.append(prompt)
        if messages:
            self.messages_received.append(list(messages))
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = "Default fallback response, boss."
        return ModelResponse(content=content)

    def stream(self, prompt: str, system_prompt: str = None, messages=None, options=None):
        yield self.generate(prompt, system_prompt, messages).content

    def structured(self, prompt: str, schema: dict, system_prompt: str = None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)


# ---------------------------------------------------------------------------
# 1. Truncate Tool Output Unit Tests
# ---------------------------------------------------------------------------

class TestToolOutputTruncation:
    def test_short_output_preserved_intact(self):
        short = "File content line 1\nFile content line 2"
        res = truncate_tool_output(short, max_chars=4000)
        assert res == short

    def test_long_output_truncated_with_head_and_tail(self):
        head_sample = "START_MARKER_" + ("A" * 2600)
        tail_sample = ("Z" * 1200) + "_END_MARKER"
        giant_text = head_sample + ("M" * 5000) + tail_sample
        assert len(giant_text) > 8000

        res = truncate_tool_output(giant_text, max_chars=4000, head_chars=2500, tail_chars=1000)

        assert "START_MARKER_" in res
        assert "_END_MARKER" in res
        assert "characters omitted to preserve context budget" in res
        assert len(res) < len(giant_text)

    def test_non_string_conversion(self):
        data = {"status": "ok", "count": 42}
        res = truncate_tool_output(data, max_chars=100)
        assert "status" in res


# ---------------------------------------------------------------------------
# 2. Repetition Guard Unit Tests
# ---------------------------------------------------------------------------

class TestRepetitionGuard:
    def test_short_or_varied_text_not_flagged(self):
        assert not is_repetition_dominated("Hello boss, how can I help you?")
        assert not is_repetition_dominated("A" * 50)

        varied_text = (
            "I checked the system diagnostics and all services are running properly. "
            "Next, we should examine the memory foundation and conversation loop tests. "
            "Everything is operating within acceptable parameters and ready for deployment. "
            "We have full visibility into the execution traces and system status reports. "
            "Standing by for further instructions whenever you are ready, boss."
        )
        assert len(varied_text) >= MIN_FRAGMENT_LENGTH
        assert not is_repetition_dominated(varied_text)

    def test_line_repetition_loop_flagged(self):
        repeated_line = "I am checking the status of the file system now.\n"
        pathological_loop = repeated_line * 20
        assert len(pathological_loop) > MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(pathological_loop)

        cleaned = clean_repetition_if_dominated(pathological_loop)
        assert REPETITION_LOOP_INTERRUPTED in cleaned
        assert len(cleaned) < len(pathological_loop)

    def test_window_repetition_loop_flagged(self):
        phrase = "The quick brown fox jumps over the lazy dog repeatedly! "
        chunk_loop = phrase * 15
        assert len(chunk_loop) > MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(chunk_loop)

        cleaned = clean_repetition_if_dominated(chunk_loop)
        assert REPETITION_LOOP_INTERRUPTED in cleaned


# ---------------------------------------------------------------------------
# 3. Turn State Unit Tests
# ---------------------------------------------------------------------------

class TestTurnState:
    def test_runaway_identical_tool_calls_detected(self):
        state = TurnState(max_repeated_tool_calls=3)

        assert state.record_tool_call("file_read", {"path": "test.txt"}) is True
        assert state.record_tool_call("file_read", {"path": "test.txt"}) is True
        # 3rd identical call trips the runaway guard
        assert state.record_tool_call("file_read", {"path": "test.txt"}) is False

    def test_different_tool_calls_or_args_allowed(self):
        state = TurnState(max_repeated_tool_calls=3)

        assert state.record_tool_call("file_read", {"path": "a.txt"}) is True
        assert state.record_tool_call("file_read", {"path": "b.txt"}) is True
        assert state.record_tool_call("file_read", {"path": "c.txt"}) is True
        assert state.record_tool_call("echo_tool", {"text": "hello"}) is True

    def test_context_budget_compaction(self):
        state = TurnState(token_budget_chars=1000)

        # Build messages exceeding 1000 chars
        messages = [
            ModelMessage(role="system", content="System instructions"),
            ModelMessage(role="user", content="User goal"),
            ModelMessage(role="assistant", content="Calling tool"),
            ModelMessage(role="tool", content="X" * 1500),
            ModelMessage(role="assistant", content="Next step"),
            ModelMessage(role="tool", content="Y" * 1500),
            ModelMessage(role="assistant", content="Final step"),
            ModelMessage(role="user", content="Recent feedback"),
        ]

        compacted = state.manage_context_budget(messages)
        # Verify system and user goal preserved
        assert compacted[0].role == "system"
        assert compacted[1].content == "User goal"
        # Verify older tool messages were truncated
        tool_msg_1 = [m for m in compacted if m.role == "tool"][0]
        assert len(tool_msg_1.content) < 600
        assert "Output truncated" in tool_msg_1.content
        # Verify most recent 2 messages kept intact
        assert compacted[-2].content == "Final step"
        assert compacted[-1].content == "Recent feedback"


# ---------------------------------------------------------------------------
# 4. Conversation Loop Integration Guards
# ---------------------------------------------------------------------------

class TestAgentLoopGuards:
    @pytest.fixture
    def setup_loop(self):
        registry = ToolRegistry()
        registry.register(create_echo_tool())
        executor = ToolExecutor(registry)
        return registry, executor

    def test_empty_response_recovery_and_nudge(self, setup_loop):
        registry, executor = setup_loop
        # Sequence: empty response, empty response, empty response (trips fallback)
        provider = MockProvider(["", "", ""])
        loop = MaddyConversationLoop(
            model_provider=provider,
            executor=executor,
            registry=registry,
            max_steps=5,
        )

        result = loop.execute("Hello M.A.D.D.Y.")
        assert result.status == AgentStatus.SUCCESS
        assert "I am here and ready, boss" in result.final_answer

    def test_repetition_dominated_generation_interrupted(self, setup_loop):
        registry, executor = setup_loop
        repeating_line = "Checking status of the core power matrix.\n"
        dominated_response = repeating_line * 25
        provider = MockProvider([dominated_response])
        loop = MaddyConversationLoop(
            model_provider=provider,
            executor=executor,
            registry=registry,
            max_steps=3,
        )

        result = loop.execute("Check system")
        assert result.status == AgentStatus.SUCCESS
        assert REPETITION_LOOP_INTERRUPTED in result.final_answer

    def test_unclosed_tool_call_tag_tolerance(self, setup_loop):
        registry, executor = setup_loop
        # Model emits <tool_call> without closing tag at turn boundary
        unclosed_output = (
            "<thought>Let's echo back the input.</thought>\n"
            '<tool_call>{"name": "echo_tool", "arguments": {"text": "live stream test"}}'
        )
        final_answer = "Echo confirmed, boss."
        provider = MockProvider([unclosed_output, final_answer])

        loop = MaddyConversationLoop(
            model_provider=provider,
            executor=executor,
            registry=registry,
            max_steps=3,
        )

        result = loop.execute("Echo message")
        assert result.status == AgentStatus.SUCCESS
        assert result.final_answer == "Echo confirmed, boss."
        assert len(result.steps) == 2
        assert result.steps[0].tool_call.name == "echo_tool"
        assert result.steps[0].tool_result.data == {"echo": "live stream test"}

    def test_function_call_tag_synonym_parsed(self, setup_loop):
        registry, executor = setup_loop
        fc_output = (
            "<thought>Testing function_call tag.</thought>\n"
            '<function_call>\n{"name": "echo_tool", "arguments": {"text": "open ai tag"}}\n</function_call>'
        )
        provider = MockProvider([fc_output, "Task complete, boss."])

        loop = MaddyConversationLoop(
            model_provider=provider,
            executor=executor,
            registry=registry,
            max_steps=3,
        )

        result = loop.execute("Test function call")
        assert result.status == AgentStatus.SUCCESS
        assert result.steps[0].tool_call.name == "echo_tool"
        assert result.steps[0].tool_result.data["echo"] == "open ai tag"

    def test_runaway_tool_call_interception(self, setup_loop):
        registry, executor = setup_loop
        # Model gets stuck in infinite tool loop calling identical arguments
        tool_step = '<tool_call>{"name": "echo_tool", "arguments": {"text": "stuck"}}</tool_call>'
        # Send identical tool call 3 times, then model realizes and gives final answer
        provider = MockProvider([
            tool_step,
            tool_step,
            tool_step,
            "Recovered from runaway loop, boss.",
        ])

        loop = MaddyConversationLoop(
            model_provider=provider,
            executor=executor,
            registry=registry,
            max_steps=5,
        )

        result = loop.execute("Run test")
        assert result.status == AgentStatus.SUCCESS
        # Step 3 must have tripped the runaway loop detector
        assert len(result.steps) >= 3
        third_step = result.steps[2]
        assert third_step.tool_result.success is False
        assert "repeated execution loop" in third_step.tool_result.error.lower()
