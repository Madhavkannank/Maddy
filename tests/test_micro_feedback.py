"""Tests for Verbal Micro-Feedback Engine for Voice Grounding in M.A.D.D.Y.

Verifies tool latency classification, persona-aligned cue formatting,
turn debouncing, asynchronous non-blocking speech dispatch, and agent integration.
"""
import time
import pytest
from unittest.mock import MagicMock

from voice.micro_feedback import MicroFeedbackEngine, SLOW_TOOL_CUES, FAST_TOOLS
from agents.maddy_agent import MaddyAgent
from agents.base import AgentStatus
from tools.registry import ToolRegistry
from tools.executor import ToolExecutor
from tools.base import Tool, RiskLevel


def dummy_slow_handler(args):
    return {"result": "search results"}


def dummy_fast_handler(args):
    return {"result": "42"}


def make_dummy_tool(name: str, handler):
    return Tool(
        name=name,
        description=f"Dummy tool {name}",
        parameter_schema={"type": "object", "properties": {}},
        handler=handler,
        risk_level=RiskLevel.LOW,
    )


from models.base import ModelResponse


class MockModelProvider:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, **kwargs):
        if self.responses:
            return ModelResponse(content=self.responses.pop(0))
        return ModelResponse(content="Done.")


def test_slow_tool_cues():
    engine = MicroFeedbackEngine(speak_fn=MagicMock(), address_title="boss", enabled=True)
    for tool_name in ["browser_search", "arxiv_search", "delegate_task", "spotify_play"]:
        cue = engine.get_cue(tool_name)
        assert cue is not None
        assert "boss" in cue


def test_fast_tool_silent():
    engine = MicroFeedbackEngine(speak_fn=MagicMock(), address_title="boss", enabled=True)
    for tool_name in ["calculator", "memory", "skills_list", "filesystem_read"]:
        cue = engine.get_cue(tool_name)
        assert cue is None


def test_turn_debounce():
    mock_speak = MagicMock()
    engine = MicroFeedbackEngine(speak_fn=mock_speak, address_title="sir", enabled=True)

    # First slow tool triggers cue
    cue1 = engine.on_tool_start("browser_search")
    assert cue1 is not None
    assert "sir" in cue1

    # Second slow tool in same turn is debounced (silent)
    cue2 = engine.on_tool_start("arxiv_search")
    assert cue2 is None

    # Third slow tool in same turn is also debounced
    cue3 = engine.on_tool_start("delegate_task")
    assert cue3 is None

    # Reset turn for next turn
    engine.reset_turn()
    cue4 = engine.on_tool_start("arxiv_search")
    assert cue4 is not None
    assert "sir" in cue4


def test_disabled_engine_is_silent():
    mock_speak = MagicMock()
    engine = MicroFeedbackEngine(speak_fn=mock_speak, address_title="boss", enabled=False)
    cue = engine.on_tool_start("browser_search")
    assert cue is None
    mock_speak.assert_not_called()


def test_async_non_blocking_dispatch():
    spoken_cues = []

    def slow_speak(text):
        time.sleep(0.05)
        spoken_cues.append(text)

    engine = MicroFeedbackEngine(speak_fn=slow_speak, address_title="boss", enabled=True)
    start_t = time.time()
    cue = engine.on_tool_start("browser_search")
    elapsed = time.time() - start_t

    # on_tool_start must return immediately (< 0.03s), not waiting for slow_speak
    assert elapsed < 0.03
    assert cue is not None

    # Wait briefly for daemon thread to complete speech
    time.sleep(0.1)
    assert len(spoken_cues) == 1
    assert spoken_cues[0] == cue


def test_maddy_agent_triggers_micro_feedback_on_slow_tool():
    registry = ToolRegistry()
    registry.register(make_dummy_tool("browser_search", dummy_slow_handler))
    executor = ToolExecutor(registry=registry)

    spoken_cues = []
    engine = MicroFeedbackEngine(speak_fn=lambda t: spoken_cues.append(t), address_title="boss", enabled=True)

    # Model calls browser_search then terminates
    responses = [
        '<thought>Need to search the web.</thought>\n<tool_call>{"name": "browser_search", "arguments": {"query": "AI news"}}</tool_call>',
        '<thought>I have the result.</thought>\nHere is the latest news, boss.',
    ]
    model = MockModelProvider(responses)

    agent = MaddyAgent(
        model_provider=model,
        executor=executor,
        registry=registry,
        max_steps=5,
        on_tool_start=engine.on_tool_start,
    )

    result = agent.run("What is the latest AI news?")
    assert result.status == AgentStatus.SUCCESS
    assert "Here is the latest news" in result.final_answer

    # Give thread a split second to record spoken cue
    time.sleep(0.05)
    assert len(spoken_cues) == 1
    assert "boss" in spoken_cues[0]
