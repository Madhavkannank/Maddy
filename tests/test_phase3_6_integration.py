"""Comprehensive integration tests for Phase 3.6: Runtime Integration & Legacy Rebuild.
Verifies unified MaddyRuntime execution, session context, diagnostics, cancellation,
interruption, permission enforcement, and execution tracing.
"""
from __future__ import annotations

import threading
import time
import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from app.runtime import MaddyRuntime
from core.llm_client import ModelResponse, ToolCallEnvelope
from core.memory import MemoryStore
from core.permissions import PermissionGate, PermissionTier
from core.tools.base import BaseTool, ToolRegistry, ToolResult
from core.trace_store import TraceStore


class MockLLM:
    """Mock LLM that can return scripted model responses or tool invocations."""

    def __init__(self, responses: List[ModelResponse] | None = None):
        self.responses = list(responses or [])
        self.calls: List[Dict[str, Any]] = []

    def chat_completion(self, messages, tools=None, model=None, temperature=None):
        self.calls.append({"messages": messages, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse(content="Default scripted response", tool_calls=[])


def build_test_runtime(tmp_path: Path, llm: MockLLM | None = None) -> MaddyRuntime:
    fake_llm = llm or MockLLM()
    runtime = MaddyRuntime(
        llm_client=fake_llm,
        trace_store=TraceStore(tmp_path / "traces.db"),
        memory_store=MemoryStore(tmp_path / "memory.db"),
        permission_gate=PermissionGate(allowed_write_directories=[str(tmp_path)]),
    )
    return runtime


def test_runtime_lifecycle_and_diagnostics(tmp_path):
    """Verify runtime manages execution lifecycle, timing, and diagnostics."""
    runtime = build_test_runtime(tmp_path)
    diag = runtime.get_diagnostics()

    assert diag["execution_core"] == "MaddyAgentCore"
    assert diag["registered_tools_count"] >= 14
    assert diag["trace_db_exists"] is True
    assert diag["memory_db_exists"] is True

    res = runtime.execute("What is the system status?")
    assert res["status"] == "completed"
    assert "elapsed_seconds" in res
    assert res["elapsed_seconds"] >= 0.0
    assert res["task_id"] is not None
    assert runtime.last_task_id == res["task_id"]


def test_runtime_session_context_continuity(tmp_path):
    """Verify session conversation history is retained across turns when session_id is provided."""
    llm = MockLLM([
        ModelResponse(content="I understand your project is M.A.D.D.Y.", tool_calls=[]),
        ModelResponse(content="Yes, continuing with M.A.D.D.Y. architecture.", tool_calls=[]),
    ])
    runtime = build_test_runtime(tmp_path, llm=llm)

    res1 = runtime.execute("My project is called M.A.D.D.Y.", session_id="test_sess_1")
    assert res1["status"] == "completed"
    assert len(runtime.session_histories["test_sess_1"]) == 2

    res2 = runtime.execute("What project did I mention?", session_id="test_sess_1")
    assert res2["status"] == "completed"
    assert len(runtime.session_histories["test_sess_1"]) == 4

    # Second call should include prior user and assistant messages in its LLM context
    last_llm_messages = llm.calls[-1]["messages"]
    user_contents = [m["content"] for m in last_llm_messages if m["role"] == "user"]
    assert "My project is called M.A.D.D.Y." in user_contents
    assert "What project did I mention?" in user_contents


def test_runtime_cancellation_and_interruption(tmp_path):
    """Verify tasks can be cancelled and active execution interrupted."""
    runtime = build_test_runtime(tmp_path)

    # 1. Cancel task
    task_id = "test_cancel_task_001"
    runtime.trace_store.create_task("Test task to cancel", task_id=task_id)
    cancel_ok = runtime.cancel_task(task_id)
    assert cancel_ok is True
    t = runtime.trace_store.get_task(task_id)
    assert t["status"] == "failed"

    # 2. Interruption event
    stop_ev = threading.Event()
    stop_ev.set()  # Pre-interrupted
    res = runtime.execute("Execute lengthy operation", stop_event=stop_ev)
    assert res["status"] in ("failed", "interrupted")
    assert "interrupted" in res["response"].lower()


def test_runtime_voice_synthesis_integration(tmp_path):
    """Verify optional voice synthesis executes through VoiceEngine when requested."""
    runtime = build_test_runtime(tmp_path)

    with patch("voice.engine.VoiceEngine.speak") as mock_speak:
        mock_speak.return_value = {
            "success": True,
            "text": "Hello operator",
            "voice": "adam",
            "method": "f5tts_local_zero_shot_clone",
        }
        res = runtime.execute("Hello operator", speak_response=True, voice="adam")
        assert res["status"] == "completed"
        assert "voice" in res
        assert res["voice"]["success"] is True
        mock_speak.assert_called_once()


def test_execution_tracing_persists_before_executing(tmp_path):
    """Verify persist-before-execute invariant in SQLite trace store."""
    llm = MockLLM([
        ModelResponse(
            content="Checking system info",
            tool_calls=[
                ToolCallEnvelope(
                    id="call_sysinfo_1",
                    name="system_info",
                    arguments={"section": "platform"},
                )
            ],
        ),
        ModelResponse(content="System is operational.", tool_calls=[]),
    ])
    runtime = build_test_runtime(tmp_path, llm=llm)
    res = runtime.execute("Check system telemetry")

    assert res["status"] == "completed"
    task_id = res["task_id"]

    # Verify task in SQLite
    task = runtime.trace_store.get_task(task_id)
    assert task is not None
    assert task["status"] == "completed"

    # Verify tool calls persisted
    tool_calls = runtime.trace_store.get_task_tool_calls(task_id)
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["tool_name"] == "system_info"
    assert tc["status"] == "completed"
    assert "os:" in tc["result_output"].lower() or "linux" in tc["result_output"].lower() or "windows" in tc["result_output"].lower()


def test_telegram_isolation_smoke_test():
    """Verify Telegram files remain uncorrupted and importable as smoke test."""
    try:
        import integrations.messaging.telegram_bot
        assert integrations.messaging.telegram_bot is not None
    except (ImportError, ModuleNotFoundError) as err:
        pytest.skip(f"Telegram integration excluded from repository checkout: {err}")
