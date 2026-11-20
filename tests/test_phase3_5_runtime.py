"""Runtime integration tests for the rebuilt non-Telegram application boundary."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from core.llm_client import ModelResponse
from core.memory import MemoryStore
from core.permissions import PermissionGate
from core.tools.base import ToolRegistry
from core.trace_store import TraceStore
from app.runtime import MaddyRuntime


class FakeLLM:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def chat_completion(self, messages, tools=None, model=None, temperature=None):
        self.calls.append({"messages": messages, "tools": tools})
        return ModelResponse(content="Core runtime response", tool_calls=[])


def make_runtime(tmp_path: Path) -> tuple[MaddyRuntime, FakeLLM]:
    llm = FakeLLM()
    runtime = MaddyRuntime(
        llm_client=llm,
        trace_store=TraceStore(tmp_path / "traces.db"),
        memory_store=MemoryStore(tmp_path / "memory.db"),
        permission_gate=PermissionGate(allowed_write_directories=[str(tmp_path)]),
    )
    return runtime, llm


def test_runtime_registers_phase3_5_tools(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    names = set(runtime.list_tools())
    assert {
        "system_control",
        "app_control",
        "media_control",
        "web_browser",
        "camera_tool",
        "arxiv_tool",
        "timer_tool",
    }.issubset(names)
    assert "remember" in names
    assert "recall" in names
    assert "forget" in names


def test_runtime_executes_exclusively_through_core(tmp_path):
    runtime, llm = make_runtime(tmp_path)
    result = runtime.execute("report runtime status")
    assert result["status"] == "completed"
    assert result["response"] == "Core runtime response"
    assert len(llm.calls) == 1
    assert any(msg["role"] == "user" and msg["content"] == "report runtime status" for msg in llm.calls[0]["messages"])


def test_runtime_empty_prompt_is_structured_error(tmp_path):
    runtime, llm = make_runtime(tmp_path)
    result = runtime.execute("  ")
    assert result["status"] == "error"
    assert llm.calls == []


def test_runtime_resume_delegates_to_core(tmp_path):
    runtime, _ = make_runtime(tmp_path)
    called = {}

    def fake_resume(task_id):
        called["task_id"] = task_id
        return {"status": "completed", "task_id": task_id, "response": "resumed"}

    runtime.core.resume_turn = fake_resume
    result = runtime.resume("task_123")
    assert called["task_id"] == "task_123"
    assert result["response"] == "resumed"
