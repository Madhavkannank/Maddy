"""End-to-End Stress Testing & Hardening for M.A.D.D.Y. Agent Architecture.

Validates:
1. Multi-turn autonomous execution with memory persistence and context preservation.
2. Compound Supervisor task graph execution with dynamic worker lifecycle.
3. Concurrent multi-threaded memory access and atomic mutation under load.
4. Leaf worker isolation and delegation depth safety under stress.
5. Sub-second readiness and component initialization benchmarks (< 0.8s).
"""
import concurrent.futures
import os
import shutil
import tempfile
import time
from unittest.mock import MagicMock
import pytest

from agents.base import AgentStatus
from agents.bus import MessageBus
from agents.maddy_agent import MaddyAgent
from agents.supervisor import Supervisor
from memory.episodic_memory import EpisodicMemory
from memory.task_memory import TaskMemory
from models.base import ModelMessage, ModelProvider, ModelResponse
from models.queue import ModelInferenceQueue
from recovery.engine import RecoveryEngine
from tasks.graph import TaskGraph
from tasks.task import Task
from tools.base import RiskLevel, Tool
from tools.delegate_tool import create_delegate_tool, DELEGATE_BLOCKED_TOOLS
from tools.executor import ToolExecutor
from tools.memory_tool_store import MemoryStore
from tools.registry import ToolRegistry
from voice.micro_feedback import MicroFeedbackEngine


class ScriptedModelProvider(ModelProvider):
    def __init__(self, script):
        self.script = list(script)
        self.history = []

    def generate(self, prompt, system_prompt=None, messages=None, options=None):
        self.history.append({"prompt": prompt, "messages": messages})
        if self.script:
            return ModelResponse(content=self.script.pop(0))
        return ModelResponse(content="Goal achieved, boss.")

    def stream(self, prompt, system_prompt=None, messages=None, options=None):
        yield self.generate(prompt, system_prompt, messages).content

    def structured(self, prompt, schema, system_prompt=None, options=None):
        if self.script:
            import json
            txt = self.script.pop(0)
            try:
                return json.loads(txt)
            except Exception:
                return {"result": txt}
        return {}


def test_concurrent_memory_store_stress():
    """Verify concurrent reads and writes across multiple threads without file corruption."""
    temp_dir = tempfile.mkdtemp()
    try:
        store = MemoryStore(memory_dir=temp_dir)
        num_threads = 10
        ops_per_thread = 5

        def worker_task(tid: int):
            for i in range(ops_per_thread):
                store.add("memory", f"Thread {tid} entry {i}")
                content = store.load("memory")
                assert f"Thread {tid} entry {i}" in content

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, t) for t in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        final_content = store.load("memory")
        # Ensure delimiter parsing works and all entries exist
        entries = store.view("memory")["entries"]
        assert len(entries) == num_threads * ops_per_thread
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_end_to_end_agent_multi_turn_with_memory():
    """Verify agent executes a multi-turn conversation interacting with memory."""
    temp_dir = tempfile.mkdtemp()
    try:
        from tools.memory_tool import memory_aci_tool, set_memory_store
        mem_store = MemoryStore(memory_dir=temp_dir)
        set_memory_store(mem_store)

        registry = ToolRegistry()
        registry.register(memory_aci_tool)
        executor = ToolExecutor(registry=registry)

        # Turn 1: Save preference
        # Turn 2: Query preference
        responses = [
            '<thought>Remember user likes dark mode</thought>\n<tool_call>{"name": "memory", "arguments": {"action": "add", "target": "user", "content": "Prefers dark mode UI"}}</tool_call>',
            '<thought>Preference recorded.</thought>\nI have saved your preference for dark mode, boss.',
            '<thought>Check user preference.</thought>\n<tool_call>{"name": "memory", "arguments": {"action": "view", "target": "user"}}</tool_call>',
            '<thought>I see the dark mode entry.</thought>\nYour preferred interface theme is dark mode, boss.',
        ]
        model = ScriptedModelProvider(responses)

        agent = MaddyAgent(
            model_provider=model,
            executor=executor,
            registry=registry,
            max_steps=5,
            address_title="boss",
        )

        res1 = agent.run("Please remember that I prefer dark mode.")
        assert res1.status == AgentStatus.SUCCESS
        assert "dark mode" in res1.final_answer.lower()
        assert "Prefers dark mode UI" in mem_store.load("user")

        res2 = agent.run("What theme do I prefer?")
        assert res2.status == AgentStatus.SUCCESS
        assert "dark mode" in res2.final_answer.lower()
    finally:
        set_memory_store(None)
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_supervisor_compound_graph_with_workers():
    """Verify Supervisor decomposes compound task and dynamic workers execute cleanly."""
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="system_info",
            description="Get system info",
            parameter_schema={"type": "object", "properties": {}},
            handler=lambda args: {"os": "Windows 11", "status": "nominal"},
            risk_level=RiskLevel.LOW,
        )
    )
    executor = ToolExecutor(registry=registry)

    # 1. Decomposition response: 2 tasks (T1: get info, T2: summarize)
    decomp_json = (
        '{\n'
        '  "thought": "Decompose into info retrieval and summary",\n'
        '  "tasks": [\n'
        '    {"id": "T1", "description": "Check system info", "dependencies": [], "required_tools": ["system_info"]},\n'
        '    {"id": "T2", "description": "Synthesize status", "dependencies": ["T1"], "required_tools": []}\n'
        '  ]\n'
        '}'
    )
    # 2. Worker 1 response: calls tool then answers
    w1_call = '<thought>Check info</thought>\n<tool_call>{"name": "system_info", "arguments": {}}</tool_call>'
    w1_done = '<thought>Info gathered</thought>\nWindows 11 status is nominal.'
    # 3. Worker 2 response: direct answer
    w2_done = '<thought>Summarize info</thought>\nAll systems are operating normally on Windows 11, boss.'

    scripted_model = ScriptedModelProvider([decomp_json, w1_call, w1_done, w2_done])

    supervisor = Supervisor(
        model_provider=scripted_model,
        executor=executor,
        registry=registry,
        max_retries=1,
    )

    result = supervisor.run("Check system status and summarize.")
    assert result.status == "SUCCESS"
    assert len(result.completed_tasks) == 2
    assert any(t["id"] == "T1" for t in result.completed_tasks)
    assert any(t["id"] == "T2" for t in result.completed_tasks)


def test_subagent_recursion_barrier():
    """Verify that child worker cannot spawn further workers beyond MAX_SPAWN_DEPTH."""
    from tools.delegate_tool import (
        DELEGATE_BLOCKED_TOOLS,
        MAX_SPAWN_DEPTH,
        _build_child_registry,
        delegate_task_tool,
    )
    registry = ToolRegistry()
    registry.register(delegate_task_tool)
    child_reg = _build_child_registry(registry, role="orchestrator", current_depth=MAX_SPAWN_DEPTH)
    tool_names = [t.name for t in child_reg.list_tools()]
    assert "delegate_task" not in tool_names


def test_subsecond_startup_benchmark():
    """Verify that importing and instantiating MaddyAssistant / MaddyAssistant executes in under 0.8 seconds."""
    t0 = time.perf_counter()
    from app import MaddyAssistant
    asst = MaddyAssistant()
    elapsed = time.perf_counter() - t0

    assert elapsed < 0.8, f"Startup took {elapsed:.3f}s, exceeding 0.8s threshold!"
    assert hasattr(asst, "maddy_agent")
    assert hasattr(asst, "supervisor")
    assert hasattr(asst, "micro_feedback")
