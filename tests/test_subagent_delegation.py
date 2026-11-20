"""Comprehensive Tests for Subagent Delegation & Orchestration Architecture in M.A.D.D.Y.

Verifies:
1. Single task delegation in isolated child agent session.
2. Parallel batch delegation across concurrent subagents.
3. Toolset filtering (leaf workers cannot spawn recursive subagents).
4. Spawn depth limiting (prevents infinite delegation trees).
5. Timeout handling and diagnostic error recovery.
6. End-to-end delegation within MaddyConversationLoop.
"""
from __future__ import annotations

import json
import time
import pytest

from agents.base import AgentStatus
from agents.core.conversation_loop import MaddyConversationLoop
from models.base import ModelMessage, ModelProvider, ModelResponse
from tools.base import Tool
from tools.call import ToolCall
from tools.delegate_tool import (
    DELEGATE_BLOCKED_TOOLS,
    MAX_SPAWN_DEPTH,
    DelegateToolManager,
    _build_child_registry,
    create_delegate_tool,
    delegate_task_tool,
    execute_single_delegation,
)
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from tools.result import ToolResult


class MockProvider(ModelProvider):
    def __init__(self, response_map: dict[str, list[str]] | None = None, default_responses: list[str] | None = None):
        self.response_map = response_map or {}
        self.default_responses = list(default_responses or [])
        self.call_history: list[str] = []

    def generate(self, prompt: str, system_prompt: str = "", messages=None, options=None) -> ModelResponse:
        self.call_history.append(prompt)
        # Check if any key in response_map matches prompt or goal
        for key, resps in self.response_map.items():
            if key in prompt or (messages and any(key in (m.content or "") for m in messages)):
                if resps:
                    return ModelResponse(content=resps.pop(0))

        if self.default_responses:
            return ModelResponse(content=self.default_responses.pop(0))

        return ModelResponse(content="Default child task completed, boss.")

    def stream(self, prompt: str, system_prompt: str = None, messages=None, options=None):
        yield self.generate(prompt, system_prompt, messages).content

    def structured(self, prompt: str, schema: dict, system_prompt: str = None, options=None):
        return json.loads(self.generate(prompt, system_prompt).content)


def create_mock_tool(name: str, result_data: dict) -> Tool:
    return Tool(
        name=name,
        description=f"Mock tool {name}",
        parameter_schema={"type": "object", "properties": {}},
        handler=lambda args: result_data,
    )


class TestSubagentDelegation:
    @pytest.fixture
    def setup_environment(self):
        registry = ToolRegistry()
        registry.register(create_mock_tool("system_info", {"cpu": "15%", "ram": "45%"}))
        registry.register(create_mock_tool("network_ping", {"latency": "12ms"}))
        registry.register(delegate_task_tool)
        executor = ToolExecutor(registry)
        return registry, executor

    def test_child_registry_strips_delegation_for_leaf_worker(self, setup_environment):
        registry, _ = setup_environment

        # Leaf child registry
        leaf_reg = _build_child_registry(registry, role="leaf", current_depth=1)
        tool_names = [t.name for t in leaf_reg.list_tools()]

        assert "system_info" in tool_names
        assert "network_ping" in tool_names
        # Must not contain delegation tools
        for blocked in DELEGATE_BLOCKED_TOOLS:
            assert blocked not in tool_names

    def test_child_registry_allows_delegation_for_orchestrator_below_max_depth(self, setup_environment):
        registry, _ = setup_environment

        # Orchestrator child registry below max depth
        orch_reg = _build_child_registry(registry, role="orchestrator", current_depth=1)
        tool_names = [t.name for t in orch_reg.list_tools()]
        assert "delegate_task" in tool_names

        # Orchestrator at max depth must be stripped
        deep_reg = _build_child_registry(registry, role="orchestrator", current_depth=MAX_SPAWN_DEPTH)
        deep_names = [t.name for t in deep_reg.list_tools()]
        assert "delegate_task" not in deep_names

    def test_single_task_delegation_execution(self, setup_environment):
        registry, executor = setup_environment
        provider = MockProvider(
            response_map={
                "Check hardware": [
                    "<thought>Checking hardware.</thought>\n<tool_call>{\"name\": \"system_info\", \"arguments\": {}}</tool_call>",
                    "Hardware check complete: CPU is at 15% and RAM is at 45%, boss.",
                ]
            }
        )

        outcome = execute_single_delegation(
            task_description="Check hardware",
            context="Routine system inspection",
            role="leaf",
            timeout=30.0,
            model_provider=provider,
            parent_executor=executor,
            parent_registry=registry,
        )

        assert outcome["status"] == "SUCCESS"
        assert "Hardware check complete" in outcome["summary"]
        assert outcome["total_steps"] == 2

    def test_batch_parallel_delegation(self, setup_environment):
        registry, executor = setup_environment
        provider = MockProvider(
            response_map={
                "Task 1": ["Task 1 finished successfully."],
                "Task 2": ["Task 2 finished successfully."],
            }
        )

        manager = DelegateToolManager(
            model_provider=provider,
            registry=registry,
            executor=executor,
            current_depth=1,
        )

        batch_result = manager.delegate(
            tasks=[
                {"id": "t1", "task": "Execute Task 1"},
                {"id": "t2", "task": "Execute Task 2"},
            ]
        )

        assert batch_result["success"] is True
        assert batch_result["batch_size"] == 2
        results = batch_result["results"]
        assert len(results) == 2
        statuses = [r["status"] for r in results]
        assert all(s == "SUCCESS" for s in statuses)

    def test_subagent_timeout_handling(self, setup_environment):
        registry, executor = setup_environment

        class HangingProvider(ModelProvider):
            def generate(self, prompt, system_prompt="", messages=None, options=None):
                time.sleep(1.0)
                return ModelResponse(content="Too late")

            def stream(self, prompt, system_prompt=None, messages=None, options=None):
                yield "Too late"

            def structured(self, prompt, schema, system_prompt=None, options=None):
                return {}

        outcome = execute_single_delegation(
            task_description="Slow task",
            timeout=0.2,  # 200ms timeout
            model_provider=HangingProvider(),
            parent_executor=executor,
            parent_registry=registry,
        )

        assert outcome["status"] == "TIMED_OUT"
        assert "exceeded deadline" in outcome["error"]

    def test_end_to_end_conversation_loop_delegation(self, setup_environment):
        registry, executor = setup_environment

        # Parent calls delegate_task tool, receives summary, then responds to user
        parent_responses = [
            '<thought>Delegating diagnostic check to subagent.</thought>\n<tool_call>{"name": "delegate_task", "arguments": {"task": "Check system diagnostics"}}</tool_call>',
            "All subagent diagnostics reported healthy, boss.",
        ]
        # Child subagent responses
        child_responses = [
            "Subagent analysis: all subsystems operating within optimal nominal bounds, boss.",
        ]

        provider = MockProvider(
            response_map={
                "Check system diagnostics": child_responses,
            },
            default_responses=parent_responses,
        )

        loop = MaddyConversationLoop(
            model_provider=provider,
            executor=executor,
            registry=registry,
            max_steps=5,
            address_title="boss",
        )

        res = loop.execute("Run deep system diagnostics")
        assert res.status == AgentStatus.SUCCESS
        assert "All subagent diagnostics reported healthy" in res.final_answer
        assert len(res.steps) == 2
        assert res.steps[0].tool_call.name == "delegate_task"
