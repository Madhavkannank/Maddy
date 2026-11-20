"""Single Agent core loop implementation for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Combines structured tool calling (<tools>, <tool_call>) with M.A.D.D.Y. personality prompts
and multi-turn verification.
"""
from typing import Any, Dict, List, Optional
from agents.base import AgentResult, AgentStatus, AgentStep
from agents.maddy_agent import MaddyAgent, build_maddy_system_prompt
from models.base import ModelProvider
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

from memory.working_memory import WorkingMemory
from memory.task_memory import TaskMemory
from memory.episodic_memory import EpisodicMemory


class SingleAgent(MaddyAgent):
    """M.A.D.D.Y. SingleAgent core execution loop.
    Subclasses MaddyAgent for function-calling compatibility,
    multi-turn message history, action verification, and personality preservation.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        executor: ToolExecutor,
        registry: ToolRegistry,
        max_steps: int = 10,
        task_memory: Optional[TaskMemory] = None,
        working_memory: Optional[WorkingMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        verification_engine: Optional[Any] = None,
        address_title: str = "boss",
    ):
        super().__init__(
            model_provider=model_provider,
            executor=executor,
            registry=registry,
            max_steps=max_steps,
            task_memory=task_memory,
            working_memory=working_memory,
            episodic_memory=episodic_memory,
            verification_engine=verification_engine,
            address_title=address_title,
        )
