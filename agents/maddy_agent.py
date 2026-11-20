"""M.A.D.D.Y. Agent Facade (Modular Autonomous Digital Daemon for You).

Integrates core execution engine (prompt builder, turn state, conversation loop)
with M.A.D.D.Y. ToolExecutor, VerificationEngine, and Personality preservation.
"""
import logging
from typing import Any, Callable, Dict, List, Optional

from agents.base import AgentResult, AgentStatus, AgentStep
from agents.core.conversation_loop import MaddyConversationLoop
from agents.core.prompt_builder import MaddyPromptBuilder
from memory.episodic_memory import EpisodicMemory
from memory.task_memory import TaskMemory
from memory.working_memory import WorkingMemory
from models.base import ModelProvider
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


def build_maddy_system_prompt(
    registry: ToolRegistry,
    base_personality_prompt: Optional[str] = None,
    address_title: str = "boss",
    user_md_content: Optional[str] = None,
    memory_md_content: Optional[str] = None,
) -> str:
    """Helper to build 3-tier M.A.D.D.Y. system prompt."""
    builder = MaddyPromptBuilder(registry=registry, address_title=address_title)
    return builder.build_system_prompt(
        context="general",
        user_md_content=user_md_content,
        memory_md_content=memory_md_content,
    )


class MaddyAgent:
    """Agent for M.A.D.D.Y.

    Combines function-calling protocol (<tools>, <tool_call>, <tool_response>)
    with persistent multi-turn history, action verification, and M.A.D.D.Y. personality.
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
        on_tool_start: Optional[Callable[[str], None]] = None,
        memory_manager: Optional[Any] = None,
    ):
        self.model_provider = model_provider
        self.executor = executor
        self.registry = registry
        self.max_steps = max_steps
        self.task_memory = task_memory
        self.working_memory = working_memory
        self.episodic_memory = episodic_memory
        self.address_title = address_title
        self.on_tool_start = on_tool_start
        self.memory_manager = memory_manager

        if verification_engine is None:
            try:
                from verification.engine import VerificationEngine
                self.verification_engine = VerificationEngine()
            except Exception:
                self.verification_engine = None
        else:
            self.verification_engine = verification_engine

        self.loop = MaddyConversationLoop(
            model_provider=self.model_provider,
            executor=self.executor,
            registry=self.registry,
            max_steps=self.max_steps,
            address_title=self.address_title,
            verification_engine=self.verification_engine,
            on_tool_start=self.on_tool_start,
            memory_manager=self.memory_manager,
        )

    def run(self, goal: str, base_system_prompt: Optional[str] = None) -> AgentResult:
        """Executes the goal using multi-turn reasoning and tool calling."""
        return self.loop.execute(
            goal=goal,
            base_system_prompt=base_system_prompt,
            task_memory=self.task_memory,
            working_memory=self.working_memory,
            episodic_memory=self.episodic_memory,
        )
