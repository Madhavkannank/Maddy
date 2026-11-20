"""
Dynamic Worker Agent wrapper for isolated capability execution and structured message communication.
"""
from typing import Any, Dict, List, Optional
from agents.base import AgentResult, AgentStatus
from agents.bus import MessageBus
from agents.message import AgentMessage, MessageType
from agents.single_agent import SingleAgent
from models.base import ModelProvider
from tasks.task import Task
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class DynamicWorkerAgent:
    """
    Capability-isolated dynamic worker executing assigned tasks and communicating over MessageBus.
    """

    def __init__(
        self,
        worker_id: str,
        model_provider: ModelProvider,
        executor: ToolExecutor,
        registry: ToolRegistry,
        message_bus: MessageBus,
        max_steps: int = 10,
        verification_engine: Optional[Any] = None,
        on_tool_start: Optional[Any] = None,
    ):
        self.worker_id = worker_id
        self.model_provider = model_provider
        self.executor = executor
        self.registry = registry
        self.message_bus = message_bus
        self.max_steps = max_steps
        self.verification_engine = verification_engine
        self.on_tool_start = on_tool_start

        # Register worker on bus
        self.message_bus.register_agent(self.worker_id)

    def execute_task(self, task: Task, task_goal: str, supervisor_id: str = "supervisor") -> AgentResult:
        """
        Executes an assigned task and publishes structured outcome messages to the supervisor.
        """
        # 1. Post TASK_PROGRESS message
        progress_msg = AgentMessage(
            sender=self.worker_id,
            receiver=supervisor_id,
            msg_type=MessageType.TASK_PROGRESS,
            task_id=task.id,
            payload={"status": "RUNNING", "goal": task_goal},
        )
        self.message_bus.publish(progress_msg)

        # 2. Build isolated ToolRegistry matching required_tools
        worker_registry = self._build_isolated_registry(task.required_tools)
        worker_executor = ToolExecutor(
            registry=worker_registry,
            permission_engine=getattr(self.executor, "permission_engine", None),
        )

        # 3. Instantiate MaddyAgent for execution (with SingleAgent fallback)
        try:
            from agents.maddy_agent import MaddyAgent
            agent = MaddyAgent(
                model_provider=self.model_provider,
                executor=worker_executor,
                registry=worker_registry,
                max_steps=self.max_steps,
                verification_engine=self.verification_engine,
                on_tool_start=self.on_tool_start,
            )
        except Exception:
            agent = SingleAgent(
                model_provider=self.model_provider,
                executor=worker_executor,
                registry=worker_registry,
                max_steps=self.max_steps,
                verification_engine=self.verification_engine,
            )

        agent_result = agent.run(task_goal)

        # 4. Publish structured result message
        if agent_result.status == AgentStatus.SUCCESS:
            result_msg = AgentMessage(
                sender=self.worker_id,
                receiver=supervisor_id,
                msg_type=MessageType.TASK_RESULT,
                task_id=task.id,
                payload={
                    "status": "SUCCESS",
                    "final_answer": agent_result.final_answer,
                    "total_steps": agent_result.total_steps,
                },
            )
            self.message_bus.publish(result_msg)
        else:
            failure_msg = AgentMessage(
                sender=self.worker_id,
                receiver=supervisor_id,
                msg_type=MessageType.TASK_FAILURE,
                task_id=task.id,
                payload={
                    "status": "FAILED",
                    "error_message": agent_result.error_message or "Worker task execution failed",
                },
            )
            self.message_bus.publish(failure_msg)

        return agent_result

    def _build_isolated_registry(self, required_tools: Optional[List[str]]) -> ToolRegistry:
        """
        Builds a task-isolated ToolRegistry containing tools required for the task.
        If required_tools is None or empty, defaults to full registry so worker always has capabilities.
        """
        if not required_tools:
            return self.registry

        isolated = ToolRegistry()
        for tool_name in required_tools:
            tool = self.registry.get(tool_name)
            if tool:
                isolated.register(tool)

        # Fallback to full registry if no tools were registered
        if len(isolated.list_tools()) == 0:
            return self.registry

        return isolated
