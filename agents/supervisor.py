"""
Supervisor Orchestration Engine for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Handles request decomposition, task graph execution, dynamic worker spawning,
context isolation, atomic checkpointing, stuck-task timeout handling, and crash resumption.
"""
import concurrent.futures
import json
import logging
import time
from typing import Any, Dict, List, Optional

from agents.base import AgentStatus
from agents.bus import MessageBus
from agents.dynamic_worker import DynamicWorkerAgent
from agents.message import AgentMessage, MessageType
from agents.single_agent import SingleAgent
from agents.supervisor_result import SupervisorResult
from memory.episodic_memory import EpisodicMemory
from memory.task_memory import TaskMemory
from models.base import ModelProvider
from models.queue import ModelInferenceQueue
from recovery.engine import ActionState, ErrorType, RecoveryAction, RecoveryEngine
from recovery.journal import JournalEntry, JournalEventType, RecoveryJournal
from tasks.graph import TaskGraph, TaskGraphError
from tasks.task import Task, TaskState
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SUPERVISOR_DECOMPOSITION_PROMPT = """You are the M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) Supervisor Orchestrator.
Your job is to analyze the user request and decompose it into a structured graph of tasks with explicit dependencies.

RULES:
1. Return a valid JSON object matching the output schema below.
2. Break the request into 1 or more logical tasks.
3. If a task depends on the output of a prior task, list its task ID in 'dependencies'.
4. Do NOT create circular dependencies.
5. Provide a clear description and list of required tools for each task.
6. FAST-PATH RULE: If the user request is a simple conversational query, greeting, general knowledge question, or direct Q&A that does NOT require OS commands or web automation, return exactly ONE task with required_tools = [].

OUTPUT JSON SCHEMA:
{
  "thought": "Analysis of user request and decomposition strategy",
  "tasks": [
    {
      "id": "T1",
      "description": "Clear task description",
      "dependencies": [],
      "required_tools": []
    }
  ]
}
"""

class Supervisor:
    """
    Supervisor engine responsible for request decomposition, task graph execution,
    checkpoint persistence at state boundaries, stuck-task detection, and crash resumption.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        executor: ToolExecutor,
        registry: ToolRegistry,
        max_retries: int = 2,
        default_timeout: float = 120.0,
        message_bus: Optional[MessageBus] = None,
        task_memory: Optional[TaskMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        recovery_engine: Optional[RecoveryEngine] = None,
        verification_engine: Optional[Any] = None,
        on_tool_start: Optional[Any] = None,
    ):
        if not isinstance(model_provider, ModelInferenceQueue):
            self.model_provider = ModelInferenceQueue(model_provider)
        else:
            self.model_provider = model_provider

        self.executor = executor
        self.registry = registry
        self.max_retries = max_retries
        self.default_timeout = default_timeout
        self.message_bus = message_bus or MessageBus()
        self.message_bus.register_agent("supervisor")
        self.task_memory = task_memory
        self.episodic_memory = episodic_memory
        self.recovery_engine = recovery_engine or RecoveryEngine(max_retries=self.max_retries)
        self.on_tool_start = on_tool_start
        if verification_engine is None:
            from verification.engine import VerificationEngine
            self.verification_engine = VerificationEngine()
        else:
            self.verification_engine = verification_engine
        self.journal: Optional[RecoveryJournal] = None
        self._current_graph: Optional[TaskGraph] = None

    def run(self, user_request: str) -> SupervisorResult:
        """
        Decompose a user request into a TaskGraph, checkpoint initial state,
        and execute to completion.
        """
        # 1. Decompose request into TaskGraph
        graph = self._decompose_and_build_graph(user_request)
        self._current_graph = graph

        if not self.task_memory:
            request_id = f"request_{int(time.time())}"
            self.task_memory = TaskMemory(task_id=request_id)

        self.journal = RecoveryJournal(task_id=self.task_memory.task_id)
        self.journal.record(JournalEventType.TASK_INITIALIZED, f"Goal initialized: {user_request}")

        self.task_memory.save_goal(user_request)
        self.task_memory.save_plan([t.to_dict() for t in graph.list_tasks()])

        # Initial checkpoint
        self.task_memory.save_checkpoint(graph, journal_entries=self.journal.to_dict_list())

        # 2. Execution loop
        return self._execute_graph_loop(user_request, graph)

    def _execute_graph_loop(self, user_request: str, graph: TaskGraph) -> SupervisorResult:
        """
        Core task execution loop with stuck task monitoring, context isolation,
        and atomic checkpointing at state boundaries.
        """
        while True:
            # Check for stuck tasks that exceeded timeout
            stuck_tasks = graph.find_stuck_tasks(default_timeout_seconds=self.default_timeout)
            for st in stuck_tasks:
                if self.journal:
                    self.journal.record(JournalEventType.TASK_TIMED_OUT, f"Task '{st.id}' exceeded timeout deadline", task_id=st.id)
                graph.update_status(st.id, TaskState.TIMED_OUT, error="Task exceeded execution deadline", force=True)
                retried = graph.retry_task(st.id)
                if not retried:
                    graph.update_status(st.id, TaskState.FAILED, error="Task timed out and exhausted retries", force=True)
                if self.task_memory:
                    self.task_memory.save_checkpoint(graph, journal_entries=self.journal.to_dict_list() if self.journal else None)

            ready_tasks = graph.get_ready_tasks()
            if not ready_tasks:
                break

            def _run_task(task: Task):
                # Mark RUNNING and checkpoint immediately before execution
                graph.update_status(task.id, TaskState.RUNNING)
                if self.journal:
                    self.journal.record(JournalEventType.TASK_RUNNING, f"Task '{task.id}' started execution", task_id=task.id)
                if self.task_memory:
                    self.task_memory.save_checkpoint(graph, journal_entries=self.journal.to_dict_list() if self.journal else None)

                # Context Isolation: Build task-specific goal with prerequisite results
                task_goal = self._build_isolated_task_prompt(task, graph)
                worker_id = f"worker_{task.id}"

                # Publish TASK_ASSIGN message over MessageBus
                assign_msg = AgentMessage(
                    sender="supervisor",
                    receiver=worker_id,
                    msg_type=MessageType.TASK_ASSIGN,
                    task_id=task.id,
                    payload={"goal": task_goal, "required_tools": task.required_tools},
                )
                self.message_bus.publish(assign_msg)

                # Spawn dynamic worker agent (stateless runtime reconstruction)
                worker = DynamicWorkerAgent(
                    worker_id=worker_id,
                    model_provider=self.model_provider,
                    executor=self.executor,
                    registry=self.registry,
                    message_bus=self.message_bus,
                    max_steps=10,
                    verification_engine=self.verification_engine,
                    on_tool_start=self.on_tool_start,
                )

                agent_result = worker.execute_task(task=task, task_goal=task_goal, supervisor_id="supervisor")
                return task, agent_result

            # Execute parallel if multiple independent tasks are ready, otherwise execute inline
            if len(ready_tasks) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ready_tasks), 4)) as executor:
                    task_futures = [executor.submit(_run_task, t) for t in ready_tasks]
                    task_results = [f.result() for f in concurrent.futures.as_completed(task_futures)]
            else:
                task_results = [_run_task(ready_tasks[0])]

            for task, agent_result in task_results:
                # Read worker messages sent back to supervisor
                bus_messages = self.message_bus.receive("supervisor")
                result_msg = next((m for m in bus_messages if m.msg_type == MessageType.TASK_RESULT and m.task_id == task.id), None)
                failure_msg = next((m for m in bus_messages if m.msg_type == MessageType.TASK_FAILURE and m.task_id == task.id), None)

                if agent_result.status == AgentStatus.SUCCESS:
                    final_answer = result_msg.payload.get("final_answer") if result_msg else agent_result.final_answer
                    v_report = self.verification_engine.verify_task_completion(task.id, task.description, final_answer)
                    task.verification_state = v_report.to_dict()
                    graph.update_status(
                        task.id,
                        TaskState.SUCCESS,
                        result=final_answer,
                    )
                    if self.journal:
                        self.journal.record(JournalEventType.TASK_SUCCESS, f"Task '{task.id}' completed successfully", task_id=task.id)
                else:
                    # Retry logic with bounded retries
                    retried = graph.retry_task(task.id)
                    if retried:
                        if self.journal:
                            self.journal.record(
                                JournalEventType.RETRY_ATTEMPT,
                                f"Task '{task.id}' failed, retrying (attempt {task.retry_count}/{task.max_retries})",
                                task_id=task.id,
                            )
                    else:
                        err_text = failure_msg.payload.get("error_message") if failure_msg else (agent_result.error_message or "Agent task execution failed")
                        graph.update_status(
                            task.id,
                            TaskState.FAILED,
                            error=err_text,
                        )
                        if self.journal:
                            self.journal.record(JournalEventType.TASK_FAILED, f"Task '{task.id}' failed permanently: {err_text}", task_id=task.id)

                # Checkpoint at task completion boundary
                if self.task_memory:
                    self.task_memory.save_checkpoint(graph, journal_entries=self.journal.to_dict_list() if self.journal else None)

        # 3. Aggregate results
        result = self._aggregate_results(user_request, graph)
        if self.task_memory:
            self.task_memory.save_result(result.status, result.final_response)
        if self.episodic_memory:
            ep_mem = self.episodic_memory or EpisodicMemory()
            ep_mem.record_episode(
                task_id=self.task_memory.task_id if self.task_memory else f"req_{int(time.time())}",
                goal=user_request,
                status=result.status,
                summary=result.final_response,
            )
        return result

    def resume_task(self, task_id: str, base_dir: Optional[str] = None) -> SupervisorResult:
        """
        Rehydrate and resume an interrupted task from its last authoritative checkpoint.
        Recovers interrupted tasks, verifies external side effects before retrying,
        and skips already completed tasks.
        """
        self.task_memory = TaskMemory(task_id=task_id, base_dir=base_dir)
        checkpoint = self.task_memory.load_checkpoint()
        if not checkpoint:
            return SupervisorResult(
                status="FAILED",
                user_request=task_id,
                final_response=f"Cannot resume: No valid checkpoint found for task '{task_id}'.",
                completed_tasks=[],
                failed_tasks=[],
                blocked_tasks=[],
                total_tasks=0,
            )

        graph = TaskGraph.from_dict(checkpoint["graph"])
        self._current_graph = graph
        self.journal = RecoveryJournal.from_dict_list(task_id, checkpoint.get("journal", []))
        self.journal.record(JournalEventType.CHECKPOINT_LOADED, f"Checkpoint loaded for task '{task_id}'")

        goal_data = self.task_memory.load_goal()
        user_request = goal_data.get("goal", task_id)

        # Interrupted task recovery analysis
        for task in graph.list_tasks():
            if task.status in (TaskState.RUNNING, TaskState.VERIFYING):
                self.journal.record(JournalEventType.TASK_INTERRUPTED, f"Task '{task.id}' was running during interruption", task_id=task.id)
                task.transition_to(TaskState.INTERRUPTED, force=True)

                strategy = self.recovery_engine.analyze_interrupted_task(task)
                if strategy.action == RecoveryAction.VERIFY_FIRST:
                    # Check external side effect state before repeating
                    tool_name = task.checkpoint_data.get("tool_name") or (task.required_tools[0] if task.required_tools else None)
                    tool_args = task.checkpoint_data.get("arguments", {})
                    if tool_name:
                        action_state, evidence = self.recovery_engine.verify_external_action(tool_name, tool_args)
                        if action_state == ActionState.COMPLETED:
                            # Action already happened! Mark SUCCESS without duplicate execution
                            task.verification_state = {"state": "verified_success", "evidence": evidence}
                            graph.update_status(task.id, TaskState.SUCCESS, result=evidence, force=True)
                            self.journal.record(
                                JournalEventType.EXTERNAL_STATE_VERIFIED,
                                f"Task '{task.id}' action '{tool_name}' verified as already completed: {evidence}",
                                task_id=task.id,
                            )
                        else:
                            # Not completed or unknown; reset to READY for safe execution
                            graph.update_status(task.id, TaskState.READY, force=True)
                            self.journal.record(
                                JournalEventType.RETRY_ATTEMPT,
                                f"Task '{task.id}' side effect not detected; resetting to READY",
                                task_id=task.id,
                            )
                    else:
                        graph.update_status(task.id, TaskState.READY, force=True)
                elif strategy.can_retry:
                    graph.update_status(task.id, TaskState.READY, force=True)
                    self.journal.record(JournalEventType.TASK_RESUMED, f"Task '{task.id}' resumed to READY", task_id=task.id)
                else:
                    graph.update_status(task.id, TaskState.FAILED, error="Exhausted retries during recovery", force=True)

            elif task.status == TaskState.PAUSED:
                # Re-evaluate dependencies and resume
                deps_met = all(
                    graph.get_task(dep_id) and graph.get_task(dep_id).status == TaskState.SUCCESS
                    for dep_id in task.dependencies
                )
                target = TaskState.READY if deps_met else TaskState.PENDING
                graph.update_status(task.id, target, force=True)
                self.journal.record(JournalEventType.TASK_RESUMED, f"Paused task '{task.id}' reset to {target.value}", task_id=task.id)

        # Checkpoint recovered state before resuming execution
        self.task_memory.save_checkpoint(graph, journal_entries=self.journal.to_dict_list())

        if graph.is_complete():
            return self._aggregate_results(user_request, graph)

        return self._execute_graph_loop(user_request, graph)

    def pause_task(self, task_id: Optional[str] = None) -> None:
        """Cooperatively pause current graph execution and checkpoint state."""
        if self._current_graph:
            self._current_graph.pause()
            if self.journal:
                self.journal.record(JournalEventType.TASK_PAUSED, "Task execution paused by supervisor")
            if self.task_memory:
                self.task_memory.save_checkpoint(
                    self._current_graph,
                    journal_entries=self.journal.to_dict_list() if self.journal else None,
                )

    def cancel_task(self, task_id: Optional[str] = None, reason: str = "Cancelled by user/supervisor") -> None:
        """Cancel graph execution, mark remaining tasks, and checkpoint cancellation."""
        if self._current_graph:
            self._current_graph.cancel_all(reason)
            if self.journal:
                self.journal.record(JournalEventType.TASK_CANCELLED, f"Task execution cancelled: {reason}")
            if self.task_memory:
                self.task_memory.save_checkpoint(
                    self._current_graph,
                    journal_entries=self.journal.to_dict_list() if self.journal else None,
                )

    def _decompose_and_build_graph(self, user_request: str) -> TaskGraph:
        """Decompose user request via LLM or fallback single-task graph."""
        graph = TaskGraph()

        try:
            response = self.model_provider.generate(
                prompt=f"USER REQUEST: {user_request}",
                system_prompt=SUPERVISOR_DECOMPOSITION_PROMPT,
            )
            raw_text = response.content.strip()

            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()

            decomp_data = json.loads(raw_text)
            tasks_list = decomp_data.get("tasks", [])

            if isinstance(tasks_list, list) and len(tasks_list) > 0:
                for t_data in tasks_list:
                    if isinstance(t_data, dict) and "id" in t_data and "description" in t_data:
                        task = Task(
                            id=str(t_data["id"]),
                            description=str(t_data["description"]),
                            dependencies=[str(d) for d in t_data.get("dependencies", [])],
                            required_tools=[str(rt) for rt in t_data.get("required_tools", [])],
                            max_retries=self.max_retries,
                            timeout_seconds=self.default_timeout,
                        )
                        graph.add_task(task)
                if len(graph.list_tasks()) > 0:
                    return graph
        except Exception:
            # Fallback on LLM parsing error or malformed plan: create single default task graph
            pass

        # Fallback single task graph
        fallback_task = Task(
            id="T1",
            description=user_request,
            dependencies=[],
            max_retries=self.max_retries,
            timeout_seconds=self.default_timeout,
        )
        graph.add_task(fallback_task)
        return graph

    def _build_isolated_task_prompt(self, task: Task, graph: TaskGraph) -> str:
        """
        Build an isolated prompt for a task containing ONLY its description and output results
        of direct prerequisite dependencies.
        """
        context_parts = [f"TASK DESCRIPTION: {task.description}"]

        if task.dependencies:
            prereq_info = []
            for dep_id in task.dependencies:
                dep_task = graph.get_task(dep_id)
                if dep_task and dep_task.status == TaskState.SUCCESS:
                    prereq_info.append(f"- Dependency '{dep_id}' result: {dep_task.result}")
            if prereq_info:
                context_parts.append("PREREQUISITE DEPENDENCY RESULTS:\n" + "\n".join(prereq_info))

        return "\n\n".join(context_parts)

    def _aggregate_results(self, user_request: str, graph: TaskGraph) -> SupervisorResult:
        all_tasks = graph.list_tasks()
        completed = [t.to_dict() for t in all_tasks if t.status == TaskState.SUCCESS]
        failed = [t.to_dict() for t in all_tasks if t.status == TaskState.FAILED]
        blocked = [t.to_dict() for t in all_tasks if t.status == TaskState.BLOCKED]

        if len(failed) == 0 and len(blocked) == 0:
            status = "SUCCESS"
            final_responses = [str(t.result) for t in all_tasks if t.result is not None]
            final_response = "\n\n".join(final_responses) if final_responses else "Goal executed successfully."
        elif len(completed) > 0:
            status = "PARTIAL_SUCCESS"
            final_response = f"Completed {len(completed)} of {len(all_tasks)} tasks. Failures occurred."
        else:
            status = "FAILED"
            final_response = "Request failed: required tasks failed or were blocked."

        # Rule 19 & Rule 32: Truthful verification disclaimers for indeterminate results
        indeterminate_tasks = [
            t for t in all_tasks
            if t.verification_state.get("outcome") == "INDETERMINATE"
            or "could not verify" in str(t.result or "").lower()
        ]
        if indeterminate_tasks:
            disclaimers = [
                f"I attempted {t.description}, but I could not verify whether it succeeded."
                for t in indeterminate_tasks
            ]
            final_response = (final_response + "\n\n" + "\n".join(disclaimers)).strip()

        return SupervisorResult(
            status=status,
            user_request=user_request,
            final_response=final_response,
            completed_tasks=completed,
            failed_tasks=failed,
            blocked_tasks=blocked,
            total_tasks=len(all_tasks),
        )
