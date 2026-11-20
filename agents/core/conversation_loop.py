"""Conversation Loop for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

The execution engine coordinating turn-by-turn LLM generation,
action rationale extraction (<thought>), tool execution (<tool_call>),
verbal micro-feedback triggers, and verification grounding.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.base import AgentResult, AgentStatus, AgentStep
from agents.core.prompt_builder import MaddyPromptBuilder
from agents.core.repetition_guard import clean_repetition_if_dominated, is_repetition_dominated
from agents.core.turn_state import TurnState, truncate_tool_output
from models.base import ModelMessage, ModelProvider
from tools.call import ToolCall, parse_tool_call
from tools.result import ToolResult
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MaddyConversationLoop:
    """Executes an autonomous goal using a robust turn-based state machine."""

    def __init__(
        self,
        model_provider: ModelProvider,
        executor: ToolExecutor,
        registry: ToolRegistry,
        max_steps: int = 10,
        address_title: str = "boss",
        verification_engine: Optional[Any] = None,
        on_tool_start: Optional[Callable[[str], None]] = None,
        memory_manager: Optional[Any] = None,
    ):
        self.model_provider = model_provider
        self.executor = executor
        self.registry = registry
        self.max_steps = max_steps
        self.address_title = address_title
        self.verification_engine = verification_engine
        self.on_tool_start = on_tool_start
        self.memory_manager = memory_manager
        self.prompt_builder = MaddyPromptBuilder(registry=self.registry, address_title=self.address_title)

        # Bind delegation manager for subagent execution
        try:
            from tools.delegate_tool import DelegateToolManager, bind_delegate_manager
            bind_delegate_manager(DelegateToolManager(
                model_provider=self.model_provider,
                registry=self.registry,
                executor=self.executor,
            ))
        except Exception:
            pass

    def execute(
        self,
        goal: str,
        base_system_prompt: Optional[str] = None,
        task_memory: Optional[Any] = None,
        working_memory: Optional[Any] = None,
        episodic_memory: Optional[Any] = None,
    ) -> AgentResult:
        """Runs the full conversation loop until completion or max steps reached."""
        steps: List[AgentStep] = []
        turn_state = TurnState(max_turns=self.max_steps)

        # Retrieve dynamic memory snapshots for Volatile Tier 3
        user_md = ""
        memory_md = ""
        if self.memory_manager:
            if hasattr(self.memory_manager, "get_user_md_content"):
                user_md = self.memory_manager.get_user_md_content()
            if hasattr(self.memory_manager, "get_memory_md_content"):
                memory_md = self.memory_manager.get_memory_md_content()

        system_prompt = self.prompt_builder.build_system_prompt(
            context="general",
            user_md_content=user_md,
            memory_md_content=memory_md,
        )

        if task_memory:
            task_memory.save_goal(goal)
        if working_memory:
            working_memory.goal = goal

        def _finish(res: AgentResult) -> AgentResult:
            if task_memory:
                try:
                    task_memory.save_result(res.status.name, res.final_answer)
                except Exception as me:
                    logger.debug(f"TaskMemory save_result error: {me}")
            if episodic_memory and res.status == AgentStatus.SUCCESS:
                try:
                    import time
                    t_id = getattr(task_memory, "task_id", f"task_{int(time.time())}")
                    episodic_memory.record_episode(
                        task_id=t_id,
                        goal=goal,
                        status=res.status.name,
                        summary=res.final_answer or "Success",
                    )
                except Exception as ee:
                    logger.debug(f"EpisodicMemory record_episode error: {ee}")
            return res

        messages: List[ModelMessage] = [
            ModelMessage(role="system", content=system_prompt),
            ModelMessage(role="user", content=goal),
        ]

        from agents.prompts import build_user_prompt

        for step_num in range(1, self.max_steps + 1):
            logger.info(f"[ConversationLoop] Step {step_num}/{self.max_steps} beginning turn generation.")

            # Context budget compaction
            messages = turn_state.manage_context_budget(messages)

            # Generate model completion
            try:
                response = self.model_provider.generate(
                    prompt=messages[-1].content or "",
                    system_prompt=messages[0].content or "",
                    messages=messages,
                )
                raw_output = response.content or ""
            except Exception as e:
                logger.error(f"[ConversationLoop] Model inference failed at step {step_num}: {e}")
                return _finish(AgentResult(
                    status=AgentStatus.ERROR,
                    goal=goal,
                    error=f"Model inference failed: {str(e)}",
                    steps=steps,
                    total_steps=len(steps),
                ))

            # Check for repetition loops
            if is_repetition_dominated(raw_output):
                logger.warning(f"[ConversationLoop] Step {step_num}: Repetition loop detected! Cleaning output.")
                raw_output = clean_repetition_if_dominated(raw_output)

            # Empty response nudge
            if not raw_output.strip():
                turn_state.consecutive_empty_responses += 1
                if turn_state.consecutive_empty_responses >= 3:
                    logger.warning("[ConversationLoop] 3 consecutive empty outputs; forcing graceful termination.")
                    return _finish(AgentResult(
                        status=AgentStatus.SUCCESS,
                        goal=goal,
                        final_answer=f"I am here and ready, {self.address_title}. How may I assist you?",
                        steps=steps,
                        total_steps=len(steps),
                    ))
                messages.append(ModelMessage(
                    role="user",
                    content="Please state your thought and next action, or your final answer if complete.",
                ))
                continue
            else:
                turn_state.consecutive_empty_responses = 0

            # Parse thought, tool calls, and potential direct answer
            thought, tool_calls, final_answer = self._parse_model_turn(raw_output)

            # Append assistant's turn
            messages.append(ModelMessage(role="assistant", content=raw_output))

            # Case A: Model emitted tool calls
            if tool_calls:
                for tc in tool_calls:
                    logger.info(f"[ConversationLoop] Step {step_num}: Executing tool '{tc.name}'")

                    # Runaway tool-call detection
                    if not turn_state.record_tool_call(tc.name, tc.arguments):
                        error_msg = f"Repeated execution loop intercepted: {tc.name} was called multiple times with identical arguments. Please adjust your parameters or synthesize your final response."
                        logger.warning(f"[ConversationLoop] Runaway loop blocked for tool '{tc.name}'.")
                        t_res = ToolResult.fail(
                            error=error_msg,
                            data={"blocked": True, "reason": "repeated_call_loop"},
                        )
                    else:
                        if self.on_tool_start:
                            try:
                                self.on_tool_start(tc.name)
                            except Exception as te:
                                logger.debug(f"on_tool_start callback error: {te}")

                        try:
                            t_res = self.executor.execute(tc)
                        except Exception as e:
                            logger.error(f"[ConversationLoop] Tool execution raised exception: {e}")
                            t_res = ToolResult.fail(
                                error_type="EXECUTION_ERROR",
                                message=str(e),
                                tool_name=tc.name,
                            )

                    # Verification grounding
                    if self.verification_engine:
                        try:
                            v_report = self.verification_engine.verify_step(
                                step=AgentStep(
                                    step_number=step_num,
                                    thought=thought,
                                    tool_call=tc,
                                    tool_result=t_res,
                                ),
                                goal=goal,
                            )
                            if not v_report.passed:
                                logger.warning(f"[ConversationLoop] Verification warning: {v_report.message}")
                        except Exception as ve:
                            logger.debug(f"[ConversationLoop] Verification step skipped: {ve}")

                    # Truncate overly verbose tool output
                    raw_result_str = json.dumps(t_res.data if t_res.data is not None else t_res.error)
                    truncated_result_str = truncate_tool_output(raw_result_str)

                    # Append tool observation
                    obs_content = f"Observation: <tool_response>\n{{\"name\": \"{tc.name}\", \"content\": {truncated_result_str}}}\n</tool_response>"
                    messages.append(ModelMessage(role="tool", content=obs_content))

                    steps.append(AgentStep(
                        step_number=step_num,
                        thought=thought,
                        tool_call=tc,
                        tool_result=t_res,
                    ))

                # After executing tool calls, continue loop to let model observe results
                continue

            # Case B: Model returned final answer (no tools)
            if final_answer is not None or not tool_calls:
                ans = final_answer if final_answer is not None else self._clean_raw_text(raw_output)
                if not ans.strip():
                    ans = "Task completed."

                steps.append(AgentStep(
                    step_number=step_num,
                    thought=thought,
                    final_answer=ans,
                ))

                logger.info(f"[ConversationLoop] Task resolved successfully at step {step_num}.")
                return _finish(AgentResult(
                    status=AgentStatus.SUCCESS,
                    goal=goal,
                    final_answer=ans,
                    steps=steps,
                    total_steps=len(steps),
                ))

        logger.warning(f"[ConversationLoop] Exceeded maximum allowed steps ({self.max_steps}).")
        return _finish(AgentResult(
            status=AgentStatus.MAX_STEPS,
            goal=goal,
            final_answer=None,
            steps=steps,
            total_steps=len(steps),
        ))

    def _parse_model_turn(
        self,
        raw: str,
    ) -> Tuple[Optional[str], List[ToolCall], Optional[str]]:
        """Parses model generation for thoughts, tool calls, and final answers.

        Supports:
        - <thought>...</thought> and <scratch_pad>...</scratch_pad>
        - <tool_call>...</tool_call> and <function_call>...</function_call>
        - Direct final text response without tool calls.
        """
        thought: Optional[str] = None
        tool_calls: List[ToolCall] = []

        # 1. Thought extraction
        thought_match = re.search(r"<(?:thought|scratch_pad)>(.*?)</(?:thought|scratch_pad)>", raw, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            # Check for unclosed thought tag at beginning
            unclosed_thought = re.match(r"^\s*<(?:thought|scratch_pad)>(.*)", raw, re.DOTALL | re.IGNORECASE)
            if unclosed_thought and "<tool_call>" in unclosed_thought.group(1):
                thought = unclosed_thought.group(1).split("<tool_call>")[0].strip()

        # 2. Tool calls extraction (<tool_call> or <function_call>)
        tc_matches = list(re.finditer(r"<(?:tool_call|function_call)>(.*?)(?:</(?:tool_call|function_call)>|$)", raw, re.DOTALL | re.IGNORECASE))
        for m in tc_matches:
            block = m.group(1).strip()
            if not block:
                continue
            try:
                tc = parse_tool_call(block)
                if tc:
                    tool_calls.append(tc)
            except Exception:
                # Fallback to direct json dict parsing
                try:
                    data = json.loads(block)
                    name = data.get("name") or data.get("tool") or (data.get("function", {}).get("name") if isinstance(data.get("function"), dict) else None)
                    args = data.get("arguments") or data.get("parameters") or (data.get("function", {}).get("arguments") if isinstance(data.get("function"), dict) else {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    if name:
                        tool_calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
                except Exception:
                    pass

        # 2b. Direct JSON dictionary format parsing (e.g. {"thought": "...", "tool_call": {...}})
        if not tool_calls:
            trimmed = raw.strip()
            if trimmed.startswith("{") and trimmed.endswith("}"):
                try:
                    data = json.loads(trimmed)
                    if isinstance(data, dict):
                        if not thought:
                            thought = data.get("thought") or data.get("scratchpad")
                        raw_tc = data.get("tool_call") or data.get("tool_calls") or data.get("action")
                        if raw_tc:
                            if isinstance(raw_tc, dict):
                                t_name = raw_tc.get("name") or raw_tc.get("tool")
                                t_args = raw_tc.get("arguments") or raw_tc.get("parameters") or {}
                                if t_name:
                                    tool_calls.append(ToolCall(name=t_name, arguments=t_args if isinstance(t_args, dict) else {}))
                            elif isinstance(raw_tc, list):
                                for tc_item in raw_tc:
                                    if isinstance(tc_item, dict):
                                        t_name = tc_item.get("name") or tc_item.get("tool")
                                        t_args = tc_item.get("arguments") or tc_item.get("parameters") or {}
                                        if t_name:
                                            tool_calls.append(ToolCall(name=t_name, arguments=t_args if isinstance(t_args, dict) else {}))
                except Exception:
                    pass

        if tool_calls:
            return thought, tool_calls, None

        # 3. Final answer extraction
        # Check for explicit json wrapped final answer
        trimmed = raw.strip()
        if trimmed.startswith("{") and trimmed.endswith("}"):
            try:
                data = json.loads(trimmed)
                fa = (
                    data.get("final_answer")
                    or data.get("finalanswer")
                    or data.get("answer")
                    or data.get("response")
                )
                if fa is not None:
                    return thought, [], str(fa)
            except Exception:
                pass

        final_text = self._clean_raw_text(raw)
        if final_text:
            return thought, [], final_text

        return thought, [], None

    @staticmethod
    def _clean_raw_text(text: str) -> str:
        """Strips tags to isolate user-facing response."""
        cleaned = re.sub(r"<(?:thought|scratch_pad)>.*?</(?:thought|scratch_pad)>", "", text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<(?:tool_call|function_call)>.*?(?:</(?:tool_call|function_call)>|$)", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<tool_response>.*?</tool_response>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<tools>.*?</tools>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()
