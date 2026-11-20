"""Dynamic Agent Execution Loop for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You).

Core execution model:
Plan → Act → Observe → Verify → Replan

Fulfills CLAUDE.md Engineering Rules 4, 5, 6, 7, 8, 9, 11, 13.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from core.budget import IterationBudget
from core.errors import (
    BudgetExceededError,
    DegenerateLoopError,
    MaddyError,
    ModelApiError,
    PermissionDeniedError,
)
from core.llm_client import CloudLLMClient, ToolCallEnvelope
from core.loop_guard import LoopGuard
from core.memory import MemoryStore, WorkingMemory
from core.memory_extractor import ConversationMemoryExtractor
from core.permissions import PermissionGate, PermissionTier
from core.redact import redact_structure, redact_text
from core.skills.executor import SkillExecutor
from core.skills.registry import SkillRegistry
from core.tools.base import BaseTool, ToolRegistry, ToolResult
from core.tools.memory_tools import ForgetTool, RecallTool, RememberTool, UpdatePreferenceTool
from core.tools.skill_tools import ListSkillsTool, LoadSkillTool, SearchSkillsTool, ValidateSkillTool
from core.toolsets import resolve_active_tool_names
from core.trace_store import TraceStore

logger = logging.getLogger(__name__)


def truncate_tool_output(
    content: str,
    max_chars: int = 1500,
    head_chars: int = 1000,
    tail_chars: int = 400,
) -> str:
    """Truncates verbose tool output while preserving head and tail context.
    Prevents context blowup and token exhaustion.
    """
    if not isinstance(content, str):
        content = str(content)

    if len(content) <= max_chars:
        return content

    omitted = len(content) - (head_chars + tail_chars)
    if omitted <= 0:
        return content[:max_chars]

    head = content[:head_chars].rstrip()
    tail = content[-tail_chars:].lstrip()
    return f"{head}\n\n... [Output truncated: {omitted:,} characters omitted to preserve token budget] ...\n\n{tail}"


def build_temporal_context() -> str:
    """Construct byte-stable host date and timezone context block.
    Enforces prompt caching invariants by maintaining a byte-stable system prompt
    prefix across the calendar day, avoiding token recomputation and cache churn,
    while delegating live clock queries to the system_info tool.
    """
    now = datetime.datetime.now().astimezone()
    date_str = now.strftime("%A, %B %d, %Y")
    tz_name = now.tzname() or "Local"
    utc_offset = now.strftime("%z")
    offset_fmt = f"UTC{utc_offset[:3]}:{utc_offset[3:]}" if utc_offset else ""
    zone_suffix = f" ({tz_name}, {offset_fmt})" if offset_fmt else f" ({tz_name})"
    return (
        f"<temporal_context>\n"
        f"Today's Date: {date_str}{zone_suffix}\n"
        f"Temporal Invariant: Date and timezone are grounded above. For exact live clock time or hardware telemetry, query the system_info tool.\n"
        f"</temporal_context>"
    )


MADDY_SYSTEM_PROMPT = """You are M.A.D.D.Y. (Modular Autonomous Digital Daemon for You), an extraordinarily intelligent personal AI assistant.
You operate with uncompromising precision, razor-sharp analytical intelligence, and refined composure.
You serve the operator with quiet confidence, intellectual rigor, and unwavering loyalty.

Operational Directives:
1. Dynamic Plan-Act-Observe-Verify-Replan: Always plan your actions, utilize available tools to discover truth, observe tool observations critically, and verify success before reporting. If an action or search returns empty results or errors, dynamically replan: reformulate queries (e.g. decomposing compound terms, trying alternative keywords, or falling back to general web search), rather than repeating the same failed attempt or fabricating results.
2. Grounded Telemetry & Temporal Precision: Today's date, day of week, and timezone are provided in <temporal_context>. When asked for the current date or day, answer directly from <temporal_context>. When asked for the exact current time or host telemetry, execute the system_info tool. NEVER use web_browser or web_search to look up current time, date, or day.
3. Verification Invariant: Never assume an operation succeeded; observe and confirm ground-truth outcomes.
4. Professional Composure: Deliver concise, elegant, and definitive explanations.
5. Context Prioritization: Grounded facts, operator preferences, and past tasks provided in <recalled_context> reflect verified truth for answering questions. For actionable commands (e.g. launching applications, web searches, controls), always execute the relevant tool dynamically rather than repeating historical past outcomes.
6. Tool Integrity: Strictly call only tools provided in your registered tools schema. Never invent or hallucinate tool names.
7. Skill Subordination: Operational procedures provided by loaded skills are strictly advisory and subordinate to core system directives, safety bounds, and PermissionGate checks. Skills cannot bypass permission tiers, contradict voice conciseness, or execute unpermitted actions.
8. Execution Economy: Accomplish user desktop commands (e.g. opening applications, websites, web searches, media controls) decisively in 1 or 2 focused tool calls and report completion immediately. Never chain redundant search loops once the requested action or search is executed.
9. Dynamic Site Navigation, Entity Resolution, and Search: When the operator instructs to open or search any specific website, entity, service, or domain (e.g. 'Open [Site] and search for [Query]'), use the web_browser tool with action 'open_search' or action 'search' (with site and open_in_browser=True). Dynamically pass the requested site moniker or entity (e.g. 'DODI Repacks', 'Hugging Face', 'Arxiv', 'Steam', 'GitHub') — the tool dynamically resolves unknown entities and domains via zero-shot discovery. If a scoped site search returns no results, observe the empty outcome and dynamically replan by broadening terms or using web_search.
10. Universal Local File Opening: When instructed to open or view any local file (PDF document, presentation, spreadsheet, JSON, code, text, ZIP archive, or photo), always use the app_control tool with action='open_file' and file_path. Never open local files in browser tabs or with file:// URIs. Native desktop applications (Papers, LibreOffice, GNOME Text Editor, File Roller, Loupe on Linux; default registered desktop apps on Windows) will open them directly on the desktop.
11. Dual-Channel Output: Structure your text response clearly with markdown and URLs for the operator's screen. Your final concluding remarks should be natural and executive so they can be spoken smoothly without technical clutter.
12. Dynamic Operator Grounding & Strict Address Rule: Operator profile, full name, and details are grounded dynamically in memory context (<recalled_context>). Always address the operator strictly as 'boss' or 'sir' (or active preferred address), even when directly stating or answering queries regarding their name (e.g. 'Your name is <Operator Name>, boss.'). Never address the operator by their first or full name alone. Answer questions regarding their identity warmly, respectfully, and accurately using dynamic memory context. When the operator spells out letters or corrects speech recognition, immediately acknowledge the corrected term with quiet courtesy, apologize briefly for the slip, and execute the requested query. Never claim to be a text-based model, assert that you lack ears/hearing, or argue about audio transcription.
"""


class MaddyAgentCore:
    """The central dynamic agent execution engine for M.A.D.D.Y.
    (Modular Autonomous Digital Daemon for You).
    """

    def __init__(
        self,
        llm_client: Optional[CloudLLMClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        trace_store: Optional[TraceStore] = None,
        permission_gate: Optional[PermissionGate] = None,
        memory_store: Optional[MemoryStore] = None,
        memory_extractor: Optional[ConversationMemoryExtractor] = None,
        skill_registry: Optional[SkillRegistry] = None,
        skill_executor: Optional[SkillExecutor] = None,
        system_prompt: Optional[str] = None,
        personality: Optional[Any] = None,
        max_iterations_per_turn: int = 6,
        tool_timeout_seconds: float = 10.0,
        current_depth: int = 1,
        max_spawn_depth: int = 3,
        enable_delegation: bool = True,
    ):
        self.llm_client = llm_client or CloudLLMClient()
        self.tool_registry = tool_registry or ToolRegistry()
        self.trace_store = trace_store or TraceStore()
        self.permission_gate = permission_gate or PermissionGate()
        self.memory_store = memory_store or MemoryStore()
        self.memory_extractor = memory_extractor or ConversationMemoryExtractor(
            memory_store=self.memory_store,
            llm_client=None,
        )
        self.skill_registry = skill_registry or SkillRegistry()
        self.skill_executor = skill_executor or SkillExecutor(
            registry=self.skill_registry,
            tool_registry=self.tool_registry,
            permission_gate=self.permission_gate,
        )
        self._custom_system_prompt = system_prompt
        self.personality = personality
        self.max_iterations_per_turn = max_iterations_per_turn
        self.tool_timeout_seconds = tool_timeout_seconds
        self.current_depth = current_depth
        self.max_spawn_depth = max_spawn_depth
        self.enable_delegation = enable_delegation

        # Auto-register Phase 4/3.8 native memory tools if not already present
        if not self.tool_registry.get_tool("remember"):
            self.tool_registry.register(RememberTool(memory_store=self.memory_store))
        if not self.tool_registry.get_tool("recall"):
            self.tool_registry.register(RecallTool(memory_store=self.memory_store))
        if not self.tool_registry.get_tool("forget"):
            self.tool_registry.register(ForgetTool(memory_store=self.memory_store))
        if not self.tool_registry.get_tool("update_preference"):
            self.tool_registry.register(UpdatePreferenceTool(memory_store=self.memory_store))

        # Auto-register Phase 5 native skill tools if not already present
        if not self.tool_registry.get_tool("list_skills"):
            self.tool_registry.register(ListSkillsTool(skill_registry=self.skill_registry))
        if not self.tool_registry.get_tool("search_skills"):
            self.tool_registry.register(SearchSkillsTool(skill_registry=self.skill_registry))
        if not self.tool_registry.get_tool("load_skill"):
            self.tool_registry.register(LoadSkillTool(skill_executor=self.skill_executor))
        if not self.tool_registry.get_tool("validate_skill"):
            self.tool_registry.register(ValidateSkillTool())

        # Auto-register Phase 6 delegation tool if allowed by recursion depth
        if self.enable_delegation and self.current_depth < self.max_spawn_depth:
            if not self.tool_registry.get_tool("delegate_task"):
                from core.tools.delegate_tool import DelegateTool
                self.tool_registry.register(
                    DelegateTool(
                        llm_client=self.llm_client,
                        tool_registry=self.tool_registry,
                        permission_gate=self.permission_gate,
                        trace_store=self.trace_store,
                        current_depth=self.current_depth,
                    )
                )

        # Auto-register Phase 7 native background autonomy/cron tool
        if not self.tool_registry.get_tool("cronjob_manage"):
            from core.tools.cron_tool import CronJobManageTool
            self.tool_registry.register(CronJobManageTool())

        try:
            from verification.engine import VerificationEngine
            self.verification_engine: Optional[Any] = VerificationEngine()
        except Exception:
            self.verification_engine = None

        self.working_memory: Optional[WorkingMemory] = None

    @property
    def system_prompt(self) -> str:
        """
        Dynamically constructs the full system prompt. If a personality engine is attached,
        personality and operator addressing directives are dynamically sourced from PersonalityEngine,
        keeping the agent core decoupled from hardcoded persona logic.
        """
        if self._custom_system_prompt is not None:
            return self._custom_system_prompt
        if self.personality is not None and hasattr(self.personality, "get_llm_system_prompt"):
            personality_prompt = self.personality.get_llm_system_prompt()
            return f"{personality_prompt}\n\n{MADDY_SYSTEM_PROMPT}"
        return MADDY_SYSTEM_PROMPT

    @system_prompt.setter
    def system_prompt(self, val: str) -> None:
        self._custom_system_prompt = val

    def get_relevant_tool_schemas(
        self,
        user_prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dynamically filter registered tool schemas based on intent toolset gating.
        Maintains a narrow core waist (_MADDY_CORE_TOOLS) to keep base payload under 900 tokens
        (preventing Groq HTTP 413 / 7000 TPM limit breaches) while dynamically activating domain
        toolsets on demand. Injected custom tools are always preserved.
        """
        all_schemas = {
            s["function"]["name"]: s
            for s in self.tool_registry.get_openai_tool_schemas()
        }

        active_names = resolve_active_tool_names(
            user_prompt=user_prompt,
            registered_tools=set(all_schemas.keys()),
            conversation_history=conversation_history,
        )

        return [all_schemas[name] for name in active_names if name in all_schemas]

    def run_turn(
        self,
        user_prompt: str,
        task_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        stop_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute one complete conversational turn following Plan → Act → Observe → Verify → Replan."""
        active_task_id = self.trace_store.create_task(
            description=user_prompt,
            metadata={"system_prompt_length": len(self.system_prompt)},
            task_id=task_id,
        )

        budget = IterationBudget(
            max_iterations=self.max_iterations_per_turn,
            max_tool_timeout_seconds=self.tool_timeout_seconds,
        )
        loop_guard = LoopGuard()
        self.working_memory = WorkingMemory(goal=user_prompt)
        if self.skill_executor:
            self.skill_executor.reset_turn_budget()

        # Build prompt incorporating Tier 1 Core Memory and 5-layer bounded context
        core_block = self.memory_store.get_core_memory_block() if self.memory_store else ""
        recalled_context = (
            self.memory_store.retrieve_relevant_context(user_prompt, working_memory=self.working_memory)
            if self.memory_store
            else ""
        )

        temporal_block = build_temporal_context()
        full_system_prompt = f"{self.system_prompt}\n\n{temporal_block}"
        if core_block:
            full_system_prompt += f"\n\n{core_block}"
        if recalled_context:
            full_system_prompt += f"\n\n{recalled_context}"

        # Phase 5: Inject prompt-cache preserving skills catalog if skills exist (sparse gating when > 15)
        if self.skill_registry and self.skill_registry.count() > 0:
            skills_catalog = self.skill_registry.generate_prompt_catalog(user_query=user_prompt)
            if skills_catalog:
                full_system_prompt += f"\n\n{skills_catalog}"

        # Build message history
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": full_system_prompt}
        ]
        if conversation_history:
            messages.extend(conversation_history)

        messages.append({"role": "user", "content": user_prompt})
        self.trace_store.append_message(active_task_id, role="user", content=user_prompt)

        final_response: str = ""
        completed = False
        step_index = 0
        had_tool_failures = False

        try:
            while budget.remaining > 0:
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    final_response = "Execution interrupted by operator."
                    self.trace_store.record_event(
                        active_task_id, "interrupted", {"step": step_index}
                    )
                    break

                iteration_step = budget.consume()
                step_index += 1

                # ── 1. PLAN / MODEL INVOCATION ────────────────────────────────────
                tool_schemas = self.get_relevant_tool_schemas(
                    user_prompt, conversation_history=conversation_history
                )
                model_resp = self.llm_client.chat_completion(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )

                # Record step in trace store
                thought_preview = model_resp.content or f"Invoking {len(model_resp.tool_calls)} tool(s)"
                step_id = self.trace_store.record_step(
                    task_id=active_task_id,
                    step_index=step_index,
                    thought=thought_preview,
                    action_type="tool_round" if model_resp.tool_calls else "final_response",
                )

                # If no tool calls, model has produced final response
                if not model_resp.tool_calls:
                    final_response = model_resp.content or ""
                    if final_response:
                        loop_guard.check_model_text(final_response)

                    # Enforce Rule 16 (Never Trust Model Output) and Rule 32 (No Fake Completion)
                    is_empty = not final_response.strip()
                    if is_empty or had_tool_failures:
                        completed = False
                    else:
                        completed = True

                    # Append assistant message
                    messages.append({"role": "assistant", "content": final_response})
                    self.trace_store.append_message(active_task_id, role="assistant", content=final_response)
                    break

                if model_resp.content:
                    loop_guard.check_model_text(model_resp.content)

                # ── 2. PREPARE ASSISTANT TOOL MESSAGE ─────────────────────────────
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": model_resp.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in model_resp.tool_calls
                    ],
                }
                messages.append(assistant_tool_msg)
                self.trace_store.append_message(
                    active_task_id,
                    role="assistant",
                    content=model_resp.content,
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in model_resp.tool_calls
                    ],
                )

                # ── 3. EXECUTE TOOLS (PERSIST-BEFORE-EXECUTE) ─────────────────────
                for tc in model_resp.tool_calls:
                    tool_obj = self.tool_registry.get_tool(tc.name)
                    tier = (
                        tool_obj.operation_tier(tc.arguments)
                        if tool_obj
                        else PermissionTier.SYSTEM_IMPACTING
                    )

                    # a. PERSIST-BEFORE-EXECUTE INVARIANT
                    self.trace_store.record_tool_call_planned(
                        task_id=active_task_id,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        tier=tier.value,
                        step_id=step_id,
                    )

                    tool_result_content: str = ""
                    execution_status = "completed"
                    error_detail: Optional[str] = None

                    try:
                        # b. LOOP GUARD CHECK
                        loop_guard.record_and_check(tc.name, tc.arguments)

                        # c. DETERMINISTIC PERMISSION GATE CHECK
                        self.permission_gate.check_permission(tc.name, tier, tc.arguments)

                        # d. ACT: EXECUTE TOOL
                        result: ToolResult = self.tool_registry.execute_tool(
                            name=tc.name,
                            arguments=tc.arguments,
                            timeout_seconds=self.tool_timeout_seconds,
                        )

                        if result.success:
                            # Verification grounding (Rule 16 & Rule 32)
                            if self.verification_engine:
                                try:
                                    from tools.call import ToolCall as VToolCall
                                    v_tc = VToolCall(call_id=tc.id, name=tc.name, arguments=tc.arguments)
                                    v_res, v_report = self.verification_engine.verify_tool_execution(v_tc, result)
                                    if not v_res.success:
                                        execution_status = "failed"
                                        had_tool_failures = True
                                        error_detail = v_res.error.get("message") if isinstance(v_res.error, dict) else str(v_res.error)
                                        tool_result_content = f"Verification Failure: {getattr(v_report, 'evidence_summary', error_detail)}"
                                except Exception as ve:
                                    logger.debug("Verification check skipped: %s", ve)

                            if execution_status != "failed":
                                tool_result_content = result.output
                                # Mutating success resets loop streak
                                if tier == PermissionTier.SYSTEM_IMPACTING:
                                    loop_guard.reset_streak_on_progress()
                        else:
                            execution_status = "failed"
                            had_tool_failures = True
                            error_detail = result.error
                            tool_result_content = f"Tool Error: {result.error}\nOutput: {result.output}".strip()

                    except DegenerateLoopError as dle:
                        execution_status = "blocked"
                        had_tool_failures = True
                        error_detail = dle.message
                        tool_result_content = f"Guardrail Blocked: {dle.message}"
                    except PermissionDeniedError as pde:
                        execution_status = "blocked"
                        had_tool_failures = True
                        error_detail = pde.message
                        tool_result_content = f"Permission Denied: {pde.message}"
                    except Exception as exc:
                        execution_status = "failed"
                        had_tool_failures = True
                        error_detail = str(exc)
                        tool_result_content = f"System Error: {str(exc)}"

                    # e. PERSIST EXECUTION RESULT IN TRACE STORE
                    self.trace_store.record_tool_call_completed(
                        tool_call_id=tc.id,
                        result_output=tool_result_content,
                        error_info=error_detail,
                        status=execution_status,
                    )

                    # f. OBSERVE: STRICT ROLE-ALTERNATION TOOL MESSAGE
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": redact_text(truncate_tool_output(tool_result_content)),
                    }
                    messages.append(tool_msg)
                    self.trace_store.append_message(
                        active_task_id,
                        role="tool",
                        content=tool_result_content,
                        tool_call_id=tc.id,
                    )
                    if self.working_memory:
                        self.working_memory.add_observation(
                            step=step_index,
                            tool=tc.name,
                            output=tool_result_content,
                            success=(execution_status == "completed"),
                        )

                # ── 3.5 CHECKPOINT STATE SNAPSHOT ─────────────────────────────────
                self.trace_store.save_checkpoint(
                    task_id=active_task_id,
                    step_index=step_index,
                    state_snapshot={
                        "task_id": active_task_id,
                        "step_index": step_index,
                        "messages": messages,
                        "budget_remaining": budget.remaining,
                        "system_prompt": self.system_prompt,
                        "loop_guard": loop_guard.to_dict(),
                        "working_memory": self.working_memory.to_dict() if self.working_memory else {},
                    },
                )

                # ── 4. VERIFY & REPLAN LOOP CONTINUES ─────────────────────────────

        except DegenerateLoopError as dle:
            final_response = f"M.A.D.D.Y. Loop Guard Alert: {dle.message}"
            self.trace_store.record_event(
                active_task_id,
                "degenerate_loop_halted",
                {"message": dle.message, "pattern": dle.pattern},
            )
        except BudgetExceededError as bee:
            final_response = f"M.A.D.D.Y. Resource Alert: {bee.message}"
            self.trace_store.record_event(
                active_task_id,
                "budget_exhausted",
                {"used": bee.error_context.get("used_iterations")},
            )
        except ModelApiError as mae:
            logger.error("Model API error in MaddyAgentCore: %s", mae)
            addr = self.personality.get_address_title() if hasattr(self.personality, "get_address_title") else "boss"
            low_msg = str(mae).lower()
            if "unconfigured" in low_msg or "missing groq_api_key" in low_msg or "missing api key" in low_msg:
                final_response = f"My cloud LLM credentials are unconfigured, {addr}. Please configure your GROQ_API_KEY in the .env file."
            elif mae.status_code == 429:
                final_response = f"My neural connection is temporarily experiencing high traffic with the cloud provider, {addr}. Please give me just a moment and try again."
            else:
                final_response = f"I encountered a disruption communicating with my cloud neural core, {addr}. Please try again in a moment."
            self.trace_store.record_event(
                active_task_id,
                "model_api_error",
                {"status_code": mae.status_code, "message": str(mae)},
            )
        except Exception as exc:
            logger.exception("Fatal turn error in MaddyAgentCore")
            err_msg = str(exc)
            if "429" in err_msg or "rate limit" in err_msg.lower():
                final_response = "My neural connection is temporarily experiencing high traffic, boss. Please give me just a moment."
            else:
                final_response = f"I experienced an unexpected internal disruption while processing that request: {type(exc).__name__}."
            self.trace_store.record_event(
                active_task_id,
                "fatal_error",
                {"error": str(exc)},
            )
        finally:
            self.trace_store.update_task_status(
                active_task_id,
                status="completed" if completed else "failed",
            )
            # Record completed episode to Layer 3 Episodic Memory
            if self.memory_store and completed and hasattr(self.memory_store, "episodic"):
                try:
                    tool_calls = self.trace_store.get_task_tool_calls(active_task_id)
                    if tool_calls or len(user_prompt) > 15:
                        tools_used = [tc.get("tool_name", "") for tc in tool_calls]
                        self.memory_store.episodic.record_episode(
                            task_id=active_task_id,
                            goal=user_prompt,
                            summary=final_response[:250],
                            status="completed" if completed else "failed",
                            tools_used=tools_used,
                        )
                except Exception as ep_exc:
                    logger.debug("Episodic recording skipped: %s", ep_exc)

            # Autonomous Conversational Memory Extraction & Invalidation
            if self.memory_store and completed and hasattr(self, "memory_extractor") and self.memory_extractor:
                try:
                    tool_calls = self.trace_store.get_task_tool_calls(active_task_id)
                    extracted = self.memory_extractor.extract_and_sync(
                        user_prompt=user_prompt,
                        assistant_response=final_response,
                        tool_calls=tool_calls,
                        provenance_source="operator",
                    )
                    if extracted:
                        self.trace_store.record_event(
                            active_task_id,
                            "conversational_memory_extracted",
                            {"extracted_count": len(extracted), "facts": extracted},
                        )
                except Exception as ext_exc:
                    logger.debug("Conversational memory extraction skipped: %s", ext_exc)

        # Personality formatting (sanitize cliches/markdown and apply tone policy)
        if self.personality and hasattr(self.personality, "format_response") and final_response:
            final_response = self.personality.format_response(final_response)

        return {
            "task_id": active_task_id,
            "response": final_response,
            "completed": completed,
            "steps_taken": step_index,
            "messages": messages,
        }

    def resume_turn(
        self,
        task_id: str,
        checkpoint_id: Optional[str] = None,
        stop_event: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Resume an interrupted or paused turn from the most recent or specified checkpoint.

        Fulfills CLAUDE.md Phase 3 recovery and task resumption invariant.
        Restores conversational messages, iteration budget, and loop guard state,
        then continues Plan → Act → Observe → Verify → Replan without re-executing
        previously completed tool calls.
        """
        if checkpoint_id:
            checkpoint = self.trace_store.get_checkpoint(checkpoint_id)
        else:
            checkpoint = self.trace_store.get_latest_checkpoint(task_id)

        if not checkpoint:
            raise MaddyError(
                message=f"Cannot resume task '{task_id}': no valid checkpoint discovered.",
                details={"task_id": task_id, "checkpoint_id": checkpoint_id},
            )

        snapshot = checkpoint.get("state_snapshot", {})
        messages = snapshot.get("messages", [])
        step_index = snapshot.get("step_index", 0)
        saved_budget_remaining = snapshot.get("budget_remaining", self.max_iterations_per_turn)
        saved_wm = snapshot.get("working_memory")
        self.working_memory = WorkingMemory.from_dict(saved_wm) if saved_wm else WorkingMemory(goal=task_id)

        # Record resume event
        self.trace_store.record_event(
            task_id,
            "task_resumed",
            {
                "checkpoint_id": checkpoint.get("checkpoint_id"),
                "resumed_from_step": step_index,
                "budget_remaining": saved_budget_remaining,
            },
        )
        self.trace_store.update_task_status(task_id, "running")

        budget = IterationBudget(
            max_iterations=max(1, saved_budget_remaining),
            max_tool_timeout_seconds=self.tool_timeout_seconds,
        )
        saved_lg = snapshot.get("loop_guard")
        loop_guard = LoopGuard.from_dict(saved_lg) if saved_lg else LoopGuard()

        final_response: str = ""
        completed = False
        had_tool_failures = False

        try:
            while budget.remaining > 0:
                if stop_event is not None and getattr(stop_event, "is_set", lambda: False)():
                    final_response = "Execution interrupted by operator."
                    self.trace_store.record_event(task_id, "interrupted", {"step": step_index})
                    break

                iteration_step = budget.consume()
                step_index += 1

                # ── 1. PLAN / MODEL INVOCATION ────────────────────────────────────
                resume_prompt = self.working_memory.goal if self.working_memory else ""
                tool_schemas = self.get_relevant_tool_schemas(resume_prompt)
                model_resp = self.llm_client.chat_completion(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )

                thought_preview = model_resp.content or f"Invoking {len(model_resp.tool_calls)} tool(s)"
                step_id = self.trace_store.record_step(
                    task_id=task_id,
                    step_index=step_index,
                    thought=thought_preview,
                    action_type="tool_round" if model_resp.tool_calls else "final_response",
                )

                if not model_resp.tool_calls:
                    final_response = model_resp.content or ""
                    is_empty = not final_response.strip()
                    if is_empty or had_tool_failures:
                        completed = False
                    else:
                        completed = True
                    messages.append({"role": "assistant", "content": final_response})
                    self.trace_store.append_message(task_id, role="assistant", content=final_response)
                    break

                # ── 2. PREPARE ASSISTANT TOOL MESSAGE ─────────────────────────────
                assistant_tool_msg = {
                    "role": "assistant",
                    "content": model_resp.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for tc in model_resp.tool_calls
                    ],
                }
                messages.append(assistant_tool_msg)
                self.trace_store.append_message(
                    task_id,
                    role="assistant",
                    content=model_resp.content,
                    tool_calls=[
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in model_resp.tool_calls
                    ],
                )

                # ── 3. EXECUTE TOOLS (PERSIST-BEFORE-EXECUTE) ─────────────────────
                for tc in model_resp.tool_calls:
                    tool_obj = self.tool_registry.get_tool(tc.name)
                    tier = (
                        tool_obj.operation_tier(tc.arguments)
                        if tool_obj
                        else PermissionTier.SYSTEM_IMPACTING
                    )

                    self.trace_store.record_tool_call_planned(
                        task_id=task_id,
                        tool_call_id=tc.id,
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        tier=tier.value,
                        step_id=step_id,
                    )

                    tool_result_content: str = ""
                    execution_status = "completed"
                    error_detail: Optional[str] = None

                    try:
                        loop_guard.record_and_check(tc.name, tc.arguments)
                        self.permission_gate.check_permission(tc.name, tier, tc.arguments)

                        result: ToolResult = self.tool_registry.execute_tool(
                            name=tc.name,
                            arguments=tc.arguments,
                            timeout_seconds=self.tool_timeout_seconds,
                        )

                        if result.success:
                            tool_result_content = result.output
                            if tier == PermissionTier.SYSTEM_IMPACTING:
                                loop_guard.reset_streak_on_progress()
                        else:
                            execution_status = "failed"
                            had_tool_failures = True
                            error_detail = result.error
                            tool_result_content = f"Tool Error: {result.error}\nOutput: {result.output}".strip()

                    except DegenerateLoopError as dle:
                        execution_status = "blocked"
                        had_tool_failures = True
                        error_detail = dle.message
                        tool_result_content = f"Guardrail Blocked: {dle.message}"
                    except PermissionDeniedError as pde:
                        execution_status = "blocked"
                        had_tool_failures = True
                        error_detail = pde.message
                        tool_result_content = f"Permission Denied: {pde.message}"
                    except Exception as exc:
                        execution_status = "failed"
                        had_tool_failures = True
                        error_detail = str(exc)
                        tool_result_content = f"System Error: {str(exc)}"

                    self.trace_store.record_tool_call_completed(
                        tool_call_id=tc.id,
                        result_output=tool_result_content,
                        error_info=error_detail,
                        status=execution_status,
                    )

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": redact_text(truncate_tool_output(tool_result_content)),
                    }
                    messages.append(tool_msg)
                    self.trace_store.append_message(
                        task_id,
                        role="tool",
                        content=tool_result_content,
                        tool_call_id=tc.id,
                    )
                    if self.working_memory:
                        self.working_memory.add_observation(
                            step=step_index,
                            tool=tc.name,
                            output=tool_result_content,
                            success=(execution_status == "completed"),
                        )

                # Checkpoint step
                self.trace_store.save_checkpoint(
                    task_id=task_id,
                    step_index=step_index,
                    state_snapshot={
                        "task_id": task_id,
                        "step_index": step_index,
                        "messages": messages,
                        "budget_remaining": budget.remaining,
                        "system_prompt": self.system_prompt,
                        "loop_guard": loop_guard.to_dict(),
                        "working_memory": self.working_memory.to_dict() if self.working_memory else {},
                    },
                )

        except BudgetExceededError as bee:
            final_response = f"M.A.D.D.Y. Resource Alert: {bee.message}"
            self.trace_store.record_event(
                task_id,
                "budget_exhausted",
                {"used": bee.error_context.get("used_iterations")},
            )
        except Exception as exc:
            logger.exception("Fatal turn error in resumed MaddyAgentCore")
            final_response = f"M.A.D.D.Y. Internal Disruption: {type(exc).__name__}: {str(exc)}"
            self.trace_store.record_event(
                task_id,
                "fatal_error",
                {"error": str(exc)},
            )
        finally:
            self.trace_store.update_task_status(
                task_id,
                status="completed" if completed else "failed",
            )
            if self.memory_store and completed and hasattr(self.memory_store, "episodic"):
                try:
                    tool_calls = self.trace_store.get_task_tool_calls(task_id)
                    tools_used = [tc.get("tool_name", "") for tc in tool_calls]
                    self.memory_store.episodic.record_episode(
                        task_id=task_id,
                        goal=f"Resumed Task {task_id}",
                        summary=final_response[:250],
                        status="completed" if completed else "failed",
                        tools_used=tools_used,
                    )
                except Exception as ep_exc:
                    logger.debug("Episodic recording skipped on resume: %s", ep_exc)

            # Autonomous Conversational Memory Extraction & Invalidation
            if self.memory_store and completed and hasattr(self, "memory_extractor") and self.memory_extractor:
                try:
                    tool_calls = self.trace_store.get_task_tool_calls(task_id)
                    extracted = self.memory_extractor.extract_and_sync(
                        user_prompt=f"Resumed Task {task_id}",
                        assistant_response=final_response,
                        tool_calls=tool_calls,
                        provenance_source="operator",
                    )
                    if extracted:
                        self.trace_store.record_event(
                            task_id,
                            "conversational_memory_extracted",
                            {"extracted_count": len(extracted), "facts": extracted},
                        )
                except Exception as ext_exc:
                    logger.debug("Conversational memory extraction in resumed turn skipped: %s", ext_exc)

        if self.personality and hasattr(self.personality, "format_response") and final_response:
            final_response = self.personality.format_response(final_response)

        return {
            "task_id": task_id,
            "response": final_response,
            "completed": completed,
            "steps_taken": step_index,
            "messages": messages,
            "resumed": True,
        }


