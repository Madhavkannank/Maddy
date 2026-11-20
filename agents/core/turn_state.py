"""Turn State Machine and Context Manager for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Monitors token limits, handles message truncation, prevents infinite tool loops,
and recovers from malformed model outputs.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from models.base import ModelMessage

logger = logging.getLogger(__name__)

DEFAULT_TOOL_OUTPUT_MAX_CHARS = 4000
DEFAULT_TOKEN_BUDGET_CHARS = 24000


def truncate_tool_output(
    content: str,
    max_chars: int = DEFAULT_TOOL_OUTPUT_MAX_CHARS,
    head_chars: int = 2500,
    tail_chars: int = 1000,
) -> str:
    """
    Truncates large tool output while preserving both the start and ending sections.
    Prevents context blowups from verbose filesystem reads or system logs.
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
    return f"{head}\n\n... [Output truncated: {omitted:,} characters omitted to preserve context budget] ...\n\n{tail}"


class TurnState:
    """
    Manages conversational turn state, context limits, and tool execution history.
    """

    def __init__(
        self,
        max_turns: int = 15,
        max_repeated_tool_calls: int = 3,
        token_budget_chars: int = DEFAULT_TOKEN_BUDGET_CHARS,
    ):
        self.max_turns = max_turns
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.token_budget_chars = token_budget_chars
        self.tool_call_history: List[str] = []
        self.consecutive_empty_responses = 0

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        Record a tool call. Returns False if the same tool call with identical arguments
        has been repeated consecutively beyond max_repeated_tool_calls (detecting runaway loops).
        """
        try:
            signature = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        except Exception:
            signature = f"{tool_name}:{str(arguments)}"

        self.tool_call_history.append(signature)

        # Check consecutive repeats
        if len(self.tool_call_history) >= self.max_repeated_tool_calls:
            recent = self.tool_call_history[-self.max_repeated_tool_calls :]
            if all(s == signature for s in recent):
                logger.warning(f"[TurnState] Detected runaway execution loop on tool: {tool_name}")
                return False
        return True

    def manage_context_budget(
        self,
        messages: List[ModelMessage],
    ) -> List[ModelMessage]:
        """
        Ensure messages remain within reasonable character/token bounds.
        If exceeding budget, compacts older intermediate tool responses
        while preserving the initial system prompt, original goal, and recent turns.
        """
        total_chars = sum(len(m.content or "") for m in messages)
        if total_chars <= self.token_budget_chars or len(messages) <= 4:
            return messages

        # Preserve index 0 (System) and index 1 (Initial user goal), plus last 2 turns
        compacted: List[ModelMessage] = [messages[0], messages[1]]
        middle = messages[2:-2]

        for m in middle:
            if m.role == "tool" and len(m.content or "") > 500:
                # Compact lengthy older tool responses
                truncated_content = truncate_tool_output(m.content or "", max_chars=500, head_chars=300, tail_chars=150)
                compacted.append(ModelMessage(role=m.role, content=truncated_content))
            else:
                compacted.append(m)

        # Append most recent turns intact
        compacted.extend(messages[-2:])
        return compacted
