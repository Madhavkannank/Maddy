"""Tiered Prompt Builder for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Constructs ordered prompts separating:
- Tier 1: Stable Persona (Immutable identity, address policies, speech cleanliness)
- Tier 2: System & Environmental Context (OS, hardware, constraints, capabilities)
- Tier 3: Volatile Context (Byte-stable calendar date & timezone, USER.md, MEMORY.md, and <tools> schema)

Prompt Prefix Caching Invariant:
The system prompt remains byte-identical across all turns during a calendar day.
Date and timezone information are anchored stably to allow LLM provider KV-cache prefix hits,
while real-time seconds/clock queries are dynamically executed via the system_info tool.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
from typing import Any, Dict, List, Optional

from personality.maddy_persona import build_stable_persona_prompt
from tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class MaddyPromptBuilder:
    """Tiered prompt assembly for M.A.D.D.Y. agent execution."""

    def __init__(
        self,
        registry: ToolRegistry,
        address_title: str = "boss",
        workspace_dir: Optional[str] = None,
    ):
        self.registry = registry
        self.address_title = address_title
        self.workspace_dir = workspace_dir or os.getcwd()

    def build_system_prompt(
        self,
        context: str = "general",
        user_md_content: Optional[str] = None,
        memory_md_content: Optional[str] = None,
        memory_store: Optional[Any] = None,
        skill_engine: Optional[Any] = None,
    ) -> str:
        """Assembles the complete system prompt across all three tiers."""
        # Tier 1: Stable Persona
        tier1_stable = build_stable_persona_prompt(
            address_title=self.address_title,
            context=context,
        )

        # Tier 2: System & Environmental Context
        os_name = "Windows 11 x64" if os.name == "nt" else "Fedora Linux"
        tier2_context = f"""==================================================
ENVIRONMENT & EXECUTION CONTEXT
==================================================
- Operating System: {os_name}
- Active Workspace: {self.workspace_dir}
- Execution Mode: Single-model queue with concurrent environment I/O.
- Model Instructions: Reason clearly in <thought></thought> tags. Execute operations via <tool_call></tool_call>.
"""

        # Tier 3: Volatile Context (Byte-stable Date, Dynamic Memory Snapshots, Skills Index, Tools Schema)
        # Anchor date-only for prefix cache stability across turns
        now = datetime.datetime.now().astimezone()
        tz_name = now.tzname() or "UTC"
        tz_offset = now.strftime("%z")
        tz_formatted_offset = f"UTC{tz_offset[:3]}:{tz_offset[3:]}" if tz_offset else "UTC"
        date_str = now.strftime("%A, %B %d, %Y")

        user_section = ""
        memory_section = ""

        if user_md_content is not None or memory_md_content is not None:
            if user_md_content and user_md_content.strip():
                user_section = f"\n{user_md_content.strip()}\n"
            if memory_md_content and memory_md_content.strip():
                memory_section = f"\n{memory_md_content.strip()}\n"
        else:
            # Default to active MemoryStore snapshot
            try:
                from tools.memory_tool import get_memory_store
                store = memory_store or get_memory_store()
                user_snap = store.format_for_system_prompt("user")
                mem_snap = store.format_for_system_prompt("memory")
                if user_snap:
                    user_section = f"\n{user_snap}\n"
                if mem_snap:
                    memory_section = f"\n{mem_snap}\n"
            except Exception as e:
                logger.debug(f"MemoryStore snapshot render skipped: {e}")

        # Progressive disclosure skills index
        skills_section = ""
        try:
            from agents.core.skill_engine import SkillEngine
            engine = skill_engine or SkillEngine()
            formatted_skills = engine.format_skills_for_prompt()
            if formatted_skills:
                skills_section = f"\n{formatted_skills}\n"
        except Exception as e:
            logger.debug(f"Skills prompt index render skipped: {e}")

        # Format tools schema in standard JSON schema format
        schemas = self.registry.get_tool_schemas()
        formatted_tools = []
        for s in schemas:
            if "function" in s:
                formatted_tools.append({
                    "type": "function",
                    "function": s["function"],
                })
            else:
                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": s.get("name", ""),
                        "description": s.get("description", ""),
                        "parameters": s.get("parameters", s.get("parameter_schema", {})),
                    }
                })

        tools_json = json.dumps(formatted_tools, indent=2)

        tier3_volatile = f"""==================================================
ACTIVE RUNTIME STATE & TOOLS
==================================================
- Today's Date: {date_str} ({tz_name}, {tz_formatted_offset})
- Real-Time Clock: Dynamic (call system_info tool when exact seconds/time of day is requested)
{user_section}{memory_section}{skills_section}
<tools>
{tools_json}
</tools>

TOOL INVOCATION PROTOCOL:
1. When external actions, filesystem modifications, searches, or checks are needed, emit:
   <tool_call>
   {{"name": "tool_name", "arguments": {{"arg_name": "arg_value"}}}}
   </tool_call>
2. You may emit multiple <tool_call> tags in a single step if actions are independent.
3. Tool observations will return wrapped in <tool_response></tool_response> tags:
   <tool_response>
   {{"name": "tool_name", "content": {{...}}}}
   </tool_response>
4. When finished, or if no tools are required, output your final answer directly in your natural M.A.D.D.Y. persona (addressing the user as {self.address_title}, concise, speech-friendly, zero corporate cliches).
"""

        return f"{tier1_stable}\n\n{tier2_context}\n\n{tier3_volatile}".strip()
