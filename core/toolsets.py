"""Toolset and Dynamic Tool Gating Architecture for M.A.D.D.Y.
(Modular Autonomous Digital Daemon for You).

Enforces a narrow core waist to keep baseline prompt overhead under 900 tokens
(preventing Groq HTTP 413 / 7000 TPM limit breaches) while dynamically gating
specialized toolsets based on conversational intent and turn history.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


# Narrow core waist: tools that are fundamental to assistant execution and
# always present in every turn. All other capabilities are gated into domain toolsets.
_MADDY_CORE_TOOLS: Tuple[str, ...] = (
    "app_control",      # Open apps, local files, URLs
    "web_browser",     # Web navigation, site search, domain resolution
    "web_search",      # General web information discovery
    "system_control",   # System settings, volume, display
    "remember",        # Core episodic/user memory writing
    "recall",          # Explicit memory querying
)

# Domain toolsets defining specialized capabilities and their trigger patterns
TOOLSETS: Dict[str, Dict[str, Any]] = {
    "math": {
        "description": "Mathematical calculation, arithmetic, and quantitative reasoning",
        "tools": ["calculator"],
        "keywords": (
            "calculate", "calculator", "math", "add", "subtract", "multiply",
            "divide", "sum", "difference", "product", "quotient", "equation",
            "percent", "percentage", "sqrt", "power", "pi", "factorial",
        ),
        "regex": r"\b\d+[\s]*[\+\-\*\/\^][\s]*\d+\b",
    },
    "weather": {
        "description": "Live weather telemetry, forecasts, and atmospheric conditions",
        "tools": ["weather"],
        "keywords": (
            "weather", "forecast", "temperature", "temp", "rain", "snow",
            "humidity", "wind", "sunny", "cloudy", "degrees", "celsius", "fahrenheit",
        ),
    },
    "media": {
        "description": "Media player controls, music playback, and audio tracks",
        "tools": ["media_control"],
        "keywords": (
            "media", "music", "song", "play", "pause", "resume", "stop",
            "next", "previous", "track", "spotify", "playback",
        ),
    },
    "system_info": {
        "description": "Detailed hardware, CPU, RAM, battery, OS, platform telemetry, and temporal clock state",
        "tools": ["system_info"],
        "keywords": (
            "specs", "specifications", "cpu", "ram", "memory usage", "battery",
            "hardware", "os", "operating system", "disk", "storage space", "uptime",
            "system info", "system status", "time", "date", "clock", "current time",
            "what time", "what date", "today", "day", "timezone", "timestamp",
        ),
        "regex": r"\b(time|date|clock|today|timezone|timestamp)\b",
    },
    "camera": {
        "description": "Camera capture and photo snapshots",
        "tools": ["camera_tool"],
        "keywords": (
            "camera", "photo", "picture", "snap", "snapshot", "selfie", "take photo",
        ),
    },
    "timer": {
        "description": "Timers, alarms, stopwatches, and countdowns",
        "tools": ["timer_tool"],
        "keywords": (
            "timer", "alarm", "countdown", "stopwatch", "remind in", "wake me in",
        ),
    },
    "messaging": {
        "description": "Outbound messaging and notification relays",
        "tools": ["telegram"],
        "keywords": (
            "telegram", "message", "send message", "chat",
        ),
    },
    "memory_management": {
        "description": "Advanced memory deletion and preference overrides",
        "tools": ["forget", "update_preference"],
        "keywords": (
            "forget", "delete memory", "clear memory", "preference",
            "call me", "address me as", "my preferred", "update preference",
        ),
    },
    "file_system": {
        "description": "File manipulation, directory listing, and command execution",
        "tools": ["read_file", "write_file", "list_directory", "execute_command"],
        "keywords": (
            "file", "files", "folder", "directory", "dir", "read", "write",
            "create file", "script", "command", "terminal", "bash", "shell",
            "run", "exec", "git", "code", "log", "python", "path",
            "json", "txt", "csv",
        ),
    },
    "arxiv": {
        "description": "Academic papers and scientific literature retrieval",
        "tools": ["arxiv_tool"],
        "keywords": (
            "arxiv", "paper", "papers", "research", "journal", "citation", "academic",
        ),
    },
    "computer_use": {
        "description": "Visual screen observation, coordinate interaction, and browser automation",
        "tools": ["computer_screen", "computer_interaction", "browser_automation"],
        "keywords": (
            "screen", "screenshot", "click", "mouse", "cursor", "type",
            "coordinate", "drag", "scroll", "dom", "element", "visual", "gui",
        ),
    },
    "skills": {
        "description": "Procedural skill document management, discovery, and execution",
        "tools": ["list_skills", "search_skills", "load_skill", "validate_skill"],
        "keywords": (
            "skill", "skills", "procedure", "recipe", "load skill", "run skill",
        ),
    },
    "delegation": {
        "description": "Subagent task delegation and parallel decomposition",
        "tools": ["delegate_task"],
        "keywords": (
            "delegate", "subagent", "worker", "parallel task", "decompose",
        ),
    },
    "cronjob": {
        "description": "Background task scheduling and recurring daemon jobs",
        "tools": ["cronjob_manage"],
        "keywords": (
            "cron", "schedule", "recurring", "periodic", "remind me at",
            "every day", "every hour", "schedule task",
        ),
    },
}

# Fast lookup index mapping known builtin tool names
ALL_BUILTIN_TOOLS: Set[str] = set(_MADDY_CORE_TOOLS)
for ts in TOOLSETS.values():
    ALL_BUILTIN_TOOLS.update(ts.get("tools", []))


def resolve_active_tool_names(
    user_prompt: str,
    registered_tools: Set[str],
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> Set[str]:
    """Dynamically resolves the optimal set of tool names for a turn.

    Invariants:
    1. Narrow Core: Always includes _MADDY_CORE_TOOLS (if registered).
    2. Intent Gating: Includes domain toolsets matching the current user prompt.
    3. Conversational Continuity: Preserves tools invoked in recent conversation history.
    4. Custom Extensibility: Preserves any injected tools not part of standard builtins (e.g. unit test mocks).
    """
    selected: Set[str] = set()

    # 1. Base Core Waist (only add if actually registered in ToolRegistry)
    for tool_name in _MADDY_CORE_TOOLS:
        if tool_name in registered_tools:
            selected.add(tool_name)

    # 2. Intent-based domain toolset activation
    p_lower = (user_prompt or "").lower()
    for ts_name, ts_spec in TOOLSETS.items():
        matched = False
        # Check keywords with word-boundary start matching to prevent false substring hits (e.g. 'add' in 'maddy')
        keywords = ts_spec.get("keywords", ())
        for kw in keywords:
            if len(kw) <= 3:
                # Exact word boundary for ultra-short tokens (e.g. 'os', 'ram', 'cpu')
                if re.search(r"\b" + re.escape(kw) + r"\b", p_lower):
                    matched = True
                    break
            else:
                # Word-start matching for words (e.g. 'rain' matches 'raining', but not 'brain')
                if re.search(r"\b" + re.escape(kw), p_lower):
                    matched = True
                    break

        # Check optional regex
        regex_pattern = ts_spec.get("regex")
        if not matched and regex_pattern and re.search(regex_pattern, p_lower):
            matched = True

        if matched:
            for t in ts_spec.get("tools", []):
                if t in registered_tools:
                    selected.add(t)

    # 3. Conversational Continuity: preserve tools recently called in message history
    if conversation_history:
        for msg in conversation_history[-6:]:
            # If assistant recently called a tool, keep that tool active for follow-up turns
            tool_calls = msg.get("tool_calls") or []
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if fn_name and fn_name in registered_tools:
                    selected.add(fn_name)

    # 4. Custom/Injected Tools: Always preserve tools outside ALL_BUILTIN_TOOLS (e.g. test doubles)
    for registered_name in registered_tools:
        if registered_name not in ALL_BUILTIN_TOOLS:
            selected.add(registered_name)

    return selected
