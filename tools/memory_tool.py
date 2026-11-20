"""Persistent Memory Tool for M.A.D.D.Y.
Exposes the unified `memory` tool for managing persistent MEMORY.md and USER.md stores.
Supports add, patch, remove, replace, view, and atomic batch operations.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from tools.base import RiskLevel, Tool
from tools.memory_tool_store import (
    DEFAULT_MEMORY_CHAR_LIMIT,
    DEFAULT_USER_CHAR_LIMIT,
    MemoryStore,
)

logger = logging.getLogger("tools.memory_tool")

_GLOBAL_STORE: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """Returns the process-wide active MemoryStore singleton."""
    global _GLOBAL_STORE
    if _GLOBAL_STORE is None:
        _GLOBAL_STORE = MemoryStore()
        _GLOBAL_STORE.load_from_disk()
    return _GLOBAL_STORE


def set_memory_store(store: Optional[MemoryStore]) -> None:
    """Overrides the process-wide MemoryStore (used for test isolation)."""
    global _GLOBAL_STORE
    _GLOBAL_STORE = store


MEMORY_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {
            "type": "string",
            "enum": ["memory", "user"],
            "description": "Which memory store: 'memory' for personal/system notes, 'user' for user profile and preferences.",
        },
        "action": {
            "type": "string",
            "enum": ["add", "patch", "remove", "view", "replace"],
            "description": "The action to perform: 'add' (append entry), 'patch' (modify span in entry), 'remove' (delete entry), 'view' (inspect entries/usage). Omit when using 'operations'.",
        },
        "content": {
            "type": "string",
            "description": "Required for 'add' and 'patch'. For 'add', the new entry text. For 'patch', the replacement text for old_text. Alias: 'new_text'.",
        },
        "old_text": {
            "type": "string",
            "description": "Required for 'patch' and 'remove': unique substring from one entry. 'patch' replaces this span; 'remove' deletes the whole entry.",
        },
        "new_text": {
            "type": "string",
            "description": "Alias for 'content'. If both are set, 'content' takes precedence.",
        },
        "auto_trim": {
            "type": "boolean",
            "description": "Optional (default false). When true and store is over budget on 'add', automatically evicts the oldest entry/entries (FIFO) to make room for this new entry.",
        },
        "operations": {
            "type": "array",
            "description": "Batch shape: list of operations {action, content?, old_text?} applied atomically in one call against the final budget.",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "patch", "remove", "replace"]},
                    "content": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    "required": ["target"],
}


def memory_tool(
    target: str = "memory",
    action: Optional[str] = None,
    content: Optional[str] = None,
    old_text: Optional[str] = None,
    new_text: Optional[str] = None,
    auto_trim: bool = False,
    operations: Optional[List[Dict[str, Any]]] = None,
    store: Optional[MemoryStore] = None,
) -> Dict[str, Any]:
    """
    Core handler executing memory operations against the active MemoryStore.
    """
    active_store = store or get_memory_store()

    if target not in ("memory", "user"):
        return {
            "success": False,
            "error": f"Invalid target '{target}'. Target must be 'memory' or 'user'.",
            "failure_class": "invalid_args",
        }

    # Handle content alias
    effective_content = content or new_text

    # Batch operations take priority
    if operations:
        return active_store.apply_batch(target, operations)

    if not action:
        return {
            "success": False,
            "error": "Missing required action: specify 'action' (add, patch, remove, view) or 'operations' batch.",
            "failure_class": "invalid_args",
        }

    act = action.lower().strip()
    if act == "view":
        return active_store.view(target)
    elif act == "add":
        return active_store.add(target, effective_content or "", auto_trim=auto_trim)
    elif act == "patch":
        return active_store.patch(target, old_text or "", effective_content or "")
    elif act == "remove":
        return active_store.remove(target, old_text or "")
    elif act == "replace":
        return active_store.replace(target, old_text or "", effective_content or "")
    else:
        return {
            "success": False,
            "error": f"Unknown action '{action}'. Supported actions: add, patch, remove, view, replace.",
            "failure_class": "invalid_args",
        }


def _memory_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Tool execution adapter for M.A.D.D.Y. ToolRegistry."""
    return memory_tool(
        target=params.get("target", "memory"),
        action=params.get("action"),
        content=params.get("content"),
        old_text=params.get("old_text"),
        new_text=params.get("new_text"),
        auto_trim=params.get("auto_trim", False),
        operations=params.get("operations"),
    )


memory_aci_tool = Tool(
    name="memory",
    description=(
        "Save, patch, view, or remove durable facts in persistent memory that survive across sessions.\n"
        "TARGETS:\n"
        "- 'user': who the user is (name, preferred address, favorite music, location, style).\n"
        "- 'memory': personal notes (hardware specs, OS context, speech pipeline, environmental facts).\n"
        "ACTIONS:\n"
        "- 'add': append a new entry (requires 'content').\n"
        "- 'patch': update a span in an existing entry (requires 'old_text' and 'content').\n"
        "- 'remove': delete an entire entry (requires 'old_text').\n"
        "- 'view': inspect current entries and character budget.\n"
        "- 'operations': batch list of atomic operations applied together against final budget."
    ),
    parameter_schema=MEMORY_SCHEMA,
    handler=_memory_handler,
    risk_level=RiskLevel.LOW,
)
