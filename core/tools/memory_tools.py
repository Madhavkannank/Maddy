"""Native Memory Tools for M.A.D.D.Y. Agent Core (Modular Autonomous Digital Daemon for You): remember, recall, forget, update_preference.

Provides typed tool definitions and parameter schemas for:
- remember: Preserving facts, preferences, project architecture in SQLite FTS5.
- recall: Searching long-term memory, past episodes, and completed task histories using multi-factor FTS5 ranking.
- forget: Soft-deleting memory entries.
- update_preference: Explicitly storing or updating stable operator preferences with precedence guarantees.

All memory tools operate under PermissionTier.SAFE (sandboxed SQLite storage).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from core.memory import MemoryCategory, MemoryStore
from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class RememberTool(BaseTool):
    """Tool for saving facts, preferences, project details, and observations to long-term memory."""

    name = "remember"
    description = (
        "Preserve a user preference, system fact, project architecture decision, or observation "
        "into M.A.D.D.Y. long-term memory with FTS5 search indexing."
    )
    tier = PermissionTier.SAFE

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory_store = memory_store or MemoryStore()

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The exact fact, preference, rule, or observation to preserve in memory.",
                },
                "category": {
                    "type": "string",
                    "enum": ["user_preference", "system_fact", "project", "episode", "observation"],
                    "description": "Classification of the memory item.",
                    "default": "user_preference",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of searchable keyword tags.",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance weighting factor (default 1.0).",
                    "default": 1.0,
                },
                "supersedes_id": {
                    "type": "string",
                    "description": "Optional memory_id of an older fact that this new fact replaces/contradicts.",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence level from 0.0 to 1.0 (default 1.0).",
                    "default": 1.0,
                },
                "is_explicit": {
                    "type": "boolean",
                    "description": "Whether this is an explicit operator instruction (default True).",
                    "default": True,
                },
            },
            "required": ["content"],
        }

    def execute(
        self,
        content: str,
        category: str = "user_preference",
        tags: Optional[List[str]] = None,
        importance: float = 1.0,
        supersedes_id: Optional[str] = None,
        confidence: float = 1.0,
        is_explicit: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        if not content or not content.strip():
            return ToolResult(
                success=False,
                output="",
                error="Cannot remember empty or whitespace content.",
            )

        try:
            mem_id = self.memory_store.remember(
                content=content.strip(),
                category=category,
                tags=tags or [],
                importance=importance,
                confidence=confidence,
                is_explicit=is_explicit,
                supersedes_id=supersedes_id,
            )
            return ToolResult(
                success=True,
                output=f"Memory preserved [ID: {mem_id}] in category '{category}'.",
                metadata={"memory_id": mem_id, "category": category},
            )
        except Exception as exc:
            logger.exception("RememberTool execution failure")
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to preserve memory: {str(exc)}",
            )


class RecallTool(BaseTool):
    """Tool for searching long-term memory, past episodes, and task histories using multi-factor FTS5 ranking."""

    name = "recall"
    description = (
        "Search M.A.D.D.Y. long-term memory, past episodes, and past task history using full-text search (FTS5 BM25). "
        "Returns the most relevant memories, preferences, and completed task summaries."
    )
    tier = PermissionTier.SAFE

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory_store = memory_store or MemoryStore()

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or phrase to search across memories and past task logs.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memory items to return (default 5).",
                    "default": 5,
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter (e.g. user_preference, system_fact, project).",
                },
            },
            "required": ["query"],
        }

    def execute(self, query: str, limit: int = 5, category: Optional[str] = None, **kwargs: Any) -> ToolResult:
        if not query or not query.strip():
            return ToolResult(
                success=False,
                output="",
                error="Search query cannot be empty.",
            )

        try:
            memories = self.memory_store.recall_multi_factor(query=query.strip(), limit=limit, category=category)
            episodes = self.memory_store.episodic.search_episodes(query=query.strip(), limit=max(2, limit // 2))
            tasks = self.memory_store.search_task_history(query=query.strip(), limit=max(2, limit // 2))

            if not memories and not episodes and not tasks:
                return ToolResult(
                    success=True,
                    output=f"No relevant memories or past tasks found matching query: '{query}'.",
                    metadata={"query": query, "found": 0},
                )

            lines = [f"Recalled {len(memories)} memory item(s), {len(episodes)} episode(s), and {len(tasks)} past task(s) for query: '{query}':"]
            for i, m in enumerate(memories, 1):
                cat = m.get("category", "note")
                cid = m.get("id", "")
                txt = m.get("content", "")
                exp_label = " [EXPLICIT]" if m.get("is_explicit", 1) else ""
                lines.append(f"{i}. [{cat.upper()}{exp_label}] (ID: {cid}) {txt}")

            if episodes:
                lines.append("\nRelevant Past Episodes:")
                for ep in episodes:
                    lines.append(f"- Episode {ep.get('episode_id')}: {ep.get('goal')} -> {ep.get('summary')}")

            if tasks:
                lines.append("\nRelevant Past Tasks:")
                for t in tasks:
                    tid = t.get("task_id", "")
                    desc = t.get("description", "")
                    lines.append(f"- Task {tid}: {desc}")

            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={
                    "memories_count": len(memories),
                    "episodes_count": len(episodes),
                    "tasks_count": len(tasks),
                },
            )
        except Exception as exc:
            logger.exception("RecallTool execution failure")
            return ToolResult(
                success=False,
                output="",
                error=f"Memory recall error: {str(exc)}",
            )


class ForgetTool(BaseTool):
    """Tool for soft-deleting or invalidating memory entries."""

    name = "forget"
    description = (
        "Soft-delete a specific memory item by its memory_id, excluding it from future active recall."
    )
    tier = PermissionTier.SAFE

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory_store = memory_store or MemoryStore()

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The exact ID of the memory to forget (e.g. 'mem_abc123').",
                },
            },
            "required": ["memory_id"],
        }

    def execute(self, memory_id: str, **kwargs: Any) -> ToolResult:
        if not memory_id or not memory_id.strip():
            return ToolResult(
                success=False,
                output="",
                error="Memory ID must be provided.",
            )

        try:
            success = self.memory_store.forget(memory_id=memory_id.strip())
            if success:
                return ToolResult(
                    success=True,
                    output=f"Memory forgotten [ID: {memory_id}]. It will no longer appear in active recall.",
                    metadata={"memory_id": memory_id, "forgotten": True},
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"No active memory found with ID: '{memory_id}'.",
                )
        except Exception as exc:
            logger.exception("ForgetTool execution failure")
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to forget memory: {str(exc)}",
            )


class UpdatePreferenceTool(BaseTool):
    """Tool for explicitly recording or updating an operator preference with precedence guarantees."""

    name = "update_preference"
    description = (
        "Explicitly record or update a stable operator preference (e.g. code style, editor, default browser, address title). "
        "Guarantees explicit operator preferences take absolute precedence over model inferences."
    )
    tier = PermissionTier.SAFE

    def __init__(self, memory_store: Optional[MemoryStore] = None):
        self.memory_store = memory_store or MemoryStore()

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The preference key (e.g. 'code_style.indentation', 'default_browser', 'editor').",
                },
                "value": {
                    "type": "string",
                    "description": "The exact preference value (e.g. '4 spaces', 'firefox', 'vscode').",
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence score between 0.0 and 1.0 (default 1.0 for explicit user statements).",
                    "default": 1.0,
                },
                "is_explicit": {
                    "type": "boolean",
                    "description": "Whether this preference was explicitly stated by the user (default True).",
                    "default": True,
                },
            },
            "required": ["key", "value"],
        }

    def execute(self, key: str, value: Any, confidence: float = 1.0, is_explicit: bool = True, **kwargs: Any) -> ToolResult:
        if not key or not str(key).strip():
            return ToolResult(
                success=False,
                output="",
                error="Preference 'key' cannot be empty.",
            )

        try:
            self.memory_store.preferences.set_preference(
                key=str(key).strip(),
                value=value,
                confidence=confidence,
                is_explicit=is_explicit,
            )
            return ToolResult(
                success=True,
                output=f"Preference '{key}' successfully saved as: {value}.",
                metadata={"key": key, "value": value, "is_explicit": is_explicit},
            )
        except Exception as exc:
            logger.exception("UpdatePreferenceTool execution failure")
            return ToolResult(
                success=False,
                output="",
                error=f"Failed to save preference: {str(exc)}",
            )
