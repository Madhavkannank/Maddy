"""Base tool interface and registry for M.A.D.D.Y. Agent Core (Modular Autonomous Digital Daemon for You).

Provides typed tool definitions, OpenAI-compatible JSON schema exports,
timeout-bounded execution, and classified error handling.
"""

from __future__ import annotations

import abc
import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.errors import ToolExecutionError, ToolTimeoutError
from core.permissions import PermissionTier
from core.redact import redact_structure, redact_text

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Structured result returned by tool execution."""
    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": redact_text(self.output),
            "error": redact_text(self.error) if self.error else None,
            "metadata": redact_structure(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class BaseTool(abc.ABC):
    """Abstract base class for all M.A.D.D.Y. tools."""

    name: str
    description: str
    tier: PermissionTier = PermissionTier.SAFE

    @abc.abstractmethod
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Returns JSON schema for the tool arguments."""
        pass

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        """Return the deterministic permission tier for one operation.

        Tools with mixed read/write actions override this method so a read-only
        operation is not unnecessarily escalated to a mutating permission tier.
        """
        return self.tier

    @abc.abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Executes the tool with given keyword arguments."""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """Export tool definition in OpenAI/Groq function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_schema(),
            },
        }


class ToolRegistry:
    """Central registry and dispatcher for model tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def register_tool(self, tool: BaseTool) -> None:
        """Alias for register to support both register and register_tool conventions."""
        self._tools[tool.name] = tool

    def register_alias(self, alias_name: str, tool: BaseTool) -> None:
        """Register an alternative invocation name for an existing tool."""
        self._tools[alias_name] = tool

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def get_all_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def list_tools(self) -> List[BaseTool]:
        """Convenience alias for get_all_tools."""
        return self.get_all_tools()

    def get_openai_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas: List[Dict[str, Any]] = []
        seen_names = set()
        for tool in self._tools.values():
            s = tool.to_openai_schema()
            t_name = s.get("function", {}).get("name")
            if t_name and t_name not in seen_names:
                seen_names.add(t_name)
                schemas.append(s)
        return schemas

    def execute_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        timeout_seconds: float = 10.0,
    ) -> ToolResult:
        """Execute a tool with strict timeout bounding and error classification."""
        tool = self.get_tool(name)
        if not tool:
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' is not registered in the active toolset.",
            )

        start_time = time.monotonic()
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.execute, **arguments)

        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise ToolTimeoutError(
                message=f"Tool '{name}' exceeded maximum timeout of {timeout_seconds}s.",
                tool_name=name,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logger.exception("Error executing tool %s", name)
            return ToolResult(
                success=False,
                output="",
                error=f"Exception during tool execution: {type(exc).__name__}: {str(exc)}",
                metadata={"duration_ms": (time.monotonic() - start_time) * 1000.0},
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
