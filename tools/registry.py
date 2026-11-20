"""
Central Tool Registry for registering, retrieving, and inspecting ACI tools.
"""
from typing import Any, Dict, List, Optional
from tools.base import Tool

class ToolRegistryError(ValueError):
    """Exception raised for tool registry errors."""
    pass

class ToolRegistry:
    """
    Central registry for managing registered ACI tools.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a new tool. Raises ToolRegistryError on duplicate or invalid registration.
        """
        if not tool or not isinstance(tool, Tool):
            raise ToolRegistryError("Invalid tool instance provided")

        if not tool.name or not isinstance(tool.name, str):
            raise ToolRegistryError("Tool name must be a non-empty string")

        if tool.name in self._tools:
            raise ToolRegistryError(f"Tool with name '{tool.name}' is already registered")

        if not callable(tool.handler):
            raise ToolRegistryError(f"Tool '{tool.name}' must have a callable execution handler")

        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name. Returns True if tool was present and removed, False otherwise.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[Tool]:
        """
        Retrieve a tool by name.
        """
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """
        Return a list of all registered tools.
        """
        return list(self._tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        Return JSON schema representations of all registered tools for LLM prompts/tools configuration.
        """
        return [tool.to_schema_dict() for tool in self._tools.values()]
