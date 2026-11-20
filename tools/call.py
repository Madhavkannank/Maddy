"""
Structured representation and parsing for model tool calls.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

@dataclass
class ToolCall:
    """
    Representation of a requested tool call from an agent/model.
    """
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    call_id: Optional[str] = None

class ToolCallParseError(ValueError):
    """Exception raised when raw model output cannot be parsed into a ToolCall."""
    pass

def parse_tool_call(raw_input: Union[str, Dict[str, Any]]) -> ToolCall:
    """
    Parse raw string output or dictionary payload into a structured ToolCall instance.
    """
    if isinstance(raw_input, str):
        try:
            payload = json.loads(raw_input)
        except json.JSONDecodeError as e:
            raise ToolCallParseError(f"Failed to parse tool call JSON: {e}") from e
    elif isinstance(raw_input, dict):
        payload = raw_input
    else:
        raise ToolCallParseError(f"Expected str or dict for tool call, got {type(raw_input).__name__}")

    if not isinstance(payload, dict):
        raise ToolCallParseError("Tool call payload must be a JSON object")

    # Handle standard tool call formats: {"tool": "name", "arguments": {...}} or {"name": "name", "arguments": {...}}
    tool_name = payload.get("tool") or payload.get("name") or payload.get("function")
    if isinstance(tool_name, dict):
        tool_name = tool_name.get("name")

    if not tool_name or not isinstance(tool_name, str):
        raise ToolCallParseError("Tool call payload missing valid 'tool' or 'name' string field")

    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ToolCallParseError("Tool call 'arguments' field must be a dictionary")

    call_id = payload.get("call_id") or payload.get("id")

    return ToolCall(
        name=tool_name,
        arguments=arguments,
        call_id=str(call_id) if call_id else None,
    )
