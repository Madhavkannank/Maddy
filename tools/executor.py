"""
Tool Executor for running validated ACI tool calls.
"""
import time
from typing import Any, Dict, Optional, Union
from tools.call import ToolCall, ToolCallParseError, parse_tool_call
from tools.registry import ToolRegistry
from tools.result import ToolResult
from tools.validator import ToolValidator, ValidationError

class ToolExecutor:
    """
    Executor responsible for resolving, validating, checking permissions, and executing tool calls safely.
    """

    def __init__(self, registry: ToolRegistry, permission_engine: Optional[Any] = None):
        self.registry = registry
        self.permission_engine = permission_engine

    def execute(self, tool_call_input: Union[ToolCall, str, Dict[str, Any]]) -> ToolResult:
        """
        Execute a tool call safely, returning a structured ToolResult.
        """
        start_time = time.time()

        # 1. Parse raw call if needed
        if isinstance(tool_call_input, (str, dict)):
            try:
                call = parse_tool_call(tool_call_input)
            except ToolCallParseError as e:
                execution_time_ms = (time.time() - start_time) * 1000.0
                return ToolResult.fail(
                    error_type="MALFORMED_TOOL_CALL",
                    message=str(e),
                    tool_name="unknown",
                    execution_time_ms=execution_time_ms,
                )
        elif isinstance(tool_call_input, ToolCall):
            call = tool_call_input
        else:
            execution_time_ms = (time.time() - start_time) * 1000.0
            return ToolResult.fail(
                error_type="INVALID_INPUT",
                message=f"Expected ToolCall, str, or dict; got {type(tool_call_input).__name__}",
                tool_name="unknown",
                execution_time_ms=execution_time_ms,
            )

        tool_name = call.name

        # 2. Resolve tool from registry
        tool = self.registry.get(tool_name)
        if not tool:
            execution_time_ms = (time.time() - start_time) * 1000.0
            return ToolResult.fail(
                error_type="UNKNOWN_TOOL",
                message=f"Tool '{tool_name}' is not registered in the ACI tool registry",
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
            )

        # 3. Validate call arguments against schema
        try:
            ToolValidator.validate(call, tool)
        except ValidationError as ve:
            execution_time_ms = (time.time() - start_time) * 1000.0
            return ToolResult.fail(
                error_type=ve.error_type,
                message=ve.message,
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
            )

        # 4. Check permissions via PermissionEngine if configured
        if self.permission_engine:
            decision = self.permission_engine.check_permission(call, tool)
            if not decision.allowed:
                execution_time_ms = (time.time() - start_time) * 1000.0
                return ToolResult.fail(
                    error_type="PERMISSION_DENIED",
                    message=decision.reason,
                    tool_name=tool_name,
                    execution_time_ms=execution_time_ms,
                )

        # 4. Execute tool handler safely
        try:
            result_data = tool.handler(call.arguments)
            execution_time_ms = (time.time() - start_time) * 1000.0
            
            # If handler returns an explicit error structure
            if isinstance(result_data, dict) and result_data.get("success") is False:
                raw_err = result_data.get("error", {})
                if isinstance(raw_err, dict):
                    err_type = raw_err.get("type", "EXECUTION_ERROR")
                    err_msg = raw_err.get("message", result_data.get("message", "Tool handler reported failure"))
                else:
                    err_type = str(raw_err) if raw_err else "EXECUTION_ERROR"
                    err_msg = result_data.get("message", str(raw_err))
                return ToolResult.fail(
                    error_type=err_type,
                    message=err_msg,
                    tool_name=tool_name,
                    execution_time_ms=execution_time_ms,
                )

            return ToolResult.ok(
                data=result_data if isinstance(result_data, dict) else {"result": result_data},
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
            )
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000.0
            return ToolResult.fail(
                error_type="EXECUTION_EXCEPTION",
                message=f"Tool handler threw an exception: {str(e)}",
                tool_name=tool_name,
                execution_time_ms=execution_time_ms,
            )
