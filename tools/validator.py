"""
Tool call argument and schema validator for ACI.
"""
from typing import Any, Dict, List, Optional
from tools.base import Tool
from tools.call import ToolCall

class ValidationError(ValueError):
    """Exception raised when a tool call fails validation."""
    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.message = message

class ToolValidator:
    """
    Validates ToolCall parameters against the JSON Schema defined in Tool.parameter_schema.
    """

    TYPE_MAPPINGS = {
        "string": (str,),
        "integer": (int,),
        "number": (int, float),
        "boolean": (bool,),
        "array": (list,),
        "object": (dict,),
    }

    @classmethod
    def validate(cls, tool_call: ToolCall, tool: Tool) -> None:
        """
        Validate tool call arguments against target tool schema.
        Raises ValidationError if validation fails.
        """
        schema = tool.parameter_schema
        args = tool_call.arguments

        if not isinstance(args, dict):
            raise ValidationError("INVALID_ARGUMENTS", "Arguments must be a key-value object")

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        # 1. Check missing required arguments
        for req_param in required:
            if req_param not in args or args[req_param] is None:
                raise ValidationError(
                    "MISSING_ARGUMENT",
                    f"Missing required argument '{req_param}' for tool '{tool.name}'",
                )

        # 2. Check for unexpected arguments (if properties are defined)
        if properties:
            for arg_key in args.keys():
                if arg_key not in properties:
                    raise ValidationError(
                        "UNEXPECTED_ARGUMENT",
                        f"Unexpected argument '{arg_key}' passed to tool '{tool.name}'",
                    )

        # 3. Check argument types
        for param_name, param_schema in properties.items():
            if param_name in args and args[param_name] is not None:
                expected_type_str = param_schema.get("type")
                if expected_type_str in cls.TYPE_MAPPINGS:
                    expected_py_types = cls.TYPE_MAPPINGS[expected_type_str]
                    val = args[param_name]
                    # Python bool is a subclass of int, so handle boolean explicitly
                    if expected_type_str == "integer" and isinstance(val, bool):
                        raise ValidationError(
                            "INVALID_TYPE",
                            f"Argument '{param_name}' expected integer, got boolean",
                        )
                    if not isinstance(val, expected_py_types):
                        raise ValidationError(
                            "INVALID_TYPE",
                            f"Argument '{param_name}' expected type '{expected_type_str}', got '{type(val).__name__}'",
                        )
