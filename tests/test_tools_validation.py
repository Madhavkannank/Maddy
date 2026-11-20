"""
Unit tests for Tool call parsing and schema validation.
"""
import unittest
from tools.base import Tool
from tools.call import ToolCall, ToolCallParseError, parse_tool_call
from tools.validator import ToolValidator, ValidationError

def dummy_handler(args):
    return {}

class TestToolValidation(unittest.TestCase):

    def setUp(self):
        self.tool = Tool(
            name="test.math",
            description="Math test tool",
            parameter_schema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                    "operation": {"type": "string"},
                },
                "required": ["a", "b"],
            },
            handler=dummy_handler,
        )

    def test_parse_tool_call_valid_json(self):
        json_str = '{"tool": "test.math", "arguments": {"a": 5, "b": 10}}'
        call = parse_tool_call(json_str)
        self.assertEqual(call.name, "test.math")
        self.assertEqual(call.arguments, {"a": 5, "b": 10})

    def test_parse_tool_call_malformed_json_raises_error(self):
        with self.assertRaises(ToolCallParseError):
            parse_tool_call("not a valid json")

    def test_validate_success(self):
        call = ToolCall(name="test.math", arguments={"a": 5, "b": 10, "operation": "add"})
        ToolValidator.validate(call, self.tool)

    def test_validate_missing_required_argument(self):
        call = ToolCall(name="test.math", arguments={"a": 5})
        with self.assertRaises(ValidationError) as ctx:
            ToolValidator.validate(call, self.tool)
        self.assertEqual(ctx.exception.error_type, "MISSING_ARGUMENT")

    def test_validate_unexpected_argument(self):
        call = ToolCall(name="test.math", arguments={"a": 5, "b": 10, "invalid_param": True})
        with self.assertRaises(ValidationError) as ctx:
            ToolValidator.validate(call, self.tool)
        self.assertEqual(ctx.exception.error_type, "UNEXPECTED_ARGUMENT")

    def test_validate_invalid_argument_type(self):
        call = ToolCall(name="test.math", arguments={"a": "not_an_int", "b": 10})
        with self.assertRaises(ValidationError) as ctx:
            ToolValidator.validate(call, self.tool)
        self.assertEqual(ctx.exception.error_type, "INVALID_TYPE")

if __name__ == "__main__":
    unittest.main()
