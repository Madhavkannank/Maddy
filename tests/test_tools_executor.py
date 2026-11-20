"""
Unit tests for ToolExecutor execution boundary and structured results.
"""
import unittest
from tools.base import Tool
from tools.call import ToolCall
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

def echo_handler(args):
    return {"echo": args["message"]}

def exception_handler(args):
    raise ValueError("Internal handler error")

class TestToolExecutor(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        self.echo_tool = Tool(
            name="test.echo",
            description="Echo tool",
            parameter_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=echo_handler,
        )
        self.fail_tool = Tool(
            name="test.fail",
            description="Failing tool",
            parameter_schema={"type": "object", "properties": {}},
            handler=exception_handler,
        )
        self.registry.register(self.echo_tool)
        self.registry.register(self.fail_tool)
        self.executor = ToolExecutor(self.registry)

    def test_execute_success(self):
        call = ToolCall(name="test.echo", arguments={"message": "hello"})
        result = self.executor.execute(call)
        self.assertTrue(result.success)
        self.assertEqual(result.tool_name, "test.echo")
        self.assertEqual(result.data, {"echo": "hello"})
        self.assertGreaterEqual(result.execution_time_ms, 0.0)

    def test_execute_unknown_tool_fails_structured(self):
        call = ToolCall(name="test.unknown", arguments={})
        result = self.executor.execute(call)
        self.assertFalse(result.success)
        self.assertEqual(result.error["type"], "UNKNOWN_TOOL")

    def test_execute_validation_failure_fails_structured(self):
        call = ToolCall(name="test.echo", arguments={})  # Missing required "message"
        result = self.executor.execute(call)
        self.assertFalse(result.success)
        self.assertEqual(result.error["type"], "MISSING_ARGUMENT")

    def test_execute_exception_fails_structured(self):
        call = ToolCall(name="test.fail", arguments={})
        result = self.executor.execute(call)
        self.assertFalse(result.success)
        self.assertEqual(result.error["type"], "EXECUTION_EXCEPTION")
        self.assertIn("Internal handler error", result.error["message"])

if __name__ == "__main__":
    unittest.main()
