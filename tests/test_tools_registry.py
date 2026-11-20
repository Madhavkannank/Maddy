"""
Unit tests for ToolRegistry.
"""
import unittest
from tools.base import RiskLevel, Tool
from tools.registry import ToolRegistry, ToolRegistryError

def dummy_handler(args):
    return {"status": "ok"}

class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        self.dummy_tool = Tool(
            name="test.dummy",
            description="Dummy tool for testing",
            parameter_schema={"type": "object", "properties": {}},
            handler=dummy_handler,
            risk_level=RiskLevel.LOW,
        )

    def test_register_and_get_tool(self):
        self.registry.register(self.dummy_tool)
        retrieved = self.registry.get("test.dummy")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test.dummy")

    def test_duplicate_registration_raises_error(self):
        self.registry.register(self.dummy_tool)
        with self.assertRaises(ToolRegistryError):
            self.registry.register(self.dummy_tool)

    def test_list_tools(self):
        self.registry.register(self.dummy_tool)
        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "test.dummy")

    def test_get_tool_schemas(self):
        self.registry.register(self.dummy_tool)
        schemas = self.registry.get_tool_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "test.dummy")
        self.assertEqual(schemas[0]["risk_level"], "low")

    def test_unregister_tool(self):
        self.registry.register(self.dummy_tool)
        success = self.registry.unregister("test.dummy")
        self.assertTrue(success)
        self.assertIsNone(self.registry.get("test.dummy"))

if __name__ == "__main__":
    unittest.main()
