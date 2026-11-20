"""
Unit tests for ToolExecutor integrated with PermissionEngine security checks.
"""
import unittest
from security.confirmation import AutoDenyConfirmationHandler, InteractiveConfirmationHandler
from security.permission_engine import PermissionEngine
from security.policy import PolicyMode
from tools.base import RiskLevel, Tool
from tools.call import ToolCall
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

def safe_handler(args):
    return {"message": "executed"}

class TestSecureExecutor(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        self.low_tool = Tool(
            name="test.safe",
            description="Safe tool",
            parameter_schema={"type": "object", "properties": {}},
            handler=safe_handler,
            risk_level=RiskLevel.LOW,
        )
        self.high_tool = Tool(
            name="test.risky",
            description="Risky tool",
            parameter_schema={"type": "object", "properties": {}},
            handler=safe_handler,
            risk_level=RiskLevel.HIGH,
        )
        self.registry.register(self.low_tool)
        self.registry.register(self.high_tool)

    def test_executor_with_permission_engine_allows_low_risk(self):
        perm_engine = PermissionEngine(policy_mode=PolicyMode.BALANCED)
        executor = ToolExecutor(self.registry, permission_engine=perm_engine)

        call = ToolCall(name="test.safe", arguments={})
        result = executor.execute(call)
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"message": "executed"})

    def test_executor_with_permission_engine_blocks_unconfirmed_high_risk(self):
        perm_engine = PermissionEngine(
            policy_mode=PolicyMode.BALANCED,
            confirmation_handler=AutoDenyConfirmationHandler(),
        )
        executor = ToolExecutor(self.registry, permission_engine=perm_engine)

        call = ToolCall(name="test.risky", arguments={})
        result = executor.execute(call)
        self.assertFalse(result.success)
        self.assertEqual(result.error["type"], "PERMISSION_DENIED")
        self.assertIn("Permission denied", result.error["message"])

    def test_executor_with_permission_engine_executes_confirmed_high_risk(self):
        handler = InteractiveConfirmationHandler(callback=lambda call, tool, eval: True)
        perm_engine = PermissionEngine(
            policy_mode=PolicyMode.BALANCED,
            confirmation_handler=handler,
        )
        executor = ToolExecutor(self.registry, permission_engine=perm_engine)

        call = ToolCall(name="test.risky", arguments={})
        result = executor.execute(call)
        self.assertTrue(result.success)
        self.assertEqual(result.data, {"message": "executed"})

if __name__ == "__main__":
    unittest.main()
