"""
Unit tests for PermissionEngine and security policies.
"""
import unittest
from security.confirmation import (
    AutoApproveLowRiskConfirmationHandler,
    AutoDenyConfirmationHandler,
    InteractiveConfirmationHandler,
)
from security.permission_engine import PermissionEngine
from security.policy import PolicyMode
from tools.base import RiskLevel, Tool
from tools.call import ToolCall

def dummy_handler(args):
    return {}

class TestPermissionEngine(unittest.TestCase):

    def setUp(self):
        self.low_tool = Tool(
            name="test.low",
            description="Low risk tool",
            parameter_schema={"type": "object", "properties": {}},
            handler=dummy_handler,
            risk_level=RiskLevel.LOW,
        )
        self.high_tool = Tool(
            name="test.high",
            description="High risk tool",
            parameter_schema={"type": "object", "properties": {}},
            handler=dummy_handler,
            risk_level=RiskLevel.HIGH,
        )

    def test_balanced_policy_auto_allows_low_risk(self):
        engine = PermissionEngine(policy_mode=PolicyMode.BALANCED)
        call = ToolCall(name="test.low", arguments={})
        decision = engine.check_permission(call, self.low_tool)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.decision_code, "ALLOWED")

    def test_balanced_policy_denies_unconfirmed_high_risk(self):
        engine = PermissionEngine(
            policy_mode=PolicyMode.BALANCED,
            confirmation_handler=AutoDenyConfirmationHandler(),
        )
        call = ToolCall(name="test.high", arguments={})
        decision = engine.check_permission(call, self.high_tool)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.decision_code, "REQUIRES_USER_CONFIRMATION")

    def test_interactive_confirmation_approved(self):
        handler = InteractiveConfirmationHandler(callback=lambda call, tool, eval: True)
        engine = PermissionEngine(
            policy_mode=PolicyMode.BALANCED,
            confirmation_handler=handler,
        )
        call = ToolCall(name="test.high", arguments={})
        decision = engine.check_permission(call, self.high_tool)
        self.assertTrue(decision.allowed)
        self.assertTrue(decision.requires_confirmation)

    def test_dry_run_mode_blocks_high_risk(self):
        engine = PermissionEngine(policy_mode=PolicyMode.DRY_RUN)
        call = ToolCall(name="test.high", arguments={})
        decision = engine.check_permission(call, self.high_tool)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.decision_code, "DRY_RUN_BLOCKED")

    def test_audit_logs_recorded(self):
        engine = PermissionEngine(policy_mode=PolicyMode.BALANCED)
        call = ToolCall(name="test.low", arguments={})
        engine.check_permission(call, self.low_tool)
        records = engine.audit_logger.get_records()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].tool_name, "test.low")
        self.assertTrue(records[0].allowed)

if __name__ == "__main__":
    unittest.main()
