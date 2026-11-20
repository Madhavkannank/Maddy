"""
Unit tests for RiskEvaluator.
"""
import unittest
from security.evaluator import RiskEvaluator
from tools.base import RiskLevel, Tool
from tools.call import ToolCall

def dummy_handler(args):
    return {}

class TestRiskEvaluator(unittest.TestCase):

    def setUp(self):
        self.low_tool = Tool(
            name="test.read",
            description="Low risk read",
            parameter_schema={"type": "object", "properties": {"path": {"type": "string"}}},
            handler=dummy_handler,
            risk_level=RiskLevel.LOW,
        )
        self.high_tool = Tool(
            name="test.sys",
            description="High risk sys action",
            parameter_schema={"type": "object", "properties": {}},
            handler=dummy_handler,
            risk_level=RiskLevel.HIGH,
        )

    def test_low_risk_evaluation(self):
        call = ToolCall(name="test.read", arguments={"path": "data/sample.txt"})
        evaluation = RiskEvaluator.evaluate(call, self.low_tool)
        self.assertEqual(evaluation.risk_level, RiskLevel.LOW)
        self.assertFalse(evaluation.requires_confirmation)

    def test_system_path_escalates_risk_to_high(self):
        call = ToolCall(name="test.read", arguments={"path": "/etc/passwd"})
        evaluation = RiskEvaluator.evaluate(call, self.low_tool)
        self.assertEqual(evaluation.risk_level, RiskLevel.HIGH)
        self.assertTrue(evaluation.requires_confirmation)

    def test_high_risk_tool_requires_confirmation(self):
        call = ToolCall(name="test.sys", arguments={})
        evaluation = RiskEvaluator.evaluate(call, self.high_tool)
        self.assertEqual(evaluation.risk_level, RiskLevel.HIGH)
        self.assertTrue(evaluation.requires_confirmation)

if __name__ == "__main__":
    unittest.main()
