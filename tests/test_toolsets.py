"""Unit tests for M.A.D.D.Y. toolsets gating architecture.
"""

import unittest
from core.toolsets import (
    _MADDY_CORE_TOOLS,
    TOOLSETS,
    ALL_BUILTIN_TOOLS,
    resolve_active_tool_names,
)


class TestToolsetsGating(unittest.TestCase):

    def setUp(self):
        self.mock_registry = {
            "app_control",
            "web_browser",
            "web_search",
            "system_control",
            "remember",
            "recall",
            "calculator",
            "weather",
            "media_control",
            "system_info",
            "camera_tool",
            "timer_tool",
            "telegram",
            "forget",
            "update_preference",
            "read_file",
            "write_file",
            "list_directory",
            "execute_command",
            "arxiv_tool",
            "computer_screen",
            "computer_interaction",
            "browser_automation",
            "list_skills",
            "search_skills",
            "load_skill",
            "validate_skill",
            "delegate_task",
            "cronjob_manage",
        }

    def test_01_narrow_core_waist_selected_by_default(self):
        """Invariant: General prompt only activates narrow core waist tools."""
        active = resolve_active_tool_names("hello maddy how are you", self.mock_registry)
        for core_tool in _MADDY_CORE_TOOLS:
            self.assertIn(core_tool, active)
        # Domain tools should not be active
        self.assertNotIn("calculator", active)
        self.assertNotIn("weather", active)
        self.assertNotIn("media_control", active)
        self.assertNotIn("system_info", active)
        self.assertNotIn("read_file", active)
        self.assertNotIn("arxiv_tool", active)
        self.assertEqual(len(active), len(_MADDY_CORE_TOOLS))

    def test_02_no_false_positive_substring_matching(self):
        """Invariant: 'add' in 'maddy' or 'math' in 'aftermath' does not trigger calculator."""
        active = resolve_active_tool_names("hey maddy listen to me", self.mock_registry)
        self.assertNotIn("calculator", active)

    def test_03_math_domain_activation(self):
        """Invariant: Quantitative queries activate calculator."""
        active_kw = resolve_active_tool_names("calculate the tip on 50 dollars", self.mock_registry)
        self.assertIn("calculator", active_kw)

        active_expr = resolve_active_tool_names("what is 45 * 12", self.mock_registry)
        self.assertIn("calculator", active_expr)

    def test_04_weather_domain_activation(self):
        """Invariant: Atmospheric/forecast queries activate weather."""
        active = resolve_active_tool_names("is it raining in Seattle today?", self.mock_registry)
        self.assertIn("weather", active)
        self.assertNotIn("calculator", active)

    def test_05_file_system_domain_activation(self):
        """Invariant: File/shell queries activate file manipulation tools."""
        active = resolve_active_tool_names("read the contents of config.json", self.mock_registry)
        self.assertIn("read_file", active)
        self.assertIn("write_file", active)
        self.assertIn("list_directory", active)
        self.assertIn("execute_command", active)

    def test_06_conversational_continuity_preserves_recent_tool_calls(self):
        """Invariant: Tools invoked in recent conversation turns remain active in follow-up turns."""
        history = [
            {"role": "user", "content": "calculate 5 + 5"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "calculator", "arguments": '{"expr": "5 + 5"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "10"},
        ]
        # User says "now multiply that by 2" without explicitly saying "calculate"
        active = resolve_active_tool_names(
            "now do that times 2",
            self.mock_registry,
            conversation_history=history,
        )
        self.assertIn("calculator", active)

    def test_07_custom_injected_tools_preserved(self):
        """Invariant: Custom or test-injected tools outside ALL_BUILTIN_TOOLS are always preserved."""
        registry_with_mock = set(self.mock_registry)
        registry_with_mock.add("custom_test_eval_tool")
        active = resolve_active_tool_names("general query", registry_with_mock)
        self.assertIn("custom_test_eval_tool", active)


if __name__ == "__main__":
    unittest.main()
