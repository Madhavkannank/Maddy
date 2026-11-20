"""
Unit tests for Phase 7 Computer & Browser Automation System.
"""
import os
import tempfile
import unittest

from browser import (
    BrowserEngine,
    browser_extract_tool,
    browser_navigate_tool,
    browser_search_tool,
)
from computer import (
    ComputerController,
    computer_click_tool,
    computer_screenshot_tool,
    computer_type_tool,
)
import importlib.util
import pathlib
import sys

from platform_layer import LinuxPlatform, WindowsPlatform, get_platform
from security import PermissionEngine
from security.confirmation import AutoApproveLowRiskConfirmationHandler
from tools import register_default_tools
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry

class TestPhase7ComputerBrowser(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()
        register_default_tools(self.registry)
        self.perm_engine = PermissionEngine(confirmation_handler=AutoApproveLowRiskConfirmationHandler())
        self.executor = ToolExecutor(self.registry, permission_engine=self.perm_engine)
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_browser_tools_registration(self):
        tool_names = [t.name for t in self.registry.list_tools()]
        self.assertIn("browser.navigate", tool_names)
        self.assertIn("browser.search", tool_names)
        self.assertIn("browser.extract_content", tool_names)
        self.assertIn("computer.screenshot", tool_names)
        self.assertIn("computer.click", tool_names)
        self.assertIn("computer.type", tool_names)

    def test_computer_screenshot_tool_execution(self):
        shot_path = os.path.join(self.temp_dir.name, "test_shot.png")
        call = {
            "tool": "computer.screenshot",
            "arguments": {"path": shot_path},
        }
        res = self.executor.execute(call)
        self.assertTrue(res.success)
        self.assertTrue(os.path.exists(shot_path))
        self.assertGreater(os.path.getsize(shot_path), 0)

    def test_computer_click_and_type_tools(self):
        click_call = {
            "tool": "computer.click",
            "arguments": {"x": 100, "y": 200},
        }
        click_res = self.executor.execute(click_call)
        self.assertTrue(click_res.success)
        self.assertEqual(click_res.data["x"], 100)

        type_call = {
            "tool": "computer.type",
            "arguments": {"text": "M.A.D.D.Y. Automation"},
        }
        type_res = self.executor.execute(type_call)
        self.assertTrue(type_res.success)
        self.assertEqual(type_res.data["text_length"], len("M.A.D.D.Y. Automation"))

    def test_platform_application_lifecycle_interface(self):
        plat = get_platform()
        os_name = plat.get_os_name()
        self.assertIn(os_name, ("linux", "windows"))

        # Test checking non-existent application process
        self.assertFalse(plat.is_application_running("non_existent_fake_app_12345"))

if __name__ == "__main__":
    unittest.main()
