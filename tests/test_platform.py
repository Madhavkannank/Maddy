"""
Unit tests for PlatformInterface and platform OS detection.
"""
import sys
import unittest
import importlib.util
import pathlib

from platform_layer import LinuxPlatform, WindowsPlatform, get_platform

class TestPlatformInterface(unittest.TestCase):

    def test_get_platform_returns_driver(self):
        plat = get_platform()
        if sys.platform.startswith("win"):
            self.assertIsInstance(plat, WindowsPlatform)
            self.assertEqual(plat.get_os_name(), "windows")
        else:
            self.assertIsInstance(plat, LinuxPlatform)
            self.assertEqual(plat.get_os_name(), "linux")

    def test_phase7_application_control(self):
        plat = get_platform()
        # Verify checking non-existent application process
        self.assertFalse(plat.is_application_running("non_existent_fake_app_99999"))

    def test_brightness_methods(self):
        plat = get_platform()
        if sys.platform.startswith("win"):
            # Verify interface exposes brightness controls
            self.assertTrue(hasattr(plat, "set_brightness"))
            self.assertTrue(hasattr(plat, "get_brightness"))

    def test_maddy_brightness_execution(self):
        from unittest.mock import MagicMock
        from app.maddy_assistant import MaddyAssistant
        assistant = MaddyAssistant()
        mock_plat = MagicMock()
        mock_plat.set_brightness.return_value = True
        mock_plat.get_brightness.return_value = 80
        assistant.platform = mock_plat

        res_max = assistant.execute("I need maximum screen brightness.")
        self.assertTrue(res_max["success"])
        self.assertIn("maximum", res_max["response"].lower())
        mock_plat.set_brightness.assert_called_with(100)

        res_50 = assistant.execute("Can you reduce the screen brightness to 50%?")
        self.assertTrue(res_50["success"])
        self.assertIn("50%", res_50["response"])
        mock_plat.set_brightness.assert_called_with(50)

    def test_volume_methods(self):
        plat = get_platform()
        if sys.platform.startswith("win"):
            self.assertTrue(hasattr(plat, "set_volume"))
            self.assertTrue(hasattr(plat, "get_volume"))
            self.assertTrue(hasattr(plat, "set_mute"))

    def test_maddy_volume_execution(self):
        from unittest.mock import MagicMock
        from app.maddy_assistant import MaddyAssistant
        assistant = MaddyAssistant()
        mock_plat = MagicMock()
        mock_plat.set_volume.return_value = True
        mock_plat.get_volume.return_value = 50
        mock_plat.set_mute.return_value = True
        assistant.platform = mock_plat

        res_max = assistant.execute("increase volume to maximum.")
        self.assertTrue(res_max["success"])
        self.assertIn("maximum", res_max["response"].lower())
        mock_plat.set_volume.assert_called_with(100)

        res_50 = assistant.execute("reduce volume to 50%")
        self.assertTrue(res_50["success"])
        self.assertIn("50%", res_50["response"])
        mock_plat.set_volume.assert_called_with(50)

        res_mute = assistant.execute("mute")
        self.assertTrue(res_mute["success"])
        self.assertIn("muted", res_mute["response"].lower())
        mock_plat.set_mute.assert_called_with(True)

        res_unmute = assistant.execute("unmute")
        self.assertTrue(res_unmute["success"])
        self.assertIn("unmuted", res_unmute["response"].lower())
        mock_plat.set_mute.assert_called_with(False)

if __name__ == "__main__":
    unittest.main()
