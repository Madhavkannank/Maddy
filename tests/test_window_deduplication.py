"""
Unit tests for Window & Application Deduplication and Focus Switching.
Ensures that if an application, website, or browser is already open,
M.A.D.D.Y. brings the existing window to focus rather than launching duplicate instances.
"""
import unittest
from unittest.mock import MagicMock, patch
from app.maddy_assistant import MaddyAssistant

class TestWindowDeduplication(unittest.TestCase):

    def setUp(self):
        self.mock_plat = MagicMock()
        self.patcher = patch("platform_layer.get_platform", return_value=self.mock_plat)
        self.patcher.start()
        self.assistant = MaddyAssistant()
        self.assistant.platform = self.mock_plat

    def tearDown(self):
        self.patcher.stop()

    def test_app_launch_deduplication_when_already_open(self):
        """When an application is already open, it must switch to it without launching a new process."""
        self.mock_plat.focus_window_or_application.return_value = True

        res = self.assistant.execute("open notepad", speak_response=False)
        self.assertTrue(res["success"])
        self.mock_plat.focus_window_or_application.assert_called_with("notepad")
        self.mock_plat.open_application.assert_not_called()
        self.assertTrue(any(w in res["response"].lower() for w in ["switched", "open", "notepad"]))

    def test_app_launch_when_not_already_open(self):
        """When an application is not already open, it should launch it."""
        self.mock_plat.focus_window_or_application.return_value = False
        self.mock_plat.is_application_available.return_value = True

        res = self.assistant.execute("open calculator", speak_response=False)
        self.assertTrue(res["success"])
        self.mock_plat.focus_window_or_application.assert_called_with("calculator")
        self.mock_plat.open_application.assert_called_with("calculator")

    def test_browser_page_deduplication_when_already_open(self):
        """When a website/page is already open in a window, focus it without opening duplicate tabs."""
        from core.intent_taxonomy import IntentPayload, SemanticIntent
        self.mock_plat.focus_window_or_application.return_value = True
        self.assistant.executor.execute = MagicMock()
        mock_disp = MagicMock()
        mock_disp.dispatch.return_value = IntentPayload(intent=SemanticIntent.UNKNOWN)
        self.assistant.dispatcher = mock_disp
        self.assistant.semantic_dispatcher = mock_disp

        res = self.assistant.execute("open wikipedia", speak_response=False)
        self.assertTrue(res["success"])
        self.mock_plat.focus_window_or_application.assert_called_with("Wikipedia")
        self.assistant.executor.execute.assert_not_called()
        self.assertTrue(any(w in res["response"].lower() for w in ["switched", "open", "wikipedia"]))

    def test_browser_window_deduplication_when_already_open(self):
        """When user says 'open chrome' and Chrome is already running, focus Chrome window."""
        self.mock_plat.focus_window_or_application.return_value = True
        self.assistant.executor.execute = MagicMock()

        res = self.assistant.execute("open chrome", speak_response=False)
        self.assertTrue(res["success"])
        self.mock_plat.focus_window_or_application.assert_called_with("chrome")
        self.assistant.executor.execute.assert_not_called()

    def test_spotify_open_deduplication_when_already_open(self):
        """When user says 'open spotify' and Spotify is already running, switch to it."""
        self.mock_plat.focus_window_or_application.return_value = True

        res = self.assistant.execute("open spotify", speak_response=False)
        self.assertTrue(res["success"])
        self.mock_plat.focus_window_or_application.assert_called_with("spotify")
        self.mock_plat.open_application.assert_not_called()
        self.assertIn("switched", res["response"].lower())

    def test_switch_to_command_variation(self):
        """Commands like 'switch to notepad' or 'bring up terminal' should focus window."""
        self.mock_plat.focus_window_or_application.return_value = True

        res = self.assistant.execute("switch to notepad", speak_response=False)
        self.assertTrue(res["success"])
        self.mock_plat.focus_window_or_application.assert_called_with("notepad")
        self.mock_plat.open_application.assert_not_called()

if __name__ == "__main__":
    unittest.main()
