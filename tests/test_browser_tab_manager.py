"""
Unit tests for BrowserTabManager (detecting and reusing pre-opened tabs for WhatsApp, YouTube, Spotify).
"""
import unittest
from unittest.mock import MagicMock, patch

from browser.tab_manager import BrowserTabManager


class TestBrowserTabManager(unittest.TestCase):

    def setUp(self):
        self.mgr = BrowserTabManager()
        self.mock_platform = MagicMock()
        self.mgr._platform = self.mock_platform

    def test_find_service_window_whatsapp_unread(self):
        self.mock_platform.get_open_windows.return_value = [
            {"hwnd": 101, "title": "(4) WhatsApp - Google Chrome", "pid": 1234, "process": "chrome.exe"},
            {"hwnd": 102, "title": "Settings", "pid": 5678, "process": "systemsettings.exe"},
        ]
        win = self.mgr.find_service_window("whatsapp")
        self.assertIsNotNone(win)
        self.assertEqual(win["hwnd"], 101)

    def test_find_service_window_youtube(self):
        self.mock_platform.get_open_windows.return_value = [
            {"hwnd": 201, "title": "Interstellar Main Theme - YouTube - Google Chrome", "pid": 1234, "process": "chrome.exe"},
        ]
        win = self.mgr.find_service_window("youtube")
        self.assertIsNotNone(win)
        self.assertEqual(win["hwnd"], 201)

    def test_find_service_window_spotify(self):
        self.mock_platform.get_open_windows.return_value = [
            {"hwnd": 301, "title": "Spotify - Web Player: Music for everyone - Google Chrome", "pid": 1234, "process": "chrome.exe"},
        ]
        win = self.mgr.find_service_window("spotify")
        self.assertIsNotNone(win)
        self.assertEqual(win["hwnd"], 301)

    def test_find_service_window_not_found(self):
        self.mock_platform.get_open_windows.return_value = [
            {"hwnd": 401, "title": "Notepad", "pid": 9999, "process": "notepad.exe"},
        ]
        win = self.mgr.find_service_window("spotify")
        self.assertIsNone(win)

    def test_open_or_reuse_existing_tab_no_new_tab(self):
        self.mock_platform.get_open_windows.return_value = [
            {"hwnd": 101, "title": "WhatsApp - Google Chrome", "pid": 1234, "process": "chrome.exe"},
        ]
        with patch.object(self.mgr, "navigate_active_tab") as mock_nav, \
             patch("webbrowser.open") as mock_webbrowser:
            
            res = self.mgr.open_or_reuse("whatsapp", "https://web.whatsapp.com/send?phone=919876543210&text=Hi")
            
            self.assertTrue(res["success"])
            self.assertTrue(res["reused"])
            self.assertEqual(res["hwnd"], 101)
            self.mock_platform.focus_window_by_hwnd.assert_called_with(101)
            mock_nav.assert_called_once_with(101, "https://web.whatsapp.com/send?phone=919876543210&text=Hi")
            mock_webbrowser.assert_not_called()

    def test_open_or_reuse_generic_home_does_not_reload(self):
        self.mock_platform.get_open_windows.return_value = [
            {"hwnd": 301, "title": "Spotify - Web Player", "pid": 1234, "process": "chrome.exe"},
        ]
        with patch.object(self.mgr, "navigate_active_tab") as mock_nav, \
             patch("webbrowser.open") as mock_webbrowser:
            
            res = self.mgr.open_or_reuse("spotify", "https://open.spotify.com", force_navigate=False)
            
            self.assertTrue(res["success"])
            self.assertTrue(res["reused"])
            self.assertFalse(res["navigated"])
            mock_nav.assert_not_called()
            mock_webbrowser.assert_not_called()

    def test_open_or_reuse_fallback_to_browser_when_not_open(self):
        self.mock_platform.get_open_windows.return_value = [
            {"hwnd": 401, "title": "Calculator", "pid": 8888, "process": "calc.exe"},
        ]
        with patch("webbrowser.open") as mock_webbrowser:
            res = self.mgr.open_or_reuse("youtube", "https://www.youtube.com/results?search_query=test")
            
            self.assertTrue(res["success"])
            self.assertFalse(res["reused"])
            mock_webbrowser.assert_called_once_with("https://www.youtube.com/results?search_query=test")


if __name__ == "__main__":
    unittest.main()
