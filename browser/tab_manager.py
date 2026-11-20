"""
Browser Tab Manager for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Detects and focuses pre-opened windows/tabs for services like WhatsApp, YouTube, and Spotify.
Prevents creating unwanted duplicate tabs in user browsers (Chrome, Edge, Brave, etc.).
Strictly avoids synthetic keystroke injection (Ctrl+Shift+A, Ctrl+L, Ctrl+V) to eliminate race conditions.
"""
import logging
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BrowserTabManager:
    """
    Intelligent tab manager that detects pre-opened windows/tabs in browsers (Chrome, Edge, Brave, etc.)
    and switches to them in-place instead of creating duplicate tabs.
    """

    SERVICE_KEYWORDS: Dict[str, List[str]] = {
        "whatsapp": ["whatsapp", "web.whatsapp.com"],
        "youtube": ["youtube", "youtu.be"],
        "spotify": ["spotify", "open.spotify.com"],
    }

    def __init__(self):
        self._platform = None

    @property
    def platform(self):
        if self._platform is None:
            try:
                from platform_layer import get_platform
                self._platform = get_platform()
            except Exception:
                from platform_layer.windows.platform_windows import WindowsPlatform
                self._platform = WindowsPlatform()
        return self._platform

    def find_service_window(self, service: str) -> Optional[Dict[str, Any]]:
        """
        Check if any currently open top-level window has the service in its title or process name.
        """
        if not hasattr(self.platform, "get_open_windows"):
            return None

        clean_svc = service.lower().strip()
        keywords = self.SERVICE_KEYWORDS.get(clean_svc, [clean_svc])
        windows = self.platform.get_open_windows()

        for w in windows:
            title_lower = w.get("title", "").lower()
            proc_lower = w.get("process", "").lower()
            for kw in keywords:
                if kw in title_lower or kw in proc_lower:
                    return w
        return None

    def find_and_focus_service(self, service: str) -> Optional[Dict[str, Any]]:
        """
        Find pre-opened window or tab for service and bring it to foreground cleanly.
        Uses native OS window activation; never injects synthetic hotkeys.
        """
        win = self.find_service_window(service)
        if win:
            if hasattr(self.platform, "focus_window_by_hwnd"):
                self.platform.focus_window_by_hwnd(win["hwnd"])
            return win
        return None

    def trigger_spotify_search_play(self, hwnd: Optional[int] = None):
        """
        Background automation to automatically trigger playback of the Top Result
        after navigating to a search page on Spotify Web.
        """
        import threading

        def _worker():
            try:
                import ctypes
                import win32gui
                from PIL import ImageGrab
                import numpy as np
                from scipy.ndimage import label

                user32 = ctypes.windll.user32
                try:
                    user32.SetProcessDPIAware()
                except Exception:
                    pass
                h_desk = user32.OpenInputDesktop(0, False, 0x01FF)
                if h_desk:
                    user32.SetThreadDesktop(h_desk)

                time.sleep(3.2)

                target_hwnd = hwnd
                if not target_hwnd or not win32gui.IsWindow(target_hwnd):
                    win = self.find_service_window("spotify")
                    if win:
                        target_hwnd = win["hwnd"]

                if not target_hwnd:
                    return

                if hasattr(self.platform, "focus_window_by_hwnd"):
                    self.platform.focus_window_by_hwnd(target_hwnd)
                time.sleep(0.35)

                rect = win32gui.GetWindowRect(target_hwnd)
                w = rect[2] - rect[0]
                h = rect[3] - rect[1]
                if w <= 100 or h <= 100 or rect[0] <= -1000:
                    return

                hover_x = rect[0] + int(w * 0.45)
                hover_y = rect[1] + int(h * 0.40)

                user32.SetCursorPos(hover_x, hover_y)
                time.sleep(0.35)

                clicked = False
                try:
                    img = ImageGrab.grab(bbox=rect)
                    arr = np.array(img.convert("RGB"))
                    mask = (arr[:, :, 0] >= 10) & (arr[:, :, 0] <= 55) & (arr[:, :, 1] >= 170) & (arr[:, :, 1] <= 255) & (arr[:, :, 2] >= 60) & (arr[:, :, 2] <= 135)
                    lbl, num = label(mask)
                    best_target = None
                    for i in range(1, num + 1):
                        c_ys, c_xs = np.where(lbl == i)
                        cx, cy = int(c_xs.mean()), int(c_ys.mean())
                        if len(c_xs) > 400 and cx < int(w * 0.90) and cy < int(h * 0.65):
                            best_target = (rect[0] + cx, rect[1] + cy)
                            break
                    if best_target:
                        user32.SetCursorPos(best_target[0], best_target[1])
                        time.sleep(0.06)
                        user32.mouse_event(0x0002, 0, 0, 0, 0)
                        time.sleep(0.04)
                        user32.mouse_event(0x0004, 0, 0, 0, 0)
                        clicked = True
                except Exception:
                    pass

                if not clicked:
                    user32.SetCursorPos(hover_x, hover_y)
                    time.sleep(0.06)
                    user32.mouse_event(0x0002, 0, 0, 0, 0)
                    time.sleep(0.04)
                    user32.mouse_event(0x0004, 0, 0, 0, 0)
                    time.sleep(0.08)
                    user32.mouse_event(0x0002, 0, 0, 0, 0)
                    time.sleep(0.04)
                    user32.mouse_event(0x0004, 0, 0, 0, 0)
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def navigate_active_tab(self, hwnd: int, target_url: str) -> bool:
        """Navigate window or tab cleanly to target URL."""
        if not target_url:
            return False
        default_browser = getattr(self.platform, "get_default_browser", lambda: "chrome")()
        resolved_exe = getattr(self.platform, "resolve_application_path", lambda _: "")(default_browser)
        if resolved_exe and os.path.isabs(resolved_exe) and os.path.exists(resolved_exe):
            subprocess.Popen([resolved_exe, target_url])
            return True
        elif sys.platform.startswith("win"):
            os.startfile(target_url)
            return True
        else:
            import webbrowser
            webbrowser.open(target_url)
            return True

    def open_or_reuse(self, service: str, target_url: str = "", force_navigate: Optional[bool] = None) -> Dict[str, Any]:
        """
        Main entry point for service window/tab reuse.
        If service is already pre-opened in a window/tab:
        - Brings it to the foreground cleanly.
        - Navigates if a non-generic destination/query URL is provided.
        - Returns reused=True to prevent opening duplicate windows/tabs.
        If service is NOT open:
        - Falls back to opening destination in default browser.
        """
        clean_svc = service.lower().strip()
        win = self.find_and_focus_service(clean_svc)

        if win:
            hwnd = win.get("hwnd") if isinstance(win, dict) else None
            title = win.get("title", "") if isinstance(win, dict) else ""

            # Check if this is a generic homepage
            is_generic = True
            if target_url:
                target_clean = target_url.strip().lower().rstrip("/")
                generic_urls = (
                    "https://web.whatsapp.com", "http://web.whatsapp.com",
                    "https://www.youtube.com", "http://www.youtube.com", "https://youtube.com",
                    "https://open.spotify.com", "http://open.spotify.com",
                )
                is_generic = target_clean in generic_urls

            navigated = False
            should_navigate = force_navigate if force_navigate is not None else (not is_generic and bool(target_url))
            if should_navigate and target_url:
                navigated = bool(self.navigate_active_tab(hwnd, target_url))
                if clean_svc == "spotify" and "/search/" in target_url:
                    self.trigger_spotify_search_play(hwnd)

            return {
                "success": True,
                "reused": True,
                "service": clean_svc,
                "hwnd": hwnd,
                "title": title,
                "navigated": navigated,
                "message": f"Activated pre-opened {clean_svc.capitalize()} window.",
            }

        # Service is not pre-opened: launch via default browser
        if target_url:
            import webbrowser
            webbrowser.open(target_url)

        return {
            "success": True if target_url else False,
            "reused": False,
            "service": clean_svc,
            "message": f"Opened {clean_svc.capitalize()} in browser." if target_url else f"{clean_svc.capitalize()} is not pre-opened.",
        }


# Global singleton instance
browser_tab_manager = BrowserTabManager()
