"""
Windows platform driver implementation for M.A.D.D.Y.
Handles Windows application management and system operations safely.
"""
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional
from ..interface import PlatformInterface

class WindowsPlatform(PlatformInterface):
    """Windows-specific platform abstraction implementation."""

    def get_os_name(self) -> str:
        return "windows"

    KNOWN_APP_PATHS = {
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "firefox": [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ],
        "edge": [
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ],
        "spotify": [
            os.path.expandvars(r"%APPDATA%\Spotify\Spotify.exe"),
            os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps\Spotify.exe"),
        ],
    }

    def get_default_browser(self) -> str:
        """Determine default or primary installed browser on Windows by inspecting UserChoice registry."""
        try:
            import winreg
            k = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice",
            )
            progid = str(winreg.QueryValueEx(k, "ProgId")[0]).lower()
            if "firefox" in progid:
                return "Firefox"
            elif "chrome" in progid:
                return "Chrome"
            elif "edge" in progid or "msedge" in progid:
                return "Edge"
            elif "brave" in progid:
                return "Brave"
        except Exception:
            pass

        for name, label in [("edge", "Edge"), ("firefox", "Firefox"), ("chrome", "Chrome")]:
            for candidate in self.KNOWN_APP_PATHS.get(name, []):
                if os.path.exists(candidate):
                    return label
        return "Edge"

    SYSTEM_APP_ALIASES = {
        "notepad": "notepad",
        "note pad": "notepad",
        "calculator": "calc",
        "calc": "calc",
        "terminal": "wt",
        "windows terminal": "wt",
        "cmd": "cmd",
        "command prompt": "cmd",
        "powershell": "powershell",
        "file explorer": "explorer",
        "explorer": "explorer",
        "my computer": "explorer",
        "this pc": "explorer",
        "paint": "mspaint",
        "settings": "ms-settings:",
        "control panel": "control",
        "task manager": "taskmgr",
        "taskmgr": "taskmgr",
        "regedit": "regedit",
        "vs code": "code",
        "vscode": "code",
        "visual studio code": "code",
        "camera": "microsoft.windows.camera:",
        "webcam": "microsoft.windows.camera:",
        "windows camera": "microsoft.windows.camera:",
        "photos": "ms-photos:",
        "clock": "ms-clock:",
        "alarms": "ms-clock:",
    }

    def resolve_application_path(self, app_name: str) -> str:
        """Resolve executable path, shortcut, or command for standard applications."""
        clean = app_name.strip().lower()
        if not clean:
            return ""
        if clean in ("browser", "default browser"):
            clean = self.get_default_browser()

        # 1. Check known aliases
        if clean in self.SYSTEM_APP_ALIASES:
            alias_val = self.SYSTEM_APP_ALIASES[clean]
            if alias_val.startswith(("ms-", "microsoft.windows.")) or shutil.which(alias_val) or shutil.which(f"{alias_val}.exe"):
                return alias_val

        # 2. Check known path tables
        if clean in self.KNOWN_APP_PATHS:
            for p in self.KNOWN_APP_PATHS[clean]:
                if os.path.exists(p):
                    return p

        # Check with .exe stripped or appended
        base = clean.replace(".exe", "")
        if base in self.KNOWN_APP_PATHS:
            for p in self.KNOWN_APP_PATHS[base]:
                if os.path.exists(p):
                    return p

        # 3. Check system PATH
        which_path = shutil.which(clean) or shutil.which(f"{clean}.exe")
        if which_path:
            return which_path

        # 4. Check Windows Start Menu shortcuts (.lnk)
        import glob
        shortcut_dirs = [
            r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
        ]
        for sdir in shortcut_dirs:
            if os.path.exists(sdir):
                for lnk in glob.glob(sdir + "/**/*.lnk", recursive=True):
                    lnk_name = os.path.basename(lnk).lower().replace(".lnk", "")
                    if clean == lnk_name or (len(clean) >= 3 and clean in lnk_name):
                        return lnk

        # 5. Fallback for Spotify Web Player when desktop app is not installed
        if clean in ("spotify", "spotify.exe", "spotify web"):
            return "https://open.spotify.com"

        return ""

    def is_application_available(self, app_name: str) -> bool:
        """Check if an application is installed or available locally on Windows."""
        if not app_name or not app_name.strip():
            return False
        res = self.resolve_application_path(app_name)
        if not res:
            return False
        if res.startswith(("ms-", "microsoft.windows.")):
            return True
        if os.path.isabs(res) and os.path.exists(res):
            return True
        if shutil.which(res) or shutil.which(f"{res}.exe"):
            return True
        return False

    def open_url(self, url: str) -> bool:
        """Open destination URL in the default browser on Windows."""
        clean_url = (url or "").strip()
        if not clean_url:
            return False
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            clean_url = f"https://{clean_url}"
        try:
            os.startfile(clean_url)
            return True
        except Exception:
            try:
                import webbrowser
                return webbrowser.open(clean_url)
            except Exception:
                return False

    def open_file(self, file_path: str) -> bool:
        """Open a local file, image, document, presentation, spreadsheet, archive, or code
        using native OS desktop applications (e.g. Photos, Acrobat/Edge for PDF, PowerPoint for PPT/PPTX,
        Excel for XLSX/CSV, Notepad/VS Code for JSON/TXT/code). Never opens local files in browser tabs.
        """
        clean_path = (file_path or "").strip()
        if not clean_path:
            return False
        if clean_path.startswith("file:///"):
            import urllib.parse
            clean_path = urllib.parse.unquote(clean_path[8:])
        elif clean_path.startswith("file://"):
            import urllib.parse
            clean_path = urllib.parse.unquote(clean_path[7:])

        target = Path(clean_path).resolve()
        if not target.exists():
            return False
        try:
            os.startfile(str(target))
            return True
        except Exception:
            try:
                subprocess.Popen(f'start "" "{str(target)}"', shell=True)
                return True
            except Exception:
                return False

    def open_application(self, app_name: str) -> None:
        """Open an application by name or path on Windows."""
        resolved = self.resolve_application_path(app_name)
        try:
            if resolved.startswith(("http://", "https://", "ms-")) or resolved.endswith(".lnk"):
                os.startfile(resolved)
            elif os.path.isabs(resolved) and os.path.exists(resolved):
                subprocess.Popen([resolved], shell=False)
            else:
                subprocess.Popen(f'start "" "{resolved}"', shell=True)
        except Exception as err:
            raise RuntimeError(f"Could not launch application '{app_name}' on Windows: {str(err)}")

    UWP_PROCESS_NAMES = {
        "camera": "WindowsCamera.exe",
        "webcam": "WindowsCamera.exe",
        "windowscamera": "WindowsCamera.exe",
        "photos": "Microsoft.Photos.exe",
        "clock": "Time.exe",
        "settings": "SystemSettings.exe",
    }

    def close_application(self, app_name: str) -> None:
        """Terminate a running process by image name on Windows."""
        clean = app_name.strip().lower()
        if clean in self.UWP_PROCESS_NAMES:
            clean_name = self.UWP_PROCESS_NAMES[clean]
        else:
            clean_name = os.path.basename(app_name.strip())
            low = clean_name.lower()
            if low in ("edge", "msedge"):
                clean_name = "msedge.exe"
            elif low in ("task manager", "taskmanager", "taskmgr"):
                clean_name = "Taskmgr.exe"
            elif low in ("chrome", "google chrome"):
                clean_name = "chrome.exe"
            elif low in ("antigravity",):
                clean_name = "Antigravity.exe"
            elif not clean_name.endswith(".exe"):
                clean_name += ".exe"

        try:
            res = subprocess.run(["taskkill", "/F", "/IM", clean_name], capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"No running process found for '{app_name}' on Windows")
        except Exception as err:
            raise RuntimeError(f"Failed to close application '{app_name}' on Windows: {str(err)}")

    def is_application_running(self, app_name: str) -> bool:
        """Check if process is currently running on Windows."""
        clean = app_name.strip().lower()
        if clean in self.UWP_PROCESS_NAMES:
            clean_name = self.UWP_PROCESS_NAMES[clean].lower()
        else:
            clean_name = os.path.basename(clean)
            if clean_name in ("browser", "default browser"):
                clean_name = self.get_default_browser()

            if clean_name in ("edge", "msedge"):
                clean_name = "msedge.exe"
            elif not clean_name.endswith(".exe"):
                clean_name += ".exe"

        try:
            import psutil
            for proc in psutil.process_iter(['name']):
                p_name = proc.info.get('name')
                if p_name and p_name.lower() == clean_name:
                    return True
        except Exception:
            pass

        try:
            res = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {clean_name}"], capture_output=True, text=True)
            return clean_name.lower() in res.stdout.lower() and "no tasks are running" not in res.stdout.lower()
        except Exception:
            return False

    def get_open_windows(self) -> list:
        """
        Enumerate all open and visible windows on the interactive Windows desktop.
        Returns list of dicts: [{'hwnd': int, 'title': str, 'pid': int, 'process': str}]
        """
        import ctypes
        from ctypes import wintypes
        import psutil

        user32 = ctypes.windll.user32
        try:
            # Ensure calling thread is attached to interactive desktop
            h_desk = user32.OpenInputDesktop(0, False, 0x01FF)
            if h_desk:
                user32.SetThreadDesktop(h_desk)
        except Exception:
            pass

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        windows = []

        def enum_cb(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.strip()
            if not title:
                return True

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pname = ""
            try:
                pname = psutil.Process(pid.value).name().lower()
            except Exception:
                pass

            windows.append({
                "hwnd": hwnd,
                "title": title,
                "pid": pid.value,
                "process": pname,
            })
            return True

        try:
            user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        except Exception:
            pass
        return windows

    def get_active_window(self) -> Optional[dict]:
        """Retrieve title, process and bounds of the current active window on Windows."""
        try:
            import ctypes
            from ctypes import wintypes
            import psutil

            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None

            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value.strip() or "Desktop"

            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pname = ""
            try:
                pname = psutil.Process(pid.value).name().lower()
            except Exception:
                pass

            w = max(0, rect.right - rect.left)
            h = max(0, rect.bottom - rect.top)

            return {
                "id": hwnd,
                "hwnd": hwnd,
                "title": title,
                "pid": pid.value,
                "process": pname,
                "bounds": {"x": rect.left, "y": rect.top, "width": w, "height": h},
            }
        except Exception:
            return {
                "id": 0,
                "hwnd": 0,
                "title": "Active Desktop Screen",
                "pid": 0,
                "process": "",
                "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
            }

    def close_window(self, target: str) -> bool:
        """Close an open window matching target on Windows by sending WM_CLOSE."""
        clean = target.strip().lower()
        if not clean:
            return False

        open_windows = self.get_open_windows()
        for w in open_windows:
            if clean in w["title"].lower() or clean in w.get("process", "").lower():
                try:
                    import ctypes
                    WM_CLOSE = 0x0010
                    ctypes.windll.user32.PostMessageW(w["hwnd"], WM_CLOSE, 0, 0)
                    return True
                except Exception:
                    pass

        try:
            self.close_application(target)
            return True
        except Exception:
            return False

    def discover_applications(self) -> list:
        """Discover installed or available applications on Windows."""
        discovered = []
        seen = set()

        # 1. System app aliases
        for name, cmd in self.SYSTEM_APP_ALIASES.items():
            if name not in seen:
                seen.add(name)
                discovered.append({
                    "name": name.title(),
                    "exec": cmd,
                    "category": "System",
                    "path": cmd,
                })

        # 2. Known app paths
        for name, paths in self.KNOWN_APP_PATHS.items():
            for p in paths:
                if os.path.exists(p) and name not in seen:
                    seen.add(name)
                    discovered.append({
                        "name": name.title(),
                        "exec": p,
                        "category": "Applications",
                        "path": p,
                    })

        # 3. Start menu shortcuts
        import glob
        shortcut_dirs = [
            r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs",
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"),
        ]
        for sdir in shortcut_dirs:
            if os.path.exists(sdir):
                for lnk in glob.glob(sdir + "/**/*.lnk", recursive=True):
                    lnk_name = os.path.basename(lnk).replace(".lnk", "")
                    clean_k = lnk_name.lower()
                    if clean_k not in seen:
                        seen.add(clean_k)
                        discovered.append({
                            "name": lnk_name,
                            "exec": lnk,
                            "category": "StartMenu",
                            "path": lnk,
                        })

        return discovered

    def get_screen_geometry(self) -> dict:
        """Return screen dimensions on Windows."""
        try:
            import ctypes
            u = ctypes.windll.user32
            w = u.GetSystemMetrics(0)
            h = u.GetSystemMetrics(1)
            return {"width": w if w > 0 else 1920, "height": h if h > 0 else 1080, "scale_factor": 1.0}
        except Exception:
            return {"width": 1920, "height": 1080, "scale_factor": 1.0}

    def take_screenshot(self, output_path: Optional[str] = None) -> dict:
        """Capture screen display image and save to output_path."""
        from computer.controller import ComputerController
        ctrl = ComputerController()
        return ctrl.take_screenshot(output_path)

    def move_cursor(self, x: int, y: int, smooth: bool = True) -> dict:
        """Move mouse cursor to (x, y)."""
        from computer.controller import ComputerController
        ctrl = ComputerController()
        return ctrl.move_cursor(x, y, smooth=smooth)

    def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> dict:
        """Simulate mouse click."""
        from computer.controller import ComputerController
        ctrl = ComputerController()
        return ctrl.click(x=x, y=y, button=button, clicks=clicks)

    def scroll_mouse(self, direction: str = "down", amount: int = 3) -> dict:
        """Scroll mouse wheel."""
        from computer.controller import ComputerController
        ctrl = ComputerController()
        return ctrl.scroll(direction=direction, amount=amount)

    def type_text(self, text: str, press_enter: bool = False) -> dict:
        """Type text into active foreground window."""
        from computer.controller import ComputerController
        ctrl = ComputerController()
        return ctrl.type_text(text, press_enter=press_enter)

    def press_key(self, key_combination: str) -> dict:
        """Press key or key combination."""
        from computer.controller import ComputerController
        ctrl = ComputerController()
        return ctrl.press_key(key_combination)

    def focus_window_by_hwnd(self, hwnd: int) -> bool:
        """
        Restore (if minimized) and bring an existing window to the foreground.
        Uses AttachThreadInput + Alt-key bypass + ShowWindow(SW_RESTORE) + SwitchToThisWindow + SetForegroundWindow.
        """
        import ctypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        try:
            SW_RESTORE = 9
            user32.ShowWindow(hwnd, SW_RESTORE)

            # Windows 11 foreground activation requires attaching input thread to target and foreground window
            curr_tid = kernel32.GetCurrentThreadId()
            fg_hwnd = user32.GetForegroundWindow()
            fg_tid = user32.GetWindowThreadProcessId(fg_hwnd, None) if fg_hwnd else 0
            target_tid = user32.GetWindowThreadProcessId(hwnd, None)

            attached_fg = False
            if fg_tid and fg_tid != curr_tid:
                attached_fg = bool(user32.AttachThreadInput(curr_tid, fg_tid, True))
            attached_target = False
            if target_tid and target_tid != curr_tid:
                attached_target = bool(user32.AttachThreadInput(curr_tid, target_tid, True))

            # Windows Alt-key bypass allows background threads to successfully set foreground
            VK_MENU = 0x12
            KEYEVENTF_KEYUP = 0x0002
            user32.keybd_event(VK_MENU, 0, 0, 0)
            user32.SetForegroundWindow(hwnd)
            user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)

            user32.BringWindowToTop(hwnd)
            user32.SetFocus(hwnd)
            user32.SwitchToThisWindow(hwnd, True)

            if attached_target:
                user32.AttachThreadInput(curr_tid, target_tid, False)
            if attached_fg:
                user32.AttachThreadInput(curr_tid, fg_tid, False)
            return True
        except Exception:
            try:
                import win32com.client
                shell = win32com.client.Dispatch("WScript.Shell")
                shell.AppActivate(hwnd)
                return True
            except Exception:
                return False

    def focus_window_or_application(self, target: str) -> bool:
        """
        Check if an application or window matching target (app name, process name, or page/window title)
        is already open. If so, bring it to the foreground and return True.
        """
        clean_target = target.strip().lower()
        if not clean_target:
            return False

        # Map common aliases to process and window keywords
        alias_map = {
            "calculator": ["calc", "calculator"],
            "notepad": ["notepad"],
            "vs code": ["code", "visual studio code"],
            "vscode": ["code", "visual studio code"],
            "terminal": ["windowsterminal", "cmd", "powershell", "terminal"],
            "browser": ["chrome", "edge", "firefox", "msedge"],
            "chrome": ["chrome"],
            "google chrome": ["chrome"],
            "edge": ["msedge", "edge"],
            "firefox": ["firefox"],
            "spotify": ["spotify"],
            "youtube": ["youtube"],
            "amazon": ["amazon"],
            "wikipedia": ["wikipedia"],
            "github": ["github"],
            "gmail": ["gmail", "google mail"],
            "antigravity": ["antigravity"],
            "task manager": ["taskmgr", "task manager"],
            "taskmgr": ["taskmgr", "task manager"],
        }
        keywords = alias_map.get(clean_target, [clean_target])

        open_windows = self.get_open_windows()

        # 1. Match against window title (handles browser tabs/pages like "YouTube - Google Chrome", "Spotify - Web Player")
        for kw in keywords:
            for w in open_windows:
                w_title_lower = w["title"].lower()
                if kw in w_title_lower:
                    if self.focus_window_by_hwnd(w["hwnd"]):
                        return True

        # 2. Match against process name (e.g. notepad.exe, spotify.exe, code.exe)
        for kw in keywords:
            for w in open_windows:
                pname = w.get("process", "").lower()
                if kw in pname or pname.startswith(kw):
                    if self.focus_window_by_hwnd(w["hwnd"]):
                        return True

        return False

    def speak(self, text: str) -> None:
        """Synthesize text-to-speech output on Windows using neural VoiceEngine with fallback."""
        clean_text = text.strip()
        try:
            from voice.engine import VoiceEngine
            ve = VoiceEngine(platform_driver=self)
            res = ve.speak(clean_text)
            if res.get("success"):
                return
        except Exception:
            pass

        escaped_text = clean_text.replace('"', '\\"').replace("'", "''")
        ps_script = f'Add-Type -AssemblyName System.Speech; $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; $synth.Speak("{escaped_text}")'
        try:
            subprocess.run(["powershell", "-Command", ps_script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception as err:
            raise RuntimeError(f"Windows TTS failed: {str(err)}")

    def set_brightness(self, level: int) -> bool:
        """Set display brightness percentage (0-100) on Windows via WMI."""
        try:
            target = max(0, min(100, int(level)))
            cmd = f"(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, {target})"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=5)
            return res.returncode == 0
        except Exception:
            return False

    def get_brightness(self) -> Optional[int]:
        """Get current display brightness percentage (0-100) on Windows via WMI."""
        try:
            cmd = "(Get-WmiObject -Namespace root/wmi -Class WmiMonitorBrightness).CurrentBrightness"
            res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=5)
            if res.returncode == 0 and res.stdout.strip():
                return int(res.stdout.strip().splitlines()[-1])
        except Exception:
            pass
        return None

    def set_volume(self, level: int) -> bool:
        """Set master audio volume level (0-100) on Windows."""
        try:
            target = max(0, min(100, int(level)))
            from pycaw.pycaw import AudioUtilities
            device = AudioUtilities.GetSpeakers()
            volume = device.EndpointVolume
            volume.SetMasterVolumeLevelScalar(target / 100.0, None)
            return True
        except Exception:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                if level >= 100:
                    for _ in range(50):
                        user32.keybd_event(0xAF, 0, 0, 0)
                        user32.keybd_event(0xAF, 0, 2, 0)
                    return True
                elif level <= 0:
                    for _ in range(50):
                        user32.keybd_event(0xAE, 0, 0, 0)
                        user32.keybd_event(0xAE, 0, 2, 0)
                    return True
            except Exception:
                pass
            return False

    def get_volume(self) -> Optional[int]:
        """Get current master audio volume level (0-100) on Windows."""
        try:
            from pycaw.pycaw import AudioUtilities
            device = AudioUtilities.GetSpeakers()
            volume = device.EndpointVolume
            scalar = volume.GetMasterVolumeLevelScalar()
            return int(round(scalar * 100))
        except Exception:
            return None

    def set_mute(self, mute: bool) -> bool:
        """Mute or unmute master audio on Windows."""
        try:
            from pycaw.pycaw import AudioUtilities
            device = AudioUtilities.GetSpeakers()
            volume = device.EndpointVolume
            volume.SetMute(1 if mute else 0, None)
            return True
        except Exception:
            try:
                import ctypes
                user32 = ctypes.windll.user32
                user32.keybd_event(0xAD, 0, 0, 0)
                user32.keybd_event(0xAD, 0, 2, 0)
                return True
            except Exception:
                return False
