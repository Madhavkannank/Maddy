"""
Linux platform driver implementation for M.A.D.D.Y.
Handles Linux application management and system operations safely.
"""
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Optional
from ..interface import PlatformInterface

_PROTECTED_PROCESS_NAMES = frozenset({
    "python", "python3", "pytest", "maddy", "systemd", "init", "sshd", "bash", "zsh", "sh", "xorg", "wayland", "gnome-shell"
})

class LinuxPlatform(PlatformInterface):
    """Linux-specific platform abstraction implementation."""

    def get_os_name(self) -> str:
        return "linux"

    def get_gui_environment(self) -> dict:
        """Auto-discover and construct active Wayland/X11/DBus desktop environment variables."""
        env = dict(os.environ)
        uid = os.getuid()
        runtime_dir = f"/run/user/{uid}"
        if "XDG_RUNTIME_DIR" not in env and os.path.isdir(runtime_dir):
            env["XDG_RUNTIME_DIR"] = runtime_dir

        if "DBUS_SESSION_BUS_ADDRESS" not in env:
            dbus_sock = os.path.join(env.get("XDG_RUNTIME_DIR", runtime_dir), "bus")
            if os.path.exists(dbus_sock):
                env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={dbus_sock}"

        if "WAYLAND_DISPLAY" not in env:
            rt = env.get("XDG_RUNTIME_DIR", runtime_dir)
            for w_sock in ["wayland-0", "wayland-1"]:
                if os.path.exists(os.path.join(rt, w_sock)):
                    env["WAYLAND_DISPLAY"] = w_sock
                    break

        if "DISPLAY" not in env:
            env["DISPLAY"] = ":0"

        # Export to os.environ so Python libraries (e.g. webbrowser) find runnable displays
        for k in ["XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "WAYLAND_DISPLAY", "DISPLAY"]:
            if k in env and k not in os.environ:
                os.environ[k] = env[k]

        return env

    def get_default_browser(self) -> str:
        """Determine default web browser on Linux dynamically via xdg-settings or MIME associations."""
        try:
            out = subprocess.check_output(
                ["xdg-settings", "get", "default-web-browser"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip().lower()
            if "firefox" in out:
                return "Firefox"
            elif "chrome" in out:
                return "Chrome"
            elif "brave" in out:
                return "Brave"
            elif "edge" in out:
                return "Edge"
            elif "chromium" in out:
                return "Chromium"
            elif out:
                clean_name = out.replace(".desktop", "").replace("-", " ").title()
                return clean_name
        except Exception:
            pass

        # Try gio mime query
        try:
            out = subprocess.check_output(
                ["gio", "mime", "x-scheme-handler/http"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip().lower()
            if "firefox" in out:
                return "Firefox"
            elif "chrome" in out:
                return "Chrome"
            elif "brave" in out:
                return "Brave"
            elif "edge" in out:
                return "Edge"
        except Exception:
            pass

        # Fallback to installed browser binaries in priority order
        for bin_name, label in [
            ("firefox", "Firefox"),
            ("google-chrome", "Chrome"),
            ("chromium", "Chromium"),
            ("brave-browser", "Brave"),
        ]:
            if shutil.which(bin_name):
                return label

        return "Firefox"

    def open_url(self, url: str) -> bool:
        """Open destination URL in the desktop browser with verified GUI environment."""
        clean_url = (url or "").strip()
        if not clean_url:
            return False
        if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
            clean_url = f"https://{clean_url}"

        gui_env = self.get_gui_environment()

        # Try xdg-open
        if shutil.which("xdg-open"):
            try:
                subprocess.Popen(
                    ["xdg-open", clean_url],
                    env=gui_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                pass

        # Try gio open
        if shutil.which("gio"):
            try:
                subprocess.Popen(
                    ["gio", "open", clean_url],
                    env=gui_env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return True
            except Exception:
                pass

        # Direct browser binary fallback
        for b in ["firefox", "google-chrome", "google-chrome-stable", "chromium"]:
            bin_path = shutil.which(b)
            if bin_path:
                try:
                    subprocess.Popen(
                        [bin_path, clean_url],
                        env=gui_env,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except Exception:
                    pass

        try:
            import webbrowser
            return webbrowser.open(clean_url)
        except Exception:
            return False

    def open_file(self, file_path: str) -> bool:
        """Open a local file, image, document, presentation, spreadsheet, archive, or code
        using native OS desktop applications (e.g. Papers/Evince for PDF, LibreOffice for PPT/PPTX/ODP/XLS/XLSX,
        File-Roller/Ark for ZIP/archives, GNOME Text Editor for code/JSON/text, Loupe for photos).
        Never opens local files in browser tabs.
        """
        clean_path = (file_path or "").strip()
        if not clean_path:
            return False
        if clean_path.startswith("file://"):
            import urllib.parse
            clean_path = urllib.parse.unquote(clean_path[7:])

        target = Path(clean_path).expanduser().resolve()
        if not target.exists():
            logger.warning("Target file does not exist: %s", target)
            return False

        gui_env = self.get_gui_environment()
        ext = target.suffix.lower()

        # 1. Images & photos: prioritize dedicated native image viewers (Loupe on Fedora)
        if ext in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".svg", ".tiff"}:
            for iv in ["loupe", "eog", "gwenview", "viewnior"]:
                iv_bin = shutil.which(iv)
                if iv_bin:
                    try:
                        subprocess.Popen([iv_bin, str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass

        # 2. PDF & documents: prioritize native document viewer (Papers on GNOME/Fedora, Evince, Okular)
        elif ext in {".pdf"}:
            for pv in ["papers", "evince", "okular", "atril", "xreader", "qpdfview"]:
                pv_bin = shutil.which(pv)
                if pv_bin:
                    try:
                        subprocess.Popen([pv_bin, str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass

        # 3. Presentations & Slides: prioritize office presentation suite (LibreOffice Impress)
        elif ext in {".ppt", ".pptx", ".odp", ".pps", ".ppsx"}:
            for ov in ["libreoffice", "soffice"]:
                ov_bin = shutil.which(ov)
                if ov_bin:
                    try:
                        subprocess.Popen([ov_bin, "--impress", str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass

        # 4. Spreadsheets: prioritize office spreadsheet suite (LibreOffice Calc)
        elif ext in {".xls", ".xlsx", ".ods", ".csv"}:
            for ov in ["libreoffice", "soffice"]:
                ov_bin = shutil.which(ov)
                if ov_bin:
                    try:
                        subprocess.Popen([ov_bin, "--calc", str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass

        # 5. Office Documents: prioritize office writer suite (LibreOffice Writer)
        elif ext in {".doc", ".docx", ".odt", ".rtf"}:
            for ov in ["libreoffice", "soffice"]:
                ov_bin = shutil.which(ov)
                if ov_bin:
                    try:
                        subprocess.Popen([ov_bin, "--writer", str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass

        # 6. Archives & Compressed files: prioritize Archive Manager (file-roller, ark, nautilus)
        elif ext in {".zip", ".tar", ".gz", ".xz", ".7z", ".bz2", ".tgz", ".rar"}:
            for av in ["file-roller", "ark", "nautilus"]:
                av_bin = shutil.which(av)
                if av_bin:
                    try:
                        subprocess.Popen([av_bin, str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass

        # 7. Code, JSON, and Text files: prioritize native desktop text editor (GNOME Text Editor, Gedit, Kate, Code)
        elif ext in {".json", ".txt", ".py", ".md", ".yaml", ".yml", ".xml", ".sh", ".conf", ".ini", ".log"}:
            for tv in ["gnome-text-editor", "gedit", "kate", "kwrite", "code"]:
                tv_bin = shutil.which(tv)
                if tv_bin:
                    try:
                        subprocess.Popen([tv_bin, str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return True
                    except Exception:
                        pass

        # 8. Standard desktop MIME dispatch via gio open (GNOME / FreeDesktop default association)
        if shutil.which("gio"):
            try:
                subprocess.Popen(["gio", "open", str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

        # 9. Fallback to xdg-open
        if shutil.which("xdg-open"):
            try:
                subprocess.Popen(["xdg-open", str(target)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass

        return False

    def open_application(self, app_name: str) -> None:
        """Open an application by executable, alias, or desktop entry name on Linux."""
        clean_name = app_name.strip().lower()
        if not clean_name:
            raise ValueError("Application name cannot be empty")

        gui_env = self.get_gui_environment()

        # 1. Check if direct executable exists in PATH
        executable = shutil.which(clean_name)
        if executable:
            subprocess.Popen([executable], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return

        # 2. Comprehensive Linux / Fedora desktop application mapping
        common_apps = {
            "camera": ["snapshot", "cheese"],
            "terminal": ["ptyxis", "gnome-terminal", "konsole", "xterm"],
            "browser": ["firefox", "google-chrome", "google-chrome-stable", "chromium", "brave-browser"],
            "chrome": ["google-chrome", "google-chrome-stable", "chromium"],
            "firefox": ["firefox"],
            "code": ["code", "codium"],
            "vscode": ["code", "codium"],
            "calculator": ["gnome-calculator", "kcalc", "calc"],
            "files": ["nautilus", "dolphin", "thunar"],
            "file manager": ["nautilus", "dolphin", "thunar"],
            "settings": ["gnome-control-center", "systemsettings"],
            "editor": ["gnome-text-editor", "gedit", "kate"],
            "text editor": ["gnome-text-editor", "gedit", "kate"],
            "spotify": ["spotify"],
            "music": ["spotify", "rhythmbox"],
            "videos": ["totem", "celluloid", "vlc"],
            "system monitor": ["gnome-system-monitor", "ksysguard"],
        }

        candidates = common_apps.get(clean_name, [clean_name])
        for cand in candidates:
            cand_bin = shutil.which(cand)
            if cand_bin:
                subprocess.Popen([cand_bin], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return

        # 3. Dynamic .desktop file resolution across standard XDG application directories
        import glob
        desktop_dirs = [
            "/usr/share/applications",
            os.path.expanduser("~/.local/share/applications"),
            "/var/lib/flatpak/exports/share/applications",
        ]
        for d in desktop_dirs:
            if not os.path.isdir(d):
                continue
            for f in glob.glob(os.path.join(d, "*.desktop")):
                try:
                    with open(f, "r", encoding="utf-8", errors="ignore") as fp:
                        content = fp.read()
                    name_m = re.search(r"^Name=(.+)$", content, re.MULTILINE)
                    exec_m = re.search(r"^Exec=([^ \n]+)", content, re.MULTILINE)
                    d_name = name_m.group(1).strip().lower() if name_m else ""
                    d_exe = exec_m.group(1).strip() if exec_m else ""
                    desktop_id = os.path.basename(f)

                    if clean_name == d_name or clean_name in d_name or clean_name == d_exe.lower() or clean_name in desktop_id.lower():
                        if d_exe and shutil.which(d_exe):
                            subprocess.Popen([shutil.which(d_exe)], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            return
                        if shutil.which("gtk-launch"):
                            res = subprocess.run(["gtk-launch", desktop_id], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            if res.returncode == 0:
                                return
                except Exception:
                    pass

        # 4. Fallback to gtk-launch or xdg-open
        try:
            subprocess.Popen(["gtk-launch", clean_name], env=gui_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as err:
            raise RuntimeError(f"Could not launch application '{app_name}' on Linux: {str(err)}")

    def close_application(self, app_name: str) -> None:
        """Terminate a running process by application name on Linux with self-exclusion safety."""
        clean_name = app_name.strip().lower()
        if not clean_name:
            raise ValueError("Application name cannot be empty")

        if clean_name in _PROTECTED_PROCESS_NAMES:
            raise RuntimeError(f"Cannot close protected system or agent runtime process '{app_name}'")

        current_pid = os.getpid()
        parent_pid = os.getppid()

        # Gather target PIDs excluding current process, parent process, and system daemons
        target_pids = []
        try:
            res = subprocess.run(["pgrep", "-f", clean_name], capture_output=True, text=True)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    p_str = line.strip()
                    if p_str.isdigit():
                        pid = int(p_str)
                        if pid not in (current_pid, parent_pid, 1):
                            target_pids.append(pid)
        except Exception:
            pass

        if not target_pids:
            raise RuntimeError(f"No non-protected running process found for '{app_name}' on Linux")

        # Terminate gracefully with SIGTERM first
        terminated = False
        for pid in target_pids:
            try:
                os.kill(pid, signal.SIGTERM)
                terminated = True
            except OSError:
                pass

        if not terminated:
            raise RuntimeError(f"Failed to terminate any processes for '{app_name}'")

    def is_application_running(self, app_name: str) -> bool:
        """Check if process is currently running on Linux."""
        clean_name = app_name.strip().lower()
        try:
            res = subprocess.run(["pgrep", "-f", clean_name], capture_output=True, text=True)
            return res.returncode == 0
        except Exception:
            return False

    def focus_window_or_application(self, target: str) -> bool:
        """Bring matching application or window to focus on Linux using wmctrl or xdotool if available."""
        clean = target.strip().lower()
        if not clean:
            return False
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-a", clean], capture_output=True, text=True)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
        if shutil.which("xdotool"):
            try:
                res = subprocess.run(["xdotool", "search", "--onlyvisible", "--name", clean, "windowactivate"], capture_output=True, text=True)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
        return False

    def get_open_windows(self) -> list:
        """Enumerate currently open and visible windows on Linux."""
        windows = []
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-l", "-p", "-G"], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        parts = line.split(None, 8)
                        if len(parts) >= 9:
                            wid, d_num, pid_str, x, y, w, h, host, title = parts
                            windows.append({
                                "id": wid,
                                "hwnd": wid,
                                "pid": int(pid_str) if pid_str.isdigit() else 0,
                                "title": title.strip(),
                                "bounds": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                                "process": "",
                            })
                    if windows:
                        return windows
            except Exception:
                pass

        # Fallback: check active running GUI/system processes via psutil
        try:
            import psutil
            gui_keywords = ["chrome", "firefox", "terminal", "code", "spotify", "nautilus", "calc", "gedit", "bash"]
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    name = (proc.info.get('name') or "").lower()
                    if any(kw in name for kw in gui_keywords):
                        windows.append({
                            "id": str(proc.info.get('pid', 0)),
                            "hwnd": proc.info.get('pid', 0),
                            "pid": proc.info.get('pid', 0),
                            "title": proc.info.get('name') or "Application Window",
                            "process": name,
                            "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
                        })
                except Exception:
                    continue
        except Exception:
            pass
        return windows

    def get_active_window(self) -> Optional[dict]:
        """Retrieve title and bounds of the current active window on Linux."""
        if shutil.which("xdotool"):
            try:
                res = subprocess.run(["xdotool", "getactivewindow"], capture_output=True, text=True, timeout=2)
                if res.returncode == 0 and res.stdout.strip():
                    wid = res.stdout.strip()
                    title = ""
                    t_res = subprocess.run(["xdotool", "getwindowname", wid], capture_output=True, text=True, timeout=2)
                    if t_res.returncode == 0:
                        title = t_res.stdout.strip()
                    pid = 0
                    p_res = subprocess.run(["xdotool", "getwindowpid", wid], capture_output=True, text=True, timeout=2)
                    if p_res.returncode == 0 and p_res.stdout.strip().isdigit():
                        pid = int(p_res.stdout.strip())
                    return {
                        "id": wid,
                        "hwnd": wid,
                        "title": title or "Active Window",
                        "pid": pid,
                        "process": "",
                        "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
                    }
            except Exception:
                pass

        open_wins = self.get_open_windows()
        if open_wins:
            return open_wins[0]

        return {
            "id": "desktop",
            "hwnd": 0,
            "title": "Desktop Workspace",
            "pid": 0,
            "process": "",
            "bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080},
        }

    def close_window(self, target: str) -> bool:
        """Close an open window matching target on Linux."""
        clean = target.strip().lower()
        if not clean:
            return False
        if shutil.which("wmctrl"):
            try:
                res = subprocess.run(["wmctrl", "-c", clean], capture_output=True, text=True, timeout=3)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
        try:
            self.close_application(target)
            return True
        except Exception:
            return False

    def discover_applications(self) -> list:
        """Discover installed Linux applications from desktop entries."""
        discovered = []
        seen_names = set()
        desktop_dirs = [
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
            Path(os.path.expanduser("~/.local/share/applications")),
        ]

        for d in desktop_dirs:
            if not d.exists() or not d.is_dir():
                continue
            for entry_file in d.glob("*.desktop"):
                try:
                    content = entry_file.read_text(encoding="utf-8", errors="replace")
                    name = None
                    exec_cmd = None
                    category = "Utility"
                    no_display = False

                    for line in content.splitlines():
                        line = line.strip()
                        if line.startswith("Name=") and not name:
                            name = line[5:].strip()
                        elif line.startswith("Exec=") and not exec_cmd:
                            exec_cmd = line[5:].strip().split()[0]
                        elif line.startswith("Categories="):
                            category = line[11:].strip().split(";")[0] or category
                        elif line.startswith("NoDisplay=true"):
                            no_display = True

                    if name and exec_cmd and not no_display:
                        clean_key = name.lower()
                        if clean_key not in seen_names:
                            seen_names.add(clean_key)
                            discovered.append({
                                "name": name,
                                "exec": exec_cmd,
                                "category": category,
                                "path": str(entry_file),
                            })
                except Exception:
                    continue

        if not discovered:
            # Standard curated applications fallback
            curated = [
                {"name": "Web Browser", "exec": "firefox", "category": "Network"},
                {"name": "Terminal", "exec": "gnome-terminal", "category": "System"},
                {"name": "Calculator", "exec": "gnome-calculator", "category": "Utility"},
                {"name": "Files", "exec": "nautilus", "category": "System"},
                {"name": "Text Editor", "exec": "gedit", "category": "Utility"},
            ]
            return curated

        return discovered

    def get_screen_geometry(self) -> dict:
        """Return screen dimensions on Linux."""
        if shutil.which("xrandr"):
            try:
                res = subprocess.run(["xrandr", "--current"], capture_output=True, text=True, timeout=2)
                if res.returncode == 0:
                    m = re.search(r"current\s+(\d+)\s*x\s*(\d+)", res.stdout)
                    if m:
                        return {"width": int(m.group(1)), "height": int(m.group(2)), "scale_factor": 1.0}
            except Exception:
                pass

        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return {"width": img.width, "height": img.height, "scale_factor": 1.0}
        except Exception:
            pass

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

    def speak(self, text: str) -> None:
        """Synthesize text-to-speech output on Linux using neural VoiceEngine with fallback."""
        clean_text = text.strip()
        try:
            from voice.engine import VoiceEngine
            ve = VoiceEngine(platform_driver=self)
            res = ve.speak(clean_text)
            if res.get("success"):
                return
        except Exception:
            pass

        for tts_cmd in ["spd-say", "espeak-ng", "espeak"]:
            if shutil.which(tts_cmd):
                try:
                    subprocess.run([tts_cmd, clean_text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    return
                except Exception:
                    pass
        pass

    def set_volume(self, level: int) -> bool:
        """Set master audio volume level (0-100) on Linux using pactl or amixer."""
        target = max(0, min(100, int(level)))
        if shutil.which("pactl"):
            try:
                res = subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{target}%"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        if shutil.which("amixer"):
            try:
                res = subprocess.run(
                    ["amixer", "set", "Master", f"{target}%"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        return False

    def get_volume(self) -> Optional[int]:
        """Get current master audio volume level (0-100) on Linux using pactl or amixer."""
        if shutil.which("pactl"):
            try:
                res = subprocess.run(
                    ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0 and res.stdout:
                    match = re.search(r"(\d+)%", res.stdout)
                    if match:
                        return int(match.group(1))
            except Exception:
                pass

        if shutil.which("amixer"):
            try:
                res = subprocess.run(
                    ["amixer", "get", "Master"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0 and res.stdout:
                    match = re.search(r"\[(\d+)%\]", res.stdout)
                    if match:
                        return int(match.group(1))
            except Exception:
                pass

        return None

    def set_mute(self, mute: bool) -> bool:
        """Mute or unmute master audio on Linux using pactl set-sink-mute @DEFAULT_SINK@ or amixer."""
        mute_val = "1" if mute else "0"
        if shutil.which("pactl"):
            try:
                res = subprocess.run(
                    ["pactl", "set-sink-mute", "@DEFAULT_SINK@", mute_val],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        if shutil.which("amixer"):
            try:
                amixer_arg = "mute" if mute else "unmute"
                res = subprocess.run(
                    ["amixer", "set", "Master", amixer_arg],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        return False

    def set_brightness(self, level: int) -> bool:
        """Set screen display brightness (0-100) on Linux using brightnessctl or sysfs."""
        target = max(0, min(100, int(level)))

        if shutil.which("brightnessctl"):
            try:
                res = subprocess.run(
                    ["brightnessctl", "set", f"{target}%"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return True
            except Exception:
                pass

        backlight_dir = Path("/sys/class/backlight")
        if backlight_dir.exists():
            for device in backlight_dir.iterdir():
                b_file = device / "brightness"
                mb_file = device / "max_brightness"
                if b_file.exists() and mb_file.exists():
                    try:
                        max_b = int(mb_file.read_text().strip())
                        val = int(round((target / 100.0) * max_b))
                        b_file.write_text(str(val))
                        return True
                    except Exception:
                        pass

        return False

    def get_brightness(self) -> Optional[int]:
        """Get current screen display brightness (0-100) on Linux using brightnessctl or sysfs."""
        if shutil.which("brightnessctl"):
            try:
                res = subprocess.run(
                    ["brightnessctl", "-m"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0 and res.stdout:
                    parts = res.stdout.strip().split(",")
                    for part in parts:
                        if part.endswith("%"):
                            return int(part[:-1])
            except Exception:
                pass

        backlight_dir = Path("/sys/class/backlight")
        if backlight_dir.exists():
            for device in backlight_dir.iterdir():
                b_file = device / "brightness"
                mb_file = device / "max_brightness"
                if b_file.exists() and mb_file.exists():
                    try:
                        cur = int(b_file.read_text().strip())
                        max_b = int(mb_file.read_text().strip())
                        if max_b > 0:
                            return int(round((cur / max_b) * 100))
                    except Exception:
                        pass

        return None
