"""
Cross-platform system interface for M.A.D.D.Y.
Defines platform-neutral contract implemented by OS-specific drivers.
"""
from abc import ABC, abstractmethod
from typing import Optional

class PlatformInterface(ABC):
    """
    Platform Interface providing OS abstraction.
    Core agent logic interacts with this interface rather than invoking raw system commands.
    """

    @abstractmethod
    def get_os_name(self) -> str:
        """Return the canonical operating system name ('windows' or 'linux')."""
        pass

    @abstractmethod
    def open_application(self, app_name: str) -> None:
        """Open an application by name."""
        pass

    def open_url(self, url: str) -> bool:
        """Open a destination URL in the default system browser."""
        return False

    def get_default_browser(self) -> str:
        """Determine default or primary installed browser dynamically from OS settings."""
        return "Default Browser"

    def open_file(self, file_path: str) -> bool:
        """Open a local file, image, or document using the native OS desktop viewer (e.g. Loupe/Photos)."""
        return False

    @abstractmethod
    def close_application(self, app_name: str) -> None:
        """Close an application by name."""
        pass

    @abstractmethod
    def is_application_running(self, app_name: str) -> bool:
        """Check if an application process is running."""
        pass

    def is_application_available(self, app_name: str) -> bool:
        """Check if an application is installed or available locally on the system."""
        return False

    def focus_window_or_application(self, target: str) -> bool:
        """
        If a window or application matching target (app name or window/tab title) is already open,
        bring that existing window to the foreground instead of opening a duplicate instance.
        Returns True if an existing window was found and focused, False otherwise.
        """
        return False

    def get_open_windows(self) -> list:
        """Enumerate currently open and visible windows."""
        return []

    def get_active_window(self) -> Optional[dict]:
        """Retrieve details and bounds of the currently focused foreground window."""
        return None

    def close_window(self, target: str) -> bool:
        """Close an open window matching target (window title, id, or process)."""
        return False

    def discover_applications(self) -> list:
        """Enumerate installed or discovered applications on the system."""
        return []

    def get_screen_geometry(self) -> dict:
        """Return screen dimensions and display scale factor."""
        return {"width": 1920, "height": 1080, "scale_factor": 1.0}

    def take_screenshot(self, output_path: Optional[str] = None) -> dict:
        """Capture screen display image and save to output_path."""
        return {"success": False, "error": "Not implemented"}

    def move_cursor(self, x: int, y: int, smooth: bool = True) -> dict:
        """Move mouse cursor to coordinates (x, y)."""
        return {"success": False, "error": "Not implemented"}

    def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> dict:
        """Simulate physical mouse click at coordinates or current position."""
        return {"success": False, "error": "Not implemented"}

    def scroll_mouse(self, direction: str = "down", amount: int = 3) -> dict:
        """Scroll mouse wheel in given direction."""
        return {"success": False, "error": "Not implemented"}

    def type_text(self, text: str, press_enter: bool = False) -> dict:
        """Type text into active foreground window."""
        return {"success": False, "error": "Not implemented"}

    def press_key(self, key_combination: str) -> dict:
        """Press a keyboard key or hotkey combination (e.g. 'ctrl+c', 'enter', 'alt+tab')."""
        return {"success": False, "error": "Not implemented"}

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesize spoken speech output from text input."""
        pass

    def set_brightness(self, level: int) -> bool:
        """Set screen display brightness (0-100)."""
        return False

    def get_brightness(self) -> Optional[int]:
        """Get current screen display brightness (0-100)."""
        return None

    def set_volume(self, level: int) -> bool:
        """Set master audio volume level (0-100)."""
        return False

    def get_volume(self) -> Optional[int]:
        """Get current master audio volume level (0-100)."""
        return None

    def set_mute(self, mute: bool) -> bool:
        """Set master audio mute state."""
        return False
