"""Application Lifecycle and Window Control Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides structured operations to launch, close, focus, and query desktop applications and windows.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult
from platform_layer import PlatformInterface, get_platform


class AppControlTool(BaseTool):
    """Controls desktop applications and windows across Windows and Linux."""

    name = "app_control"
    description = (
        "Launch, close, focus, or inspect desktop applications, or open any local file in its native desktop application. "
        "Allows opening installed software (Spotify, VS Code, Browser, Notepad, Terminal), "
        "opening local files (PDF documents, PowerPoint presentations, Excel spreadsheets, JSON files, code, text, ZIP archives, photos) "
        "in their native desktop applications (Papers/Evince, LibreOffice, GNOME Text Editor, File Roller, Loupe on Linux; registered default apps on Windows), "
        "focusing existing open windows, and checking process states. Never launches local files in browser tabs."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    def __init__(self, platform: Optional[PlatformInterface] = None):
        self.platform = platform or get_platform()

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        action = str(arguments.get("action", "")).strip().lower()
        if action in {"is_running", "list_windows", "open_file"}:
            return PermissionTier.SAFE
        return PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["launch", "close", "focus", "is_running", "list_windows", "open_file"],
                    "description": "Application control action to perform. Use 'open_file' to open any local document, presentation, spreadsheet, JSON, archive, or photo in its native desktop application.",
                },
                "app_name": {
                    "type": "string",
                    "description": "Name or alias of application (e.g. 'spotify', 'code', 'calculator', 'chrome').",
                },
                "target": {
                    "type": "string",
                    "description": "Target window title, process, or tab name to focus.",
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to local file, document, PDF, presentation (.ppt/.pptx), spreadsheet (.xls/.xlsx/.csv), JSON (.json), code, archive (.zip), or photo to open in native desktop viewer (e.g. '~/document.pdf', 'report.pptx', 'data.json', 'archive.zip', 'photo.jpg').",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def execute(
        self,
        action: str,
        app_name: Optional[str] = None,
        target: Optional[str] = None,
        **kwargs: Any,
    ) -> ToolResult:
        act = action.strip().lower()
        try:
            if act == "launch":
                name = (app_name or target or "").strip()
                if not name:
                    return ToolResult(success=False, output="", error="Parameter 'app_name' is required to launch an application.")
                # First check if window is already open to avoid redundant launch
                if hasattr(self.platform, "focus_window_or_application") and self.platform.focus_window_or_application(name):
                    return ToolResult(
                        success=True,
                        output=f"Existing window for '{name}' was already open; brought to foreground.",
                        metadata={"action": "focused_existing", "app_name": name},
                    )
                self.platform.open_application(name)
                return ToolResult(
                    success=True,
                    output=f"Dispatched launch request for application '{name}'.",
                    metadata={"action": "launched", "app_name": name},
                )

            elif act == "close":
                name = (app_name or target or "").strip()
                if not name:
                    return ToolResult(success=False, output="", error="Parameter 'app_name' is required to close an application.")
                self.platform.close_application(name)
                return ToolResult(
                    success=True,
                    output=f"Dispatched termination request for application '{name}'.",
                    metadata={"action": "closed", "app_name": name},
                )

            elif act == "focus":
                win_target = (target or app_name or "").strip()
                if not win_target:
                    return ToolResult(success=False, output="", error="Parameter 'target' or 'app_name' is required to focus a window.")
                if hasattr(self.platform, "focus_window_or_application"):
                    focused = self.platform.focus_window_or_application(win_target)
                    if focused:
                        return ToolResult(
                            success=True,
                            output=f"Window matching '{win_target}' focused successfully.",
                            metadata={"focused": True, "target": win_target},
                        )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"No matching active window found for '{win_target}'.",
                )

            elif act == "is_running":
                name = (app_name or target or "").strip()
                if not name:
                    return ToolResult(success=False, output="", error="Parameter 'app_name' is required to check running status.")
                running = self.platform.is_application_running(name)
                state_str = "is currently running" if running else "is not running"
                return ToolResult(
                    success=True,
                    output=f"Application '{name}' {state_str}.",
                    metadata={"app_name": name, "running": running},
                )

            elif act == "list_windows":
                windows = []
                if hasattr(self.platform, "get_open_windows"):
                    windows = self.platform.get_open_windows()
                return ToolResult(
                    success=True,
                    output=f"Found {len(windows)} active window(s) on desktop.",
                    metadata={"windows": windows, "count": len(windows)},
                )

            elif act == "open_file":
                path_str = (kwargs.get("file_path") or target or app_name or "").strip()
                if not path_str:
                    return ToolResult(
                        success=False,
                        output="",
                        error="Parameter 'file_path' is required to open a local file.",
                    )
                if hasattr(self.platform, "open_file"):
                    opened = self.platform.open_file(path_str)
                    if opened:
                        return ToolResult(
                            success=True,
                            output=f"Opened '{path_str}' in native desktop viewer.",
                            metadata={"action": "open_file", "file_path": path_str},
                        )
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Could not open file '{path_str}' in native desktop viewer. Check that the file exists.",
                )

            return ToolResult(success=False, output="", error=f"Unknown app_control action: '{action}'.")

        except Exception as exc:
            return ToolResult(success=False, output="", error=f"App control action '{action}' failed: {str(exc)}")
