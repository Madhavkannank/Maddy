"""System Control Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).
Provides structured operations for host volume, brightness, mute, lock, sleep, reboot, and shutdown.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Any, Dict, Optional

from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult
from platform_layer import PlatformInterface, get_platform


class SystemControlTool(BaseTool):
    """Controls workstation hardware and OS state: audio volume, brightness, mute, power."""

    name = "system_control"
    description = (
        "Control workstation system state including master audio volume (0-100), screen brightness (0-100), "
        "audio mute/unmute, and workstation power/lock state (lock, sleep, reboot, shutdown). "
        "Read operations are safe; mutating operations require system-impacting permissions."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    def __init__(self, platform: Optional[PlatformInterface] = None):
        self.platform = platform or get_platform()

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        action = str(arguments.get("action", "")).strip().lower()
        if action in {"get_volume", "get_brightness", "get_mute"}:
            return PermissionTier.SAFE
        return PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "get_volume",
                        "set_volume",
                        "get_brightness",
                        "set_brightness",
                        "set_mute",
                        "get_mute",
                        "lock_workstation",
                        "sleep_workstation",
                        "reboot_workstation",
                        "shutdown_workstation",
                    ],
                    "description": "The system action to execute.",
                },
                "level": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Level percentage (0-100) for set_volume or set_brightness.",
                },
                "mute": {
                    "type": "boolean",
                    "description": "Whether to mute (True) or unmute (False) master audio.",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def execute(self, action: str, level: Optional[int] = None, mute: Optional[bool] = None, **kwargs: Any) -> ToolResult:
        act = action.strip().lower()
        try:
            if act == "get_volume":
                vol = self.platform.get_volume()
                if vol is not None:
                    return ToolResult(success=True, output=f"Current master volume is {vol}%.", metadata={"volume": vol})
                return ToolResult(success=False, output="", error="Unable to query master audio volume.")

            elif act == "set_volume":
                if level is None:
                    return ToolResult(success=False, output="", error="Parameter 'level' (0-100) is required for 'set_volume'.")
                target = max(0, min(100, int(level)))
                success = self.platform.set_volume(target)
                if success:
                    return ToolResult(success=True, output=f"Master volume adjusted to {target}%.", metadata={"volume": target})
                return ToolResult(success=False, output="", error=f"Failed to set volume to {target}%.")

            elif act == "get_brightness":
                br = self.platform.get_brightness()
                if br is not None:
                    return ToolResult(success=True, output=f"Current display brightness is {br}%.", metadata={"brightness": br})
                return ToolResult(success=False, output="", error="Unable to query display brightness.")

            elif act == "set_brightness":
                if level is None:
                    return ToolResult(success=False, output="", error="Parameter 'level' (0-100) is required for 'set_brightness'.")
                target = max(0, min(100, int(level)))
                success = self.platform.set_brightness(target)
                if success:
                    return ToolResult(success=True, output=f"Display brightness adjusted to {target}%.", metadata={"brightness": target})
                return ToolResult(success=False, output="", error=f"Failed to set brightness to {target}%.")

            elif act == "set_mute":
                is_mute = bool(mute) if mute is not None else True
                success = self.platform.set_mute(is_mute)
                state_str = "muted" if is_mute else "unmuted"
                if success:
                    return ToolResult(success=True, output=f"Master audio {state_str}.", metadata={"muted": is_mute})
                return ToolResult(success=False, output="", error=f"Failed to set audio {state_str}.")

            elif act == "get_mute":
                vol = self.platform.get_volume()
                return ToolResult(success=True, output=f"Master audio query completed (volume: {vol}%).", metadata={"volume": vol})

            elif act == "lock_workstation":
                if self.platform.get_os_name() == "windows":
                    import ctypes
                    res = ctypes.windll.user32.LockWorkStation()
                    return ToolResult(success=bool(res), output="Workstation locked successfully." if res else "", error=None if res else "Failed to lock Windows workstation.")
                else:
                    for cmd in [["loginctl", "lock-session"], ["xdg-screensaver", "lock"]]:
                        if shutil.which(cmd[0]):
                            res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                            if res.returncode == 0:
                                return ToolResult(success=True, output="Linux session locked successfully.")
                    return ToolResult(success=False, output="", error="No supported screen locker tool (loginctl/xdg-screensaver) found.")

            elif act in ("sleep_workstation", "reboot_workstation", "shutdown_workstation"):
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Power action '{act}' is not implemented; no power state was changed.",
                    metadata={"action": act, "executed": False},
                )

            return ToolResult(success=False, output="", error=f"Unknown system_control action: '{action}'.")

        except Exception as exc:
            return ToolResult(success=False, output="", error=f"System control action '{action}' failed: {str(exc)}")
