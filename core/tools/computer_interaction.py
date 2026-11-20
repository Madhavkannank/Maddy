"""Computer Interaction and Layered Input Automation Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Implements Phase 3.7:
- Layered interaction hierarchy: Native OS APIs -> Accessibility APIs -> Visual Screen -> Coordinate clicking
- Deterministic operation-level permissions
- Independent outcome verification (Observe -> Act -> Observe -> Verify)
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.computer_automation import ComputerAutomationController
from core.permissions import PermissionTier
from core.tools.base import BaseTool, ToolResult
from platform_layer import PlatformInterface, get_platform
from vision.engine import VisionEngine


class ComputerInteractionTool(BaseTool):
    """Executes layered computer control interactions with before-and-after outcome verification."""

    name = "computer_interaction"
    description = (
        "Interact with computer applications, windows, and display elements using the layered interaction hierarchy. "
        "Safe read actions: 'list_windows' (enumerates open windows), 'get_active_window' (gets current foreground window), "
        "'discover_apps' (finds installed applications). "
        "Mutating actions: 'focus_window' (brings target window to foreground), 'close_window' (closes window), "
        "'click_element' (layered click using Native OS API -> Semantic UI -> Vision/OCR -> Coordinate fallback), "
        "'click_coordinates' (simulates mouse click at x, y), 'type_text' (types text into focused window), "
        "'press_key' (sends keyboard key/shortcut e.g. 'ctrl+s', 'enter'), 'scroll' (scrolls mouse wheel)."
    )
    tier = PermissionTier.SYSTEM_IMPACTING

    SAFE_ACTIONS = {"list_windows", "get_active_window", "discover_apps"}

    def __init__(
        self,
        platform: Optional[PlatformInterface] = None,
        vision_engine: Optional[VisionEngine] = None,
        controller: Optional[ComputerAutomationController] = None,
    ):
        self.platform = platform or get_platform()
        self.vision = vision_engine or VisionEngine()
        self.controller = controller or ComputerAutomationController(
            platform=self.platform,
            vision_engine=self.vision,
        )

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        action = str(arguments.get("action", "")).strip().lower()
        if action in self.SAFE_ACTIONS:
            return PermissionTier.SAFE
        return PermissionTier.SYSTEM_IMPACTING

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_windows",
                        "get_active_window",
                        "discover_apps",
                        "focus_window",
                        "close_window",
                        "click_element",
                        "click_coordinates",
                        "type_text",
                        "press_key",
                        "scroll",
                    ],
                    "description": "Interaction action to execute.",
                },
                "target": {
                    "type": "string",
                    "description": "Target application name, window title, or element description for focus_window, close_window, or click_element.",
                },
                "x": {
                    "type": "integer",
                    "description": "X coordinate for click_coordinates.",
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate for click_coordinates.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type for action 'type_text'.",
                },
                "key": {
                    "type": "string",
                    "description": "Key or shortcut combination to press for action 'press_key' (e.g. 'ctrl+c', 'enter', 'alt+tab').",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right", "middle"],
                    "default": "left",
                    "description": "Mouse button to click.",
                },
                "clicks": {
                    "type": "integer",
                    "default": 1,
                    "description": "Number of mouse clicks (1 for single click, 2 for double click).",
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "default": "down",
                    "description": "Scroll direction for action 'scroll'.",
                },
                "amount": {
                    "type": "integer",
                    "default": 3,
                    "description": "Number of scroll clicks.",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()

        # ── READ-ONLY ACTIONS (SAFE) ──────────────────────────────────────────

        if action == "list_windows":
            windows = self.platform.get_open_windows() or []
            if not windows:
                return ToolResult(
                    success=True,
                    output="No visible application windows currently open.",
                    metadata={"windows": []},
                )
            lines = [f"Open Windows ({len(windows)}):"]
            for w in windows:
                wid = w.get("id") or w.get("hwnd", "0")
                lines.append(f"  - [{wid}] '{w.get('title')}' (Process: {w.get('process', 'N/A')})")
            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"windows": windows, "count": len(windows)},
            )

        elif action == "get_active_window":
            active_win = self.platform.get_active_window() or {}
            title = active_win.get("title", "Desktop")
            return ToolResult(
                success=True,
                output=f"Active Foreground Window: '{title}' (ID: {active_win.get('id', 0)})",
                metadata={"active_window": active_win},
            )

        elif action == "discover_apps":
            apps = self.platform.discover_applications() or []
            lines = [f"Discovered Applications ({len(apps)}):"]
            for app in apps[:20]:
                lines.append(f"  - {app.get('name')} [{app.get('category', 'App')}]: {app.get('exec')}")
            if len(apps) > 20:
                lines.append(f"  ... and {len(apps) - 20} more applications.")
            return ToolResult(
                success=True,
                output="\n".join(lines),
                metadata={"applications": apps, "total": len(apps)},
            )

        # ── MUTATING ACTIONS (SYSTEM_IMPACTING) WITH VERIFICATION ──────────────

        elif action == "focus_window":
            target = str(kwargs.get("target", "")).strip()
            if not target:
                return ToolResult(success=False, output="", error="Parameter 'target' is required for focus_window.")

            ver_res = self.controller.focus_window(target)
            return ToolResult(
                success=ver_res.verified,
                output=ver_res.message,
                metadata={
                    "verified": ver_res.verified,
                    "layer_used": ver_res.layer_used,
                    "evidence": ver_res.evidence,
                },
            )

        elif action == "close_window":
            target = str(kwargs.get("target", "")).strip()
            if not target:
                return ToolResult(success=False, output="", error="Parameter 'target' is required for close_window.")

            ver_res = self.controller.close_window(target)
            return ToolResult(
                success=ver_res.verified,
                output=ver_res.message,
                metadata={
                    "verified": ver_res.verified,
                    "layer_used": ver_res.layer_used,
                    "evidence": ver_res.evidence,
                },
            )

        elif action == "click_element":
            target = str(kwargs.get("target", "")).strip()
            if not target:
                return ToolResult(success=False, output="", error="Parameter 'target' (element description) is required for click_element.")

            button = str(kwargs.get("button", "left")).lower()
            clicks = int(kwargs.get("clicks", 1))

            ver_res = self.controller.click_element_layered(
                element_description=target,
                button=button,
                clicks=clicks,
            )
            return ToolResult(
                success=ver_res.verified,
                output=ver_res.message,
                metadata={
                    "verified": ver_res.verified,
                    "layer_used": ver_res.layer_used,
                    "evidence": ver_res.evidence,
                },
            )

        elif action == "click_coordinates":
            if "x" not in kwargs or "y" not in kwargs:
                return ToolResult(success=False, output="", error="Parameters 'x' and 'y' are required for click_coordinates.")

            x = int(kwargs["x"])
            y = int(kwargs["y"])
            button = str(kwargs.get("button", "left")).lower()
            clicks = int(kwargs.get("clicks", 1))

            ver_res = self.controller.click_coordinates(x=x, y=y, button=button, clicks=clicks)
            return ToolResult(
                success=ver_res.verified,
                output=ver_res.message,
                metadata={
                    "verified": ver_res.verified,
                    "layer_used": ver_res.layer_used,
                    "evidence": ver_res.evidence,
                },
            )

        elif action == "type_text":
            text = str(kwargs.get("text", ""))
            if not text:
                return ToolResult(success=False, output="", error="Parameter 'text' is required for type_text.")

            press_enter = bool(kwargs.get("press_enter", False))
            ver_res = self.controller.type_text(text, press_enter=press_enter)
            return ToolResult(
                success=ver_res.verified,
                output=ver_res.message,
                metadata={
                    "verified": ver_res.verified,
                    "layer_used": ver_res.layer_used,
                    "evidence": ver_res.evidence,
                },
            )

        elif action == "press_key":
            key = str(kwargs.get("key", "")).strip()
            if not key:
                return ToolResult(success=False, output="", error="Parameter 'key' is required for press_key.")

            ver_res = self.controller.press_key(key)
            return ToolResult(
                success=ver_res.verified,
                output=ver_res.message,
                metadata={
                    "verified": ver_res.verified,
                    "layer_used": ver_res.layer_used,
                    "evidence": ver_res.evidence,
                },
            )

        elif action == "scroll":
            direction = str(kwargs.get("direction", "down")).lower()
            amount = int(kwargs.get("amount", 3))
            scroll_res = self.platform.scroll_mouse(direction=direction, amount=amount)
            return ToolResult(
                success=scroll_res.get("success", True),
                output=f"Scrolled mouse {direction} by {amount} units.",
                metadata=scroll_res,
            )

        return ToolResult(
            success=False,
            output="",
            error=f"Unknown interaction action '{action}'.",
        )
