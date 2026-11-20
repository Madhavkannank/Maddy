"""Computer Screen Perception and Screenshot Tool for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You).

Provides display observation, visual element localization, OCR error detection,
and sandboxed screenshot capture.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from core.permissions import PermissionGate, PermissionTier
from core.tools.base import BaseTool, ToolResult
from platform_layer import PlatformInterface, get_platform
from vision.engine import VisionEngine


class ComputerScreenTool(BaseTool):
    """Tool for capturing screenshots, observing screen state, and locating UI elements visually."""

    name = "computer_screen"
    description = (
        "Observe the computer display, capture screenshots, and locate UI elements visually. "
        "Actions: 'observe' (reads active window, screen geometry, OCR text, and visual error lines), "
        "'screenshot' (captures display image to a sandboxed file path), "
        "'locate_element' (finds UI element bounding box and centroid coordinates for visual interaction), "
        "'get_active_window' (retrieves active foreground window info)."
    )
    tier = PermissionTier.SAFE

    def __init__(
        self,
        platform: Optional[PlatformInterface] = None,
        vision_engine: Optional[VisionEngine] = None,
        permission_gate: Optional[PermissionGate] = None,
    ):
        self.platform = platform or get_platform()
        self.vision = vision_engine or VisionEngine()
        self.permission_gate = permission_gate or PermissionGate()

    def operation_tier(self, arguments: Dict[str, Any]) -> PermissionTier:
        action = str(arguments.get("action", "")).strip().lower()
        if action == "screenshot":
            # If path is outside standard temp/capture dir, treat as SYSTEM_IMPACTING
            path_str = str(arguments.get("path", "")).strip()
            if path_str:
                p = Path(path_str).resolve()
                temp_dir = Path(tempfile.gettempdir()).resolve()
                if not (p == temp_dir or temp_dir in p.parents):
                    return PermissionTier.SYSTEM_IMPACTING
        return PermissionTier.SAFE

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["observe", "screenshot", "locate_element", "get_active_window"],
                    "description": "Screen action to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Optional destination path for 'screenshot' action (PNG format).",
                },
                "element_description": {
                    "type": "string",
                    "description": "Description or text of visual element to locate for 'locate_element' action.",
                },
            },
            "required": ["action"],
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()

        if action == "observe":
            active_win = self.platform.get_active_window() or {}
            screen_geo = self.platform.get_screen_geometry() or {}

            # Capture temporary observation shot
            temp_path = os.path.join(tempfile.gettempdir(), f"observe_{os.getpid()}.png")
            shot_res = self.platform.take_screenshot(temp_path)
            actual_shot = shot_res.get("path", temp_path)

            ocr_res = self.vision.extract_screen_text_and_errors(actual_shot)

            output_lines = [
                f"Display: {screen_geo.get('width', 1920)}x{screen_geo.get('height', 1080)}",
                f"Active Window: '{active_win.get('title', 'Desktop')}' (PID: {active_win.get('pid', 0)})",
                f"Visual Errors: {ocr_res.get('summary', 'No error lines detected.')}",
            ]
            if ocr_res.get("error_lines"):
                output_lines.append("Detected Error Snippets:")
                for err in ocr_res["error_lines"][:3]:
                    output_lines.append(f"  - {err}")

            return ToolResult(
                success=True,
                output="\n".join(output_lines),
                metadata={
                    "active_window": active_win,
                    "geometry": screen_geo,
                    "has_error": ocr_res.get("has_error", False),
                    "error_lines": ocr_res.get("error_lines", []),
                    "screenshot_path": actual_shot,
                },
            )

        elif action == "get_active_window":
            active_win = self.platform.get_active_window() or {}
            title = active_win.get("title", "Unknown")
            pid = active_win.get("pid", 0)
            bounds = active_win.get("bounds", {})
            return ToolResult(
                success=True,
                output=f"Active Foreground Window: '{title}' (PID: {pid}, Bounds: {bounds})",
                metadata={"active_window": active_win},
            )

        elif action == "screenshot":
            out_path = kwargs.get("path")
            if out_path:
                clean_path = Path(str(out_path)).resolve()
                # Verify sandboxed write directory
                if not self.permission_gate.is_path_in_allowed_sandbox(clean_path):
                    return ToolResult(
                        success=False,
                        output="",
                        error=f"Destination screenshot path '{clean_path}' violates filesystem sandbox restrictions.",
                    )
                target_file = str(clean_path)
            else:
                pictures_dir = os.path.join(os.path.expanduser("~"), "Pictures", "MADDY_Captures")
                os.makedirs(pictures_dir, exist_ok=True)
                target_file = os.path.join(pictures_dir, f"screenshot_{int(os.getpid())}.png")

            shot_res = self.platform.take_screenshot(target_file)
            if shot_res.get("success", True):
                actual_path = shot_res.get("path", target_file)
                size_kb = round(os.path.getsize(actual_path) / 1024, 1) if os.path.exists(actual_path) else 0
                return ToolResult(
                    success=True,
                    output=f"Screenshot captured and saved to '{actual_path}' ({size_kb} KB).",
                    metadata=shot_res,
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Screenshot capture failed: {shot_res.get('error', 'Unknown error')}",
                )

        elif action == "locate_element":
            elem_desc = str(kwargs.get("element_description", "")).strip()
            if not elem_desc:
                return ToolResult(
                    success=False,
                    output="",
                    error="Parameter 'element_description' is required for locate_element.",
                )

            temp_path = os.path.join(tempfile.gettempdir(), f"locate_{os.getpid()}.png")
            shot_res = self.platform.take_screenshot(temp_path)
            actual_shot = shot_res.get("path", temp_path)

            loc_res = self.vision.locate_element(actual_shot, elem_desc)
            if loc_res.get("success"):
                cx, cy = loc_res.get("centroid", (0, 0))
                bb = loc_res.get("bounding_box", {})
                return ToolResult(
                    success=True,
                    output=(
                        f"Located '{elem_desc}': Centroid=({cx}, {cy}), "
                        f"Bounds={bb}, Match Type={loc_res.get('match_type')}, "
                        f"Confidence={loc_res.get('confidence')}."
                    ),
                    metadata=loc_res,
                )
            else:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Could not locate visual element '{elem_desc}' on screen.",
                )

        return ToolResult(
            success=False,
            output="",
            error=f"Unknown screen action '{action}'. Valid actions: observe, screenshot, locate_element, get_active_window.",
        )
