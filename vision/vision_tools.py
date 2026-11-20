"""Multimodal Vision & Active Window Grounding Tools for M.A.D.D.Y.

Tools:
1. `screen_inspect`: Captures display screenshot, extracts active foreground window info,
   and runs OCR to read visible on-screen text, code, or terminal error traces.
2. `active_window_inspect`: Retrieves title, process, and bounding geometry of the active window.
3. `screen_interact`: Unified desktop GUI interaction (click, type, cursor move).
4. `vision_analyze`: Direct image inspection (dimensions, format, aspect ratio).
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from computer.controller import ComputerController
from tools.base import RiskLevel, Tool
from vision.engine import VisionEngine

logger = logging.getLogger("vision.vision_tools")

_default_computer = ComputerController()
_default_vision = VisionEngine()


# ---------------------------------------------------------------------------
# 1. Screen Inspect Tool (Screenshot + Active Window + OCR Text)
# ---------------------------------------------------------------------------
def _screen_inspect_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    save_path = args.get("save_path")
    perform_ocr = args.get("extract_text", True)

    if not save_path:
        temp_dir = tempfile.gettempdir()
        save_path = os.path.join(temp_dir, f"maddy_screen_inspect_{os.getpid()}.png")

    # 1. Capture screen
    shot_result = _default_computer.take_screenshot(save_path)
    if not shot_result.get("success"):
        return shot_result

    actual_path = shot_result.get("path", save_path)

    # 2. Get active window metadata
    active_win = _default_vision.get_active_window_info()

    # 3. Extract OCR text if requested
    ocr_result = {}
    ocr_text = ""
    error_traces = []
    if perform_ocr:
        ocr_result = _default_vision.extract_screen_text_and_errors(actual_path)
        raw_text = ocr_result.get("extracted_text", "")
        error_traces = ocr_result.get("error_lines", [])
        # Bound OCR text length to prevent context flooding
        if len(raw_text) > 2000:
            ocr_text = raw_text[:2000] + f"\n... [Truncated: {len(raw_text):,} chars total]"
        else:
            ocr_text = raw_text

    return {
        "success": True,
        "screenshot_path": actual_path,
        "active_window": {
            "title": active_win.get("title", "Unknown"),
            "width": active_win.get("width", 0),
            "height": active_win.get("height", 0),
        },
        "dimensions": {
            "width": shot_result.get("width"),
            "height": shot_result.get("height"),
        },
        "visible_text": ocr_text,
        "error_traces": error_traces,
    }


screen_inspect_tool = Tool(
    name="screen_inspect",
    description=(
        "Capture current screen display, identify the active foreground window title, "
        "and run OCR to extract visible on-screen text, code, or terminal error messages."
    ),
    parameter_schema={
        "type": "object",
        "properties": {
            "extract_text": {
                "type": "boolean",
                "description": "Whether to perform OCR on the screenshot to extract visible text (default True).",
            },
            "save_path": {
                "type": "string",
                "description": "Optional file path where screenshot PNG will be stored.",
            },
        },
        "required": [],
    },
    handler=_screen_inspect_handler,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)


# ---------------------------------------------------------------------------
# 2. Active Window Inspect Tool
# ---------------------------------------------------------------------------
def _active_window_inspect_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    info = _default_vision.get_active_window_info()
    return {
        "success": True,
        "window_title": info.get("title", "Desktop"),
        "hwnd": info.get("hwnd", 0),
        "bounds": info.get("rect", {}),
        "width": info.get("width", 0),
        "height": info.get("height", 0),
    }


active_window_inspect_tool = Tool(
    name="active_window_inspect",
    description="Retrieve the title and geometry bounds of the currently focused foreground application window.",
    parameter_schema={"type": "object", "properties": {}},
    handler=_active_window_inspect_handler,
    risk_level=RiskLevel.LOW,
    timeout=5.0,
)


# ---------------------------------------------------------------------------
# 3. Screen Interact Tool (Click, Type, Move)
# ---------------------------------------------------------------------------
def _screen_interact_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    action = args.get("action", "").lower().strip()

    if action == "click":
        x = args.get("x")
        y = args.get("y")
        button = args.get("button", "left")
        clicks = int(args.get("clicks", 1))
        if x is None or y is None:
            return {"success": False, "error": "Coordinates 'x' and 'y' are required for click action."}
        return _default_computer.click(int(x), int(y), button=button, clicks=clicks)

    elif action == "type":
        text = args.get("text", "")
        if not text:
            return {"success": False, "error": "'text' is required for type action."}
        return _default_computer.type_text(text)

    elif action == "move":
        x = args.get("x")
        y = args.get("y")
        if x is None or y is None:
            return {"success": False, "error": "Coordinates 'x' and 'y' are required for move action."}
        return _default_computer.move_cursor(int(x), int(y), smooth=True)

    return {
        "success": False,
        "error": f"Unknown action '{action}'. Supported actions: click, type, move.",
    }


screen_interact_tool = Tool(
    name="screen_interact",
    description=(
        "Simulate physical desktop interactions at specific screen coordinates.\n"
        "ACTIONS:\n"
        "- 'click': Mouse click at (x, y) with optional button ('left'/'right') and clicks (1 or 2).\n"
        "- 'type': Type text into the active focused field.\n"
        "- 'move': Move mouse cursor smoothly to (x, y)."
    ),
    parameter_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["click", "type", "move"],
                "description": "Interaction action to execute.",
            },
            "x": {"type": "integer", "description": "Horizontal X coordinate."},
            "y": {"type": "integer", "description": "Vertical Y coordinate."},
            "button": {"type": "string", "enum": ["left", "right", "middle"], "description": "Mouse button for click."},
            "clicks": {"type": "integer", "description": "Number of clicks (1 for single, 2 for double)."},
            "text": {"type": "string", "description": "Text string to type into focused element."},
        },
        "required": ["action"],
    },
    handler=_screen_interact_handler,
    risk_level=RiskLevel.MEDIUM,
    timeout=10.0,
)


# ---------------------------------------------------------------------------
# 4. Vision Analyze Tool
# ---------------------------------------------------------------------------
def _vision_analyze_handler(args: Dict[str, Any]) -> Dict[str, Any]:
    image_path = args.get("image_path", "").strip()
    if not image_path:
        return {"success": False, "error": "image_path is required."}
    return _default_vision.analyze_image(image_path)


vision_analyze_tool = Tool(
    name="vision_analyze",
    description="Inspect image file properties: format, pixel dimensions, file size, and aspect ratio.",
    parameter_schema={
        "type": "object",
        "properties": {
            "image_path": {"type": "string", "description": "Path to image file to analyze."},
        },
        "required": ["image_path"],
    },
    handler=_vision_analyze_handler,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)
