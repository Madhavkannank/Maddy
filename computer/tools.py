"""
Computer control tool definitions for M.A.D.D.Y. (Modular Autonomous Digital Daemon for You) ACI (computer.screenshot, computer.click, computer.type).
"""
import os
import tempfile
from typing import Any, Dict
from computer.controller import ComputerController
from tools.base import RiskLevel, Tool

_default_computer_controller = ComputerController()

def _handle_computer_screenshot(args: Dict[str, Any]) -> Dict[str, Any]:
    path = args.get("path")
    if not path:
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, f"maddy_screenshot_{int(os.getpid())}.png")

    return _default_computer_controller.take_screenshot(path)

def _handle_computer_click(args: Dict[str, Any]) -> Dict[str, Any]:
    x = int(args["x"])
    y = int(args["y"])
    return _default_computer_controller.click(x, y)

def _handle_computer_type(args: Dict[str, Any]) -> Dict[str, Any]:
    text = str(args["text"])
    return _default_computer_controller.type_text(text)

computer_screenshot_tool = Tool(
    name="computer.screenshot",
    description="Capture current desktop screen display and save screenshot image.",
    parameter_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Optional file path where screenshot PNG image will be saved.",
            },
        },
    },
    handler=_handle_computer_screenshot,
    risk_level=RiskLevel.LOW,
    timeout=10.0,
)

computer_click_tool = Tool(
    name="computer.click",
    description="Simulate mouse click at screen coordinates (x, y).",
    parameter_schema={
        "type": "object",
        "properties": {
            "x": {
                "type": "integer",
                "description": "Horizontal X screen coordinate.",
            },
            "y": {
                "type": "integer",
                "description": "Vertical Y screen coordinate.",
            },
        },
        "required": ["x", "y"],
    },
    handler=_handle_computer_click,
    risk_level=RiskLevel.MEDIUM,
    timeout=10.0,
)

computer_type_tool = Tool(
    name="computer.type",
    description="Simulate typing text characters using keyboard automation.",
    parameter_schema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text string to type out.",
            },
        },
        "required": ["text"],
    },
    handler=_handle_computer_type,
    risk_level=RiskLevel.MEDIUM,
    timeout=10.0,
)
