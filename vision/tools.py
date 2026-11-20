"""
Vision tool definitions for M.A.D.D.Y. ACI (vision.analyze_image, vision.locate_element, vision.verify_visual).
"""
from typing import Any, Dict
from tools.base import RiskLevel, Tool
from vision.engine import VisionEngine

_default_vision_engine = VisionEngine()

def _handle_vision_analyze_image(args: Dict[str, Any]) -> Dict[str, Any]:
    image_path = args["image_path"]
    return _default_vision_engine.analyze_image(image_path)

def _handle_vision_locate_element(args: Dict[str, Any]) -> Dict[str, Any]:
    image_path = args["image_path"]
    element_description = args["element_description"]
    return _default_vision_engine.locate_element(image_path, element_description)

def _handle_vision_verify_visual(args: Dict[str, Any]) -> Dict[str, Any]:
    image_path = args["image_path"]
    expected_visual = args["expected_visual"]
    return _default_vision_engine.verify_visual(image_path, expected_visual)

vision_analyze_image_tool = Tool(
    name="vision.analyze_image",
    description="Analyze image properties such as dimensions, file format, aspect ratio, and size.",
    parameter_schema={
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to the target image or display screenshot file.",
            }
        },
        "required": ["image_path"],
    },
    handler=_handle_vision_analyze_image,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

vision_locate_element_tool = Tool(
    name="vision.locate_element",
    description="Locate visual UI element coordinates (bounding box) on a display screenshot.",
    parameter_schema={
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to display screenshot image file.",
            },
            "element_description": {
                "type": "string",
                "description": "Description of the visual UI element to locate (e.g. 'Submit button', 'Search field').",
            },
        },
        "required": ["image_path", "element_description"],
    },
    handler=_handle_vision_locate_element,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)

vision_verify_visual_tool = Tool(
    name="vision.verify_visual",
    description="Verify if a visual element or expected UI state is visible on a display screenshot.",
    parameter_schema={
        "type": "object",
        "properties": {
            "image_path": {
                "type": "string",
                "description": "Path to display screenshot image file.",
            },
            "expected_visual": {
                "type": "string",
                "description": "Expected visual element or state description.",
            },
        },
        "required": ["image_path", "expected_visual"],
    },
    handler=_handle_vision_verify_visual,
    risk_level=RiskLevel.LOW,
    timeout=15.0,
)
