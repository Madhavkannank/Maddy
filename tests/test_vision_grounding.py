"""Tests for Multimodal Vision & Active Window Grounding Tools in M.A.D.D.Y.

Verifies:
1. screen_inspect: captures screen, extracts active window info, and bounded OCR visible text.
2. active_window_inspect: queries foreground window title, bounds, and geometry.
3. screen_interact: executes click, type, and move actions with parameter validation.
4. vision_analyze: inspects image file metadata and format.
5. MaddyAgent integration: model calling screen_inspect and grounding answer on visible screen content.
"""
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from agents.base import AgentStatus
from agents.maddy_agent import MaddyAgent
from models.base import ModelProvider, ModelResponse
from tools.result import ToolResult
from tools.executor import ToolExecutor
from tools.registry import ToolRegistry
from vision.vision_tools import (
    active_window_inspect_tool,
    screen_inspect_tool,
    screen_interact_tool,
    vision_analyze_tool,
)


class MockProvider(ModelProvider):
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, prompt, **kwargs):
        if self.responses:
            return ModelResponse(content=self.responses.pop(0))
        return ModelResponse(content="Done.")

    def stream(self, prompt, **kwargs):
        yield self.generate(prompt).content

    def structured(self, prompt, schema, **kwargs):
        return {}


def test_active_window_inspect_tool():
    with patch("vision.vision_tools._default_vision.get_active_window_info") as mock_win:
        mock_win.return_value = {
            "title": "Visual Studio Code - Maddy",
            "hwnd": 12345,
            "rect": {"left": 0, "top": 0, "right": 1920, "bottom": 1080},
            "width": 1920,
            "height": 1080,
        }
        res = active_window_inspect_tool.handler({})
        assert res["success"] is True
        assert res["window_title"] == "Visual Studio Code - Maddy"
        assert res["width"] == 1920
        assert res["height"] == 1080
        assert res["bounds"]["right"] == 1920


def test_screen_inspect_tool_with_ocr():
    with patch("vision.vision_tools._default_computer.take_screenshot") as mock_shot, \
         patch("vision.vision_tools._default_vision.get_active_window_info") as mock_win, \
         patch("vision.vision_tools._default_vision.extract_screen_text_and_errors") as mock_ocr:

        mock_shot.return_value = {
            "success": True,
            "path": "C:\\temp\\fake_shot.png",
            "width": 1920,
            "height": 1080,
        }
        mock_win.return_value = {
            "title": "Windows Terminal - PowerShell",
            "width": 1200,
            "height": 800,
        }
        mock_ocr.return_value = {
            "extracted_text": "Traceback (most recent call last):\n  File 'app.py', line 42, in <module>\nZeroDivisionError: division by zero",
            "error_lines": ["ZeroDivisionError: division by zero"],
        }

        res = screen_inspect_tool.handler({"extract_text": True})
        assert res["success"] is True
        assert res["screenshot_path"] == "C:\\temp\\fake_shot.png"
        assert res["active_window"]["title"] == "Windows Terminal - PowerShell"
        assert "ZeroDivisionError" in res["visible_text"]
        assert len(res["error_traces"]) == 1


def test_screen_inspect_truncates_large_ocr():
    with patch("vision.vision_tools._default_computer.take_screenshot") as mock_shot, \
         patch("vision.vision_tools._default_vision.get_active_window_info") as mock_win, \
         patch("vision.vision_tools._default_vision.extract_screen_text_and_errors") as mock_ocr:

        mock_shot.return_value = {"success": True, "path": "test.png", "width": 800, "height": 600}
        mock_win.return_value = {"title": "Doc", "width": 800, "height": 600}
        mock_ocr.return_value = {
            "extracted_text": "A" * 5000,
            "error_lines": [],
        }

        res = screen_inspect_tool.handler({"extract_text": True})
        assert res["success"] is True
        assert len(res["visible_text"]) < 2200
        assert "Truncated" in res["visible_text"]


def test_screen_interact_click_type_move():
    with patch("vision.vision_tools._default_computer.click") as mock_click, \
         patch("vision.vision_tools._default_computer.type_text") as mock_type, \
         patch("vision.vision_tools._default_computer.move_cursor") as mock_move:

        mock_click.return_value = {"success": True, "action": "click", "x": 100, "y": 200}
        mock_type.return_value = {"success": True, "action": "type", "text": "hello"}
        mock_move.return_value = {"success": True, "action": "move", "x": 500, "y": 300}

        # Click
        res_click = screen_interact_tool.handler({"action": "click", "x": 100, "y": 200})
        assert res_click["success"] is True
        mock_click.assert_called_once_with(100, 200, button="left", clicks=1)

        # Type
        res_type = screen_interact_tool.handler({"action": "type", "text": "hello"})
        assert res_type["success"] is True
        mock_type.assert_called_once_with("hello")

        # Move
        res_move = screen_interact_tool.handler({"action": "move", "x": 500, "y": 300})
        assert res_move["success"] is True
        mock_move.assert_called_once_with(500, 300, smooth=True)

        # Missing required parameter errors
        res_bad_click = screen_interact_tool.handler({"action": "click"})
        assert res_bad_click["success"] is False
        assert "Coordinates 'x' and 'y' are required" in res_bad_click["error"]

        res_bad_type = screen_interact_tool.handler({"action": "type"})
        assert res_bad_type["success"] is False

        res_unknown = screen_interact_tool.handler({"action": "drag"})
        assert res_unknown["success"] is False
        assert "Unknown action" in res_unknown["error"]


def test_vision_analyze_tool():
    with patch("vision.vision_tools._default_vision.analyze_image") as mock_analyze:
        mock_analyze.return_value = {
            "success": True,
            "format": "PNG",
            "width": 1920,
            "height": 1080,
            "aspect_ratio": 1.7778,
            "size_bytes": 102400,
        }
        res = vision_analyze_tool.handler({"image_path": "test.png"})
        assert res["success"] is True
        assert res["format"] == "PNG"
        assert res["aspect_ratio"] == 1.7778

        # Missing image_path
        res_err = vision_analyze_tool.handler({"image_path": ""})
        assert res_err["success"] is False


def test_maddy_agent_visual_grounding_turn():
    """Verify MaddyAgent can inspect the screen, extract OCR and window, and answer based on visual feedback."""
    registry = ToolRegistry()
    registry.register(screen_inspect_tool)
    registry.register(active_window_inspect_tool)
    registry.register(screen_interact_tool)
    registry.register(vision_analyze_tool)

    executor = ToolExecutor(registry)

    with patch("vision.vision_tools._default_computer.take_screenshot") as mock_shot, \
         patch("vision.vision_tools._default_vision.get_active_window_info") as mock_win, \
         patch("vision.vision_tools._default_vision.extract_screen_text_and_errors") as mock_ocr:

        mock_shot.return_value = {"success": True, "path": "mock.png", "width": 1920, "height": 1080}
        mock_win.return_value = {"title": "VS Code - main.py", "width": 1920, "height": 1080}
        mock_ocr.return_value = {
            "extracted_text": "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
            "error_lines": ["TypeError: unsupported operand type(s) for +: 'int' and 'str'"],
        }

        mock_responses = [
            '<tool_call>{"name": "screen_inspect", "arguments": {"extract_text": true}}</tool_call>',
            "I checked your screen, boss. You're in VS Code on main.py, and there's a TypeError on line 12 where an integer is being concatenated with a string.",
        ]
        provider = MockProvider(mock_responses)

        agent = MaddyAgent(
            model_provider=provider,
            executor=executor,
            registry=registry,
            max_steps=5,
        )

        result = agent.run("What error is currently displayed on my screen?")
        assert result.status == AgentStatus.SUCCESS
        assert "TypeError" in (result.final_answer or "")
        assert "VS Code" in (result.final_answer or "")
