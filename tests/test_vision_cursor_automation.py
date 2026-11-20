"""
Unit and integration tests for Antigravity-grade Vision and Cursor Automation.
Tests:
1. UI Element Localization & Centroid Calculation
2. Deep Screen Content Analysis with Active Window Context
3. Native Cursor Movements, Clicks, Scrolling, and Typing
4. Perception-Action Verification Loop (click_element)
5. MaddyAssistant natural language integration for GUI commands
"""
import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image, ImageDraw

from vision.engine import VisionEngine
from computer.controller import ComputerController
from app.maddy_assistant import MaddyAssistant


def create_sample_ui_image(path: str, width: int = 800, height: int = 600) -> str:
    """Create a synthetic UI image with buttons and text for vision testing."""
    img = Image.new("RGB", (width, height), color=(240, 240, 245))
    draw = ImageDraw.Draw(img)
    # Draw a mock "Search" button
    draw.rectangle([(100, 50), (250, 90)], fill=(70, 130, 180))
    # Draw a mock "Login" button
    draw.rectangle([(300, 50), (420, 90)], fill=(34, 139, 34))
    img.save(path)
    return path


@pytest.fixture
def vision_engine():
    return VisionEngine()


@pytest.fixture
def computer_controller():
    return ComputerController()


def test_element_localization_centroid(vision_engine):
    """Verify locate_element calculates bounding boxes and clickable centroids."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "ui_test.png")
        create_sample_ui_image(img_path, width=1000, height=800)

        res = vision_engine.locate_element(img_path, "search")
        assert res["success"] is True
        assert "centroid" in res
        cx, cy = res["centroid"]
        assert isinstance(cx, int) and isinstance(cy, int)
        assert 0 <= cx <= 1000
        assert 0 <= cy <= 800

        bb = res["bounding_box"]
        assert bb["width"] > 0
        assert bb["height"] > 0


def test_active_window_detection(vision_engine):
    """Verify active window information retrieval."""
    win_info = vision_engine.get_active_window_info()
    assert isinstance(win_info, dict)
    assert "title" in win_info
    assert "hwnd" in win_info


def test_analyze_screen_content_synthetic(vision_engine):
    """Verify deep screen content analysis without external LLM."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "screen_test.png")
        create_sample_ui_image(img_path, width=800, height=600)

        res = vision_engine.analyze_screen_content(img_path, model_provider=None)
        assert res["success"] is True
        assert "active_window" in res
        assert "summary" in res
        assert len(res["summary"]) > 0


def test_analyze_screen_content_with_llm(vision_engine):
    """Verify screen content analysis with mocked LLM reasoning."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "screen_test.png")
        create_sample_ui_image(img_path, width=800, height=600)

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "You are currently viewing a browser window with active search tabs."
        mock_llm.generate.return_value = mock_resp

        res = vision_engine.analyze_screen_content(img_path, model_provider=mock_llm, user_query="What is on screen?")
        assert res["success"] is True
        assert "browser window" in res["summary"]
        mock_llm.generate.assert_called_once()


def test_cursor_movements_mocked(computer_controller):
    """Verify move_cursor dispatches smooth coordinates without crashing."""
    res = computer_controller.move_cursor(500, 300, smooth=False)
    assert res["success"] is True
    assert res["x"] == 500
    assert res["y"] == 300


def test_mouse_clicks_and_scroll(computer_controller):
    """Verify click, double click, right click, and scroll dispatching."""
    # Click
    c_res = computer_controller.click(x=100, y=200, button="left", clicks=1)
    assert c_res["success"] is True

    # Double click
    dc_res = computer_controller.double_click(x=150, y=250)
    assert dc_res["success"] is True
    assert dc_res["clicks"] == 2

    # Right click
    rc_res = computer_controller.right_click(x=200, y=300)
    assert rc_res["success"] is True
    assert rc_res["button"] == "right"

    # Scroll
    s_down = computer_controller.scroll(direction="down", amount=3)
    assert s_down["success"] is True
    assert s_down["direction"] == "down"

    s_up = computer_controller.scroll(direction="up", amount=2)
    assert s_up["success"] is True
    assert s_up["direction"] == "up"


def test_type_text(computer_controller):
    """Verify text typing / clipboard dispatch."""
    t_res = computer_controller.type_text("Hello M.A.D.D.Y.")
    assert t_res["success"] is True
    assert t_res["text_length"] == len("Hello M.A.D.D.Y.")


def test_click_element_perception_loop(computer_controller):
    """Verify click_element perception-action loop."""
    with patch.object(computer_controller, "take_screenshot") as mock_shot:
        with patch.object(computer_controller, "click") as mock_click:
            mock_shot.return_value = {"path": "/fake/shot.png"}
            mock_click.return_value = {"success": True, "action": "clicked"}

            res = computer_controller.click_element("search box")
            assert res["success"] is True
            assert res["element_description"] == "search box"
            assert "centroid" in res
            mock_click.assert_called_once()


def test_maddy_assistant_vision_cursor_integration():
    """Verify MaddyAssistant executes natural language vision and cursor commands."""
    assistant = MaddyAssistant()

    # Mock hardware platform and controller so tests never hijack real user mouse
    mock_controller = MagicMock()
    mock_controller.click_element.return_value = {"success": True}
    mock_controller.scroll.return_value = {"success": True}
    mock_controller.type_text.return_value = {"success": True}
    mock_controller.take_screenshot.return_value = {"path": "/fake/screen.png"}
    assistant.computer_controller = mock_controller

    mock_vision = MagicMock()
    mock_vision.analyze_screen_content.return_value = {
        "success": True,
        "summary": "You are currently running VS Code with an active Python workspace."
    }
    assistant.vision_engine = mock_vision

    # 1. Click command
    res_click = assistant.execute("click on the search bar", speak_response=False)
    assert res_click["success"] is True
    assert "clicked on search bar" in res_click["response"].lower()
    mock_controller.click_element.assert_called_with("search bar", vision_engine=mock_vision)

    # 2. Double click command
    mock_controller.click_element.reset_mock()
    res_dbl = assistant.execute("double click on project folder", speak_response=False)
    assert res_dbl["success"] is True
    assert "double-clicked on project folder" in res_dbl["response"].lower()
    mock_controller.click_element.assert_called_with("project folder", vision_engine=mock_vision, clicks=2)

    # 3. Right click command
    mock_controller.click_element.reset_mock()
    res_rc = assistant.execute("right click on desktop", speak_response=False)
    assert res_rc["success"] is True
    assert "right-clicked on desktop" in res_rc["response"].lower()
    mock_controller.click_element.assert_called_with("desktop", vision_engine=mock_vision, button="right")

    # 4. Scroll commands
    res_sd = assistant.execute("scroll down", speak_response=False)
    assert res_sd["success"] is True
    assert "scrolled down" in res_sd["response"].lower()
    mock_controller.scroll.assert_called_with(direction="down", amount=4)

    res_su = assistant.execute("scroll up", speak_response=False)
    assert res_su["success"] is True
    assert "scrolled up" in res_su["response"].lower()
    mock_controller.scroll.assert_called_with(direction="up", amount=4)

    # 5. Type into element
    res_type = assistant.execute("type 'hello world' into the search input", speak_response=False)
    assert res_type["success"] is True
    assert "typed" in res_type["response"].lower()
    mock_controller.type_text.assert_called_with("hello world", press_enter=False)

    # 6. Screen analysis query
    res_screen = assistant.execute("what is on my screen right now?", speak_response=False)
    assert res_screen["success"] is True
    assert "vs code" in res_screen["response"].lower()
