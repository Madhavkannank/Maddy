"""
Tests for Telegram-compatible slash commands implemented in the new architecture.
Verifies /status, /help, /screenshot, /photo, /lock, and /capture_* commands.
"""
from unittest.mock import MagicMock, patch
import pytest

from app.runtime import MaddyRuntime
from core.tools.base import BaseTool, ToolRegistry, ToolResult


def test_slash_help(tmp_path):
    runtime = MaddyRuntime(trace_store=MagicMock(), memory_store=MagicMock())
    res = runtime.execute("/help")
    assert res["status"] == "completed"
    assert "/status" in res["response"]
    assert "/screenshot" in res["response"]
    assert "/lock" in res["response"]


def test_slash_status(tmp_path):
    runtime = MaddyRuntime(trace_store=MagicMock(), memory_store=MagicMock())
    res = runtime.execute("/status")
    assert res["status"] == "completed"
    assert "M.A.D.D.Y. PC SYSTEM STATUS" in res["response"]
    assert "CPU Usage" in res["response"]
    assert "RAM Usage" in res["response"]


def test_slash_lock(tmp_path):
    runtime = MaddyRuntime(trace_store=MagicMock(), memory_store=MagicMock())
    tool = runtime.tool_registry.get_tool("system_control")
    if tool:
        with patch.object(tool, "execute", return_value=ToolResult(success=True, output="Workstation locked successfully.")):
            res = runtime.execute("/lock")
            assert res["status"] == "completed"
            assert "locked" in res["response"].lower()


def test_slash_screenshot(tmp_path):
    runtime = MaddyRuntime(trace_store=MagicMock(), memory_store=MagicMock())
    screen_tool = runtime.tool_registry.get_tool("computer_screen")
    if screen_tool:
        with patch.object(screen_tool, "execute", return_value=ToolResult(success=True, output="/tmp/test_screen.png", metadata={"path": "/tmp/test_screen.png"})):
            res = runtime.execute("/screenshot")
            assert res["status"] == "completed"
            assert "screenshot captured successfully" in res["response"].lower()


def test_slash_photo(tmp_path):
    from core.permissions import PermissionGate
    gate = PermissionGate(approval_callback=lambda *a, **k: True)
    runtime = MaddyRuntime(trace_store=MagicMock(), memory_store=MagicMock(), permission_gate=gate)
    cam_tool = runtime.tool_registry.get_tool("camera")
    if cam_tool:
        with patch.object(cam_tool, "execute", return_value=ToolResult(success=True, output="/tmp/test_cam.jpg", metadata={"path": "/tmp/test_cam.jpg"})):
            res = runtime.execute("/photo")
            assert res["status"] == "completed"
            assert "photo captured successfully" in res["response"].lower()


def test_slash_capture_start_stop(tmp_path):
    from core.permissions import PermissionGate
    gate = PermissionGate(approval_callback=lambda *a, **k: True)
    runtime = MaddyRuntime(trace_store=MagicMock(), memory_store=MagicMock(), permission_gate=gate)
    with patch("tools.session_capture_tool.handle_session_capture_start", return_value={"success": True, "spoken_response": "Session capture activated."}):
        res = runtime.execute("/capture_start")
        assert res["status"] == "completed"
        assert "Session capture activated" in res["response"]

    with patch("tools.session_capture_tool.handle_session_capture_stop", return_value={"success": True, "spoken_response": "Session capture stopped."}):
        res = runtime.execute("stop stop")
        assert res["status"] == "completed"
        assert "Session capture stopped" in res["response"]
