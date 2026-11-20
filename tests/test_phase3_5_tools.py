"""Unit tests for Phase 3.5 migrated BaseTool subclasses:
SystemControlTool, AppControlTool, MediaControlTool, WebBrowserTool,
CameraTool, ArxivTool, and TimerTool.
"""
from __future__ import annotations

import io
import tempfile
import time
from unittest.mock import MagicMock, patch

import pytest

from core.permissions import PermissionGate, PermissionTier
from core.tools import (
    AppControlTool,
    ArxivTool,
    CameraTool,
    MediaControlTool,
    SystemControlTool,
    TimerTool,
    ToolRegistry,
    WebBrowserTool,
)
from platform_layer.interface import PlatformInterface


class MockPlatform(PlatformInterface):
    def __init__(self, os_name: str = "linux"):
        self.os_name = os_name
        self.volume = 50
        self.brightness = 75
        self.muted = False
        self.apps_opened = []
        self.apps_closed = []
        self.windows_focused = []

    def get_os_name(self) -> str:
        return self.os_name

    def open_application(self, app_name: str) -> None:
        self.apps_opened.append(app_name)

    def close_application(self, app_name: str) -> None:
        self.apps_closed.append(app_name)

    def is_application_running(self, app_name: str) -> bool:
        return app_name.lower() in ("spotify", "code")

    def focus_window_or_application(self, target: str) -> bool:
        self.windows_focused.append(target)
        return True

    def get_open_windows(self) -> list:
        return [{"hwnd": 1, "title": "Spotify", "process": "spotify.exe"}]

    def speak(self, text: str) -> None:
        pass

    def set_volume(self, level: int) -> bool:
        self.volume = level
        return True

    def get_volume(self) -> int:
        return self.volume

    def set_brightness(self, level: int) -> bool:
        self.brightness = level
        return True

    def get_brightness(self) -> int:
        return self.brightness

    def set_mute(self, mute: bool) -> bool:
        self.muted = mute
        return True


def test_system_control_tool():
    platform = MockPlatform()
    tool = SystemControlTool(platform=platform)

    assert tool.tier == PermissionTier.SYSTEM_IMPACTING
    schema = tool.get_parameters_schema()
    assert "action" in schema["required"]

    # 1. Get / Set Volume
    res = tool.execute(action="get_volume")
    assert res.success is True
    assert res.metadata["volume"] == 50

    res = tool.execute(action="set_volume", level=80)
    assert res.success is True
    assert res.metadata["volume"] == 80
    assert platform.volume == 80

    # 2. Get / Set Brightness
    res = tool.execute(action="get_brightness")
    assert res.success is True
    assert res.metadata["brightness"] == 75

    res = tool.execute(action="set_brightness", level=90)
    assert res.success is True
    assert res.metadata["brightness"] == 90
    assert platform.brightness == 90

    # 3. Set Mute
    res = tool.execute(action="set_mute", mute=True)
    assert res.success is True
    assert res.metadata["muted"] is True
    assert platform.muted is True

    # 4. Lock workstation
    with patch("shutil.which", return_value="/bin/loginctl"), patch("subprocess.run") as mock_sub:
        mock_sub.return_value.returncode = 0
        res = tool.execute(action="lock_workstation")
        assert res.success is True

    # 5. Invalid action
    res = tool.execute(action="invalid_action")
    assert res.success is False
    assert "Unknown system_control action" in res.error


def test_app_control_tool():
    platform = MockPlatform()
    tool = AppControlTool(platform=platform)

    # 1. Launch app
    res = tool.execute(action="launch", app_name="spotify")
    assert res.success is True

    # 2. Close app
    res = tool.execute(action="close", app_name="spotify")
    assert res.success is True
    assert "spotify" in platform.apps_closed

    # 3. Is running
    res = tool.execute(action="is_running", app_name="spotify")
    assert res.success is True
    assert res.metadata["running"] is True

    res = tool.execute(action="is_running", app_name="nonexistent")
    assert res.success is True
    assert res.metadata["running"] is False

    # 4. Focus window
    res = tool.execute(action="focus", target="Spotify")
    assert res.success is True
    assert "Spotify" in platform.windows_focused

    # 5. List windows
    res = tool.execute(action="list_windows")
    assert res.success is True
    assert res.metadata["count"] == 1


def test_media_control_tool():
    # Test Windows media control
    platform_win = MockPlatform(os_name="windows")
    tool_win = MediaControlTool(platform=platform_win)

    # Keep the Windows-specific ctypes path isolated from the Linux test host.
    with patch.object(
        tool_win,
        "_execute_windows",
        return_value=__import__("core.tools.base", fromlist=["ToolResult"]).ToolResult(
            success=True,
            output="Dispatched Windows media key event 'play_pause'.",
            metadata={"action": "play_pause", "platform": "windows"},
        ),
    ) as mock_windows:
        res = tool_win.execute(action="play_pause")
        assert res.success is True
        assert res.metadata["platform"] == "windows"
        mock_windows.assert_called_once_with("play_pause")

    # Test Linux media control
    platform_lin = MockPlatform(os_name="linux")
    tool_lin = MediaControlTool(platform=platform_lin)

    with patch("shutil.which", side_effect=lambda cmd: "/bin/dbus-send" if cmd == "dbus-send" else None), \
         patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        res = tool_lin.execute(action="next", player="spotify")
        assert res.success is True
        assert res.metadata["method"] == "dbus-send"


def test_web_browser_tool():
    tool = WebBrowserTool(platform=MockPlatform())

    # 1. Open URL
    with patch("webbrowser.open") as mock_open:
        res = tool.execute(action="open_url", url="https://example.com")
        assert res.success is True
        mock_open.assert_called_with("https://example.com")

    # 2. Open Domain
    with patch("webbrowser.open") as mock_open:
        res = tool.execute(action="open_domain", domain="github")
        assert res.success is True
        mock_open.assert_called_with("https://github.com")

    # 3. Search DuckDuckGo Lite (mocked)
    sample_html = """
    <html><body>
    <a href="/l/?uddg=https%3A%2F%2Fwww.python.org%2F" class='result-link'>Python Official</a>
    <td class='result-snippet'>Python programming language home page.</td>
    </body></html>
    """
    mock_resp = MagicMock()
    mock_resp.read.return_value = sample_html.encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = tool.execute(action="search", query="python")
        assert res.success is True
        assert res.metadata["count"] == 1
        assert res.metadata["results"][0]["title"] == "Python Official"
        assert "python.org" in res.metadata["results"][0]["url"]


def test_camera_tool():
    gate = PermissionGate(allowed_write_directories=[tempfile.gettempdir()])
    tool = CameraTool(permission_gate=gate)

    assert tool.tier == PermissionTier.SYSTEM_IMPACTING

    # 1. Mock availability check
    mock_cv2 = MagicMock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, MagicMock())
    mock_cv2.VideoCapture.return_value = mock_cap

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        res = tool.execute(action="check_availability")
        assert res.success is True
        assert res.metadata["available"] is True

    # 2. Mock capture frame
    fake_frame = MagicMock()
    fake_frame.shape = (480, 640, 3)
    mock_cap.read.return_value = (True, fake_frame)
    mock_cv2.imwrite.return_value = True

    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        res = tool.execute(action="capture_frame")
        assert res.success is True
        assert res.metadata["width"] == 640
        assert res.metadata["height"] == 480

    # 3. Permission denied on unauthorized path outside sandbox
    with patch.dict("sys.modules", {"cv2": mock_cv2}):
        res = tool.execute(action="capture_frame", output_path="/etc/unauthorized_snapshot.jpg")
        assert res.success is False
        assert "Permission denied" in res.error


def test_arxiv_tool():
    tool = ArxivTool()
    assert tool.tier == PermissionTier.SAFE

    # 1. Empty query validation
    res = tool.execute(query="   ")
    assert res.success is False

    # 2. Mocked ArXiv Atom XML
    sample_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <title>Attention Is All You Need</title>
            <summary>The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.</summary>
            <published>2017-06-12T17:47:58Z</published>
            <author><name>Ashish Vaswani</name></author>
            <author><name>Noam Shazeer</name></author>
            <link rel="alternate" href="http://arxiv.org/abs/1706.03762v7"/>
            <link title="pdf" href="http://arxiv.org/pdf/1706.03762v7"/>
        </entry>
    </feed>
    """
    mock_resp = MagicMock()
    mock_resp.read.return_value = sample_xml
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = tool.execute(query="Attention Is All You Need", max_results=1)
        assert res.success is True
        assert res.metadata["count"] == 1
        paper = res.metadata["papers"][0]
        assert paper["title"] == "Attention Is All You Need"
        assert "Ashish Vaswani" in paper["authors"]
        assert "1706.03762" in paper["arxiv_url"]


def test_timer_tool():
    expired_timers = []
    tool = TimerTool(on_timer_expired=lambda t: expired_timers.append(t))

    assert tool.tier == PermissionTier.SAFE

    # 1. Set Timer (very short)
    res = tool.execute(action="set_timer", duration_seconds=10, label="Pasta")
    assert res.success is True
    tid = res.metadata["timer_id"]
    assert tid.startswith("timer_")

    # 2. List Timers
    res = tool.execute(action="list_timers")
    assert res.success is True
    assert res.metadata["count"] >= 1
    assert any(t["id"] == tid for t in res.metadata["timers"])

    # 3. Cancel Timer
    res = tool.execute(action="cancel_timer", timer_id=tid)
    assert res.success is True
    assert res.metadata["cancelled"] is True


def test_tool_registry_with_phase3_5_tools():
    reg = ToolRegistry()
    tools = [
        SystemControlTool(),
        AppControlTool(),
        MediaControlTool(),
        WebBrowserTool(),
        CameraTool(),
        ArxivTool(),
        TimerTool(),
    ]
    for t in tools:
        reg.register(t)

    schemas = reg.get_openai_tool_schemas()
    assert len(schemas) == 7
    schema_names = {s["function"]["name"] for s in schemas}
    assert schema_names == {
        "system_control",
        "app_control",
        "media_control",
        "web_browser",
        "camera_tool",
        "arxiv_tool",
        "timer_tool",
    }
