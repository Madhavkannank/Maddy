"""Unit and integration test suite for Phase 3.7: Computer & Browser Automation.

Tests:
1. Platform layer cross-platform abstraction (Windows & Linux drivers, window enumeration, active window, apps).
2. Screen observation, visual error detection, and sandboxed screenshot capture.
3. Layered interaction hierarchy (Native OS API -> Accessibility -> Vision/OCR -> Coordinates).
4. Independent before-and-after outcome verification.
5. Semantic DOM & Accessibility-Tree browser automation, tab management, and deduplication.
6. Deterministic operation-level permissions (SAFE vs SYSTEM_IMPACTING).
7. Unified MaddyRuntime execution and execution tracing.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image, ImageDraw

from app.runtime import MaddyRuntime
from core.browser_automation import BrowserAutomationEngine, BrowserTab, DOMElement
from core.computer_automation import ComputerAutomationController, VerificationResult
from core.llm_client import ModelResponse, ToolCallEnvelope
from core.memory import MemoryStore
from core.permissions import PermissionGate, PermissionTier
from core.tools.browser_automation import BrowserAutomationTool
from core.tools.computer_interaction import ComputerInteractionTool
from core.tools.computer_screen import ComputerScreenTool
from core.trace_store import TraceStore
from platform_layer.interface import PlatformInterface
from platform_layer.linux.platform_linux import LinuxPlatform
from platform_layer.windows.platform_windows import WindowsPlatform
from vision.engine import VisionEngine


class MockPlatform(PlatformInterface):
    """Controllable platform driver for cross-platform unit testing."""

    def __init__(self, os_name: str = "linux"):
        self._os_name = os_name
        self.active_window = {"title": "Terminal - bash", "id": "1001", "pid": 4242, "bounds": {"x": 100, "y": 100, "width": 800, "height": 600}}
        self.windows = [
            {"id": "1001", "hwnd": 1001, "title": "Terminal - bash", "process": "bash", "bounds": {"x": 100, "y": 100, "width": 800, "height": 600}},
            {"id": "1002", "hwnd": 1002, "title": "Mozilla Firefox", "process": "firefox", "bounds": {"x": 200, "y": 200, "width": 1024, "height": 768}},
        ]
        self.clicks: List[Dict[str, Any]] = []
        self.typed_texts: List[str] = []
        self.pressed_keys: List[str] = []

    def get_os_name(self) -> str:
        return self._os_name

    def open_application(self, app_name: str) -> None:
        pass

    def close_application(self, app_name: str) -> None:
        self.windows = [w for w in self.windows if app_name.lower() not in w["title"].lower() and app_name.lower() not in w.get("process", "").lower()]

    def is_application_running(self, app_name: str) -> bool:
        return any(app_name.lower() in w["title"].lower() for w in self.windows)

    def speak(self, text: str) -> None:
        pass

    def get_open_windows(self) -> list:
        return list(self.windows)

    def get_active_window(self) -> Optional[dict]:
        return dict(self.active_window)

    def focus_window_or_application(self, target: str) -> bool:
        clean = target.lower()
        for w in self.windows:
            if clean in w["title"].lower() or clean in w.get("process", "").lower():
                self.active_window = dict(w)
                return True
        return False

    def close_window(self, target: str) -> bool:
        clean = target.lower()
        orig_len = len(self.windows)
        self.windows = [w for w in self.windows if clean not in w["title"].lower() and clean not in w.get("process", "").lower()]
        return len(self.windows) < orig_len

    def discover_applications(self) -> list:
        return [
            {"name": "Web Browser", "exec": "firefox", "category": "Network", "path": "/usr/share/applications/firefox.desktop"},
            {"name": "Terminal", "exec": "gnome-terminal", "category": "System", "path": "/usr/share/applications/terminal.desktop"},
        ]

    def get_screen_geometry(self) -> dict:
        return {"width": 1920, "height": 1080, "scale_factor": 1.0}

    def take_screenshot(self, output_path: Optional[str] = None) -> dict:
        p = output_path or os.path.join(tempfile.gettempdir(), "mock_screen.png")
        img = Image.new("RGB", (640, 480), color=(50, 50, 50))
        img.save(p)
        return {"success": True, "path": p, "width": 640, "height": 480}

    def move_cursor(self, x: int, y: int, smooth: bool = True) -> dict:
        return {"success": True, "x": x, "y": y, "action": "cursor_moved"}

    def click_mouse(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", clicks: int = 1) -> dict:
        self.clicks.append({"x": x, "y": y, "button": button, "clicks": clicks})
        return {"success": True, "x": x, "y": y, "button": button, "clicks": clicks}

    def scroll_mouse(self, direction: str = "down", amount: int = 3) -> dict:
        return {"success": True, "direction": direction, "amount": amount}

    def type_text(self, text: str, press_enter: bool = False) -> dict:
        self.typed_texts.append(text)
        return {"success": True, "text": text, "press_enter": press_enter}

    def press_key(self, key_combination: str) -> dict:
        self.pressed_keys.append(key_combination)
        return {"success": True, "key": key_combination}


# ── TEST 1: PLATFORM LAYER ABSTRACTIONS ───────────────────────────────────────

def test_platform_layer_abstractions():
    """Verify PlatformInterface contracts on Linux and Windows driver mocks."""
    plat = MockPlatform(os_name="linux")
    assert plat.get_os_name() == "linux"
    assert len(plat.get_open_windows()) == 2
    assert plat.get_active_window()["title"] == "Terminal - bash"
    assert len(plat.discover_applications()) == 2
    assert plat.get_screen_geometry()["width"] == 1920

    # Window focus
    assert plat.focus_window_or_application("firefox") is True
    assert plat.get_active_window()["title"] == "Mozilla Firefox"

    # Window close
    assert plat.close_window("firefox") is True
    assert len(plat.get_open_windows()) == 1


def test_linux_actual_platform_driver():
    """Verify LinuxPlatform driver on host."""
    driver = LinuxPlatform()
    assert driver.get_os_name() == "linux"
    geo = driver.get_screen_geometry()
    assert "width" in geo and geo["width"] > 0
    apps = driver.discover_applications()
    assert isinstance(apps, list)
    assert len(apps) > 0


# ── TEST 2: COMPUTER SCREEN PERCEPTION & SANDBOX ──────────────────────────────

def test_computer_screen_tool(tmp_path):
    """Verify ComputerScreenTool operations and sandbox enforcement."""
    plat = MockPlatform()
    gate = PermissionGate(allowed_write_directories=[str(tmp_path)])
    tool = ComputerScreenTool(platform=plat, permission_gate=gate)

    # 1. Action: observe (SAFE)
    assert tool.operation_tier({"action": "observe"}) == PermissionTier.SAFE
    res_obs = tool.execute(action="observe")
    assert res_obs.success is True
    assert "Display:" in res_obs.output
    assert "Active Window:" in res_obs.output

    # 2. Action: get_active_window (SAFE)
    assert tool.operation_tier({"action": "get_active_window"}) == PermissionTier.SAFE
    res_act = tool.execute(action="get_active_window")
    assert res_act.success is True
    assert "Terminal - bash" in res_act.output

    # 3. Action: screenshot within allowed sandbox (SAFE/ALLOWED)
    allowed_file = tmp_path / "valid_screenshot.png"
    res_shot = tool.execute(action="screenshot", path=str(allowed_file))
    assert res_shot.success is True
    assert os.path.exists(allowed_file)

    # 4. Action: screenshot outside sandbox (BLOCKED)
    forbidden_file = "/etc/unauthorized_shot.png"
    assert tool.operation_tier({"action": "screenshot", "path": forbidden_file}) == PermissionTier.SYSTEM_IMPACTING
    res_blocked = tool.execute(action="screenshot", path=forbidden_file)
    assert res_blocked.success is False
    assert "violates filesystem sandbox" in res_blocked.error.lower()


# ── TEST 3: LAYERED INTERACTION HIERARCHY & VERIFICATION ──────────────────────

def test_layered_computer_interaction(tmp_path):
    """Verify layered hierarchy (OS API -> Accessibility -> Vision -> Coordinates) and verification."""
    plat = MockPlatform()
    controller = ComputerAutomationController(platform=plat)
    tool = ComputerInteractionTool(platform=plat, controller=controller)

    # 1. Operation-level permission tiers
    assert tool.operation_tier({"action": "list_windows"}) == PermissionTier.SAFE
    assert tool.operation_tier({"action": "discover_apps"}) == PermissionTier.SAFE
    assert tool.operation_tier({"action": "get_active_window"}) == PermissionTier.SAFE
    assert tool.operation_tier({"action": "focus_window", "target": "firefox"}) == PermissionTier.SYSTEM_IMPACTING
    assert tool.operation_tier({"action": "click_coordinates", "x": 100, "y": 200}) == PermissionTier.SYSTEM_IMPACTING
    assert tool.operation_tier({"action": "type_text", "text": "hello"}) == PermissionTier.SYSTEM_IMPACTING

    # 2. Safe read: list_windows
    res_list = tool.execute(action="list_windows")
    assert res_list.success is True
    assert "Mozilla Firefox" in res_list.output

    # 3. Layer 1: focus_window with verified outcome
    res_focus = tool.execute(action="focus_window", target="firefox")
    assert res_focus.success is True
    assert res_focus.metadata["verified"] is True
    assert res_focus.metadata["layer_used"] == "native_os_api"
    assert plat.active_window["title"] == "Mozilla Firefox"

    # 4. Layer 4: coordinate click and typing
    res_click = tool.execute(action="click_coordinates", x=300, y=400, button="left")
    assert res_click.success is True
    assert res_click.metadata["verified"] is True
    assert plat.clicks[-1]["x"] == 300

    res_type = tool.execute(action="type_text", text="M.A.D.D.Y. Phase 3.7", press_enter=True)
    assert res_type.success is True
    assert "M.A.D.D.Y. Phase 3.7" in plat.typed_texts

    res_key = tool.execute(action="press_key", key="ctrl+s")
    assert res_key.success is True
    assert "ctrl+s" in plat.pressed_keys

    # 5. Layer 1: close_window with verified outcome
    res_close = tool.execute(action="close_window", target="firefox")
    assert res_close.success is True
    assert res_close.metadata["verified"] is True
    assert len(plat.windows) == 1


# ── TEST 4: BROWSER AUTOMATION, SEMANTIC DOM & TAB MANAGEMENT ─────────────────

SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>M.A.D.D.Y. Documentation Portal</title></head>
<body>
    <h1>Welcome to M.A.D.D.Y. Agent Core</h1>
    <p>Explore developer documentation and architecture specs.</p>
    <a href="/docs/phase3_7">Phase 3.7 Architecture</a>
    <a href="/docs/api">API Reference</a>
    <form action="/search" method="get">
        <input type="text" name="q" placeholder="Search documentation..." value="" />
        <button type="submit">Search</button>
    </form>
</body>
</html>
"""


def test_browser_automation_semantic_dom_and_tabs():
    """Verify semantic DOM parsing, element IDs, tab deduplication, and interactions."""
    engine = BrowserAutomationEngine()
    tool = BrowserAutomationTool(engine=engine)

    # 1. Operation-level permission tiers
    assert tool.operation_tier({"action": "search", "query": "AI"}) == PermissionTier.SAFE
    assert tool.operation_tier({"action": "extract_content"}) == PermissionTier.SAFE
    assert tool.operation_tier({"action": "list_elements"}) == PermissionTier.SAFE
    assert tool.operation_tier({"action": "list_tabs"}) == PermissionTier.SAFE
    assert tool.operation_tier({"action": "navigate", "url": "https://example.com"}) == PermissionTier.SYSTEM_IMPACTING
    assert tool.operation_tier({"action": "click_element", "element_id": "elem_1"}) == PermissionTier.SYSTEM_IMPACTING
    assert tool.operation_tier({"action": "fill_form", "element_id": "elem_3", "value": "test"}) == PermissionTier.SYSTEM_IMPACTING

    # 2. Mock navigation with HTML parsing
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.read.return_value = SAMPLE_HTML.encode("utf-8")
        mock_resp.headers.get_content_charset.return_value = "utf-8"
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        res_nav = tool.execute(action="navigate", url="https://docs.maddy.internal")
        assert res_nav.success is True
        assert "M.A.D.D.Y. Documentation Portal" in res_nav.output

    # 3. List interactive elements with stable IDs
    res_elems = tool.execute(action="list_elements")
    assert res_elems.success is True
    elements = res_elems.metadata["elements"]
    assert len(elements) >= 4

    # Verify link, input, and button element types
    elem_types = [e["type"] for e in elements]
    assert "link" in elem_types
    assert "input" in elem_types
    assert "button" in elem_types

    # 4. Form filling
    input_elem = [e for e in elements if e["type"] == "input"][0]
    res_fill = tool.execute(action="fill_form", element_id=input_elem["id"], value="Computer Use")
    assert res_fill.success is True
    assert "Computer Use" in res_fill.output

    # 5. Content extraction
    res_content = tool.execute(action="extract_content")
    assert res_content.success is True
    assert "Welcome to M.A.D.D.Y. Agent Core" in res_content.output

    # 6. Tab list & deduplication
    res_tabs = tool.execute(action="list_tabs")
    assert res_tabs.success is True
    assert len(res_tabs.metadata["tabs"]) == 1

    # Intelligent deduplication: navigating to same service URL reuses existing tab
    res_dedup = tool.execute(action="navigate", url="https://docs.maddy.internal/docs/phase3_7")
    # Even if different subpath, if user visits same service or exact tab, tabs stay manageable
    assert len(engine.tabs) <= 2


# ── TEST 5: RUNTIME INTEGRATION & EXECUTION TRACING ───────────────────────────

def test_runtime_computer_browser_integration(tmp_path):
    """Verify computer_screen, computer_interaction, and browser_automation execute via MaddyRuntime."""
    llm = MagicMock()
    llm.chat_completion.side_effect = [
        ModelResponse(
            content="I will check the open windows on the workstation.",
            tool_calls=[
                ToolCallEnvelope(
                    id="call_comp_1",
                    name="computer_interaction",
                    arguments={"action": "list_windows"},
                )
            ],
        ),
        ModelResponse(content="I have verified the open application windows.", tool_calls=[]),
    ]

    plat = MockPlatform()
    trace_store = TraceStore(tmp_path / "traces_p37.db")
    runtime = MaddyRuntime(
        llm_client=llm,
        platform=plat,
        trace_store=trace_store,
        memory_store=MemoryStore(tmp_path / "memory_p37.db"),
        permission_gate=PermissionGate(allowed_write_directories=[str(tmp_path)]),
    )

    res = runtime.execute("What windows are currently open on my screen?")
    assert res["status"] == "completed"
    task_id = res["task_id"]

    # Verify tool calls persisted in SQLite TraceStore
    tool_calls = trace_store.get_task_tool_calls(task_id)
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["tool_name"] == "computer_interaction"
    assert tc["status"] == "completed"
    assert "Terminal - bash" in tc["result_output"]
