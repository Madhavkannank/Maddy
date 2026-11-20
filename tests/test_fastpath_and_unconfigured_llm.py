"""
Unit tests for deterministic fast-path execution and unconfigured cloud LLM error messaging.
Verifies:
1. Spoken commands with lead-in vocatives ("Channel, open camera.", "Maddy, open camera")
   execute directly without requiring cloud LLM dispatch.
2. Fast-path app launch, workstation lock, status, and time queries return structured success.
3. Unconfigured Cloud LLM raises clear, informative guidance instead of opaque generic disruption messages.
"""
from unittest.mock import MagicMock
import pytest
from app.runtime import MaddyRuntime
from core.agent_loop import MaddyAgentCore
from core.errors import ModelApiError
from core.llm_client import CloudLLMClient


def test_fastpath_open_camera_with_vocative():
    """Verify that 'Channel, open camera.' executes open_camera fast-path immediately."""
    rt = MaddyRuntime()
    rt.platform.open_application = MagicMock(return_value=True)

    res = rt.execute("Channel, open camera.")
    assert res.get("status") == "completed"
    assert res.get("data", {}).get("command") == "/open_camera"
    assert "Opening camera" in res.get("response", "")
    rt.platform.open_application.assert_called_with("camera")


def test_fastpath_open_camera_direct():
    """Verify that 'open camera' executes open_camera fast-path."""
    rt = MaddyRuntime()
    rt.platform.open_application = MagicMock(return_value=True)

    res = rt.execute("open camera")
    assert res.get("status") == "completed"
    assert res.get("data", {}).get("command") == "/open_camera"
    assert "Opening camera" in res.get("response", "")
    rt.platform.open_application.assert_called_with("camera")


def test_fastpath_lock_workstation():
    """Verify that 'Hey Maddy, lock workstation' executes lock workstation fast-path."""
    rt = MaddyRuntime()
    rt.platform.lock_workstation = MagicMock(return_value=True)

    res = rt.execute("Hey Maddy, lock workstation")
    assert res.get("status") == "completed"
    assert res.get("data", {}).get("command") == "/lock"
    assert "locked" in res.get("response", "")


def test_fastpath_live_time_query():
    """Verify that 'what time is it' returns local live time fast-path."""
    rt = MaddyRuntime()
    res = rt.execute("what time is it")
    assert res.get("status") == "completed"
    assert res.get("data", {}).get("command") == "/time"
    assert "It is" in res.get("response", "")


def test_fastpath_app_launch():
    """Verify that 'Maddy, open calculator' launches calculator without calling cloud LLM."""
    rt = MaddyRuntime()
    rt.platform.open_application = MagicMock(return_value=True)

    res = rt.execute("Maddy, open calculator")
    assert res.get("status") == "completed"
    assert res.get("data", {}).get("command") == "/open_calculator"
    assert "Opening Calculator" in res.get("response", "")
    rt.platform.open_application.assert_called_with("calculator")


def test_unconfigured_cloud_llm_explicit_error_response():
    """Verify that unconfigured Cloud LLM produces clear actionable error text."""
    mock_llm = MagicMock(spec=CloudLLMClient)
    mock_llm.chat_completion.side_effect = ModelApiError(
        message="Cloud LLM client unconfigured: missing GROQ_API_KEY.",
        status_code=401,
        provider="groq",
        retryable=False,
    )

    core = MaddyAgentCore(llm_client=mock_llm)
    res = core.run_turn("What is quantum computing?")
    resp_text = res.get("response", "") if isinstance(res, dict) else str(res)
    assert "unconfigured" in resp_text.lower()
    assert "groq_api_key" in resp_text.lower()


def test_camera_tool_registered_with_alias_and_executes():
    """Verify that camera tool is accessible via 'camera' and 'camera_tool' and captures snapshot with authorization."""
    from core.tools.camera_tool import CameraTool
    from core.permissions import PermissionGate
    gate = PermissionGate(approval_callback=lambda tool, args: True)
    rt = MaddyRuntime(permission_gate=gate)
    assert rt.tool_registry.get_tool("camera") is not None
    assert rt.tool_registry.get_tool("camera_tool") is not None

    # Test exact prompt from user report
    res = rt.execute("Open camera, take a photo and send it to telegram")
    assert res.get("status") == "completed"
    assert "Camera tool is unavailable" not in res.get("response", "")
    assert res.get("data", {}).get("command") == "/photo"
    assert "Webcam photo captured successfully" in res.get("response", "")
    # Cleanup any captured test image
    p = res.get("data", {}).get("path")
    if p:
        import os
        if os.path.exists(p):
            os.remove(p)


def test_camera_tool_supports_snapshot_action_alias():
    """Verify CameraTool handles both 'snapshot' and 'capture_frame' action names."""
    from core.tools.camera_tool import CameraTool
    cam = CameraTool()
    # Test availability
    res_avail = cam.execute(action="check_availability")
    assert res_avail.success is True
    assert "connected" in res_avail.output or "unavailable" in res_avail.output

    # Test snapshot action
    res_snap = cam.execute(action="snapshot")
    assert res_snap.success is True
    assert "Camera snapshot captured successfully" in res_snap.output
    p = res_snap.metadata.get("path")
    if p:
        import os
        if os.path.exists(p):
            os.remove(p)
