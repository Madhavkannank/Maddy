"""
Unit and integration tests for Phase 9: Vision & Voice System.
Validates VisionEngine, VoiceEngine, vision tools, voice tools, and platform audio integration.
"""
import os
import tempfile
import struct
import pytest
from unittest.mock import MagicMock

from vision.engine import VisionEngine
from vision.tools import (
    vision_analyze_image_tool,
    vision_locate_element_tool,
    vision_verify_visual_tool,
)
from voice.engine import VoiceEngine
from voice.tools import (
    voice_speak_tool,
    voice_transcribe_tool,
    voice_listen_tool,
)
from tools.registry import ToolRegistry
from tools.__init__ import register_default_tools

def create_sample_png(path: str, width: int = 100, height: int = 50):
    """Generate a minimal valid PNG binary file header for testing."""
    header = b"\x89PNG\r\n\x1a\n"
    ihdr_chunk = b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    with open(path, "wb") as f:
        f.write(header)
        f.write(ihdr_chunk)

def test_vision_engine_analysis():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "test.png")
        create_sample_png(img_path, width=640, height=480)

        ve = VisionEngine()
        res = ve.analyze_image(img_path)

        assert res["success"] is True
        assert res["format"] == "PNG"
        assert res["width"] == 640
        assert res["height"] == 480
        assert res["aspect_ratio"] == 1.33

        # Base64 encoding
        b64 = ve.encode_image_base64(img_path)
        assert b64 is not None
        assert len(b64) > 0

def test_vision_engine_element_location_and_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "screen.png")
        create_sample_png(img_path, width=1920, height=1080)

        ve = VisionEngine()
        loc_res = ve.locate_element(img_path, "Submit button")
        assert loc_res["success"] is True
        assert "bounding_box" in loc_res
        bbox = loc_res["bounding_box"]
        assert "x" in bbox and "y" in bbox and "width" in bbox and "height" in bbox

        ver_res = ve.verify_visual(img_path, "Dialog box visible")
        assert ver_res["success"] is True
        assert ver_res["verified"] is True

def test_voice_engine_speak_and_transcribe():
    mock_platform = MagicMock()
    mock_provider = MagicMock()
    mock_provider.synthesize.return_value = {
        "success": True,
        "text": "Hello, I am M.A.D.D.Y.",
        "voice": "eve",
        "method": "f5tts_local_zero_shot_clone",
        "verification": "LOCAL CLONE — MATCH QUALITY VERIFIED",
    }
    ve = VoiceEngine(platform_driver=mock_platform, provider=mock_provider)

    # Test Speak via provider mock
    speak_res = ve.speak("Hello, I am M.A.D.D.Y.")
    assert speak_res["success"] is True
    mock_provider.synthesize.assert_called_once()

    # Test Transcribe non-existent audio file
    tr_fail = ve.transcribe("/tmp/nonexistent_audio.wav")
    assert tr_fail["success"] is False

    # Test Listen stub
    listen_res = ve.listen(duration_seconds=2.0)
    assert listen_res["success"] is True
    assert listen_res["duration_seconds"] == 2.0

def test_vision_voice_tools_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, "tool_test.png")
        create_sample_png(img_path, width=800, height=600)

        # Vision tool handlers
        an_res = vision_analyze_image_tool.handler({"image_path": img_path})
        assert an_res["success"] is True
        assert an_res["width"] == 800

        loc_res = vision_locate_element_tool.handler({"image_path": img_path, "element_description": "search bar"})
        assert loc_res["success"] is True

        ver_res = vision_verify_visual_tool.handler({"image_path": img_path, "expected_visual": "menu bar"})
        assert ver_res["success"] is True

        # Voice tool handlers
        sp_res = voice_speak_tool.handler({"text": "Testing voice tool execution."})
        assert sp_res["success"] is True

        lis_res = voice_listen_tool.handler({"duration_seconds": 1.5})
        assert lis_res["success"] is True

def test_tool_registry_includes_vision_and_voice():
    registry = ToolRegistry()
    register_default_tools(registry)

    # Check vision tools registered
    assert registry.get("vision.analyze_image") is not None
    assert registry.get("vision.locate_element") is not None
    assert registry.get("vision.verify_visual") is not None

    # Check voice tools registered
    assert registry.get("voice.speak") is not None
    assert registry.get("voice.transcribe") is not None
    assert registry.get("voice.listen") is not None

    schemas = registry.get_tool_schemas()
    schema_names = [s["function"]["name"] for s in schemas]
    assert "vision.analyze_image" in schema_names
    assert "voice.speak" in schema_names
