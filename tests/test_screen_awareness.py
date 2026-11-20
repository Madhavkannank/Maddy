"""
Tests for Screen-Aware Conversation and Vision Engine analysis.
Verifies screen referent recognition, text/error extraction, and visual effect verification.
"""
import os
import pytest
from core.router import CapabilityRouter
from vision.engine import VisionEngine
from PIL import Image, ImageDraw

def test_screen_referent_detection():
    router = CapabilityRouter()

    assert router.is_screen_dependent("What does this error mean?") is True
    assert router.is_screen_dependent("What's wrong on my screen?") is True
    assert router.is_screen_dependent("What am I looking at?") is True
    assert router.is_screen_dependent("What button should I press?") is True

    # Negative checks: do NOT screenshot unnecessarily
    assert router.is_screen_dependent("What is 2 + 2?") is False
    assert router.is_screen_dependent("Open Spotify.") is False
    assert router.is_screen_dependent("What is the capital of France?") is False

def test_vision_engine_text_and_error_extraction(tmp_path):
    img_path = str(tmp_path / "mock_error_screen.png")
    img = Image.new("RGB", (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Traceback (most recent call last):\nValueError: Invalid parameter", fill=(255, 80, 80))
    img.save(img_path)

    ve = VisionEngine()
    analysis = ve.extract_screen_text_and_errors(img_path)
    assert analysis["success"] is True
    assert "dimensions" in analysis
    assert analysis["image_path"] == img_path

def test_vision_action_effect_verification(tmp_path):
    p1 = str(tmp_path / "screen_before.png")
    p2 = str(tmp_path / "screen_after.png")

    # Before screenshot
    img1 = Image.new("RGB", (200, 200), color=(0, 0, 0))
    img1.save(p1)

    # After screenshot with visual change
    img2 = Image.new("RGB", (200, 200), color=(255, 255, 255))
    img2.save(p2)

    ve = VisionEngine()
    res = ve.verify_action_effect(p1, p2, expected_change="Color turned white")
    assert res["verified"] is True
    assert res["change_detected"] is True

    # Test identical screenshots -> no change detected
    res_ident = ve.verify_action_effect(p1, p1, expected_change="No change")
    assert res_ident["change_detected"] is False
