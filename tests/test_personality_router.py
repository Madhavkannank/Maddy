"""
Tests for PersonalityEngine and CapabilityRouter.
Verifies tone refinement, cliche removal, address preference lifecycle,
and capability classification across all domains.
"""
import pytest
from personality.engine import PersonalityEngine
from core.router import CapabilityRouter, CapabilityType
from memory.user_preferences import UserPreferences

def test_personality_cliche_removal(tmp_path):
    prefs_file = str(tmp_path / "prefs.json")
    prefs = UserPreferences(storage_path=prefs_file)
    engine = PersonalityEngine(user_preferences=prefs)

    raw = "As an AI language model, I'm delighted to help! The answer is 42."
    cleaned = engine.format_response(raw)
    assert "as an ai language model" not in cleaned.lower()
    assert "delighted to help" not in cleaned.lower()
    assert "42" in cleaned

def test_preferred_address_lifecycle(tmp_path):
    prefs_file = str(tmp_path / "prefs.json")
    prefs = UserPreferences(storage_path=prefs_file)
    engine = PersonalityEngine(user_preferences=prefs)

    assert engine.needs_address_onboarding() is True

    # User configures name
    chosen = engine.check_address_setup_input("Call me Madhav")
    assert chosen == "Madhav"
    assert engine.needs_address_onboarding() is False
    assert engine.get_address_title() == "Madhav"

    # Subsequent check
    assert prefs.get_preferred_address() == "Madhav"

def test_capability_router_classification():
    router = CapabilityRouter()

    # Math
    r_math = router.route("What is 23891 times 472?")
    assert r_math["primary"] == CapabilityType.MATH
    assert r_math["fast_path_eligible"] is True

    # Screen
    r_screen = router.route("What does this error on my screen mean?")
    assert r_screen["primary"] == CapabilityType.SCREEN_VISION
    assert r_screen["is_screen_dependent"] is True

    # Spotify
    r_spotify = router.route("Play my workout playlist on Spotify")
    assert r_spotify["primary"] == CapabilityType.SPOTIFY

    # Browser
    r_browser = router.route("Open the browser")
    assert r_browser["primary"] == CapabilityType.BROWSER

    # Conversation
    r_conv = router.route("Hey M.A.D.D.Y., how are you?")
    assert r_conv["primary"] == CapabilityType.CONVERSATION
    assert r_conv["fast_path_eligible"] is True

    # Compound
    r_comp = router.route("Open Spotify and then play my playlist and search Firefox for lyrics")
    assert r_comp["primary"] == CapabilityType.COMPOUND_TASK
