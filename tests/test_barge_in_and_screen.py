"""
Unit and Integration Tests for:
1. Hands-Free Conversational Voice (Barge-in / interruption handling & Continuous Dialogue).
2. Multimodal Screen Awareness & Visual Perception routing.
3. Distinct Long-Term Memory Independence (ensuring multiple facts coexist without collapsing).
"""
import threading
import sys
import pytest
from unittest.mock import MagicMock, patch

try:
    import sounddevice
except Exception:
    sys.modules["sounddevice"] = MagicMock()

from core.router import CapabilityRouter, CapabilityType
from core.intent_taxonomy import IntentPayload, SemanticIntent
from memory.manager import MemoryManager
from memory.schema import MemoryType, MemoryStatus
from memory.relation import classify_memory_relation, MemoryRelation
from voice.engine import VoiceEngine
from voice_listener import should_listen_for_followup, is_valid_speech_transcription
from app.maddy_assistant import MaddyAssistant


def test_barge_in_stop_event_cancels_playback():
    """Verify that setting stop_event immediately halts audio playback."""
    engine = VoiceEngine()
    stop_event = threading.Event()

    with patch("soundfile.read", return_value=(MagicMock(), 24000)), \
         patch("sounddevice.play") as mock_play, \
         patch("sounddevice.stop") as mock_stop:

        # Mock an active stream
        mock_stream = MagicMock()
        mock_stream.active = True
        
        with patch("sounddevice.get_stream", return_value=mock_stream):
            # Set stop event after a brief delay
            stop_event.set()
            result = engine._play_audio_file("fake_audio.wav", stop_event=stop_event)
            assert result is False
            mock_play.assert_called_once()
            mock_stop.assert_called_once()


def test_continuous_dialogue_should_listen_for_followup():
    """Verify continuous dialogue mode remains open for normal conversational responses."""
    # Standard responses should keep conversation open
    assert should_listen_for_followup("That simplifies to 6x plus 5.") is True
    assert should_listen_for_followup("Your machine has an RTX 3050.") is True
    assert should_listen_for_followup("What would you like to do next?") is True
    assert should_listen_for_followup("I have updated your script.") is True

    # Terminal standby phrases should gracefully end the follow-up window
    assert should_listen_for_followup("Standing by.") is False
    assert should_listen_for_followup("Goodbye boss.") is False
    assert should_listen_for_followup("Good night sir, signing off.") is False


def test_screen_awareness_router_and_intent():
    """Verify natural screen inspection phrases and SCREEN_ANALYZE intent route to screen perception."""
    router = CapabilityRouter()
    
    # Natural screen referents
    assert router.is_screen_dependent("look at my screen") is True
    assert router.is_screen_dependent("what is on my screen") is True
    assert router.is_screen_dependent("explain this error on my screen") is True
    assert router.is_screen_dependent("describe my screen") is True
    assert router.is_screen_dependent("what am i looking at") is True

    # Capability classification
    res = router.route("look at my screen and tell me what is happening")
    assert CapabilityType.SCREEN_VISION.value in res["all_capabilities"]
    assert res["is_screen_dependent"] is True


def test_memory_independence_no_collapse():
    """Verify that multiple facts about the user are stored as separate memories without being swallowed."""
    mm = MemoryManager()
    
    # Store Fact 1: Project
    rec1 = mm.remember(
        content="User is participating in Smart India Hackathon 2026",
        memory_type=MemoryType.SEMANTIC_FACT,
    )
    assert rec1 is not None

    # Store Fact 2: Name
    rec2 = mm.remember(
        content="User name is Madhav",
        memory_type=MemoryType.SEMANTIC_FACT,
    )
    assert rec2 is not None
    # Must NOT be merged into rec1!
    assert rec2.memory_id != rec1.memory_id
    assert "Madhav" in rec2.content

    # Store Fact 3: Work
    rec3 = mm.remember(
        content="User is working on AI defect detection",
        memory_type=MemoryType.SEMANTIC_FACT,
    )
    assert rec3 is not None
    assert rec3.memory_id != rec1.memory_id
    assert rec3.memory_id != rec2.memory_id
    assert "defect detection" in rec3.content

    # Verify all 3 active records exist in store
    active = mm.store.list_records(status=MemoryStatus.ACTIVE)
    contents = [r.content for r in active]
    assert any("Hackathon" in c for c in contents)
    assert any("Madhav" in c for c in contents)
    assert any("defect detection" in c for c in contents)


def test_noise_hallucination_filtering():
    """Verify acoustic noise hallucinations (like ZJ-3-2-1) and filler sounds are discarded."""
    # Reject noise artifacts and letter-number acronyms
    assert is_valid_speech_transcription("ZJ-3-2-1.") is False
    assert is_valid_speech_transcription("zj") is False
    assert is_valid_speech_transcription("psst") is False
    assert is_valid_speech_transcription("A-1") is False
    assert is_valid_speech_transcription("thank you") is False
    assert is_valid_speech_transcription("bye") is False
    assert is_valid_speech_transcription("") is False

    # Accept genuine spoken commands
    assert is_valid_speech_transcription("What is the weather today?") is True
    assert is_valid_speech_transcription("Open Spotify.") is True
    assert is_valid_speech_transcription("Can you check for any newspapers on concrete defects") is True

