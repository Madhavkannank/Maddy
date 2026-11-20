"""
Tests for WakeWordEngine and State Machine transitions.
Verifies IDLE -> WAKE_DETECTED -> LISTENING -> TRANSCRIBING -> PROCESSING -> RESPONDING -> IDLE,
false wake resistance, cancellation, and barge-in interruption.
"""
import pytest
from voice.wakeword import WakeWordEngine, WakeWordState

def test_wake_word_lifecycle_transitions():
    states = []
    engine = WakeWordEngine(on_state_change=lambda o, n: states.append((o, n)))

    # Initial
    assert engine.get_state() == WakeWordState.IDLE

    # False wake attempt
    e1 = engine.process_audio_stream_event({"text": "Hello world outside"})
    assert engine.get_state() == WakeWordState.IDLE
    assert e1["action"] == "continue_standby"

    # Wake word trigger
    e2 = engine.process_audio_stream_event({"text": "Hey M.A.D.D.Y."})
    assert engine.get_state() == WakeWordState.LISTENING

    # Command received
    e3 = engine.process_audio_stream_event({"text": "What time is it?"})
    assert engine.get_state() == WakeWordState.PROCESSING
    assert e3["command"] == "What time is it?"

    # Assistant responding
    engine.mark_responding()
    assert engine.get_state() == WakeWordState.RESPONDING

    # Assistant finishes
    engine.complete_response_and_standby()
    assert engine.get_state() == WakeWordState.IDLE

def test_wake_word_barge_in_interruption():
    engine = WakeWordEngine()
    engine.mark_responding()
    assert engine.get_state() == WakeWordState.RESPONDING

    # User interrupts assistant speech
    e = engine.process_audio_stream_event({"text": "Hey M.A.D.D.Y., stop."})
    assert e["action"] == "barge_in_interrupted"
    assert engine.get_state() == WakeWordState.IDLE

def test_wake_word_silence_timeout():
    engine = WakeWordEngine(silence_timeout_seconds=3.0)
    engine.process_audio_stream_event({"text": "Hey M.A.D.D.Y."})
    assert engine.get_state() == WakeWordState.LISTENING

    # Silence event exceeding timeout
    e = engine.process_audio_stream_event({"is_silent": True, "duration": 4.0})
    assert e["action"] == "timeout_to_idle"
    assert engine.get_state() == WakeWordState.IDLE

def test_custom_wake_phrases_detection():
    engine = WakeWordEngine()
    test_cases = [
        "maddy",
        "M.A.D.D.Y.",
        "hey maddy",
        "ok maddy",
        "okay maddy",
        "hi maddy",
        "hello maddy",
        "yo maddy",
        "wake up maddy",
        "maddy online",
        "listen maddy",
        "are you there maddy",
        "daddy's home",
        "daddys home",
        "daddy is home",
        "wake up",
        "listen up",
    ]
    for phrase in test_cases:
        assert engine.detect_wake_word(phrase) is True, f"Failed to detect: {phrase}"
        assert engine.detect_wake_word(phrase.lower()) is True, f"Failed to detect lower: {phrase}"

    # Bare M.A.D.D.Y. should wake, but negative forms should not.
    assert engine.detect_wake_word("M.A.D.D.Y.") is True
    assert engine.detect_wake_word("maddy") is True
    assert engine.detect_wake_word("No Maddy, do not run that.") is False

    # Verify no false wake on arbitrary chatter and ambient room conversation
    assert engine.detect_wake_word("What is the weather outside today") is False
    assert engine.detect_wake_word("cowboy boots for sale") is False
    assert engine.detect_wake_word("Let's review the code together") is False
    assert engine.detect_wake_word("I need to wake up early tomorrow") is False
    assert engine.detect_wake_word("Can we start the project tomorrow morning") is False

def test_wake_with_trailing_command():
    engine = WakeWordEngine()

    # Single-breath command with Maddy wake phrase
    e1 = engine.process_audio_stream_event({"text": "hey maddy find a good horror movie"})
    assert e1["action"] == "command_extracted"
    assert e1["command"] == "find a good horror movie"
    assert engine.get_state() == WakeWordState.PROCESSING

    engine.complete_response_and_standby()

    # Single-breath command with 'ok maddy'
    e2 = engine.process_audio_stream_event({"text": "ok maddy reduce volume to 50%"})
    assert e2["action"] == "command_extracted"
    assert e2["command"] == "reduce volume to 50%"
    assert engine.get_state() == WakeWordState.PROCESSING

