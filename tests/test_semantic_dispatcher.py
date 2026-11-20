"""
Tests for Granular Semantic Intent Taxonomy, SemanticDispatcher, and MathPhoneticizer.
Verifies:
1. MathPhoneticizer translates symbolic math (isqrt2, powers, operators) into natural spoken English.
2. MathEngine handles Whisper speech idioms ("x square minus 2 x to 3", "equal to 3").
3. SemanticDispatcher classifies granular intents (QA, Media, Commercial, Consensus, Algebra).
4. Zero-error fallback resilience when model is offline or unconfigured.
5. End-to-end MaddyAssistant execution of algebra queries avoids browser search hijacking.
"""
import pytest
from unittest.mock import MagicMock, patch

from core.intent_taxonomy import IntentPayload, SemanticIntent
from core.math_phonetics import MathPhoneticizer
from core.math_engine import MathEngine
from core.semantic_dispatcher import SemanticDispatcher
from models.base import ModelProvider, ModelResponse


def test_math_phoneticizer_complex_roots():
    """Verify isqrt2 / I*sqrt(2) is spoken as 'i times the square root of 2'."""
    raw = "x = 1 - I*sqrt(2), x = 1 + I*sqrt(2)"
    display, speech = MathPhoneticizer.format_math_result(raw)

    assert "√2" in display or "sqrt" in display
    # Voice speech must pronounce 'square root' and not 'isqrt'
    assert "i times the square root of 2" in speech
    assert "isqrt" not in speech.lower()
    assert "equals" in speech


def test_math_phoneticizer_powers_and_operators():
    """Verify x^2, +/-, and arithmetic operators are naturally vocalized."""
    raw = "x = 1 +/- isqrt2"
    display, speech = MathPhoneticizer.format_math_result(raw)

    assert "±" in display
    assert "plus or minus" in speech
    assert "i times the square root of 2" in speech

    spoken_eq = MathPhoneticizer.to_spoken_english("x^2 - 2x + 3 = 0")
    assert "x squared" in spoken_eq
    assert "minus" in spoken_eq
    assert "plus" in spoken_eq
    assert "equals 0" in spoken_eq


def test_math_engine_whisper_idiom_algebra():
    """Verify MathEngine handles spoken Whisper transcription: 'x square minus 2 x to 3'."""
    engine = MathEngine()

    # Whisper transcription: "Can you find the value of x in x square minus 2 x to 3?"
    res = engine.evaluate_detailed("Can you find the value of x in x square minus 2 x to 3?")
    assert res is not None
    assert "x = -1" in res["display"] or "x = 3" in res["display"]
    assert "x equals" in res["speech"]

    # Spoken complex roots: "solve x square minus 2x plus 3 = 0"
    res_complex = engine.evaluate_detailed("solve x square minus 2x plus 3 = 0")
    assert res_complex is not None
    assert "i times the square root of 2" in res_complex["speech"]


def test_semantic_dispatcher_mock_classification():
    """Verify dispatcher properly parses structured JSON response from Groq."""
    mock_provider = MagicMock()
    mock_provider.is_configured.return_value = True
    mock_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "MATH_ALGEBRA",
            "slots": {
                "equation": "x^2 - 2*x - 3 = 0",
                "variable": "x"
            },
            "direct_speech": ""
        }'''
    )

    dispatcher = SemanticDispatcher(model_provider=mock_provider)
    payload = dispatcher.dispatch("Can you find the value of x in x square minus 2 x to 3?")

    assert payload.intent == SemanticIntent.MATH_ALGEBRA
    assert payload.equation == "x^2 - 2*x - 3 = 0"


def test_semantic_dispatcher_factoid_direct_qa():
    """Verify QA_FACTOID intent extracts direct speech answer."""
    mock_provider = MagicMock()
    mock_provider.is_configured.return_value = True
    mock_provider.generate.return_value = ModelResponse(
        content='''{
            "intent": "QA_FACTOID",
            "slots": {
                "entity": "water"
            },
            "direct_speech": "Water boils at 100 degrees Celsius at standard atmospheric pressure."
        }'''
    )

    dispatcher = SemanticDispatcher(model_provider=mock_provider)
    payload = dispatcher.dispatch("What is the boiling point of water?")

    assert payload.intent == SemanticIntent.QA_FACTOID
    assert "100 degrees Celsius" in payload.direct_speech


def test_semantic_dispatcher_offline_fallback():
    """Verify dispatcher safely falls back to deterministic rules when LLM is offline."""
    mock_provider = MagicMock()
    mock_provider.is_configured.return_value = False

    dispatcher = SemanticDispatcher(model_provider=mock_provider)
    payload_math = dispatcher.dispatch("find x in x^2 - 2x = 3")
    assert payload_math.intent == SemanticIntent.MATH_ALGEBRA

    payload_yt = dispatcher.dispatch("search youtube for lofi beats")
    assert payload_yt.intent == SemanticIntent.SEARCH_MEDIA_VIDEO


def test_maddy_end_to_end_algebra_no_browser():
    """Verify that algebra queries execute through Math and do NOT open browser search."""
    from app.maddy_assistant import MaddyAssistant

    assistant = MaddyAssistant()
    # Mock browser launch to ensure it is NEVER called
    with patch.object(assistant, "_handle_direct_web_search") as mock_web_search:
        result = assistant.execute("Can you find the value of x in x square minus 2 x to 3?", speak_response=False)
        assert result["success"] is True
        assert "-1" in result["response"] and "3" in result["response"]
        assert "equals" in result["response"] or "=" in result["response"]
        # Crucial check: Browser search must NOT be invoked!
        mock_web_search.assert_not_called()
