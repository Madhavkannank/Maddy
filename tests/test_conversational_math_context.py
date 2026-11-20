"""
Comprehensive Test Suite for Conversational Multi-Turn Context & Speech Sanitization.
Verifies:
1. LaTeX and math delimiter ($x$, $3x$) sanitization for TTS.
2. Algebraic expression simplification without '=' (e.g. 'Solve, 3x plus 2x minus, 3x plus 4x plus 5').
3. Multi-turn variable substitution ('x is equal to 5 in that equation').
4. Rolling conversational history persistence in MaddyAssistant.
5. Speech sanitization defense-in-depth across VoiceEngine and MaddyAssistant.
"""
import pytest
from unittest.mock import MagicMock, patch

from core.math_phonetics import MathPhoneticizer
from core.math_engine import MathEngine
from app.maddy_assistant import MaddyAssistant


def test_latex_and_dollar_phonetic_sanitization():
    """Verify that LaTeX math delimiters and stray dollar signs are never read as 'dollar'."""
    raw_response = (
        "Let's break that down. We have: $3x plus 2x minus 3x plus 4x plus 5$ "
        "Combine the $x$ terms: $3x plus 2x equals 5x$ $5x minus 3x equals 2x$ "
        "$2x plus 4x equals 6x$ So the expression simplifies to: $6x plus 5$ "
        "If you're looking for a specific value, you'll need to tell me what $x$ is. "
        "Otherwise, that's the simplified form, boss."
    )

    sanitized = MathPhoneticizer.sanitize_for_speech(raw_response)

    # Must NOT contain any dollar signs
    assert "$" not in sanitized
    # Must NOT contain "dollar" (since there is no monetary amount here)
    assert "dollar" not in sanitized.lower()
    # Must have preserved x, 6x plus 5, etc.
    assert "Combine the x terms" in sanitized
    assert "simplifies to: 6x plus 5" in sanitized
    assert "what x is" in sanitized


def test_currency_vs_math_dollar_differentiation():
    """Verify that real currency ($50) converts to '50 dollars' while math $x$ strips the dollar sign."""
    currency_text = "The item costs $50 with a $15 shipping fee for $x$ items."
    sanitized = MathPhoneticizer.sanitize_for_speech(currency_text)

    assert "$" not in sanitized
    assert "50 dollars" in sanitized
    assert "15 dollars" in sanitized
    assert "x items" in sanitized


def test_markdown_and_latex_math_commands_sanitized():
    """Verify LaTeX commands and markdown formatting are phoneticized for voice output."""
    raw = "**Result:** The fraction is `\\frac{a}{b}` and `x^2 + y^2 = 25` with `\\pm 5`."
    sanitized = MathPhoneticizer.sanitize_for_speech(raw)

    assert "**" not in sanitized
    assert "`" not in sanitized
    assert "Result: The fraction is a divided by b and x squared plus y squared equals 25 with plus or minus 5." in sanitized


def test_algebraic_expression_simplification():
    """Verify that expressions with commas and no '=' are simplified deterministically."""
    engine = MathEngine()

    # Case 1: The user's exact query with commas from Whisper
    query = "Solve, 3x plus 2x minus, 3x plus 4x plus 5."
    res = engine.evaluate_detailed(query)

    assert res is not None
    assert "6x + 5" in res["display"]
    assert "6x plus 5" in res["speech"]
    assert engine.last_expression is not None
    assert "x" in engine.last_variables


def test_multi_turn_variable_substitution():
    """Verify that a follow-up 'x is equal to 5 in that equation' evaluates accurately."""
    engine = MathEngine()

    # Turn 1: Simplify expression
    res1 = engine.evaluate_detailed("Solve, 3x plus 2x minus, 3x plus 4x plus 5.")
    assert "6x + 5" in res1["display"]

    # Turn 2: Follow-up substitution
    res2 = engine.evaluate_detailed("x is equal to 5 in that equation.")
    assert res2 is not None
    assert "35" in res2["display"]
    assert "Substituting x = 5 into 6x + 5 gives 35" in res2["display"]
    assert "Substituting x equals 5 into 6x plus 5 gives 35" in res2["speech"]

    # Turn 2 variation: 'what if x = 10'
    res3 = engine.evaluate_detailed("what if x is 10")
    assert res3 is not None
    assert "65" in res3["display"]


def test_maddy_assistant_end_to_end_conversational_math():
    """Verify end-to-end two-turn conversational math execution through MaddyAssistant."""
    assistant = MaddyAssistant()

    # Turn 1: User asks to solve expression
    res1 = assistant.execute("Solve, 3x plus 2x minus, 3x plus 4x plus 5.", speak_response=False)
    assert res1["success"] is True
    assert "6x + 5" in res1["response"] or "6x" in res1["response"]

    # Turn 2: Follow-up substitution
    res2 = assistant.execute("x is equal to 5 in that equation.", speak_response=False)
    assert res2["success"] is True
    assert "35" in res2["response"]

    # Verify conversation history was tracked
    assert len(assistant.conversation_history) >= 4
    assert any("3x" in turn["content"] for turn in assistant.conversation_history)
    assert any("35" in turn["content"] for turn in assistant.conversation_history)


def test_voice_engine_automatic_speech_sanitization():
    """Verify VoiceEngine.speak sanitizes LaTeX and dollar signs before provider synthesis."""
    assistant = MaddyAssistant()
    mock_provider = MagicMock()
    mock_provider.synthesize.return_value = {"success": True, "output_path": "fake.wav"}
    assistant.voice_engine.provider = mock_provider
    assistant.voice_engine._play_audio_file = MagicMock(return_value=True)

    assistant.voice_engine.speak("We simplify $3x$ to get $x = 5$ at a price of $20.")
    call_args = mock_provider.synthesize.call_args
    passed_text = call_args.kwargs["text"]

    assert "$" not in passed_text
    assert "3x" in passed_text
    assert "x equals 5" in passed_text
    assert "20 dollars" in passed_text
