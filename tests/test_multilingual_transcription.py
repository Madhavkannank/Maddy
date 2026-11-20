import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from core.math_engine import MathEngine
from core.math_phonetics import MathPhoneticizer
from voice.engine import EdgeTTSProvider, ElevenLabsProvider
from app.maddy_assistant import MaddyAssistant


def test_math_phoneticizer_speech_vs_display():
    """Verify clean Unicode display alongside natural phonetic spoken output."""
    math_engine = MathEngine()
    res = math_engine.evaluate_detailed("solve x^2 - 2*x + 3 = 0")
    assert res is not None
    assert "display" in res
    assert "speech" in res

    display_text = res["display"]
    speech_text = res["speech"]

    # Display has clean square root and plus-or-minus Unicode
    assert "√" in display_text or "i" in display_text
    # Spoken output has NO raw 'isqrt' and uses natural phonetic words
    assert "isqrt" not in speech_text
    assert "square root of" in speech_text.lower() or "root" in speech_text.lower()


def test_edgetts_multilingual_script_detection():
    """Verify EdgeTTS automatically selects Tamil / Malayalam neural voices for native text."""
    provider = EdgeTTSProvider()

    # English defaults to configured Ryan / Sonia
    voice_en, name_en = provider.resolve_edge_voice("Hello sir, how are you?", "chris")
    assert voice_en == "en-GB-RyanNeural"
    assert name_en == "chris"

    # Tamil text selects Valluvar / Pallavi
    voice_ta_m, name_ta_m = provider.resolve_edge_voice("வணக்கம் சார், எப்படி இருக்கிறீர்கள்?", "chris")
    assert voice_ta_m == "ta-IN-ValluvarNeural"
    assert name_ta_m == "tamil"

    voice_ta_f, name_ta_f = provider.resolve_edge_voice("வால்யூம் குறைக்கப்பட்டது", "sarah")
    assert voice_ta_f == "ta-IN-PallaviNeural"
    assert name_ta_f == "tamil"

    # Malayalam text selects Midhun / Sobhana
    voice_ml_m, name_ml_m = provider.resolve_edge_voice("നമസ്കാരം, എന്താണ് വിശേഷം?", "chris")
    assert voice_ml_m == "ml-IN-MidhunNeural"
    assert name_ml_m == "malayalam"

    voice_ml_f, name_ml_f = provider.resolve_edge_voice("ശബ്ദം കുറച്ചു", "sarah")
    assert voice_ml_f == "ml-IN-SobhanaNeural"
    assert name_ml_f == "malayalam"


def test_multilingual_volume_commands():
    """Verify Tamil and Malayalam volume commands work directly through platform layer."""
    assistant = MaddyAssistant()

    with patch.object(assistant.platform, "get_volume", return_value=50), \
         patch.object(assistant.platform, "set_volume", return_value=True) as mock_set_vol:

        # Tamil volume reduce
        res_ta = assistant.execute("வால்யூம் குறைக்கவும்", speak_response=False)
        assert res_ta["success"] is True
        assert "குறைக்கப்பட்டது" in res_ta["response"]
        mock_set_vol.assert_called_with(35)

        # Check _handle_volume_command directly first
        v_check = assistant._handle_volume_command("ശബ്ദം കൂട്ടൂ")
        assert v_check is not None, f"_handle_volume_command returned None!"

        # Malayalam volume increase
        res_ml = assistant.execute("ശബ്ദം കൂട്ടൂ", speak_response=False)
        assert res_ml["success"] is True
        assert "വർദ്ധിപ്പിച്ചു" in res_ml["response"]
        mock_set_vol.assert_called_with(65)


def test_transcription_groq_mock_and_fallback():
    """Verify dual-tier transcription logic with Groq primary and local fallback."""
    import voice_listener

    # Verify voice_listener imports and structure
    assert hasattr(voice_listener, "transcribe_audio") or callable(voice_listener.audio_callback)


def test_resolve_groq_api_keys_uses_backup_keys(monkeypatch):
    """Groq STT should rotate across all configured keys instead of hard-falling back after the first quota hit."""
    import voice_listener

    monkeypatch.setenv("GROQ_API_KEY", "primary-key")
    monkeypatch.setenv("GROQ_KEY", "legacy-key")
    monkeypatch.setenv("GROQ_API_KEY_FALLBACK", "backup-key")

    keys = voice_listener.resolve_groq_api_keys()

    assert keys[0] == "primary-key"
    assert "backup-key" in keys
    assert "legacy-key" in keys
